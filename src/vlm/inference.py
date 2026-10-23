"""
VLM Inference Engine for humor detection.

IMPORTANT: Confidence is derived from token-level logits, NOT from VLM self-reporting.
When the VLM generates 'true' or 'false' for the 'humorous' field, we extract
the softmax probabilities of 'true' vs 'false' tokens from the model's logits.
This gives a genuine model-derived probability rather than a hallucinated number.
"""

import time
import yaml
import json
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import StoppingCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info
from src.vlm.model_loader import load_model_and_processor
from src.vlm.prompts import (
    build_humor_analysis_prompt,
    build_cultural_analysis_prompt,
    build_cultural_analysis_prompt_config_c,
    build_multi_image_prompt,
)
from src.vlm.output_parser import parse_json_response
from src.cultural.category_detector import detect_categories

logger = logging.getLogger(__name__)


class DiagnosticStoppingCriteria(StoppingCriteria):
    def __init__(self, start_time, timeout_seconds=900.0, initial_input_length=0, generation_timeout_seconds=None):
        self.prefill_start_time = start_time
        self.generation_timeout_seconds = generation_timeout_seconds if generation_timeout_seconds is not None else timeout_seconds
        self.initial_input_length = initial_input_length
        self.generation_start_time = None
        self.prefill_duration = None
        self.last_log_time = start_time
        self.tokens_generated = 0
        self.timed_out = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        current_time = time.time()
        self.tokens_generated = input_ids.shape[1] - self.initial_input_length

        if self.generation_start_time is None:
            self.generation_start_time = current_time
            self.prefill_duration = current_time - self.prefill_start_time
            self.last_log_time = current_time
            logger.info("Prefill completed in %.1fs. Decoding started (timeout=%.0fs)...",
                        self.prefill_duration, self.generation_timeout_seconds)

        # Log progress every 30 seconds of decoding
        if current_time - self.last_log_time > 30.0:
            logger.info("Generation progress: %d tokens in %.1fs (prefill: %.1fs)...", 
                        self.tokens_generated, current_time - self.generation_start_time, self.prefill_duration)
            self.last_log_time = current_time

        if (current_time - self.generation_start_time) > self.generation_timeout_seconds:
            logger.warning("Generation TIMED OUT after %.1fs decoding (prefill: %.1fs, tokens: %d)!", 
                           current_time - self.generation_start_time, self.prefill_duration, self.tokens_generated)
            self.timed_out = True
            return True
        return False


class VLMInferenceEngine:
    def __init__(self, config_path: str = "configs/model.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_id = self.config["model_id"]
        self.generation_params = self.config.get("generation", {})

        self.model = None
        self.processor = None
        self.device = None

        # Token IDs for 'true' and 'false' — resolved after processor loads
        self._true_token_ids = None
        self._false_token_ids = None

    def load(self):
        """Loads model and processor into memory."""
        self.model, self.processor, self.device = load_model_and_processor(self.model_id)
        self._resolve_humor_token_ids()
        logger.info("VLM Engine loaded successfully on %s", self.device)

    def _resolve_humor_token_ids(self):
        """
        Pre-resolve token IDs for 'true' and 'false' strings.
        These are used to extract logit-based probability for the humorous field.
        """
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor

        # Try multiple token representations that the model might use
        # Note: BPE tokenizers produce different IDs for words with leading spaces
        # (e.g. ' true' is token 830, while 'true' is token 1866)
        true_candidates = ["true", " true", "True", " True", "TRUE", " TRUE"]
        false_candidates = ["false", " false", "False", " False", "FALSE", " FALSE"]

        self._true_token_ids = set()
        self._false_token_ids = set()

        for candidate in true_candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                self._true_token_ids.add(ids[0])

        for candidate in false_candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                self._false_token_ids.add(ids[0])

        logger.info("Resolved true token IDs: %s, false token IDs: %s",
                     self._true_token_ids, self._false_token_ids)

    def _extract_humor_probability(self, scores, generated_ids_trimmed):
        """
        Extract P(humorous) from the model's logits at the position where
        it generates 'true' or 'false' for the humorous field.

        The prompt asks the VLM to output JSON with "humorous": bool as the first field.
        We scan the generated token sequence to find where 'true'/'false' was generated
        and use the logits at that position to compute a genuine probability.

        Returns:
            tuple: (humor_prob: float, non_humor_prob: float)
                   Both sum to ~1.0. Returns (None, None) if extraction fails.
        """
        if not scores or not self._true_token_ids or not self._false_token_ids:
            logger.warning("Cannot extract logit probability: scores=%s, true_ids=%s, false_ids=%s",
                           bool(scores), bool(self._true_token_ids), bool(self._false_token_ids))
            return None, None

        try:
            # Find the token position where true/false was generated
            humor_token_pos = None
            generated_token_list = generated_ids_trimmed[0].tolist() if len(generated_ids_trimmed) > 0 else []

            for pos, token_id in enumerate(generated_token_list):
                if token_id in self._true_token_ids or token_id in self._false_token_ids:
                    humor_token_pos = pos
                    break  # First occurrence is for the 'humorous' field at start of JSON

            if humor_token_pos is None:
                logger.warning("Could not find true/false token in generated sequence")
                return None, None

            if humor_token_pos >= len(scores):
                logger.warning("Token position %d exceeds scores length %d",
                               humor_token_pos, len(scores))
                return None, None

            # Get logits at that position
            logits_at_pos = scores[humor_token_pos]
            if logits_at_pos.dim() > 1:
                logits_at_pos = logits_at_pos[0]  # batch dim

            # Extract logits for true and false token IDs
            all_relevant_ids = list(self._true_token_ids | self._false_token_ids)
            relevant_logits = logits_at_pos[all_relevant_ids]

            # Softmax over just the true/false token logits
            probs = F.softmax(relevant_logits.float(), dim=0)

            # Sum probabilities for all true-like and false-like tokens
            true_prob = 0.0
            false_prob = 0.0
            for i, tid in enumerate(all_relevant_ids):
                if tid in self._true_token_ids:
                    true_prob += probs[i].item()
                else:
                    false_prob += probs[i].item()

            # Normalize to ensure they sum to 1
            total = true_prob + false_prob
            if total > 0:
                true_prob /= total
                false_prob /= total

            logger.info("Logit-based probability: P(humorous)=%.4f, P(non-humorous)=%.4f",
                         true_prob, false_prob)
            return true_prob, false_prob

        except Exception as e:
            logger.error("Failed to extract humor probability from logits: %s", e)
            return None, None

    def _run_generation(self, messages):
        """
        Runs the VLM generation with logit extraction.
        Returns (parsed_result_dict, humor_prob, non_humor_prob).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate WITH score output for logit-based confidence
        gen_params = dict(self.generation_params)
        if not gen_params.get("do_sample", False):
            gen_params.pop("temperature", None)
        gen_params["return_dict_in_generate"] = True
        gen_params["output_scores"] = True

        # Safety: Add timeout stopping criteria
        # Decouple vision-encoder prefill duration from autoregressive token generation timeout.
        gen_timeout = float(self.generation_params.get("generation_timeout_seconds", 900.0))
        initial_length = inputs.input_ids.shape[1]
        start_time = time.time()
        stopping_criteria = DiagnosticStoppingCriteria(
            start_time,
            timeout_seconds=gen_timeout,
            initial_input_length=initial_length,
            generation_timeout_seconds=gen_timeout
        )
        gen_params["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])

        logger.info("[ANALYSIS] VLM inference started (decoding timeout=%.0fs)", gen_timeout)
        outputs = self.model.generate(**inputs, **gen_params)
        inference_time = time.time() - start_time
        logger.info("[ANALYSIS] VLM inference completed in %.1fs", inference_time)

        if stopping_criteria.timed_out:
            logger.error("VLM Inference forcefully terminated due to generation timeout. Output may be truncated/malformed.")

        prefill_dur = (
            round(stopping_criteria.prefill_duration, 2)
            if stopping_criteria.prefill_duration is not None
            else round(inference_time, 2)
        )
        decoding_dur = round(max(0.0, inference_time - prefill_dur), 2)

        # Extract generated text
        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        gen_token_count = len(generated_ids_trimmed[0])
        max_tokens = gen_params.get("max_new_tokens", -1)
        hit_max = (gen_token_count >= max_tokens) if max_tokens > 0 else False
        logger.info("Generation Diagnostics: %d tokens generated in %.1fs (prefill: %.1fs). Hit max_new_tokens: %s. Timed out: %s.", 
                    gen_token_count, decoding_dur, prefill_dur, hit_max, stopping_criteria.timed_out)

        logger.info("VLM raw output: %s", output_text[:500])

        # Parse JSON from VLM output
        logger.info("[ANALYSIS] Output parsing started")
        parsed = parse_json_response(output_text)
        logger.info("[ANALYSIS] Output parsing completed")

        # Extract logit-based probability
        logger.info("[ANALYSIS] Probability extraction started")
        scores = outputs.scores  # tuple of (vocab_size,) tensors, one per generated token
        humor_prob, non_humor_prob = self._extract_humor_probability(scores, generated_ids_trimmed)
        logger.info("[ANALYSIS] Probability extraction completed: P(humor)=%s, P(non_humor)=%s",
                    humor_prob, non_humor_prob)

        diagnostics = {
            "timed_out": stopping_criteria.timed_out,
            "hit_max_new_tokens": hit_max,
            "generated_token_count": gen_token_count,
            "raw_response": output_text,
            "inference_duration_s": round(inference_time, 2),
            "prefill_duration_s": prefill_dur,
            "decoding_duration_s": decoding_dur,
        }

        return parsed, humor_prob, non_humor_prob, diagnostics

    def _build_result(self, parsed, humor_prob, non_humor_prob, mode="general", images_count=1, cultural_sources=None, diagnostics=None):
        """
        Combines the parsed VLM output with logit-based probabilities
        into a structured result dictionary.
        """
        result = dict(parsed)
        result["mode"] = mode
        result["images_analyzed"] = images_count
        result["cultural_sources"] = cultural_sources if cultural_sources is not None else []
        if diagnostics:
            result.update(diagnostics)

        # Check if inference timed out
        is_timed_out = diagnostics.get("timed_out", False) if diagnostics else False

        # Set logit-based confidence
        if not is_timed_out and humor_prob is not None and non_humor_prob is not None:
            result["humor_probability"] = round(humor_prob, 4)
            result["non_humor_probability"] = round(non_humor_prob, 4)

            # Determine prediction from logits (overrides VLM text if they disagree)
            logit_humorous = humor_prob >= 0.5
            vlm_humorous = result.get("humorous")

            if vlm_humorous is not None and bool(vlm_humorous) != logit_humorous:
                logger.warning(
                    "VLM text says humorous=%s but logits say P(humorous)=%.4f. Using logit-based prediction.",
                    vlm_humorous, humor_prob
                )

            result["humorous"] = logit_humorous
            result["prediction"] = "Humorous" if logit_humorous else "Not Humorous"
            result["confidence"] = round(max(humor_prob, non_humor_prob), 4)
        else:
            # Fallback when logits unavailable or timed out
            vlm_humorous = result.get("humorous")
            if not is_timed_out and vlm_humorous is not None:
                result["humorous"] = bool(vlm_humorous)
                result["prediction"] = "Humorous" if result["humorous"] else "Not Humorous"
            else:
                # Genuinely unparsed / failed / timed-out inference
                result["humorous"] = None
                result["prediction"] = "Unavailable"
            result["humor_probability"] = None
            result["non_humor_probability"] = None
            # Do NOT invent a confidence number — mark it as unavailable
            result["confidence"] = None
            result["confidence_method"] = "unavailable"

        # Set confidence method
        if result.get("confidence") is not None and result.get("confidence_method") is None:
            result["confidence_method"] = "logit_derived"

        # Cultural fields: provide meaningful defaults for general mode
        if mode == "general":
            result.setdefault("cultural_category", "Not analyzed (General mode)")
            result.setdefault("cultural_dependency", "Not analyzed (General mode)")
            result.setdefault("cultural_context", "Not analyzed (General mode)")
        else:
            # Cultural mode: ensure fields have meaningful values
            cat = result.get("cultural_category")
            if not cat or str(cat).lower() in ("none", "null", ""):
                result["cultural_category"] = "No specific cultural category detected"
            dep = result.get("cultural_dependency")
            if not dep or str(dep).lower() in ("null", ""):
                result["cultural_dependency"] = "low"
            ctx = result.get("cultural_context")
            if not ctx or str(ctx).lower() in ("none", "null", ""):
                result["cultural_context"] = "No significant cultural context detected"

        # Ensure detected_text is meaningful
        dt = result.get("detected_text")
        if not dt or str(dt).lower() in ("none", "null", ""):
            result["detected_text"] = "No text detected in image"

        return result

    @staticmethod
    def _unpack_generation_result(gen_res):
        if isinstance(gen_res, (tuple, list)):
            if len(gen_res) >= 4:
                return gen_res[0], gen_res[1], gen_res[2], gen_res[3]
            elif len(gen_res) == 3:
                return gen_res[0], gen_res[1], gen_res[2], {}
        return gen_res, None, None, {}

    def infer(self, image_path: str, mode: str = "general",
              ocr_text: str = "", retrieved_context: str = "",
              cultural_sources: list = None,
              stage1_result: dict = None,
              two_stage: bool = True,
              context_builder = None):
        """
        Runs inference on a single image.

        In Cultural-Aware mode with two_stage=True:
        Stage 1: Assesses whether humour exists based on image + OCR (no cultural injection).
                 If NON-HUMOROUS -> final result immediately (short-circuits cultural RAG).
        Stage 2: If HUMOROUS or uncertain, determines whether cultural context is actually required.
                 Only if cultural context is present/relevant does it run Stage-2 cultural reasoning.

        Args:
            image_path: Path to the meme image.
            mode: "general" or "cultural".
            ocr_text: Pre-extracted OCR text (if available).
            retrieved_context: Cultural context string.
            cultural_sources: List of sources used (e.g. ['PROJECT_JSON', 'DRISHTIKON']).
            stage1_result: Optional precomputed Stage-1 result dict.
            two_stage: Whether to use two-stage decision flow in cultural mode.
            context_builder: Optional CulturalContextBuilder for dynamic Stage 2 retrieval.

        Returns:
            dict: Structured result with genuine logit-based confidence.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        logger.info("Running %s mode inference on: %s", mode, path.name)

        if mode in ("cultural", "cultural_c") and two_stage:
            # Stage 1: Humour Assessment (Image + OCR, without cultural injection)
            if stage1_result is None:
                logger.info("[TWO-STAGE] Executing Stage-1 general humour assessment...")
                stage1_result = self.infer(
                    str(path), mode="general", ocr_text=ocr_text, two_stage=False
                )
            else:
                logger.info("[TWO-STAGE] Using provided Stage-1 assessment.")

            stage1_humorous = bool(stage1_result.get("humorous", False))

            # Dynamic retrieval from context_builder if retrieved_context not pre-supplied
            if (not retrieved_context or not retrieved_context.strip()) and context_builder is not None:
                detected_text = stage1_result.get("detected_text", "")
                query = f"{ocr_text} {detected_text}".replace("No text detected in image", "").strip()
                if query:
                    retrieved_context, dyn_sources, _ = context_builder.build_context(query)
                    cultural_sources = dyn_sources

            has_valid_cultural_context = bool(
                retrieved_context
                and retrieved_context.strip()
                and "No specific cultural reference detected" not in retrieved_context
            )

            # Detect meaningful cultural categories from OCR or detected text
            eval_text = ocr_text or stage1_result.get("detected_text", "")
            detected_cats = detect_categories(eval_text)
            meaningful_cats = [c for c in detected_cats if c and c.lower() != "none"]

            # If Stage-1 predicts NON-HUMOROUS:
            # Short-circuit only when there are NO meaningful cultural categories and valid context.
            # When meaningful cultural markers and valid context exist, allow Cultural-aware Stage 2 to evaluate.
            if not stage1_humorous:
                if not (meaningful_cats and has_valid_cultural_context):
                    logger.info("[TWO-STAGE] Stage-1 assessed as NON-HUMOROUS and no relevant cultural evidence. Short-circuiting cultural RAG.")
                    final_res = dict(stage1_result)
                    final_res["mode"] = mode
                    final_res["stage1_humorous"] = False
                    final_res["two_stage_applied"] = True
                    final_res["context_passed_to_qwen"] = False
                    final_res["cultural_sources"] = []
                    final_res["cultural_dependency"] = "low"
                    final_res["cultural_category"] = "No specific cultural category detected"
                    final_res["cultural_context"] = "Stage-1 determined content is not humorous. Cultural context not required."
                    return final_res
                logger.info(
                    "[TWO-STAGE] Stage-1 assessed as NON-HUMOROUS, but meaningful cultural markers detected (%s). Proceeding to Stage-2 cultural reasoning...",
                    ", ".join(meaningful_cats)
                )

            if not has_valid_cultural_context:
                logger.info("[TWO-STAGE] Stage-1 humorous, but no cultural context required. Using Stage-1 result.")
                final_res = dict(stage1_result)
                final_res["mode"] = mode
                final_res["stage1_humorous"] = True
                final_res["two_stage_applied"] = True
                final_res["context_passed_to_qwen"] = False
                final_res["cultural_sources"] = cultural_sources or []
                final_res["cultural_dependency"] = "low"
                return final_res

            # Cultural context is present and required: Stage-2 cultural-aware reasoning
            logger.info("[TWO-STAGE] Cultural context required. Executing Stage-2 cultural reasoning...")
            if mode == "cultural_c":
                messages = build_cultural_analysis_prompt_config_c(
                    str(path), ocr_text=ocr_text, retrieved_context=retrieved_context
                )
            else:
                messages = build_cultural_analysis_prompt(
                    str(path), ocr_text=ocr_text, retrieved_context=retrieved_context
                )
            parsed, humor_prob, non_humor_prob, diagnostics = self._unpack_generation_result(
                self._run_generation(messages)
            )
            
            # Check if Stage-2 failed (unparsed JSON / timeout without recovery)
            stage2_failed = parsed.get("error") is not None or (parsed.get("humorous") is None and humor_prob is None)
            
            if stage2_failed and stage1_result and stage1_result.get("humorous") is not None:
                logger.warning("[TWO-STAGE] Stage-2 cultural reasoning failed (%s). Falling back to valid Stage-1 assessment.", parsed.get("error"))
                final_res = dict(stage1_result)
                final_res["mode"] = mode
                final_res["stage1_humorous"] = stage1_humorous
                final_res["two_stage_applied"] = True
                final_res["stage2_failed"] = True
                final_res["stage2_error"] = parsed.get("error")
                final_res["stage2_fallback_to_stage1"] = True
                final_res["cultural_sources"] = cultural_sources or []
                if diagnostics:
                    final_res["timed_out"] = diagnostics.get("timed_out", False)
                    final_res["hit_max_new_tokens"] = diagnostics.get("hit_max_new_tokens", False)
                    final_res["generated_token_count"] = diagnostics.get("generated_token_count")
                    final_res["raw_response"] = diagnostics.get("raw_response", "")
                    final_res["inference_duration_s"] = diagnostics.get("inference_duration_s", 0)
                return final_res

            final_res = self._build_result(
                parsed, humor_prob, non_humor_prob,
                mode=mode, cultural_sources=cultural_sources,
                diagnostics=diagnostics
            )
            final_res["stage1_humorous"] = stage1_humorous
            final_res["two_stage_applied"] = True
            final_res["context_passed_to_qwen"] = True
            final_res["stage2_failed"] = stage2_failed
            return final_res

        # Standard single-stage path (General mode, or explicit two_stage=False)
        if mode == "cultural_c":
            messages = build_cultural_analysis_prompt_config_c(
                str(path), ocr_text=ocr_text, retrieved_context=retrieved_context
            )
        elif mode == "cultural":
            messages = build_cultural_analysis_prompt(
                str(path), ocr_text=ocr_text, retrieved_context=retrieved_context
            )
        else:
            messages = build_humor_analysis_prompt(str(path))

        parsed, humor_prob, non_humor_prob, diagnostics = self._unpack_generation_result(
            self._run_generation(messages)
        )
        return self._build_result(
            parsed, humor_prob, non_humor_prob,
            mode=mode, cultural_sources=cultural_sources,
            diagnostics=diagnostics
        )

    def infer_multi(self, image_paths: list, mode: str = "general",
                    ocr_text: str = "", retrieved_context: str = "",
                    cultural_sources: list = None,
                    stage1_result: dict = None,
                    two_stage: bool = True):
        """
        Runs inference on multiple images as a combined set.
        Uses Qwen2.5-VL's native multi-image support with two-stage decision flow.
        """
        resolved_paths = []
        for p in image_paths:
            rp = Path(p).resolve()
            if not rp.exists():
                raise FileNotFoundError(f"Image not found at {rp}")
            resolved_paths.append(str(rp))

        logger.info("Running %s mode multi-image inference on %d images", mode, len(resolved_paths))

        if mode == "cultural" and two_stage:
            if stage1_result is None:
                logger.info("[TWO-STAGE] Executing multi-image Stage-1 general assessment...")
                stage1_result = self.infer_multi(
                    resolved_paths, mode="general", ocr_text=ocr_text, two_stage=False
                )
            stage1_humorous = bool(stage1_result.get("humorous", False))

            has_valid_cultural_context = bool(
                retrieved_context
                and retrieved_context.strip()
                and "No specific cultural reference detected" not in retrieved_context
            )

            eval_text = ocr_text or stage1_result.get("detected_text", "")
            detected_cats = detect_categories(eval_text)
            meaningful_cats = [c for c in detected_cats if c and c.lower() != "none"]

            if not stage1_humorous:
                if not (meaningful_cats and has_valid_cultural_context):
                    logger.info("[TWO-STAGE] Multi-image Stage-1 assessed as NON-HUMOROUS and no relevant cultural evidence. Short-circuiting cultural RAG.")
                    final_res = dict(stage1_result)
                    final_res["mode"] = "cultural"
                    final_res["stage1_humorous"] = False
                    final_res["two_stage_applied"] = True
                    final_res["context_passed_to_qwen"] = False
                    final_res["cultural_sources"] = []
                    final_res["cultural_dependency"] = "low"
                    final_res["cultural_category"] = "No specific cultural category detected"
                    final_res["cultural_context"] = "Stage-1 determined content is not humorous. Cultural context not required."
                    return final_res
                logger.info(
                    "[TWO-STAGE] Multi-image Stage-1 assessed as NON-HUMOROUS, but meaningful cultural markers detected (%s). Proceeding to Stage-2 cultural reasoning...",
                    ", ".join(meaningful_cats)
                )

            if not has_valid_cultural_context:
                final_res = dict(stage1_result)
                final_res["mode"] = "cultural"
                final_res["stage1_humorous"] = True
                final_res["two_stage_applied"] = True
                final_res["context_passed_to_qwen"] = False
                final_res["cultural_sources"] = cultural_sources or []
                final_res["cultural_dependency"] = "low"
                return final_res

            messages = build_multi_image_prompt(
                resolved_paths, mode="cultural",
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )
            parsed, humor_prob, non_humor_prob, diagnostics = self._unpack_generation_result(
                self._run_generation(messages)
            )

            stage2_failed = parsed.get("error") is not None or (parsed.get("humorous") is None and humor_prob is None)

            if stage2_failed and stage1_result and stage1_result.get("humorous") is not None:
                logger.warning("[TWO-STAGE] Multi-image Stage-2 cultural reasoning failed (%s). Falling back to valid Stage-1 assessment.", parsed.get("error"))
                final_res = dict(stage1_result)
                final_res["mode"] = "cultural"
                final_res["stage1_humorous"] = stage1_humorous
                final_res["two_stage_applied"] = True
                final_res["stage2_failed"] = True
                final_res["stage2_error"] = parsed.get("error")
                final_res["stage2_fallback_to_stage1"] = True
                final_res["cultural_sources"] = cultural_sources or []
                if diagnostics:
                    final_res["timed_out"] = diagnostics.get("timed_out", False)
                    final_res["hit_max_new_tokens"] = diagnostics.get("hit_max_new_tokens", False)
                    final_res["generated_token_count"] = diagnostics.get("generated_token_count")
                    final_res["raw_response"] = diagnostics.get("raw_response", "")
                    final_res["inference_duration_s"] = diagnostics.get("inference_duration_s", 0)
                return final_res

            final_res = self._build_result(
                parsed, humor_prob, non_humor_prob,
                mode="cultural", images_count=len(resolved_paths),
                cultural_sources=cultural_sources,
                diagnostics=diagnostics
            )
            final_res["stage1_humorous"] = stage1_humorous
            final_res["two_stage_applied"] = True
            final_res["context_passed_to_qwen"] = True
            final_res["stage2_failed"] = stage2_failed
            return final_res

        if mode == "cultural":
            messages = build_multi_image_prompt(
                resolved_paths, mode="cultural",
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )
        else:
            messages = build_multi_image_prompt(
                resolved_paths, mode="general",
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )

        parsed, humor_prob, non_humor_prob, diagnostics = self._unpack_generation_result(
            self._run_generation(messages)
        )
        return self._build_result(
            parsed, humor_prob, non_humor_prob,
            mode=mode, images_count=len(resolved_paths),
            cultural_sources=cultural_sources,
            diagnostics=diagnostics
        )

    @staticmethod
    def to_evaluation_dict(result: dict, meme_id: str = "") -> dict:
        """
        Formats result into a standardized evaluation record for research benchmarking.
        """
        return {
            "meme_id": meme_id,
            "mode": result.get("mode", "general"),
            "humor_prediction": "Humorous" if result.get("humorous") else "Not Humorous",
            "model_probability": result.get("humor_probability"),
            "cultural_category": result.get("cultural_category", "N/A"),
            "cultural_dependency": result.get("cultural_dependency", "N/A"),
            "cultural_sources": result.get("cultural_sources", []),
            "retrieved_context": result.get("cultural_context", ""),
        }

