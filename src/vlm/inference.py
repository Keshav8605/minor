"""
VLM Inference Engine for humor detection.

IMPORTANT: Confidence is derived from token-level logits, NOT from VLM self-reporting.
When the VLM generates 'true' or 'false' for the 'humorous' field, we extract
the softmax probabilities of 'true' vs 'false' tokens from the model's logits.
This gives a genuine model-derived probability rather than a hallucinated number.
"""

import yaml
import json
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from qwen_vl_utils import process_vision_info
from src.vlm.model_loader import load_model_and_processor
from src.vlm.prompts import (
    build_humor_analysis_prompt,
    build_cultural_analysis_prompt,
    build_multi_image_prompt,
)
from src.vlm.output_parser import parse_json_response

logger = logging.getLogger(__name__)


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
                    break  # First occurrence is for the 'humorous' field

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

        logger.info("[ANALYSIS] VLM inference started")
        outputs = self.model.generate(**inputs, **gen_params)
        logger.info("[ANALYSIS] VLM inference completed")

        # Extract generated text
        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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

        return parsed, humor_prob, non_humor_prob

    def _build_result(self, parsed, humor_prob, non_humor_prob, mode="general", images_count=1):
        """
        Combines the parsed VLM output with logit-based probabilities
        into a structured result dictionary.
        """
        result = dict(parsed)
        result["mode"] = mode
        result["images_analyzed"] = images_count

        # Set logit-based confidence
        if humor_prob is not None and non_humor_prob is not None:
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
            result["confidence"] = round(max(humor_prob, non_humor_prob), 4)
        else:
            # Fallback: logit extraction failed, use VLM text prediction without fake confidence
            logger.warning("Logit extraction failed. Using VLM text prediction with marked confidence.")
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

    def infer(self, image_path: str, mode: str = "general",
              ocr_text: str = "", retrieved_context: str = ""):
        """
        Runs inference on a single image.

        Args:
            image_path: Path to the meme image.
            mode: "general" or "cultural".
            ocr_text: Pre-extracted OCR text (if available).
            retrieved_context: Cultural context string from CulturalRetriever.

        Returns:
            dict: Structured result with genuine logit-based confidence.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        logger.info("Running %s mode inference on: %s", mode, path.name)

        if mode == "cultural":
            messages = build_cultural_analysis_prompt(
                str(path), ocr_text=ocr_text, retrieved_context=retrieved_context
            )
        else:
            messages = build_humor_analysis_prompt(str(path))

        parsed, humor_prob, non_humor_prob = self._run_generation(messages)
        return self._build_result(parsed, humor_prob, non_humor_prob, mode=mode)

    def infer_multi(self, image_paths: list, mode: str = "general",
                    ocr_text: str = "", retrieved_context: str = ""):
        """
        Runs inference on multiple images as a combined set.
        Uses Qwen2.5-VL's native multi-image support.

        Args:
            image_paths: List of image file paths.
            mode: "general" or "cultural".
            ocr_text: Combined OCR text from all images.
            retrieved_context: Cultural context string.

        Returns:
            dict: Structured result for the combined analysis.
        """
        resolved_paths = []
        for p in image_paths:
            rp = Path(p).resolve()
            if not rp.exists():
                raise FileNotFoundError(f"Image not found at {rp}")
            resolved_paths.append(str(rp))

        logger.info("Running %s mode multi-image inference on %d images", mode, len(resolved_paths))

        messages = build_multi_image_prompt(
            resolved_paths, mode=mode,
            ocr_text=ocr_text, retrieved_context=retrieved_context
        )

        parsed, humor_prob, non_humor_prob = self._run_generation(messages)
        return self._build_result(
            parsed, humor_prob, non_humor_prob,
            mode=mode, images_count=len(resolved_paths)
        )
