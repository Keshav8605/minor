"""
Main application entry point for the Gradio-based humor detection interface.

Wires the UI to the VLM inference engine with proper cultural mode support.
Implements SHA-256 content-based caching with mode-aware keys for incremental
multi-image analysis — previously analyzed images are reused from cache.
"""

import sys
import os
import logging
import yaml

# Starlette compatibility patch: ensure HTTP_422_UNPROCESSABLE_CONTENT is defined
import starlette.status
if not hasattr(starlette.status, "HTTP_422_UNPROCESSABLE_CONTENT"):
    starlette.status.HTTP_422_UNPROCESSABLE_CONTENT = getattr(starlette.status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)

# Ensure both root and app directories are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app_dir = os.path.abspath(os.path.dirname(__file__))
for d in [root_dir, app_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    from app.formatting import parse_vlm_output, format_multi_meme_cards
except (ImportError, ModuleNotFoundError):
    from formatting import parse_vlm_output, format_multi_meme_cards

try:
    from app.state_manager import get_image_identity, get_cached_result, store_cached_result, clear_cache
except (ImportError, ModuleNotFoundError):
    from state_manager import get_image_identity, get_cached_result, store_cached_result, clear_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("app")

VLM_ENGINE = None
CULTURAL_RETRIEVER = None


def _extract_file_path(item):
    """Safely extracts a valid filesystem path string from various Gradio input types."""
    if not item:
        return None
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, tuple) and len(item) > 0:
        return _extract_file_path(item[0])
    if isinstance(item, dict):
        p = item.get("path") or item.get("name")
        if p and os.path.exists(p):
            return str(p)
        return str(p) if p else None
    if hasattr(item, "path") and item.path:
        return str(item.path)
    if hasattr(item, "name") and item.name:
        return str(item.name)
    return str(item)


def init_engine():
    """Initialize the VLM engine (singleton)."""
    global VLM_ENGINE
    if VLM_ENGINE is not None:
        return True

    try:
        from src.vlm.inference import VLMInferenceEngine
        VLM_ENGINE = VLMInferenceEngine("configs/model.yaml")
        VLM_ENGINE.load()
        logger.info("VLM Engine initialized successfully on %s", VLM_ENGINE.device)
        return True
    except Exception as e:
        logger.error("Could not initialize VLM Engine: %s", e)
        return False


def init_cultural_retriever():
    """Initialize the cultural context retriever (singleton)."""
    global CULTURAL_RETRIEVER
    if CULTURAL_RETRIEVER is not None:
        return True

    try:
        cultural_config_path = "configs/cultural.yaml"
        if os.path.exists(cultural_config_path):
            with open(cultural_config_path, "r") as f:
                cultural_config = yaml.safe_load(f)
            from src.cultural.context_retriever import CulturalRetriever
            CULTURAL_RETRIEVER = CulturalRetriever(
                cultural_config["categories_path"],
                cultural_config["knowledge_path"]
            )
            logger.info("Cultural Retriever initialized successfully")
            return True
        else:
            logger.warning("Cultural config not found at %s", cultural_config_path)
            return False
    except Exception as e:
        logger.error("Could not initialize Cultural Retriever: %s", e)
        return False


class AnalysisResponse(dict):
    """
    Response container that behaves both as a structured dictionary (with 'memes', 'is_multi', 'multi_html')
    and as an iterable 7-tuple for seamless backward compatibility with existing UI unpacking.
    """
    def __iter__(self):
        return iter(self.get("single_output", ()))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get("single_output", ())[item]
        return super().__getitem__(item)

    def __len__(self):
        return len(self.get("single_output", ()))


def _build_meme_record(result, idx, path):
    """
    Build a per-meme record from a VLM result dict.

    Extracts the standard fields needed for multi-meme display
    without the redundant 'confidence' field.
    """
    return {
        "meme_number": idx,
        "image_path": str(path),
        "humorous": result.get("humorous", False),
        "prediction": result.get("prediction", "Not Humorous"),
        "humor_probability": result.get("humor_probability", 0.5),
        "non_humor_probability": result.get("non_humor_probability", 0.5),
        "detected_text": result.get("detected_text", "No text detected in image"),
        "cultural_category": result.get("cultural_category", "Not analyzed"),
        "cultural_dependency": result.get("cultural_dependency", "Not analyzed"),
        "cultural_context": result.get("cultural_context", "Not analyzed"),
        "reasoning": result.get("reasoning", ""),
        "timing": result.get("timing", {})
    }


def _enrich_cultural_result(result, mode, retrieved_context):
    """
    Post-inference cultural enrichment: re-queries cultural retriever
    using the VLM-detected text for potentially better context.
    Applied consistently in both single and multi-meme paths.
    """
    if mode != "cultural" or CULTURAL_RETRIEVER is None:
        return

    detected = result.get("detected_text", "")
    if not detected or detected == "No text detected in image":
        return

    better_context = CULTURAL_RETRIEVER.retrieve_context(detected)
    if better_context and not retrieved_context:
        from src.cultural.category_detector import detect_categories
        categories = detect_categories(detected)
        if "none" not in categories:
            result["cultural_category"] = " / ".join(
                cat.replace("_", " ").title() for cat in categories
            )
        if result.get("cultural_context") == "No significant cultural context detected":
            result["cultural_context"] = better_context.replace(
                "EXTERNAL CULTURAL CONTEXT:\n", ""
            ).strip()


def analyze_meme(image_paths, cultural_mode, progress=None):
    """
    Main analysis function called by the Gradio UI and evaluation pipelines.
    For single images: runs standard single-image inference pipeline.
    For multiple images: runs independent per-image processing in exact uploaded order
    designed to prevent cross-meme data mixing.

    Uses content-based SHA-256 caching with mode-aware keys to avoid redundant
    VLM inference for previously analyzed images.

    Args:
        image_paths: Single image path (str) or list of image paths from Gallery/File.
        cultural_mode: "General" or "Cultural-Aware".
        progress: Optional Gradio progress instance (gr.Progress).

    Returns:
        AnalysisResponse: Dictionary supporting both 7-tuple unpacking and .get("memes").
    """
    # Normalize input
    if image_paths is None:
        err_tuple = ("Unavailable", "N/A", "No image uploaded", "No specific cultural category detected", "Not analyzed", "Not analyzed", "Please upload at least one image.")
        return AnalysisResponse({
            "is_multi": False, "memes_count": 0, "memes": [],
            "single_output": err_tuple, "multi_html": ""
        })

    logger.info("[ANALYSIS] Started")
    logger.info("[ANALYSIS] Image received: type=%s, val=%s", type(image_paths), str(image_paths)[:200])

    # Handle single item vs list of items
    if isinstance(image_paths, (list, tuple)):
        raw_items = list(image_paths)
    else:
        raw_items = [image_paths]

    paths = []
    for item in raw_items:
        p = _extract_file_path(item)
        if p and os.path.exists(p):
            paths.append(p)

    if not paths:
        logger.warning("[ANALYSIS] No valid image paths found from input: %s", image_paths)
        err_tuple = ("Unavailable", "N/A", "No valid image path found", "No specific cultural category detected", "Not analyzed", "Not analyzed", "Please upload a valid image file.")
        return AnalysisResponse({
            "is_multi": False, "memes_count": 0, "memes": [],
            "single_output": err_tuple, "multi_html": ""
        })

    logger.info("[ANALYSIS] Image path resolved (count=%d): %s", len(paths), paths)
    logger.info("[ANALYSIS] Mode = %s", cultural_mode)

    if not init_engine():
        logger.error("[ANALYSIS] Failed to initialize VLM engine")
        err_tuple = ("Error", "N/A", "N/A", "N/A", "N/A", "N/A", "Model is currently unavailable or missing dependencies locally.")
        return AnalysisResponse({
            "is_multi": False, "memes_count": 0, "memes": [],
            "single_output": err_tuple, "multi_html": ""
        })

    try:
        from src.cultural.ocr_engine import extract_text_ocr
        mode = "cultural" if cultural_mode == "Cultural-Aware" else "general"

        if mode == "cultural":
            init_cultural_retriever()

        # ── SINGLE MEME ANALYSIS (len == 1) ──
        if len(paths) == 1:
            p = paths[0]
            logger.info("[ANALYSIS] Single image path: %s", p)

            # Check cache first (mode-aware)
            img_identity = get_image_identity(p)
            cached = get_cached_result(img_identity, mode)

            if cached is not None:
                logger.info("[ANALYSIS] Reusing cached single-image result")
                single_output = parse_vlm_output(cached, include_context=True)
                meme_record = _build_meme_record(cached, idx=1, path=p)
                return AnalysisResponse({
                    "is_multi": False,
                    "memes_count": 1,
                    "memes": [meme_record],
                    "single_output": single_output,
                    "multi_html": "",
                    "result": cached
                })

            # Cache MISS — full pipeline
            # Step 1: Dedicated OCR extraction
            ocr_text, _ = extract_text_ocr(p)
            if ocr_text:
                logger.info("[ANALYSIS] Dedicated OCR extracted: %s", ocr_text[:120])

            # Step 2: Cultural Knowledge Retrieval
            retrieved_context = ""
            if mode == "cultural" and CULTURAL_RETRIEVER is not None:
                search_text = ocr_text if ocr_text else os.path.basename(p)
                retrieved_context = CULTURAL_RETRIEVER.retrieve_context(search_text)

            # Step 3: VLM inference
            result = VLM_ENGINE.infer(
                p, mode=mode,
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )

            # Step 4: Cultural post-inference enrichment
            _enrich_cultural_result(result, mode, retrieved_context)

            # Log reasoning status
            reasoning_val = result.get("reasoning", "")
            if not reasoning_val or reasoning_val == "No reasoning provided.":
                logger.warning("[REASONING] Missing reasoning for single image=%s...", img_identity[:12])

            # Store complete result in cache
            store_cached_result(img_identity, mode, result)

            single_output = parse_vlm_output(result, include_context=True)
            meme_record = _build_meme_record(result, idx=1, path=p)

            return AnalysisResponse({
                "is_multi": False,
                "memes_count": 1,
                "memes": [meme_record],
                "single_output": single_output,
                "multi_html": "",
                "result": result
            })

        # ── MULTI-MEME ANALYSIS (len > 1): INDEPENDENT PER-IMAGE PROCESSING ──
        total_memes = len(paths)
        logger.info("[ANALYSIS] Independent per-image multi-meme analysis started for %d memes", total_memes)
        memes_list = []
        cache_hits = 0

        for idx, p in enumerate(paths, start=1):
            # Compute content-based identity
            img_identity = get_image_identity(p)
            cached = get_cached_result(img_identity, mode)

            if cached is not None:
                # Cache HIT — reuse result with current ordering (deep copy already done)
                cache_hits += 1
                meme_record = _build_meme_record(cached, idx=idx, path=p)
                memes_list.append(meme_record)

                if progress is not None:
                    progress(float(idx) / float(total_memes),
                             desc=f"Reusing cached result for Meme {idx} of {total_memes}...")
                logger.info("[ANALYSIS] Reusing cached result for Meme %d of %d", idx, total_memes)
                continue

            # Cache MISS — full analysis pipeline for this image
            if progress is not None:
                progress(float(idx - 1) / float(total_memes),
                         desc=f"Analyzing Meme {idx} of {total_memes}...")

            logger.info("[MULTI-MEME] Processing Meme %d of %d: %s", idx, total_memes, os.path.basename(p))

            # Isolated Step A: Dedicated OCR for this meme ONLY
            iso_ocr, _ = extract_text_ocr(p)
            if iso_ocr:
                logger.info("[MULTI-MEME] Meme %d OCR: %s", idx, iso_ocr[:100])

            # Isolated Step B: Cultural retrieval for this meme ONLY
            iso_ctx = ""
            if mode == "cultural" and CULTURAL_RETRIEVER is not None:
                search_text = iso_ocr if iso_ocr else os.path.basename(p)
                iso_ctx = CULTURAL_RETRIEVER.retrieve_context(search_text)
                if iso_ctx:
                    logger.info("[MULTI-MEME] Meme %d Cultural Context retrieved: %s", idx, iso_ctx[:120])

            # Isolated Step C: VLM inference strictly on this meme
            res_single = VLM_ENGINE.infer(
                p, mode=mode,
                ocr_text=iso_ocr, retrieved_context=iso_ctx
            )

            # Isolated Step D: Cultural post-inference enrichment for this meme
            _enrich_cultural_result(res_single, mode, iso_ctx)

            # Log reasoning status
            reasoning_val = res_single.get("reasoning", "")
            if not reasoning_val or reasoning_val == "No reasoning provided.":
                logger.warning("[REASONING] Missing reasoning for Meme %d image=%s...", idx, img_identity[:12])

            # Store complete result in cache
            store_cached_result(img_identity, mode, res_single)

            # Build per-meme schema (no redundant 'confidence' field)
            meme_record = _build_meme_record(res_single, idx=idx, path=p)
            memes_list.append(meme_record)
            logger.info("[MULTI-MEME] Completed Meme %d: %s", idx, meme_record["prediction"])

        if progress is not None:
            progress(1.0, desc=f"Completed analysis of {total_memes} memes.")

        logger.info("[ANALYSIS] Multi-meme complete: %d total, %d cached, %d newly analyzed",
                    total_memes, cache_hits, total_memes - cache_hits)

        multi_result = {
            "is_multi": True,
            "memes_count": len(memes_list),
            "memes": memes_list
        }
        multi_html = format_multi_meme_cards(multi_result, include_context=True)
        single_output_fallback = parse_vlm_output(memes_list[0], include_context=True)

        return AnalysisResponse({
            "is_multi": True,
            "memes_count": len(memes_list),
            "memes": memes_list,
            "single_output": single_output_fallback,
            "multi_html": multi_html,
            "result": multi_result
        })

    except Exception as e:
        logger.error("[ANALYSIS] Exception type: %s", type(e).__name__)
        logger.error("[ANALYSIS] Exception message: %s", str(e))
        logger.exception("[ANALYSIS] Complete traceback:")
        err_tuple = ("Error", "N/A", "N/A", "N/A", "N/A", "N/A", f"An internal inference error occurred: {str(e)}")
        return AnalysisResponse({
            "is_multi": False, "memes_count": 0, "memes": [],
            "single_output": err_tuple,
            "multi_html": f"<div style='color:#EF4444;padding:16px;'>Error: {str(e)}</div>",
            "error": True
        })



def main():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please run: uv pip install gradio")
        sys.exit(0)

    # Pre-initialize engine and retriever if pre-warming is enabled
    if os.environ.get("PREWARM", "1") == "1":
        logger.info("Pre-warming VLM Engine and Cultural Retriever...")
        init_engine()
        init_cultural_retriever()
    else:
        logger.info("Pre-warming skipped via PREWARM=0.")

    try:
        from app.ui_components import create_ui, CUSTOM_CSS, theme
    except ImportError:
        from ui_components import create_ui, CUSTOM_CSS, theme

    demo = create_ui(analyze_meme)
    # Enable Gradio queue with concurrency limit to prevent request timeouts and OOM
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, css=CUSTOM_CSS, theme=theme)


if __name__ == "__main__":
    print("Gradio app entrypoint initialized.")
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        sys.exit(0)
    else:
        main()
