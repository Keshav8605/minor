"""
Main application entry point for the Gradio-based humor detection interface.

Wires the UI to the VLM inference engine with proper cultural mode support.
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
    from app.formatting import parse_vlm_output
except ImportError:
    from formatting import parse_vlm_output

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


def analyze_meme(image_paths, cultural_mode):
    """
    Main analysis function called by the Gradio UI.

    Args:
        image_paths: Single image path (str) or list of image paths from Gallery/File.
        cultural_mode: "General" or "Cultural-Aware".

    Returns:
        Tuple of UI output values.
    """
    # Normalize input
    if image_paths is None:
        return "Error", "N/A", "N/A", "N/A", "N/A", "Please upload at least one image."

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
        return "Error", "N/A", "N/A", "N/A", "N/A", "Please upload a valid image."

    logger.info("[ANALYSIS] Image path resolved: %s", paths)
    logger.info("[ANALYSIS] Mode = %s", cultural_mode)

    if not init_engine():
        logger.error("[ANALYSIS] Failed to initialize VLM engine")
        return "Error", "N/A", "N/A", "N/A", "N/A", "Model is currently unavailable or missing dependencies locally."

    try:
        mode = "cultural" if cultural_mode == "Cultural-Aware" else "general"
        ocr_text = ""
        retrieved_context = ""

        logger.info("[ANALYSIS] OCR started")
        # Text reading is handled by multimodal VLM vision understanding
        logger.info("[ANALYSIS] OCR completed (integrated vision-language text detection)")

        # For Cultural-Aware mode, use the cultural retriever
        if mode == "cultural":
            init_cultural_retriever()
            if CULTURAL_RETRIEVER is not None:
                file_context = " ".join([os.path.basename(p) for p in paths])
                retrieved_context = CULTURAL_RETRIEVER.retrieve_context(file_context)
                logger.info("[ANALYSIS] Cultural context retrieved: %s",
                           retrieved_context[:200] if retrieved_context else "none")

        logger.info("[ANALYSIS] Prompt construction started")
        # Prompt is constructed inside engine.infer()
        logger.info("[ANALYSIS] Prompt construction completed")

        # Run inference
        if len(paths) == 1:
            result = VLM_ENGINE.infer(
                paths[0], mode=mode,
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )
        else:
            result = VLM_ENGINE.infer_multi(
                paths, mode=mode,
                ocr_text=ocr_text, retrieved_context=retrieved_context
            )

        # For Cultural-Aware mode, if we got detected_text from VLM,
        # do a second cultural retrieval pass with actual OCR text
        if mode == "cultural" and CULTURAL_RETRIEVER is not None:
            detected = result.get("detected_text", "")
            if detected and detected != "No text detected in image":
                better_context = CULTURAL_RETRIEVER.retrieve_context(detected)
                if better_context and not retrieved_context:
                    logger.info("[ANALYSIS] Additional cultural context from OCR: %s",
                               better_context[:200])
                    from src.cultural.category_detector import detect_categories
                    categories = detect_categories(detected)
                    if "none" not in categories:
                        current_cat = result.get("cultural_category", "")
                        if not current_cat or "not" in str(current_cat).lower() or "none" in str(current_cat).lower():
                            result["cultural_category"] = " / ".join(
                                cat.replace("_", " ").title() for cat in categories
                            )
                            logger.info("[ANALYSIS] Updated cultural category from OCR keywords: %s",
                                       result["cultural_category"])

        logger.info("[ANALYSIS] Formatting started")
        ui_output = parse_vlm_output(result)
        logger.info("[ANALYSIS] Formatting completed: prediction=%s, confidence=%s",
                    result.get("humorous"), result.get("confidence"))

        return ui_output

    except Exception as e:
        logger.error("[ANALYSIS] Exception type: %s", type(e).__name__)
        logger.error("[ANALYSIS] Exception message: %s", str(e))
        logger.exception("[ANALYSIS] Complete traceback:")
        return "Error", "N/A", "N/A", "N/A", "N/A", f"An internal inference error occurred: {str(e)}"


def main():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please run: uv pip install gradio")
        sys.exit(0)

    # Pre-initialize engine and retriever so the first request doesn't suffer cold-start delay
    logger.info("Pre-warming VLM Engine and Cultural Retriever...")
    init_engine()
    init_cultural_retriever()

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
