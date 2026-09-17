import os
import sys
import base64
import logging

try:
    import gradio as gr
except ImportError:
    gr = None

logger = logging.getLogger("ui_components")

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ──────────────────────────────────────────────
   BASE CONTAINER AND RESET
   ────────────────────────────────────────────── */
*, *::before, *::after {
    box-sizing: border-box !important;
}

body, html {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #0B1020 !important;
    overflow-x: hidden !important;
}

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B1020 !important;
    color: #F4F6FA !important;
    width: 100% !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
    border: none !important;
    box-shadow: none !important;
    overflow-x: hidden !important;
}

/* Hide Gradio footer */
footer {
    display: none !important;
}

/* ──────────────────────────────────────────────
   HEADER SECTION
   ────────────────────────────────────────────── */
.header-section {
    margin-bottom: 16px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #222B42 !important;
}

.header-section h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #F4F6FA !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.02em !important;
}

.header-section p {
    color: #AAB4C5 !important;
    font-size: 13px !important;
    margin: 0 !important;
}

/* ──────────────────────────────────────────────
   MAIN GRID LAYOUT - RESPONSIVE
   ────────────────────────────────────────────── */
.main-grid {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 16px !important;
    width: 100% !important;
}

/* Left Input Card / Right Output Card panel styling */
.panel-card {
    background-color: #12182A !important;
    border: 1px solid #222B42 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}

/* Default Desktop Column split */
.input-panel {
    flex: 0 0 calc(42% - 8px) !important;
    min-width: 320px !important;
}

.results-panel {
    flex: 0 0 calc(58% - 8px) !important;
    min-width: 360px !important;
}

/* ──────────────────────────────────────────────
   TYPOGRAPHY & HEADINGS
   ────────────────────────────────────────────── */
.panel-title {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #F4F6FA !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 14px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    border-bottom: 1px solid #222B42 !important;
    padding-bottom: 6px !important;
}

.results-subtitle {
    font-size: 13px !important;
    color: #AAB4C5 !important;
    margin-top: -6px !important;
    margin-bottom: 12px !important;
}

/* ──────────────────────────────────────────────
   INNER RESULTS CARDS
   ────────────────────────────────────────────── */
.result-card {
    background-color: #171E31 !important;
    border: 1px solid #222B42 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
    height: auto !important;
    transition: border-color 0.2s ease !important;
}

.result-card:hover {
    border-color: #38BDF8 !important;
}

.card-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #AAB4C5 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
}

/* Textarea / Input styling within result cards */
.result-card textarea {
    background-color: transparent !important;
    border: none !important;
    color: #F4F6FA !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.45 !important;
    padding: 0 !important;
    resize: none !important;
    width: 100% !important;
}

.result-card textarea:focus {
    box-shadow: none !important;
}

/* Specific styling for Humor prediction text */
.prediction-box textarea {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #22C55E !important;
}

/* ──────────────────────────────────────────────
   IMAGE UPLOAD COMPONENT (SINGLE & MULTI)
   ────────────────────────────────────────────── */
.meme-dropzone {
    border: 1px dashed #29324A !important;
    background-color: #171E31 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 14px !important;
    transition: border-color 0.25s ease !important;
}

.meme-dropzone:hover {
    border-color: #8B5CF6 !important;
}

/* Single Meme Dropzone & Aspect-Ratio-Preserving Image Preview */
.single-meme-dropzone {
    min-height: 220px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

.single-meme-dropzone .image-container,
.single-meme-dropzone .image-frame,
.single-meme-dropzone [data-testid="image"],
.single-meme-dropzone .wrap {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-height: 200px !important;
    max-height: 320px !important;
    background: transparent !important;
}

.single-meme-dropzone img {
    max-width: 100% !important;
    max-height: 280px !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 6px !important;
    display: block !important;
    margin: 0 auto !important;
}

/* ──────────────────────────────────────────────
   UNIFIED MULTI-MEME GALLERY & PREVIEW CARDS
   ────────────────────────────────────────────── */
.multi-preview-section {
    background: #171E31 !important;
    border: 1px solid #222B42 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-top: 4px !important;
    margin-bottom: 14px !important;
}

.multi-preview-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #222B42 !important;
}

.multi-preview-title {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #F4F6FA !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.multi-order-tag {
    font-size: 11px !important;
    font-weight: 400 !important;
    color: #8E9BAE !important;
}

.multi-preview-count {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #8B5CF6 !important;
    background: rgba(139, 92, 246, 0.12) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
}

.multi-preview-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 12px !important;
    width: 100% !important;
}

@media (max-width: 900px) {
    .multi-preview-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}

@media (max-width: 500px) {
    .multi-preview-grid {
        grid-template-columns: 1fr !important;
    }
}

.multi-image-card {
    background: #12182A !important;
    border: 1px solid #222B42 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    transition: all 0.2s ease !important;
}

.multi-image-card:hover {
    border-color: #38BDF8 !important;
    box-shadow: 0 2px 8px rgba(56, 189, 248, 0.15) !important;
}

.multi-card-badge-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 6px 8px !important;
    background: #171E31 !important;
    border-bottom: 1px solid #222B42 !important;
}

.multi-card-badge {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: #FFFFFF !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

.multi-card-del-btn {
    background: transparent !important;
    border: none !important;
    color: #AAB4C5 !important;
    font-size: 16px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    padding: 0 4px !important;
    border-radius: 4px !important;
    transition: all 0.15s ease !important;
}

.multi-card-del-btn:hover {
    color: #EF4444 !important;
    background: rgba(239, 68, 68, 0.15) !important;
}

.multi-card-thumb-wrap {
    width: 100% !important;
    height: 110px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #0B1020 !important;
    overflow: hidden !important;
    padding: 4px !important;
}

.multi-card-thumb {
    max-width: 100% !important;
    max-height: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 4px !important;
    display: block !important;
}

.multi-card-info-box {
    padding: 6px 8px !important;
    border-top: 1px solid #222B42 !important;
    background: #12182A !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
}

.multi-card-filename {
    font-size: 11px !important;
    color: #E2E8F0 !important;
    font-weight: 500 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    width: 100% !important;
}

.multi-card-meta-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

.multi-card-filesize {
    font-size: 10px !important;
    color: #8E9BAE !important;
}

/* Hide duplicate redundant internal file list inside the Gradio File component */
#multi-file-uploader .file-preview-holder,
#multi-file-uploader ul,
#multi-file-uploader table {
    display: none !important;
}

/* ──────────────────────────────────────────────
   SEGMENTED CONTROL (Cultural Toggle)
   ────────────────────────────────────────────── */
.segmented-control {
    background-color: #171E31 !important;
    border: 1px solid #222B42 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    margin-bottom: 12px !important;
}

.segmented-control label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 12px !important;
    color: #AAB4C5 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}

.segmented-control label:has(input[type="radio"]:checked) {
    background-color: rgba(139, 92, 246, 0.2) !important;
    color: #F4F6FA !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
}

.mode-explanation {
    font-size: 12px !important;
    color: #8E9BAE !important;
    line-height: 1.4 !important;
    margin-bottom: 14px !important;
}

.mode-explanation strong {
    color: #D1D5DB !important;
    font-weight: 600 !important;
}

/* ──────────────────────────────────────────────
   ANALYZE BUTTON
   ────────────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    height: 46px !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 6px !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25) !important;
}

#analyze-btn:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4) !important;
}

#analyze-btn:active {
    transform: translateY(1px) !important;
}

/* ──────────────────────────────────────────────
   GRID ROWS AND EQUAL WIDTHS
   ────────────────────────────────────────────── */
.sub-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
    margin-bottom: 10px !important;
}

.sub-row > * {
    flex: 1 !important;
    min-width: 0 !important;
    margin-bottom: 0 !important;
}

/* Multi-meme responsive styling */
.multi-cards-container {
    width: 100% !important;
}
.multi-meme-section {
    background: rgba(17, 22, 37, 0.6) !important;
    border: 1px solid #1E2640 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* ──────────────────────────────────────────────
   PROCESSING / LOADING STATE
   ────────────────────────────────────────────── */
.pending {
    background-color: rgba(18, 24, 42, 0.95) !important;
    border-radius: 10px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 20px !important;
    gap: 10px !important;
}

.pending .eta-bar, .pending .loading {
    display: none !important;
}

.pending::after {
    content: "Processing meme analysis..." !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #AAB4C5 !important;
    display: block !important;
}

/* ──────────────────────────────────────────────
   RESPONSIVENESS (Media Queries)
   ────────────────────────────────────────────── */
@media (max-width: 1080px) {
    .main-grid {
        flex-direction: column !important;
    }
    .input-panel, .results-panel {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 12px 10px !important;
    }
    .panel-card {
        padding: 14px !important;
    }
    .header-section h1 {
        font-size: 22px !important;
    }
    .sub-row {
        flex-direction: column !important;
        gap: 8px !important;
    }
    .sub-row > * {
        margin-bottom: 0 !important;
    }
    .multi-meme-section {
        padding: 12px 10px !important;
    }
    .multi-meme-section [style*="grid-template-columns"] {
        grid-template-columns: 1fr !important;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
)


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
        return str(p) if p else None
    if hasattr(item, "path") and item.path:
        return str(item.path)
    if hasattr(item, "name") and item.name:
        return str(item.name)
    return str(item)


def _format_file_size(size_bytes):
    """Formats bytes into human-readable string (e.g. 87.2 KB)."""
    if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
        return ""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_image_data_uri(path):
    """Reads local image and converts to base64 data URI for reliable browser rendering."""
    if not path or not os.path.exists(path):
        return ""
    try:
        ext = os.path.splitext(path)[1].lower().replace(".", "")
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp"
        }
        mime = mime_map.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64_data}"
    except Exception as e:
        logger.warning("Failed to encode image data URI: %s", e)
        return ""


def _render_multi_meme_cards_html(files):
    """
    Renders unified Multi-Meme preview gallery HTML.
    Merges Meme Number Badge + Actual Image + Filename + File Size + Remove Button
    into a single unified card component per image.
    """
    if not files or len(files) == 0:
        return ""

    cards_html = []
    total_count = len(files)

    for i, f in enumerate(files, 1):
        path = _extract_file_path(f)
        fname = os.path.basename(path) if path else f"Meme {i}.png"
        
        file_size_str = ""
        data_uri = ""
        if path and os.path.exists(path):
            try:
                sz = os.path.getsize(path)
                file_size_str = _format_file_size(sz)
            except Exception:
                file_size_str = ""
            data_uri = _get_image_data_uri(path)

        card_html = f"""
        <div class="multi-image-card" data-index="{i-1}">
            <div class="multi-card-badge-row">
                <span class="multi-card-badge">MEME {i}</span>
                <button type="button" class="multi-card-del-btn" onclick="removeMultiMemeFile({i-1})" title="Remove Meme {i}">×</button>
            </div>
            <div class="multi-card-thumb-wrap">
                <img src="{data_uri}" alt="Meme {i}" class="multi-card-thumb" />
            </div>
            <div class="multi-card-info-box">
                <div class="multi-card-filename" title="{fname}">{fname}</div>
                <div class="multi-card-meta-row">
                    <span class="multi-card-filesize">{file_size_str}</span>
                </div>
            </div>
        </div>
        """
        cards_html.append(card_html)

    count_label = f"{total_count} {'meme' if total_count == 1 else 'memes'}"

    js_helper = """
    <script>
    if (!window.removeMultiMemeFile) {
        window.removeMultiMemeFile = function(idx) {
            var container = document.getElementById('multi-file-uploader');
            if (!container) return;
            var delBtns = container.querySelectorAll('button[aria-label="Clear"], button[aria-label="Delete"], button[title="Delete"], button[title="Clear"], .file-row button, .file-item button, [data-testid="clear-button"], button.delete');
            if (delBtns && delBtns[idx]) {
                delBtns[idx].click();
                return;
            }
            var allBtns = container.querySelectorAll('button');
            if (allBtns && allBtns[idx]) {
                allBtns[idx].click();
            }
        };
    }
    </script>
    """

    return f"""
    <div class="multi-preview-section">
        <div class="multi-preview-header">
            <div class="multi-preview-title">
                <span>🖼️ Uploaded Memes</span>
                <span class="multi-order-tag">(Preserved Input Order)</span>
            </div>
            <span class="multi-preview-count">{count_label}</span>
        </div>
        <div class="multi-preview-grid">
            {''.join(cards_html)}
        </div>
    </div>
    {js_helper}
    """


def create_ui(analyze_fn):
    if gr is None:
        print("Gradio is not installed. Cannot create UI.")
        sys.exit(1)

    with gr.Blocks(title="Culturally Aware Multimodal Humor Detection") as demo:

        # ── HEADER ──
        with gr.Column(elem_classes=["header-section"]):
            gr.Markdown("# 🎭 Culturally Aware Multimodal Humor Detection")
            gr.Markdown("Vision-Language Humor Analysis for Hindi & Hinglish Memes (Qwen2.5-VL)")

        # ── MAIN LAYOUT GRID ──
        with gr.Row(elem_classes=["main-grid"]):

            # ── LEFT COLUMN: INPUT PANEL (42%) ──
            with gr.Column(scale=42, elem_classes=["panel-card", "input-panel"]):
                gr.Markdown("📥 MEME INPUT", elem_classes=["panel-title"])

                # Tab state: tracks which tab is currently selected (source of truth)
                current_tab = gr.State(value="single")

                with gr.Tabs():
                    with gr.TabItem("🖼️ Single Meme") as single_tab:
                        single_image = gr.Image(
                            type="filepath",
                            label="Upload Meme Image",
                            sources=["upload", "clipboard"],
                            elem_classes=["single-meme-dropzone", "meme-dropzone"],
                            height=260
                        )
                    with gr.TabItem("📚 Multi-Meme / Strip") as multi_tab:
                        multi_images = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Upload Multiple Meme Images",
                            elem_id="multi-file-uploader",
                            elem_classes=["meme-dropzone"]
                        )
                        # Unified Multi-Meme Preview Component (replaces duplicate labels + separate gallery)
                        multi_preview_gallery = gr.HTML(value="", visible=False)

                # Track tab selection — current_tab is the source of truth
                single_tab.select(fn=lambda: "single", inputs=[], outputs=[current_tab])
                multi_tab.select(fn=lambda: "multi", inputs=[], outputs=[current_tab])

                gr.Markdown("🌐 ANALYSIS MODE", elem_classes=["panel-title"])

                cultural_toggle = gr.Radio(
                    choices=["General", "Cultural-Aware"],
                    value="Cultural-Aware",
                    show_label=False,
                    elem_classes=["segmented-control"]
                )

                gr.Markdown(
                    "**General Mode**: Standard VLM visual & linguistic reasoning without external cultural knowledge injection.\n\n"
                    "**Cultural-Aware Mode**: Detects cultural entities, retrieves Indian cultural knowledge (family, exams, cricket, cinema, slang), and conditions VLM reasoning.",
                    elem_classes=["mode-explanation"]
                )

                analyze_btn = gr.Button("✨ Analyze Meme", elem_id="analyze-btn")

            # ── RIGHT COLUMN: RESULTS PANEL (58%) ──
            with gr.Column(scale=58, elem_classes=["panel-card", "results-panel"]):
                gr.Markdown("📊 ANALYSIS RESULTS", elem_classes=["panel-title"])
                gr.Markdown("Multimodal inference insights & structured cultural reasoning", elem_classes=["results-subtitle"])

                # View A: Single-Meme Results (Clean 7 Cards)
                with gr.Group(visible=True) as single_results_group:
                    # Row 1: Humor Prediction + Model Probability
                    with gr.Row(elem_classes=["sub-row"]):
                        with gr.Column(elem_classes=["result-card"]):
                            gr.Markdown("🎯 HUMOR PREDICTION", elem_classes=["card-label"])
                            pred_out = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                elem_classes=["prediction-box"]
                            )
                        with gr.Column(elem_classes=["result-card"]):
                            gr.Markdown("◉ MODEL PROBABILITY", elem_classes=["card-label"])
                            conf_out = gr.HTML(elem_classes=["confidence-display"])

                    # Row 2: Detected Text (OCR)
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("📄 DETECTED TEXT (OCR)", elem_classes=["card-label"])
                        text_out = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=2
                        )

                    # Row 3: Cultural Category + Cultural Dependency
                    with gr.Row(elem_classes=["sub-row"]):
                        with gr.Column(elem_classes=["result-card"]):
                            gr.Markdown("🏷️ CULTURAL CATEGORY", elem_classes=["card-label"])
                            cat_out = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                lines=1
                            )
                        with gr.Column(elem_classes=["result-card"]):
                            gr.Markdown("🔗 CULTURAL DEPENDENCY", elem_classes=["card-label"])
                            dep_out = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                lines=1
                            )

                    # Row 4: Cultural Context (Dedicated Card)
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["card-label"])
                        ctx_out = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=2
                        )

                    # Row 5: AI Reasoning (Reasoning only, no raw JSON)
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("🧠 AI REASONING", elem_classes=["card-label"])
                        reason_out = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=3
                        )

                # View B: Multi-Meme Results (Independent Per-Image Cards)
                with gr.Group(visible=False) as multi_results_group:
                    multi_cards_display = gr.HTML(elem_classes=["multi-cards-container"])

        # ── UNIFIED MULTI-MEME PREVIEW UPDATE ──
        def _update_multi_cards(files):
            """Update unified multi-meme cards when files change in multi-meme upload."""
            if not files or len(files) == 0:
                return gr.update(value="", visible=False)

            html = _render_multi_meme_cards_html(files)
            return gr.update(value=html, visible=bool(html))

        multi_images.change(
            fn=_update_multi_cards,
            inputs=[multi_images],
            outputs=[multi_preview_gallery]
        )

        # ── CLICK ACTION BINDING (TAB-AWARE DISPATCH WITH RESULT CLEARING) ──

        def _clear_results():
            """Clear all visible results before starting a new analysis.
            This prevents old results from remaining visible under new loading state."""
            return (
                gr.update(visible=True),   # Show single results (default view during loading)
                gr.update(visible=False),  # Hide multi results
                "",                         # Clear multi HTML
                "⏳ Analyzing...",          # pred_out — loading indicator
                "",                         # conf_out
                "",                         # text_out
                "",                         # cat_out
                "",                         # dep_out
                "",                         # ctx_out
                ""                          # reason_out
            )

        def _dispatch_analysis(single_path, multi_paths, mode, tab_state, progress=gr.Progress(track_tqdm=False)):
            """
            Tab-aware dispatch: uses current_tab as the source of truth
            for whether to run single-image or multi-image analysis.
            Current uploaded images determine the image count — never previous state.
            """
            is_single_tab = (tab_state == "single")

            if is_single_tab:
                # ── SINGLE MEME MODE ──
                # Use ONLY the current single image; ignore any multi-image state
                if not single_path:
                    return (
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "",
                        "Unavailable",
                        "<span style='color:#EF4444;font-size:14px;'>No image uploaded</span>",
                        "No image provided",
                        "No specific cultural category detected",
                        "Not analyzed",
                        "Not analyzed",
                        "Please upload a meme image in the 'Single Meme' tab before clicking Analyze."
                    )

                resp = analyze_fn(single_path, mode, progress=progress)
                p0 = resp[0] if len(resp) > 0 else "Not Humorous"
                p1 = resp[1] if len(resp) > 1 else ""
                p2 = resp[2] if len(resp) > 2 else ""
                p3 = resp[3] if len(resp) > 3 else ""
                p4 = resp[4] if len(resp) > 4 else ""
                p5 = resp[5] if len(resp) > 5 else ""
                p6 = resp[6] if len(resp) > 6 else ""
                return (
                    gr.update(visible=True),    # Show single view
                    gr.update(visible=False),   # Hide multi view — no stale multi cards
                    "",                          # Clear multi HTML
                    p0, p1, p2, p3, p4, p5, p6
                )

            else:
                # ── MULTI-MEME MODE ──
                # Use ONLY the CURRENT multi-image file list
                has_multi = multi_paths and isinstance(multi_paths, list) and len(multi_paths) > 0

                if not has_multi:
                    return (
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "",
                        "Unavailable",
                        "<span style='color:#EF4444;font-size:14px;'>No images uploaded</span>",
                        "No images provided",
                        "No specific cultural category detected",
                        "Not analyzed",
                        "Not analyzed",
                        "Please upload meme images in the 'Multi-Meme / Strip' tab before clicking Analyze."
                    )

                current_count = len(multi_paths)

                if current_count == 1:
                    # Single image in multi tab — use single-image pipeline, show single view
                    resp = analyze_fn(multi_paths[0], mode, progress=progress)
                    p0 = resp[0] if len(resp) > 0 else "Not Humorous"
                    p1 = resp[1] if len(resp) > 1 else ""
                    p2 = resp[2] if len(resp) > 2 else ""
                    p3 = resp[3] if len(resp) > 3 else ""
                    p4 = resp[4] if len(resp) > 4 else ""
                    p5 = resp[5] if len(resp) > 5 else ""
                    p6 = resp[6] if len(resp) > 6 else ""
                    return (
                        gr.update(visible=True),    # Show single view
                        gr.update(visible=False),   # Hide multi view
                        "",
                        p0, p1, p2, p3, p4, p5, p6
                    )

                # Multiple images — use multi-image pipeline
                resp = analyze_fn(multi_paths, mode, progress=progress)
                multi_html = resp.get("multi_html", "") if isinstance(resp, dict) else ""
                p0 = resp[0] if len(resp) > 0 else "Not Humorous"
                p1 = resp[1] if len(resp) > 1 else ""
                p2 = resp[2] if len(resp) > 2 else ""
                p3 = resp[3] if len(resp) > 3 else ""
                p4 = resp[4] if len(resp) > 4 else ""
                p5 = resp[5] if len(resp) > 5 else ""
                p6 = resp[6] if len(resp) > 6 else ""
                return (
                    gr.update(visible=False),   # Hide single view
                    gr.update(visible=True),    # Show multi view
                    multi_html,
                    p0, p1, p2, p3, p4, p5, p6
                )

        # Chain: clear old results → run analysis
        # The first callback clears visible UI immediately, then the analysis fills in results
        analyze_btn.click(
            fn=_clear_results,
            inputs=[],
            outputs=[
                single_results_group,
                multi_results_group,
                multi_cards_display,
                pred_out, conf_out, text_out, cat_out, dep_out, ctx_out, reason_out
            ]
        ).then(
            fn=_dispatch_analysis,
            inputs=[single_image, multi_images, cultural_toggle, current_tab],
            outputs=[
                single_results_group,
                multi_results_group,
                multi_cards_display,
                pred_out, conf_out, text_out, cat_out, dep_out, ctx_out, reason_out
            ]
        )

    return demo

