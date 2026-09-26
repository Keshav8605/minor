import sys
import os
import html as html_mod

try:
    import gradio as gr
except ImportError:
    gr = None

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B1020 !important;
    color: #F4F6FA !important;
    width: calc(100% - 64px) !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 24px 32px !important;
    border: none !important;
    box-shadow: none !important;
}
footer { display: none !important; }

/* ── Header ── */
.header-section { margin-bottom: 20px !important; padding-bottom: 12px !important; border-bottom: 1px solid #29324A !important; }
.header-section h1 { font-size: 32px !important; font-weight: 700 !important; color: #F4F6FA !important; margin: 0 0 6px 0 !important; letter-spacing: -0.02em !important; }
.header-section p { color: #AAB4C5 !important; font-size: 14px !important; margin: 0 !important; }

/* ── Main Grid / Panels ── */
.main-grid { gap: 20px !important; }
.panel-card {
    background-color: #12182A !important;
    border: 1px solid #29324A !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}
.panel-title {
    font-size: 18px !important; font-weight: 600 !important; color: #F4F6FA !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
    margin-bottom: 18px !important; display: flex !important; align-items: center !important;
    gap: 8px !important; border-bottom: 1px solid #29324A !important; padding-bottom: 8px !important;
}

/* ── Upload ── */
.meme-dropzone {
    border: 1px dashed #29324A !important; background-color: #171E31 !important;
    border-radius: 12px !important; overflow: hidden !important;
    margin-bottom: 20px !important; transition: border-color 0.25s ease !important;
}
.meme-dropzone:hover { border-color: #8B5CF6 !important; }

/* ── Multi-Gallery: compact horizontal single-row ── */
.multi-gallery { counter-reset: meme-counter !important; }

.multi-gallery .grid-wrap,
.multi-gallery .grid-container {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    gap: 10px !important;
    padding: 10px 8px !important;
    align-items: flex-start !important;
}

.multi-gallery .thumbnail-item,
.multi-gallery button.thumbnail-item {
    flex: 0 0 auto !important;
    width: 90px !important;
    height: 90px !important;
    position: relative !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #29324A !important;
}

.multi-gallery .thumbnail-item img,
.multi-gallery button.thumbnail-item img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}

/* Badge at BOTTOM-LEFT to avoid Gradio delete-button overlap */
.multi-gallery .thumbnail-item::after,
.multi-gallery button.thumbnail-item::after {
    counter-increment: meme-counter;
    content: "MEME " counter(meme-counter);
    position: absolute;
    bottom: 4px;
    left: 4px;
    background: rgba(139, 92, 246, 0.92);
    color: white;
    font-size: 8px;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 3px;
    z-index: 10;
    pointer-events: none;
    letter-spacing: 0.03em;
    line-height: 1;
}

/* ── Segmented Control ── */
.segmented-control {
    background-color: #171E31 !important; border: 1px solid #29324A !important;
    border-radius: 10px !important; padding: 4px !important; margin-bottom: 16px !important;
}
.segmented-control label {
    flex: 1 !important; text-align: center !important; padding: 10px 16px !important;
    color: #AAB4C5 !important; font-weight: 500 !important; font-size: 14px !important;
    cursor: pointer !important; border-radius: 8px !important;
    transition: all 0.2s ease !important; border: 1px solid transparent !important;
    background: transparent !important;
}
.segmented-control label:has(input[type="radio"]:checked) {
    background-color: rgba(139, 92, 246, 0.15) !important;
    color: #F4F6FA !important; border: 1px solid rgba(139, 92, 246, 0.35) !important;
}
.mode-explanation { font-size: 13px !important; color: #AAB4C5 !important; line-height: 1.5 !important; margin-bottom: 20px !important; }
.mode-explanation strong { color: #F4F6FA !important; font-weight: 600 !important; display: inline-block !important; margin-bottom: 2px !important; }

/* ── Analyze Button ── */
#analyze-btn {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: #FFFFFF !important; font-weight: 600 !important; font-size: 16px !important;
    height: 52px !important; border: none !important; border-radius: 10px !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
    width: 100% !important; margin-top: 10px !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
}
#analyze-btn:hover { background: linear-gradient(135deg, #6D28D9, #7C3AED) !important; box-shadow: 0 4px 16px rgba(139, 92, 246, 0.35) !important; }
#analyze-btn:active { transform: translateY(1px) !important; }

/* ── Focused Viewer ── */
.focused-viewer {
    background-color: #171E31 !important; border: 1px solid #29324A !important;
    border-radius: 12px !important; padding: 16px !important;
}
.focused-viewer img { border-radius: 8px !important; }
.nav-btn {
    background-color: #171E31 !important; border: 1px solid #29324A !important;
    color: #F4F6FA !important; border-radius: 8px !important;
}
.nav-btn:hover { border-color: #8B5CF6 !important; }
.back-btn {
    background-color: transparent !important; border: 1px solid #29324A !important;
    color: #AAB4C5 !important; border-radius: 8px !important;
}
.back-btn:hover { border-color: #8B5CF6 !important; color: #F4F6FA !important; }

/* ── Responsiveness ── */
@media (max-width: 1024px) {
    .gradio-container { width: calc(100% - 32px) !important; padding: 20px 16px !important; }
    .main-grid { flex-direction: column !important; }
}
@media (max-width: 768px) {
    .gradio-container { width: 100% !important; padding: 16px 12px !important; }
}

/* ── FIX: Confidence background-image leak ── */
.progress-text { display: none !important; }

/* ═══════════════════════════════════════════════
   RESULT CARDS  (rendered server-side as HTML)
   ═══════════════════════════════════════════════ */
.meme-result-block {
    background-color: #0F1525; border: 1px solid #29324A;
    border-radius: 14px; padding: 20px; margin-bottom: 20px;
}
.meme-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid #29324A;
}
.meme-badge {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6);
    color: #fff; font-size: 12px; font-weight: 700;
    padding: 4px 12px; border-radius: 6px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.meme-thumb {
    width: 50px; height: 50px; border-radius: 6px;
    object-fit: cover; border: 1px solid #29324A;
}
.card-row { display: flex; gap: 16px; margin-bottom: 16px; }
.card-row > .r-card { flex: 1; min-width: 0; }
.r-card {
    background-color: #171E31; border: 1px solid #29324A;
    border-radius: 12px; padding: 16px 20px;
    transition: border-color 0.2s ease;
}
.r-card:hover { border-color: #38BDF8; }
.r-card-label {
    font-size: 13px; font-weight: 600; color: #AAB4C5;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}
.r-card-value {
    color: #F4F6FA; font-family: 'Inter', sans-serif;
    font-size: 15px; line-height: 1.55;
    white-space: pre-wrap; word-break: break-word;
}
.r-card-value.prediction { font-size: 22px; font-weight: 700; }
.r-card-value.placeholder { color: #6B7A90; font-style: italic; }
.r-card-fullwidth { margin-bottom: 16px; }
.status-text { font-size: 14px; color: #AAB4C5; padding: 8px 0 12px 0; }
"""

# ═══════════════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════════════
theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
_PLACEHOLDER_CONF = "<span style='color:#6B7A90;font-size:15px;font-style:italic;'>—</span>"
_PROCESSING_CONF = "<span style='color:#AAB4C5;font-size:14px;font-style:italic;'>Processing…</span>"

EMPTY_RESULT = {
    "prediction": "—", "confidence": _PLACEHOLDER_CONF,
    "ocr": "Awaiting image", "category": "Awaiting analysis",
    "dependency": "—", "reasoning": "Awaiting analysis",
    "index": 1, "is_multi": False, "image_path": "",
    "_placeholder": True,
}

PROCESSING_RESULT = {
    "prediction": "Analyzing…", "confidence": _PROCESSING_CONF,
    "ocr": "Extracting text…", "category": "Retrieving context…",
    "dependency": "Analyzing…", "reasoning": "Generating reasoning…",
    "index": 1, "is_multi": False, "image_path": "",
    "_placeholder": True,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _escape(text):
    """HTML-escape safely."""
    if text is None:
        return ""
    return html_mod.escape(str(text))


def _extract_gallery_path(item):
    """Extract a filepath string from various Gradio Gallery item formats."""
    if isinstance(item, (tuple, list)):
        return str(item[0]) if item else ""
    if isinstance(item, dict):
        return str(item.get("name") or item.get("path", ""))
    if hasattr(item, "name"):
        return str(item.name)
    return str(item) if item else ""


def _build_result_card_html(res):
    """
    Build the full HTML for one meme's 6-card result block.

    The card structure is ALWAYS the same (4 rows, 6 cards).
    Only the values change between empty / processing / completed states.
    """
    prediction = _escape(res.get("prediction", "—"))
    confidence_html = res.get("confidence", _PLACEHOLDER_CONF)  # already HTML
    ocr = _escape(res.get("ocr", "—"))
    category = _escape(res.get("category", "—"))
    dependency = _escape(res.get("dependency", "—"))
    reasoning = _escape(res.get("reasoning", "—"))
    meme_idx = res.get("index", 1)
    is_multi = res.get("is_multi", False)
    image_path = res.get("image_path", "")
    is_placeholder = res.get("_placeholder", False)

    # Prediction colour
    pl = prediction.lower()
    if pl in ("humorous", "humor"):
        pred_color = "#22C55E"
    elif pl in ("not humorous", "not humor"):
        pred_color = "#EF4444"
    else:
        pred_color = "#6B7A90"

    val_class = "r-card-value placeholder" if is_placeholder else "r-card-value"

    # Multi-meme header badge
    header = ""
    if is_multi:
        header = (
            f'<div class="meme-header">'
            f'<span class="meme-badge">Meme {meme_idx}</span>'
            f'</div>'
        )

    return f"""<div class="meme-result-block">
  {header}
  <div class="card-row">
    <div class="r-card">
      <div class="r-card-label">🎯 HUMOR PREDICTION</div>
      <div class="{val_class} prediction" style="color:{pred_color};">{prediction}</div>
    </div>
    <div class="r-card">
      <div class="r-card-label">◉ CONFIDENCE SCORE</div>
      <div class="{val_class}">{confidence_html}</div>
    </div>
  </div>
  <div class="r-card r-card-fullwidth">
    <div class="r-card-label">📄 DETECTED TEXT (OCR)</div>
    <div class="{val_class}">{ocr}</div>
  </div>
  <div class="card-row">
    <div class="r-card">
      <div class="r-card-label">🌐 CULTURAL CONTEXT</div>
      <div class="{val_class}">{category}</div>
    </div>
    <div class="r-card">
      <div class="r-card-label">🔗 CULTURAL DEPENDENCY</div>
      <div class="{val_class}">{dependency}</div>
    </div>
  </div>
  <div class="r-card r-card-fullwidth">
    <div class="r-card-label">🧠 AI REASONING</div>
    <div class="{val_class}">{reasoning}</div>
  </div>
</div>"""


def _build_all_results_html(results_list=None, status_text=""):
    """
    Build the complete results-panel HTML.

    If *results_list* is empty / None the permanent card structure is still
    rendered with placeholder content so the panel is never visually empty.
    """
    parts = []
    if status_text:
        parts.append(f'<div class="status-text">{_escape(status_text)}</div>')

    if not results_list:
        parts.append(_build_result_card_html(EMPTY_RESULT))
    else:
        for res in results_list:
            parts.append(_build_result_card_html(res))

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# UI FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def create_ui(analyze_fn):
    if gr is None:
        print("Gradio is not installed. Cannot create UI.")
        sys.exit(1)

    with gr.Blocks(title="Culturally Aware Humor Detection") as demo:

        # ── HEADER ──────────────────────────────────────────────────────────
        with gr.Column(elem_classes=["header-section"]):
            gr.Markdown("# 🎭 Culturally Aware Humor Detection")
            gr.Markdown("AI-Powered Hindi/Hinglish Meme Analysis")

        # ── STATE ───────────────────────────────────────────────────────────
        active_tab = gr.State("single")
        focused_idx = gr.State(None)

        # ── MAIN LAYOUT ────────────────────────────────────────────────────
        with gr.Row(elem_classes=["main-grid"]):

            # ── LEFT: INPUT PANEL (42 %) ────────────────────────────────────
            with gr.Column(scale=42, elem_classes=["panel-card"]):
                gr.Markdown("📥 MEME INPUT", elem_classes=["panel-title"])

                with gr.Tabs():
                    with gr.TabItem("🖼️ Single Meme") as tab_single:
                        single_image = gr.Image(
                            type="filepath",
                            label="Upload Single Meme",
                            height=250,
                            elem_classes=["meme-dropzone"],
                            show_label=False,
                        )

                    with gr.TabItem("📚 Multi-Meme / Comic Strip") as tab_multi:
                        # -- gallery row (toggled off when focused) --
                        with gr.Column() as gallery_panel:
                            multi_images = gr.Gallery(
                                label="Upload Meme Images",
                                elem_classes=["meme-dropzone", "multi-gallery"],
                                columns=8,
                                rows=1,
                                height=120,
                                type="filepath",
                                interactive=True,
                            )

                        # -- focused viewer (toggled on when a thumb is clicked) --
                        with gr.Column(
                            visible=False,
                            elem_classes=["focused-viewer"],
                        ) as focused_panel:
                            focused_label = gr.Markdown("### MEME 1")
                            focused_img = gr.Image(
                                interactive=False,
                                height=300,
                                show_label=False,
                            )
                            with gr.Row():
                                prev_btn = gr.Button(
                                    "← Previous",
                                    size="sm",
                                    elem_classes=["nav-btn"],
                                )
                                next_btn = gr.Button(
                                    "Next →",
                                    size="sm",
                                    elem_classes=["nav-btn"],
                                )
                            back_btn = gr.Button(
                                "↩ Back to All Memes",
                                size="sm",
                                elem_classes=["back-btn"],
                            )

                tab_single.select(lambda: "single", outputs=active_tab)
                tab_multi.select(lambda: "multi", outputs=active_tab)

                gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["panel-title"])

                cultural_toggle = gr.Radio(
                    choices=["General", "Cultural-Aware"],
                    value="General",
                    show_label=False,
                    elem_classes=["segmented-control"],
                )
                gr.Markdown(
                    "**General**  \n"
                    "Standard multimodal VLM reasoning without external "
                    "cultural context.  \n\n"
                    "**Cultural-Aware**  \n"
                    "Retrieves relevant Indian cultural knowledge, slang, "
                    "and context during reasoning.",
                    elem_classes=["mode-explanation"],
                )

                analyze_btn = gr.Button(
                    "✨ Analyze Meme", elem_id="analyze-btn"
                )

            # ── RIGHT: RESULTS PANEL (58 %) ─────────────────────────────────
            with gr.Column(scale=58, elem_classes=["panel-card"]):
                gr.Markdown(
                    "📊 ANALYSIS RESULTS", elem_classes=["panel-title"]
                )
                # Permanent card structure — always visible, content changes
                results_html = gr.HTML(value=_build_all_results_html())

        # ═══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS — Gallery Focused Viewer
        # ═══════════════════════════════════════════════════════════════════

        def _on_gallery_select(gallery_data, evt: gr.SelectData):
            """Open focused viewer for the clicked thumbnail."""
            idx = evt.index
            if not gallery_data or idx >= len(gallery_data):
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), None,
                )
            img = _extract_gallery_path(gallery_data[idx])
            total = len(gallery_data)
            return (
                gr.update(visible=False),                  # gallery_panel
                gr.update(visible=True),                   # focused_panel
                f"### MEME {idx + 1}",                     # focused_label
                img,                                       # focused_img
                gr.update(interactive=(idx > 0)),          # prev_btn
                gr.update(interactive=(idx < total - 1)),  # next_btn
                idx,                                       # focused_idx
            )

        multi_images.select(
            fn=_on_gallery_select,
            inputs=[multi_images],
            outputs=[
                gallery_panel, focused_panel, focused_label,
                focused_img, prev_btn, next_btn, focused_idx,
            ],
        )

        def _on_prev(gallery_data, cur):
            if cur is None or cur <= 0:
                return gr.update(), gr.update(), gr.update(), gr.update(), cur
            nxt = cur - 1
            total = len(gallery_data)
            return (
                f"### MEME {nxt + 1}",
                _extract_gallery_path(gallery_data[nxt]),
                gr.update(interactive=(nxt > 0)),
                gr.update(interactive=(nxt < total - 1)),
                nxt,
            )

        prev_btn.click(
            fn=_on_prev,
            inputs=[multi_images, focused_idx],
            outputs=[focused_label, focused_img, prev_btn, next_btn, focused_idx],
        )

        def _on_next(gallery_data, cur):
            if cur is None or gallery_data is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), cur
            total = len(gallery_data)
            if cur >= total - 1:
                return gr.update(), gr.update(), gr.update(), gr.update(), cur
            nxt = cur + 1
            return (
                f"### MEME {nxt + 1}",
                _extract_gallery_path(gallery_data[nxt]),
                gr.update(interactive=(nxt > 0)),
                gr.update(interactive=(nxt < total - 1)),
                nxt,
            )

        next_btn.click(
            fn=_on_next,
            inputs=[multi_images, focused_idx],
            outputs=[focused_label, focused_img, prev_btn, next_btn, focused_idx],
        )

        def _on_back():
            return gr.update(visible=True), gr.update(visible=False), None

        back_btn.click(
            fn=_on_back,
            outputs=[gallery_panel, focused_panel, focused_idx],
        )

        # ═══════════════════════════════════════════════════════════════════
        # EVENT HANDLER — Analysis  (only fires on explicit button click)
        # ═══════════════════════════════════════════════════════════════════

        def process_analysis(single_path, multi_paths, mode, tab):
            """
            Generator that yields updated results HTML after each meme.

            * Single-meme: one iteration.
            * Multi-meme:  one iteration per meme, yielding cumulative HTML
              so previously-completed results remain visible.
            """
            paths = []
            is_multi = False

            if tab == "multi" and multi_paths:
                for item in multi_paths:
                    p = _extract_gallery_path(item)
                    if p:
                        paths.append(p)
                is_multi = True
            elif tab == "single" and single_path:
                paths = [single_path]

            if not paths:
                yield _build_all_results_html(
                    None, "⚠️ Please upload a meme image first."
                )
                return

            results_list = []
            total = len(paths)

            for i, p in enumerate(paths, 1):
                # ── Show processing-state card for this meme ────────────
                proc = dict(PROCESSING_RESULT)
                proc["index"] = i
                proc["is_multi"] = is_multi
                if is_multi:
                    proc["image_path"] = p
                status = (
                    f"⏳ Analyzing Meme {i} of {total}…"
                    if total > 1
                    else "⏳ Analyzing meme…"
                )
                yield _build_all_results_html(
                    results_list + [proc], status
                )

                # ── Run inference (backend untouched) ───────────────────
                try:
                    out = analyze_fn(p, mode)
                except Exception as e:
                    out = (
                        "Error", "N/A", "N/A", "N/A", "N/A",
                        f"Error: {str(e)}",
                    )

                res_dict = {
                    "index": i,
                    "is_multi": is_multi,
                    "image_path": p,
                    "prediction": out[0],
                    "confidence": out[1],
                    "ocr": out[2],
                    "category": out[3],
                    "dependency": out[4],
                    "reasoning": out[5],
                }
                results_list.append(res_dict)

                # ── Yield cumulative results so far ─────────────────────
                if total > 1:
                    yield _build_all_results_html(
                        results_list, f"✅ Meme {i} Complete"
                    )

            final = (
                f"✅ {total}/{total} Memes Analyzed"
                if total > 1
                else "✅ Analysis Complete"
            )
            yield _build_all_results_html(results_list, final)

        analyze_btn.click(
            fn=process_analysis,
            inputs=[single_image, multi_images, cultural_toggle, active_tab],
            outputs=[results_html],
            show_progress="hidden",       # prevents Gradio progress thumbnail leak
        )

    return demo
