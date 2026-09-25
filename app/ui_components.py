import sys
import os
try:
    import gradio as gr
except ImportError:
    gr = None

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ──────────────────────────────────────────────
   BASE CONTAINER AND RESET
   ────────────────────────────────────────────── */
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

/* ──────────────────────────────────────────────
   HEADER SECTION
   ────────────────────────────────────────────── */
.header-section {
    margin-bottom: 20px !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #29324A !important;
}

.header-section h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #F4F6FA !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em !important;
}

.header-section p {
    color: #AAB4C5 !important;
    font-size: 14px !important;
    margin: 0 !important;
}

/* ──────────────────────────────────────────────
   MAIN GRID LAYOUT
   ────────────────────────────────────────────── */
.main-grid { gap: 20px !important; }

.panel-card {
    background-color: #12182A !important;
    border: 1px solid #29324A !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* ──────────────────────────────────────────────
   TYPOGRAPHY & HEADINGS
   ────────────────────────────────────────────── */
.panel-title {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #F4F6FA !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 18px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    border-bottom: 1px solid #29324A !important;
    padding-bottom: 8px !important;
}

.results-subtitle {
    font-size: 14px !important;
    color: #AAB4C5 !important;
    margin-top: 12px !important;
    margin-bottom: 14px !important;
}

/* ──────────────────────────────────────────────
   INNER RESULTS CARDS
   ────────────────────────────────────────────── */
.result-block {
    background-color: #0B1020 !important;
    border: 1px solid #29324A !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin-bottom: 24px !important;
}

.result-card {
    background-color: #171E31 !important;
    border: 1px solid #29324A !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
    height: auto !important;
    transition: all 0.2s ease !important;
}

.result-card:hover { border-color: #38BDF8 !important; }

.card-label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #AAB4C5 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.result-card textarea {
    background-color: transparent !important;
    border: none !important;
    color: #F4F6FA !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.55 !important;
    padding: 0 !important;
    resize: none !important;
    width: 100% !important;
}

.result-card textarea:focus { box-shadow: none !important; }

.prediction-box textarea {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #22C55E !important;
}

/* ──────────────────────────────────────────────
   IMAGE UPLOAD COMPONENT
   ────────────────────────────────────────────── */
.meme-dropzone {
    border: 1px dashed #29324A !important;
    background-color: #171E31 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    margin-bottom: 20px !important;
    transition: border-color 0.25s ease !important;
}

.meme-dropzone:hover { border-color: #8B5CF6 !important; }

/* CSS Counter for Multi-Meme Upload Gallery */
.multi-gallery { counter-reset: meme-counter; }
.multi-gallery button.thumbnail-item::before, .multi-gallery .thumbnail-item::before {
    counter-increment: meme-counter;
    content: "MEME " counter(meme-counter);
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(139, 92, 246, 0.9);
    color: white;
    font-size: 11px;
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 6px;
    z-index: 10;
    pointer-events: none;
}

/* ──────────────────────────────────────────────
   SEGMENTED CONTROL (Cultural Toggle)
   ────────────────────────────────────────────── */
.segmented-control {
    background-color: #171E31 !important;
    border: 1px solid #29324A !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 16px !important;
}

.segmented-control label {
    flex: 1 !important;
    text-align: center !important;
    padding: 10px 16px !important;
    color: #AAB4C5 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}

.segmented-control label:has(input[type="radio"]:checked) {
    background-color: rgba(139, 92, 246, 0.15) !important;
    color: #F4F6FA !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
}

.mode-explanation {
    font-size: 13px !important;
    color: #AAB4C5 !important;
    line-height: 1.5 !important;
    margin-bottom: 20px !important;
}

.mode-explanation strong {
    color: #F4F6FA !important;
    font-weight: 600 !important;
    display: inline-block !important;
    margin-bottom: 2px !important;
}

/* ──────────────────────────────────────────────
   ANALYZE BUTTON
   ────────────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    height: 52px !important;
    border: none !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 10px !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
}

#analyze-btn:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.35) !important;
}

#analyze-btn:active { transform: translateY(1px) !important; }

/* ──────────────────────────────────────────────
   GRID ROWS AND EQUAL WIDTHS
   ────────────────────────────────────────────── */
.top-row, .middle-row { gap: 16px !important; }
.top-row > *, .middle-row > * {
    flex: 1 !important;
    min-width: 0 !important;
    margin-bottom: 0 !important;
}

/* ──────────────────────────────────────────────
   RESPONSIVENESS
   ────────────────────────────────────────────── */
@media (max-width: 1024px) {
    .gradio-container {
        width: calc(100% - 32px) !important;
        padding: 20px 16px !important;
    }
    .main-grid { flex-direction: column !important; }
}

@media (max-width: 768px) {
    .gradio-container {
        width: 100% !important;
        padding: 16px 12px !important;
    }
    .top-row, .middle-row {
        flex-direction: column !important;
        gap: 0 !important;
    }
    .top-row > *, .middle-row > * {
        margin-bottom: 16px !important;
    }
}

/* FIX BACKGROUND LEAK: Hide Gradio's internal progress image */
.progress-text { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
)

def create_ui(analyze_fn):
    if gr is None:
        print("Gradio is not installed. Cannot create UI.")
        sys.exit(1)
        
    with gr.Blocks(title="Culturally Aware Humor Detection") as demo:
        
        # ── HEADER ──
        with gr.Column(elem_classes=["header-section"]):
            gr.Markdown("# 🎭 Culturally Aware Humor Detection")
            gr.Markdown("AI-Powered Hindi/Hinglish Meme Analysis")
        
        # State for single and multi meme results
        current_results = gr.State([])
        
        # ── MAIN LAYOUT GRID ──
        with gr.Row(elem_classes=["main-grid"]):
            
            # ── LEFT COLUMN: INPUT PANEL (42%) ──
            with gr.Column(scale=42, elem_classes=["panel-card"]):
                gr.Markdown("📥 MEME INPUT", elem_classes=["panel-title"])
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Single Meme") as tab_single:
                        single_image = gr.Image(
                            type="filepath", 
                            label="Upload Single Meme", 
                            height=250,
                            elem_classes=["meme-dropzone"],
                            show_label=False
                        )
                    with gr.TabItem("📚 Multi-Meme / Comic Strip") as tab_multi:
                        multi_images = gr.Gallery(
                            label="Upload Meme Images",
                            elem_classes=["meme-dropzone", "multi-gallery"],
                            columns=2,
                            type="filepath",
                            allow_active_elements=False
                        )
                
                active_tab = gr.State("single")
                tab_single.select(lambda: "single", outputs=active_tab)
                tab_multi.select(lambda: "multi", outputs=active_tab)

                gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["panel-title"])
                
                cultural_toggle = gr.Radio(
                    choices=["General", "Cultural-Aware"], 
                    value="General", 
                    show_label=False,
                    elem_classes=["segmented-control"]
                )
                
                gr.Markdown(
                    "**General**  \\n"
                    "Standard multimodal VLM reasoning without external cultural context.  \\n\\n"
                    "**Cultural-Aware**  \\n"
                    "Retrieves relevant Indian cultural knowledge, slang, and context during reasoning.",
                    elem_classes=["mode-explanation"]
                )
                
                analyze_btn = gr.Button("✨ Analyze Meme", elem_id="analyze-btn")
            
            # ── RIGHT COLUMN: RESULTS PANEL (58%) ──
            with gr.Column(scale=58, elem_classes=["panel-card"]):
                gr.Markdown("📊 ANALYSIS RESULTS", elem_classes=["panel-title"])
                
                status_box = gr.Markdown("Waiting for input...", elem_classes=["results-subtitle"])
                
                @gr.render(inputs=[current_results])
                def render_results(results):
                    if not results:
                        gr.Markdown("No results yet. Upload a meme and click Analyze.")
                        return
                    
                    for res in results:
                        meme_idx = res.get("index", 1)
                        if res.get("is_multi", False):
                            gr.Markdown(f"### MEME {meme_idx}")
                            gr.Image(value=res.get("image_path"), height=150, interactive=False, show_label=False)
                            
                        with gr.Column(elem_classes=["result-block"]):
                            # Row 1: Humor Prediction + Confidence Score
                            with gr.Row(elem_classes=["top-row"], equal_height=True):
                                with gr.Column(elem_classes=["result-card"]):
                                    gr.Markdown("🎯 HUMOR PREDICTION", elem_classes=["card-label"])
                                    gr.Textbox(value=res["prediction"], show_label=False, interactive=False, elem_classes=["prediction-box"])
                                with gr.Column(elem_classes=["result-card"]):
                                    gr.Markdown("◉ CONFIDENCE SCORE", elem_classes=["card-label"])
                                    gr.HTML(value=res["confidence"], elem_classes=["confidence-display"])
                            
                            # Row 2: Detected Text (OCR)
                            with gr.Column(elem_classes=["result-card"]):
                                gr.Markdown("📄 DETECTED TEXT (OCR)", elem_classes=["card-label"])
                                gr.Textbox(value=res["ocr"], show_label=False, interactive=False, lines=2)
                            
                            # Row 3: Cultural Context + Cultural Dependency
                            with gr.Row(elem_classes=["middle-row"], equal_height=True):
                                with gr.Column(elem_classes=["result-card"]):
                                    gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["card-label"])
                                    gr.Textbox(value=res["category"], show_label=False, interactive=False, lines=2)
                                with gr.Column(elem_classes=["result-card"]):
                                    gr.Markdown("🔗 CULTURAL DEPENDENCY", elem_classes=["card-label"])
                                    gr.Textbox(value=res["dependency"], show_label=False, interactive=False, lines=2)
                            
                            # Row 4: AI Reasoning
                            with gr.Column(elem_classes=["result-card"]):
                                gr.Markdown("🧠 AI REASONING", elem_classes=["card-label"])
                                gr.Textbox(value=res["reasoning"], show_label=False, interactive=False, lines=4)
                
        # ── CLICK ACTION BINDING ──
        def process_analysis(single_path, multi_paths, mode, tab):
            paths = []
            is_multi = False
            if tab == "multi" and multi_paths:
                # Gallery returns a list of tuples (filepath, caption) or dicts or strings
                for item in multi_paths:
                    p = item[0] if isinstance(item, (tuple, list)) else (item.get("name") if isinstance(item, dict) else str(item))
                    if p: paths.append(p)
                is_multi = True
            elif tab == "single" and single_path:
                paths = [single_path]
            
            if not paths:
                yield "Error: Please upload a meme image.", []
                return
            
            results_list = []
            total = len(paths)
            
            for i, p in enumerate(paths, 1):
                if total > 1:
                    yield f"Analyzing Meme {i} of {total}...", results_list
                else:
                    yield "Analyzing meme...", results_list
                
                try:
                    out = analyze_fn(p, mode)
                except Exception as e:
                    out = ("Error", "N/A", "N/A", "N/A", "N/A", f"Error: {str(e)}")
                
                res_dict = {
                    "index": i,
                    "is_multi": is_multi,
                    "image_path": p,
                    "prediction": out[0],
                    "confidence": out[1],
                    "ocr": out[2],
                    "category": out[3],
                    "dependency": out[4],
                    "reasoning": out[5]
                }
                results_list.append(res_dict)
                
                if total > 1:
                    yield f"Meme {i} Complete", results_list
            
            if total > 1:
                yield f"{total}/{total} Memes Analyzed", results_list
            else:
                yield "Analysis Complete", results_list

        analyze_btn.click(
            fn=process_analysis,
            inputs=[single_image, multi_images, cultural_toggle, active_tab],
            outputs=[status_box, current_results],
            show_progress="hidden"
        )
        
    return demo
