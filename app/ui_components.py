import sys
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

/* Hide Gradio footer */
footer {
    display: none !important;
}

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
.main-grid {
    gap: 20px !important;
}

/* Left Input Card / Right Output Card panel styling */
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
.result-card {
    background-color: #171E31 !important;
    border: 1px solid #29324A !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
    height: auto !important;
    transition: all 0.2s ease !important;
}

.result-card:hover {
    border-color: #38BDF8 !important;
}

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

/* Textarea / Input styling within result cards */
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

.result-card textarea:focus {
    box-shadow: none !important;
}

/* Specific styling for Humor prediction text */
.prediction-box textarea {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #22C55E !important; /* Green for positive humor, adjusted on output */
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

.meme-dropzone:hover {
    border-color: #8B5CF6 !important;
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

#analyze-btn:active {
    transform: translateY(1px) !important;
}

/* ──────────────────────────────────────────────
   GRID ROWS AND EQUAL WIDTHS
   ────────────────────────────────────────────── */
.top-row, .middle-row {
    gap: 16px !important;
}

.top-row > *, .middle-row > * {
    flex: 1 !important;
    min-width: 0 !important;
    margin-bottom: 0 !important;
}

/* ──────────────────────────────────────────────
   PROCESSING / LOADING STATE OVERRIDES
   ────────────────────────────────────────────── */
.pending {
    background-color: rgba(18, 24, 42, 0.95) !important;
    border-radius: 12px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 24px !important;
    gap: 12px !important;
}

.pending .eta-bar, .pending .loading {
    display: none !important;
}

/* Analyzing text */
.pending::after {
    content: "Analyzing meme..." !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #AAB4C5 !important;
    display: block !important;
}

/* Indeterminate progress bar track */
.pending::before {
    content: "" !important;
    display: block !important;
    width: 160px !important;
    height: 4px !important;
    background: #29324A !important;
    border-radius: 999px !important;
    position: relative !important;
    overflow: hidden !important;
}

/* ──────────────────────────────────────────────
   RESPONSIVENESS
   ────────────────────────────────────────────── */
@media (max-width: 1024px) {
    .gradio-container {
        width: calc(100% - 32px) !important;
        padding: 20px 16px !important;
    }
    .main-grid {
        flex-direction: column !important;
    }
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
        
        # ── MAIN LAYOUT GRID ──
        with gr.Row(elem_classes=["main-grid"]):
            
            # ── LEFT COLUMN: INPUT PANEL (42%) ──
            with gr.Column(scale=42, elem_classes=["panel-card"]):
                gr.Markdown("📥 MEME INPUT", elem_classes=["panel-title"])
                
                image_input = gr.Image(
                    type="filepath", 
                    label="Upload Meme", 
                    height=240,
                    elem_classes=["meme-dropzone"],
                    show_label=False
                )
                
                gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["panel-title"])
                
                cultural_toggle = gr.Radio(
                    choices=["General", "Cultural-Aware"], 
                    value="General", 
                    show_label=False,
                    elem_classes=["segmented-control"]
                )
                
                gr.Markdown(
                    "**General**  \n"
                    "Standard VLM reasoning without explicit cultural context.  \n\n"
                    "**Cultural-Aware**  \n"
                    "Uses relevant Indian cultural context during reasoning.",
                    elem_classes=["mode-explanation"]
                )
                
                analyze_btn = gr.Button("✨ Analyze Meme", elem_id="analyze-btn")
            
            # ── RIGHT COLUMN: RESULTS PANEL (58%) ──
            with gr.Column(scale=58, elem_classes=["panel-card"]):
                gr.Markdown("📊 ANALYSIS RESULTS", elem_classes=["panel-title"])
                gr.Markdown("AI-generated insights from your meme", elem_classes=["results-subtitle"])
                
                # Row 1: Humor Prediction + Confidence Score
                with gr.Row(elem_classes=["top-row"], equal_height=True):
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("🎯 HUMOR PREDICTION", elem_classes=["card-label"])
                        pred_out = gr.Textbox(
                            show_label=False, 
                            interactive=False, 
                            elem_classes=["prediction-box"]
                        )
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("◉ CONFIDENCE SCORE", elem_classes=["card-label"])
                        conf_out = gr.HTML(elem_classes=["confidence-display"])
                
                # Row 2: Detected Text (OCR)
                with gr.Column(elem_classes=["result-card"]):
                    gr.Markdown("📄 DETECTED TEXT (OCR)", elem_classes=["card-label"])
                    text_out = gr.Textbox(
                        show_label=False, 
                        interactive=False, 
                        lines=2
                    )
                
                # Row 3: Cultural Context + Cultural Dependency
                with gr.Row(elem_classes=["middle-row"], equal_height=True):
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("🌐 CULTURAL CONTEXT", elem_classes=["card-label"])
                        cat_out = gr.Textbox(
                            show_label=False, 
                            interactive=False, 
                            lines=2
                        )
                    with gr.Column(elem_classes=["result-card"]):
                        gr.Markdown("🔗 CULTURAL DEPENDENCY", elem_classes=["card-label"])
                        dep_out = gr.Textbox(
                            show_label=False, 
                            interactive=False, 
                            lines=2
                        )
                
                # Row 4: AI Reasoning
                with gr.Column(elem_classes=["result-card"]):
                    gr.Markdown("🧠 AI REASONING", elem_classes=["card-label"])
                    reason_out = gr.Textbox(
                        show_label=False, 
                        interactive=False, 
                        lines=4
                    )
                
        # ── CLICK ACTION BINDING ──
        analyze_btn.click(
            fn=analyze_fn,
            inputs=[image_input, cultural_toggle],
            outputs=[pred_out, conf_out, text_out, cat_out, dep_out, reason_out]
        )
        
    return demo
