import sys
try:
    import gradio as gr
except ImportError:
    gr = None

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply clean academic font & charcoal/dark-navy theme */
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B0D18 !important;
    color: #F3F4F6 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 24px 32px !important;
    border: none !important;
    box-shadow: none !important;
}

/* Header Spacing and Hierarchy */
.header-container {
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid #252A40;
    text-align: left;
}

.header-container h1 {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #F3F4F6 !important;
    margin-bottom: 6px !important;
    letter-spacing: -0.02em !important;
}

.header-container p {
    color: #9CA3AF !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}

/* Two-Column Grid Setup */
.two-column-layout {
    display: grid !important;
    grid-template-columns: minmax(380px, 0.42fr) minmax(550px, 0.58fr) !important;
    gap: 24px !important;
    align-items: start !important;
}

/* Cards (Input and Output) */
.input-panel, .output-panel {
    background-color: #111426 !important;
    border: 1px solid #252A40 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.3) !important;
}

/* Card Heading Header format */
.card-title {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #9CA3AF !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 16px !important;
}

/* Image Dropzone Customization */
.meme-dropzone {
    border: 1px dashed #252A40 !important;
    background-color: #15182A !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.meme-dropzone .image-container {
    max-height: 230px !important;
}

/* Radio Button Segmented Control */
.segmented-control {
    background-color: #15182A !important;
    border: 1px solid #252A40 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    margin-top: 8px !important;
}

.segmented-control label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 14px !important;
    color: #9CA3AF !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}

/* Hide native circular checkboxes */
.segmented-control input[type="radio"] {
    display: none !important;
}

/* Styled Selected State via CSS has() */
.segmented-control label:has(input[type="radio"]:checked) {
    background-color: rgba(139, 92, 246, 0.12) !important;
    color: #8B5CF6 !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
}

.toggle-explanation {
    margin-top: 10px !important;
    font-size: 12px !important;
    color: #9CA3AF !important;
    line-height: 1.45 !important;
}

.toggle-explanation strong {
    color: #F3F4F6 !important;
    display: block !important;
    margin-top: 6px !important;
    font-weight: 600 !important;
}

/* Custom styled action button with subtle purple */
#analyze-btn {
    background: #8B5CF6 !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    height: 50px !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 18px !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
}

#analyze-btn:hover {
    background: #7C3AED !important;
    box-shadow: 0 6px 16px rgba(139, 92, 246, 0.3) !important;
}

#analyze-btn:active {
    transform: translateY(1px) !important;
}

/* Outputs & Results Box layouts */
.results-subhead {
    font-size: 13px !important;
    color: #9CA3AF !important;
    margin-bottom: 20px !important;
    border-bottom: 1px solid #252A40;
    padding-bottom: 12px;
}

.result-card-item {
    background-color: #15182A !important;
    border: 1px solid #252A40 !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
}

/* Prediction Block text values */
.prediction-box textarea {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #F3F4F6 !important;
    background-color: #15182A !important;
    border: none !important;
    text-align: left !important;
    padding: 12px !important;
}

/* Confidence score box centring */
.confidence-box {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    padding: 12px !important;
    min-height: 48px !important;
}

/* Equalise top row card heights */
.top-results-row > * {
    min-height: 80px !important;
}

/* OCR Box */
.ocr-box textarea {
    font-size: 14px !important;
    line-height: 1.5 !important;
    padding: 14px !important;
    background-color: #15182A !important;
    border: none !important;
}

/* Reasoning Box */
.reasoning-box textarea {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #F3F4F6 !important;
    padding: 14px !important;
    background-color: #15182A !important;
    border: none !important;
}

/* Customized Gradio loader overlay */
.loading {
    background-color: rgba(11, 13, 24, 0.85) !important;
}

.loading .loading-icon {
    border-color: #8B5CF6 !important;
    border-right-color: transparent !important;
}

.loading .meta-text {
    font-size: 13px !important;
    color: #9CA3AF !important;
    font-weight: 500 !important;
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
        with gr.Column(elem_classes=["header-container"]):
            gr.Markdown("# 🎭 Culturally Aware Humor Detection")
            gr.Markdown("AI-Powered Hindi/Hinglish Meme Analysis")
        
        with gr.Row(elem_classes=["two-column-layout"]):
            with gr.Column(scale=42, elem_classes=["input-panel"]):
                gr.Markdown("### MEME INPUT", elem_classes=["card-title"])
                
                image_input = gr.Image(
                    type="filepath", 
                    label="Upload Meme", 
                    height=230,
                    elem_classes=["meme-dropzone"],
                    show_label=False
                )
                
                gr.Markdown("### Cultural Context", elem_classes=["card-title"])
                
                cultural_toggle = gr.Radio(
                    choices=["General", "Cultural-Aware"], 
                    value="General", 
                    show_label=False,
                    elem_classes=["segmented-control"]
                )
                
                gr.Markdown(
                    "**General**\n"
                    "Standard VLM reasoning without explicit cultural context.\n\n"
                    "**Cultural-Aware**\n"
                    "Uses relevant Indian cultural context during reasoning.",
                    elem_classes=["toggle-explanation"]
                )
                
                analyze_btn = gr.Button("Analyze Meme", elem_id="analyze-btn")
                
            with gr.Column(scale=58, elem_classes=["output-panel"]):
                gr.Markdown("### Analysis Results", elem_classes=["card-title"])
                gr.Markdown("AI-generated insights from your meme", elem_classes=["results-subhead"])
                
                with gr.Row(elem_classes=["top-results-row"]):
                    pred_out = gr.Textbox(
                        label="HUMOR PREDICTION", 
                        interactive=False, 
                        elem_classes=["prediction-box", "result-card-item"]
                    )
                    with gr.Column(elem_classes=["result-card-item"]):
                        gr.Markdown("CONFIDENCE LEVEL", elem_classes=["card-title"])
                        conf_out = gr.HTML(elem_classes=["confidence-box"])
                        
                text_out = gr.Textbox(
                    label="DETECTED TEXT (OCR)", 
                    interactive=False, 
                    lines=2,
                    elem_classes=["ocr-box", "result-card-item"]
                )
                
                # cat_out maps to CULTURAL CONTEXT. dep_out is hidden to present a unified context view.
                cat_out = gr.Textbox(
                    label="CULTURAL CONTEXT", 
                    interactive=False, 
                    lines=2,
                    elem_classes=["ocr-box", "result-card-item"]
                )
                dep_out = gr.Textbox(visible=False)
                
                reason_out = gr.Textbox(
                    label="AI REASONING", 
                    interactive=False, 
                    lines=5,
                    elem_classes=["reasoning-box", "result-card-item"]
                )
                
        analyze_btn.click(
            fn=analyze_fn,
            inputs=[image_input, cultural_toggle],
            outputs=[pred_out, conf_out, text_out, cat_out, dep_out, reason_out]
        )
        
    return demo
