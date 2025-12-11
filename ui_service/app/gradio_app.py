import os
import requests
import gradio as gr

API_URL = os.environ.get("API_URL", "http://api:8000")

# ---------------- GLOBAL HISTORY LIST ----------------
history = []


# ---------- PREDICT FUNCTION ----------
def predict_and_index(text):
    payload = {"texts": [text], "add_to_index": True}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    r.raise_for_status()
    res = r.json()

    if "results" in res and len(res["results"]) > 0:
        r0 = res["results"][0]
        label = r0.get("label")
        probs = r0.get("probs")

        positive_prob = probs[1]
        if positive_prob >= 0.8:
            level = "Strongly Positive"
        elif positive_prob >= 0.6:
            level = "Positive"
        elif positive_prob >= 0.4:
            level = "Neutral"
        elif positive_prob >= 0.2:
            level = "Negative"
        else:
            level = "Strongly Negative"

        sentiment_label = "Positive" if label == 1 else "Negative"

        # main output result
        result = (
            f"Label: {sentiment_label}\n"
            f"Probability: {positive_prob:.4f}\n"
            f"Sentiment Level: {level}\n"
            f"Text: {r0.get('text')}"
        )

        # SAVE TO HISTORY
        history.append(result)

        # ---------------- THEME-AWARE HISTORY CARDS (LATEST FIRST) ----------------
        history_html = ""
        for item in reversed(history):     # latest first ✔
            lines = item.split("\n")
            sentiment = lines[0].replace("Label: ", "")
            probability = float(lines[1].replace("Probability: ", "")) * 100
            level_val = lines[2].replace("Sentiment Level: ", "")
            text_val = lines[3].replace("Text: ", "")

            history_html += f"""
<div style="
    padding:12px;
    margin-top:12px;
    border-radius:10px;
    background: var(--block-background);
    border:1px solid var(--border-color-primary);
">
    <div style="font-size:15px; margin-bottom:6px; color: var(--body-text-color);">
         {text_val}
    </div>
    <div style="font-size:13px; color: var(--body-text-color); opacity:0.8;">
        <b>Label:</b> {sentiment} · 
        <b>Confidence:</b> {probability:.2f}% · 
        <b>Level:</b> {level_val}
    </div>
</div>
"""
        # --------------------------------------------------------------------

        return result, history_html

    return "No result", ""


# ---------------- CLEAR (ONLY INPUT) ----------------
def clear_input_only():
    return ""


# -------------------------- UI LAYOUT --------------------------
with gr.Blocks() as demo:

    # Inject custom ORANGE button CSS
    gr.HTML("""
    <style>
    .custom-orange-btn {
        background-color: #ff9800 !important;
        color: black !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid #e68900 !important;
    }
    .custom-orange-btn:hover {
        background-color: #ffa726 !important;
    }
    </style>
    """)

    gr.Markdown("# Sentiment Analysis")

    with gr.Row():

        # LEFT COLUMN → INPUT
        with gr.Column(scale=1):
            txt = gr.Textbox(
                lines=6,
                label="Input Text",
                placeholder="Enter text...",
            )

            clear_btn = gr.Button("Clear")
            btn = gr.Button("Analyze", elem_classes=["custom-orange-btn"])  # ORANGE BUTTON ✔

        # RIGHT COLUMN → OUTPUT + HISTORY
        with gr.Column(scale=1):
            out = gr.Textbox(
                lines=6,
                label="Output",
                placeholder="Result will appear here..."
            )

            out_history = gr.HTML(
                "<div style='color: var(--body-text-color); opacity:0.7;'>History will appear here...</div>"
            )

    # BUTTON CONNECTIONS
    btn.click(predict_and_index, inputs=[txt], outputs=[out, out_history])

    clear_btn.click(clear_input_only, inputs=[], outputs=[txt])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7880, share=False)