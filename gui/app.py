"""
GUI Module.

Antarmuka sederhana untuk inference menggunakan model final.
TIDAK berisi logika training.

GUI Minimum Requirements:
- Input: text area untuk komentar
- Tombol: Predict
- Output:
  - 7 emosi dengan probability score
  - Threshold yang digunakan
  - Label aktif yang terprediksi
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.inference import EmotionPredictor
from src.config import LABEL_COLUMNS


# Global predictor (loaded once)
predictor = None


def load_predictor():
    """Load model predictor (singleton)."""
    global predictor
    if predictor is None:
        predictor = EmotionPredictor()
    return predictor


def predict_emotions(text):
    """
    Callback untuk GUI: prediksi emosi dari teks input.
    
    Args:
        text: Input komentar dari user
        
    Returns:
        tuple: (probabilities_text, predicted_labels_text, threshold_text)
    """
    if not text or text.strip() == "":
        return "Please enter a comment.", "", ""
    
    pred = load_predictor()
    result = pred.predict(text)
    
    # Format probabilities
    prob_lines = []
    for label in LABEL_COLUMNS:
        prob = result["probabilities"][label]
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        is_active = "  ✅" if label in result["predicted_labels"] else ""
        prob_lines.append(f"{label:<10} {bar} {prob:.4f}{is_active}")
    prob_text = "\n".join(prob_lines)
    
    # Format predicted labels
    if result["predicted_labels"]:
        labels_text = ", ".join(result["predicted_labels"])
    else:
        labels_text = "(no emotion detected above threshold)"
    
    threshold_text = f"Threshold: {result['threshold']}"
    
    return prob_text, labels_text, threshold_text


def create_app():
    """Buat Gradio app."""
    with gr.Blocks(
        title="GoEmotions-Ekman: Emotion Classification",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("# 🎭 Multi-label Emotion Classification")
        gr.Markdown("Predict emotions from English Reddit comments using a fine-tuned BERT model.")
        gr.Markdown(
            "> ⚠️ **Note:** This is a model prediction and may be incorrect. "
            "The model was trained on Reddit comments and may not generalize perfectly to all text types."
        )
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Comment",
                    placeholder="Type or paste a comment here...",
                    lines=4,
                )
                predict_btn = gr.Button("🔍 Predict", variant="primary")
            
            with gr.Column():
                prob_output = gr.Textbox(
                    label="Emotion Probabilities",
                    lines=8,
                    interactive=False,
                )
                labels_output = gr.Textbox(
                    label="Predicted Labels (above threshold)",
                    interactive=False,
                )
                threshold_output = gr.Textbox(
                    label="Threshold Used",
                    interactive=False,
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["I'm so happy today! Everything is going great!"],
                ["This makes me really angry and frustrated."],
                ["I can't believe this happened, what a surprise!"],
                ["I feel sad and disappointed about the news."],
            ],
            inputs=text_input,
        )
        
        predict_btn.click(
            fn=predict_emotions,
            inputs=text_input,
            outputs=[prob_output, labels_output, threshold_output],
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
