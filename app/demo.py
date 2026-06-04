import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from data_loader import clean_text

# Load the model
pipeline = joblib.load("models/pipeline.joblib")
tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]

# Build a lighweight explainer with small background
_bg_texts = pd.read_csv("data/X_test.csv").iloc[:, 0].astype(str).sample(500, random_state=0)
_bg_vec = tfidf.transform(_bg_texts)
_bg_kmean = shap.kmeans(_bg_vec, 10)
explainer = shap.LinearExplainer(clf, _bg_kmean)

# Prediction + explanation
def predict_and_explain(raw_text: str):
  if not raw_text.strip():
    return "-", "-", None
  
  text = clean_text(raw_text)
  vec = tfidf.transform([text])

  # Prediction
  proba = pipeline.predict_proba([text])[0]
  pred = int(pipeline.predict([text])[0])
  label = "Real" if pred == 1 else "Fake"
  conf = f"{proba[pred]*100:.1f}% confident"

  # SHAP values
  sv = explainer.shap_values(vec)[0]  # 1D
  feature_names = tfidf.get_feature_names_out()
  nonzero_idx = vec.nonzero()[1]

  word_shap = sorted(
    [(feature_names[i], float(sv[i])) for i in nonzero_idx if sv[i] != 0],
    key = lambda x: abs(x[1]), reverse=True
  )[:15]

  if not word_shap:
    return label, conf, None
  
  words, values = zip(*word_shap)
  colors = ["#d73027" if v < 0 else "#4575b4" for v in values]

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.barh(list(words), list(values), color=colors)
  ax.axvline(0, color="black", linewidth=0.8)
  ax.set_xlabel("<- pushes toward FAKE  |  pushes toward REAL ->")
  ax.set_title("Word-level SHAP contributions")
  ax.invert_yaxis()
  plt.tight_layout()

  return label, conf, fig

# Gradio UI
with gr.Blocks(title="Fake News Detector + SHAP") as demo:
  gr.Markdown("""
  # Fake News Detectorwith SHAP Explanations
  Paste a news headline or article . The model predicts Real vs Fake
  **and shows which words drove the decision.**
  """)

  with gr.Row():
    text_input = gr.Textbox(
      lines=6,
      placeholder="Paste a news article or headline here...",
      label="Input text"
    )

  with gr.Row():
    btn = gr.Button("Analyse", variant="primary")

  with gr.Row():
    lable_out = gr.Textbox(label="Predictions")
    conf_out = gr.Textbox(label="Confidence")

  shap_plot = gr.Plot(label="SHAP explanation - wich words matter?")

  btn.click(
    fn=predict_and_explain,
    inputs=text_input,
    outputs=[lable_out, conf_out, shap_plot]
  )

  gr.Examples(
    examples=[
      ["Scientists discover new species of deep-sea creature near hydrothermal vents, publishing findings in Nature journal"],
      ["SHOCKING: Government SECRETLY replacing water supply with 5G nanobots, leaked documents reveal TRUTH they don't want you to know"],
      ["Federal Reserve raises interest rates by 25 basis points amid inflation concerns, markets respond cautiously"],
    ],
    inputs=text_input,
  )

if __name__ == "__main__":
  demo.launch(share=True)