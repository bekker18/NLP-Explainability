import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import joblib
import shap
from shap.maskers import Independent
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import clean_text

# Load the model
pipeline = joblib.load(PROJECT_ROOT / "models" / "pipeline.joblib")
tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]

# Build a lightweight SHAP explainer
X_test_df = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

if "text" not in X_test_df.columns:
  raise ValueError("X_test.csv must contain a 'text' column.")

_bg_texts = X_test_df["text"].fillna("").astype(str).sample(
  n=min(500, len(X_test_df)),
  random_state=0
)

_bg_vec = tfidf.transform(_bg_texts)

if _bg_vec.nnz == 0:
  raise ValueError(
    "Background TF-IDF matrix is empty. "
    "Rerun training after fixing clean_text() so words keep spaces."
  )

masker = Independent(_bg_vec, max_samples=100)
explainer = shap.LinearExplainer(clf, masker)

def predict_and_explain(raw_text: str):
  """Predict Real/Fake and return SHAP word-level explanation plot."""

  if not raw_text.strip():
    return "-", "-", None

  text = clean_text(raw_text)
  vec = tfidf.transform([text])

  # Prediction
  proba = pipeline.predict_proba([text])[0]
  pred = int(pipeline.predict([text])[0])

  label = "Real" if pred == 1 else "Fake"
  conf = f"{proba[pred] * 100:.1f}% confident"

  # SHAP values
  shap_values = explainer.shap_values(vec)

  if isinstance(shap_values, list):
    sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
  else:
    sv = shap_values[0]

  feature_names = tfidf.get_feature_names_out()
  nonzero_idx = vec.nonzero()[1]

  word_shap = sorted(
    [
      (feature_names[i], float(sv[i]))
      for i in nonzero_idx
      if sv[i] != 0
    ],
    key=lambda x: abs(x[1]),
    reverse=True
  )[:15]

  if not word_shap:
    return label, conf, None

  words, values = zip(*word_shap)
  colors = ["#d73027" if v < 0 else "#4575b4" for v in values]

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.barh(list(words), list(values), color=colors)
  ax.axvline(0, color="black", linewidth=0.8)
  ax.set_xlabel("<- pushes toward FAKE | pushes toward REAL ->")
  ax.set_title("Word-level SHAP contributions")
  ax.invert_yaxis()
  plt.tight_layout()

  return label, conf, fig

# Gradio UI
with gr.Blocks(title="Fake News Detector + SHAP") as demo:
  gr.Markdown("""
  # Fake News Detector with SHAP Explanations

  Paste a news headline or article. The model predicts **Real** vs **Fake**
  and shows which words drove the decision.
  """)

  with gr.Row():
    text_input = gr.Textbox(
      lines=6,
      placeholder="Paste a news article or headline here...",
      label="Input text"
    )

  with gr.Row():
    btn = gr.Button("Analyze", variant="primary")

  with gr.Row():
    label_out = gr.Textbox(label="Prediction")
    conf_out = gr.Textbox(label="Confidence")

  shap_plot = gr.Plot(label="SHAP explanation - which words matter?")

  btn.click(
    fn=predict_and_explain,
    inputs=text_input,
    outputs=[label_out, conf_out, shap_plot]
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
  demo.launch(share=False)