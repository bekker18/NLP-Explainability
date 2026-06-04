import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt

def load_artefacts():
  """Load pipeline and test data saved during training"""

  pipeline = joblib.load("../models/pipeline.joblib")
  tfidf = pipeline.named_steps["tfidf"]

  X_test_df = pd.read_csv("../data/X_test.csv")
  y_test_df = pd.read_csv("../data/y_test.csv")

  print("X_test shape:", X_test_df.shape)
  print("X_test columns:", X_test_df.columns.tolist())
  print(X_test_df.head())

  # Remove saved pandas index columns
  X_test_df = X_test_df.loc[:, ~X_test_df.columns.str.contains("^Unnamed")]

  best_col = None
  best_nnz = -1

  # Find which column produces non-zero TF-IDF vectors
  for col in X_test_df.columns:
    texts = X_test_df[col].fillna("").astype(str)

    test_sample = texts.head(min(200, len(texts)))
    mat = tfidf.transform(test_sample)

    print(f"Column: {col} | TF-IDF non-zero values: {mat.nnz}")
    print("Example:", texts.iloc[0][:200])
    print("-" * 50)

    if mat.nnz > best_nnz:
      best_nnz = mat.nnz
      best_col = col

  if best_col is None or best_nnz == 0:
    raise ValueError(
      "No valid text column found in X_test.csv. "
      "Your X_test.csv probably does not contain the real news text."
    )

  print(f"Using text column: {best_col}")

  X_test = X_test_df[best_col].fillna("").astype(str)
  y_test = y_test_df.iloc[:, 0]

  return pipeline, X_test, y_test

def get_explainer(pipeline):
  """
  Build a SHAP LinearExplainer directly on the TF-IDF + LR pipeline.
  
  LinearExplainer is exact (not sampling-based) and works because
  Logistic Regressor is a linear model
  """

  tfidf = pipeline.named_steps["tfidf"]
  clf = pipeline.named_steps["clf"]

  # Transform a backgroud sample so SHAP knows the features distribution
  # We use the mean of a 1000-sample background (standard practice)
  if X_test_global is None:
    raise ValueError("X_test is not loaded. Call load_artefacts() firts.")
  
  background_texts = X_test_global.sample(
    n=min(1000, len(X_test_global)),
    random_state=0
  )

  background_matrix = tfidf.transform(background_texts)

  if background_matrix.nnz == 0:
    raise ValueError(
      "TF-IDF background matrix is all zeros. "
      "You are probably using the wrong text column from X_test.csv."
    )
  
  explainer = shap.LinearExplainer(clf, background_matrix)

  return explainer, tfidf

# We need X_test at module level for background sampling
pipeline, X_test_global, y_test_global = None, None, None

def explain_single(text: str, pipeline, explainer, tfidf):
  """
  Return (prediction_label, confidence, word_shap_pairs)
  word_shap_pairs: list of (word, shap_value) sorted by |value| descending
  """

  # Vectorize
  vec = tfidf.transform([text])

  # SHAP values - shape (1, n_features); positive = toward class 1 (Real)
  shap_values = explainer.shap_values(vec)  # shape: (1, vocab_size)
  sv = shap_values[0]  # 1D array, per feature

  # Prediction
  proba = pipeline.predict_proba([text])[0]
  pred = int(pipeline.predict([text])[0])
  label = "Real" if pred == 1 else "Fake"
  conf = proba[pred]

  # Map non-zero SHAP values back to feature names
  feature_names = tfidf.get_feature_names_out()
  nonzero_idx = vec.nonzero()[1]  # indices of present features

  word_shap = [
    (feature_names[i], float(sv[i]))
    for i in nonzero_idx
    if sv[i] != 0
  ]
  word_shap.sort(key = lambda x: abs(x[1]), reverse=True)

  return label, conf, word_shap[:20]  # top-20 most influential words

def plot_top_words(word_shap, title="Top contributing words", save_path=None):
  """
  Horizontal bar chart - red = pushed toward Fake, blue = pushed toward Real.
  """

  words, values = zip(*word_shap[:15])
  colors = ["#d73027" if v < 0 else "#4575b4" for v in values]

  fig, ax = plt.subplots(figsize=(7, 4))
  bars = ax.barh(words, values, colors=colors)
  ax.axvline(0, color="black", linewidth=0.8)
  ax.set_xlabel("SHAP value (negative -> Fake | positive -> Real)")
  ax.set_title(title)
  ax.invert_yaxis()
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
  else:
    plt.show()

def run_batch_analysis(n_samples=200):
  """
  Compute SHAP values for a sample of the test set and produce
  a global summary plot.
  """

  global pipeline, X_test_global, y_test_global
  pipeline, X_test_global, y_test_global = load_artefacts()

  tfidf = pipeline.named_steps["tfidf"]
  clf = pipeline.named_steps["clf"]

  sample = X_test_global.sample(
    n=min(n_samples, len(X_test_global)),
    random_state=42
  )

  X_vec = tfidf.transform(sample)

  background_texts = X_test_global.sample(
    n=min(1000, len(X_test_global)),
    random_state=0
  )

  background_matrix = tfidf.transform(background_texts)

  if background_matrix.nnz == 0:
    raise ValueError(
      "TF-IDF background matrix is all zeros. "
      "X_test still does not contain proper text."
    )

  explainer = shap.LinearExplainer(clf, background_matrix)
  shap_values = explainer.shap_values(X_vec)

  mean_abs = np.abs(shap_values).mean(axis=0)
  top_idx = np.argsort(mean_abs)[::-1][:20]
  feature_names = tfidf.get_feature_names_out()

  top_words = [feature_names[i] for i in top_idx]
  top_values = [mean_abs[i] for i in top_idx]

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.barh(top_words[::-1], top_values[::-1], color="#4575b4", alpha=0.8)
  ax.set_xlabel("Mean |SHAP value| across test sample")
  ax.set_title("Global feature importance (top 20 words)")
  plt.tight_layout()
  plt.savefig("../models/global_shap_summary.png", dpi=150, bbox_inches="tight")
  plt.close()

  return explainer, tfidf

if __name__ == "__main__":
  run_batch_analysis()