import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless - no display needed
import matplotlib.pyplot as plt

def load_artefacts():
  """Load pipeline and test data saved during training"""

  pipeline = joblib.load('models/pipeline.joblib')
  X_test = pd.read_csv('data/X_test.csv')
  y_test = pd.read_csv('data/y_test.csv')

  return pipeline, X_test, y_test

def get_explainer(pipeline):
  """
  Build a SHAP LinearExplainer directly on the TF-IDF + LR pipeline.
  
  LinearExplainer is exact (not sampling-based) and works because
  Logistic Regressor is a linear model
  """

  tfidf = pipeline.named_steps['tfidf']
  clf = pipeline.named_steps['clf']

  # Transform a backgroud sample so SHAP knows the features distribution
  # We use the mean of a 1000-sample background (standard practice)
  if X_test_global is None:
    raise ValueError('X_test is not loaded. Call load_artefacts() firts.')
  background_texts = X_test_global.sample(1000, random_state=42)
  background_matrix = tfidf.transform(background_texts)
  background_mean = shap.kmeans(background_matrix, 10)

  explainer = shap.LinearExplainer(clf, background_mean)

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
  label = 'Real' if pred == 1 else 'Fake'
  conf = proba[pred]

  # Map non-zero SHAP values back to feature names
  feature_names = tfidf.get_features_names_out()
  nonzero_idx = vec.nonzero()[1]  # indices of present features

  word_shap = [
    (feature_names[i], float(sv[i]))
    for i in nonzero_idx
    if sv[i] != 0
  ]
  word_shap.sort(key = lambda x: abs(x[1]), reverse=True)

  return label, conf, word_shap[:20]  # top-20 most influential words

def plot_top_words(word_shap, title='Top contributing words', save_path=None):
  """
  Horizontal bar chart - red = pushed toward Fake, blue = pushed toward Real.
  """

  words, values = zip(*word_shap[:15])
  colors = ['#d73027' if v < 0 else '#4575b4' for v in values]

  fig, ax = plt.subplots(figsize=(7, 4))
  bars = ax.barh(words, values, colors=colors)
  ax.axvline(0, color='black', linewidth=0.8)
  ax.set_xlabel('SHAP value (negative -> Fake | positive -> Real)')
  ax.set_title(title)
  ax.invert_yaxis()
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
  else:
    plt.show()

def run_batch_analysis(n_samples=200):
  """
  Compute SHAP valuesfor a sample of the test set and produce
  a global summery plot (which features matter most overall)
  """

  global pipeline, X_test_global, y_test_global
  pipeline, X_test_global, y_test_global = load_artefacts()

  tfidf = pipeline.named_steps['tfidf']
  clf = pipeline.named_steps['clf']
  sample = X_test_global.sample(n_samples, random_state=42)
  X_vec = tfidf.transform(sample)

  background = shap.kmeans(
    tfidf.transform(X_test_global.sample(1000, random_state=0)), 10
  )
  explainer = shap.LinearExplainer(clf, background)
  shap_values = explainer.shap_values(X_vec)  # (n_samples, vocab_size)

  # Global bar summary - top-20 features by mean |SHAP|
  mean_abs = np.abs(shap_values).mean(axis=0)
  top_idx = np.argsort(mean_abs)[::-1][:20]
  feature_names = tfidf.get_feature_names_out()

  top_words = [feature_names[i] for i in top_idx]
  top_values = [mean_abs[i] for i in top_idx]
  
  fig, ax = plt.subplots(figsize=(7, 5))
  ax.barh(top_words[::-1], top_values[::-1], color='#4575b4', alpha=0.8)
  ax.set_xlabel('Mean |SHAP value| across test sample')
  ax.set_title('Global feature importance (top 20 words)')
  plt.tight_layout()
  plt.savefig('models/global_shap_summary.png', dpi=150, bbox_inches='tight')
  plt.close()
  print('Global SHAP summary saved.')

  return explainer, tfidf

if __name__ == '__main__':
  run_batch_analysis()