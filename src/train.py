import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  classification_report, confusion_matrix, f1_score
)

from data_loader import load_raw_data, preprocess

def build_pipeline() -> Pipeline:
  """Define the sklearn Pipeline."""

  return Pipeline([
    ('tfidf', TfidfVectorizer(
      max_features=30_000,  # vocabulary cap
      ngram_range=(1, 2),  # unigrams + bigrams
      sublinear_tf=True,  # log-scale TF, hepls with long docs
      min_df=3,  # ignore very rare terms
    )),
    ('clf', LogisticRegression(
      max_iter=1000,
      C=5.0,  # inverse regularization strength
      solver='lbfgs',
      n_jobs=-1
    )),
  ])

def plot_confusion_matrix(y_test, y_pred, save_path: str):
  """Save a labelled confusion matrix PNG."""

  cm = confusion_matrix(y_test, y_pred)
  fig, ax = plt.subplots(figsize=(5, 4))
  sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Fake', 'Real'],
    yticklabels=['Fake', 'Real'],
    ax=ax
  )
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  ax.set_title('Confusion Matrix')
  plt.tight_layout()
  plt.savefig(save_path, dpi=150)
  plt.close()
  print(f'Confusion matrix saved to {save_path}')

def train(fake_path='data/Fake.csv', real_path='data/Real.csv'):
  # Load & preprocess
  print('Loading data...')
  df = load_raw_data(fake_path, real_path)
  df = preprocess(df)
  print(f'Dataset size: {len(df):,} rows -'
        f'{df['label'].value_counts().to_dict()}')
  
  # Split
  X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
  )
  print(f'Train: {len(X_train):,} Test: {len(y_train):,}')

  # Train
  print('Training pipeline...')
  pipeline = build_pipeline()
  pipeline.fit(X_train, y_train)

  # Evaluate
  y_pred = pipeline.predict(X_test)
  f1 = f1_score(y_test, y_pred)
  print(f'\nTest F1: {f1:4f}')
  print(classification_report(y_test,y_pred,target_names=['Fake', 'Real']))

  # Persist artefacts
  plot_confusion_matrix(y_test, y_pred, 'models/confusion_matrix.png')
  joblib.dump(pipeline, 'models/pipeline.joblib')
  print('Model saved to models/pipeline.joblib')

  # Also save the test split for SHAP analysis
  X_test.to_csv('data/X_test.csv', index=False)
  y_test.to_csv('data/y_test.csv', index=False)

  return pipeline, X_test, y_test

if __name__ == '__main__':
  train()
