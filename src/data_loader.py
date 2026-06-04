import pandas as pd
import re

def load_raw_data(fake_path: str, real_path: str) -> pd.DataFrame:
  """Load and merge fake/real CSVs into one labelled dataframe"""

  fake = pd.read_csv(fake_path)
  real = pd.read_csv(real_path)

  fake["label"] = 0  # 0 = Fake
  real["label"] = 1  # 1 = Real

  df = pd.concat([fake, real], ignore_index=True)

  # Combine title + body for richer text signals
  df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
  df = df[["text", "label"]].dropna()

  return df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

def clean_text(text: str) -> str:
  """Basic text cleaning: lowercase, strip URLs, extra spaces"""

  text = text.lower()
  text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
  text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
  text = re.sub(r"\s+", "", text).strip()  # collapse whitespace

  return text

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  """Apply cleaning to the text column."""

  df = df.copy()

  df["text"] = df["text"].apply(clean_text)

  return df