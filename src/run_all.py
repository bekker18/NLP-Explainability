"""Run the full pipeline: train → SHAP analysis → launch Gradio demo."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from train import train
from explain import run_batch_analysis

if __name__ == "__main__":
  print("=" * 50)
  print("STEP 1: Training")
  print("=" * 50)
  train()

  print("\n" + "=" * 50)
  print("STEP 2: SHAP batch analysis")
  print("=" * 50)
  run_batch_analysis()

  print("\n" + "=" * 50)
  print("STEP 3: Launching Gradio demo")
  print("=" * 50)
  os.chdir(os.path.join(os.path.dirname(__file__), ".."))
  os.system("python app/demo.py")