"""Test an unseen leaf photo without FastAPI or Streamlit.
Usage: python predict_local.py data/unseen/leaf.jpg
"""
import argparse
from pathlib import Path
from src.inference import predict_image_bytes
from src.advice import get_advice

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()
result = predict_image_bytes(args.image.read_bytes())
print("Predicted:", result["label"])
print("Model score:", f"{result['confidence']:.1%}")
print("General guidance:", get_advice(result["label"]))
