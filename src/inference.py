"""One shared inference routine for local CLI + FastAPI.

The trained Keras model contains Rescaling(1/255) internally.
Therefore inference must NOT divide the image by 255 again.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "potato_cnn.keras"
LABELS_FILE = ROOT / "models" / "class_names.json"
IMG_SIZE = (128, 128)


@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_FILE.is_file() or not LABELS_FILE.is_file():
        raise FileNotFoundError("Model missing. Run the training notebook first.")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_FILE)
    classes = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return model, classes


def predict_image_bytes(content: bytes) -> dict:
    model, classes = load_assets()
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB").resize(IMG_SIZE)
        x = np.expand_dims(np.asarray(image, dtype=np.float32), 0)
    probs = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probs))
    return {
        "label": classes[index].replace("Potato___", "").replace("_", " ").title(),
        "confidence": round(float(probs[index]), 4),
        "all_probabilities": {
            classes[i].replace("Potato___", "").replace("_", " ").title():
            round(float(p), 4) for i, p in enumerate(probs)
        },
    }
