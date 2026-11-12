"""FastAPI service. Start: uvicorn api.main:app --reload"""
import io
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.inference import MODEL_FILE, LABELS_FILE, predict_image_bytes
from src.advice import get_advice

app = FastAPI(title="Potato Leaf CNN", version="1.0")
MAX_UPLOAD = 5 * 1024 * 1024


@app.get("/health")
def health():
    assets_exist = MODEL_FILE.is_file() and LABELS_FILE.is_file()
    return {"status": "model-files-found" if assets_exist else "train-model-first"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        with Image.open(io.BytesIO(content)) as check:
            check.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Upload a valid JPG or PNG image")
    try:
        result = predict_image_bytes(content)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    result["advice"] = get_advice(result["label"])
    result["disclaimer"] = (
        "Educational prototype only. Results are not confirmed plant disease diagnoses."
    )
    return result
