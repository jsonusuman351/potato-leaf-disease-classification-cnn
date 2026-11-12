"""Tests using synthetic images and a fake model; never tests trained CNN quality."""
from pathlib import Path
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient


def test_clean_split_synthetic(tmp_path, monkeypatch):
    import prepare_data
    raw = tmp_path / "raw"
    split = tmp_path / "split"
    for ci, name in enumerate(prepare_data.CLASSES):
        folder = raw / name
        folder.mkdir(parents=True)
        for i in range(25):
            # Unique content for each image (avoids hash deduplication)
            arr = np.zeros((24, 24, 3), dtype=np.uint8)
            arr[..., :] = (i*7+ci, i*3, ci*80)
            Image.fromarray(arr).save(folder / f"{i:03d}.png")
        (folder / "broken.jpg").write_bytes(b"this is not a real image")
    monkeypatch.setattr(prepare_data, "RAW", raw)
    monkeypatch.setattr(prepare_data, "DEST", split)
    summary = prepare_data.prepare()
    for name in prepare_data.CLASSES:
        assert summary[name] == {"train": 17, "val": 3, "test": 5}
    # Repeat with no overwrite
    assert prepare_data.prepare() == {}
    assert len(list(split.rglob("*.jpg"))) == 75


def test_api_prediction_fake_model(monkeypatch):
    from api.main import app
    import src.inference as inference
    class FakeModel:
        def predict(self, x, verbose=0):
            assert x.shape == (1, 128, 128, 3)
            # No second scaling: API passes raw pixel values to model.
            assert float(x.max()) == 220.0
            return np.array([[.05, .90, .05]])
    monkeypatch.setattr(inference, "load_assets", lambda: (
        FakeModel(), ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    ))
    Image.new("RGB", (64, 64), (220, 10, 10)).save("/tmp/potato_smoke.png")
    content=Path("/tmp/potato_smoke.png").read_bytes()
    client=TestClient(app)
    result=client.post("/predict",files={"file": ("leaf.png",content,"image/png")})
    assert result.status_code==200, result.text
    payload=result.json()
    assert payload["label"]=="Late Blight"
    assert payload["confidence"]==0.9
    assert "agricultural" in payload["advice"].lower()
    bad=client.post("/predict",files={"file": ("x.jpg",b"not a picture","image/jpeg")})
    assert bad.status_code==400
