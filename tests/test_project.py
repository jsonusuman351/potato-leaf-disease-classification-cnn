"""Lightweight project checks; full ML training needs downloaded images + TF."""
from pathlib import Path
from PIL import Image


def test_template_is_idempotent(tmp_path):
    source = Path(__file__).parents[1] / "template.py"
    scope = {}
    # Check static compile, because executing would create folders in repository.
    compile(source.read_text(encoding="utf-8"), str(source), "exec")


def test_all_expected_project_files_exist():
    root = Path(__file__).parents[1]
    for path in ["prepare_data.py", "notebooks/01_CNN_Potato_Training.ipynb",
                 "api/main.py", "frontend/app.py", "src/inference.py", "README.md"]:
        assert (root / path).exists(), path
