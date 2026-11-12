"""Prepare project folders and safe missing boilerplate files without overwrite."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDERS = [
    "data/raw", "data/split", "data/unseen", "models", "reports",
    "notebooks", "api", "frontend", "tests", "src"
]

# The downloadable starter project already contains functional source files.
# This script creates missing structural files only; it never replaces lesson code.
FILES = {
    "src/__init__.py": "",
    "api/__init__.py": "",
    "frontend/__init__.py": "",
    "models/.gitkeep": "",
    "reports/.gitkeep": "",
    "data/raw/.gitkeep": "",
    "data/split/.gitkeep": "",
    "data/unseen/.gitkeep": "",
    ".env.example": "GEMINI_API_KEY=\nGEMINI_MODEL=gemini-3.5-flash-lite\nAPI_URL=http://127.0.0.1:8000/predict\n",
    "README.md": "# Potato Leaf CNN\n\nAdd setup and usage instructions here.\n",
    ".gitignore": ".venv/\n.env\n__pycache__/\ndata/raw/\ndata/split/\ndata/unseen/\nmodels/*.keras\nmodels/*.json\n",
    "requirements.txt": "# Install the dependencies specified in the distributed starter project.\n",
}

for folder in FOLDERS:
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"Folder ready: {path.relative_to(ROOT)}")

for relative_path, starter_text in FILES.items():
    path = ROOT / relative_path
    if path.exists():
        print(f"Existing file kept: {relative_path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(starter_text, encoding="utf-8")
        print(f"Created missing file: {relative_path}")

print("Project structure ready. Nothing was overwritten.")
