import os
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle import api

COMPETITION = "lnu-deep-learn-2-text-classification-2025"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_credentials():
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if user and key:
        return True

def download():
    if not ensure_credentials():
        raise SystemExit(
            "Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json in ~/.kaggle/"
        )
    # Download all files as zip archive
    api.competition_download_files(COMPETITION, path=str(OUT_DIR), quiet=False)
    # Optionally unzip
    for z in OUT_DIR.glob("*.zip"):
        import zipfile
        with zipfile.ZipFile(z) as zipf:
            zipf.extractall(OUT_DIR)
        z.unlink()

if __name__ == "__main__":
    download()
    print("Download complete:", OUT_DIR)