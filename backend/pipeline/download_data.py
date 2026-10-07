"""
Reproducible Data Ingestion Script for hERG Benchmark Datasets.
Sources:
1. Karim et al. (2021) / Therapeutics Data Commons (TDC) hERG dataset: Harvard Dataverse ID 6822246
   - 13,445 curated molecules with binary hERG activity classification (IC50 <= 10 uM -> 1, IC50 > 10 uM -> 0)
2. Doddareddy et al. (2010) / TDC Tox.herg dataset: Harvard Dataverse ID 4259588
   - Independent external benchmark dataset (655 molecules)
"""

import os
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "herg_karim.tsv": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/6822246",
        "description": "Primary dataset (Karim et al. 2021, TDC) ~13,445 compounds",
    },
    "herg_doddareddy.tsv": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/4259588",
        "description": "Independent external validation dataset (Doddareddy et al. 2010, TDC) ~655 compounds",
    }
}


def download_dataset(filename: str, info: dict) -> Path:
    dest_path = DATA_DIR / filename
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[SKIP] {filename} already exists ({dest_path.stat().st_size} bytes).")
        return dest_path

    print(f"[DOWNLOAD] Fetching {filename} from {info['url']}...")
    resp = requests.get(info["url"], stream=True, timeout=60)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    print(f"[SUCCESS] Saved {filename} ({dest_path.stat().st_size} bytes) to {dest_path}")
    return dest_path


def verify_downloads():
    for filename, info in SOURCES.items():
        path = download_dataset(filename, info)
        df = pd.read_csv(path, sep="\t")
        print(f"Verified {filename}: {df.shape[0]} rows, columns: {df.columns.tolist()}")
        print(f"  Distribution: {df['Y'].value_counts().to_dict()}")


if __name__ == "__main__":
    verify_downloads()
