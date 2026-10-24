"""Load the competition files and the original survey they were synthesised from."""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ORIGINAL = ROOT / "data" / "original"
COMPETITION = "playground-series-s6e10"
ORIGINAL_SLUG = "teejmahal20/airline-passenger-satisfaction"
TARGET = "satisfaction"

RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
FEATURES = NUMERIC + RATINGS + CATEGORICAL


def load_competition(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (train features, train labels as 0/1, test features); test keeps its id."""
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    y = train[TARGET].astype(int).rename(TARGET)
    return train[["id"] + FEATURES], y, test[["id"] + FEATURES]


def load_original(orig_dir: Path = ORIGINAL) -> tuple[pd.DataFrame, pd.Series]:
    """The 129,880-passenger survey (its own train+test), mapped to the competition schema.

    The competition dropped the original 'Inflight service' rating, so it is dropped here too.
    """
    df = pd.concat(
        [pd.read_csv(orig_dir / "train.csv"), pd.read_csv(orig_dir / "test.csv")],
        ignore_index=True,
    )
    y = (df[TARGET] == "satisfied").astype(int).rename(TARGET)
    return df[FEATURES].reset_index(drop=True), y.reset_index(drop=True)


def _kaggle(*args: str) -> None:
    subprocess.run(["kaggle", *args], check=True)


def download() -> None:
    """Fetch both sources with the Kaggle CLI (needs accepted competition rules + API key)."""
    RAW.mkdir(parents=True, exist_ok=True)
    _kaggle("competitions", "download", "-c", COMPETITION, "-p", str(RAW))
    archive = RAW / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW)
    archive.unlink()
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    _kaggle("datasets", "download", "-d", ORIGINAL_SLUG, "-p", str(ORIGINAL), "--unzip")


if __name__ == "__main__":
    download()
