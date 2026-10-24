"""Submit prediction files with the Kaggle CLI and record the scores.

    python -m airline.submit logreg lgbm_raw stack ...   # submit, then record scores
    python -m airline.submit --scores-only               # just refresh reports/leaderboard.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .data import COMPETITION, ROOT

SUBS = ROOT / "submissions"
OUT = ROOT / "reports" / "leaderboard.json"
TEST_ROWS = 299_844


def kaggle() -> str:
    return shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")


def submit(name: str) -> None:
    path = SUBS / f"{name}.csv"
    rows = path.read_text().strip().splitlines()
    assert rows[0] == "id,satisfaction" and len(rows) == TEST_ROWS + 1, f"bad file {path}"
    subprocess.run([kaggle(), "competitions", "submit", "-c", COMPETITION, "-f", str(path),
                    "-m", name], check=True)


def scores() -> list[dict]:
    out = subprocess.run([kaggle(), "competitions", "submissions", "-c", COMPETITION, "--csv"],
                         check=True, capture_output=True, text=True).stdout
    start = out.find("fileName")
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    return [{"name": r["description"], "date": r["date"], "status": r["status"],
             "public_score": r["publicScore"], "private_score": r.get("privateScore", "")}
            for r in rows]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--scores-only", action="store_true")
    a = ap.parse_args()
    if not a.scores_only:
        for n in a.names:
            submit(n)
        time.sleep(60)
    rows = scores()
    OUT.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
