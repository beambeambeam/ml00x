"""Download the Task 1 Kaggle files into data/.

Reads KAGGLE_API_TOKEN from the .env file next to this script.

Usage (from task1/):
    uv run get_data.py           # skip files that already exist
    uv run get_data.py --force   # download every file again
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
COMPETITION = "player-intelligence-system-cpe-342-ml"
REMOTE_DIR = "cpe342-kaggle-main/01_binary_classification"
FILES = ["datacard.txt", "train.csv", "test.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download files that already exist")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit(f"KAGGLE_API_TOKEN is not set. Add it to {HERE / '.env'}")

    # The kaggle package authenticates on import, so import it after .env is loaded.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        if target.exists() and not args.force:
            print(f"skip {target.relative_to(HERE)} (exists)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            api.competition_download_file(COMPETITION, f"{REMOTE_DIR}/{name}", path=tmp, force=True, quiet=True)
            for zipped in Path(tmp).rglob("*.zip"):
                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(zipped.parent)
            downloaded = next(Path(tmp).rglob(name))
            shutil.move(downloaded, target)
        print(f"saved {target.relative_to(HERE)} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
