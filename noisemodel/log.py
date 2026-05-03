# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
import logging, sys
from pathlib import Path
import csv

def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )
    return log_dir


class CSVLogger:
    """Appends one row per step to a CSV file."""

    def __init__(self, path: Path):
        self.path = path
        self._file  = open(path, "w", newline="", buffering=1)
        self._writer = None

    def write(self, row: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self):
        self._file.close()