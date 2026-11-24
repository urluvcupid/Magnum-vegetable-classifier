"""Smoke test: python -m unittest discover -s tests -v (no Kaggle download)."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PredictorCLITest(unittest.TestCase):
    def test_batch_csv_format_and_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            src = temp / "photos"
            (src / "nested").mkdir(parents=True)
            Image.new("RGB", (320, 200), (30, 130, 45)).save(src / "demo.jpg")
            Image.new("RGB", (250, 300), (190, 120, 40)).save(src / "nested" / "second.png")
            result_file = temp / "preds.csv"
            run = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "predict.py"),
                 "--input", str(src), "--output", str(result_file), "--device", "cpu"],
                cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stderr)
            with result_file.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, ["filename", "label", "confidence"])
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["filename"] for r in rows}, {"demo.jpg", "nested/second.png"})
            for row in rows:
                self.assertIn(row["label"], {"Capsicum","Carrot","Cucumber","Potato","Raddish","Squash"})
                self.assertTrue(0 <= float(row["confidence"]) <= 1)


if __name__ == "__main__": unittest.main()
