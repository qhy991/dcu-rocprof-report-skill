"""Contract checks for multi-dispatch rocprof counter aggregation."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "helpers/analyze_csv.py"


class AnalyzeCsvTests(unittest.TestCase):
    def test_percentages_are_mean_and_wave_expectation_covers_all_dispatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            fields = ["Index", "KernelName", "grd", "wgr", "lds", "arch_vgpr",
                      "sgpr", "wave_size", "Wavefronts", "FETCH_SIZE", "GPUBusy"]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index, busy in enumerate((50, 100)):
                    writer.writerow({"Index": index,
                                     "KernelName": "Rmsnorm2dFwd [clone .kd]",
                                     "grd": 64, "wgr": 64, "lds": 0,
                                     "arch_vgpr": 32, "sgpr": 32,
                                     "wave_size": 64, "Wavefronts": 1,
                                     "FETCH_SIZE": 10, "GPUBusy": busy})
            result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                    text=True, capture_output=True, check=True)
            self.assertIn("2 dispatch(es)", result.stdout)
            self.assertIn("GPUBusy           75.0 (per-dispatch mean, n=2)",
                          result.stdout)
            self.assertIn("Wavefronts 2 vs summed grd/wave_size 2 -> OK",
                          result.stdout)
            self.assertIn("total traffic:   unavailable", result.stdout)
            self.assertNotIn("WRITE_SIZE:      0 KB", result.stdout)
            self.assertNotIn("bandwidth floor:", result.stdout)


if __name__ == "__main__":
    unittest.main()
