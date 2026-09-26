"""Compose preflight: verify checked-in model, or train reproducibly if absent."""
import json
import subprocess
import sys
from pathlib import Path
from ml.data_audit import audit
from ml.make_submission import make_submission


def main():
    data = Path("data/raw")
    required = ["train/traffic.csv", "train/schedule.csv", "labels/labels_train.csv",
                "test/traffic.csv", "test/schedule.csv", "labels/labels_test.csv",
                "validate/traffic.csv", "validate/schedule_plan.csv", "validate/points.csv", "sample_submission.csv"]
    missing = [str(data / p) for p in required if not (data / p).exists()]
    if missing:
        raise SystemExit("Extract the supplied dataset into data/raw first. Missing: " + ", ".join(missing))
    audit(data, Path("artifacts"))
    if not Path("models/best_model.cbm").exists() or not Path("models/metadata.json").exists():
        subprocess.run([sys.executable, "-m", "ml.build_dataset"], check=True)
        subprocess.run([sys.executable, "-m", "ml.train"], check=True)
    make_submission(data)
    print("ML preflight complete", flush=True)


if __name__ == "__main__":
    main()
