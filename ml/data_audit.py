"""Reproducible full CSV audit; never infers trip/route keys from proximity."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def dates(series):
    return pd.to_datetime(series, format="mixed", errors="coerce")


def audit(data, output):
    output.mkdir(parents=True, exist_ok=True)
    tables, profiles = {}, {}
    for path in sorted(data.rglob("*.csv")):
        name = path.relative_to(data).as_posix()
        frame = pd.read_csv(path, sep=";" if path.name == "sample_submission.csv" else ",", low_memory=False)
        tables[name] = frame
        profile = {"shape": list(frame.shape), "columns": list(frame.columns),
                   "dtypes": frame.dtypes.astype(str).to_dict(),
                   "missing": frame.isna().sum().astype(int).to_dict(),
                   "duplicates": int(frame.duplicated().sum()),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "unique_ids": {c: int(frame[c].nunique()) for c in frame if c.endswith("_id")},
                   "time_ranges": {}, "coordinate_ranges": {}}
        for c in frame:
            if c in ["T", "event_time", "time_begin", "time_fact_begin", "target_time_begin", "gps_time", "receive_time", "order_date"]:
                dt = dates(frame[c])
                profile["time_ranges"][c] = {"min": str(dt.min()), "max": str(dt.max()), "unparseable": int((frame[c].notna() & dt.isna()).sum())}
            if c in ["lon", "lat", "speed"]:
                profile["coordinate_ranges"][c] = {"min": float(frame[c].min()), "max": float(frame[c].max())}
        if "event_time" in frame:
            profile["duplicate_vehicle_time"] = int(frame.duplicated(["tr_id", "event_time"]).sum())
            profile["invalid_location_rows"] = int((frame.location_valid != True).sum())
        profiles[name] = profile
    checks = {}
    for split, point_path, sched_path in [("train", "labels/labels_train.csv", "train/schedule.csv"), ("test", "labels/labels_test.csv", "test/schedule.csv"), ("validate", "validate/points.csv", "validate/schedule_plan.csv")]:
        pts, schedule = tables[point_path], tables[sched_path]
        joined = pts.merge(schedule, left_on=["tr_id", "target_stop_id"], right_on=["tr_id", "tt_action_item_id"], how="left", validate="many_to_one", indicator=True)
        horizon = (dates(pts.target_time_begin) - dates(pts["T"])).dt.total_seconds()
        item = {"samples": len(pts), "missing_target_join": int((joined._merge != "both").sum()),
                "plan_time_mismatch": int((dates(joined.target_time_begin) != dates(joined.time_begin)).sum()),
                "horizon_seconds": horizon.describe().to_dict(), "horizon_outside_10_15": int((~((horizon > 600) & (horizon <= 900))).sum()),
                "unknown_vehicles": int((~pts.tr_id.isin(tables[split + "/traffic.csv"].tr_id)).sum())}
        if "target_delay_s" in pts:
            computed = (dates(joined.time_fact_begin) - dates(joined.time_begin)).dt.total_seconds()
            item["target_mismatch"] = int(((computed - joined.target_delay_s).abs() > 1e-6).sum())
            item["missing_target_fact"] = int(computed.isna().sum())
            item["target_distribution"] = pts.target_delay_s.describe().to_dict()
        checks[split] = item
    checks["test_validate_identical_traffic"] = profiles["test/traffic.csv"]["sha256"] == profiles["validate/traffic.csv"]["sha256"]
    checks["test_validate_sample_overlap"] = len(set(tables["labels/labels_test.csv"].sample_id) & set(tables["validate/points.csv"].sample_id))
    checks["test_validate_target_overlap"] = len(set(tables["labels/labels_test.csv"].target_stop_id) & set(tables["validate/points.csv"].target_stop_id))
    checks["train_test_sample_overlap"] = len(set(tables["labels/labels_train.csv"].sample_id) & set(tables["labels/labels_test.csv"].sample_id))
    checks["submission_matches_points"] = set(tables["sample_submission.csv"].sample_id) == set(tables["validate/points.csv"].sample_id)
    result = {"files": profiles, "checks": checks}
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("Full profiles:", output / "audit.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    audit(args.data, args.output)
