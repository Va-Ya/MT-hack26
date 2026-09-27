"""Use supplied historical analogues of points; verify targets against schedule."""
import argparse
from pathlib import Path
import pandas as pd
from ml.features import FeatureBuilder


def load_contexts(data, split):
    is_submit = split == "validate"
    p = pd.read_csv(data / ("validate/points.csv" if is_submit else f"labels/labels_{split}.csv"))
    s = pd.read_csv(data / split / ("schedule_plan.csv" if is_submit else "schedule.csv"))
    # Facts are deliberately excluded from the feature context.
    p = p.merge(s[["tr_id", "tt_action_item_id", "time_begin", "geom", "building_address"]], left_on=["tr_id","target_stop_id"], right_on=["tr_id","tt_action_item_id"], validate="many_to_one", how="left")
    if p.tt_action_item_id.isna().any() or p.sample_id.duplicated().any():
        raise ValueError("Missing or ambiguous schedule join/sample ID")
    for c in ["T", "target_time_begin", "time_begin"]:
        p[c] = pd.to_datetime(p[c], format="mixed")
    if not (p.target_time_begin == p.time_begin).all():
        raise ValueError("Target plan mismatch")
    return p


def build(data, split):
    pts = load_contexts(data, split)
    telemetry = pd.read_csv(data / split / "traffic.csv", low_memory=False)
    schedule = pd.read_csv(data / split / ('schedule_plan.csv' if split == 'validate' else 'schedule.csv'), usecols=['tr_id','tt_action_item_id','time_begin','geom'])
    builder = FeatureBuilder(telemetry, schedule)
    records = []
    for ctx in pts.to_dict("records"):
        features = builder.build(ctx)
        row = {**features, **{k:ctx[k] for k in ["sample_id","tr_id","T","target_time_begin"]}}
        row["prediction_horizon_seconds"] = features["scheduled_time_to_target"]
        if "target_delay_s" in ctx:
            row["target_delay_s"] = ctx["target_delay_s"]
            row["target_event_time"] = ctx["target_time_begin"] + pd.Timedelta(seconds=ctx["target_delay_s"])
        records.append(row)
    return pd.DataFrame(records)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args=parser.parse_args()
    Path("artifacts").mkdir(exist_ok=True)
    for split in ["train","test","validate"]:
        frame=build(args.data, split)
        frame.to_parquet(f"artifacts/{split}_dataset.parquet", index=False)
        print(split, frame.shape, flush=True)
