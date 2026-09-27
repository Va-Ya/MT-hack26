import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ml.build_dataset import build
from ml.model import Predictor

def make_submission(data=Path("data/raw")):
    frame=build(data,"validate")
    pred=Predictor().predict_frame(frame)
    template=pd.read_csv(data/"sample_submission.csv",sep=";")
    if frame.sample_id.duplicated().any() or template.sample_id.duplicated().any(): raise ValueError("Duplicate sample IDs")
    if set(frame.sample_id)!=set(template.sample_id): raise ValueError("Sample coverage mismatch")
    lookup=dict(zip(frame.sample_id,pred))
    template["prediction"]=template.sample_id.map(lookup)
    if list(template.columns)!=["sample_id","prediction"] or not np.isfinite(template.prediction).all(): raise ValueError("Invalid submission")
    template.to_csv("artifacts/submission.csv",sep=";",index=False,encoding="utf-8")
    print(f"Verified submission: {len(template)} unique samples, all finite")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--data",type=Path,default=Path("data/raw")); args=parser.parse_args()
    make_submission(args.data)
