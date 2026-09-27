"""Purged temporal validation, bounded experiments and final train-only fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ml.features import FEATURE_VERSION

EXCLUDE={"sample_id","tr_id","T","target_time_begin","target_event_time","prediction_horizon_seconds","target_delay_s"}
def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

def train():
    data=pd.read_parquet("artifacts/train_dataset.parquet")
    test=pd.read_parquet("artifacts/test_dataset.parquet")
    cutoff=data["T"].quantile(.70)
    early=data[(data["T"]<cutoff)&(data.target_event_time<cutoff)&(data.target_time_begin<cutoff)]
    late=data[data["T"]>=cutoff]
    # Membership in official real-only test identifies real vehicles, not an inferred route.
    real=late.tr_id.isin(test.tr_id.unique()).to_numpy()
    if not real.any(): raise ValueError("No real validation samples")
    full=[c for c in data if c not in EXCLUDE]
    configs=[("context", ["cur_dev_s","hour","weekday","is_weekend","scheduled_time_to_target"], True),
             ("telemetry", [c for c in full if c!="target_stop_id"], True),
             ("telemetry_stop", full, True)]
    results=[]; candidates=[]
    Path("models").mkdir(exist_ok=True)
    for name, columns, residual in configs:
        cats=[c for c in columns if c=="target_stop_id"]
        model=CatBoostRegressor(iterations=1000,depth=6,learning_rate=.05,loss_function="MAE",eval_metric="MAE",random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        y=early.target_delay_s-early.cur_dev_s if residual else early.target_delay_s
        yv=late.target_delay_s-late.cur_dev_s if residual else late.target_delay_s
        model.fit(early[columns],y,cat_features=cats,eval_set=(late[columns],yv),early_stopping_rounds=80)
        pred=np.asarray(model.predict(late[columns]))+(late.cur_dev_s.to_numpy() if residual else 0)
        result=dict(experiment=name,iterations=model.tree_count_,temporal_mae=mae(late.target_delay_s,pred),temporal_real_mae=mae(late.target_delay_s.to_numpy()[real],pred[real]),baseline_mae=mae(late.target_delay_s,late.cur_dev_s),baseline_real_mae=mae(late.target_delay_s.to_numpy()[real],late.cur_dev_s.to_numpy()[real]))
        print(result,flush=True); results.append(result); candidates.append((result,model,columns,cats,residual,pred))
    pd.DataFrame(results).to_csv("artifacts/experiments.csv",index=False)
    result,model,columns,cats,residual,pred=min(candidates,key=lambda c:c[0]["temporal_real_mae"])
    model.save_model("models/temporal_model.cbm")
    evaluated=late[["sample_id","tr_id","T","target_stop_id","target_delay_s","cur_dev_s","target_event_time","prediction_horizon_seconds","speed_now"]].copy()
    evaluated["prediction"]=pred
    evaluated["absolute_error"]=np.abs(evaluated.target_delay_s-pred)
    evaluated.to_parquet("artifacts/temporal_predictions.parquet",index=False)
    pd.DataFrame({"feature":columns,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv("artifacts/feature_importance.csv",index=False)
    metrics={**result,"cutoff":str(cutoff),"early_count":len(early),"late_count":len(late),"late_real_count":int(real.sum()),"purged_count":int((data["T"]<cutoff).sum()-len(early))}
    late_test=test[test["T"]>=cutoff]
    if len(late_test):
        tp=model.predict(late_test[columns])+(late_test.cur_dev_s.to_numpy() if residual else 0)
        metrics["temporal_test_mae"]=mae(late_test.target_delay_s,tp)
        metrics["temporal_test_baseline_mae"]=mae(late_test.target_delay_s,late_test.cur_dev_s)
        metrics["temporal_test_count"]=len(late_test)
    final=CatBoostRegressor(iterations=max(50,model.tree_count_),depth=6,learning_rate=.05,loss_function="MAE",random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    final.fit(data[columns],data.target_delay_s-data.cur_dev_s if residual else data.target_delay_s,cat_features=cats)
    final.save_model("models/best_model.cbm")
    version=hashlib.sha256(Path("models/best_model.cbm").read_bytes()).hexdigest()[:12]
    metadata={"model_version":version,"feature_version":FEATURE_VERSION,"features":columns,"categorical_features":cats,"residual":residual,"horizon_seconds":[600,900],"training_source":"labels_train only", "metrics":metrics}
    Path("models/metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    diagnostic=final.predict(test[columns])+(test.cur_dev_s.to_numpy() if residual else 0)
    metrics["non_temporal_test_mae"]=mae(test.target_delay_s,diagnostic)
    metrics["non_temporal_test_baseline_mae"]=mae(test.target_delay_s,test.cur_dev_s)
    Path("artifacts/metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    rows=[]
    for name, values in [("hour",evaluated["T"].dt.hour), ("target_stop",evaluated.target_stop_id), ("vehicle",evaluated.tr_id), ("current_delay",pd.cut(evaluated.cur_dev_s,[-np.inf,-60,0,120,300,np.inf]).astype(str)), ("target_delay",pd.cut(evaluated.target_delay_s,[-np.inf,-60,0,120,300,np.inf]).astype(str))]:
        grouped=evaluated.assign(segment=values).groupby("segment").absolute_error.agg(["size","mean"])
        for key,row in grouped.iterrows(): rows.append({"dimension":name,"segment":str(key),"count":int(row["size"]),"mae":row["mean"]})
    pd.DataFrame(rows).to_csv("artifacts/error_analysis.csv",index=False)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=="__main__": train()
