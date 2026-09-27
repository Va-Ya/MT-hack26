from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from ml.streaming import StreamingPredictor
from ml.features import FeatureBuilder
from ml.model import Predictor
from tests.test_no_future_leakage import T, CTX, telemetry

class ConstantModel:
    version="test"
    def predict(self, features): return 200.0

def engine(tmp_path):
    schedule=pd.DataFrame([dict(tr_id=1,tt_action_item_id=20,time_begin=CTX["target_time_begin"],geom=CTX["geom"],building_address="test")])
    return StreamingPredictor(schedule,ConstantModel(),tmp_path/"log.parquet")

def test_full_log_flushes_every_200_new_predictions(tmp_path,monkeypatch):
    e=engine(tmp_path)
    e.log.extend([{} for _ in range(20000)])
    writes=[]
    monkeypatch.setattr(e,'flush',lambda:writes.append(len(e.log)))
    for _ in range(201):e.predict_at(CTX)
    assert writes==[20000]
    assert e.pending_log_writes==1

def test_issued_prediction_is_immutable_and_fact_delayed(tmp_path):
    e=engine(tmp_path)
    for row in telemetry().sort_values("event_time").to_dict("records"): e.ingest(row)
    p=e.predict_at(CTX).copy()
    assert all(r["actual"] is None for r in e.log)
    fact=CTX["target_time_begin"]+pd.Timedelta(seconds=60)
    with pytest.raises(ValueError): e.observe(dict(tr_id=1,target_stop_id=20,actual_time=fact,observed_at=T))
    e.observe(dict(tr_id=1,target_stop_id=20,actual_time=fact,observed_at=fact))
    assert e.log[-1]["actual"]==60
    assert e.log[-1]["absolute_error"]==140
    assert e.log[-1]["lead_time_seconds"]==720
    assert p["predicted_delay_s"]==200
    e.flush()
    assert pd.read_parquet(tmp_path/"log.parquet").iloc[-1].actual==60


def test_actual_lead_metric_distinguishes_horizon_and_after_event(tmp_path):
    e=engine(tmp_path)
    e.log.extend(dict(actual=10,absolute_error=1,lead_time_seconds=lead) for lead in [-1,0,599,600,720,900,901])
    result=e.metrics()['actual_lead_time']
    assert result['matched_predictions']==7
    assert result['within_10_15_minutes']==3
    assert result['after_event']==2
    assert result['p50_seconds']==600

def test_history_bounded_and_duplicates_idempotent(tmp_path):
    e=engine(tmp_path); row=telemetry().iloc[-1].to_dict()
    e.ingest(row)
    assert not e.ingest(row)["accepted"]
    later={**row,"event_time":T+pd.Timedelta(hours=1),"receive_time":T+pd.Timedelta(hours=1),"gps_time":T+pd.Timedelta(hours=1),"packet_id":"new"}
    e.ingest(later)
    assert len(e.history["1"])==1

def test_online_offline_parity(tmp_path):
    from ml.build_dataset import load_contexts
    data=Path("data/raw")
    ctx=load_contexts(data,"test").iloc[40].to_dict()
    traffic=pd.read_csv(data/"test/traffic.csv",low_memory=False)
    traffic=traffic[traffic.tr_id==ctx["tr_id"]]
    model=Predictor()
    schedule=pd.read_csv(data/"test/schedule.csv")
    expected=model.predict(FeatureBuilder(traffic,schedule).build(ctx))
    e=StreamingPredictor(schedule,model,tmp_path/"log.parquet")
    times=pd.to_datetime(traffic.event_time,format="mixed")
    history=traffic[(times<=ctx["T"])&(times>=ctx["T"]-pd.Timedelta(minutes=20))].copy()
    history=history.astype(object).where(pd.notna(history),None)
    # Seed the identical buffer; ingestion itself is tested separately.
    e.history[str(ctx["tr_id"])]=[{**r,"event_time":pd.Timestamp(r["event_time"])} for r in history.to_dict("records")]
    assert e.predict_at(ctx)["predicted_delay_s"]==pytest.approx(expected)

def test_api_validation_and_empty_state(tmp_path,monkeypatch):
    from backend import app as module
    monkeypatch.setenv("INGEST_TOKEN","test-token")
    monkeypatch.setenv("PREDICTION_LOG_PATH",str(tmp_path/"api.parquet"))
    with TestClient(module.app) as client:
        module.engine.log_path=tmp_path/"api.parquet"
        assert client.get("/health").json()["status"]=="ok"
        assert client.get("/vehicles/missing").status_code==404
        assert client.get("/network/heatmap?bbox=nan,0,1,1").status_code==422
        assert client.get("/network/heatmap?horizon_minutes=11").status_code==422
        assert client.post("/telemetry",json={}).status_code==401
        assert client.post("/telemetry",json={},headers={"X-Ingest-Token":"test-token"}).status_code==422
        assert client.get("/metrics").json()["total_ms"] is None
        assert client.get("/docs").status_code==200
