"""Correctness contracts: these four tests fail on a4433b1 before fixes.
Run from repository root: python -m pytest /path/to/test_review_regressions.py -q
"""
from collections import deque
import pandas as pd
import pytest
from tests.test_streaming_api import engine
from tests.test_no_future_leakage import T, CTX, telemetry
from ml.streaming import StreamingPredictor

def test_rejected_context_does_not_mutate_clock(tmp_path):
    e=engine(tmp_path)
    e.ingest(telemetry().iloc[0].to_dict())
    before=e.clock
    with pytest.raises(ValueError):
        e.predict_at({**CTX,'T':T+pd.Timedelta(days=1000)})
    assert e.clock==before, 'Rejected request changed the global clock'

def test_rejected_outcome_does_not_mutate_clock(tmp_path):
    e=engine(tmp_path)
    e.ingest(telemetry().iloc[0].to_dict())
    before=e.clock
    with pytest.raises(ValueError):
        e.observe(dict(tr_id='unknown',target_stop_id='unknown',actual_time=str(T+pd.Timedelta(days=1000)),observed_at=str(T+pd.Timedelta(days=1000))))
    assert e.clock==before, 'Rejected outcome changed the global clock'

def test_no_backdated_prediction_with_future_delay(tmp_path):
    s=pd.DataFrame([
        dict(tr_id=1,tt_action_item_id=10,time_begin=T-pd.Timedelta(seconds=60),geom=CTX['geom'],building_address='x'),
        dict(tr_id=1,tt_action_item_id=20,time_begin=CTX['target_time_begin'],geom=CTX['geom'],building_address='x')])
    class Echo:
        version='test'
        def predict(self,features):return features['cur_dev_s']
    e=StreamingPredictor(s,Echo(),tmp_path/'log.parquet')
    e.observe(dict(tr_id=1,target_stop_id=10,actual_time=str(T+pd.Timedelta(seconds=60)),observed_at=str(T+pd.Timedelta(seconds=60))))
    result=e.ingest(telemetry().iloc[0].to_dict())
    prediction=result.get('prediction')
    # An implementation may reject/buffer old data or issue a forecast at current time.
    # A historical forecast is legal only if it uses a historical delay snapshot.
    assert prediction is None or pd.Timestamp(prediction['timestamp'])>=T+pd.Timedelta(seconds=60) or prediction['predicted_delay_s']!=120

def test_full_log_does_not_flush_every_prediction(tmp_path):
    e=engine(tmp_path)
    e.log=deque([{}]*20000,maxlen=20000)
    calls=[]
    e.flush=lambda:calls.append(1)
    for _ in range(3):e.predict_at(CTX)
    assert len(calls)<=1, 'A saturated deque triggers a full Parquet rewrite per forecast'

def test_delay_is_available_at_observation_time_not_actual_time(tmp_path):
    s=pd.DataFrame([
        dict(tr_id=1,tt_action_item_id=10,time_begin=T-pd.Timedelta(minutes=2),geom=CTX['geom'],building_address='x'),
        dict(tr_id=1,tt_action_item_id=20,time_begin=CTX['target_time_begin'],geom=CTX['geom'],building_address='x')])
    e=StreamingPredictor(s,engine(tmp_path).model,tmp_path/'late.parquet')
    e.observe(dict(tr_id=1,target_stop_id=10,actual_time=T-pd.Timedelta(minutes=1),observed_at=T+pd.Timedelta(seconds=30)))
    before=e.context('1',T)
    assert before['cur_dev_s']==0 and not before['current_delay_known']
    after=e.context('1',T+pd.Timedelta(seconds=30))
    assert after['cur_dev_s']==60 and after['current_delay_known']
