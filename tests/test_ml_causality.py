"""Standalone tests shipped with the Kaggle ML package."""
import numpy as np
import pandas as pd
import pytest
from ml.features import FeatureBuilder,timestamp
from ml.kaggle_train import past,columns

T=pd.Timestamp('2026-01-06 14:00:00')
def context():return dict(tr_id='1',T=T,target_stop_id=3,target_time_begin=T+pd.Timedelta(minutes=11),cur_dev_s=30,geom='POINT (37.61 55.75)')
def traffic():
    return pd.DataFrame([dict(tr_id='1',event_time=T-pd.Timedelta(seconds=i*15),receive_time=T-pd.Timedelta(seconds=i*15),gps_time=T-pd.Timedelta(seconds=i*15),location_valid=True,lat=55.75,lon=37.6-i*.0001,speed=20+i*.1,heading=90,packet_id=str(i)) for i in range(80)])
def schedule():
    return pd.DataFrame([dict(tr_id='1',tt_action_item_id=i,time_begin=T+pd.Timedelta(minutes=offset),geom=f'POINT ({37.59+i*.01} 55.75)',time_fact_begin=T+pd.Timedelta(days=3)) for i,offset in [(1,-5),(2,5),(3,11)]])
def same(a,b):pd.testing.assert_series_equal(pd.Series(a),pd.Series(b))
def test_1400_never_sees_1401_to_1600():
    f=traffic();future=f.copy()
    for c in ['event_time','receive_time','gps_time']:future[c]+=pd.Timedelta(hours=2)
    future['speed']=129;future['lon']=38
    a=FeatureBuilder(f,schedule());b=FeatureBuilder(pd.concat([f,future]),schedule())
    same(a.build(context()),b.build(context()));np.testing.assert_array_equal(a.sequence(context()),b.sequence(context()))
def test_late_received_or_gps_timestamp_is_not_available():
    f=traffic();late=f.copy();late['speed']=129;late['receive_time']=T+pd.Timedelta(seconds=1)
    same(FeatureBuilder(f).build(context()),FeatureBuilder(pd.concat([f,late])).build(context()))
    late['receive_time']=T;late['gps_time']=T+pd.Timedelta(seconds=1)
    same(FeatureBuilder(f).build(context()),FeatureBuilder(pd.concat([f,late])).build(context()))
def test_fact_columns_are_never_features():
    s=schedule();modified=s.copy();modified['time_fact_begin']=T-pd.Timedelta(days=50)
    a=FeatureBuilder(traffic(),s);b=FeatureBuilder(traffic(),modified)
    same(a.build(context()),b.build({**context(),'target_delay_s':1e9,'time_fact_begin':T-pd.Timedelta(days=1)}))
def test_only_observed_labels_enter_training_fold():
    frame=pd.DataFrame({'T':[T-pd.Timedelta(minutes=30)]*3,'target_time_begin':[T-pd.Timedelta(minutes=10),T+pd.Timedelta(seconds=1),T-pd.Timedelta(minutes=10)],'target_event_time':[T-pd.Timedelta(minutes=5),T-pd.Timedelta(minutes=5),T+pd.Timedelta(seconds=1)]})
    assert past(frame,T).index.tolist()==[0]
@pytest.mark.parametrize('seconds',[0,600,901])
def test_strict_horizon(seconds):
    ctx={**context(),'target_time_begin':T+pd.Timedelta(seconds=seconds)}
    with pytest.raises(ValueError):FeatureBuilder(traffic()).build(ctx)
def test_empty_telemetry_has_explicit_masks():
    b=FeatureBuilder(pd.DataFrame(),schedule());f=b.build(context())
    assert f['gps_missing']==1 and f['telemetry_count_5m']==0
    assert b.sequence(context()).shape==(80,6)
def test_metadata_cannot_enter_model_columns():
    frame=pd.DataFrame(columns=['sample_id','tr_id','T','target_time_begin','target_event_time','target_delay_s','target_stop_id','prediction_horizon_seconds','cur_dev_s','speed_now'])
    assert columns(frame,'extended')==['cur_dev_s','speed_now']
def test_packet_at_T_is_eligible():
    assert FeatureBuilder(traffic()).build(context())['speed_now']==20

def test_compiled_plan_matches_dataframe_plan():
    plans=FeatureBuilder(pd.DataFrame(),schedule()).plans
    same(FeatureBuilder(traffic(),schedule()).build(context()),FeatureBuilder(traffic(),plans).build(context()))
def test_truncated_history_matches_full_dataset():
    all_rows=traffic();all_rows=pd.concat([all_rows,all_rows.assign(event_time=all_rows.event_time+pd.Timedelta(hours=1),receive_time=all_rows.receive_time+pd.Timedelta(hours=1),gps_time=all_rows.gps_time+pd.Timedelta(hours=1))])
    online=all_rows[pd.to_datetime(all_rows.event_time)<=T]
    same(FeatureBuilder(all_rows,schedule()).build(context()),FeatureBuilder(online,schedule()).build(context()))

def test_stateless_inference_filters_future_and_uses_plan_geometry(monkeypatch):
    from ml import inference
    class Echo:
        version='test'
        def __init__(self,path):pass
        def predict(self,features):return features['distance_to_target_stop']
    monkeypatch.setattr(inference,'Predictor',Echo)
    runtime=inference.CausalInference('unused',schedule())
    f=traffic();future=f.copy()
    for c in ['event_time','receive_time','gps_time']:future[c]+=pd.Timedelta(hours=2)
    future['lon']=38
    a=runtime.predict_at(f,context())
    b=runtime.predict_at(pd.concat([f,future]),{**context(),'geom':'POINT (0 0)','target_delay_s':999999})
    assert a==b

def test_inference_rejects_target_plan_mismatch(monkeypatch):
    from ml import inference
    class Dummy:
        def __init__(self,path):pass
    monkeypatch.setattr(inference,'Predictor',Dummy)
    runtime=inference.CausalInference('unused',schedule())
    with pytest.raises(ValueError,match='Target plan mismatch'):
        runtime.predict_at(traffic(),{**context(),'target_time_begin':T+pd.Timedelta(minutes=12)})

def test_ensemble_restores_seconds_and_categorical_columns(tmp_path,monkeypatch):
    import json
    from ml import model as module
    class Dummy:
        def load_model(self,path):pass
        def predict(self,frame,**kwargs):
            assert frame.vehicle_key.dtype==object or str(frame.vehicle_key.dtype).startswith('str')
            return np.ones(len(frame))
    monkeypatch.setattr(module,'CatBoostRegressor',Dummy)
    metadata={'features':['cur_dev_s','vehicle_key'],'categorical_features':['vehicle_key'],'model_version':'test','residual':True,'members':[{'file':'one.cbm','weight':.25,'residual':True,'target_scale':60},{'file':'two.cbm','weight':.75,'residual':True,'target_scale':120}]}
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    predictor=module.Predictor(tmp_path)
    np.testing.assert_allclose(predictor.predict_frame(pd.DataFrame({'cur_dev_s':[10.],'vehicle_key':[1]})),[115.])
