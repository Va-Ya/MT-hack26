import numpy as np
import pandas as pd
from ml.features import FeatureBuilder
from ml.features_v4 import augment,augment_frame,FEATURES
from ml.train_v4 import choose_strategies
from tests.test_ml_causality import context,traffic,schedule,same,T
from ml.fast_features import FastFeatureBuilder
import pytest

@pytest.mark.parametrize('variant',range(8))
def test_numpy_feature_contract(variant):
    f=traffic();ctx=context()
    if variant==0:f=f.iloc[:0]
    if variant==1:f.loc[f.index[:15],'location_valid']=False
    if variant==2:
        dup=f.iloc[:8].copy();dup['packet_id']='zz';dup['location_valid']=False
        f=pd.concat([f,dup])
    if variant==3:
        f.loc[f.index[:10],'receive_time']=T+pd.Timedelta(minutes=1)
        f.loc[f.index[10:20],'gps_time']=T+pd.Timedelta(minutes=1)
        f.loc[f.index[20:25],'receive_time']=pd.NaT
    if variant==4:
        f.loc[f.index[:4],'speed']=[-1,150,np.nan,0]
        f.loc[f.index[4:8],'lat']=np.nan
    if variant==5:
        f=f.iloc[30:];ctx={**ctx,'cur_dev_s':10000}
    if variant==6:
        f['speed']=0;f['heading']=np.nan
    if variant==7:
        future=f.copy()
        for c in ['event_time','receive_time','gps_time']:future[c]+=pd.Timedelta(hours=2)
        f=pd.concat([f,future])
    a=FeatureBuilder(f,schedule()).build(ctx);b=FastFeatureBuilder(f,schedule()).build(ctx)
    pd.testing.assert_series_equal(pd.Series(a).sort_index(),pd.Series(b).sort_index(),check_exact=False,rtol=1e-12,atol=1e-10)

def test_v4_future_packets_and_target_labels_are_irrelevant():
    f=traffic();future=f.copy()
    for col in ['event_time','receive_time','gps_time']:future[col]+=pd.Timedelta(hours=1)
    future['speed']=120
    a=FeatureBuilder(f,schedule()).build(context())
    b=FeatureBuilder(pd.concat([f,future]),schedule()).build(context())
    same(augment(a),augment({**b,'target_delay_s':99999,'target_event_time':T}))

def test_v4_online_and_batch_features_match_with_missing_values():
    rows=[FeatureBuilder(traffic(),schedule()).build(context()),FeatureBuilder(pd.DataFrame(),schedule()).build(context())]
    f=augment_frame(pd.DataFrame(rows))
    for i,row in enumerate(rows):same(pd.Series(augment(row)).sort_index(),f.iloc[i][FEATURES].sort_index().rename(None))
    numeric=f.select_dtypes(include='number').to_numpy()
    assert not np.isinf(numeric).any()

def test_blending_uses_complementary_oof_errors_and_fast_tolerance():
    records=[dict(candidate='a',cv_mae=2.,complexity=10),dict(candidate='b',cv_mae=2.,complexity=20),dict(candidate='small',cv_mae=2.5,complexity=1)]
    oof={'a':np.array([2.,-2.]),'b':np.array([-2.,2.]),'small':np.array([2.5,2.5])}
    strategies,scores=choose_strategies(records,oof,np.zeros(2))
    assert scores['quality']==0
    assert len(strategies['quality'])==2
    assert strategies['fast']==[('small',1.)]

def test_heterogeneous_saved_models_single_batch_parity(tmp_path):
    import json
    from catboost import CatBoostRegressor
    from ml.model import Predictor
    frame=pd.DataFrame({'cur_dev_s':[-30.,0.,10.,20.,50.,-5.],'vehicle_key':['a','b','a','b','a','c']})
    extra=augment_frame(frame)
    first=CatBoostRegressor(iterations=8,depth=2,verbose=False,allow_writing_files=False,thread_count=1)
    first.fit(frame,[1.,2.,3.,4.,5.,6.],cat_features=['vehicle_key'])
    second=CatBoostRegressor(iterations=8,depth=2,verbose=False,allow_writing_files=False,thread_count=1)
    second.fit(extra[['v4_delay_abs']],[-1.,3.,2.,5.,8.,9.])
    first.save_model(str(tmp_path/'best_model.cbm'));second.save_model(str(tmp_path/'second.cbm'))
    metadata=dict(feature_version='causal-v4',model_version='test',features=['cur_dev_s','vehicle_key','v4_delay_abs'],categorical_features=['vehicle_key'],members=[dict(file='best_model.cbm',features=['cur_dev_s','vehicle_key'],weight=.4,residual=True,target_scale=60),dict(file='second.cbm',features=['v4_delay_abs'],weight=.6,residual=False,target_scale=30)])
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    runtime=Predictor(tmp_path)
    expected=.4*(first.predict(frame)*60+frame.cur_dev_s.to_numpy())+.6*second.predict(extra[['v4_delay_abs']])*30
    np.testing.assert_allclose(runtime.predict_frame(frame),expected)
    np.testing.assert_allclose([runtime.predict(r) for r in frame.to_dict('records')],expected)
