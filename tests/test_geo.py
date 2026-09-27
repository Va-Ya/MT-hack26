import pandas as pd
from backend.geo import aggregate, risk

def test_score_is_bounded_and_explained():
    score,factors=risk(10000,10000,1000,999,"forecast")
    assert score==100
    assert sum(factors.values())==100
    assert risk(-500,-600,0,0,"forecast")[0]==0

def test_current_forecast_and_staleness():
    t=pd.Timestamp('2026-01-06 12:00:00')
    v=dict(tr_id='1',lat=55.75,lon=37.61,speed=20,current_delay=0,current_delay_known=True,location_timestamp=str(t),prediction=dict(timestamp=str(t),predicted_delay_s=600,prediction_horizon_s=660))
    now=aggregate([v],t,'current')[0]
    future=aggregate([v],t,'forecast')[0]
    assert now['risk_score']==0
    assert future['risk_score']>now['risk_score']
    assert future['routes_available'] is False
    assert future['vehicles']==1
    assert aggregate([v],t+pd.Timedelta(minutes=6),'forecast')==[]
    assert aggregate([v],t,'forecast',[38,55,39,56])==[]
