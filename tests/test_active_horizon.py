import pandas as pd
from backend.geo import active_prediction

def test_forecast_expires_when_less_than_ten_minutes_remain():
    at=pd.Timestamp('2026-01-06 14:00:00')
    prediction=dict(timestamp=str(at),prediction_horizon_s=660)
    assert active_prediction(prediction,at)
    assert not active_prediction(prediction,at+pd.Timedelta(minutes=2))
    assert not active_prediction(prediction,at-pd.Timedelta(seconds=1))
