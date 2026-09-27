import numpy as np
import pandas as pd
from ml.features import FeatureBuilder, point

T = pd.Timestamp("2026-01-06 12:00:00")
CTX = dict(T=T, tr_id=1, target_stop_id=20, target_time_begin=T+pd.Timedelta(minutes=11), cur_dev_s=60, geom="POINT (37.6 55.75)")

def telemetry():
    return pd.DataFrame([dict(tr_id=1, event_time=T-pd.Timedelta(seconds=15*i), receive_time=T-pd.Timedelta(seconds=15*i), gps_time=T-pd.Timedelta(seconds=15*i), lat=55.75, lon=37.6+i*0.0001, speed=20+i, heading=10, location_valid=True, packet_id=str(i)) for i in range(30)])

def test_future_rows_do_not_change_features():
    old = telemetry()
    future = old.copy()
    for c in ["event_time","receive_time","gps_time"]:
        future[c] += pd.Timedelta(hours=1)
    future["speed"] = 129
    future["lon"] = 38
    a=FeatureBuilder(old).build(CTX)
    b=FeatureBuilder(pd.concat([old,future])).build(CTX)
    pd.testing.assert_series_equal(pd.Series(a), pd.Series(b))

def test_late_received_rows_do_not_change_features():
    old=telemetry()
    late=old.copy()
    late["receive_time"]=T+pd.Timedelta(seconds=1)
    late["speed"]=129
    pd.testing.assert_series_equal(pd.Series(FeatureBuilder(old).build(CTX)), pd.Series(FeatureBuilder(pd.concat([old,late])).build(CTX)))

def test_target_fact_cannot_become_a_feature():
    a=FeatureBuilder(telemetry()).build(CTX)
    b=FeatureBuilder(telemetry()).build({**CTX, "time_fact_begin": T+pd.Timedelta(days=1), "target_delay_s": 999999})
    assert set(a) == set(b)
    pd.testing.assert_series_equal(pd.Series(a), pd.Series(b))

def test_geometry_and_window():
    assert point("POINT (37.6 55.75)") == (55.75,37.6)
    f=FeatureBuilder(telemetry()).build(CTX)
    assert f["telemetry_count_5m"]==21
    assert f["speed_now"]==20
    assert f["speed_mean_1m"]==22
    assert f["distance_to_target_stop"]==0

def test_future_rows_do_not_change_prediction():
    from ml.model import Predictor
    model=Predictor()
    old=telemetry()
    future=old.copy()
    for c in ["event_time","receive_time","gps_time"]:
        future[c]+=pd.Timedelta(days=1)
    a=model.predict(FeatureBuilder(old).build(CTX))
    b=model.predict(FeatureBuilder(pd.concat([old,future])).build(CTX))
    assert a == b
