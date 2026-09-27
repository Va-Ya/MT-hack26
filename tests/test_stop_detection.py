import pandas as pd
from ml.stop_detection import StopDetector

def schedule(ambiguous=False):
    at=pd.Timestamp('2026-01-06 14:00:00')
    return pd.DataFrame([dict(tr_id=1,tt_action_item_id=i,time_begin=at,geom='POINT (37.61 55.75)') for i in ([20,21] if ambiguous else [20])])

def row(seconds):
    return dict(tr_id=1,event_time=pd.Timestamp('2026-01-06 14:00:00')+pd.Timedelta(seconds=seconds),location_valid=True,lat=55.75,lon=37.61,speed=0)

def test_confirmation_availability_and_retransmission():
    detector=StopDetector(schedule())
    assert detector.ingest(row(0),row(0)['event_time']) is None
    assert detector.ingest(row(10),row(10)['event_time']) is None
    known=row(40)['event_time']
    observed=detector.ingest(row(20),known)
    assert observed['actual_time']==row(0)['event_time']
    assert observed['observed_at']==known
    assert detector.ingest(row(20),known) is None

def test_ambiguous_and_interrupted_dwell():
    detector=StopDetector(schedule(True))
    for seconds in (0,20):assert detector.ingest(row(seconds),row(seconds)['event_time']) is None
    detector=StopDetector(schedule())
    assert detector.ingest(row(0),row(0)['event_time']) is None
    assert detector.ingest(row(40),row(40)['event_time']) is None
