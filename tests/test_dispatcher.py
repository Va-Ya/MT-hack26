import time
import pandas as pd
from fastapi.testclient import TestClient
from backend.history import ZoneHistory
from backend.replay import ReplayController
from tests.test_streaming_api import engine
from tests.test_no_future_leakage import T, CTX, telemetry


def wait_state(controller, status):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        state=controller.state()
        if state['status']==status: return state
        assert state['status']!='error',state
        time.sleep(.01)
    raise AssertionError(controller.state())


def test_seek_rebuilds_and_does_not_reveal_future_facts(tmp_path):
    holder=[engine(tmp_path)]
    at=pd.Timestamp('2026-01-06 06:50:00')
    seen=[]
    pending=[(at,0,'telemetry',{}),(at+pd.Timedelta(minutes=1),1,'fact',{})]
    def reset(): seen.clear()
    c=ReplayController(None,None,lambda path,payload:seen.append(path),reset,lambda t:None,pending)
    try:
        c.control('seek',timestamp=str(at))
        wait_state(c,'paused')
        assert seen==['telemetry']
        c.control('seek',timestamp=str(at+pd.Timedelta(minutes=2)))
        wait_state(c,'paused')
        assert seen==['telemetry','fact']
        c.control('seek',timestamp=str(at))
        wait_state(c,'paused')
        assert seen==['telemetry']
        c.control('play',speed=10)
        assert c.state()['status']=='playing'
        c.control('pause')
        assert c.state()['status']=='paused'
        c.control('live')
        assert not seen and c.state()['mode']=='LIVE'
    finally: c.close()


def test_zone_history_uses_five_minute_past_and_expires(tmp_path):
    e=engine(tmp_path); h=ZoneHistory()
    row=telemetry().iloc[-1].to_dict()
    e.ingest(row); h.capture(e)
    from backend.geo import aggregate
    first=aggregate(list(e.vehicles.values()),e.clock,'current')[0]
    assert h.decorate([first.copy()],e.clock,'current')[0]['risk_trend'] is None
    e.clock+=pd.Timedelta(minutes=5)
    e.vehicles['1']['location_timestamp']=str(e.clock)
    e.vehicles['1']['current_delay']=300
    e.vehicles['1']['current_delay_known']=True
    h.capture(e)
    current=aggregate(list(e.vehicles.values()),e.clock,'current')
    decorated=h.decorate(current,e.clock,'current')[0]
    assert decorated['risk_trend']==round(decorated['risk_score']-first['risk_score'],1)
    assert decorated['trend']=='worsening'
    assert len(h.rows(first['cell_id'],'current'))==2
    e.clock+=pd.Timedelta(minutes=31);h.capture(e)
    assert len(h.snapshots)==1


def test_dispatcher_api_filters_and_control(tmp_path,monkeypatch):
    from backend import app as module
    monkeypatch.setenv('PREDICTION_LOG_PATH',str(tmp_path/'api.parquet'))
    monkeypatch.delenv('INGEST_TOKEN',raising=False)
    with TestClient(module.app) as client:
        module.engine=engine(tmp_path)
        for row in telemetry().to_dict('records'): module.engine.ingest(row)
        cells=client.get('/network/heatmap').json()
        assert cells and 0<=cells[0]['risk_score']<=100
        assert client.get('/network/heatmap?min_risk=100').json()==[]
        assert client.get('/hotspots').status_code==200
        assert client.get('/vehicles?bbox=0,0,1,1').json()==[]
        assert client.get('/vehicles?bbox=bad').status_code==422
        assert client.get('/network/zones/missing/history').json()==[]
        assert client.get('/search?q=1').json()[0]['type']=='vehicle'
        assert client.get('/predictions?tr_id=missing').json()==[]
        assert client.post('/replay/control',json={'action':'seek','timestamp':'2030-01-01'}).status_code==422
        assert client.post('/replay/control',json={'action':'speed','speed':10}).json()['speed']==10
