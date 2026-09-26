import os
import time
import shutil
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from catboost import CatBoostError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ml.streaming import StreamingPredictor
from backend.geo import aggregate
from backend.history import ZoneHistory
from backend.replay import ReplayController

DATA=Path(os.getenv("DATA_DIR","data/raw"))
engine=None
last_received=None
zone_history=ZoneHistory()
replay=None
model_error=None

@asynccontextmanager
async def lifespan(app):
    global engine, replay, model_error, last_received, zone_history
    last_received=None
    zone_history=ZoneHistory()
    schedule=pd.read_csv(DATA/os.getenv("REPLAY_SPLIT","test")/"schedule.csv")
    log_path=Path(os.getenv("PREDICTION_LOG_PATH","artifacts/prediction_log.parquet"))
    if log_path.exists():
        archive=log_path.parent/"logs"
        archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(log_path,archive/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".parquet"))
    try:
        engine=StreamingPredictor(schedule,log_path=log_path)
        engine.flush()
        model_error=None
    except (FileNotFoundError, ValueError, RuntimeError, CatBoostError) as exc:
        engine=None
        model_error=str(exc)
    def reset():
        global engine, zone_history, last_received
        with engine.lock:
            engine.flush()
            engine=StreamingPredictor(schedule, model=engine.model, log_path=log_path)
            zone_history=ZoneHistory()
            last_received=None
    def dispatch(path, payload):
        fn={'/telemetry':engine.ingest, '/replay/context':engine.predict_at, '/replay/outcome':engine.observe}[path]
        perform(fn, payload)
    def advance(at):
        with engine.lock:
            engine.advance(at)
            zone_history.capture(engine)
    replay=ReplayController(DATA,os.getenv('REPLAY_SPLIT','test'),dispatch,reset,advance) if engine else None
    yield
    if replay: replay.close()
    if engine: engine.flush()

app=FastAPI(title="Предиктор · Московский транспорт",version="1.0.0",lifespan=lifespan,
            servers=[{"url":"/","description":"Direct backend :8000"},
                     {"url":"/api","description":"Read-only dashboard proxy :8080"}])

class Telemetry(BaseModel):
    model_config=ConfigDict(extra="ignore",allow_inf_nan=False,coerce_numbers_to_str=True)
    tr_id:str
    event_time:str
    location_valid:bool
    lat:float|None=Field(None,ge=-90,le=90)
    lon:float|None=Field(None,ge=-180,le=180)
    speed:float|None=None
    heading:float|None=None
    receive_time:str|None=None
    gps_time:str|None=None
    packet_id:str|None=None
    @model_validator(mode="after")
    def times(self):
        from ml.features import timestamp
        timestamp(self.event_time)
        for value in [self.receive_time,self.gps_time]:
            if value is not None: timestamp(value)
        return self

class Context(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False,coerce_numbers_to_str=True)
    tr_id:str
    T:str
    target_stop_id:str
    target_time_begin:str
    cur_dev_s:float
    sample_id:str|None=None

class Outcome(BaseModel):
    model_config=ConfigDict(coerce_numbers_to_str=True)
    tr_id:str
    target_stop_id:str
    actual_time:str
    observed_at:str

def authorization(x_ingest_token:str|None=Header(None)):
    token=os.getenv("INGEST_TOKEN")
    if token and x_ingest_token!=token: raise HTTPException(401,"Ingest token required")

def perform(fn, obj):
    global last_received
    try:
        result=fn(obj)
        last_received=time.monotonic()
        with engine.lock: zone_history.capture(engine)
        return result
    except (ValueError,KeyError,TypeError) as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post("/telemetry",dependencies=[Depends(authorization)])
def telemetry(record:Telemetry):
    require_engine()
    if replay and replay.mode=='REPLAY': raise HTTPException(409,'Managed replay owns telemetry; switch to LIVE first')
    return perform(engine.ingest,record.model_dump(exclude_none=True))

@app.post("/replay/context",dependencies=[Depends(authorization)])
def context(record:Context):
    require_external_source()
    return perform(engine.predict_at,record.model_dump())

@app.post("/replay/outcome",dependencies=[Depends(authorization)])
def outcome(record:Outcome):
    require_external_source()
    return perform(engine.observe,record.model_dump())

@app.post("/replay/checkpoint",dependencies=[Depends(authorization)])
def checkpoint():
    require_engine()
    with engine.lock: engine.flush()
    return {"saved":len(engine.log)}

@app.get("/vehicles")
def vehicles(bbox:str|None=None,cell_id:str|None=None,limit:int=Query(500,ge=1,le=2000)):
    require_engine()
    bounds=parse_bbox(bbox)
    with engine.lock:
        ids={i for c in cells('current',bbox) if cell_id is None or c['cell_id']==cell_id for i in c['vehicle_ids']}
        return [v for k,v in engine.vehicles.items() if k in ids][:limit]

@app.get("/vehicles/{id}")
def vehicle(id:str):
    require_engine()
    with engine.lock:
        if id not in engine.vehicles: raise HTTPException(404,"Vehicle not found")
        return engine.vehicles[id]

def require_engine():
    if engine is None: raise HTTPException(503,'ML unavailable')

def require_external_source():
    require_engine()
    if replay and replay.mode=='REPLAY': raise HTTPException(409,'Managed replay owns events; switch to LIVE first')

def parse_bbox(bbox):
    bounds=None
    if bbox:
        try:
            bounds=[float(x) for x in bbox.split(",")]
            if len(bounds)!=4 or bounds[0]>=bounds[2] or bounds[1]>=bounds[3] or not all(__import__('math').isfinite(v) for v in bounds): raise ValueError()
        except ValueError: raise HTTPException(422,"bbox must be west,south,east,north")
    return bounds

def cells(mode,bbox):
    require_engine()
    bounds=parse_bbox(bbox)
    with engine.lock:
        return zone_history.decorate(aggregate(list(engine.vehicles.values()),engine.clock,mode,bounds),engine.clock,mode)

@app.get("/network/heatmap")
def heatmap(mode:Literal["current","forecast"]="forecast",horizon_minutes:int=Query(15,ge=11,le=15),bbox:str|None=None,zoom:int=Query(10,ge=1,le=20),min_risk:float=Query(0,ge=0,le=100)):
    # The trained model covers the full interval; do not pretend distinct horizon models exist.
    if horizon_minutes!=15: raise HTTPException(422,"Model supports the joint (10,15] minute window only; use 15")
    return [c for c in cells(mode,bbox) if c['risk_score']>=min_risk]

@app.get("/hotspots")
def hotspots(mode:Literal["current","forecast"]="forecast",limit:int=Query(8,ge=1,le=50)):
    return [c for c in cells(mode,None) if c["risk_score"]>=20][:limit]

@app.get("/predictions")
def predictions(limit:int=Query(100,ge=1,le=1000),observed:bool=False,tr_id:str|None=None,target_stop_id:str|None=None):
    require_engine()
    with engine.lock:
        rows=list(engine.log)
        if observed: rows=[r for r in rows if r["actual"] is not None]
        if tr_id: rows=[r for r in rows if r['tr_id']==tr_id]
        if target_stop_id: rows=[r for r in rows if r['target_stop_id']==target_stop_id]
        return rows[-limit:]

@app.get("/health")
def health():
    gap=time.monotonic()-last_received if last_received is not None else None
    return {"status":"ok" if engine else "degraded","ml_status":"ONLINE" if engine else "UNAVAILABLE","ml_error":model_error,"mode":replay.mode if replay else 'LIVE',"active_vehicles":sum(c['vehicles'] for c in cells('current',None)) if engine else 0,"hotspots_count":sum(c['risk_score']>=20 for c in cells('forecast',None)) if engine else 0,"connection":"WAITING" if gap is None else "CONNECTION LOST" if gap>20 else "LIVE","seconds_since_last_packet":gap,"last_telemetry_timestamp":max((str(v['location_timestamp']) for v in engine.vehicles.values() if v.get('location_timestamp')),default=None) if engine else None,"model_version":engine.model.version if engine else None,"horizon_seconds":[600,900],"time_scale":"source timestamps (timezone unspecified)","routes_available":False}

@app.get("/metrics")
def metrics():
    require_engine()
    with engine.lock: return engine.metrics()

@app.get("/external/status")
def external_status():
    return {"google_maps":{"provider":"Google Maps JavaScript API","configuration":"frontend VITE_GOOGLE_MAPS_API_KEY","runtime_status":"reported by browser","required_for_inference":False},"weather":{"enabled":False},"ndtp":{"enabled":False,"reason":"Replay uses decoded CSV telemetry"}}

class ReplayCommand(BaseModel):
    action:Literal['play','pause','seek','speed','live']
    speed:Literal[1,10,50]|None=None
    timestamp:str|None=None

@app.get('/replay/state')
def replay_state():
    require_engine()
    return replay.state()

@app.post('/replay/control',dependencies=[Depends(authorization)])
def replay_control(command:ReplayCommand):
    require_engine()
    try: return replay.control(command.action,command.speed,command.timestamp)
    except (ValueError,TypeError) as exc: raise HTTPException(422,str(exc)) from exc

@app.get('/network/zones/{cell_id}/history')
def history(cell_id:str,mode:Literal['current','forecast']='forecast'):
    require_engine()
    with engine.lock: return zone_history.rows(cell_id,mode)

@app.get('/search')
def search(q:str=Query(...,min_length=1,max_length=100)):
    require_engine()
    with engine.lock:
        result=[dict(type='vehicle',id=k,label='ТС '+k,lat=v.get('lat'),lon=v.get('lon')) for k,v in engine.vehicles.items() if q.casefold() in k.casefold()]
        for tr, schedule in engine.schedule.items():
            matches=schedule[schedule.tt_action_item_id.astype(str).str.contains(q,regex=False,case=False)|schedule.building_address.fillna('').str.contains(q,regex=False,case=False)]
            for row in matches.head(20).itertuples():
                import re
                coords=re.findall(r'[-+]?\d+(?:\.\d+)?',str(row.geom))
                result.append(dict(type='stop_event',id=str(row.tt_action_item_id),tr_id=tr,label=str(row.building_address),lon=float(coords[0]) if len(coords)==2 else None,lat=float(coords[1]) if len(coords)==2 else None))
            if len(result)>=30: break
        return result[:30]
