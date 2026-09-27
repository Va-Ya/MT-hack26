import os
import time
import shutil
import asyncio
from contextlib import suppress
from threading import RLock
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends, Query, Request, Response
from catboost import CatBoostError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ml.streaming import StreamingPredictor
from backend.geo import aggregate
from backend.history import ZoneHistory
from backend.context import ExternalContext, EventFeed
from backend.scenarios import Scenario, calculate
from backend.routes import RouteNetwork, MatchRequest, match
from backend.replay import ReplayController

DATA=Path(os.getenv("DATA_DIR","data/raw"))
engine=None
last_received=None
zone_history=ZoneHistory()
replay=None
model_error=None
ndtp_server=None
external=ExternalContext()
route_network=RouteNetwork(routes=[])
route_lock=RLock()
schedule_available=False

@asynccontextmanager
async def lifespan(app):
    global engine, replay, model_error, last_received, zone_history, ndtp_server, external, route_network, schedule_available
    last_received=None
    zone_history=ZoneHistory()
    schedule_path=Path(os.getenv('SCHEDULE_FILE') or str(DATA/os.getenv("REPLAY_SPLIT","test")/"schedule.csv"))
    schedule_available=schedule_path.is_file()
    schedule=pd.read_csv(schedule_path) if schedule_available else pd.DataFrame(columns=['tr_id','tt_action_item_id','time_begin','geom','building_address'])
    external=ExternalContext()
    try:
        route_network=RouteNetwork.model_validate_json(Path('artifacts/route-network.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        route_network=RouteNetwork(routes=[])
    detect_stops=os.getenv('DETECT_STOPS','0')=='1'
    log_path=Path(os.getenv("PREDICTION_LOG_PATH","artifacts/prediction_log.parquet"))
    if log_path.exists():
        archive=log_path.parent/"logs"
        archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(log_path,archive/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".parquet"))
    try:
        engine=StreamingPredictor(schedule,log_path=log_path,detect_stops=detect_stops)
        engine.flush()
        model_error=None
    except (FileNotFoundError, ValueError, RuntimeError, CatBoostError) as exc:
        engine=None
        model_error=str(exc)
    def reset():
        global engine, zone_history, last_received
        with engine.lock:
            engine.flush()
            engine=StreamingPredictor(schedule, model=engine.model, log_path=log_path,detect_stops=detect_stops)
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
    if replay and schedule_available and os.getenv('DEMO_AUTOSTART','0')=='1':
        replay.control('play')
    ndtp_server=None
    if engine and os.getenv('NDTP_ENABLED','0')=='1':
        from backend.ndtp import NDTPServer
        def ingest_ndtp(record):
            if replay and replay.mode=='REPLAY':raise ValueError('Switch replay to LIVE before NDTP ingestion')
            return perform(engine.ingest,record)
        ndtp_server=NDTPServer(ingest_ndtp,host=os.getenv('NDTP_HOST','127.0.0.1'),port=int(os.getenv('NDTP_PORT','9201')),unit_map=__import__('json').loads(os.getenv('NDTP_UNIT_MAP','{}')),tz=os.getenv('NDTP_TIMEZONE','UTC'),offset_seconds=float(os.getenv('NDTP_CLOCK_OFFSET_SECONDS','0')))
        try:ndtp_server.start()
        except RuntimeError:pass # ML stays available; /external/status reports failure.
    external_task=asyncio.create_task(external.run())
    try:
        yield
    finally:
        external_task.cancel()
        with suppress(asyncio.CancelledError): await external_task
        if ndtp_server:ndtp_server.close()
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

@app.get('/health/live')
def liveness():return {'status':'alive'}

@app.get("/health")
def health(response:Response):
    if engine is None:response.status_code=503
    gap=time.monotonic()-last_received if last_received is not None else None
    return {"demo_enabled":os.getenv("DEMO_ENABLED","0")=="1","schedule_available":schedule_available,"status":"ok" if engine else "degraded","ml_status":"ONLINE" if engine else "UNAVAILABLE","ml_error":model_error,"mode":replay.mode if replay else 'LIVE',"active_vehicles":sum(c['vehicles'] for c in cells('current',None)) if engine else 0,"hotspots_count":sum(c['risk_score']>=20 for c in cells('forecast',None)) if engine else 0,"connection":"WAITING" if gap is None else "CONNECTION LOST" if gap>20 else "LIVE","seconds_since_last_packet":gap,"last_telemetry_timestamp":max((str(v['location_timestamp']) for v in engine.vehicles.values() if v.get('location_timestamp')),default=None) if engine else None,"model_version":engine.model.version if engine else None,"horizon_seconds":[600,900],"time_scale":"source timestamps (timezone unspecified)","routes_available":bool(route_network.routes)}

@app.get("/metrics")
def metrics():
    require_engine()
    with engine.lock: return engine.metrics()

@app.get("/external/status")
def external_status():
    return {"yandex_maps":{"provider":"Yandex Maps JS API v3","configuration":"VITE_YANDEX_MAPS_API_KEY"},"context":external.snapshot(mode=replay.mode if replay else 'LIVE'),"ndtp":ndtp_server.state() if ndtp_server else {"enabled":False,"reason":"NDTP_ENABLED=1 or POST /telemetry/ndtp"},"schedule_available":schedule_available,"routes_count":len(route_network.routes)}

@app.post('/telemetry/ndtp',dependencies=[Depends(authorization)])
async def ndtp_http(request:Request):
    """A bridge batch contains a handshake followed by complete NDTP frames."""
    require_external_source()
    from backend.ndtp import FrameDecoder,Session
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>1048576:raise HTTPException(413,'NDTP batch exceeds 1 MiB')
    try:
        decoder=FrameDecoder();packets=decoder.feed(data);session=Session();records=[]
        if decoder.buffer:raise ValueError('Incomplete final NDTP frame')
        if not packets:raise ValueError('Empty NDTP batch')
        mapping=__import__('json').loads(os.getenv('NDTP_UNIT_MAP','{}'))
        received=datetime.now(timezone.utc)
        for packet in packets:
            if session.accept(packet):records.append(packet.record(received,unit_map=mapping,tz=os.getenv('NDTP_TIMEZONE','UTC'),offset_seconds=float(os.getenv('NDTP_CLOCK_OFFSET_SECONDS','0'))))
        # Validate the whole batch before any model/state mutation.
        parsed=[Telemetry.model_validate(r).model_dump(exclude_none=True) for r in records]
    except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
    return {'frames':len(records),'results':[perform(engine.ingest,r) for r in parsed]}

@app.get('/external/context')
def context_evidence(lat:float=Query(55.7558,ge=54.5,le=57),lon:float=Query(37.6173,ge=36,le=39),radius_m:int=Query(1000,ge=100,le=10000)):
    mode = replay.mode if replay else 'LIVE'
    source_at = None
    if mode == 'REPLAY' and replay:
        from zoneinfo import ZoneInfo
        try:
            source_at = datetime.fromisoformat(replay.state()['timestamp']).replace(tzinfo=ZoneInfo(os.getenv('TELEMETRY_TIMEZONE', 'UTC')))
        except (ValueError, KeyError):
            pass
    result=external.snapshot(lat,lon,radius_m,mode=mode,source_at=source_at)
    # LIVE is also used by the historical CLI emulator. It is not proof of clock alignment.
    if result['usable_with_telemetry']:
        source_zone=os.getenv('TELEMETRY_TIMEZONE')
        if engine is None or engine.clock is None:
            result.update(usable_with_telemetry=False,replay_notice='Телеметрия ещё не поступила; показан отдельный текущий контекст Москвы')
        elif not source_zone:
            result.update(usable_with_telemetry=False,replay_notice='Часовой пояс телеметрии не подтверждён; текущий контекст не совмещается с потоком')
        else:
            try:
                from zoneinfo import ZoneInfo
                source_at=engine.clock.to_pydatetime().replace(tzinfo=ZoneInfo(source_zone))
                gap=abs((datetime.now(timezone.utc)-source_at).total_seconds())
                if gap>300: result.update(usable_with_telemetry=False,replay_notice='Время телеметрии отличается от текущего более чем на 5 минут; контекст отделён от потока')
            except (ValueError,KeyError):
                result.update(usable_with_telemetry=False,replay_notice='Некорректная настройка часового пояса телеметрии')
    return result

@app.post('/external/events/import',dependencies=[Depends(authorization)])
def import_events(feed:EventFeed):
    try: return external.import_feed(feed)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc

@app.post('/analysis/what-if')
def what_if(scenario:Scenario):
    try: return calculate(scenario)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc

@app.get('/network/routes')
def network_routes():
    return route_network.model_dump()

@app.post('/network/routes/import',dependencies=[Depends(authorization)])
def import_routes(network:RouteNetwork):
    global route_network
    with route_lock:
        path=Path('artifacts/route-network.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        temp.write_text(network.model_dump_json(),encoding='utf-8')
        temp.replace(path)
        route_network=network
    return {'accepted':len(network.routes)}

@app.post('/network/map-match')
def map_match(point:MatchRequest):
    return match(route_network,point)

@app.get('/ndtp/status')
def ndtp_status():
    return ndtp_server.state() if ndtp_server else {'enabled':False}

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
    if command.action=='seek':
        try:
            target=pd.Timestamp(command.timestamp)
            if pd.isna(target) or target.tzinfo is not None or not replay.start<=target<=replay.end:
                raise ValueError('Seek timestamp outside replay range')
        except (ValueError,TypeError) as exc: raise HTTPException(422,str(exc)) from exc
    if command.action in ('play','seek') and not schedule_available:
        raise HTTPException(409,'Датасет не установлен. Укажите DATA_DIR с test/schedule.csv и телеметрией; внешние факторы и what-if доступны без него.')
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

# Public controls are restricted to the installed historical dataset.
from collections import deque
_demo_commands=deque(maxlen=120)
_demo_lock=RLock()

@app.post('/demo/control')
def demo_control(command:ReplayCommand):
    if os.getenv('DEMO_ENABLED','0')!='1':raise HTTPException(404,'Public demo disabled')
    if command.action=='live':raise HTTPException(403,'LIVE requires operator access')
    with _demo_lock:
        now=time.monotonic()
        while _demo_commands and now-_demo_commands[0]>=60:_demo_commands.popleft()
        if len(_demo_commands)>=60:raise HTTPException(429,'Too many demo controls; retry in a minute')
        _demo_commands.append(now)
        return replay_control(command)
