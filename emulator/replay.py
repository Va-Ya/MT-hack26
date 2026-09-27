"""CSV replay in availability-time order; outcome events arrive only after fact."""
import argparse
import json
import os
import time
from pathlib import Path
import httpx
import pandas as pd


def events(data, split, start=None, end=None):
    traffic=pd.read_csv(data/split/"traffic.csv",low_memory=False)
    schedule=pd.read_csv(data/split/"schedule.csv")
    points=pd.read_csv(data/"labels"/f"labels_{split}.csv")
    traffic=traffic[traffic.tr_id.isin(schedule.tr_id.unique())]
    pending=[]
    for row in traffic.to_dict("records"):
        row={k:v for k,v in row.items() if pd.notna(v)}
        at=max(pd.Timestamp(row[c]) for c in ["event_time","receive_time","gps_time"] if c in row)
        pending.append((at,0,"/telemetry",row))
    for row in points.to_dict("records"):
        payload={k:row[k] for k in ["sample_id","tr_id","T","target_stop_id","target_time_begin","cur_dev_s"]}
        pending.append((pd.Timestamp(row["T"]),2,"/replay/context",payload))
    for row in schedule.to_dict("records"):
        at=pd.Timestamp(row["time_fact_begin"])
        payload=dict(tr_id=row["tr_id"],target_stop_id=row["tt_action_item_id"],actual_time=str(at),observed_at=str(at))
        pending.append((at,1,"/replay/outcome",payload))
    pending.sort(key=lambda e:(e[0],e[1]))
    lo=pd.Timestamp(start) if start else None; hi=pd.Timestamp(end) if end else None
    for e in pending:
        if lo is not None and e[0]<lo: continue
        if hi is not None and e[0]>hi: continue
        yield e


def replay(args):
    token=os.getenv("INGEST_TOKEN","")
    with httpx.Client(base_url=args.url,timeout=30,headers={"X-Ingest-Token":token}) as client:
        for attempt in range(60):
            try:
                client.get("/health").raise_for_status(); break
            except httpx.HTTPError:
                time.sleep(1)
        else: raise RuntimeError("Backend did not become healthy")
        first=None; wall=time.monotonic(); count=0
        for at,_,path,payload in events(args.data,args.split,args.start,args.end):
            if first is None: first=at; wall=time.monotonic()
            if not args.no_wait:
                wait=(at-first).total_seconds()/args.speed-(time.monotonic()-wall)
                if wait>0: time.sleep(wait)
            response=client.post(path,json=payload)
            response.raise_for_status()
            count+=1
            if count%500==0: print(f"{count} events | {at} | x{args.speed}",flush=True)
            if args.max_events and count>=args.max_events: break
        client.post("/replay/checkpoint").raise_for_status()
        print(json.dumps(client.get("/metrics").json(),indent=2),flush=True)
        print(f"Replay finished: {count} events; last state remains available",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/raw")); p.add_argument("--split",choices=["train","test"],default="test")
    p.add_argument("--url",default="http://127.0.0.1:8000"); p.add_argument("--speed",type=int,choices=[1,10,50],default=50)
    p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--max-events",type=int); p.add_argument("--no-wait",action="store_true")
    replay(p.parse_args())
