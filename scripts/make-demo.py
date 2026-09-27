"""Package original events, never generated targets, for a bounded public replay."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

def build(source,target):
    labels=pd.read_csv(source/'labels/labels_test.csv')
    times=pd.to_datetime(labels['T'],format='mixed')
    start=times.min();lo=start-pd.Timedelta(minutes=20);hi=times.max()+pd.Timedelta(minutes=15)
    ids=sorted(labels.tr_id.unique().tolist())
    if not ids:raise ValueError('No demonstration vehicles in the source window')
    schedule=pd.read_csv(source/'test/schedule.csv')
    schedule=schedule[schedule.tr_id.isin(ids)]
    labels=labels[labels.tr_id.isin(ids)]
    chunks=[]
    for chunk in pd.read_csv(source/'test/traffic.csv',chunksize=50000,low_memory=False):
        chunk=chunk[chunk.tr_id.isin(ids)]
        times=pd.DataFrame({c:pd.to_datetime(chunk[c],format='mixed',errors='coerce') for c in ['event_time','receive_time','gps_time'] if c in chunk})
        chunks.append(chunk[times.max(axis=1).between(lo,hi)])
    traffic=pd.concat(chunks,ignore_index=True)
    (target/'test').mkdir(parents=True,exist_ok=True);(target/'labels').mkdir(exist_ok=True)
    for frame,path in [(traffic,'test/traffic.csv'),(schedule,'test/schedule.csv'),(labels,'labels/labels_test.csv')]:
        frame.to_csv(target/path,index=False)
    hashes={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['test/traffic.csv','test/schedule.csv','labels/labels_test.csv']}
    manifest=dict(source='Original hackathon test data; not training data',source_sha256=hashes,history_start=str(lo),start=str(start),end=str(hi),vehicles=[str(x) for x in ids],telemetry_rows=len(traffic),contexts=len(labels),schedule_rows=len(schedule))
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('data/raw'));p.add_argument('--output',type=Path,default=Path('demo/data'))
    args=p.parse_args();build(args.source,args.output)
