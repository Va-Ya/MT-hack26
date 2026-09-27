"""Reproducible local CPU benchmark; does not train or modify the model."""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from ml.streaming import StreamingPredictor
from emulator.replay import events

def main():
    data=Path('demo/data');engine=StreamingPredictor(pd.read_csv(data/'test/schedule.csv'),log_path='artifacts/benchmark.parquet')
    timings=[];first=time.perf_counter();count=0
    for _,_,path,payload in events(data,'test','2026-01-06 06:30:00','2026-01-06 08:00:00'):
        start=time.perf_counter()
        {'/telemetry':engine.ingest,'/replay/context':engine.predict_at,'/replay/outcome':engine.observe}[path](payload)
        timings.append((time.perf_counter()-start)*1000);count+=1
    elapsed=time.perf_counter()-first
    def summary(values):return dict(n=len(values),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),max_ms=float(max(values)))
    metrics=engine.metrics()
    result=dict(platform=platform.platform(),processor=platform.processor(),model_version=engine.model.version,vehicles=len(engine.schedule),events=count,elapsed_s=elapsed,events_per_s=count/elapsed,full_event_processing=summary(timings),forecast_features=summary([r['feature_ms'] for r in engine.latencies]),forecast_inference=summary([r['inference_ms'] for r in engine.latencies]),forecast_total=summary([r['total_ms'] for r in engine.latencies]),metrics=metrics,limitations='Single-process local CPU, no HTTP/network; includes periodic log persistence, not Docker/Render or official emulator throughput')
    try:
        import ctypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile']]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.byref(counters),counters.cb):
            result['peak_working_set_mb']=counters.peak/1048576;result['working_set_mb']=counters.working/1048576
    except (AttributeError,OSError):pass
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/performance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
