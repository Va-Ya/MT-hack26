"""Bounded telemetry histories; immutable issued forecasts and delayed outcomes."""
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
import math
import threading
import numpy as np
import pandas as pd
from ml.features import FeatureBuilder, timestamp, HISTORY_SECONDS
from ml.model import Predictor

class StreamingPredictor:
    def __init__(self, schedule, model=None, log_path="artifacts/prediction_log.parquet"):
        self.model=model or Predictor()
        s=schedule[["tr_id","tt_action_item_id","time_begin","geom","building_address"]].copy()
        s["time_begin"]=pd.to_datetime(s.time_begin,format="mixed")
        self.schedule={str(k):v.sort_values("time_begin") for k,v in s.groupby("tr_id")}
        self.history=defaultdict(list)
        self.vehicles={}; self.watermarks={}; self.current_delay={}; self.delay_times={}
        self.log=deque(maxlen=20000); self.observed={}
        self.latencies=deque(maxlen=5000); self.log_path=Path(log_path)
        self.lock=threading.RLock(); self.clock=None

    def advance(self, value):
        t=timestamp(value)
        if self.clock is None or t>self.clock: self.clock=t
        return t

    def context(self, tr_id, T):
        s=self.schedule.get(str(tr_id))
        if s is None: return None
        available=s[(s.time_begin>T+pd.Timedelta(minutes=10))&(s.time_begin<=T+pd.Timedelta(minutes=15))]
        if available.empty: return None
        stop=available.iloc[0]
        return dict(T=T,tr_id=str(tr_id),target_stop_id=str(stop.tt_action_item_id),target_time_begin=stop.time_begin,geom=stop.geom,cur_dev_s=self.current_delay.get(str(tr_id),0.0), current_delay_known=str(tr_id) in self.current_delay)

    def ingest(self, record):
        with self.lock:
            t=timestamp(record["event_time"]); tr=str(record["tr_id"])
            if "location_valid" not in record: raise ValueError("location_valid is required")
            available_at=max([t]+[timestamp(record[c]) for c in ["receive_time","gps_time"] if record.get(c) is not None])
            self.advance(available_at)
            watermark=self.watermarks.get(tr)
            if watermark is not None and t < watermark-pd.Timedelta(seconds=HISTORY_SECONDS):
                return {"accepted":False,"reason":"outside_history"}
            rows=self.history[tr]
            record={**record,"tr_id":tr,"event_time":t}
            # Idempotent retransmission when packet ID is present.
            if record.get("packet_id") is not None and any(str(r.get("packet_id"))==str(record["packet_id"]) for r in rows):
                return {"accepted":False,"reason":"duplicate_packet"}
            rows.append(record)
            newest=max(t,watermark) if watermark is not None else t
            self.watermarks[tr]=newest
            self.history[tr]=[r for r in rows if r["event_time"]>=newest-pd.Timedelta(seconds=HISTORY_SECONDS)]
            if watermark is not None and t<watermark:
                return {"accepted":True,"prediction":None,"reason":"out_of_order_buffered"}
            previous=self.vehicles.get(tr,{})
            valid=record.get("location_valid") and record.get("lat") is not None and record.get("lon") is not None and 54.5<=record["lat"]<=57 and 36<=record["lon"]<=39
            state={**previous,"tr_id":tr,"timestamp":str(t),"current_delay":self.current_delay.get(tr,0.0),"current_delay_known":tr in self.current_delay,"route_id":None}
            if valid:
                speed=record.get("speed")
                state.update(lat=record["lat"],lon=record["lon"],speed=speed if speed is not None and 0<=speed<=130 else None,location_timestamp=str(t))
            self.vehicles[tr]=state
            context=self.context(tr,available_at)
            prediction=self.predict_at(context) if context else None
            if not context: self.vehicles[tr].pop("prediction",None)
            return {"accepted":True,"prediction":prediction}

    def predict_at(self, context):
        with self.lock:
            if context is None: return None
            start=perf_counter(); tr=str(context["tr_id"]); T=timestamp(context["T"])
            self.advance(T)
            s=self.schedule.get(tr)
            if s is None: raise ValueError("Unknown vehicle schedule")
            match=s[s.tt_action_item_id.astype(str)==str(context["target_stop_id"])]
            if len(match)!=1: raise ValueError("Unknown or ambiguous target")
            stop=match.iloc[0]
            if timestamp(context["target_time_begin"])!=stop.time_begin: raise ValueError("Target plan mismatch")
            context={**context,"geom":stop.geom}
            rows=[r for r in self.history.get(tr,[]) if r["event_time"]<=T]
            features=FeatureBuilder(pd.DataFrame(rows)).build(context)
            after_features=perf_counter()
            value=self.model.predict(features)
            after_inference=perf_counter()
            if not math.isfinite(value): raise ValueError("Non-finite model prediction")
            self.latencies.append({"feature_ms":(after_features-start)*1000,"inference_ms":(after_inference-after_features)*1000,"total_ms":(after_inference-start)*1000})
            prediction={"timestamp":str(T),"tr_id":tr,"target_stop_id":str(context["target_stop_id"]),"predicted_delay_s":value,"prediction_horizon_s":features["scheduled_time_to_target"],"target_time_begin":str(stop.time_begin),"model_version":self.model.version}
            known=self.delay_times.get(tr)
            if context.get("current_delay_known",True) and (known is None or T>=known):
                self.current_delay[tr]=float(context["cur_dev_s"]); self.delay_times[tr]=T
            state=self.vehicles.setdefault(tr,{"tr_id":tr,"route_id":None})
            if "timestamp" not in state or T>=timestamp(state["timestamp"]):
                state.update(timestamp=str(T),current_delay=float(context["cur_dev_s"]),current_delay_known=context.get("current_delay_known",True),prediction=prediction,speed_drop=features["speed_drop"] if math.isfinite(features["speed_drop"]) else 0)
            key=(tr,str(context["target_stop_id"]))
            actual=self.observed.get(key)
            # Facts are never used as features. Only already observed outcomes may be attached.
            row={"T":str(T),"tr_id":tr,"route_id":None,"target_stop_id":key[1],"prediction":value,"actual":actual["delay"] if actual else None,"absolute_error":abs(value-actual["delay"]) if actual else None,"lead_time_seconds":(actual["time"]-T).total_seconds() if actual else None,"prediction_horizon_s":features["scheduled_time_to_target"],"model_version":self.model.version,"speed":features["speed_now"] if math.isfinite(features["speed_now"]) else None,"sample_id":context.get("sample_id")}
            self.log.append(row)
            if len(self.log)%200==0: self.flush()
            return prediction

    def observe(self, event):
        with self.lock:
            observed=timestamp(event["observed_at"]); fact=timestamp(event["actual_time"])
            if fact>observed: raise ValueError("Future outcome cannot be revealed")
            self.advance(observed); tr=str(event["tr_id"]); key=(tr,str(event["target_stop_id"]))
            s=self.schedule.get(tr)
            if s is None: raise ValueError("Unknown vehicle")
            matched=s[s.tt_action_item_id.astype(str)==key[1]]
            if len(matched)!=1: raise ValueError("Unknown target")
            delay=(fact-matched.iloc[0].time_begin).total_seconds()
            self.observed[key]={"time":fact,"delay":delay}
            if tr not in self.delay_times or fact>=self.delay_times[tr]:
                self.current_delay[tr]=delay; self.delay_times[tr]=fact
                if tr in self.vehicles:
                    self.vehicles[tr].update(current_delay=delay,current_delay_known=True)
            for row in self.log:
                if (row["tr_id"],row["target_stop_id"])==key and timestamp(row["T"])<=observed:
                    row.update(actual=delay,absolute_error=abs(row["prediction"]-delay),lead_time_seconds=(fact-timestamp(row["T"])).total_seconds())
            return {"observed":True,"delay_s":delay}

    def flush(self):
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.log_path.with_suffix(".tmp.parquet")
        columns=["T","tr_id","route_id","target_stop_id","prediction","actual","absolute_error","lead_time_seconds","prediction_horizon_s","model_version","speed","sample_id"]
        pd.DataFrame(list(self.log),columns=columns).to_parquet(tmp,index=False)
        tmp.replace(self.log_path)

    def metrics(self):
        out={"samples":len(self.latencies),"logged_predictions":len(self.log),"model_version":self.model.version}
        for key in ["feature_ms","inference_ms","total_ms"]:
            vals=[r[key] for r in self.latencies]
            out[key]={"mean":float(np.mean(vals)),"p50":float(np.percentile(vals,50)),"p95":float(np.percentile(vals,95))} if vals else None
        matched=[r for r in self.log if r["actual"] is not None]
        out["observed_outcomes"]=len(matched)
        out["replay_mae"]=float(np.mean([r["absolute_error"] for r in matched])) if matched else None
        return out
