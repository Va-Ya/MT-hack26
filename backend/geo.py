"""Explainable geographic aggregation. Scores are indices, not probabilities."""
from collections import defaultdict
import math
import numpy as np
import pandas as pd

def risk(current,predicted,drop,affected,mode):
    primary=predicted if mode=="forecast" else current
    factors={"delay":50*min(max(primary,0)/600,1),"current_delay":20*min(max(current,0)/300,1),"speed_drop":15*min(max(drop,0)/20,1),"affected_vehicles":15*min(affected/5,1)}
    return round(sum(factors.values()),1),{k:round(v,1) for k,v in factors.items()}

def aggregate(vehicles, clock, mode="forecast", bbox=None):
    groups=defaultdict(list)
    if clock is None: return []
    for v in vehicles:
        if "lat" not in v or "location_timestamp" not in v: continue
        if not 0<=(clock-pd.Timestamp(v["location_timestamp"])).total_seconds()<=300: continue
        if bbox and not (bbox[0]<=v["lon"]<=bbox[2] and bbox[1]<=v["lat"]<=bbox[3]): continue
        p=v.get("prediction")
        if mode=="forecast" and not active_prediction(p,clock): continue
        # ~650m cells near Moscow; only centers are exposed to the map.
        key=(math.floor(v["lat"]/.006),math.floor(v["lon"]/.01))
        groups[key].append(v)
    output=[]
    for key, rows in groups.items():
        known_delays=[v.get("current_delay",0) for v in rows if v.get('current_delay_known',True)]
        current=float(np.mean(known_delays)) if known_delays else 0.0
        predicted_values=[v["prediction"]["predicted_delay_s"] for v in rows if active_prediction(v.get('prediction'),clock)]
        predicted=float(np.mean(predicted_values)) if predicted_values else None
        speeds=[v["speed"] for v in rows if v.get("speed") is not None]
        drop=float(np.mean([v.get("speed_drop",0) for v in rows]))
        affected=sum((v.get("prediction",{}).get("predicted_delay_s",0) if mode=="forecast" else v.get("current_delay",0))>120 for v in rows)
        score,factors=risk(current,predicted or 0,drop,affected,mode)
        current_affected=sum(v.get("current_delay",0)>120 for v in rows)
        now,_=risk(current,predicted or 0,drop,current_affected,"current")
        horizons=[remaining_horizon(v['prediction'],clock) for v in rows if active_prediction(v.get('prediction'),clock)]
        output.append({"cell_id":f"{key[0]}:{key[1]}","lat":(key[0]+.5)*.006,"lon":(key[1]+.5)*.01,"risk_score":score,"vehicles":len(rows),"vehicles_count":len(rows),"avg_speed":float(np.mean(speeds)) if speeds else None,"median_speed":float(np.median(speeds)) if speeds else None,"current_delay":current,"predicted_delay":predicted,"max_predicted_delay":max(predicted_values) if predicted_values else None,"problem_vehicles_count":affected,"risk_factors":factors,"risk_trend":round(score-now,1),"trend_definition":"forecast minus current index" if mode=="forecast" else "current view","vehicle_ids":[v["tr_id"] for v in rows],"affected_routes":None,"routes_available":False,"current_delay_known_count":sum(v.get("current_delay_known",False) for v in rows),"horizon_min_s":min(horizons) if horizons else None,"horizon_max_s":max(horizons) if horizons else None})
    return sorted(output,key=lambda c:c["risk_score"],reverse=True)

def remaining_horizon(prediction,clock):
    issued=pd.Timestamp(prediction['timestamp'])
    target=pd.Timestamp(prediction['target_time_begin']) if prediction.get('target_time_begin') else issued+pd.Timedelta(seconds=prediction['prediction_horizon_s'])
    return (target-clock).total_seconds()

def active_prediction(prediction,clock):
    return bool(prediction and 0<=(clock-pd.Timestamp(prediction['timestamp'])).total_seconds()<=300 and 600<remaining_horizon(prediction,clock)<=900)
