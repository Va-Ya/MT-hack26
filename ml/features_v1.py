"""Causal feature extraction shared by training, submission and streaming."""
import math
import re
import numpy as np
import pandas as pd

HISTORY_SECONDS = 1200
FEATURE_VERSION = "causal-v1"

def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None:
        raise ValueError("Expected a valid timezone-naive timestamp in the dataset time scale")
    return t

def distance(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2)
    h = np.sin((b-a)/2)**2 + np.cos(a)*np.cos(b)*np.sin(np.radians(lon2-lon1)/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def point(geom):
    nums = re.findall(r"[-+]?[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?", str(geom))
    return (float(nums[1]), float(nums[0])) if len(nums) == 2 else (np.nan, np.nan)

class FeatureBuilder:
    def __init__(self, telemetry):
        f = telemetry.copy()
        if f.empty:
            self.groups = {}
            return
        for col in ["lat", "lon", "speed", "heading"]:
            if col not in f:
                f[col] = np.nan
        f["event_time"] = pd.to_datetime(f.event_time, format="mixed")
        for col in ["receive_time", "gps_time"]:
            if col in f:
                f[col] = pd.to_datetime(f[col], format="mixed", errors="coerce")
        f["tr_id"] = f.tr_id.astype(str)
        if "packet_id" in f:
            f["packet_id"] = f.packet_id.astype(str)
        f = f.sort_values(["event_time"] + [c for c in ["receive_time", "packet_id"] if c in f], kind="stable")
        self.groups = {str(k): v for k, v in f.groupby("tr_id", sort=False)}

    def build(self, context):
        T = timestamp(context["T"])
        horizon = (timestamp(context["target_time_begin"]) - T).total_seconds()
        if not 600 < horizon <= 900:
            raise ValueError("Target must lie in (T+10min, T+15min]")
        out = {"cur_dev_s": float(context["cur_dev_s"]), "target_stop_id": str(context["target_stop_id"]),
               "hour": T.hour + T.minute/60, "weekday": T.weekday(), "is_weekend": int(T.weekday() >= 5),
               "scheduled_time_to_target": horizon}
        cols = ["speed_now", "speed_mean_1m", "speed_mean_3m", "speed_mean_5m", "speed_std_5m", "speed_min_5m", "speed_max_5m", "speed_delta_1m", "speed_delta_3m", "speed_delta_5m", "acceleration", "stationary_time", "distance_travelled_1m", "distance_travelled_5m", "heading", "heading_delta", "distance_to_target_stop", "telemetry_age_s", "telemetry_count_5m", "speed_drop"]
        out.update({c: np.nan for c in cols})
        f = self.groups.get(str(context["tr_id"]))
        if f is None:
            out["telemetry_count_5m"] = 0
            return out
        times = f.event_time
        f = f.iloc[times.searchsorted(T-pd.Timedelta(seconds=HISTORY_SECONDS)):times.searchsorted(T, side="right")]
        for col in ["receive_time", "gps_time"]:
            if col in f:
                f = f[f[col].isna() | (f[col] <= T)]
        valid = f.location_valid.astype(str).str.lower().isin(["true", "1"])
        f = f[valid & f.lat.between(54.5,57) & f.lon.between(36,39)].drop_duplicates("event_time", keep="last").copy()
        if f.empty:
            out["telemetry_count_5m"] = 0
            return out
        f.loc[~f.speed.between(0,130), "speed"] = np.nan
        sec = (f.event_time-T).dt.total_seconds().to_numpy()
        speed = f.speed.to_numpy(dtype=float)
        age = -sec[-1]
        out["telemetry_age_s"] = age
        out["telemetry_count_5m"] = int((sec >= -300).sum())
        if age > 300:
            return out
        out["speed_now"] = speed[-1]
        for minute in [1,3,5]:
            w = speed[(sec >= -minute*60) & np.isfinite(speed)]
            out[f"speed_mean_{minute}m"] = float(w.mean()) if len(w) else np.nan
            old = np.flatnonzero((sec <= -minute*60) & (sec >= -minute*60-60) & np.isfinite(speed))
            out[f"speed_delta_{minute}m"] = speed[-1]-speed[old[-1]] if len(old) else np.nan
            if minute == 5 and len(w):
                out.update(speed_std_5m=float(w.std()), speed_min_5m=float(w.min()), speed_max_5m=float(w.max()))
        dt = np.diff(sec)
        if len(f) >= 2 and 0 < dt[-1] <= 60:
            out["acceleration"] = (speed[-1]-speed[-2])/3.6/dt[-1]
            out["heading_delta"] = (float(f.heading.iloc[-1])-float(f.heading.iloc[-2])+180)%360-180
        out["heading"] = float(f.heading.iloc[-1])
        stationary = 0.0
        for j in range(len(speed)-1,0,-1):
            if not (speed[j] <= 1 and speed[j-1] <= 1 and dt[j-1] <= 60):
                break
            stationary += dt[j-1]
        out["stationary_time"] = stationary
        if len(f) >= 2:
            steps = distance(f.lat.to_numpy()[:-1], f.lon.to_numpy()[:-1], f.lat.to_numpy()[1:], f.lon.to_numpy()[1:])
            good = (dt > 0) & (dt <= 60) & (steps/np.maximum(dt,1) <= 130/3.6)
            for minute in [1,5]:
                out[f"distance_travelled_{minute}m"] = float(steps[good & (sec[:-1] >= -minute*60)].sum())
        lat, lon = point(context.get("geom", ""))
        out["distance_to_target_stop"] = float(distance(float(f.lat.iloc[-1]),float(f.lon.iloc[-1]),lat,lon))
        out["speed_drop"] = max(0.0,out["speed_mean_5m"]-out["speed_now"]) if math.isfinite(out["speed_mean_5m"]) and math.isfinite(out["speed_now"]) else np.nan
        return out
