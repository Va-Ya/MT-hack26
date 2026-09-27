"""Causal feature extraction shared by training, submission and streaming."""
import math
import re
import numpy as np
import pandas as pd

HISTORY_SECONDS = 1200
FEATURE_VERSION = "causal-v3"

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

class LegacyBuilder:
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


EXTENDED = ['hour_sin','hour_cos','target_lat','target_lon','gps_missing',
            'invalid_gps_fraction_5m','speed_missing_fraction_5m','receive_lag_s',
            'telemetry_span_5m','max_gap_5m','speed_slope_5m','heading_sin','heading_cos',
            'distance_change_5m','approach_speed_mps','distance_over_horizon_mps',
            'expected_position_error_m','segment_progress','planned_progress',
            'segment_cross_track_m','planned_segment_m','planned_segment_s',
            'planned_stops_to_target','plan_distance_to_target_m','planned_speed_kmh']
for _m in [1,3,5,10,20]:
    EXTENDED += [f'speed_median_{_m}m', f'speed_p10_{_m}m',f'speed_p90_{_m}m',
                 f'stop_fraction_{_m}m',f'speed_count_{_m}m']
EXTENDED += ['vehicle_key','target_geo_key','segment_geo_key','gps_geo_key','gps_lat_now','gps_lon_now']
CAT_FEATURES = ['vehicle_key','target_geo_key','segment_geo_key','gps_geo_key']

class CompiledPlans(dict):
    """Plan-only arrays: immutable for the lifetime of a schedule version."""
    def __init__(self,groups):
        super().__init__(groups)
        self.arrays={}
        for key,s in self.items():
            xy=np.asarray([point(g) for g in s.geom],dtype=float).reshape(-1,2)
            self.arrays[key]=(s.time_begin.to_numpy(dtype='datetime64[ns]'),xy)

class FeatureBuilder(LegacyBuilder):
    """Causal v2. Known plans are allowed; future observations are never read."""
    def __init__(self, telemetry, schedule=None):
        super().__init__(telemetry)
        self.plans={}
        if isinstance(schedule,dict):
            # Compiled immutable plans shared by all live requests.
            self.plans=schedule if isinstance(schedule,CompiledPlans) else CompiledPlans(schedule)
            return
        if schedule is not None:
            s=schedule[['tr_id','tt_action_item_id','time_begin','geom']].copy()
            s['time_begin']=pd.to_datetime(s.time_begin,format='mixed')
            for tr,g in s.groupby('tr_id',sort=False):
                self.plans[str(tr)]=g.sort_values(['time_begin','tt_action_item_id'],kind='stable')
        self.plans=CompiledPlans(self.plans)

    def available(self, context):
        T=timestamp(context['T']);f=self.groups.get(str(context['tr_id']))
        if f is None:return pd.DataFrame()
        times=f.event_time
        f=f.iloc[times.searchsorted(T-pd.Timedelta(seconds=HISTORY_SECONDS)):times.searchsorted(T,side='right')]
        for col in ['receive_time','gps_time']:
            if col in f:f=f[f[col].isna() | (f[col]<=T)]
        return f.drop_duplicates('event_time',keep='last')

    def build(self,context):
        out=super().build(context)
        out.update({c:np.nan for c in EXTENDED})
        h=out['hour']*2*np.pi/24
        lat,lon=point(context.get('geom',''))
        out.update(hour_sin=np.sin(h),hour_cos=np.cos(h),target_lat=lat,target_lon=lon,gps_missing=1)
        out.update(vehicle_key=str(context['tr_id']),target_geo_key=f'{lat:.4f}:{lon:.4f}' if np.isfinite([lat,lon]).all() else 'missing',segment_geo_key='missing',gps_geo_key='missing')
        f=self.available(context)
        if f.empty:return out
        T=timestamp(context['T']);sec=(f.event_time-T).dt.total_seconds().to_numpy()
        speed=pd.to_numeric(f.speed,errors='coerce').to_numpy(dtype=float)
        speed=np.where((speed>=0)&(speed<=130),speed,np.nan)
        valid=f.location_valid.astype(str).str.lower().isin(['true','1']).to_numpy()
        valid = valid & f.lat.between(54.5,57).to_numpy() & f.lon.between(36,39).to_numpy()
        recent=sec>=-300
        if recent.any():
            out['invalid_gps_fraction_5m']=float(1-valid[recent].mean())
            out['speed_missing_fraction_5m']=float(1-np.isfinite(speed[recent]).mean())
            st=sec[recent];out['telemetry_span_5m']=float(st[-1]-st[0])
            out['max_gap_5m']=float(np.diff(st).max()) if len(st)>1 else np.nan
        if 'receive_time' in f and pd.notna(f.receive_time.iloc[-1]):
            out['receive_lag_s']=max(0,(f.receive_time.iloc[-1]-f.event_time.iloc[-1]).total_seconds())
        for m in [1,3,5,10,20]:
            mask=(sec>=-60*m)&np.isfinite(speed);w=speed[mask]
            out[f'speed_count_{m}m']=int(len(w))
            if len(w):
                p10,median,p90=np.quantile(w,[.1,.5,.9])
                out.update({f'speed_median_{m}m':float(median),f'speed_p10_{m}m':float(p10),f'speed_p90_{m}m':float(p90),f'stop_fraction_{m}m':float((w<=1).mean())})
            if m==5 and len(w)>1 and np.ptp(sec[mask])>0:
                x=sec[mask];out['speed_slope_5m']=float(np.dot(x-x.mean(),w-w.mean())/np.dot(x-x.mean(),x-x.mean()))
        g=f.iloc[np.flatnonzero(valid)]
        if g.empty or (T-g.event_time.iloc[-1]).total_seconds()>300:return out
        out['gps_missing']=0
        out.update(gps_lat_now=float(g.lat.iloc[-1]),gps_lon_now=float(g.lon.iloc[-1]),gps_geo_key=f'{g.lat.iloc[-1]:.3f}:{g.lon.iloc[-1]:.3f}')
        heading=float(g.heading.iloc[-1]);out.update(heading_sin=np.sin(np.radians(heading)),heading_cos=np.cos(np.radians(heading)))
        ds=distance(g.lat.to_numpy(),g.lon.to_numpy(),lat,lon)
        out['distance_over_horizon_mps']=float(ds[-1]/out['scheduled_time_to_target'])
        recent_g=g.event_time>=T-pd.Timedelta(minutes=5)
        if recent_g.sum()>1:
            d=ds[recent_g];gt=g.loc[recent_g,'event_time'];dt=(gt.iloc[-1]-gt.iloc[0]).total_seconds()
            out['distance_change_5m']=float(d[0]-d[-1])
            out['approach_speed_mps']=float((d[0]-d[-1])/dt) if dt>0 else np.nan
        self._plan(out,context,g.iloc[-1])
        return out

    def _plan(self,out,context,last):
        arrays=self.plans.arrays.get(str(context['tr_id']))
        if arrays is None:return
        times,coords=arrays
        T=timestamp(context['T']);target=timestamp(context['target_time_begin'])
        aligned=T-pd.Timedelta(seconds=float(context['cur_dev_s']))
        j=int(times.searchsorted(aligned.to_datetime64(),side='right'))
        end=int(times.searchsorted(target.to_datetime64(),side='right'))
        now=int(times.searchsorted(T.to_datetime64(),side='right'))
        out['planned_stops_to_target']=max(0,end-now)
        if j==0 or j==len(times):return
        alat,alon=coords[j-1];blat,blon=coords[j]
        a_time=pd.Timestamp(times[j-1]);b_time=pd.Timestamp(times[j])
        dt=(b_time-a_time).total_seconds()
        if not np.isfinite([alat,alon,blat,blon]).all() or dt<=0:return
        out['segment_geo_key']=f'{alat:.4f}:{alon:.4f}>{blat:.4f}:{blon:.4f}'
        length=float(distance(alat,alon,blat,blon))
        alpha=np.clip((aligned-a_time).total_seconds()/dt,0,1)
        out.update(expected_position_error_m=float(distance(last.lat,last.lon,alat+alpha*(blat-alat),alon+alpha*(blon-alon))),planned_progress=float(alpha),planned_segment_m=length,planned_segment_s=dt,planned_speed_kmh=length/dt*3.6)
        # Stop-to-stop chord is a feature, not claimed road/route map matching.
        scale=np.cos(np.radians((alat+blat)/2))
        v=np.array([(blon-alon)*scale,blat-alat])*111195
        p=np.array([(last.lon-alon)*scale,last.lat-alat])*111195
        norm=np.dot(v,v)
        if norm>0:
            progress=float(np.dot(p,v)/norm)
            out.update(segment_progress=progress,segment_cross_track_m=float(np.linalg.norm(p-np.clip(progress,0,1)*v)))
        xy=coords[j:end]
        if len(xy) and np.isfinite(xy).all():
            out['plan_distance_to_target_m']=float(distance(last.lat,last.lon,xy[0,0],xy[0,1])+distance(xy[:-1,0],xy[:-1,1],xy[1:,0],xy[1:,1]).sum())

    def sequence(self,context,steps=80,step_seconds=15):
        """Bins known at T, six channels with explicit speed/GPS validity masks."""
        f=self.available(context);result=np.zeros((steps,6),dtype=np.float32)
        if f.empty:return result
        age=(timestamp(context['T'])-f.event_time).dt.total_seconds().to_numpy()
        bins=steps-1-np.floor(age/step_seconds).astype(int)
        speed=pd.to_numeric(f.speed,errors='coerce').to_numpy(dtype=float)
        valid=f.location_valid.astype(str).str.lower().isin(['true','1']).to_numpy()
        valid = valid & f.lat.between(54.5,57).to_numpy() & f.lon.between(36,39).to_numpy()
        lat,lon=point(context.get('geom',''))
        for i in range(steps):
            mask=bins==i
            if not mask.any():continue
            w=speed[mask];good=np.isfinite(w)&(w>=0)&(w<=130)
            result[i,0]=float(w[good].mean()/50) if good.any() else 0
            result[i,1]=float((w[good]<=1).mean()) if good.any() else 0
            result[i,2]=float(good.mean());result[i,3]=1
            gps=np.flatnonzero(mask&valid)
            if len(gps) and np.isfinite([lat,lon]).all():
                row=f.iloc[gps[-1]];result[i,4]=float(distance(row.lat,row.lon,lat,lon)/10000);result[i,5]=1
        return result
