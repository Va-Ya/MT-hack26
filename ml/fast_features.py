"""NumPy implementation of the v3 feature contract for online requests.

Both duplicate policies of v3 are preserved: legacy filters GPS before dedup,
extended features dedup before filtering GPS. Plans are shared and immutable.
"""
from types import SimpleNamespace
import math
import numpy as np
from ml.features import FeatureBuilder,EXTENDED,HISTORY_SECONDS,timestamp,point,distance

NAT=np.iinfo(np.int64).min

class FastFeatureBuilder(FeatureBuilder):
    def __init__(self,telemetry,schedule=None):
        super().__init__(telemetry,schedule)
        self.arrays={}
        for key,f in self.groups.items():
            a={c:f[c].to_numpy(dtype=float) for c in ['speed','lat','lon','heading']}
            a['t']=f.event_time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
            for c in ['receive_time','gps_time']:
                if c in f:a[c]=f[c].to_numpy(dtype='datetime64[ns]').astype(np.int64)
            a['valid']=f.location_valid.astype(str).str.lower().isin(['true','1']).to_numpy() & (a['lat']>=54.5)&(a['lat']<=57)&(a['lon']>=36)&(a['lon']<=39)
            a['speed']=np.where((a['speed']>=0)&(a['speed']<=130),a['speed'],np.nan)
            self.arrays[key]=a

    def build(self,context):
        T=timestamp(context['T']);tn=T.value
        horizon=(timestamp(context['target_time_begin'])-T).total_seconds()
        if not 600<horizon<=900:raise ValueError('Target must lie in (T+10min, T+15min]')
        out=dict(cur_dev_s=float(context['cur_dev_s']),target_stop_id=str(context['target_stop_id']),hour=T.hour+T.minute/60,weekday=T.weekday(),is_weekend=int(T.weekday()>=5),scheduled_time_to_target=horizon)
        cols=['speed_now','speed_mean_1m','speed_mean_3m','speed_mean_5m','speed_std_5m','speed_min_5m','speed_max_5m','speed_delta_1m','speed_delta_3m','speed_delta_5m','acceleration','stationary_time','distance_travelled_1m','distance_travelled_5m','heading','heading_delta','distance_to_target_stop','telemetry_age_s','telemetry_count_5m','speed_drop']
        out.update({c:np.nan for c in cols+EXTENDED})
        lat,lon=point(context.get('geom',''));h=out['hour']*2*np.pi/24
        out.update(hour_sin=np.sin(h),hour_cos=np.cos(h),target_lat=lat,target_lon=lon,gps_missing=1,vehicle_key=str(context['tr_id']),target_geo_key=f'{lat:.4f}:{lon:.4f}' if np.isfinite([lat,lon]).all() else 'missing',segment_geo_key='missing',gps_geo_key='missing')
        a=self.arrays.get(str(context['tr_id']))
        if a is None:
            out['telemetry_count_5m']=0;return out
        t=a['t'];lo=t.searchsorted(tn-HISTORY_SECONDS*10**9);hi=t.searchsorted(tn,side='right')
        idx=np.arange(lo,hi)
        for c in ['receive_time','gps_time']:
            if c in a:idx=idx[(a[c][idx]==NAT)|(a[c][idx]<=tn)]
        def dedup(rows):
            if not len(rows):return rows
            times=t[rows];return rows[np.r_[times[:-1]!=times[1:],True]]
        legacy=dedup(idx[a['valid'][idx]])
        if not len(legacy):out['telemetry_count_5m']=0
        else:self._legacy(out,a,legacy,tn,lat,lon)
        idx=dedup(idx)
        if not len(idx):return out
        sec=(t[idx]-tn)/1e9;speed=a['speed'][idx];valid=a['valid'][idx];recent=sec>=-300
        if recent.any():
            st=sec[recent]
            out.update(invalid_gps_fraction_5m=float(1-valid[recent].mean()),speed_missing_fraction_5m=float(1-np.isfinite(speed[recent]).mean()),telemetry_span_5m=float(st[-1]-st[0]),max_gap_5m=float(np.diff(st).max()) if len(st)>1 else np.nan)
        if 'receive_time' in a and a['receive_time'][idx[-1]]!=NAT:
            out['receive_lag_s']=max(0,(int(a['receive_time'][idx[-1]])-int(t[idx[-1]]))/1e9)
        for m in [1,3,5,10,20]:
            mask=(sec>=-60*m)&np.isfinite(speed);w=speed[mask]
            out[f'speed_count_{m}m']=int(len(w))
            if len(w):
                p10,median,p90=np.quantile(w,[.1,.5,.9])
                out.update({f'speed_median_{m}m':float(median),f'speed_p10_{m}m':float(p10),f'speed_p90_{m}m':float(p90),f'stop_fraction_{m}m':float((w<=1).mean())})
            if m==5 and len(w)>1 and np.ptp(sec[mask])>0:
                x=sec[mask];out['speed_slope_5m']=float(np.dot(x-x.mean(),w-w.mean())/np.dot(x-x.mean(),x-x.mean()))
        gps=idx[valid]
        if not len(gps) or (tn-t[gps[-1]])/1e9>300:return out
        last=gps[-1];heading=a['heading'][last]
        out.update(gps_missing=0,gps_lat_now=float(a['lat'][last]),gps_lon_now=float(a['lon'][last]),gps_geo_key=f"{a['lat'][last]:.3f}:{a['lon'][last]:.3f}",heading_sin=np.sin(np.radians(heading)),heading_cos=np.cos(np.radians(heading)))
        ds=distance(a['lat'][gps],a['lon'][gps],lat,lon)
        out['distance_over_horizon_mps']=float(ds[-1]/horizon)
        recent_g=t[gps]>=tn-300*10**9
        if recent_g.sum()>1:
            d=ds[recent_g];gt=t[gps][recent_g];dt=(gt[-1]-gt[0])/1e9
            out['distance_change_5m']=float(d[0]-d[-1]);out['approach_speed_mps']=float((d[0]-d[-1])/dt) if dt>0 else np.nan
        self._plan(out,context,SimpleNamespace(lat=float(a['lat'][last]),lon=float(a['lon'][last])))
        return out

    def _legacy(self,out,a,idx,tn,lat,lon):
        sec=(a['t'][idx]-tn)/1e9;speed=a['speed'][idx];age=-sec[-1]
        out['telemetry_age_s']=age;out['telemetry_count_5m']=int((sec>=-300).sum())
        if age>300:return
        out['speed_now']=speed[-1]
        for m in [1,3,5]:
            w=speed[(sec>=-m*60)&np.isfinite(speed)]
            out[f'speed_mean_{m}m']=float(w.mean()) if len(w) else np.nan
            old=np.flatnonzero((sec<=-m*60)&(sec>=-m*60-60)&np.isfinite(speed))
            out[f'speed_delta_{m}m']=speed[-1]-speed[old[-1]] if len(old) else np.nan
            if m==5 and len(w):out.update(speed_std_5m=float(w.std()),speed_min_5m=float(w.min()),speed_max_5m=float(w.max()))
        dt=np.diff(sec)
        if len(idx)>=2 and 0<dt[-1]<=60:
            out['acceleration']=(speed[-1]-speed[-2])/3.6/dt[-1]
            out['heading_delta']=(float(a['heading'][idx[-1]])-float(a['heading'][idx[-2]])+180)%360-180
        out['heading']=float(a['heading'][idx[-1]])
        stationary=0.
        for j in range(len(speed)-1,0,-1):
            if not (speed[j]<=1 and speed[j-1]<=1 and dt[j-1]<=60):break
            stationary+=dt[j-1]
        out['stationary_time']=stationary
        if len(idx)>=2:
            lats=a['lat'][idx];lons=a['lon'][idx]
            steps=distance(lats[:-1],lons[:-1],lats[1:],lons[1:]);good=(dt>0)&(dt<=60)&(steps/np.maximum(dt,1)<=130/3.6)
            for m in [1,5]:out[f'distance_travelled_{m}m']=float(steps[good&(sec[:-1]>=-m*60)].sum())
        out['distance_to_target_stop']=float(distance(float(a['lat'][idx[-1]]),float(a['lon'][idx[-1]]),lat,lon))
        out['speed_drop']=max(0.,out['speed_mean_5m']-out['speed_now']) if math.isfinite(out['speed_mean_5m']) and math.isfinite(out['speed_now']) else np.nan
