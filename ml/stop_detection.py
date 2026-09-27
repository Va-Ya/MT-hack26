"""Conservative observed stop arrival detection; planned times are known inputs."""
import numpy as np
import pandas as pd
from ml.features import timestamp,point,distance

class StopDetector:
    def __init__(self,schedule,radius_m=50,dwell_s=15,max_gap_s=30):
        self.radius=radius_m;self.dwell=dwell_s;self.max_gap=max_gap_s
        self.groups={};self.pending={};self.confirmed={};self.last_time={}
        for tr,group in schedule.groupby('tr_id',sort=False):
            plans=group[['tt_action_item_id','time_begin','geom']].copy()
            plans['time_begin']=pd.to_datetime(plans.time_begin,format='mixed')
            plans=plans.sort_values('time_begin',kind='stable')
            coords=np.asarray([point(g) for g in plans.geom],dtype=float)
            self.groups[str(tr)]=(plans,coords)
    def ingest(self,record,available_at):
        tr=str(record['tr_id']);at=timestamp(record['event_time']);known=timestamp(available_at)
        if tr in self.last_time and at<=self.last_time[tr]:return None
        self.last_time[tr]=at
        if tr not in self.groups:return None
        valid=str(record.get('location_valid')).lower() in ('true','1')
        lat,lon,speed=record.get('lat'),record.get('lon'),record.get('speed')
        if not valid or any(v is None or not np.isfinite(v) for v in [lat,lon,speed]) or not (0<=speed<=3):
            self.pending.pop(tr,None);return None
        plans,xy=self.groups[tr]
        mask=np.isfinite(xy).all(axis=1)
        mask &= (np.abs((plans.time_begin-at).dt.total_seconds().to_numpy())<=1800)
        mask &= ~plans.tt_action_item_id.astype(str).isin(self.confirmed.get(tr,set())).to_numpy()
        candidates=np.flatnonzero(mask & (distance(lat,lon,xy[:,0],xy[:,1])<=self.radius))
        # Nearby repeated plan events cannot be disambiguated from GPS alone.
        if len(candidates)!=1:
            self.pending.pop(tr,None);return None
        stop=plans.iloc[candidates[0]];key=str(stop.tt_action_item_id);pending=self.pending.get(tr)
        if pending is None or pending['key']!=key or (at-pending['last']).total_seconds()>self.max_gap:
            pending=dict(key=key,first=at,last=at);self.pending[tr]=pending
        else:pending['last']=at
        if (at-pending['first']).total_seconds()<self.dwell:return None
        self.confirmed.setdefault(tr,set()).add(key);self.pending.pop(tr,None)
        return dict(tr_id=tr,target_stop_id=key,actual_time=pending['first'],observed_at=known,source='gps_geofence',confidence='confirmed_dwell')
