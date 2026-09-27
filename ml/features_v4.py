"""Cheap deterministic transforms of causal v3 features, shared online/offline.

No labels, future observations, sample IDs or historical target lookup tables.
Kinematic quantities are model INPUTS, never substituted for an ML prediction.
"""
import math
import numpy as np
import pandas as pd

FEATURE_VERSION='causal-v4'
CAT_FEATURES=['v4_target_zone','v4_position_zone']

def augment(row):
    def n(key):
        value=row.get(key,np.nan)
        return float(value) if value is not None else float('nan')
    def ratio(a,b):return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-6 else float('nan')
    def zone(a,b):return f'{a:.3f}:{b:.3f}' if math.isfinite(a) and math.isfinite(b) else 'missing'
    delay=n('cur_dev_s');horizon=n('scheduled_time_to_target')
    progress=n('segment_progress');planned=n('planned_progress');segment_s=n('planned_segment_s')
    out={'v4_delay_abs':abs(delay),'v4_delay_horizon':ratio(delay,horizon),
         'v4_remaining_plan_s':horizon+delay,
         'v4_position_lag_s':(planned-progress)*segment_s,
         'v4_position_lag_m':(planned-progress)*n('planned_segment_m'),
         'v4_cross_track_ratio':ratio(n('segment_cross_track_m'),n('planned_segment_m')),
         'v4_speed_ratio_1_5':ratio(n('speed_mean_1m'),n('speed_mean_5m')),
         'v4_speed_ratio_3_20':ratio(n('speed_mean_3m'),n('speed_median_20m')),
         'v4_stop_change_1_5':n('stop_fraction_1m')-n('stop_fraction_5m'),
         'v4_stop_change_5_20':n('stop_fraction_5m')-n('stop_fraction_20m'),
         'v4_speed_spread_5m':n('speed_p90_5m')-n('speed_p10_5m'),
         'v4_speed_vs_plan':n('speed_mean_3m')-n('planned_speed_kmh'),
         'v4_approach_vs_speed':ratio(n('approach_speed_mps')*3.6,n('speed_mean_5m')),
         'v4_distance_path_ratio':ratio(n('plan_distance_to_target_m'),n('distance_to_target_stop')),
         'v4_idle_horizon':ratio(n('stationary_time'),horizon),
         'v4_target_zone':zone(n('target_lat'),n('target_lon')),
         'v4_position_zone':zone(n('gps_lat_now'),n('gps_lon_now'))}
    for minute in [1,3,5]:
        speed=n(f'speed_mean_{minute}m')
        # Cap extreme ratios in low-speed conditions; still just features.
        eta=ratio(n('plan_distance_to_target_m'),max(speed,2)/3.6)
        out[f'v4_eta_{minute}m_s']=min(eta,7200) if math.isfinite(eta) else float('nan')
        out[f'v4_eta_margin_{minute}m_s']=out[f'v4_eta_{minute}m_s']-horizon
    return out

NUMERIC_FEATURES=[k for k in augment({}) if k not in CAT_FEATURES]
FEATURES=NUMERIC_FEATURES+CAT_FEATURES

def augment_frame(frame):
    extra=pd.DataFrame([augment(row) for row in frame.to_dict('records')],index=frame.index)
    result=frame.drop(columns=[c for c in FEATURES if c in frame]).copy()
    return pd.concat([result,extra],axis=1)
