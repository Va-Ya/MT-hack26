"""Stateless streaming-ready inference on a telemetry snapshot known at T."""
import pandas as pd
from ml.features import timestamp
from ml.fast_features import FastFeatureBuilder as FeatureBuilder
from ml.model import Predictor

class CausalInference:
    def __init__(self,models,schedule):
        self.predictor=Predictor(models)
        self.schedule=schedule[['tr_id','tt_action_item_id','time_begin','geom']].copy()
        self.plans=FeatureBuilder(pd.DataFrame(),self.schedule).plans
        self.targets={}
        for row in self.schedule.itertuples():
            self.targets.setdefault((str(row.tr_id),str(row.tt_action_item_id)),[]).append(row)

    def predict_at(self,telemetry,context):
        """Caller supplies cur_dev_s observed at T; history may contain future rows.

        No outcomes, labels or future actual arrival times are accessed. The
        backend is responsible for deriving an honest current-delay snapshot.
        """
        match=self.targets.get((str(context['tr_id']),str(context['target_stop_id'])),[])
        if len(match)!=1:raise ValueError('Unknown or ambiguous planned target')
        stop=match[0]
        if timestamp(stop.time_begin)!=timestamp(context['target_time_begin']):raise ValueError('Target plan mismatch')
        ctx={**context,'geom':stop.geom}
        features=FeatureBuilder(telemetry,self.plans).build(ctx)
        return {'timestamp':str(timestamp(ctx['T'])),'target_stop_id':str(ctx['target_stop_id']),'predicted_delay_s':self.predictor.predict(features),'prediction_horizon_s':features['scheduled_time_to_target'],'model_version':self.predictor.version}
