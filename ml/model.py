import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

class Predictor:
    def __init__(self, directory="models"):
        directory=Path(directory)
        self.metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        self.model=CatBoostRegressor()
        self.model.load_model(str(directory/"best_model.cbm"))
        self.members=[]
        for member in self.metadata.get('members',[]):
            model=self.model if member['file']=='best_model.cbm' else CatBoostRegressor()
            if model is not self.model:model.load_model(str(directory/member['file']))
            self.members.append((model,float(member['weight']),bool(member['residual']),float(member.get('target_scale',1))))
        self.columns=self.metadata["features"]
        self.version=self.metadata["model_version"]
        self.inference_threads=int(self.metadata.get('inference_threads',1))
        self.cats=set(self.metadata['categorical_features'])
        self.member_columns=[tuple(m.get('features',self.columns)) for m in self.metadata.get('members',[])]
        self.v4=self.metadata.get('feature_version')=='causal-v4'
    def frame(self, frame,columns=None):
        columns=self.columns if columns is None else list(columns)
        frame=frame[columns].copy()
        for c in self.metadata["categorical_features"]:
            if c in frame:frame[c]=frame[c].astype(str)
        return frame
    def predict_frame(self, frame):
        if self.v4 and any(c not in frame for c in self.columns):
            from ml.features_v4 import augment_frame
            frame=augment_frame(frame)
        if self.members:
            prepared={cols:self.frame(frame,cols) for cols in set(self.member_columns)}
            values=np.zeros(len(frame),dtype=float)
            for i,(model,weight,residual,scale) in enumerate(self.members):
                p=np.asarray(model.predict(prepared[self.member_columns[i]],thread_count=self.inference_threads))*scale
                if residual:p=p+frame.cur_dev_s.to_numpy()
                values+=weight*p
            return values
        values=np.asarray(self.model.predict(self.frame(frame),thread_count=self.inference_threads))*float(self.metadata.get('target_scale',1))
        if self.metadata["residual"]:
            values += frame.cur_dev_s.to_numpy()
        return values
    def predict(self, features):
        if self.v4:
            from ml.features_v4 import augment
            features={**features,**augment(features)}
        def row(cols):return [[str(features[c]) if c in self.cats else features[c] for c in cols]]
        delay=float(features.get('cur_dev_s',0))
        if self.members:
            prepared={cols:row(cols) for cols in set(self.member_columns)}
            return float(sum(weight*(float(model.predict(prepared[self.member_columns[i]],thread_count=self.inference_threads)[0])*scale+(delay if residual else 0)) for i,(model,weight,residual,scale) in enumerate(self.members)))
        value=float(self.model.predict(row(self.columns),thread_count=self.inference_threads)[0])*float(self.metadata.get('target_scale',1))
        return value+(delay if self.metadata['residual'] else 0)
