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
        self.columns=self.metadata["features"]
        self.version=self.metadata["model_version"]
    def frame(self, frame):
        frame=frame[self.columns].copy()
        for c in self.metadata["categorical_features"]:
            frame[c]=frame[c].astype(str)
        return frame
    def predict_frame(self, frame):
        values=np.asarray(self.model.predict(self.frame(frame)))
        if self.metadata["residual"]:
            values += frame.cur_dev_s.to_numpy()
        return values
    def predict(self, features):
        return float(self.predict_frame(pd.DataFrame([features]))[0])
