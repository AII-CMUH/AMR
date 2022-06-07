import os
import numpy as np
import pandas as pd
import lightgbm as lgb

def pred_results(filepath, model):
    container = np.load(pathtofile)
    ms_data = [container[key] for key in container]
    ms_df = pd.DataFrame(ms_data).fillna(0)
    ms_df = ms_df.to_numpy()
    return model.predict(ms_df)
