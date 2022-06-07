import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import modf
import lightgbm as lgb

def lgb_train(fdata, flabel):
    container = np.load(pathtofile)
    ms_data = [container[key] for key in container]
    ms_df = pd.DataFrame(ms_data).fillna(0).to_numpy()
    ms_label = pd.read_csv(pathtofile)

    # train-val split
    while 1: 
        dtrain, dtest, ytrain, ytest = train_test_split(ms_df, ms_label[ms_label.columns[4]],
                                                        test_size=0.2, 
                                                        random_state=np.random.randint(10000))

        if modf(100*(np.sum(ytrain)/len(ytrain)))[1] == modf(100*(np.sum(ytest)/len(ytest)))[1]:
            break

    print("train: ", np.sum(ytrain)/len(ytrain), "test: ", np.sum(ytest)/len(ytest))
    return dtrain, dtest, ytrain, ytest
  
def lr_decay_power_099(current_iter):
    base_learning_rate = (np.random.randint(10) + 1)*0.01 
    lr = base_learning_rate * np.power(0.995, current_iter) 
    return lr if lr > 0.0001 else 0.0001


def model_training(params, dtrain, dtest, ytrain, ytest, modelpath, ini_criteria):
    # loading data
    train_data = lgb.Dataset(dtrain, ytrain, free_raw_data = False)
    val_data = lgb.Dataset(dtest, ytest, free_raw_data = False)
    
    evals_result = {}
    ini_model = modelpath
    model1 = lgb.train(params, 
                      train_data, 
                      valid_sets=val_data,
                      evals_result=evals_result,
                      verbose_eval=False,
                      early_stopping_rounds=100,
                      callbacks=[lgb.reset_parameter(learning_rate=lr_decay_power_099)])
        
    return evals_result
