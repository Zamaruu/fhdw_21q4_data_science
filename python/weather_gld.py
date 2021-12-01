"""
@author: Kevin Liss
"""

import numpy as np
import pandas as pd
import os, glob

from weather_api import saveDictToJSON

def durchschnitt(x,w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_data():
    path = os.path.dirname(__file__)
    filenames = glob.glob(path + "/data" + "/export*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, usecols=[0,1]))

    df = pd.concat(dfs, ignore_index=True)
    return df


def getGLD(start = 0, end = 1):
    df = get_data()
    return durchschnitt(df['tavg'], 30)[start-30:end-30].tolist()