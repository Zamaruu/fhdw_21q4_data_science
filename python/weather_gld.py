"""
@author: Kevin Liss
"""

import numpy as np
import pandas as pd
import os, glob
from weather_api import saveDFtoJSON, removeOutputFile

divider = 10000000000000000
def durchschnitt(x,w):
    return np.convolve(x, np.ones(w), 'valid') / w

df = pd.read_csv('../backend/import.csv', header=None)


df[0] = pd.to_datetime(df[0])
df = df.sort_values(0)
df[0] = pd.to_numeric(df[0]) / divider

ds = durchschnitt(df[0], 4)

saveDFtoJSON(pd.DataFrame(data=ds))