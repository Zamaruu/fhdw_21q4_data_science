"""
@author: Maximilian Ditz
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from weather_api import convertDateTimeListToString, getApiArguments, saveDictToJSON

# Argumente für diese Funktion Lag, Zeitspanne

# __________________________Importieren und Transformieren der AKF-Daten__________________________
path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

weather_pb_df = pd.concat(dfs, ignore_index=True)
weather_pb_df['date'] = pd.to_datetime(weather_pb_df.date, format='%Y/%m/%d')
weather_pb_df = weather_pb_df.sort_values('date')
weather_pb_df = weather_pb_df[['date', 'tavg']].set_index(['date'])

print(weather_pb_df.head())
print(weather_pb_df.tail())
print(weather_pb_df.isnull().sum())


plt.plot(weather_pb_df['1993-01-01':'1994-01-01'])
plt.show()
acf = weather_pb_df['1993-01-01':'1994-01-01']

# __________________________Durchführung der AKF auf die Wetter-Daten__________________________
tsaplots.plot_acf(acf, lags=35)
plt.show()

acf_erg, ci = sm.tsa.acf(acf, nlags= 35, alpha=0.05) 

# __________________________Zentrieren der Confidence Level__________________________
for i in range(0, len(acf_erg)):
    ci[i] = ci[i] - acf_erg[i]
    i = i + 1

ci_pos = [x[0] for x in ci]
ci_neg = [x[1] for x in ci]

# __________________________Rückgabe-Dictionary erstellen__________________________
plt.plot(acf_erg)
plt.plot(ci)
plt.show()

acf_output = {
    "acf": acf_erg.tolist(),
    "ci_pos": ci_pos,
    "ci_neg": ci_neg 
}

# __________________________AKF-Details zurückgeben__________________________
saveDictToJSON("output.json", acf_output)