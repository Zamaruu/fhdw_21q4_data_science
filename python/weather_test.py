import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse.sputils import getdata
from statsmodels.tsa import tsatools
import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split

import os, glob

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*20.csv")

def get_data(cols_array=[0,1]):
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, usecols=cols_array))

    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

df = get_data()
ts = pd.Series(df['tavg'].values, index=df['date'], name="tavg")
ts.index = pd.DatetimeIndex(ts.index, freq=ts.index.inferred_freq)

tavg_df = tsatools.add_trend(ts, trend='ct')

tavg_lm = sm.ols(formula="tavg ~ trend", data=tavg_df).fit()
print(tavg_lm.summary())

# oh wunder, ist schlecht

tavg_rmse = np.sqrt(np.mean(tavg_lm.resid)**2)
print("rmse : ", tavg_rmse)

ax = ts.plot()
ax.set_xlabel('time')
ax.set_ylabel('tavg')
tavg_lm.predict(tavg_df).plot()
plt.show()

tavg_df = tsatools.add_trend(ts, trend='ctt')
tavg_lmq = sm.ols(formula='tavg ~ trend + trend_squared', data=tavg_df).fit()
print("-------------")
print(tavg_lmq.summary())

plt.figure()
ax = ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('tavg')
tavg_lmq.predict(tavg_df).plot()
plt.show()


######################################
from sklearn.neural_network import MLPRegressor

df = get_data()
date = df.pop('date')

n = len(df)
train = df[0:int(n*0.8)]
test = df[int(n*0.8):]

df_shift = pd.concat([train['tavg'], train['tavg'].shift(), train['tavg'].shift(2)], axis=1)
df_shift['tavg'] = df_shift['tavg'].replace(np.nan, 0)

clf = MLPRegressor(hidden_layer_sizes=2, max_iter=5000, random_state=1)

clf.fit(train, df_shift)
pre = clf.predict(test)

plt.figure()
plt.plot(test, 'blue')
plt.show()
plt.figure()
plt.plot(pre, 'red')
plt.show()