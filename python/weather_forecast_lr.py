import os
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools
import glob

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=[0,1]))

df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

tavg_ts = pd.Series(df['tavg'].values, index=df['date'], name='tavg')
tavg_ts.index = pd.DatetimeIndex(tavg_ts.index, freq=tavg_ts.index.inferred_freq)

# tsatools.add_trend adds variables const and trend to ridership_ts
tavg_df = tsatools.add_trend(tavg_ts, trend='ct')  # ct=const + trend
print(tavg_df.head())

tavg_lm = sm.ols(formula='tavg ~ trend', data=tavg_df).fit()

# # shorter and longer time series
ax = tavg_ts[1000:].plot()
tavg_lm.predict(tavg_df[1000:]).plot(ax=ax)

ax.set_xlabel('date')
ax.set_ylabel('tavg')
plt.show()

# # Informationen über die Qualität der Regression  (R^2, F-Stat, ...)
print(tavg_lm.summary()) 

 

# (3) Forecast

# Die Regr.Koeffizienten stehen in ridership_lm.params
# Der Faktor am Schluss der Formel ist die Nummer des Monats, z. B. 2. Monat->1694.08. 
# # Dies ist auch der Wert, der in ridership_lm.fittedvalues[1] steht (Index um 1 verschoben))
# print(ridership_lm.params[0] + ridership_lm.params[1] * 2)  # --> 1694.08
# # Zum Vergleich
# print(ridership_lm.fittedvalues[1])                         # --> 1694.08
# # Zukünftige Werte (nicht in fittedvalues enthalten)
# print(ridership_lm.params[0] + ridership_lm.params[1] * range(150,170))
