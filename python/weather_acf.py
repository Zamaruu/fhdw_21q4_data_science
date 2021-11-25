import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

# Argumente für diese Funktion Lag, Zeitspanne

# get data file names
path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

# Concatenate all data into one DataFrame
weather_pb_df = pd.concat(dfs, ignore_index=True)
weather_pb_df['date'] = pd.to_datetime(weather_pb_df.date, format='%Y/%m/%d')
weather_pb_df = weather_pb_df.sort_values('date')

print(weather_pb_df.head())
print(weather_pb_df.tail())

# Check if any null or NaN values in data
print(weather_pb_df.isnull().sum())

# convert the date information to a datetime object
weather_pb_df = weather_pb_df[['date', 'tavg']].set_index(['date'])

end = 365

plt.plot(weather_pb_df['1993-01-01':'1994-01-01'])
acf = weather_pb_df['1993-01-01':'1994-01-01']


#print(acf)

# Je mehr Lags desto weiter reicht die Zeit zurück und desto weniger hängen die Werte voneinander ab
x = tsaplots.plot_acf(acf, lags=35)
print(x)

acf = acf["tavg"].autocorr(lag=35)
print("ACF: " + str(acf))

plt.show()

# convert dataframe column to series (name is used to label the data)
# weather_ts = pd.Series(weather_pb_df.tavg.values, index=weather_pb_df.date, name='tavg')
# # define the time series frequency
# weather_ts.index = pd.DatetimeIndex(weather_ts.index, freq=weather_ts.index.inferred_freq)

# #  linearen Trend hinzufügen
# weather_pb_df = tsatools.add_trend(weather_ts, trend='ct')
# # Monate hinzufügen
# weather_pb_df['date'] = weather_pb_df.index.date

# # Alles hier drunter funktioniert noch nicht!!!!!!!

# # partition the data
# # Split into train and valid 
# nTrain = len(weather_pb_df) - len(weather_pb_df) / 3    # Anzahl Trainingsdaten
# train_df = weather_pb_df[:nTrain]
# valid_df = weather_pb_df[nTrain:]

# # acf = autocorrelation function
# tsaplots.plot_acf(train_df['1993-01-01':'1994-01-01'].tavg) 
# # der blaue Bereich im Bild besteht aus den Konf. Intervallen. Kann mit
# # Parameter alpha geändert werden. Default: alpha=0.05 (95% Konf. Int.)
# plt.show()