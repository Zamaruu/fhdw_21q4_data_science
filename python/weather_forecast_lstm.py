import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#Zeitserienforecast mit LSTM Neural-Netzwerk

#Methoden

#Main-Programm

# get data file names
path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*20.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
weather_pb_df = pd.concat(dfs, ignore_index=True)
print(weather_pb_df.head())
# weather_pb_df['date'] = pd.to_datetime(weather_pb_df.date, format='%Y/%m/%d')
# weather_pb_df = weather_pb_df.sort_values('date')

# print(weather_pb_df.head())
# print(weather_pb_df.tail())
# # Check if any null or NaN values in data
# print(weather_pb_df.isnull().sum())

# convert the date information to a datetime object
# weather_pb_df = weather_pb_df[['tavg']]
