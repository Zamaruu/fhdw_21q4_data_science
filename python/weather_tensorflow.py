import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
import tensorflow as tf
import glob

IPython.display.clear_output()

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/export*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=[0,1,9]))
df = pd.concat(dfs, ignore_index=True)

df['date'] = pd.to_numeric(pd.to_datetime(df['date']))
df['sin'] = np.sin(df['date'] / 1e9 / 3600 / 60 / 24)
df['date'] = df['date'] / 1e16

df['tavg'] = pd.to_numeric(df['tavg'])

df = df.fillna(0)
df = df.sort_values(by=['date'], ascending=True)
print(df.head())

n = len(df)
train = df[0:int(n*0.9)]
test = df[int(n*0.9):]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(1)
])

history = model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

dataset = train.copy()

train_x = dataset[['date', 'pres']]
train_y = dataset['tavg']

model.fit(train_x, train_y, epochs=100, validation_split=0.2)

pred = model.predict(test['date'])

plt.figure()
plt.plot(pred, "red")
plt.plot(test['tavg'].values, "green")
