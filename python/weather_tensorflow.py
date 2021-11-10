import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import glob
import time
import math

from sklearn.model_selection import train_test_split

IPython.display.clear_output()

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=[0,1]))
df = pd.concat(dfs, ignore_index=True)

df['date'] = pd.to_numeric(pd.to_datetime(df['date']))
df['tavg'] = pd.to_numeric(df['tavg'])
df = df.sort_values(by=['date'], ascending=True)

df['sin'] = np.sin(df['date'] * 3600 * 24)

df['date'] = pd.to_datetime(df['date'])
df.pop('date')

train, test = train_test_split(df, test_size=0.2)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min'
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=tf.optimizers.Adam(),
    metrics=[tf.metrics.MeanAbsoluteError()]
)

history = model.fit(
    train['tavg'], train['sin'], epochs=100,
    validation_data=(test['tavg'], test['sin']),
    callbacks=[early_stopping]
)


pred = model.predict(df['tavg'])
print(pred)

plt.figure()
df.pop('sin')
plt.plot(df['tavg'])

plt.figure()
plt.plot(pred)