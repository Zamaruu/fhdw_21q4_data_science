import os
import datetime
from numpy.random.mtrand import shuffle

from pandas.core.algorithms import mode

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv('./data/timestamp.CSV', usecols=[0,1], sep=";")

df['date'] = pd.to_numeric(df['date'])
df['tavg'] = pd.to_numeric(df['tavg'])

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', metrics=['accuracy'])
model.fit(train_df['tavg'], train_df['date'], epochs=10, batch_size=20, shuffle=False)

pred = model.predict(test_df['tavg'])

plt.figure()
plt.plot(test_df['date'], test_df['tavg'])

plt.figure()
plt.plot(test_df['date'], pred)


pred = model.predict(val_df['tavg'])