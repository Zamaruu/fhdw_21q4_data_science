"""
@author: Kevin Liss
"""

# __________________________Import Section__________________________
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import glob

from weather_api import getApiArguments, saveDictToJSON

# __________________________Format Section__________________________
# Ausgabe formatieren für bessere lesbarkeit
np.set_printoptions(precision=3, suppress=True)

# kontrolle der Tensorflow Version
print("TF Version: ", tf.__version__)

# Konstante zum einfachen rechnen von Daten in sekunden (160 * divier ~ 2020)
divider = 10000000000000000
test_results = {}

# Funktion
# eingabe: array aus Integer zur definition der einzulesenden Spalten
# ausgabe: Gibt ein DataFrame zurück mit den eingelesenen Daten


def get_data(cols_array):
    path = os.path.dirname(__file__)
    filenames = glob.glob(path + "/data" + "/export*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, usecols=cols_array))

    df = pd.concat(dfs, ignore_index=True)
    return df


df = get_data([0, 1, 9])

df['date'] = pd.to_datetime(df['date'])  # Formatiere Spalte "Date" zu DateTim
df = df.sort_values('date')  # Sortiere nach Datum
# Index des DataFrames nach sortieren falsch => neu bauen
df = df.reset_index(drop=True)
df['date'] = pd.to_numeric(df['date'])
# datum in sekunden in menschenlesbar gestalten
df['date'] = df['date'] / divider

df['tavg'] = pd.to_numeric(df['tavg'])
df['pres'] = pd.to_numeric(df['pres'])

for col in df:
    df[col] = df[col].fillna(0)  # alle "NaN" werte mit "0" ersetzen

# Aufteilen der Daten in "Training" und "Test". Die ersten 90% => Trainingsdaten
n = len(df)
train_dataset = df[0:int(n*0.9)]
test_dataset = df[int(n*0.9):]

# Aufteilen der Trainings/Test Daten in X, Y Werte
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('tavg')
test_labels = test_features.pop('tavg')

date_min = np.amin(train_features['date'])
date_max = np.amax(train_features['date'])
date_dif = date_max - date_min

# __________________________Einlesen der Forecast-X Werte__________________________
args = getApiArguments()
print(args)
df_forecast = pd.date_range(args['start'], args['end'], freq="d").to_frame()
df_forecast.reset_index(drop=True, inplace=True)

forecast_dates = pd.DataFrame()
forecast_dates[0] = df_forecast[0].dt.strftime("%Y-%m-%d")

past_dates = pd.DataFrame()
past_dates[0] = test_features['date'] * divider
past_dates[0] = pd.to_datetime(past_dates[0]).dt.strftime("%Y-%m-%d")

df_forecast[0] = pd.to_numeric(pd.to_datetime(df_forecast[0].values)) / divider
df_forecast[1] = 1000
print(df_forecast.head())

# __________________________Performance Plotter__________________________


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [tavg]')
    plt.legend()
    plt.grid(True)


def plot_tavg(x, y):
    plt.figure()
    plt.scatter(train_features['date'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('date')
    plt.ylabel('tavg')
    plt.legend()


# Simpler Normalizer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Lin Reg 1 Var
tavg = np.array(train_features['date'])

# Normalizer anhand von X-Achse
tavg_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
tavg_normalizer.adapt(tavg)


# __________________________Linear Regression nach "Datum"__________________________
tavg_date = tf.keras.Sequential([
    tavg_normalizer,
    layers.Dense(1)
])

tavg_date.compile(
    optimizer=tf.optimizers.Adam(0.1),
    loss="mse"
)

history = tavg_date.fit(
    train_features['date'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

test_results['date_model'] = tavg_date.evaluate(
    test_features['date'],
    test_labels,
    verbose=0
)

print(df_forecast[0])
prediction_linear = tavg_date.predict(df_forecast[0])

# __________________________Linear Regression nach "Datum" und "Pressure"__________________________
# linear_model = tf.keras.Sequential([
#     normalizer,
#     layers.Dense(1)
# ])

# linear_model.compile(
#     optimizer=tf.optimizers.Adam(learning_rate=0.1),
#     loss='mse'
# )

# history = linear_model.fit(
#     train_features,
#     train_labels,
#     epochs=100,
#     verbose=0,
#     validation_split=0.2
# )

# test_results['linear_model'] = linear_model.evaluate(
#     test_features['date'], test_labels, verbose=0
# )


def build_and_compile_model(norm):
    model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1)),
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=tf.keras.metrics.MeanSquaredLogarithmicError()
    )
    return model

# __________________________Neurales Netzt mit "Datum" als eingabe__________________________
# dnn_date_model = build_and_compile_model(tavg_normalizer)
# history = dnn_date_model.fit(
#     train_features['date'],
#     train_labels,
#     validation_split=0.2,
#     epochs=100,
#     verbose=0
# )

# x = tf.linspace(date_min, date_max, date_dif)
# y = dnn_date_model.predict(x)

# test_results['dnn_date_model'] = dnn_date_model.evaluate(
#     test_features['date'], test_labels,
#     verbose=0
# )


# __________________________Neurales Netzt mit "Datum" und "Pressure" als eingabe__________________________
dnn_model = build_and_compile_model(normalizer)
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=250,
    verbose=0
)
test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)

# Perfofmance ausgabe
print(pd.DataFrame(test_results).T)

print(test_features.head())
test_predictions = dnn_model.predict(test_features)

plt.figure()
plt.plot(test_labels.values, "green")
plt.plot(test_predictions, "red")

# vergleich training
train_predictions = dnn_model.predict(train_features.values)

plt.figure()
plt.plot(train_labels.values, "green")
plt.plot(train_predictions, "red")

predict = dnn_model.predict(df_forecast)

# __________________________Speichern der Ausgabe__________________________


def toSimpleList(list):
    result = []
    for i in range(0, len(list)):
        result.append(list[i][0])
    return result


plt.figure()
plt.plot(predict)

result = DataFrame()
df_forecast[0] = pd.to_datetime(df_forecast[0] * divider)

past_len = 365*2 # ~ 2 Jahre
result = {
    "forecast_tavg": toSimpleList(predict.tolist()),
    "forecast_linear": toSimpleList(prediction_linear.tolist()),
    "forecast_dates": toSimpleList(forecast_dates.values.tolist()),
    "past_date": toSimpleList(past_dates[past_len:].values.tolist()),
    "past_tavg": test_labels[past_len:].values.tolist()
}

saveDictToJSON("output.json", result)
