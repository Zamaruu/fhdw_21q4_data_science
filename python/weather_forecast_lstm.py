import os
import glob
import numpy as np
import pandas as pd
from scipy.sparse import data
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout, Bidirectional
from numpy import array
from tensorflow.python.ops.gen_array_ops import reshape


def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# ------------------------------------------------------------
# Daten einlesen
# Trainingsdaten
path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*19.csv")

dfs = []
for filename in filenames:
    print(filename)
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

# Concatenate all data into one DataFrame
weather_pb_df = pd.concat(dfs, ignore_index=True)
weather_pb_df['date'] = pd.to_datetime(weather_pb_df['date'])
weather_pb_df = weather_pb_df.sort_values(by=['date'], ascending=True)
print(weather_pb_df.head())
print(weather_pb_df.tail())

# Predictiondaten
testfile = glob.glob(path + "/data" + "/*20.csv")
pred_pb_df = pd.read_csv(testfile[0], usecols=["date", "tavg"])
pred_pb_df['date'] = pd.to_datetime(pred_pb_df['date'])
pred_pb_df = pred_pb_df.sort_values(by=['date'], ascending=True)
print(pred_pb_df.head())
# ---------------------------------------------------------------
# Daten für Model vorbereiten

n_timestamp = 10
train_days = int(len(weather_pb_df) * 0.66)  # number of days to train from
n_epochs = 25
filter_on = 1



train_set = weather_pb_df[0:train_days].reset_index(drop=True)
test_set = weather_pb_df[train_days:].reset_index(drop=True)
pred_set = pred_pb_df[:].reset_index(drop=True)
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values
prediction_set = pred_set.iloc[:, 1:2].values

print(training_set)
print(testing_set)
print(prediction_set)
# ------------------------------------------------------------------
# Daten Normalisiern

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)
prediction_set_scaled = sc.fit_transform(prediction_set)

X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_pred, y_predi = data_split(prediction_set_scaled, n_timestamp)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[1], 1)

# ------------------------------------------------------------------
# Stacked LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
print(model.summary())

#
# Start training
#
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
loss = history.history['loss']
epochs = range(len(loss))
print(history)

# ------------------------------------------------------------------
# Prediction
# y_predicted = model.predict(array(weather_pb_df["tavg"]))
y_predicted = model.predict(X_pred)

#
# 'De-normalize' the data
#
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()

# ------------------------------------------------------------------
# Plotten
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(weather_pb_df['tavg'], color = 'black', linewidth=1, label = 'True value')
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("All data")


plt.subplot(3, 2, 3)
plt.plot(pred_pb_df["tavg"], color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled, color = 'red',  linewidth=1, label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (n days)")

plt.subplot(3, 2, 4)
plt.plot(pred_pb_df["tavg"][0:75], color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled[0:75], color = 'red', label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (first 75 days)")