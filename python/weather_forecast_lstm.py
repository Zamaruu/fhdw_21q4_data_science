import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout, Bidirectional

#Zeitserienforecast mit LSTM Neural-Netzwerk

#Methoden
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
 series = tf.expand_dims(series, axis=-1)
 ds = tf.data.Dataset.from_tensor_slices(series)
 ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
 ds = ds.flat_map(lambda w: w.batch(window_size + 1))
 ds = ds.shuffle(shuffle_buffer)
 ds = ds.map(lambda w: (w[:-1], w[1:]))
 return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
 ds = tf.data.Dataset.from_tensor_slices(series)
 ds = ds.window(window_size, shift=1, drop_remainder=True)
 ds = ds.flat_map(lambda w: w.batch(window_size))
 ds = ds.batch(32).prefetch(1)
 forecast = model.predict(ds)
 return forecast

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

#Main-Programm

# get data file names
path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/*19.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

# Concatenate all data into one DataFrame
weather_pb_df = pd.concat(dfs, ignore_index=True)
print(weather_pb_df.head())

series = np.array(weather_pb_df["tavg"])
time = np.array(weather_pb_df["date"])

plt.plot(time, series)
plt.show()

size = len(series)
split_time = int(size * 0.66)

time_train = time[:split_time]
time_valid = time[split_time:]

temp_train = series[:split_time]
temp_valid = series[split_time:]

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
shuffle_buffer_size = 1000
window_size = 64
batch_size = 256
train_set = windowed_dataset(temp_train, window_size, batch_size, shuffle_buffer_size)
print(temp_train.shape)

# Creating Model
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
 tf.keras.layers.Conv1D(filters=60, kernel_size=5,
 strides=1, padding="causal",
 activation="relu",
 input_shape=[None, 1]),
 tf.keras.layers.LSTM(60, return_sequences=True),
 tf.keras.layers.LSTM(60, return_sequences=True),
 tf.keras.layers.Dense(30, activation="relu"),
 tf.keras.layers.Dense(10, activation="relu"),
 tf.keras.layers.Dense(1),
 tf.keras.layers.Lambda(lambda x: x * 400)
])
model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
 optimizer=optimizer,
 metrics=["acc"])
history = model.fit(train_set,epochs=100)


# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -
# Retrieve a list of list results on training and test data
# sets for each training epoch
# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -
loss=history.history['loss']
epochs=range(len(loss)) # Get number of epochs
# — — — — — — — — — — — — — — — — — — — — — — — — 
# Plot training and validation loss per epoch
# — — — — — — — — — — — — — — — — — — — — — — — — 
plt.plot(epochs, loss, "r")
plt.title("Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])
plt.figure()
zoomed_loss = loss[50:]
zoomed_epochs = range(50,100)
# — — — — — — — — — — — — — — — — — — — — — — — — 
# Plot training and validation loss per epoch
# — — — — — — — — — — — — — — — — — — — — — — — — 
plt.plot(zoomed_epochs, zoomed_loss, "r")
plt.title("Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])
plt.figure()


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, temp_valid)
plot_series(time_valid, rnn_forecast)

#weather_pb_df = weather_pb_df[["date", "tavg"]].set_index("date")
# print(weather_pb_df.head())
# -----------------------------------------------
# training_set = weather_pb_df.iloc[:,1:2].values

# # print(training_set)

# sc = MinMaxScaler(feature_range=(0,1))
# training_set_scaled = sc.fit_transform(training_set)

# print(training_set_scaled)

# x_train = []
# y_train = []
# n_future = 4 # next 4 days temperature forecast
# n_past = 30 # Past 30 days 
# for i in range(0,len(training_set_scaled)-n_past-n_future+1):
#     x_train.append(training_set_scaled[i : i + n_past , 0])     
#     y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])

# x_train , y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )


# regressor = Sequential()
# regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 30 , return_sequences=True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 30 , return_sequences=True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 30))
# regressor.add(Dropout(0.2))
# regressor.add(Dense(units = n_future,activation='sigmoid'))
# regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
# regressor.fit(x_train, y_train, epochs=20,batch_size=32 )


# # read test dataset
# filename = glob.glob(path + "/data" + "/*20.csv")
# print(filename)
# testdataset = pd.read_csv(filename[0])
# #get only the temperature column
# testdataset = testdataset.iloc[:30,1:2].values

# real_temperature = pd.read_csv(filename[0])
# real_temperature = real_temperature.iloc[30:,1:2].values

# testing = sc.transform(testdataset)
# testing = np.array(testing)
# testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))

# predicted_temperature = regressor.predict("")
# predicted_temperature = sc.inverse_transform(predicted_temperature)
# predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))

# plt.plot(real_temperature)
# plt.plot(predicted_temperature)
# plt.show()
#-----------------------------------
