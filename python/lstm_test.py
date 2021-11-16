import os
import glob
from types import LambdaType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/export*.csv")

dfs = []
for filename in filenames:
    print(filename)
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

# Concatenate all data into one DataFrame
weather_pb_df = pd.concat(dfs, ignore_index=True)
weather_pb_df.dropna()
weather_pb_df['date'] = pd.to_datetime(weather_pb_df['date'])
weather_pb_df = weather_pb_df.sort_values(by=['date'], ascending=True)
print(weather_pb_df.info())
print(weather_pb_df.head())
print(weather_pb_df.tail())

plt.plot(weather_pb_df['date'], weather_pb_df['tavg'])
plt.show()

# --------------------------------------------------------
# Data Preprocessing

temp_data = weather_pb_df['tavg'].values
temp_data = temp_data.reshape((-1,1))

split_percent = 0.67
split = int(split_percent*len(temp_data))

temp_train = temp_data[:split]
temp_test = temp_data[split:]

date_train = weather_pb_df['date'][:split]
date_test = weather_pb_df['date'][split:]

print(len(temp_train))
print(len(temp_test))

# --------------------------------------------------
look_back = 30

train_generator = TimeseriesGenerator(temp_train, temp_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(temp_test, temp_test, length=look_back, batch_size=1)

# ----------------------------------------------------

model = Sequential()
model.add(LSTM(50, activation='sigmoid', return_sequences=True, input_shape=(look_back,1)))
model.add(LSTM(50, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

num_epochs = 10
history = model.fit(train_generator, epochs=num_epochs, verbose=1, batch_size = 32)

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# ----------------------------------------------------
# Prediction
prediction = model.predict(test_generator)

close_train = date_train.reshape((-1))
close_test = temp_test.reshape((-1))
prediction = prediction.reshape((-1))

# plt.plot(prediction)
# plt.title("Prediction")
print(len(prediction))
plt.plot(date_train, close_train, 'blue', label="Data")
plt.plot(date_test[:len(prediction)], prediction, 'green', label="Prediction")
plt.plot(date_test, close_test, "orange", label="Real Value")

# If you don't like the break in the graph, change 90 to 89 in the above line
plt.gcf().autofmt_xdate()
plt.show()

# ----------------------------------------------------
# Forecasting

def predict(num_prediction, model):
    prediction_list = temp_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = weather_pb_df['date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 365
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

print(forecast_dates)
print(forecast)

# plt.plot(date_train, close_train, 'blue', label="Data")
# plt.plot(date_test[:len(prediction)], prediction, 'blue', label="Prediction")
plt.plot(date_test[3000:], close_test[3000:], "blue", label="Real Value")
plt.plot(forecast_dates, forecast, "orange", label="Forecast")
# If you don't like the break in the graph, change 90 to 89 in the above line
plt.gcf().autofmt_xdate()
plt.show()
