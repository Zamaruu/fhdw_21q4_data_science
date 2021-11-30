"""
@author: Maximilian Ditz
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from math import sqrt
from weather_api import convertDateTimeListToString, getApiArguments, saveDictToJSON
import time

print_values = True

# __________________________Forecast-Methoden__________________________
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

start_time = time.time()

# __________________________Importieren und Formatieren der LSTM-Daten__________________________
args = getApiArguments()
print(args)
num_epochs = int(args["epochs"])
num_prediction = int(args["days"])

# num_epochs = 30
# num_prediction = 180

path = os.path.dirname(__file__)
filenames = glob.glob(path + "/data" + "/export*.csv")

dfs = []
for filename in filenames:
    print(filename)
    dfs.append(pd.read_csv(filename, usecols=["date", "tavg"]))

weather_pb_df = pd.concat(dfs, ignore_index=True)
weather_pb_df.dropna()
weather_pb_df['date'] = pd.to_datetime(weather_pb_df['date'])
weather_pb_df = weather_pb_df.sort_values(by=['date'], ascending=True)
print(weather_pb_df.info())
print(weather_pb_df.head())
print(weather_pb_df.tail())

plt.plot(weather_pb_df['date'], weather_pb_df['tavg'])
plt.show()


# __________________________Data Preprocessing__________________________

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

look_back = 30
batch_size = 100

train_generator = TimeseriesGenerator(temp_train, temp_train, length=look_back, batch_size=batch_size)     
test_generator = TimeseriesGenerator(temp_test, temp_test, length=look_back, batch_size=30)

# __________________________Prediction__________________________

model = Sequential()
model.add(LSTM(60, activation='sigmoid', return_sequences=True, input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(train_generator, epochs=num_epochs, verbose=1, batch_size = batch_size)

# __________________________Prediction__________________________
prediction = model.predict(test_generator)

temp_train = temp_train.reshape((-1))
temp_test = temp_test.reshape((-1))
prediction = prediction.reshape((-1))

rmse = sqrt(mean_squared_error(temp_test[:len(prediction)], prediction[:len(temp_test)]))
print('RMSE: %.3f' % rmse)

print(len(prediction))
if(print_values):
    plt.plot(date_train, temp_train, 'blue', label="Data")
    plt.plot(date_test, temp_test, "green", label="Real Value")
    plt.plot(date_test[:len(prediction)], prediction[:len(date_test)], 'orange', label="Prediction")
    plt.gcf().autofmt_xdate()
    plt.show()

train_test_dict = {
    "train_date": convertDateTimeListToString(date_train).tolist(),
    "train_tavg": temp_train.tolist(),
    "prediction_date": convertDateTimeListToString(date_test).tolist(),
    "prediction_tavg": prediction[:len(date_test)].tolist(),
    "valid_date": convertDateTimeListToString(date_test).tolist(),
    "valid_tavg": temp_test.tolist()
}

# __________________________Forecasting__________________________
forecast = predict(num_prediction, model)
forecast = forecast.reshape((-1))
forecast_dates = predict_dates(num_prediction)


if(print_values):
    plt.plot(date_test[3000:], temp_test[3000:], "blue", label="Real Value")
    plt.plot(forecast_dates, forecast, "orange", label="Forecast")

    plt.gcf().autofmt_xdate()
    plt.show()

runtime = time.time() - start_time

print("--- %s seconds ---" % (runtime))

# __________________________Rückgabe-Dictionary erstellen__________________________

forecast_dates = [item.to_pydatetime() for item in forecast_dates]
forecast_dates = convertDateTimeListToString(forecast_dates)

date_test = date_test[3000:].values
date_test = convertDateTimeListToString(date_test)

date_train = convertDateTimeListToString(date_train)
validation_date = convertDateTimeListToString(date_test)

api_dict = {
  "forecast_tavg": forecast.tolist(),
  "forecast_dates": forecast_dates.tolist(),
  "past_date": date_test.tolist(),
  "past_tavg": temp_test[3000:].tolist(),
}

analyse_dict = {
  "epochs": num_epochs,
  "days":num_prediction,
  "rmse": rmse,
  "loss_history": history.history['loss'],
  "mae_history": history.history['mae'],
  "runtime": runtime
}
api_dict.update(analyse_dict)
api_dict.update(train_test_dict)

# print(api_dict)

# __________________________Analysedetails zurückgeben__________________________
saveDictToJSON("output.json", api_dict)