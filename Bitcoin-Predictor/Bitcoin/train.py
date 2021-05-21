from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from itertools import product
import warnings
# import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
# plt.style.use('seaborn-darkgrid')

bitstamp = pd.read_csv(
    r"C:\Users\afzal\Desktop\Bitcoin-Predictor\Bitcoin-Predictor\Bitcoin\trained_models\bitstampUSD.csv")
bitstamp['Timestamp'] = [datetime.fromtimestamp(
    x) for x in bitstamp['Timestamp']]


def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()

    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()


fill_missing(bitstamp)


bitstamp_non_indexed = bitstamp.copy()

bitstamp = bitstamp.set_index('Timestamp')

hourly_data = bitstamp.resample('1H').mean()
hourly_data = hourly_data.reset_index()

bitstamp_daily = bitstamp.resample("24H").mean()
bitstamp_daily.reset_index(inplace=True)

fill_missing(bitstamp_daily)

df = bitstamp_daily.set_index("Timestamp")
df.reset_index(drop=False, inplace=True)

lag_features = ["Open", "High", "Low", "Close", "Volume_(BTC)"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

df_std_3d = df_rolled_3d.std().shift(1).reset_index()
df_std_7d = df_rolled_7d.std().shift(1).reset_index()
df_std_30d = df_rolled_30d.std().shift(1).reset_index()

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Timestamp", drop=False, inplace=True)

df["month"] = df.Timestamp.dt.month
df["week"] = df.Timestamp.dt.isocalendar().week
df["day"] = df.Timestamp.dt.day
df["day_of_week"] = df.Timestamp.dt.dayofweek


df_train = df[df.Timestamp < "2020"]
df_valid = df[df.Timestamp >= "2020"]

price_series = bitstamp_daily.reset_index().Weighted_Price.values


scaler = MinMaxScaler(feature_range=(0, 1))
price_series_scaled = scaler.fit_transform(price_series.reshape(-1, 1))

train_data, test_data = price_series_scaled[0:2923], price_series_scaled[2923:]


def windowed_dataset(series, time_step):
    dataX, dataY = [], []
    for i in range(len(series) - time_step-1):
        a = series[i: (i+time_step), 0]
        dataX.append(a)
        dataY.append(series[i + time_step, 0])

    return np.array(dataX), np.array(dataY)


X_train, y_train = windowed_dataset(train_data, time_step=100)
X_test, y_test = windowed_dataset(test_data, time_step=100)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

history = regressor.fit(X_train, y_train, validation_split=0.1,
                        epochs=50, batch_size=32, verbose=1, shuffle=False)

model_json = regressor.to_json()
with open("bitcoin.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("weights.h5")
print("Saved model to disk")
