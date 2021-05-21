import tensorflow as tf
import numpy as np
from nsepy import get_history
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
import os
import math
from numpy import array
import pandas as pd
from random import random


class RunModel:

    def __loadModel(self):
        path = 'Bitcoin/bitcoin.json'
        weights = 'Bitcoin/weights.h5'
        # print(weights)
        # print(path)
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights)
        print("Loaded model from disk")
        return model

    def __getEndDate(self, today):
        end = str(today).split("-")
        end[-1] = str(int(end[-1])-1)

        end = list(map(int, end))
        return end

    def getNextQDays(self, q):
        lst_output = []
        n_steps = 348
        i = 0

        bitstamp = pd.read_csv(
            r"C:\Users\afzal\Desktop\Bitcoin-Predictor\Bitcoin-Predictor\Bitcoin\trained_models\bitt.csv")

        price_series = bitstamp.reset_index().Weighted_Price.values

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))
        price_series_scaled = scaler.fit_transform(price_series.reshape(-1, 1))

        le = len(price_series_scaled)

        x_input = price_series_scaled.reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        q = int(q)
        while(i < q):

            if(len(temp_input) > 100):
                # print(temp_input
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input), flush=True)
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)

                self.model = self.__loadModel()
                yhat = self.model.predict(x_input, verbose=0)
                print("{} day output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat)
                i = i+1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat)
                i = i+1

        df_invscaled = scaler.inverse_transform(lst_output).tolist()
        
        return df_invscaled
