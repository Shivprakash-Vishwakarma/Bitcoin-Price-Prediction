#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import plotly.express as px
from itertools import product
import warnings
import statsmodels.api as sm
plt.style.use('seaborn-darkgrid')

#matplotlib inline


# <a id="subsection-one"></a>
# # A first look at Bitcoin Prices
# 
# Let’s check what the first 5 lines of our time-series data look like:

# In[3]:


# Reading the csv file
bitstamp = pd.read_csv(r"C:\Users\afzal\ML\bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")
bitstamp.head()


# Let us now look at the datatypes of the various components.

# In[4]:


bitstamp.info()


# It appears that the Timestamp column is being treated as a integer rather than as dates. To fix this, we’ll use the fromtimestamp() function which converts the arguments to dates.

# In[5]:


# Converting the Timestamp column from string to datetime
bitstamp['Timestamp'] = [datetime.fromtimestamp(x) for x in bitstamp['Timestamp']]


# In[6]:



bitstamp.head()


# In[7]:


print('Dataset Shape: ',  bitstamp.shape)


# * The **Open and Close** columns indicate the opening and closing price on a particular day.
# * The **High and Low** columns provide the highest and the lowest price on a particular day, respectively.
# * The **Volume** column tells us the total volume of traded on a particular day.
# * The **Weighted price** is a trading benchmark used by traders that gives the weighted price a security has traded at throughout the day, based on both volume and price.
# 

# # Visualising the Time Series data

# In[8]:


bitstamp.set_index("Timestamp").Weighted_Price.plot(figsize=(14,7), title="Bitcoin Weighted Price")


# <a id="section-two"></a>
# # Handling Missing Values in Time-series Data
# 

# In[9]:


#calculating missing values in the dataset

missing_values = bitstamp.isnull().sum()
missing_per = (missing_values/bitstamp.shape[0])*100
missing_table = pd.concat([missing_values,missing_per], axis=1, ignore_index=True) 
missing_table.rename(columns={0:'Total Missing Values',1:'Missing %'}, inplace=True)
missing_table


# **Imputation using Linear Interpolation method**
# 
# Time series data has a lot of variations against time. Hence, imputing using backfill and forward fill isn't the best possible solution to address the missing value problem. A more apt alternative would be to use interpolation methods, where the values are filled with incrementing or decrementing values.
# 
# Linear interpolation is an imputation technique that assumes a linear relationship between data points and utilises non-missing values from adjacent data points to compute a value for a missing data point.
# 
# In our dataset, we will be performing Linear interpolation on the missing value columns.

# In[12]:


def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()

    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()

    print(df.head())
    print(df.isnull().sum())


# In[15]:


fill_missing(bitstamp)


# No Null values in the final output. Now we will move to **Exploratory Data Analysis**.
# 

# In[16]:


#created a copy 
bitstamp_non_indexed = bitstamp.copy()


# In[17]:


bitstamp = bitstamp.set_index('Timestamp')
bitstamp.head()


# <a id="section-three"></a>
# 
# # Exploratory Data Analysis
# 
# **Visualizing the weighted price using markers**

# In[18]:


ax = bitstamp['Weighted_Price'].plot(title='Bitcoin Prices', grid=True, figsize=(14,7))
ax.set_xlabel('Year')
ax.set_ylabel('Weighted Price')

ax.axvspan('2018-12-01','2019-01-31',color='red', alpha=0.3)
ax.axhspan(17500,20000, color='green',alpha=0.3)


# In[19]:


#Zooming in

ax = bitstamp.loc['2017-10':'2019-03','Weighted_Price'].plot(marker='o', linestyle='-',figsize=(15,6), title="Oct-17 to March-19 Trend", grid=True)
ax.set_xlabel('Month')
ax.set_ylabel('Weighted_Price')


# There has been a increase in Bitcoin's weighted price except a slump in late 2018 and early 2019. Also, we can  observe a spike in weighted price in December 2017. We shall use Pandas to investigate it further in the coming sections.

# # Visualising using KDEs
# 
# Summarizing the data with Density plots to see where the mass of the data is located.

# In[20]:


sns.kdeplot(bitstamp['Weighted_Price'], shade=True)


# So there is a downward trend in stock prices from Dec-17 onwards till March 2019. We will investigate it further by investigation and with some findings during that period.

# We can see that there is a positive correlation for minute, hour and daily lag plots. We observe absolutely no correlation for month lag plots.
# 
# It makes sense to re-sample our data atmost at the Daily level, thereby preserving the autocorrelation as well. 

# <a id="subsection-three"></a>
# 
# # Time resampling
# 
# Examining stock price data for every single day isn’t of much use to financial institutions, who are more interested in spotting market trends. To make it easier, we use a process called time resampling to aggregate data into a defined time period, such as by month or by quarter. Institutions can then see an overview of stock prices and make decisions according to these trends.
# 

# In[21]:


hourly_data = bitstamp.resample('1H').mean()
hourly_data = hourly_data.reset_index()

hourly_data.head()


# **To summarize what happened above:**
# 
# * data.resample() is used to resample the stock data.
# * The ‘1H’ stands for hourly frequency, and denotes the offset values by which we want to resample the data.
# * mean() indicates that we want the average stock price during this period.
# 

# In[22]:


bitstamp_daily = bitstamp.resample("24H").mean() #daily resampling


# # Plotting using Plotly
# 
# Plotly allows us to make interactve charts which are pretty useful in financial analysis. 
# 
# * The **range-sliders** can be used to zoom-in and zoom-out.
# * The **range-selectors** can be used to select the range.

# In[23]:


import plotly.express as px

bitstamp_daily.reset_index(inplace=True)
fig = px.line(bitstamp_daily, x='Timestamp', y='Weighted_Price', title='Weighted Price with Range Slider and Selectors')
fig.update_layout(hovermode="x")

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
            
        ])
    )
)
fig.show()


# # Simple Candlestick Graph

# In[24]:


ploty = bitstamp_daily.set_index("Timestamp")["2017-12"]


# In[25]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Candlestick(x= ploty.index,
                    open=ploty['Open'],
                    high=ploty['High'],
                    low=ploty['Low'],
                    close=ploty['Close']))
fig.show()


# <a id="section-four"></a>
# # Time Series Decomposition & Statistical Tests
# 
# We can decompose a time series into trend, seasonal amd remainder components, as mentioned in the earlier section. The series can be decomposed as an additive or multiplicative combination of the base level, trend, seasonal index and the residual.
# The seasonal_decompose in statsmodels is used to implements the decomposition.

# In[26]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Let's ensure there are no missing values before you perform statistical tests.

# In[27]:


fill_missing(bitstamp_daily)


# In[28]:


plt.figure(figsize=(15,12))
series = bitstamp_daily.Weighted_Price
result = seasonal_decompose(series, model='additive',period=1)
result.plot()


# Post time series decomposition we don't observe any seasonality. Also, there is no constant mean, variance and covariance, hence the series is **Non Stationary.** 
# We will perform statistical tests like KPSS and ADF to confirm our understanding.
# 
# But first, let's plot ACF and PACF graphs.

# In[29]:


acf = plot_acf(series, lags=50, alpha=0.05)
plt.title("ACF for Weighted Price", size=20)
plt.show()


# The above graph shows that effect barely detoriate over time, so past values affect the present ones. The more lags we include, the better our model will fit the dataset, now the risk is coefficients might predict the dataset too well, cause an overfitting.
# In our model, we always try to include only those lags which have a direct effect on our present value. Hence, let's try PACF.

# In[30]:


plot_pacf(series, lags=50, alpha=0.05, method='ols')
plt.title("PACF for Weighted Price", size=20)
plt.show()


# Coefficients values for lag>5 are statistically not significant and their impact on the model is minimal, except a few spikes at 8,11,22 and beyond.

# <a id="subsection-four"></a>
# # KPSS Test
# 
# The KPSS test, short for, Kwiatkowski-Phillips-Schmidt-Shin (KPSS), is a type of Unit root test that tests for the stationarity of a given series around a deterministic trend.
# 
# Here, the null hypothesis is that the series is **stationary**.
# 
# That is, if p-value is < signif level (say 0.05), then the series is non-stationary and vice versa.

# In[31]:


stats, p, lags, critical_values = kpss(series, 'ct')


# In[32]:


print(f'Test Statistics : {stats}')
print(f'p-value : {p}')
print(f'Critical Values : {critical_values}')

if p < 0.05:
    print('Series is not Stationary')
else:
    print('Series is Stationary')


# # Interpreting KPSS test results
# 
# The output of the KPSS test contains 4 things:
# 
# * The KPSS statistic
# * p-value
# * Number of lags used by the test
# * Critical values
# 
# The **p-value** reported by the test is the probability score based on which you can decide whether to reject the null hypothesis or not. If the p-value is less than a predefined alpha level (typically 0.05), we reject the null hypothesis.
# 
# The **KPSS statistic** is the actual test statistic that is computed while performing the test.
# 
# The number of **lags** reported is the number of lags of the series that was actually used by the model equation of the kpss test.
# 
# In order to reject the null hypothesis, the test statistic should be greater than the provided critical values. If it is in fact higher than the target critical value, then that should automatically reflect in a low p-value.
# That is, if the p-value is less than 0.05, the kpss statistic will be greater than the 5% critical value.

# <a id="subsection-five"></a>
# # ADF Test
# 
# The only difference here is the Null hypothesis which is just opposite of KPSS.
# 
# The null hypothesis of the test is the presence of **unit root**, that is, the series is **non-stationary**.

# In[33]:


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    
    print (dfoutput)
    
    if p > 0.05:
        print('Series is not Stationary')
    else:
        print('Series is Stationary')


# In[34]:


adf_test(series)


# # Conclusion
# 
# 
# KPSS says series is not stationary and ADF says series is stationary. It means series is **difference stationary**, we will use **differencing** to make series stationary.

# <a id="section-five"></a>
# # Feature Extraction

# # Rolling windows
# 
# Time series data can be noisy due to high fluctuations in the market. As a result, it becomes difficult to gauge a trend or pattern in the data. 
# 
# As we’re looking at daily data, there’s quite a bit of noise present. we should average this out by a week, which is where a rolling mean comes in. 
# 
# A rolling mean, or moving average, is a transformation method which helps average out noise from data. It works by simply splitting and aggregating the data into windows according to function, such as mean(), median(), count(), etc. For this example, we’ll use a rolling mean for 3, 7 and 30 days.

# In[35]:


df = bitstamp_daily.set_index("Timestamp")


# In[36]:


df.reset_index(drop=False, inplace=True)

lag_features = ["Open", "High", "Low", "Close","Volume_(BTC)"]
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
df.head()


# **Benefits :**
# 
# So, what are the key benefits of calculating a moving average or using this rolling mean method? Our data becomes a lot less noisy and more reflective of the trend than the data itself.

# Let's extract time and date features from the Date column.

# In[37]:


df["month"] = df.Timestamp.dt.month
df["week"] = df.Timestamp.dt.isocalendar().week
df["day"] = df.Timestamp.dt.day
df["day_of_week"] = df.Timestamp.dt.dayofweek
df.head()


# <a id="section-six"></a>
# # Model Building

# # Important Note on Cross Validation
# 
# To measure the performance of our forecasting model, We typically want to split the time series into a training period and a validation period. This is called fixed partitioning. 
# 
# * If the time series has some seasonality, you generally want to ensure that each period contains a whole number of seasons. For example, one year, or two years, or three years, if the time series has a yearly seasonality. 
# You generally don't want one year and a half, or else some months will be represented more than others. 
#  
# * We'll train our model on the training period, we'll evaluate it on the validation period. Here's where you can experiment to find the right architecture for training. And work on it and your hyper parameters, until you get the desired performance, measured using the validation set. Often, once you've done that, you can retrain using both the training and validation data.And then test on the test(or forecast) period to see if your model will perform just as well.
#  
# * And if it does, then you could take the unusual step of retraining again, using also the test data. But why would you do that? Well, it's because the test data is the closest data you have to the current point in time. And as such it's often the strongest signal in determining future values. If your model is not trained using that data, too, then it may not be optimal.
# 
# Here, we we will opt for a ***hold-out based validation***. 
# 
# Hold-out is used very frequently with time-series data. In this case, we will select all the data for 2020 as a hold-out and train our model on all the data from 2012 to 2019. 
# 

# In[38]:


df_train = df[df.Timestamp < "2020"]
df_valid = df[df.Timestamp >= "2020"]

print('train shape :', df_train.shape)
print('validation shape :', df_valid.shape)


# <a id="subsection-nine"></a>
# # LSTM
# ![LSTM.JPG](attachment:LSTM.JPG)

# * **Long Short Term Memory networks** – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. 
# * LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn like RNNs!
# * All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.
# * Also, they don't suffer from problems like **vanishing/exploding gradient descent**. 

# In[39]:


price_series = bitstamp_daily.reset_index().Weighted_Price.values
price_series


# In[40]:


price_series.shape


# In[41]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
price_series_scaled = scaler.fit_transform(price_series.reshape(-1,1))


# In[42]:


price_series_scaled, price_series_scaled.shape


# In[43]:


train_data, test_data = price_series_scaled[0:2923], price_series_scaled[2923:]


# In[44]:


test_data


# In[45]:


# train_data = train_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)


# In[46]:


train_data.shape, test_data.shape


# In[47]:


def windowed_dataset(series, time_step):
    dataX, dataY = [], []
    for i in range(len(series)- time_step-1):
        a = series[i : (i+time_step), 0]
        dataX.append(a)
        dataY.append(series[i+ time_step, 0])
        
    return np.array(dataX), np.array(dataY)


# In[48]:


X_train, y_train = windowed_dataset(train_data, time_step=100)
X_test, y_test = windowed_dataset(test_data, time_step=100)


# In[49]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[50]:


#reshape inputs to be [samples, timesteps, features] which is requred for LSTM

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


print(X_train.shape) 
print(X_test.shape)


# In[51]:


print(y_train.shape) 
print(y_test.shape)


# In[52]:


#Create Stacked LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[53]:


# Initialising the LSTM
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[54]:


regressor.summary()


# In[55]:


# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, validation_split=0.1, epochs = 50, batch_size = 32, verbose=1, shuffle=False)


# In[56]:


plt.figure(figsize=(16,7))
plt.plot(history.history["loss"], label= "train loss")
plt.plot(history.history["val_loss"], label= "validation loss")
plt.legend()


# In[57]:


#Lets do the prediction and performance checking

train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)


# In[58]:


#transformation to original form

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)


# In[59]:


plt.figure(figsize=(16,7))
plt.plot(y_train_inv.flatten(), marker='.', label="Actual")
plt.plot(train_predict_inv.flatten(), 'r', marker='.', label="Predicted")
plt.legend()


# In[60]:


plt.figure(figsize=(16,7))
plt.plot(y_test_inv.flatten(), marker='.', label="Actual")
plt.plot(test_predict_inv.flatten(), 'r', marker='.', label="Predicted")
plt.legend()


# In[61]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

train_RMSE = np.sqrt(mean_squared_error(y_train, train_predict))
test_RMSE = np.sqrt(mean_squared_error(y_test, test_predict))
train_MAE = np.sqrt(mean_absolute_error(y_train, train_predict))
test_MAE = np.sqrt(mean_absolute_error(y_test, test_predict))


print(f"Train RMSE: {train_RMSE}")
print(f"Train MAE: {train_MAE}")

print(f"Test RMSE: {test_RMSE}")
print(f"Test MAE: {test_MAE}")


# 
# *Prediction
# 
# We observed remarkable results using LSTMs. They really work a lot better for most sequence tasks! 
# 
# Let's predict weighted price for next 30 days. 

# In[62]:


test_data.shape


# In[63]:


lookback = len(test_data) - 100
x_input=test_data[lookback:].reshape(1,-1)
x_input.shape


# In[64]:


x_input


# In[65]:


lookback, len(test_data)


# In[66]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[67]:


len(temp_input)


# In[68]:


# demonstrate prediction for next 100 days
from numpy import array

lst_output=[]
n_steps=100
i=0
q=10
while(i<q):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = regressor.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = regressor.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[69]:


len(price_series_scaled)


# In[70]:


df_=price_series_scaled.tolist()
df_.extend(lst_output)
plt.plot(df_)


# In[71]:


plt.figure(figsize=(14,7))
df_invscaled=scaler.inverse_transform(df_).tolist()
plt.plot(df_invscaled)

