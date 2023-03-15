
# Importing all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import yfinance as yf
import streamlit as st
import datetime

yf.pdr_override()

from keras.models import load_model 

import math
from sklearn.metrics import mean_squared_error

# Input parameters
st.sidebar.header('Input Parameters')

today = datetime.date.today()

def input_para():
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.date_input('Start Date', value = pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))
    return ticker, start_date, end_date

symbol, start, end = input_para()

# Fetch data 
def fetch_data():
    data = yf.download(symbol, start, end)
    df = pd.DataFrame(data)
    output_name = ''+symbol+'.csv'
    df.to_csv("./stocks data/" + output_name)
    return data, df

data, df = fetch_data()

# Adjusted Close Price
st.subheader(f"Adjusted Close Price\n {symbol}")
st.line_chart(data['Adj Close'])


# """************************* LSTM SECTION ********************************"""

def lstm_analysis(df):

    # Splitting data into train and test data
    data_train = df.iloc[0: int(len(df)*.80)]
    data_test = df.iloc[int(len(df)*.80): int(len(df))]

    training_set = df.iloc[:,4:5].values

    #Scaling down the model data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range = (0,1))

    data_train_arr = scaler.fit_transform(training_set)

    # # Loading the Model
    # model = load_model('model.h5')

    # Dividing the data into xtrain and ytrain

    x_train = []
    y_train = []

    for i in range(7, data_train_arr.shape[0]):
        x_train.append(data_train_arr[i-7: i])
        y_train.append(data_train_arr[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_forecast = np.array(x_train[-1,1:])
    x_forecast = np.append(x_forecast, y_train[-1])

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    x_forecast = np.reshape(x_forecast, (1, x_forecast.shape[0],1))

    # Machine Learning Model

    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential

    model = Sequential()

    # 1st layer
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.1))

    # 2nd Layer
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
    model.add(Dropout(0.1))

    # 3rd Layer
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
    model.add(Dropout(0.1))

    # 4th Layer
    model.add(LSTM(units = 50, activation = 'relu'))
    model.add(Dropout(0.1))

    # Dense Layer which is to connect all the layers together
    model.add(Dense(units = 1))  

    # Compiling the model and loss is defined as mse 
    model.compile(optimizer = 'adam', loss = 'mean_squared_error') 

    model.fit(x_train, y_train, epochs = 30)

    # Testing Parameters

    real_stock_price = data_test.iloc[:,4:5].values

    data_total = pd.concat((data_train['Close'], data_test['Close']), axis=0)
    data_input = data_total[ len(data_total) - len(data_test) - 7: ].values
    data_input = data_input.reshape(-1,1)

    data_input = scaler.transform(data_input)

    x_test = []

    for i in range(7, data_input.shape[0]):
        x_test.append(data_input[i-7: i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    y_predict = model.predict(x_test)

    y_predict = scaler.inverse_transform(y_predict)

    # scaler = scaler.scale_

    # scale_factor = 1 / scaler[0]
    # y_predict = y_predict * scale_factor

    # Final Graph
    # st.subheader('Predictions vs Original Price')

    fig2 = plt.figure(figsize = (8, 5), dpi=70)
    plt.plot(real_stock_price, 'red', label = 'Original Price')
    plt.plot(y_predict, 'blue', label = 'Predicted Price')

    # plt.xlabel('Time')
    # plt.ylabel('Price')

    plt.legend()
    # Here we are directly showing the results to the client and not images for better UE
    st.pyplot(fig2)

    # Calibrating the Errors
    lstm_err = math.sqrt(mean_squared_error(real_stock_price, y_predict))

    forecast_price = model.predict(x_forecast)
    forecast_price = scaler.inverse_transform(forecast_price)

    lstm_pred = forecast_price[0,0]

    st.write("Tomorrow's ",symbol," Closing Price Prediction by LSTM: ",lstm_pred)
    # st.write("LSTM RMSE: ", lstm_err)

    return lstm_err

lstm_err = lstm_analysis(df)


# """************************* LinReg SECTION ********************************"""

def lin_reg_analysis(df):

    # since we are forecast next 7 days data
    forecast_out = int(7)

    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after n days']]

    x = np.array(df_new.iloc[:-forecast_out,0:-1])

    y = np.array(df_new.iloc[:-forecast_out, -1]) #35 rows discard
    y = np.reshape(y, (-1,1))

    x_forecast = np.array(df_new.iloc[-forecast_out:,0:-1])

    # Splitting data into train and test
    x_train = x[0:int(0.8*len(df))]
    x_test = x[int(0.8*len(df)):,:]

    y_train = y[0:int(0.8*len(df)),:]
    y_test = y[int(0.8*len(df)):,:]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
            
    x_forecast = scaler.transform(x_forecast)

    from sklearn.linear_model import LinearRegression

    # Linear Model Training
    lin_mod = LinearRegression()
    lin_mod.fit(x_train, y_train) # function to pass the training ds

    # Testing
    y_predict = lin_mod.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1.04
    y_predict = y_predict * scale_factor
    y_test = y_test * scale_factor

    fig = plt.figure(figsize = (8, 5), dpi=70)
    plt.plot(y_test, 'red', label = 'Original Price')
    plt.plot(y_predict, 'blue', label = 'Predicted Price')

    plt.legend()
    st.pyplot(fig)

    lin_reg_err = math.sqrt(mean_squared_error(y_test, y_predict))

    # Forecasted data
    forecast_set = lin_mod.predict(x_forecast)

    forecast_set = forecast_set * scale_factor
    
    mean = forecast_set.mean()
    lin_reg_pred = forecast_set[0, 0]

    st.write("Tomorrow's ", symbol," Closing Price Prediction by Linear Regression: ", lin_reg_pred)
    # st.write("Linear Regression RMSE:", lin_reg_err)

    return df, lin_reg_err, mean, forecast_set, lin_reg_pred

df, lin_reg_err, mean, forecast_set, lin_reg_pred = lin_reg_analysis(df)



# st.write("Forecasted Prices for Next 7 days:")
# st.write(forecast_set)
