import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import joblib
from flask import jsonify
from StatusCode import ServerStatusCodes
from Utils import Utils
import math
yf.pdr_override()

class Stocks:
    dataFolderPath = os.getcwd()
    
    @classmethod
    def CheckTickerValidity(cls, ticker):
        tickerr = yf.Ticker(str(ticker)).history(period="7d", interval="1d")
        if len(tickerr) > 0:
            reponse = {
                "status": ServerStatusCodes.SUCCESS.value,
                "message": f"{ticker} is a valid ticker"
            }
            return jsonify(reponse)
        else:
            reponse = {
                "status": ServerStatusCodes.BADREQUEST.value,
                "message": f"{ticker} is not a valid ticker or data is not available"
            }
            return jsonify(reponse)
        
    @classmethod
    def PredictionLinReg(cls, ticker, today):
        Path("models").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(f"models\{ticker}-LR.pkl"):
            cls.trainLinearRegression(ticker, today)
        #prediction code
        print("Predicting please wait")
        df = pd.read_csv(f"data/{ticker}.csv")
        model = joblib.load(f"./models/{ticker}-LR.pkl")
        
        data_test = df.iloc[int(len(df)*.80): int(len(df))]
        categories = data_test["Date"].values
        
        forecast_out = int(1)

        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new = df[['Close', 'Close after n days']]

        x = np.array(df_new.iloc[:-forecast_out,0:-1])

        y = np.array(df_new.iloc[:-forecast_out, -1]) #35 rows discard
        y = np.reshape(y, (-1,1))

        x_forecast = np.array(df_new.iloc[-forecast_out:,0:-1])

        # Splitting data into train and test
        x_train = x[0:int(0.8*len(df))]
        x_test = x[int(0.8*len(df)):,:]

        y_test = y[int(0.8*len(df)):,:]
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
                
        x_forecast = scaler.transform(x_forecast)
        
        y_predict = model.predict(x_test)
        
        scale_factor = 1.04
        
        lin_reg_err = math.sqrt(mean_squared_error(y_test, y_predict))
        
        forecast_set = model.predict(x_forecast)

        forecast_set = forecast_set * scale_factor
        
        lin_reg_pred = forecast_set[0, 0]
        
        # data sanitisation
        real_data = []
        tempOrg = y_test.tolist()
        for i in range(len(tempOrg)):
            data = round(tempOrg[i][0], 2)
            real_data.append(data)
            
        pred_data = []
        tempPred = y_predict.tolist()
        for i in range(len(tempPred)):
            data = round(tempPred[i][0], 2)
            pred_data.append(data)
        
        final_categories = categories.tolist()
        print(len(final_categories))
        print(len(pred_data))
        print(len(real_data))
        response = {
            "status": ServerStatusCodes.SUCCESS.value,
            "predictionLinReg": lin_reg_pred.astype("float").round(2),
            "errorPercentage": round(lin_reg_err, 2),
            "realData": real_data,
            "predicatedData": pred_data,
            "categories": final_categories
        }
        return jsonify(response)
    
    @classmethod
    def PredictionLSTM(cls, ticker, today):
        Path("models").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(f"models\{ticker}-LSTM.pkl"):
            cls.trainLstmModel(ticker, today)
        # prediction code
        print("Predicting please wait")
        df = pd.read_csv(f"data/{ticker}.csv")
        model = joblib.load(f"./models/{ticker}-LSTM.pkl")
        
        data_train = df.iloc[0: int(len(df)*.80)]
        data_test = df.iloc[int(len(df)*.80): int(len(df))]
    
        training_set = df.iloc[:,4:5].values
        
        scaler = MinMaxScaler(feature_range = (0,1))
        data_train_arr = scaler.fit_transform(training_set)
        
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
        
        real_stock_price = data_test.iloc[:,4:5].values
        categories = data_test["Date"].values

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

        # Calibrating the Errors
        lstm_err = math.sqrt(mean_squared_error(real_stock_price, y_predict))

        forecast_price = model.predict(x_forecast)
        forecast_price = scaler.inverse_transform(forecast_price)

        lstm_pred = forecast_price[0,0]
        
        real_data = []
        temp_data = real_stock_price.tolist()
        for i in range(len(temp_data)):
            data = round(temp_data[i][0], 2)
            real_data.append(data)
        
        pred_data = []
        temp_pred_data = y_predict.tolist()
        for i in range(len(temp_pred_data)):
            data = round(temp_pred_data[i][0], 2)
            pred_data.append(data)
        
        final_category = categories.tolist()
        response = {
            "status": ServerStatusCodes.SUCCESS.value,
            "predictionLstm": lstm_pred.astype("float").round(2),
            "errorPercentage": round(lstm_err, 2),
            "realData": real_data,
            "predicatedData": pred_data,
            "categories": final_category
        }
        return jsonify(response)

    @classmethod
    def trainLstmModel(cls,ticker, today):
        df = Utils.downloadData(ticker, "2021-01-01", today)
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"data/{ticker}.csv")
        
        training_set = df.iloc[:,4:5].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        data_train_arr = scaler.fit_transform(training_set)
        
        # Dividing the data into x_train and y_train
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
        
        # Machine Learning LSTM model
        
        model = Sequential()
        
        # 1st layer
        model.add(LSTM(units = 50, activation = "relu", return_sequences = True,input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.1))
        
        # 2nd Layer
        model.add(LSTM(units = 50, activation = "relu", return_sequences = True))
        model.add(Dropout(0.1))

        # 3rd Layer
        model.add(LSTM(units = 50, activation = "relu", return_sequences = True))
        model.add(Dropout(0.1))

        # 4th Layer
        model.add(LSTM(units = 50, activation = "relu"))
        model.add(Dropout(0.1))
        
        # Dense Layer which is to connect all the layers together
        model.add(Dense(units = 1))  

        # Compiling the model and loss is defined as mse 
        model.compile(optimizer = 'adam', loss = 'mean_squared_error') 

        model.fit(x_train, y_train, epochs = 60)

        Path("models").mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, f"models\{ticker}-LSTM.pkl")
        
    @classmethod
    def trainLinearRegression(cls, ticker, today):
        df = Utils.downloadData(ticker, "2021-01-01", today)
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"data/{ticker}.csv")
        
        forecast_out = int(7)
        
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new = df[['Close', 'Close after n days']]
        
        x = np.array(df_new.iloc[:-forecast_out,0:-1])

        y = np.array(df_new.iloc[:-forecast_out, -1]) #35 rows discard
        y = np.reshape(y, (-1,1))

        x_train = x[0:int(0.8*len(df))]
        y_train = y[0:int(0.8*len(df)),:]
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        
        lin_mod = LinearRegression()
        lin_mod.fit(x_train, y_train)
        
        Path("models").mkdir(parents=True, exist_ok=True)
        
        joblib.dump(lin_mod, f"models\{ticker}-LR.pkl")
