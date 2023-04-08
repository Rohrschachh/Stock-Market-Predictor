import os
from flask import jsonify
from StatusCode import ServerStatusCodes
import pandas as pd
import yfinance as yf
from Utils import Utils

class Chart:
    dataFolderPath = os.getcwd()
    
    @classmethod
    def SendChartData(cls, ticker, today):
        if not os.path.exists(f"data\{ticker}.csv"):
            df = Utils.downloadData(ticker, "2021-01-01", today)
            df.to_csv(f"data/{ticker}.csv")
            response = {
                "status": ServerStatusCodes.NOTIMPLEMENTED.value,
                "message": "Rerun same request resource were being loaded"
            }
            return jsonify(response)
            # return Chart.GetChartData(df)
        else:
            df = pd.read_csv(f"data/{ticker}.csv")
            return Chart.GetChartData(df)
    
    @classmethod
    def GetChartData(cls, df):
        data = df.iloc[-56:]
        candleChartData = []
        for i in range(56):
            dict_list = {
                'x': data.iloc[i, 0],
                'y': data.iloc[i, 1:].astype("float").values.round(2).tolist()
            }
            candleChartData.append(dict_list)
            
        adjacentCloseChart = []
        for i in range(56):
            tempList = {
                "x":data.iloc[i, 0],
                "y":data.iloc[i]["Adj Close"].round(2),
            }
            adjacentCloseChart.append(tempList)
            
        priceStatus = (df.iloc[-1]["Adj Close"].astype("float") - df.iloc[-2]["Adj Close"].astype("float")).round(2)
        currentPrice = df.iloc[-1]['Adj Close'].round(2).astype("float")
        dayOpen = df.iloc[-1]['Open'].round(2).astype("float")
        dayHigh = df.iloc[-1]['High'].round(2).astype("float")
        dayLow = df.iloc[-1]['Low'].round(2).astype("float")

        response = {
            "status": ServerStatusCodes.SUCCESS.value,
            "candleChart": candleChartData,
            "adjacentCloseChart": adjacentCloseChart,
            "priceStatus": priceStatus,
            "currentPrice": currentPrice,
            "dayOpen": dayOpen,
            "dayHigh": dayHigh,
            "dayLow": dayLow,
        }
        return jsonify(response)
            
        