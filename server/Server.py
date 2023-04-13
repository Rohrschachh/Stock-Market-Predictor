from flask import Flask, request, jsonify
from flask_cors import CORS
from Stocks import Stocks
from Chart import Chart
from News import News
from StatusCode import ServerStatusCodes
import datetime

today = datetime.date.today()

application = Flask(__name__)
cors = CORS(application, origins="*")
CORS(application)

@application.route("/")
def index():
    response = {
        "status": ServerStatusCodes.SUCCESS.value,
        "message": "Hello from Stock Price Prediction API"
    }
    return jsonify(response)

@application.route("/api/checkticker")
def checkTicker():
    stockname = request.args.get("name")
    return Stocks.CheckTickerValidity(stockname)

@application.route("/api/chartdata")
def getCandleChartData():
    stockname = request.args.get("name")
    return Chart.SendChartData(stockname, today)

@application.route("/api/stocknews")
def getStockNews():
    stockname = request.args.get("name")
    page = request.args.get("page")
    pageSize = request.args.get("pagesize")
    return News.GetStockNews(stockname, page, pageSize)

@application.route("/api/predict/lstm")
def predictLSTM():
    stockName = request.args.get("name")
    return Stocks.PredictionLSTM(stockName, today)

@application.route("/api/predict/linreg")
def predictLinReg():
    stockName = request.args.get("name")
    return Stocks.PredictionLinReg(stockName, today)

if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=5000)