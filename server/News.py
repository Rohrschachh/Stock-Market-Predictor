from flask import jsonify
import pandas as pd
import os
from stocknews import StockNews
from StatusCode import ServerStatusCodes
from pathlib import Path

class News:
    MIN_PAGE_BUNDLESIZE = 10
    @classmethod
    def IsPageValid(cls, ticker, page, pageSize):
        newsDF = pd.read_csv(f"news/{ticker}.csv")
        if int(pageSize) < cls.MIN_PAGE_BUNDLESIZE:
            response = {
                "status": ServerStatusCodes.BADREQUEST.value,
                "messgae": f"Page size should to greater than {cls.MIN_PAGE_BUNDLESIZE}"
            }
            return jsonify(response)
        if len(newsDF) - int(page)*int(pageSize) <= 0:
            response = {
                "status": ServerStatusCodes.NORESOURCE.value,
                "messgae": "Please reduce the page number you have exhausted all news"
            }
            return jsonify(response)

    @classmethod
    def GetStockNews(cls, ticker, page, pagesize):
        Path("news").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(f"news\{ticker}.csv"):
            saved = StockNews(ticker, save_news=False)
            newsDF = saved.read_rss()
            newsDF.to_csv(f"news/{ticker}.csv")
        tempDF = pd.read_csv(f"news/{ticker}.csv")
        totalPage = int(page)*int(pagesize)
        news = tempDF.iloc[totalPage: totalPage + int(pagesize)]
        response = cls.IsPageValid(ticker, page, pagesize)
        if response is not None:
            return response
        finalNews = []
        for _, row in news.iterrows():
            data = {
                "title": row["title"],
                "summary": row["summary"],
                "date": row["published"],
                "sentimentSummary": row["sentiment_summary"],
                "sentimentTitle": row["sentiment_title"]
            }
            finalNews.append(data)
        # Send filterred news
        response = {
                "status": ServerStatusCodes.SUCCESS.value,
                "news": finalNews
            }
        return jsonify(response)
            
            