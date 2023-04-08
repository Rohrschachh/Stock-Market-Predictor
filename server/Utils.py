import pandas as pd
import yfinance as yf

class Utils:
    @classmethod
    def downloadData(cls, ticker, start, end):
        data = yf.download(ticker, start, end)
        df = pd.DataFrame(data)
        return df