export type APIChartResponse = {
  status: 200 | 501;
  candleChart: {
    x: string;
    y: number[];
  }[];
  adjacentCloseChart: {
    x: string;
    y: number;
  }[];
  priceStatus: number;
  currentPrice: number;
  dayOpen: number;
  dayHigh: number;
  dayLow: number;
  date: string;
};

export type APITickerValidityResponse = {
  status: 400 | 200;
  message: string;
};

export type APILSTMPredictionResponse = {
  status: 200;
  predictionLstm: number;
  errorPercentage: number;
  categories: string[];
  realData: number[];
  predicatedData: number[];
  date: string;
};

export type APILinRegPredictionResponse = {
  status: 200;
  predictionLinReg: number;
  errorPercentage: number;
  categories: string[];
  realData: number[];
  predicatedData: number[];
  date: string;
};

export type APINewsStocksResponse = {
  status: 200 | 403 | 400;
  news: {
    title: string;
    summary: string;
    date: string;
    sentimentSummary: number;
    sentimentTitle: number;
  }[];
};
