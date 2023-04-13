import { useContext, useEffect, useState } from "react";
import { CandleChart } from "./components/CandleChart";
import { DetailsCard } from "./components/DetailsCard";
import { LineChart } from "./components/LineChart";
import Header from "./layout/Header";
import {
  APIChartResponse,
  APILSTMPredictionResponse,
  APILinRegPredictionResponse,
} from "./types/StockPricePredictionAPI";
import AppContext from "./context/AppContext";
import { AreaChart } from "./components/AreaChart";
import { NewsComponent } from "./components/News";

export default function App() {
  const { ticker } = useContext(AppContext);
  const [chartState, setChartState] = useState<null | APIChartResponse>(null);
  const [lstmState, setLstmState] = useState<null | APILSTMPredictionResponse>(
    null
  );
  const [linregState, setLinregState] =
    useState<null | APILinRegPredictionResponse>(null);

  const getChartData = async () => {
    const url = new URL("/api/chartdata", `${import.meta.env.VITE_SERVER_URL}`);
    url.searchParams.set("name", ticker);
    const data: APIChartResponse = await fetch(url).then((res) => res.json());
    if (data.status === 501) {
      getChartData();
    }
    setChartState(data);
  };

  const getLSTMPrediction = async () => {
    const url = new URL(
      "/api/predict/lstm",
      `${import.meta.env.VITE_SERVER_URL}`
    );
    url.searchParams.set("name", ticker);
    const data: APILSTMPredictionResponse = await fetch(url).then((res) =>
      res.json()
    );
    setLstmState(data);
  };

  const getLinRegPrediction = async () => {
    const url = new URL(
      "/api/predict/linreg",
      `${import.meta.env.VITE_SERVER_URL}`
    );
    url.searchParams.set("name", ticker);
    const data: APILinRegPredictionResponse = await fetch(url).then((res) =>
      res.json()
    );
    setLinregState(data);
  };

  useEffect(() => {
    getChartData();
    getLSTMPrediction();
    getLinRegPrediction();
  }, [ticker]);

  return (
    <div className="lg:mr-72 xl:ml-72">
      <Header />
      <div className="flex min-h-screen flex-col px-4 sm:px-6 lg:px-8 xl:pt-8">
        <main className="flex py-6 sm:py-10 xl:pt-0">
          <div className="flex w-full flex-col">
            <h1 className="mb-4 text-3xl font-bold text-slate-900">{ticker}</h1>
            <div className="grid w-auto grid-cols-2 gap-2 lg:grid-cols-4">
              <DetailsCard
                title="Current Price"
                price={chartState ? chartState.currentPrice : 0.0}
                priceChange={chartState?.priceStatus}
              />
              <DetailsCard
                title="Day Open"
                price={chartState ? chartState.dayOpen : 0.0}
              />
              <DetailsCard
                title="Day High"
                price={chartState ? chartState.dayHigh : 0.0}
              />
              <DetailsCard
                title="Day Low"
                price={chartState ? chartState.dayLow : 0.0}
              />
            </div>
            <CandleChart
              series={[
                {
                  data: chartState ? chartState.candleChart : [],
                },
              ]}
            />
            <AreaChart
              id="lstmpred"
              text={`LSTM Prediction: ${
                lstmState ? lstmState.predictionLstm : ""
              }`}
              categories={lstmState ? lstmState.categories : []}
              series={[
                {
                  name: "Original Price",
                  data: lstmState ? lstmState.realData : [],
                },
                {
                  name: "Predicted Price",
                  data: lstmState ? lstmState.predicatedData : [],
                },
              ]}
            />
            <AreaChart
              id="linregpred"
              text={`Linear Regression Prediction: ${
                linregState ? linregState.predictionLinReg : ""
              }`}
              categories={linregState ? linregState.categories : []}
              series={[
                {
                  name: "Original Price",
                  data: linregState ? linregState.realData : [],
                },
                {
                  name: "Predicted Price",
                  data: linregState ? linregState.predicatedData : [],
                },
              ]}
            />
            <LineChart
              series={[
                {
                  name: "Adj Close",
                  data: chartState ? chartState.adjacentCloseChart : [],
                },
              ]}
            />
          </div>
        </main>
        <NewsComponent />
      </div>
    </div>
  );
}
