import { AreaChart } from "./AreaChart";
import { CandleChart } from "./CandleChart";
import { LineChart } from "./LineChart";
import {
  APIChartResponse,
  APILSTMPredictionResponse,
  APILinRegPredictionResponse,
} from "../types/StockPricePredictionAPI";

export interface IChartsProps {
  chartState: APIChartResponse | null;
  lstmState: APILSTMPredictionResponse | null;
  linregState: APILinRegPredictionResponse | null;
}

export default function Charts({
  chartState,
  lstmState,
  linregState,
}: IChartsProps) {
  return (
    <div>
      <CandleChart
        series={[
          {
            data: chartState ? chartState.candleChart : [],
          },
        ]}
      />
      <AreaChart
        id="lstmpred"
        text={`LSTM Prediction: ${lstmState ? lstmState.predictionLstm : ""}`}
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
  );
}
