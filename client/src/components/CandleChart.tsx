import ReactApexChart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { CandleChartSeries } from "../types/Chart";

const options: ApexOptions = {
  chart: {
    id: "CandleChart",
    type: "candlestick",
    height: 400,
  },
  title: {
    text: "Candlestick Chart",
    align: "left",
    style: {
      fontSize: "16px",
    },
  },
  xaxis: {
    type: "datetime",
  },
  yaxis: {
    title: {
      text: "Price Rs",
      style: {
        fontSize: "14px",
      },
    },
    tooltip: {
      enabled: true,
    },
  },
  theme: {
    mode: "light",
  },
};

export interface ICandleChartProps {
  series: CandleChartSeries;
}

export function CandleChart(props: ICandleChartProps) {
  return (
    <div className="mt-6 w-full rounded-xl bg-white p-1 pt-3 shadow-md dark:bg-slate-950 sm:p-4 sm:pt-6">
      {props.series && (
        <ReactApexChart
          options={options}
          series={props.series}
          type="candlestick"
          width="100%"
        />
      )}
    </div>
  );
}
