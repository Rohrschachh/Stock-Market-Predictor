import { LineChartSeries } from "../types/Chart";
import { ApexOptions } from "apexcharts";
import ReactApexChart from "react-apexcharts";

const options: ApexOptions = {
  chart: {
    id: "LineAdjacentClose",
    type: "line",
    height: 400,
    zoom: {
      enabled: false,
    },
  },
  title: {
    text: `Ajcacent Close: TCS.NS`,
    align: "left",
    style: {
      fontSize: "16px",
      fontWeight: 800,
    },
  },
  stroke: {
    curve: "straight",
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
  },
  theme: {
    mode: "light",
  },
};

export interface ILineChartProps {
  series: LineChartSeries;
}

export function LineChart(props: ILineChartProps) {
  return (
    <div className="mt-6 w-full rounded-xl bg-white p-1 pt-3 shadow-md dark:bg-slate-950 sm:p-4 sm:pt-6">
      {props.series && (
        <div>
          <ReactApexChart
            options={options}
            type="line"
            width="100%"
            series={props.series}
          />
        </div>
      )}
    </div>
  );
}
