import { ApexOptions } from "apexcharts";

export type CandleChartSeries = {
  data: {
    x: string;
    y: number[];
  }[];
}[];

export type LineChartSeries = {
  name: string;
  data: {
    x: string;
    y: number;
  }[];
}[];

export type AreaChartSeries = {
  name: string;
  data: {
    x: string;
    y: number;
  }[];
}[];

export type ApexChartOptions = ApexOptions;
