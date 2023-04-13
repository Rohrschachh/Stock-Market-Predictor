import ReactApexChart from "react-apexcharts";

export interface IAreaChartProps {
  id: string;
  text: string;
  categories: string[];
  series: [
    {
      name: "Original Price";
      data: number[];
    },
    {
      name: "Predicted Price";
      data: number[];
    }
  ];
}

export function AreaChart(props: IAreaChartProps) {
  return (
    <div className="mt-6 w-full rounded-xl bg-white p-1 pt-3 shadow-md dark:bg-slate-950 sm:p-4 sm:pt-6">
      <ReactApexChart
        options={{
          chart: {
            id: props.id,
            type: "area",
            height: 400,
            zoom: {
              enabled: false,
            },
          },
          title: {
            text: props.text,
            align: "left",
            style: {
              fontSize: "16px",
            },
          },
          xaxis: {
            type: "datetime",
            categories: props.categories,
          },
          dataLabels: {
            enabled: false,
          },
          stroke: {
            curve: "smooth",
            width: [2, 2],
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
        }}
        series={props.series}
        type="area"
        width="100%"
      />
    </div>
  );
}
