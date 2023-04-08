import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from "@heroicons/react/24/outline";
import { useContext, useEffect, useState } from "react";
import AppContext from "../context/AppContext";

export interface IDetailsCardProps {
  title: string;
  price: number;
  priceChange?: number;
}

export function DetailsCard(props: IDetailsCardProps) {
  const { ticker } = useContext(AppContext);
  const [isNSE, setISNSE] = useState<boolean>(true);
  const checkNSE = (tr: string) => {
    const names = tr.split(".");
    if (names[1]) {
      if (names[1].toLocaleLowerCase() === "ns") {
        setISNSE(true);
      } else setISNSE(false);
    } else setISNSE(false);
  };

  useEffect(() => {
    checkNSE(ticker);
  }, [ticker]);

  return (
    <div className="w-full rounded-lg bg-white p-4 shadow-md">
      <span className="text-sm font-semibold text-gray-500">{props.title}</span>
      <div className="flex items-center gap-2">
        <p className="text-lg font-bold">
          {isNSE ? "₹" : "$"} {props.price}
        </p>
        {props.priceChange !== undefined ? (
          props.priceChange >= 0 ? (
            <>
              <ArrowTrendingUpIcon className="h-6 w-6 stroke-emerald-500" />
              <p className="font-bold text-emerald-500">{props.priceChange}</p>
            </>
          ) : (
            <>
              <ArrowTrendingDownIcon className="h-6 w-6 stroke-red-500" />
              <p className="font-bold text-red-500">{props.priceChange}</p>
            </>
          )
        ) : null}
      </div>
    </div>
  );
}
