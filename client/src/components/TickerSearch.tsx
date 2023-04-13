import { useCallback, useContext, useEffect, useRef, useState } from "react";
import AppContext from "../context/AppContext";
import { APITickerValidityResponse } from "../types/StockPricePredictionAPI";
import useDebounce from "../hooks/useDebounce";

export interface ITickerSearchProps {}

export function TickerSearch(props: ITickerSearchProps) {
  const { ticker, handleSetTicker, handleShowSnackbar } =
    useContext(AppContext);
  const [query, setQuery] = useState<string>(ticker);
  const debouncedValue = useDebounce(query, 1000);

  const checkTickerValidity = async (tkr: string) => {
    const url = new URL(
      "/api/checkticker",
      `${import.meta.env.VITE_SERVER_URL}`
    );
    url.searchParams.set("name", tkr);

    const res: APITickerValidityResponse = await fetch(url).then((res) =>
      res.json()
    );

    if (res.status === 200) {
      handleSetTicker(tkr);
    } else {
      handleShowSnackbar();
    }
  };

  useEffect(() => {
    checkTickerValidity(debouncedValue);
  }, [debouncedValue]);

  useEffect(() => {
    setQuery(ticker);
  }, [ticker]);

  return (
    <input
      type="text"
      name="first-name"
      id="first-name"
      autoComplete="given-name"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      className="block w-full max-w-xs rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-cyan-400 sm:text-sm sm:leading-6"
      placeholder="Ticker name..."
    />
  );
}
