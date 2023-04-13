import { useEffect, useRef, useState, useContext } from "react";
import { APINewsStocksResponse } from "../types/StockPricePredictionAPI";
import AppContext from "../context/AppContext";

export interface INewsComponentProps {}

export function NewsComponent(props: INewsComponentProps) {
  const { ticker } = useContext(AppContext);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(10);
  const [endOfNews, setEndOfNews] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [news, setNews] = useState<[] | APINewsStocksResponse["news"]>([]);
  const refNewsContainer = useRef<HTMLDivElement>(null);

  const fetchNews = async (tkr: string) => {
    setLoading(true);
    const url = new URL("/api/stocknews", import.meta.env.VITE_SERVER_URL);
    url.searchParams.set("name", tkr);
    url.searchParams.set("page", pageNumber.toString());
    url.searchParams.set("pagesize", pageSize.toString());
    const data: APINewsStocksResponse = await fetch(url).then((res) =>
      res.json()
    );
    if (data.status === 200) {
      if (pageNumber === 1) {
        setNews(data.news);
      } else {
        setNews((prevNews) => [...prevNews, ...data.news]);
      }
      setLoading(false);
    } else {
      setEndOfNews(true);
    }
  };

  const handleNewsLoad = () => {
    const container = refNewsContainer.current!;
    if (
      container.scrollTop + container.clientHeight >= container.scrollHeight &&
      loading
    ) {
      setLoading(true);
      if (!endOfNews) {
        setPageNumber((prev) => prev + 1);
      }
    }
  };

  useEffect(() => {
    const container = refNewsContainer.current!;
    container.addEventListener("scroll", handleNewsLoad, false);
    return () => {
      container.removeEventListener("scroll", handleNewsLoad);
    };
  }, []);

  useEffect(() => {
    fetchNews(ticker);
  }, [pageNumber]);

  useEffect(() => {
    refNewsContainer.current?.scrollTo({
      top: 0,
      behavior: "smooth",
    });
    setPageNumber(1);
    setEndOfNews(false);
    fetchNews(ticker);
  }, [ticker]);

  return (
    <div
      ref={refNewsContainer}
      className="fixed inset-y-0 right-0 top-0 hidden w-72 overflow-y-scroll pt-14 lg:inline-block lg:border-l lg:border-zinc-900/10 xl:pt-0"
    >
      <div className="fixed right-0 top-14 w-72 border-l bg-gray-100 px-3 py-4 text-center text-lg font-semibold text-slate-900 lg:border-zinc-900/10 xl:top-0">
        News for{" "}
        <span className=" text-lg font-bold text-cyan-600">{ticker}</span>
      </div>
      <div className="mt-4 pb-16 pt-10">
        {news &&
          news.map((n, index) => (
            <div
              key={index}
              className="mx-4 mt-3 rounded-md bg-white p-3 shadow-sm"
            >
              <span className="text-sm font-bold">{n.title}</span>
              <p className="truncate text-sm">{n.summary}</p>
              <p className="">{new Date(n.date).toLocaleDateString()}</p>
              <div className="flex justify-end gap-4">
                <span>{n.sentimentSummary}</span>
                <span>{n.sentimentTitle}</span>
              </div>
            </div>
          ))}
        <div className="mt-4 text-center">
          {endOfNews ? "End of feed" : "Loading..."}
        </div>
      </div>
    </div>
  );
}
