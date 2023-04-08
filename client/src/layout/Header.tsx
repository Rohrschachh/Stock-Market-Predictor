import { TickerSearch } from "../components/TickerSearch";
import Navbar from "./Navbar";

export default function Header() {
  return (
    <header className="contents lg:inset-0 xl:pointer-events-none xl:fixed xl:z-40 xl:flex">
      <div className="contents lg:pointer-events-auto lg:block xl:w-72 xl:overflow-y-auto xl:border-r xl:border-zinc-900/10 xl:px-6 xl:pb-8 xl:pt-4 xl:dark:border-white/10">
        <div className="hidden items-center gap-2 xl:flex">
          <img src="/icon.svg" alt="Logo" className="h-8 w-8" />
          <span className="text-lg font-bold">Stock Predictor</span>
        </div>
        <Navbar />
        <div className="xl:mt-6">
          <TickerSearch />
        </div>
      </div>
    </header>
  );
}
