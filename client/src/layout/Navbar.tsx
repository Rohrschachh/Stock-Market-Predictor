import { TickerSearch } from "../components/TickerSearch";

export interface INavbarProps {}

export default function Navbar(props: INavbarProps) {
  return (
    <div className="fixed z-50 flex h-14 w-full items-center gap-4 border-b bg-white px-6 py-3 shadow-sm xl:hidden">
      <div className="flex items-center gap-2">
        <img src="/icon.svg" alt="Logo" className="h-8 w-8" />
        <span className="text-lg font-bold">Stock Predictor</span>
      </div>
      <div className="flex w-full flex-1 justify-center">
        <div className="inline-block">
          <TickerSearch />
        </div>
      </div>
    </div>
  );
}
