import { createContext } from "react";

export type AppTheme = "light" | "dark";

export type AppContext = {
  theme: AppTheme;
  ticker: string;
  handleCheckTickerValidity: (ticker: string) => void;
  handleSetTicker: (ticker: string) => void;
  handleSwitchTheme: () => void;
  handleShowSnackbar: () => void;
};

const initialContextState: AppContext = {
  theme: "light",
  ticker: "TCS.NS",
  handleCheckTickerValidity: () => {},
  handleSetTicker: () => {},
  handleSwitchTheme: () => {},
  handleShowSnackbar: () => {},
};

const AppContext = createContext<AppContext>(initialContextState);

export const AppContextProvider = AppContext.Provider;
export const AppContextConsumer = AppContext.Consumer;

export default AppContext;
