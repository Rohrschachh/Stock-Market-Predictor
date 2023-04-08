import { PropsWithChildren, useDeferredValue, useState } from "react";
import { AppContextProvider, AppTheme } from "./AppContext";
import useLocalStorage from "../hooks/useLocalStorage";
import { Snackbar } from "../components/Snackbar";

export interface IAppContextComponentProps extends PropsWithChildren {}

export function AppContextComponent(props: IAppContextComponentProps) {
  const { children } = props;
  const [theme, setTheme] = useState<AppTheme>("light");
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false);
  const [ticker, setTicker] = useLocalStorage<string>("ticker", "TCS.NS");

  const handleCheckTickerValidity = (tkr: string) => {};

  const handleSetTicker = (tkr: string) => {
    setTicker(tkr);
  };

  const handleSwitchTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const handleShowSnackbar = () => {
    setShowSnackbar(true);
    setTimeout(() => {
      setShowSnackbar(false);
    }, 4000);
  };

  return (
    <AppContextProvider
      value={{
        theme,
        ticker,
        handleCheckTickerValidity,
        handleSetTicker,
        handleSwitchTheme,
        handleShowSnackbar,
      }}
    >
      {showSnackbar && <Snackbar />}
      {children}
    </AppContextProvider>
  );
}
