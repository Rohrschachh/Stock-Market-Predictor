import { ExclamationTriangleIcon } from "@heroicons/react/24/solid";

export interface ISnackbarProps {}

export function Snackbar(props: ISnackbarProps) {
  return (
    <div className="fixed bottom-4 left-4 z-[100] flex animate-bounce overflow-hidden rounded-md bg-red-500 p-3 px-5">
      <div className="flex min-w-[200px] items-center gap-2 text-white">
        <ExclamationTriangleIcon className="h-6 w-6" />
        <span className="text-lg font-semibold">Invalid ticker</span>
      </div>
    </div>
  );
}
