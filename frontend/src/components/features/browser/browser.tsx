import { useSelector } from "react-redux";
import { RootState } from "#/store";
import { BrowserSnapshot } from "./browser-snapshot";
import { EmptyBrowserMessage } from "./empty-browser-message";

export function BrowserPanel() {
  const { url, screenshotSrc } = useSelector(
    (state: RootState) => state.browser,
  );
  let imgSrc = "";
  if (screenshotSrc){
    if (screenshotSrc.startsWith("http")) {
      imgSrc = screenshotSrc;
    } else if (screenshotSrc.startsWith("data:image/png;base64,")) {
      imgSrc = screenshotSrc;
    } else {
      imgSrc = `data:image/png;base64,${screenshotSrc}`;
    }
  }

  return (
    <div className="h-full w-full flex flex-col text-neutral-400">
      <div className="w-full p-2 truncate border-b border-neutral-600">
        {url}
      </div>
      <div className="overflow-y-auto grow scrollbar-hide rounded-xl">
        {screenshotSrc ? (
          <BrowserSnapshot src={imgSrc} />
        ) : (
          <EmptyBrowserMessage />
        )}
      </div>
    </div>
  );
}
