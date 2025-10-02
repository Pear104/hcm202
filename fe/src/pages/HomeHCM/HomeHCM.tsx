import ReactLenis from "lenis/react";
import Chat from "../Chat/Chat";
import Banner from "./Landing/Banner";
import Introduction from "./Landing/Introduction";
import Document from "./Landing/Document";

export default function HomeHCM() {
  return (
    <ReactLenis root>
      <div className="">
        <Banner />
        <Introduction />
        <Document />
      </div>
      <Chat />
    </ReactLenis>
  );
}
