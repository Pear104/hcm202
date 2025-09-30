import ReactLenis from "lenis/react";
import HeroSection from "./HeroSection";
import QuanNiem from "./QuanNiem/QuanNiem";
import RaDoi from "./RaDoi/RaDoi";
import LienHe from "./LienHe/LienHe";
import ParralaxBackground from "./ParralaxBackground";
import Credit from "./Credit/Credit";
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
