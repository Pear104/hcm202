import ReactLenis from "lenis/react";
import HeroSection from "./HeroSection";
import QuanNiem from "./QuanNiem/QuanNiem";
import RaDoi from "./RaDoi/RaDoi";
import LienHe from "./LienHe/LienHe";
import ParralaxBackground from "./ParralaxBackground";
import Credit from "./Credit/Credit";
import Chat from "../Chat/Chat";

export default function HomeNew() {
  return (
    <ReactLenis root>
      <ParralaxBackground />
      <div className="bg-black/10 backdrop-blur-xs">
        <HeroSection />
        <QuanNiem />
        <RaDoi />
        <LienHe />
        <Credit />
      </div>
      <Chat />
    </ReactLenis>
  );
}
