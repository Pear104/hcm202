import ReactLenis from "lenis/react";
import HeroSection from "./HeroSection";
import QuanNiem from "./QuanNiem/QuanNiem";
import RaDoi from "./RaDoi/RaDoi";
import LienHe from "./LienHe/LienHe";
import ParralaxBackground from "./ParralaxBackground";

export default function Home() {
  return (
    <ReactLenis root>
      <ParralaxBackground />
      <div className="bg-black/10 backdrop-blur-xs">
        <HeroSection />
        <QuanNiem />
        <RaDoi />
        <LienHe />
      </div>
    </ReactLenis>
  );
}
