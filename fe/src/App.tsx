import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { MotionPathPlugin, ScrollTrigger, SplitText } from "gsap/all";
import ReactLenis from "lenis/react";
import MainRoutes from "./routes/MainRoutes";

gsap.registerPlugin(ScrollTrigger, SplitText, MotionPathPlugin);

function App() {
  return (
    <ReactLenis root>
      <MainRoutes />
    </ReactLenis>
  );
}

export default App;
