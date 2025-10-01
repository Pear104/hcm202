import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import CongSanNguyenThuy from "./CongSanNguyenThuy";
import ChiemHuuNoLe from "./ChiemHuuNoLe";
import PhongKien from "./PhongKien";
import TuBanChuNghia from "./TuBanChuNghia";
import XaHoiChuNghia from "./XaHoiChuNghia";

export default function SectionTwo() {
  useGSAP(() => {
    gsap.to("#rect", {
      motionPath: {
        path: "#path",
        align: "#path",
        autoRotate: true,
        alignOrigin: [0.5, 0.5],
      },
      ease: "power1.inOut",
      scrollTrigger: {
        trigger: "#svg",
        start: "top center",
        end: "bottom center",
        scrub: true,
      },
    });
  }, []);

  return (
    <>
      <div id="event-section" className="z-10 px-[2vw] h-[220vw]">
        <div className="relative">
          <div className="pt-[12vh]">
            <CongSanNguyenThuy />
            <ChiemHuuNoLe />
            <PhongKien />
            <TuBanChuNghia />
            <XaHoiChuNghia />
          </div>

          <svg
            className="w-full absolute top-0 left-0 -z-10"
            viewBox="0 0 500 1200"
            id="svg"
          >
            <path
              id="path"
              d="
    M 460 30
    L 460 230
    C 460 260, 440 260, 420 260
    L 70 260
    C 50 260, 30 280, 30 300
    L 30 440
    C 30 460, 50 470, 70 470
    L 420 470
    C 440 470, 460 480, 460 500
    L 460 670
    C 460 690, 440 700, 420 700
    L 70 700
    C 50 700, 30 720, 30 740
    L 30 880
    C 30 900, 50 910, 70 910
    L 420 910
    C 440 910, 460 920, 460 940
    L 460 1130
  "
              stroke="white"
              fill="none"
              stroke-width="1"
              stroke-linecap="round"
            />

            <g
              width="100px"
              height="100px"
              id="rect"
              viewBox="-100 -100 200 200"
              fill="white"
            >
              <circle
                className="eye"
                cx="40"
                cy="0"
                r={24}
                fill="#f5b800"
                // stroke="black"
              />
              <ellipse
                className="eye-ball"
                cx="40"
                cy="0"
                rx={14}
                ry={14}
                fill="black"
              />
            </g>
          </svg>
        </div>
      </div>
    </>
  );
}
