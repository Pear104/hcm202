import { useGSAP } from "@gsap/react";
import gsap from "gsap";

export default function LienHe() {
  return (
    <div className="w-screen h-screen flex items-center justify-center">
      <div className="group flex flex-col space-x-10">
        <div
          id="qn-title"
          className="text-[10rem] uppercase font-semibold leading-[10rem] cursor-pointer relative transition-all duration-300 group-hover:scale-105 group-hover:text-blue-500 group-hover:translate-x-[30vw] group-hover:-translate-y-[20vh] text-nowrap"
        >
          Liên hệ
          <div className="absolute"></div>
        </div>
        <div className="transition-all duration-300 scale-0 group-hover:scale-100">
          Ahihi
        </div>
        <div
          id="qn-title"
          className="w-[28rem] text-[10rem] uppercase font-semibold leading-[10rem] cursor-pointer relative transition-all duration-300 group-hover:scale-105 group-hover:text-blue-500 group-hover:-translate-x-[30vw] group-hover:translate-y-[20vh] text-nowrap"
        >
          Việt Nam
          <div className="absolute"></div>
        </div>
      </div>
    </div>
  );
}
