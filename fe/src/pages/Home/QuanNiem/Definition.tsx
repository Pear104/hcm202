import DanChuText from "@/components/DanChuText";
import React from "react";

export default function Definition() {
  return (
    <div className="w-screen h-screen flex items-center justify-center relative group">
      <div className="flex flex-col">
        <QnItem positionClass="group-hover:translate-x-[30vw] group-hover:-translate-y-[20vh]">
          Quan
        </QnItem>

        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:-translate-x-[42vw] group-hover:-translate-y-[40vh]">
          <img
            className="transition-all duration-400 opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 h-auto rounded-xl shadow-lg aspect-square object-cover object-center w-[20vw]"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png?auto=compress&cs=tinysrgb&w=600"
            loading="eager"
            alt=""
          />
        </div>

        <div
          className="transition-all duration-400 absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 group-hover:opacity-100 opacity-0"
          style={{ transform: "translate3d(0, 0, 0)" }}
        >
          <div className="flex px-[5vh] bg-black/10 rounded-full border border-white/30 backdrop-blur-md items-center cursor-pointer">
            <div className="py-[8vh] transition-all duration-400 w-[24vw] text-[2vw] text-center uppercase font-semibold tracking-widest">
              xuất hiện vào
              <br />
              thế kỷ VII-VI TCN
              <br />ở Hy Lạp cổ đại
            </div>
          </div>
        </div>
        <div className="text-[34vh] absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:translate-x-[2vw] group-hover:translate-y-[30vh] group-hover:text-[20vh] leading-[10vh]">
          <DanChuText />
        </div>

        <QnItem positionClass="group-hover:-translate-x-[30vw] group-hover:translate-y-[20vh]">
          niệm
        </QnItem>
      </div>
    </div>
  );
}

const QnItem = ({
  positionClass,
  children,
}: {
  positionClass: string;
  children: React.ReactNode;
}) => {
  return (
    <div
      id="qn-title"
      className={`w-[28vw] text-[20vh] uppercase font-semibold cursor-pointer relative transition-all duration-400 group-hover:scale-105 group-hover:text-blue-500 text-center ${positionClass}`}
      style={{
        transform: "translate3d(0, 0, 0)",
        backfaceVisibility: "hidden",
        willChange: "transform",
      }}
    >
      {children}
      <div className="absolute"></div>
    </div>
  );
};
