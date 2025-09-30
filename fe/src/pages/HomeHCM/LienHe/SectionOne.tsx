import React from "react";

export default function SectionOne() {
  return (
    <>
      <div className="w-screen h-[100vh] relative group">
        <div className="w-screen h-screen flex items-center justify-center relative group">
          <div className="flex flex-col text-zinc-200">
            <QnItem positionClass="group-hover:translate-x-[37vw] group-hover:-translate-y-[20vh]">
              dân
            </QnItem>

            <div className="text-[30vh] absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:translate-x-[24vw] group-hover:translate-y-[30vh] group-hover:text-[12vh] leading-[10vh] uppercase font-semibold text-nowrap text-yellow-400 group-hover:opacity-100 opacity-0">
              hiện tại
            </div>

            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400">
              <img
                className="transition-all duration-400 opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 h-auto rounded-xl shadow-lg aspect-square object-cover object-center w-[32vw]"
                src="/images/dan-chu-vn.jpg"
                loading="eager"
                alt=""
              />
            </div>
            <div className="text-[30vh] absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:-translate-x-[47vw] group-hover:-translate-y-[40vh] group-hover:text-[12vh] leading-[10vh] uppercase font-semibold text-nowrap text-yellow-400">
              việt nam
            </div>

            <QnItem positionClass="group-hover:-translate-x-[36vw] group-hover:translate-y-[20vh]">
              chủ
            </QnItem>
          </div>
        </div>
      </div>
    </>
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
      className={`w-[28vw] text-[20vh] uppercase font-semibold cursor-pointer relative transition-all duration-400 group-hover:scale-105 group-hover:text-zinc-100 text-center ${positionClass}`}
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
