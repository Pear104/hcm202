import React from "react";

export default function SectionOne() {
  return (
    <>
      <div className="w-screen h-[100vh] relative group">
        <div className="w-screen h-screen flex items-center justify-center relative group">
          <div className="flex flex-col text-zinc-200">
            <QnItem positionClass="group-hover:translate-x-[30vw] group-hover:-translate-y-[20vh]">
              lịch
            </QnItem>
            <div className="text-[34vh] absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:-translate-x-[48vw] group-hover:-translate-y-[40vh] group-hover:text-[20vh] leading-[10vh] uppercase font-semibold text-nowrap text-yellow-400">
              Ra đời
            </div>
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400">
              <img
                className="transition-all duration-400 opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 h-auto rounded-xl shadow-lg aspect-square object-cover object-center w-[32vw]"
                src="/images/ra-doi.png"
                loading="eager"
                alt=""
              />
            </div>
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-400 group-hover:translate-x-[0vw] group-hover:translate-y-[30vh] text-[7vw] opacity-0 group-hover:opacity-100 leading-[10vh] uppercase font-semibold text-nowrap text-yellow-400">
              và phát triển
            </div>

            <QnItem positionClass="group-hover:-translate-x-[40vw] group-hover:translate-y-[20vh]">
              sử
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
