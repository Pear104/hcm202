import React from "react";

export default function Footer() {
  return (
    <div className="w-screen h-screen px-[2vw] py-[1vw] pt-[6vw] relative">
      <div className="col-span-4">
        <div className="text-[4.7vw] font-bold grid-cols-2 unbounded">
          <div className="flex justify-between">
            <span>Nguyễn Lê Hạnh Uyên</span>
            <span className="text-red-500 text-[3.4vw]">SS180629</span>
          </div>
          <div className="flex justify-between">
            <span>Võ Đỗ Quang Dương</span>
            <span className="text-red-500 text-[3.4vw]">SE181769</span>
          </div>
          <div className="flex justify-between">
            <span>Đỗ Long Ánh</span>
            <span className="text-red-500 text-[3.4vw]">SE181818</span>
          </div>
          <div className="flex justify-between">
            <span>Lê Thế Trường</span>
            <span className="text-red-500 text-[3.4vw]">SE182338</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2">
        <div className="uppercase font-semibold text-[8vh] col-span-2 text-yellow-500"></div>
        <div>
          <div className="text-[2vh] flex flex-col gap-[1vh] mt-[2vh]"></div>
        </div>
        <div className="absolute bottom-0 right-[1vw] unbounded">
          Cooked by
          <div
            style={{
              textUnderlineOffset: "0.8rem",
            }}
            className="uppercase font-semibold text-[8vh] col-span-2 underline leading-[12vh]"
          >
            Nhóm 6
          </div>
        </div>
      </div>
    </div>
  );
}
