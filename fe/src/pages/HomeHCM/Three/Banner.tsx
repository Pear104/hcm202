import React from "react";
import { GrGroup } from "react-icons/gr";
import { RiOrganizationChart } from "react-icons/ri";

export default function Banner() {
  return (
    <>
      <div className="w-full grid grid-cols-2 p-[4vw] gap-[4vw]">
        <div className="">
          <div className="unbounded text-[3.4vw] text-red-500 uppercase mt-[6vw]">
            Nguyên tắc đoàn kết quốc tế
          </div>
          <div className="inter italic mt-[2vw] text-[1.4vw]">
             “Tự lực cánh sinh, dựa vào sức mình là chính” “Muốn người ta giúp
            cho, thì trước mình phải tự giúp lấy mình đã”
          </div>
        </div>
        <div className="relative">
          <div className="z-10 absolute top-0 left-0 w-[24vw] aspect-square rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[-10deg]">
            <img
              className="w-full aspect-square object-cover object-center rounded-xl"
              src="images/3_banner.jpg"
              alt=""
            />
          </div>
          <div className="absolute top-[2vw] left-[16vw] w-[22vw] aspect-square rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[14deg]">
            <img
              className="w-full aspect-square object-cover object-left rounded-xl"
              src="images/3_banner2.jpg"
              alt=""
            />
          </div>
        </div>

        <div className="col-span-2 grid grid-cols-2 gap-[4vw] mt-[4vw]">
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <GrGroup size={40} />
              </div>
              A. Đoàn kết trên cơ sở thống nhất mục tiêu và lợi ích; có lý, có
              tình
            </div>
          </div>
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <RiOrganizationChart size={40} />
              </div>
              B. Đoàn kết trên cơ sở độc lập, tự chủ
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
