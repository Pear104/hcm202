import React from "react";
import { GrGroup } from "react-icons/gr";
import { RiOrganizationChart } from "react-icons/ri";

export default function Banner() {
  return (
    <>
      <div className="w-full p-[4vw] gap-[4vw]">
        <div className="flex justify-center items-center">
          <div className="relative h-[32vw] w-full">
            <div className="z-10 absolute top-0 left-1/2 -translate-x-[20%] w-[22vw] aspect-[9/11] rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[-18deg]">
              <img
                className="w-full aspect-[9/11] object-cover object-center rounded-xl"
                src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
                alt=""
              />
            </div>
            <div className="absolute top-[2vw] left-1/2 -translate-x-[90%] w-[20vw] aspect-[9/11] rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[34deg]">
              <img
                className="w-full aspect-[9/11] object-cover object-center rounded-xl"
                src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
                alt=""
              />
            </div>
          </div>
        </div>

        <div className="">
          <div className="unbounded text-[4vw] text-red-500 text-center uppercase">
             Lực lượng đoàn kết quốc tế và
            <br />
            hình thức tổ chức
          </div>
          <div className="inter italic text-center text-[1.4vw] mt-[2vw]">
            “Đoàn kết, đoàn kết, đại đoàn kết. Thành công, thành công, đại thành
            công”
          </div>
        </div>
        <div className="col-span-2 grid grid-cols-2 gap-[4vw] mt-[4vw]">
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <GrGroup size={40} />
              </div>
              A. Các lực lượng cần đoàn kết
            </div>
          </div>
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <RiOrganizationChart size={40} />
              </div>
              B. Hình thức tổ chức
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
