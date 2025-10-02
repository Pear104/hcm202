import React from "react";
import { GrGroup } from "react-icons/gr";
import { RiOrganizationChart } from "react-icons/ri";

export default function Banner() {
  return (
    <>
      <div className="w-full grid grid-cols-2 mt-[8vw] p-[4vw] gap-[4vw]">
        <div className="relative">
          <div className="z-10 absolute top-0 left-0 w-[28vw] aspect-square rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[-10deg]">
            <img
              className="w-full aspect-square object-cover object-center rounded-xl"
              src="images/1_banner.jpg"
              alt=""
            />
          </div>
          <div className="absolute top-[2vw] left-[16vw] w-[24vw] aspect-square rounded-3xl flex justify-center items-center p-[2vw] bg-white/20 backdrop-blur-2xl rotate-[14deg]">
            <img
              className="w-full aspect-square object-cover object-center rounded-xl"
              src="images/1_banner2.jpg"
              alt=""
            />
          </div>
        </div>
        <div className="">
          <div className="unbounded text-[4vw] text-red-500 text-end">
            Sự cần thiết
            <br />
            phải đoàn kết
            <br />
            quốc tế
          </div>
          <div className="inter italic text-end mt-[2vw]">
            "Dù màu da có khác nhau, trên đời này chỉ có hai giống người: Giống
            người bóc lột và giống người bị bóc lột. Mà cũng chỉ có một mối
            tình hữu ái là thật mà thôi: tình hữu ái vô sản".
          </div>
        </div>
        <div className="col-span-2 grid grid-cols-2 gap-[4vw] mt-[4vw]">
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <GrGroup size={40} />
              </div>
              a. Thực hiện đoàn kết quốc tế nhằm kết hợp sức mạnh dân tộc với
              sức mạnh thời đại, tạo sức mạnh tổng hợp cho cách mạng
            </div>
          </div>
          <div className="rounded-3xl uppercase unbounded text-[1.2vw] bg-red-500 text-white text-center flex justify-center items-center p-[2vw]">
            <div className="">
              <div className="flex justify-center items-center mb-[1vw]">
                <RiOrganizationChart size={40} />
              </div>
              b. Thực hiện đoàn kết quốc tế nhằm góp phần cùng nhân dân thế giới
              thực hiện thắng lợi các mục tiêu cách mạng của thời đại
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
