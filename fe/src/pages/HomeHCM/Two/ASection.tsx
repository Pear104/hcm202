import { div } from "motion/react-client";
import React from "react";

export default function ASection() {
  return (
    <>
      <div className="px-[4vw]">
        <div className="grid grid-cols-2 gap-[6vw]">
          <div>
            <img
              className="w-full aspect-[12/9] object-cover object-center rounded-xl"
              src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
              alt=""
            />
          </div>
          <div className="">
            <div className="unbounded text-[4vw] text-[#FF2F2F] capitalize leading-[4.4vw] pl-[1.5vw] border-l-[1vw]">
              A
            </div>
            <div className="unbounded text-[4vw] text-red-500/90 uppercase my-[1.6vw]">
              Các lực lượng cần đoàn kết
            </div>
            <div className="inter italic text-[1.2vw]">
              Lực lượng đoàn kết quốc tế trong tư tưởng Hồ Chí Minh bao gồm:
              phong trào cộng sản và công nhân quốc tế; phong trào đấu tranh
              giải phóng dân tộc và phong trào hoà bình, dân chủ thế giới, trước
              hết là phong trào chống chiến tranh của nhân dân các nước đang xâm
              lược Việt Nam.
            </div>
          </div>
        </div>
        <div className="mt-[4vw] flex flex-col items-center">
          <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize">
            Hình thức tổ chức
          </div>
          <div className="w-[64vw] space-y-[1vw]">
            <div className="h-fit text-center">
              Đoàn kết quốc tế trong tư tưởng Hồ Chí Minh không phải là một vấn
              đề sách lược hay thủ đoạn chính trị nhất thời, mà là vấn đề có
              tính nguyên tắc, là một đòi hỏi khách quan của cách mạng Việt Nam.
              Người luôn khẳng định, đoàn kết quốc tế vừa là nhu cầu tất yếu,
              vừa là điều kiện để cách mạng Việt Nam gắn bó chặt chẽ với phong
              trào cách mạng thế giới.
            </div>
            <div className="grid grid-cols-3 gap-x-[6vw] my-[2vw] gap-y-[2vw]">
              <div className="border border-dashed border-red-500"></div>
              <div className="border border-dashed border-red-500"></div>
              <div className="border border-dashed border-red-500"></div>
            </div>
          </div>
        </div>
        <div className="w-full grid grid-cols-4 gap-[4vw] mt-[2vw] mb-[4vw]">
          <Item />
          <Item />
          <Item />
          <Item />
        </div>
      </div>
    </>
  );
}

const Item = () => {
  return (
    <div className="flex flex-col items-center">
      <div className="font-bold text-[1.4vw] text-center">
        Mặt trận đại đoàn kết dân tộc
      </div>
      <div className="my-[2vw] text-center">
        Hồ Chí Minh chủ trương xây dựng mặt trận thống nhất để khơi dậy sức mạnh
        toàn dân trong đấu tranh chống đế quốc.
      </div>
      <div className="text-red-500">Xem thêm </div>
    </div>
  );
};
