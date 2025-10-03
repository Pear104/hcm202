import React from "react";
import { BiX } from "react-icons/bi";

export default function TheGioi() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
      <div className="flex flex-col items-center">
        <div className="font-bold text-[1.4vw] text-center">
          Mặt trận nhân dân thế giới đoàn kết với Việt Nam chống đế quốc xâm
          lược
        </div>
        <div className="my-[2vw] text-center">
          Phong trào phản chiến và ủng hộ quốc tế lan rộng, tạo chỗ dựa vững
          chắc cho kháng chiến Việt Nam.
        </div>
        <div
          className="text-red-500"
          onClick={() => setIsOpen((prev) => !prev)}
        >
          Xem thêm
        </div>
      </div>
      {isOpen && (
        <div
          data-lenis-prevent-wheel
          // data-lenis-prevent-touch
          className="fixed gap-x-[6vw] gap-y-[2vw] w-screen h-screen bg-zinc-900 top-0 left-0 z-50 overflow-y-scroll"
        >
          <div
            onClick={() => setIsOpen((prev) => !prev)}
            className="fixed top-4 left-4 hover:scale-[1.1] transition-all duration-300 cursor-pointer"
          >
            <BiX className="text-white text-4xl" />
          </div>
          <div className="text-red-500 unbounded text-[2vw] uppercase text-center mt-[2vw]">
            B
          </div>
          <div className="text-red-500 unbounded text-[2vw] uppercase text-center my-[1vw]">
            Mặt trận đại đoàn kết dân tộc
          </div>
          <div className="mb-[4vw] w-[70vw] mx-auto">
            Mặt trận đại đoàn kết dân tộc nhằm khơi dậy sức mạnh và quyền tự
            quyết của mỗi dân tộc trong sự nghiệp đấu tranh cách mạng. Ngay từ
            năm 1924, Hồ Chí Minh đã đưa ra quan điểm về việc thành lập “Mặt
            trận thống nhất của nhân dân chính quốc và thuộc địa” để chống chủ
            nghĩa đế quốc, đồng thời kiến nghị Quốc tế Cộng sản cần có những
            giải pháp cụ thể nhằm biến quan điểm này thành hiện thực. Từ ngày
            18/11/1930 đến nay, tổ chức Mặt trận Dân tộc thống nhất ở Việt Nam
            đã nhiều lần thay đổi tên gọi để phù hợp với từng giai đoạn cách
            mạng:
          </div>
          <div className="grid grid-cols-3 gap-x-[2vw] gap-y-[2vw] w-[70vw] mx-auto mb-[2vw]">
            <Item />
            <Item />
            <Item />
            <Item />
            <Item />
            <Item />
            <Item />
            <Item />
          </div>
        </div>
      )}
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
