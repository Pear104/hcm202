import React from "react";

export default function PhongKien() {
  return (
    <div className="relative pl-[4vw] flex flex-col h-[42.3vw]">
      <div className="absolute bottom-[5vw] right-[16vw]">
        <span className="text-[4vw] leading-[90%] italic uppercase font-bold">
          Phong kiến
          {/* 🌪 Phong kiến 🐜 */}
        </span>
      </div>

      <div className="absolute right-[15vw] top-[5vw] transition-all duration-400">
        <img
          className="transition-all duration-400 opacity-100 rounded-xl shadow-lg aspect-square object-cover object-center w-[24vw]"
          src="/images/phong-kien.png"
          loading="eager"
          alt=""
        />
      </div>
      <div className="absolute top-[7vw] transition-all duration-400 w-[48vw] text-[1.5vw]">
        <div>
          Sau khi chế độ chiếm hữu nô lệ tan rã, dân chủ chủ nô biến mất, thay
          bằng{" "}
          <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
            chuyên chế phong kiến
          </span>{" "}
          hay còn gọi là
          <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
            thời kỳ trung cổ
          </span>{" "}
        </div>
        <div className="mt-[1vw]">
          <span className="uppercase mr-[0.4vw] text-yellow-400 font-bold">
            Vua chúa và quý tộc nắm toàn quyền
          </span>{" "}
          , quyết định mọi việc như những kẻ độc tài, bạo chúa.
        </div>
        <div className="mt-[1vw]">
          Sự thống trị được thần quyền hóa, khoác áo
          <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
            “ý chí của đấng tối cao”
          </span>{" "}
          để hợp thức hóa quyền lực.
        </div>
        <div className="mt-[1vw]">
          Nhân dân bị xem như phải phục tùng; không có quyền dân chủ hay đấu
          tranh chính trị.
        </div>
        <div className="mt-[1vw]">
          Vì thế, ý thức về dân chủ không có bước tiến đáng kể trong giai đoạn
          này hay gần như là không tồn tại và thời kỳ này được gọi là
          <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
            nền quân chủ phong kiến.
          </span>{" "}
        </div>
      </div>
    </div>
  );
}
