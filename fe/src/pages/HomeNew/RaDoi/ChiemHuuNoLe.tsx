import React from "react";

export default function ChiemHuuNoLe() {
  return (
    <div className="pl-[4vw] flex flex-col relative h-[42.3vw]">
      <span className="uppercase font-bold text-right pt-[10vw]">
        <span className="text-[3.2vw] border-zinc-300 border-t-4 pt-[1vh]">
          Chiếm
        </span>
        <br />
        <span className="text-[4.8vw] text-blue-500 leading-[4vh]">hữu</span>
        <br />
        <span className="text-[7.2vw] leading-[10vh] text-orange-500">nô</span>
        <br />
        <span
          className="text-[19.6vh] leading-[14vh] underline text-green-500"
          style={{
            textUnderlineOffset: "0.8rem",
          }}
        >
          lệ
        </span>
      </span>
      <div className="absolute top-[4vw] left-[10vw] transition-all duration-400">
        <img
          className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-[9/12] object-contain object-center w-[24vw]"
          src="/images/chiem-huu-no-le.png"
          loading="eager"
          alt=""
        />
      </div>
      <div className="absolute left-[38vw] top-[6vw] transition-all duration-400 w-[40vw] text-[1.2vw]">
        <div>
          Xuất hiện khi
          <span className="uppercase text-yellow-400 font-bold text-[1.6vw] mx-[0.4vw]">
            chế độ tư hữu
          </span>
          và
          <span className="uppercase text-yellow-400 font-bold text-[1.6vw] mx-[0.4vw]">
            giai cấp
          </span>
          ra đời, thay thế dân chủ nguyên thủy, và nguồn gốc lịch sử của nền dân
          chủ có thể bắt nguồn từ các nền văn minh cổ đại, đặc biệt là Hy Lạp cổ
          đại.
        </div>
        <div>
          Được tổ chức thành nhà nước dân chủ chủ nô, có cơ chế công dân tham
          gia vào đại hội nhân dân.
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase text-[1.5vw] font-bold">Công dân:</div>
          <ul className="list-inside">
            <li className="list-disc">Chủ nô</li>
            <li className="list-disc">
              Một số công dân tự do (tăng lữ, thương gia, trí thức).
            </li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase text-[1.5vw] font-bold">Cách thức:</div>
          <ul className="list-inside">
            <li className="list-disc">
              Thành viên đại hội được chọn bằng bốc thăm.
            </li>
            <li className="list-disc">
              Quyết định thông qua theo đa số phiếu.
            </li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase text-[1.5vw] font-bold">Hạn chế:</div>
          <ul className="list-inside">
            <li className="list-disc">
              phụ nữ, nô lệ, người ngoại kiều không có quyền chính trị.
            </li>
            <li className="list-disc">
              Thực chất dân chủ chủ nô là dân chủ cho thiểu số, nhằm duy trì
              quyền lợi của giai cấp thống trị.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
