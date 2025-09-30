import React from "react";

export default function XaHoiChuNghia() {
  return (
    <div className="relative pl-[4vw] flex flex-col h-[42.3vw]">
      <div className="absolute right-[16vw] bottom-[2vh] text-right">
        <span className="text-[3vw] uppercase font-bold text-yellow-400">
          xã hội
        </span>
        <br />
        <span
          className="text-[10vh] leading-[12vh] italic underline uppercase font-bold"
          style={{
            textUnderlineOffset: "0.8rem",
          }}
        >
          chủ nghĩa
        </span>
      </div>

      <div className="absolute top-[10vh] right-[18vw] transition-all duration-400">
        <img
          className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-square object-contain object-center w-[24vw]"
          src="/images/xa-hoi-chu-nghia.png"
          loading="eager"
          alt=""
        />
      </div>
      <div className="absolute top-[3vw] transition-all duration-400 w-[48vw] text-[1.05vw]">
        <div>
          <div className="uppercase mx-[0.4vw] font-bold text-[1.5vw]">
            Cách mạng Tháng Mười Nga (1917) thắng lợi
          </div>
          <ul className="list-disc list-inside">
            <li>
              <span className="uppercase mr-[0.4vw] text-yellow-400 font-bold">
                Lần đầu tiên
              </span>{" "}
              trong lịch sử, giai cấp công nhân và nhân dân lao động lật đổ ách
              thống trị tư sản.
            </li>
            <li>
              Mở ra thời đại quá độ từ chủ nghĩa tư bản lên chủ nghĩa xã hội
              trên phạm vi thế giới.
            </li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase mx-[0.4vw] font-bold text-[1.5vw]">
            Giai cấp công nhân, nông dân và nhân dân lao động giành quyền làm
            chủ
          </div>
          <ul className="list-disc list-inside">
            <li>
              Lần đầu tiên đa số quần chúng lao động nắm quyền lực chính trị.
            </li>
            <li>Họ trở thành chủ thể quản lý nhà nước, quản lý xã hội.</li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase mx-[0.4vw] font-bold text-[1.5vw]">
            Sự ra đời của
            <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
              Nhà nước công – nông
            </span>
            (Nhà nước XHCN)
          </div>
          <ul className="list-disc list-inside">
            <li>Đây là kiểu nhà nước mới, khác với nhà nước tư sản.</li>
            <li>
              Là công cụ
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
                chính trị của đại đa số
              </span>
              , nhằm thực thi và bảo vệ quyền lực của nhân dân lao động.
            </li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase mx-[0.4vw] font-bold text-[1.5vw]">
            Nền dân chủ vô sản (dân chủ XHCN) được thiết lập
          </div>
          <ul className="list-disc list-inside">
            <li>Thay thế nền dân chủ tư sản vốn chỉ phục vụ thiểu số.</li>
            <li>Trở thành nền dân chủ thực sự của đại đa số.</li>
            <li>
              Mở rộng các quyền tự do, bình đẳng, tham gia chính trị cho quần
              chúng.
            </li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase mx-[0.4vw] font-bold text-[1.5vw]">
            Đặc trưng cơ bản của dân chủ XHCN
          </div>
          <ul className="list-disc list-inside">
            <li>
              Quyền lực thuộc về nhân dân:
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
                “dân làm chủ nhà nước và xã hội”.
              </span>
            </li>
            <li>
              Gắn với mục tiêu
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
                giải phóng giai cấp, giải phóng xã hội.
              </span>
            </li>
            <li>
              Nhằm
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold">
                bảo vệ lợi ích căn bản và lâu dài
              </span>
              của người lao động.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
