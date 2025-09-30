import React from "react";

export default function TuBanChuNghia() {
  return (
    <div className="relative pl-[4vw] flex flex-col h-[42.3vw]">
      <span className="text-[4vh] uppercase font-bold text-right pt-[10vh]">
        <span className="text-[16.6vh] border-zinc-300 border-t-4 pt-[1vh]">
          tư
        </span>
        <br />
        <span className="text-[5.4vw] text-blue-500 leading-[0vh]">bản</span>
        <br />
        <span className="text-[5.2vw] leading-[10vh] text-orange-500">chủ</span>
        <br />
        <span
          className="text-[3.5vw] leading-[8vh] underline text-green-500"
          style={{
            textUnderlineOffset: "0.8rem",
          }}
        >
          nghĩa
        </span>
      </span>
      <div className="absolute top-[3vw] right-[18vw] transition-all duration-400">
        <img
          className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-square object-contain object-center w-[12vw]"
          src="/images/tu-ban-chu-nghia.png"
          loading="eager"
          alt=""
        />
      </div>
      <div className="absolute left-[12vw] top-[12vh] transition-all duration-400 w-[66vw] text-[1vw]">
        <div>
          <div className="uppercase font-bold text-[2vw]">
            Cuối thế kỷ XIV - XV:
          </div>
          <ul className="list-disc list-inside">
            <li>
              Giai cấp tư sản ra đời, mang theo tư tưởng tiến bộ về tự do, công
              bằng, dân chủ.
            </li>
            <li>Đây là tiền đề cho sự xuất hiện của nền dân chủ tư sản</li>
          </ul>
        </div>
        <div className="mt-[1vw]">
          <div className="uppercase font-bold text-[2vw]">
            Thế kỷ XVIII – Cách mạng tư sản điển hình:
          </div>
          <div className="w-full grid grid-cols-2 gap-4">
            <ul className="list-disc list-inside">
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold text-[1.5vw]">
                Cách mạng Mỹ (1775–1783):
              </span>{" "}
              <li>Giành độc lập khỏi Anh, lập cộng hòa dân chủ.</li>
              <li>
                Hiến pháp 1787 quy định tam quyền phân lập, Bill of Rights
                (1791) bảo vệ quyền công dân.
              </li>
              <li>
                Trở thành mô hình dân chủ đại diện tiêu biểu, ảnh hưởng toàn
                cầu.
              </li>
            </ul>
            <ul className="list-disc list-inside">
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold text-[1.5vw]">
                Cách mạng Pháp (1789–1799):
              </span>{" "}
              <li>
                Lật đổ chế độ quân chủ, đề cao khẩu hiệu “Tự do – Bình đẳng –
                Bác ái”.
              </li>
              <li>
                Quốc hội và các chính thể dân chủ được thiết lập nhưng bất ổn
                dẫn đến sự nổi lên của Napoleon.
              </li>
              <li>
                Dù thăng trầm, nhưng đã mở đường cho cải cách dân chủ ở châu Âu,
                truyền cảm hứng cho phong trào dân chủ thế giới.
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-[1vw]">
          <div className="uppercase font-bold text-[2vw]">
            Đặc trưng của dân chủ tư sản:
          </div>
          <ul className="list-disc list-inside">
            <li>Là bước tiến lớn của lịch sử loài người (theo Mác - Lênin).</li>
            <li>Mang lại các giá trị tự do, bình đẳng, dân chủ.</li>
            <li>
              Tuy nhiên, vẫn là
              <span className="uppercase mx-[0.4vw] text-yellow-400 font-bold text-[1.2vw]">
                dân chủ của thiểu số – những người sở hữu tư liệu sản xuất
              </span>
              – đối với đại đa số nhân dân lao động.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
