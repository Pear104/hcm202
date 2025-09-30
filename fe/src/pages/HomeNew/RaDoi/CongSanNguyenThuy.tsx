import React from "react";

export default function CongSanNguyenThuy() {
  return (
    <div className="relative pl-[4vw] flex flex-col h-[42.3vw]">
      <span className="text-[4vh] uppercase font-bold text-yellow-400">
        Cộng sản
      </span>
      <span
        className="text-[10vh] leading-[6vw] italic underline uppercase font-bold"
        style={{
          textUnderlineOffset: "0.8rem",
        }}
      >
        Nguyên thủy
      </span>
      <div className="absolute top-[24vh] transition-all duration-400">
        <img
          className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-square object-cover object-center w-[24vw]"
          src="/images/cong-san-nguyen-thuy.png"
          loading="eager"
          alt=""
        />
      </div>
      <div className="absolute left-[32vw] top-[12vw] transition-all duration-400 w-[48vw] text-[1.4vw]">
        <div className="text-[1.5vw] font-semibold">
          Nhu cầu về
          <span className="text-[1.5vw] font-bold mx-[0.6vw] uppercase text-yellow-400">
            dân chủ
          </span>
          nảy sinh từ xã hội tự quản của thị tộc, bộ lạc
        </div>
        <div className="text-zinc-300">
          Trong cộng sản nguyên thủy, đã có hình thức manh nha của dân chủ - gọi
          là “dân chủ nguyên thủy” hay “dân chủ quân sự”
        </div>
        <div className="mt-[3vw]">
          <div className="uppercase text-[1.5vw] font-bold">Đặc trưng:</div>
          <ul className="list-inside">
            <li className="list-disc">
              Nhân dân bầu thủ lĩnh quân sự qua “Đại hội nhân dân”
            </li>
            <li className="list-disc">
              Mọi người có quyền phát biểu, quyết định (giơ tay, hoan hô).
            </li>
            <li className="list-disc">
              Quyền lực thật sự thuộc về nhân dân, dù trình độ sản xuất còn
              thấp. Tuy nhiên{" "}
              <span
                className="underline italic"
                style={{ textUnderlineOffset: "0.4rem" }}
              >
                chưa thật sự
              </span>{" "}
              có nền
              <span className="text-[1.5vw] font-bold mx-[0.4vw] uppercase text-yellow-400">
                dân chủ
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
