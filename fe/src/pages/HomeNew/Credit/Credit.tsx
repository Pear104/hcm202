import React from "react";

export default function Credit() {
  return (
    <div className="w-screen h-screen px-[2vw] py-[1vw]">
      <div className="col-span-4">
        <div className="text-yellow-400 text-[4.7vw] font-bold grid-cols-2">
          <div className="flex justify-between">
            <span>Nguyễn Lê Hạnh Uyên</span>
            <span className="text-zinc-300 text-[3.4vw]">SS180629</span>
          </div>
          <div className="flex justify-between">
            <span>Võ Đỗ Quang Dương</span>
            <span className="text-zinc-300 text-[3.4vw]">SE181769</span>
          </div>
          <div className="flex justify-between">
            <span>Đỗ Long Ánh</span>
            <span className="text-zinc-300 text-[3.4vw]">SE181818</span>
          </div>
          <div className="flex justify-between">
            <span>Lê Thế Trường</span>
            <span className="text-zinc-300 text-[3.4vw]">SE182338</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2">
        <div className="uppercase font-semibold text-[8vh] col-span-2">
          Sản phẩm sáng tạo môn
          <span className="text-blue-500 ml-4">MLN131</span>
        </div>
        <div>
          <div>Nội dung sản phảm được tham khảo từ:</div>
          <div className="text-[2vh] flex flex-col gap-[1vh] mt-[2vh]">
            <div>Giáo trình chủ nghĩa xã hội khoa học, NXB CTQGST, HN.2021</div>
            <div>HCM toàn tập, NXB CTQG, HN. 2000 Tập 5</div>
            <div>
              Đảng Cộng sản Việt Nam. Văn kiện Đại hội đại biểu toàn quốc lần
              thứ XIII. Tập I. H. NXB CTQGST, 2021
            </div>
            <div>
              Đảng Cộng sản Việt Nam. Văn kiện Đại hội đại biểu toàn quốc lần
              thứ XIII. Tập II. H. NXB CTQGST, 2021
            </div>
          </div>
        </div>
        <div className="relative">
          <div className="absolute bottom-0 right-[5vw]">
            Cooked by
            <div
              style={{
                textUnderlineOffset: "0.8rem",
              }}
              className="uppercase font-semibold text-[8vh] col-span-2 underline leading-[12vh]"
            >
              Nhóm 3
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
