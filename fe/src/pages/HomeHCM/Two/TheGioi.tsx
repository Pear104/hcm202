import React from "react";
import { BiX } from "react-icons/bi";

export default function TheGioi() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
      {/* Trigger */}
      <div className="flex flex-col items-center">
        <div className="font-bold text-[1.4vw] text-center">
          Mặt trận nhân dân thế giới đoàn kết với Việt Nam chống đế quốc xâm lược
        </div>
        <div className="my-[2vw] text-center text-[1.1vw]">
          Phong trào phản chiến và ủng hộ quốc tế lan rộng, tạo chỗ dựa vững chắc
          cho kháng chiến Việt Nam.
        </div>
        <button
          className="text-red-500 cursor-pointer hover:scale-[1.1] duration-300 transition-all"
          onClick={() => setIsOpen(v => !v)}
        >
          Xem thêm
        </button>
      </div>

      {/* Modal */}
      {isOpen && (
        <div
          data-lenis-prevent-wheel
          className="fixed inset-0 z-50 w-screen h-screen overflow-y-auto bg-zinc-900"
        >
          {/* Close */}
          <button
            onClick={() => setIsOpen(false)}
            className="fixed top-4 left-4 hover:scale-[1.1] transition-all duration-300 cursor-pointer"
          >
            <BiX className="text-white text-4xl" />
          </button>

          {/* Header */}
          <div className="max-w-[1100px] mx-auto px-6 pt-[3.5vw] pb-6">
            <div className="text-red-500 unbounded uppercase text-center text-[1.2vw] tracking-wider">
              IB
            </div>
            <h1 className="unbounded text-center text-red-500 text-[2.3vw] font-extrabold mt-2 leading-snug">
              MẶT TRẬN NHÂN DÂN THẾ GIỚI
              <br /> ĐOÀN KẾT VỚI VIỆT NAM <br /> CHỐNG ĐẾ QUỐC XÂM LƯỢC
            </h1>
            <p className="text-zinc-300 text-[1vw] text-center mt-3 max-w-[950px] mx-auto">
              Mặt trận nhân dân thế giới đoàn kết với Việt Nam được hình thành nhằm tranh thủ sự đồng tình, ủng hộ của các nước xã hội chủ nghĩa và bạn bè quốc tế, tạo thế dựa vững chắc cho cách mạng Việt Nam.
            </p>
          </div>

          {/* Content */}
          <div className="max-w-[1100px] mx-auto px-6 pb-[5vw] space-y-12 text-zinc-200 text-[0.95vw] leading-relaxed">
            {/* Block 1: Ảnh trái - text phải */}
            <div className="grid grid-cols-12 gap-6 items-start">
              <div className="col-span-12 md:col-span-7">
                <ImageCard src="/images/2-4-1.png" alt="Biểu tình ở Moskva 1965" />
              </div>
              <div className="col-span-12 md:col-span-5">
                <TextCard>
                  Ngày 08/02/1965, nhân dân thủ đô Moskva (Liên Xô) đã tổ chức một
                  cuộc mít-tinh lớn để bày tỏ sự ủng hộ mạnh mẽ đối với cuộc kháng
                  chiến chính nghĩa của nhân dân Việt Nam.
                </TextCard>
              </div>
            </div>

            {/* Block 2: text trái - ảnh phải */}
            <div className="grid grid-cols-12 gap-6 items-start">
              <div className="col-span-12 md:col-span-5">
                <TextCard>
                  Tại Pháp, phong trào phản chiến trong binh lính ngày càng lan rộng,
                  lên án cuộc chiến tranh phi nghĩa ở Việt Nam.
                </TextCard>
              </div>
              <div className="col-span-12 md:col-span-7">
                <ImageCard src="/images/2-4-2.png" alt="Phong trào phản chiến tại Pháp" />
              </div>
            </div>

            {/* Block 3: ảnh trái - text phải */}
            <div className="grid grid-cols-12 gap-6 items-start">
              <div className="col-span-12 md:col-span-7">
                <ImageCard src="/images/2-4-3.png" alt="Phong trào phản chiến tại Mỹ" />
              </div>
              <div className="col-span-12 md:col-span-5">
                <TextCard>
                  Tại Mỹ, phong trào phản chiến phát triển mạnh mẽ với nhiều hình
                  thức đa dạng, trong đó có các hoạt động tiêu biểu như “Ngày ngừng
                  hoạt động” năm 1968 và “Tạm ngưng hòa bình” năm 1969.
                </TextCard>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* --- UI helpers --- */
function TextCard({ children }: { children: React.ReactNode }) {
  return (
    <div className=" rounded-xl p-5">
      <p className="text-zinc-200 text-2xl">{children}</p>
    </div>
  );
}

function ImageCard({ src, alt }: { src: string; alt?: string }) {
  return (
    <div className="rounded-xl">
      <div className="w-full rounded-lg overflow-hidden">
        {src ? (
          <img src={src} alt={alt || ""} className="w-full h-auto object-cover" />
        ) : (
          <div className="w-full aspect-[16/9]  grid place-items-center text-zinc-300">
            Thêm ảnh ở đây
          </div>
        )}
      </div>
    </div>
  );
}
