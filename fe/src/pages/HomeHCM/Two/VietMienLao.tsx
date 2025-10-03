import React from "react";
import { BiX } from "react-icons/bi";

export default function VietMienLao() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
      {/* Trigger block */}
      <div className="flex flex-col items-center">
        <div className="font-bold text-[1.4vw] text-center">
          Mặt trận đoàn kết
          <br />
          Việt – Miên – Lào
        </div>
        <div className="my-[2vw] text-center text-[1.2vw]">
          Khối liên minh nhân dân ba nước hình thành, đoàn kết chiến đấu chống
          Pháp thắng lợi.
        </div>
        <button
          className="text-red-500 hover:underline"
          onClick={() => setIsOpen((v) => !v)}
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
            aria-label="Đóng"
          >
            <BiX className="text-white text-4xl" />
          </button>

          {/* Header */}
          <div className="max-w-[1100px] mx-auto px-5 pt-[3.5vw] pb-6">
            <div className="text-red-500 unbounded uppercase text-center text-[1.2vw] tracking-wider">
              IB
            </div>
            <h1 className="unbounded text-center text-[#ec3343] text-[2.4vw] font-extrabold mt-2">
              MẶT TRẬN ĐOÀN KẾT
              <br />
              VIỆT – MIÊN – LÀO
            </h1>
            <p className="text-zinc-300 text-[1vw] text-center mt-3">
              Mặt trận đoàn kết Việt – Miên – Lào được thành lập nhằm phối hợp
              và giúp đỡ lẫn nhau trong cuộc đấu tranh giải phóng dân tộc, cùng
              hướng tới thắng lợi.
            </p>
          </div>

          {/* Content grid */}
          <div className="max-w-[1100px] mx-auto px-6 pb-[5vw]">
            <div className="grid grid-cols-12 gap-6">
              {/* Row 1: Left = big image, Right = text */}
              <div className="col-span-12 md:col-span-6 space-y-[1vw]">
                <ImageCard
                  src="/images/viet-mien-lao1.png"
                  alt="Đại hội/Kháng chiến"
                  caption=""
                />
                <TextCard>
                  <p className="mb-3">
                    Tiếp đó, vào tháng 3/1951, Hội nghị liên minh nhân dân ba
                    nước Đông Dương được tổ chức tại xã Vinh Quang với sự tham
                    dự của đại diện Mặt trận Liên Việt (Việt Nam), Mặt trận Lào
                    Ítxala và Mặt trận Khơ-me Ítxarắc (Campuchia). Hội nghị đã
                    thống nhất thành lập khối liên minh nhân dân Việt – Miên –
                    Lào trên nguyên tắc tự nguyện, bình đẳng, tôn trọng chủ
                    quyền và tương trợ lẫn nhau.
                  </p>
                  <p>
                    Sự ra đời của khối liên minh góp phần củng cố sức mạnh đại
                    đoàn kết, đẩy mạnh cuộc kháng chiến chống thực dân giành
                    thắng lợi, đồng thời củng cố và phát triển chính quyền dân
                    tộc, chính quyền nhân dân ở cả ba nước.
                  </p>
                </TextCard>
              </div>
              <div className="col-span-12 md:col-span-6 space-y-[1vw]">
                <TextCard>
                  <p className="mb-3">
                    Bước sang năm 1951, cuộc kháng chiến chống thực dân Pháp của
                    nhân dân ba nước Đông Dương bước vào giai đoạn phát triển
                    mới. Cục diện chiến tranh có nhiều chuyển biến sâu sắc, xuất
                    hiện nhiều thuận lợi cơ bản nhưng cũng không ít khó khăn,
                    phức tạp, đòi hỏi sự liên kết chặt chẽ giữa ba dân tộc Việt
                    Nam, Lào và Campuchia.
                  </p>
                  <p>
                    Trong bối cảnh đó, Đại hội đại biểu lần thứ II của Đảng Cộng
                    sản Đông Dương (11-19/02/1951) tại xã Vinh Quang (Chiêm Hóa,
                    Tuyên Quang) đã đưa ra nhiều quyết sách quan trọng. Chủ tịch
                    Hồ Chí Minh nhấn mạnh yêu cầu đoàn kết chiến đấu của ba dân
                    tộc bạn Miên, Lào, Việt và tiến hành thành lập mặt trận
                    thống nhất.
                  </p>
                </TextCard>
                <div className="col-span-12 md:col-span-6 space-y-6">
                  <ImageCard
                    src="/images/viet-lao2.png"
                    alt="Liên minh Việt – Miên – Lào"
                    caption=""
                  />
                </div>
              </div>

              {/* Row 2: Left = text, Right = two stacked images */}
              <div className="col-span-12 md:col-span-6"></div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* ----------------- Small UI helpers ----------------- */

function TextCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-zinc-800/70 rounded-xl p-5 text-zinc-200 text-[0.95vw] leading-relaxed shadow-[0_0_0_1px_rgba(255,255,255,0.04)]">
      {children}
    </div>
  );
}

function ImageCard({
  src,
  alt,
  caption,
}: {
  src: string;
  alt?: string;
  caption?: string;
}) {
  return (
    <figure className="bg-zinc-800/70 rounded-xl p-3 shadow-[0_0_0_1px_rgba(255,255,255,0.04)]">
      <div className="w-full rounded-lg overflow-hidden">
        {src ? (
          // Actual image
          <img
            src={src}
            alt={alt || ""}
            className="w-full h-auto object-cover"
          />
        ) : (
          // Placeholder frame with aspect
          <div className="w-full aspect-[16/9] bg-zinc-700/60 grid place-items-center text-zinc-300">
            Thêm ảnh ở đây
          </div>
        )}
      </div>
      {caption ? (
        <figcaption className="text-[0.8vw] text-zinc-400 mt-2">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
