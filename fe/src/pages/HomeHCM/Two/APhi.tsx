import React from "react";
import { BiX } from "react-icons/bi";

export default function APhi() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
      {/* Trigger */}
      <div className="flex flex-col items-center">
        <div className="font-bold text-[1.4vw] text-center">
          Mặt trận nhân dân Á - Phi <br /> đoàn kết với Việt Nam
        </div>
        <div className="my-[2vw] text-center">
          Khẳng định mối liên hệ vận mệnh giữa Việt Nam với châu Á – Phi, mở rộng
          mặt trận đoàn kết chống đế quốc.
        </div>
        <button
          className="text-red-500 hover:underline"
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
            aria-label="Đóng"
          >
            <BiX className="text-white text-4xl" />
          </button>

          {/* Header */}
          <div className="max-w-[1100px] mx-auto px-6 pt-[3.5vw] pb-6">
            <div className="text-red-500 unbounded uppercase text-center text-[1.2vw] tracking-wider">
              IB
            </div>
            <h1 className="unbounded text-center text-red-500 text-[2.4vw] font-extrabold mt-2">
              MẶT TRẬN NHÂN DÂN Á - PHI
              <br /> ĐOÀN KẾT VỚI VIỆT NAM
            </h1>
            <p className="text-zinc-300 text-[1vw] text-center mt-3 max-w-[820px] mx-auto">
              Mặt trận nhân dân Á – Phi được hình thành nhằm mở rộng và củng cố
              mối quan hệ đoàn kết hữu nghị giữa Việt Nam với các dân tộc trong khu vực.
            </p>
          </div>

          {/* Content */}
          <div className="max-w-[1100px] mx-auto px-6 pb-[5vw] space-y-8 text-zinc-200 text-[0.95vw] leading-relaxed">
            {/* Đoạn mở đầu */}
            <p>
              Hồ Chí Minh đặc biệt coi trọng hợp tác nhiều mặt với Trung Quốc – nước láng giềng
              có quan hệ lịch sử, văn hóa lâu dài với Việt Nam – để tăng cường sức mạnh đoàn kết
              trong cuộc đấu tranh chống chủ nghĩa đế quốc.
            </p>

            {/* HÀNG 1: Ảnh (bạn có thể dùng ảnh collage 3 khung) + khối chú thích bên phải */}
            <div className="grid grid-cols-12 gap-6 items-start">
              <div className="col-span-12 md:col-span-8">
                <ImageCard
                  src="/images/a-phi.png"
                  alt="Collage ảnh đoàn kết Á - Phi"
                  // caption để trống vì caption dài đặt ở khối bên phải
                />
              </div>
              <div className="col-span-12 md:col-span-4">
                <CaptionCard>
                  <p className="mb-2">
                    <span className="italic">Ảnh trên:</span> Chủ tịch Hồ Chí Minh gặp và hội đàm với
                    Chủ tịch Mao Trạch Đông, tháng 2–1960 tại Bắc Kinh; Tổng thư ký Đảng Cộng sản Trung
                    Quốc từ ngày 2 đến ngày 4–11–1960. Ảnh: TTXVN
                  </p>
                  <p className="mb-2">
                    <span className="italic">Ảnh giữa:</span> Chủ tịch Hồ Chí Minh và Chủ tịch Mao Trạch
                    Đông năm 1957.
                  </p>
                  <p>
                    <span className="italic">Ảnh dưới:</span> Nhân dân Trung Quốc chào mừng đoàn đại biểu
                    chính phủ Việt Nam, có Chủ tịch Hồ Chí Minh dẫn đầu, thăm hữu nghị Trung Quốc, tháng 6–1955.
                    Ảnh tư liệu/TTXVN
                  </p>
                </CaptionCard>
              </div>
            </div>

            {/* HÀNG 2: Văn bản bên trái + Ảnh báo Nhân Dân bên phải */}
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12 md:col-span-4">
                <TextCard>
                  <p className="mb-3">
                    Đồng thời, Hồ Chí Minh chủ trương thực hiện đoàn kết với các dân tộc châu Á và châu
                    Phi trong phong trào đấu tranh giành độc lập dân tộc. Người chỉ rõ: các dân tộc châu Á
                    có độc lập thì hòa bình thế giới mới có thể thực hiện, và vận mệnh của các dân tộc châu Á
                    gắn bó mật thiết với vận mệnh của dân tộc Việt Nam. Đây là quan điểm mang tính chiến lược,
                    đặt cách mạng Việt Nam trong mối liên hệ rộng lớn với phong trào giải phóng dân tộc toàn cầu.
                  </p>
                  {/* <p className="text-zinc-400 text-[0.85vw]">
                    Nguồn chú thích: Bài viết của Bác Hồ trên báo Nhân dân số 422, 28/04/1955
                  </p> */}
                </TextCard>
              </div>
              <div className="col-span-12 md:col-span-8">
                <ImageCard
                  
                  src="/images/a-phi2.png"
                  alt="Báo Nhân Dân 1955"
                  caption="Bài viết của Bác Hồ trên báo Nhân dân số 422, 28/04/1955"
                />
              </div>
            </div>

            {/* Đoạn kết */}
            <p>
              Ngay từ rất sớm, Hồ Chí Minh đã có những hoạt động thực tiễn để đặt nền móng cho phong trào
              đoàn kết này. Năm 1923, Người tham gia sáng lập Hội Liên hiệp thuộc địa tại Pháp; đến tháng
              7/1925, Người tiếp tục tham gia Hội Liên hiệp các dân tộc bị áp bức tại Trung Quốc. Những
              hoạt động này góp phần quan trọng đặt cơ sở cho sự ra đời của Mặt trận nhân dân Á – Phi đoàn
              kết với Việt Nam, trở thành chỗ dựa quốc tế to lớn cho cách mạng Việt Nam.
            </p>
          </div>
        </div>
      )}
    </>
  );
}

/* ------------ UI helpers ------------- */

function TextCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-none p-5 ">
      <div className="text-zinc-200">{children}</div>
    </div>
  );
}

function CaptionCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-none p-5 text-zinc-300 text-[0.9vw] leading-relaxed">
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
          <img src={src} alt={alt || ""} className="w-full h-auto object-cover" />
        ) : (
          <div className="w-full aspect-[4/3] bg-zinc-700/60 grid place-items-center text-zinc-300">
            Thêm ảnh ở đây
          </div>
        )}
      </div>
      {caption ? (
        <figcaption className="text-[0.8vw] text-zinc-400 mt-2 leading-snug">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
