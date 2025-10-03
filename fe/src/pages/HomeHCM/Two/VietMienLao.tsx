import React from "react";
import { BiX } from "react-icons/bi";

export default function VietMienLao() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
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
        <div
          className="text-red-500 cursor-pointer hover:scale-[1.1] duration-300 transition-all"
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
            Mặt trận đoàn kết Việt – Miên – Lào
          </div>
          <div className="mb-[1vw] w-[70vw] mx-auto">
            Mặt trận đoàn kết Việt – Miên – Lào được thành lập nhằm phối hợp và
            giúp đỡ lẫn nhau trong cuộc đấu tranh giải phóng dân tộc, cùng hướng
            tới thắng lợi.
          </div>
          <div className="w-[70vw] mx-auto mb-[2vw]">
            <div>
              Bước sang năm 1951, cuộc kháng chiến chống thực dân Pháp của nhân
              dân ba nước Đông Dương bước vào giai đoạn phát triển mới. Cục diện
              chiến tranh có nhiều chuyển biến sâu sắc, xuất hiện nhiều thuận
              lợi cơ bản nhưng cũng không ít khó khăn, phức tạp, đòi hỏi sự liên
              kết chặt chẽ giữa ba dân tộc Việt Nam, Lào và Campuchia.
            </div>
            <div className="flex gap-[2vw] my-[2vw]">
              <img
                className="aspect-[17/9] object-cover object-center rounded-xl"
                src="images/daihoi.jpg"
                alt=""
              />
              <div>
                Trong bối cảnh đó, Đại hội đại biểu lần thứ II của Đảng Cộng sản
                Đông Dương (11–19/02/1951) tại xã Vinh Quang (nay thuộc xã Kim
                Bình, huyện Chiêm Hóa, tỉnh Tuyên Quang) đã đưa ra nhiều quyết
                sách quan trọng. Tại Đại hội, Chủ tịch Hồ Chí Minh nhấn mạnh:
                “Chúng ta kháng chiến, dân tộc bạn Miên, Lào cũng kháng chiến.
                Bọn thực dân Pháp và bọn can thiệp Mỹ, là kẻ thù của ta và của
                dân tộc Miên, Lào. Vì vậy, ta phải ra sức giúp đỡ anh em Miên,
                Lào, giúp đỡ kháng chiến Miên, Lào. Và tiến hành thành lập Mặt
                trận thống nhất các dân tộc Việt – Miên – Lào.”
              </div>
            </div>
            <div className="flex gap-[2vw] my-[2vw]">
              <div>
                Tiếp đó, vào tháng 3/1951, Hội nghị liên minh ba nước Đông Dương
                được tổ chức tại xã Vinh Quang với sự tham dự của đại diện Mặt
                trận Liên Việt (Việt Nam), Mặt trận Lào Ítxala và Mặt trận
                Khơ-me Ítxarắc (Campuchia). Hội nghị đã thống nhất thành lập
                khối liên minh nhân dân Việt – Miên – Lào trên nguyên tắc tự
                nguyện, bình đẳng, tôn trọng chủ quyền và tương trợ lẫn nhau.
                Hội nghị cũng kêu gọi nhân dân ba nước đoàn kết chặt chẽ trong
                từng mặt trận của mình (Liên Việt ở Việt Nam, Ítxala ở Lào,
                Ítxarắc ở Campuchia), củng cố khối liên minh ngày càng vững
                chắc, đẩy mạnh cuộc kháng chiến mau chóng giành thắng lợi, đồng
                thời củng cố và phát triển chính quyền dân tộc, chính quyền nhân
                dân ở cả ba nước.
              </div>
              <img
                className="aspect-[14/9] object-cover object-center rounded-xl"
                src="images/viet-mien-lao.png"
                alt=""
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
