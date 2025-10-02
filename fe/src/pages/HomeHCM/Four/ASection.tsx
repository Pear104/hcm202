import React from "react";

export default function ASection() {
  return (
    <>
      <div className="px-[4vw] pb-[3vw]">
        <div className="gap-[6vw]">
          <div className="unbounded text-[4vw] text-red-500/90 uppercase my-[1vw]">
            Sinh viên trong thời kỳ hộp nhập
          </div>
          <div className="inter mt-[2vw] flex flex-col gap-x-[3vw] text-[1.4vw]">
            <div className="grid grid-cols-2 gap-x-[4vw] pb-[2vw]">
              <div className="col-span-2 text-[1.4vw]">
                Với sinh viên – lực lượng năng động nhất trong giao lưu quốc tế
                – điều đó có nghĩa là: khi tham gia trao đổi học thuật, du học,
                làm việc trong các tập đoàn đa quốc gia hay các chiến dịch toàn
                cầu, họ vừa là người học hỏi tri thức, vừa là “sứ giả văn hóa”,
                mang bản sắc Việt Nam để tạo đồng cảm và xây dựng cầu nối hiểu
                biết giữa các dân tộc.
              </div>
              <div className="mt-[2vw]">
                <img
                  className="w-full aspect-[14/9] object-cover object-center rounded-xl"
                  src="images/4_4.png"
                  alt=""
                />
                <div className="mt-[2vw] text-[1.4vw]">
                  Học sinh FPT đạt giải nhất cuộc thi quốc tế Enjoy AI 2024
                </div>
              </div>
              <div className="mt-[2vw]">
                <div className="mb-[2vw] text-[1.4vw]">
                  Học sinh Việt Nam đạt HCV tại Cuộc thi Olympic Phát minh và
                  Sáng chế thế giới
                </div>
                <img
                  className="w-full aspect-[15/9] object-cover object-center rounded-xl"
                  src="images/4_6.png"
                  alt=""
                />
              </div>
              <img
                className="col-span-2 w-full aspect-[27/9] object-cover object-center rounded-xl mt-[4vw]"
                src="images/4_7.png"
                alt=""
              />
            </div>
            <div className="">
              <div className="mb-[2vw]">
                Hoa hậu Bảo Ngọc với tổ chức Gen Zero đã phát biểu tại ACYCS
                2025 và COP29, đưa tiếng nói thanh niên Việt Nam ra toàn cầu,
                thể hiện tinh thần gắn lợi ích dân tộc với lợi ích chung; từ đó
                khẳng định kim chỉ nam cho sinh viên Việt Nam là hội nhập có
                chọn lọc, gắn bó với nhân loại nhưng luôn giữ vững độc lập và
                bản sắc.
              </div>
              <div>
                Đoàn kết quốc tế theo tư tưởng Hồ Chí Minh, khi vận dụng cho
                sinh viên hôm nay, chính là năng lực hội nhập có chọn lọc: học
                hỏi và đóng góp cho nhân loại nhưng không hòa tan; giữ gìn bản
                sắc và độc lập tự chủ nhưng không cô lập. Đây là kim chỉ nam để
                thế hệ trẻ Việt Nam vừa gắn bó với nhân loại, vừa làm tròn trách
                nhiệm xây dựng, bảo vệ và phát triển đất nước trong kỷ nguyên
                toàn cầu hóa.
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
