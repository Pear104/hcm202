import { div } from "motion/react-client";
import React from "react";
import DanToc from "./DanToc";
import VietMienLao from "./VietMienLao";
import APhi from "./APhi";
import TheGioi from "./TheGioi";
import { ToggleCard } from "../One/ASection";
import { NumberedToggleCard } from "./numberToggleCard";

export default function ASection() {
  return (
    <>
      <div className="px-[4vw]">
        <div className="grid grid-cols-2 gap-x-[6vw] gap-y-[1vw]">
          <div>
            <img
              className="w-full aspect-[12/9] object-cover object-center rounded-xl"
              src="images/2_A.png"
              alt=""
            />
          </div>
          <div className="">
            <div className="unbounded text-[4vw] text-[#FF2F2F] capitalize leading-[4.4vw] pl-[1.5vw] border-l-[1vw]">
              A
            </div>
            <div className="unbounded text-[4vw] text-red-500/90 uppercase my-[1.6vw]">
              Các lực lượng cần đoàn kết
            </div>
            <div className="inter italic text-[1.2vw]">
              Lực lượng đoàn kết quốc tế trong tư tưởng Hồ Chí Minh bao gồm:
              phong trào cộng sản và công nhân quốc tế; phong trào đấu tranh
              giải phóng dân tộc và phong trào hoà bình, dân chủ thế giới, trước
              hết là phong trào chống chiến tranh của nhân dân các nước đang xâm
              lược Việt Nam.
            </div>
          </div>
          <div className="col-span-2 flex flex-col items-center">
            <div className="space-y-[1vw] mt-[2vw]">
              {/* Toggle giữ nguyên CSS gốc từng khối */}
              <NumberedToggleCard
                number={1}

                title="Phong trào cộng sản và công nhân thế giới"
                defaultOpen={true}
              >
                Sự đoàn kết giữa giai cấp công nhân quốc tế là một bảo đảm vững
                chắc cho chủ nghĩa cộng sản. Hồ Chí Minh cho rằng chủ nghĩa tư
                bản là một lực lượng phản động quốc tế, là kẻ thù chung của nhân
                dân lao động toàn thế giới. Trong hoàn cảnh đó, chỉ có sức mạnh
                của sự đoàn kết, nhất trí, sự đồng tình và ủng hộ lẫn nhau của
                giai cấp lao động toàn thế giới, theo tinh thần “bốn phương vô
                sản đều là anh em”, mới có thể chống lại được những âm mưu thâm
                độc của chủ nghĩa đế quốc thực dân.
              </NumberedToggleCard>

              <NumberedToggleCard
                number={2}
                title="Phong trào đấu tranh giải phóng dân tộc"
                defaultOpen={false /* mở sẵn như block đỏ ban đầu */}
              >
                Hồ Chí Minh đặc biệt lưu ý Quốc tế Cộng sản về những biện pháp
                nhằm “làm cho các dân tộc thuộc địa, từ trước đến nay vẫn cách
                biệt nhau, hiểu biết nhau hơn và đoàn kết lại để đặt cơ sở cho
                một liên minh phương Đông tương lai”. Người nhấn mạnh rằng, khối
                liên minh này sẽ là một trong những cánh quan trọng của cách
                mạng vô sản. Đồng thời, Người còn đề nghị Quốc tế Cộng sản bằng
                mọi cách phải “làm cho đội quân tiên phong của lao động thuộc
                địa tiếp xúc mật thiết với giai cấp vô sản phương Tây để dọn
                đường cho một sự hợp tác thật sự sau này”. Theo Hồ Chí Minh, chỉ
                có sự hợp tác này mới bảo đảm cho giai cấp công nhân quốc tế
                giành thắng lợi cuối cùng.
              </NumberedToggleCard>
              <NumberedToggleCard
                number={3}
                title="Các lực lượng tiến bộ, yêu chuộng hòa bình, dân chủ, tự do và công lý"
                defaultOpen={false /* mở sẵn như block đỏ ban đầu */}
              >
                Hồ Chí Minh luôn gắn cuộc đấu tranh vì độc lập ở Việt Nam với
                mục tiêu bảo vệ hòa bình, tự do, công lý và bình đẳng, từ đó tập
                hợp và tranh thủ sự ủng hộ của các lực lượng tiến bộ trên thế
                giới. Người nhiều lần khẳng định: Chính nhờ biết kết hợp phong
                trào cách mạng trong nước với phong trào cách mạng của giai cấp
                công nhân quốc tế và của các dân tộc bị áp bức, Đảng ta đã vượt
                qua được mọi khó khăn, thử thách, đưa giai cấp công nhân và nhân
                dân ta đến những thắng lợi vẻ vang.
              </NumberedToggleCard>
            </div>

            <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize py-[2vw]">
              Đại đoàn kết dân tộc gắn liền với
              <br />
              đoàn kết quốc tế
            </div>
            <div className="w-[64vw] text-[1.4vw] text-center">
              Theo Hồ Chí Minh, đại đoàn kết toàn dân tộc phải gắn liền với đoàn
              kết quốc tế. Đại đoàn kết dân tộc chính là cơ sở, là tiền đề vững
              chắc để triển khai thành công đoàn kết quốc tế. Chỉ khi sức mạnh
              dân tộc kết hợp hài hòa với sức mạnh thời đại thì mới có thể tạo
              nên sức mạnh tổng hợp, bảo đảm cho thắng lợi của cách mạng Việt
              Nam.
            </div>
          </div>
        </div>
        <div className="mt-[4vw] flex flex-col items-center">
          <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize">
            Hình thức tổ chức
          </div>
          <div className="w-[64vw] space-y-[1vw]">
            <div className="h-fit text-center text-[1.4vw] my-[1vw]">
              Đoàn kết quốc tế trong tư tưởng Hồ Chí Minh không phải là một vấn
              đề sách lược hay thủ đoạn chính trị nhất thời, mà là vấn đề có
              tính nguyên tắc, là một đòi hỏi khách quan của cách mạng Việt Nam.
              Người luôn khẳng định, đoàn kết quốc tế vừa là nhu cầu tất yếu,
              vừa là điều kiện để cách mạng Việt Nam gắn bó chặt chẽ với phong
              trào cách mạng thế giới.
            </div>
            <div className="grid grid-cols-3 gap-x-[6vw] my-[2vw] gap-y-[2vw]">
              <div className="border border-dashed border-red-500"></div>
              <div className="border border-dashed border-red-500"></div>
              <div className="border border-dashed border-red-500"></div>
            </div>
          </div>
        </div>
        <div className="w-full grid grid-cols-4 gap-[4vw] mt-[2vw] mb-[4vw]">
          <DanToc />
          <VietMienLao />
          <APhi />
          <TheGioi />
        </div>
      </div>
    </>
  );
}

// const Item = () => {
//   return (
//     <div className="flex flex-col items-center">
//       <div className="font-bold text-[1.4vw] text-center">
//         Mặt trận đại đoàn kết dân tộc
//       </div>
//       <div className="my-[2vw] text-center">
//         Hồ Chí Minh chủ trương xây dựng mặt trận thống nhất để khơi dậy sức mạnh
//         toàn dân trong đấu tranh chống đế quốc.
//       </div>
//       <div className="text-red-500">Xem thêm </div>
//     </div>
//   );
// };
