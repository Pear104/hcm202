import React from "react";
import { FaCheck } from "react-icons/fa";

export default function BSection() {
  return (
    <>
      <>
        <div className="p-[2vw]">
          <div className="unbounded text-[3vw] text-[#FF2F2F] capitalize leading-[3.4vw] pl-[1.5vw] border-l-[1vw]">
            B
          </div>
          <div className="unbounded text-[4vw] text-red-500/90 capitalize mt-[1vw]">
            Đoàn kết quốc tế <br/> và sự nghiệp chung của nhân loại
          </div>
          <div className="w-full border border-dashed translate-y-[6vw]"></div>
          <div className="grid grid-cols-3 gap-x-[6vw] mt-[4vw] gap-y-[2vw]">
            <div className="relative unbounded w-[4vw] aspect-square bg-red-500 rounded-full flex items-center justify-center text-[2vw]">
              1
              <div className="right-[-1vw] top-[0.1vw] absolute unbounded w-[2vw] h-[2vw] text-black bg-white border border-black rounded-full flex items-center justify-center">
                <FaCheck size={14} />
              </div>
            </div>
            <div className="relative unbounded w-[4vw] aspect-square bg-red-500 rounded-full flex items-center justify-center text-[2vw]">
              2
              <div className="right-[-1vw] top-[0.1vw] absolute unbounded w-[2vw] h-[2vw] text-black bg-white border border-black rounded-full flex items-center justify-center">
                <FaCheck size={14} />
              </div>
            </div>
            <div className="relative unbounded w-[4vw] aspect-square bg-red-500 rounded-full flex items-center justify-center text-[2vw]">
              3
              <div className="right-[-1vw] top-[0.1vw] absolute unbounded w-[2vw] h-[2vw] text-black bg-white border border-black rounded-full flex items-center justify-center">
                <FaCheck size={14} />
              </div>
            </div>
            <div className="bg-zinc-700 rounded-xl p-[2vw]">
              <div className="text-red-500 text-[2vw] font-bold unbounded">
                1.
              </div>
              <div className="unbounded text-[1.4vw] font-bold mb-[1vw]">
                Đoàn kết quốc tế vì mục tiêu chung của nhân loại
              </div>
              <div>
                Thực hiện đoàn kết quốc tế không chỉ vì thắng lợi của cách mạng
                mỗi nước, mà còn vì sự nghiệp chung của nhân loại tiến bộ trong
                cuộc đấu tranh chống chủ nghĩa đế quốc và các thế lực phản động.
                Thời đại Hồ Chí Minh sống và hoạt động chính trị đã chấm dứt sự
                biệt lập của các quốc gia, mở ra quan hệ quốc tế sâu rộng, khiến
                vận mệnh của mỗi dân tộc gắn liền với vận mệnh chung của toàn
                nhân loại.
              </div>
            </div>
            <div className="bg-zinc-700 rounded-xl p-[2vw]">
              <div className="text-red-500 text-[2vw] font-bold unbounded">
                2.
              </div>
              <div className="unbounded text-[1.4vw] font-bold mb-[1vw]">
                Kết nối cách mạng Việt Nam với mục tiêu cao cả của thời đại
              </div>
              <div>
                Hồ Chí Minh kiên trì đấu tranh, không ngừng củng cố và tăng
                cường đoàn kết giữa các lực lượng cách mạng thế giới vì hòa
                bình, độc lập dân tộc, dân chủ và tiến bộ xã hội. Người luôn gắn
                cách mạng Việt Nam với mục tiêu chung cao cả của nhân loại. Nhân
                dân Việt Nam không chỉ chiến đấu cho độc lập, tự do của dân tộc
                mình mà còn vì độc lập, tự do của các dân tộc khác, không chỉ
                bảo vệ lợi ích quốc gia mà còn cùng nhân loại hướng tới hòa
                bình, dân chủ và chủ nghĩa xã hội.
              </div>
            </div>
            <div className="bg-zinc-700 rounded-xl p-[2vw]">
              <div className="text-red-500 text-[2vw] font-bold unbounded">
                3.
              </div>
              <div className="unbounded text-[1.4vw] font-bold mb-[1vw]">
                Giáo dục chủ nghĩa yêu nước gắn với quốc tế vô sản
              </div>
              <div>
                Để thực hiện thắng lợi sự nghiệp cách mạng, cần kết hợp giáo dục
                chủ nghĩa yêu nước chân chính với chủ nghĩa quốc tế vô sản cho
                nhân dân, đồng thời đấu tranh chống chủ nghĩa sô vanh, cơ hội,
                vị kỷ dân tộc. Đây là cơ sở tư tưởng để xây dựng sự gắn kết giữa
                lợi ích dân tộc với lợi ích nhân loại.
              </div>
            </div>
          </div>
          <div className="mt-[4vw] flex flex-col items-center">
            <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize">
              Thắng lợi của tư tưởng Hồ Chí Minh
            </div>
            <div className="w-[64vw] space-y-[1vw] mt-[2vw] text-center text-[1.6vw]">
              Thắng lợi của cách mạng Việt Nam trong nhiều thập kỷ qua chính là
              minh chứng sinh động cho tư tưởng Hồ Chí Minh: độc lập dân tộc
              phải luôn gắn liền với chủ nghĩa xã hội. Đây vừa là đường lối
              chiến lược, vừa là kết tinh của sự kết hợp giữa sức mạnh dân tộc
              với sức mạnh thời đại, giữa chủ nghĩa yêu nước và chủ nghĩa quốc
              tế vô sản.
            </div>
          </div>
        </div>
      </>
    </>
  );
}
