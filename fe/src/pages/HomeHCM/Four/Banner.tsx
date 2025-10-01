import React from "react";
import { GrGroup } from "react-icons/gr";
import { RiOrganizationChart } from "react-icons/ri";

export default function Banner() {
  return (
    <>
      <div className="w-full grid grid-cols-12 px-[4vw] py-[4vw] gap-[4vw]">
        <div className="col-span-7">
          <div className="unbounded text-[4vw] text-red-500 uppercase mt-[6vw]">
            Đoàn kết quốc tế trong thời kỳ hội nhập
          </div>
          <div className="inter italic mt-[2vw] text-[1.2vw]">
            Hồ Chí Minh từng khẳng định: “Cách mạng Việt Nam là một bộ phận
            khăng khít của cách mạng thế giới”. Người luôn đề cao sức mạnh đoàn
            kết quốc tế trên cơ sở hòa bình, công lý, hợp tác và tiến bộ. Ngày
            nay, sinh viên Việt Nam tham gia các chương trình trao đổi quốc tế,
            du học, làm việc trong các tập đoàn đa quốc gia, hay tham gia chiến
            dịch vì môi trường, vì nhân quyền toàn cầu. Nhưng đồng thời, cũng có
            những xung đột lợi ích, va chạm văn hóa, khác biệt trong hệ giá trị.
            Điều này đặt ra câu hỏi: đoàn kết quốc tế cần được hiểu và thực hành
            như thế nào để vừa gắn bó với nhân loại, vừa giữ vững bản sắc và lợi
            ích dân tộc?
          </div>
        </div>
        <div className="col-span-5">
          <img
            className="w-full aspect-[9/10] object-cover object-center rounded-xl"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
            alt=""
          />
        </div>
        <div className="col-span-6">
          <img
            className="w-full aspect-[9/6] object-cover object-center rounded-xl"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
            alt=""
          />
          <div className="text-red-500 text-[2.6vw] unbounded my-[4vw]">
            Đoàn kết trên cơ sở thống nhất mục tiêu và lợi ích; có lý, có tình
          </div>
          <img
            className="w-full aspect-[9/6] object-cover object-center rounded-xl"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
            alt=""
          />
        </div>
        <div className="col-span-6">
          <div className="text-red-500 text-[2.6vw] unbounded">
            Sức mạnh dân tộc chỉ bền vững khi gắn với sức mạnh thời đại và lợi
            ích chung của nhân loại
          </div>
          <img
            className="w-full aspect-[9/6] object-cover object-center rounded-xl my-[6vw]"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
            alt=""
          />
          <div className="text-red-500 text-[2.6vw] unbounded">
            Đoàn kết trên cơ sở độc lập, tự chủ
          </div>
        </div>
      </div>
    </>
  );
}
