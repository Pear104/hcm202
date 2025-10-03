import React, { useState } from "react";
import { BiX } from "react-icons/bi";
import { FaX } from "react-icons/fa6";

export default function DanToc() {
  const [isOpen, setIsOpen] = useState(false);
  const itemsData = [
    {
      date: "19/11/1930",
      content:
        "Ban Thường vụ Trung ương Đảng Cộng sản Đông Dương ra Chỉ thị thành lập Hội Phản đế Đồng minh, hình thức đầu tiên của Mặt trận Dân tộc Thống nhất Việt Nam dưới sự lãnh đạo của Đảng",
    },
    {
      date: "03/1935 - 10/1936",
      content:
        "Đảng ra Nghị quyết thành lập các Hội Phản đế Liên minh, lôi kéo rộng rãi các tổ chức cá nhân có tính chất phản đế phải liên kết cuộc vận động Phản đế Liên minh với các cuộc đầu  tranh đòi quyền lợi hằng ngày",
    },
    {
      date: "10/1936",
      content:
        "Đảng chủ trương thành lập Mặt trận Thống nhất Nhân dân Phản đế Đông Dương, tập hợp các lực lượng toàn Đông Dương vào cuộc đấu tranh chống đế quốc",
    },
    {
      date: "06/1938",
      content:
        "Đảng Cộng sản Đông Dương gửi thư công khai cho các đảng phái đề nghị gác các ý kiến bất đồng để \"bước tới thành lập Mặt trận Dân chủ Đông Dương\"",
    },
    {
      date: "11/1939",
      content:
        "Đảng Cộng sản Đông Dương đã kịp thời chuyển hướng chỉ đạo, chuyển cuộc vận động Mặt trận Dân chủ thành Mặt trận Dân tộc thống nhất chống chiến tranh đế quốc với tên gọi chính thức: Mặt trận Dân tộc thống nhất phản đế Đông Dương",
    },
    {
      date: "05/1941",
      content:
        "Thành lập Mặt trận Việt Nam Độc lập Đồng minh (Mặt trận Việt Minh) nhằm tổ chức lực lượng đấu tranh để thực hiện mục đích cứu nước",
    },
    {
      date: "05/1946",
      content:
        "Giữa lúc nước Việt nam dân chủ cộng hoà vừa ra đời phải đương đầu với nhiều khó khăn lớn, một Ban vận động thành lập Hội liên hiệp quốc dân Việt Nam nhằm mở rộng hơn nữa khối đoàn kết dân tộc",
    },
    {
      date: "03/1951",
      content:
        "Với các chủ trương đường lối đúng đắn Đảng Lao động Việt Nam và sự ủng hộ tích cực của các Đảng Xã hội, Đảng Dân chủ, các tổ chức chính trị, các nhân sĩ trí thức trong Mặt trận Việt Minh và Hội Liên Việt, hai tổ chức Mặt trận được hợp nhất thành Mặt trận Liên Việt",
    },
    {
      date: "09/1955",
      content:
        "Mặt trận Tổ quốc Việt nam ra đời với mục đích đoàn kết mọi lực lượng dân tộc và dân chủ, đấu tranh đánh bại đế quốc Mỹ xâm lược và tay sai, xây dựng một nước Việt nam hoà bình thống nhất, độc lập, dân chủ và giàu mạnh.",
    },
    {
      date: "12/1960",
      content:
        "Trong cao trào đồng khởi của đồng bào miền Nam, Mặt trận Dân tộc Giải phóng miền Nam ra đời nhằm đoàn kết toàn dân đánh bại chiến tranh xâm lược của đế quốc Mỹ, đánh đổ nguỵ quyền tay sai của chúng, giải phóng miền Nam, tiến tới thống nhất Tổ quốc",
    },
    {
      date: "04/1968",
      content:
        "Trong cao trào tiến công và nổi dậy đầu xuân Mậu Thân (1968) Liên minh các lực lượng Dân tộc, Dân chủ và Hoà bình Việt nam ra đời đã góp sức động viên xúc tiến các phong trào đấu tranh yêu nước, tăng thêm sức mạnh cho khối đoàn kết toàn dân, chống Mỹ cứu nước",
    },
    {
      date: "04/02/1977",
      content:
        "Nhằm đáp ứng yêu cầu của giai đoạn cách mạng mới, Đại hội Mặt trận Dân tộc thống nhất họp tại thành phố Hồ Chí Minh đã thống nhất ba tổ chức Mặt trận ở hai miền Nam Bắc nước ta thành một tổ chức Mặt trận Dân tộc thống nhất duy nhất lấy tên là Mặt trận Tổ quốc Việt Nam",
    }

  ];
  return (
    <>
      <div className="flex flex-col items-center">
        <div className="font-bold text-[1.4vw] text-center">
          Mặt trận đại đoàn kết
          <br />
          dân tộc
        </div>
        <div className="my-[2vw] text-center">
          Hồ Chí Minh chủ trương xây dựng mặt trận thống nhất để khơi dậy sức
          mạnh toàn dân trong đấu tranh chống đế quốc.
        </div>
        <div
          className="text-red-500"
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
            Mặt trận đại đoàn kết dân tộc
          </div>
          <div className="mb-[4vw] w-[70vw] mx-auto text-2xl">
            Mặt trận đại đoàn kết dân tộc nhằm khơi dậy sức mạnh và quyền tự
            quyết của mỗi dân tộc trong sự nghiệp đấu tranh cách mạng. Ngay từ
            năm 1924, Hồ Chí Minh đã đưa ra quan điểm về việc thành lập “Mặt
            trận thống nhất của nhân dân chính quốc và thuộc địa” để chống chủ
            nghĩa đế quốc, đồng thời kiến nghị Quốc tế Cộng sản cần có những
            giải pháp cụ thể nhằm biến quan điểm này thành hiện thực. Từ ngày
            18/11/1930 đến nay, tổ chức Mặt trận Dân tộc thống nhất ở Việt Nam
            đã nhiều lần thay đổi tên gọi để phù hợp với từng giai đoạn cách
            mạng:
          </div>
          <div className="grid grid-cols-3 gap-x-[2vw] gap-y-[2vw] w-[70vw] mx-auto mb-[2vw]">
            {itemsData.map((item, index) => (
              <Item key={index} date={item.date} content={item.content} />
            ))}
          </div>
        </div>
      )}
    </>
  );
}

const Item = ({ date, content }: { date: string; content: string }) => {
  return (
    <div>
      <div className="bg-[#952d2d] text-white rounded-r-2xl px-[1vw] py-[0.4vw] mb-[1vw] text-[1vw] unbounded">
        {date}
      </div>
      <div>{content}</div>
    </div>
  );
};