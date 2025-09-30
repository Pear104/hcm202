import React, { useState, useEffect } from "react";

export default function SectionThree() {
  const [index, setIndex] = useState(0);
  const [isFading, setIsFading] = useState(true);

  const items = [
    {
      title: "Dân chủ đại diện",
      description:
        "Dân chủ đại diện là hình thức chính phủ trong đó nhân dân bầu ra những người đại diện để thay mặt mình quyết định và quản lý. Nền tảng của dân chủ đại diện là bầu cử, thông qua các cơ quan như Quốc hội và Hội đồng nhân dân các cấp.",
      image: "/images/trung-cau-y-dan.jpg",
    },
    {
      title: "Dân chủ trực tiếp",
      description:
        "Dân chủ trực tiếp là việc nhân dân tự mình thể hiện ý chí và tham gia quyết định các vấn đề của Nhà nước, không qua trung gian đại diện. Hình thức phổ biến gồm: ứng cử, bầu cử, trưng cầu dân ý, thực hiện quy chế dân chủ ở cơ sở và đối thoại trực tiếp với cơ quan nhà nước.",
      image: "/images/a-muong.jpg",
    },
    {
      title: "Dân chủ ở cơ sở",
      description:
        "Dân chủ ở cơ sở được thực hiện theo phương châm “Dân biết, dân bàn, dân làm, dân kiểm tra”. Luật Thực hiện dân chủ ở cơ sở năm 2022 (có hiệu lực từ 01/7/2023, thay thế Pháp lệnh 34/2007) đã mở rộng phạm vi áp dụng, không chỉ trong xã, phường mà còn ở các đơn vị sự nghiệp và nơi làm việc. Luật cũng bổ sung nguyên tắc quan trọng là tôn trọng ý kiến đóng góp của nhân dân và kịp thời giải quyết kiến nghị, phản ánh, nhằm bảo đảm quyền làm chủ thực chất của người dân.",
      image: "/images/dan-chu-co-so.jpg",
    },
  ];

  const handleMouseEnter = (newIndex) => {
    if (index !== newIndex) {
      setIsFading(false); // Start fade-out
      setTimeout(() => {
        setIndex(newIndex);
        setIsFading(true); // Start fade-in
      }, 300); // Wait for the transition to finish
    }
  };

  return (
    <>
      <div className="w-screen h-[100vh] relative group p-[4vw]">
        <div className="font-semibold text-[8vh]">
          Hình thức thực hiện{" "}
          <span className="text-yellow-400 uppercase font-bold text-[14vh]">
            Dân chủ
          </span>{" "}
          hiện nay
        </div>
        <div className="flex gap-[4vh] mt-[4vh]">
          <div className="flex flex-col gap-[4vh] mt-[4vh] text-[3vh] col-span-3 w-[14vw] font-semibold">
            {items.map((item, i) => (
              <div
                key={i}
                onMouseEnter={() => handleMouseEnter(i)}
                className={`relative cursor-pointer pb-2 transition-all duration-300 ${
                  index === i ? "text-yellow-400" : ""
                }`}
              >
                {item.title}
                <div
                  className={
                    "h-[0.3vw] absolute bottom-0 left-0 bg-yellow-400 transition-all duration-300 " +
                    (index === i ? "w-full" : "w-0")
                  }
                ></div>
              </div>
            ))}
          </div>

          <div className="w-[38vw]">
            <img
              className={`transition-all duration-400 rounded-xl shadow-lg aspect-[12/9] object-cover object-center w-full ${
                isFading ? "opacity-100" : "opacity-0"
              }`}
              src={items[index].image}
              loading="eager"
              alt={items[index].title}
            />
          </div>

          <div className="w-[38vw]">
            <div
              className={`uppercase text-[8vh] text-nowrap transition-all duration-400 ${
                isFading ? "opacity-100" : "opacity-0"
              }`}
            >
              {items[index].title}
            </div>
            <div
              className={`text-[3vh] mt-[6vh] transition-all duration-400 ${
                isFading ? "opacity-100" : "opacity-0"
              }`}
            >
              {items[index].description}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
