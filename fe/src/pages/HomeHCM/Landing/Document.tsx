import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/all";
import React from "react";

export default function Document() {
  useGSAP(() => {
    ScrollTrigger.create({
      animation: gsap.to("#scroll-wrapper", {
        x: "-30.5%",
        ease: "power1.inOut",
      }),
      trigger: "#scroll-section",
      start: "top top",
      end: "bottom bottom",
      scrub: true,
      anticipatePin: 1,
      // markers: true,
    });
  }, []);

  const items = [
    {
      title: "Quyền làm chủ của nhân dân ngày càng được mở rộng",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      title: "Nhân dân tham gia trực tiếp vào đời sống chính trị",
      description:
        "Người dân ngày càng có nhiều cơ hội đóng góp ý kiến vào các dự thảo chính sách, pháp luật, cũng như thực hiện quyền bầu cử Quốc hội và Hội đồng nhân dân các cấp. Hàng chục nghìn hội nghị đối thoại giữa chính quyền và nhân dân được tổ chức, tạo sự đồng thuận và gắn kết giữa Đảng, Nhà nước và nhân dân.",
      image: "/images/tham-gia.jpg",
    },
    {
      title: "Khung pháp lý về dân chủ được hoàn thiện",
      description:
        "Nhiều đạo luật và nghị định quan trọng đã được ban hành nhằm phát huy quyền làm chủ của nhân dân, như Luật Mặt trận Tổ quốc Việt Nam (2015), Luật Trưng cầu ý dân (2015), Luật Thực hiện dân chủ ở cơ sở (2022). Đây là cơ sở pháp lý để nhân dân tham gia quản lý xã hội và giám sát hoạt động của Nhà nước.",
      image: "/images/phap-ly.jpg",
    },
    {
      title: "Hệ thống chính trị đổi mới phương thức hoạt động",
      description:
        "Hoạt động của các cơ quan quyền lực được tăng cường hiệu lực và hiệu quả. Nghị quyết số 27-NQ/TW (2022) khẳng định quyền con người, quyền công dân đã được cụ thể hóa bằng pháp luật và thực hiện tốt hơn trên thực tế; đồng thời, cả dân chủ trực tiếp và dân chủ đại diện đều được phát huy.",
      image: "/images/doi-moi.jpg",
    },
  ];

  return (
    <>
      <div
        id="scroll-section"
        className="w-screen h-[240vh] relative group py-[4vw]"
      >
        <div
          id="scroll-title"
          className="sticky top-[8vw] font-semibold text-[6vh] uppercase overflow-x-scroll w-screen"
        >
          <div className="px-[4vw] unbounded text-red-500">
            Tài liệu tham khảo
          </div>
          <div
            id="scroll-wrapper"
            className="flex gap-[4vw] w-[136vw] mx-[4vw] overflow-x-scroll mt-[2vw]"
          >
            {items.map((item, i) => (
              <SlideItem key={i} item={item} />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

const SlideItem = ({ item }: { item: any }) => {
  return (
    <div className="horizontal-scroll flex flex-col items-center justify-center w-[20vw] gap-[2vh]">
      <img
        className="transition-all duration-400 opacity-100 rounded-xl shadow-lg aspect-[9/12] object-cover object-center w-full"
        src={item.image}
        loading="eager"
        alt=""
      />
      <div className="text-[1.1vw] text-red-400 font-bold text-ellipsis text-wrap">
        {item.title}
      </div>
      <div className="text-[1.1vw] text-yellow-400 text-nowrap font-bold">
        {item.title}
      </div>
    </div>
  );
};
