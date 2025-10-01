import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/all";
import React from "react";
import { FaUser } from "react-icons/fa";

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
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
    {
      type: "Giáo trình",
      title: "Tư tưởng Hồ Chí Minh",
      time: "2021",
      author: "NXBCTQGST",
      link: "https://www.google.com/",
      description:
        "Gần 40 năm đổi mới cho thấy quyền làm chủ của nhân dân được bảo đảm trên mọi lĩnh vực kinh tế, chính trị, văn hóa – xã hội. Dân chủ không chỉ thể hiện trong Hiến pháp, pháp luật mà còn hiện diện sinh động trong đời sống xã hội, góp phần củng cố sức mạnh đại đoàn kết toàn dân tộc.",
      image: "/images/mo-rong.jpg",
    },
  ];

  return (
    <>
      <div
        id="scroll-section"
        className="w-screen h-[140vw] relative group py-[4vw]"
      >
        <div
          id="scroll-title"
          className="sticky top-[8vw] text-[6vh] overflow-x-scroll w-screen"
        >
          <div className="px-[4vw] unbounded text-red-500 font-semibold">
            Tài liệu tham khảo
          </div>
          <div
            id="scroll-wrapper"
            className="flex gap-[4vw] w-[210vw] mx-[4vw] overflow-x-scroll mt-[2vw]"
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
    <div className="horizontal-scroll flex flex-col justify-center w-[14vw] gap-[0.5vw] inter">
      <img
        className="transition-all duration-400 opacity-100 rounded-xl shadow-lg aspect-[9/11] object-cover object-center w-full"
        src={item.image}
        loading="eager"
        alt=""
      />
      <div className="w-full text-[1.1vw] text-red-500 text-ellipsis text-wrap mt-[0.4vw]">
        {item.type}
      </div>
      <div className="w-full font-semibold text-[1.1vw] text-white text-ellipsis text-wrap">
        {item.title}
      </div>
      <div className="w-full font-semibold text-[0.8vw] text-zinc-400 text-ellipsis text-wrap flex gap-4 items-center">
        <div>{item.time}</div>|
        <div className="flex gap-2 items-center">
          <FaUser />
          {item.author}
        </div>
      </div>
      <div className="-translate-y-[1.4vw]">
        <span className="text-[0.8vw] font-semibold hover:font-bold text-red-500 px-[1.4vw] py-[0.5vw] border hover:border-2 border-red-500 rounded-2xl cursor-pointer">
          Truy cập
        </span>
      </div>
    </div>
  );
};
