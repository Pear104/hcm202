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
      image: "/images/giao_trinh.jpg",
    },
    {
      type: "Website",
      title: "hochiminh.vn",
      time: "2022",
      author: "TS Minh Dương",
      link: "https://hochiminh.vn/tu-tuong-dao-duc-ho-chi-minh/nghien-cuu-tu-tuong-dao-duc-ho-chi-minh/van-dung-tu-tuong-ho-chi-minh-ve-mat-tran-dan-toc-thong-nhat-trong-xay-dung-khoi-dai-doan-ket-dan-toc-hien-nay-7775",
      image: "/images/web1.png",
    },
    {
      type: "Website",
      title: "tuyenquang.dcs.vn",
      time: "2024",
      author: "--",
      link: "https://tuyenquang.dcs.vn/DetailView/148090/40/Tuyen-Quang---Noi-ghi-dau-tinh-doan-ket-Viet-Nam---Lao---Campuchia.html",
      image: "/images/web2.png",
    },

    {
      type: "Website",
      title: "hochiminh.nhandan.vn",
      time: "2023",
      author: "Lê Quốc Minh",
      link: "https://hochiminh.nhandan.vn/mat-tran-thong-nhat-a-phi-977.html",
      image: "/images/web3.png",
    },
    {
      type: "Website",
      title: "ct.qdnd.vn",
      time: "2015",
      author: "Ngô Văn Lương",
      link: "https://ct.qdnd.vn/quoc-te/phong-trao-phan-chien-cua-binh-linh-phap-trong-chien-tranh-o-viet-nam-517527",
      image: "/images/web4.png",
    },
    {
      type: "Website",
      title: "nhandan.vn",
      time: "2025",
      author: "Trần Anh Tuấn",
      link: "https://nhandan.vn/phong-trao-phan-chien-chong-chien-tranh-xam-luoc-viet-nam-cua-nhan-dan-my-post866549.html",
      image: "/images/web6.png",
    },
    {
      type: "Website",
      title: "tapchicongsan.org.vn",
      time: "2023",
      author: "Đỗ Ngọc Hanh",
      link: "https://www.tapchicongsan.org.vn/web/guest/quoc-phong-an-ninh-oi-ngoai1/-/2018/827273/van-dung-tu-tuong-ho-chi-minh-ve-doan-ket%2C-hop-tac-quoc-te-trong-duong-loi-doi-ngoai-cua-viet-nam-hien-nay.aspx",
      image: "/images/web7.png",
    },
    {
      type: "Website",
      title: "baocaovien.vn",
      time: "2025",
      author: "--",
      link: "https://hcmiu.edu.vn/sinh-vien-le-nguyen-bao-ngoc-tham-gia-cop29-hoi-nghi-thuong-dinh-ve-bien-doi-khi-hau-cua-lien-hop-quoc-nam-2024/",
      image: "/images/web8.png",
    },
  ];

  return (
    <>
      <div
        id="scroll-section"
        className="w-screen h-[140vw] relative group pt-[4vw]"
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
            className="flex gap-[3vw] w-[240vw] mx-[4vw] overflow-x-scroll mt-[2vw]"
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
    <div className="horizontal-scroll flex flex-col justify-center w-[18vw] gap-[0.5vw] inter">
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
        <a
          href={item.link}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[0.8vw] font-semibold hover:font-bold text-red-500 px-[1.4vw] py-[0.5vw] border hover:border-2 border-red-500 rounded-2xl cursor-pointer inline-block"
        >
          Truy cập
        </a>
      </div>
    </div>
  );
};
