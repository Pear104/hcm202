import { useGSAP } from "@gsap/react";
import { Float, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { ScrollTrigger } from "gsap/all";
import React, { useState } from "react";

export default function HoChiMinh() {
  const [index, setIndex] = useState(0);
  const [isFading, setIsFading] = useState(true);
  useGSAP(() => {
    // const tl = gsap
    //   .timeline()
    //   .fromTo(
    //     "#demos",
    //     {
    //       x: "-100%",
    //       opacity: 0,
    //     },
    //     {
    //       x: "0%",
    //       opacity: 1,
    //       duration: 1,
    //     }
    //   )
    //   .fromTo(
    //     "#kratos",
    //     {
    //       y: "100%",
    //       opacity: 0,
    //     },
    //     {
    //       y: "0%",
    //       opacity: 1,
    //       duration: 1,
    //     }
    //   );
    // ScrollTrigger.create({
    //   trigger: "#qn-container",
    //   start: "top top",
    //   end: "bottom bottom",
    //   scrub: true,
    //   markers: true,
    //   animation: tl,
    // });
  }, []);

  const handleMouseEnter = (newIndex) => {
    if (index !== newIndex) {
      setIsFading(false); // Start fade-out
      setTimeout(() => {
        setIndex(newIndex);
        setIsFading(true); // Start fade-in
      }, 300); // Wait for the transition to finish
    }
  };

  const items = [
    {
      name: "thể chế chính trị",
      title: (
        <div className="uppercase text-yellow-400 text-[2.5vw] font-bold">
          Dân chủ là một thể chế chính trị,
          <br />
          một chế độ xã hội
        </div>
      ),
      description: (
        <div>
          <div className="text-[2.1vw] mb-[2vw]">Người khẳng định:</div>
          <div className="text-zinc-200 text-[2.1vw] italic w-[80%]">
            Dân chủ là dân là chủ và dân làm chủ. Người nói: “Nước ta là nước
            dân chủ, địa vị cao nhất là dân, vì dân là chủ”
          </div>
        </div>
      ),
      image: "/images/trung-cau-y-dan.jpg",
    },
    {
      name: "giá trị nhân loại chung",
      title: (
        <div className="uppercase text-yellow-400 text-[2.5vw] font-bold">
          Dân chủ là một
          <br />
          giá trị nhân loại chung
          <br />
        </div>
      ),
      description: (
        <div>
          <div className="text-[2.1vw] mb-[2vw]">Người khẳng định:</div>
          <div className="text-zinc-200 text-[2.1vw] italic w-[80%]">
            “Chế độ ta là chế độ dân chủ, tức là nhân dân là người chủ, mà Chính
            phủ là người đày tớ trung thành của nhân dân”
          </div>
        </div>
      ),
      image: "/images/trung-cau-y-dan.jpg",
    },
  ];

  return (
    <>
      <div className="w-screen h-[100vh] relative group p-[4vw]">
        <div className="text-zinc-200 text-[3vw] text-left uppercase font-bold">
          Tư tưởng về dân chủ của
          <br />
          <span className="text-yellow-400 text-[5vw]">Hồ Chí Minh</span>
        </div>
        <div className="flex flex-col gap-[4vh] mt-[4vh] !text-[1vw] font-semibold text-right absolute right-[8vw] top-[6vw] transition-all duration-300">
          {items.map((item, i) => (
            <div
              key={i}
              onMouseEnter={() => handleMouseEnter(i)}
              className={`relative cursor-pointer pb-2 transition-all duration-300 uppercase ${
                index === i ? "text-yellow-400" : ""
              }`}
            >
              {item.name}
              <div
                className={
                  "h-[0.3vw] absolute bottom-0 left-0 bg-yellow-400 transition-all duration-300 " +
                  (index === i ? "w-full" : "w-0")
                }
              ></div>
            </div>
          ))}
        </div>
        <div className="absolute top-[2vw] left-[8vw] transition-all duration-400">
          <img
            className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-[9/16] object-contain object-center w-[30vw]"
            src="/images/hcm1.png"
            loading="eager"
            alt=""
          />
        </div>
        <div className="absolute bottom-[4vw] right-[4vw] transition-all duration-300 w-[50%]">
          <div
            className={`flex flex-col gap-[8vh] mt-[4vh] transition-all duration-300 ${
              isFading ? "opacity-100" : "opacity-0"
            }`}
          >
            {items[index].title}

            {items[index].description}
          </div>
        </div>
      </div>
    </>
  );
}
