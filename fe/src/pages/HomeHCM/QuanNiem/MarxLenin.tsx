import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from "gsap/all";
import React from "react";
import gsap from "gsap";

export default function MarxLenin() {
  useGSAP(() => {
    const tl = gsap
      .timeline()
      .fromTo(
        "#dc-1",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#dc-2",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#dc-3",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#dc-4",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#dc-5",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#dc-6",
        {
          y: "40%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      );
    ScrollTrigger.create({
      trigger: "#mln-container",
      start: "top 50%",
      end: "bottom bottom",
      // pin: true,
      scrub: true,
      animation: tl,
    });
  }, []);

  return (
    <>
      <div
        id="mln-container"
        className="w-screen h-[200vw] relative group p-[4vw]"
      >
        <div className="relative"></div>
        <div className="sticky top-[7vw] left-[6vw] transition-all duration-400 flex gap-[4vw]">
          <div className="w-[50%]">
            <img
              className="transition-all duration-400 opacity-100 h-auto rounded-xl shadow-lg aspect-[16/9] object-contain object-center w-[40vw]"
              src="/images/marx-lenin.png"
              loading="eager"
              alt=""
            />
            <div className="text-zinc-200 text-[3vw] uppercase font-bold">
              Quan niệm về dân chủ của
              <br />
              <span className="text-yellow-400 text-[7vw] leading-[8vw]">
                Mác - Lênin
              </span>
            </div>
          </div>
          <div className="flex sticky top-[7vw] right-[6vw] w-[50%]">
            <div className="flex flex-col gap-[1vw] mt-[4vh]">
              <div
                id="dc-1"
                className="text-blue-400 text-[2vw] uppercase font-bold"
              >
                Phương diện quyền lực:
              </div>
              <div
                id="dc-2"
                className="text-zinc-200 text-[1.6vw] italic leading-[2vw]"
              >
                Dân chủ là quyền lực thuộc về nhân dân; nhân dân là chủ nhân của
                nhà nước
              </div>
              <div
                id="dc-3"
                className="text-orange-400 text-[2vw] uppercase font-bold"
              >
                Chế độ xã hội - chính trị:
              </div>
              <div
                id="dc-4"
                className="text-zinc-200 text-[1.6vw] italic leading-[2vw]"
              >
                Dân chủ là một hình thức chính thể, chế độ chính trị
              </div>
              <div
                id="dc-5"
                className="text-green-400 text-[2vw] uppercase font-bold"
              >
                Tổ chức và quản lý xã hội:
              </div>
              <div
                id="dc-6"
                className="text-zinc-200 text-[1.6vw] italic leading-[2vw]"
              >
                Dân chủ là một nguyên tắc, kết hợp với nguyên tắc tập trung
                thành nguyên tắc tập trung dân chủ
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
