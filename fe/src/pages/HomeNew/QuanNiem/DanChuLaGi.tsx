import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { SplitText } from "gsap/all";
import React from "react";

export default function DanChuLaGi() {
  useGSAP(() => {
    let dcWords = new SplitText("#dclg-title", {
      type: "lines",
    });
    gsap.fromTo(
      dcWords.lines,
      {
        opacity: 0,
        y: "100%",
      },
      {
        opacity: 1,
        y: "0%",
        stagger: 0.3,
        scrollTrigger: {
          trigger: "#dclg",
          start: "20% 50%",
          end: "bottom bottom",
          // markers: true,
        },
      }
    );
  }, []);

  return (
    <>
      <div
        id="dclg"
        className="w-screen h-[100vh] relative group p-[4vw] mt-[4vw]"
      >
        <div className="text-yellow-400 text-[12vh] font-bold">
          {" "}
          <span className="uppercase text-[10vw]">Dân chủ</span>
          <span className="ml-8 text-zinc-200">là gì?</span>
        </div>

        <div id="dclg-title" className="text-[3.5vw] mt-[2vw]">
          “Dân chủ là một
          <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
            giá trị xã hội
          </span>
          phản ánh những
          <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
            quyền cơ bản
          </span>
          của con người; là một
          <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
            hình thức tổ chức nhà nước
          </span>
          của giai cấp cầm quyền;
          <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
            có quá trình ra đời, phát triển
          </span>
          cùng với lịch sử xã hội nhân loại.”
        </div>
      </div>
    </>
  );
}
