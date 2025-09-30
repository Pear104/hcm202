import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { SplitText } from "gsap/all";
import React from "react";

export default function SectionThree() {
  // useGSAP(() => {
  //   let dcWords = new SplitText("#nv-text", {
  //     type: "lines",
  //   });
  //   gsap.fromTo(
  //     dcWords.lines,
  //     {
  //       opacity: 0,
  //       y: "100%",
  //     },
  //     {
  //       opacity: 1,
  //       y: "0%",
  //       stagger: 0.5,
  //       scrollTrigger: {
  //         trigger: "#nv",
  //         start: "20% 50%",
  //         end: "bottom bottom",
  //         markers: true,
  //       },
  //     }
  //   );
  // }, []);

  return (
    <>
      <div
        id="nv"
        className="w-screen h-[100vh] relative group p-[4vw] mt-[4vw]"
      >
        <div className="text-yellow-400 text-[12vh] font-bold">Như vậy</div>

        <div id="nv-text" className="text-[2.4vw] mt-[1.4vw] leading-[5vw]">
          Với tư cách là một hình thái nhà nước, một chế độ chính trị trong lịch
          sử nhân loại, cho đến nay có ba chế độ dân chủ:
          <ul className="list-disc list-inside">
            <li>
              <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
                Nền dân chủ chủ nô
              </span>
              , gắn với chế độ chiếm hữu nô lệ;
            </li>
            <li>
              <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
                nền dân chủ tư sản
              </span>
              , gắn với chế độ tư bản chủ nghĩa;
            </li>
            <li>
              <span className="uppercase text-yellow-400 font-bold mx-[0.8vw]">
                nền dân chủ xã hội chủ nghĩa
              </span>
              , gắn với chế độ xã hội chủ nghĩa.
            </li>
          </ul>
          <div>
            Tuy nhiên, muốn biết một nhà nước dân chủ có thực sự dân chủ hay
            không, phải xem trong nhà nước ấy dân là ai và bản chất của chế độ
            xã hội ấy như thế nào?
          </div>
        </div>
      </div>
    </>
  );
}
