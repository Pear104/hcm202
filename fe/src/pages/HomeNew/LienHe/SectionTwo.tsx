import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { SplitText } from "gsap/all";
import React from "react";

export default function SectionTwo() {
  // useGSAP(() => {
  //   let dcWords = new SplitText("#cthcm-text", {
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
  //       stagger: 0.3,
  //       scrollTrigger: {
  //         trigger: "#cthcm",
  //         start: "20% 50%",
  //         end: "bottom bottom",
  //         // markers: true,
  //       },
  //     }
  //   );
  // }, []);

  return (
    <>
      <div id="cthcm" className="w-screen h-[110vh] relative group p-[4vw]">
        <div>
          <span className="text-yellow-400 text-[6vw] font-bold">
            Chủ tịch Hồ Chí Minh
          </span>
          <span className="text-[4vw] font-bold mx-[2vw]">Khẳng định:</span>
        </div>
        <div id="cthcm-text" className="text-[2vw] my-[2vw] leading-[3vw]">
          <div className="mt-[2vw]">
            Nhà nước là{" "}
            <span className="uppercase text-yellow-400 font-bold">
              của dân, do dân và vì dân
            </span>
            . Cán bộ chỉ là
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              “công bộc”
            </span>
            của nhân dân, còn nhân dân mới thực sự là
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              chủ thể quyền lực.
            </span>
          </div>
          <div className="mt-[2vw]">
            Người coi dân chủ là
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              “của quý báu nhất của nhân dân”
            </span>
            ,
            <br />
            Là
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              chìa khóa
            </span>
            <span className="ml-[0.4vw]">giải quyết mọi</span>
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              khó khăn
            </span>
            .
            <br />
            Mọi chính sách phải vì lợi ích của dân: “Việc gì có
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              lợi
            </span>
            cho dân thì
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              làm
            </span>
            , việc gì có
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              hại
            </span>
            cho dân thì
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              tránh
            </span>
            ”
          </div>
          <div className="mt-[2vw]">
            Dân chủ phải bắt đầu từ cơ sở với phương châm
            <span className="uppercase text-yellow-400 font-bold mx-[0.4vw]">
              “Dân biết, dân bàn, dân làm, dân kiểm tra”.
            </span>
            Nhân dân được thông tin, tham gia quyết định, trực tiếp thực hiện và
            giám sát mọi công việc.
          </div>
          <div className="mt-[2vw]">
            Tư tưởng dân chủ của
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              Hồ Chí Minh
            </span>
            đã đặt
            <span className="uppercase text-yellow-400 font-bold mx-[0.6vw]">
              nền tảng
            </span>
            cho việc xây dựng Nhà nước pháp quyền xã hội chủ nghĩa
            <span className="uppercase text-yellow-400 font-bold mx-[0.2vw]">
              Việt Nam
            </span>
            , nơi quyền lực thuộc về nhân dân và phục vụ nhân dân.
          </div>
        </div>
      </div>
    </>
  );
}
