import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/all";
import React from "react";
import { IoClose } from "react-icons/io5";

export default function Explaination() {
  useGSAP(() => {
    const tl = gsap
      .timeline()
      .fromTo(
        "#demos",
        {
          x: "-100%",
          opacity: 0,
        },
        {
          x: "0%",
          opacity: 1,
          duration: 1,
        }
      )
      .fromTo(
        "#kratos",
        {
          y: "100%",
          opacity: 0,
        },
        {
          y: "0%",
          opacity: 1,
          duration: 1,
        }
      );

    ScrollTrigger.create({
      trigger: "#qn-container",
      start: "top top",
      end: "bottom bottom",
      scrub: true,
      // markers: true,
      animation: tl,
    });
  }, []);

  return (
    <div id="qn-container" className="w-screen h-[200vh] group">
      <div id="kn-title" className="text-[24vh] text-center sticky top-0">
        <span className="text-[20vh] font-semibold border-b-4 text-orange-500">
          demos
        </span>
        <span className="text-[20vh] font-semibold border-t-4 text-green-500">
          kratos
        </span>
        <div className="flex justify-between px-[10vw] items-center">
          <div
            id="demos"
            className="relative text-[12vh] font-semibold text-blue-500"
          >
            nhân dân
            <img
              className="transition-all duration-400 scale-95 opacity-100 h-[16vw] rounded-xl shadow-lg aspect-video object-cover object-center"
              src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png?auto=compress&cs=tinysrgb&w=600"
              loading="eager"
              alt=""
            />
          </div>
          <IoClose className="text-[20vh] text-zinc-900" />
          <div
            id="kratos"
            className="text-[12vh] font-semibold text-orange-500"
          >
            <img
              className="transition-all duration-400 scale-95 opacity-100 h-[16vw] rounded-xl shadow-lg aspect-video object-cover object-center"
              src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png?auto=compress&cs=tinysrgb&w=600"
              loading="eager"
              alt=""
            />
            cai trị
          </div>
        </div>
      </div>
    </div>
  );
}
