import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { SplitText } from "gsap/all";

export default function HeroSection() {
  useGSAP(() => {
    let nsWord = new SplitText("#ns-title", {
      type: "words,chars",
    });

    const tl = gsap
      .timeline()
      .fromTo(
        ".letter",
        { y: "100%", opacity: 0 },
        {
          y: "-10%",
          opacity: 1,
          stagger: 0.05,
          ease: "bounce.inOut",
          duration: 1,
        }
      )
      .to("#ns-line", {
        width: "100%",
        duration: 1,
        ease: "power2.inOut",
      })
      .fromTo(
        nsWord.chars,
        { y: "40%", opacity: 0 },
        {
          y: "0",
          opacity: 1,
          stagger: 0.05,
          duration: 0.3,
        }
      )
      .fromTo(
        "#ns-group",
        { x: "40%", opacity: 0 },
        {
          x: "0",
          opacity: 1,
          stagger: 0.05,
          duration: 0.3,
        }
      );
    // tl.play();
  }, []);

  return (
    <div
      id="hero-section"
      className="w-screen h-screen flex items-center justify-center overflow-x-hidden"
    >
      <div>
        <div
          id="dc-title"
          className="text-[34vh] uppercase font-semibold leading-[10vh] text-zinc-800 flex gap-[3vw]"
        >
          <div>
            <span className="letter text-yellow-500">D</span>
            <span className="letter text-yellow-500">â</span>
            <span className="letter text-yellow-500">n</span>
          </div>
          <div>
            <span className="letter text-yellow-500">c</span>
            <span className="letter text-yellow-500">h</span>
            <span className="letter text-yellow-500">ủ</span>
          </div>
        </div>
        <div className="relative flex justify-between mt-[10vh]">
          <div
            id="ns-line"
            className="top-0 left-0 absolute h-[1vw] w-0 bg-orange-500"
          ></div>
          <div
            id="ns-title"
            className="text-[12vh] leading-[18vh] text-blue-500 font-bold"
          >
            sự ra đời
          </div>
          <div
            id="ns-group"
            className="text-[12vh] leading-[18vh] text-green-500 font-bold"
          >
            và phát triển
          </div>
        </div>
      </div>
    </div>
  );
}
