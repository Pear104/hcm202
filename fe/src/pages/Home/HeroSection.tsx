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
      className="w-screen h-screen flex items-center justify-center"
    >
      <div>
        <div
          id="dc-title"
          className="text-[34vh] uppercase font-semibold leading-[10vh] text-zinc-800 flex gap-[3vw]"
        >
          <div>
            <span className="letter text-blue-500">D</span>
            <span className="letter text-orange-500">â</span>
            <span className="letter text-green-500">n</span>
          </div>
          <div>
            <span className="letter text-blue-500">c</span>
            <span className="letter text-orange-500">h</span>
            <span className="letter text-green-500">ủ</span>
          </div>
        </div>
        <div className="relative flex justify-between mt-[10vh]">
          <div
            id="ns-line"
            className="top-0 left-0 absolute h-[1vh] w-0 bg-black"
          ></div>
          <div
            id="ns-title"
            className="text-[12vh] italic leading-[18vh] text-zinc-900"
          >
            sự ra đời
          </div>
          <div
            id="ns-group"
            className="text-[12vh] italic leading-[18vh] text-zinc-900"
          >
            và phát triển
          </div>
        </div>
      </div>
    </div>
  );
}
