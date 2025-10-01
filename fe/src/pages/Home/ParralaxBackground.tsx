import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import React from "react";

export default function ParralaxBackground() {
  useGSAP(() => {
    gsap.to(".parallax-item", {
      y: "-300vh",
      scrollTrigger: {
        trigger: "html",
        start: "top",
        end: "bottom",
        scrub: true,
      },
    });
  }, []);

  return (
    <div
      id="parallax-container"
      className="w-screen h-screen fixed top-0 left-0 -z-10 overflow-hidden"
    >
      <ImageItem
        src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
        className="h-[20vh] left-[50%] top-[33%] rotate-[15deg]"
      />
      <ImageItem
        src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
        className="h-[20vh] left-[75%] top-[33%] rotate-[35deg]"
      />
      <ImageItem
        src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
        className="h-[20vh] left-[25%] top-[33%] rotate-[-15deg]"
      />
      <ImageItem
        src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
        className="h-[20vh] left-[86%] top-[5%] rotate-[15deg]"
      />
      <ImageItem
        src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
        className="h-[30vh] left-[80%] top-[70%] rotate-[-25deg]"
      />
    </div>
  );
}

const ImageItem = ({ src, className }: { src: string; className: string }) => {
  return (
    <img
      src={src}
      className={`parallax-item rounded-xl absolute aspect-square object-cover object-center ${className}`}
    />
  );
};
