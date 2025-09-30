import React from "react";

export default function Banner() {
  return (
    <>
      <div className="w-screen p-[2vw]">
        <div className="relative">
          <img
            className="w-full aspect-[20/9] object-cover object-center rounded-xl"
            src="https://images.pexels.com/photos/12001644/pexels-photo-12001644.png"
            alt=""
          />
          <div className="z-10 absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center bg-black/50">
            <div className="text-[4vw] font-bold unbounded">
              Sự cần thiết phải <br /> đoàn kết quốc tế
            </div>
            <div className="text-center">
              "Dù màu da có khác nhau, trên đời này chỉ có hai giống người:
              Giống người bóc lột và giống người bị bóc lột.
              <br />
              Mà cũng chỉ có một mối tình hữu ái là thật mà thôi: tình hữu ái vô
              sản".
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
