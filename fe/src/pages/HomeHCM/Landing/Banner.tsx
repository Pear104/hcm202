import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const slides = [
  {
    id: 1,
    path: "/1",
    image: "/images/landing_canthiet.png",
    title: " Lực lượng đoàn kết quốc tế và hình thức tổ chức",
    subtitle: "“Dù màu da có khác nhau, trên đời này chỉ có hai giống người: Giống người bóc lột và giống người bị bóc lột. Mà cũng chỉ có một mối tình hữu ái là thật mà thôi: tình hữu ái vô sản”.",
  },
  {
    id: 2,
    path: "/2",

    image: "/images/landing_lucluong.png",
    title: " Lực lượng đoàn kết quốc tế và hình thức tổ chức",
    subtitle: "“Đoàn kết, đoàn kết, đại đoàn kết.Thành công, thành công, đại thành công”",
  },
  {
    id: 3,
    path: "/3",

    image: "/images/landing_nguyentac.png",
    title: "Nguyên tắc đoàn kết quốc tế",
    subtitle: "Nguyên tắc cơ bản trong quan hệ quốc tế hiện đại",
  },
  {
    id: 4,
    path: "/4",
    image: "/images/landing_hoinhap.png",
    title: "Đoàn kết quốc tế trong thời kỳ hội nhập",
    subtitle: "Đoàn kết quốc tế theo tư tưởng Hồ Chí Minh – Kim chỉ nam cho sinh viên Việt Nam trong thời kỳ hội nhập",
  },
];

export default function Banner() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const goToSlide = (index) => {
    setCurrentIndex(index);
  };

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev === 0 ? slides.length - 1 : prev - 1));
  };

  const goToNext = () => {
    setCurrentIndex((prev) => (prev === slides.length - 1 ? 0 : prev + 1));
  };
  const handleViewMore = (path) => {
    window.location.href = path;
  };

  return (
    <div className="w-screen bg-black pb-[4vw] overflow-hidden">
      <div className="relative h-[50vw] flex items-center justify-center perspective-[2000px]">
        {/* Slides Container */}
        <div className="relative w-full h-full flex items-center justify-center">
          {slides.map((slide, index) => {
            const offset = index - currentIndex;
            const isActive = index === currentIndex;

            return (
              <div
                key={slide.id}
                onClick={() => goToSlide(index)}
                className={`absolute transition-all duration-700 ease-out cursor-pointer ${isActive ? 'z-20' : 'z-10'
                  }`}
                style={{
                  transform: `
                    translateX(${offset * 35}vw) 
                    translateZ(${isActive ? '0px' : '-300px'}) 
                    rotateY(${offset * -25}deg)
                    scale(${isActive ? 1 : 0.8})
                  `,
                  opacity: isActive ? 1 : 0.6,
                  filter: isActive ? 'brightness(1)' : 'brightness(0.5)',
                  transformStyle: 'preserve-3d'
                }}
              >
                <div className="relative w-[80vw] h-[40vw] rounded-2xl overflow-hidden shadow-2xl">
                  <div
                    className="w-full h-full object-cover z-0"
                  >
                    <div className='absolute top-0 left-0 w-full h-full bg-black/30'>

                    </div>
                    <img
                      src={slide.image}
                      alt={slide.title}
                      className="w-full h-full object-cover -z-10"
                    />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent" />

                  {isActive && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white text-center px-[4vw] animate-fadeIn">
                      <h2 className="text-[3.5vw] font-bold unbounded leading-tight mb-[1vw]">
                        {slide.title}
                      </h2>
                      <p className="text-[1.2vw] opacity-90 max-w-[40vw] italic">
                        {slide.subtitle}
                      </p>
                      <button
                        onClick={() => handleViewMore(slide.path)}
                        className="mt-[2vw] px-[2.5vw] py-[0.8vw] bg-white text-black rounded-lg font-semibold hover:bg-gray-200 transition-colors text-[1.1vw]"
                      >
                        Xem thêm
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Navigation Arrows */}
        <button
          onClick={goToPrevious}
          className="absolute left-[4vw] z-30 w-[4vw] h-[4vw] rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/40 transition-all flex items-center justify-center text-white"
        >
          <ChevronLeft className="w-[2.5vw] h-[2.5vw]" />
        </button>

        <button
          onClick={goToNext}
          className="absolute right-[4vw] z-30 w-[4vw] h-[4vw] rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/40 transition-all flex items-center justify-center text-white"
        >
          <ChevronRight className="w-[2.5vw] h-[2.5vw]" />
        </button>

        {/* Dots Indicator */}
        <div className="absolute bottom-[-3vw] left-1/2 transform -translate-x-1/2 flex gap-[0.8vw] z-30">
          {slides.map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`w-[0.8vw] h-[0.8vw] rounded-full transition-all ${index === currentIndex
                ? 'bg-red-500 w-[2.5vw]'
                : 'bg-white/40 hover:bg-white/60'
                }`}
            />
          ))}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out;
        }
        .perspective-[2000px] {
          perspective: 2000px;
        }
      `}</style>
    </div>
  );
}