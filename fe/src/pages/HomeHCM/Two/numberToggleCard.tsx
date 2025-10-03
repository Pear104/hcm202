import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

export const NumberedToggleCard = ({ number, title, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="w-full relative">
      {/* Vertical Line Connector */}
      {number > 1 && (
        <div className="absolute left-[1.5vw] top-[-1vw] w-[0.15vw] h-[1vw] bg-red-500/40" 
             style={{ borderLeft: '2px dashed #ef4444' }} />
      )}
      
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center gap-[1.5vw] px-[2vw] py-[1vw] rounded-2xl transition-all duration-300 ease-in-out ${
          isOpen 
            ? 'bg-[#602222] text-white shadow-lg' 
            : 'bg-[#151515] text-white hover:bg-gray-700'
        }`}
      >
        {/* Number Circle */}
        <div className={`relative flex-shrink-0 w-[4vw] h-[4vw] rounded-full flex items-center justify-center text-[1.6vw] font-bold transition-all duration-300 ${
          isOpen 
            ? 'bg-white text-red-500' 
            : 'bg-[#ff2f2f] text-white'
        }`}>
          {number}
          {/* Checkmark for closed state */}
          {!isOpen && (
            <svg 
              className="absolute top-[-0.3vw] right-[-0.3vw] w-[1.4vw] h-[1.4vw] bg-white rounded-full p-[0.15vw] transition-opacity duration-200" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={3} 
                d="M5 13l4 4L19 7" 
                className="text-green-500"
              />
            </svg>
          )}
        </div>

        {/* Title */}
        <div className="flex-1 text-left text-[1.3vw] font-bold uppercase">
          {title}
        </div>

        {/* Arrow Icon */}
        <div className="flex-shrink-0 transition-transform duration-300">
          {isOpen ? (
            <ChevronUp className="w-[2vw] h-[2vw]" />
          ) : (
            <ChevronDown className="w-[2vw] h-[2vw]" />
          )}
        </div>
      </button>

      {/* Content with smooth animation */}
      <div 
        className={`overflow-hidden transition-all duration-500 ease-in-out ${
          isOpen ? 'max-h-[500px] opacity-100 mt-[1vw]' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="ml-[4vw] px-[2vw] py-[1.5vw] bg-[#363636] text-white rounded-2xl text-[1.2vw] leading-relaxed">
          {children}
        </div>
      </div>
      
      {/* Bottom connector line - extends through content when open */}
      <div 
        className={`absolute left-[1.5vw] w-[0.15vw] transition-all duration-500 ease-in-out ${
          isOpen ? 'h-[calc(100%+1vw)]' : 'h-0'
        }`}
        style={{ 
          borderLeft: '2px dashed #ff2f2f',
          top: '3vw',
          opacity: isOpen ? 1 : 0
        }} 
      />
    </div>
  );
};
