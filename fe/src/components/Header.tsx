import React from "react";
import { FaVolumeOff } from "react-icons/fa";
import { useNavigate, useLocation } from "react-router";

export default function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) =>
    location.pathname === path ? "text-red-500" : "text-white";

  return (
    <div className="flex justify-between items-center px-[4vw] py-[1vw] fixed top-0 left-0 z-20 w-full bg-zinc-800 unbounded">
      <div className="flex gap-x-[2vw] items-center">
        <div
          onClick={() => navigate("/")}
          className={`${isActive("/")} cursor-pointer`}
        >
          Trang chủ
        </div>
        <div
          className={`${isActive(
            "/1"
          )} cursor-pointer aspect-square rounded-full flex justify-center items-center transition-all duration-300 hover:text-red-500 hover:bg-red-50 hover:px-[1vw]`}
          onClick={() => navigate("/1")}
        >
          1.
        </div>
        <div
          className={`${isActive(
            "/2"
          )} cursor-pointer aspect-square rounded-full flex justify-center items-center transition-all duration-300 hover:text-red-500 hover:bg-red-50 hover:px-[1vw]`}
          onClick={() => navigate("/2")}
        >
          2.
        </div>
        <div
          className={`${isActive(
            "/3"
          )} cursor-pointer aspect-square rounded-full flex justify-center items-center transition-all duration-300 hover:text-red-500 hover:bg-red-50 hover:px-[1vw]`}
          onClick={() => navigate("/3")}
        >
          3.
        </div>
        <div
          className={`${isActive(
            "/4"
          )} cursor-pointer aspect-square rounded-full flex justify-center items-center transition-all duration-300 hover:text-red-500 hover:bg-red-50 hover:px-[1vw]`}
          onClick={() => navigate("/4")}
        >
          4.
        </div>
      </div>
      <div className="flex gap-[0.2vw] items-center unbounded">
        <FaVolumeOff className="rotate-180 text-red-500" size={30} />
        <div>Nhóm 6</div>
      </div>
      <div className="flex gap-x-[1vw] items-center">
        <div
         className={`${isActive(
            "/chat"
          )} cursor-pointer aspect-auto rounded-3xl flex justify-center items-center transition-all duration-300 hover:text-red-500 hover:bg-red-50 hover:p-[1vw]`}
          onClick={() => navigate("/chat")}
        >Chat với AI</div>
        {/* <div className="px-[1vw] py-[0.4vw] bg-red-500 rounded-2xl">
          Kiểm tra kiến thức
        </div> */}
      </div>
    </div>
  );
}
