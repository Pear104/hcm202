import React from "react";
import { FaVolumeOff } from "react-icons/fa";
import { useNavigate } from "react-router";

export default function Header() {
  const navigate = useNavigate();

  return (
    <div className="flex justify-between items-center px-[2vw] py-[1vw] fixed top-0 left-0 z-20 w-full bg-zinc-800 unbounded">
      <div className="flex gap-x-[2vw]">
        <div onClick={() => navigate("/")} className="text-red-500">
          Trang chủ
        </div>
        <div onClick={() => navigate("/1")}>1.</div>
        <div onClick={() => navigate("/2")}>2.</div>
        <div onClick={() => navigate("/3")}>3.</div>
        <div onClick={() => navigate("/4")}>4.</div>
      </div>
      <div className="flex gap-[0.2vw] items-center unbounded">
        <FaVolumeOff className="rotate-180 text-red-500" size={30} />
        <div>Nhóm 6</div>
      </div>
      <div className="flex gap-x-[1vw] items-center">
        <div>Chat với AI</div>
        <div className="px-[1vw] py-[0.4vw] bg-red-500 rounded-2xl">
          Kiểm tra kiến thức
        </div>
      </div>
    </div>
  );
}
