import DanChuText from "@/components/DanChuText";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/all";
import { useState } from "react";
import { IoClose } from "react-icons/io5";
import Definition from "./Definition";
import Explaination from "./Explaination";

export default function QuanNiem() {
  return (
    <>
      <Definition />
      <Explaination />
    </>
  );
}
