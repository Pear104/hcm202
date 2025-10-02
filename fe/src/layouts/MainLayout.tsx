import Footer from "@/components/Footer";
import Header from "@/components/Header";
import React from "react";
import { Outlet } from "react-router";

export default function MainLayout() {
  return (
    <div className="text-[1vw]">
      <Header />
      <div className="mt-[3vw]">
        <Outlet />
      </div>
      {/* <Footer /> */}
    </div>
  );
}
