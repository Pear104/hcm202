// import Footer from "@/components/Footer";
import Header from "@/components/Header";
import React from "react";
import { Outlet } from "react-router";

export default function MainLayout() {
  return (
    <div className="text-[1vw] min-h-0">
      <Header />
      <div className="mt-[3vw] min-h-0">
        <Outlet />
      </div>
      {/* <Footer /> */}
    </div>
  );
}
