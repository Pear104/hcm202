import React, { useEffect } from "react";
import Banner from "./Banner";
import ASection from "./ASection";

export default function Two() {
  useEffect(() => {
    scrollTo(0, 0);
  }, []);

  return (
    <>
      <div>
        <Banner />
        <ASection />
      </div>
    </>
  );
}
