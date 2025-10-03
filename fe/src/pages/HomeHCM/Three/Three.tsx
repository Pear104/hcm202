import React, { useEffect } from "react";
import Banner from "./Banner";
import ASection from "./ASection";
import BSection from "./BSection";

export default function Three() {
  useEffect(() => {
    scrollTo(0, 0);
  }, []);

  return (
    <>
      <div>
        <Banner />
        <ASection />
        <BSection />
      </div>
    </>
  );
}
