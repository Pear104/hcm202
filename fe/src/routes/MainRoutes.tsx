import MainLayout from "@/layouts/MainLayout";
import Chat from "@/pages/Chat/Chat";
import Five from "@/pages/HomeHCM/Five/Five";
import Four from "@/pages/HomeHCM/Four/Four";
import HomeHCM from "@/pages/HomeHCM/HomeHCM";
import One from "@/pages/HomeHCM/One/One";
import Three from "@/pages/HomeHCM/Three/Three";
import Two from "@/pages/HomeHCM/Two/Two";
import ChatFullPage from "./../pages/Chat/ChatFullPage";

import { BrowserRouter, Route, Routes } from "react-router";

export default function MainRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<HomeHCM />} />
          <Route path="1" element={<One />} />
          <Route path="2" element={<Two />} />
          <Route path="3" element={<Three />} />
          <Route path="4" element={<Four />} />
          <Route path="5" element={<Five />} />
          <Route path="/chat" element={<ChatFullPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
