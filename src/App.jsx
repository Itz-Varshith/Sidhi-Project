import React, { useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Annotate from "./pages/Annotate.jsx";
import Navbar from "./components/Navbar.jsx";
import VerifyImages from "./pages/VerifyImages.jsx";

const App = () => {
  useEffect(() => {
    let lastSuccessfulPing = Date.now();

    const interval = setInterval(() => {
      fetch("http://10.2.35.160:5000/ping")
        .then((response) => {
          if (response.ok) {
            lastSuccessfulPing = Date.now();
          }
        })
        .catch(() => {
          const now = Date.now();
          if (now - lastSuccessfulPing > 30000) {
            console.warn("Backend unreachable. Reloading...");
            // Optional: Store current route before reload
            localStorage.setItem("lastRoute", window.location.pathname);
            window.location.reload();
          }
        });
    }, 3000);

    return () => clearInterval(interval); // Cleanup
  }, []);

  useEffect(() => {
    const lastRoute = localStorage.getItem("lastRoute");
    if (lastRoute && window.location.pathname !== lastRoute) {
      localStorage.removeItem("lastRoute");
      window.location.href = lastRoute;
    }
  }, []);

  return (
    <div className="flex flex-col h-screen space-y-4">
      
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/annotate" element={<Annotate />} />
        <Route path="/verify-images" element={<VerifyImages/>}></Route>
      </Routes>
    </div>
  );
};

export default App;
