import React from "react";
import { Link, Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Annotate from "./pages/Annotate.jsx";
import Navbar from "./components/Navbar.jsx";


const App = () => {
  return (
    <div>
      {/* This will be the basic design only for the building of the annotation site of the website part from which we can easily annotate and send data to our backend where we save it to the Amazon s3 bucket for the storage of images*/}
      {/* The basic setup for the notification system currently uses the Toatify npm package and it can used at will in the application to show the data required. */}
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/annotate" element={<Annotate />} />
      </Routes>
      
    </div>
  );
};

export default App;
