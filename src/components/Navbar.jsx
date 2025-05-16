import React from "react";

const Navbar = () => {
  return (
    <nav className=" navbar flex justify-between flex-col sm:flex-row items-center top-0 left-0 px-3 py-3 m-0 w-full bg-[#E5ECE9]">
      <div className="">
        <div className=" flex justify-between ">
          <a href="https://www.iiti.ac.in">
            <img
              src="/IITI Logo.png"
              className=" px-2 py-2 m-2 h-15"
              alt="IITI logo"
            />
          </a>
          <a href="https://charakcenter.iiti.ac.in/">
            <img
              src="/charak.png"
              className=" px-2 py-2 m-2 h-15"
              alt="Charak logo"
            />
          </a>
          <a href="https://drishticps.iiti.ac.in/">
            <img
              src="/Drishti.png"
              className=" px-2 py-2 m-2 h-15"
              alt="Drishti logo"
            />
          </a>
        </div>
        <h1 className="text-center"> CHARAK DIGITAL TWIN PLATFORM</h1>
      </div>
      <div className="w-[80%] p-3 text-center text-3xl font-extrabold">
        <h1 >Data Collection Interface</h1>
      </div>
    </nav>
  );
};

export default Navbar;
