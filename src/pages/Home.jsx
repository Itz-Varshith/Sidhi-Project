import React from "react";
import Camera from "../components/Camera.jsx";
import Paramonitor from "../components/Paramonitor.jsx";
import RecentLogs from "../components/RecentLogs.jsx";
import Ventilator from "../components/Ventilator.jsx";

const Home = () => {
  //We will use a component based system so that we can easily manage the data fetching part of the system.
  //The styling for the layut will be added here.
  //The components will only contain the code for the data of the container.
  return (
    <div className="bg-blue-500 grow-0 basis-[90%] rounded-lg shadow-lg p-4 overflow-auto">
      <div className="flex flex-col md:flex-row h-full md:space-x-4 space-y-4 md:space-y-0">
        <Camera />
        <div className="md:w-[45%] w-full bg-white rounded-lg shadow-lg p-4 flex flex-col space-y-4">
            <RecentLogs />
            <Ventilator />
            <Paramonitor />
        </div>
      </div>
    </div>
  );
};

export default Home;
