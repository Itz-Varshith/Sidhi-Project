import React from "react";
import Camera from "../components/Camera.jsx";
import Paramonitor from "../components/Paramonitor.jsx";
import RecentLogs from "../components/RecentLogs.jsx";
import Ventilator from "../components/Ventilator.jsx";
import DailyInteractionReport from '../components/DailyInteractionReport.jsx';
import Navbar from "../components/Navbar.jsx";

const Home = () => {
  //We will use a component based system so that we can easily manage the data fetching part of the system.
  //The styling for the layut will be added here.
  //The components will only contain the code for the data of the container.
  return (
    <>
    <Navbar heading="Data Collection Interface"/>
    <div className="bg-500 grow-0 basis-[90%] shadow-lg p-8 overflow-auto min-h-fit">
      <div className="flex flex-col md:flex-row h-full md:space-x-8 space-y-4 md:space-y-0 py-0.5">
        <Camera />
        <div className="md:w-[45%] min-h-[500px] w-full bg-white rounded-2xl shadow-lg p-4 flex flex-col space-y-4">
            <RecentLogs />
            <DailyInteractionReport/>
            <Ventilator />
            <Paramonitor />
        </div>
      </div>
    </div>
    </>
  );
};

export default Home;
