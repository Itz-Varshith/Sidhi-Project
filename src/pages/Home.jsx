import React from 'react'
import Camera from '../components/Camera.jsx';
import Paramonitor from '../components/Paramonitor.jsx';
import RecentLogs from '../components/RecentLogs.jsx';
import Ventilator from '../components/Ventilator.jsx';


const Home = () => {
  //We will use a component based system so that we can easily manage the data fetching part of the system.
  //The styling for the layut will be added here.
  //The components will only contain the code for the data of the container.
  return (
    <div>
      <div>
        <Camera/>
      </div>
      <div>
        <div>
          <RecentLogs/>
        </div>
        <div>
          <Ventilator/>
        </div>
        <div>
          <Paramonitor/>
        </div>
      </div>
    </div>
  )
}

export default Home
