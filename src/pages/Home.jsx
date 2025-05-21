import React from 'react'
import Camera from '../components/Camera.jsx';
import Paramonitor from '../components.jsx';
import RecentLogs from '../components/RecentLogs.jsx';
import Ventilator from '../components/Ventilator.jsx';


const Home = () => {
  //We will use a component based system so that we can easily manage the data fetching part of the system.
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
