import React, { useState, useEffect } from 'react';
import {io} from 'socket.io-client';
import PromptBox from './PromptBox';

const SOCKET_URL = 'http://10.2.35.160:5000';


const socket = io(SOCKET_URL);

const Camera = () => {

  const [privacyEnabled, setPrivacyEnabled] = useState(false);
  const [patientPrivacyEnabled, setPatientPrivacyEnabled] = useState(false);
  

  const handleVerifyImages = async () => {
      console.log('Redirecting to verification');

      window.open(`${window.location.origin}/verify-images`, '_blank');
  }

  const handlePatientPrivacy = () => {
    const newState = !patientPrivacyEnabled;
    setPatientPrivacyEnabled(newState);
    socket.emit('toggle_patient_privacy', { enabled: newState });
  };

  const handlePrivacy = () => {
    const newState = !privacyEnabled;
    setPrivacyEnabled(newState);
    socket.emit('toggle_privacy', { enabled: newState });
  };

  
  const [bed, setBed] = useState('');
  const [time, setTime] = useState('');
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [timerId, setTimerId] = useState(null);
  const [buttonLabel, setButtonLabel] = useState('Go');

  // Enable/disable button
  const isFormValid = bed !== '' && time !== '';

  // Handle form submit
  const handleMonitoringSubmit = async (e) => {
    e.preventDefault();

    const action = isMonitoring ? 'cancel' : 'start';

    try {
      const response = await fetch('http://10.2.35.160:5000/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          beds: bed,
          appt: time,
          action: action,
        }),
      });

      const data = await response.json();
      console.log('Server response:', data);

      // Clear any previous countdown
      if (timerId) {
        clearTimeout(timerId);
        setTimerId(null);
      }

      if (isMonitoring) {
        // Stop monitoring
        setIsMonitoring(false);
        setButtonLabel('Go');
      } else {
        // Start monitoring
        setIsMonitoring(true);
        setButtonLabel('Stop Monitoring');

        const secondsLeft = data.seconds_left;
        if (secondsLeft > 0) {
          const timeout = setTimeout(() => {
            setIsMonitoring(false);
            setButtonLabel('Go');
            console.log('Monitoring ended automatically.');
          }, secondsLeft * 1000);
          setTimerId(timeout);
        } else {
          // Expired already
          setIsMonitoring(false);
          setButtonLabel('Go');
        }
      }
    } catch (err) {
      console.error('Error:', err);
    }
  };
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    socket.on("notification", (data) => {
      setNotifications((prev) => {
        const newNotifs = [...prev, data].slice(-3); // max 5
        return newNotifs;
      });

      // Remove the notification after 10 seconds
      setTimeout(() => {
        setNotifications((prev) => prev.filter((n) => n !== data));
      }, 10000);
    });

    // Cleanup
    return () => {
      socket.off("notification");
    };
  }, []);  
  useEffect( ()=>{

   socket.on("connect", ()=> console.log('Connected to the socket server')); 
  });
  useEffect(() => {
    let lastSuccessfulPing = Date.now();

    const interval = setInterval(() => {
      fetch('http://10.2.35.160:5000/ping')
        .then((response) => {
          if (response.ok) {
            lastSuccessfulPing = Date.now();
          }
        })
        .catch(() => {
          const now = Date.now();
          if (now - lastSuccessfulPing > 30000) {
            console.warn("Backend unreachable. Reloading...");
            window.location.reload();
          }
        });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  

  return (
    <div className="md:w-[55%] min-h-[500px] w-full bg-white rounded-2xl shadow-lg p-4 flex flex-col space-y-4 overflow-auto">
      <h2 className="text-gray-900 font-semibold text-center mb-2">🔴Live Video Stream!</h2>

      <form
        onSubmit={handleMonitoringSubmit}
        className="flex flex-wrap items-center justify-center gap-4"
      >
        <div className="flex items-center space-x-2">
          <label htmlFor="select-bed" className="text-gray-700 font-medium whitespace-nowrap">
            Select Bed Number:
          </label>
          <select
            id="select-bed"
            value={bed}
            onChange={(e) => setBed(e.target.value)}
            className="border border-gray-300 rounded-md p-2 w-24 text-gray-700"
            required
          >
            <option value="">Select</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <label htmlFor="time-input" className="text-gray-700 font-medium whitespace-nowrap">
            Monitor till:
          </label>
          <input
            type="time"
            id="time-input"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            className="border border-gray-300 rounded-md p-2"
            required
          />
        </div>

        <button
          type="submit"
          disabled={!isFormValid}
          className={`px-4 py-2 text-white rounded transition ${
            isMonitoring ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-amber-600'
          }`}
        >
          {buttonLabel}
        </button>
      </form>

      <div className="flex-1 w-full h-[10vh] flex justify-center items-center border border-gray-300 rounded-md bg-gray-100 p-4">
        <img
          src="http://10.2.35.160:5000/video"
          alt="Live Camera Feed"
          className="w-full h-full object-contain rounded-md shadow-md"
        />
      </div>

      <div className="m-4 flex flex-wrap justify-center space-x-4 px-5 gap-4">
        <button
          onClick={handleVerifyImages}
          className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
          aria-label="Verify Images"
        >
          Verify Images
        </button>
        <button
          onClick={handlePrivacy}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          {privacyEnabled ? 'Disable Privacy Mode' : 'Enable Privacy Mode'}
        </button>

        <button
          onClick={handlePatientPrivacy}
          className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
          aria-label="Toggle Patient Privacy Mode"
        >
          {patientPrivacyEnabled ? 'Disable Patient Privacy' : 'Enable Patient Privacy'}
        </button>
        <PromptBox/>
      </div>
      <div
          style={{
            position: 'fixed',
            top: '10px',
            right: '10px',
            width: '350px',
            zIndex: 1001,
          }}
        >
          {notifications.map((notif, index) => (
            <div
              key={index}
              style={{
                background: 'red',
                color: 'white',
                padding: '10px',
                borderRadius: '5px',
                marginBottom: '10px',
                position: 'relative',
              }}
            >
              <span>{notif.message}</span>
              <button
                style={{
                  marginLeft: '10px',
                  background: 'white',
                  color: 'red',
                  border: 'none',
                  padding: '5px',
                  cursor: 'pointer',
                  borderRadius: '3px',
                  position: 'absolute',
                  right: '2px',
                  top: '2px',
                }}
                onClick={() =>
                  setNotifications((prev) =>
                    prev.filter((_, i) => i !== index)
                  )
                }
              >
                X
              </button>
            </div>
          ))}
        </div>

    </div>
    
  );
};

export default Camera;