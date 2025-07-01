import React, { useEffect, useState } from 'react';
import axios from 'axios';

const RecentLogs = () => {
  const [logs, setLogs] = useState([]);

  const fetchRecentLogs=()=>{
    axios.get('http://localhost:5000/get_daily_logs')
      .then(response => setLogs(response.data))
      .catch(error => console.error('Error fetching logs:', error));
  }

  useEffect(() => {
    fetchRecentLogs();
    const interval=setInterval(fetchRecentLogs,5000);
    return ()=> clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth()+1).toString().padStart(2, '0');
    const year = date.getFullYear().toString().slice(-2);
    const hours = date.getHours();
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `[${day}-${month}-${year}]- ${hours}:${minutes}`;
  };

  return (
    <div className="flex-1 border border-gray-300 rounded-md p-3 bg-gray-50 overflow-auto">
      <h2 className="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">Logs</h2>
      <pre className="text-xs text-gray-600 whitespace-pre-wrap max-h-[150px] overflow-auto">
        {logs.map((log, index) => (
          <div key={index}>
            {`${formatTimestamp(log.timestamp)}- ${log.message}`}
          </div>
        ))}
      </pre>
    </div>
  );
};

export default RecentLogs;
