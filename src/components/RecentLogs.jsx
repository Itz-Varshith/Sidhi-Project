import React, { useEffect, useState } from 'react';
import axios from 'axios';

const RecentLogs = () => {
  const [logs, setLogs] = useState([]);

  const fetchRecentLogs = async () => {
    axios.get('http://10.2.35.160:5000/get_log_data')
      .then(response => setLogs(response.data.logs))
      .catch(error => console.error('Error fetching logs:', error));
  }

  useEffect(() => {
    fetchRecentLogs();
    const interval=setInterval(fetchRecentLogs,5000);
    return ()=> clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 border border-gray-300 rounded-md p-3 bg-gray-50 overflow-auto">
      <h2 className="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">Logs</h2>
      <div className="text-xs text-gray-600 whitespace-pre-wrap max-h-[150px] overflow-auto space-y-1">
        {logs.map((log, index) => (
          <div key={index}>{log}</div>
        ))}
      </div>
    </div>
  );
};

export default RecentLogs;