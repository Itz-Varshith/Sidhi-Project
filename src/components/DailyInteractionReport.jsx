import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DailyData = () => {
  const [data, setData] = useState([]);
  const [error, setError] = useState('');

  const fetchDailyData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/get_log_data');
      // Assuming each entry has all keys as strings or numbers
      // You can parse or format dates if needed here
      setData(response.data.reverse()); // latest first
    } catch (err) {
      console.error(err);
      setError('Error fetching daily data');
    }
  };

  useEffect(() => {
    fetchDailyData();
    const interval = setInterval(fetchDailyData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="overflow-x-auto max-h-[220px] border border-gray-300 rounded-2xl p-3 bg-gray-50">
      <h3 className="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">
        Daily Data
      </h3>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <table className="min-w-full table-auto border-collapse border border-gray-300 text-sm">
        <thead className="sticky top-0 bg-gray-200">
          <tr>
            <th className="border px-2 py-1 text-left">Date</th>
            <th className="border px-2 py-1 text-left">Start Time</th>
            <th className="border px-2 py-1 text-left">Last Capture</th>
            <th className="border px-2 py-1 text-left">Device Usage Count</th>
            <th className="border px-2 py-1 text-left">Data Capture Count</th>
            <th className="border px-2 py-1 text-left">Total Duration</th>
            <th className="border px-2 py-1 text-left">Longest Duration</th>
            <th className="border px-2 py-1 text-left">No of Person</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row, index) => (
            <tr key={index} className={index % 2 ? 'bg-gray-50' : ''}>
              <td className="border px-2 py-1">{row.Date}</td>
              <td className="border px-2 py-1">{row.Start_time}</td>
              <td className="border px-2 py-1">{row.Last_capture}</td>
              <td className="border px-2 py-1">{row.Device_usage_count}</td>
              <td className="border px-2 py-1">{row.Data_capture_count}</td>
              <td className="border px-2 py-1">{row.Total_duration}</td>
              <td className="border px-2 py-1">{row.Longest_duration}</td>
              <td className="border px-2 py-1">{row.No_of_person}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DailyData;
