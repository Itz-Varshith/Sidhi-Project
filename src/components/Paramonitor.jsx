import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Paramonitor = () => {
  const [data, setData] = useState([]);
  const [error, setError] = useState('');

  const fetchParamonitorData = async () => {
    try {
      const response = await axios.get('http://10.2.35.160:5000/get_paramonitor_data');
      const formatted = response.data.map(entry => {
        const dateObj = new Date(`${entry.Date}T${entry.Time}`);
        return {
          date: dateObj.toLocaleDateString('en-IN'), // or entry.Date
          time: dateObj.toLocaleTimeString('en-IN'), // or entry.Time
          spo2: entry.SpO2 ?? 'N/A',
          pr: entry.PR ?? 'N/A',
          temp: entry.TEMP ?? 'N/A',
          bp: entry.BP ?? 'N/A'
        };
      });
      setData(formatted.reverse()); // show latest first
    } catch (err) {
      console.error(err);
      setError('Error fetching paramonitor data');
    }
  };

  useEffect(() => {
    fetchParamonitorData(); // initial fetch
    const interval = setInterval(fetchParamonitorData, 5000); // refresh every 5 sec
    return () => clearInterval(interval); // cleanup
  }, []);

  return (
    <div className="overflow-x-auto max-h-[220px] border border-gray-300 rounded-2xl p-3 bg-gray-50">
      <h3 className="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">
        Paramonitor Data 
      </h3>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <table className="min-w-full table-auto border-collapse border border-gray-300 text-sm">
        <thead className="sticky top-0 bg-gray-200">
          <tr>
            <th className="border px-2 py-1 text-left">Date</th>
            <th className="border px-2 py-1 text-left">Time</th>
            <th className="border px-2 py-1 text-left">PR</th>
            <th className="border px-2 py-1 text-left">SPO2</th>
            <th className="border px-2 py-1 text-left">Temp (°C)</th>
            <th className="border px-2 py-1 text-left">BP</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row, index) => (
            <tr key={index} className={index % 2 ? 'bg-gray-50' : ''}>
              <td className="border px-2 py-1">{row.date}</td>
              <td className="border px-2 py-1">{row.time}</td>
              <td className="border px-2 py-1">{row.pr}</td>
              <td className="border px-2 py-1">{row.spo2}</td>
              <td className="border px-2 py-1">{row.temp}</td>
              <td className="border px-2 py-1">{row.bp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Paramonitor;
