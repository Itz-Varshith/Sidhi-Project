import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Ventilator = () => {
  const [data, setData] = useState([]);
  const [error, setError] = useState('');

  const fetchVentilatorData = async () => {
    try {
      const response = await axios.get('http://10.2.35.160:5000/get_ventilator_data');
      // Assuming each entry has a timestamp and ventilator params like peak, pmean, peep1, ie, etc.
      const formatted = response.data.map(entry => {
        const dateObj = new Date(`${entry.Date}T${entry.Time}`);
        return {
          date: dateObj.toLocaleDateString('en-IN'), // or entry.Date
          time: dateObj.toLocaleTimeString('en-IN'), // or entry.Time
          peak: entry.peak,
          pmean: entry.pmean,
          peep1: entry.peep1,
          ie: entry.ie,
          ftot: entry.ftot,
          vte: entry.vte,
          vetot: entry.vetot,
          peep2: entry.peep2,
          vt: entry.vt,
          o2: entry.o2,
        };
      });
      setData(formatted.reverse()); // latest first
    } catch (err) {
      console.error(err);
      setError('Error fetching ventilator data');
    }
  };

  useEffect(() => {
    fetchVentilatorData();
    const interval = setInterval(fetchVentilatorData, 5000); // refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="overflow-x-auto max-h-[220px] border border-gray-300 rounded-2xl p-3 bg-gray-50">
      <h3 className="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">
        Ventilator Data
      </h3>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <table className="min-w-full table-auto border-collapse border border-gray-300 text-sm">
        <thead className="sticky top-0 bg-gray-200">
          <tr>
            <th className="border px-2 py-1 text-left w-auto">Date</th>
            <th className="border px-2 py-1 text-left">Time</th>
            <th className="border px-2 py-1 text-left">Peak</th>
            <th className="border px-2 py-1 text-left">PMean</th>
            <th className="border px-2 py-1 text-left">Peep1</th>
            <th className="border px-2 py-1 text-left">I:E</th>
            <th className="border px-2 py-1 text-left">FTOT</th>
            <th className="border px-2 py-1 text-left">VTE</th>
            <th className="border px-2 py-1 text-left">VETOT</th>
            <th className="border px-2 py-1 text-left">Peep2</th>
            <th className="border px-2 py-1 text-left">VT</th>
            <th className="border px-2 py-1 text-left">O2</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row, index) => (
            <tr key={index} className={index % 2 ? 'bg-gray-50' : ''}>
              <td className="border px-2 py-1">{row.date}</td>
              <td className="border px-2 py-1">{row.time}</td>
              <td className="border px-2 py-1">{row.peak}</td>
              <td className="border px-2 py-1">{row.pmean}</td>
              <td className="border px-2 py-1">{row.peep1}</td>
              <td className="border px-2 py-1">{row.ie}</td>
              <td className="border px-2 py-1">{row.ftot}</td>
              <td className="border px-2 py-1">{row.vte}</td>
              <td className="border px-2 py-1">{row.vetot}</td>
              <td className="border px-2 py-1">{row.peep2}</td>
              <td className="border px-2 py-1">{row.vt}</td>
              <td className="border px-2 py-1">{row.o2}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Ventilator;
