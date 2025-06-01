import React from 'react';

const Camera = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    alert('Form submitted!');
  };

  return (
    <div className="md:w-[55%] min-h-[500px] w-full bg-white rounded-2xl shadow-lg p-4 flex flex-col space-y-4 overflow-auto">
      <h2 className="text-gray-800 font-semibold text-center mb-2">Live Video Stream</h2>

      <form
        className="flex flex-wrap items-center justify-center gap-4"
        onSubmit={handleSubmit}
      >
        <div className="flex items-center space-x-2">
          <label htmlFor="select-number" className="text-gray-700 font-medium whitespace-nowrap">
            Select Number:
          </label>
          <select
            id="select-number"
            name="select-number"
            className="border border-gray-300 rounded-md p-2 w-24 text-gray-700"
            required
            aria-label="Select number"
          >
            <option value="" disabled>Select</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <label htmlFor="time-input" className="text-gray-700 font-medium whitespace-nowrap">
            Select Time:
          </label>
          <input
            type="time"
            id="time-input"
            name="time-input"
            className="border border-gray-300 rounded-md p-2"
            required
            aria-label="Select time"
          />
        </div>

        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-amber-600 transition"
        >
          Go
        </button>
      </form>

      <div className="flex-1 min-w-0 flex justify-center items-center border border-gray-300 rounded-md bg-gray-50 p-4">
        <img
          src="https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?auto=format&fit=crop&w=400&q=80"
          alt="Camera Roll Placeholder"
          className="w-full h-auto max-h-[40vh] object-contain rounded-md shadow-md"
        />
      </div>

      <div className="m-4 flex flex-wrap justify-center space-x-4 px-5 gap-4">
        <button>Enable Privacy Mode</button>
        <button>Enable Patient Privacy</button>
      </div>
    </div>
  );
};

export default Camera;