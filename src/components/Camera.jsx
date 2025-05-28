//This is the file for showing the Camera roll.

import React from 'react'

const Camera = () => {
  return (
    <div class="md:w-[55%] w-full bg-white rounded-2xl shadow-lg p-4 flex flex-col space-y-4">
          <h2 class="text-gray-800 font-semibold text-center mb-2">Live Video Stream</h2>
          
          <form
            class="flex flex-wrap items-center justify-center gap-4"
            onsubmit="event.preventDefault(); alert('Form submitted!');"
          >
            <div class="flex items-center space-x-2">
              <label for="select-number" class="text-gray-700 font-medium whitespace-nowrap">Select Number:</label>
              <select
                id="select-number"
                name="select-number"
                class="border border-gray-300 rounded-md p-2 w-24 text-gray-700"
                required
                aria-label="Select number"
              >
                <option value="" disabled selected>Select</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
              </select>
            </div>

            <div class="flex items-center space-x-2">
              <label for="time-input" class="text-gray-700 font-medium whitespace-nowrap">Select Time:</label>
              <input
                type="time"
                id="time-input"
                name="time-input"
                class="border border-gray-300 rounded-md p-2"
                required
                aria-label="Select time"
              />
            </div>

            <button
              type="submit"
              class="px-4 py-2 bg-green-500 text-white rounded hover:bg-amber-600 transition"
            >
              Go
            </button>
          </form>

          <div class="flex-1 flex justify-center items-center border border-gray-300 rounded-md bg-gray-50 p-4">
            <img
              src="https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?auto=format&fit=crop&w=400&q=80"
              alt="Camera Roll Placeholder"
              class="max-w-full max-h-[300px] object-contain rounded-md shadow-md"
            />
          </div>
          <div class="m-4 flex justify-between space-x-4 px-5">
            <button>Enable Privacy Mode</button>
            {/* <button class="px-4 py-2 bg-amber-500 text-white rounded hover:bg-amber-600 transition">Button 2</button> */}
            <button>Enable Patient Privacy</button>
          </div>
        </div>
  )
}

export default Camera
