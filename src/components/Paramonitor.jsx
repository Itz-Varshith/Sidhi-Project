import React from 'react'

const Paramonitor = () => {
  return (
    <div class="overflow-x-auto max-h-[180px] border border-gray-300 rounded-2xl p-3 bg-gray-50">
            <h3 class="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">Table 1: User Data</h3>
            <table class="min-w-full table-auto border-collapse border border-gray-300 text-sm">
              <thead>
                <tr class="bg-gray-200">
                  <th class="border border-gray-300 px-2 py-1 text-left">ID</th>
                  <th class="border border-gray-300 px-2 py-1 text-left">Name</th>
                  <th class="border border-gray-300 px-2 py-1 text-left">Email</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="border border-gray-300 px-2 py-1">1</td>
                  <td class="border border-gray-300 px-2 py-1">Alice</td>
                  <td class="border border-gray-300 px-2 py-1">alice@example.com</td>
                </tr>
                <tr class="bg-gray-50">
                  <td class="border border-gray-300 px-2 py-1">2</td>
                  <td class="border border-gray-300 px-2 py-1">Bob</td>
                  <td class="border border-gray-300 px-2 py-1">bob@example.com</td>
                </tr>
                <tr>
                  <td class="border border-gray-300 px-2 py-1">3</td>
                  <td class="border border-gray-300 px-2 py-1">Charlie</td>
                  <td class="border border-gray-300 px-2 py-1">charlie@example.com</td>
                </tr>
              </tbody>
            </table>
          </div>

  )
}

export default Paramonitor
