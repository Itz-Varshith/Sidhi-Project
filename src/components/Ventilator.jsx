import React from 'react'

const Ventilator = () => {
  return (
    <div class="overflow-x-auto max-h-[180px] border border-gray-300 rounded-md p-3 bg-gray-50">
            <h3 class="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">Table 2: Transaction Data</h3>
            <table class="min-w-full table-auto border-collapse border border-gray-300 text-sm">
              <thead>
                <tr class="bg-gray-200">
                  <th class="border border-gray-300 px-2 py-1 text-left">Txn ID</th>
                  <th class="border border-gray-300 px-2 py-1 text-left">User ID</th>
                  <th class="border border-gray-300 px-2 py-1 text-left">Amount</th>
                  <th class="border border-gray-300 px-2 py-1 text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="border border-gray-300 px-2 py-1">TX1001</td>
                  <td class="border border-gray-300 px-2 py-1">1</td>
                  <td class="border border-gray-300 px-2 py-1">$150.00</td>
                  <td class="border border-gray-300 px-2 py-1">Completed</td>
                </tr>
                <tr class="bg-gray-50">
                  <td class="border border-gray-300 px-2 py-1">TX1002</td>
                  <td class="border border-gray-300 px-2 py-1">2</td>
                  <td class="border border-gray-300 px-2 py-1">$320.00</td>
                  <td class="border border-gray-300 px-2 py-1">Pending</td>
                </tr>
                <tr>
                  <td class="border border-gray-300 px-2 py-1">TX1003</td>
                  <td class="border border-gray-300 px-2 py-1">3</td>
                  <td class="border border-gray-300 px-2 py-1">$75.00</td>
                  <td class="border border-gray-300 px-2 py-1">Failed</td>
                </tr>
              </tbody>
            </table>
          </div>
  )
}

export default Ventilator
