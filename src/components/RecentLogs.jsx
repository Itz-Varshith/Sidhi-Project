//This is the file for fetching and showing the Recent logs in the webpage.


import React from 'react'

const RecentLogs = () => {
  return (
    <div class="flex-1 border border-gray-300 rounded-md p-3 bg-gray-50 overflow-auto">
            <h2 class="text-gray-700 font-semibold mb-2 border-b border-gray-300 pb-1">Logs</h2>
            <pre className="text-xs text-gray-600 whitespace-pre-wrap max-h-[150px] overflow-auto">
{`[2025-05-27 10:00:00] System started
[2025-05-27 10:05:23] User logged in
[2025-05-27 10:07:45] Data sync complete
[2025-05-27 10:10:11] Warning: Low disk space
[2025-05-27 10:15:00] User logged out`}
</pre>

          </div>
  )
}

export default RecentLogs