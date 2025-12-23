/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// AllocationLogger - Logs all memory allocations for debugging.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_ALLOCATIONLOGGER_H
#define LIBND4J_ALLOCATIONLOGGER_H

#include <string>
#include <atomic>
#include <mutex>
#include <deque>
#include <sstream>
#include <fstream>

namespace sd {
namespace array {

/**
 * AllocationLogger - Singleton class for logging memory allocations.
 * Used for debugging memory-related issues by tracking all allocations.
 *
 * When SD_GCC_FUNCTRACE is enabled, this logs allocations for debugging.
 */
class AllocationLogger {
public:
    /**
     * Get singleton instance
     */
    static AllocationLogger& getInstance() {
        static AllocationLogger instance;
        return instance;
    }

    /**
     * Enable allocation logging
     */
    void enable() {
        _enabled.store(true);
    }

    /**
     * Disable allocation logging
     */
    void disable() {
        _enabled.store(false);
    }

    /**
     * Check if logging is enabled
     */
    bool isEnabled() const {
        return _enabled.load();
    }

    /**
     * Get the log file path
     */
    std::string getLogPath() const {
        return _logPath;
    }

    /**
     * Set the log file path
     * @param path Path for the log file
     */
    void setLogPath(const std::string& path) {
        _logPath = path;
    }

    /**
     * Log an allocation
     * @param ptr Pointer to allocated memory
     * @param size Size in bytes
     * @param type Type description
     */
    void logAllocation(void* ptr, size_t size, const std::string& type) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _totalAllocations++;
        _totalBytesAllocated += size;

        std::stringstream ss;
        ss << "[ALLOC] " << ptr << " | size=" << size << " | type=" << type;
        addLogEntry(ss.str());
    }

    /**
     * Log a deallocation
     * @param ptr Pointer being freed
     * @param type Type description
     */
    void logDeallocation(void* ptr, const std::string& type) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _totalDeallocations++;

        std::stringstream ss;
        ss << "[FREE] " << ptr << " | type=" << type;
        addLogEntry(ss.str());
    }

    /**
     * Get log contents
     * @param maxBytes Maximum bytes to return
     */
    std::string getLogContents(size_t maxBytes) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::stringstream ss;
        size_t totalSize = 0;

        for (const auto& entry : _logEntries) {
            if (totalSize + entry.size() > maxBytes) break;
            ss << entry << "\n";
            totalSize += entry.size() + 1;
        }
        return ss.str();
    }

    /**
     * Get statistics
     */
    void getStats(size_t& allocations, size_t& deallocations, size_t& bytesAllocated) const {
        std::lock_guard<std::mutex> lock(_mutex);
        allocations = _totalAllocations;
        deallocations = _totalDeallocations;
        bytesAllocated = _totalBytesAllocated;
    }

    /**
     * Flush log to disk (if log path is set)
     */
    void flush() {
        if (_logPath.empty()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        std::ofstream out(_logPath, std::ios::app);
        if (out.is_open()) {
            for (const auto& entry : _logEntries) {
                out << entry << "\n";
            }
            out.close();
            _logEntries.clear();
        }
    }

private:
    AllocationLogger() : _enabled(false), _logPath(""), _totalAllocations(0), _totalDeallocations(0), _totalBytesAllocated(0) {}
    ~AllocationLogger() = default;

    // Disable copy and move
    AllocationLogger(const AllocationLogger&) = delete;
    AllocationLogger& operator=(const AllocationLogger&) = delete;
    AllocationLogger(AllocationLogger&&) = delete;
    AllocationLogger& operator=(AllocationLogger&&) = delete;

    void addLogEntry(const std::string& entry) {
        _logEntries.push_back(entry);
        // Keep only last 10000 entries to avoid memory growth
        while (_logEntries.size() > 10000) {
            _logEntries.pop_front();
        }
    }

    std::atomic<bool> _enabled;
    std::string _logPath;
    mutable std::mutex _mutex;
    std::deque<std::string> _logEntries;
    size_t _totalAllocations;
    size_t _totalDeallocations;
    size_t _totalBytesAllocated;
};

} // namespace array
} // namespace sd

#endif // LIBND4J_ALLOCATIONLOGGER_H
