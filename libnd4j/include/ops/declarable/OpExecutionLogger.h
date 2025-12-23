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
// OpExecutionLogger - Logs operation executions for crash detection and debugging.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_OPEXECUTIONLOGGER_H
#define LIBND4J_OPEXECUTIONLOGGER_H

#include <string>
#include <atomic>
#include <mutex>
#include <deque>
#include <sstream>
#include <chrono>

// Forward declaration of Context
namespace sd {
namespace graph {
class Context;
}
}

namespace sd {
namespace ops {

/**
 * OpExecutionLogger - Singleton class for logging operation executions.
 * Used for crash detection and debugging by tracking which operations
 * are currently executing.
 *
 * When SD_GCC_FUNCTRACE is enabled, this logs operation starts, successes,
 * and failures to help diagnose crashes.
 */
class OpExecutionLogger {
public:
    /**
     * Get singleton instance
     */
    static OpExecutionLogger& getInstance() {
        static OpExecutionLogger instance;
        return instance;
    }

    /**
     * Enable operation execution logging
     */
    void enable() {
        _enabled.store(true);
    }

    /**
     * Disable operation execution logging
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
     */
    void setLogPath(const std::string& path) {
        _logPath = path;
    }

    /**
     * Get log contents
     * @param maxBytes Maximum number of bytes to return
     * @param fromEnd If true, get from end of log; otherwise from beginning
     */
    std::string getLogContents(size_t maxBytes, bool fromEnd) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::stringstream ss;
        size_t totalSize = 0;

        if (fromEnd) {
            // Get from end
            for (auto it = _logEntries.rbegin(); it != _logEntries.rend() && totalSize < maxBytes; ++it) {
                if (totalSize + it->size() > maxBytes) break;
                ss << *it << "\n";
                totalSize += it->size() + 1;
            }
        } else {
            // Get from beginning
            for (const auto& entry : _logEntries) {
                if (totalSize + entry.size() > maxBytes) break;
                ss << entry << "\n";
                totalSize += entry.size() + 1;
            }
        }
        return ss.str();
    }

    /**
     * Flush log to disk
     */
    void flush() {
        // In-memory logging - nothing to flush
    }

    /**
     * Dump current state to log
     * @param message Optional message to include
     */
    void dumpCurrentState(const std::string& message) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        std::stringstream ss;
        ss << "[STATE] " << message;
        ss << " | Total ops logged: " << _totalOps;
        ss << " | Successes: " << _successCount;
        ss << " | Failures: " << _failureCount;
        addLogEntry(ss.str());
    }

    /**
     * Log operation start
     * @param opName Name of the operation
     * @param context Execution context
     * @param javaStackTrace Optional Java stack trace
     */
    void logOpStart(const char* opName, graph::Context* context, const std::string& javaStackTrace) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _totalOps++;
        std::stringstream ss;
        ss << "[START] " << (opName ? opName : "unknown");
        ss << " | context=" << (void*)context;
        if (!javaStackTrace.empty()) {
            ss << " | java_stack=" << javaStackTrace.substr(0, 100);
        }
        addLogEntry(ss.str());
        setCurrentOpName(opName ? opName : "unknown");
    }

    /**
     * Log successful operation completion
     * @param opName Name of the operation
     * @param context Execution context
     */
    void logOpSuccess(const char* opName, graph::Context* context) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _successCount++;
        std::stringstream ss;
        ss << "[SUCCESS] " << (opName ? opName : "unknown");
        ss << " | context=" << (void*)context;
        addLogEntry(ss.str());
        clearCurrentOpName();
    }

    /**
     * Log operation failure
     * @param opName Name of the operation
     * @param context Execution context
     * @param errorMessage Error message
     */
    void logOpFailure(const char* opName, graph::Context* context, const char* errorMessage) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _failureCount++;
        std::stringstream ss;
        ss << "[FAILURE] " << (opName ? opName : "unknown");
        ss << " | context=" << (void*)context;
        ss << " | error=" << (errorMessage ? errorMessage : "unknown");
        addLogEntry(ss.str());
        clearCurrentOpName();
    }

    /**
     * Get statistics
     */
    void getStats(size_t& totalOps, size_t& successes, size_t& failures) const {
        std::lock_guard<std::mutex> lock(_mutex);
        totalOps = _totalOps;
        successes = _successCount;
        failures = _failureCount;
    }

    /**
     * Set the current operation name for this thread.
     * Used by lifecycle trackers to tag allocations with the operation that triggered them.
     * @param opName Name of the current operation
     */
    static void setCurrentOpName(const std::string& opName) {
        _currentOpName = opName;
    }

    /**
     * Clear the current operation name for this thread.
     */
    static void clearCurrentOpName() {
        _currentOpName.clear();
    }

    /**
     * Get the current operation name for this thread.
     */
    static const std::string& getCurrentOpName() {
        return _currentOpName;
    }

private:
    OpExecutionLogger() : _enabled(false), _logPath(""), _totalOps(0), _successCount(0), _failureCount(0) {}
    ~OpExecutionLogger() = default;

    // Disable copy and move
    OpExecutionLogger(const OpExecutionLogger&) = delete;
    OpExecutionLogger& operator=(const OpExecutionLogger&) = delete;
    OpExecutionLogger(OpExecutionLogger&&) = delete;
    OpExecutionLogger& operator=(OpExecutionLogger&&) = delete;

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
    size_t _totalOps;
    size_t _successCount;
    size_t _failureCount;

    // Thread-local storage for current operation name
    static thread_local std::string _currentOpName;
};

} // namespace ops
} // namespace sd

#endif // LIBND4J_OPEXECUTIONLOGGER_H
