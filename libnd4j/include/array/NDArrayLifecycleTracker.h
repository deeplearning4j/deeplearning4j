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
// NDArrayLifecycleTracker - Tracks NDArray allocations and deallocations for memory leak detection.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_NDARRAYLIFECYCLETRACKER_H
#define LIBND4J_NDARRAYLIFECYCLETRACKER_H

#include <cstddef>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <map>
#include <unordered_map>
#include <ostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <system/common.h>

#if defined(SD_GCC_FUNCTRACE)
#include <exceptions/backward.hpp>
#endif

namespace sd {

// Forward declaration
class NDArray;

namespace analysis {
class ComprehensiveLeakAnalyzer;
}

namespace array {

    enum class NDArraySegment {    CPP,
    JAVA
};

/**
 * Statistics structure for NDArray lifecycle tracking
 */
struct NDArrayStats {
    size_t totalAllocations = 0;
    size_t totalDeallocations = 0;
    size_t currentLive = 0;
    size_t peakLive = 0;
    size_t totalBytesAllocated = 0;
    size_t totalBytesDeallocated = 0;

    // C++ segment stats
    size_t cppAllocations = 0;
    size_t cppDeallocations = 0;
    size_t cppCurrentLive = 0;
    size_t cppPeakLive = 0;
    size_t cppBytesAllocated = 0;
    size_t cppBytesDeallocated = 0;

    // Java segment stats
    size_t javaAllocations = 0;
    size_t javaDeallocations = 0;
    size_t javaCurrentLive = 0;
    size_t javaPeakLive = 0;
    size_t javaBytesAllocated = 0;
    size_t javaBytesDeallocated = 0;
};

/**
 * Per-operation statistics for allocation analysis
 */
struct PerOpAllocationStats {
    size_t allocations = 0;
    size_t deallocations = 0;
    size_t currentLive = 0;
    size_t bytesAllocated = 0;
    size_t bytesDeallocated = 0;

    // C++ segment stats
    size_t cpp_allocations = 0;
    size_t cpp_deallocations = 0;
    size_t cpp_currentLive = 0;
    size_t cpp_bytesAllocated = 0;
    size_t cpp_bytesDeallocated = 0;

    // Java segment stats
    size_t java_allocations = 0;
    size_t java_deallocations = 0;
    size_t java_currentLive = 0;
    size_t java_bytesAllocated = 0;
    size_t java_bytesDeallocated = 0;
};

/**
 * Record of a single NDArray allocation with stack trace
 */
struct NDArrayAllocationRecord {
    void* pointer = nullptr;
    size_t sizeBytes = 0;
    DataType dataType = DataType::FLOAT32;
    std::vector<LongType> shape;
    bool isView = false;
    std::chrono::steady_clock::time_point timestamp;
    std::string stackTrace;        // Captured C++ stack trace
    std::string javaStackTrace;    // Java stack trace (if provided)
    std::string opContext;         // Operation context (if available)
    NDArraySegment segment = NDArraySegment::CPP; // Origin of the allocation
};

/**
 * NDArrayLifecycleTracker - Singleton class for tracking NDArray allocations
 * and deallocations for memory leak detection.
 *
 * This is a stub implementation that provides the expected interface.
 * When SD_GCC_FUNCTRACE is enabled, this tracks NDArray lifecycle events.
 */
class NDArrayLifecycleTracker {
    friend class sd::analysis::ComprehensiveLeakAnalyzer;

public:
    /**
     * Get singleton instance
     */
    static NDArrayLifecycleTracker& getInstance() {
        static NDArrayLifecycleTracker instance;
        return instance;
    }

    /**
     * Enable or disable tracking
     */
    void setEnabled(bool enabled) {
        _enabled.store(enabled);
    }

    /**
     * Check if tracking is enabled
     */
    bool isEnabled() const {
        return _enabled.load();
    }

    /**
     * Set the current operation context for this thread.
     * Allocations made while an op context is set will be tagged with it.
     */
    static void setCurrentOpContext(const std::string& opName) {
        _currentOpContext = opName;
    }

    /**
     * Clear the current operation context for this thread.
     */
    static void clearCurrentOpContext() {
        _currentOpContext.clear();
    }

    /**
     * Get the current operation context for this thread.
     */
    static const std::string& getCurrentOpContext() {
        return _currentOpContext;
    }

    /**
     * Record an NDArray allocation
     * @param array Pointer to the NDArray being allocated
     * @param sizeBytes Size of the array in bytes
     * @param dataType Data type of the array
     * @param shape Shape of the array
     * @param isView Whether this is a view
     */
    void recordAllocation(void* array, size_t sizeBytes, DataType dataType,
                         const std::vector<LongType>& shape, bool isView) {
        recordAllocation(array, sizeBytes, dataType, shape, isView, NDArraySegment::CPP);
    }

    /**
     * Record an NDArray allocation
     * @param array Pointer to the NDArray being allocated
     * @param sizeBytes Size of the array in bytes
     * @param dataType Data type of the array
     * @param shape Shape of the array
     * @param isView Whether this is a view
     * @param segment The allocation origin (CPP or JAVA)
     */
    void recordAllocation(void* array, size_t sizeBytes, DataType dataType,
                         const std::vector<LongType>& shape, bool isView, NDArraySegment segment) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalAllocations++;
        _stats.currentLive++;
        _stats.totalBytesAllocated += sizeBytes;
        if (_stats.currentLive > _stats.peakLive) {
            _stats.peakLive = _stats.currentLive;
        }

        if (segment == NDArraySegment::JAVA) {
            _stats.javaAllocations++;
            _stats.javaCurrentLive++;
            _stats.javaBytesAllocated += sizeBytes;
            if (_stats.javaCurrentLive > _stats.javaPeakLive) {
                _stats.javaPeakLive = _stats.javaCurrentLive;
            }
        } else {
            _stats.cppAllocations++;
            _stats.cppCurrentLive++;
            _stats.cppBytesAllocated += sizeBytes;
            if (_stats.cppCurrentLive > _stats.cppPeakLive) {
                _stats.cppPeakLive = _stats.cppCurrentLive;
            }
        }

        // Create allocation record with stack trace
        NDArrayAllocationRecord record;
        record.pointer = array;
        record.sizeBytes = sizeBytes;
        record.dataType = dataType;
        record.shape = shape;
        record.isView = isView;
        record.timestamp = std::chrono::steady_clock::now();
        record.segment = segment;

        // Capture current operation context
        record.opContext = _currentOpContext;

        // Update per-op statistics
        auto& opStats = record.opContext.empty() ? _perOpStats["(unknown)"] : _perOpStats[record.opContext];
        opStats.allocations++;
        opStats.bytesAllocated += sizeBytes;
        opStats.currentLive++;

        if (segment == NDArraySegment::JAVA) {
            opStats.java_allocations++;
            opStats.java_bytesAllocated += sizeBytes;
            opStats.java_currentLive++;
        } else {
            opStats.cpp_allocations++;
            opStats.cpp_bytesAllocated += sizeBytes;
            opStats.cpp_currentLive++;
        }

#if defined(SD_GCC_FUNCTRACE)
        // Capture C++ stack trace, skipping internal frames to show actual caller
        backward::StackTrace st;
        st.load_here(48);  // Capture more frames to ensure we get meaningful ones

        // Skip initial frames that are just lifecycle tracking internals
        // We want to show the actual operation that triggered the allocation
        size_t skipFrames = 0;
        backward::TraceResolver resolver;
        resolver.load_stacktrace(st);

        // Find the first frame that's not part of the tracking infrastructure
        for (size_t i = 0; i < st.size() && skipFrames == 0; i++) {
            backward::ResolvedTrace trace = resolver.resolve(st[i]);
            const std::string& funcName = trace.object_function;
            // Skip frames from lifecycle tracking, NDArray constructors that just delegate
            if (funcName.find("recordAllocation") != std::string::npos ||
                funcName.find("LifecycleTracker") != std::string::npos ||
                funcName.find("load_here") != std::string::npos) {
                continue;
            }
            // Found first meaningful frame - we'll start display from a few frames before
            skipFrames = (i > 2) ? i - 2 : 0;
            break;
        }

        std::ostringstream oss;
        backward::Printer printer;
        printer.snippet = false;  // Don't include source snippets for compactness
        printer.color_mode = backward::ColorMode::never;
        printer.address = true;

        // Print a filtered subset of the stack trace
        oss << "--- Allocation Stack Trace ---\n";
        if (!record.opContext.empty()) {
            oss << "Operation: " << record.opContext << "\n";
        }

        // Print the stack trace, starting from skipFrames
        size_t printed = 0;
        const size_t maxFrames = 15;  // Limit output for readability
        for (size_t i = skipFrames; i < st.size() && printed < maxFrames; i++) {
            backward::ResolvedTrace trace = resolver.resolve(st[i]);
            oss << "#" << printed << " ";
            if (!trace.object_function.empty()) {
                oss << trace.object_function;
            } else {
                oss << "[unknown function]";
            }
            if (!trace.source.filename.empty()) {
                oss << " at " << trace.source.filename << ":" << trace.source.line;
            }
            oss << "\n";
            printed++;
        }
        record.stackTrace = oss.str();
#else
        record.stackTrace = "(stack trace capture requires SD_GCC_FUNCTRACE)";
#endif

        // Store the allocation record
        _liveAllocations[array] = record;
    }

    /**
     * Record an NDArray deallocation
     * @param array Pointer to the NDArray being deallocated
     */
    void recordDeallocation(void* array) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalDeallocations++;
        if (_stats.currentLive > 0) {
            _stats.currentLive--;
        }

        // Remove allocation record and update per-op stats
        auto it = _liveAllocations.find(array);
        if (it != _liveAllocations.end()) {
            const auto& record = it->second;
            _stats.totalBytesDeallocated += record.sizeBytes;

            if (record.segment == NDArraySegment::JAVA) {
                _stats.javaDeallocations++;
                if (_stats.javaCurrentLive > 0) {
                    _stats.javaCurrentLive--;
                }
                _stats.javaBytesDeallocated += record.sizeBytes;
            } else {
                _stats.cppDeallocations++;
                if (_stats.cppCurrentLive > 0) {
                    _stats.cppCurrentLive--;
                }
                _stats.cppBytesDeallocated += record.sizeBytes;
            }

            // Update per-op statistics
            const std::string& opCtx = record.opContext.empty() ? "(unknown)" : record.opContext;
            auto opIt = _perOpStats.find(opCtx);
            if (opIt != _perOpStats.end()) {
                auto& opStats = opIt->second;
                opStats.deallocations++;
                opStats.bytesDeallocated += record.sizeBytes;
                if (opStats.currentLive > 0) {
                    opStats.currentLive--;
                }

                if (record.segment == NDArraySegment::JAVA) {
                    opStats.java_deallocations++;
                    opStats.java_bytesDeallocated += record.sizeBytes;
                    if (opStats.java_currentLive > 0) {
                        opStats.java_currentLive--;
                    }
                } else {
                    opStats.cpp_deallocations++;
                    opStats.cpp_bytesDeallocated += record.sizeBytes;
                    if (opStats.cpp_currentLive > 0) {
                        opStats.cpp_currentLive--;
                    }
                }
            }

            _liveAllocations.erase(it);
        }
    }

    /**
     * Update the Java stack trace for an existing allocation record
     * @param array Pointer to the NDArray
     * @param javaStackTrace Java stack trace string
     */
    void updateJavaStackTrace(void* array, const std::string& javaStackTrace) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _liveAllocations.find(array);
        if (it != _liveAllocations.end()) {
            it->second.javaStackTrace = javaStackTrace;
        }
    }

    /**
     * Update the operation context for an existing allocation record.
     * This can be used to retroactively tag an allocation with the op that created it.
     * @param array Pointer to the NDArray
     * @param opContext Operation name
     */
    void updateOpContext(void* array, const std::string& opContext) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _liveAllocations.find(array);
        if (it != _liveAllocations.end()) {
            // Also update per-op stats if this allocation was previously unknown
            const std::string& oldOpCtx = it->second.opContext.empty() ? "(unknown)" : it->second.opContext;
            const std::string& newOpCtx = opContext.empty() ? "(unknown)" : opContext;

            if (oldOpCtx != newOpCtx) {
                // Update per-op stats: decrement old, increment new
                auto oldIt = _perOpStats.find(oldOpCtx);
                if (oldIt != _perOpStats.end() && oldIt->second.currentLive > 0) {
                    oldIt->second.currentLive--;
                }
                _perOpStats[newOpCtx].currentLive++;
            }

            it->second.opContext = opContext;
        }
    }

    /**
     * Get statistics
     */
    NDArrayStats getStats() const {
        return _stats;
    }

    /**
     * Print statistics to output stream
     */
    void printStatistics(std::ostream& out) const {
        out << "NDArray Statistics:\n"
            << "  Total: allocs=" << _stats.totalAllocations
            << ", deallocs=" << _stats.totalDeallocations
            << ", live=" << _stats.currentLive
            << ", peak=" << _stats.peakLive
            << ", bytes alloc=" << _stats.totalBytesAllocated
            << ", bytes dealloc=" << _stats.totalBytesDeallocated << "\n"
            << "  CPP:   allocs=" << _stats.cppAllocations
            << ", deallocs=" << _stats.cppDeallocations
            << ", live=" << _stats.cppCurrentLive
            << ", peak=" << _stats.cppPeakLive
            << ", bytes alloc=" << _stats.cppBytesAllocated
            << ", bytes dealloc=" << _stats.cppBytesDeallocated << "\n"
            << "  JAVA:  allocs=" << _stats.javaAllocations
            << ", deallocs=" << _stats.javaDeallocations
            << ", live=" << _stats.javaCurrentLive
            << ", peak=" << _stats.javaPeakLive
            << ", bytes alloc=" << _stats.javaBytesAllocated
            << ", bytes dealloc=" << _stats.javaBytesDeallocated << "\n";
    }

    /**
     * Print current memory leaks to output stream with stack traces
     * @param out Output stream
     * @param maxSamples Maximum number of sample leaks to show (default 10)
     */
    void printCurrentLeaks(std::ostream& out, size_t maxSamples = 10) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out << "NDArray Current Leaks: " << _liveAllocations.size() << " arrays\n";

        if (_liveAllocations.empty()) {
            return;
        }

        out << "\n=== SAMPLE LEAKED NDArrays (showing up to " << maxSamples << " of " << _liveAllocations.size() << ") ===\n";

        size_t count = 0;
        for (const auto& entry : _liveAllocations) {
            if (count >= maxSamples) break;

            const NDArrayAllocationRecord& rec = entry.second;
            out << "\n--- Leak #" << (count + 1) << " ---\n";
            out << "  Pointer: " << rec.pointer << "\n";
            out << "  Segment: " << (rec.segment == NDArraySegment::JAVA ? "JAVA" : "CPP") << "\n";
            out << "  Size: " << rec.sizeBytes << " bytes\n";
            out << "  DataType: " << static_cast<int>(rec.dataType) << "\n";
            out << "  Shape: [";
            for (size_t i = 0; i < rec.shape.size(); i++) {
                if (i > 0) out << ", ";
                out << rec.shape[i];
            }
            out << "]\n";
            out << "  IsView: " << (rec.isView ? "true" : "false") << "\n";

            // Calculate age
            auto now = std::chrono::steady_clock::now();
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec.timestamp).count();
            out << "  Age: " << age << " ms\n";

            if (!rec.opContext.empty()) {
                out << "  Operation Context: " << rec.opContext << "\n";
            }

            if (!rec.javaStackTrace.empty()) {
                out << "  Java Stack Trace:\n" << rec.javaStackTrace << "\n";
            }

            if (!rec.stackTrace.empty()) {
                out << "  C++ Stack Trace:\n" << rec.stackTrace << "\n";
            }

            count++;
        }

        if (_liveAllocations.size() > maxSamples) {
            out << "\n... and " << (_liveAllocations.size() - maxSamples) << " more leaked NDArrays\n";
        }
    }

    /**
     * Log allocation info for a specific pointer (for crash debugging)
     * @param ptr Pointer to look up
     * @param out Output stream
     * @return true if allocation was found
     */
    bool logAllocationForPointer(void* ptr, std::ostream& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out << "NDArray pointer lookup for " << ptr << ": ";
        out << "tracking " << (_enabled.load() ? "enabled" : "disabled");
        out << ", live arrays: " << _liveAllocations.size() << "\n";

        auto it = _liveAllocations.find(ptr);
        if (it != _liveAllocations.end()) {
            const NDArrayAllocationRecord& rec = it->second;
            out << "  FOUND - Allocation record:\n";
            out << "    Size: " << rec.sizeBytes << " bytes\n";
            out << "    DataType: " << static_cast<int>(rec.dataType) << "\n";
            out << "    Shape: [";
            for (size_t i = 0; i < rec.shape.size(); i++) {
                if (i > 0) out << ", ";
                out << rec.shape[i];
            }
            out << "]\n";
            out << "    IsView: " << (rec.isView ? "true" : "false") << "\n";

            auto now = std::chrono::steady_clock::now();
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec.timestamp).count();
            out << "    Age: " << age << " ms\n";

            if (!rec.javaStackTrace.empty()) {
                out << "    Java Stack Trace:\n" << rec.javaStackTrace << "\n";
            }
            if (!rec.stackTrace.empty()) {
                out << "    C++ Stack Trace:\n" << rec.stackTrace << "\n";
            }
            return true;
        }
        out << "  NOT FOUND in live allocations\n";
        return false;
    }

    /**
     * Generate flamegraph for allocations
     * @param outputPath Path to write flamegraph
     */
    void generateFlamegraph(const std::string& outputPath) const {
        std::ofstream out(outputPath);
        if (out.is_open()) {
            out << "# NDArray Allocation Flamegraph\n";
            out << "# Stack trace capture not yet implemented\n";
            out << "# Statistics: allocations=" << _stats.totalAllocations;
            out << ", deallocations=" << _stats.totalDeallocations;
            out << ", live=" << _stats.currentLive << "\n";
            out.close();
        }
    }

    /**
     * Generate flamegraph for deletions
     * @param outputPath Path to write flamegraph
     */
    void generateDeletionFlamegraph(const std::string& outputPath) const {
        std::ofstream out(outputPath);
        if (out.is_open()) {
            out << "# NDArray Deallocation Flamegraph\n";
            out << "# Stack trace capture not yet implemented\n";
            out << "# Statistics: deallocations=" << _stats.totalDeallocations << "\n";
            out.close();
        }
    }

    /**
     * Generate leak report with sample stack traces
     * @param outputPath Path to write report
     * @param maxSamples Maximum number of sample leaks to show (default 10)
     */
    void generateLeakReport(const std::string& outputPath, size_t maxSamples = 10) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::ofstream out(outputPath);
        if (out.is_open()) {
            out << "=== NDArray Leak Report ===\n";
            out << "Tracking Enabled: " << (_enabled.load() ? "YES" : "NO") << "\n\n";

            out << "--- Overall Statistics ---\n";
            out << "Total Allocations: " << _stats.totalAllocations << "\n";
            out << "Total Deallocations: " << _stats.totalDeallocations << "\n";
            out << "Current Live: " << _liveAllocations.size() << "\n";
            out << "Peak Live: " << _stats.peakLive << "\n";
            out << "Total Bytes Allocated: " << _stats.totalBytesAllocated << "\n";
            out << "Total Bytes Deallocated: " << _stats.totalBytesDeallocated << "\n\n";

            out << "--- C++ Segment Statistics ---\n";
            out << "CPP Allocations: " << _stats.cppAllocations << "\n";
            out << "CPP Deallocations: " << _stats.cppDeallocations << "\n";
            out << "CPP Current Live: " << _stats.cppCurrentLive << "\n";
            out << "CPP Peak Live: " << _stats.cppPeakLive << "\n";
            out << "CPP Bytes Allocated: " << _stats.cppBytesAllocated << "\n";
            out << "CPP Bytes Deallocated: " << _stats.cppBytesDeallocated << "\n\n";

            out << "--- Java Segment Statistics ---\n";
            out << "Java Allocations: " << _stats.javaAllocations << "\n";
            out << "Java Deallocations: " << _stats.javaDeallocations << "\n";
            out << "Java Current Live: " << _stats.javaCurrentLive << "\n";
            out << "Java Peak Live: " << _stats.javaPeakLive << "\n";
            out << "Java Bytes Allocated: " << _stats.javaBytesAllocated << "\n";
            out << "Java Bytes Deallocated: " << _stats.javaBytesDeallocated << "\n";


            if (!_liveAllocations.empty()) {
                out << "\n** POTENTIAL LEAK: " << _liveAllocations.size() << " arrays not deallocated **\n";

                // Output sample leaks with stack traces
                out << "\n=== SAMPLE LEAKED NDArrays (showing up to " << maxSamples << " of " << _liveAllocations.size() << ") ===\n";

                size_t count = 0;
                for (const auto& entry : _liveAllocations) {
                    if (count >= maxSamples) break;

                    const NDArrayAllocationRecord& rec = entry.second;
                    out << "\n--- Leak #" << (count + 1) << " ---\n";
                    out << "  Pointer: " << rec.pointer << "\n";
                    out << "  Segment: " << (rec.segment == NDArraySegment::JAVA ? "JAVA" : "CPP") << "\n";
                    out << "  Size: " << rec.sizeBytes << " bytes\n";
                    out << "  DataType: " << static_cast<int>(rec.dataType) << "\n";
                    out << "  Shape: [";
                    for (size_t i = 0; i < rec.shape.size(); i++) {
                        if (i > 0) out << ", ";
                        out << rec.shape[i];
                    }
                    out << "]\n";
                    out << "  IsView: " << (rec.isView ? "true" : "false") << "\n";

                    auto now = std::chrono::steady_clock::now();
                    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec.timestamp).count();
                    out << "  Age: " << age << " ms\n";

                    if (!rec.opContext.empty()) {
                        out << "  Operation Context: " << rec.opContext << "\n";
                    }
                    if (!rec.javaStackTrace.empty()) {
                        out << "  Java Stack Trace:\n" << rec.javaStackTrace << "\n";
                    }
                    if (!rec.stackTrace.empty()) {
                        out << "  C++ Stack Trace:\n" << rec.stackTrace << "\n";
                    }

                    count++;
                }

                if (_liveAllocations.size() > maxSamples) {
                    out << "\n... and " << (_liveAllocations.size() - maxSamples) << " more leaked NDArrays\n";
                }
            }
            out.close();
        }
    }

    /**
     * Generate temporal leak report
     * @param outputPath Path to write report
     * @param windowCount Number of time windows
     * @param windowDurationSec Duration of each window in seconds
     */
    void generateTemporalLeakReport(const std::string& outputPath, int windowCount, double windowDurationSec) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::ofstream out(outputPath);
        if (out.is_open()) {
            out << "=== NDArray Temporal Leak Report ===\n";
            out << "Note: Temporal tracking requires timestamp storage - not yet implemented\n";
            out << "Current Statistics:\n";
            out << "  Total Allocations: " << _stats.totalAllocations << "\n";
            out << "  Total Deallocations: " << _stats.totalDeallocations << "\n";
            out << "  Current Live: " << _stats.currentLive << "\n";
            out.close();
        }
    }

    /**
     * Capture a leak snapshot for later comparison
     * @return Snapshot ID
     */
    LongType captureLeakSnapshot() {
        std::lock_guard<std::mutex> lock(_mutex);
        LongType id = _nextSnapshotId++;
        // Store current stats with snapshot ID
        _snapshots[id] = _stats;
        return id;
    }

    /**
     * Generate diff between two snapshots
     * @param snapshot1 First snapshot ID
     * @param snapshot2 Second snapshot ID
     * @param outputPath Path to write diff report
     */
    void generateSnapshotDiff(LongType snapshot1, LongType snapshot2, const std::string& outputPath) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::ofstream out(outputPath);
        if (out.is_open()) {
            out << "=== NDArray Snapshot Diff Report ===\n";
            out << "Comparing snapshot " << snapshot1 << " to snapshot " << snapshot2 << "\n\n";

            auto it1 = _snapshots.find(snapshot1);
            auto it2 = _snapshots.find(snapshot2);

            if (it1 == _snapshots.end()) {
                out << "ERROR: Snapshot " << snapshot1 << " not found\n";
            } else if (it2 == _snapshots.end()) {
                out << "ERROR: Snapshot " << snapshot2 << " not found\n";
            } else {
                const auto& s1 = it1->second;
                const auto& s2 = it2->second;
                out << "Allocations:   " << s1.totalAllocations << " -> " << s2.totalAllocations;
                out << " (diff: " << (long long)(s2.totalAllocations - s1.totalAllocations) << ")\n";
                out << "Deallocations: " << s1.totalDeallocations << " -> " << s2.totalDeallocations;
                out << " (diff: " << (long long)(s2.totalDeallocations - s1.totalDeallocations) << ")\n";
                out << "Live:          " << s1.currentLive << " -> " << s2.currentLive;
                out << " (diff: " << (long long)(s2.currentLive - s1.currentLive) << ")\n";
            }
            out.close();
        }
    }

    /**
     * Clear all snapshots
     */
    void clearSnapshots() {
        std::lock_guard<std::mutex> lock(_mutex);
        _snapshots.clear();
    }

    /**
     * Print per-operation allocation analysis to output stream.
     * Groups all allocations by operation name and shows statistics and sample stack traces.
     * @param out Output stream
     * @param maxSamplesPerOp Maximum sample allocations to show per operation (default 3)
     */
    void printPerOpAnalysis(std::ostream& out, size_t maxSamplesPerOp = 3) const {
        std::lock_guard<std::mutex> lock(_mutex);

        out << "\n============================================\n";
        out << "  PER-OPERATION NDArray ALLOCATION ANALYSIS\n";
        out << "============================================\n\n";

        if (_liveAllocations.empty()) {
            out << "No live allocations to analyze.\n";
            return;
        }

        // Group allocations by operation context
        std::map<std::string, std::vector<const NDArrayAllocationRecord*>> byOp;
        for (const auto& entry : _liveAllocations) {
            const std::string& opName = entry.second.opContext.empty() ? "(unknown)" : entry.second.opContext;
            byOp[opName].push_back(&entry.second);
        }

        out << "Total operations with live allocations: " << byOp.size() << "\n";
        out << "Total live NDArrays: " << _liveAllocations.size() << "\n\n";

        // Sort operations by number of live allocations (descending)
        std::vector<std::pair<std::string, std::vector<const NDArrayAllocationRecord*>>> sortedOps(byOp.begin(), byOp.end());
        std::sort(sortedOps.begin(), sortedOps.end(),
            [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });

        // Print analysis for each operation
        for (const auto& opEntry : sortedOps) {
            const std::string& opName = opEntry.first;
            const auto& records = opEntry.second;

            // Calculate total bytes for this op
            size_t totalBytes = 0;
            size_t javaBytes = 0;
            size_t cppBytes = 0;
            size_t javaCount = 0;
            size_t cppCount = 0;

            for (const auto* rec : records) {
                totalBytes += rec->sizeBytes;
                if (rec->segment == NDArraySegment::JAVA) {
                    javaBytes += rec->sizeBytes;
                    javaCount++;
                } else {
                    cppBytes += rec->sizeBytes;
                    cppCount++;
                }
            }

            out << "────────────────────────────────────────────\n";
            out << "OPERATION: " << opName << "\n";
            out << "────────────────────────────────────────────\n";
            out << "  Live Allocations: " << records.size() << " (Java: " << javaCount << ", CPP: " << cppCount << ")\n";
            out << "  Total Live Bytes: " << totalBytes << " (Java: " << javaBytes << ", CPP: " << cppBytes << ")\n";


            // Show per-op stats if available
            auto statsIt = _perOpStats.find(opName);
            if (statsIt != _perOpStats.end()) {
                const auto& stats = statsIt->second;
                out << "  Total Stats (all time):\n";
                out << "    allocs=" << stats.allocations << ", deallocs=" << stats.deallocations
                    << ", bytes alloc=" << stats.bytesAllocated << ", bytes dealloc=" << stats.bytesDeallocated << "\n";
                if (stats.allocations > stats.deallocations) {
                    out << "    ** LEAK INDICATOR: " << (stats.allocations - stats.deallocations)
                        << " total allocations not deallocated **\n";
                }

                out << "  CPP Stats (all time):\n";
                out << "    allocs=" << stats.cpp_allocations << ", deallocs=" << stats.cpp_deallocations
                    << ", bytes alloc=" << stats.cpp_bytesAllocated << ", bytes dealloc=" << stats.cpp_bytesDeallocated << "\n";
                 if (stats.cpp_allocations > stats.cpp_deallocations) {
                    out << "    ** LEAK INDICATOR (CPP): " << (stats.cpp_allocations - stats.cpp_deallocations)
                        << " allocations not deallocated **\n";
                }

                out << "  Java Stats (all time):\n";
                out << "    allocs=" << stats.java_allocations << ", deallocs=" << stats.java_deallocations
                    << ", bytes alloc=" << stats.java_bytesAllocated << ", bytes dealloc=" << stats.java_bytesDeallocated << "\n";
                if (stats.java_allocations > stats.java_deallocations) {
                    out << "    ** LEAK INDICATOR (JAVA): " << (stats.java_allocations - stats.java_deallocations)
                        << " allocations not deallocated **\n";
                }
            }

            // Show sample allocations with stack traces
            out << "\n  Sample Allocations (showing " << std::min(maxSamplesPerOp, records.size())
                << " of " << records.size() << "):\n";

            size_t sampleCount = 0;
            for (const auto* rec : records) {
                if (sampleCount >= maxSamplesPerOp) break;

                out << "\n  --- Sample #" << (sampleCount + 1) << " ---\n";
                out << "    Pointer: " << rec->pointer << "\n";
                out << "    Segment: " << (rec->segment == NDArraySegment::JAVA ? "JAVA" : "CPP") << "\n";
                out << "    Size: " << rec->sizeBytes << " bytes\n";
                out << "    DataType: " << static_cast<int>(rec->dataType) << "\n";
                out << "    Shape: [";
                for (size_t i = 0; i < rec->shape.size(); i++) {
                    if (i > 0) out << ", ";
                    out << rec->shape[i];
                }
                out << "]\n";
                out << "    IsView: " << (rec->isView ? "true" : "false") << "\n";

                auto now = std::chrono::steady_clock::now();
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec->timestamp).count();
                out << "    Age: " << age << " ms\n";

                if (!rec->javaStackTrace.empty()) {
                    out << "    Java Stack Trace:\n" << rec->javaStackTrace << "\n";
                }
                if (!rec->stackTrace.empty()) {
                    out << "    C++ Stack Trace:\n" << rec->stackTrace << "\n";
                }

                sampleCount++;
            }

            if (records.size() > maxSamplesPerOp) {
                out << "\n  ... and " << (records.size() - maxSamplesPerOp) << " more allocations from this operation\n";
            }
            out << "\n";
        }
    }

    /**
     * Generate per-operation analysis report to file.
     * @param outputPath Path to write report
     * @param maxSamplesPerOp Maximum sample allocations to show per operation (default 3)
     */
    void generatePerOpAnalysisReport(const std::string& outputPath, size_t maxSamplesPerOp = 3) const {
        std::ofstream out(outputPath);
        if (out.is_open()) {
            printPerOpAnalysis(out, maxSamplesPerOp);
            out.close();
        }
    }

    /**
     * Get a summary of allocations grouped by operation.
     * Returns a map of operation name to (count, bytes).
     */
    std::map<std::string, std::pair<size_t, size_t>> getPerOpSummary() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::map<std::string, std::pair<size_t, size_t>> summary;

        for (const auto& entry : _liveAllocations) {
            const std::string& opName = entry.second.opContext.empty() ? "(unknown)" : entry.second.opContext;
            summary[opName].first++;
            summary[opName].second += entry.second.sizeBytes;
        }

        return summary;
    }

    /**
     * Get top operations by live allocation count.
     */
    struct OpLiveStats {
        std::string opName;
        size_t liveCount;
        size_t liveBytes;
        size_t javaCount;
        size_t cppCount;
    };

    std::vector<OpLiveStats> getTopOpsByLiveCount(size_t n = 10) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::map<std::string, OpLiveStats> opMap;

        for (const auto& entry : _liveAllocations) {
            const std::string& opName = entry.second.opContext.empty() ? "(unknown)" : entry.second.opContext;
            auto& stats = opMap[opName];
            stats.opName = opName;
            stats.liveCount++;
            stats.liveBytes += entry.second.sizeBytes;
            if (entry.second.segment == NDArraySegment::JAVA) {
                stats.javaCount++;
            } else {
                stats.cppCount++;
            }
        }

        std::vector<OpLiveStats> result;
        for (auto& pair : opMap) {
            result.push_back(pair.second);
        }

        std::sort(result.begin(), result.end(), 
                  [](const OpLiveStats& a, const OpLiveStats& b) { 
                    return a.liveCount > b.liveCount; 
                  });

        if (result.size() > n) {
            result.resize(n);
        }
        return result;
    }

    /**
     * Get allocation age statistics
     */
    struct AllocationAgeStats {
        size_t olderThan30Sec = 0;
        size_t olderThan5Min = 0;
        size_t javaOlderThan30Sec = 0;
        size_t cppOlderThan30Sec = 0;
        double oldestAgeSec = 0.0;
    };

    AllocationAgeStats getAgeStats() const {
        std::lock_guard<std::mutex> lock(_mutex);
        AllocationAgeStats stats;
        auto now = std::chrono::steady_clock::now();

        for (const auto& entry : _liveAllocations) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - entry.second.timestamp).count();
            
            if (age > stats.oldestAgeSec) {
                stats.oldestAgeSec = static_cast<double>(age);
            }
            
            if (age > 30) {
                stats.olderThan30Sec++;
                if (entry.second.segment == NDArraySegment::JAVA) {
                    stats.javaOlderThan30Sec++;
                } else {
                    stats.cppOlderThan30Sec++;
                }
            }
            if (age > 300) {
                stats.olderThan5Min++;
            }
        }
        return stats;
    }

    /**
     * Print actionable analysis with specific recommendations
     */
    void printActionableAnalysis(std::ostream& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto now = std::chrono::steady_clock::now();

        out << "--- NDArray Actionable Analysis ---\n";
        
        // Segment breakdown
        out << "\n  SEGMENT BREAKDOWN:\n";
        size_t javaBytes = _stats.javaBytesAllocated - _stats.javaBytesDeallocated;
        size_t cppBytes = _stats.cppBytesAllocated - _stats.cppBytesDeallocated;
        out << "    Java-owned:  " << _stats.javaCurrentLive << " arrays (" 
            << (javaBytes / (1024*1024)) << " MB)\n";
        out << "    C++ internal: " << _stats.cppCurrentLive << " arrays (" 
            << (cppBytes / (1024*1024)) << " MB)\n";

        // Age analysis
        size_t olderThan30Sec = 0, olderThan5Min = 0;
        size_t javaOld = 0, cppOld = 0;
        std::string oldestOp;
        double oldestAge = 0;

        for (const auto& entry : _liveAllocations) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - entry.second.timestamp).count();
            
            if (age > oldestAge) {
                oldestAge = static_cast<double>(age);
                oldestOp = entry.second.opContext.empty() ? "(unknown)" : entry.second.opContext;
            }
            
            if (age > 30) {
                olderThan30Sec++;
                if (entry.second.segment == NDArraySegment::JAVA) javaOld++;
                else cppOld++;
            }
            if (age > 300) olderThan5Min++;
        }

        out << "\n  AGE ANALYSIS:\n";
        out << "    Oldest allocation: " << oldestAge << " sec (op: " << oldestOp << ")\n";
        out << "    Older than 30 sec: " << olderThan30Sec << " (Java: " << javaOld 
            << ", C++: " << cppOld << ")\n";
        out << "    Older than 5 min:  " << olderThan5Min << "\n";

        // Actionable recommendations
        out << "\n  ACTIONS:\n";
        
        if (olderThan30Sec > 0) {
            double javaPct = olderThan30Sec > 0 ? (100.0 * javaOld / olderThan30Sec) : 0;
            
            if (javaPct > 80) {
                out << "    [ACTION] Java NDArrays not being .close()'d - " << javaPct 
                    << "% of old allocations are Java-owned\n";
                out << "    [ACTION] Use try-with-resources or explicit .close()\n";
            } else if (cppOld > 10) {
                out << "    [WARNING] Possible op leak - " << cppOld 
                    << " C++ allocations older than 30s\n";
                out << "    [ACTION] Review op: " << oldestOp << "\n";
            }
        }

        if (olderThan30Sec == 0 && _stats.currentLive < 100) {
            out << "    [OK] Memory appears healthy\n";
        }
    }


private:
    NDArrayLifecycleTracker() : _enabled(false), _nextSnapshotId(1) {}
    ~NDArrayLifecycleTracker() = default;

    // Disable copy and move
    NDArrayLifecycleTracker(const NDArrayLifecycleTracker&) = delete;
    NDArrayLifecycleTracker& operator=(const NDArrayLifecycleTracker&) = delete;
    NDArrayLifecycleTracker(NDArrayLifecycleTracker&&) = delete;
    NDArrayLifecycleTracker& operator=(NDArrayLifecycleTracker&&) = delete;

    std::atomic<bool> _enabled;
    mutable std::mutex _mutex;
    NDArrayStats _stats;
    std::atomic<LongType> _nextSnapshotId;
    std::map<LongType, NDArrayStats> _snapshots;

    // Live allocation records with stack traces
    std::unordered_map<void*, NDArrayAllocationRecord> _liveAllocations;

    // Per-operation statistics
    mutable std::map<std::string, PerOpAllocationStats> _perOpStats;

    // Thread-local operation context
    static thread_local std::string _currentOpContext;
};

// Define thread-local storage
inline thread_local std::string NDArrayLifecycleTracker::_currentOpContext;

} // namespace array
} // namespace sd

#endif // LIBND4J_NDARRAYLIFECYCLETRACKER_H
