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
// OpContext lifecycle tracking for memory leak detection
// Only active when SD_GCC_FUNCTRACE is defined
//

#ifndef LIBND4J_OPCONTEXTLIFECYCLETRACKER_H
#define LIBND4J_OPCONTEXTLIFECYCLETRACKER_H

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

#if defined(SD_GCC_FUNCTRACE)
#include <exceptions/backward.hpp>
#endif

namespace sd {
namespace graph {

// Forward declaration
class Context;

enum class OpContextSegment {
    CPP,
    JAVA
};


/**
 * Statistics structure for OpContext lifecycle tracking
 */
struct OpContextStats {
    size_t totalAllocations = 0;
    size_t totalDeallocations = 0;
    size_t currentLive = 0;
    size_t peakLive = 0;

    // C++ segment stats
    size_t cppAllocations = 0;
    size_t cppDeallocations = 0;
    size_t cppCurrentLive = 0;
    size_t cppPeakLive = 0;

    // Java segment stats
    size_t javaAllocations = 0;
    size_t javaDeallocations = 0;
    size_t javaCurrentLive = 0;
    size_t javaPeakLive = 0;
};

/**
 * Per-operation statistics for OpContext allocation analysis
 */
struct OpContextPerOpStats {
    size_t allocations = 0;
    size_t deallocations = 0;
    size_t currentLive = 0;

    // C++ segment stats
    size_t cpp_allocations = 0;
    size_t cpp_deallocations = 0;
    size_t cpp_currentLive = 0;

    // Java segment stats
    size_t java_allocations = 0;
    size_t java_deallocations = 0;
    size_t java_currentLive = 0;
};

/**
 * Record of a single OpContext allocation with stack trace
 */
struct OpContextAllocationRecord {
    void* context = nullptr;
    int nodeId = 0;
    size_t fastpathInSize = 0;
    size_t fastpathOutSize = 0;
    size_t intermediateResultsSize = 0;
    size_t handlesSize = 0;
    bool hasWorkspace = false;
    bool isFastPath = false;
    std::string opName;
    std::chrono::steady_clock::time_point timestamp;
    std::string stackTrace;        // Captured C++ stack trace
    OpContextSegment segment = OpContextSegment::CPP; // Origin of the allocation
};

/**
 * OpContextLifecycleTracker - Singleton class for tracking OpContext allocations
 * and deallocations for memory leak detection.
 *
 * When SD_GCC_FUNCTRACE is enabled, this tracks context lifecycle events.
 */
class OpContextLifecycleTracker {
public:
    /**
     * Get singleton instance
     */
    static OpContextLifecycleTracker& getInstance() {
        static OpContextLifecycleTracker instance;
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
     * Record an OpContext allocation
     * @param context Pointer to the Context being allocated
     * @param nodeId Node ID for this context
     * @param fastpathInSize Number of fastpath input arrays
     * @param fastpathOutSize Number of fastpath output arrays
     * @param intermediateResultsSize Number of intermediate results
     * @param handlesSize Number of handles
     * @param hasWorkspace Whether the context has a workspace
     * @param isFastPath Whether this is a fastpath execution
     */
    void recordAllocation(void* context, int nodeId,
                         size_t fastpathInSize, size_t fastpathOutSize,
                         size_t intermediateResultsSize, size_t handlesSize,
                         bool hasWorkspace, bool isFastPath) {
        recordAllocation(context, nodeId, fastpathInSize, fastpathOutSize, intermediateResultsSize, handlesSize, hasWorkspace, isFastPath, OpContextSegment::CPP);
    }

    /**
     * Record an OpContext allocation
     * @param context Pointer to the Context being allocated
     * @param nodeId Node ID for this context
     * @param fastpathInSize Number of fastpath input arrays
     * @param fastpathOutSize Number of fastpath output arrays
     * @param intermediateResultsSize Number of intermediate results
     * @param handlesSize Number of handles
     * @param hasWorkspace Whether the context has a workspace
     * @param isFastPath Whether this is a fastpath execution
     * @param segment The allocation origin (CPP or JAVA)
     */
    void recordAllocation(void* context, int nodeId,
                         size_t fastpathInSize, size_t fastpathOutSize,
                         size_t intermediateResultsSize, size_t handlesSize,
                         bool hasWorkspace, bool isFastPath, OpContextSegment segment) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalAllocations++;
        _stats.currentLive++;
        if (_stats.currentLive > _stats.peakLive) {
            _stats.peakLive = _stats.currentLive;
        }

        if (segment == OpContextSegment::JAVA) {
            _stats.javaAllocations++;
            _stats.javaCurrentLive++;
            if (_stats.javaCurrentLive > _stats.javaPeakLive) {
                _stats.javaPeakLive = _stats.javaCurrentLive;
            }
        } else {
            _stats.cppAllocations++;
            _stats.cppCurrentLive++;
            if (_stats.cppCurrentLive > _stats.cppPeakLive) {
                _stats.cppPeakLive = _stats.cppCurrentLive;
            }
        }

        // Create allocation record with stack trace
        OpContextAllocationRecord record;
        record.context = context;
        record.nodeId = nodeId;
        record.fastpathInSize = fastpathInSize;
        record.fastpathOutSize = fastpathOutSize;
        record.intermediateResultsSize = intermediateResultsSize;
        record.handlesSize = handlesSize;
        record.hasWorkspace = hasWorkspace;
        record.isFastPath = isFastPath;
        record.timestamp = std::chrono::steady_clock::now();
        record.segment = segment;

        // Capture current operation context if not already set
        if (record.opName.empty()) {
            record.opName = _currentOpContext;
        }

        // Update per-op statistics
        const std::string& opKey = record.opName.empty() ? "(unknown)" : record.opName;
        auto& opStats = _perOpStats[opKey];
        opStats.allocations++;
        opStats.currentLive++;

        if (segment == OpContextSegment::JAVA) {
            opStats.java_allocations++;
            opStats.java_currentLive++;
        } else {
            opStats.cpp_allocations++;
            opStats.cpp_currentLive++;
        }


#if defined(SD_GCC_FUNCTRACE)
        // Capture C++ stack trace, skipping internal frames to show actual caller
        backward::StackTrace st;
        st.load_here(48);  // Capture more frames to ensure we get meaningful ones

        // Skip initial frames that are just lifecycle tracking internals
        size_t skipFrames = 0;
        backward::TraceResolver resolver;
        resolver.load_stacktrace(st);

        // Find the first frame that's not part of the tracking infrastructure
        for (size_t i = 0; i < st.size() && skipFrames == 0; i++) {
            backward::ResolvedTrace trace = resolver.resolve(st[i]);
            const std::string& funcName = trace.object_function;
            // Skip frames from lifecycle tracking, OpContext internals
            if (funcName.find("recordAllocation") != std::string::npos ||
                funcName.find("LifecycleTracker") != std::string::npos ||
                funcName.find("load_here") != std::string::npos) {
                continue;
            }
            // Found first meaningful frame - start from a few frames before for context
            skipFrames = (i > 2) ? i - 2 : 0;
            break;
        }

        std::ostringstream oss;
        oss << "--- OpContext Allocation Stack Trace ---\n";
        if (!record.opName.empty()) {
            oss << "Operation: " << record.opName << "\n";
        }

        // Print the stack trace, starting from skipFrames
        size_t printed = 0;
        const size_t maxFrames = 15;
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
        _liveAllocations[context] = record;
    }

    /**
     * Record an OpContext deallocation
     * @param context Pointer to the Context being deallocated
     */
    void recordDeallocation(void* context) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalDeallocations++;
        if (_stats.currentLive > 0) {
            _stats.currentLive--;
        }

        // Update per-op stats before removing allocation record
        auto it = _liveAllocations.find(context);
        if (it != _liveAllocations.end()) {
            const auto& record = it->second;

            if (record.segment == OpContextSegment::JAVA) {
                _stats.javaDeallocations++;
                if (_stats.javaCurrentLive > 0) {
                    _stats.javaCurrentLive--;
                }
            } else {
                _stats.cppDeallocations++;
                if (_stats.cppCurrentLive > 0) {
                    _stats.cppCurrentLive--;
                }
            }

            const std::string& opKey = record.opName.empty() ? "(unknown)" : record.opName;
            auto opIt = _perOpStats.find(opKey);
            if (opIt != _perOpStats.end()) {
                auto& opStats = opIt->second;
                opStats.deallocations++;
                if (opStats.currentLive > 0) {
                    opStats.currentLive--;
                }

                if (record.segment == OpContextSegment::JAVA) {
                    opStats.java_deallocations++;
                    if (opStats.java_currentLive > 0) {
                        opStats.java_currentLive--;
                    }
                } else {
                    opStats.cpp_deallocations++;
                    if (opStats.cpp_currentLive > 0) {
                        opStats.cpp_currentLive--;
                    }
                }
            }
            _liveAllocations.erase(it);
        }
    }

    /**
     * Update the operation name for a tracked context
     * @param context Pointer to the Context
     * @param opName Name of the operation
     */
    void updateContextOpName(void* context, const std::string& opName) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _liveAllocations.find(context);
        if (it != _liveAllocations.end()) {
            it->second.opName = opName;
        }
    }

    /**
     * Get statistics
     */
    OpContextStats getStats() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _stats;
    }

    /**
     * Print current memory leaks to output stream with stack traces
     * @param out Output stream
     * @param maxSamples Maximum number of sample leaks to show (default 10)
     */
    void printCurrentLeaks(std::ostream& out, size_t maxSamples = 10) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out << "OpContext Current Leaks: " << _liveAllocations.size() << " contexts\n";

        if (_liveAllocations.empty()) {
            return;
        }

        out << "\n=== SAMPLE LEAKED OpContexts (showing up to " << maxSamples << " of " << _liveAllocations.size() << ") ===\n";

        size_t count = 0;
        for (const auto& entry : _liveAllocations) {
            if (count >= maxSamples) break;

            const OpContextAllocationRecord& rec = entry.second;
            out << "\n--- Leak #" << (count + 1) << " ---\n";
            out << "  Context: " << rec.context << "\n";
            out << "  Segment: " << (rec.segment == OpContextSegment::JAVA ? "JAVA" : "CPP") << "\n";
            out << "  NodeId: " << rec.nodeId << "\n";
            if (!rec.opName.empty()) {
                out << "  OpName: " << rec.opName << "\n";
            }
            out << "  FastpathIn: " << rec.fastpathInSize << "\n";
            out << "  FastpathOut: " << rec.fastpathOutSize << "\n";
            out << "  IntermediateResults: " << rec.intermediateResultsSize << "\n";
            out << "  Handles: " << rec.handlesSize << "\n";
            out << "  HasWorkspace: " << (rec.hasWorkspace ? "true" : "false") << "\n";
            out << "  IsFastPath: " << (rec.isFastPath ? "true" : "false") << "\n";

            auto now = std::chrono::steady_clock::now();
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec.timestamp).count();
            out << "  Age: " << age << " ms\n";

            if (!rec.stackTrace.empty()) {
                out << "  C++ Stack Trace:\n" << rec.stackTrace << "\n";
            }

            count++;
        }

        if (_liveAllocations.size() > maxSamples) {
            out << "\n... and " << (_liveAllocations.size() - maxSamples) << " more leaked OpContexts\n";
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
            out << "=== OpContext Leak Report ===\n";
            out << "Tracking Enabled: " << (_enabled.load() ? "YES" : "NO") << "\n\n";

            out << "--- Overall Statistics ---\n";
            out << "Total Allocations: " << _stats.totalAllocations << "\n";
            out << "Total Deallocations: " << _stats.totalDeallocations << "\n";
            out << "Current Live: " << _liveAllocations.size() << "\n";
            out << "Peak Live: " << _stats.peakLive << "\n\n";

            out << "--- C++ Segment Statistics ---\n";
            out << "CPP Allocations: " << _stats.cppAllocations << "\n";
            out << "CPP Deallocations: " << _stats.cppDeallocations << "\n";
            out << "CPP Current Live: " << _stats.cppCurrentLive << "\n";
            out << "CPP Peak Live: " << _stats.cppPeakLive << "\n\n";

            out << "--- Java Segment Statistics ---\n";
            out << "Java Allocations: " << _stats.javaAllocations << "\n";
            out << "Java Deallocations: " << _stats.javaDeallocations << "\n";
            out << "Java Current Live: " << _stats.javaCurrentLive << "\n";
            out << "Java Peak Live: " << _stats.javaPeakLive << "\n";

            if (!_liveAllocations.empty()) {
                out << "\n** POTENTIAL LEAK: " << _liveAllocations.size() << " contexts not deallocated **\n";

                out << "\n=== SAMPLE LEAKED OpContexts (showing up to " << maxSamples << " of " << _liveAllocations.size() << ") ===\n";

                size_t count = 0;
                for (const auto& entry : _liveAllocations) {
                    if (count >= maxSamples) break;

                    const OpContextAllocationRecord& rec = entry.second;
                    out << "\n--- Leak #" << (count + 1) << " ---\n";
                    out << "  Context: " << rec.context << "\n";
                    out << "  Segment: " << (rec.segment == OpContextSegment::JAVA ? "JAVA" : "CPP") << "\n";
                    out << "  NodeId: " << rec.nodeId << "\n";
                    if (!rec.opName.empty()) {
                        out << "  OpName: " << rec.opName << "\n";
                    }
                    out << "  IsFastPath: " << (rec.isFastPath ? "true" : "false") << "\n";

                    auto now = std::chrono::steady_clock::now();
                    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec.timestamp).count();
                    out << "  Age: " << age << " ms\n";

                    if (!rec.stackTrace.empty()) {
                        out << "  C++ Stack Trace:\n" << rec.stackTrace << "\n";
                    }

                    count++;
                }

                if (_liveAllocations.size() > maxSamples) {
                    out << "\n... and " << (_liveAllocations.size() - maxSamples) << " more leaked OpContexts\n";
                }
            }
            out.close();
        }
    }

    /**
     * Print per-operation allocation analysis to output stream.
     * Groups all OpContext allocations by operation name and shows statistics and sample stack traces.
     * @param out Output stream
     * @param maxSamplesPerOp Maximum sample allocations to show per operation (default 3)
     */
    void printPerOpAnalysis(std::ostream& out, size_t maxSamplesPerOp = 3) const {
        std::lock_guard<std::mutex> lock(_mutex);

        out << "\n============================================\n";
        out << "  PER-OPERATION OpContext ALLOCATION ANALYSIS\n";
        out << "============================================\n\n";

        if (_liveAllocations.empty()) {
            out << "No live OpContext allocations to analyze.\n";
            return;
        }

        // Group allocations by operation context
        std::map<std::string, std::vector<const OpContextAllocationRecord*>> byOp;
        for (const auto& entry : _liveAllocations) {
            const std::string& opName = entry.second.opName.empty() ? "(unknown)" : entry.second.opName;
            byOp[opName].push_back(&entry.second);
        }

        out << "Total operations with live OpContexts: " << byOp.size() << "\n";
        out << "Total live OpContexts: " << _liveAllocations.size() << "\n\n";

        // Sort operations by number of live allocations (descending)
        std::vector<std::pair<std::string, std::vector<const OpContextAllocationRecord*>>> sortedOps(byOp.begin(), byOp.end());
        std::sort(sortedOps.begin(), sortedOps.end(),
            [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });

        // Print analysis for each operation
        for (const auto& opEntry : sortedOps) {
            const std::string& opName = opEntry.first;
            const auto& records = opEntry.second;

            size_t javaCount = 0;
            size_t cppCount = 0;
            for (const auto* rec : records) {
                if (rec->segment == OpContextSegment::JAVA) {
                    javaCount++;
                } else {
                    cppCount++;
                }
            }


            out << "────────────────────────────────────────────\n";
            out << "OPERATION: " << opName << "\n";
            out << "────────────────────────────────────────────\n";
            out << "  Live OpContexts: " << records.size() << " (Java: " << javaCount << ", CPP: " << cppCount << ")\n";

            // Show per-op stats if available
            auto statsIt = _perOpStats.find(opName);
            if (statsIt != _perOpStats.end()) {
                const auto& stats = statsIt->second;
                out << "  Total Allocations (all time): " << stats.allocations << "\n";
                out << "  Total Deallocations (all time): " << stats.deallocations << "\n";
                if (stats.allocations > stats.deallocations) {
                    out << "  ** LEAK INDICATOR: " << (stats.allocations - stats.deallocations)
                        << " OpContexts not deallocated **\n";
                }

                out << "  CPP Stats (all time):\n";
                out << "    allocs=" << stats.cpp_allocations << ", deallocs=" << stats.cpp_deallocations << "\n";
                 if (stats.cpp_allocations > stats.cpp_deallocations) {
                    out << "    ** LEAK INDICATOR (CPP): " << (stats.cpp_allocations - stats.cpp_deallocations)
                        << " allocations not deallocated **\n";
                }

                out << "  Java Stats (all time):\n";
                out << "    allocs=" << stats.java_allocations << ", deallocs=" << stats.java_deallocations << "\n";
                if (stats.java_allocations > stats.java_deallocations) {
                    out << "    ** LEAK INDICATOR (JAVA): " << (stats.java_allocations - stats.java_deallocations)
                        << " allocations not deallocated **\n";
                }
            }

            // Show sample allocations with stack traces
            out << "\n  Sample OpContexts (showing " << std::min(maxSamplesPerOp, records.size())
                << " of " << records.size() << "):\n";

            size_t sampleCount = 0;
            for (const auto* rec : records) {
                if (sampleCount >= maxSamplesPerOp) break;

                out << "\n  --- Sample #" << (sampleCount + 1) << " ---\n";
                out << "    Context: " << rec->context << "\n";
                out << "    Segment: " << (rec->segment == OpContextSegment::JAVA ? "JAVA" : "CPP") << "\n";
                out << "    NodeId: " << rec->nodeId << "\n";
                out << "    IsFastPath: " << (rec->isFastPath ? "true" : "false") << "\n";
                out << "    FastpathIn: " << rec->fastpathInSize << "\n";
                out << "    FastpathOut: " << rec->fastpathOutSize << "\n";
                out << "    IntermediateResults: " << rec->intermediateResultsSize << "\n";
                out << "    HasWorkspace: " << (rec->hasWorkspace ? "true" : "false") << "\n";

                auto now = std::chrono::steady_clock::now();
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - rec->timestamp).count();
                out << "    Age: " << age << " ms\n";

                if (!rec->stackTrace.empty()) {
                    out << "    C++ Stack Trace:\n" << rec->stackTrace << "\n";
                }

                sampleCount++;
            }

            if (records.size() > maxSamplesPerOp) {
                out << "\n  ... and " << (records.size() - maxSamplesPerOp) << " more OpContexts from this operation\n";
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
     * Get a summary of OpContext allocations grouped by operation.
     * Returns a map of operation name to count.
     */
    std::map<std::string, size_t> getPerOpSummary() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::map<std::string, size_t> summary;

        for (const auto& entry : _liveAllocations) {
            const std::string& opName = entry.second.opName.empty() ? "(unknown)" : entry.second.opName;
            summary[opName]++;
        }

        return summary;
    }

private:
    OpContextLifecycleTracker() : _enabled(false) {}
    ~OpContextLifecycleTracker() = default;

    // Disable copy and move
    OpContextLifecycleTracker(const OpContextLifecycleTracker&) = delete;
    OpContextLifecycleTracker& operator=(const OpContextLifecycleTracker&) = delete;
    OpContextLifecycleTracker(OpContextLifecycleTracker&&) = delete;
    OpContextLifecycleTracker& operator=(OpContextLifecycleTracker&&) = delete;

    std::atomic<bool> _enabled;
    mutable std::mutex _mutex;
    OpContextStats _stats;

    // Live allocation records with stack traces
    std::unordered_map<void*, OpContextAllocationRecord> _liveAllocations;

    // Per-operation statistics
    mutable std::map<std::string, OpContextPerOpStats> _perOpStats;

    // Thread-local operation context
    static inline thread_local std::string _currentOpContext;
};
} // namespace graph
} // namespace sd

#endif // LIBND4J_OPCONTEXTLIFECYCLETRACKER_H
