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
// TADCacheLifecycleTracker - Tracks TAD (Tensor Along Dimension) cache allocations
// and deallocations for memory leak detection.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_TADCACHELIFECYCLETRACKER_H
#define LIBND4J_TADCACHELIFECYCLETRACKER_H

#include <cstddef>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <map>
#include <ostream>
#include <fstream>
#include <system/common.h>

namespace sd {

// Forward declaration
class TadPack;

namespace analysis {
class ComprehensiveLeakAnalyzer;
}

namespace array {

/**
 * Statistics structure for TAD cache lifecycle tracking
 */
struct TADCacheStats {
    size_t totalAllocations = 0;
    size_t totalDeallocations = 0;
    size_t currentLive = 0;
    size_t peakLive = 0;
    size_t totalBytesAllocated = 0;
    size_t totalBytesDeallocated = 0;
};

/**
 * TADCacheLifecycleTracker - Singleton class for tracking TAD cache allocations
 * and deallocations for memory leak detection.
 *
 * This is a stub implementation that provides the expected interface.
 * When SD_GCC_FUNCTRACE is enabled, this tracks TAD cache lifecycle events.
 */
class TADCacheLifecycleTracker {
    friend class sd::analysis::ComprehensiveLeakAnalyzer;

public:
    /**
     * Get singleton instance
     */
    static TADCacheLifecycleTracker& getInstance() {
        static TADCacheLifecycleTracker instance;
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
     * Record a TAD cache allocation
     * @param tadPack Pointer to the TadPack being allocated
     * @param numTads Number of TADs
     * @param shapeInfoBytes Size of shape info in bytes
     * @param offsetsBytes Size of offsets in bytes
     * @param dimensions Dimensions vector
     */
    void recordAllocation(void* tadPack, LongType numTads,
                         size_t shapeInfoBytes, size_t offsetsBytes,
                         const std::vector<LongType>& dimensions) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalAllocations++;
        _stats.currentLive++;
        size_t totalBytes = shapeInfoBytes + offsetsBytes;
        _stats.totalBytesAllocated += totalBytes;
        if (_stats.currentLive > _stats.peakLive) {
            _stats.peakLive = _stats.currentLive;
        }
    }

    /**
     * Record a TAD cache deallocation
     * @param tadPack Pointer to the TadPack being deallocated
     */
    void recordDeallocation(void* tadPack) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalDeallocations++;
        if (_stats.currentLive > 0) {
            _stats.currentLive--;
        }
    }

    /**
     * Get statistics
     */
    TADCacheStats getStats() const {
        return _stats;
    }

    /**
     * Print statistics to output stream
     */
    void printStatistics(std::ostream& out) const {
        out << "TADCache Statistics: allocations=" << _stats.totalAllocations
            << ", deallocations=" << _stats.totalDeallocations
            << ", live=" << _stats.currentLive << "\n";
    }

    /**
     * Print current memory leaks to output stream
     */
    void printCurrentLeaks(std::ostream& out) const {
        out << "TADCache Current Leaks: " << _stats.currentLive << " TadPacks\n";
    }

    /**
     * Log TAD info for a specific address (for crash debugging)
     * @param address Address to look up
     * @param out Output stream
     * @return true if TAD was found
     */
    bool logTADForAddress(void* address, std::ostream& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out << "TADCache address lookup for " << address << ": ";
        out << "tracking " << (_enabled.load() ? "enabled" : "disabled");
        out << ", live TadPacks: " << _stats.currentLive << "\n";
        return false;
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
            out << "=== TADCache Temporal Leak Report ===\n";
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
            out << "=== TADCache Snapshot Diff Report ===\n";
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

private:
    TADCacheLifecycleTracker() : _enabled(false), _nextSnapshotId(1) {}
    ~TADCacheLifecycleTracker() = default;

    // Disable copy and move
    TADCacheLifecycleTracker(const TADCacheLifecycleTracker&) = delete;
    TADCacheLifecycleTracker& operator=(const TADCacheLifecycleTracker&) = delete;
    TADCacheLifecycleTracker(TADCacheLifecycleTracker&&) = delete;
    TADCacheLifecycleTracker& operator=(TADCacheLifecycleTracker&&) = delete;

    std::atomic<bool> _enabled;
    mutable std::mutex _mutex;
    TADCacheStats _stats;
    std::atomic<LongType> _nextSnapshotId;
    std::map<LongType, TADCacheStats> _snapshots;
};

} // namespace array
} // namespace sd

#endif // LIBND4J_TADCACHELIFECYCLETRACKER_H
