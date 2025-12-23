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
// ShapeCacheLifecycleTracker - Tracks shape cache allocations and deallocations
// for memory leak detection.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_SHAPECACHELIFECYCLETRACKER_H
#define LIBND4J_SHAPECACHELIFECYCLETRACKER_H

#include <cstddef>
#include <string>
#include <atomic>
#include <mutex>
#include <ostream>
#include <system/common.h>

namespace sd {

namespace analysis {
class ComprehensiveLeakAnalyzer;
}

namespace array {

/**
 * Statistics structure for shape cache lifecycle tracking
 */
struct ShapeCacheStats {
    size_t totalAllocations = 0;
    size_t totalDeallocations = 0;
    size_t currentLive = 0;
    size_t peakLive = 0;
    size_t totalBytesAllocated = 0;
    size_t totalBytesDeallocated = 0;
};

/**
 * ShapeCacheLifecycleTracker - Singleton class for tracking shape cache allocations
 * and deallocations for memory leak detection.
 *
 * This is a stub implementation that provides the expected interface.
 * When SD_GCC_FUNCTRACE is enabled, this tracks shape cache lifecycle events.
 */
class ShapeCacheLifecycleTracker {
    friend class sd::analysis::ComprehensiveLeakAnalyzer;

public:
    /**
     * Get singleton instance
     */
    static ShapeCacheLifecycleTracker& getInstance() {
        static ShapeCacheLifecycleTracker instance;
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
     * Record a shape cache allocation
     * @param shapeInfo Pointer to the shape info being allocated
     */
    void recordAllocation(LongType* shapeInfo) {
        if (!_enabled.load()) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _stats.totalAllocations++;
        _stats.currentLive++;
        if (_stats.currentLive > _stats.peakLive) {
            _stats.peakLive = _stats.currentLive;
        }
    }

    /**
     * Record a shape cache deallocation
     * @param shapeInfo Pointer to the shape info being deallocated
     */
    void recordDeallocation(LongType* shapeInfo) {
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
    ShapeCacheStats getStats() const {
        return _stats;
    }

    /**
     * Print statistics to output stream
     */
    void printStatistics(std::ostream& out) const {
        out << "ShapeCache Statistics: allocations=" << _stats.totalAllocations
            << ", deallocations=" << _stats.totalDeallocations
            << ", live=" << _stats.currentLive << "\n";
    }

    /**
     * Print current memory leaks to output stream
     */
    void printCurrentLeaks(std::ostream& out) const {
        out << "ShapeCache Current Leaks: " << _stats.currentLive << " shapes\n";
    }

    /**
     * Log shape info for a specific address (for crash debugging)
     * @param address Address to look up
     * @param out Output stream
     * @return true if shape was found
     */
    bool logShapeForAddress(void* address, std::ostream& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out << "ShapeCache address lookup for " << address << ": ";
        out << "tracking " << (_enabled.load() ? "enabled" : "disabled");
        out << ", live shapes: " << _stats.currentLive << "\n";
        return false;
    }

private:
    ShapeCacheLifecycleTracker() : _enabled(false) {}
    ~ShapeCacheLifecycleTracker() = default;

    // Disable copy and move
    ShapeCacheLifecycleTracker(const ShapeCacheLifecycleTracker&) = delete;
    ShapeCacheLifecycleTracker& operator=(const ShapeCacheLifecycleTracker&) = delete;
    ShapeCacheLifecycleTracker(ShapeCacheLifecycleTracker&&) = delete;
    ShapeCacheLifecycleTracker& operator=(ShapeCacheLifecycleTracker&&) = delete;

    std::atomic<bool> _enabled;
    mutable std::mutex _mutex;
    ShapeCacheStats _stats;
};

} // namespace array
} // namespace sd

#endif // LIBND4J_SHAPECACHELIFECYCLETRACKER_H
