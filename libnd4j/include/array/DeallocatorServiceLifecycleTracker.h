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
// DeallocatorServiceLifecycleTracker - Tracks DeallocatorService statistics
// from the Java side for memory leak detection.
// Only active when SD_GCC_FUNCTRACE is defined.
//

#ifndef LIBND4J_DEALLOCATORSERVICELIFECYCLETRACKER_H
#define LIBND4J_DEALLOCATORSERVICELIFECYCLETRACKER_H

#include <cstddef>
#include <atomic>
#include <system/common.h>

namespace sd {

namespace analysis {
class ComprehensiveLeakAnalyzer;
}

namespace array {

/**
 * DeallocatorServiceLifecycleTracker - Singleton class for tracking DeallocatorService
 * statistics from the Java side for memory leak detection.
 *
 * This receives snapshot data from Java's DeallocatorService and tracks
 * memory management patterns.
 */
class DeallocatorServiceLifecycleTracker {
    friend class sd::analysis::ComprehensiveLeakAnalyzer;

public:
    /**
     * Get singleton instance
     */
    static DeallocatorServiceLifecycleTracker& getInstance() {
        static DeallocatorServiceLifecycleTracker instance;
        return instance;
    }

    /**
     * Enable tracking
     */
    void enable() {
        _enabled.store(true);
    }

    /**
     * Disable tracking
     */
    void disable() {
        _enabled.store(false);
    }

    /**
     * Check if tracking is enabled
     */
    bool isEnabled() const {
        return _enabled.load();
    }

    /**
     * Record a snapshot from Java DeallocatorService
     * @param totalAllocations Total allocations count
     * @param totalDeallocations Total deallocations count
     * @param totalBytesAllocated Total bytes allocated
     * @param totalBytesDeallocated Total bytes deallocated
     * @param peakLiveCount Peak live object count
     * @param peakBytes Peak bytes in use
     */
    void recordSnapshot(LongType totalAllocations, LongType totalDeallocations,
                       LongType totalBytesAllocated, LongType totalBytesDeallocated,
                       LongType peakLiveCount, LongType peakBytes) {
        if (!_enabled.load()) return;

        _totalAllocations.store(totalAllocations);
        _totalDeallocations.store(totalDeallocations);
        _totalBytesAllocated.store(totalBytesAllocated);
        _totalBytesDeallocated.store(totalBytesDeallocated);
        _peakLiveCount.store(peakLiveCount);
        _peakBytes.store(peakBytes);
    }

    /**
     * Get current live count (allocations - deallocations)
     */
    LongType getCurrentLiveCount() const {
        return _totalAllocations.load() - _totalDeallocations.load();
    }

    /**
     * Get current bytes in use
     */
    LongType getCurrentBytesInUse() const {
        return _totalBytesAllocated.load() - _totalBytesDeallocated.load();
    }

    /**
     * Get total allocations
     */
    LongType getTotalAllocations() const {
        return _totalAllocations.load();
    }

    /**
     * Get total deallocations
     */
    LongType getTotalDeallocations() const {
        return _totalDeallocations.load();
    }

    /**
     * Get peak live count
     */
    LongType getPeakLiveCount() const {
        return _peakLiveCount.load();
    }

    /**
     * Get peak bytes
     */
    LongType getPeakBytes() const {
        return _peakBytes.load();
    }

private:
    DeallocatorServiceLifecycleTracker() : _enabled(false),
        _totalAllocations(0), _totalDeallocations(0),
        _totalBytesAllocated(0), _totalBytesDeallocated(0),
        _peakLiveCount(0), _peakBytes(0) {}
    ~DeallocatorServiceLifecycleTracker() = default;

    // Disable copy and move
    DeallocatorServiceLifecycleTracker(const DeallocatorServiceLifecycleTracker&) = delete;
    DeallocatorServiceLifecycleTracker& operator=(const DeallocatorServiceLifecycleTracker&) = delete;
    DeallocatorServiceLifecycleTracker(DeallocatorServiceLifecycleTracker&&) = delete;
    DeallocatorServiceLifecycleTracker& operator=(DeallocatorServiceLifecycleTracker&&) = delete;

    std::atomic<bool> _enabled;
    std::atomic<LongType> _totalAllocations;
    std::atomic<LongType> _totalDeallocations;
    std::atomic<LongType> _totalBytesAllocated;
    std::atomic<LongType> _totalBytesDeallocated;
    std::atomic<LongType> _peakLiveCount;
    std::atomic<LongType> _peakBytes;
};

} // namespace array
} // namespace sd

#endif // LIBND4J_DEALLOCATORSERVICELIFECYCLETRACKER_H
