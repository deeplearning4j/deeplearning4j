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

#ifndef LIBND4J_TRANSFER_METRICS_H
#define LIBND4J_TRANSFER_METRICS_H

#include <system/common.h>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sd {

/**
 * Transfer type enumeration for tracking different kinds of data movements
 */
enum class TransferType {
    HOST_TO_DEVICE,      // CPU -> GPU
    DEVICE_TO_HOST,      // GPU -> CPU
    DEVICE_TO_DEVICE,    // GPU -> GPU (same device, e.g., buffer migration)
    PEER_TO_PEER         // GPU -> GPU (different devices via P2P or staged)
};

/**
 * Statistics for a specific transfer route (e.g., GPU0 -> GPU1)
 */
struct TransferRouteStats {
    std::atomic<uint64_t> transferCount{0};
    std::atomic<uint64_t> totalBytes{0};
    std::atomic<uint64_t> totalTimeNs{0};  // nanoseconds

    TransferRouteStats() = default;
    TransferRouteStats(const TransferRouteStats& other)
        : transferCount(other.transferCount.load())
        , totalBytes(other.totalBytes.load())
        , totalTimeNs(other.totalTimeNs.load()) {}
};

/**
 * TransferMetrics provides centralized tracking of data transfers between devices.
 * This is useful for profiling multi-GPU workloads and identifying bottlenecks.
 *
 * Thread-safe singleton pattern for global access.
 */
class SD_LIB_EXPORT TransferMetrics {
private:
    // Singleton instance
    static TransferMetrics* _instance;
    static std::mutex _instanceMutex;

    // Per-type aggregate stats
    std::atomic<uint64_t> _h2dCount{0};
    std::atomic<uint64_t> _h2dBytes{0};
    std::atomic<uint64_t> _h2dTimeNs{0};

    std::atomic<uint64_t> _d2hCount{0};
    std::atomic<uint64_t> _d2hBytes{0};
    std::atomic<uint64_t> _d2hTimeNs{0};

    std::atomic<uint64_t> _d2dCount{0};
    std::atomic<uint64_t> _d2dBytes{0};
    std::atomic<uint64_t> _d2dTimeNs{0};

    std::atomic<uint64_t> _p2pCount{0};
    std::atomic<uint64_t> _p2pBytes{0};
    std::atomic<uint64_t> _p2pTimeNs{0};

    // Per-route stats (e.g., "0->1" for GPU0 to GPU1)
    std::unordered_map<std::string, TransferRouteStats> _routeStats;
    std::mutex _routeMutex;

    // Total op execution time for overhead calculation
    std::atomic<uint64_t> _totalOpTimeNs{0};

    // Configuration
    std::atomic<bool> _enabled{true};
    std::atomic<bool> _logTransfers{false};
    std::atomic<uint64_t> _minBytesForLogging{1024 * 1024};  // 1MB default

    TransferMetrics() = default;

public:
    /**
     * Get the singleton instance
     */
    static TransferMetrics& getInstance();

    /**
     * Record a data transfer
     *
     * @param type The type of transfer
     * @param bytes Number of bytes transferred
     * @param timeNs Transfer time in nanoseconds
     * @param srcDevice Source device ID (-1 for host)
     * @param dstDevice Destination device ID (-1 for host)
     */
    void recordTransfer(TransferType type, uint64_t bytes, uint64_t timeNs,
                       int srcDevice = -1, int dstDevice = -1);

    /**
     * Record operation execution time (for overhead calculation)
     */
    void recordOpExecution(uint64_t timeNs);

    /**
     * Get total bytes transferred by type
     */
    uint64_t getTotalBytes(TransferType type) const;

    /**
     * Get total transfer count by type
     */
    uint64_t getTransferCount(TransferType type) const;

    /**
     * Get total transfer time in nanoseconds by type
     */
    uint64_t getTotalTimeNs(TransferType type) const;

    /**
     * Get average bandwidth in GB/s for a transfer type
     */
    double getAverageBandwidthGBps(TransferType type) const;

    /**
     * Get transfer overhead as percentage of total op time
     */
    double getOverheadPercent() const;

    /**
     * Get total bytes across all transfer types
     */
    uint64_t getTotalBytesAllTypes() const;

    /**
     * Get total time across all transfer types
     */
    uint64_t getTotalTimeNsAllTypes() const;

    /**
     * Reset all statistics
     */
    void reset();

    /**
     * Enable/disable metrics collection
     */
    void setEnabled(bool enabled);
    bool isEnabled() const;

    /**
     * Enable/disable transfer logging
     */
    void setLogTransfers(bool log);
    bool isLogTransfers() const;

    /**
     * Set minimum bytes for logging individual transfers
     */
    void setMinBytesForLogging(uint64_t bytes);

    /**
     * Print a summary of transfer statistics
     */
    void printSummary() const;

    /**
     * Get a formatted summary string
     */
    std::string getSummary() const;

    /**
     * Helper to create route key
     */
    static std::string makeRouteKey(int srcDevice, int dstDevice);
};

/**
 * RAII helper for timing transfers
 */
class TransferTimer {
private:
    TransferType _type;
    uint64_t _bytes;
    int _srcDevice;
    int _dstDevice;
    std::chrono::high_resolution_clock::time_point _start;
    bool _recorded;

public:
    TransferTimer(TransferType type, uint64_t bytes, int srcDevice = -1, int dstDevice = -1)
        : _type(type)
        , _bytes(bytes)
        , _srcDevice(srcDevice)
        , _dstDevice(dstDevice)
        , _start(std::chrono::high_resolution_clock::now())
        , _recorded(false) {}

    ~TransferTimer() {
        if (!_recorded) {
            record();
        }
    }

    void record() {
        if (_recorded) return;
        _recorded = true;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start);
        TransferMetrics::getInstance().recordTransfer(_type, _bytes, duration.count(),
                                                      _srcDevice, _dstDevice);
    }
};

}  // namespace sd

#endif  // LIBND4J_TRANSFER_METRICS_H
