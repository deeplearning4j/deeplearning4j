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
// @author Adam Gibson
//
// Data Transfer Manager - P2P and host-device transfer handling for model parallelism
//

#ifndef SD_DATA_TRANSFER_MANAGER_H
#define SD_DATA_TRANSFER_MANAGER_H

#include <system/common.h>
#include <array/NDArray.h>
#include <execution/ModelParallelConfig.h>
#include <execution/DeviceManager.h>

#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <atomic>
#include <future>
#include <functional>
#include <condition_variable>

namespace sd {
namespace modelparallel {

/**
 * Transfer direction
 */
enum class TransferDirection {
    HOST_TO_DEVICE = 0,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    HOST_TO_HOST
};

/**
 * Transfer priority
 */
enum class TransferPriority {
    LOW = 0,
    NORMAL,
    HIGH,
    CRITICAL
};

/**
 * Transfer status
 */
enum class TransferStatus {
    PENDING = 0,
    IN_PROGRESS,
    COMPLETED,
    FAILED,
    CANCELLED
};

/**
 * Transfer request descriptor
 */
struct SD_LIB_EXPORT TransferRequest {
    // Identification
    uint64_t transferId = 0;
    std::string name;                 // Optional descriptive name

    // Source and destination
    void* srcPtr = nullptr;
    void* dstPtr = nullptr;
    size_t bytes = 0;

    int srcDevice = -1;               // -1 for host
    int dstDevice = -1;               // -1 for host

    // Transfer options
    TransferDirection direction = TransferDirection::HOST_TO_DEVICE;
    TransferPriority priority = TransferPriority::NORMAL;
    bool async = true;                // Asynchronous transfer
    bool blocking = false;            // Block until complete

    // Callbacks
    std::function<void(uint64_t, TransferStatus)> callback;

    // Status tracking
    TransferStatus status = TransferStatus::PENDING;
    std::string errorMessage;
    double startTime = 0.0;
    double endTime = 0.0;
    double bandwidth = 0.0;           // Measured bandwidth in GB/s
};

/**
 * Transfer result
 */
struct SD_LIB_EXPORT TransferResult {
    bool success = false;
    uint64_t transferId = 0;
    double durationMs = 0.0;
    double bandwidthGBps = 0.0;
    std::string errorMessage;
};

/**
 * Batch transfer result
 */
struct SD_LIB_EXPORT BatchTransferResult {
    bool allSucceeded = false;
    int successCount = 0;
    int failCount = 0;
    double totalDurationMs = 0.0;
    std::vector<TransferResult> results;
};

/**
 * Transfer statistics
 */
struct SD_LIB_EXPORT TransferStats {
    uint64_t totalTransfers = 0;
    uint64_t successfulTransfers = 0;
    uint64_t failedTransfers = 0;
    size_t totalBytesTransferred = 0;
    double totalTimeMs = 0.0;
    double averageBandwidthGBps = 0.0;
    double peakBandwidthGBps = 0.0;

    // Per-direction stats
    std::unordered_map<TransferDirection, uint64_t> transfersByDirection;
    std::unordered_map<TransferDirection, size_t> bytesByDirection;
};

/**
 * Ring buffer for P2P communication
 */
class SD_LIB_EXPORT RingBuffer {
public:
    RingBuffer(size_t capacity, int numParticipants);
    ~RingBuffer();

    /**
     * Get buffer pointer for a participant
     */
    void* getBuffer(int rank);

    /**
     * Get capacity per participant
     */
    size_t getCapacity() const { return _capacityPerParticipant; }

    /**
     * Check if buffers are allocated
     */
    bool isAllocated() const { return _allocated; }

    /**
     * Synchronize ring buffer
     */
    void synchronize();

private:
    size_t _capacityPerParticipant;
    int _numParticipants;
    bool _allocated = false;
    std::vector<void*> _buffers;
    std::vector<int> _deviceIds;
};

/**
 * Data Transfer Manager - Handles all data movement for model parallelism
 *
 * This class provides:
 * - Synchronous and asynchronous transfers
 * - P2P (peer-to-peer) direct GPU-to-GPU transfers
 * - Host-device transfers
 * - Batched and pipelined transfers
 * - Transfer scheduling and prioritization
 * - Bandwidth tracking and optimization
 */
class SD_LIB_EXPORT DataTransferManager {
public:
    /**
     * Get the singleton instance
     */
    static DataTransferManager& getInstance();

    /**
     * Initialize the transfer manager
     */
    bool initialize(const ModelParallelConfig& config = ModelParallelConfig());

    /**
     * Shutdown and release resources
     */
    void shutdown();

    /**
     * Check if initialized
     */
    bool isInitialized() const { return _initialized.load(); }

    // =========================================================================
    // Basic Transfers
    // =========================================================================

    /**
     * Transfer data between devices/host
     * @param srcPtr Source pointer
     * @param dstPtr Destination pointer
     * @param bytes Number of bytes to transfer
     * @param srcDevice Source device (-1 for host)
     * @param dstDevice Destination device (-1 for host)
     * @param async If true, return immediately
     */
    TransferResult transfer(
        void* srcPtr,
        void* dstPtr,
        size_t bytes,
        int srcDevice,
        int dstDevice,
        bool async = false
    );

    /**
     * Copy NDArray to another device
     */
    TransferResult copyToDevice(
        const NDArray* src,
        NDArray* dst,
        int targetDevice,
        bool async = false
    );

    /**
     * Copy NDArray from device to host
     */
    TransferResult copyToHost(
        const NDArray* src,
        NDArray* dst,
        bool async = false
    );

    /**
     * Copy NDArray from host to device
     */
    TransferResult copyFromHost(
        const NDArray* src,
        NDArray* dst,
        int targetDevice,
        bool async = false
    );

    // =========================================================================
    // P2P Transfers
    // =========================================================================

    /**
     * Direct GPU-to-GPU transfer using P2P
     */
    TransferResult p2pTransfer(
        void* srcPtr,
        void* dstPtr,
        size_t bytes,
        int srcDevice,
        int dstDevice,
        bool async = false
    );

    /**
     * P2P copy of NDArray
     */
    TransferResult p2pCopy(
        const NDArray* src,
        NDArray* dst,
        int srcDevice,
        int dstDevice,
        bool async = false
    );

    /**
     * Check if P2P is available between devices
     */
    bool isP2PAvailable(int device1, int device2) const;

    /**
     * Enable P2P between devices
     */
    bool enableP2P(int device1, int device2);

    /**
     * Disable P2P between devices
     */
    void disableP2P(int device1, int device2);

    /**
     * Get P2P bandwidth between devices
     */
    float getP2PBandwidth(int device1, int device2) const;

    // =========================================================================
    // Batch Transfers
    // =========================================================================

    /**
     * Submit a batch of transfer requests
     */
    BatchTransferResult submitBatch(std::vector<TransferRequest>& requests);

    /**
     * Copy multiple NDArrays to a device
     */
    BatchTransferResult copyBatchToDevice(
        const std::vector<NDArray*>& sources,
        std::vector<NDArray*>& destinations,
        int targetDevice,
        bool async = false
    );

    /**
     * Scatter data to multiple devices
     */
    BatchTransferResult scatter(
        const NDArray* source,
        std::vector<NDArray*>& destinations,
        const std::vector<int>& devices
    );

    /**
     * Gather data from multiple devices
     */
    TransferResult gatherFrom(
        const std::vector<NDArray*>& sources,
        NDArray* destination,
        const std::vector<int>& sourceDevices,
        int targetDevice
    );

    // =========================================================================
    // Collective Operations
    // =========================================================================

    /**
     * All-reduce across devices
     */
    BatchTransferResult allReduce(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices,
        CommunicationPattern pattern = CommunicationPattern::RING
    );

    /**
     * All-gather across devices
     */
    BatchTransferResult allGather(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices
    );

    /**
     * Reduce-scatter across devices
     */
    BatchTransferResult reduceScatter(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices
    );

    /**
     * Broadcast from one device to all others
     */
    BatchTransferResult broadcast(
        NDArray* source,
        std::vector<NDArray*>& destinations,
        int sourceDevice,
        const std::vector<int>& targetDevices
    );

    // =========================================================================
    // Ring Communication
    // =========================================================================

    /**
     * Create a ring buffer for ring-based communication
     */
    std::shared_ptr<RingBuffer> createRingBuffer(
        size_t capacity,
        const std::vector<int>& devices
    );

    /**
     * Ring all-reduce using ring buffer
     */
    BatchTransferResult ringAllReduce(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices,
        std::shared_ptr<RingBuffer> ringBuffer = nullptr
    );

    /**
     * Ring all-gather using ring buffer
     */
    BatchTransferResult ringAllGather(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices,
        std::shared_ptr<RingBuffer> ringBuffer = nullptr
    );

    // =========================================================================
    // Async Operations
    // =========================================================================

    /**
     * Submit an async transfer request
     */
    uint64_t submitAsync(TransferRequest& request);

    /**
     * Wait for a specific transfer to complete
     */
    TransferResult waitFor(uint64_t transferId, int timeoutMs = -1);

    /**
     * Wait for all pending transfers
     */
    void waitAll();

    /**
     * Check if a transfer is complete
     */
    bool isComplete(uint64_t transferId) const;

    /**
     * Cancel a pending transfer
     */
    bool cancel(uint64_t transferId);

    /**
     * Get number of pending transfers
     */
    int getPendingCount() const;

    // =========================================================================
    // Synchronization
    // =========================================================================

    /**
     * Synchronize a specific device
     */
    void synchronizeDevice(int deviceId);

    /**
     * Synchronize all devices
     */
    void synchronizeAll();

    /**
     * Create a synchronization barrier for devices
     */
    void barrier(const std::vector<int>& devices);

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Allocate pinned host memory for faster transfers
     */
    void* allocatePinnedMemory(size_t bytes);

    /**
     * Free pinned host memory
     */
    void freePinnedMemory(void* ptr);

    /**
     * Allocate device memory
     */
    void* allocateDeviceMemory(int deviceId, size_t bytes);

    /**
     * Free device memory
     */
    void freeDeviceMemory(int deviceId, void* ptr);

    /**
     * Get a staging buffer for transfers
     */
    void* getStagingBuffer(size_t minBytes, int deviceId = -1);

    /**
     * Release a staging buffer
     */
    void releaseStagingBuffer(void* ptr);

    // =========================================================================
    // Statistics and Monitoring
    // =========================================================================

    /**
     * Get transfer statistics
     */
    TransferStats getStats() const;

    /**
     * Reset statistics
     */
    void resetStats();

    /**
     * Get estimated transfer time
     */
    double estimateTransferTime(size_t bytes, int srcDevice, int dstDevice) const;

    /**
     * Benchmark transfer bandwidth between devices
     */
    float benchmarkBandwidth(int srcDevice, int dstDevice, size_t bytes = 64 * 1024 * 1024);

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * Set maximum concurrent transfers
     */
    void setMaxConcurrentTransfers(int count);

    /**
     * Set staging buffer size
     */
    void setStagingBufferSize(size_t bytes);

    /**
     * Enable/disable bandwidth tracking
     */
    void setBandwidthTracking(bool enabled);

    /**
     * Get current configuration
     */
    const ModelParallelConfig& getConfig() const { return _config; }

private:
    DataTransferManager();
    ~DataTransferManager();

    // Prevent copying
    DataTransferManager(const DataTransferManager&) = delete;
    DataTransferManager& operator=(const DataTransferManager&) = delete;

    // Internal transfer methods
    TransferResult doSyncTransfer(TransferRequest& request);
    void doAsyncTransfer(TransferRequest request);
    void processTransferQueue();
    void updateStats(const TransferRequest& request, TransferStatus status);

    // P2P helpers
    void initializeP2PConnections();
    bool setupP2PAccess(int device1, int device2);

    // Ring communication helpers
    void ringStep(
        std::vector<NDArray*>& tensors,
        const std::vector<int>& devices,
        int step,
        bool reduce
    );

    // Member variables
    std::atomic<bool> _initialized{false};
    std::atomic<bool> _shutdownRequested{false};
    ModelParallelConfig _config;

    // Transfer queue
    mutable std::mutex _queueMutex;
    std::condition_variable _queueCondition;
    std::queue<TransferRequest> _transferQueue;
    std::unordered_map<uint64_t, TransferRequest> _activeTransfers;
    std::atomic<uint64_t> _nextTransferId{1};

    // Worker thread
    std::unique_ptr<std::thread> _workerThread;

    // P2P state
    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 16);
        }
    };

    mutable std::mutex _p2pMutex;
    std::unordered_map<std::pair<int,int>, bool, PairHash> _p2pEnabled;
    std::unordered_map<std::pair<int,int>, float, PairHash> _p2pBandwidth;

    // Staging buffers
    mutable std::mutex _stagingMutex;
    std::vector<std::pair<void*, size_t>> _stagingBuffers;
    std::vector<bool> _stagingInUse;

    // Statistics
    mutable std::mutex _statsMutex;
    TransferStats _stats;
    bool _trackBandwidth = true;

    // Configuration
    int _maxConcurrentTransfers = 4;
    size_t _stagingBufferSize = 64 * 1024 * 1024;  // 64MB
};

/**
 * RAII wrapper for async transfer
 */
class SD_LIB_EXPORT AsyncTransfer {
public:
    AsyncTransfer(
        void* srcPtr,
        void* dstPtr,
        size_t bytes,
        int srcDevice,
        int dstDevice
    );

    ~AsyncTransfer();

    /**
     * Wait for transfer to complete
     */
    TransferResult wait(int timeoutMs = -1);

    /**
     * Check if transfer is complete
     */
    bool isComplete() const;

    /**
     * Get transfer ID
     */
    uint64_t getId() const { return _transferId; }

private:
    uint64_t _transferId;
    bool _completed = false;
};

/**
 * Helper function to get the transfer manager
 */
inline DataTransferManager& getTransferManager() {
    return DataTransferManager::getInstance();
}

/**
 * Convenience function for device-to-device copy
 */
inline TransferResult copyD2D(
    const NDArray* src,
    NDArray* dst,
    int srcDevice,
    int dstDevice,
    bool async = false
) {
    return DataTransferManager::getInstance().p2pCopy(src, dst, srcDevice, dstDevice, async);
}

/**
 * Convenience function for host-to-device copy
 */
inline TransferResult copyH2D(
    const NDArray* src,
    NDArray* dst,
    int device,
    bool async = false
) {
    return DataTransferManager::getInstance().copyFromHost(src, dst, device, async);
}

/**
 * Convenience function for device-to-host copy
 */
inline TransferResult copyD2H(
    const NDArray* src,
    NDArray* dst,
    bool async = false
) {
    return DataTransferManager::getInstance().copyToHost(src, dst, async);
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_DATA_TRANSFER_MANAGER_H
