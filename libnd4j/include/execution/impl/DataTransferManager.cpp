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

#include <execution/DataTransferManager.h>
#include <execution/DeviceManager.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>

namespace sd {
namespace modelparallel {

// RingBuffer implementation
RingBuffer::RingBuffer(size_t capacity, int numParticipants)
    : _capacityPerParticipant(capacity)
    , _numParticipants(numParticipants)
    , _allocated(false) {

    _buffers.resize(numParticipants, nullptr);
    _deviceIds.resize(numParticipants, -1);
}

#ifndef SD_CUDA
RingBuffer::~RingBuffer() {
    for (int i = 0; i < _numParticipants; ++i) {
        if (_buffers[i] != nullptr) {
            free(_buffers[i]);
        }
    }
}

void RingBuffer::synchronize() {
    // No-op on CPU
}
#endif

void* RingBuffer::getBuffer(int rank) {
    if (rank < 0 || rank >= _numParticipants) {
        return nullptr;
    }
    return _buffers[rank];
}

// DataTransferManager implementation
DataTransferManager& DataTransferManager::getInstance() {
    static DataTransferManager* instance = nullptr;
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        instance = new DataTransferManager();
    });
    return *instance;
}

DataTransferManager::DataTransferManager() {
}

DataTransferManager::~DataTransferManager() {
    shutdown();
}

#ifndef SD_CUDA
bool DataTransferManager::initialize(const ModelParallelConfig& config) {
    std::lock_guard<std::mutex> lock(_queueMutex);

    if (_initialized.load()) {
        return true;
    }

    _config = config;
    _shutdownRequested.store(false);

    // Initialize P2P connections if enabled
    if (config.enableP2P) {
        initializeP2PConnections();
    }

    // Allocate staging buffers
    for (int i = 0; i < _maxConcurrentTransfers; ++i) {
        void* buffer = nullptr;
        buffer = malloc(_stagingBufferSize);
        if (buffer) {
            _stagingBuffers.push_back({buffer, _stagingBufferSize});
            _stagingInUse.push_back(false);
        }
    }

    _initialized.store(true);
    return true;
}

void DataTransferManager::shutdown() {
    _shutdownRequested.store(true);

    // Wait for pending transfers
    waitAll();

    std::lock_guard<std::mutex> lock(_queueMutex);

    // Free staging buffers
    for (auto& [buffer, size] : _stagingBuffers) {
        if (buffer) {
            free(buffer);
        }
    }
    _stagingBuffers.clear();
    _stagingInUse.clear();

    _initialized.store(false);
}

void DataTransferManager::initializeP2PConnections() {
    // No-op on CPU
}

TransferResult DataTransferManager::doSyncTransfer(TransferRequest& request) {
    TransferResult result;
    result.transferId = request.transferId;

    auto startTime = std::chrono::high_resolution_clock::now();

    // CPU-only path
    std::memcpy(request.dstPtr, request.srcPtr, request.bytes);

    auto endTime = std::chrono::high_resolution_clock::now();
    result.durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    result.bandwidthGBps = (request.bytes / 1e9) / (result.durationMs / 1000.0);
    result.success = true;

    // Update stats
    updateStats(request, TransferStatus::COMPLETED);

    return result;
}

TransferResult DataTransferManager::copyToHost(
    const NDArray* src,
    NDArray* dst,
    bool async
) {
    auto srcNonConst = const_cast<NDArray*>(src);
    int srcDevice = 0;  // Would need device tracking in NDArray
    return transfer(
        srcNonConst->buffer(),
        dst->buffer(),
        srcNonConst->lengthOf() * DataTypeUtils::sizeOf(srcNonConst->dataType()),
        srcDevice,
        -1,  // Host
        async
    );
}

TransferResult DataTransferManager::p2pTransfer(
    void* srcPtr,
    void* dstPtr,
    size_t bytes,
    int srcDevice,
    int dstDevice,
    bool async
) {
    TransferResult result;
    result.transferId = _nextTransferId.fetch_add(1);

    if (!isP2PAvailable(srcDevice, dstDevice)) {
        result.success = false;
        result.errorMessage = "P2P not available between devices";
        return result;
    }

    result.success = false;
    result.errorMessage = "P2P requires CUDA";

    return result;
}

void DataTransferManager::waitAll() {
    // No-op on CPU
}

void DataTransferManager::synchronizeDevice(int deviceId) {
    // No-op on CPU
}

void DataTransferManager::barrier(const std::vector<int>& devices) {
    // No-op on CPU
}

void* DataTransferManager::allocatePinnedMemory(size_t bytes) {
    void* ptr = nullptr;
    ptr = malloc(bytes);
    return ptr;
}

void DataTransferManager::freePinnedMemory(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

void* DataTransferManager::allocateDeviceMemory(int deviceId, size_t bytes) {
    void* ptr = nullptr;
    // No-op on CPU
    return ptr;
}

void DataTransferManager::freeDeviceMemory(int deviceId, void* ptr) {
    // No-op on CPU
}

float DataTransferManager::benchmarkBandwidth(int srcDevice, int dstDevice, size_t bytes) {
    return 0.0f;
}
#endif  // !SD_CUDA

TransferResult DataTransferManager::transfer(
    void* srcPtr,
    void* dstPtr,
    size_t bytes,
    int srcDevice,
    int dstDevice,
    bool async
) {
    TransferRequest request;
    request.transferId = _nextTransferId.fetch_add(1);
    request.srcPtr = srcPtr;
    request.dstPtr = dstPtr;
    request.bytes = bytes;
    request.srcDevice = srcDevice;
    request.dstDevice = dstDevice;
    request.async = async;

    // Determine direction
    if (srcDevice < 0 && dstDevice >= 0) {
        request.direction = TransferDirection::HOST_TO_DEVICE;
    } else if (srcDevice >= 0 && dstDevice < 0) {
        request.direction = TransferDirection::DEVICE_TO_HOST;
    } else if (srcDevice >= 0 && dstDevice >= 0) {
        request.direction = TransferDirection::DEVICE_TO_DEVICE;
    } else {
        request.direction = TransferDirection::HOST_TO_HOST;
    }

    return doSyncTransfer(request);
}

TransferResult DataTransferManager::copyToDevice(
    const NDArray* src,
    NDArray* dst,
    int targetDevice,
    bool async
) {
    auto srcNonConst = const_cast<NDArray*>(src);
    return transfer(
        srcNonConst->buffer(),
        dst->buffer(),
        srcNonConst->lengthOf() * DataTypeUtils::sizeOf(srcNonConst->dataType()),
        -1,  // Host
        targetDevice,
        async
    );
}

TransferResult DataTransferManager::copyFromHost(
    const NDArray* src,
    NDArray* dst,
    int targetDevice,
    bool async
) {
    auto srcNonConst = const_cast<NDArray*>(src);
    return transfer(
        srcNonConst->buffer(),
        dst->buffer(),
        srcNonConst->lengthOf() * DataTypeUtils::sizeOf(srcNonConst->dataType()),
        -1,  // Host
        targetDevice,
        async
    );
}

TransferResult DataTransferManager::p2pCopy(
    const NDArray* src,
    NDArray* dst,
    int srcDevice,
    int dstDevice,
    bool async
) {
    auto srcNonConst = const_cast<NDArray*>(src);
    return p2pTransfer(
        srcNonConst->buffer(),
        dst->buffer(),
        srcNonConst->lengthOf() * DataTypeUtils::sizeOf(srcNonConst->dataType()),
        srcDevice,
        dstDevice,
        async
    );
}

bool DataTransferManager::isP2PAvailable(int device1, int device2) const {
    std::lock_guard<std::mutex> lock(_p2pMutex);
    auto it = _p2pEnabled.find({device1, device2});
    return it != _p2pEnabled.end() && it->second;
}

bool DataTransferManager::enableP2P(int device1, int device2) {
    return DeviceManager::getInstance().enableP2P(device1, device2);
}

void DataTransferManager::disableP2P(int device1, int device2) {
    DeviceManager::getInstance().disableP2P(device1, device2);

    std::lock_guard<std::mutex> lock(_p2pMutex);
    _p2pEnabled[{device1, device2}] = false;
    _p2pEnabled[{device2, device1}] = false;
}

float DataTransferManager::getP2PBandwidth(int device1, int device2) const {
    std::lock_guard<std::mutex> lock(_p2pMutex);
    auto it = _p2pBandwidth.find({device1, device2});
    if (it != _p2pBandwidth.end()) {
        return it->second;
    }
    return 0.0f;
}

void DataTransferManager::synchronizeAll() {
    waitAll();
}

void* DataTransferManager::getStagingBuffer(size_t minBytes, int deviceId) {
    std::lock_guard<std::mutex> lock(_stagingMutex);

    for (size_t i = 0; i < _stagingBuffers.size(); ++i) {
        if (!_stagingInUse[i] && _stagingBuffers[i].second >= minBytes) {
            _stagingInUse[i] = true;
            return _stagingBuffers[i].first;
        }
    }

    // No available buffer, allocate a new one
    void* buffer = allocatePinnedMemory(minBytes);
    if (buffer) {
        _stagingBuffers.push_back({buffer, minBytes});
        _stagingInUse.push_back(true);
    }
    return buffer;
}

void DataTransferManager::releaseStagingBuffer(void* ptr) {
    std::lock_guard<std::mutex> lock(_stagingMutex);

    for (size_t i = 0; i < _stagingBuffers.size(); ++i) {
        if (_stagingBuffers[i].first == ptr) {
            _stagingInUse[i] = false;
            return;
        }
    }
}

TransferStats DataTransferManager::getStats() const {
    std::lock_guard<std::mutex> lock(_statsMutex);
    return _stats;
}

void DataTransferManager::resetStats() {
    std::lock_guard<std::mutex> lock(_statsMutex);
    _stats = TransferStats();
}

void DataTransferManager::updateStats(const TransferRequest& request, TransferStatus status) {
    std::lock_guard<std::mutex> lock(_statsMutex);

    _stats.totalTransfers++;

    if (status == TransferStatus::COMPLETED) {
        _stats.successfulTransfers++;
        _stats.totalBytesTransferred += request.bytes;

        double duration = request.endTime - request.startTime;
        if (duration > 0) {
            _stats.totalTimeMs += duration;
            double bandwidth = (request.bytes / 1e9) / (duration / 1000.0);
            if (bandwidth > _stats.peakBandwidthGBps) {
                _stats.peakBandwidthGBps = bandwidth;
            }
        }

        _stats.transfersByDirection[request.direction]++;
        _stats.bytesByDirection[request.direction] += request.bytes;
    } else {
        _stats.failedTransfers++;
    }

    if (_stats.totalTimeMs > 0) {
        _stats.averageBandwidthGBps = (_stats.totalBytesTransferred / 1e9) /
                                       (_stats.totalTimeMs / 1000.0);
    }
}

double DataTransferManager::estimateTransferTime(size_t bytes, int srcDevice, int dstDevice) const {
    // Rough estimates based on typical bandwidths
    double bandwidthGBps = 10.0;  // Default: ~10 GB/s

    if (srcDevice < 0 && dstDevice >= 0) {
        bandwidthGBps = 12.0;  // PCIe 3.0 x16: ~12 GB/s
    } else if (srcDevice >= 0 && dstDevice < 0) {
        bandwidthGBps = 12.0;
    } else if (srcDevice >= 0 && dstDevice >= 0) {
        if (isP2PAvailable(srcDevice, dstDevice)) {
            bandwidthGBps = 25.0;  // NVLink: ~25-50 GB/s
        } else {
            bandwidthGBps = 6.0;  // Staging through host
        }
    }

    return (bytes / 1e9) / bandwidthGBps * 1000.0;  // Return milliseconds
}

void DataTransferManager::setMaxConcurrentTransfers(int count) {
    _maxConcurrentTransfers = count;
}

void DataTransferManager::setStagingBufferSize(size_t bytes) {
    _stagingBufferSize = bytes;
}

void DataTransferManager::setBandwidthTracking(bool enabled) {
    _trackBandwidth = enabled;
}

// AsyncTransfer implementation
AsyncTransfer::AsyncTransfer(
    void* srcPtr,
    void* dstPtr,
    size_t bytes,
    int srcDevice,
    int dstDevice
) {
    TransferRequest request;
    request.srcPtr = srcPtr;
    request.dstPtr = dstPtr;
    request.bytes = bytes;
    request.srcDevice = srcDevice;
    request.dstDevice = dstDevice;
    request.async = true;

    _transferId = DataTransferManager::getInstance().submitAsync(request);
}

AsyncTransfer::~AsyncTransfer() {
    if (!_completed) {
        wait();
    }
}

TransferResult AsyncTransfer::wait(int timeoutMs) {
    auto result = DataTransferManager::getInstance().waitFor(_transferId, timeoutMs);
    _completed = true;
    return result;
}

bool AsyncTransfer::isComplete() const {
    return _completed || DataTransferManager::getInstance().isComplete(_transferId);
}

uint64_t DataTransferManager::submitAsync(TransferRequest& request) {
    request.transferId = _nextTransferId.fetch_add(1);
    request.async = true;

    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _activeTransfers[request.transferId] = request;
    }

    // For now, just do a sync transfer (full async would need worker thread)
    doSyncTransfer(request);

    return request.transferId;
}

TransferResult DataTransferManager::waitFor(uint64_t transferId, int timeoutMs) {
    std::lock_guard<std::mutex> lock(_queueMutex);

    auto it = _activeTransfers.find(transferId);
    if (it != _activeTransfers.end()) {
        TransferResult result;
        result.transferId = transferId;
        result.success = (it->second.status == TransferStatus::COMPLETED);
        _activeTransfers.erase(it);
        return result;
    }

    TransferResult result;
    result.success = false;
    result.errorMessage = "Transfer not found";
    return result;
}

bool DataTransferManager::isComplete(uint64_t transferId) const {
    std::lock_guard<std::mutex> lock(_queueMutex);
    auto it = _activeTransfers.find(transferId);
    return it == _activeTransfers.end() || it->second.status == TransferStatus::COMPLETED;
}

bool DataTransferManager::cancel(uint64_t transferId) {
    std::lock_guard<std::mutex> lock(_queueMutex);

    auto it = _activeTransfers.find(transferId);
    if (it != _activeTransfers.end() && it->second.status == TransferStatus::PENDING) {
        it->second.status = TransferStatus::CANCELLED;
        _activeTransfers.erase(it);
        return true;
    }
    return false;
}

int DataTransferManager::getPendingCount() const {
    std::lock_guard<std::mutex> lock(_queueMutex);
    int count = 0;
    for (const auto& [id, request] : _activeTransfers) {
        if (request.status == TransferStatus::PENDING ||
            request.status == TransferStatus::IN_PROGRESS) {
            count++;
        }
    }
    return count;
}

}  // namespace modelparallel
}  // namespace sd
