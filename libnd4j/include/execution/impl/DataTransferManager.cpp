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

#include <chrono>
#include <cstring>
#include <algorithm>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

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

RingBuffer::~RingBuffer() {
    for (int i = 0; i < _numParticipants; ++i) {
        if (_buffers[i] != nullptr) {
#ifdef SD_CUDA
            if (_deviceIds[i] >= 0) {
                cudaSetDevice(_deviceIds[i]);
                cudaFree(_buffers[i]);
            } else {
                free(_buffers[i]);
            }
#else
            free(_buffers[i]);
#endif
        }
    }
}

void* RingBuffer::getBuffer(int rank) {
    if (rank < 0 || rank >= _numParticipants) {
        return nullptr;
    }
    return _buffers[rank];
}

void RingBuffer::synchronize() {
#ifdef SD_CUDA
    for (int i = 0; i < _numParticipants; ++i) {
        if (_deviceIds[i] >= 0) {
            cudaSetDevice(_deviceIds[i]);
            cudaDeviceSynchronize();
        }
    }
#endif
}

// DataTransferManager implementation
DataTransferManager& DataTransferManager::getInstance() {
    static DataTransferManager instance;
    return instance;
}

DataTransferManager::DataTransferManager() {
}

DataTransferManager::~DataTransferManager() {
    shutdown();
}

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
#ifdef SD_CUDA
        cudaMallocHost(&buffer, _stagingBufferSize);
#else
        buffer = malloc(_stagingBufferSize);
#endif
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
#ifdef SD_CUDA
            cudaFreeHost(buffer);
#else
            free(buffer);
#endif
        }
    }
    _stagingBuffers.clear();
    _stagingInUse.clear();

    _initialized.store(false);
}

void DataTransferManager::initializeP2PConnections() {
#ifdef SD_CUDA
    auto& dm = DeviceManager::getInstance();
    int gpuCount = dm.getCudaGpuCount();

    for (int i = 0; i < gpuCount; ++i) {
        for (int j = i + 1; j < gpuCount; ++j) {
            if (dm.supportsP2P(i, j)) {
                if (dm.enableP2P(i, j)) {
                    std::lock_guard<std::mutex> lock(_p2pMutex);
                    _p2pEnabled[{i, j}] = true;
                    _p2pEnabled[{j, i}] = true;
                }
            }
        }
    }
#endif
}

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

TransferResult DataTransferManager::doSyncTransfer(TransferRequest& request) {
    TransferResult result;
    result.transferId = request.transferId;

    auto startTime = std::chrono::high_resolution_clock::now();

#ifdef SD_CUDA
    cudaError_t err = cudaSuccess;

    switch (request.direction) {
        case TransferDirection::HOST_TO_DEVICE:
            cudaSetDevice(request.dstDevice);
            if (request.async) {
                err = cudaMemcpyAsync(request.dstPtr, request.srcPtr, request.bytes,
                                      cudaMemcpyHostToDevice);
            } else {
                err = cudaMemcpy(request.dstPtr, request.srcPtr, request.bytes,
                                 cudaMemcpyHostToDevice);
            }
            break;

        case TransferDirection::DEVICE_TO_HOST:
            cudaSetDevice(request.srcDevice);
            if (request.async) {
                err = cudaMemcpyAsync(request.dstPtr, request.srcPtr, request.bytes,
                                      cudaMemcpyDeviceToHost);
            } else {
                err = cudaMemcpy(request.dstPtr, request.srcPtr, request.bytes,
                                 cudaMemcpyDeviceToHost);
            }
            break;

        case TransferDirection::DEVICE_TO_DEVICE:
            if (isP2PAvailable(request.srcDevice, request.dstDevice)) {
                // Use P2P transfer
                cudaSetDevice(request.srcDevice);
                if (request.async) {
                    err = cudaMemcpyPeerAsync(request.dstPtr, request.dstDevice,
                                              request.srcPtr, request.srcDevice,
                                              request.bytes);
                } else {
                    err = cudaMemcpyPeer(request.dstPtr, request.dstDevice,
                                         request.srcPtr, request.srcDevice,
                                         request.bytes);
                }
            } else {
                // Fall back to staging through host
                void* staging = getStagingBuffer(request.bytes);
                if (staging) {
                    cudaSetDevice(request.srcDevice);
                    err = cudaMemcpy(staging, request.srcPtr, request.bytes, cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        cudaSetDevice(request.dstDevice);
                        err = cudaMemcpy(request.dstPtr, staging, request.bytes, cudaMemcpyHostToDevice);
                    }
                    releaseStagingBuffer(staging);
                } else {
                    result.success = false;
                    result.errorMessage = "Failed to get staging buffer for D2D transfer";
                    return result;
                }
            }
            break;

        case TransferDirection::HOST_TO_HOST:
            std::memcpy(request.dstPtr, request.srcPtr, request.bytes);
            break;
    }

    if (err != cudaSuccess) {
        result.success = false;
        result.errorMessage = cudaGetErrorString(err);
        return result;
    }

    if (!request.async) {
        cudaDeviceSynchronize();
    }
#else
    // CPU-only path
    std::memcpy(request.dstPtr, request.srcPtr, request.bytes);
#endif

    auto endTime = std::chrono::high_resolution_clock::now();
    result.durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    result.bandwidthGBps = (request.bytes / 1e9) / (result.durationMs / 1000.0);
    result.success = true;

    // Update stats
    updateStats(request, TransferStatus::COMPLETED);

    return result;
}

TransferResult DataTransferManager::copyToDevice(
    const NDArray* src,
    NDArray* dst,
    int targetDevice,
    bool async
) {
    return transfer(
        src->buffer(),
        dst->buffer(),
        src->lengthOf() * DataTypeUtils::sizeOf(src->dataType()),
        -1,  // Host
        targetDevice,
        async
    );
}

TransferResult DataTransferManager::copyToHost(
    const NDArray* src,
    NDArray* dst,
    bool async
) {
    int srcDevice = 0;  // Would need device tracking in NDArray
#ifdef SD_CUDA
    // Get device from CUDA pointer
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, src->buffer()) == cudaSuccess) {
        srcDevice = attrs.device;
    }
#endif
    return transfer(
        src->buffer(),
        dst->buffer(),
        src->lengthOf() * DataTypeUtils::sizeOf(src->dataType()),
        srcDevice,
        -1,  // Host
        async
    );
}

TransferResult DataTransferManager::copyFromHost(
    const NDArray* src,
    NDArray* dst,
    int targetDevice,
    bool async
) {
    return transfer(
        src->buffer(),
        dst->buffer(),
        src->lengthOf() * DataTypeUtils::sizeOf(src->dataType()),
        -1,  // Host
        targetDevice,
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

#ifdef SD_CUDA
    auto startTime = std::chrono::high_resolution_clock::now();

    cudaError_t err;
    if (async) {
        err = cudaMemcpyPeerAsync(dstPtr, dstDevice, srcPtr, srcDevice, bytes);
    } else {
        err = cudaMemcpyPeer(dstPtr, dstDevice, srcPtr, srcDevice, bytes);
    }

    if (err != cudaSuccess) {
        result.success = false;
        result.errorMessage = cudaGetErrorString(err);
        return result;
    }

    if (!async) {
        cudaDeviceSynchronize();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    result.bandwidthGBps = (bytes / 1e9) / (result.durationMs / 1000.0);
    result.success = true;
#else
    result.success = false;
    result.errorMessage = "P2P requires CUDA";
#endif

    return result;
}

TransferResult DataTransferManager::p2pCopy(
    const NDArray* src,
    NDArray* dst,
    int srcDevice,
    int dstDevice,
    bool async
) {
    return p2pTransfer(
        src->buffer(),
        dst->buffer(),
        src->lengthOf() * DataTypeUtils::sizeOf(src->dataType()),
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

void DataTransferManager::waitAll() {
#ifdef SD_CUDA
    auto& dm = DeviceManager::getInstance();
    int gpuCount = dm.getCudaGpuCount();
    for (int i = 0; i < gpuCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
#endif
}

void DataTransferManager::synchronizeDevice(int deviceId) {
#ifdef SD_CUDA
    cudaSetDevice(deviceId);
    cudaDeviceSynchronize();
#endif
}

void DataTransferManager::synchronizeAll() {
    waitAll();
}

void DataTransferManager::barrier(const std::vector<int>& devices) {
#ifdef SD_CUDA
    for (int device : devices) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
#endif
}

void* DataTransferManager::allocatePinnedMemory(size_t bytes) {
    void* ptr = nullptr;
#ifdef SD_CUDA
    cudaMallocHost(&ptr, bytes);
#else
    ptr = malloc(bytes);
#endif
    return ptr;
}

void DataTransferManager::freePinnedMemory(void* ptr) {
    if (ptr) {
#ifdef SD_CUDA
        cudaFreeHost(ptr);
#else
        free(ptr);
#endif
    }
}

void* DataTransferManager::allocateDeviceMemory(int deviceId, size_t bytes) {
    void* ptr = nullptr;
#ifdef SD_CUDA
    cudaSetDevice(deviceId);
    cudaMalloc(&ptr, bytes);
#endif
    return ptr;
}

void DataTransferManager::freeDeviceMemory(int deviceId, void* ptr) {
#ifdef SD_CUDA
    if (ptr) {
        cudaSetDevice(deviceId);
        cudaFree(ptr);
    }
#endif
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

float DataTransferManager::benchmarkBandwidth(int srcDevice, int dstDevice, size_t bytes) {
    // Allocate test buffers
    void* srcPtr = nullptr;
    void* dstPtr = nullptr;

#ifdef SD_CUDA
    if (srcDevice >= 0) {
        cudaSetDevice(srcDevice);
        cudaMalloc(&srcPtr, bytes);
    } else {
        cudaMallocHost(&srcPtr, bytes);
    }

    if (dstDevice >= 0) {
        cudaSetDevice(dstDevice);
        cudaMalloc(&dstPtr, bytes);
    } else {
        cudaMallocHost(&dstPtr, bytes);
    }

    if (!srcPtr || !dstPtr) {
        if (srcPtr) srcDevice >= 0 ? cudaFree(srcPtr) : cudaFreeHost(srcPtr);
        if (dstPtr) dstDevice >= 0 ? cudaFree(dstPtr) : cudaFreeHost(dstPtr);
        return 0.0f;
    }

    // Warm up
    transfer(srcPtr, dstPtr, bytes, srcDevice, dstDevice, false);

    // Benchmark
    const int numIterations = 10;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        transfer(srcPtr, dstPtr, bytes, srcDevice, dstDevice, false);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Cleanup
    if (srcDevice >= 0) cudaFree(srcPtr); else cudaFreeHost(srcPtr);
    if (dstDevice >= 0) cudaFree(dstPtr); else cudaFreeHost(dstPtr);

    // Calculate bandwidth
    float bandwidth = static_cast<float>((bytes * numIterations) / 1e9) /
                      static_cast<float>(totalMs / 1000.0);

    // Cache the result
    {
        std::lock_guard<std::mutex> lock(_p2pMutex);
        _p2pBandwidth[{srcDevice, dstDevice}] = bandwidth;
    }

    return bandwidth;
#else
    return 0.0f;
#endif
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
