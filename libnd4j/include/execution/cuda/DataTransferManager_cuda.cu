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

#ifdef SD_CUDA

#include <execution/DataTransferManager.h>
#include <execution/DeviceManager.h>
#include <execution/LaunchContext.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>

namespace sd {
namespace modelparallel {

RingBuffer::~RingBuffer() {
    for (int i = 0; i < _numParticipants; ++i) {
        if (_buffers[i] != nullptr) {
            if (_deviceIds[i] >= 0) {
                cudaSetDevice(_deviceIds[i]);
                cudaFree(_buffers[i]);
            } else {
                free(_buffers[i]);
            }
        }
    }
}

void RingBuffer::synchronize() {
    auto* stream = LaunchContext::defaultContext()->getCudaStream();
    if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
    }
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
        cudaMallocHost(&buffer, _stagingBufferSize);
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
            cudaFreeHost(buffer);
        }
    }
    _stagingBuffers.clear();
    _stagingInUse.clear();

    _initialized.store(false);
}

void DataTransferManager::initializeP2PConnections() {
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
}

TransferResult DataTransferManager::doSyncTransfer(TransferRequest& request) {
    TransferResult result;
    result.transferId = request.transferId;

    auto startTime = std::chrono::high_resolution_clock::now();

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
        auto* stream = LaunchContext::defaultContext()->getCudaStream();
        if (stream != nullptr) {
            cudaStreamSynchronize(*stream);
        }
    }

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
    // Get device from CUDA pointer
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, srcNonConst->buffer()) == cudaSuccess) {
        srcDevice = attrs.device;
    }
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
        auto* stream = LaunchContext::defaultContext()->getCudaStream();
        if (stream != nullptr) {
            cudaStreamSynchronize(*stream);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    result.bandwidthGBps = (bytes / 1e9) / (result.durationMs / 1000.0);
    result.success = true;

    return result;
}

void DataTransferManager::waitAll() {
    auto* stream = LaunchContext::defaultContext()->getCudaStream();
    if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
    }
}

void DataTransferManager::synchronizeDevice(int deviceId) {
    auto* stream = LaunchContext::defaultContext()->getCudaStream();
    if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
    }
}

void DataTransferManager::barrier(const std::vector<int>& devices) {
    auto* stream = LaunchContext::defaultContext()->getCudaStream();
    if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
    }
}

void* DataTransferManager::allocatePinnedMemory(size_t bytes) {
    void* ptr = nullptr;
    cudaMallocHost(&ptr, bytes);
    return ptr;
}

void DataTransferManager::freePinnedMemory(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void* DataTransferManager::allocateDeviceMemory(int deviceId, size_t bytes) {
    cudaSetDevice(deviceId);
    return memory::CudaMemoryPool::getInstance().allocate(bytes, deviceId);
}

void DataTransferManager::freeDeviceMemory(int deviceId, void* ptr) {
    if (ptr) {
        cudaSetDevice(deviceId);
        memory::CudaMemoryPool::getInstance().free(ptr, deviceId);
    }
}

float DataTransferManager::benchmarkBandwidth(int srcDevice, int dstDevice, size_t bytes) {
    // Allocate test buffers
    void* srcPtr = nullptr;
    void* dstPtr = nullptr;

    auto& pool = memory::CudaMemoryPool::getInstance();
    if (srcDevice >= 0) {
        cudaSetDevice(srcDevice);
        srcPtr = pool.allocate(bytes, srcDevice);
    } else {
        cudaMallocHost(&srcPtr, bytes);
    }

    if (dstDevice >= 0) {
        cudaSetDevice(dstDevice);
        dstPtr = pool.allocate(bytes, dstDevice);
    } else {
        cudaMallocHost(&dstPtr, bytes);
    }

    if (!srcPtr || !dstPtr) {
        if (srcPtr) { if (srcDevice >= 0) pool.free(srcPtr, srcDevice); else cudaFreeHost(srcPtr); }
        if (dstPtr) { if (dstDevice >= 0) pool.free(dstPtr, dstDevice); else cudaFreeHost(dstPtr); }
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
    if (srcDevice >= 0) pool.free(srcPtr, srcDevice); else cudaFreeHost(srcPtr);
    if (dstDevice >= 0) pool.free(dstPtr, dstDevice); else cudaFreeHost(dstPtr);

    // Calculate bandwidth
    float bandwidth = static_cast<float>((bytes * numIterations) / 1e9) /
                      static_cast<float>(totalMs / 1000.0);

    // Cache the result
    {
        std::lock_guard<std::mutex> lock(_p2pMutex);
        _p2pBandwidth[{srcDevice, dstDevice}] = bandwidth;
    }

    return bandwidth;
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_CUDA
