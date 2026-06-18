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

// CPU-specific implementations for DataTransferManager.
// CUDA implementations are in execution/cuda/DataTransferManager.cu.

#include <execution/DataTransferManager.h>
#include <execution/DeviceManager.h>

#include <chrono>
#include <cstring>

namespace sd {
namespace modelparallel {

// RingBuffer CPU implementation
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

// DataTransferManager CPU implementation
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

}  // namespace modelparallel
}  // namespace sd
