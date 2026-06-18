/* ******************************************************************************
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
// Split from CudaGraphScheduler.cu to reduce object file size for SD_GCC_FUNCTRACE builds
// Contains: MultiDeviceGraph class implementation
//

#include <execution/cuda/CudaGraphScheduler.h>

#ifdef SD_CUDA

#include <execution/AffinityManager.h>
#include <helpers/logger.h>
#include <exceptions/cuda_exception.h>

#include <chrono>
#include <algorithm>
#include <sstream>

namespace sd {
namespace cuda {

// ============================================================================
// MultiDeviceGraph Implementation
// ============================================================================

MultiDeviceGraph::MultiDeviceGraph() {}

MultiDeviceGraph::MultiDeviceGraph(const std::vector<int>& deviceIds) : _deviceIds(deviceIds) {
    for (int deviceId : _deviceIds) {
        _graphs[deviceId] = std::make_unique<CudaGraphHandle>(deviceId);
    }
}

MultiDeviceGraph::~MultiDeviceGraph() = default;

void MultiDeviceGraph::addDevice(int deviceId) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graphs.find(deviceId) == _graphs.end()) {
        _deviceIds.push_back(deviceId);
        _graphs[deviceId] = std::make_unique<CudaGraphHandle>(deviceId);
    }
}

void MultiDeviceGraph::removeDevice(int deviceId) {
    std::lock_guard<std::mutex> lock(_mutex);

    _graphs.erase(deviceId);
    _deviceIds.erase(
        std::remove(_deviceIds.begin(), _deviceIds.end(), deviceId),
        _deviceIds.end()
    );
}

bool MultiDeviceGraph::beginCaptureOnDevice(int deviceId, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _graphs.find(deviceId);
    if (it == _graphs.end()) {
        addDevice(deviceId);
        it = _graphs.find(deviceId);
    }

    return it->second->beginCapture(stream);
}

bool MultiDeviceGraph::endCaptureOnDevice(int deviceId, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _graphs.find(deviceId);
    if (it == _graphs.end()) return false;

    return it->second->endCapture(stream);
}

bool MultiDeviceGraph::beginCaptureAll(const std::vector<cudaStream_t>& streams) {
    if (streams.size() != _deviceIds.size()) return false;

    for (size_t i = 0; i < _deviceIds.size(); i++) {
        if (!beginCaptureOnDevice(_deviceIds[i], streams[i])) {
            // Abort all captures on failure
            for (size_t j = 0; j < i; j++) {
                cudaStreamEndCapture(streams[j], nullptr);
            }
            return false;
        }
    }
    return true;
}

bool MultiDeviceGraph::endCaptureAll(const std::vector<cudaStream_t>& streams) {
    if (streams.size() != _deviceIds.size()) return false;

    bool success = true;
    for (size_t i = 0; i < _deviceIds.size(); i++) {
        if (!endCaptureOnDevice(_deviceIds[i], streams[i])) {
            success = false;
        }
    }
    return success;
}

void MultiDeviceGraph::addDependency(int srcDevice, int dstDevice) {
    std::lock_guard<std::mutex> lock(_mutex);
    _dependencies.push_back({srcDevice, dstDevice});
}

void MultiDeviceGraph::addP2PTransfer(int srcDevice, int dstDevice, void* srcPtr, void* dstPtr, size_t bytes) {
    addDependency(srcDevice, dstDevice);
    // The actual P2P transfer would be captured as part of the graph
}

bool MultiDeviceGraph::instantiateAll() {
    std::lock_guard<std::mutex> lock(_mutex);

    for (auto it = _graphs.begin(); it != _graphs.end(); ++it) {
        if (!it->second->instantiate()) {
            return false;
        }
    }
    return true;
}

bool MultiDeviceGraph::launchAll(const std::vector<cudaStream_t>& streams) {
    if (streams.size() != _deviceIds.size()) return false;

    // Launch all graphs asynchronously
    for (size_t i = 0; i < _deviceIds.size(); i++) {
        auto it = _graphs.find(_deviceIds[i]);
        if (it != _graphs.end()) {
            if (!it->second->launchAsync(streams[i])) {
                return false;
            }
        }
    }

    // Synchronize all streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
    }

    return true;
}

bool MultiDeviceGraph::launchSequential(const std::vector<cudaStream_t>& streams) {
    if (streams.size() != _deviceIds.size()) return false;

    for (size_t i = 0; i < _deviceIds.size(); i++) {
        auto it = _graphs.find(_deviceIds[i]);
        if (it != _graphs.end()) {
            if (!it->second->launch(streams[i])) {
                return false;
            }
        }
    }

    return true;
}

GraphState MultiDeviceGraph::getStateForDevice(int deviceId) const {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _graphs.find(deviceId);
    if (it == _graphs.end()) return GraphState::EMPTY;
    return it->second->getState();
}

bool MultiDeviceGraph::allReady() const {
    std::lock_guard<std::mutex> lock(_mutex);

    for (auto it = _graphs.begin(); it != _graphs.end(); ++it) {
        if (it->second->getState() != GraphState::INSTANTIATED &&
            it->second->getState() != GraphState::COMPLETED) {
            return false;
        }
    }
    return true;
}

GraphStatistics MultiDeviceGraph::getAggregateStatistics() const {
    std::lock_guard<std::mutex> lock(_mutex);

    GraphStatistics aggregate;
    for (auto it = _graphs.begin(); it != _graphs.end(); ++it) {
        auto stats = it->second->getStatistics();
        aggregate.numKernels += stats.numKernels;
        aggregate.numMemcpyH2D += stats.numMemcpyH2D;
        aggregate.numMemcpyD2H += stats.numMemcpyD2H;
        aggregate.numMemcpyD2D += stats.numMemcpyD2D;
        aggregate.numMemsets += stats.numMemsets;
        aggregate.numEvents += stats.numEvents;
        aggregate.numHostCallbacks += stats.numHostCallbacks;
        aggregate.numChildGraphs += stats.numChildGraphs;
        aggregate.totalMemoryOps += stats.totalMemoryOps;
    }
    return aggregate;
}

}  // namespace cuda
}  // namespace sd

#endif  // SD_CUDA
