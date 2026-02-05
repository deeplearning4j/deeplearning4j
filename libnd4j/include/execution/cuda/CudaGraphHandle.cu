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
// Contains: CudaGraphHandle class implementation
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
// CudaGraphHandle Implementation
// ============================================================================

CudaGraphHandle::CudaGraphHandle() : _deviceId(AffinityManager::currentDeviceId()) {}

CudaGraphHandle::CudaGraphHandle(int deviceId) : _deviceId(deviceId) {}

CudaGraphHandle::~CudaGraphHandle() {
    cleanup();
}

CudaGraphHandle::CudaGraphHandle(CudaGraphHandle&& other) noexcept
    : _graph(other._graph),
      _graphExec(other._graphExec),
      _state(other._state),
      _deviceId(other._deviceId),
      _stats(other._stats) {
    other._graph = nullptr;
    other._graphExec = nullptr;
    other._state = GraphState::EMPTY;
}

CudaGraphHandle& CudaGraphHandle::operator=(CudaGraphHandle&& other) noexcept {
    if (this != &other) {
        cleanup();
        _graph = other._graph;
        _graphExec = other._graphExec;
        _state = other._state;
        _deviceId = other._deviceId;
        _stats = other._stats;
        other._graph = nullptr;
        other._graphExec = nullptr;
        other._state = GraphState::EMPTY;
    }
    return *this;
}

void CudaGraphHandle::cleanup() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graphExec != nullptr) {
        cudaGraphExecDestroy(_graphExec);
        _graphExec = nullptr;
    }

    if (_graph != nullptr) {
        cudaGraphDestroy(_graph);
        _graph = nullptr;
    }

    _state = GraphState::EMPTY;
}

bool CudaGraphHandle::beginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state == GraphState::CAPTURING) {
        sd_print("CudaGraphHandle::beginCapture - Already capturing\n");
        return false;
    }

    // Set device
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    cudaError_t err = cudaStreamBeginCapture(stream, mode);

    // Restore device
    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
    }

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::beginCapture failed: %s\n", cudaGetErrorString(err));
        _state = GraphState::ERROR;
        return false;
    }

    _state = GraphState::CAPTURING;
    return true;
}

bool CudaGraphHandle::endCapture(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state != GraphState::CAPTURING) {
        sd_print("CudaGraphHandle::endCapture - Not currently capturing\n");
        return false;
    }

    // Set device
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    cudaError_t err = cudaStreamEndCapture(stream, &_graph);

    // Restore device
    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
    }

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::endCapture failed: %s\n", cudaGetErrorString(err));
        _state = GraphState::ERROR;
        return false;
    }

    if (_graph == nullptr) {
        sd_print("CudaGraphHandle::endCapture - Capture failed (null graph)\n");
        _state = GraphState::ERROR;
        return false;
    }

    _state = GraphState::CAPTURED;
    updateStatistics();
    return true;
}

bool CudaGraphHandle::instantiate() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state != GraphState::CAPTURED) {
        sd_print("CudaGraphHandle::instantiate - Graph not captured\n");
        return false;
    }

    // Set device
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    cudaGraphNode_t errorNode;
    char logBuffer[1024] = {0};

    cudaError_t err = cudaGraphInstantiate(&_graphExec, _graph, &errorNode, logBuffer, sizeof(logBuffer));

    // Restore device
    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
    }

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::instantiate failed: %s\n", cudaGetErrorString(err));
        if (strlen(logBuffer) > 0) {
            sd_printf("Graph instantiation log: %s\n", logBuffer);
        }
        _state = GraphState::ERROR;
        return false;
    }

    _state = GraphState::INSTANTIATED;
    return true;
}

bool CudaGraphHandle::updateFromGraph() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graphExec == nullptr || _graph == nullptr) {
        return false;
    }

    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;

    cudaError_t err = cudaGraphExecUpdate(_graphExec, _graph, &errorNode, &updateResult);

    if (err != cudaSuccess || updateResult != cudaGraphExecUpdateSuccess) {
        sd_print("CudaGraphHandle::updateFromGraph failed, need to re-instantiate\n");
        return instantiate();
    }

    return true;
}

bool CudaGraphHandle::launch(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state != GraphState::INSTANTIATED) {
        sd_print("CudaGraphHandle::launch - Graph not instantiated\n");
        return false;
    }

    _state = GraphState::EXECUTING;

    cudaError_t err = cudaGraphLaunch(_graphExec, stream);

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::launch failed: %s\n", cudaGetErrorString(err));
        _state = GraphState::ERROR;
        return false;
    }

    // Sync to ensure completion
    err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::launch sync failed: %s\n", cudaGetErrorString(err));
        _state = GraphState::ERROR;
        return false;
    }

    _state = GraphState::COMPLETED;
    return true;
}

bool CudaGraphHandle::launchAsync(cudaStream_t stream, cudaEvent_t completionEvent) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state != GraphState::INSTANTIATED && _state != GraphState::COMPLETED) {
        sd_print("CudaGraphHandle::launchAsync - Graph not ready\n");
        return false;
    }

    _state = GraphState::EXECUTING;

    cudaError_t err = cudaGraphLaunch(_graphExec, stream);

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::launchAsync failed: %s\n", cudaGetErrorString(err));
        _state = GraphState::ERROR;
        return false;
    }

    if (completionEvent != nullptr) {
        cudaEventRecord(completionEvent, stream);
    }

    _state = GraphState::INSTANTIATED;  // Ready for next launch
    return true;
}

size_t CudaGraphHandle::getNumNodes() const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) return 0;

    size_t numNodes;
    cudaError_t err = cudaGraphGetNodes(_graph, nullptr, &numNodes);

    if (err != cudaSuccess) return 0;
    return numNodes;
}

size_t CudaGraphHandle::getNumEdges() const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) return 0;

    size_t numEdges;
    cudaError_t err = cudaGraphGetEdges(_graph, nullptr, nullptr, &numEdges);

    if (err != cudaSuccess) return 0;
    return numEdges;
}

void CudaGraphHandle::updateStatistics() {
    if (_graph == nullptr) return;

    size_t numNodes = 0;
    cudaGraphGetNodes(_graph, nullptr, &numNodes);

    if (numNodes == 0) return;

    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(_graph, nodes.data(), &numNodes);

    _stats = GraphStatistics();

    for (const auto& node : nodes) {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(node, &nodeType);

        switch (nodeType) {
            case cudaGraphNodeTypeKernel:
                _stats.numKernels++;
                break;
            case cudaGraphNodeTypeMemcpy:
                // Could differentiate H2D, D2H, D2D but simplified here
                _stats.numMemcpyH2D++;
                _stats.totalMemoryOps++;
                break;
            case cudaGraphNodeTypeMemset:
                _stats.numMemsets++;
                _stats.totalMemoryOps++;
                break;
            case cudaGraphNodeTypeHost:
                _stats.numHostCallbacks++;
                break;
            case cudaGraphNodeTypeGraph:
                _stats.numChildGraphs++;
                break;
            case cudaGraphNodeTypeEventRecord:
            case cudaGraphNodeTypeWaitEvent:
                _stats.numEvents++;
                break;
            default:
                break;
        }
    }
}

GraphStatistics CudaGraphHandle::getStatistics() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _stats;
}

void CudaGraphHandle::printDebugInfo() const {
    std::lock_guard<std::mutex> lock(_mutex);

    sd_print("=== CUDA Graph Debug Info ===\n");
    sd_printf("Device ID: %d\n", _deviceId);
    sd_printf("State: %d\n", static_cast<int>(_state));
    sd_printf("Graph: %p\n", _graph);
    sd_printf("GraphExec: %p\n", _graphExec);
    sd_printf("Num Nodes: %zu\n", getNumNodes());
    sd_printf("Num Edges: %zu\n", getNumEdges());
    sd_printf("Kernels: %d\n", _stats.numKernels);
    sd_printf("Memory Ops: %zu\n", _stats.totalMemoryOps);
}

bool CudaGraphHandle::exportToDot(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) return false;

    cudaError_t err = cudaGraphDebugDotPrint(_graph, filename.c_str(), 0);
    return err == cudaSuccess;
}

}  // namespace cuda
}  // namespace sd

#endif  // SD_CUDA
