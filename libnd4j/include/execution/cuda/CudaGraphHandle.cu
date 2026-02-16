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
#include <cxxabi.h>  // For __cxa_demangle (kernel name demangling)

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

    // Free pinned host buffers that were used for H2D copies during capture.
    // These persist for graph lifetime so replay can read from them.
    for (auto* ptr : _capturedHostPtrs) {
        if (ptr != nullptr) cudaFreeHost(ptr);
    }
    _capturedHostPtrs.clear();

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
        // Clear sticky CUDA error state so subsequent operations aren't affected
        cudaGetLastError();
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

    // Clear any sticky error before launch (e.g., from updateStatistics graph queries)
    cudaGetLastError();

    cudaError_t err = cudaGraphLaunch(_graphExec, stream);

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::launchAsync failed: %s (err=%d)\n",
                  cudaGetErrorString(err), (int)err);
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
    auto err = cudaGraphGetNodes(_graph, nullptr, &numNodes);
    if (err != cudaSuccess) {
        cudaGetLastError();  // Clear sticky error
        return;
    }

    if (numNodes == 0) return;

    std::vector<cudaGraphNode_t> nodes(numNodes);
    err = cudaGraphGetNodes(_graph, nodes.data(), &numNodes);
    if (err != cudaSuccess) {
        cudaGetLastError();  // Clear sticky error
        return;
    }

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

// ============================================================================
// Detailed graph node introspection
// ============================================================================

static std::string nodeTypeName(cudaGraphNodeType type) {
    switch (type) {
        case cudaGraphNodeTypeKernel:       return "Kernel";
        case cudaGraphNodeTypeMemcpy:       return "Memcpy";
        case cudaGraphNodeTypeMemset:       return "Memset";
        case cudaGraphNodeTypeHost:         return "HostCallback";
        case cudaGraphNodeTypeGraph:        return "ChildGraph";
        case cudaGraphNodeTypeEmpty:        return "Empty";
        case cudaGraphNodeTypeEventRecord:  return "EventRecord";
        case cudaGraphNodeTypeWaitEvent:    return "WaitEvent";
        default:                            return "Unknown(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

static std::string memcpyKindName(cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToDevice:   return "H2D";
        case cudaMemcpyDeviceToHost:   return "D2H";
        case cudaMemcpyDeviceToDevice: return "D2D";
        case cudaMemcpyHostToHost:     return "H2H";
        case cudaMemcpyDefault:        return "Default";
        default:                       return "Unknown";
    }
}

static std::string demangleKernelName(const char* mangled) {
    if (mangled == nullptr || mangled[0] == '\0') return "<unknown>";

    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled != nullptr) {
        std::string result(demangled);
        free(demangled);
        // Truncate long template parameters for readability
        auto pos = result.find('<');
        if (pos != std::string::npos && result.size() > 120) {
            result = result.substr(0, pos) + "<...>";
        }
        return result;
    }
    // Demangling failed — return raw name
    return std::string(mangled);
}

std::vector<CudaGraphNodeInfo> CudaGraphHandle::getDetailedNodeInfo() const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<CudaGraphNodeInfo> result;

    if (_graph == nullptr) return result;

    size_t numNodes = 0;
    cudaError_t err = cudaGraphGetNodes(_graph, nullptr, &numNodes);
    if (err != cudaSuccess || numNodes == 0) return result;

    std::vector<cudaGraphNode_t> nodes(numNodes);
    err = cudaGraphGetNodes(_graph, nodes.data(), &numNodes);
    if (err != cudaSuccess) return result;

    result.reserve(numNodes);

    for (size_t i = 0; i < numNodes; i++) {
        CudaGraphNodeInfo info;
        info.nodeIndex = i;

        cudaGraphNodeType nodeType;
        err = cudaGraphNodeGetType(nodes[i], &nodeType);
        if (err != cudaSuccess) continue;

        info.type = nodeType;
        info.typeName = nodeTypeName(nodeType);

        switch (nodeType) {
            case cudaGraphNodeTypeKernel: {
                cudaKernelNodeParams params;
                memset(&params, 0, sizeof(params));
                err = cudaGraphKernelNodeGetParams(nodes[i], &params);
                if (err == cudaSuccess && params.func != nullptr) {
                    // Get kernel name from function pointer
                    const char* name = nullptr;
                    auto nameErr = cudaFuncGetName(&name, params.func);
                    if (nameErr == cudaSuccess && name != nullptr) {
                        info.kernelName = demangleKernelName(name);
                    } else {
                        // Fallback: show raw function pointer
                        std::ostringstream oss;
                        oss << "func@" << params.func;
                        info.kernelName = oss.str();
                    }
                }
                break;
            }

            case cudaGraphNodeTypeMemcpy: {
                // Use cudaGraphMemcpyNodeGetParams (CUDA 10+)
                cudaMemcpy3DParms mcpyParams;
                memset(&mcpyParams, 0, sizeof(mcpyParams));
                err = cudaGraphMemcpyNodeGetParams(nodes[i], &mcpyParams);
                if (err == cudaSuccess) {
                    info.memcpyBytes = mcpyParams.extent.width *
                                       std::max(mcpyParams.extent.height, (size_t)1) *
                                       std::max(mcpyParams.extent.depth, (size_t)1);
                    info.memcpyKind = memcpyKindName(mcpyParams.kind);
                }
                break;
            }

            case cudaGraphNodeTypeMemset: {
                cudaMemsetParams msParams;
                memset(&msParams, 0, sizeof(msParams));
                err = cudaGraphMemsetNodeGetParams(nodes[i], &msParams);
                if (err == cudaSuccess) {
                    info.memsetBytes = msParams.width * std::max(msParams.height, (size_t)1) *
                                       msParams.elementSize;
                    info.memsetValue = msParams.value;
                }
                break;
            }

            default:
                break;
        }

        result.push_back(std::move(info));
    }

    return result;
}

void CudaGraphHandle::printGraphContents() const {
    auto nodes = getDetailedNodeInfo();

    sd_print("╔══════════════════════════════════════════════════════════════════╗\n");
    sd_print("║              CUDA GRAPH CONTENTS                               ║\n");
    sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
    sd_printf("║ Device: %d  State: %d  Nodes: %zu  Edges: %zu\n",
              _deviceId, static_cast<int>(_state), nodes.size(), getNumEdges());
    sd_print("╠══════════════════════════════════════════════════════════════════╣\n");

    // Summary counts by type
    int kernelCount = 0, memcpyH2D = 0, memcpyD2H = 0, memcpyD2D = 0, memcpyOther = 0;
    int memsetCount = 0, hostCount = 0, childCount = 0, eventCount = 0, emptyCount = 0;

    for (const auto& n : nodes) {
        switch (n.type) {
            case cudaGraphNodeTypeKernel:      kernelCount++; break;
            case cudaGraphNodeTypeMemcpy:
                if (n.memcpyKind == "H2D") memcpyH2D++;
                else if (n.memcpyKind == "D2H") memcpyD2H++;
                else if (n.memcpyKind == "D2D") memcpyD2D++;
                else memcpyOther++;
                break;
            case cudaGraphNodeTypeMemset:      memsetCount++; break;
            case cudaGraphNodeTypeHost:        hostCount++; break;
            case cudaGraphNodeTypeGraph:       childCount++; break;
            case cudaGraphNodeTypeEmpty:       emptyCount++; break;
            case cudaGraphNodeTypeEventRecord:
            case cudaGraphNodeTypeWaitEvent:   eventCount++; break;
            default: break;
        }
    }

    sd_printf("║ Summary: %d Kernels, %d Memcpy(H2D), %d Memcpy(D2H), %d Memcpy(D2D)\n",
              kernelCount, memcpyH2D, memcpyD2H, memcpyD2D);
    sd_printf("║          %d Memsets, %d HostCallbacks, %d Events, %d Empty\n",
              memsetCount, hostCount, eventCount, emptyCount);
    sd_print("╠══════════════════════════════════════════════════════════════════╣\n");
    sd_print("║ Node Details:\n");

    for (const auto& n : nodes) {
        switch (n.type) {
            case cudaGraphNodeTypeKernel:
                sd_printf("║  [%3zu] Kernel: %s\n", n.nodeIndex, n.kernelName.c_str());
                break;
            case cudaGraphNodeTypeMemcpy:
                sd_printf("║  [%3zu] Memcpy %s: %zu bytes\n",
                          n.nodeIndex, n.memcpyKind.c_str(), n.memcpyBytes);
                break;
            case cudaGraphNodeTypeMemset:
                sd_printf("║  [%3zu] Memset: %zu bytes, value=%d\n",
                          n.nodeIndex, n.memsetBytes, n.memsetValue);
                break;
            default:
                sd_printf("║  [%3zu] %s\n", n.nodeIndex, n.typeName.c_str());
                break;
        }
    }

    sd_print("╚══════════════════════════════════════════════════════════════════╝\n");
}

size_t CudaGraphHandle::getNumNodesDuringCapture(cudaStream_t captureStream) const {
    if (captureStream == nullptr) return 0;

    // During capture, we can query the in-progress graph from the stream
    cudaStreamCaptureStatus status;
    cudaGraph_t capGraph = nullptr;

    // cudaStreamGetCaptureInfo_v2 is available since CUDA 11.3
    // It returns the graph being captured without ending the capture.
    unsigned long long captureId = 0;
    cudaError_t err = cudaStreamGetCaptureInfo_v2(captureStream, &status, &captureId,
                                                   &capGraph, nullptr, nullptr);
    if (err != cudaSuccess || status != cudaStreamCaptureStatusActive || capGraph == nullptr) {
        cudaGetLastError();  // Clear any error
        return 0;
    }

    size_t numNodes = 0;
    err = cudaGraphGetNodes(capGraph, nullptr, &numNodes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }

    return numNodes;
}

}  // namespace cuda
}  // namespace sd

#endif  // SD_CUDA
