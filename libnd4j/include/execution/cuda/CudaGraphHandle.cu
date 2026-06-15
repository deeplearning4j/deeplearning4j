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
#include <system/env_functions.h>

#include <chrono>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#if !defined(_MSC_VER)
#if !defined(_MSC_VER)
#include <cxxabi.h>  // For __cxa_demangle (kernel name demangling)
#endif
#endif

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
    _captureStartTimeMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    return true;
}

bool CudaGraphHandle::endCapture(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_state != GraphState::CAPTURING) {
        if (_state == GraphState::CAPTURED || _state == GraphState::INSTANTIATED ||
            _state == GraphState::EMPTY) {
            // Benign: endCapture called on a handle that already completed capture
            // (composite replay cleanup, abort paths). Not an error — return false.
            return false;
        }
        // ERROR or EXECUTING state — a real lifecycle bug. Throw with stack trace.
        std::string errMsg = "CudaGraphHandle::endCapture — NOT currently capturing (state=" +
            std::to_string(static_cast<int>(_state)) +
            "). Likely cause: capture workspace OOM aborted capture mid-stream. "
            "Increase capture workspace: -Dnd4j.dsp.captureWorkspaceMb=512";
        THROW_EXCEPTION(errMsg.c_str());
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

    _captureEndTimeMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
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

    double startTime = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    // Set device
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    // Use non-deprecated cudaGraphInstantiateWithFlags (CUDA 12.0+).
    // AutoFreeOnLaunch ensures cudaMallocAsync nodes captured in the graph release
    // their memory between launches. Without this, each cudaGraphLaunch accumulates
    // ~256 MB of unreleased internal allocations, causing OOM after ~100 steps.
    // Default ON — can be disabled via ND4J_TRITON_GRAPH_AUTOFREE=false env var.
    unsigned long long instantiateFlags = cudaGraphInstantiateFlagAutoFreeOnLaunch;
    cudaError_t err = cudaGraphInstantiateWithFlags(&_graphExec, _graph, instantiateFlags);

    // Store the error code for the caller to inspect (OOM detection)
    _lastInstantiateError = static_cast<int>(err);

    // Restore device
    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
    }

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::instantiate failed: %s (err=%d)\n",
                  cudaGetErrorString(err), (int)err);
        // Try to get more info from graph validation
        size_t numNodes = 0;
        cudaGraphGetNodes(_graph, nullptr, &numNodes);
        sd_printf("  Graph has %zu nodes, deviceId=%d\n", numNodes, _deviceId);
        // Clear sticky CUDA error state
        cudaGetLastError();
        // Destroy the captured graph immediately — it consumes GPU memory even
        // without a successful instantiation. Keeping it alive leaks GPU memory
        // proportional to the number of nodes and their associated resources.
        if (_graph != nullptr) {
            cudaGraphDestroy(_graph);
            _graph = nullptr;
        }
        _state = GraphState::ERROR;
        return false;
    }

    double endTime = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    _instantiateTimeMs = endTime - startTime;

    _state = GraphState::INSTANTIATED;
    return true;
}

bool CudaGraphHandle::reInstantiate() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) {
        sd_print("CudaGraphHandle::reInstantiate - No graph to instantiate from\n");
        return false;
    }

    // Destroy old exec
    if (_graphExec != nullptr) {
        cudaGraphExecDestroy(_graphExec);
        _graphExec = nullptr;
    }

    // Set device
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    // Re-create exec from the same captured graph
    unsigned long long instantiateFlags = sd::env_tritonGraphAutoFree()
        ? cudaGraphInstantiateFlagAutoFreeOnLaunch : 0;
    cudaError_t err = cudaGraphInstantiateWithFlags(&_graphExec, _graph, instantiateFlags);

    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
    }

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::reInstantiate failed: %s (err=%d)\n",
                  cudaGetErrorString(err), (int)err);
        // Clear sticky CUDA error state
        cudaGetLastError();
        // Destroy the captured graph — no point keeping it if instantiation failed,
        // and it holds GPU memory.
        if (_graph != nullptr) {
            cudaGraphDestroy(_graph);
            _graph = nullptr;
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

    // Ensure correct device before launch — graph was captured on _deviceId
    int prevDevice;
    cudaGetDevice(&prevDevice);
    if (_deviceId != prevDevice) {
        cudaSetDevice(_deviceId);
    }

    // Record start time for execution timeline
    double startTime = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    // Clear any sticky error before launch (e.g., from updateStatistics graph queries)
    cudaGetLastError();

    // ── TRIPWIRE: validate graph exec handle before launch ──
    if (_graphExec == nullptr) {
        sd_printf("TRIPWIRE: cudaGraphLaunch — _graphExec is NULL! device=%d state=%d\n",
                  _deviceId, (int)_state);
        _state = GraphState::ERROR;
        return false;
    }
    if (stream == nullptr) {
        sd_printf("TRIPWIRE: cudaGraphLaunch — stream is NULL! device=%d\n", _deviceId);
    }
    // ── END TRIPWIRE ───────────────────────────────────────────────────

    cudaError_t err = cudaGraphLaunch(_graphExec, stream);

    double endTime = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    if (err != cudaSuccess) {
        sd_printf("CudaGraphHandle::launchAsync failed: %s (err=%d)\n",
                  cudaGetErrorString(err), (int)err);
        sd_printf("  Graph stats: %d kernels, %d memcpys, %d memsets, %d memAllocs, %d memFrees, "
                  "%d hostCallbacks, %d events, %d empty, %d childGraphs\n",
                  _stats.numKernels, _stats.numMemcpyH2D, _stats.numMemsets,
                  _stats.numMemAllocs, _stats.numMemFrees,
                  _stats.numHostCallbacks, _stats.numEvents, _stats.numEmpty, _stats.numChildGraphs);
        if (_stats.numMemAllocs > 0 || _stats.numMemFrees > 0) {
            sd_printf("  WARNING: Graph contains %d MemAlloc + %d MemFree nodes from cudaMallocAsync/cudaFreeAsync. "
                      "These may cause replay failures if virtual addresses are unstable.\n",
                      _stats.numMemAllocs, _stats.numMemFrees);
        }
        if (_stats.numHostCallbacks > 0) {
            sd_printf("  WARNING: Graph contains %d host callback nodes. "
                      "Host callbacks may cause replay failures.\n", _stats.numHostCallbacks);
        }
        // Try exporting DOT file for debugging
        if (_graph != nullptr) {
            std::string dotPath = "/tmp/cuda_graph_failed_launch.dot";
            auto dotErr = cudaGraphDebugDotPrint(_graph, dotPath.c_str(),
                                                  cudaGraphDebugDotFlagsVerbose);
            if (dotErr == cudaSuccess) {
                sd_printf("  Exported failed graph DOT to %s\n", dotPath.c_str());
            }
            cudaGetLastError(); // clear any error from dot print
        }
        _state = GraphState::ERROR;
        // Record failed execution in timeline
        if (!_executionTimeline.empty()) {
            auto& lastEntry = _executionTimeline.back();
            lastEntry.durationMs = endTime - startTime;
            lastEntry.success = false;
            lastEntry.errorMessage = cudaGetErrorString(err);
        }
        // Restore device
        if (_deviceId != prevDevice) {
            cudaSetDevice(prevDevice);
        }
        return false;
    }

    if (completionEvent != nullptr) {
        cudaEventRecord(completionEvent, stream);
    }

    // Update last execution timeline entry with duration
    if (!_executionTimeline.empty()) {
        auto& lastEntry = _executionTimeline.back();
        lastEntry.durationMs = endTime - startTime;
        lastEntry.success = true;
    }

    // Restore device
    if (_deviceId != prevDevice) {
        cudaSetDevice(prevDevice);
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
            case cudaGraphNodeTypeEmpty:
                _stats.numEmpty++;
                break;
            // CUDA 11.4+ memory allocation nodes from cudaMallocAsync/cudaFreeAsync during capture
            case cudaGraphNodeTypeMemAlloc:
                _stats.numMemAllocs++;
                _stats.totalMemoryOps++;
                break;
            case cudaGraphNodeTypeMemFree:
                _stats.numMemFrees++;
                _stats.totalMemoryOps++;
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

#if !defined(_MSC_VER)
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
#endif
    // Demangling not available or failed — return raw name
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
                if (err != cudaSuccess) {
                    cudaGetLastError();  // Clear sticky error from failed param query
                }
                if (err == cudaSuccess && params.func != nullptr) {
                    // Get kernel name from function pointer
                    const char* name = nullptr;
                    auto nameErr = cudaFuncGetName(&name, params.func);
                    if (nameErr == cudaSuccess && name != nullptr) {
                        info.kernelName = demangleKernelName(name);
                    } else {
                        // Clear sticky error from cudaFuncGetName failure.
                        // Driver-API CUfunction handles (e.g., from Triton's cuModuleGetFunction)
                        // are not recognized by the runtime API's cudaFuncGetName, which sets
                        // cudaErrorInvalidDeviceFunction. If not cleared, this sticky error
                        // propagates to cudaGraphLaunch and causes SIGSEGV.
                        cudaGetLastError();
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

// ============================================================================
// Chrome Trace Export (PyTorch-style visualization)
// ============================================================================

static double getTimestampMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

static std::string escapeJson(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 1.1);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

std::string CudaGraphHandle::getChromeTraceJson() const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) return "{}";

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);

    auto nodes = getDetailedNodeInfo();
    if (nodes.empty()) return "{\"traceEvents\":[]}";

    // Get dependency edges for flow events
    size_t numEdges = 0;
    cudaGraphGetEdges(_graph, nullptr, nullptr, &numEdges);
    std::vector<cudaGraphNode_t> fromNodes(numEdges), toNodes(numEdges);
    if (numEdges > 0) {
        cudaGraphGetEdges(_graph, fromNodes.data(), toNodes.data(), &numEdges);
    }

    // Build node index map for flow events
    std::unordered_map<cudaGraphNode_t, size_t> nodeToIndex;
    size_t numNodes = 0;
    cudaGraphGetNodes(_graph, nullptr, &numNodes);
    if (numNodes > 0) {
        std::vector<cudaGraphNode_t> graphNodes(numNodes);
        cudaGraphGetNodes(_graph, graphNodes.data(), &numNodes);
        for (size_t i = 0; i < numNodes; i++) {
            nodeToIndex[graphNodes[i]] = i;
        }
    }

    json << "{\n";
    json << "  \"traceEvents\": [\n";

    // Process start event (metadata)
    double baseTime = _captureStartTimeMs > 0 ? _captureStartTimeMs : getTimestampMs();
    json << "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": " << _deviceId
         << ", \"tid\": 0, \"ts\": 0, \"args\": {\"name\": \"CUDA Graph (Device " << _deviceId << ")\"}},\n";

    // Thread metadata
    json << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": " << _deviceId
         << ", \"tid\": 1, \"ts\": 0, \"args\": {\"name\": \"CUDA Graph Execution\"}},\n";

    // Graph capture event
    json << "    {\"name\": \"Graph Capture\", \"ph\": \"X\", \"pid\": " << _deviceId
         << ", \"tid\": 0, \"ts\": 0, \"dur\": "
         << (_captureEndTimeMs - _captureStartTimeMs)
         << ", \"args\": {\"nodes\": " << nodes.size() << "}},\n";

    // Graph instantiation event
    if (_instantiateTimeMs > 0) {
        json << "    {\"name\": \"Graph Instantiate\", \"ph\": \"X\", \"pid\": " << _deviceId
             << ", \"tid\": 0, \"ts\": " << (_captureEndTimeMs - _captureStartTimeMs)
             << ", \"dur\": " << _instantiateTimeMs << "},\n";
    }

    // Add node events
    double nodeTime = 0.0;
    double nodeDuration = 10.0;  // Estimated duration per node in microseconds

    for (size_t i = 0; i < nodes.size(); i++) {
        const auto& n = nodes[i];

        std::string eventName;
        std::string category = "gpu";

        switch (n.type) {
            case cudaGraphNodeTypeKernel:
                eventName = n.kernelName.empty() ? "Kernel" : n.kernelName;
                category = "kernel";
                nodeDuration = 50.0;  // Kernels typically longer
                break;
            case cudaGraphNodeTypeMemcpy:
                eventName = "Memcpy " + n.memcpyKind;
                category = "memcpy";
                nodeDuration = 5.0 + (n.memcpyBytes / 1000000.0);  // Scale with size
                break;
            case cudaGraphNodeTypeMemset:
                eventName = "Memset";
                category = "memset";
                nodeDuration = 2.0 + (n.memsetBytes / 10000000.0);
                break;
            case cudaGraphNodeTypeHost:
                eventName = "HostCallback";
                category = "host";
                nodeDuration = 20.0;
                break;
            case cudaGraphNodeTypeGraph:
                eventName = "ChildGraph";
                category = "graph";
                nodeDuration = 100.0;
                break;
            default:
                eventName = n.typeName;
                break;
        }

        json << "    {\"name\": \"" << escapeJson(eventName) << "\", \"ph\": \"X\", \"pid\": " << _deviceId
             << ", \"tid\": 1, \"ts\": " << nodeTime << ", \"dur\": " << nodeDuration
             << ", \"cat\": \"" << category << "\", \"args\": {\"node_index\": " << i;

        // Add extra args based on type
        if (n.type == cudaGraphNodeTypeMemcpy) {
            json << ", \"bytes\": " << n.memcpyBytes << ", \"kind\": \"" << n.memcpyKind << "\"";
        } else if (n.type == cudaGraphNodeTypeMemset) {
            json << ", \"bytes\": " << n.memsetBytes << ", \"value\": " << n.memsetValue;
        }

        json << "}}";
        if (i < nodes.size() - 1 || numEdges > 0 || !_executionTimeline.empty()) {
            json << ",";
        }
        json << "\n";

        nodeTime += nodeDuration + 1.0;  // Small gap between events
    }

    // Add flow events for dependencies
    size_t flowId = 0;
    for (size_t e = 0; e < numEdges; e++) {
        auto fromIt = nodeToIndex.find(fromNodes[e]);
        auto toIt = nodeToIndex.find(toNodes[e]);
        if (fromIt != nodeToIndex.end() && toIt != nodeToIndex.end()) {
            json << "    {\"ph\": \"s\", \"name\": \"dependency\", \"pid\": " << _deviceId
                 << ", \"tid\": 1, \"id\": " << flowId << ", \"ts\": "
                 << (fromIt->second * nodeDuration) << "},\n";
            json << "    {\"ph\": \"f\", \"name\": \"dependency\", \"pid\": " << _deviceId
                 << ", \"tid\": 1, \"id\": " << flowId << ", \"ts\": "
                 << (toIt->second * nodeDuration) << ", \"bp\": \"e\"}";
            if (e < numEdges - 1 || !_executionTimeline.empty()) {
                json << ",";
            }
            json << "\n";
            flowId++;
        }
    }

    // Add execution timeline (replay history)
    double replayBaseTime = nodeTime + 100.0;
    size_t replayIdx = 0;
    for (const auto& entry : _executionTimeline) {
        json << "    {\"name\": \"Graph Replay #" << (replayIdx + 1) << "\", \"ph\": \"X\", \"pid\": " << _deviceId
             << ", \"tid\": 2, \"ts\": " << replayBaseTime << ", \"dur\": " << entry.durationMs * 1000.0
             << ", \"cat\": \"replay\", \"args\": {\"success\": " << (entry.success ? "true" : "false")
             << ", \"kernels\": " << entry.numKernels << ", \"memops\": " << entry.numMemops;
        if (!entry.errorMessage.empty()) {
            json << ", \"error\": \"" << escapeJson(entry.errorMessage) << "\"";
        }
        json << "}}";
        if (replayIdx < _executionTimeline.size() - 1) {
            json << ",";
        }
        json << "\n";
        replayBaseTime += entry.durationMs * 1000.0 + 50.0;
        replayIdx++;
    }

    json << "  ],\n";

    // Add metadata
    json << "  \"displayName\": \"CUDA Graph Visualization\",\n";
    json << "  \"metadata\": {\n";
    json << "    \"device\": " << _deviceId << ",\n";
    json << "    \"numNodes\": " << nodes.size() << ",\n";
    json << "    \"numEdges\": " << numEdges << ",\n";
    json << "    \"numReplays\": " << _executionTimeline.size() << ",\n";
    json << "    \"kernels\": " << _stats.numKernels << ",\n";
    json << "    \"memcpyOps\": " << _stats.numMemcpyH2D + _stats.numMemcpyD2H + _stats.numMemcpyD2D << ",\n";
    json << "    \"memsetOps\": " << _stats.numMemsets << "\n";
    json << "  }\n";
    json << "}\n";

    return json.str();
}

bool CudaGraphHandle::exportToChromeTrace(const std::string& filename) const {
    std::unordered_map<std::string, std::string> emptyMetadata;
    return exportToChromeTrace(filename, emptyMetadata);
}

bool CudaGraphHandle::exportToChromeTrace(
    const std::string& filename,
    const std::unordered_map<std::string, std::string>& metadata) const {

    std::string json = getChromeTraceJson();
    if (json == "{}") {
        sd_printf("CudaGraphHandle::exportToChromeTrace - No graph data to export\n");
        return false;
    }

    // Inject custom metadata if provided
    if (!metadata.empty()) {
        std::ostringstream metaJson;
        metaJson << ",\n    \"custom\": {\n";
        bool first = true;
        for (const auto& [key, value] : metadata) {
            if (!first) metaJson << ",\n";
            metaJson << "      \"" << escapeJson(key) << "\": \"" << escapeJson(value) << "\"";
            first = false;
        }
        metaJson << "\n    }\n";

        // Insert before closing brace of metadata section
        size_t metaPos = json.rfind("  }\n}");
        if (metaPos != std::string::npos) {
            json = json.substr(0, metaPos) + metaJson.str() + "}\n";
        }
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        sd_printf("CudaGraphHandle::exportToChromeTrace - Failed to open file: %s\n", filename.c_str());
        return false;
    }

    file << json;
    file.close();

    sd_printf("CudaGraphHandle: Exported Chrome trace to %s (%zu bytes)\n",
              filename.c_str(), json.size());
    return true;
}

bool CudaGraphHandle::exportToHtml(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_graph == nullptr) return false;

    auto nodes = getDetailedNodeInfo();

    std::ofstream file(filename);
    if (!file.is_open()) {
        sd_printf("CudaGraphHandle::exportToHtml - Failed to open file: %s\n", filename.c_str());
        return false;
    }

    file << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CUDA Graph Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00d4ff; margin-bottom: 10px; }
        .stats { display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }
        .stat-card { background: #16213e; padding: 20px 30px; border-radius: 10px; text-align: center; min-width: 120px; }
        .stat-value { font-size: 2em; color: #00d4ff; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.9em; margin-top: 5px; }
        .graph-container { background: #16213e; border-radius: 15px; padding: 20px; margin: 20px 0; overflow-x: auto; }
        .node-grid { display: flex; flex-direction: column; gap: 8px; }
        .node-row { display: flex; align-items: center; gap: 15px; padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 4px solid; }
        .node-row.kernel { border-color: #00ff88; }
        .node-row.memcpy { border-color: #ffaa00; }
        .node-row.memset { border-color: #ff6b6b; }
        .node-row.host { border-color: #a855f7; }
        .node-row.graph { border-color: #3b82f6; }
        .node-row.empty { border-color: #666; }
        .node-index { font-family: monospace; color: #666; min-width: 50px; }
        .node-type { font-weight: bold; min-width: 100px; }
        .node-name { flex: 1; font-family: monospace; font-size: 0.9em; }
        .node-bytes { color: #888; font-size: 0.85em; }
        .timeline { margin-top: 30px; }
        .timeline-bar { height: 30px; background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%); border-radius: 5px; position: relative; margin: 5px 0; }
        .timeline-section { position: absolute; height: 100%; opacity: 0.8; }
        .legend { display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-color { width: 20px; height: 20px; border-radius: 4px; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CUDA Graph Visualization</h1>
        <p>Interactive visualization of captured CUDA graph</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">)" << _deviceId << R"(</div>
            <div class="stat-label">Device ID</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">)" << nodes.size() << R"(</div>
            <div class="stat-label">Nodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">)" << _stats.numKernels << R"(</div>
            <div class="stat-label">Kernels</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">)" << (_stats.numMemcpyH2D + _stats.numMemcpyD2H + _stats.numMemcpyD2D) << R"(</div>
            <div class="stat-label">Memcopies</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">)" << _executionTimeline.size() << R"(</div>
            <div class="stat-label">Replays</div>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #00ff88;"></div> Kernel</div>
        <div class="legend-item"><div class="legend-color" style="background: #ffaa00;"></div> Memcpy</div>
        <div class="legend-item"><div class="legend-color" style="background: #ff6b6b;"></div> Memset</div>
        <div class="legend-item"><div class="legend-color" style="background: #a855f7;"></div> Host Callback</div>
        <div class="legend-item"><div class="legend-color" style="background: #3b82f6;"></div> Child Graph</div>
    </div>

    <div class="graph-container">
        <h2>Graph Nodes</h2>
        <div class="node-grid">
)";

    for (const auto& n : nodes) {
        std::string cssClass = "node-row ";
        switch (n.type) {
            case cudaGraphNodeTypeKernel: cssClass += "kernel"; break;
            case cudaGraphNodeTypeMemcpy: cssClass += "memcpy"; break;
            case cudaGraphNodeTypeMemset: cssClass += "memset"; break;
            case cudaGraphNodeTypeHost: cssClass += "host"; break;
            case cudaGraphNodeTypeGraph: cssClass += "graph"; break;
            default: cssClass += "empty"; break;
        }

        file << "            <div class=\"" << cssClass << "\">\n";
        file << "                <span class=\"node-index\">[" << n.nodeIndex << "]</span>\n";
        file << "                <span class=\"node-type\">" << escapeJson(n.typeName) << "</span>\n";
        file << "                <span class=\"node-name\">";
        if (n.type == cudaGraphNodeTypeKernel) {
            file << escapeJson(n.kernelName);
        } else if (n.type == cudaGraphNodeTypeMemcpy) {
            file << n.memcpyKind << " transfer";
        }
        file << "</span>\n";
        if (n.type == cudaGraphNodeTypeMemcpy && n.memcpyBytes > 0) {
            file << "                <span class=\"node-bytes\">" << n.memcpyBytes << " bytes</span>\n";
        } else if (n.type == cudaGraphNodeTypeMemset && n.memsetBytes > 0) {
            file << "                <span class=\"node-bytes\">" << n.memsetBytes << " bytes (value=" << n.memsetValue << ")</span>\n";
        }
        file << "            </div>\n";
    }

    file << R"(        </div>
    </div>

    <div class="graph-container">
        <h2>Execution Timeline</h2>
        <div class="timeline">
)";

    if (!_executionTimeline.empty()) {
        double maxDuration = 0;
        for (const auto& e : _executionTimeline) {
            maxDuration = std::max(maxDuration, e.durationMs);
        }
        double totalWidth = 0;
        for (const auto& entry : _executionTimeline) {
            double width = (entry.durationMs / maxDuration) * 100.0;
            std::string bgColor = entry.success ? "#00ff88" : "#ff6b6b";
            file << "            <div class=\"timeline-bar\" style=\"width: " << width << "%; background: " << bgColor << ";\" title=\"Duration: " << entry.durationMs << " ms\"></div>\n";
            totalWidth += width;
        }
    } else {
        file << "            <p style=\"color: #888;\">No replay executions recorded yet.</p>\n";
    }

    file << R"(        </div>
    </div>

    <div class="footer">
        <p>Generated by ND4J CUDA Graph Visualization</p>
        <p>Open chrome://tracing and load the .json trace file for detailed timeline analysis</p>
    </div>
</body>
</html>
)";

    file.close();
    sd_printf("CudaGraphHandle: Exported HTML visualization to %s\n", filename.c_str());
    return true;
}

bool CudaGraphHandle::debugDump(const std::string& debugPath) const {
    bool success = true;

    // 1. Export DOT file for graphviz
    if (!exportToDot(debugPath + ".dot")) {
        sd_printf("CudaGraphHandle::debugDump - Failed to export DOT file\n");
        success = false;
    }

    // 2. Export Chrome trace JSON
    if (!exportToChromeTrace(debugPath + ".json")) {
        sd_printf("CudaGraphHandle::debugDump - Failed to export Chrome trace\n");
        success = false;
    }

    // 3. Export HTML visualization
    if (!exportToHtml(debugPath + ".html")) {
        sd_printf("CudaGraphHandle::debugDump - Failed to export HTML\n");
        success = false;
    }

    // 4. Export node details as JSON
    {
        std::ofstream file(debugPath + "_nodes.json");
        if (file.is_open()) {
            auto nodes = getDetailedNodeInfo();
            file << "{\n  \"nodes\": [\n";
            for (size_t i = 0; i < nodes.size(); i++) {
                const auto& n = nodes[i];
                file << "    {\"index\": " << n.nodeIndex
                     << ", \"type\": \"" << escapeJson(n.typeName) << "\""
                     << ", \"typeEnum\": " << static_cast<int>(n.type);
                if (n.type == cudaGraphNodeTypeKernel) {
                    file << ", \"kernel\": \"" << escapeJson(n.kernelName) << "\"";
                } else if (n.type == cudaGraphNodeTypeMemcpy) {
                    file << ", \"memcpyKind\": \"" << escapeJson(n.memcpyKind) << "\""
                         << ", \"bytes\": " << n.memcpyBytes;
                } else if (n.type == cudaGraphNodeTypeMemset) {
                    file << ", \"bytes\": " << n.memsetBytes
                         << ", \"value\": " << n.memsetValue;
                }
                file << "}";
                if (i < nodes.size() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";
            file << "  \"statistics\": {\n";
            file << "    \"numKernels\": " << _stats.numKernels << ",\n";
            file << "    \"numMemcpyH2D\": " << _stats.numMemcpyH2D << ",\n";
            file << "    \"numMemcpyD2H\": " << _stats.numMemcpyD2H << ",\n";
            file << "    \"numMemcpyD2D\": " << _stats.numMemcpyD2D << ",\n";
            file << "    \"numMemsets\": " << _stats.numMemsets << ",\n";
            file << "    \"numHostCallbacks\": " << _stats.numHostCallbacks << ",\n";
            file << "    \"numChildGraphs\": " << _stats.numChildGraphs << ",\n";
            file << "    \"totalMemoryOps\": " << _stats.totalMemoryOps << "\n";
            file << "  }\n}\n";
            file.close();
        }
    }

    sd_printf("CudaGraphHandle::debugDump - Exported debug files to %s.{dot,json,html}\n",
              debugPath.c_str());
    return success;
}

void CudaGraphHandle::recordReplayStart(cudaStream_t stream) {
    // Record timestamp for execution timeline
    // The actual duration will be recorded at launch completion
    ExecutionTimelineEntry entry;
    entry.timestampMs = getTimestampMs();
    entry.numKernels = _stats.numKernels;
    entry.numMemops = _stats.totalMemoryOps;
    entry.success = true;

    // Store for later update (in launchAsync)
    _executionTimeline.push_back(entry);

    // Keep only last 1000 entries to avoid memory growth
    if (_executionTimeline.size() > 1000) {
        _executionTimeline.erase(_executionTimeline.begin());
    }
}

}  // namespace cuda
}  // namespace sd

#endif  // SD_CUDA
