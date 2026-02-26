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
// @author Adam Gibson
//
// CUDA Graph Scheduler - Manages CUDA graph capture and execution for multi-device scenarios
//

#ifndef SD_CUDA_GRAPH_SCHEDULER_H
#define SD_CUDA_GRAPH_SCHEDULER_H

#include <system/common.h>

#ifdef SD_CUDA

#include <cuda_runtime.h>
#include <execution/LaunchContext.h>
#include <execution/DeviceManager.h>
#include <array/NDArray.h>
#include <graph/Context.h>

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <string>
#include <functional>
#include <queue>
#include <future>
#include <list>

namespace sd {
namespace cuda {

/**
 * CUDA Graph capture mode
 */
enum class GraphCaptureMode {
    DISABLED = 0,       // No graph capture
    GLOBAL,             // Global capture mode - capture entire execution
    THREAD_LOCAL,       // Thread-local capture
    RELAXED             // Relaxed capture - allows some operations outside graph
};

/**
 * Graph execution mode
 */
enum class GraphExecutionMode {
    SYNC = 0,           // Synchronous execution
    ASYNC,              // Asynchronous execution
    PIPELINED           // Pipelined execution with multiple graphs
};

/**
 * State of a CUDA graph instance
 */
enum class GraphState {
    EMPTY = 0,          // Graph is empty
    CAPTURING,          // Currently capturing
    CAPTURED,           // Capture complete, not yet instantiated
    INSTANTIATED,       // Graph exec created
    EXECUTING,          // Currently executing
    COMPLETED,          // Execution completed
    ERROR               // Error state
};

/**
 * Graph node type for tracking
 */
enum class GraphNodeType {
    KERNEL = 0,         // Kernel launch
    MEMCPY_H2D,         // Host to device memcpy
    MEMCPY_D2H,         // Device to host memcpy
    MEMCPY_D2D,         // Device to device memcpy
    MEMSET,             // Memory set
    EVENT_RECORD,       // Event record
    EVENT_WAIT,         // Event wait
    HOST_CALLBACK,      // Host callback
    CHILD_GRAPH,        // Child graph
    EXTERNAL_SEMAPHORE  // External semaphore
};

/**
 * Descriptor for a graph node
 */
struct SD_LIB_EXPORT GraphNodeDescriptor {
    std::string name;
    GraphNodeType type = GraphNodeType::KERNEL;
    int deviceId = 0;
    size_t estimatedTimeNs = 0;
    size_t memoryBytes = 0;
    std::vector<int> dependencies;  // Node indices this depends on
};

/**
 * Detailed info about a single CUDA graph node, extracted from cudaGraph_t.
 * Used for debug/verbose printing of the full captured graph.
 */
struct SD_LIB_EXPORT CudaGraphNodeInfo {
    cudaGraphNodeType type;         // CUDA node type enum
    std::string typeName;           // Human-readable type name
    std::string kernelName;         // Demangled kernel function name (kernels only)
    size_t memcpyBytes = 0;        // Bytes transferred (memcpy only)
    int memcpySrcDevice = -1;      // Source device (memcpy only)
    int memcpyDstDevice = -1;      // Dest device (memcpy only)
    std::string memcpyKind;        // "H2D", "D2H", "D2D", "H2H" (memcpy only)
    size_t memsetBytes = 0;        // Bytes set (memset only)
    int memsetValue = 0;           // Value set (memset only)
    size_t nodeIndex = 0;          // Index in the graph node array
};

/**
 * Per-op capture audit entry: tracks how many CUDA graph nodes each op contributed.
 * Used for validating that every op in a segment actually captured work.
 */
struct SD_LIB_EXPORT CaptureAuditEntry {
    int slotIndex = -1;            // Slot index in the NativeDynamicShapePlan
    std::string opName;            // Op name (e.g. "shape_of", "gather", "multiply")
    size_t nodesBefore = 0;        // Graph node count before this op executed
    size_t nodesAfter = 0;         // Graph node count after this op executed
    size_t nodesContributed = 0;   // nodesAfter - nodesBefore

    // Per-op node type breakdown (populated post-capture)
    int kernels = 0;
    int memcpys = 0;
    int memsets = 0;
    int memAllocs = 0;
    int memFrees = 0;

    bool isHostOnly() const { return nodesContributed == 0; }
};

/**
 * Statistics for a captured graph
 */
struct SD_LIB_EXPORT GraphStatistics {
    int numKernels = 0;
    int numMemcpyH2D = 0;
    int numMemcpyD2H = 0;
    int numMemcpyD2D = 0;
    int numMemsets = 0;
    int numEvents = 0;
    int numHostCallbacks = 0;
    int numChildGraphs = 0;
    int numMemAllocs = 0;    // cudaMallocAsync during capture → cudaGraphMemAllocNode
    int numMemFrees = 0;     // cudaFreeAsync during capture → cudaGraphMemFreeNode
    int numEmpty = 0;
    size_t totalMemoryOps = 0;
    double estimatedTimeMs = 0.0;
};

/**
 * Configuration for CUDA graph scheduler
 */
struct SD_LIB_EXPORT CudaGraphConfig {
    GraphCaptureMode captureMode = GraphCaptureMode::GLOBAL;
    GraphExecutionMode executionMode = GraphExecutionMode::ASYNC;

    // Capture options
    bool captureMemcpyAsync = true;
    bool captureEvents = true;
    bool captureHostCallbacks = false;  // Host callbacks can cause issues

    // Execution options
    int maxConcurrentGraphs = 4;
    bool enableGraphCaching = true;
    size_t graphCacheSize = 32;  // Number of graphs to cache

    // Multi-device options
    bool enableMultiDevice = true;
    bool enableP2P = true;
    bool preferUnifiedMemory = false;

    // Optimization options
    bool enableGraphOptimization = true;
    bool enableKernelFusion = false;  // Experimental
    bool enablePrefetch = true;

    // Debug options
    bool verbose = false;
    bool validateGraphs = false;
    bool recordTiming = true;
};

/**
 * Result of graph execution
 */
struct SD_LIB_EXPORT GraphExecutionResult {
    bool success = false;
    std::string errorMessage;
    GraphState finalState = GraphState::EMPTY;

    // Timing
    double captureTimeMs = 0.0;
    double instantiateTimeMs = 0.0;
    double executeTimeMs = 0.0;
    double totalTimeMs = 0.0;

    // Statistics
    GraphStatistics stats;
    int executionCount = 0;
};

/**
 * CUDA Graph Handle - Wrapper for cudaGraph_t and cudaGraphExec_t
 */
class SD_LIB_EXPORT CudaGraphHandle {
public:
    CudaGraphHandle();
    explicit CudaGraphHandle(int deviceId);
    ~CudaGraphHandle();

    // Non-copyable
    CudaGraphHandle(const CudaGraphHandle&) = delete;
    CudaGraphHandle& operator=(const CudaGraphHandle&) = delete;

    // Movable
    CudaGraphHandle(CudaGraphHandle&& other) noexcept;
    CudaGraphHandle& operator=(CudaGraphHandle&& other) noexcept;

    // Capture control
    bool beginCapture(cudaStream_t stream, cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal);
    bool endCapture(cudaStream_t stream);
    bool isCapturing() const { return _state == GraphState::CAPTURING; }

    // Instantiation
    bool instantiate();
    bool updateFromGraph();  // Update exec from modified graph

    // Execution
    bool launch(cudaStream_t stream);
    bool launchAsync(cudaStream_t stream, cudaEvent_t completionEvent = nullptr);

    // State
    GraphState getState() const { return _state; }
    int getDeviceId() const { return _deviceId; }
    bool isValid() const { return _graph != nullptr || _graphExec != nullptr; }

    // Graph info
    size_t getNumNodes() const;
    size_t getNumEdges() const;
    GraphStatistics getStatistics() const;

    // Debug / Diagnostics
    void printDebugInfo() const;
    bool exportToDot(const std::string& filename) const;

    /**
     * Export graph execution timeline to Chrome trace format.
     * The output JSON file can be loaded in chrome://tracing for visualization.
     * Includes kernel launches, memory operations, and timing data.
     * @param filename Output file path (should end with .json)
     * @return true on success
     */
    bool exportToChromeTrace(const std::string& filename) const;

    /**
     * Export graph execution timeline to Chrome trace format with custom metadata.
     * @param filename Output file path
     * @param metadata Custom metadata to include in the trace
     * @return true on success
     */
    bool exportToChromeTrace(const std::string& filename,
                            const std::unordered_map<std::string, std::string>& metadata) const;

    /**
     * Get the Chrome trace JSON string directly (for programmatic access).
     * @return JSON string in Chrome trace format
     */
    std::string getChromeTraceJson() const;

    /**
     * Dump graph visualization to multiple formats for debugging.
     * Creates: {debug_path}.dot, {debug_path}_nodes.json, {debug_path}.html
     * PyTorch-style debug dump similar to torch.cuda.CUDAGraph.debug_dump().
     * @param debugPath Base path for output files (without extension)
     * @return true on success
     */
    bool debugDump(const std::string& debugPath) const;

    /**
     * Export graph to HTML visualization with interactive SVG.
     * Creates a standalone HTML file with embedded JavaScript for visualization.
     * @param filename Output HTML file path
     * @return true on success
     */
    bool exportToHtml(const std::string& filename) const;

    /**
     * Record a graph replay event for Chrome trace timeline.
     * Call this before each graph launch to track replay history.
     * @param stream The CUDA stream used for launch
     */
    void recordReplayStart(cudaStream_t stream);

    /**
     * Get execution timeline statistics.
     */
    struct ExecutionTimelineEntry {
        double timestampMs = 0.0;
        double durationMs = 0.0;
        int numKernels = 0;
        int numMemops = 0;
        bool success = false;
        std::string errorMessage;
    };
    const std::vector<ExecutionTimelineEntry>& getExecutionTimeline() const { return _executionTimeline; }
    void clearExecutionTimeline() { _executionTimeline.clear(); }

    /**
     * Extract detailed info about every node in the captured graph.
     * Includes kernel names, memcpy directions/sizes, memset values, etc.
     * Only valid after endCapture() succeeds (state >= CAPTURED).
     */
    std::vector<CudaGraphNodeInfo> getDetailedNodeInfo() const;

    /**
     * Print the full contents of the captured CUDA graph to stderr.
     * Shows every node with its type, kernel name, memcpy details, etc.
     * Gated on debug/verbose mode in the caller — this method always prints.
     */
    void printGraphContents() const;

    /**
     * Get the current number of nodes in the underlying cudaGraph_t.
     * Can be called during capture to track incremental node additions.
     * Returns 0 if no graph exists or capture hasn't started.
     *
     * NOTE: During active capture (between beginCapture and endCapture),
     * this queries the in-progress graph. The CUDA driver allows this
     * via cudaStreamGetCaptureInfo to get the graph being captured.
     */
    size_t getNumNodesDuringCapture(cudaStream_t captureStream) const;

    // Native handles (use with caution)
    cudaGraph_t getGraph() const { return _graph; }
    cudaGraphExec_t getGraphExec() const { return _graphExec; }

private:
    void cleanup();
    void updateStatistics();

    cudaGraph_t _graph = nullptr;
    cudaGraphExec_t _graphExec = nullptr;
    GraphState _state = GraphState::EMPTY;
    int _deviceId = 0;
    GraphStatistics _stats;
    mutable std::mutex _mutex;

    // Execution timeline tracking for Chrome trace visualization
    std::vector<ExecutionTimelineEntry> _executionTimeline;
    double _captureStartTimeMs = 0.0;
    double _captureEndTimeMs = 0.0;
    double _instantiateTimeMs = 0.0;

    // Persistent pinned host buffers used during graph capture for H2D copies.
    // These must persist for the lifetime of the graph because graph replay
    // reads from the recorded host addresses. Freed in cleanup().
    std::vector<void*> _capturedHostPtrs;

public:
    // Add persistent pinned host buffers (called during graph capture)
    void addCapturedHostPtr(void* ptr) { _capturedHostPtrs.push_back(ptr); }
private:
};

/**
 * Multi-device graph - Coordinates graphs across multiple GPUs
 */
class SD_LIB_EXPORT MultiDeviceGraph {
public:
    MultiDeviceGraph();
    explicit MultiDeviceGraph(const std::vector<int>& deviceIds);
    ~MultiDeviceGraph();

    // Setup
    void addDevice(int deviceId);
    void removeDevice(int deviceId);
    const std::vector<int>& getDevices() const { return _deviceIds; }

    // Capture
    bool beginCaptureOnDevice(int deviceId, cudaStream_t stream);
    bool endCaptureOnDevice(int deviceId, cudaStream_t stream);
    bool beginCaptureAll(const std::vector<cudaStream_t>& streams);
    bool endCaptureAll(const std::vector<cudaStream_t>& streams);

    // Add cross-device dependencies
    void addDependency(int srcDevice, int dstDevice);
    void addP2PTransfer(int srcDevice, int dstDevice, void* srcPtr, void* dstPtr, size_t bytes);

    // Instantiate
    bool instantiateAll();

    // Execute
    bool launchAll(const std::vector<cudaStream_t>& streams);
    bool launchSequential(const std::vector<cudaStream_t>& streams);

    // State
    GraphState getStateForDevice(int deviceId) const;
    bool allReady() const;

    // Statistics
    GraphStatistics getAggregateStatistics() const;
    GraphExecutionResult getLastResult() const { return _lastResult; }

private:
    std::vector<int> _deviceIds;
    std::unordered_map<int, std::unique_ptr<CudaGraphHandle>> _graphs;
    std::vector<std::pair<int, int>> _dependencies;  // (src, dst) device pairs
    GraphExecutionResult _lastResult;
    mutable std::mutex _mutex;
};

/**
 * Graph cache entry
 */
struct SD_LIB_EXPORT GraphCacheEntry {
    std::string key;
    std::shared_ptr<CudaGraphHandle> graph;
    int hitCount = 0;
    double lastAccessTime = 0.0;
    double creationTime = 0.0;
};

/**
 * CUDA Graph Scheduler - Main class for managing CUDA graphs
 *
 * This scheduler provides:
 * - Automatic graph capture for operation sequences
 * - Multi-device graph coordination
 * - Graph caching for repeated patterns
 * - Integration with LaunchContext
 * - Pipeline support with multiple concurrent graphs
 */
class SD_LIB_EXPORT CudaGraphScheduler {
public:
    /**
     * Get singleton instance
     */
    static CudaGraphScheduler& getInstance();

    // Disable copy/move
    CudaGraphScheduler(const CudaGraphScheduler&) = delete;
    CudaGraphScheduler& operator=(const CudaGraphScheduler&) = delete;

    // Configuration
    void configure(const CudaGraphConfig& config);
    const CudaGraphConfig& getConfig() const { return _config; }

    // =========================================================================
    // Graph Capture API
    // =========================================================================

    /**
     * Begin capturing operations on current device's stream
     * @param context LaunchContext to use (nullptr for default)
     * @return true if capture started successfully
     */
    bool beginCapture(LaunchContext* context = nullptr);

    /**
     * Begin capturing on a specific stream
     */
    bool beginCapture(cudaStream_t stream, int deviceId = -1);

    /**
     * End capturing and return graph handle
     * @param context LaunchContext used for capture
     * @return Graph handle, or nullptr on failure
     */
    std::shared_ptr<CudaGraphHandle> endCapture(LaunchContext* context = nullptr);

    /**
     * End capturing on a specific stream
     */
    std::shared_ptr<CudaGraphHandle> endCapture(cudaStream_t stream);

    /**
     * Check if currently capturing on a stream
     */
    bool isCapturing(cudaStream_t stream) const;
    bool isCapturingOnDevice(int deviceId) const;
    bool isCapturingAny() const;

    /**
     * Abort current capture (discard captured operations)
     */
    void abortCapture(cudaStream_t stream);
    void abortAllCaptures();

    // =========================================================================
    // Multi-device Capture
    // =========================================================================

    /**
     * Begin multi-device capture
     * @param contexts LaunchContexts for each device
     * @return Multi-device graph handle
     */
    std::shared_ptr<MultiDeviceGraph> beginMultiDeviceCapture(
        const std::vector<LaunchContext*>& contexts
    );

    /**
     * End multi-device capture
     */
    bool endMultiDeviceCapture(std::shared_ptr<MultiDeviceGraph> graph);

    // =========================================================================
    // Graph Execution
    // =========================================================================

    /**
     * Execute a captured graph
     * @param graph Graph to execute
     * @param context LaunchContext for execution (nullptr for default)
     * @return Execution result
     */
    GraphExecutionResult execute(
        std::shared_ptr<CudaGraphHandle> graph,
        LaunchContext* context = nullptr
    );

    /**
     * Execute a graph asynchronously
     */
    std::future<GraphExecutionResult> executeAsync(
        std::shared_ptr<CudaGraphHandle> graph,
        LaunchContext* context = nullptr
    );

    /**
     * Execute a multi-device graph
     */
    GraphExecutionResult execute(
        std::shared_ptr<MultiDeviceGraph> graph,
        const std::vector<LaunchContext*>& contexts
    );

    // =========================================================================
    // Graph Caching
    // =========================================================================

    /**
     * Get or create a cached graph for a key
     * @param key Unique identifier for the graph pattern
     * @param captureFunc Function to capture the graph if not cached
     * @return Cached or newly captured graph
     */
    std::shared_ptr<CudaGraphHandle> getOrCapture(
        const std::string& key,
        std::function<void()> captureFunc,
        LaunchContext* context = nullptr
    );

    /**
     * Cache a graph with a key
     */
    void cacheGraph(const std::string& key, std::shared_ptr<CudaGraphHandle> graph);

    /**
     * Get cached graph by key
     */
    std::shared_ptr<CudaGraphHandle> getCachedGraph(const std::string& key);

    /**
     * Remove cached graph
     */
    void removeCachedGraph(const std::string& key);

    /**
     * Clear all cached graphs
     */
    void clearCache();

    /**
     * Get cache statistics
     */
    size_t getCacheSize() const;
    size_t getCacheHits() const { return _cacheHits; }
    size_t getCacheMisses() const { return _cacheMisses; }

    // =========================================================================
    // Pipeline Support
    // =========================================================================

    /**
     * Create a pipeline of graphs for double/triple buffering
     * @param numStages Number of pipeline stages (graphs)
     * @param captureFunc Function to capture each stage
     * @return Pipeline ID
     */
    int createPipeline(
        int numStages,
        std::function<void(int stage)> captureFunc,
        LaunchContext* context = nullptr
    );

    /**
     * Execute next stage in pipeline
     */
    bool executePipelineStage(int pipelineId, int stage);

    /**
     * Synchronize pipeline
     */
    void syncPipeline(int pipelineId);

    /**
     * Destroy pipeline
     */
    void destroyPipeline(int pipelineId);

    // =========================================================================
    // Integration with Op Execution
    // =========================================================================

    /**
     * Record an operation for potential graph capture
     * Call this before executing an operation
     */
    void recordOpBegin(const std::string& opName, graph::Context* context);

    /**
     * Mark operation as complete
     */
    void recordOpEnd(const std::string& opName, graph::Context* context);

    /**
     * Check if an operation should be captured
     */
    bool shouldCaptureOp(const std::string& opName) const;

    // =========================================================================
    // Device Management Integration
    // =========================================================================

    /**
     * Set the current device for capture
     */
    void setCurrentDevice(int deviceId);

    /**
     * Get current capture device
     */
    int getCurrentDevice() const;

    /**
     * Check if a device supports CUDA graphs
     */
    bool deviceSupportsGraphs(int deviceId) const;

    /**
     * Get devices that support graphs
     */
    std::vector<int> getGraphCapableDevices() const;

    // =========================================================================
    // Statistics & Debug
    // =========================================================================

    /**
     * Get total number of graphs created
     */
    size_t getTotalGraphsCreated() const { return _totalGraphsCreated; }

    /**
     * Get total graph executions
     */
    size_t getTotalExecutions() const { return _totalExecutions; }

    /**
     * Get average execution time
     */
    double getAverageExecutionTimeMs() const;

    /**
     * Print scheduler statistics
     */
    void printStats() const;

    /**
     * Enable/disable verbose logging
     */
    void setVerbose(bool verbose) { _config.verbose = verbose; }

    /**
     * Reset all statistics
     */
    void resetStats();

private:
    CudaGraphScheduler();
    ~CudaGraphScheduler();

    // Internal helpers
    void initializeDeviceCapabilities();
    cudaStream_t getStreamForContext(LaunchContext* context);
    int getDeviceForContext(LaunchContext* context);
    void cleanupExpiredCacheEntries();

    // Configuration
    CudaGraphConfig _config;

    // Capture state per stream
    struct CaptureState {
        bool isCapturing = false;
        int deviceId = 0;
        cudaStream_t stream = nullptr;
        double startTime = 0.0;
        std::vector<std::string> capturedOps;
    };
    std::unordered_map<cudaStream_t, CaptureState> _captureStates;

    // Graph cache (LRU)
    std::unordered_map<std::string, std::unique_ptr<GraphCacheEntry>> _graphCache;
    std::list<std::string> _cacheAccessOrder;  // For LRU eviction

    // Pipeline management
    struct Pipeline {
        std::vector<std::shared_ptr<CudaGraphHandle>> stages;
        std::vector<cudaEvent_t> events;
        int currentStage = 0;
        bool isActive = false;
    };
    std::unordered_map<int, Pipeline> _pipelines;
    std::atomic<int> _nextPipelineId{0};

    // Device capabilities
    std::unordered_map<int, bool> _deviceGraphSupport;

    // Statistics
    std::atomic<size_t> _totalGraphsCreated{0};
    std::atomic<size_t> _totalExecutions{0};
    std::atomic<size_t> _cacheHits{0};
    std::atomic<size_t> _cacheMisses{0};
    double _totalExecutionTimeMs = 0.0;

    // Synchronization
    mutable std::mutex _mutex;
    mutable std::mutex _cacheMutex;
    mutable std::mutex _captureMutex;

    // Thread-local current device (accessor pattern for MSVC C2492 compatibility)
    static int& currentDeviceRef();
};

/**
 * RAII wrapper for graph capture
 * Usage:
 *   {
 *     CudaGraphCaptureScope scope(context);
 *     // ... operations to capture ...
 *   } // Graph captured and ready
 *   auto graph = scope.getGraph();
 */
class SD_LIB_EXPORT CudaGraphCaptureScope {
public:
    explicit CudaGraphCaptureScope(LaunchContext* context = nullptr);
    CudaGraphCaptureScope(cudaStream_t stream, int deviceId);
    ~CudaGraphCaptureScope();

    // Non-copyable
    CudaGraphCaptureScope(const CudaGraphCaptureScope&) = delete;
    CudaGraphCaptureScope& operator=(const CudaGraphCaptureScope&) = delete;

    /**
     * Get the captured graph (available after scope ends)
     */
    std::shared_ptr<CudaGraphHandle> getGraph();

    /**
     * Check if capture was successful
     */
    bool isValid() const { return _captureStarted; }

    /**
     * Abort capture (don't create graph)
     */
    void abort();

private:
    LaunchContext* _context;
    cudaStream_t _stream;
    int _deviceId;
    bool _captureStarted = false;
    bool _aborted = false;
    std::shared_ptr<CudaGraphHandle> _graph;
};

/**
 * Helper function to execute with graph capture
 * @param key Cache key for the graph
 * @param func Function containing operations to capture/execute
 * @param context LaunchContext for execution
 */
inline GraphExecutionResult executeWithGraph(
    const std::string& key,
    std::function<void()> func,
    LaunchContext* context = nullptr
) {
    auto& scheduler = CudaGraphScheduler::getInstance();
    auto graph = scheduler.getOrCapture(key, func, context);
    return scheduler.execute(graph, context);
}

}  // namespace cuda
}  // namespace sd

#endif  // SD_CUDA

#endif  // SD_CUDA_GRAPH_SCHEDULER_H
