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

#ifndef LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
#define LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <graph/generated/graph_generated.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/common.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef SD_CUDA
#include <execution/cuda/CudaGraphScheduler.h>
#endif

// GraphBackend.h defines CompilationAuditEntry and the GraphBackend base class.
// Concrete backends (OneDnnGraphBackend, AclGraphBackend) are included conditionally in .cpp.
#include <graph/GraphBackend.h>

namespace sd {
namespace graph {

// FlatGraph is in the ::graph namespace (FlatBuffer-generated)

/**
 * Source type constants for inputSourceTypes, mirroring DynamicShapeSlot.java.
 */
enum NativeSourceType : int8_t {
  SOURCE_CONSTANT = 0,
  SOURCE_VARIABLE = 1,
  SOURCE_PLACEHOLDER = 2,
  SOURCE_OP_OUTPUT = 3,
};

/**
 * Per-op descriptor with pre-compiled wiring for the native plan executor.
 * Mirrors DynamicShapeSlot.java but uses C++ types.
 *
 * Index conventions for inputSourceIndices:
 *   >= 0: index into the flat outputSlots array (from a prior op's output)
 *   <  0: index into external arrays: -(index + 1) into the external input array
 */
struct NativeSlot {
  // Op identification
  LongType opHash;                         // Op hash for lookup
  sd::ops::DeclarableOp* op;       // Resolved at compile time (not owned)
  std::string opName;                      // For diagnostics

  // Input wiring
  int numInputs;
  int* inputSourceIndices;                 // >=0: prior slot, <0: external (-(idx+1))
  int8_t* inputSourceTypes;               // NativeSourceType values

  // Output wiring
  int numOutputs;
  int* outputSlotIndices;                  // Flat slot indices for each output

  // Frozen op arguments
  LongType* iArgs;    int numIArgs;
  double* tArgs;      int numTArgs;
  bool* bArgs;        int numBArgs;
  DataType* dArgs;    int numDArgs;

  // Execution flags
  bool needsZeroedOutput;
  bool isDataDependent;
  bool outputShapeDependsOnInputValues;
  bool needsIntLongSync;
  bool isCustomOp;
  bool isIdentityOp;                       // Identity: output = input, skip execution
  bool inPlaceFused;                       // In-place fused: output reuses input buffer (set by FusionPass)
  int inPlaceFusedInputIdx;                // Which input index to reuse as output (-1 = not fused)
  int targetDeviceId;                      // -1 = auto

  // Legacy op support for ops not registered in OpRegistrator
  // (exp, log, abs, neg, sqrt, sin, cos, etc.)
  // legacyOpType: 0=not legacy, 1=TransformSame, 2=TransformStrict,
  //               3=TransformFloat, 4=TransformBool, 5=Scalar, 6=PairwiseTransform
  int legacyOpType;
  int legacyOpNum;

  // Per-slot shape cache
  LongType cachedShapeKey;
  std::vector<const LongType*> cachedOutputShapes;   // Cached shape infos (not owned)
  bool shapeCacheValid;

  // Shape static analysis: true if output shape never changes between executions.
  // Determined at plan construction time by analyzing input dependencies.
  // Shape-static slots have their caches preserved across clearShapeCaches() calls.
  bool shapeStatic;

  // Frozen context: after the first shapes-frozen execution, the context is
  // fully configured with the same input/output arrays and arguments.
  // On subsequent executions, skip all setup and just call op->execute().
  bool frozenContextReady;

  NativeSlot()
      : opHash(0), op(nullptr), numInputs(0), inputSourceIndices(nullptr),
        inputSourceTypes(nullptr), numOutputs(0), outputSlotIndices(nullptr),
        iArgs(nullptr), numIArgs(0), tArgs(nullptr), numTArgs(0),
        bArgs(nullptr), numBArgs(0), dArgs(nullptr), numDArgs(0),
        needsZeroedOutput(true), isDataDependent(false),
        outputShapeDependsOnInputValues(false), needsIntLongSync(false),
        isCustomOp(true), isIdentityOp(false),
        inPlaceFused(false), inPlaceFusedInputIdx(-1), targetDeviceId(-1),
        legacyOpType(0), legacyOpNum(-1),
        cachedShapeKey(0), shapeCacheValid(false), shapeStatic(false),
        frozenContextReady(false) {}

  ~NativeSlot() {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
    delete[] iArgs;
    delete[] tArgs;
    delete[] bArgs;
    delete[] dArgs;
  }

  // No copy
  NativeSlot(const NativeSlot&) = delete;
  NativeSlot& operator=(const NativeSlot&) = delete;

  // Move OK
  NativeSlot(NativeSlot&& other) noexcept;
  NativeSlot& operator=(NativeSlot&& other) noexcept;
};

/**
 * Graph segment for CUDA Graph capture.
 * A contiguous range of slots that can be captured as a single CUDA graph.
 *
 * Lifecycle:
 *   executionCount == 0: warm-up pass (slot-by-slot, populates slot cache)
 *   executionCount == 1: capture pass (ops recorded into CUDA graph, then launched)
 *   executionCount >= 2: replay pass (cached graph launched directly)
 */
struct GraphSegment {
  int startSlot;
  int endSlot;
  bool isCapturable;

  // Shape key for cache invalidation
  LongType shapeKey;

  // Execution tracking
  int executionCount;

  // If true, never attempt graph capture/compilation for this segment.
  // Set for permanent failures (capture invalidation, host-only ops, address instability).
  // NOT set for OOM failures — those use the retry mechanism below.
  bool captureFailed;

  // OOM retry mechanism: instead of permanently disabling capture on allocation failures,
  // retry after a cooldown period. Memory pressure decreases as other segments get captured
  // (graph replay uses less memory than slot-by-slot due to kernel fusion).
  int captureOomRetries;              // Number of OOM retries so far
  int captureRetryAfterExec;          // Don't attempt capture until executionCount >= this
  static constexpr int MAX_OOM_RETRIES = 3;
  static constexpr int RETRY_INTERVAL = 4;  // Retry every N executions

#ifdef SD_CUDA
  // Cached CUDA graph for replay
  std::shared_ptr<sd::cuda::CudaGraphHandle> cachedGraph;
  LongType cachedShapeKey;

  // ── Capture buffers ──────────────────────────────────────────────────
  // CUDA graphs record exact GPU memory addresses during capture.
  // External inputs (position_ids, attention_mask) get recreated each
  // decoder step with new GPU addresses. Instead of checking/invalidating
  // on address change, we allocate fixed-address "capture buffers" and
  // copy external input data into them before each graph replay.
  // The graph always references these stable addresses.
  struct CaptureBuffer {
    NDArray* buffer;           // Fixed-address GPU buffer (owned)
    int externalInputIndex;    // Which external input this maps to (-1 = cross-segment)
    int crossSegmentSlotIdx;   // Which output slot this maps to (for cross-segment inputs)
    size_t capturedSize;       // Size in bytes at capture time
    const void* lastSourcePtr; // Last source GPU address — skip copy if unchanged (static weights)
    bool initialCopyDone;      // True after first successful copy

    CaptureBuffer() : buffer(nullptr), externalInputIndex(-1),
                      crossSegmentSlotIdx(-1), capturedSize(0),
                      lastSourcePtr(nullptr), initialCopyDone(false) {}
  };
  std::vector<CaptureBuffer> captureBuffers;

  // Legacy address key (kept for fallback diagnostics but no longer used for invalidation)
  LongType capturedInputAddrKey;

  // Capture workspace: pre-allocated GPU buffer for PointersManager temporaries.
  // Eliminates cudaMallocAsync/cudaFreeAsync graph nodes during capture.
  // Persists for graph lifetime — kernel params reference these addresses.
  void* captureWorkspacePtr = nullptr;
  size_t captureWorkspaceBytes = 0;
#endif

  // ── Adaptive splitting for unstable segments ──────────────────────────────
  // If a segment's shape key changes for INSTABILITY_THRESHOLD consecutive
  // executions after warmup, it likely contains mixed stable + unstable ops
  // (e.g. KV-cache-growing concat inside a large FFN+attention block).
  // Instead of permanently marking it slot-by-slot, split it at the midpoint
  // so stable halves can eventually capture and unstable halves get further
  // split until small enough to run slot-by-slot with minimal overhead.
  // MIN_SPLIT_SIZE prevents infinite splitting: segments at this size or below
  // are permanently marked slot-by-slot.
  int consecutiveShapeChanges;
  bool needsSplit;
  // Require 2 consecutive shape changes before splitting, to filter one-time
  // events (e.g., first-step initialization differences).  KV-growing segments
  // change every step and will reliably trigger at threshold=2.
  static constexpr int INSTABILITY_THRESHOLD = 2;
  static constexpr int MIN_SPLIT_SIZE = 5;

  GraphSegment()
      : startSlot(0), endSlot(0), isCapturable(false), shapeKey(0),
        executionCount(0), captureFailed(false),
        captureOomRetries(0), captureRetryAfterExec(0),
        consecutiveShapeChanges(0), needsSplit(false)
#ifdef SD_CUDA
        , cachedShapeKey(0), capturedInputAddrKey(0)
#endif
  {}
};

/**
 * Describes a single KV cache output→input mapping for native KV cache retention.
 * After execute(), the new KV entry at the last position of the present output
 * is scattered to the specified position in the static past input buffer.
 */
struct KvCacheMapping {
  int presentOutputSlotIdx;   // Absolute index into outputSlots_ for the present KV output
  int pastInputExternalIdx;   // Index into external inputs for the past KV buffer
  int seqDim;                 // Which dimension is the sequence dim (typically 2)

  KvCacheMapping() : presentOutputSlotIdx(-1), pastInputExternalIdx(-1), seqDim(2) {}
};

/**
 * Native C++ plan executor that replaces the Java DynamicShapePlanExecutor.
 *
 * Executes the entire pre-compiled plan in C++ with a single JNI call,
 * eliminating per-op Java→JNI→C++ round-trip overhead (~15-20μs per op).
 *
 * Two construction paths:
 * 1. fromSerializedPlan() - from serialized binary plan (sent from Java via JNI)
 * 2. fromFlatGraph() - from FlatGraph + variables (C++-native loading via SdzReader)
 */
class SD_LIB_EXPORT NativeDynamicShapePlan {
 public:
  /**
   * Construct from a serialized binary plan (sent from Java).
   *
   * Binary format:
   *   Header: magic("DSP1") + version(int32) + numSlots(int32) + totalOutputSlots(int32)
   *           + numExternalInputs(int32) + numRequestedOutputs(int32)
   *   Per-slot: opHash(int64), numInputs(int32), numOutputs(int32),
   *             inputSourceIndices[numInputs](int32),
   *             inputSourceTypes[numInputs](int8),
   *             outputSlotIndices[numOutputs](int32),
   *             numIArgs/numTArgs/numBArgs/numDArgs(int32 each),
   *             iArgs[](int64), tArgs[](double), bArgs[](bool), dArgs[](int32),
   *             flags: needsZeroedOutput(bool), isDataDependent(bool),
   *                    outputShapeDependsOnInputValues(bool), needsIntLongSync(bool),
   *                    isCustomOp(bool), targetDeviceId(int32)
   *   Release schedule: for each step: count(int32) + slotIndices[count](int32)
   *   Requested outputs: for each output: slotIndex(int32)
   */
  static NativeDynamicShapePlan* fromSerializedPlan(const void* data, LongType size);

  /**
   * Construct from a FlatGraph + pre-loaded variables.
   * Used for pure C++ model loading (no Java needed).
   */
  static NativeDynamicShapePlan* fromFlatGraph(
      const ::graph::FlatGraph* graph,
      const std::unordered_map<std::string, NDArray*>& variables,
      const std::vector<std::string>& requestedOutputs);

  ~NativeDynamicShapePlan();

  // No copy
  NativeDynamicShapePlan(const NativeDynamicShapePlan&) = delete;
  NativeDynamicShapePlan& operator=(const NativeDynamicShapePlan&) = delete;

  /**
   * Execute the full plan.
   *
   * @param externalInputs  Array of NDArray* for constants/variables/placeholders
   * @param numExternalInputs  Number of external inputs
   * @param requestedOutputs  Pre-allocated array to receive output NDArrays (caller owns)
   * @param numRequestedOutputs  Number of requested outputs
   * @param stream  CUDA stream (nullptr for CPU)
   * @return Status::OK on success
   */
  Status execute(
      NDArray** externalInputs, int numExternalInputs,
      NDArray** requestedOutputs, int numRequestedOutputs,
      void* stream);

  /**
   * Clear per-slot shape caches for shape-dynamic slots only.
   * Shape-static slots (those with no transitive dependency on placeholders)
   * retain their caches, avoiding redundant shape inference on every step.
   * Must be called when placeholder shapes change between executions.
   */
  void clearShapeCaches();

  /**
   * Force-clear ALL shape caches unconditionally (including static slots).
   * Use for session reset or model reload scenarios.
   */
  void clearAllShapeCachesForce();

  /**
   * Configure KV cache retention. After this, execute() will scatter new KV entries
   * from present output slots into static past input buffers, avoiding 60 copyBuffer
   * round-trips per decode step.
   *
   * @param mappings       Flat array of (presentSlotIdx, pastExtIdx, seqDim) triples
   * @param numMappings    Number of KV cache mappings (e.g. 60 for 30-layer model)
   * @param maxKvLen       Maximum KV cache length (static buffer size along seqDim)
   * @param initialPos     Initial write position (prefillLen)
   */
  void configureKvCacheRetention(const int* mappings, int numMappings, int maxKvLen, int initialPos);

  /**
   * Advance the KV cache write position by 1.
   * @return the new position value
   */
  int advanceKvCachePosition();

  /**
   * Reset the KV cache write position.
   */
  void resetKvCachePosition(int newPos);

  /**
   * Get the number of external inputs expected by this plan.
   */
  int getNumExternalInputs() const { return numExternalInputs_; }

  /**
   * Get the number of requested outputs.
   */
  int getNumRequestedOutputs() const { return numRequestedOutputs_; }

  /**
   * Get the total number of slots (ops) in the plan.
   */
  int getNumSlots() const { return numSlots_; }

  /**
   * Get the total number of output slots (intermediate + final).
   */
  int getTotalOutputSlots() const { return totalOutputSlots_; }

  /**
   * Get plan segments (for CUDA Graphs integration).
   */
  const std::vector<GraphSegment>& getSegments() const { return segments_; }

  /**
   * Get mutable plan segments (for clearing CUDA graph timelines, etc.).
   */
  std::vector<GraphSegment>& getSegmentsMutable() { return segments_; }

  /**
   * Get CUDA Graph execution statistics.
   * Returns number of segments captured as CUDA graphs.
   */
  int getNumCapturedGraphSegments() const;

  /**
   * Get total number of CUDA graph replays across all segments.
   */
  int getTotalGraphReplays() const;

  /**
   * Enable/disable CUDA Graphs for this plan.
   * Default: disabled (slot-by-slot execution).
   */
  void setCudaGraphsEnabled(bool enabled) { cudaGraphsEnabled_ = enabled; }
  bool isCudaGraphsEnabled() const { return cudaGraphsEnabled_; }


  /**
   * Enable/disable "shapes frozen" mode. When enabled:
   * - clearShapeCaches() becomes a no-op (shapes are known constant)
   * - Shape key computation is skipped for slots with valid cached shapes
   * - nullify() is skipped for slots with needsZeroedOutput=false
   *
   * Use this during static KV decode where all external input shapes
   * are guaranteed to be constant across decode steps.
   * The first execution after enabling will still do full shape inference
   * to populate the cache; subsequent executions skip shape work entirely.
   */
  void setShapesFrozen(bool frozen) {
    bool wasFrozen = shapesFrozen_;
    shapesFrozen_ = frozen;
    if (frozen && !wasFrozen) {
      // Merge segments now that value-dep-shape ops are safe to capture
      // (their input values never change → output shapes are constant).
      int oldSegCount = (int)segments_.size();
      rebuildSegmentsForFrozenShapes();
      sd_printf("NativeDSP::setShapesFrozen: merged %d -> %d segments\n",
                oldSegCount, (int)segments_.size());
      // Reset executeCount so step 0 = warmup, step 1+ = capture/replay
      executeCount_ = 0;
    }
    if (!frozen) {
      // Reset frozen context state when unfreezing — shapes may change
      for (int i = 0; i < numSlots_; i++) {
        slots_[i].frozenContextReady = false;
      }
    }
  }
  bool isShapesFrozen() const { return shapesFrozen_; }

  /**
   * Enable/disable per-execution timing breakdown logging.
   * When enabled, prints phase-level timing after each execute() call.
   */
  void setExecutionTimingEnabled(bool enabled) { executionTimingEnabled_ = enabled; }
  bool isExecutionTimingEnabled() const { return executionTimingEnabled_; }

  /**
   * Enable/disable trace logging for DSP execution decisions.
   * When enabled, logs segment dispatch, graph capture/replay decisions,
   * and error paths via sd_printf (to stderr).
   * Controlled by -Dnd4j.dsp.trace system property.
   */
  void setTraceEnabled(bool enabled) { traceEnabled_ = enabled; }
  bool isTraceEnabled() const { return traceEnabled_; }

  /**
   * Get the capture audit for the most recent CUDA graph capture.
   * Each entry shows which op contributed how many CUDA graph nodes.
   * Empty if no capture has been performed or CUDA graphs are disabled.
   */
#ifdef SD_CUDA
  const std::vector<sd::cuda::CaptureAuditEntry>& getLastCaptureAudit() const { return lastCaptureAudit_; }

  /**
   * Get ops that contributed zero CUDA graph nodes during the last capture.
   * These are "host-only" ops whose work is NOT replayed on graph replay,
   * which means their outputs will be STALE on the 2nd+ execution.
   * This is a critical diagnostic for debugging graph correctness issues.
   */
  std::vector<sd::cuda::CaptureAuditEntry> getHostOnlyOps() const;

  /**
   * Print the full capture audit to stderr.
   * Shows every op in the last captured segment with its CUDA node contribution.
   * Flags host-only ops with a warning marker.
   */
  void printCaptureAudit() const;

  /**
   * Validate that the captured CUDA graph covers all ops in the segment.
   * Returns true if every op contributed at least one CUDA graph node.
   * When debug/verbose mode is on, this also prints the full audit and
   * asserts that no host-only ops exist.
   *
   * @param segmentIndex  Which segment to validate (-1 for all segments)
   * @return true if all ops contributed CUDA graph nodes, false if any host-only ops found
   */
   bool validateCapturedGraph(int segmentIndex = -1) const;

  /**
   * Set maximum sizes for specific output slots (KV cache pre-allocation).
   * 
   * When set, these slots will be pre-allocated at the specified maximum size during
   * the first execution, keeping buffer addresses stable across all subsequent steps.
   * This enables CUDA graph capture for models with growing KV caches.
   * 
   * @param slotIndices   Array of output slot indices to pre-allocate
   * @param maxSizes      Array of maximum sizes (in number of elements, not bytes)
   * @param numSlots      Number of entries in the arrays
   * 
   * Usage for KV cache:
   *   - Call once before first execution
   *   - For each KV cache output slot, set max size = batch * numHeads * maxSeqLen * headDim
   */
  void setOutputSlotMaxSizes(const int* slotIndices, const LongType* maxSizes, int numSlots);
  
  /**
   * Get the current KV cache sequence position (for slice-based KV cache access).
   * This is the position where new KV entries will be written.
   */
  int getKvCachePosition() const { return kvCachePosition_; }
  
  /**
   * Set the KV cache sequence position.
   * Call before each decode step to tell the attention op where to write new KV.
   */
  void setKvCachePosition(int pos);
  
  /**
   * Set the maximum KV cache length (used with pre-allocated KV cache).
   * When set, attention outputs are pre-allocated at [batch, numHeads, maxKvLen, headDim].
   */
  void setMaxKvCacheLength(int maxLen);
#endif

  /**
   * Validate that a compiled CPU graph (oneDNN/ACL) covers all ops in the segment.
   * Returns true if every op was compiled by the backend.
   * When any ops are missing, logs warnings and returns false.
   *
   * @param segmentIndex  Which segment to validate (-1 for all segments)
   * @return true if all ops were compiled, false if any were skipped
   */
  bool validateCompiledCpuGraph(int segmentIndex = -1) const;

  /**
   * Print the compilation audit for CPU graph backends.
   * Shows every op with its compilation status (compiled vs skipped).
   */
  void printCompilationAudit() const;

  friend class NativePlanCompiler;

 private:
  NativeDynamicShapePlan();

  // Slot data
  NativeSlot* slots_;
  int numSlots_;
  int totalOutputSlots_;
  int numExternalInputs_;

  // Release schedule: releaseAtStep_[stepIdx] = array of slot indices to release
  int** releaseAtStep_;
  int* releaseAtStepCounts_;

  // Requested output mapping
  int* requestedOutputSlotIndices_;
  int numRequestedOutputs_;

  // Execution state (reused across calls)
  NDArray** outputSlots_;              // Current output slot values
  NDArray** slotArrayCache_;           // Per-slot cached arrays for reuse
  bool* slotIsViewProducer_;           // View producer flags (learned from first exec)
  Context** contextPool_;              // Pre-allocated Context pool
  bool viewProducerDetectionDone_;

  // Graph segments for CUDA Graphs
  std::vector<GraphSegment> segments_;

  // Memory management: evicted NDArrays awaiting deletion
  std::vector<NDArray*> pendingClose_;
  size_t pendingCloseBytes_;

  // CUDA Graphs control
  bool cudaGraphsEnabled_;
  int totalGraphReplays_;

  // Shapes-frozen optimization: when enabled, skip shape cache clearing,
  // shape key computation, and unnecessary output zeroing between executions.
  // Use when all external input shapes are guaranteed constant (e.g., static KV decode).
  bool shapesFrozen_;
  int executeCount_;  // Tracks executions since shapes were frozen

  // Per-execution timing breakdown (enabled by setExecutionTimingEnabled)
  bool executionTimingEnabled_;

  // Trace logging for execution decisions (enabled by setTraceEnabled / -Dnd4j.dsp.trace)
  bool traceEnabled_ = false;

#ifdef SD_CUDA
  // Capture audit: per-op CUDA node contribution tracking
  std::vector<sd::cuda::CaptureAuditEntry> lastCaptureAudit_;
#endif

  // Compilation audit: per-op compilation status for CPU graph backends
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Owned legacy ops created during deserialization
  // (for ops not registered in OpRegistrator)
  std::vector<sd::ops::DeclarableOp*> ownedLegacyOps_;

  // ── Untracked output cache ──────────────────────────────────────────
  // Ops with untracked outputs (outputSlotIndices[i] < 0) allocate a
  // temporary buffer every execution. During CUDA graph capture, these
  // allocations fail because cudaMallocAsync on the captured stream is
  // deferred. This cache stores untracked outputs from the warmup pass
  // so they can be reused during capture and shapes-frozen execution.
  // Indexed as untrackedOutputCache_[slotIndex * maxOutputsPerSlot + outputIndex].
  NDArray** untrackedOutputCache_;
  int untrackedOutputCacheSize_;
  static constexpr int MAX_OUTPUTS_PER_SLOT = 8;  // Max outputs per op (most ops have 1-3)

  // ── KV cache retention ──────────────────────────────────────────────
  // When configured, present KV outputs are not returned to Java.
  // Instead, C++ extracts the new KV entry and scatters it into the
  // static input buffer, avoiding 60 copyBuffer round-trips per step.
  bool kvCacheRetentionEnabled_;
  int kvCachePosition_;           // Current write position in static buffers
  int kvCacheMaxLen_;             // Maximum length of static KV buffers (dim along seqDim)
  int kvCacheNumMappings_;
  KvCacheMapping* kvCacheMappings_;  // Array of output→input mappings (owned)

  // Internal methods
  void scatterKvEntries(NDArray** externalInputs, int numExt, void* stream);
  Status executeSlot(int slotIdx, NDArray** externalArrays, int numExt, void* stream);
  LongType computeShapeKey(NativeSlot& slot, NDArray** inputs, int numInputs);
  LongType computeSegmentShapeKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  LongType computeSegmentInputAddrKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  void flushPendingClose(void* stream);
  void buildSegments();
  void rebuildSegmentsForFrozenShapes();
  void maybeSplitUnstableSegments();

  // Segment execution strategies
  Status executeSegmentSlotBySlot(GraphSegment& seg, NDArray** externalArrays,
                                  int numExt, void* stream);
#ifdef SD_CUDA
  Status executeSegmentWithGraph(GraphSegment& seg, NDArray** externalArrays,
                                 int numExt, void* stream);
#endif

  // CPU graph backend (oneDNN Graph or ACL Dynamic Fusion)
  GraphBackend* cpuGraphBackend_;
  bool cpuGraphBackendChecked_;

  Status executeSegmentWithCpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getCpuGraphBackend();

  // GPU graph backend (Triton GPU compiler)
  GraphBackend* gpuGraphBackend_;
  bool gpuGraphBackendChecked_;

  Status executeSegmentWithGpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getGpuGraphBackend();

#ifdef SD_CUDA
  // Pre-allocated cuBLAS workspace for CUDA graph capture.
  // During graph capture, cuBLAS internal cudaMalloc calls on stream 0
  // break capture on the named stream. Providing an explicit workspace
  // prevents cuBLAS from doing any internal allocations.
  void* cublasWorkspaceBuffer_;
  size_t cublasWorkspaceSize_;
  void ensureCublasWorkspace(size_t minBytes);
  void setCublasWorkspaceForCapture(void* stream);
  void restoreCublasWorkspaceAfterCapture(void* stream);
#endif

  // Max-allocation mode for KV cache outputs
  // Maps output slot index -> max number of elements to pre-allocate
  std::unordered_map<int, LongType> outputSlotMaxSizes_;
  // Tracks which slots have been pre-allocated at max size
  std::unordered_set<int> maxAllocatedSlots_;
  // Maximum KV cache length (for pre-allocated attention outputs)
  int maxKvCacheLen_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
