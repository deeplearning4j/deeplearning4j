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
#include <graph/DspDiagnostics.h>
#include <graph/generated/graph_generated.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/common.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <graph/GraphReplayHandle.h>
#include <graph/SlotBufferOwnership.h>
#include <graph/PlanDefinition.h>
#include <graph/ExecutionState.h>

// Forward declaration — full definition in DspStreamGuard.h
// (included only in .cpp/.cu files that use the guard)

#ifdef SD_CUDA
#include <execution/cuda/CudaGraphScheduler.h>
#endif

// GraphBackend.h defines CompilationAuditEntry and the GraphBackend base class.
// Concrete backends are included conditionally in .cpp:
//   GPU: TritonGraphBackend, NvrtcGraphBackend, PtxGraphBackend
//   CPU: MlxGraphBackend, OneDnnGraphBackend, AclGraphBackend,
//        NnapiGraphBackend, ArmHybridGraphBackend, MlirCpuGraphBackend
#include <graph/GraphBackend.h>

namespace sd {
namespace graph {

// Forward declaration for NVRTC JIT kernel handle (defined in NvrtcKernelCache.h)
#ifdef SD_CUDA
struct NvrtcKernelHandle;
#endif

/**
 * JIT compilation mode for DSP segment execution.
 * Configured via -Dnd4j.dsp.jitMode system property.
 */
enum class JitMode : int {
  GRAPH_ONLY = 0,    // CUDA graph capture/replay only (default)
  JIT_ONLY = 1,      // NVRTC JIT only (skip graph capture for element-wise segments)
  GRAPH_PLUS_JIT = 2 // Try JIT first, fall back to graph capture for non-fusible segments
};

/**
 * Graph execution mode — controls which backend the DSP executor uses.
 * Set from Java via SameDiff.setGraphExecutionMode().
 * Must match GraphExecutionMode.java native codes.
 */
enum class GraphExecutionMode : int {
  GEM_AUTO = 0,         // Try Triton → NVRTC → PTX → CUDA Graphs → slot-by-slot
  GEM_SLOT_BY_SLOT = 1, // Execute each op individually (no fusion, no graphs)
  GEM_CUDA_GRAPHS = 2,  // CUDA graph capture/replay only
  GEM_NVRTC_JIT = 3,    // Force NVRTC JIT backend for fusible segments
  GEM_PTX_JIT = 4,      // Force PTX template backend for fusible segments
  GEM_TRITON = 5,       // Force Triton MLIR backend for fusible segments
  GEM_MLX = 6,          // Force MLX Apple Silicon backend for fusible segments
  GEM_ARM_HYBRID = 7,   // Force ARM Hybrid (MLIR CPU + Vulkan) backend
  GEM_NNAPI = 8,        // Force Android NNAPI backend for hardware acceleration
  GEM_HIP_GRAPHS = 9,   // HIP graph capture/replay (AMD ROCm) — mirrors CUDA graphs
  GEM_LEVELZERO = 10,   // Intel Level Zero mutable command list replay
  GEM_VULKAN = 11,      // Vulkan compute command buffer replay
  GEM_METAL = 12,       // Metal indirect command buffer replay (Apple GPU)
  GEM_TPU = 13,         // TPU HLO compilation + PJRT execution caching
  GEM_HEXAGON = 14      // Hexagon-MLIR NPU compilation + command list replay
};

/**
 * ExecutionPhase — the ACTUAL runtime execution mode of a segment.
 *
 * Unlike GraphExecutionMode (which is the user's PREFERENCE), ExecutionPhase
 * tracks what a segment is ACTUALLY doing right now. This enables programmatic
 * assertions about execution stage at both C++ and Java levels.
 *
 * Lifecycle: WARMUP → COMPILING → COMPILED → REPLAYING (capturable segments)
 *            SLOT_BY_SLOT (non-capturable segments, always)
 */
enum class ExecutionPhase : uint8_t {
  WARMUP = 0,          // First execution — slot-by-slot for shape population
  COMPILING = 1,       // Backend is compiling (Triton, NVRTC, CUDA graph capture)
  COMPILED = 2,        // Compiled, first post-compile execution
  REPLAYING = 3,       // Steady state — graph replay or compiled kernel reuse
  SLOT_BY_SLOT = 4,    // Non-capturable segment — always slot-by-slot
};

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
 * Control flow type for DSP slots, mirroring DynamicShapeSlot.java CF constants.
 */
enum ControlFlowType : int8_t {
  CF_NONE = 0,
  CF_SWITCH = 1,
  CF_MERGE = 2,
  CF_ENTER = 3,
  CF_EXIT = 4,
  CF_NEXT_ITERATION = 5,
  CF_LOOP_COND = 6,
};

/**
 * Describes a while-loop region in the DSP plan.
 * Mirrors DynamicShapePlan.LoopRegion in Java.
 */
struct LoopRegion {
  int mergeSlot;        // Jump-back target (Merge)
  int switchSlot;       // Switch that gates the loop
  int nextIterSlot;     // NextIteration (triggers jump-back)
  int exitSlot;         // Exit (output when loop ends)
  int bodyStartSlot;    // First body slot after Switch
  int bodyEndSlot;      // Last body slot (= nextIterSlot)

  LoopRegion() : mergeSlot(-1), switchSlot(-1), nextIterSlot(-1),
                 exitSlot(-1), bodyStartSlot(-1), bodyEndSlot(-1) {}
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
  std::string* sArgs; int numSArgs;

  // Execution flags
  bool needsZeroedOutput;
  bool isDataDependent;
  bool outputShapeDependsOnInputValues;
  bool needsIntLongSync;
  bool isCustomOp;
  bool isIdentityOp;                       // Identity: output = input, skip execution
  bool isViewCapableOp;                    // View-capable: reshape/expand_dims/squeeze — output can share input buffer
  bool inPlaceFused;                       // In-place fused: output reuses input buffer (set by FusionPass)
  int inPlaceFusedInputIdx;                // Which input index to reuse as output (-1 = not fused)

  // Structural iArg count: first N iArgs are structural (masks, mode flags, axis),
  // rest are data that could come from input tensors.
  // -1 = all iArgs are structural (default for most ops).
  // Used by plan compiler to cap iArgs when data comes from input tensors.
  int structuralIArgCount;

  // Fused elementwise chain: head slot dispatches single fused kernel for entire chain
  bool isFusedChainHead;                     // Head — dispatches fused kernel
  int fusedChainLength;                      // Chain length (including head)
  int fusedChainOpCodes[8] = {};              // FusedElemOp codes for each op in chain
  int fusedChainSlots[8] = {};               // Slot indices of all chain members
  int fusedChainSecondaryInputSources[8] = {};  // inputSourceIndices for binary ops' 2nd input (-1 if unary)
  bool isFusedChainTail;                     // Tail — skip execution entirely (head already computed result)

  int targetDeviceId;                      // -1 = auto

  // Control flow support
  ControlFlowType controlFlowType;         // CF_NONE for regular ops
  int loopBackTarget;                      // For NextIteration: step index of Merge to jump back to (-1 otherwise)
  int loopRegionIndex;                     // Index into loopRegions_ (-1 if not in a loop)

  // Legacy op support for ops not registered in OpRegistrator
  // (exp, log, abs, neg, sqrt, sin, cos, etc.)
  // legacyOpType: 0=not legacy, 1=TransformSame, 2=TransformStrict,
  //               3=TransformFloat, 4=TransformBool, 5=Scalar, 6=PairwiseTransform
  int legacyOpType;
  int legacyOpNum;

  // Per-slot shape cache
  LongType cachedShapeKey;
  std::vector<const LongType*> cachedOutputShapes;   // Cached shape infos (not owned)

  // Shape static analysis: true if output shape never changes between executions.
  // Determined at plan construction time by analyzing input dependencies.
  // Shape-static slots have their caches preserved across clearShapeCaches() calls.
  // This is an inherent property, NOT a lifecycle state.
  bool shapeStatic;

  /**
   * Slot lifecycle state machine. Replaces 3 independent booleans
   * (shapeCacheValid, frozenContextReady, frozenConstantSlot) with
   * explicit ordered states and documented transitions.
   *
   * State transitions (ordered — each state includes all prior guarantees):
   *   UNINITIALIZED → WARMUP:         First execution begins
   *   WARMUP → SHAPE_CACHED:          Shape cache populated, view status determined
   *   SHAPE_CACHED → COMPILED:        Segment compiled/captured
   *   COMPILED → FROZEN:              Shapes frozen, context reuse enabled
   *   FROZEN → FROZEN_CONSTANT:       Output never changes, skip execution entirely
   *
   * Backward transitions:
   *   Any → WARMUP:                   Plan invalidation (shape change, etc.)
   *   FROZEN/FROZEN_CONSTANT → SHAPE_CACHED:  Unfreeze
   */
  enum class SlotState : uint8_t {
    UNINITIALIZED = 0,
    WARMUP,           // First execution (shape inference + view detection)
    SHAPE_CACHED,     // Shape cache populated, view status determined
    COMPILED,         // Segment compiled/captured
    FROZEN,           // Shapes frozen, context reuse enabled
    FROZEN_CONSTANT,  // Output never changes, skip execution entirely
  };
  SlotState state_;

  NativeSlot()
      : opHash(0), op(nullptr), numInputs(0), inputSourceIndices(nullptr),
        inputSourceTypes(nullptr), numOutputs(0), outputSlotIndices(nullptr),
        iArgs(nullptr), numIArgs(0), tArgs(nullptr), numTArgs(0),
        bArgs(nullptr), numBArgs(0), dArgs(nullptr), numDArgs(0),
        sArgs(nullptr), numSArgs(0),
        needsZeroedOutput(true), isDataDependent(false),
        outputShapeDependsOnInputValues(false), needsIntLongSync(false),
        isCustomOp(true), isIdentityOp(false), isViewCapableOp(false),
        inPlaceFused(false), inPlaceFusedInputIdx(-1),
        structuralIArgCount(-1),
        isFusedChainHead(false), fusedChainLength(0), isFusedChainTail(false),
        targetDeviceId(-1),
        controlFlowType(CF_NONE), loopBackTarget(-1), loopRegionIndex(-1),
        legacyOpType(0), legacyOpNum(-1),
        cachedShapeKey(0), shapeStatic(false),
        state_(SlotState::UNINITIALIZED) {}

  // Convenience accessors that map SlotState to the old boolean semantics.
  // These allow gradual migration — callers can use these until fully converted.
  bool shapeCacheValid() const { return state_ >= SlotState::SHAPE_CACHED; }
  bool frozenContextReady() const { return state_ >= SlotState::FROZEN; }
  bool frozenConstantSlot() const { return state_ == SlotState::FROZEN_CONSTANT; }

  ~NativeSlot() {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
    delete[] iArgs;
    delete[] tArgs;
    delete[] bArgs;
    delete[] dArgs;
    delete[] sArgs;
  }

  // No copy
  NativeSlot(const NativeSlot&) = delete;
  NativeSlot& operator=(const NativeSlot&) = delete;

  // Move OK
  NativeSlot(NativeSlot&& other) noexcept;
  NativeSlot& operator=(NativeSlot&& other) noexcept;
};

/**
 * Graph segment for graph capture / backend compilation.
 * A contiguous range of slots that can be captured as a single graph.
 *
 * Split into:
 *   - Immutable definition (set at buildSegments time, never changes)
 *   - Mutable ExecState (changes every execution)
 *
 * Lifecycle (via exec.executionCount):
 *   == 0: warm-up pass (slot-by-slot, populates slot cache)
 *   == 1: capture pass (ops recorded into graph, then launched)
 *   >= 2: replay pass (cached graph launched directly)
 */
struct GraphSegment {
  // ── Immutable definition (set at buildSegments, never changes) ─────
  int startSlot;
  int endSlot;
  bool isCapturable;
  LongType shapeKey;                   // Initial shape key from buildSegments

  // User-forced backend override (empty = automatic selection via priority chain)
  std::string backendOverride;

  // Pointer to NativeDynamicShapePlan slot array cache — allows GPU backends
  // to update the slot cache when pre-allocating output arrays.
  NDArray** slotArrayCache = nullptr;

  /**
   * Formal contract for segment integrity. Validated after buildSegments()
   * and used to detect ownership/device boundary violations.
   */
  struct SegmentContract {
    bool allOpsSameCapturability = true;
    bool noMidSegmentOwnershipTransitions = true;
    bool noMidSegmentDeviceTransitions = true;
    bool shapeKeyStableOrInvalidates = true;
  };
  SegmentContract contract;

  // Constants
  static constexpr int MAX_OOM_RETRIES = 3;
  static constexpr int RETRY_INTERVAL = 4;  // Retry every N executions

  // Batch-zero entry definition (used by exec.segBatchZeroEntries)
  struct BatchZeroEntry { void* ptr; int bytes; };

  // ── Mutable execution state (changes per-execution) ────────────────
  struct ExecState {
    int executionCount = 0;

    // If true, never attempt graph capture/compilation for this segment.
    // Set for permanent failures (capture invalidation, host-only ops, address instability).
    // NOT set for OOM failures — those use the retry mechanism below.
    bool captureFailed = false;

    // OOM retry mechanism
    int captureOomRetries = 0;
    int captureRetryAfterExec = 0;

    // ── Platform-agnostic graph replay handle ────────────────────────
    // CUDA: CudaGraphReplayHandle (wraps cudaGraph_t/cudaGraphExec_t)
    // CPU: FunctionalReplayHandle (cached op dispatch, skip shape inference)
    // nullptr until first capture attempt.
    std::unique_ptr<GraphReplayHandle> replayHandle;
    LongType cachedShapeKey = 0;

    // Legacy address key (kept for fallback diagnostics)
    LongType capturedInputAddrKey = 0;

    // Hash of input DATA values for 'create' (ConstantOfShape) ops.
    LongType capturedCreateValueKey = 0;

    // ── NVRTC JIT kernel (CUDA-only) ─────────────────────────────────
#ifdef SD_CUDA
    NvrtcKernelHandle* jitKernel = nullptr;
    LongType jitShapeKey = 0;
    bool jitCompileFailed = false;
#endif

    // Symbolic shape ranges
    bool symbolicShapeEnabled = false;
    int symbolicWarmupRemaining = 0;
    void* symbolicRangeData = nullptr;  // opaque ptr to SegmentShapeProfile

    // Backend that compiled this segment ("Triton", "oneDNN", "CUDA", "slot-by-slot", etc.)
    std::string compiledByBackend;

    // Fast-replay: skip arg table refresh and EXT_INPUT_SYNC on replay.
    bool argTableStable = false;

    // Triton fallback gap ops captured in graph — must not be re-executed after replay.
    bool gapOpsCapturedInGraph = false;

    // Per-segment batch-zero entries for replay
    std::vector<BatchZeroEntry> segBatchZeroEntries;

    // Execution phase tracking — ACTUAL runtime mode (not user preference).
    ExecutionPhase currentPhase = ExecutionPhase::WARMUP;

    void reset() {
      executionCount = 0;
      captureFailed = false;
      captureOomRetries = 0;
      captureRetryAfterExec = 0;
      replayHandle.reset();
      cachedShapeKey = 0;
      capturedInputAddrKey = 0;
      capturedCreateValueKey = 0;
#ifdef SD_CUDA
      jitKernel = nullptr;
      jitShapeKey = 0;
      jitCompileFailed = false;
#endif
      symbolicShapeEnabled = false;
      symbolicWarmupRemaining = 0;
      symbolicRangeData = nullptr;
      compiledByBackend.clear();
      argTableStable = false;
      gapOpsCapturedInGraph = false;
      segBatchZeroEntries.clear();
      currentPhase = ExecutionPhase::WARMUP;
    }
  };

  ExecState exec;

  GraphSegment()
      : startSlot(0), endSlot(0), isCapturable(false), shapeKey(0)
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
   *             numIArgs/numTArgs/numBArgs/numDArgs/numSArgs(int32 each),
   *             iArgs[](int64), tArgs[](double), bArgs[](bool), dArgs[](int32),
   *             sArgs[](int32 len + UTF-8),
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
   * Configure decode input indices for direct device-side updates.
   * Call once after plan compilation. The plan will use these indices
   * to update input_ids, position_ids, and attention_mask directly on
   * device memory — no JNI putScalar or host↔device round-trips.
   *
   * @param inputIdsExtIdx    External input index for input_ids (or -1 if N/A)
   * @param positionIdsExtIdx External input index for position_ids (or -1 if N/A)
   * @param attentionMaskExtIdx External input index for attention_mask (or -1 if N/A)
   * @param maxKvLen          Maximum KV cache length (attention mask width minus 1)
   */
  void configureDecodeInputs(int inputIdsExtIdx, int positionIdsExtIdx,
                              int attentionMaskExtIdx, int maxKvLen);

  /**
   * Update decode inputs directly on device. Single call replaces 3+ putScalar calls.
   * Writes tokenId → input_ids[0,0], cachePos → position_ids[0,0],
   * and sets attention_mask[0, cachePos-1] = 1 on the GPU.
   *
   * @param externalInputs  External input array (same as passed to execute())
   * @param numExt          Number of external inputs
   * @param tokenId         The next token ID to write into input_ids
   * @param cachePos        Current cache position (for position_ids and mask update)
   * @param stream          CUDA stream for async writes
   */
  void updateDecodeInputs(NDArray** externalInputs, int numExt,
                           long long tokenId, int cachePos, void* stream);

  /**
   * Set the next decode token and cache position for automatic device-side update.
   * Call before execute(). If decode inputs are configured, execute() will
   * write these values directly to device memory before graph replay.
   *
   * @param tokenId   Next token ID
   * @param cachePos  Current cache position
   */
  void setNextDecodeToken(long long tokenId, int cachePos) {
    pendingTokenId_ = tokenId;
    pendingCachePos_ = cachePos;
    hasPendingDecodeUpdate_ = true;
  }

  /**
   * Check if decode inputs have been configured.
   */
  bool isDecodeInputsConfigured() const { return decodeInputIdsExtIdx_ >= 0 || decodeAttentionMaskExtIdx_ >= 0; }

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
   * Get raw slot array for inspection (read-only).
   */
  const NativeSlot* getSlots() const { return slots_; }

  /**
   * Get the shared immutable PlanDefinition (Phase 3).
   * Returns nullptr if plan was not compiled via standard path.
   */
  PlanDefinition* getPlanDefinition() const { return planDef_; }

  /**
   * Get the per-instance ExecutionState (Phase 4).
   * Returns nullptr if plan was not compiled via standard path.
   */
  ExecutionState* getExecutionState() const { return execState_; }

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
   * Get compilation audit for a specific segment (JSON string).
   */
  std::string getSegmentCompilationAudit(int segIdx) const;

  /**
   * Set backend priority order for segment compilation.
   */
  void setBackendPriority(const std::vector<std::string>& priority);

  /**
   * Get the current backend priority order.
   */
  const std::vector<std::string>& getBackendPriority() const { return backendPriority_; }

  /**
   * Enable/disable GPU graph capture for this plan.
   * Default: disabled (slot-by-slot execution).
   */
  void setCudaGraphsEnabled(bool enabled) { gpuGraphCaptureEnabled_ = enabled; }
  bool isCudaGraphsEnabled() const { return gpuGraphCaptureEnabled_; }

  /**
   * Set JIT compilation mode for segment execution.
   * - GRAPH_ONLY: CUDA graph capture/replay only (default)
   * - JIT_ONLY: NVRTC JIT only for element-wise segments
   * - GRAPH_PLUS_JIT: Try JIT first, fall back to graph capture
   */
  void setJitMode(JitMode mode) { jitMode_ = mode; }
  JitMode getJitMode() const { return jitMode_; }

  void setGraphExecutionMode(GraphExecutionMode mode) {
    graphExecutionMode_ = mode;
    // Reset cached backends so mode changes take effect immediately.
    gpuGraphBackendChecked_ = false;
    gpuGraphBackend_ = nullptr;
    cpuGraphBackendChecked_ = false;
    cpuGraphBackend_ = nullptr;
    // Enable GPU graph capture as fallback for all modes except SLOT_BY_SLOT.
    // JIT backends (Triton/NVRTC/PTX) need graph capture fallback when they
    // can't handle a segment (unsupported ops, etc).
    if (mode != GraphExecutionMode::GEM_SLOT_BY_SLOT) {
      gpuGraphCaptureEnabled_ = true;
    }
    // Clear GPU backend failed-compilation cache so segments that failed with
    // incomplete shapes (e.g., attention with seqK=0 before KV setup)
    // can retry when called again with correct external input shapes.
    clearGpuBackendFailedCache();
  }
  GraphExecutionMode getGraphExecutionMode() const { return graphExecutionMode_; }


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
      auto& env = Environment::getInstance();
      bool mergeSegments = env.dspFreezeMergeSegments();

      int oldSegCount = (int)segments_.size();
      if (mergeSegments && oldSegCount > 1) {
        rebuildSegmentsForFrozenShapes();
        DSP_DIAG(SEGMENT, "setShapesFrozen: merged %d -> %d segments (ND4J_DSP_FREEZE_MERGE_SEGMENTS=1)",
                  oldSegCount, (int)segments_.size());
      }

      DSP_DIAG(EXECUTE, "FROZEN_TRANSITION: unfrozen → FROZEN, "
                "%d segments, %d slots, %d extInputs, mergeSegments=%d, recompile=%d",
                (int)segments_.size(), numSlots_, numExternalInputs_,
                mergeSegments ? 1 : 0, env.dspFreezeRecompile() ? 1 : 0);

      // Clear stale cast cache from prefill phase
      MmulHelper::clearCastCache();

      // Arrays persist — NO freeing, NO zeroing.
      // Reset segment state so CUDA graph capture starts fresh.
      executeCount_ = 0;
      for (auto& seg : segments_) {
        seg.exec.executionCount = 0;
        // Invalidate CUDA graph replay handles — buffer addresses from
        // non-frozen execution may differ from frozen steady-state.
        if (seg.exec.replayHandle) {
          for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
            if (!cb.directReference) delete cb.buffer;
          }
          seg.exec.replayHandle->getCaptureBuffers().clear();
          seg.exec.replayHandle.reset();
        }
        seg.exec.argTableStable = false;
        seg.exec.gapOpsCapturedInGraph = false;
        seg.exec.capturedInputAddrKey = 0;
        seg.exec.captureFailed = false;
      }
    }
    if (!frozen) {
      // Unfreeze: demote any FROZEN/FROZEN_CONSTANT slots back to SHAPE_CACHED
      for (int i = 0; i < numSlots_; i++) {
        if (slots_[i].state_ >= NativeSlot::SlotState::FROZEN) {
          slots_[i].state_ = NativeSlot::SlotState::SHAPE_CACHED;
        }
      }
      frozenConstantDetectionDone_ = false;
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
   * and error paths via DSP_DIAG macros (to stderr).
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
#endif

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

  // ── Shared immutable plan definition (Phase 3) ────────────────────────
  // Contains all data that does NOT change between executions:
  // numSlots, totalOutputSlots, numExternalInputs, requestedOutputSlotIndices,
  // externalInputNames, externalInputIsVariable, hasControlFlow, numLoopRegions,
  // backendPriority. Created during compilation, ref-counted for sharing.
  // Currently populated alongside existing fields (behavioral no-op).
  // Future phases will migrate reads to use planDef_ instead.
  PlanDefinition* planDef_ = nullptr;

  // ── Per-plan-instance mutable execution state (Phase 4) ───────────────
  // Contains slotArrays, ownership, protectedWeightBuffers, executeCount,
  // shapesFrozen. Thread-bound (error if called from different thread).
  // Currently populated alongside existing fields (behavioral no-op).
  // Future phases will migrate mutable state into execState_.
  ExecutionState* execState_ = nullptr;

  // Slot data
  NativeSlot* slots_;
  int numSlots_;
  int totalOutputSlots_;
  int numExternalInputs_;
  std::vector<std::string> externalInputNames_;  // name for each external input index
  std::vector<bool> externalInputIsVariable_;    // true if VARIABLE or PLACEHOLDER (needs forced H2D before replay)

  // Release schedule: releaseAtStep_[stepIdx] = array of slot indices to release
  int** releaseAtStep_;
  int* releaseAtStepCounts_;

  // Requested output mapping
  int* requestedOutputSlotIndices_;
  int numRequestedOutputs_;

  // Execution state (reused across calls)
  NDArray** outputSlots_;              // Current output slot values
  NDArray** slotArrayCache_;           // UNIFIED with outputSlots_ (same pointer, DO NOT delete[] separately)
  bool* slotIsViewProducer_;           // View producer flags (learned from first exec)
  // slotViewOutputs_ removed (Phase 2): views go directly into outputSlots_/slotArrayCache_
  Context** contextPool_;              // Pre-allocated Context pool
  bool viewProducerDetectionDone_;
  bool frozenConstantDetectionDone_;

  // Unified buffer ownership tracking (Phase 1A).
  // One SlotBufferInfo per totalOutputSlots_. Replaces ad-hoc tracking:
  //   protectedWeightBuffers_ → ownership == WEIGHT/VIEW_OF_WEIGHT
  //   slotViewOutputs_ → ownership == VIEW_OF_SLOT with parentSlotIdx
  //   per-execute dedup HashSet → O(1) ownership check
  SlotBufferInfo* slotOwnership_;

  // Graph segments for CUDA Graphs
  std::vector<GraphSegment> segments_;

  // pendingClose_ and deferredClose_ REMOVED: arrays persist (one array per slot).
  // View wrappers deleted inline in slotexec when replaced. No batched close needed.

  // Protected DataBuffers: model weights and shapeStatic outputs whose
  // DataBuffers must NEVER be freed during cleanup. Built on first execute().
  // Mirrors Java-side protectedWeightBuffers in DynamicShapePlanExecutor.
  std::unordered_set<DataBuffer*> protectedWeightBuffers_;

  // GPU graph capture control
  bool gpuGraphCaptureEnabled_;
  int totalGraphReplays_;

  // NVRTC JIT mode
  JitMode jitMode_;

  // Graph execution mode (controls which backend to use)
  GraphExecutionMode graphExecutionMode_;

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

  // ── Control flow support ─────────────────────────────────────────────
  bool hasControlFlow_;                    // True if any slot has controlFlowType != CF_NONE
  LoopRegion* loopRegions_;                // Array of loop region descriptors (owned)
  int numLoopRegions_;
  bool* slotIsDead_;                       // Per-output-slot dead flag (reset each execute)
  int slotIsDeadSize_;                     // = totalOutputSlots_
  static constexpr int MAX_LOOP_ITERATIONS = 10000000;  // Safety limit

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

  // ── Decode input direct-update ───────────────────────────────────────
  int decodeInputIdsExtIdx_ = -1;
  int decodePositionIdsExtIdx_ = -1;
  int decodeAttentionMaskExtIdx_ = -1;
  int decodeMaxKvLen_ = 0;
  long long pendingTokenId_ = 0;
  int pendingCachePos_ = 0;
  bool hasPendingDecodeUpdate_ = false;

  // Internal methods
  void scatterKvEntries(NDArray** externalInputs, int numExt, void* stream);
  // flushPendingClose REMOVED: arrays persist, view wrappers deleted inline
  void buildSegments();
  void rebuildSegmentsForFrozenShapes();

  // ── Slot execution (NativeDynamicShapePlan_slotexec.cpp) ──
  Status executeSlot(int slotIdx, NDArray** externalArrays, int numExt, void* stream);
  LongType computeShapeKey(NativeSlot& slot, NDArray** inputs, int numInputs);
  void detectFrozenConstants();

  // ── Segment management (NativeDynamicShapePlan_segments.cpp) ──
  LongType computeSegmentShapeKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  Status executeSegmentSlotBySlot(GraphSegment& seg, NDArray** externalArrays,
                                  int numExt, void* stream);
  Status executeSegmentWithCpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getCpuGraphBackend();

  // ── CUDA graph capture/replay (NativeDynamicShapePlan_cudagraph.cu) ──
#ifdef SD_CUDA
  LongType computeSegmentInputAddrKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  LongType computeCreateOpValueKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  void snapshotExternalAddrs(GraphSegment& seg, NDArray** externalInputs, int numExt);
  bool externalAddrsMatch(const GraphSegment& seg, NDArray** externalInputs, int numExt) const;
  Status executeSegmentWithGraph(GraphSegment& seg, NDArray** externalArrays,
                                 int numExt, void* stream);
  Status executeSegmentWithJit(GraphSegment& seg, NDArray** externalArrays,
                               int numExt, void* stream);

  /**
   * Replay verification: snapshot replay outputs, re-execute slot-by-slot with
   * frozen context reset, and compare to detect divergence.
   *
   * Reusable by both the Triton path (gpubackend.cpp) and the CUDA_GRAPHS
   * frozen fast path (cuda.cu). Requires cudaStreamSynchronize before calling.
   *
   * @param seg        The segment that was just replayed
   * @param externalArrays External input arrays
   * @param numExt     Number of external inputs
   * @param stream     CUDA stream (for sync/memcpy)
   * @param pathLabel  Label for log output (e.g. "TRITON" or "CUDA_GRAPHS")
   */
  void performReplayVerify(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream, const char* pathLabel);
#endif

  // ── Platform dispatch (NativeDynamicShapePlan_cuda.cu / _cuda_stubs.cpp) ──
  // These methods abstract platform-specific (CUDA vs CPU) behavior.
  // CUDA implementations are in _cuda.cu; CPU fallbacks in _cuda_stubs.cpp.
  // The linker picks the right version based on the build configuration.
  Status platformTryFrozenFastPath(NDArray** externalInputs, int numExternalInputs,
                                    NDArray** requestedOutputs, int numRequestedOutputs, void* stream);
  void platformPreExecuteSetup(NDArray** externalInputs, int numExternalInputs, void* stream);
  bool platformShouldKeepSegmentCache(const GraphSegment& seg) const;
  void platformPrecompileSegments(NDArray** externalInputs, int numExternalInputs);
  bool platformBindSegmentDevice(const GraphSegment& seg);
  bool platformShouldUseGraph(const GraphSegment& seg);
  Status platformExecuteSegmentWithBackends(GraphSegment& seg, NDArray** externalInputs,
                                             int numExternalInputs, void* stream, bool& usedGraph);
  Status platformCheckPostSegment(GraphSegment& seg);
  void platformScatterKvEntry(NDArray* presentKv, NDArray* staticBuf, int seqDim, int pos, void* stream);
  void* platformBeginKvScatter(void* stream);
  void platformEndKvScatter(void* savedState);
  void platformMarkKvCaptureBuffersNeverSkip();
  void platformCleanupSegmentForRebuild(GraphSegment& seg);
  void platformFreePlanResources();
  int platformCountCapturedGraphSegments() const;
  void platformMaybeSplitIfEnabled();

  // ── GPU graph backend (NativeDynamicShapePlan_gpubackend.cpp) ──
  Status executeSegmentWithGpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getGpuGraphBackend();
  void clearGpuBackendFailedCache();

  // CPU graph backend (oneDNN Graph or ACL Dynamic Fusion)
  GraphBackend* cpuGraphBackend_;
  bool cpuGraphBackendChecked_;

  // GPU graph backend (Triton GPU compiler)
  GraphBackend* gpuGraphBackend_;
  bool gpuGraphBackendChecked_;

  // Backend priority order (user-configurable)
  std::vector<std::string> backendPriority_;

#ifdef SD_CUDA
  // Pre-allocated cuBLAS workspace for GPU graph capture.
  void* cublasWorkspaceBuffer_ = nullptr;
  size_t cublasWorkspaceSize_ = 0;
  void ensureCublasWorkspace(size_t minBytes);
  void setCublasWorkspaceForCapture(void* stream);
  void setCublasWorkspaceForWarmup();
  void restoreCublasWorkspaceAfterCapture(void* stream);
#endif

  // Max-allocation mode for KV cache outputs
  // Maps output slot index -> max number of elements to pre-allocate
  std::unordered_map<int, LongType> outputSlotMaxSizes_;
  // Tracks which slots have been pre-allocated at max size
  std::unordered_set<int> maxAllocatedSlots_;
  // Maximum KV cache length (for pre-allocated attention outputs)
  int maxKvCacheLen_;

#ifdef SD_CUDA
  // ── Batch-zero optimization ────────────────────────────────────────────
  // Replaces ~1000 individual cudaMemsetAsync graph nodes with a single
  // kernel launch that zeros all output buffers in parallel.
  // Reduces CUDA graph node count by ~28%, saving ~1-2ms per graph replay.
  struct BatchZeroEntry { void* ptr; int bytes; };
  std::vector<BatchZeroEntry> batchZeroEntries_;
  void* batchZeroDevicePtrs_ = nullptr;   // Device array of void* pointers
  void* batchZeroDeviceSizes_ = nullptr;  // Device array of int sizes
  void* batchZeroHostPtrs_ = nullptr;     // Pinned host mirror of pointers
  void* batchZeroHostSizes_ = nullptr;    // Pinned host mirror of sizes
  int batchZeroDeviceCount_ = 0;

  void collectBatchZeroTargets(const std::unordered_set<int>& gapSlots);
  void prepareBatchZeroDevice(cudaStream_t stream);
  void launchBatchZero(cudaStream_t stream);
  void freeBatchZeroResources();
  static void setBatchZeroActive(bool active);

  // Registration-based batch-zero: during warmup, observe which buffers
  // actually get nullified and save that exact list for capture.
  // This replaces the pre-scan approach (collectBatchZeroTargets) which
  // collected ~143 extra buffers for slots that don't actually execute.
  void startBatchZeroRegistration();
  void finishBatchZeroRegistration();

  // Capture buffer pool sharing: routes capture workspace allocation
  // through CudaMemoryPool for cross-segment reuse.
  void* captureBufferRegistry_ = nullptr;  // opaque ptr to CaptureBufferRegistry

  // ── Batch D2D copy optimization ─────────────────────────────────────────
  // Replaces ~357 individual cudaMemcpyAsync D2D calls for capture buffer
  // updates with a single kernel launch (same pattern as batch-zero).
  // dst pointers and sizes are static (capture buffers are fixed-address);
  // src pointers are updated each step from external input specialBuffer().
  void* batchD2DDeviceSrcPtrs_ = nullptr;   // Device: void*[count]
  void* batchD2DDeviceDstPtrs_ = nullptr;   // Device: void*[count]
  void* batchD2DDeviceSizes_ = nullptr;     // Device: size_t[count]
  void* batchD2DHostSrcPtrs_ = nullptr;     // Pinned host: void*[count]
  void* batchD2DHostDstPtrs_ = nullptr;     // Pinned host: void*[count] (static)
  void* batchD2DHostSizes_ = nullptr;       // Pinned host: size_t[count] (static)
  int batchD2DCount_ = 0;                   // Number of valid entries
  int batchD2DAllocated_ = 0;               // Allocated capacity
  // Map from capture buffer index → batchD2D index (-1 = skipped)
  std::vector<int> captureBufferToBatchIdx_;

  void prepareBatchD2DDevice(int count, cudaStream_t stream);
  void launchBatchD2D(cudaStream_t stream);
  void freeBatchD2DResources();
#else
  void freeBatchZeroResources() {}
  void freeBatchD2DResources() {}
#endif

  // Available on all platforms (returns false on non-CUDA, no-op stubs)
  static bool isBatchZeroActive();
  static bool isBatchZeroRegistering();
  static void registerBatchZeroBuffer(void* ptr, size_t bytes);

#ifdef SD_CUDA
  // ── Batched GEMM optimization ──────────────────────────────────────────
  // Groups consecutive matmul slots with identical (M,N,K,transA,transB,dtype)
  // into single cublasGemmBatchedEx calls, reducing CUDA graph node count.
  // For SmolDocling (24 layers × 9 matmuls), reduces ~211 → ~120 matmul nodes.
  struct BatchedGemmGroup {
    std::vector<int> slotIndices;  // matmul slot indices in this group (non-consecutive OK)
    int triggerSlot;    // last slot in group — execution happens here
    int M, N, K;        // shared dimensions
    int transA, transB;
    sd::DataType dtype;
    void** d_A_ptrs;    // device pointer array
    void** d_B_ptrs;
    void** d_C_ptrs;
    void** h_A_ptrs;    // pinned host staging
    void** h_B_ptrs;
    void** h_C_ptrs;
    int maxBatchSize;   // allocated capacity
  };
  std::vector<BatchedGemmGroup> batchedGemmGroups_;
  // Maps slot index → index into batchedGemmGroups_ (-1 if not part of a group)
  std::vector<int> slotToBatchedGemmGroup_;

  const LongType* resolveInputShapeInfo(int srcIdx, NDArray** externalArrays, int numExt) const;
  void detectBatchedGemmGroups(NDArray** externalArrays, int numExt);
  void prepareBatchedGemmDevice(cudaStream_t stream);
  Status executeBatchedGemmGroup(int groupIdx, NDArray** externalArrays, int numExt, cudaStream_t stream);
  void freeBatchedGemmResources();
#else
  void freeBatchedGemmResources() {}
#endif
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
