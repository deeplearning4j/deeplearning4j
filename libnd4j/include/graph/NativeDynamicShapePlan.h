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
#include <graph/DspExecutionTrace.h>
#include <graph/DspPhaseUtils.h>
#include <graph/FusionPass.h>
#include <graph/generated/graph_generated.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/common.h>

#include <atomic>
#include <cstdint>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <graph/GraphReplayHandle.h>
#include <graph/SlotBufferOwnership.h>
#include <graph/PlanDefinition.h>
#include <graph/ExecutionState.h>
#include <graph/gpu/ViewRecipe.h>
#include <system/env_functions.h>

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
  GEM_HEXAGON = 14,     // Hexagon-MLIR NPU compilation + command list replay
  GEM_OPENVINO = 15,    // Force OpenVINO CPU graph backend (Intel x86, broad op coverage)
  GEM_TVM = 16,         // Deprecated: TVM removed, use triton-cpu instead
  GEM_EMULATED_REPLAY = 17,  // Emulated graph replay: slot-by-slot with replay lifecycle diagnostics
  GEM_SHAPE_INFERENCE_ONLY = 18  // Shape inference only: calculates output shapes without executing ops
};

/**
 * SelectedBackend — resolved once at build time, stored per-segment.
 * Each GraphExecutionMode maps to exactly one SelectedBackend.
 * GEM_AUTO resolves to the best available backend at build time.
 * After resolution, dispatch is a simple switch — no cascade.
 */
enum class SelectedBackend : uint8_t {
  SLOT_BY_SLOT = 0,    // Execute each op individually (no fusion, no graphs)
  CUDA_GRAPHS = 1,     // CUDA graph capture/replay
  GPU_COMPILER = 2,    // Triton/NVRTC/PTX/TPU/Hexagon compiler backend
  CPU_GRAPH = 3,       // CPU graph backend (oneDNN/ACL/MLIR/MLX/NNAPI/ARM)
  EMULATED_REPLAY = 4, // Emulated graph replay: slot-by-slot with replay lifecycle tracking
};

/**
 * PlanDestructionReason — records WHY a plan was destroyed or reset.
 * Set via setDestructionReason() before releasing resources.
 * Useful for post-mortem diagnostics and distinguishing expected vs unexpected teardown.
 */
enum class PlanDestructionReason : uint8_t {
  NORMAL_CLOSE = 0,       // Normal plan lifecycle end
  SESSION_RESET = 1,      // Session reset (e.g., new page, new prompt)
  OOM_RECOVERY = 2,       // Out-of-memory recovery — freeing resources to reclaim GPU memory
  DEVICE_SWITCH = 3,      // Switching to a different GPU device
  CAPTURE_FAILURE = 4,    // CUDA graph capture failed — plan must be rebuilt
  SHAPE_CHANGE = 5,       // Input shapes changed — plan invalidated
  ERROR_RECOVERY = 6,     // Error recovery (e.g., CUDA error 700) — plan must be rebuilt
};

/**
 * PlanPhase — plan-level lifecycle phase for the entire NativeDynamicShapePlan.
 *
 * Enforces a strict progression that makes assumptions easier at each level:
 *   SLOT_BY_SLOT → SHAPES_FROZEN → POINTERS_STABLE → REPLAYING
 *
 * Each phase guarantees everything from prior phases plus additional invariants:
 *   SLOT_BY_SLOT:      No assumptions. Shapes may change, pointers may move.
 *   SHAPES_FROZEN:     All output shapes are constant. Shape inference skipped.
 *   POINTERS_STABLE:   Shapes frozen + all buffer pointers stable across steps.
 *                      Graph capture is safe.
 *   REPLAYING:         Shapes frozen + pointers stable + graph replay active.
 *                      Only D2D copies + graph launch needed.
 *
 * Phase is automatically advanced by execute() based on observed stability.
 * Can be manually set backward (e.g., unfreeze → SLOT_BY_SLOT).
 */
enum class PlanPhase : uint8_t {
  SLOT_BY_SLOT = 0,      // No guarantees — shapes and pointers may change
  SHAPES_FROZEN = 1,     // Shapes are constant across executions
  POINTERS_STABLE = 2,   // Shapes frozen + buffer pointers stable
  REPLAYING = 3,         // Steady state — graph replay active
};

// ExecutionPhase REMOVED — unified into SegmentLifecycleState (in GraphSegmentExec).
// Mapping for callers that previously used ExecutionPhase:
//   WARMUP      → lifecycleState == NEEDS_WARMUP
//   COMPILING   → lifecycleState == NEEDS_COMPILE
//   COMPILED    → lifecycleState == CAPTURE_PENDING or CAPTURED
//   REPLAYING   → lifecycleState == REPLAYING
//   SLOT_BY_SLOT → lifecycleState == FAILED (or non-capturable segment)

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
 * Slot sub-structs — extracted from the monolithic NativeSlot to reduce
 * cognitive load and make the header navigable.
 *
 * Access pattern after refactor:
 *   slot.ident.opHash      (was slot.opHash)
 *   slot.wiring.numInputs  (was slot.numInputs)
 *   slot.args.iArgs        (was slot.iArgs)
 *   slot.flags.isDataDependent  (was slot.isDataDependent)
 *   slot.fusedChain.isFusedChainHead  (was slot.isFusedChainHead)
 *   slot.cf.controlFlowType  (was slot.controlFlowType)
 *   slot.legacy.legacyOpType (was slot.legacyOpType)
 *   slot.shapeCache.cachedShapeKey (was slot.cachedShapeKey)
 */

static constexpr int MAX_FUSED_CHAIN = 8;  // Max ops in a fused elementwise chain

/** Op identification. */
struct SlotIdent {
  LongType opHash = 0;
  sd::ops::DeclarableOp* op = nullptr;  // Resolved at compile time (not owned)
  std::string opName;                    // For diagnostics
};

/** Input/output wiring. */
struct SlotWiring {
  int numInputs = 0;
  int* inputSourceIndices = nullptr;     // >=0: prior slot, <0: external (-(idx+1))
  int8_t* inputSourceTypes = nullptr;    // NativeSourceType values
  int numOutputs = 0;
  int* outputSlotIndices = nullptr;      // Flat slot indices for each output

  ~SlotWiring() {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
  }
  SlotWiring() = default;
  SlotWiring(SlotWiring&& o) noexcept
      : numInputs(o.numInputs), inputSourceIndices(o.inputSourceIndices),
        inputSourceTypes(o.inputSourceTypes), numOutputs(o.numOutputs),
        outputSlotIndices(o.outputSlotIndices) {
    o.numInputs = 0; o.inputSourceIndices = nullptr; o.inputSourceTypes = nullptr;
    o.numOutputs = 0; o.outputSlotIndices = nullptr;
  }
  SlotWiring& operator=(SlotWiring&& o) noexcept {
    if (this != &o) {
      delete[] inputSourceIndices; delete[] inputSourceTypes; delete[] outputSlotIndices;
      numInputs = o.numInputs; inputSourceIndices = o.inputSourceIndices;
      inputSourceTypes = o.inputSourceTypes; numOutputs = o.numOutputs;
      outputSlotIndices = o.outputSlotIndices;
      o.numInputs = 0; o.inputSourceIndices = nullptr; o.inputSourceTypes = nullptr;
      o.numOutputs = 0; o.outputSlotIndices = nullptr;
    }
    return *this;
  }
  SlotWiring(const SlotWiring&) = delete;
  SlotWiring& operator=(const SlotWiring&) = delete;
};

/** Frozen op arguments. */
struct SlotArgs {
  LongType* iArgs = nullptr;  int numIArgs = 0;
  double* tArgs = nullptr;    int numTArgs = 0;
  bool* bArgs = nullptr;      int numBArgs = 0;
  DataType* dArgs = nullptr;  int numDArgs = 0;
  std::string* sArgs = nullptr; int numSArgs = 0;

  ~SlotArgs() {
    delete[] iArgs; delete[] tArgs; delete[] bArgs; delete[] dArgs; delete[] sArgs;
  }
  SlotArgs() = default;
  SlotArgs(SlotArgs&& o) noexcept
      : iArgs(o.iArgs), numIArgs(o.numIArgs), tArgs(o.tArgs), numTArgs(o.numTArgs),
        bArgs(o.bArgs), numBArgs(o.numBArgs), dArgs(o.dArgs), numDArgs(o.numDArgs),
        sArgs(o.sArgs), numSArgs(o.numSArgs) {
    o.iArgs = nullptr; o.numIArgs = 0; o.tArgs = nullptr; o.numTArgs = 0;
    o.bArgs = nullptr; o.numBArgs = 0; o.dArgs = nullptr; o.numDArgs = 0;
    o.sArgs = nullptr; o.numSArgs = 0;
  }
  SlotArgs& operator=(SlotArgs&& o) noexcept {
    if (this != &o) {
      delete[] iArgs; delete[] tArgs; delete[] bArgs; delete[] dArgs; delete[] sArgs;
      iArgs = o.iArgs; numIArgs = o.numIArgs; tArgs = o.tArgs; numTArgs = o.numTArgs;
      bArgs = o.bArgs; numBArgs = o.numBArgs; dArgs = o.dArgs; numDArgs = o.numDArgs;
      sArgs = o.sArgs; numSArgs = o.numSArgs;
      o.iArgs = nullptr; o.numIArgs = 0; o.tArgs = nullptr; o.numTArgs = 0;
      o.bArgs = nullptr; o.numBArgs = 0; o.dArgs = nullptr; o.numDArgs = 0;
      o.sArgs = nullptr; o.numSArgs = 0;
    }
    return *this;
  }
  SlotArgs(const SlotArgs&) = delete;
  SlotArgs& operator=(const SlotArgs&) = delete;
};

/** Execution flags and fusion metadata. */
struct SlotFlags {
  bool needsZeroedOutput = true;
  bool isDataDependent = false;
  bool outputShapeDependsOnInputValues = false;
  bool needsIntLongSync = false;
  bool isCustomOp = true;
  bool isIdentityOp = false;
  bool isViewCapableOp = false;
  bool isFullyWriting = false;
  bool inPlaceFused = false;
  int inPlaceFusedInputIdx = -1;
  // cublasLt epilogue fusion
  int ltEpilogueType = 0;            // 0=none, 1=bias, 2=bias+relu, 3=bias+gelu
  int ltEpilogueBiasSourceIdx = -1;
  // Structural iArg count
  int structuralIArgCount = -1;       // -1 = all iArgs are structural

  /**
   * True if this slot's output shape changes between decode steps (e.g. attention
   * mask growth, KV cache views with growing sequence dimension, position IDs).
   *
   * Set during phaseWarmup when a shape-reassign fires (step3-warmup-reassign or
   * fused-chain-warmup-reassign paths), or propagated transitively from any input
   * slot that is already dynamic.
   *
   * Effect on execution:
   *   - Slot is excluded from the frozen fast path (executeSlot early-return).
   *   - Slot's output DataBuffer is NOT frozen (addFrozenRef skipped in phaseWarmup).
   *   - Slot always runs via the normal (non-frozen) execution path every step.
   *
   * This flag is set once during warmup and never cleared (immutable after freeze).
   */
  bool isDynamicShape = false;
};

/** Fused elementwise chain metadata. */
struct FusedChain {
  bool isFusedChainHead = false;
  int fusedChainLength = 0;
  int fusedChainOpCodes[MAX_FUSED_CHAIN] = {};
  int fusedChainSlots[MAX_FUSED_CHAIN] = {};
  int fusedChainSecondaryInputSources[MAX_FUSED_CHAIN] = {};
  bool isFusedChainTail = false;
};

/** Control flow support. */
struct ControlFlowInfo {
  ControlFlowType controlFlowType = CF_NONE;
  int loopBackTarget = -1;
  int loopRegionIndex = -1;
};

/** Legacy op support for ops not registered in OpRegistrator. */
struct LegacyOpInfo {
  int legacyOpType = 0;   // 0=not legacy, 1=TransformSame, 2=TransformStrict, ...
  int legacyOpNum = -1;
};

/** Per-slot shape cache and static analysis. */
struct ShapeCache {
  LongType cachedShapeKey = 0;
  std::vector<const LongType*> cachedOutputShapes;   // Cached shape infos (not owned)
  bool shapeStatic = false;  // True if output shape never changes between executions
};

/**
 * Per-op descriptor with pre-compiled wiring for the native plan executor.
 * Mirrors DynamicShapeSlot.java but uses C++ types.
 *
 * Index conventions for inputSourceIndices:
 *   >= 0: index into the flat outputSlots array (from a prior op's output)
 *   <  0: index into external arrays: -(index + 1) into the external input array
 *
 * Sub-structs extracted to reduce cognitive load (was 50+ flat fields).
 */
struct NativeSlot {
  // ── Sub-structs (extracted for organization) ──────────────────────
  SlotIdent ident;
  SlotWiring wiring;
  SlotArgs args;
  SlotFlags flags;
  FusedChain fusedChain;
  ControlFlowInfo cf;
  LegacyOpInfo legacy;
  ShapeCache shapeCache;

  // ── Top-level fields (not grouped) ────────────────────────────────
  int targetDeviceId = -1;             // -1 = auto

  /**
   * Slot lifecycle state machine. Replaces 3 independent booleans
   * (shapeCacheValid, frozenContextReady, frozenConstantSlot) with
   * explicit ordered states and documented transitions.
   *
   * State transitions (ordered — each state includes all prior guarantees):
   *   WARMUP → SHAPE_CACHED:          Shape cache populated, view status determined
   *   SHAPE_CACHED → FROZEN:          Shapes frozen, context reuse enabled
   *   FROZEN → FROZEN_CONSTANT:       Output never changes, skip execution entirely
   *
   * Backward transitions:
   *   Any → WARMUP:                   Plan invalidation (shape change, etc.)
   *   FROZEN/FROZEN_CONSTANT → SHAPE_CACHED:  Unfreeze
   */
  // SlotState simplified: UNINITIALIZED and COMPILED removed (both were dead states).
  // UNINITIALIZED was only used as default init, never tested. COMPILED had zero references.
  // Progression: WARMUP → SHAPE_CACHED → FROZEN → FROZEN_CONSTANT
  enum class SlotState : uint8_t {
    WARMUP = 0,           // Initial + invalidation state (shape inference + view detection)
    SHAPE_CACHED,         // Shape cache populated, view status determined
    FROZEN,               // Shapes frozen, context reuse enabled
    FROZEN_CONSTANT,      // Output never changes, skip execution entirely
  };
  SlotState state_ = SlotState::WARMUP;

  /**
   * Per-slot monotonic write generation.
   *
   * Incremented every time this slot's output buffers are written (kernel
   * dispatch, prezero memset, nullify, fused chain emit, view install, etc.).
   * Readers may record the generation they saw at read time and later call
   * dspAssertSlotGeneration() to detect stale reads — e.g. "I read slot 42 at
   * generation 17, but after my consumer ran the slot is at generation 19
   * without my expecting a rewrite."
   *
   * Plain uint32_t (not atomic): plan execution is single-threaded per plan;
   * cross-plan replay happens under an external serialization point. If
   * concurrent access is ever added, promote to std::atomic<uint32_t>.
   */
  uint32_t generation_ = 0;

  NativeSlot() = default;

  // Convenience accessors that map SlotState to the old boolean semantics.
  bool shapeCacheValid() const { return state_ >= SlotState::SHAPE_CACHED; }
  bool frozenContextReady() const { return state_ >= SlotState::FROZEN; }
  bool frozenConstantSlot() const { return state_ == SlotState::FROZEN_CONSTANT; }

  // ── Generation counter accessors ─────────────────────────────────
  uint32_t generation() const { return generation_; }

  /**
   * Bump the write generation and return the new value.
   * Call AFTER a slot-write completes (kernel dispatch submitted, memset
   * enqueued, etc.) so that any reader that sees the new generation is
   * guaranteed the write was at least initiated.
   */
  uint32_t bumpGeneration() { return ++generation_; }

  ~NativeSlot() = default;

  // No copy, no move (sub-structs manage their own memory)
  NativeSlot(const NativeSlot&) = delete;
  NativeSlot& operator=(const NativeSlot&) = delete;
  NativeSlot(NativeSlot&&) = delete;
  NativeSlot& operator=(NativeSlot&&) = delete;
};

/**
 * Graph segment for graph capture / backend compilation.
 * A contiguous range of slots that can be captured as a single graph.
 *
 * Split into:
 *   - GraphSegmentDef: immutable definition (set at buildSegments, never changes)
 *   - GraphSegmentExec: mutable execution state (changes every execution)
 *
 * Lifecycle (via exec.executionCount):
 *   == 0: warm-up pass (slot-by-slot, populates slot cache)
 *   == 1: capture pass (ops recorded into graph, then launched)
 *   >= 2: replay pass (cached graph launched directly)
 */

/**
 * Immutable segment definition — set at buildSegments(), never changes.
 */
/**
 * ShapeKeyState — explicit lifecycle for the segment shape key.
 *
 * The shape key identifies a unique set of shapes flowing through a segment.
 * It determines whether a cached compilation/graph can be reused or must be
 * rebuilt. The lifecycle is:
 *
 *   UNSET (0)        → after buildSegments() or invalidateForRebuild()
 *   WARMUP_COMPUTED  → after first executeSegmentSlotBySlot() computes it
 *   COMPILED_WITH    → after backend->compileSegment() succeeds (this is the "reference" key)
 *   STABLE           → subsequent executions compute same key → no recompile
 *   DRIFTED          → subsequent execution computes DIFFERENT key → triggers recompile
 *
 * Invariant: if compiledShapeKey != 0 and currentShapeKey != compiledShapeKey,
 * the segment MUST recompile before executing compiled code.
 */
struct ShapeKeyState {
  /// The shape key that was used when the segment was last successfully compiled.
  /// Zero means "never compiled" — the segment needs its first compile.
  LongType compiledShapeKey = 0;

  /// The shape key computed on the most recent execution (may differ from compiled).
  /// Used for comparison: if current != compiled, shapes drifted.
  LongType lastComputedKey = 0;

  /// Frozen cache: when shapes are frozen, this caches the key to avoid recomputation.
  /// Zero means "not cached" — must compute fresh.
  LongType frozenCacheKey = 0;

  /// Returns true if shapes have been compiled and current key matches.
  bool isStable() const {
    return compiledShapeKey != 0 && lastComputedKey == compiledShapeKey;
  }

  /// Returns true if shapes drifted since last compile (needs recompile).
  bool hasDrifted() const {
    return compiledShapeKey != 0 && lastComputedKey != 0 && lastComputedKey != compiledShapeKey;
  }

  /// Returns true if the segment has never been compiled.
  bool neverCompiled() const { return compiledShapeKey == 0; }

  /// Record a fresh compilation with this key.
  void markCompiled(LongType key) {
    compiledShapeKey = key;
    lastComputedKey = key;
  }

  /// Record a freshly computed key (before deciding compile vs replay).
  void recordComputed(LongType key) { lastComputedKey = key; }

  /// Full reset (invalidation).
  void reset() {
    compiledShapeKey = 0;
    lastComputedKey = 0;
    frozenCacheKey = 0;
  }

  /// Diagnostic string for logging.
  const char* statusName() const {
    if (compiledShapeKey == 0) return "UNSET";
    if (lastComputedKey == 0) return "COMPILED_NO_CURRENT";
    if (lastComputedKey == compiledShapeKey) return "STABLE";
    return "DRIFTED";
  }
};

struct GraphSegmentDef {
  int startSlot = 0;
  int endSlot = 0;
  bool isCapturable = false;
  bool hasValueDepOps = false;         // True if any slot has outputShapeDependsOnInputValues
  /// Structured shape key lifecycle. All shape key reads/writes go through this.
  ShapeKeyState shapeKeyState;

  // User-forced backend override (empty = automatic selection via priority chain)
  std::string backendOverride;

  // Resolved backend — set once at buildSegments() time, never changes.
  // For non-capturable segments, always SLOT_BY_SLOT.
  // For capturable: resolved from graphExecutionMode_ at build time.
  SelectedBackend selectedBackend = SelectedBackend::SLOT_BY_SLOT;

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
};

/**
 * SegmentDispatchEvent — significant events during segment execution.
 * Used for structured logging: when any of these fires, it's always logged
 * regardless of diagnostic level, so post-mortem analysis can trace exactly
 * what happened and why.
 */
enum class SegmentDispatchEvent : uint8_t {
  WARMUP_START,           // Entering warmup (first slot-by-slot execution)
  WARMUP_DONE,           // Warmup completed OK, transitioning to NEEDS_COMPILE
  COMPILE_START,         // Starting compilation
  COMPILE_DONE,          // Compilation succeeded
  COMPILE_FAILED,        // Compilation failed
  SHAPE_KEY_COMPUTED,    // Shape key freshly computed
  SHAPE_KEY_MATCHED,     // Computed key matches compiled key (no recompile needed)
  SHAPE_KEY_DRIFTED,     // Computed key differs from compiled key (recompile triggered)
  SHAPE_KEY_STORED,      // Shape key stored after successful execution
  RECOMPILE_TRIGGERED,   // Mid-execution recompile due to shape drift
  CAPTURE_START,         // CUDA graph capture starting
  CAPTURE_DONE,         // Capture succeeded
  REPLAY_START,         // Graph replay starting
  REPLAY_DONE,          // Replay succeeded
  DIRECT_EXEC_START,    // Direct (non-captured) execution starting
  DIRECT_EXEC_DONE,     // Direct execution succeeded
  INVALIDATE,           // Segment invalidated (full reset)
};

// One-line event logger: always prints for significant events regardless of DSP_DIAG level.
// Format: [DSP_EVENT] seg[start-end] EVENT execCount=N shapeKey=compiled/current detail
#define DSP_SEG_EVENT(seg, event, ...) do { \
  sd_printf("[DSP_EVENT] seg[%d-%d] %s execCount=%d shapeKey=%s compiled=%lld current=%lld " , \
            (seg).def.startSlot, (seg).def.endSlot, \
            #event, (seg).exec.executionCount, \
            (seg).def.shapeKeyState.statusName(), \
            (long long)(seg).def.shapeKeyState.compiledShapeKey, \
            (long long)(seg).def.shapeKeyState.lastComputedKey); \
  sd_printf(__VA_ARGS__); \
  sd_printf("\n"); \
} while (0)

/**
 * Composite replay unit — one element of a mixed segment's replay schedule.
 *
 * When a segment contains both Triton-capturable ops and unsupported gap ops
 * (interleaved as "island -> gap -> island"), it cannot be captured as one
 * monolithic graph. Instead, we build an ordered replay schedule:
 *
 *   unit 0: Triton island [200-297]  → replayHandle[0]->replay()
 *   unit 1: gap unit [298-312]       → execute slots directly
 *   unit 2: Triton island [313-346]  → replayHandle[1]->replay()
 *   unit 3: gap unit [347-369]       → execute slots directly
 *   unit 4: Triton island [370-399]  → replayHandle[2]->replay()
 *
 * Gap units are executed via direct slot dispatch. Triton island units are
 * executed via their individual CudaGraphReplayHandle.
 */
enum ReplayUnitKind {
  REPLAY_UNIT_TRITON_ISLAND,   // Captured Triton graph — replay via handle
  REPLAY_UNIT_GAP,              // Gap range — execute slots directly
};
struct ReplayScheduleUnit {
  ReplayUnitKind kind;
  int startSlot;
  int endSlot;
  int islandIndex;  // For TRITON_ISLAND: index into compositeReplayHandles

  // ── Island merging through capture-safe gaps ──
  // A gap is "capture-safe" if ALL its slots launch CUDA kernels (no view ops,
  // no identity ops, no shape-only ops, no frozen constants). Such gaps can be
  // captured into the preceding island's CUDA graph instead of running natively.
  bool isCaptureSafe = false;

  // Merged capture group tracking. Units with the same non-negative mergedGroupId
  // share one merged CudaGraphReplayHandle. The isMergedLeader unit triggers the
  // graph launch; other units in the group are skipped during replay.
  int mergedGroupId = -1;    // -1 = not merged. >=0 = index into mergedReplayHandles
  bool isMergedLeader = false;

  ReplayScheduleUnit() : kind(REPLAY_UNIT_GAP), startSlot(0), endSlot(0), islandIndex(-1) {}
  ReplayScheduleUnit(ReplayUnitKind k, int s, int e, int idx)
      : kind(k), startSlot(s), endSlot(e), islandIndex(idx) {}
};
struct ReplaySchedule {
  std::vector<ReplayScheduleUnit> units;
  // Individual capture handles for each Triton island in the schedule.
  std::vector<std::unique_ptr<GraphReplayHandle>> compositeReplayHandles;
  // Merged capture handles — one per merged group (island + capture-safe gap sequences).
  // Indexed by ReplayScheduleUnit::mergedGroupId.
  std::vector<std::unique_ptr<GraphReplayHandle>> mergedReplayHandles;

  // Cast-cache high-water marks recorded right after all merged-group captures
  // complete.  The merged CUDA graphs bake in device pointers for cast-cache
  // slots [0, mergedCastHwmA) (A-side) and [0, mergedCastHwmB) (B-side).
  // At replay time, unmerged gap matmuls must start their cast-cache indices
  // from these values so they don't alias the merged graphs' baked pointers.
  // Zero when no merged captures exist (falls back to full reset to 0).
  size_t mergedCastHwmA = 0;
  size_t mergedCastHwmB = 0;

  // Pre-computed per-merged-group slot ranges [minSlot, maxSlot].
  // Indexed by mergedGroupId. Populated once during capture so the replay
  // hot loop can dirty-mark + tickWriteDevice in a single O(range) pass
  // instead of scanning all units per leader.
  struct MergedGroupRange { int minSlot; int maxSlot; };
  std::vector<MergedGroupRange> mergedGroupSlotRanges;
};

// Batch-zero entry used by NativeDynamicShapePlan::batchZeroEntries_.
// outputSlotIndex tracks which slot the pointer came from.
struct BatchZeroEntry { void* ptr; int bytes; int outputSlotIndex; };

/**
 * Mutable execution state — changes per-execution.
 */
struct GraphSegmentExec {
  // Explicit lifecycle state — replaces the implicit state machine derived from
  // executionCount thresholds, nullable handles, and boolean flags.
  // All transitions go through SegmentLifecycle functions in _gpubackend.cpp.
  enum class SegmentLifecycleState : uint8_t {
    NEEDS_WARMUP    = 0,  // First slot-by-slot run to populate shape caches
    NEEDS_COMPILE   = 1,  // Backend compile pass needed (Triton, NVRTC)
    CAPTURE_PENDING = 2,  // Compiled, waiting for CUDA graph capture
    CAPTURED        = 3,  // Graph handles valid and ready
    REPLAYING       = 4,  // Steady-state graph replay every step
    FAILED          = 5,  // Permanent failure — never attempt again
    OOM_DEFERRED    = 6,  // OOM during capture — deferred retry pending
  };
  SegmentLifecycleState lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;

  int executionCount = 0;

  // If true, never attempt graph capture/compilation for this segment.
  // Set for permanent failures (capture invalidation, host-only ops, address instability).
  // NOT set for OOM failures — those use the retry mechanism below.
  bool compilationFailed = false;

  // OOM retry mechanism
  int captureOomRetries = 0;
  int captureRetryAfterExec = 0;

  // LRU tracking: last executeCount_ at which this segment was replayed.
  // Used by proactive eviction to target least-recently-used graphs.
  int lastReplayExecCount = 0;

  // ── Platform-agnostic graph replay handle ────────────────────────
  // CUDA: CudaGraphReplayHandle (wraps cudaGraph_t/cudaGraphExec_t)
  // CPU: FunctionalReplayHandle (cached op dispatch, skip shape inference)
  // nullptr until first capture attempt.
  std::unique_ptr<GraphReplayHandle> replayHandle;
  LongType cachedShapeKey = 0;

  // Legacy address key (kept for replay diagnostics)
  LongType capturedInputAddrKey = 0;

  // Hash of input DATA values for 'create' (ConstantOfShape) ops.
  LongType capturedCreateValueKey = 0;

  // Hash of slot output specialBuffer() addresses at capture time.
  // Verified before replay — mismatch means output buffers were reallocated
  // and the CUDA graph has stale baked-in addresses (would SIGSEGV or corrupt).
  LongType capturedSlotAddrHash = 0;

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

  // Consecutive-stable pass counters for the defensive addr-key and slot-hash
  // checks in compositeReplay. Tracked for diagnostic telemetry only — the
  // checks themselves only run when DSP_DIAG VERIFY is enabled.
  int addrKeyStableCount = 0;   // counts consecutive "ext-input key matched" passes
  int slotAddrStableCount = 0;  // counts consecutive "slot addr hash matched" passes

  // Native ordered range ops captured in graph — must not be re-executed after replay.
  bool gapOpsCapturedInGraph = false;

  // View recipe chain — captures view-producing ops (reshape, permute,
  // expand_dims, squeeze, strided_slice) so they can be installed during
  // REPLAYING without launching a kernel or executing a native ordered range.
  // Populated during SHAPES_FROZEN, validated during POINTERS_STABLE,
  // installed before consumer replay during REPLAYING.
  ViewRecipeChain viewRecipes;

  // Composite replay schedule for mixed Triton/gap segments.
  // When a segment has interleaved gap slots, it's captured as multiple
  // Triton island graphs + gap slot ranges, replayed in program order.
  // Only populated for segments with hasUnsupportedTritonReplayGaps.
  ReplaySchedule compositeReplaySchedule;

  // Phase 2: Replay schedule signature hash (FNV-1a) from the last execution.
  // Computed from ordered replay units (kinds, slot ranges, op categories).
  // Zero if the segment has no replay or consolidation has not yet run.
  unsigned long long replaySignatureHash = 0;

  // Phase 2: Number of replay units after consolidation for this segment.
  // Updated each execution when the consolidation pass runs.
  // Zero if the segment is not capturable or consolidation hasn't run.
  int replayUnitCount = 0;

  // ExecutionPhase REMOVED — use lifecycleState for all phase queries.
  // Convenience: maps lifecycle state to display name for diagnostics.
  const char* displayPhaseName() const {
    switch (lifecycleState) {
      case SegmentLifecycleState::NEEDS_WARMUP:    return "WARMUP";
      case SegmentLifecycleState::NEEDS_COMPILE:   return "COMPILING";
      case SegmentLifecycleState::CAPTURE_PENDING: return "COMPILED";
      case SegmentLifecycleState::CAPTURED:        return "COMPILED";
      case SegmentLifecycleState::REPLAYING:       return "REPLAYING";
      case SegmentLifecycleState::FAILED:          return "SLOT_BY_SLOT";
      case SegmentLifecycleState::OOM_DEFERRED:    return "OOM_DEFERRED";
      default:                                     return "UNKNOWN";
    }
  }

  // JNI-compatible integer encoding matching the old ExecutionPhase values:
  //   0=WARMUP, 1=COMPILING, 2=COMPILED, 3=REPLAYING, 4=SLOT_BY_SLOT
  int getExecutionPhaseCode() const {
    switch (lifecycleState) {
      case SegmentLifecycleState::NEEDS_WARMUP:    return 0;
      case SegmentLifecycleState::NEEDS_COMPILE:   return 1;
      case SegmentLifecycleState::CAPTURE_PENDING: return 2;
      case SegmentLifecycleState::CAPTURED:        return 2;
      case SegmentLifecycleState::REPLAYING:       return 3;
      case SegmentLifecycleState::FAILED:          return 4;
      case SegmentLifecycleState::OOM_DEFERRED:    return 0;
      default:                                     return -1;
    }
  }

  void reset() {
    lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;
    executionCount = 0;
    compilationFailed = false;
    captureOomRetries = 0;
    captureRetryAfterExec = 0;
    lastReplayExecCount = 0;
    replayHandle.reset();
    cachedShapeKey = 0;
    capturedInputAddrKey = 0;
    capturedCreateValueKey = 0;
    capturedSlotAddrHash = 0;
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
    addrKeyStableCount = 0;
    slotAddrStableCount = 0;
    gapOpsCapturedInGraph = false;
    viewRecipes = ViewRecipeChain();
    compositeReplaySchedule = ReplaySchedule();
    replaySignatureHash = 0;
    replayUnitCount = 0;
  }
};

/**
 * Graph segment — combines immutable definition with mutable execution state.
 * Access pattern: seg.def.startSlot, seg.exec.executionCount
 */
struct GraphSegment {
  GraphSegmentDef def;
  GraphSegmentExec exec;

  // Per-segment resolved CPU backend — set on first successful compile via cascade.
  // Subsequent executions reuse this without re-cascading.
  GraphBackend* resolvedCpuBackend = nullptr;

  // Cached backend type to avoid dynamic_cast per token in frozen path.
  // Set once when resolvedCpuBackend is assigned.
  enum class CpuBackendType { UNKNOWN, ONEDNN, OPENVINO } resolvedCpuBackendType = CpuBackendType::UNKNOWN;

  // Pointer to NativeDynamicShapePlan slot array cache — allows GPU backends
  // to update the slot cache when pre-allocating output arrays.
  NDArray** slotArrayCache = nullptr;

  // Runtime-configurable OOM retry constants (read from Environment)
  static int maxOomRetries();
  static int retryInterval();

  // Reset resolvedCpuBackend when exec state is reset (e.g., shape change rebuild)
  void resetCpuBackend() { resolvedCpuBackend = nullptr; }

  GraphSegment() = default;
};

/**
 * Per-phase execution timing breakdown for diagnostics.
 * Populated by the phase helpers and consumed by execute().
 */
struct PhaseExecutionStats {
  long long graphReplayUs = 0;
  long long slotBySlotUs = 0;
  int graphReplaySegs = 0;
  int slotBySlotSegs = 0;
  int graphReplaySlots = 0;
  int slotBySlotSlots = 0;
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
  static NativeDynamicShapePlan* fromSerializedPlan(const void* data, LongType size,
                                                     GraphExecutionMode mode = GraphExecutionMode::GEM_AUTO);

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
   * Steady-state fast path for autoregressive decode loops.
   *
   * PRECONDITIONS (caller must ensure):
   *   - planPhase_ >= REPLAYING (plan is in steady-state graph replay)
   *   - shapesFrozen_ == true
   *   - executeCount_ >= 3 (past warmup + capture + first replay)
   *   - tritonVerifyKernels is false (no golden comparison needed)
   *
   * Skips: lifecycle validation, buffer scanning, fingerprinting, view wrapper
   * refresh, frozen snapshot detection, closed-buffer detection, external input
   * validation, ownership reclassification, phase advancement, diagnostics.
   *
   * Keeps: platformBeginExecution (stream setup), phaseReplay (segment dispatch
   * + compositeReplay), output collection, executeCount_ increment,
   * platformEndExecution (completion event).
   *
   * Returns Status::BAD_ARGUMENTS if preconditions are not met (falls back to
   * full execute()). This is NOT an error — it means the plan has not yet
   * reached steady state.
   */
  Status executeSteadyState(
      NDArray** externalInputs, int numExternalInputs,
      NDArray** requestedOutputs, int numRequestedOutputs,
      void* stream);

  // ─── Phase execution containers ─────────────────────────────────────────
  // The monolithic execute() is decomposed into these clearly-scoped methods.
  // Each encapsulates all work for its phase — no scattered logic.

  /**
   * PHASE: Warmup — first frozen execution (executeCount_ == 0 after freeze).
   *
   * Populates shapes in outputSlots_ by running all segments slot-by-slot.
   * Captures shapes needed for subsequent compilation. Invalidates shape
   * caches for non-capturable segments (their shapes may change across steps).
   *
   * Called automatically by execute() when shapesFrozen_ && executeCount_ == 0.
   * @return Status::OK on success, error on segment execution failure.
   */
  Status phaseWarmup(NDArray** externalInputs, int numExternalInputs, void* stream,
                     PhaseExecutionStats* stats = nullptr);

  /**
   * PHASE: Compile — precompile all GPU-compilable segments.
   *
   * Fires async compilation threads for Triton/NVRTC modules. On CPU, no-op.
   * Compilation results populate seg.exec.compiledByBackend and seg.exec.replayHandle.
   *
   * Called automatically by execute() after warmup (executeCount_ == 1).
   * Can also be called eagerly after setShapesFrozen() for ahead-of-time compilation.
   */
  void phaseCompile(NDArray** externalInputs, int numExternalInputs);

  /**
   * Ahead-of-time precompile entry point for benchmarks and tests.
   *
   * Performs the full compile lifecycle up front so subsequent execute() calls
   * measure only replay time, never compilation. Runs: phaseFreeze (if not yet
   * frozen), one warmup execution to populate shape caches, then phaseCompile.
   * After return, compilationSealed() is true and any subsequent compileSegment
   * call logs a COMPILE_VIOLATION and increments midExecutionCompileCount().
   *
   * Use this from benchmarks to separate compile time from execution time —
   * without it, cold-start compile time leaks into the first measured execute().
   */
  Status precompilePlan(NDArray** externalInputs, int numExternalInputs, void* stream);

  /**
   * PHASE: Freeze — transition from dynamic to frozen shapes.
   *
   * Called by setShapesFrozen(true). Runs the fusion pass, rebuilds segments
   * if merge is enabled, resets execution counters, and advances planPhase_
   * to SHAPES_FROZEN.
   *
   * This is NOT called from execute() — it's the freeze transition itself.
   */
  Status phaseFreeze();

  /**
   * PHASE: Replay — steady-state graph replay execution.
   *
   * Dispatches all segments via their replay handles (CUDA graph replay,
   * Triton compiled kernels, or emulated replay). Post-segment checks include
   * NaN detection, trace slot reporting, and pool trimming.
   *
   * Called automatically by execute() when planPhase_ == REPLAYING.
   * Also called for POINTERS_STABLE when segments are compiled but not yet
   * fully replaying (transitional state).
   *
   * @return Status::OK on success, error on segment execution failure.
   */
  Status phaseReplay(NDArray** externalInputs, int numExternalInputs,
                     NDArray** requestedOutputs, int numRequestedOutputs,
                     void* stream, PhaseExecutionStats* stats = nullptr);

  /**
   * PHASE: Slot-by-slot — non-capturable segment execution.
   *
   * Executes segments that cannot be captured (control flow, CPU fallback,
   * compilation failures). Each op runs individually with full shape inference.
   *
   * Called automatically by execute() when the plan is forced into full
   * slot-by-slot mode.
   * @return Status::OK on success.
   */
  Status phaseSlotBySlot(NDArray** externalInputs, int numExternalInputs, void* stream,
                         PhaseExecutionStats* stats = nullptr);

  /**
   * PHASE: Shape inference only — propagates shapes without executing ops.
   *
   * Iterates all slots in order, gathering inputs, calling calculateOutputShape()
   * to determine output shapes, and allocating output arrays with the correct
   * shapes. No op kernels are executed, no host/device sync is performed, and
   * no phase advancement or frozen detection occurs.
   *
   * Use this for pre-computing output shapes, memory planning, or validating
   * shape compatibility across the graph without paying compute cost.
   *
   * @return Status::OK on success, KERNEL_FAILURE if shape inference fails.
   */
  Status phaseShapeInferenceOnly(NDArray** externalInputs, int numExternalInputs, void* stream);

  /**
   * Advance plan phase based on observed stability.
   * Automatic — called at end of execute(). Transitions:
   *   SHAPES_FROZEN → POINTERS_STABLE (after 2+ stable frozen executions)
   *   POINTERS_STABLE → REPLAYING (when all segments reach replay steady state)
   */
  void advancePlanPhase();

  /**
   * Demote plan phase (manual override for error recovery).
   * Used when segment drops out of replay steady state.
   */
  void demotePlanPhase(PlanPhase targetPhase, const char* reason);

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
   * Release all GPU memory held by intermediate computation results while keeping
   * the plan structure alive. This frees:
   *  1. Non-weight NDArrays from outputSlots_ (SLOT_OWNED buffers only)
   *  2. Per-segment CUDA graph replay handles (workspaces, host pointers)
   *  3. cuBLAS workspace (256 MB)
   *  4. Batch-zero, batch-D2D, and batched-GEMM device arrays
   *  5. MmulHelper cast cache (thread-local FP16→FP32 staging)
   *
   * After this call the plan is in a "cold" state — the next execute() will
   * re-warm (re-detect view producers, re-capture CUDA graphs, re-allocate
   * cuBLAS workspace, etc.) just like the very first execution.
   *
   * Use this between execution runs to reclaim GPU memory without
   * destroying the plan handle (avoiding costly re-compilation).
   *
   * @return the number of intermediate NDArrays freed
   */
  int releaseGpuIntermediates();

  /**
   * Get the number of external inputs expected by this plan.
   */
  int getNumExternalInputs() const { return numExternalInputs_; }

  /**
   * Get the last external inputs array passed to execute().
   * Only valid after at least one execute() call. Returns nullptr before first call.
   * The returned pointer is owned by the caller of execute() — the plan does NOT own it.
   */
  NDArray** getLastExternalInputs() const { return lastExternalInputs_; }
  int getLastNumExternalInputs() const { return lastNumExternalInputs_; }

  /**
   * Get the number of requested outputs.
   */
  int getNumRequestedOutputs() const { return numRequestedOutputs_; }

  /**
   * Get the total number of slots (ops) in the plan.
   */
  int getNumSlots() const { return numSlots_; }

  /**
   * Identity fingerprint: FNV-1a hash of (numSlots, all opNames, all output wiring).
   * Computed at deserialization time. Plans deserialized from identical bytes
   * produce identical fingerprints. Different fingerprints = different plan structure.
   */
  uint64_t identityFingerprint() const { return identityFingerprint_; }

  /**
   * Get the total number of output slots (intermediate + final).
   */
  int getTotalOutputSlots() const { return totalOutputSlots_; }

  /**
   * Estimate memory used by this plan's owned intermediate arrays.
   * Sums NDArray::memoryFootprint() for every plan-owned array in outputSlots_.
   * Returns 0 if slots have not been populated yet.
   */
  size_t estimatedOwnedBytes() const {
    if (outputSlots_ == nullptr || totalOutputSlots_ <= 0) return 0;
    size_t total = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      NDArray* arr = outputSlots_[i];
      if (arr == nullptr) continue;
      if (planOwnedArrays_.count(arr) > 0) {
        total += static_cast<size_t>(arr->memoryFootprint());
      }
    }
    return total;
  }

  /**
   * Get the output slots array (NDArray pointers for all slots).
   * Used by validation/diagnostic functions to inspect outputs after execution.
   */
  NDArray** getOutputSlots() const { return outputSlots_; }

  /**
   * Get plan segments (for CUDA Graphs integration).
   */
  const std::vector<GraphSegment>& getSegments() const { return segments_; }

  /**
   * Get mutable plan segments (for clearing CUDA graph timelines, etc.).
   */
  std::vector<GraphSegment>& getSegmentsMutable() {
    if (planPhase_ > PlanPhase::SLOT_BY_SLOT) {
      sd_printf("DSP PHASE VIOLATION: getSegmentsMutable called in phase %d\n", (int)planPhase_);
      assert(false && "DSP phase violation: getSegmentsMutable");
    }
    return segments_;
  }

  /**
   * Reset all segment execution state back to WARMUP.
   * Demotes plan phase to SLOT_BY_SLOT and clears compilation caches,
   * replay handles, and execution counters.
   * Use for session reset or model reload without full plan rebuild.
   */
  void resetSegmentExecutionState();

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
  void setCudaGraphsEnabled(bool enabled) {
    if (gpuGraphCaptureEnabled_ == enabled) return;  // idempotent: no-op if unchanged
    if (planPhase_ > PlanPhase::SLOT_BY_SLOT) {
      sd_printf("DSP PHASE VIOLATION: setCudaGraphsEnabled called in phase %d\n", (int)planPhase_);
      assert(false && "DSP phase violation: setCudaGraphsEnabled");
      return;
    }
    gpuGraphCaptureEnabled_ = enabled;
  }
  bool isCudaGraphsEnabled() const { return gpuGraphCaptureEnabled_; }

  /**
   * Set JIT compilation mode for segment execution.
   * - GRAPH_ONLY: CUDA graph capture/replay only (default)
   * - JIT_ONLY: NVRTC JIT only for element-wise segments
   * - GRAPH_PLUS_JIT: Try JIT first, fall back to graph capture
   */
  void setJitMode(JitMode mode) {
    if (jitMode_ == mode) return;  // idempotent: no-op if unchanged
    if (planPhase_ > PlanPhase::SLOT_BY_SLOT) {
      sd_printf("DSP PHASE VIOLATION: setJitMode called in phase %d\n", (int)planPhase_);
      assert(false && "DSP phase violation: setJitMode");
      return;
    }
    jitMode_ = mode;
  }
  JitMode getJitMode() const { return jitMode_; }

  void setGraphExecutionMode(GraphExecutionMode mode);
  GraphExecutionMode getGraphExecutionMode() const { return graphExecutionMode_; }


  /**
   * Enable/disable "shapes frozen" mode. When enabled:
   * - clearShapeCaches() becomes a no-op (shapes are known constant)
   * - Shape key computation is skipped for slots with valid cached shapes
   * - nullify() is skipped for slots with needsZeroedOutput=false
   *
   * Use when all external input shapes are guaranteed constant across steps.
   * The first execution after enabling will still do full shape inference
   * to populate the cache; subsequent executions skip shape work entirely.
   */
  void setShapesFrozen(bool frozen);
  bool isShapesFrozen() const { return shapesFrozen_; }

  /**
   * Enable/disable shape-only dry-run mode.
   *
   * When enabled, executeSlot() runs the full DSP dispatch machinery —
   * slot iteration, shape caching, frozen-constant checks, identity/fusion
   * detection, segment dispatch, output allocation — but SKIPS the actual
   * op->execute() call.  The outputs retain whatever values they held from
   * the previous real execution (or are uninitialized on the first pass).
   *
   * Purpose: measure pure dispatch/infrastructure overhead independently
   * from compute.  With 1683 ops per CPU decode step and 185 ms of kernel
   * time versus 367 ms of dispatch overhead, this mode lets dispatch
   * optimizations be profiled and iterated ~100x faster.
   *
   * Safe to toggle between executions; does NOT affect shape inference,
   * context-pool state, or frozen-constant detection.
   */
  SD_INLINE void setShapeOnlyMode(bool enabled) { shapeOnlyMode_ = enabled; }
  SD_INLINE bool isShapeOnlyMode() const { return shapeOnlyMode_; }

  /**
   * Get the current plan-level phase.
   * Phase progresses: SLOT_BY_SLOT → SHAPES_FROZEN → POINTERS_STABLE → REPLAYING
   */
  PlanPhase getPlanPhase() const { return planPhase_; }

  /**
   * Get the plan-level phase as an integer (for JNI).
   */
  int getPlanPhaseCode() const { return static_cast<int>(planPhase_); }

  /**
   * Set the reason for plan destruction/reset (for diagnostics).
   * Should be called before releaseGpuIntermediates() or destroy().
   */
  void setDestructionReason(PlanDestructionReason reason) { destructionReason_ = reason; }
  PlanDestructionReason getDestructionReason() const { return destructionReason_; }

  /**
   * Check if all buffer pointers are stable (same addresses as previous execution).
   * Pointer stability is a prerequisite for graph capture/replay.
   * Returns true only after at least 2 executions with frozen shapes where
   * all segment arg tables have stable pointers.
   */
  bool arePointersStable() const { return pointersStable_; }

  // ── Compilation seal & mid-execution violation tracking ──────────────
  // A "compilation seal" is placed after phaseCompile() succeeds. Any subsequent
  // call into compileSegment()/platformPrecompileSegments() is a contract
  // violation — it means compile time is leaking into measured execution time
  // and skewing benchmark numbers. Violations increment midExecutionCompileCount_
  // and emit a loud sd_printf log with a [COMPILE_VIOLATION] tag.
  //
  // The shape-change recompile path in executeSegmentWithGpuGraph is the one
  // legitimate case where we must unseal, recompile, and reseal. It counts as
  // a violation too — benchmarks should not observe any — but the plan
  // transparently recovers instead of hard-faulting.

  /** True once phaseCompile() has finalized. Reset by phaseFreeze/reset paths. */
  bool isCompilationSealed() const { return compilationDone_; }

  /**
   * Number of times compileSegment() ran after the compilation seal was placed.
   * Expected to be 0 for well-behaved benchmarks that call precompilePlan()
   * before their measured execution loop. A non-zero value means compile time
   * was measured as execution time.
   */
  int64_t getMidExecutionCompileCount() const {
    return midExecutionCompileCount_.load(std::memory_order_relaxed);
  }

  /**
   * Reset the mid-execution compile counter. Used by tests and by benchmark
   * harnesses that want to assert zero violations across a measured window.
   */
  void resetMidExecutionCompileCount() {
    midExecutionCompileCount_.store(0, std::memory_order_relaxed);
  }

  /**
   * Record a mid-execution compilation event. Called by the shape-change
   * recompile path before it unseal/recompile/reseals. Emits a loud
   * [COMPILE_VIOLATION] log visible at any DSP diagnostic level so benchmarks
   * never silently regress.
   */
  void recordMidExecutionCompile(int startSlot, int endSlot, const char* reason);

  // ── Lifecycle enforcement ──────────────────────────────────────────────
  // These methods are the ONLY way to write output slots, sync external inputs,
  // and transition phases. All validation is centralized here.

  /** Violation types for hard error diagnostics. */
  enum class LifecycleViolation {
    BUFFER_REPLACED_POST_FREEZE,
    STALE_WRITE,
    STALE_READ,
    PHASE_VIOLATION
  };

  /**
   * Write to an output slot. The ONLY way to modify outputSlots_.
   * Validates against phase invariants. Hard error on violation.
   * Tracks plan ownership of new arrays.
   */
  void writeOutputSlot(int slotIdx, NDArray* value, const char* tag);

  /**
   * Get the slot state for a specific slot index (for JNI).
   * Returns -1 if slotIdx is out of range.
   */
  int getSlotStateCode(int slotIdx) const {
    if (slotIdx < 0 || slotIdx >= numSlots_) return -1;
    return static_cast<int>(slots_[slotIdx].state_);
  }

  /**
   * Enable/disable per-execution timing breakdown logging.
   * When enabled, prints phase-level timing after each execute() call.
   */
  void setExecutionTimingEnabled(bool enabled) { executionTimingEnabled_ = enabled; }
  bool isExecutionTimingEnabled() const { return executionTimingEnabled_; }

  /**
   * Access the active PlanExecutionContext during execute() lifetime.
   * Returns nullptr outside of execute(). Cast to PlanExecutionContext*
   * in .cpp/.cu files that include PlanExecutionContext.h.
   */
  void* activeExecutionContext() const { return activeExecCtx_; }

  /**
   * Ensure plan-owned staging buffers exist for variable external inputs,
   * then D2D copy current data into them. Returns pointer to internal
   * effectiveExternals_ array (staging buffers for variable inputs,
   * original pointers for non-variable inputs). Only active when frozen.
   * Called once per step (gated by PlanExecutionContext dedup flag).
   */
  NDArray** ensureAndSyncStagingBuffers(NDArray** externalArrays, int numExt, void* stream);

  /**
   * Enable/disable trace logging for DSP execution decisions.
   * When enabled, logs segment dispatch, graph capture/replay decisions,
   * and error paths via DSP_DIAG macros (to stderr).
   * Controlled by -Dnd4j.dsp.trace system property.
   */
  void setTraceEnabled(bool enabled) { traceEnabled_ = enabled; }
  bool isTraceEnabled() const { return traceEnabled_; }

  /**
   * Access the structured execution trace ring buffer.
   * Always non-null after construction. Records segment dispatches,
   * slot writes, graph captures/replays, phase transitions, and errors.
   * Useful for post-mortem diagnostics — call dumpTrace() on crash/error.
   */
  DspExecutionTrace* getTrace() const { return trace_; }

  /**
   * Dump the last `count` trace events to `out` (default: stderr, 64 events).
   * No-op if trace is null.
   */
  void dumpTrace(FILE* out = stderr, int count = 64) const {
    DSP_TRACE_DUMP(trace_, out, count);
  }

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
   * Configure native-plan KV scatter: run scatter as a post-execution step so
   * KV cache updates happen inside the plan without a Java round-trip.
   *
   * Call once after initialization, before the first execute() call.
   * The plan takes ownership of the position buffer (frees it on destroy/reset).
   *
   * @param presentSlotIndices  Output slot indices that hold "present" KV tensors
   * @param staticKvBuffers     Corresponding static KV buffer pointers (must outlive the plan)
   * @param numPairs            Number of (present, static) pairs
   * @param dtype               Data type of all KV tensors (must be uniform)
   * @param heads               Number of attention heads (uniform across all pairs)
   * @param srcSeqLen           Present sequence length (uniform — always 1 for decode)
   * @param dstSeqLen           Static buffer sequence length (= maxKvLen)
   * @param dim                 Head dimension (uniform)
   * @param kvPositionDevice    Device-accessible int64 scalar pointer holding the current
   *                            cache position. Updated by the plan after each execute().
   *                            On CUDA: must be a device pointer; on CPU: host pointer is fine.
   *                            The plan increments the value at this address after each scatter.
   */
  void configureKvScatter(const int* presentSlotIndices,
                           NDArray** staticKvBuffers,
                           int numPairs,
                           DataType dtype,
                           LongType heads,
                           LongType srcSeqLen,
                           LongType dstSeqLen,
                           LongType dim,
                           LongType* kvPositionDevice);

  /**
   * Reset the KV cache position to a specific value (e.g., after prefill).
   * No-op if KV scatter is not configured.
   */
  void resetKvCachePosition(LongType position);

  /**
   * Get the current KV cache position managed by the plan.
   * Returns -1 if KV scatter is not configured.
   */
  LongType getKvCachePosition() const;

  /**
   * Validate that a compiled CPU graph (oneDNN/ACL) covers all ops in the segment.
   * Returns true if every op was compiled by the backend.
   * When any ops are missing, logs warnings and returns false.
   *
   * @param segmentIndex  Which segment to validate (-1 for all segments)
   * @return true if all ops were compiled, false if any were skipped
   */
  bool validateCompiledCpuGraph(int segmentIndex = -1) const;

  // Segment cleanup — public so SegmentLifecycle::invalidateForRebuild can call it.
  void cleanupSegmentForRebuild(GraphSegment& seg, const char* reason);

  // Reset plan-level execute count — public so invalidateForRebuild can
  // re-enable warmup gates (input validation, shape reassignment, DSP diagnostics).
  void resetExecuteCount() { executeCount_ = 0; }

  // Reset frozenConstantDetectionDone_ so detectFrozenConstants() re-runs
  // after the next warmup.  Without this, stale frozen classifications
  // persist through invalidation cycles.
  void resetFrozenConstantDetection() { frozenConstantDetectionDone_ = false; }

  // Reset slot states within a segment range back to WARMUP so frozen
  // contexts with stale dtypes are not reused after invalidation.
  // Keeps cached shapes intact — they're needed by the normal warmup
  // path for output allocation.  Only clears the frozen-context gate.
  void resetSlotStatesForSegment(int startSlot, int endSlot) {
    for (int i = startSlot; i <= endSlot && i < numSlots_; i++) {
      slots_[i].state_ = NativeSlot::SlotState::WARMUP;
    }
  }

  // Public so NativeOps_dsp.cpp diagnostics can query composite state.
  bool hasCompositeHandles(const GraphSegment& seg) const;

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
  NDArray** lastExternalInputs_ = nullptr;       // last ext inputs passed to execute() (not owned)
  int lastNumExternalInputs_ = 0;               // size of lastExternalInputs_
  std::vector<std::string> externalInputNames_;  // name for each external input index
  std::vector<bool> externalInputIsVariable_;    // true if VARIABLE or PLACEHOLDER (needs forced H2D before replay)
  std::vector<int> variableExternalInputIndices_;  // cached indices where externalInputIsVariable_[i]=true (replay optimization)
  bool variableIndicesCached_ = false;             // true once variableExternalInputIndices_ is populated

  // External input ranks captured during the first execute() call.
  // -1 = not yet observed. Used by FusionPass at freeze transition to
  // disambiguate 1D bias vectors from N-D residual operands.
  std::vector<int> externalInputRanks_;

  // Release schedule: releaseAtStep_[stepIdx] = array of slot indices to release
  int** releaseAtStep_;
  int* releaseAtStepCounts_;

  // Requested output mapping
  int* requestedOutputSlotIndices_;
  int numRequestedOutputs_;

  // Execution state (reused across calls)
  NDArray** outputSlots_;              // THE slot arrays — current output values for all slots
  bool* slotIsViewProducer_;           // View producer flags (learned from first exec)
  Context** contextPool_;              // Pre-allocated Context pool
  bool viewProducerDetectionDone_;
  bool frozenConstantDetectionDone_;

  // Unified buffer ownership tracking (Phase 1A).
  // One SlotBufferInfo per totalOutputSlots_. Replaces ad-hoc tracking:
  //   protectedWeightBuffers_ → ownership == WEIGHT/VIEW_OF_WEIGHT
  //   slotViewOutputs_ → ownership == VIEW_OF_SLOT with parentSlotIdx
  //   per-execute dedup HashSet → O(1) ownership check
  SlotBufferInfo* slotOwnership_;

  // Dirty bitmap (generation counter): tracks which output slots were written
  // during the current execution. Used to optimize tickWriteDevice() in steady
  // state — only dirty slots need to be ticked instead of all totalOutputSlots_.
  // Generation counter avoids the O(N) std::fill per step of a bool vector.
  std::vector<uint32_t> dirtySlotGenerations_;
  uint32_t currentDirtyGeneration_ = 1;

  // Graph segments for CUDA Graphs
  std::vector<GraphSegment> segments_;

  // Persistent native-range segments for OneDNN/OpenVINO NativeSlotExecutor callbacks.
  // Keyed by (startSlot << 32 | endSlot). Enables CPU_FROZEN_REPLAY for native range
  // sub-segments that would otherwise be ephemeral stack-allocated GraphSegments.
  std::unordered_map<uint64_t, GraphSegment> nativeRangeSegments_;
  static uint64_t nativeRangeKey(int start, int end) {
    return (static_cast<uint64_t>(start) << 32) | static_cast<uint64_t>(end);
  }

  // pendingClose_ and deferredClose_ REMOVED: arrays persist (one array per slot).
  // View wrappers deleted inline in slotexec when replaced. No batched close needed.

  // Protected DataBuffers: model weights and shapeStatic outputs whose
  // DataBuffers must NEVER be freed during cleanup. Built on first execute().
  // Mirrors Java-side protectedWeightBuffers in DynamicShapePlanExecutor.
  std::unordered_set<DataBuffer*> protectedWeightBuffers_;

  // Arrays created by this plan (via new NDArray in slot execution).
  // The destructor ONLY deletes arrays in this set. Arrays that were
  // passed in as external inputs or that back model variables are NOT
  // in this set and are NOT deleted by the plan.
  std::unordered_set<NDArray*> planOwnedArrays_;

  // Register a newly allocated NDArray as plan-owned. Returns the pointer for inline use.
  SD_INLINE NDArray* registerOwned(NDArray* arr) {
    if (arr != nullptr) planOwnedArrays_.insert(arr);
    return arr;
  }

  // GPU graph capture control
  bool gpuGraphCaptureEnabled_;
  int totalGraphReplays_;

  // NVRTC JIT mode
  JitMode jitMode_;

  // Graph execution mode (controls which backend to use)
  GraphExecutionMode graphExecutionMode_;

  // Shapes-frozen optimization: when enabled, skip shape cache clearing,
  // shape key computation, and unnecessary output zeroing between executions.
  // Use when all external input shapes are guaranteed constant across steps.
  bool shapesFrozen_;
  bool shapeOnlyMode_ = false;  // When true, executeSlot skips op->execute(); all dispatch
                                 // infrastructure runs normally (shape, alloc, frozen detection).
                                 // Use to measure pure dispatch overhead without kernel cost.
  bool shapePrePassDone_ = false;  // True after phaseShapeInferenceOnly has been run automatically
                                    // during the first execute() call. Prevents redundant re-runs.
  bool inShapeChangeWarmup_ = false;  // True during segDispatchCompile's shape-change warmup pass.
                                       // Allows slot shape reassignment in step3_allocateOutputs.
  int executeCount_;  // Tracks executions since shapes were frozen
  uint64_t identityFingerprint_ = 0; // FNV-1a hash of (numSlots, opNames, wiring) — set at deserialization
  bool forceSync_;    // When true, executeSlot forces prepareSpecialUse/registerSpecialUse
                      // regardless of executeCount_. Used during pre-capture warmup at exec=2+.
  bool compilationDone_;  // True after platformPrecompileSegments succeeds; skip phaseCompile

  // Count of compileSegment() calls that occurred AFTER compilationDone_ was set
  // (i.e., mid-execution compiles). Accessed from the shape-change recompile
  // path and read by benchmarks to assert "no compile during measurement".
  // Atomic because phaseCompile() can be multi-threaded.
  std::atomic<int64_t> midExecutionCompileCount_{0};

  // ── Plan-level phase tracking ──────────────────────────────────────────
  // Automatically advanced by execute() based on observed stability.
  // Phase progression: SLOT_BY_SLOT → SHAPES_FROZEN → POINTERS_STABLE → REPLAYING
  PlanPhase planPhase_ = PlanPhase::SLOT_BY_SLOT;
  bool pointersStable_ = false;         // All segment arg tables have stable pointers
  int frozenExecutionCount_ = 0;        // Executions since shapes were frozen (for pointer stability detection)

  // Tracks why the plan was destroyed/reset (for diagnostics)
  PlanDestructionReason destructionReason_ = PlanDestructionReason::NORMAL_CLOSE;

  // ── Lifecycle validation ──────────────────────────────────────────────
  // Buffer pointer snapshot captured when shapes freeze. Used to detect
  // pointer drift (buffer migrated/freed/replaced) during frozen execution.
  // Violations are hard errors, not diagnostic logs.
  BufferPointerSnapshot frozenSnapshot_;

  // Cached steady-state execution context — reused by executeSteadyState() to
  // avoid heap allocation per step on both CPU and CUDA.
  // void* to avoid including PlanExecutionContext.h from this header.
  // Owned by the plan, created on first use, destroyed in destructor/releaseGpuIntermediates.
  void* steadyStateExecCtx_ = nullptr;

#ifdef SD_CUDA
  // CUDA event for lightweight cross-stream synchronization.
  // Recorded on the DSP execution stream after graph replay, then waited on by
  // the default stream (stream 0) before argmax/sampling. This replaces full
  // cudaStreamSynchronize (~1.4ms) with event-based ordering (~0.1ms).
  void* executionCompleteEvent_ = nullptr;  // cudaEvent_t (stored as void* to avoid cuda header)

  void* steadyStateCrossStreamEvent_ = nullptr;  // cudaEvent_t, reused across steps
#endif

  // Active execution context — valid only during execute() lifetime.
  // Set by execute() after platformBeginExecution, cleared before platformEndExecution.
  // Allows _gpubackend.cpp methods to access per-step state (dedup flags, sync tracking)
  // without threading PlanExecutionContext* through every method signature.
  // void* to avoid including PlanExecutionContext.h from this header.
  void* activeExecCtx_ = nullptr;

  // Plan-owned device buffers for variable (placeholder) external inputs.
  // When frozen, data is D2D-copied from the caller's external input into these
  // stable buffers. All GPU code paths (arg table build, capture, replay) resolve
  // external inputs through these buffers, making arg table pointers inherently
  // stable regardless of Java-side allocation patterns.
  //
  // Raw arrays sized to numExternalInputs_, allocated ONCE on first frozen
  // execute, NEVER resized. Same ownership pattern as outputSlots_.
  // Staging buffers are plan-owned NDArrays; effectiveExternals_ contains
  // staging pointers for variable inputs, original pointers for non-variable.
  NDArray** placeholderStagingBuffers_ = nullptr;
  NDArray** effectiveExternals_ = nullptr;
  std::vector<int> cachedVariableExtIndices_;  // fast-path: only iterate variable inputs

  // Per-execution timing breakdown (enabled by setExecutionTimingEnabled)
  bool executionTimingEnabled_;

  // Trace logging for execution decisions (enabled by setTraceEnabled / -Dnd4j.dsp.trace)
  bool traceEnabled_ = false;

  // Structured execution trace ring buffer — always allocated, records every
  // segment dispatch, slot write, graph capture/replay, phase transition, and
  // error. Call dumpTrace() to dump the last N events on crash or error.
  DspExecutionTrace* trace_ = nullptr;

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

  // Internal methods
  // flushPendingClose REMOVED: arrays persist, view wrappers deleted inline
  void buildSegments();
  void resegmentForFreeze();
  SelectedBackend resolveBackendForSegment(bool isCapturable) const;

  // ── Slot execution (NativeDynamicShapePlan_slotexec.cpp) ──
  Status executeSlot(int slotIdx, NDArray** externalArrays, int numExt, void* stream);
  LongType computeShapeKey(NativeSlot& slot, NDArray** inputs, int numInputs);
  void detectFrozenConstants();

  /**
   * Check if the given NDArray pointer is referenced by any OTHER output slot
   * besides the given slotIdx. Used to prevent deleting arrays that are shared
   * between slots via identity ops or in-place fusion.
   *
   * Identity ops (outputSlots_[si] = input) create pointer aliasing between slots
   * without reference counting. If we blindly delete outputSlots_[si] when
   * replacing it, we may delete an array still referenced by another slot.
   *
   * @param arr     The NDArray pointer to check
   * @param skipIdx The slot index to exclude from the check (-1 to check all)
   * @return true if any other slot references this pointer
   */
  inline bool isSlotArrayShared(const NDArray* arr, int skipIdx) const {
    if (arr == nullptr || outputSlots_ == nullptr) return false;
    auto* db = const_cast<NDArray*>(arr)->dataBuffer();
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (i == skipIdx) continue;
      if (outputSlots_[i] == nullptr) continue;
      // Check pointer identity (identity ops create aliases)
      if (outputSlots_[i] == arr) return true;
      // Check DataBuffer sharing (views share underlying GPU buffer).
      // Without this, deleting the original frees the DataBuffer while
      // views still reference it → dangling DataBuffer → heap corruption.
      if (db != nullptr && outputSlots_[i]->dataBuffer() == db) return true;
    }
    return false;
  }

  // ── Segment management (NativeDynamicShapePlan_segments.cpp) ──
  LongType computeSegmentShapeKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  Status executeSegmentSlotBySlot(GraphSegment& seg, NDArray** externalArrays,
                                  int numExt, void* stream);
  // Refresh stale view-producer NDArray wrappers in a segment.
  // When a view op wraps a placeholder (e.g., squeeze(x)) and SameDiff replaces
  // the placeholder's DataBuffer between calls, the cached view wrapper in
  // outputSlots_ points at the (now-closed) prior DataBuffer. This helper
  // walks slots in [seg.def.startSlot, seg.def.endSlot], detects view-producer
  // slots with stale/invalid DataBuffers, and re-creates the view wrapper
  // pointing at the current input's DataBuffer (using the cached output shape).
  //
  // Implementation lives in NativeDynamicShapePlan_slotexec.cpp alongside
  // tryCreateViewForSlot() so it can reuse the zero-copy view construction logic.
  //
  // Returns: refreshedCount on success (0 if nothing to refresh, positive if
  //          one or more wrappers were refreshed), or -1 if any view-producer
  //          slot could not be refreshed (caller must trigger graph invalidation).
  int refreshStaleViewWrappersInSegment(GraphSegment& seg, NDArray** externalArrays, int numExt);
  // Unified pre-segment zero pass — zeroes outputs for slots with needsZeroedOutput.
  // Safe to call during CUDA graph capture (memsets get recorded into the graph).
  void prezeroSegmentOutputs(const GraphSegment& seg, void* stream);
  Status executeSegmentWithCpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  const std::vector<GraphBackend*>& getCpuGraphBackendChain();
  Status executeSegmentWithSpecificBackend(GraphSegment& seg, GraphBackend* backend,
                                           NDArray** externalArrays, int numExt, void* stream);

  // ── Emulated graph replay (NativeDynamicShapePlan_segments.cpp) ──
  Status executeSegmentEmulatedReplay(GraphSegment& seg, NDArray** externalArrays,
                                      int numExt, void* stream);
  LongType computeSegmentInputAddrKeyPortable(GraphSegment& seg, NDArray** externalInputs, int numExt);

  // ── CUDA graph capture/replay (NativeDynamicShapePlan_cudagraph.cu) ──
#ifdef SD_CUDA
  LongType computeSegmentInputAddrKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  LongType computeCreateOpValueKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  void snapshotExternalAddrs(GraphSegment& seg, NDArray** externalInputs, int numExt);
  bool externalAddrsMatch(const GraphSegment& seg, NDArray** externalInputs, int numExt) const;
  Status executeSegmentWithGraph(GraphSegment& seg, NDArray** externalArrays,
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
  void platformMigrateSegmentInputs(const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs);
  void platformCleanupMigratedInputs();
  bool platformShouldUseGraph(const GraphSegment& seg);
  Status platformExecuteSegmentWithBackends(GraphSegment& seg, NDArray** externalInputs,
                                             int numExternalInputs, void* stream, bool& usedGraph);
  Status platformCheckPostSegment(GraphSegment& seg);
  void platformCleanupSegmentForRebuild(GraphSegment& seg);
  void platformFreePlanResources();
  int platformCountCapturedGraphSegments() const;
  void platformMaybeSplitIfEnabled();

  // ── Additional platform dispatch (extracted from NativeDynamicShapePlan.cpp) ──
  void* platformBeginExecution(void* stream, bool frozen, int execCount);
  void platformEndExecution(void* executionState, void* stream, bool frozen, int execCount);
  void platformDumpExternalInputDiagnostics(NDArray** ext, int numExt, int execCount);
  void platformDumpExtInputGpuValues(NDArray* arr, int extIdx, int execCount, void* stream);
  void platformClearCastCache();
  void platformPostSegmentPoolManagement(bool frozen, int execCount);
  void platformDumpLogitsArgmax(int execCount, void* stream);
  void platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream);
  void platformPreReplayPoolStats(size_t& poolUsedOut, size_t& poolReservedOut);
  void platformPostReplayPoolManagement(size_t poolUsedPre, bool frozen, int execCount);
  void platformTraceSlotValues(const GraphSegment& seg, void* stream, int execCount);
  SelectedBackend platformResolveBackend(bool isGraphCapture) const;
  bool platformShouldBreakSegmentAtTraitBoundary(int currIdx, int prevIdx) const;
  void platformReleaseSegmentGpuResources();
  void platformMigrateWeightsAndClearCaches();

  // ── Slot execution platform dispatch (NativeDynamicShapePlan_slotexec_cuda.cu / _cuda_stubs.cpp) ──
  // These abstract CUDA-specific slot execution behavior (prezero, actuality,
  // buffer validation, cublasLt epilogue, verify logging).
  void platformPrezeroSegmentOutputs(const GraphSegment& seg, void* stream);
  void platformReconcileOutputActuality(const char* stage, int stepIdx,
                                         const NativeSlot& slot, NDArray* output);
  bool platformValidateSlotInputBuffer(int stepIdx, const NativeSlot& slot,
                                        int inputIdx, NDArray* input);
  bool platformValidateReusableSlotBuffer(NDArray* cached);
  void platformSetLtEpilogue(const NativeSlot& slot, NDArray* biasArray);
  void platformClearLtEpilogue();
  void platformLogSlotOutput(int stepIdx, const char* opName, const char* tag,
                              const int* outputSlotIndices, int numOutputs);

  // ── GPU graph backend (NativeDynamicShapePlan_gpubackend.cpp) ──
  Status executeSegmentWithGpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getGpuGraphBackend();
  void clearGpuBackendFailedCache();

  // ── Segment lifecycle dispatch methods ──
  // Each handles one phase of the segment state machine.
  // Called from executeSegmentWithGpuGraph based on lifecycleState.
  Status segDispatchWarmup(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream);
  // segDispatchCompile — handles one compile cycle for a segment.
  // segShapeKey is passed by reference: the shape-change recompile path
  // recomputes the key after a mini-warmup and writes the new value back
  // so the caller's shapeKeyState.markCompiled() uses the correct key.
  Status segDispatchCompile(GraphSegment& seg, NDArray** externalArrays,
                            int numExt, void* stream, LongType& segShapeKey);

#ifdef SD_CUDA
  // segDispatchReplay — attempts composite replay for a segment that has
  // captured composite handles. Returns OK if replay succeeded, MAYBE if
  // replay conditions not met (caller falls through to capture/direct).
  Status segDispatchReplay(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream,
                           bool allowTritonCudaGraphReplay,
                           bool createValuesStable, bool extAddrsStable,
                           LongType segShapeKey, const char* backendName);

  // segDispatchCaptureOrDirect — handles CUDA graph capture AND direct
  // (non-capture) Triton execution. Includes TritonOrderedRangeGuard RAII,
  // capture decision, composite/monolithic capture, and direct fallback.
  // SegCaptureCtx bundles pre-computed state from the preamble.
  struct SegCaptureCtx;
  Status segDispatchCaptureOrDirect(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream,
                                    SegCaptureCtx& ctx);

  void proactivePreCaptureMemoryCleanup(GraphSegment& seg, int segIdx, void* stream);
  int evictLruGraphs(int segIdx, size_t neededBytes, void* stream);

  // ── Clean capture/replay/gap paths ──
  Status compositeCapture(GraphSegment& seg, ReplaySchedule& sched,
                          NDArray** externalArrays, int numExt, void* stream);
  Status compositeReplay(GraphSegment& seg, ReplaySchedule& sched,
                         NDArray** externalArrays, int numExt, void* stream);
#endif

  // CPU graph backend (oneDNN Graph or ACL Dynamic Fusion)
  GraphBackend* cpuGraphBackend_;
  bool cpuGraphBackendChecked_;

  // Prioritized chain of all available CPU graph backends for per-segment cascade
  std::vector<GraphBackend*> cpuGraphBackendChain_;
  bool cpuGraphBackendChainBuilt_ = false;

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
  void abortCapture(GraphSegment& seg, bool freeHostPtrs, bool didPushCtx, int captureDevice,
                    cudaStream_t prevCaptureStream,
                    const std::vector<NativeSlot::SlotState>& savedSlotState,
                    void* stream);
#endif

  // Cross-device input migration: tracks arrays copied to a different device
  // for segment execution. Cleaned up after each segment completes.
  struct MigratedInput {
    int outputSlotIdx;       // Which outputSlots_[] entry was replaced
    NDArray* original;       // Original array (on source device) - restore after segment
    NDArray* migrated;       // Migrated copy (on target device) - delete after segment
  };
  std::vector<MigratedInput> migratedInputs_;

  // Max-allocation mode for KV cache outputs
  // Maps output slot index -> max number of elements to pre-allocate
  std::unordered_map<int, LongType> outputSlotMaxSizes_;
  // Tracks which slots have been pre-allocated at max size
  std::unordered_set<int> maxAllocatedSlots_;

  // ── Native KV scatter post-execution (plan-managed KV cache updates) ────────
  // When configured, the plan runs batched KV scatter after each segment execution,
  // eliminating the Java-side scatterNewEntries() round-trip. The position counter
  // lives in a device-accessible (on CUDA) or host (on CPU) int64 scalar and is
  // incremented by the plan after each scatter.
  //
  // This is CUDA graph capture-compatible because:
  //   - Entry device pointers (srcPtr, dstPtr) are stable across steps
  //   - kvPositionDevice_ address is stable; only the VALUE changes between steps
  //   - Position update uses cudaMemcpyAsync (baked into graph at capture time)
  struct NativeKvScatterEntry {
    int presentSlotIdx;     // Output slot holding present KV tensor
    NDArray* staticBuf;     // Static KV buffer (not owned — passed by Java/caller)
    LongType heads;
    LongType srcSeqLen;
    LongType dstSeqLen;
    LongType dim;
  };
  std::vector<NativeKvScatterEntry> kvScatterEntries_;
  DataType kvScatterDtype_ = DataType::FLOAT32;
  LongType* kvPositionDevice_ = nullptr;  // Device-accessible int64 position scalar (owned)
  bool kvScatterConfigured_ = false;

  // Execute native KV scatter as a post-execution step.
  // Called from execute() after all phase dispatch completes.
  void executeKvScatterPostExec(void* stream);

  // Release KV scatter resources (called from destructor and releaseGpuIntermediates)
  void releaseKvScatterResources();

#ifdef SD_CUDA
  // ── Pre-capture batch-zero ─────────────────────────────────────────────
  // collectBatchZeroTargets walks the segment's gap slots and builds the set
  // of output buffers that must be zeroed before CUDA graph capture. The
  // consumer is a cudaMemsetAsync loop in NativeDynamicShapePlan_gpubackend.cpp
  // that runs outside capture (fill engines, no SM competition).
  std::vector<BatchZeroEntry> batchZeroEntries_;
  bool gapPrezeroTargetsCached_ = false;  // True after first collectBatchZeroTargets in frozen replay
  int cachedGapPrezeroCount_ = 0;         // Cached count for fast path (avoids recompute)

  void collectBatchZeroTargets(const std::unordered_set<int>& gapSlots);

  // ── Active gap slot cache ───────────────────────────────────────────────
  // In frozen steady state, ~97% of the 2743 gap slot iterations are skipped
  // (frozen constants, identity ops, fused tails, view-buffer-unchanged).
  // After the first full pass, cache the ~82 slots that actually need work.
  // Each entry records the slot index and what kind of action it needs:
  //   EXECUTE:       call executeSlot() — the default for non-trivial ops
  //   BATCHED_GEMM:  call executeBatchedGemmGroup() — trigger slot for a batched group
  //   IDENTITY_TICK: tick device actuality only (identity alias already installed)
  //   VIEW_TICK:     tick device actuality (view buffer unchanged from input)
  enum class ActiveSlotAction : uint8_t {
    EXECUTE = 0,
    BATCHED_GEMM = 1,
    IDENTITY_TICK = 2,
    VIEW_TICK = 3,
    SKIP = 4,        // batched GEMM non-trigger slot (handled by trigger)
  };
  struct ActiveGapSlot {
    int slotIdx;
    ActiveSlotAction action;
    int batchedGemmGroupIdx;    // only valid for BATCHED_GEMM action
    int outputSlotIdx;          // first output slot for tick actions
  };
  std::vector<ActiveGapSlot> cachedActiveGapSlots_;
  bool activeGapSlotsCached_ = false;

  // Shared capture workspace: all segments share one 128MB workspace
  // instead of each allocating their own. Since segments execute sequentially
  // and the workspace is scratch space (offset resets each capture), sharing
  // is safe. The pointer is baked into CUDA graph nodes during capture,
  // so all segments must capture with the same address (guaranteed by
  // allocating once and reusing).
  void* sharedCaptureWorkspace_ = nullptr;
  size_t sharedCaptureWorkspaceBytes_ = 0;
  int sharedCaptureWorkspaceDevice_ = -1;

  // ── Batch D2D copy optimization ─────────────────────────────────────────
  void* batchD2DDeviceSrcPtrs_ = nullptr;   // Device: void*[count]
  void* batchD2DDeviceDstPtrs_ = nullptr;   // Device: void*[count]
  void* batchD2DDeviceSizes_ = nullptr;     // Device: size_t[count]
  void* batchD2DHostSrcPtrs_ = nullptr;     // Pinned host: void*[count]
  void* batchD2DHostDstPtrs_ = nullptr;     // Pinned host: void*[count] (static)
  void* batchD2DHostSizes_ = nullptr;       // Pinned host: size_t[count] (static)
  int batchD2DCount_ = 0;                   // Number of valid entries
  int batchD2DAllocated_ = 0;               // Allocated capacity

  void freeBatchD2DResources();
  void prepareBatchD2DDevice(int count, cudaStream_t stream);
  void launchBatchD2D(cudaStream_t stream);
  void launchBatchMemset(cudaStream_t stream, void** dstPtrsHost, size_t* sizesHost, int count);
#else
  void freeBatchD2DResources() {}
#endif

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
    bool ptrStable = false;  // true when H2D pointer arrays match device-side
  };
  std::vector<BatchedGemmGroup> batchedGemmGroups_;
  // Maps slot index → index into batchedGemmGroups_ (-1 if not part of a group)
  std::vector<int> slotToBatchedGemmGroup_;

  const LongType* resolveInputShapeInfo(int srcIdx, NDArray** externalArrays, int numExt) const;
  void detectBatchedGemmGroups(NDArray** externalArrays, int numExt);
  void reconcileSlotDispatchAfterMerge(const ReplaySchedule& sched);
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
