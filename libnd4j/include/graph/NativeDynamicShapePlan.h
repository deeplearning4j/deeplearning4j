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

#include <graph/DspGraphTypes.h>
#include <graph/GraphReplayHandle.h>
// ModeContract.h included AFTER GraphExecutionMode enum (see below) to break circular dependency.
#include <graph/SlotBufferOwnership.h>
#include <graph/PlanDefinition.h>
#include <graph/ExecutionState.h>
#include <graph/DspBufferColorMap.h>
#include <graph/DspBufferPool.h>
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

#if defined(SD_VULKAN)
class VulkanExecutionStream;
#endif

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
  GEM_ARM_HYBRID = 7,   // ARM placement policy: ordered accelerator/ACL candidates + functional replay
  GEM_NNAPI = 8,        // Force complete Android NNAPI lowering (diagnostic/strict mode)
  GEM_HIP_GRAPHS = 9,   // HIP graph capture/replay (AMD ROCm) — mirrors CUDA graphs
  GEM_LEVELZERO = 10,   // Intel Level Zero mutable command list replay
  GEM_VULKAN = 11,      // Vulkan compute command buffer replay
  GEM_METAL = 12,       // Metal indirect command buffer replay (Apple GPU)
  GEM_TPU = 13,         // TPU HLO compilation + PJRT execution caching
  GEM_HEXAGON = 14,     // Hexagon-MLIR NPU compilation + command list replay
  GEM_OPENVINO = 15,    // Force OpenVINO CPU graph backend (Intel x86, broad op coverage)
  GEM_TVM = 16,         // Deprecated: TVM removed, use triton-cpu instead
  GEM_EMULATED_REPLAY = 17,  // Emulated graph replay: slot-by-slot with replay lifecycle diagnostics
  GEM_SHAPE_INFERENCE_ONLY = 18, // Shape inference only: calculates output shapes without executing ops
  GEM_PORTABLE_REPLAY = 19,      // Best executable replay recorder; never selects Triton/NVRTC/PTX
  GEM_ONEDNN = 20                // Force exact oneDNN Graph lowering; unsupported ranges fail closed
};

/**
 * Runtime-owned configuration that must be installed before segment discovery
 * and backend compilation begin.
 */
struct NativePlanCompileOptions {
  bool runtimeCompilationAllowed = true;
  std::string runtimeArtifactDirectory;
  std::string deviceCompilationCacheDirectory;
  std::string deviceCompilationCacheModelKey;
};

// Close namespace before including ModeContract.h — it opens its own sd::graph namespace.
// Including inside an open namespace would create sd::graph::sd::graph.
}  // namespace graph
}  // namespace sd

#include <graph/ModeContract.h>

namespace sd {
namespace graph {

/**
 * SelectedBackend — selects a backend family once per segment.
 * GRAPH_BACKEND delegates concrete implementation choice to the shared
 * GraphBackendResolver and caches the successful implementation per segment.
 */
enum class SelectedBackend : uint8_t {
  SLOT_BY_SLOT = 0,    // Execute each op individually (no fusion, no graphs)
  DEVICE_REPLAY = 1,   // Platform-native replay without a compiler/recorder backend
  GRAPH_BACKEND = 3,   // Resolver-selected compiler or recorder backend
  EMULATED_REPLAY = 4, // Emulated graph replay: slot-by-slot with replay lifecycle tracking
  VULKAN_REPLAY = 5,   // Legacy Vulkan replay path pending GraphBackend migration
};

/**
 * Result of preparing external inputs for a captured/replayed backend.
 *
 * A null effective-external pointer is intentionally not the only failure
 * signal: callers must distinguish a successful passthrough (no staging was
 * required) from a failed staging operation.  Graph capture/replay must abort
 * on every failure rather than baking raw caller addresses into a graph.
 */
enum class DspStagingSyncStatus : uint8_t {
  SUCCESS = 0,
  NOT_REQUIRED = 1,
  MISSING_EXECUTION_CONTEXT = 2,
  DEVICE_SELECTION_FAILED = 3,
  ALLOCATION_FAILED = 4,
  TRANSFER_FAILED = 5,
  SYNCHRONIZATION_FAILED = 6,
};

struct DspStagingSyncResult {
  NDArray** effectiveExternals = nullptr;
  DspStagingSyncStatus status = DspStagingSyncStatus::SYNCHRONIZATION_FAILED;
  int cudaError = 0;
  bool usedStaging = false;

  SD_INLINE bool ok() const {
    return status == DspStagingSyncStatus::SUCCESS ||
           status == DspStagingSyncStatus::NOT_REQUIRED;
  }
};

/**
 * Backend-neutral result for one segment execution attempt.
 *
 * BAD_GRAPH is reserved for a pre-execution backend rejection and may be
 * offered to the next admitted backend.  Once executionStarted is true, the
 * caller must not fall back because stateful/in-place/scatter operations may
 * already have mutated plan state.  invocationSatisfiedByWarmup prevents a
 * shape-drift warmup from being followed by a second compiled execution in
 * the same invocation.
 */
struct DspExecutionResult {
  Status status = Status::KERNEL_FAILURE;
  bool executionStarted = false;
  bool invocationSatisfiedByWarmup = false;

  DspExecutionResult() = default;
  DspExecutionResult(Status value, bool started = false, bool warmupSatisfied = false)
      : status(value), executionStarted(started),
        invocationSatisfiedByWarmup(warmupSatisfied) {}

  SD_INLINE bool ok() const { return status == Status::OK; }
  SD_INLINE bool preExecutionRejection() const {
    return status == Status::BAD_GRAPH && !executionStarted;
  }
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
 *   SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
 *
 * Each phase guarantees everything from prior phases plus additional invariants:
 *   SLOT_BY_SLOT:      No assumptions. Shapes may change, pointers may move.
 *   SHAPES_FROZEN:     All output shapes are constant. Shape inference skipped.
 *                      Per-segment generation counters track pointer stability.
 *   REPLAYING:         Shapes frozen + all segments in replay steady state.
 *                      Pointers stable (generation counters match). Graph replay active.
 *
 * Phase is automatically advanced by execute() based on observed stability.
 * Can be manually set backward (e.g., unfreeze → SLOT_BY_SLOT).
 */
enum class PlanPhase : uint8_t {
  SLOT_BY_SLOT = 0,      // No guarantees — shapes and pointers may change
  SHAPES_FROZEN = 1,     // Shapes are constant across executions
  REPLAYING = 2,         // Steady state — graph replay active, pointers stable
};

// ExecutionPhase REMOVED — unified into SegmentLifecycleState (in GraphSegmentExec).
// Mapping for callers that previously used ExecutionPhase:
//   WARMUP      → lifecycleState == NEEDS_WARMUP
//   COMPILING   → lifecycleState == NEEDS_COMPILE
//   COMPILED    → lifecycleState == CAPTURE_PENDING
//   REPLAYING   → lifecycleState == REPLAYING (includes captured state)
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
 *   slot.isDataDependent()  (was slot.flags.isDataDependent, now derived from opTraits_)
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
  uint8_t* optionalOutputMask = nullptr; // 1: output is logically declared but not demanded

  ~SlotWiring() {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
    delete[] optionalOutputMask;
  }
  SlotWiring() = default;
  SlotWiring(SlotWiring&& o) noexcept
      : numInputs(o.numInputs), inputSourceIndices(o.inputSourceIndices),
        inputSourceTypes(o.inputSourceTypes), numOutputs(o.numOutputs),
        outputSlotIndices(o.outputSlotIndices), optionalOutputMask(o.optionalOutputMask) {
    o.numInputs = 0; o.inputSourceIndices = nullptr; o.inputSourceTypes = nullptr;
    o.numOutputs = 0; o.outputSlotIndices = nullptr; o.optionalOutputMask = nullptr;
  }
  SlotWiring& operator=(SlotWiring&& o) noexcept {
    if (this != &o) {
      delete[] inputSourceIndices; delete[] inputSourceTypes; delete[] outputSlotIndices;
      delete[] optionalOutputMask;
      numInputs = o.numInputs; inputSourceIndices = o.inputSourceIndices;
      inputSourceTypes = o.inputSourceTypes; numOutputs = o.numOutputs;
      outputSlotIndices = o.outputSlotIndices; optionalOutputMask = o.optionalOutputMask;
      o.numInputs = 0; o.inputSourceIndices = nullptr; o.inputSourceTypes = nullptr;
      o.numOutputs = 0; o.outputSlotIndices = nullptr; o.optionalOutputMask = nullptr;
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

/**
 * Execution flags and fusion metadata.
 *
 * Fields here are either:
 *   - Compile-time resolved values that cannot be derived from OpTraits alone
 *     (outputShapeDependsOnInputValues can be cleared per-instance).
 *   - Runtime state set during warmup/fusion passes.
 *
 * Trait-derivable properties (isDataDependent, isIdentityOp, isViewCapableOp,
 * isFullyWriting, needsZeroedOutput) live as query methods on NativeSlot,
 * backed by the opTraits_ bitmask.
 */
struct SlotFlags {
  // Compile-time resolved: starts from VALUE_DEPENDENT_SHAPE || DATA_DEPENDENT,
  // then refined per-instance (e.g. concat without axis-in-last-arr cleared).
  bool outputShapeDependsOnInputValues = false;
  bool needsIntLongSync = false;
  bool isCustomOp = true;
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
   * This flag is monotonic: normally classified during warmup, but a view may
   * be promoted later when replay first observes that its internal producer has
   * a dynamic extent. It is never cleared.
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
  // Full shape-info buffers declared by zero-input ops in the Java graph.
  // Owned by the slot so native shape propagation never depends on a Java buffer lifetime.
  std::vector<std::vector<LongType>> staticOutputShapeInfos;
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

  // ── Op trait bitmask (single source of truth for op classification) ─
  //
  // Set once at compile time (NativePlanCompiler) or deserialization
  // (fromSerializedPlan). The where-hack (where with 3 inputs = elementwise
  // select, not data-dependent) is applied by clearing the DATA_DEPENDENT
  // bit from this mask rather than a runtime override.
  //
  // Query methods below derive all classification decisions from this mask.
  uint64_t opTraits_ = 0;

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
  // ── Unified slot lifecycle (SINGLE source of truth) ─────────────────
  // Replaces the old SlotState enum (WARMUP/SHAPE_CACHED/FROZEN/FROZEN_CONSTANT).
  // All state now lives in slotPhase: phase (BUILDING/SEALED), isConstant,
  // isViewProducer, shapeCacheValid. Use slotPhase methods for all queries.
  SlotPhase slotPhase;

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

  /**
   * Primary buffer pointers snapshotted at freeze time (detectFrozenConstants).
   * One entry per output slot, indexed by output ordinal (0..numOutputs-1).
   * Null pointer means "not snapshotted" (e.g. slot was not frozen or output
   * was null at freeze time).
   *
   * Used in the frozen constant skip path to detect buffer reallocation:
   * if the current primary pointer differs from the frozen snapshot, the buffer
   * was reallocated and the frozen value is lost — fall through to re-execute.
   *
   * Size: 0 until the slot is frozen, then resized to numOutputs.
   */
  std::vector<void*> frozenOutputPtrs;

  NativeSlot() = default;

  // ── Phase queries (delegate to slotPhase) ──────────────────────────
  bool shapeCacheValid() const { return slotPhase.shapeCacheValid; }
  bool frozenContextReady() const { return slotPhase.isSealed(); }
  bool frozenConstantSlot() const { return slotPhase.isSealed() && slotPhase.isConstant; }

  // ── Trait queries (derived from opTraits_ bitmask) ────────────────
  // These replace the removed SlotFlags booleans (isDataDependent,
  // isIdentityOp, isViewCapableOp, isFullyWriting, needsZeroedOutput).
  // Each is a const inline bitcheck — zero overhead vs stored booleans.

  bool hasOpTrait(uint64_t trait) const { return (opTraits_ & trait) != 0; }
  void addOpTrait(uint64_t trait) { opTraits_ |= trait; }
  void clearOpTrait(uint64_t trait) { opTraits_ &= ~trait; }
  uint64_t opTraits() const { return opTraits_; }

  bool isDataDependent() const { return hasOpTrait(sd::ops::OP_TRAIT_DATA_DEPENDENT); }
  bool isIdentityOp()    const { return hasOpTrait(sd::ops::OP_TRAIT_IDENTITY); }
  bool isViewCapableOp() const { return hasOpTrait(sd::ops::OP_TRAIT_VIEW_PRODUCING); }
  bool usesExternalWorkspace() const { return hasOpTrait(sd::ops::OP_TRAIT_EXTERNAL_WORKSPACE); }
  /** Output size depends on runtime data values (Where 1-arg, Unique, NonZero, NMS).
   *  These ops require host synchronization during execution to determine output
   *  tensor dimensions, which invalidates CUDA graph capture streams. */
  bool hasDynamicOutputSize() const { return hasOpTrait(sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE); }

  /** View or identity — aliases input buffer without computing. */
  bool aliasesInput() const { return isViewCapableOp() || isIdentityOp(); }

  /** Op writes every element of its output (no partial writes, no aliasing).
   *  DATA_DEPENDENT is orthogonal to write coverage: it means the shape
   *  function reads input values (e.g., argmax 2-input reads axes from
   *  input[1]), NOT that the write extent varies. Ops with variable output
   *  size (unique, non_max_suppression, 1-input where) simply don't carry
   *  OP_TRAIT_FULLY_WRITING — that's the correct separation of concerns. */
  bool isFullyWriting() const {
    return hasOpTrait(sd::ops::OP_TRAIT_FULLY_WRITING);
  }

  /** Output needs zero-fill before execution (not a view/identity, not fully writing). */
  bool needsZeroedOutput() const { return !aliasesInput() && !isFullyWriting(); }

  /** Shape function reads input tensor VALUES (not just shapes).
   *  The compiler initializes outputShapeDependsOnInputValues from the
   *  descriptor's OP_TRAIT_VALUE_DEPENDENT_SHAPE bit and then resolves
   *  argument- versus tensor-driven forms for this concrete slot. Consumers
   *  must use that resolved value; re-reading the intrinsic bit here would undo
   *  the per-instance resolution.
   *
   *  OP_TRAIT_DYNAMIC_OUTPUT_SIZE remains unconditional because the number of
   *  output elements itself is determined from runtime data. */
  bool hasValueDependentShape() const {
    return flags.outputShapeDependsOnInputValues ||
           hasOpTrait(sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
  }

  /**
   * Monotonically promote this slot to dynamic-shape execution.
   *
   * All classification paths use this transition instead of mutating
   * SlotFlags directly. Returning true only on the first promotion lets callers
   * attach one-time diagnostics or invalidation without duplicating state checks.
   */
  bool markDynamicShape() {
    if (flags.isDynamicShape) return false;
    flags.isDynamicShape = true;
    return true;
  }

  // ── In-place fusion state management ────────────────────────────────
  // Centralizes all reads and writes to inPlaceFused / inPlaceFusedInputIdx
  // so callers don't scatter paired field mutations across files.

  /** Whether this slot executes in-place (writes into its input buffer). */
  bool isInPlaceFused() const { return flags.inPlaceFused; }

  /** The input index whose buffer is reused as output, or -1 if not in-place. */
  int inPlaceFusedInputIdx() const { return flags.inPlaceFusedInputIdx; }

  /**
   * The source output-slot index that this in-place op will overwrite,
   * or -1 if not in-place or the input comes from an external array.
   */
  int inPlaceSourceSlot() const {
    if (!flags.inPlaceFused || flags.inPlaceFusedInputIdx < 0 ||
        flags.inPlaceFusedInputIdx >= wiring.numInputs) return -1;
    return wiring.inputSourceIndices[flags.inPlaceFusedInputIdx];
  }

  /** Mark this slot for in-place execution using the given input index. */
  void enableInPlaceFusion(int inputIdx) {
    flags.inPlaceFused = true;
    flags.inPlaceFusedInputIdx = inputIdx;
  }

  /** Clear in-place fusion (e.g. source is a requested output or frozen). */
  void disableInPlaceFusion() {
    flags.inPlaceFused = false;
    flags.inPlaceFusedInputIdx = -1;
  }

  /**
   * Whether this slot's output buffer needs prezero memset before execution.
   * Consolidates the skip-prezero logic that was duplicated across 4 call sites:
   *   - executeSlot prezero lambda (slotexec.cpp)
   *   - prezeroSegmentOutputs (slotexec_cuda.cu)
   *   - prezeroSegmentOutputs (cuda_stubs.cpp)
   *   - batchZeroSegmentOutputs (batchzero.cu)
   */
  bool needsPrezero() const {
    if (frozenConstantSlot()) return false;        // output never changes
    if (aliasesInput()) return false;              // no output buffer to zero
    if (isFullyWriting()) return false;            // op writes every element
    if (isInPlaceFused()) return false;             // handled by fusion host
    if (fusedChain.isFusedChainTail) return false; // tail of fused chain
    return true;
  }

  // ── Compiler backend eligibility ─────────────────────────────────
  // Compiler backends (OpenVINO, NNAPI, MLX, ARM, TPU, Hexagon) lower a
  // segment instead of recording its slot-by-slot execution. Host reads in a
  // slot implementation therefore do not make an op ineligible: the backend's
  // canResolveSegment()/lowering checks decide whether the op is supported.
  // Control flow and truly dynamic output extents still require plan-level
  // execution and cannot be admitted as an ordinary compiled segment.
  bool isCompilerBackendEligible() const {
    if (cf.controlFlowType != CF_NONE) return false;
    return !hasDynamicOutputSize();
  }

  // ── Capturability (single source of truth) ────────────────────────
  // Returns true if this slot can live inside a CUDA graph capture.
  //
  //  mergeViews = true  →  view/identity/frozen-constant ops are allowed
  //                        (zero-copy metadata that doesn't break capture).
  //  mergeViews = false →  those ops are excluded (e.g. gap analysis when
  //                        mergedCaptureThroughViews config is off).
  //
  // Every call site that decides "can I capture this slot?" MUST use
  // this method instead of ad-hoc flag checks.
  bool isCapturable(bool mergeViews = true) const {
    if (cf.controlFlowType != CF_NONE) return false;
    // Ops with dynamic output size (e.g. single-arg Where, Unique, NonZero, NMS)
    // require host synchronization during execution to determine output tensor
    // dimensions. This invalidates CUDA graph capture streams (cudaStreamCaptureStatusInvalidated).
    // DATA_DEPENDENT alone is not sufficient: reshape, concat, and argmax can
    // still execute as regular GPU kernels. Runtime-controlled slice-family ops
    // read begin/size tensors on the host (asVectorT), so they must remain live;
    // static argument slices have no host control read and are capture-safe.
    if (hasDynamicOutputSize() ||
        (hasOpTrait(sd::ops::OP_TRAIT_SLICE) && hasValueDependentShape())) return false;
    if (!mergeViews && (isViewCapableOp() || isIdentityOp() || frozenConstantSlot()))
      return false;
    return true;
  }

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

  /// Diagnostic snapshot of the unique cross-segment input signatures used by
  /// the last successful GPU compilation. Populated only when LIFECYCLE
  /// diagnostics are enabled, so the shape-drift path can identify the exact
  /// input that changed without adding steady-state overhead.
  std::vector<std::string> compiledInputSignatures;

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
    compiledInputSignatures.clear();
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
  int startSlot = 0;  // Inclusive slot index.
  int endSlot = 0;    // Inclusive slot index.
  // Legacy name: for graph-capture modes this means CUDA-capturable; for
  // compile-only modes it means eligible for backend admission. In both cases
  // false routes the segment to the explicit plan-level execution path.
  bool isCapturable = false;
  // Independent of graph-backend admission. True when this range can be
  // recorded as an immutable FunctionalReplay program; false ranges still use
  // the explicit plan-level lane but execute live (for control flow or dynamic
  // output extents).
  bool isFunctionalReplayEligible = false;
  bool hasValueDepOps = false;         // True if any slot has outputShapeDependsOnInputValues
  // True when a cross-segment input is produced by a slot whose output extent
  // may change at runtime. Such segments must validate their boundary shape key
  // before replay even when the plan's placeholder shapes are frozen.
  bool hasDynamicBoundaryInputs = false;
  /// Structured shape key lifecycle. All shape key reads/writes go through this.
  ShapeKeyState shapeKeyState;

  // User-forced backend override (empty = automatic selection via priority chain)
  std::string backendOverride;

  // True if EVERY slot in this segment is a frozen constant (frozenConstantSlot()).
  // Set once at buildSegments() time after warmup populates slot state.
  // These segments need no execution or capture — outputs are already populated
  // from warmup and never change.  Replaying a 0-node CUDA graph for them is
  // pure overhead; skip them entirely during dispatch.
  bool allFrozenConstants = false;

  // Resolved backend — set once at buildSegments() time, never changes.
  // Ineligible segments use the explicit SLOT_BY_SLOT plan path. Eligible
  // segments resolve from graphExecutionMode_ at build time.
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

// Segment event logger routed through DSP diagnostics.
// Format: [DSP_EVENT] seg[start-end] EVENT execCount=N shapeKey=compiled/current detail
#define DSP_SEG_EVENT(seg, event, ...) do { \
  DSP_DIAG(EXECUTE, "[DSP_EVENT] seg[%d-%d] %s execCount=%d shapeKey=%s " \
           "compiled=%lld current=%lld", \
           (seg).def.startSlot, (seg).def.endSlot, \
           #event, (seg).exec.executionCount, \
           (seg).def.shapeKeyState.statusName(), \
           (long long)(seg).def.shapeKeyState.compiledShapeKey, \
           (long long)(seg).def.shapeKeyState.lastComputedKey); \
  DSP_DIAG(EXECUTE, __VA_ARGS__); \
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

  // ── Per-unit performance ledger (Part II-G3) ─────────────────────────────
  // Cross-execution accumulated timing per replay unit, keyed by unit index.
  // Written only when the plan's executionTimingEnabled_ is set (zero cost
  // otherwise); merged-group launch+fixup time is attributed to the group's
  // LEADER unit. Feeds the replay-unit auction (G1/G2) and PGO (H1) with
  // MEASURED per-unit microseconds instead of static heuristics, and makes
  // per-unit regression diffs (op-timing-diff technique) automatic.
  struct UnitPerfStats {
    long long execs = 0;
    long long totalUs = 0;
    long long minUs = 0;
    long long maxUs = 0;
    long long lastUs = 0;
    void record(long long us) {
      execs++;
      totalUs += us;
      lastUs = us;
      if (minUs == 0 || us < minUs) minUs = us;
      if (us > maxUs) maxUs = us;
    }
  };
  std::vector<UnitPerfStats> unitPerf;  // sized to units.size() on first record
  void recordUnitPerf(size_t unitIdx, long long us) {
    if (unitPerf.size() < units.size()) unitPerf.resize(units.size());
    if (unitIdx < unitPerf.size()) unitPerf[unitIdx].record(us);
  }
};

// Batch-zero entry used by NativeDynamicShapePlan::batchZeroEntries_.
// outputSlotIndex tracks which slot the pointer came from.
struct BatchZeroEntry { void* ptr; int bytes; int outputSlotIndex; };

/**
 * Mutable execution state — changes per-execution.
 */
struct GraphSegmentExec {
  // ══════════════════════════════════════════════════════════════════════════
  // PRIMARY lifecycle: SegmentPhase (GraphNodePhase + BuildSubPhase)
  // ══════════════════════════════════════════════════════════════════════════
  // This is the SINGLE source of truth for segment lifecycle.
  // All state queries go through segPhase. The old SegmentLifecycleState is
  // derived from segPhase for backward compatibility during migration.
  SegmentPhase segPhase;

  // ══════════════════════════════════════════════════════════════════════════
  // LEGACY: SegmentLifecycleState (derived from segPhase — DO NOT SET DIRECTLY)
  // ══════════════════════════════════════════════════════════════════════════
  // Kept for backward compatibility. Code that reads lifecycleState gets the
  // correct value derived from segPhase. Code that SETS lifecycleState should
  // migrate to segPhase.advanceToCompiling() / segPhase.seal() / etc.
  enum class SegmentLifecycleState : uint8_t {
    NEEDS_WARMUP    = 0,  // First slot-by-slot run to populate shape caches
    NEEDS_COMPILE   = 1,  // Backend compile pass needed (Triton, NVRTC)
    CAPTURE_PENDING = 2,  // Compiled, waiting for CUDA graph capture
    REPLAYING       = 3,  // Graph handles valid — steady-state replay every step
    FAILED          = 4,  // Permanent failure — never attempt again
    OOM_DEFERRED    = 5,  // OOM during capture — deferred retry pending
  };

  // Derived accessor — reads from segPhase, NOT a stored field.
  SegmentLifecycleState getLifecycleState() const {
    return static_cast<SegmentLifecycleState>(segPhase.toLegacyCode());
  }

  // Legacy write-through: sets segPhase from old-style state assignment.
  // Call sites should migrate to segPhase methods, but this keeps old code working.
  void setLifecycleState(SegmentLifecycleState state) {
    switch (state) {
      case SegmentLifecycleState::NEEDS_WARMUP:
        segPhase.reset(); break;
      case SegmentLifecycleState::NEEDS_COMPILE:
        segPhase.phase = GraphNodePhase::BUILDING;
        segPhase.subPhase = BuildSubPhase::COMPILING; break;
      case SegmentLifecycleState::CAPTURE_PENDING:
        segPhase.phase = GraphNodePhase::BUILDING;
        segPhase.subPhase = BuildSubPhase::CAPTURING; break;
      case SegmentLifecycleState::REPLAYING:
        segPhase.phase = GraphNodePhase::SEALED; break;
      case SegmentLifecycleState::FAILED:
        segPhase.fail(); break;
      case SegmentLifecycleState::OOM_DEFERRED:
        segPhase.phase = GraphNodePhase::BUILDING;
        segPhase.subPhase = BuildSubPhase::CAPTURING;
        segPhase.oomRetryPending = true; break;
    }
  }

  // Legacy mirror kept in sync by SegmentLifecycle helpers for diagnostics and
  // old call sites that still inspect SegmentLifecycleState. New state changes
  // should use setLifecycleState() or segPhase methods.
  SegmentLifecycleState lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;

  int executionCount = 0;
  // Count of intentional value-producing warmups performed before a backend
  // commitment. Non-zero is allowed only when the resolved planning policy
  // explicitly requiresPrecommitFunctionalWarmup; it is never an implicit
  // slot-by-slot fallback after commitment.
  int precommitFunctionalWarmupCount = 0;

  // A successful segment-local rebuild performs one bounded functional warmup
  // and then compiles the replacement artifact without executing it again in
  // the same logical invocation. That validated transition contributes one
  // capture-readiness observation on the next call. Keep the credit explicit
  // so configured capture windows above the default are still honored.
 private:
  bool rebuildWarmupCaptureCredit_ = false;

 public:
  void markBoundedRebuildWarmupCaptureReady() {
    rebuildWarmupCaptureCredit_ = true;
    DSP_DIAG(LIFECYCLE,
             "REBUILD_WARMUP_CAPTURE_CREDIT: granted phase=%s exec=%d",
             displayPhaseName(), executionCount);
  }

  bool captureWarmupWindowSatisfied(int requiredExecutions) const {
    const int effectiveExecutions =
        executionCount + (rebuildWarmupCaptureCredit_ ? 1 : 0);
    return effectiveExecutions >= requiredExecutions;
  }

  bool hasBoundedRebuildWarmupCaptureCredit() const {
    return rebuildWarmupCaptureCredit_;
  }

  void clearBoundedRebuildWarmupCaptureReady(const char* reason) {
    if (rebuildWarmupCaptureCredit_) {
      DSP_DIAG(LIFECYCLE,
               "REBUILD_WARMUP_CAPTURE_CREDIT: cleared reason=%s phase=%s exec=%d",
               reason ? reason : "?", displayPhaseName(), executionCount);
    }
    rebuildWarmupCaptureCredit_ = false;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // OUTCOME: Why this segment executes the way it does
  // ══════════════════════════════════════════════════════════════════════════
  // Single source of truth for dispatch decisions. Replaces the scattered
  // boolean flags below (kept during migration for backward compat).
  // dispatchSegment() reads (selectedBackend, segPhase, outcome) to pick
  // exactly one execution action.
  SegmentExecOutcome outcome = SegmentExecOutcome::PENDING;

  // ── Legacy boolean flags (read from outcome, write both during migration) ──
  // These will be removed once all callers migrate to outcome.

  // Derived from segPhase.isFailed() — kept for backward compat reads.
  // Migration: use segPhase.isFailed() directly.
  bool compilationFailed = false;

  // If true, no backend could fuse this segment (all permutes/reshapes/identity).
  // The segment executes slot-by-slot every step without re-attempting the cascade.
  // Unlike compilationFailed, this is expected behavior — not an error.
  bool noFusibleOps = false;

  // True after a capture attempt produced a 0-node CUDA graph. This means the
  // segment's ops don't generate any GPU kernels (views, shapes, identity) even
  // though Triton compilation succeeded. Re-capturing is wasteful — execute
  // slot-by-slot instead. Set by ZERO_NODE_REJECT in the capture path.
  bool captureProducedNoKernels = false;

  // Why this segment reached a terminal outcome (ZERO_KERNEL_SBS, NOT_FUSIBLE,
  // COMPILE_FAILED, etc.). Set by SegmentLifecycle:: methods at transition time.
  // Persists across ring buffer overwrites — always available in diagnostic JSON.
  // nullptr while outcome is PENDING or GRAPH_REPLAY.
  const char* terminalReason = nullptr;

  // OOM retry mechanism — now derived from segPhase.oomRetryCount/oomRetryAfterExec.
  // Kept as fields during migration; will be removed when all callers use segPhase.
  int captureOomRetries = 0;
  int captureRetryAfterExec = 0;

  // Largest CUDA-pool allocation requested by a pre-capture execution. Graph
  // instantiation must leave at least this much execution headroom available,
  // otherwise the compiled backend is preserved and graph capture is deferred.
  size_t peakWarmupAllocationBytes = 0;

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

  // ── Arg table generation counter ────────────────────────────────────────────
  // Replaces the fragile argTableStable boolean. The generation counter is
  // bumped whenever ANY address (external input or slot output) changes.
  // The fast-replay path checks needsArgRefresh(): if generation matches
  // the captured generation, no refresh is needed — correct by construction.
  uint64_t argTableGeneration = 1;      // starts at 1 so initial state requires refresh
  uint64_t capturedArgGeneration = 0;   // recorded when arg table was last synced

  bool needsArgRefresh() const { return argTableGeneration != capturedArgGeneration; }
  void markArgsCurrent() { capturedArgGeneration = argTableGeneration; }
  void bumpArgGeneration() { argTableGeneration++; }

  // Inverse of markArgsCurrent(): an external input address, slot output address,
  // refreshed view wrapper, or shape changed — so the cached arg table and any
  // captured graph node arguments are stale. Forces needsArgRefresh()==true and
  // resets the stability telemetry. SINGLE source of truth for the bump+reset
  // triplet that was duplicated at ~18 call sites across
  // segments/cuda/gpubackend/lifecycle (several of which forgot the resets, or
  // the whole triplet, leaving stale telemetry or a stale graph on replay).
  void markArgsStale() {
    bumpArgGeneration();
    addrKeyStableCount = 0;
    slotAddrStableCount = 0;
    DSP_DIAG(EXECUTE, "MARK_ARGS_STALE: argGen=%llu phase=%s exec=%d",
             (unsigned long long)argTableGeneration, displayPhaseName(), executionCount);
  }

  // Clear the four captured-graph IDENTITY keys: shape, external-input addresses,
  // create (ConstantOfShape) values, and slot-output addresses. Call when
  // invalidating/evicting a capture so the next warmup re-establishes them from
  // scratch. Leaving ANY of them stale lets a drift check (slotAddrDrifted /
  // needsArgRefresh) compare against a dead capture. SINGLE source of truth for
  // the captured-key reset duplicated at ~8 sites — five of which silently
  // OMITTED capturedSlotAddrHash (evictSegmentCapture, the cuda.cu resource
  // release, and the three NativeDynamicShapePlan.cpp warmup/freeze resets) →
  // a stale slot-addr hash surviving into re-warmup → spurious drift.
  void resetCaptureKeys() {
    cachedShapeKey = 0;
    capturedInputAddrKey = 0;
    capturedCreateValueKey = 0;
    capturedSlotAddrHash = 0;
    clearBoundedRebuildWarmupCaptureReady("reset_capture_keys");
    DSP_DIAG(LIFECYCLE,
             "RESET_CAPTURE_KEYS: cleared shape/inputAddr/createValue/slotAddr/rebuildWarmupCredit phase=%s exec=%d",
             displayPhaseName(), executionCount);
  }

  void recordReplayBaselineKeys(LongType shapeKey, LongType inputAddrKey, const char* reason) {
    DSP_DIAG(LIFECYCLE,
             "REPLAY_BASELINE_KEYS: reason=%s phase=%s exec=%d "
             "shapeKey %lld -> %lld inputAddrKey %lld -> %lld",
             reason ? reason : "?", displayPhaseName(), executionCount,
             (long long)cachedShapeKey, (long long)shapeKey,
             (long long)capturedInputAddrKey, (long long)inputAddrKey);
    cachedShapeKey = shapeKey;
    capturedInputAddrKey = inputAddrKey;
  }

  void recordReplayInputAddrKey(LongType inputAddrKey, const char* reason) {
    DSP_DIAG(LIFECYCLE,
             "REPLAY_INPUT_ADDR_KEY: reason=%s phase=%s exec=%d inputAddrKey %lld -> %lld",
             reason ? reason : "?", displayPhaseName(), executionCount,
             (long long)capturedInputAddrKey, (long long)inputAddrKey);
    capturedInputAddrKey = inputAddrKey;
  }

  void recordReplayAddressKeys(LongType inputAddrKey, LongType slotAddrHash, const char* reason) {
    DSP_DIAG(LIFECYCLE,
             "REPLAY_ADDRESS_KEYS: reason=%s phase=%s exec=%d "
             "inputAddrKey %lld -> %lld slotAddrHash %lld -> %lld",
             reason ? reason : "?", displayPhaseName(), executionCount,
             (long long)capturedInputAddrKey, (long long)inputAddrKey,
             (long long)capturedSlotAddrHash, (long long)slotAddrHash);
    capturedInputAddrKey = inputAddrKey;
    capturedSlotAddrHash = slotAddrHash;
  }

  void clearGraphContentFlags(const char* reason) {
    DSP_DIAG(LIFECYCLE,
             "CLEAR_GRAPH_CONTENT_FLAGS: reason=%s phase=%s exec=%d "
             "gapsCaptured %d -> 0 createOpsExcluded %d -> 0",
             reason ? reason : "?", displayPhaseName(), executionCount,
             gapOpsCapturedInGraph ? 1 : 0,
             createOpsExcludedFromGraph ? 1 : 0);
    gapOpsCapturedInGraph = false;
    createOpsExcludedFromGraph = false;
  }

  void markCreateOpsExcludedFromGraph(bool excluded, const char* reason, int skippedCount = 0) {
    DSP_DIAG(LIFECYCLE,
             "CREATE_OPS_EXCLUDED_FROM_GRAPH: reason=%s phase=%s exec=%d "
             "skipped=%d createOpsExcluded %d -> %d",
             reason ? reason : "?", displayPhaseName(), executionCount,
             skippedCount, createOpsExcludedFromGraph ? 1 : 0,
             excluded ? 1 : 0);
    createOpsExcludedFromGraph = excluded;
  }

  // Reset a segment to its WARMUP baseline for re-capture: zero the execution /
  // OOM-retry counters and the capture-identity keys (resetCaptureKeys), and clear
  // the graph-content flags. Does NOT touch lifecycle phase/outcome (the caller
  // drives those) or the replay handle (the caller conditionally cleans that up).
  // Consolidates the two identical per-segment warmup-reset loops (execute-warmup
  // + phaseWarmup) in NativeDynamicShapePlan.cpp.
  void resetForWarmup() {
    executionCount = 0;
    precommitFunctionalWarmupCount = 0;
    captureOomRetries = 0;
    captureRetryAfterExec = 0;
    resetCaptureKeys();
    clearGraphContentFlags("reset_for_warmup");
    DSP_DIAG(LIFECYCLE, "RESET_FOR_WARMUP: counters+keys+graphflags reset phase=%s",
             displayPhaseName());
  }

  // ── Replay-readiness decision (encapsulated, single source of truth) ──────────────
  // "Are this segment's external addresses / baked values / shape stable enough to replay
  // the captured graph this step?" Computed once per step by recordReplayStability(),
  // recorded here, and consumed by the arg-generation tracking AND the plan-phase gate —
  // instead of recomputing the answer independently in each path with a different address
  // source (the desync that blocked REPLAYING). Also makes the decision OBSERVABLE
  // post-decode via getPlanSegmentStatisticsJson (DSP_DIAG is silent in the native op).
  struct ReplayStability {
    // Populate the decision in ONE call (no external field poking). Inputs are the per-step
    // stability sub-observations the caller already computed from the segment's captured keys
    // and current inputs.
    void record(bool extAddrsStable, bool createValuesStable, bool shapeKeyStable,
                bool hasValueShapeInputs, int gapSlotCount) {
      extAddrsStable_      = extAddrsStable;
      createValuesStable_  = createValuesStable;
      shapeKeyStable_      = shapeKeyStable;
      hasValueShapeInputs_ = hasValueShapeInputs;
      gapSlotCount_        = gapSlotCount;
      valid_               = true;
    }

    // The verdict: a segment with no value-shape inputs can't go stale on value/addr churn;
    // otherwise all three invariants must hold.
    bool isStable() const {
      return valid_ && (!hasValueShapeInputs_ ||
                        (extAddrsStable_ && createValuesStable_ && shapeKeyStable_));
    }

    // Read-only accessors (diagnostics / getPlanSegmentStatisticsJson).
    bool valid()               const { return valid_; }
    bool extAddrsStable()      const { return extAddrsStable_; }
    bool createValuesStable()  const { return createValuesStable_; }
    bool shapeKeyStable()      const { return shapeKeyStable_; }
    bool hasValueShapeInputs() const { return hasValueShapeInputs_; }
    int  gapSlotCount()        const { return gapSlotCount_; }

   private:
    bool extAddrsStable_      = false;  // graph's external device addresses unchanged (staging-aware)
    bool createValuesStable_  = false;  // baked ConstantOfShape values unchanged (or live as gaps)
    bool shapeKeyStable_      = false;  // segment shape unchanged since capture
    bool hasValueShapeInputs_ = false;  // has value-dependent-shape ops (gather, ...) with internal producers
    int  gapSlotCount_        = 0;      // cuBLAS gap slots; >0 while monolithic ⇒ composite capture fell back
    bool valid_               = false;  // computed at least once this run
  } replayStability;

  // argTableStable removed — use needsArgRefresh() / markArgsCurrent() instead.
  // All callers migrated to the generation counter.

  // Consecutive-stable pass counters for diagnostic telemetry only — the
  // checks themselves only run when DSP_DIAG VERIFY is enabled.
  int addrKeyStableCount = 0;
  int slotAddrStableCount = 0;

  // True when native-only monolithic capture included gap ops (cuBLAS matmul etc.)
  // in the CUDA graph. The frozen fast path checks this to know monolithic replay
  // covers ALL ops — no live gap execution needed.
  bool gapOpsCapturedInGraph = false;

  // True when native-only monolithic capture EXCLUDED value-shape create ops
  // (ops with CONSTANT_GENERATION + VALUE_DEPENDENT_SHAPE traits, e.g. range/create)
  // because their inputs change per decode step (position IDs, sequence lengths).
  // During monolithic replay, these ops are executed live BEFORE cudaGraphLaunch
  // so they produce fresh outputs (correct step position) that the graph reads via
  // stable slot pointers.  The createValuesStable invalidation path is bypassed
  // when this flag is true — value changes are handled naturally by live execution.
  bool createOpsExcludedFromGraph = false;

  // ── Capture seal: consolidated state update at capture completion ──────
  // Sets all capture-related fields atomically. Called from SegmentLifecycle::markCaptured.
  void sealCapture(LongType inputAddrKey, LongType createValueKey,
                   LongType slotAddrHash, const char* backendName,
                   bool gapsCaptured) {
    const LongType prevInputAddrKey = capturedInputAddrKey;
    const LongType prevCreateValueKey = capturedCreateValueKey;
    const LongType prevSlotAddrHash = capturedSlotAddrHash;
    const bool prevGapsCaptured = gapOpsCapturedInGraph;
    const bool backendWasEmpty = compiledByBackend.empty();
    capturedInputAddrKey = inputAddrKey;
    capturedCreateValueKey = createValueKey;
    capturedSlotAddrHash = slotAddrHash;
    gapOpsCapturedInGraph = gapsCaptured;
    if (compiledByBackend.empty()) compiledByBackend = backendName;
    DSP_DIAG(LIFECYCLE,
             "SEAL_CAPTURE: phase=%s exec=%d backend=%s backendSet=%d "
             "inputAddrKey %lld -> %lld createValueKey %lld -> %lld "
             "slotAddrHash %lld -> %lld gapsCaptured %d -> %d",
             displayPhaseName(), executionCount,
             backendName ? backendName : "?",
             backendWasEmpty ? 1 : 0,
             (long long)prevInputAddrKey, (long long)capturedInputAddrKey,
             (long long)prevCreateValueKey, (long long)capturedCreateValueKey,
             (long long)prevSlotAddrHash, (long long)capturedSlotAddrHash,
             prevGapsCaptured ? 1 : 0, gapOpsCapturedInGraph ? 1 : 0);
  }

  // Query: does the monolithic graph include gap ops?
  bool hasGapsInGraph() const { return gapOpsCapturedInGraph; }

  // Have this segment's slot-output device addresses drifted from capture?
  // A captured monolithic/composite graph bakes slot specialBuffer() pointers
  // into its nodes; if any slot's address changed since sealCapture(), replaying
  // the graph would dereference stale device pointers (CUDA error 700).
  // capturedSlotAddrHash==0 means "never captured" → no drift possible.
  // SINGLE source of truth for the drift check duplicated in the frozen fast
  // path and the normal cudagraph replay path. The caller computes the current
  // hash (needs plan-level outputSlots_) and passes it here.
  bool slotAddrDrifted(LongType currentSlotAddrHash) const {
    bool drifted = capturedSlotAddrHash != 0 && currentSlotAddrHash != capturedSlotAddrHash;
    if (drifted) {
      DSP_DIAG(MEMORY, "SLOT_ADDR_DRIFTED: captured=0x%llx current=0x%llx phase=%s",
               (unsigned long long)capturedSlotAddrHash,
               (unsigned long long)currentSlotAddrHash, displayPhaseName());
    }
    return drifted;
  }

  // ── Replay handle mutation tracker ──────────────────────────────────────────
  // Records every create/capture/replay/invalidate/destroy event on this
  // segment's replay handle. Ring buffer of 128 events + aggregate counters.
  // Gated behind DSP_DIAG_GRAPH_REPLAY category when recording via lifecycle
  // methods. dump() and toJsonSummary() always work (read-only).
  ReplayHandleTracker handleTracker;

  // View recipe chain — captures view-producing ops (reshape, permute,
  // expand_dims, squeeze, strided_slice) so they can be installed during
  // REPLAYING without launching a kernel or executing a native ordered range.
  // Populated during SHAPES_FROZEN, validated during convergence,
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

  // ── Phase queries: delegate to segPhase ──���───────────────────────────────
  // segPhase is the SINGLE source of truth. These are convenience accessors.

  const char* displayPhaseName() const { return segPhase.displayName(); }

  GraphNodePhase graphNodePhase() const { return segPhase.phase; }

  // JNI-compatible integer encoding matching the old ExecutionPhase values:
  //   0=WARMUP, 1=COMPILING, 2=COMPILED, 3=REPLAYING, 4=SLOT_BY_SLOT, 5=OOM_DEFERRED
  int getExecutionPhaseCode() const { return segPhase.toLegacyCode(); }

  void reset() {
    DSP_DIAG(LIFECYCLE,
             "SEGMENT_EXEC_RESET: before phase=%s exec=%d backend=%s "
             "inputAddrKey=%lld createValueKey=%lld slotAddrHash=%lld "
             "gapsCaptured=%d createOpsExcluded=%d",
             displayPhaseName(), executionCount,
             compiledByBackend.empty() ? "" : compiledByBackend.c_str(),
             (long long)capturedInputAddrKey,
             (long long)capturedCreateValueKey,
             (long long)capturedSlotAddrHash,
             gapOpsCapturedInGraph ? 1 : 0,
             createOpsExcludedFromGraph ? 1 : 0);
    // Primary lifecycle reset
    segPhase.reset();
    // Legacy field sync
    lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;
    executionCount = 0;
    precommitFunctionalWarmupCount = 0;
    compilationFailed = false;
    captureProducedNoKernels = false;
    noFusibleOps = false;
    terminalReason = nullptr;
    captureOomRetries = 0;
    captureRetryAfterExec = 0;
    lastReplayExecCount = 0;
    replayHandle.reset();
    outcome = SegmentExecOutcome::PENDING;
    resetCaptureKeys();
#ifdef SD_CUDA
    jitKernel = nullptr;
    jitShapeKey = 0;
    jitCompileFailed = false;
#endif
    symbolicShapeEnabled = false;
    symbolicWarmupRemaining = 0;
    symbolicRangeData = nullptr;
    compiledByBackend.clear();
    argTableGeneration = 1;
    capturedArgGeneration = 0;
    addrKeyStableCount = 0;
    slotAddrStableCount = 0;
    clearGraphContentFlags("segment_reset");
    handleTracker.reset();
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

  // Concrete implementation selected by the shared resolver and successfully
  // lowered for this segment. Subsequent executions reuse it directly. The
  // exact backend lifecycle policy is captured at the same transition so generic
  // plan logic never infers mechanics from platform or backend names.
  GraphBackend* resolvedGraphBackend = nullptr;
  GraphBackendPlanningPolicy resolvedGraphBackendPolicy;

  // A compiled backend artifact belongs to this exact plan segment. Backends
  // that are process-wide singletons must not own compiled graphs in a cache
  // keyed only by slot range and shape: two plans can have identical ranges and
  // shapes while representing different models. Keeping the strong reference
  // here makes the artifact's lifetime identical to the segment's lifetime.
  GraphBackend* compiledGraphBackendArtifactOwner = nullptr;
  LongType compiledGraphBackendArtifactShapeKey = 0;
  std::shared_ptr<void> compiledGraphBackendArtifact;
  // Host/device bytes retained exclusively by the opaque backend artifact.
  // The generic plan cache cannot inspect std::shared_ptr<void>, so the owning
  // backend records the footprint when publishing the artifact.
  size_t compiledGraphBackendArtifactOwnedBytes = 0;
  std::vector<CompilationAuditEntry> compilationAudit;

  // Pointer to NativeDynamicShapePlan slot array cache — allows GPU backends
  // to update the slot cache when pre-allocating output arrays.
  NDArray** slotArrayCache = nullptr;

  // Runtime-configurable OOM retry constants (read from Environment)
  static int maxOomRetries();
  static int retryInterval();

  void setResolvedGraphBackend(GraphBackend* backend,
                               const GraphBackendRequest& request) {
    resolvedGraphBackend = backend;
    resolvedGraphBackendPolicy =
        backend != nullptr ? backend->planningPolicy(request)
                           : GraphBackendPlanningPolicy{};
  }

  void setCompiledGraphBackendArtifact(
      GraphBackend* owner, LongType shapeKey, std::shared_ptr<void> artifact,
      size_t ownedBytes = 0) {
    compiledGraphBackendArtifactOwner = owner;
    compiledGraphBackendArtifactShapeKey = shapeKey;
    compiledGraphBackendArtifact = std::move(artifact);
    compiledGraphBackendArtifactOwnedBytes = ownedBytes;
  }

  void clearCompiledGraphBackendArtifact() {
    compiledGraphBackendArtifact.reset();
    compiledGraphBackendArtifactOwner = nullptr;
    compiledGraphBackendArtifactShapeKey = 0;
    compiledGraphBackendArtifactOwnedBytes = 0;
    compilationAudit.clear();
  }

  // Reset backend identity, policy, and its plan-owned compilation together.
  void resetGraphBackend() {
    clearCompiledGraphBackendArtifact();
    resolvedGraphBackend = nullptr;
    resolvedGraphBackendPolicy = GraphBackendPlanningPolicy{};
    // A compile key without its owning backend/artifact is not reusable. Keeping
    // it would let same-shape invalidation skip lowering and execute a missing
    // artifact on the next dispatch.
    def.shapeKeyState.reset();
  }

  // Unified lifecycle accessor — delegates to exec.graphNodePhase().
  GraphNodePhase graphNodePhase() const { return exec.graphNodePhase(); }

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
      const std::vector<std::string>& requestedOutputs,
      GraphExecutionMode mode = GraphExecutionMode::GEM_AUTO,
      std::string* errorMessage = nullptr,
      const NativePlanCompileOptions& compileOptions =
          NativePlanCompileOptions());

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
   *   - planLifecycle_.isReplaying() (plan is in steady-state graph replay)
   *   - !planLifecycle_.isSlotBySlot() (shapes are frozen)
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
   * Called automatically by execute() when planLifecycle_.isShapesFrozen() && executeCount_ == 0.
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
   * if merge is enabled, resets execution counters, and advances planLifecycle_
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
   * Called automatically by execute() when planLifecycle_.isReplaying().
   * Also called during SHAPES_FROZEN when segments are compiled but not yet
   * fully replaying (transitional state).
   *
   * @return Status::OK on success, error on segment execution failure.
   */
  Status phaseReplay(NDArray** externalInputs, int numExternalInputs,
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
   *   SHAPES_FROZEN → REPLAYING (after 2+ frozen executions with all segments
   *                              pointer-stable AND in replay steady state)
   */
  void advancePlanPhase();

  /**
   * Demote plan phase (manual override for error recovery).
   * Used when segment drops out of replay steady state.
   */
  void demotePlanPhase(PlanPhase targetPhase, const char* reason);

  // ── Sync policy (contract-driven) ──────────────────────────────────────
  // Whether executeSlot should call prepareSpecialUse/registerSpecialUse.
  // Computed from ModeContract + plan lifecycle + scoped overrides.
  // Replaces the old mutable forceSync_ boolean.
  bool needsSync() const {
    // Always sync during warmup (device buffers not yet stable)
    if (executeCount_ < 2) return true;
    // Always sync in slot-by-slot mode (no capture optimization)
    if (planLifecycle_.isSlotBySlot()) return true;
    // Scoped override (gpu backend gap execution, pre-capture warmup)
    if (syncOverrideDepth_ > 0) return true;
    // Contract: mode requires sync on every frozen execution
    auto contract = ModeContract::forMode(graphExecutionMode_);
    if (contract.forcesSyncOnFrozen && planLifecycle_.isShapesFrozen()) return true;
    return false;
  }

  // Returns a human-readable reason string for WHY needsSync() returned its value.
  // Used by DSP_DIAG logging at call sites so logs show the deciding factor.
  const char* syncReason() const {
    if (executeCount_ < 2) return "WARMUP(execCount<2)";
    if (planLifecycle_.isSlotBySlot()) return "SLOT_BY_SLOT";
    if (syncOverrideDepth_ > 0) return "SCOPED_OVERRIDE";
    auto contract = ModeContract::forMode(graphExecutionMode_);
    if (contract.forcesSyncOnFrozen && planLifecycle_.isShapesFrozen()) return "CONTRACT(forcesSyncOnFrozen)";
    return "NONE(sync_skipped)";
  }

  // RAII guard for scoped sync override. GPU backend gap execution and
  // pre-capture warmup use this to force sync within a bracket.
  // Logs entry/exit via DSP_DIAG(STREAM_SYNC) when diagnostics are enabled.
  struct SyncOverride {
    NativeDynamicShapePlan& plan_;
    const char* context_;  // caller label for logging
    SyncOverride(NativeDynamicShapePlan& plan, const char* context = "unknown")
        : plan_(plan), context_(context) {
      plan_.syncOverrideDepth_++;
      DSP_DIAG(STREAM_SYNC, "SyncOverride ENTER: context=%s depth=%d->%d execCount=%d mode=%s",
               context_, plan_.syncOverrideDepth_ - 1, plan_.syncOverrideDepth_,
               plan_.executeCount_, ModeContract::modeName(static_cast<int>(plan_.graphExecutionMode_)));
    }
    ~SyncOverride() {
      plan_.syncOverrideDepth_--;
      DSP_DIAG(STREAM_SYNC, "SyncOverride EXIT: context=%s depth=%d->%d",
               context_, plan_.syncOverrideDepth_ + 1, plan_.syncOverrideDepth_);
    }
    SyncOverride(const SyncOverride&) = delete;
    SyncOverride& operator=(const SyncOverride&) = delete;
  };

  // ── Shape change warmup guard (contract-driven) ──────────────────────
  // RAII guard that sets inShapeChangeWarmup_ for the duration of a
  // shape-change warmup pass. Replaces the bracket pattern:
  //   inShapeChangeWarmup_ = true; ... inShapeChangeWarmup_ = false;
  struct ShapeChangeWarmupGuard {
    NativeDynamicShapePlan& plan_;
    ShapeChangeWarmupGuard(NativeDynamicShapePlan& plan, int segStart, int segEnd)
        : plan_(plan) {
      plan_.inShapeChangeWarmup_ = true;
      DSP_DIAG(SHAPE, "ShapeChangeWarmup ENTER: seg[%d-%d] execCount=%d",
               segStart, segEnd, plan_.executeCount_);
    }
    ~ShapeChangeWarmupGuard() {
      plan_.inShapeChangeWarmup_ = false;
      DSP_DIAG(SHAPE, "ShapeChangeWarmup EXIT");
    }
    ShapeChangeWarmupGuard(const ShapeChangeWarmupGuard&) = delete;
    ShapeChangeWarmupGuard& operator=(const ShapeChangeWarmupGuard&) = delete;
  };

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
   * Passivate this plan: release GPU intermediates to the buffer pool.
   * The plan stays structurally valid (segments, wiring, definitions)
   * but holds no GPU memory. On next cache hit, the execute path
   * re-warms automatically (allocates buffers, recomputes coloring).
   *
   * @return bytes freed (approximate)
   */
  size_t passivate();

  /**
   * Clear the passivated flag. Called on cache hit to mark the plan
   * as needing re-warmup (the execute path handles allocation).
   */
  void reactivate();

  /** True if this plan was passivated and has not yet been re-warmed. */
  bool isPassivated() const;

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

  // ── External input introspection (for assertions and diagnostics) ──────

  /** True if ext[extIdx] was marked variable (participates in staging D2D). */
  bool isExternalInputVariable(int extIdx) const {
    if (extIdx < 0 || extIdx >= static_cast<int>(externalInputIsVariable_.size())) return false;
    return externalInputIsVariable_[extIdx];
  }

  /** True if ext[extIdx] was marked placeholder (forces H2D sync). */
  bool isExternalInputPlaceholder(int extIdx) const {
    if (extIdx < 0 || extIdx >= static_cast<int>(externalInputIsPlaceholder_.size())) return false;
    return externalInputIsPlaceholder_[extIdx];
  }

  /** Count of external inputs currently classified as variable. */
  int getNumVariableExternalInputs() const {
    int count = 0;
    for (int i = 0; i < static_cast<int>(externalInputIsVariable_.size()); i++) {
      if (externalInputIsVariable_[i]) count++;
    }
    return count;
  }

  /** Device address of the current-device plan-owned staging buffer for ext[extIdx], or 0 if none. */
  long long getStagingBufferAddress(int extIdx) const {
    NDArray** stagingBuffers = activeStagingBuffers_ != nullptr
        ? activeStagingBuffers_ : placeholderStagingBuffers_;
    if (stagingBuffers == nullptr || extIdx < 0 || extIdx >= numExternalInputs_)
      return 0;
    NDArray* staging = stagingBuffers[extIdx];
    if (staging == nullptr) return 0;
    return reinterpret_cast<long long>(staging->specialBuffer());
  }

  /** Device address the CUDA graph will actually read from for ext[extIdx].
   *  Returns staging address if variable with staging, else the original ext address, else 0. */
  long long getEffectiveExternalAddress(int extIdx) const {
    if (effectiveExternals_ != nullptr && extIdx >= 0 && extIdx < numExternalInputs_) {
      NDArray* eff = effectiveExternals_[extIdx];
      if (eff != nullptr) return reinterpret_cast<long long>(eff->specialBuffer());
    }
    NDArray** stagingBuffers = activeStagingBuffers_ != nullptr
        ? activeStagingBuffers_ : placeholderStagingBuffers_;
    if (stagingBuffers != nullptr && extIdx >= 0 && extIdx < numExternalInputs_) {
      NDArray* staging = stagingBuffers[extIdx];
      if (staging != nullptr) return reinterpret_cast<long long>(staging->specialBuffer());
    }
    return 0;
  }

  /** Number of variable ext inputs with allocated staging buffers. */
  int getNumStagingBuffers() const {
    NDArray** stagingBuffers = activeStagingBuffers_ != nullptr
        ? activeStagingBuffers_ : placeholderStagingBuffers_;
    if (stagingBuffers == nullptr) return 0;
    int count = 0;
    for (int i = 0; i < numExternalInputs_; i++) {
      if (stagingBuffers[i] != nullptr) count++;
    }
    return count;
  }

  /** Name of ext input at index, or empty string. */
  const std::string& getExternalInputName(int extIdx) const {
    static const std::string empty;
    if (extIdx < 0 || extIdx >= static_cast<int>(externalInputNames_.size())) return empty;
    return externalInputNames_[extIdx];
  }

  /** Current plan execution count. */
  int getExecuteCount() const { return executeCount_; }

  /**
   * Execution count relative to the current diagnostic epoch. A diagnostics
   * clear starts a new bounded capture without mutating this plan's lifecycle.
   */
  int diagnosticExecuteCount() {
    auto& diagnostics = DspDiagnostics::getInstance();
    const uint64_t currentEpoch = diagnostics.epoch();
    if (diagnosticEpoch_ != currentEpoch ||
        executeCount_ < diagnosticEpochBaseExecuteCount_) {
      diagnosticEpoch_ = currentEpoch;
      diagnosticEpochBaseExecuteCount_ = executeCount_;
      diagnostics.beginPlanExecution(
          numSlots_, static_cast<int>(segments_.size()));
    }
    return executeCount_ - diagnosticEpochBaseExecuteCount_;
  }

  /** Number of cached variable ext input indices (fast-path list). */
  int getNumCachedVariableExtIndices() const {
    return static_cast<int>(cachedVariableExtIndices_.size());
  }

  /** Get the i-th cached variable ext input index, or -1 if out of range. */
  int getCachedVariableExtIndex(int i) const {
    if (i < 0 || i >= static_cast<int>(cachedVariableExtIndices_.size())) return -1;
    return cachedVariableExtIndices_[i];
  }

  /** Get the staging NDArray* for ext[extIdx], or nullptr if none. For JNI introspection.
   *  Returns nullptr if the staging buffer's shapeInfo is corrupted or DataBuffer is
   *  closed/invalid (prevents Java crash from stale pointers after segment invalidation). */
  NDArray* getStagingBufferArray(int extIdx) const {
    if (placeholderStagingBuffers_ == nullptr || extIdx < 0 || extIdx >= numExternalInputs_)
      return nullptr;
    NDArray* staging = placeholderStagingBuffers_[extIdx];
    if (staging == nullptr) return nullptr;
    // Validate DataBuffer is alive — after markExternalInputVariable, staging buffers
    // are freed and reallocated. A closed DataBuffer means the specialBuffer() pointer
    // is stale and reading it would cause an illegal memory access (error 700/719).
    auto* db = staging->dataBuffer();
    if (db == nullptr || db->isClosed()) return nullptr;
    if (staging->specialBuffer() == nullptr) return nullptr;
    // Validate shapeInfo — null shapeInfo means the staging NDArray was freed/poisoned.
    // Return nullptr here to prevent Java crash (Java would call getOpaqueNDArrayShapeInfo
    // which dereferences the NDArray's shapeInfo_ field).
    if (staging->shapeInfo() == nullptr) return nullptr;
    {
      LongType rank = staging->shapeInfo()[0];
      if (rank < 0 || rank > SD_MAX_RANK) {
        DSP_DIAG(MEMORY,
                 "getStagingBufferArray: SHAPE CORRUPTION ext[%d] rank=%lld (0x%llx) "
                 "shapeInfo=%p staging=%p. Returning nullptr to prevent Java crash.",
                 extIdx, (long long)rank, (unsigned long long)rank,
                 (void*)staging->shapeInfo(), (void*)staging);
        return nullptr;
      }
    }
    return staging;
  }

  /**
   * Atomically copy staging buffer content for ext[extIdx] into dstDataBuffer.
   * This avoids the stale-pointer race of extracting specialBuffer() then copying separately.
   * Returns: 0 = success, -1 = no staging, -2 = invalid staging, -3 = copy failed.
   */
  int copyStagingToBuffer(int extIdx, sd::DataBuffer* dstDataBuffer);

  /** Get the last external input NDArray* at index, or nullptr. Stable after execute(). */
  NDArray* getLastExternalInput(int extIdx) const {
    if (lastExternalInputs_ == nullptr || extIdx < 0 || extIdx >= lastNumExternalInputs_)
      return nullptr;
    return lastExternalInputs_[extIdx];
  }

  /**
   * Get the number of requested outputs.
   */
  int getNumRequestedOutputs() const { return numRequestedOutputs_; }

  /**
   * Resolve the output-vector arity for a NativeOps execution adapter.
   *
   * A zero bound-output count is the runtime-allocation contract used by
   * sdxRunAllocating: the plan owns output allocation, so the adapter must pass
   * one null slot per requested output. A nonzero count is caller-buffer mode
   * and must match the serialized plan exactly. Returns -1 on mismatch.
   */
  int resolveExecutionOutputCount(int boundOutputCount) const {
    return boundOutputCount == 0 || boundOutputCount == numRequestedOutputs_
        ? numRequestedOutputs_ : -1;
  }

  /**
   * Get the total number of slots (ops) in the plan.
   */
  int getNumSlots() const { return numSlots_; }

  /** Whether the most recent shape-only pass resolved every live slot output. */
  bool isShapePrePassComplete() const { return shapePrePassComplete_; }
  int shapePrePassFirstIncompleteSlot() const {
    return shapePrePassFirstIncompleteSlot_;
  }
  const std::string& shapePrePassIncompleteReason() const {
    return shapePrePassIncompleteReason_;
  }

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
   * Estimate memory retained by this plan: owned intermediate arrays,
   * backend-owned compiled artifacts, replay-handle workspaces, and CUDA
   * shared-capture/cuBLAS workspaces.
   * NativePlanCache uses this value for its device-memory budget, so every
   * plan-lifetime allocation must be represented here.
   */
  size_t estimatedOwnedBytes() const {
    size_t total = 0;
    std::vector<std::pair<uintptr_t, size_t>> captureWorkspaceRanges;
    auto addReplayWorkspace = [&total, &captureWorkspaceRanges](
                                  const auto& handle) {
      if (handle == nullptr || handle->getWorkspacePtr() == nullptr ||
          handle->getWorkspaceBytes() == 0) {
        return;
      }
      captureWorkspaceRanges.emplace_back(
          reinterpret_cast<uintptr_t>(handle->getWorkspacePtr()),
          handle->getWorkspaceBytes());
      if (!handle->isWorkspaceExternal()) {
        total += handle->getWorkspaceBytes();
      }
    };
    for (const auto& segment : segments_) {
      total += segment.compiledGraphBackendArtifactOwnedBytes;
      addReplayWorkspace(segment.exec.replayHandle);
      for (const auto& handle :
           segment.exec.compositeReplaySchedule.mergedReplayHandles) {
        addReplayWorkspace(handle);
      }
      for (const auto& handle :
           segment.exec.compositeReplaySchedule.compositeReplayHandles) {
        addReplayWorkspace(handle);
      }
    }
#ifdef SD_CUDA
    if (sharedCaptureWorkspace_ != nullptr) {
      captureWorkspaceRanges.emplace_back(
          reinterpret_cast<uintptr_t>(sharedCaptureWorkspace_),
          sharedCaptureWorkspaceBytes_);
      total += sharedCaptureWorkspaceBytes_;
    }
    if (cublasWorkspaceBuffer_ != nullptr) {
      total += cublasWorkspaceSize_;
    }
#endif

    // Count every unique plan-owned DataBuffer once. Capture-workspace interior
    // pointers are already represented by their arena above and must not be
    // charged again. Include staging and untracked caches in addition to slot
    // outputs so the plan-cache budget reflects the complete retained footprint.
    std::unordered_set<DataBuffer*> countedBuffers;
    auto addArray = [&total, &captureWorkspaceRanges, &countedBuffers](
                        NDArray* arr) {
      if (arr == nullptr) return;
      DataBuffer* db = arr->dataBuffer();
      if (db == nullptr || !db->isValid() || !countedBuffers.insert(db).second) {
        return;
      }
      const uintptr_t special = reinterpret_cast<uintptr_t>(db->special());
      for (const auto& range : captureWorkspaceRanges) {
        if (special >= range.first && special - range.first < range.second) {
          return;
        }
      }
      total += static_cast<size_t>(arr->memoryFootprint());
    };
    for (NDArray* arr : planOwnedArrays_) addArray(arr);
    if (placeholderStagingBuffers_ != nullptr) {
      for (int i = 0; i < numExternalInputs_; ++i) {
        addArray(placeholderStagingBuffers_[i]);
      }
    }
    for (const auto& entry : deviceStagingBuffers_) {
      for (NDArray* arr : entry.second) addArray(arr);
    }
    if (untrackedOutputCache_ != nullptr) {
      for (int i = 0; i < untrackedOutputCacheSize_; ++i) {
        addArray(untrackedOutputCache_[i]);
      }
    }
    return total;
  }

  /**
   * Get the output slots array (NDArray pointers for all slots).
   * Used by validation/diagnostic functions to inspect outputs after execution.
   */
  NDArray** getOutputSlots() const { return outputSlots_; }

  // ── Buffer coloring accessors ────────────────────────────────────────
  DspBufferColorMap& bufferColorMap() { return colorMap_; }
  const DspBufferColorMap& bufferColorMap() const { return colorMap_; }

  /**
   * Get plan segments (for CUDA Graphs integration).
   */
  const std::vector<GraphSegment>& getSegments() const { return segments_; }

  /**
   * Get mutable plan segments (for clearing CUDA graph timelines, etc.).
   */
  std::vector<GraphSegment>& getSegmentsMutable() {
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE, "DSP PHASE VIOLATION: getSegmentsMutable called in phase %s",
               planLifecycle_.displayName());
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

  // Public cache invalidation entry points. These own the legal plan unseal and
  // segment rebuild transition; external callers must never mutate segments.
  bool invalidateSegmentCache(int segmentIndex, const char* reason);
  int invalidateBackendCaches(const std::string& backendName,
                              const char* reason);

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
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE, "DSP PHASE VIOLATION: setCudaGraphsEnabled called in phase %s",
               planLifecycle_.displayName());
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
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE, "DSP PHASE VIOLATION: setJitMode called in phase %s",
               planLifecycle_.displayName());
      assert(false && "DSP phase violation: setJitMode");
      return;
    }
    jitMode_ = mode;
  }
  JitMode getJitMode() const { return jitMode_; }

  /**
   * Control whether backend code generation may run while executing this plan.
   * Portable SDX bundles set this to false for hermetic mobile inference.
   */
  void setRuntimeCompilationAllowed(bool allowed) {
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE,
               "DSP PHASE VIOLATION: setRuntimeCompilationAllowed called in phase %s",
               planLifecycle_.displayName());
      assert(false && "DSP phase violation: setRuntimeCompilationAllowed");
      return;
    }
    runtimeCompilationAllowed_ = allowed;
  }
  bool isRuntimeCompilationAllowed() const {
    return runtimeCompilationAllowed_;
  }

  /**
   * Set the bundle-owned directory containing validated backend artifacts.
   * For Vulkan this directory contains spv_<cache-key>.spv/.meta pairs.
   */
  void setRuntimeArtifactDirectory(const std::string& directory) {
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE,
               "DSP PHASE VIOLATION: setRuntimeArtifactDirectory called in phase %s",
               planLifecycle_.displayName());
      assert(false && "DSP phase violation: setRuntimeArtifactDirectory");
      return;
    }
    runtimeArtifactDirectory_ = directory;
  }
  const std::string& getRuntimeArtifactDirectory() const {
    return runtimeArtifactDirectory_;
  }

  void setDeviceCompilationCacheDirectory(const std::string& directory) {
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE,
               "DSP PHASE VIOLATION: setDeviceCompilationCacheDirectory "
               "called in phase %s",
               planLifecycle_.displayName());
      assert(false && "DSP phase violation: setDeviceCompilationCacheDirectory");
      return;
    }
    deviceCompilationCacheDirectory_ = directory;
  }
  const std::string& getDeviceCompilationCacheDirectory() const {
    return deviceCompilationCacheDirectory_;
  }

  void setDeviceCompilationCacheModelKey(const std::string& modelKey) {
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(EXECUTE,
               "DSP PHASE VIOLATION: setDeviceCompilationCacheModelKey "
               "called in phase %s",
               planLifecycle_.displayName());
      assert(false && "DSP phase violation: setDeviceCompilationCacheModelKey");
      return;
    }
    deviceCompilationCacheModelKey_ = modelKey;
  }
  const std::string& getDeviceCompilationCacheModelKey() const {
    return deviceCompilationCacheModelKey_;
  }

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
  bool isShapesFrozen() const { return planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying(); }

  /**
   * Mark an external input as variable (changes between decode steps).
   *
   * Called by the native autoregressive_decode op before entering the decode
   * loop.  Inputs_embeds, attention_mask, position_ids, input_ids, and
   * causal_mask change every step — the decode kernels write fresh data to
   * their device buffers between plan executions.
   *
   * Marking them variable ensures:
   *   1. Plan-owned staging buffers are allocated for these inputs.
   *   2. ensureAndSyncStagingBuffers() D2D-copies fresh data into the staging
   *      buffers each step.
   *   3. Merged CUDA graphs (which bake in staging buffer addresses at capture
   *      time) see up-to-date data on every replay.
   *
   * Without this, the merged graph's gap ops read from the original ext input
   * address that was present at capture time.  If the native decode loop uses
   * different NDArray pointers than the Java warmup (via OpaqueContext), the
   * captured address becomes stale and replay produces degenerate output.
   *
   * MUST be called BEFORE the first executeSteadyState() in the decode loop
   * and AFTER shapes are frozen (so staging buffers are allocated at the
   * correct size).  Invalidates cached variable indices so the next execution
   * rebuilds the fast-path index.
   */
  void markExternalInputVariable(int extIdx);
  void markExternalInputPlaceholder(int extIdx);

  /**
   * Register an external input whose device buffer is mutated by native decode
   * infrastructure between plan executions. Such inputs are variable, but the
   * live device buffer is the source of truth and must not be replaced by a
   * generic staging buffer during CUDA graph capture/replay.
   */
  void registerDeviceManagedExternalInput(NDArray* input);
  bool isDeviceManagedExternalInput(NDArray* input) const;
  bool isDeviceManagedExternalInput(int extIdx, NDArray* input) const;
  bool hasDeviceManagedExternalInputs(NDArray** externalInputs, int numExternalInputs) const;

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
   * Get the current plan-level phase (legacy enum for JNI compatibility).
   * Derived from planLifecycle_. Phase progresses: SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
   */
  PlanPhase getPlanPhase() const {
    if (planLifecycle_.isReplaying()) return PlanPhase::REPLAYING;
    if (!planLifecycle_.isSlotBySlot()) return PlanPhase::SHAPES_FROZEN;
    return PlanPhase::SLOT_BY_SLOT;
  }

  /** PRIMARY accessor: unified plan lifecycle struct. */
  const PlanLifecycle& planLifecycle() const { return planLifecycle_; }

  /**
   * Prepare the plan for a segment-local rebuild.
   *
   * Segment-level graph invalidation can happen while the plan is sealed in
   * REPLAYING (for example, after a replay address drift). The segment is
   * reset to BUILDING:WARMUP by SegmentLifecycle, so leaving the plan sealed
   * would make the next execute fail the replay invariant before dispatch.
   * Capture-only invalidation preserves the completed plan compilation phase;
   * full artifact invalidation clears it. Keep both transitions at this plan
   * boundary so callers cannot mutate the compilation seal independently.
   */
  void prepareForSegmentRebuild(const char* reason,
                                bool invalidatePlanCompilation) {
    if (planLifecycle_.isReplaying()) {
      planLifecycle_.unseal();
    } else if (planLifecycle_.isInFrozenOrReplayState()) {
      planLifecycle_.recordPointersUnstable();
    }
    frozenSnapshot_.clear();
    if (invalidatePlanCompilation) {
      planLifecycle_.compilationDone = false;
    }
    DSP_DIAG(LIFECYCLE,
             "PLAN_REBUILD_PREPARED: reason=%s phase=%s pointersStable=%d invalidateCompilation=%d compilationDone=%d",
             reason ? reason : "?", planLifecycle_.displayName(),
             planLifecycle_.pointersStableCount,
             invalidatePlanCompilation ? 1 : 0,
             planLifecycle_.compilationDone ? 1 : 0);
  }

  /**
   * Unified lifecycle: maps to 3-state GraphNodePhase.
   * BUILDING = not yet in steady-state replay; SEALED = all segments replaying.
   */
  GraphNodePhase graphNodePhase() const { return planLifecycle_.phase; }

  /**
   * Get the plan-level phase as an integer (for JNI).
   */
  int getPlanPhaseCode() const { return planLifecycle_.toLegacyCode(); }

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
  bool arePointersStable() const { return planLifecycle_.pointersStable(); }

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
  bool isCompilationSealed() const { return planLifecycle_.compilationDone; }

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
   * Write to an output slot. The ONLY way to install an NDArray into outputSlots_.
   * Validates against phase invariants. Hard error on violation.
   * Tracks plan ownership of new arrays.
   */
  void writeOutputSlot(int slotIdx, NDArray* value, const char* tag);

  /**
   * Mark output-slot VIEW wrappers that alias EXTERNAL input buffers for
   * reacquire validation. dispatchNativePlan does not receive the current
   * NDArray* inputs, so the actual decision is deferred until execute(): views
   * whose external DataBuffer is unchanged stay live and preserve replay;
   * views over replaced or invalid external buffers are nulled and their
   * segment captures are rebuilt.
   */
  void invalidateExternalViewSlotsOnReacquire();

  /** Resolve a pending external-view reacquire once current inputs are known. */
  void processPendingExternalViewReacquire(NDArray** externalInputs, int numExternalInputs);

  /**
   * Clear an output slot to nullptr with proper cleanup.
   * Removes from planOwnedArrays_, optionally defers delete.
   * All null-assignments to outputSlots_ MUST go through this method.
   * @param deferDelete if true, queue old array for deferred deletion
   */
  void clearOutputSlot(int slotIdx, const char* tag, bool deferDelete = false);

  /**
   * Mark an operation slot as a view producer. This is a STRUCTURAL property of
   * the producing op (permute/reshape always produce views), not of its output-slot
   * index. The flag persists across reset()/unseal().
   */
  void markViewProducer(int producerSlotIdx, const char* tag);

  /**
   * Demote an operation slot from view producer and clear the specific output
   * slot whose wrapper can no longer be maintained. Operation-slot and output-slot
   * indices are separate domains whenever an op has multiple outputs.
   */
  void demoteViewProducer(
      int producerSlotIdx, int outputSlotIdx, const char* tag, bool forceClear = true);

  /**
   * Materialize a view-producer slot: replace the zero-copy view with an
   * independent copy so the plan owns its own DataBuffer. Used after execution
   * to decouple slot outputs from external (placeholder) inputs that the
   * caller may close.
   */
  void materializeViewSlot(int slotIdx, const char* tag);

  /**
   * Get the slot state for a specific slot index (for JNI).
   * Returns -1 if slotIdx is out of range.
   */
  int getSlotStateCode(int slotIdx) const {
    if (slotIdx < 0 || slotIdx >= numSlots_) return -1;
    return slots_[slotIdx].slotPhase.toLegacyCode();
  }

  // ── JNI introspection methods (NativeOpsDsp.h wrappers) ──────────────────

  /** Device address of the last external input at index, or 0.
   *
   *  Returns the address RECORDED at execute time — never dereferences the
   *  stored NDArray*. Java only guarantees the input arrays live for the
   *  duration of the execute call; callers commonly pass fresh inputs each
   *  step and free the old ones, so a query-time dereference reads a freed
   *  DataBuffer (garbage magic number → integrity-check throw). */
  long long getLastExternalInputAddress(int extIdx) const {
    if (extIdx < 0 || extIdx >= static_cast<int>(lastExternalInputAddrs_.size())) return 0;
    return lastExternalInputAddrs_[extIdx];
  }

  /** Get the output NDArray* at a specific slot index, or nullptr. */
  NDArray* getSlotOutputArray(int slotIdx) const {
    if (outputSlots_ == nullptr || slotIdx < 0 || slotIdx >= totalOutputSlots_)
      return nullptr;
    return outputSlots_[slotIdx];
  }

  /** Get the dirty generation counter for a slot (0 if out of range). */
  int getSlotGeneration(int slotIdx) const {
    if (slotIdx < 0 || slotIdx >= static_cast<int>(dirtySlotGenerations_.size())) return 0;
    return static_cast<int>(dirtySlotGenerations_[slotIdx]);
  }

  /** Get the execution phase code for a segment (via GraphSegmentExec). */
  int getSegmentReplayMode(int segIdx) const {
    if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return 0;
    return segments_[segIdx].exec.getExecutionPhaseCode();
  }

  /** Get the arg table generation counter for a segment. */
  long long getSegmentArgGeneration(int segIdx) const {
    if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return 0;
    return static_cast<long long>(segments_[segIdx].exec.argTableGeneration);
  }

  /** Get the captured arg generation counter for a segment. */
  long long getSegmentCapturedArgGeneration(int segIdx) const {
    if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return 0;
    return static_cast<long long>(segments_[segIdx].exec.capturedArgGeneration);
  }

  /** Check if a segment needs arg refresh (generation mismatch). */
  int getSegmentNeedsArgRefresh(int segIdx) const {
    if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return 0;
    return segments_[segIdx].exec.needsArgRefresh() ? 1 : 0;
  }

  /** Get the captured input address key for a segment. */
  long long getSegmentCapturedInputAddrKey(int segIdx) const {
    if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return 0;
    return static_cast<long long>(segments_[segIdx].exec.capturedInputAddrKey);
  }

  // ── Last execution stats snapshot ──────────────────────────────────────
  // Snapshotted from PlanExecutionContext at end of each execute() call.
  // Returns -1 if no execution has occurred yet.

  struct LastExecStats {
    int segmentsWarmup = -1;
    int segmentsCaptured = -1;
    int segmentsReplayed = -1;
    int segmentsSlotBySlot = -1;
    int segmentsFailed = -1;
    int segmentsTotal = -1;
    int syncLevel = -1;          // SyncLevel enum as int
    int streamSyncCount = -1;
    int consecutiveUnchangedCount = -1;
    bool valid = false;          // true after first execute()
  };

  int getLastExecSegmentsWarmup() const { return lastExecStats_.segmentsWarmup; }
  int getLastExecSegmentsCaptured() const { return lastExecStats_.segmentsCaptured; }
  int getLastExecSegmentsReplayed() const { return lastExecStats_.segmentsReplayed; }
  int getLastExecSegmentsSlotBySlot() const { return lastExecStats_.segmentsSlotBySlot; }
  int getLastExecSegmentsFailed() const { return lastExecStats_.segmentsFailed; }
  int getLastExecSegmentsTotal() const { return lastExecStats_.segmentsTotal; }
  int getLastExecSyncLevel() const { return lastExecStats_.syncLevel; }
  int getLastExecStreamSyncCount() const { return lastExecStats_.streamSyncCount; }
  int getLastExecConsecutiveUnchangedCount() const { return lastExecStats_.consecutiveUnchangedCount; }

  /** Called at end of execute() to snapshot PlanExecutionContext stats. */
  void snapshotExecStats(void* execCtxPtr);

  // ── Cross-stream testing API (CUDA only) ────────────────────────────────

  /** Write host data to a staging buffer's device memory on the default stream. */
  int writeDeviceBufferOnDefaultStream(int extIdx, void* srcHost, long long numBytes);

  /** Write host data to a staging buffer's device memory on an explicit stream. */
  int writeDeviceBufferOnExplicitStream(int extIdx, void* srcHost, long long numBytes, void* stream);

  /** Check if ext input at index has device-authoritative data. */
  int isExtInputDeviceAuthoritative(int extIdx) const {
    NDArray* arr = getLastExternalInput(extIdx);
    if (arr == nullptr) return 0;
    auto* db = arr->dataBuffer();
    if (db == nullptr) return 0;
    return db->isPrimaryActual() ? 0 : 1;  // device-authoritative = special is actual, primary is NOT
  }

  /** Get this plan's backend-owned execution stream storage.
   *  GPU plans own a stream so capture/replay is isolated from unrelated work
   *  submitted through a process-wide default stream.
   *
   *  CUDA ABI: this returns cudaStream_t* (pointer to handle), matching
   *  executeDynamicShapePlan and dbAsyncCrossDeviceCopy. APIs that accept a
   *  cudaStream_t handle value must dereference this boundary explicitly. */
  void* getExecutionStream() const {
#ifdef SD_CUDA
    if (ownedStream_ != nullptr) {
      return reinterpret_cast<void*>(ownedStream_);
    }
    auto lc = sd::LaunchContext::defaultContext();
    return lc != nullptr ? reinterpret_cast<void*>(lc->getCudaStream()) : nullptr;
#elif defined(SD_VULKAN)
    return platformGetExecutionStream();
#else
    return nullptr;
#endif
  }

  /** Get JSON summary of all plan segments (for diagnostics). */
  std::string getSegmentsSummaryJson() const;

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
   * Access the persisted steady-state execution context (survives across execute() calls).
   * Returns null before the first steady-state execute().
   */
  void* steadyStateExecutionContext() const { return steadyStateExecCtx_; }

  /** Access prev-step fingerprints map for diagnostic queries. */
  const std::unordered_map<int, uint64_t>& getPrevStepFingerprints() const { return prevStepFingerprints_; }

  /**
   * Ensure plan-owned staging buffers exist for variable external inputs,
   * then D2D copy current data into them. Returns pointer to internal
   * effectiveExternals_ array (staging buffers for variable inputs,
   * original pointers for non-variable inputs). Only active when frozen.
   * Called once per step (gated by PlanExecutionContext dedup flag).
   */
  DspStagingSyncResult ensureAndSyncStagingBuffers(NDArray** externalArrays, int numExt, void* stream);

  /**
   * Unified pre-replay synchronization. Handles all three sync concerns:
   *   1. Cross-stream ordering (default stream → DSP stream via event)
   *   2. H2D sync for variable external inputs (isPrimaryActual guard)
   *   3. D2D copy into plan-owned staging buffers
   *
   * Idempotent: PlanExecutionContext dedup flags ensure each step runs
   * at most once per execute() call. Safe to call from every replay path.
   *
   * Returns effectiveExternals (staging ptrs for variable inputs, originals
   * for weights). Callers should use the returned pointer for arg table
   * refresh and address validation.
   *
   * PRECONDITION: activeExecutionContext() returns valid PlanExecutionContext*.
   *               DspStreamGuard is active (caller owns it).
   */
  DspStagingSyncResult performPreReplaySync(NDArray** externalArrays, int numExt,
                                            void* stream, const char* diagTag);

  /**
   * Staleness detector — verifies variable inputs are fresh before graph replay.
   * Called by performPreReplaySync() when DSP VERIFY diagnostics are enabled.
   * Throws std::runtime_error on detected staleness.
   */
  void verifyStagingNotStale(NDArray** externalArrays, NDArray** effectiveArrays,
                             int numExt, void* stream, const char* diagTag);

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
   * Validate that a compiled graph backend covers all ops in the segment.
   * Returns true if every op was compiled by the resolved implementation.
   * When any ops are missing, logs warnings and returns false.
   *
   * @param segmentIndex  Which segment to validate (-1 for all segments)
   * @return true if all ops were compiled, false if any were skipped
   */
  bool validateCompiledGraphBackend(int segmentIndex = -1) const;

  // Segment cleanup — public so SegmentLifecycle::invalidateForRebuild can call it.
  void cleanupSegmentForRebuild(GraphSegment& seg, const char* reason);

  // Dump full graph state for all segments — phases, outcomes, replay handle
  // tracker summaries, captured address keys. Routes through DspDiagnostics.
  // tag: caller-supplied label (e.g. "pre-replay", "post-capture", "error-dump")
  void dumpSegmentGraphState(const char* tag) const;

  // Clear all nativeRangeSegments_ entries whose slot range overlaps [startSlot, endSlot].
  // Called from invalidateForRebuild so that CPU replay handles captured by the
  // NativeSlotExecutor lambda are not replayed with stale slot arrays after
  // the outer segment is invalidated.
  void clearNativeRangeSegmentsForSlotRange(int startSlot, int endSlot);

  // Plan-level execute count: single LOGGED authority. Every change routes through
  // these so a stray per-step reset (the "re-warms every step / never captures" bug
  // class) is traceable in DSP_DIAG, never silent. Do NOT write executeCount_ directly.
  void resetExecuteCount(const char* reason = "invalidation") {
    DSP_DIAG(EXECUTE, "executeCount: RESET %d -> 0 (%s) [phase=%s]",
             executeCount_, reason, planLifecycle_.displayName());
    executeCount_ = 0;
  }
  void incrementExecuteCount(const char* reason) {
    executeCount_++;
    DSP_DIAG(EXECUTE, "executeCount: %d -> %d (%s) [phase=%s]",
             executeCount_ - 1, executeCount_, reason, planLifecycle_.displayName());
  }

  // One-shot DSP state dump: the COMPLETE plan + per-segment picture in a single
  // DSP_DIAG block (per segment: phase/outcome/execCount/sealed/handleReady/composite/
  // compiledBy), so an odd/stuck plan state is diagnosable from one place instead of
  // correlated lines. Fired at the plan-phase decision points + the phase stall.
  void dumpPlanPhaseState(const char* context) const;

  // Reset frozenConstantDetectionDone_ so detectFrozenConstants() re-runs
  // after the next warmup.  Without this, stale frozen classifications
  // persist through invalidation cycles.
  void resetFrozenConstantDetection() {
    DSP_DIAG(EXECUTE, "resetFrozenConstantDetection: frozenConstantDetectionDone_ %d -> false "
             "(stale classifications cleared)", (int)frozenConstantDetectionDone_);
    frozenConstantDetectionDone_ = false;
  }

  // Reset slot states within a segment range back to WARMUP so frozen
  // contexts with stale dtypes are not reused after invalidation.
  // Keeps cached shapes intact — they're needed by the normal warmup
  // path for output allocation.  Only clears the frozen-context gate.
  void resetSlotStatesForSegment(int startSlot, int endSlot) {
    DSP_DIAG(EXECUTE, "resetSlotStatesForSegment: resetting slots [%d-%d] back to BUILDING",
             startSlot, endSlot);
    int clearedClosedOutputs = 0;
    for (int i = startSlot; i <= endSlot && i < numSlots_; i++) {
      // SlotPhase::reset() emits DSP_DIAG(LIFECYCLE) with old->new state per slot.
      slots_[i].slotPhase.reset();
      // Clear frozen buffer pointer snapshot so detectFrozenConstants re-snapshots
      // fresh pointers when the slot is re-frozen after re-warmup.
      slots_[i].frozenOutputPtrs.clear();
      for (int o = 0; o < slots_[i].wiring.numOutputs; o++) {
        int outSi = slots_[i].wiring.outputSlotIndices[o];
        if (outSi < 0 || outSi >= totalOutputSlots_ || outputSlots_ == nullptr) continue;
        NDArray* arr = outputSlots_[outSi];
        DataBuffer* db = arr != nullptr ? arr->dataBuffer() : nullptr;
        if (db != nullptr && db->isClosed()) {
          outputSlots_[outSi] = nullptr;
          planOwnedArrays_.erase(arr);
          if (slotOwnership_ != nullptr) {
            slotOwnership_[outSi].reset();
          }
          clearedClosedOutputs++;
        }
      }
    }
    if (clearedClosedOutputs > 0) {
      DSP_DIAG(MEMORY,
               "resetSlotStatesForSegment: cleared %d closed cached output slot(s) "
               "inside invalidated range [%d-%d]",
               clearedClosedOutputs, startSlot, endSlot);
    }
  }

  // Public so NativeOps_dsp.cpp diagnostics can query composite state.
  bool hasCompositeHandles(const GraphSegment& seg) const;

  // ── Public fingerprint ring API (sync-free diagnostics, CUDA only) ─────────
  // These are called from .cu TUs, so they are public.
#ifdef SD_CUDA

  /** Activate the fingerprint ring if env BUF_FP_RING=1. Call at execute() entry. */
  void maybeInitFingerprintRing();

  /**
   * Fire XOR-fingerprint kernel asynchronously on `stream`.
   * Writes one uint64 into d_fpRing_[step * BUF_FP_MAX_TRACKED + trackIdx].
   * No-op when ring is disabled, pointer is null, or trackIdx is out of range.
   * step is clamped to [0, BUF_FP_MAX_STEPS-1].
   */
  void recordBufFingerprintPublic(cudaStream_t stream, int step, int trackIdx,
                                  const void* devPtr, size_t numBytes);

#endif  // SD_CUDA

  /**
   * One-shot D2H drain: cudaMemcpy d_fpRing→h_fpRing (synchronous, call only
   * AFTER decode loop, never during capture/replay).
   * No-op if ring not enabled or already drained.
   */
  void drainFingerprintRingPublic();

  /**
   * Return a JSON string describing the per-step fingerprints for all tracked
   * buffers. Returns "null" if ring not drained yet or disabled.
   * The returned pointer is valid until the next call or plan destruction.
   */
  const char* getFingerprintJson();

  friend class NativePlanCompiler;

 private:
  NativeDynamicShapePlan();
  static void recordPlanFailureIfMissing(Status status, const std::string& detail);

  // ── Shared immutable plan definition (Phase 3) ────────────────────────
  // Contains all data that does NOT change between executions:
  // numSlots, totalOutputSlots, numExternalInputs, requestedOutputSlotIndices,
  // externalInputNames, externalInputIsVariable, hasControlFlow, numLoopRegions,
  // backendPriority. Created during compilation, ref-counted for sharing.
  // Currently populated alongside existing fields (behavioral no-op).
  // Future phases will migrate reads to use planDef_ instead.
  PlanDefinition* planDef_ = nullptr;

  // Immutable source bytes for an isolated replay-verification plan. Plans
  // created from FlatGraph do not have this payload and report verification as
  // UNAVAILABLE rather than mutating the live plan to manufacture a reference.
  std::vector<uint8_t> serializedPlanBytes_;

  // ── Per-plan-instance resource state ───────────────────────────────────
  // Owns slotArrays, ownership, protectedWeightBuffers, segment device/stream
  // state, and capture workspace. Thread-bound (error if called from a
  // different thread). Lifecycle state lives in planLifecycle_ above.
  ExecutionState* execState_ = nullptr;

  // Slot data
  NativeSlot* slots_;
  int numSlots_;
  int totalOutputSlots_;
  int numExternalInputs_;
  std::vector<NDArray*> lastExternalInputsCopy_;  // owned copy of ext input pointer array
  NDArray** lastExternalInputs_ = nullptr;       // points to lastExternalInputsCopy_.data() (stable after execute)
  bool externalViewReacquirePending_ = false;    // validate external-fed views at next execute()
  // Buffer addresses of the ext inputs, recorded at execute time while the Java-owned
  // NDArrays are guaranteed live. JNI queries (getLastExternalInputAddress) read THESE —
  // dereferencing lastExternalInputsCopy_ after execute() returns is a use-after-free
  // when the caller frees/replaces its input arrays between steps.
  std::vector<long long> lastExternalInputAddrs_;
  int lastNumExternalInputs_ = 0;               // size of lastExternalInputs_
  std::vector<std::string> externalInputNames_;  // name for each external input index
  std::vector<bool> externalInputIsVariable_;    // true if VARIABLE or PLACEHOLDER (needs forced H2D before replay)
  std::vector<bool> externalInputIsPlaceholder_; // true if host-written placeholder (force H2D); false if device-written (respect actuality)
  std::vector<int> variableExternalInputIndices_;  // cached indices where externalInputIsVariable_[i]=true (replay optimization)
  bool variableIndicesCached_ = false;             // true once variableExternalInputIndices_ is populated

  // Per-slot transitive variable dependency: true if slot transitively depends on any
  // variable ext input (PLACEHOLDER). Computed via forward propagation through wiring graph.
  // Used by the frozen fast-path gate to prevent reusing stale cached outputs for slots
  // that indirectly depend on changing placeholder inputs.
  std::vector<bool> slotDependsOnVariableExtInput_;

  // External input ranks captured during the first execute() call.
  // -1 = not yet observed. Used by FusionPass at freeze transition to
  // disambiguate 1D bias vectors from N-D residual operands.
  std::vector<int> externalInputRanks_;

  // Release schedule: releaseAtStep_[stepIdx] = array of slot indices to release
  int** releaseAtStep_;
  int* releaseAtStepCounts_;

  // Slot liveness data — producer/lastConsumer step indices for buffer coloring.
  // Populated by NativePlanCompiler, owned by this plan instance.
  SlotLivenessData* slotLiveness_ = nullptr;

  // Requested output mapping
  int* requestedOutputSlotIndices_;
  int numRequestedOutputs_;

  // Execution state (reused across calls)
  NDArray** outputSlots_;              // THE slot arrays — current output values for all slots
  // slotIsViewProducer_ removed — use slots_[i].slotPhase.isViewProducer instead.
  Context** contextPool_;              // Pre-allocated Context pool
  bool viewProducerDetectionDone_;
  bool frozenConstantDetectionDone_;

  // Unified buffer ownership tracking (Phase 1A).
  // One SlotBufferInfo per totalOutputSlots_. Replaces ad-hoc tracking:
  //   protectedWeightBuffers_ → ownership == WEIGHT/VIEW_OF_WEIGHT
  //   slotViewOutputs_ → ownership == VIEW_OF_SLOT with parentSlotIdx
  //   per-execute dedup HashSet → O(1) ownership check
  SlotBufferInfo* slotOwnership_;

  // Buffer coloring — compile-time buffer sharing for non-overlapping slots.
  // Computed after SHAPES_FROZEN, applied to replace per-slot buffers with
  // shared color buffers.  Ejected on shape change or validation failure.
  DspBufferColorMap colorMap_;

  // Dirty bitmap (generation counter): tracks which output slots were written
  // during the current execution. Used to optimize tickWriteDevice() in steady
  // state — only dirty slots need to be ticked instead of all totalOutputSlots_.
  // Generation counter avoids the O(N) std::fill per step of a bool vector.
  std::vector<uint32_t> dirtySlotGenerations_;
  uint32_t currentDirtyGeneration_ = 1;

  // Graph segments for CUDA Graphs
  std::vector<GraphSegment> segments_;
  bool hasDynamicSegmentBoundaries_ = false;

  // Persistent native-range segments for OneDNN/OpenVINO NativeSlotExecutor callbacks.
  // Keyed by (startSlot << 32 | endSlot). Enables CPU_FROZEN_REPLAY for native range
  // sub-segments that would otherwise be ephemeral stack-allocated GraphSegments.
  std::unordered_map<uint64_t, GraphSegment> nativeRangeSegments_;
  static uint64_t nativeRangeKey(int start, int end) {
    return (static_cast<uint64_t>(start) << 32) | static_cast<uint64_t>(end);
  }

  // Retired plan-owned NDArray wrappers from every slot/view/coloring path.
  // Deletion is deferred until the whole plan execution has finished; deleting
  // at a per-slot or per-segment boundary can corrupt adjacent allocator metadata.
  std::vector<NDArray*> deferredSlotDeletes_;
  void flushDeferredSlotDeletes();

  // Protected DataBuffers: model weights and shapeStatic outputs whose
  // DataBuffers must NEVER be freed during cleanup. Built on first execute().
  // Mirrors Java-side protectedWeightBuffers in DynamicShapePlanExecutor.
  std::unordered_set<DataBuffer*> protectedWeightBuffers_;

  // GPU addresses pinned to prevent pool reuse while baked into live CUDA graphs.
  // Each entry is {specialBuffer ptr, deviceId, segStartSlot}. Populated by writeOutputSlot
  // when a sealed segment exists. segStartSlot identifies which sealed segment baked this
  // address. Flushed segment-by-segment at platformCleanupSegmentForRebuild (only the
  // invalidated segment's pins); flushed entirely at platformFreePlanResources and
  // releaseGpuIntermediates. Raw structs, no smart pointers.
  // externalOwned: the pinned buffer is owned OUTSIDE the plan (a SOURCE_VARIABLE weight, or a
  // view whose base is one) — a later exec in the SAME execute() may still read it after a
  // weight rebind, so its deferred free MUST wait until plan teardown (platformFreePlanResources,
  // post stream-sync). Plan-owned intermediates (externalOwned=false) are released eagerly at
  // segment-rebuild invalidation (platformCleanupSegmentForRebuild) — their dead graph won't read them.
  struct GraphPinnedAddr { void* ptr; int deviceId; int segStartSlot; bool externalOwned; };
  std::vector<GraphPinnedAddr> graphPinnedAddrs_;

  // DataBuffers whose frozen refs were added by this plan. Keep exact tracked
  // ownership instead of inferring from lifecycle state at teardown; AUTO_SEAL,
  // phaseWarmup, and protected-input rebinds can all mutate the set.
  std::vector<DataBuffer*> frozenProtectedRefBuffers_;

  // DataBuffers that received output-slot frozen refs from this plan.
  // Multiplicity matters: view/identity slots can share one DataBuffer, and
  // frozen sealing adds one ref per non-null output slot.
  std::vector<DataBuffer*> frozenOutputRefBuffers_;

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

  // Portable runtime compilation policy. Direct JVM/native plan users retain
  // the historical compile-on-miss behavior; the SDX C runtime overrides it.
  bool runtimeCompilationAllowed_ = true;
  std::string runtimeArtifactDirectory_;
  std::string deviceCompilationCacheDirectory_;
  std::string deviceCompilationCacheModelKey_;

  // Graph execution mode (controls which backend to use)
  GraphExecutionMode graphExecutionMode_;

  // shapesFrozen_ removed — use planLifecycle_.isShapesFrozen() or
  // !planLifecycle_.isSlotBySlot() instead. All callers migrated.
  bool shapeOnlyMode_ = false;  // When true, executeSlot skips op->execute(); all dispatch
                                 // infrastructure runs normally (shape, alloc, frozen detection).
                                 // Use to measure pure dispatch overhead without kernel cost.
  bool shapePrePassDone_ = false;  // True after phaseShapeInferenceOnly has been run automatically
                                    // during the first execute() call. Prevents redundant re-runs.
  bool shapePrePassComplete_ = false;  // True only when that pass resolved every live slot output.
  int shapePrePassFirstIncompleteSlot_ = -1;
  std::string shapePrePassIncompleteReason_;
  bool inShapeChangeWarmup_ = false;  // True during segDispatchCompile's shape-change warmup pass.
                                       // Allows slot shape reassignment in step3_allocateOutputs.
  int executeCount_;  // Total plan execution count (monotonically increasing)
  uint64_t diagnosticEpoch_ = 0;
  int diagnosticEpochBaseExecuteCount_ = 0;
  // Set when weight DataBuffer rebinds are detected (a NEW Java executor
  // borrowed this cached plan): the next external-input H2D prepare must be
  // BROAD (all inputs, not the variable-filter subset) even though
  // executeCount_>0 makes the firstFrozenExec gate false. Without this, a
  // plan-cache hit in REPLAYING phase skips weight H2D sync for the new
  // executor's buffers → wrong results (batch-only bgeEncoder ~7% divergence,
  // testFreshInputCloseBetween[5]).
  //
  // NEVER poke this field directly — all transitions go through the
  // DSP_DIAG-logged accessors below (same consolidation contract as the
  // GraphSegmentExec state methods).
  bool needsBroadPreReplaySync_ = false;
  // Mark: weight rebind detected (refreshProtectedWeightBuffers). Logged.
  void markWeightRebindNeedsBroadSync(const char* reason);
  // Consume-if-set at a broad-prepare site: returns whether broad sync is
  // required and clears the flag. Logged when it fires.
  bool consumeBroadPreReplaySync(const char* site);
  // Read-only peek for gates that must widen but do not own consumption.
  bool weightRebindBroadSyncPending() const { return needsBroadPreReplaySync_; }
  LastExecStats lastExecStats_;  // Snapshotted from PlanExecutionContext at end of each execute()
  uint64_t identityFingerprint_ = 0; // FNV-1a hash of (numSlots, opNames, wiring) — set at deserialization
  // Sync override depth counter. When > 0, needsSync() returns true regardless
  // of contract/lifecycle state. Incremented/decremented by SyncOverride RAII guard.
  // Replaces the old mutable forceSync_ boolean — all sync policy now flows from
  // the ModeContract (forcesSyncOnFrozen, forceSyncDuringCapture) plus this
  // scoped override for gpu backend bracket execution (gap ops, pre-capture warmup).
  int syncOverrideDepth_ = 0;

  // Count of compileSegment() calls that occurred AFTER planLifecycle_.compilationDone was set
  // (i.e., mid-execution compiles). Accessed from the shape-change recompile
  // path and read by benchmarks to assert "no compile during measurement".
  // Atomic because phaseCompile() can be multi-threaded.
  std::atomic<int64_t> midExecutionCompileCount_{0};

  // ── Plan-level phase tracking ──────────────────────────────────────────
  // PRIMARY lifecycle struct — single source of truth for plan phase.
  // All reads go through planLifecycle_. Use getPlanPhase() for PlanPhase enum.
  PlanLifecycle planLifecycle_;

  // Tracks why the plan was destroyed/reset (for diagnostics)
  PlanDestructionReason destructionReason_ = PlanDestructionReason::NORMAL_CLOSE;

  // True when GPU intermediates have been released by passivation.
  // Cleared by reactivate() on next cache hit; execute path re-warms.
  bool passivated_ = false;

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
  // the default stream (stream 0) before argmax/sampling. This keeps
  // completion ordering on the GPU timeline without blocking the host.
  void* executionCompleteEvent_ = nullptr;  // cudaEvent_t (stored as void* to avoid cuda header)

  // Reusable cross-stream sync event for platformBeginExecution (WS-N4):
  // previously created + destroyed on EVERY execute (~2 driver calls per
  // token in steady-state decode). Lazily created, re-created on device
  // change, destroyed alongside executionCompleteEvent_.
  void* ownedCrossStreamEvent_ = nullptr;  // cudaEvent_t (void* to avoid cuda header)
  int ownedCrossStreamEventDeviceId_ = -1;
  int executionCompleteEventDeviceId_ = -1; // Device the event was created on

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
  // Device-0 staging array retained for compatibility with CPU/Vulkan paths.
  // CUDA switches activeStagingBuffers_ to a per-device vector before each
  // segment, so graph captures never reuse a pointer allocated on another GPU.
  NDArray** placeholderStagingBuffers_ = nullptr;
  NDArray** activeStagingBuffers_ = nullptr;
  int activeStagingDevice_ = -1;
  std::unordered_map<int, std::vector<NDArray*>> deviceStagingBuffers_;
  // True only during an execute() whose pre-replay sync ran (or intra-exec
  // deduped) the staging D2D — i.e., staging content matches THIS exec's
  // external inputs. Reset at every execute() entry. Gates the staging
  // preference in resolveViewInput().
  bool stagingMaintainedThisExec_ = false;
  NDArray** effectiveExternals_ = nullptr;
  std::vector<int> cachedVariableExtIndices_;  // fast-path: only iterate variable inputs
  std::vector<bool> deviceWritePending_;        // per-input: JNI wrote staging directly, skip D2D overwrite

  // Staleness detection state (used by verifyStagingNotStale)
  std::unordered_map<int, uint64_t> prevStepFingerprints_;  // ext idx → FNV-1a of device data
  std::unordered_map<int, void*> prevStagingAddresses_;     // ext idx → staging specialBuffer ptr

  // ── Sync-free buffer fingerprint ring ────────────────────────────────────
  // Records per-step XOR fingerprints of key device buffers (staging inputs
  // and gap-matmul inputs) without any host sync during the decode loop.
  // One D2H after the loop drains the ring for post-mortem analysis.
  // Activated by env BUF_FP_RING=1 at plan-first-execute time.
  //
  // Layout: d_fpRing_[step * BUF_FP_MAX_TRACKED + trackIdx] = XOR fingerprint.
  // step = executeCount_ when fingerprint was recorded (clamped to BUF_FP_MAX_STEPS-1).
  // trackIdx: [0, BUF_FP_TRACE_TRACK) = staging/slot/gap tracks;
  //           BUF_FP_TRACE_TRACK = configured trace-slot output;
  //           requested output 0 is followed across the writer, DSP completion,
  //           LC handoff, and each merged-capture validation replay.
  // step + BUF_FP_POST_CAPTURE_STEP_OFFSET holds post-validation slot values.
  static constexpr int BUF_FP_MAX_STEPS                = 64;
  static constexpr int BUF_FP_MAX_TRACKED              = 128;
  static constexpr int BUF_FP_MAX_STAGING              = 32;
  static constexpr int BUF_FP_TRACE_TRACK              = 96;
  static constexpr int BUF_FP_REQUESTED_OUTPUT_TRACK   = 97;
  static constexpr int BUF_FP_END_DSP_TRACK            = 98;
  static constexpr int BUF_FP_END_LC_TRACK             = 99;
  static constexpr int BUF_FP_CAPTURE_GROUP_BASE       = 100;
  static constexpr int BUF_FP_CAPTURE_GROUP_COUNT      = 28;
  static constexpr int BUF_FP_POST_CAPTURE_STEP_OFFSET = 32;

  // Label table (host-side): describes what each trackIdx covers.
  struct BufFpLabel {
    char tag[24];  // e.g. "stg[5]" or "gemm[2].A"
    int  extIdx;   // -1 for gemm entries
    int  groupIdx; // -1 for staging entries
    int  whichAB;  // 0=A,1=B; -1 for staging
  };

  bool         fpRingEnabled_       = false;  // activated at first execute if env set
  bool         fpRingDrained_       = false;  // true after drainFingerprintRingPublic()
  int          fpRingStagingCount_  = 0;      // how many staging track slots were filled
  int          fpRingGemmCount_     = 0;      // how many gemm track slots were filled
  std::atomic<int> fpInvocationCount_{0};     // unique diagnostic step even when executeCount_ resets/stalls
  BufFpLabel   fpLabels_[BUF_FP_MAX_TRACKED] = {};  // label for each trackIdx
  std::string  fpJsonBuffer_;                 // backing storage for getFingerprintJson()

#ifdef SD_CUDA
  uint64_t*    d_fpRing_ = nullptr;  // device: [BUF_FP_MAX_STEPS * BUF_FP_MAX_TRACKED]
  uint64_t*    h_fpRing_ = nullptr;  // host mirror, filled by one D2H drain

  // Allocate ring buffers (called once at first execute if env set).
  // No-op after first call.
  void initFingerprintRing();
#endif  // SD_CUDA

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

  // Compilation audit: per-op compilation status for graph backends
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Human-readable detail of last compile failure (for error messages)
  std::string lastCompileFailureDetail_;

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
  int cfLoopBackStep_;                     // Set by NextIteration to signal phaseReplay to jump back (-1 = no jump)

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
  void refreshDynamicSegmentBoundaryAnalysis();
  void resegmentForFreeze();
  SelectedBackend resolveBackendForSegment(bool isBackendEligible);

  // ── Slot execution (NativeDynamicShapePlan_slotexec.cpp) ──
  Status executeSlot(int slotIdx, NDArray** externalArrays, int numExt, void* stream);
  // Ultra-fast gap slot execution for steady-state cached path.
  // Bypasses all validation, diagnostics, view checks, sync, prezero.
  // Only call when: pointers stable, executeCount_ >= 5, action == EXECUTE.
  Status executeSlotGapFast(int slotIdx, NDArray** externalArrays, int numExt);
  LongType computeShapeKey(NativeSlot& slot, NDArray** inputs, int numInputs);
  void detectFrozenConstants();
  void computeSlotVariableDependency();

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

  // ── Consolidated segment dispatch (NativeDynamicShapePlan.cpp) ──
  // Single entry point for ALL segment execution. Uses (selectedBackend,
  // segPhase, outcome) to pick exactly one action. Replaces the scattered
  // 3-level dispatch across phaseReplay, platformExecuteSegmentWithBackends,
  // and executeSegmentWithGpuGraph internal branches.
  Status dispatchSegment(GraphSegment& seg, NDArray** externalArrays,
                         int numExt, void* stream, bool& usedGraph);

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

  /**
   * Resolve an input for a view op, preferring the staging buffer for external inputs.
   *
   * When a view op aliases an external input, the view wrapper must point into the
   * plan-owned staging buffer (stable address) rather than the Java-managed placeholder
   * (address changes every frame). cuBLAS kernels baked into the CUDA graph have staging
   * buffer addresses — if views resolve through the original placeholder, the output slot
   * addresses drift and the graph reads from freed/wrong memory (error 700).
   *
   * @param srcIdx  Slot wiring source index: >= 0 for internal slot, < 0 for external
   * @param externalArrays  Java-provided external input arrays
   * @param numExt  Number of external inputs
   * @return The staging NDArray* if available, else the original external array, else outputSlots_[srcIdx]
   */
  inline NDArray* resolveViewInput(int srcIdx, NDArray** externalArrays, int numExt) const {
    if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
      return outputSlots_[srcIdx];
    } else if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx >= numExt) return nullptr;
      // Prefer staging buffer: stable address that matches what the CUDA graph captured.
      // ONLY while staging is maintained this exec: once the exec target flips to
      // direct SBS ("staging D2D skipped"), staging content is no longer refreshed —
      // a view minted over it reads a PREVIOUS iteration's values forever
      // (deepAttentionQKV varying#4: bit-identical wrong output across NVRTC/PTX,
      // views object-matched to the exec-1 staging buffer at every later exec).
      NDArray** stagingBuffers = activeStagingBuffers_ != nullptr
          ? activeStagingBuffers_ : placeholderStagingBuffers_;
      if (stagingMaintainedThisExec_ &&
          stagingBuffers != nullptr && extIdx < numExternalInputs_) {
        NDArray* staging = stagingBuffers[extIdx];
        if (staging != nullptr && staging->dataBuffer() != nullptr
            && staging->dataBuffer()->isValid() && staging->specialBuffer() != nullptr) {
          return staging;
        }
      }
      NDArray* extArr = externalArrays[extIdx];
      // Fallback: staging buffer unavailable → use the raw external input. If that array was
      // closed/freed between replays, the captured graph can hold a stale address (close-weight err700 lead).
      return extArr;
    }
    return nullptr;
  }
  // Post-graph-replay fixup: ticks device actuality on all slot outputs (graph
  // replay writes device memory without registerSpecialUse), then re-executes
  // any host-only ops (0 CUDA graph nodes) that were recorded but not replayed.
  // Called from all graph replay paths: capture, regular replay, frozen fast path.
  Status postGraphReplayFixup(GraphSegment& seg, NDArray** externalArrays,
                              int numExt, void* stream, const char* diagTag);

  // Range+handle-parameterized core of postGraphReplayFixup. The monolithic
  // path passes the segment's replay handle and full segment range; the
  // merged/composite replay path passes the merged group's handle and slot
  // range (its handle carries its own capture audit — task #53). Audit-driven:
  // ticks device actuality only for slots the captured graph wrote, re-executes
  // host-only/setup-only slots, and falls back LOUDLY to a blanket tick when
  // no in-range audit exists.
  Status postReplayFixupRange(GraphReplayHandle* replayHandle, int startSlot,
                              int endSlot, NDArray** externalArrays, int numExt,
                              void* stream, const char* diagTag);

  // Consolidated monolithic CUDA graph replay sequence. Replaces duplicated
  // inline replay blocks in executeSegmentWithGraph (regular replay) and
  // platformTryFrozenFastPath (monolithic branch). Sequence:
  //   1. Triton arg table refresh (if needed)
  //   2. prezeroSegmentOutputs
  //   3. cuBLAS workspace zero
  //   4. replayHandle->replay()
  //   5. Counter increments (totalGraphReplays_, seg.exec.executionCount, lastReplayExecCount)
  //   6. postGraphReplayFixup (tick device actuality + host-only re-execution)
  //   7. Optional performReplayVerify (if tritonVerifyKernels enabled)
  // Returns OK on success, KERNEL_FAILURE if replay fails.
  Status replayMonolithicGraph(GraphSegment& seg, NDArray** externalArrays,
                               int numExt, void* stream, const char* diagTag);

  // Unified pre-segment zero pass — zeroes outputs for slots with needsZeroedOutput.
  // Safe to call during CUDA graph capture (memsets get recorded into the graph).
  void prezeroSegmentOutputs(const GraphSegment& seg, void* stream);
  DspExecutionResult executeSegmentWithGraphBackend(GraphSegment& seg,
                                                    NDArray** externalArrays,
                                                    int numExt, void* stream);
  GraphBackendRequest makeGraphBackendRequest() const;
  const std::vector<GraphBackend*>& getGraphBackendCandidates();
  GraphBackendPlanningPolicy getResolvedGraphBackendPlanningPolicy();
  GraphBackendExecutionPolicy getResolvedGraphBackendExecutionPolicy();
  DspExecutionResult executeSegmentWithSpecificBackend(GraphSegment& seg,
                                                       GraphBackend* backend,
                                                       NDArray** externalArrays,
                                                       int numExt, void* stream);

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
   * Verify a monolithic replay against an isolated slot-by-slot reference plan.
   * The live plan is never mutated. Verification reports VERIFIED only after
   * positive numeric comparison; unsupported/stateful cases report UNAVAILABLE.
   * Caller must provide GPU timeline ordering.
   *
   * @param seg        The segment that was just replayed
   * @param externalArrays External input arrays
   * @param numExt     Number of external inputs
   * @param stream     CUDA stream (for sync/memcpy)
   * @param pathLabel  Label for log output (e.g. "cudagraph_replay")
   */
  void performReplayVerify(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream, const char* pathLabel);
#endif

  // ── Platform dispatch (NativeDynamicShapePlan_cuda.cu / _cuda_stubs.cpp) ──
  // These methods abstract platform-specific (CUDA vs CPU) behavior.
  // CUDA implementations are in _cuda.cu; CPU fallbacks in _cuda_stubs.cpp.
  // The linker picks the right version based on the build configuration.

  // Returns true when ANY segment has executionCount < 2, meaning it needs
  // warmup before it can be considered steady-state. This is the SINGLE source
  // of truth for sync decisions after invalidateSegmentCaptures — plan-level
  // executeCount_ stays high but per-segment executionCount resets to 0.
  // Used by platformBeginExecution, platformEndExecution, and populateDerivedState.
  bool anySegmentNeedsWarmup() const;

  // Returns true when ALL replay-eligible segments have a ready replay artifact:
  // a monolithic/composite platform handle or a sealed direct graph-backend
  // compilation. Callers use this to decide whether platformTryFrozenFastPath
  // can be invoked. When false, the caller must use phaseReplay instead.
  bool allSegmentsReplayReady() const;

  // CPU/mobile steady-state readiness. Unlike allSegmentsReplayReady(), this
  // accepts exact segment-owned DIRECT_COMPILED artifacts while preserving the
  // stricter replay-handle contract used by CUDA/Vulkan paths.
  bool allFrozenDispatchUnitsReady() const;

  // Replays all segments via platform handles or sealed direct graph-backend
  // artifacts. Returns OK on success, KERNEL_FAILURE on replay error.
  // Precondition: allSegmentsReplayReady() must be true.
  Status platformTryFrozenFastPath(NDArray** externalInputs, int numExternalInputs,
                                    NDArray** requestedOutputs, int numRequestedOutputs, void* stream);
  void platformPreExecuteSetup(NDArray** externalInputs, int numExternalInputs, void* stream);
  bool platformShouldKeepSegmentCache(const GraphSegment& seg) const;
  void platformPrecompileSegments(NDArray** externalInputs, int numExternalInputs);
  bool platformBindSegmentDevice(const GraphSegment& seg);
  // Restore the plan-primary execution state (stream/workspace TLS + CUDA device) after a
  // secondary-device segment bound by platformBindSegmentDevice. No-op for single-GPU segments.
  void platformRestoreSegmentDevice();
  // Migrate internal segment inputs to the segment's target device. Callers
  // must not dispatch a segment when a source pointer, destination allocation,
  // or cross-device transfer cannot be validated.
  Status platformMigrateSegmentInputs(const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs);
  void platformCleanupMigratedInputs();
  /**
   * When op-segment sharding places the producer of a requested output on a secondary
   * device (targetDeviceId > 0), the Java side can only access device-0 memory.
   * This function checks whether `arr` lives on a non-primary device and, if so,
   * asynchronously copies it to device-0 via cudaMemcpyPeerAsync (non-blocking on the
   * device-0 stream).  The returned pointer is a NEW NDArray that the caller (Java)
   * owns and must eventually deallocate; `arr` (and outputSlots_[slotIdx]) are NOT
   * modified so the plan can continue using the device-N buffer on subsequent steps.
   *
   * On CPU or when the output is already on device-0, returns `arr` unchanged (no copy).
   * On error the original `arr` is returned so callers always get a valid pointer.
   */
  NDArray* platformGetOutputForDevice0(NDArray* arr, int slotIdx, int outputIdx);
  bool platformShouldUseGraph(const GraphSegment& seg);
  // REMOVED: platformPreSegmentExec — was a parallel sync path that competed
  // with performPreReplaySync. All sync now goes through performPreReplaySync
  // (called from dispatchSegment), tracked via PreReplaySyncPhase.
  Status platformExecuteSegmentWithBackends(GraphSegment& seg, NDArray** externalInputs,
                                             int numExternalInputs, void* stream, bool& usedGraph);
  Status platformCheckPostSegment(GraphSegment& seg);
  void platformCleanupSegmentForRebuild(GraphSegment& seg);
  void platformFreePlanResources();
  void platformFreeCaptureWorkspace();
  int platformCountCapturedGraphSegments() const;
  void platformMaybeSplitIfEnabled();

  // ── Additional platform dispatch (extracted from NativeDynamicShapePlan.cpp) ──
#if defined(SD_VULKAN)
  void* platformGetExecutionStream() const;
#endif
  void* platformBeginExecution(void* stream, bool frozen, int execCount);
  void platformEndExecution(void* executionState, void* stream, bool frozen, int execCount);
  void platformDumpExternalInputDiagnostics(NDArray** ext, int numExt, int execCount);
  void platformDumpExtInputGpuValues(NDArray* arr, int extIdx, int execCount, void* stream);
  void platformClearCastCache();
  void platformSetDeterministicCublas(bool enable);
  void platformSetupSteadyStateCuda(void* execCtxVoid, void* stream);
  void platformTeardownSteadyStateCuda(void* execCtxVoid, void* stream, void* prevDspStream);
  void platformResetGapCaches();
  void platformResetBatchD2D();
  void platformPostSegmentPoolManagement(bool frozen, int execCount);
  void platformDumpLogitsArgmax(int execCount, void* stream);
  void platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream);
  void platformPreReplayPoolStats(size_t& poolUsedOut, size_t& poolReservedOut);
  void platformPostReplayPoolManagement(size_t poolUsedPre, bool frozen, int execCount);
  void platformTraceSlotValues(const GraphSegment& seg, void* stream, int execCount);
  SelectedBackend platformResolveBackend(bool isGraphCapture) const;
  SelectedBackend platformResolvePortableReplayBackend() const;
  bool platformShouldBreakSegmentAtTraitBoundary(int currIdx, int prevIdx) const;
  size_t platformEstimateCaptureBudget() const;
  /** Estimated capture-workspace bytes a segment needs: aligned sum of its slots' output
   *  buffers (allocated from the workspace during cudaStreamCapture) plus a temporaries
   *  margin. Used to size the capture workspace adaptively for large single-segment models. */
  size_t platformEstimateSegmentCaptureBytes(int startSlot, int endSlot) const;
  void platformReleaseSegmentGpuResources();
  void platformMigrateWeightsAndClearCaches();

  /**
   * Pin a GPU address baked into a live CUDA graph segment to prevent pool reuse.
   * CUDA build: delegates to CudaMemoryPool::pinGraphBakedAddress(ptr, deviceId).
   * CPU build: no-op.
   */
  void platformPinGraphBakedAddress(void* ptr, int deviceId);

  /**
   * Flush all pending graph-baked address pins by calling unpinGraphBakedAddress
   * for each entry in graphPinnedAddrs_, then clearing the vector.
   * CUDA build: issues deferred cudaFreeAsync via CudaMemoryPool::unpinGraphBakedAddress.
   * CPU build: no-op.
   * @param stream CUDA stream to use for deferred free (nullptr = stream 0)
   */
  void platformFlushGraphBakedPins(void* stream);

  /**
   * Pin every device buffer a sealed segment will re-read on a later replay/re-exec, at SEAL
   * time, so no free()/SDVariable.close()/rebind/pool-reuse can dangle it (→ err700). Covers two
   * hazard classes: (a) VIEW slot outputs + SOURCE_VARIABLE weight inputs — externally-owned,
   * pinned in EVERY mode including slot-by-slot (NOT_FUSIBLE); (b) OWNED intermediate outputs —
   * pinned only when pinOwnedOutputs (a captured graph baked their raw address). externalArrays/
   * numExt are the segment's external table (resolves external-encoded weight inputs, identical
   * to the slot executor). Records into graphPinnedAddrs_ (released at teardown by
   * platformFlushGraphBakedPins). Idempotent per address and sealed segment.
   */
  void pinSegmentGraphBakedSlots(GraphSegment& seg, NDArray** externalArrays,
                                 int numExt, bool pinOwnedOutputs);

  // ── Slot execution platform dispatch (NativeDynamicShapePlan_slotexec_cuda.cu / _cuda_stubs.cpp) ──
  // These abstract backend-specific slot execution behavior (typed dispatch,
  // prezero, actuality, buffer validation, epilogue state, verify logging).
  Status platformExecuteSlot(const NativeSlot& slot, Context& context);
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

  // ── Device graph execution (NativeDynamicShapePlan_gpubackend.cpp) ──
  Status executeSegmentWithGpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  void clearGpuBackendFailedCache();

  // ── Segment lifecycle dispatch methods ──
  // Each handles one phase of the segment state machine.
  // Called from executeSegmentWithGpuGraph based on lifecycleState.
  Status segDispatchWarmup(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream);
  // Run one transactional functional pass after segment-local replay identity
  // drift (shape boundary, baked create value, or equivalent). The helper owns
  // invalidation, coherency, shape replacement authorization, lifecycle
  // advancement, and capture-readiness credit.
  Status runBoundedSegmentRebuildWarmup(
      GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream,
      const char* reason);
  // Recover a segment whose replay address validation failed before any replay
  // unit launched. Owns the complete segment-local invalidate -> warmup ->
  // immediate recompile transaction so every caller observes one atomic result.
  Status rebuildSegmentAfterPreLaunchReplayDrift(
      GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream,
      const char* reason);
  // segDispatchCompile — handles one compile cycle for a segment.
  // segShapeKey is passed by reference: the shape-change recompile path
  // recomputes the key after a mini-warmup and writes the new value back
  // so the caller's shapeKeyState.markCompiled() uses the correct key.
  Status segDispatchCompile(GraphSegment& seg, NDArray** externalArrays,
                            int numExt, void* stream, LongType& segShapeKey,
                            bool& invocationSatisfiedByWarmup);

#ifdef SD_CUDA
  // segDispatchReplay — attempts composite replay for a segment that has
  // captured composite handles. Sets handled=true and returns the replay
  // Status if replay was attempted. Sets handled=false when replay
  // conditions are not met (caller falls through to capture/direct).
  Status segDispatchReplay(GraphSegment& seg, NDArray** externalArrays,
                           int numExt, void* stream,
                           bool allowTritonCudaGraphReplay,
                           bool createValuesStable, bool extAddrsStable,
                           LongType segShapeKey, const char* backendName,
                           bool& handled);
  void markReplayInvariantInvalidatedForDispatch(GraphSegment& seg,
                                                 const char* reason,
                                                 bool& extAddrsStable,
                                                 bool& hasReplayHandle,
                                                 bool& replayHandleNull,
                                                 bool& hasComposite);
  void markBatchD2DInvalidated(GraphSegment& seg, const char* reason);
  void markGapOpsCapturedInGraph(GraphSegment& seg, bool captured, const char* reason);
  void markStaleGapGraphInvalidatedForDispatch(GraphSegment& seg,
                                               const char* reason,
                                               int captureMinExec,
                                               bool& extAddrsStable,
                                               bool& hasReplayHandle,
                                               bool& replayHandleNull,
                                               bool& isTritonCompiled);

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

  // Resolver-ordered implementations available for this execution request.
  std::vector<GraphBackend*> graphBackendCandidates_;
  bool graphBackendCandidatesBuilt_ = false;

  // Backend priority order (user-configurable)
  std::vector<std::string> backendPriority_;

#ifdef SD_CUDA
  // Plan-owned CUDA stream: isolates this plan's execution and capture
  // from other threads. Without this, all plans share the LaunchContext
  // default stream, causing cross-thread capture poisoning when one
  // thread captures while another thread's Java syncToDevice() runs
  // cudaMemcpyAsync on the same stream.
  cudaStream_t* ownedStream_ = nullptr;
  int ownedStreamDeviceId_ = -1;

  // Pre-allocated cuBLAS workspace for GPU graph capture.
  void* cublasWorkspaceBuffer_ = nullptr;
  size_t cublasWorkspaceSize_ = 0;
  int cublasWorkspaceDevice_ = -1;  // device on which cublasWorkspaceBuffer_ was allocated
  void ensureCublasWorkspace(size_t minBytes);
  void setCublasWorkspaceForCapture(void* stream);
  void setCublasWorkspaceForWarmup();
  void restoreCublasWorkspaceAfterCapture(void* stream);
  void abortCapture(GraphSegment& seg, bool freeHostPtrs, bool didPushCtx, int captureDevice,
                    cudaStream_t prevCaptureStream,
                    const std::vector<SlotPhase>& savedSlotPhases,
                    void* stream, bool preserveCompiledBackend = false);
#endif

#if defined(SD_VULKAN)
  // Plan-owned Vulkan stream: the Vulkan analogue of ownedStream_ above.
  // Mutable so getExecutionStream() can create it before the first execution.
  mutable VulkanExecutionStream* ownedVulkanStream_ = nullptr;
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
  std::vector<void*> deviceManagedExternalInputAddrs_;
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
  // Per-gap-unit cache keyed by the complete [startSlot, endSlot] interval.
  // Schedule recapture can retain a start slot while expanding or shrinking the
  // gap, so startSlot alone is not a stable cache identity.
  std::unordered_map<uint64_t, std::vector<ActiveGapSlot>> cachedActiveGapSlotsMap_;
  std::unordered_set<uint64_t> activeGapSlotsCachedSet_;

  // Plan-owned capture workspace: segments of this plan share one arena
  // because they execute sequentially. The arena is never shared with another
  // live plan: CUDA graph nodes retain addresses and capture-time state within
  // it for the entire cached-plan lifetime.
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

  // ── Batched GEMM optimization ──────────────────────────────────────────
  // Groups matmul slots with identical dimensions, transpose flags, and
  // A/B/C dtypes into single cublasGemmBatchedEx calls, reducing CUDA graph node count.
  // For SmolDocling (24 layers × 9 matmuls), reduces ~211 → ~120 matmul nodes.
  // Declared unconditionally; on CPU builds the vectors stay empty and
  // the implementation functions are no-ops.
  struct BatchedGemmGroup {
    std::vector<int> slotIndices;  // matmul slot indices in this group (non-consecutive OK)
    int triggerSlot;    // first slot in group — execution happens here
    int M, N, K;        // shared dimensions
    int transA, transB;
    sd::DataType aType;
    sd::DataType bType;
    sd::DataType cType;
    void** d_A_ptrs;    // device pointer array
    void** d_B_ptrs;
    void** d_C_ptrs;
    void** h_A_ptrs;    // pinned host staging
    void** h_B_ptrs;
    void** h_C_ptrs;
    int maxBatchSize;   // allocated capacity
    bool ptrStable = false;  // true when H2D pointer arrays match device-side
    // Persistent cast scratch for mixed-type (FLOAT32×HALF) groups.
    // Allocated once in prepareBatchedGemmDevice, reused every step.
    void* castScratch = nullptr;       // device buffer for casted operands
    size_t castScratchBytes = 0;       // allocated size
    void** d_castPtrs = nullptr;       // device pointer array for casted batch members
    bool needsCast = false;            // true when aType!=bType and cast is required
    int castScratchDevice = -1;        // device on which castScratch/d_castPtrs were allocated
  };
  std::vector<BatchedGemmGroup> batchedGemmGroups_;
  // Maps slot index → index into batchedGemmGroups_ (-1 if not part of a group)
  std::vector<int> slotToBatchedGemmGroup_;

  const LongType* resolveInputShapeInfo(int srcIdx, NDArray** externalArrays, int numExt) const;
  void detectBatchedGemmGroups(NDArray** externalArrays, int numExt);
  void reconcileSlotDispatchAfterMerge(const ReplaySchedule& sched);
  // stream is void* (cudaStream_t on CUDA, nullptr on CPU)
  void prepareBatchedGemmDevice(void* stream);
  Status executeBatchedGemmGroup(int groupIdx, NDArray** externalArrays, int numExt, void* stream);
  void freeBatchedGemmResources();
};

inline bool NativeDynamicShapePlan::isPassivated() const { return passivated_; }

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
