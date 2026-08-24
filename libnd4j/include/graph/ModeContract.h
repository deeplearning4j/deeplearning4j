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

/**
 * ModeContract — single source of truth for GraphExecutionMode semantics.
 * To add a new mode: add a case to forMode().
 */

#ifndef LIBND4J_MODE_CONTRACT_H
#define LIBND4J_MODE_CONTRACT_H

// Forward-declare GraphExecutionMode so ModeContract.h is self-contained.
// Full definition in NativeDynamicShapePlan.h — any TU calling forMode()
// must include that header (which it always does, since the plan is the
// only consumer of ModeContract).
namespace sd { namespace graph { enum class GraphExecutionMode : int; } }

namespace sd {
namespace graph {

struct ModeContract {
  // ═══════════════════════════════════════════════════════════════════════════
  // Compilation & Backend Selection
  // ═══════════════════════════════════════════════════════════════════════════

  // All fields default to false — each mode's factory only sets what's true.
  // Adding a field never silently breaks existing modes.

  bool requiresCompilation = false;
  bool needsJitBackend = false;
  bool usesGraphCapture = false;
  bool allowsFrozenFastPath = false;
  bool forceSyncDuringCapture = false;
  bool skipFrozenConstsDuringCapture = false;
  bool allowsFallback = false;
  bool allowsPhaseStall = false;
  bool requiresDeterministicCublas = false;
  bool isSlotBySlot = false;
  bool isShapeInferenceOnly = false;
  bool forcesSyncOnFrozen = false;
  bool requiresShapePrePass = false;
  bool requiresSuccessfulShapePrePass = false;

  // ═══════════════════════════════════════════════════════════════════════════
  // Factory
  // ═══════════════════════════════════════════════════════════════════════════

  // Primary factory — uses int to avoid dependency on GraphExecutionMode enum values.
  // Int values match GraphExecutionMode::GEM_* (defined in NativeDynamicShapePlan.h).
  static ModeContract forMode(int mode) {
    ModeContract c{};
    switch (mode) {
      case 1: // GEM_SLOT_BY_SLOT
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        // SLOT_BY_SLOT must use the same cuBLAS math mode as DSP replay modes
        // so parity tests (SLOT_BY_SLOT vs CUDA_GRAPHS/TRITON/AUTO) produce
        // matching results.  CUBLAS_PEDANTIC_MATH disables tensor-core fast
        // paths that reorder FP16 accumulation — without this, SLOT_BY_SLOT
        // uses CUBLAS_DEFAULT_MATH (tensor cores) while DSP modes use PEDANTIC,
        // causing maxDiff=48-113 for mixed-precision FP16 GEMM.
        c.requiresDeterministicCublas = true;
        c.isSlotBySlot = true;
        c.forcesSyncOnFrozen = true;
        return c;

      case 0: // GEM_AUTO
        c.needsJitBackend = true;
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.forceSyncDuringCapture = true;
        c.skipFrozenConstsDuringCapture = true;
        c.requiresDeterministicCublas = true;
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        return c;

      case 2: // GEM_CUDA_GRAPHS
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.requiresDeterministicCublas = true;
        c.forcesSyncOnFrozen = true;
        return c;

      case 5: // GEM_TRITON
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.forceSyncDuringCapture = true;
        c.skipFrozenConstsDuringCapture = true;
        c.requiresDeterministicCublas = true;
        return c;

      case 3: // GEM_NVRTC_JIT
      case 4: // GEM_PTX_JIT
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.forceSyncDuringCapture = true;
        c.skipFrozenConstsDuringCapture = true;
        // JIT modes still route gap ops and not-yet-compiled slots through
        // cuBLAS (async kernel compilation races the replay lifecycle).
        // Without PEDANTIC math those GEMMs use tensor-core fast paths while
        // the SLOT_BY_SLOT reference runs PEDANTIC — a deterministic
        // 1e-3..1e-2 drift on norm-heavy graphs that appears only when the
        // compile race is lost (load-dependent: bgeEncoder/attnSingleLayer
        // replay-accuracy failures in full-suite runs, task #53).
        c.requiresDeterministicCublas = true;
        return c;

      case 9: // GEM_HIP_GRAPHS
        // HipGraphBackend participates in the same resolver and lowering
        // lifecycle as TPU/Hexagon/Triton. It captures islands in its backend
        // implementation, unlike
        // GEM_CUDA_GRAPHS whose capture lives in the replay-handle machinery.
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.forceSyncDuringCapture = true;
        c.skipFrozenConstsDuringCapture = true;
        // Gap ops between islands run through cuBLAS/hipBLAS while islands
        // replay — same determinism reasoning as the NVRTC/PTX JIT modes.
        c.requiresDeterministicCublas = true;
        return c;

      case 10: // GEM_LEVELZERO
      case 12: // GEM_METAL
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        return c;

      case 11: // GEM_VULKAN
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.requiresShapePrePass = true;
        c.requiresSuccessfulShapePrePass = true;
        return c;

      case 6:  // GEM_MLX
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.allowsFrozenFastPath = true;
        return c;

      case 7:  // GEM_ARM_HYBRID
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.allowsFrozenFastPath = true;
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        return c;

      case 8:  // GEM_NNAPI
      case 13: // GEM_TPU
      case 14: // GEM_HEXAGON
      case 15: // GEM_OPENVINO
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.allowsFrozenFastPath = true;
        return c;

      case 17: // GEM_EMULATED_REPLAY
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        c.requiresDeterministicCublas = true;
        c.isSlotBySlot = true;
        c.forcesSyncOnFrozen = true;
        return c;

      case 19: // GEM_PORTABLE_REPLAY
        // Replay-only mode: select an integrated native recorder when one is
        // available, otherwise use the functional recorder. Never selects a
        // Triton/NVRTC/PTX compiler backend.
        c.usesGraphCapture = true;
        c.allowsFrozenFastPath = true;
        c.requiresDeterministicCublas = true;
        c.forcesSyncOnFrozen = true;
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        return c;

      case 20: // GEM_ONEDNN
        c.requiresCompilation = true;
        c.needsJitBackend = true;
        c.allowsFrozenFastPath = true;
        c.requiresSuccessfulShapePrePass = true;
        return c;

      case 18: // GEM_SHAPE_INFERENCE_ONLY
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        c.isSlotBySlot = true;
        c.isShapeInferenceOnly = true;
        return c;

      case 16: // GEM_TVM (deprecated)
      default:
        c.allowsFallback = true;
        c.allowsPhaseStall = true;
        c.isSlotBySlot = true;
        return c;
    }
  }

  // Typed overload — casts enum to int and delegates.
  static ModeContract forMode(GraphExecutionMode mode) {
    return forMode(static_cast<int>(mode));
  }

  /**
   * Human-readable mode name for logging. Returns a short string like
   * "SLOT_BY_SLOT", "CUDA_GRAPHS", "TRITON", etc.
   * Takes int to avoid circular dependency with NativeDynamicShapePlan.h
   * (which defines GraphExecutionMode). Callers cast: modeName(static_cast<int>(mode)).
   */
  static const char* modeName(int modeInt) {
    switch (modeInt) {
      case 0: return "AUTO";
      case 1: return "SLOT_BY_SLOT";
      case 2: return "CUDA_GRAPHS";
      case 3: return "NVRTC_JIT";
      case 4: return "PTX_JIT";
      case 5: return "TRITON";
      case 6: return "MLX";
      case 7: return "ARM_HYBRID";
      case 8: return "NNAPI";
      case 9: return "HIP_GRAPHS";
      case 10: return "LEVELZERO";
      case 11: return "VULKAN";
      case 12: return "METAL";
      case 13: return "TPU";
      case 14: return "HEXAGON";
      case 15: return "OPENVINO";
      case 16: return "TVM";
      case 17: return "EMULATED_REPLAY";
      case 18: return "SHAPE_INFERENCE_ONLY";
      case 19: return "PORTABLE_REPLAY";
      case 20: return "ONEDNN";
      default: return "UNKNOWN";
    }
  }
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_MODE_CONTRACT_H
