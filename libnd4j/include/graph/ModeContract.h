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

#include <graph/NativeDynamicShapePlan.h>

namespace sd {
namespace graph {

struct ModeContract {
  // ═══════════════════════════════════════════════════════════════════════════
  // Compilation & Backend Selection
  // ═══════════════════════════════════════════════════════════════════════════

  /** Mode requires a JIT/compilation backend (Triton, NVRTC, PTX, etc.). */
  bool requiresCompilation;

  /** Mode needs a gpuGraphBackend_ (JIT compiler). False for pure-replay modes. */
  bool needsJitBackend;

  // ═══════════════════════════════════════════════════════════════════════════
  // Graph Capture & Replay
  // ═══════════════════════════════════════════════════════════════════════════

  /** Mode uses CUDA/HIP/Metal/Vulkan graph capture for segment replay. */
  bool usesGraphCapture;

  /** Mode allows the frozen graph fast path (input-stable skip refresh). */
  bool allowsFrozenFastPath;

  // ═══════════════════════════════════════════════════════════════════════════
  // Fallback & Assertions
  // ═══════════════════════════════════════════════════════════════════════════

  /** Mode tolerates capturable segments executing slot-by-slot without assertion. */
  bool allowsFallback;

  /** Mode tolerates phase stalls (segments stuck in non-REPLAYING state). */
  bool allowsPhaseStall;

  // ═══════════════════════════════════════════════════════════════════════════
  // cuBLAS / Compute Determinism
  // ═══════════════════════════════════════════════════════════════════════════

  /** Mode requires deterministic cuBLAS (PEDANTIC_MATH, no workspace, no cublasLt). */
  bool requiresDeterministicCublas;

  // ═══════════════════════════════════════════════════════════════════════════
  // Execution Path Control
  // ═══════════════════════════════════════════════════════════════════════════

  /** Mode is purely slot-by-slot with no segment optimization. */
  bool isSlotBySlot;

  /** Mode is shape-inference-only (no actual op execution). */
  bool isShapeInferenceOnly;

  /** Mode forces full stream sync on every frozen execute (for diagnostics). */
  bool forcesSyncOnFrozen;

  // ═══════════════════════════════════════════════════════════════════════════
  // Factory
  // ═══════════════════════════════════════════════════════════════════════════

  static ModeContract forMode(GraphExecutionMode mode) {
    switch (mode) {
      case GraphExecutionMode::GEM_SLOT_BY_SLOT:
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          false,
          /* allowsFrozenFastPath */      false,
          /* allowsFallback */            true,
          /* allowsPhaseStall */          true,
          /* requiresDeterministicCublas */ true,
          /* isSlotBySlot */              true,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        true,
        };

      case GraphExecutionMode::GEM_AUTO:
        return {
          /* requiresCompilation */       false, // AUTO resolves dynamically
          /* needsJitBackend */           true,  // tries to find one
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            true,
          /* allowsPhaseStall */          true,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_CUDA_GRAPHS:
        return {
          /* requiresCompilation */       false, // captures inline, no JIT
          /* needsJitBackend */           false,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ true,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        true,
        };

      case GraphExecutionMode::GEM_TRITON:
        return {
          /* requiresCompilation */       true,
          /* needsJitBackend */           true,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_NVRTC_JIT:
        return {
          /* requiresCompilation */       true,
          /* needsJitBackend */           true,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_PTX_JIT:
        return {
          /* requiresCompilation */       true,
          /* needsJitBackend */           true,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_HIP_GRAPHS:
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_LEVELZERO:
      case GraphExecutionMode::GEM_VULKAN:
      case GraphExecutionMode::GEM_METAL:
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          true,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_MLX:
      case GraphExecutionMode::GEM_ARM_HYBRID:
      case GraphExecutionMode::GEM_NNAPI:
      case GraphExecutionMode::GEM_TPU:
      case GraphExecutionMode::GEM_HEXAGON:
      case GraphExecutionMode::GEM_OPENVINO:
        return {
          /* requiresCompilation */       true,
          /* needsJitBackend */           true,
          /* usesGraphCapture */          false,
          /* allowsFrozenFastPath */      true,
          /* allowsFallback */            false,
          /* allowsPhaseStall */          false,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              false,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_EMULATED_REPLAY:
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          false,
          /* allowsFrozenFastPath */      false,
          /* allowsFallback */            true,
          /* allowsPhaseStall */          true,
          /* requiresDeterministicCublas */ true,
          /* isSlotBySlot */              true,  // executes slot-by-slot with replay lifecycle
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        true,
        };

      case GraphExecutionMode::GEM_SHAPE_INFERENCE_ONLY:
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          false,
          /* allowsFrozenFastPath */      false,
          /* allowsFallback */            true,
          /* allowsPhaseStall */          true,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              true,
          /* isShapeInferenceOnly */      true,
          /* forcesSyncOnFrozen */        false,
        };

      case GraphExecutionMode::GEM_TVM:
      default:
        // Deprecated/unknown → safe fallback: slot-by-slot semantics
        return {
          /* requiresCompilation */       false,
          /* needsJitBackend */           false,
          /* usesGraphCapture */          false,
          /* allowsFrozenFastPath */      false,
          /* allowsFallback */            true,
          /* allowsPhaseStall */          true,
          /* requiresDeterministicCublas */ false,
          /* isSlotBySlot */              true,
          /* isShapeInferenceOnly */      false,
          /* forcesSyncOnFrozen */        false,
        };
    }
  }

  /**
   * Convenience: build from int (for PlanExecutionContext which stores mode as int).
   */
  static ModeContract forMode(int modeInt) {
    return forMode(static_cast<GraphExecutionMode>(modeInt));
  }
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_MODE_CONTRACT_H
