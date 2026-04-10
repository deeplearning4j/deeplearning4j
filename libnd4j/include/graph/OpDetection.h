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
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Consolidated op detection and grouping utilities for DSP execution.
// Centralizes matmul classification, dimension extraction, dependency analysis,
// and signature-based grouping that were previously scattered across plan files.
//

#ifndef LIBND4J_OP_DETECTION_H
#define LIBND4J_OP_DETECTION_H

#include <helpers/shape.h>
#include <array/ArrayOptions.h>
#include <ops/declarable/OpDescriptor.h>

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

// ─── Forward declarations ──────────────────────────────────────────────────
struct NativeSlot;

namespace op_detection {

// ─── Op classification ─────────────────────────────────────────────────────

// Check if an op is a matmul-family op via OpDescriptor traits.
// Preferred overload — uses the centralized OpTraitTable instead of brittle string sets.
SD_INLINE bool isMatmulOp(sd::ops::DeclarableOp* op) {
  return op && op->getOpDescriptor() &&
         op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_MATMUL);
}

// String-based fallback for contexts without a resolved DeclarableOp pointer.
// Prefer the DeclarableOp* overload when the op pointer is available.
SD_INLINE bool isMatmulOp(const std::string& opName) {
  return opName == "matmul" || opName == "mmul" || opName == "Mmul" ||
         opName == "tensormmul" || opName == "Tensormmul";
}

// ─── Matmul dimension extraction ───────────────────────────────────────────
//
// Extracts M, N, K, transpose flags from matmul input shape info pointers.
// Handles rank-2 and rank-3 (with batch dim) matmuls. For rank-3, uses only
// the last 2 dims.
//

SD_INLINE bool extractMatmulDims(const NativeSlot& slot,
                                  const LongType* shapeA, const LongType* shapeB,
                                  int& M, int& N, int& K,
                                  int& transA, int& transB,
                                  DataType& dtype) {
  if (shapeA == nullptr || shapeB == nullptr) return false;

  int rA = shape::rank(shapeA);
  int rB = shape::rank(shapeB);
  if (rA < 2 || rA > 3 || rB < 2 || rB > 3) return false;

  transA = (slot.args.numIArgs >= 1) ? static_cast<int>(slot.args.iArgs[0]) : 0;
  transB = (slot.args.numIArgs >= 2) ? static_cast<int>(slot.args.iArgs[1]) : 0;

  const LongType* dimsA = shape::shapeOf(shapeA);
  const LongType* dimsB = shape::shapeOf(shapeB);

  if (transA) {
    M = static_cast<int>(dimsA[rA - 1]);
    K = static_cast<int>(dimsA[rA - 2]);
  } else {
    M = static_cast<int>(dimsA[rA - 2]);
    K = static_cast<int>(dimsA[rA - 1]);
  }

  if (transB) {
    N = static_cast<int>(dimsB[rB - 2]);
  } else {
    N = static_cast<int>(dimsB[rB - 1]);
  }

  dtype = ArrayOptions::dataType(shapeA);

  // Only batch floating point matmuls (not shape computation ops using INT64 etc.)
  if (dtype != FLOAT32 && dtype != DOUBLE && dtype != HALF && dtype != BFLOAT16) return false;

  return true;
}

// ─── Matmul signature for grouping ─────────────────────────────────────────
//
// Two matmuls can share the same batched GEMM iff they have the same
// (M, N, K, transA, transB, dtype) signature.
//

struct MatmulSig {
  int M, N, K, transA, transB;
  DataType dtype;

  bool operator==(const MatmulSig& o) const {
    return M == o.M && N == o.N && K == o.K &&
           transA == o.transA && transB == o.transB && dtype == o.dtype;
  }
};

struct MatmulSigHash {
  size_t operator()(const MatmulSig& s) const {
    size_t h = std::hash<int>()(s.M);
    h ^= std::hash<int>()(s.N) << 1;
    h ^= std::hash<int>()(s.K) << 2;
    h ^= std::hash<int>()(s.transA) << 3;
    h ^= std::hash<int>()(s.transB) << 4;
    h ^= std::hash<int>()(static_cast<int>(s.dtype)) << 5;
    return h;
  }
};

// ─── Dependency analysis ───────────────────────────────────────────────────
//
// Used during batched GEMM detection to verify that grouped matmuls do not
// have transitive data dependencies on each other (which would prevent batching).
//

// Check if stepJ transitively depends on any step in targetSteps.
// BFS traces backward through inputSourceIndices → outputSlotToStep.
// Limited to steps >= minTarget for efficiency.
SD_INLINE bool hasTransitiveDependency(
    const NativeSlot* slots, int numSlots, int totalOutputSlots,
    const std::vector<int>& outputSlotToStep,
    int stepJ, const std::unordered_set<int>& targetSteps, int minTarget) {

  static constexpr int MAX_BFS_NODES = 2000;
  std::unordered_set<int> visited;
  std::queue<int> queue;

  // Seed BFS with stepJ's direct inputs
  for (int k = 0; k < slots[stepJ].wiring.numInputs; k++) {
    int src = slots[stepJ].wiring.inputSourceIndices[k];
    if (src < 0 || src >= totalOutputSlots) continue;
    int producerStep = outputSlotToStep[src];
    if (producerStep < 0) continue;
    if (targetSteps.count(producerStep)) return true;
    if (producerStep >= minTarget && visited.insert(producerStep).second) {
      queue.push(producerStep);
    }
  }

  while (!queue.empty() && (int)visited.size() < MAX_BFS_NODES) {
    int cur = queue.front();
    queue.pop();

    for (int k = 0; k < slots[cur].wiring.numInputs; k++) {
      int src = slots[cur].wiring.inputSourceIndices[k];
      if (src < 0 || src >= totalOutputSlots) continue;
      int producerStep = outputSlotToStep[src];
      if (producerStep < 0) continue;
      if (targetSteps.count(producerStep)) return true;
      if (producerStep >= minTarget && visited.insert(producerStep).second) {
        queue.push(producerStep);
      }
    }
  }

  return false;
}

// Check if all inputs of a slot are available before firstSlot
// (i.e., produced by steps < firstSlot, or external inputs).
SD_INLINE bool allInputsAvailableBefore(
    const NativeSlot& slot, int firstSlot, int totalOutputSlots,
    const std::vector<int>& outputSlotToStep) {

  for (int k = 0; k < slot.wiring.numInputs; k++) {
    int src = slot.wiring.inputSourceIndices[k];
    if (src < 0) continue;  // external, always available
    if (src >= totalOutputSlots) return false;
    int producerStep = outputSlotToStep[src];
    if (producerStep < 0) return false;
    if (producerStep >= firstSlot) return false;
  }
  return true;
}

}  // namespace op_detection
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_OP_DETECTION_H
