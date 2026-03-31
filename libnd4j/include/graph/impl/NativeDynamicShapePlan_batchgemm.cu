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

// Batched GEMM optimization for CUDA graph node reduction.
//
// Groups matmul slots with identical (M,N,K,transA,transB,dtype) into single
// cublasGemmBatchedEx calls. Uses dependency analysis to ensure only truly
// independent matmuls are batched together.
//
// Execution strategy: the FIRST member in each group is the trigger.
// When the trigger slot is reached, the entire batch executes and outputs
// for ALL members are populated. Non-first members are then skipped when
// reached, since their outputs are already computed.
//
// This correctly handles non-consecutive matmuls with intervening ops
// (reshapes, relus, etc.) that consume earlier members' outputs.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <array/ArrayOptions.h>
#include <system/Environment.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <exceptions/cuda_exception.h>

#include <queue>
#include <unordered_set>

// cuBLAS workspace for graph capture (from MmulHelper.cu)
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;
extern SD_TLS_EXPORT thread_local bool tl_graphExecutionActive;

namespace sd {
namespace graph {

// ── Detection: scan frozen slots for batchable matmul groups ─────────────────

static bool isMatmulOp(const std::string& opName) {
  return opName == "matmul" || opName == "mmul" || opName == "Mmul" ||
         opName == "tensormmul" || opName == "Tensormmul";
}

// Extract matmul dimensions from shape info pointers (no NDArray* needed).
// Handles rank-2 and rank-3 (with batch dim) matmuls. For rank-3, uses only the last 2 dims.
static bool extractMatmulDimsFromShape(const NativeSlot& slot,
                                        const LongType* shapeA, const LongType* shapeB,
                                        int& M, int& N, int& K, int& transA, int& transB,
                                        DataType& dtype) {
  if (shapeA == nullptr || shapeB == nullptr) return false;

  int rA = shape::rank(shapeA);
  int rB = shape::rank(shapeB);
  if (rA < 2 || rA > 3 || rB < 2 || rB > 3) return false;

  transA = (slot.numIArgs >= 1) ? static_cast<int>(slot.iArgs[0]) : 0;
  transB = (slot.numIArgs >= 2) ? static_cast<int>(slot.iArgs[1]) : 0;

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

// Signature key for grouping matmuls
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

// Resolve shape info for a matmul input given its source index.
// Priority: NDArray* (outputSlots_ / slotArrayCache_ / external) -> cachedOutputShapes on source slot.
const LongType* NativeDynamicShapePlan::resolveInputShapeInfo(
    int srcIdx, NDArray** externalArrays, int numExt) const {
  if (srcIdx >= 0 && srcIdx < numSlots_) {
    if (srcIdx < totalOutputSlots_) {
      NDArray* arr = outputSlots_[srcIdx];
      if (arr == nullptr) arr = slotArrayCache_[srcIdx];
      if (arr != nullptr) return arr->shapeInfo();
    }
    const NativeSlot& srcSlot = slots_[srcIdx];
    if (srcSlot.shapeCacheValid() && !srcSlot.cachedOutputShapes.empty() &&
        srcSlot.cachedOutputShapes[0] != nullptr) {
      return srcSlot.cachedOutputShapes[0];
    }
  } else if (srcIdx < 0) {
    int extIdx = -(srcIdx + 1);
    if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
      return externalArrays[extIdx]->shapeInfo();
    }
  }
  return nullptr;
}

// ── Dependency analysis ──────────────────────────────────────────────────────

// Check if stepJ transitively depends on any step in targetSteps.
// BFS traces backward through inputSourceIndices -> outputSlotToStep.
// Limited to steps >= minTarget for efficiency.
static bool hasTransitiveDependency(
    const NativeSlot* slots, int numSlots, int totalOutputSlots,
    const std::vector<int>& outputSlotToStep,
    int stepJ, const std::unordered_set<int>& targetSteps, int minTarget) {

  static constexpr int MAX_BFS_NODES = 2000;
  std::unordered_set<int> visited;
  std::queue<int> queue;

  // Seed BFS with stepJ's direct inputs
  for (int k = 0; k < slots[stepJ].numInputs; k++) {
    int src = slots[stepJ].inputSourceIndices[k];
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

    for (int k = 0; k < slots[cur].numInputs; k++) {
      int src = slots[cur].inputSourceIndices[k];
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
static bool allInputsAvailableBefore(
    const NativeSlot& slot, int firstSlot, int totalOutputSlots,
    const std::vector<int>& outputSlotToStep) {

  for (int k = 0; k < slot.numInputs; k++) {
    int src = slot.inputSourceIndices[k];
    if (src < 0) continue;  // external, always available
    if (src >= totalOutputSlots) return false;
    int producerStep = outputSlotToStep[src];
    if (producerStep < 0) return false;
    if (producerStep >= firstSlot) return false;
  }
  return true;
}

// ── Main detection ───────────────────────────────────────────────────────────

void NativeDynamicShapePlan::detectBatchedGemmGroups(NDArray** externalArrays, int numExt) {
  batchedGemmGroups_.clear();
  slotToBatchedGemmGroup_.assign(numSlots_, -1);

  if (!Environment::getInstance().dspBatchedGemm()) return;

  // Build reverse map: output slot index -> producing step index
  std::vector<int> outputSlotToStep(totalOutputSlots_, -1);
  for (int i = 0; i < numSlots_; i++) {
    for (int o = 0; o < slots_[i].numOutputs; o++) {
      int outIdx = slots_[i].outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots_) {
        outputSlotToStep[outIdx] = i;
      }
    }
  }

  int totalMatmuls = 0, resolvedMatmuls = 0, dimFailMatmuls = 0;
  int fromArray = 0, fromCache = 0;
  int depRejected = 0, inputRejected = 0;

  for (auto& seg : segments_) {
    std::unordered_map<MatmulSig, std::vector<int>, MatmulSigHash> sigBuckets;

    for (int i = seg.startSlot; i <= seg.endSlot; i++) {
      NativeSlot& slot = slots_[i];
      if (!isMatmulOp(slot.opName) || slot.numInputs < 2 ||
          slot.controlFlowType != CF_NONE || slot.frozenConstantSlot()) continue;

      totalMatmuls++;

      int srcA = slot.inputSourceIndices[0];
      int srcB = slot.inputSourceIndices[1];
      const LongType* shapeA = resolveInputShapeInfo(srcA, externalArrays, numExt);
      const LongType* shapeB = resolveInputShapeInfo(srcB, externalArrays, numExt);

      if (shapeA != nullptr) {
        bool wasFromArray = false;
        if (srcA >= 0 && srcA < totalOutputSlots_) {
          NDArray* arr = outputSlots_[srcA];
          if (arr == nullptr) arr = slotArrayCache_[srcA];
          wasFromArray = (arr != nullptr);
        } else if (srcA < 0) {
          wasFromArray = true;
        }
        if (wasFromArray) fromArray++; else fromCache++;
      }

      int M, N, K, transA, transB;
      DataType dtype;
      if (!extractMatmulDimsFromShape(slot, shapeA, shapeB, M, N, K, transA, transB, dtype)) {
        dimFailMatmuls++;
        if (dimFailMatmuls <= 10) {
          int rA = shapeA ? (int)shape::rank(shapeA) : -1;
          int rB = shapeB ? (int)shape::rank(shapeB) : -1;
          DSP_DIAG(EXECUTE, "  matmul slot %d: dim extract failed (shapeA=%p rankA=%d shapeB=%p rankB=%d srcA=%d srcB=%d)",
                    i, shapeA, rA, shapeB, rB, srcA, srcB);
        }
        continue;
      }

      resolvedMatmuls++;
      MatmulSig sig{M, N, K, transA, transB, dtype};
      sigBuckets[sig].push_back(i);
    }

    // For each bucket, split into independent sub-groups using dependency analysis.
    // Two matmuls can be in the same group iff:
    //   1. Neither transitively depends on the other's output
    //   2. All inputs of every member are available at the first member's position
    //      (produced by steps before the first member, or external)
    for (auto& [sig, slotVec] : sigBuckets) {
      if (slotVec.size() < 2) continue;

      std::sort(slotVec.begin(), slotVec.end());

      // Build independent sub-groups greedily
      std::vector<std::vector<int>> subGroups;

      for (int slot : slotVec) {
        bool addedToExisting = false;

        for (auto& sg : subGroups) {
          int firstSlot = sg.front();

          // Check 1: all inputs of 'slot' must be available at firstSlot's position
          if (!allInputsAvailableBefore(slots_[slot], firstSlot, totalOutputSlots_, outputSlotToStep)) {
            inputRejected++;
            continue;
          }

          // Check 2: no transitive dependency from 'slot' to any existing member
          std::unordered_set<int> existingMembers(sg.begin(), sg.end());
          int minExisting = sg.front();

          if (hasTransitiveDependency(slots_, numSlots_, totalOutputSlots_,
                                       outputSlotToStep, slot, existingMembers, minExisting)) {
            depRejected++;
            continue;
          }

          sg.push_back(slot);
          addedToExisting = true;
          break;
        }

        if (!addedToExisting) {
          subGroups.push_back({slot});
        }
      }

      // Create BatchedGemmGroup for each sub-group with >= 2 members
      for (auto& sg : subGroups) {
        if (sg.size() < 2) continue;

        // Cap at 32 per batch
        for (size_t start = 0; start < sg.size(); start += 32) {
          size_t end = std::min(start + 32, sg.size());
          if (end - start < 2) continue;

          int groupIdx = static_cast<int>(batchedGemmGroups_.size());
          BatchedGemmGroup group;
          group.slotIndices.assign(sg.begin() + start, sg.begin() + end);
          group.triggerSlot = group.slotIndices.front();  // FIRST member is trigger
          group.M = sig.M;
          group.N = sig.N;
          group.K = sig.K;
          group.transA = sig.transA;
          group.transB = sig.transB;
          group.dtype = sig.dtype;
          group.d_A_ptrs = nullptr;
          group.d_B_ptrs = nullptr;
          group.d_C_ptrs = nullptr;
          group.h_A_ptrs = nullptr;
          group.h_B_ptrs = nullptr;
          group.h_C_ptrs = nullptr;
          group.maxBatchSize = static_cast<int>(end - start);
          batchedGemmGroups_.push_back(std::move(group));

          for (size_t s = start; s < end; s++) {
            slotToBatchedGemmGroup_[sg[s]] = groupIdx;
          }

          DSP_DIAG(EXECUTE, "batched GEMM group %d: %d matmuls, slots [%d..%d] M=%d N=%d K=%d transA=%d transB=%d dtype=%d trigger=%d",
                    groupIdx, (int)(end - start), (int)sg[start], (int)sg[end - 1],
                    sig.M, sig.N, sig.K, sig.transA, sig.transB, (int)sig.dtype,
                    (int)sg[start]);
        }
      }
    }
  }

  DSP_DIAG(EXECUTE, "detectBatchedGemmGroups: found %d groups from %d segments "
            "(totalMatmuls=%d resolved=%d dimFail=%d depRejected=%d inputRejected=%d "
            "fromArray=%d fromCache=%d)",
            (int)batchedGemmGroups_.size(), (int)segments_.size(),
            totalMatmuls, resolvedMatmuls, dimFailMatmuls, depRejected, inputRejected,
            fromArray, fromCache);
}

// ── Device resource allocation ───────────────────────────────────────────────

void NativeDynamicShapePlan::prepareBatchedGemmDevice(cudaStream_t stream) {
  for (auto& group : batchedGemmGroups_) {
    if (group.d_A_ptrs != nullptr) continue;  // already allocated

    int bs = group.maxBatchSize;
    size_t ptrArrayBytes = bs * sizeof(void*);

    cudaMalloc(&group.d_A_ptrs, ptrArrayBytes);
    cudaMalloc(&group.d_B_ptrs, ptrArrayBytes);
    cudaMalloc(&group.d_C_ptrs, ptrArrayBytes);
    cudaMallocHost(&group.h_A_ptrs, ptrArrayBytes);
    cudaMallocHost(&group.h_B_ptrs, ptrArrayBytes);
    cudaMallocHost(&group.h_C_ptrs, ptrArrayBytes);

    DSP_DIAG(MEMORY, "batched GEMM group: allocated %d pointer arrays (%zu bytes each)",
              bs, ptrArrayBytes);
  }
}

// ── Execute a single batched GEMM group ──────────────────────────────────────

static inline void reapplyCublasWorkspaceBG(cublasHandle_t handle) {
  if (tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
    cublasSetWorkspace(handle, tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
  }
}

Status NativeDynamicShapePlan::executeBatchedGemmGroup(
    int groupIdx, NDArray** externalArrays, int numExt, cudaStream_t stream) {

  auto& group = batchedGemmGroups_[groupIdx];
  int batchCount = static_cast<int>(group.slotIndices.size());

  // 0. Pre-populate outputSlots_ for ALL members from slot cache.
  //    This ensures downstream ops can find each member's output array.
  // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore needed

  // 1. Populate host pointer arrays from current slot inputs/outputs
  for (int b = 0; b < batchCount; b++) {
    int slotIdx = group.slotIndices[b];
    NativeSlot& slot = slots_[slotIdx];

    // Resolve input A
    NDArray* inputA = nullptr;
    {
      int src = slot.inputSourceIndices[0];
      if (src >= 0) {
        inputA = outputSlots_[src];
        if (inputA == nullptr && src < totalOutputSlots_) inputA = slotArrayCache_[src];
      } else {
        int extIdx = -(src + 1);
        if (extIdx < numExt) inputA = externalArrays[extIdx];
      }
    }

    // Resolve input B
    NDArray* inputB = nullptr;
    {
      int src = slot.inputSourceIndices[1];
      if (src >= 0) {
        inputB = outputSlots_[src];
        if (inputB == nullptr && src < totalOutputSlots_) inputB = slotArrayCache_[src];
      } else {
        int extIdx = -(src + 1);
        if (extIdx < numExt) inputB = externalArrays[extIdx];
      }
    }

    // Resolve output C
    NDArray* outputC = nullptr;
    if (slot.numOutputs >= 1) {
      int outSlotIdx = slot.outputSlotIndices[0];
      if (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots_) {
        outputC = outputSlots_[outSlotIdx];
        if (outputC == nullptr) outputC = slotArrayCache_[outSlotIdx];
      }
    }

    if (inputA == nullptr || inputB == nullptr || outputC == nullptr) {
      DSP_DIAG(EXECUTE, "batched GEMM group %d slot %d: null array (A=%p B=%p C=%p), falling back",
                groupIdx, slotIdx, inputA, inputB, outputC);
      return Status::BAD_INPUT;
    }

    inputA->syncToDevice();
    inputB->syncToDevice();

    group.h_A_ptrs[b] = inputA->specialBuffer();
    group.h_B_ptrs[b] = inputB->specialBuffer();
    group.h_C_ptrs[b] = outputC->specialBuffer();
  }

  // 2. Copy pointer arrays H2D
  size_t ptrBytes = batchCount * sizeof(void*);
  cudaMemcpyAsync(group.d_A_ptrs, group.h_A_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(group.d_B_ptrs, group.h_B_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(group.d_C_ptrs, group.h_C_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);

  // 3. Dispatch cublasGemmBatchedEx
  auto* context = LaunchContext::defaultContext();
  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());
  auto handle = reinterpret_cast<cublasHandle_t*>(context->getCublasHandle());
  cublasSetStream_v2(*handle, stream);
  reapplyCublasWorkspaceBG(*handle);

  // cuBLAS uses column-major. For row-major C = op(A) * op(B), we swap:
  //   cuBLAS_A = our B, cuBLAS_B = our A
  // Row-major M×N stored in memory = column-major N×M with lda=N.
  // transAblas controls our B, transBblas controls our A.
  cublasOperation_t transAblas = group.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transBblas = group.transA ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = group.transB ? group.K : group.N;
  int ldb = group.transA ? group.M : group.K;
  int ldc = group.N;

  cublasStatus_t status;

  if (group.dtype == DOUBLE) {
    double alpha = 1.0, beta = 0.0;
    status = cublasDgemmBatched(*handle, transAblas, transBblas,
                                 group.N, group.M, group.K,
                                 &alpha, (const double**)group.d_B_ptrs, lda,
                                 (const double**)group.d_A_ptrs, ldb,
                                 &beta, (double**)group.d_C_ptrs, ldc,
                                 batchCount);
  } else if (group.dtype == FLOAT32) {
    float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemmBatched(*handle, transAblas, transBblas,
                                 group.N, group.M, group.K,
                                 &alpha, (const float**)group.d_B_ptrs, lda,
                                 (const float**)group.d_A_ptrs, ldb,
                                 &beta, (float**)group.d_C_ptrs, ldc,
                                 batchCount);
  } else if (group.dtype == HALF) {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    status = cublasHgemmBatched(*handle, transAblas, transBblas,
                                 group.N, group.M, group.K,
                                 &alpha, (const __half**)group.d_B_ptrs, lda,
                                 (const __half**)group.d_A_ptrs, ldb,
                                 &beta, (__half**)group.d_C_ptrs, ldc,
                                 batchCount);
  } else if (group.dtype == BFLOAT16) {
    float alpha = 1.0f, beta = 0.0f;
    status = cublasGemmBatchedEx(*handle, transAblas, transBblas,
                                  group.N, group.M, group.K,
                                  &alpha,
                                  (const void**)group.d_B_ptrs, CUDA_R_16BF, lda,
                                  (const void**)group.d_A_ptrs, CUDA_R_16BF, ldb,
                                  &beta,
                                  (void**)group.d_C_ptrs, CUDA_R_16BF, ldc,
                                  batchCount,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else {
    DSP_DIAG(EXECUTE, "batched GEMM group %d: unsupported dtype %d", groupIdx, (int)group.dtype);
    return Status::BAD_ARGUMENTS;
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    DSP_DIAG(EXECUTE, "batched GEMM group %d: cublasGemmBatched failed with status %d",
              groupIdx, (int)status);
    return Status::KERNEL_FAILURE;
  }

  // 4. Mark outputs as device-authoritative
  for (int b = 0; b < batchCount; b++) {
    int slotIdx = group.slotIndices[b];
    NativeSlot& slot = slots_[slotIdx];
    if (slot.numOutputs >= 1) {
      int outSlotIdx = slot.outputSlotIndices[0];
      if (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots_ && outputSlots_[outSlotIdx] != nullptr) {
        outputSlots_[outSlotIdx]->tickWriteDevice();
      }
    }
  }

  return Status::OK;
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

void NativeDynamicShapePlan::freeBatchedGemmResources() {
  for (auto& group : batchedGemmGroups_) {
    if (group.d_A_ptrs) { cudaFree(group.d_A_ptrs); group.d_A_ptrs = nullptr; }
    if (group.d_B_ptrs) { cudaFree(group.d_B_ptrs); group.d_B_ptrs = nullptr; }
    if (group.d_C_ptrs) { cudaFree(group.d_C_ptrs); group.d_C_ptrs = nullptr; }
    if (group.h_A_ptrs) { cudaFreeHost(group.h_A_ptrs); group.h_A_ptrs = nullptr; }
    if (group.h_B_ptrs) { cudaFreeHost(group.h_B_ptrs); group.h_B_ptrs = nullptr; }
    if (group.h_C_ptrs) { cudaFreeHost(group.h_C_ptrs); group.h_C_ptrs = nullptr; }
  }
  batchedGemmGroups_.clear();
  slotToBatchedGemmGroup_.clear();
}

}  // namespace graph
}  // namespace sd
