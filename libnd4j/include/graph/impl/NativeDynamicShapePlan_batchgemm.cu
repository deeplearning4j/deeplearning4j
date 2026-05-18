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
// Groups matmul slots with identical dimensions, transpose flags, and A/B/C
// dtypes into single cublasGemmBatchedEx calls. Uses dependency analysis to
// ensure only truly independent matmuls are batched together.
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
#include <graph/OpDetection.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <array/ArrayOptions.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/Environment.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <exceptions/cuda_exception.h>

#include <algorithm>
#include <queue>
#include <unordered_set>

// cuBLAS workspace for graph capture (from MmulHelper.cu)
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;
extern SD_TLS_EXPORT thread_local bool tl_graphExecutionActive;

namespace sd {
namespace graph {

using namespace op_detection;  // isMatmulOp, extractMatmulDims, hasTransitiveDependency, allInputsAvailableBefore

// Vectorized float→half conversion kernel for batched GEMM mixed-type support.
// Used when cublasGemmBatchedEx rejects mixed FLOAT32×HALF inputs (it only
// supports HALF×HALF→FLOAT32 with CUBLAS_COMPUTE_32F).
SD_KERNEL void batchedGemmCastFloat2Half(const float* __restrict__ src,
                                          __half* __restrict__ dst,
                                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __float2half_rn(src[idx]);
  }
}

namespace {

struct BatchedMatmulSig {
  int M, N, K, transA, transB;
  DataType aType, bType, cType;

  bool operator==(const BatchedMatmulSig& o) const {
    return M == o.M && N == o.N && K == o.K &&
           transA == o.transA && transB == o.transB &&
           aType == o.aType && bType == o.bType && cType == o.cType;
  }
};

struct BatchedMatmulSigHash {
  size_t operator()(const BatchedMatmulSig& s) const {
    size_t h = std::hash<int>()(s.M);
    h ^= std::hash<int>()(s.N) << 1;
    h ^= std::hash<int>()(s.K) << 2;
    h ^= std::hash<int>()(s.transA) << 3;
    h ^= std::hash<int>()(s.transB) << 4;
    h ^= std::hash<int>()(static_cast<int>(s.aType)) << 5;
    h ^= std::hash<int>()(static_cast<int>(s.bType)) << 6;
    h ^= std::hash<int>()(static_cast<int>(s.cType)) << 7;
    return h;
  }
};

static bool cudaTypeFor(DataType dt, cudaDataType& out) {
  switch (dt) {
    case FLOAT32:
      out = CUDA_R_32F;
      return true;
    case HALF:
      out = CUDA_R_16F;
      return true;
    case BFLOAT16:
      out = CUDA_R_16BF;
      return true;
    case DOUBLE:
      out = CUDA_R_64F;
      return true;
    default:
      return false;
  }
}

static bool supportedBatchedGemmTypes(DataType aType, DataType bType, DataType cType) {
  cudaDataType ignored;
  if (!cudaTypeFor(aType, ignored) || !cudaTypeFor(bType, ignored) || !cudaTypeFor(cType, ignored)) {
    return false;
  }

  const bool anyDouble = aType == DOUBLE || bType == DOUBLE || cType == DOUBLE;
  const bool allDouble = aType == DOUBLE && bType == DOUBLE && cType == DOUBLE;
  if (anyDouble && !allDouble) return false;

  // cublasGemmBatchedEx with CUBLAS_COMPUTE_32F supports mixed HALF/FLOAT32 A/B
  // in CUDA 12+. Allow mixed types within {HALF, FLOAT32, BFLOAT16} — the execution
  // path uses cublasGemmBatchedEx with per-operand cudaDataType and FP32 accumulation.
  // Only reject mixed types outside the supported set.
  if (aType != bType) {
    const bool aMixed = (aType == HALF || aType == FLOAT32 || aType == BFLOAT16);
    const bool bMixed = (bType == HALF || bType == FLOAT32 || bType == BFLOAT16);
    if (!aMixed || !bMixed) return false;
  }

  if (!allDouble && cType != FLOAT32 && cType != HALF && cType != BFLOAT16) return false;
  return true;
}

static bool singleMatrixRowMajor(NDArray* arr, int rows, int cols) {
  if (arr == nullptr || arr->rankOf() < 2 || arr->rankOf() > 3) return false;
  if (arr->rankOf() == 3 && arr->sizeAt(0) != 1) return false;
  if (arr->sizeAt(-2) != rows || arr->sizeAt(-1) != cols) return false;
  return arr->strideAt(-1) == 1 && (rows == 1 || arr->strideAt(-2) == cols);
}

}  // namespace

// Resolve shape info for a matmul input given its source index.
// Priority: NDArray* (outputSlots_ / outputSlots_ / external) -> cachedOutputShapes on source slot.
const LongType* NativeDynamicShapePlan::resolveInputShapeInfo(
    int srcIdx, NDArray** externalArrays, int numExt) const {
  if (srcIdx >= 0 && srcIdx < numSlots_) {
    if (srcIdx < totalOutputSlots_) {
      NDArray* arr = outputSlots_[srcIdx];
      if (arr == nullptr) arr = outputSlots_[srcIdx];
      if (arr != nullptr) return arr->shapeInfo();
    }
    const NativeSlot& srcSlot = slots_[srcIdx];
    if (srcSlot.shapeCacheValid() && !srcSlot.shapeCache.cachedOutputShapes.empty() &&
        srcSlot.shapeCache.cachedOutputShapes[0] != nullptr) {
      return srcSlot.shapeCache.cachedOutputShapes[0];
    }
  } else if (srcIdx < 0) {
    int extIdx = -(srcIdx + 1);
    if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
      return externalArrays[extIdx]->shapeInfo();
    }
  }
  return nullptr;
}

// ── Main detection ───────────────────────────────────────────────────────────

void NativeDynamicShapePlan::detectBatchedGemmGroups(NDArray** externalArrays, int numExt) {
  batchedGemmGroups_.clear();
  slotToBatchedGemmGroup_.assign(numSlots_, -1);

  if (!Environment::getInstance().dspBatchedGemm()) return;

  // Build reverse map: output slot index -> producing step index
  std::vector<int> outputSlotToStep(totalOutputSlots_, -1);
  for (int i = 0; i < numSlots_; i++) {
    for (int o = 0; o < slots_[i].wiring.numOutputs; o++) {
      int outIdx = slots_[i].wiring.outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots_) {
        outputSlotToStep[outIdx] = i;
      }
    }
  }

  int totalMatmuls = 0, resolvedMatmuls = 0, dimFailMatmuls = 0;
  int fromArray = 0, fromCache = 0;
  int depRejected = 0, inputRejected = 0;
  int typeRejected = 0, layoutRejected = 0;

  for (auto& seg : segments_) {
    std::unordered_map<BatchedMatmulSig, std::vector<int>, BatchedMatmulSigHash> sigBuckets;

    auto resolveArray = [&](int srcIdx) -> NDArray* {
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        return outputSlots_[srcIdx];
      }
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx >= 0 && extIdx < numExt) return externalArrays[extIdx];
      }
      return nullptr;
    };

    for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
      NativeSlot& slot = slots_[i];
      if (!isMatmulOp(slot.ident.op) || slot.wiring.numInputs != 2 ||
          slot.cf.controlFlowType != CF_NONE || slot.frozenConstantSlot()) continue;

      totalMatmuls++;

      int srcA = slot.wiring.inputSourceIndices[0];
      int srcB = slot.wiring.inputSourceIndices[1];
      const LongType* shapeA = resolveInputShapeInfo(srcA, externalArrays, numExt);
      const LongType* shapeB = resolveInputShapeInfo(srcB, externalArrays, numExt);

      if (shapeA != nullptr) {
        bool wasFromArray = false;
        if (srcA >= 0 && srcA < totalOutputSlots_) {
          NDArray* arr = outputSlots_[srcA];
          if (arr == nullptr) arr = outputSlots_[srcA];
          wasFromArray = (arr != nullptr);
        } else if (srcA < 0) {
          wasFromArray = true;
        }
        if (wasFromArray) fromArray++; else fromCache++;
      }

      int M, N, K, transA, transB;
      DataType shapeDtype;
      if (!extractMatmulDims(slot, shapeA, shapeB, M, N, K, transA, transB, shapeDtype)) {
        dimFailMatmuls++;
        if (dimFailMatmuls <= 10) {
          int rA = shapeA ? (int)shape::rank(shapeA) : -1;
          int rB = shapeB ? (int)shape::rank(shapeB) : -1;
          DSP_DIAG(EXECUTE, "  matmul slot %d: dim extract failed (shapeA=%p rankA=%d shapeB=%p rankB=%d srcA=%d srcB=%d)",
                    i, shapeA, rA, shapeB, rB, srcA, srcB);
        }
        continue;
      }
      (void)shapeDtype;

      resolvedMatmuls++;

      NDArray* arrA = resolveArray(srcA);
      NDArray* arrB = resolveArray(srcB);
      NDArray* arrC = nullptr;
      if (slot.wiring.numOutputs >= 1) {
        int outSlotIdx = slot.wiring.outputSlotIndices[0];
        if (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots_) arrC = outputSlots_[outSlotIdx];
      }
      if (arrA == nullptr || arrB == nullptr || arrC == nullptr) {
        layoutRejected++;
        continue;
      }

      const DataType aType = arrA->dataType();
      const DataType bType = arrB->dataType();
      const DataType cType = arrC->dataType();
      if (!supportedBatchedGemmTypes(aType, bType, cType)) {
        typeRejected++;
        continue;
      }

      const bool rowMajorA = singleMatrixRowMajor(arrA, transA ? K : M, transA ? M : K);
      const bool rowMajorB = singleMatrixRowMajor(arrB, transB ? N : K, transB ? K : N);
      const bool rowMajorC = singleMatrixRowMajor(arrC, M, N);
      if (!rowMajorA || !rowMajorB || !rowMajorC) {
        layoutRejected++;
        continue;
      }

      BatchedMatmulSig sig{M, N, K, transA, transB, aType, bType, cType};
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
          group.aType = sig.aType;
          group.bType = sig.bType;
          group.cType = sig.cType;
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

          DSP_DIAG(EXECUTE, "batched GEMM group %d: %d matmuls, slots [%d..%d] M=%d N=%d K=%d transA=%d transB=%d aType=%d bType=%d cType=%d trigger=%d",
                    groupIdx, (int)(end - start), (int)sg[start], (int)sg[end - 1],
                    sig.M, sig.N, sig.K, sig.transA, sig.transB,
                    (int)sig.aType, (int)sig.bType, (int)sig.cType,
                    (int)sg[start]);
        }
      }
    }
  }

  DSP_DIAG(EXECUTE, "detectBatchedGemmGroups: found %d groups from %d segments "
            "(totalMatmuls=%d resolved=%d dimFail=%d depRejected=%d inputRejected=%d "
            "typeRejected=%d layoutRejected=%d fromArray=%d fromCache=%d)",
            (int)batchedGemmGroups_.size(), (int)segments_.size(),
            totalMatmuls, resolvedMatmuls, dimFailMatmuls, depRejected, inputRejected,
            typeRejected, layoutRejected, fromArray, fromCache);
}

// ── Post-merge slot dispatch reconciliation ──────────────────────────────────
// Slot-level dispatch tables (batchedGemmGroups_, slotToBatchedGemmGroup_) are
// built before island merging and can reference slots across multiple replay
// units. After merging, slots in merged groups are replayed by a CUDA graph.
// If a batched GEMM group's trigger slot lands in a merged unit, it never fires
// during replay — orphaning non-trigger members in unmerged gaps (their matmul
// output stays as zeros from prezero → degenerate output).
//
// One pass over units to collect merged slots, one pass over groups to fix up.

void NativeDynamicShapePlan::reconcileSlotDispatchAfterMerge(const ReplaySchedule& sched) {
  if (batchedGemmGroups_.empty()) return;

  // Pass 1: collect merged slot indices into a dense boolean vector (O(1) lookup)
  int mapSize = static_cast<int>(slotToBatchedGemmGroup_.size());
  std::vector<bool> isMerged(mapSize, false);
  bool anyMerged = false;
  for (auto& u : sched.units) {
    if (u.mergedGroupId < 0) continue;
    for (int s = u.startSlot; s <= u.endSlot && s < mapSize; s++) {
      isMerged[s] = true;
      slotToBatchedGemmGroup_[s] = -1;  // clear mapping immediately
    }
    anyMerged = true;
  }
  if (!anyMerged) return;

  // Pass 2: fix up each group — remove merged members, collect orphans for re-grouping
  int disabledGroups = 0, removedSlots = 0;

  // Collect orphaned gap-side matmuls with their signature for cross-segment re-grouping
  struct OrphanInfo { int slotIdx; BatchedMatmulSig sig; };
  std::vector<OrphanInfo> orphans;

  for (int gi = 0; gi < static_cast<int>(batchedGemmGroups_.size()); gi++) {
    auto& group = batchedGemmGroups_[gi];

    auto it = std::remove_if(group.slotIndices.begin(), group.slotIndices.end(),
                             [&](int s) { return s < mapSize && isMerged[s]; });
    int removed = static_cast<int>(std::distance(it, group.slotIndices.end()));
    if (removed == 0) continue;

    group.slotIndices.erase(it, group.slotIndices.end());
    removedSlots += removed;

    if (static_cast<int>(group.slotIndices.size()) < 2) {
      // Collect orphans with their signature before clearing — they may re-group
      // with orphans from other segments that share the same (M,N,K,trans,types).
      BatchedMatmulSig sig{group.M, group.N, group.K, group.transA, group.transB,
                           group.aType, group.bType, group.cType};
      for (int s : group.slotIndices) {
        orphans.push_back({s, sig});
        slotToBatchedGemmGroup_[s] = -1;
      }
      group.slotIndices.clear();
      group.triggerSlot = -1;
      disabledGroups++;
    } else {
      // Reassign trigger + update mapping (group index unchanged, no erasure)
      group.triggerSlot = group.slotIndices.front();
      for (int s : group.slotIndices) {
        slotToBatchedGemmGroup_[s] = gi;
      }
    }
  }

  DSP_DIAG(EXECUTE, "reconcileSlotDispatchAfterMerge: %d matmul slots removed, %d groups disabled, %d orphans",
           removedSlots, disabledGroups, (int)orphans.size());

  // Pass 3: Re-group orphaned matmuls across segments by signature.
  // After island merging, each segment may have had its bgemm group reduced to 1 member.
  // Orphans from different segments with the same (M,N,K,transA,transB,types) can form
  // new cross-segment batched groups — e.g. 24 transformer layers each contributing
  // one gap-side matmul with identical dimensions.
  if (orphans.size() >= 2) {
    // Build reverse map for dependency analysis
    std::vector<int> outputSlotToStep(totalOutputSlots_, -1);
    for (int i = 0; i < numSlots_; i++) {
      for (int o = 0; o < slots_[i].wiring.numOutputs; o++) {
        int outIdx = slots_[i].wiring.outputSlotIndices[o];
        if (outIdx >= 0 && outIdx < totalOutputSlots_) {
          outputSlotToStep[outIdx] = i;
        }
      }
    }

    // Bucket orphans by signature
    std::unordered_map<BatchedMatmulSig, std::vector<int>, BatchedMatmulSigHash> orphanBuckets;
    for (auto& o : orphans) {
      orphanBuckets[o.sig].push_back(o.slotIdx);
    }

    int regroupedSlots = 0, newGroups = 0;
    for (auto& [sig, slotsInBucket] : orphanBuckets) {
      if (slotsInBucket.size() < 2) continue;

      std::sort(slotsInBucket.begin(), slotsInBucket.end());

      // Build independent sub-groups with dependency analysis (same as detectBatchedGemmGroups)
      std::vector<std::vector<int>> subGroups;

      for (int slot : slotsInBucket) {
        bool addedToExisting = false;

        for (auto& sg : subGroups) {
          int firstSlot = sg.front();

          if (!allInputsAvailableBefore(slots_[slot], firstSlot, totalOutputSlots_, outputSlotToStep)) {
            continue;
          }

          std::unordered_set<int> existingMembers(sg.begin(), sg.end());
          int minExisting = sg.front();

          if (hasTransitiveDependency(slots_, numSlots_, totalOutputSlots_,
                                       outputSlotToStep, slot, existingMembers, minExisting)) {
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

      // Create groups for sub-groups with >= 2 members (cap at 32 per batch)
      for (auto& sg : subGroups) {
        if (sg.size() < 2) continue;

        for (size_t start = 0; start < sg.size(); start += 32) {
          size_t end = std::min(start + 32, sg.size());
          if (end - start < 2) continue;

          int groupIdx = static_cast<int>(batchedGemmGroups_.size());
          BatchedGemmGroup group;
          group.slotIndices.assign(sg.begin() + start, sg.begin() + end);
          group.triggerSlot = group.slotIndices.front();
          group.M = sig.M; group.N = sig.N; group.K = sig.K;
          group.transA = sig.transA; group.transB = sig.transB;
          group.aType = sig.aType; group.bType = sig.bType; group.cType = sig.cType;
          group.d_A_ptrs = nullptr; group.d_B_ptrs = nullptr; group.d_C_ptrs = nullptr;
          group.h_A_ptrs = nullptr; group.h_B_ptrs = nullptr; group.h_C_ptrs = nullptr;
          group.maxBatchSize = static_cast<int>(end - start);
          batchedGemmGroups_.push_back(std::move(group));

          for (size_t s = start; s < end; s++) {
            slotToBatchedGemmGroup_[sg[s]] = groupIdx;
          }
          regroupedSlots += static_cast<int>(end - start);
          newGroups++;
        }
      }
    }

    if (regroupedSlots > 0) {
      DSP_DIAG(EXECUTE, "reconcileSlotDispatchAfterMerge: re-grouped %d orphans into %d cross-segment groups",
               regroupedSlots, newGroups);
    }
  }
}

// ── Device resource allocation ───────────────────────────────────────────────

void NativeDynamicShapePlan::prepareBatchedGemmDevice(cudaStream_t stream) {
  for (auto& group : batchedGemmGroups_) {
    if (group.d_A_ptrs != nullptr) continue;  // already allocated

    int bs = group.maxBatchSize;
    size_t ptrArrayBytes = bs * sizeof(void*);

    int deviceId = sd::AffinityManager::currentDeviceId();
    group.d_A_ptrs = reinterpret_cast<void**>(memory::CudaMemoryPool::getInstance().allocate(ptrArrayBytes, deviceId, stream));
    group.d_B_ptrs = reinterpret_cast<void**>(memory::CudaMemoryPool::getInstance().allocate(ptrArrayBytes, deviceId, stream));
    group.d_C_ptrs = reinterpret_cast<void**>(memory::CudaMemoryPool::getInstance().allocate(ptrArrayBytes, deviceId, stream));
    cudaMallocHost(&group.h_A_ptrs, ptrArrayBytes);
    cudaMallocHost(&group.h_B_ptrs, ptrArrayBytes);
    cudaMallocHost(&group.h_C_ptrs, ptrArrayBytes);

    // Pre-determine if this group needs mixed-type casting
    const bool usesBfloat = group.aType == BFLOAT16 || group.bType == BFLOAT16 || group.cType == BFLOAT16;
    group.needsCast = (group.aType != group.bType) && !usesBfloat;

    // Pre-allocate persistent cast scratch for mixed-type groups.
    // Eliminates per-step cudaMalloc/cudaFreeAsync (was 4 CUDA mem ops × 60 groups = 240/step).
    if (group.needsCast) {
      const bool castA = (group.aType == FLOAT32 && (group.bType == HALF || group.bType == BFLOAT16));
      const DataType targetType = castA ? group.bType : group.aType;
      const int castElem = castA ? (group.M * group.K) : (group.K * group.N);
      const size_t elemSize = (targetType == HALF) ? sizeof(__half) : sizeof(__nv_bfloat16);
      const size_t perMemberBytes = castElem * elemSize;
      group.castScratchBytes = perMemberBytes * bs;

      auto err = cudaMalloc(&group.castScratch, group.castScratchBytes);
      if (err != cudaSuccess) {
        DSP_DIAG(MEMORY, "batched GEMM group: cast scratch alloc failed (%zu bytes): %s",
                 group.castScratchBytes, cudaGetErrorString(err));
        cudaGetLastError();
        group.castScratch = nullptr;
        group.castScratchBytes = 0;
        group.needsCast = false;  // fall back to per-step alloc
      } else {
        err = cudaMalloc((void**)&group.d_castPtrs, ptrArrayBytes);
        if (err != cudaSuccess) {
          cudaFree(group.castScratch);
          cudaGetLastError();
          group.castScratch = nullptr;
          group.castScratchBytes = 0;
          group.d_castPtrs = nullptr;
          group.needsCast = false;
        } else {
          DSP_DIAG(MEMORY, "batched GEMM group: persistent cast scratch %zu bytes + %zu ptr bytes",
                   group.castScratchBytes, ptrArrayBytes);
        }
      }
    }

    DSP_DIAG(MEMORY, "batched GEMM group: allocated %d pointer arrays (%zu bytes each) needsCast=%d",
              bs, ptrArrayBytes, group.needsCast ? 1 : 0);
  }
}

// ── Execute a single batched GEMM group ──────────────────────────────────────

static inline void reapplyCublasWorkspaceBG(cublasHandle_t handle) {
  // Skip during CUDA graph capture: workspace was pre-set by setCublasWorkspaceForCapture
  // before cudaStreamBeginCapture; calling cublasSetWorkspace on a capturing stream may
  // inject a host-callback node into the graph.
  if (!tl_graphExecutionActive && tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
    cublasSetWorkspace(handle, tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
  }
}

Status NativeDynamicShapePlan::executeBatchedGemmGroup(
    int groupIdx, NDArray** externalArrays, int numExt, cudaStream_t stream) {

  auto& group = batchedGemmGroups_[groupIdx];
  int batchCount = static_cast<int>(group.slotIndices.size());

  // ── Pointer-stable fast path ───────────────────────────────────────────
  // In steady state (shapes frozen, pointers stable), the device pointer
  // arrays already contain the correct addresses from the previous step.
  // Skip input resolution, syncToDevice, and H2D copies entirely.
  bool canSkipPtrRefresh = group.ptrStable && planLifecycle_.pointersStable() && !planLifecycle_.isSlotBySlot() && executeCount_ >= 3;

  std::vector<NDArray*> inputAs;
  std::vector<NDArray*> inputBs;
  std::vector<NDArray*> outputs;
  std::vector<NDArray*> readList;
  std::vector<NDArray*> writeList;

  if (!canSkipPtrRefresh) {
    inputAs.reserve(batchCount);
    inputBs.reserve(batchCount);
    outputs.reserve(batchCount);
    readList.reserve(batchCount * 2);
    writeList.reserve(batchCount);

    auto resolveArray = [&](int srcIdx) -> NDArray* {
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        return outputSlots_[srcIdx];
      }
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx >= 0 && extIdx < numExt) return externalArrays[extIdx];
      }
      return nullptr;
    };

    // 1. Resolve and validate current arrays before preparing device buffers.
    bool anyPtrChanged = false;
    for (int b = 0; b < batchCount; b++) {
      int slotIdx = group.slotIndices[b];
      NativeSlot& slot = slots_[slotIdx];

      NDArray* inputA = resolveArray(slot.wiring.inputSourceIndices[0]);
      NDArray* inputB = resolveArray(slot.wiring.inputSourceIndices[1]);

      // Resolve output C
      NDArray* outputC = nullptr;
      if (slot.wiring.numOutputs >= 1) {
        int outSlotIdx = slot.wiring.outputSlotIndices[0];
        if (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots_) {
          outputC = outputSlots_[outSlotIdx];
          if (outputC == nullptr) outputC = outputSlots_[outSlotIdx];
        }
      }

      if (inputA == nullptr || inputB == nullptr || outputC == nullptr) {
        DSP_DIAG(EXECUTE, "batched GEMM group %d slot %d: null array (A=%p B=%p C=%p)",
                  groupIdx, slotIdx, inputA, inputB, outputC);
        return Status::BAD_INPUT;
      }

      if (inputA->dataType() != group.aType || inputB->dataType() != group.bType ||
          outputC->dataType() != group.cType) {
        DSP_DIAG(EXECUTE,
                 "batched GEMM group %d slot %d: dtype drift A=%d/%d B=%d/%d C=%d/%d "
                 "A shape=[%lld,%lld] B shape=[%lld,%lld] C shape=[%lld,%lld]",
                 groupIdx, slotIdx,
                 (int)inputA->dataType(), (int)group.aType,
                 (int)inputB->dataType(), (int)group.bType,
                 (int)outputC->dataType(), (int)group.cType,
                 (long long)(inputA->rankOf() >= 1 ? inputA->sizeAt(0) : 0),
                 (long long)(inputA->rankOf() >= 2 ? inputA->sizeAt(1) : 0),
                 (long long)(inputB->rankOf() >= 1 ? inputB->sizeAt(0) : 0),
                 (long long)(inputB->rankOf() >= 2 ? inputB->sizeAt(1) : 0),
                 (long long)(outputC->rankOf() >= 1 ? outputC->sizeAt(0) : 0),
                 (long long)(outputC->rankOf() >= 2 ? outputC->sizeAt(1) : 0));
        return Status::BAD_ARGUMENTS;
      }

      const bool rowMajorA = singleMatrixRowMajor(inputA, group.transA ? group.K : group.M,
                                                  group.transA ? group.M : group.K);
      const bool rowMajorB = singleMatrixRowMajor(inputB, group.transB ? group.N : group.K,
                                                  group.transB ? group.K : group.N);
      const bool rowMajorC = singleMatrixRowMajor(outputC, group.M, group.N);
      if (!rowMajorA || !rowMajorB || !rowMajorC) {
        DSP_DIAG(EXECUTE,
                 "batched GEMM group %d slot %d: layout drift rowMajorA=%d rowMajorB=%d rowMajorC=%d "
                 "A strides=[%lld,%lld] B strides=[%lld,%lld] C strides=[%lld,%lld]",
                 groupIdx, slotIdx, (int)rowMajorA, (int)rowMajorB, (int)rowMajorC,
                 (long long)(inputA->rankOf() >= 2 ? inputA->strideAt(0) : 0),
                 (long long)(inputA->rankOf() >= 2 ? inputA->strideAt(1) : 0),
                 (long long)(inputB->rankOf() >= 2 ? inputB->strideAt(0) : 0),
                 (long long)(inputB->rankOf() >= 2 ? inputB->strideAt(1) : 0),
                 (long long)(outputC->rankOf() >= 2 ? outputC->strideAt(0) : 0),
                 (long long)(outputC->rankOf() >= 2 ? outputC->strideAt(1) : 0));
        return Status::BAD_ARGUMENTS;
      }

      inputAs.push_back(inputA);
      inputBs.push_back(inputB);
      outputs.push_back(outputC);
      readList.push_back(inputA);
      readList.push_back(inputB);
      writeList.push_back(outputC);
    }

    NDArray::prepareSpecialUse(writeList, readList);

    // 2. Populate host pointer arrays from prepared device buffers.
    for (int b = 0; b < batchCount; b++) {
      void* aPtr = inputAs[b]->specialBuffer();
      void* bPtr = inputBs[b]->specialBuffer();
      void* cPtr = outputs[b]->specialBuffer();

      if (aPtr == nullptr || bPtr == nullptr || cPtr == nullptr) {
        DSP_DIAG(EXECUTE, "batched GEMM group %d: null device pointer at batch %d (A=%p B=%p C=%p)",
                 groupIdx, b, aPtr, bPtr, cPtr);
        return Status::BAD_INPUT;
      }

      if (group.h_A_ptrs[b] != aPtr || group.h_B_ptrs[b] != bPtr || group.h_C_ptrs[b] != cPtr) {
        anyPtrChanged = true;
      }
      group.h_A_ptrs[b] = aPtr;
      group.h_B_ptrs[b] = bPtr;
      group.h_C_ptrs[b] = cPtr;
    }

    // 3. Copy pointer arrays H2D only when pointers actually changed
    if (anyPtrChanged || !group.ptrStable) {
      size_t ptrBytes = batchCount * sizeof(void*);
      cudaMemcpyAsync(group.d_A_ptrs, group.h_A_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(group.d_B_ptrs, group.h_B_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(group.d_C_ptrs, group.h_C_ptrs, ptrBytes, cudaMemcpyHostToDevice, stream);
      group.ptrStable = !anyPtrChanged;
    } else {
      group.ptrStable = true;
    }
  }

  // 4. Dispatch cublasGemmBatchedEx
  auto* context = LaunchContext::defaultContext();
  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());
  auto handle = reinterpret_cast<cublasHandle_t*>(context->getCublasHandle());
  // Skip cublasSetStream_v2 during CUDA graph capture (see MmulHelper mmulMxM comment).
  if (!tl_graphExecutionActive) {
    cublasSetStream_v2(*handle, stream);
  }
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

  if (group.aType == DOUBLE && group.bType == DOUBLE && group.cType == DOUBLE) {
    double alpha = 1.0, beta = 0.0;
    status = cublasDgemmBatched(*handle, transAblas, transBblas,
                                 group.N, group.M, group.K,
                                 &alpha, (const double**)group.d_B_ptrs, lda,
                                 (const double**)group.d_A_ptrs, ldb,
                                 &beta, (double**)group.d_C_ptrs, ldc,
                                 batchCount);
  } else if (group.aType == FLOAT32 && group.bType == FLOAT32 && group.cType == FLOAT32) {
    float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemmBatched(*handle, transAblas, transBblas,
                                 group.N, group.M, group.K,
                                 &alpha, (const float**)group.d_B_ptrs, lda,
                                 (const float**)group.d_A_ptrs, ldb,
                                 &beta, (float**)group.d_C_ptrs, ldc,
                                 batchCount);
  } else if (supportedBatchedGemmTypes(group.aType, group.bType, group.cType)) {
    cudaDataType aCuda, bCuda, cCuda;
    if (!cudaTypeFor(group.aType, aCuda) ||
        !cudaTypeFor(group.bType, bCuda) ||
        !cudaTypeFor(group.cType, cCuda)) {
      DSP_DIAG(EXECUTE, "batched GEMM group %d: unsupported dtype combination A=%d B=%d C=%d",
               groupIdx, (int)group.aType, (int)group.bType, (int)group.cType);
      return Status::BAD_ARGUMENTS;
    }

    float alpha = 1.0f, beta = 0.0f;
    const bool usesBfloat = group.aType == BFLOAT16 || group.bType == BFLOAT16 || group.cType == BFLOAT16;
    cublasGemmAlgo_t algo = usesBfloat ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // cublasGemmBatchedEx with CUBLAS_COMPUTE_32F supports HALF×HALF→FLOAT32 but
    // NOT mixed FLOAT32×HALF (returns CUBLAS_STATUS_NOT_SUPPORTED=15). For mixed
    // types, cast the FLOAT32 operand to HALF before the call. For decode-phase
    // GEMV (M=1), the activation is 1×K elements — cast cost is negligible.
    //
    // OPTIMIZATION: Cast scratch is pre-allocated in prepareBatchedGemmDevice()
    // as persistent buffers (group.castScratch, group.d_castPtrs). This eliminates
    // 4 CUDA memory operations per group per step (was cudaMalloc×2 + cudaFreeAsync×2).
    if (group.needsCast && group.castScratch != nullptr && group.d_castPtrs != nullptr) {
      const bool castA = (group.aType == FLOAT32 && (group.bType == HALF || group.bType == BFLOAT16));
      const DataType targetType = castA ? group.bType : group.aType;
      const int castElem = castA ? (group.M * group.K) : (group.K * group.N);
      const size_t elemSize = (targetType == HALF) ? sizeof(__half) : sizeof(__nv_bfloat16);
      const size_t perMemberBytes = castElem * elemSize;

      // Launch float→half cast kernel per batch member using persistent scratch
      void* h_castPtrs_host[64] = {};
      for (int b = 0; b < batchCount; b++) {
        char* dst = static_cast<char*>(group.castScratch) + perMemberBytes * b;
        const void* src = castA ? group.h_A_ptrs[b] : group.h_B_ptrs[b];
        const int blocks = (castElem + 255) / 256;
        batchedGemmCastFloat2Half<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<__half*>(dst),
            castElem);
        h_castPtrs_host[b] = dst;
      }
      cudaMemcpyAsync(group.d_castPtrs, h_castPtrs_host, batchCount * sizeof(void*),
                       cudaMemcpyHostToDevice, stream);

      // Call cuBLAS with uniform HALF A × HALF B → FLOAT32 C
      cudaDataType halfType = CUDA_R_16F;
      cudaDataType cCudaUniform = CUDA_R_32F;
      const void** dB = castA ? (const void**)group.d_B_ptrs : (const void**)group.d_castPtrs;
      const void** dA = castA ? (const void**)group.d_castPtrs : (const void**)group.d_A_ptrs;
      status = cublasGemmBatchedEx(*handle, transAblas, transBblas,
                                    group.N, group.M, group.K,
                                    &alpha,
                                    dB, halfType, lda,
                                    dA, halfType, ldb,
                                    &beta,
                                    (void**)group.d_C_ptrs, cCudaUniform, ldc,
                                    batchCount,
                                    CUBLAS_COMPUTE_32F, algo);
      // No cudaFreeAsync — scratch is persistent
    } else if (group.aType != group.bType && !usesBfloat) {
      // Fallback: per-step alloc (only if persistent alloc failed)
      const bool castA = (group.aType == FLOAT32 && (group.bType == HALF || group.bType == BFLOAT16));
      const DataType targetType = castA ? group.bType : group.aType;
      const int castElem = castA ? (group.M * group.K) : (group.K * group.N);
      const size_t elemSize = (targetType == HALF) ? sizeof(__half) : sizeof(__nv_bfloat16);
      const size_t perMemberBytes = castElem * elemSize;
      const size_t totalCastBytes = perMemberBytes * batchCount;
      const size_t ptrArrayBytes = batchCount * sizeof(void*);

      void* fallbackScratch = nullptr;
      void** fallbackPtrs = nullptr;
      auto err = cudaMalloc(&fallbackScratch, totalCastBytes);
      if (err != cudaSuccess) {
        DSP_DIAG(EXECUTE, "batched GEMM group %d: fallback cast scratch alloc failed (%zu bytes): %s",
                 groupIdx, totalCastBytes, cudaGetErrorString(err));
        cudaGetLastError();
        return Status::KERNEL_FAILURE;
      }
      err = cudaMalloc((void**)&fallbackPtrs, ptrArrayBytes);
      if (err != cudaSuccess) {
        cudaFree(fallbackScratch);
        cudaGetLastError();
        return Status::KERNEL_FAILURE;
      }

      void* h_castPtrs_host[64] = {};
      for (int b = 0; b < batchCount; b++) {
        char* dst = static_cast<char*>(fallbackScratch) + perMemberBytes * b;
        const void* src = castA ? group.h_A_ptrs[b] : group.h_B_ptrs[b];
        const int blocks = (castElem + 255) / 256;
        batchedGemmCastFloat2Half<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<__half*>(dst),
            castElem);
        h_castPtrs_host[b] = dst;
      }
      cudaMemcpyAsync(fallbackPtrs, h_castPtrs_host, ptrArrayBytes,
                       cudaMemcpyHostToDevice, stream);

      cudaDataType halfType = CUDA_R_16F;
      cudaDataType cCudaUniform = CUDA_R_32F;
      const void** dB = castA ? (const void**)group.d_B_ptrs : (const void**)fallbackPtrs;
      const void** dA = castA ? (const void**)fallbackPtrs : (const void**)group.d_A_ptrs;
      status = cublasGemmBatchedEx(*handle, transAblas, transBblas,
                                    group.N, group.M, group.K,
                                    &alpha,
                                    dB, halfType, lda,
                                    dA, halfType, ldb,
                                    &beta,
                                    (void**)group.d_C_ptrs, cCudaUniform, ldc,
                                    batchCount,
                                    CUBLAS_COMPUTE_32F, algo);
      cudaFreeAsync(fallbackScratch, stream);
      cudaFreeAsync(fallbackPtrs, stream);
    } else {
      status = cublasGemmBatchedEx(*handle, transAblas, transBblas,
                                    group.N, group.M, group.K,
                                    &alpha,
                                    (const void**)group.d_B_ptrs, bCuda, lda,
                                    (const void**)group.d_A_ptrs, aCuda, ldb,
                                    &beta,
                                    (void**)group.d_C_ptrs, cCuda, ldc,
                                    batchCount,
                                    CUBLAS_COMPUTE_32F, algo);
    }
  } else {
    DSP_DIAG(EXECUTE, "batched GEMM group %d: unsupported dtype combination A=%d B=%d C=%d",
             groupIdx, (int)group.aType, (int)group.bType, (int)group.cType);
    return Status::BAD_ARGUMENTS;
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    DSP_DIAG(EXECUTE, "batched GEMM group %d: cublasGemmBatched FAILED cublas_status=%d "
              "aType=%d bType=%d cType=%d M=%d N=%d K=%d batch=%d mixedType=%d "
              "transA=%d transB=%d lda=%d ldb=%d ldc=%d",
              groupIdx, (int)status,
              (int)group.aType, (int)group.bType, (int)group.cType,
              group.M, group.N, group.K, batchCount,
              (int)(group.aType != group.bType),
              (int)group.transA, (int)group.transB, lda, ldb, ldc);
    return Status::KERNEL_FAILURE;
  }

  // 5. Mark outputs as device-authoritative. Pointer-stable executions skipped
  // prepare/register above, so tick their output buffers directly.
  if (!writeList.empty()) {
    NDArray::registerSpecialUse(writeList, readList);
  } else {
    for (int b = 0; b < batchCount; b++) {
      int slotIdx = group.slotIndices[b];
      NativeSlot& slot = slots_[slotIdx];
      if (slot.wiring.numOutputs >= 1) {
        int outSlotIdx = slot.wiring.outputSlotIndices[0];
        if (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots_ && outputSlots_[outSlotIdx] != nullptr) {
          outputSlots_[outSlotIdx]->tickWriteDevice();
        }
      }
    }
  }

  return Status::OK;
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

void NativeDynamicShapePlan::freeBatchedGemmResources() {
  int deviceId = sd::AffinityManager::currentDeviceId();
  for (auto& group : batchedGemmGroups_) {
    if (group.d_A_ptrs) { memory::CudaMemoryPool::getInstance().free(reinterpret_cast<void*>(group.d_A_ptrs), deviceId); group.d_A_ptrs = nullptr; }
    if (group.d_B_ptrs) { memory::CudaMemoryPool::getInstance().free(reinterpret_cast<void*>(group.d_B_ptrs), deviceId); group.d_B_ptrs = nullptr; }
    if (group.d_C_ptrs) { memory::CudaMemoryPool::getInstance().free(reinterpret_cast<void*>(group.d_C_ptrs), deviceId); group.d_C_ptrs = nullptr; }
    if (group.h_A_ptrs) { cudaFreeHost(group.h_A_ptrs); group.h_A_ptrs = nullptr; }
    if (group.h_B_ptrs) { cudaFreeHost(group.h_B_ptrs); group.h_B_ptrs = nullptr; }
    if (group.h_C_ptrs) { cudaFreeHost(group.h_C_ptrs); group.h_C_ptrs = nullptr; }
    if (group.castScratch) { cudaFree(group.castScratch); group.castScratch = nullptr; group.castScratchBytes = 0; }
    if (group.d_castPtrs) { cudaFree(group.d_castPtrs); group.d_castPtrs = nullptr; }
  }
  batchedGemmGroups_.clear();
  slotToBatchedGemmGroup_.clear();
}

}  // namespace graph
}  // namespace sd
