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

/**
 * NativeDynamicShapePlan — Segment Management
 *
 * Contains computeSegmentShapeKey(),
 * executeSegmentWithCpuGraph(), and executeSegmentSlotBySlot().
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif
#include <graph/cpu/FunctionalReplayHandle.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/Environment.h>

#include <algorithm>
#include <cstring>
#include <unordered_set>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

// Include CPU graph backends conditionally
#include <config.h>
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
#endif
#if HAVE_MLIR
#include <graph/cpu/MlirCpuGraphBackend.h>
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
#include <graph/cpu/ArmHybridGraphBackend.h>
#endif
#endif
#if HAVE_NNAPI
#include <graph/cpu/NnapiGraphBackend.h>
#endif
#if HAVE_MLX
#include <graph/cpu/MlxGraphBackend.h>
#endif
#if HAVE_OPENVINO
#include <graph/cpu/OpenVinoGraphBackend.h>
#endif
namespace sd {
namespace graph {

namespace {
const char* statusName_seg(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    default: return "UNKNOWN";
  }
}

uint32_t resolveSegmentShapeTraits(const NativeSlot& slot) {
  uint32_t traits = 0;
  if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
    traits |= slot.ident.op->getOpDescriptor()->getTraits();
  }
  if (slot.flags.isViewCapableOp) traits |= sd::ops::OP_TRAIT_VIEW_PRODUCING;
  if (slot.flags.isIdentityOp) traits |= sd::ops::OP_TRAIT_IDENTITY;
  if (slot.flags.outputShapeDependsOnInputValues) traits |= sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE;
  if (slot.flags.isDataDependent) traits |= sd::ops::OP_TRAIT_DATA_DEPENDENT;
  return traits;
}

int findProducerStep(const GraphSegment& seg, NativeSlot* slots, int outputSlotIdx) {
  if (slots == nullptr || outputSlotIdx < 0) return -1;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    const auto& slot = slots[s];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      if (slot.wiring.outputSlotIndices[o] == outputSlotIdx) {
        return s;
      }
    }
  }
  return -1;
}
}  // namespace

// ─── Segment shape key computation ──────────────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentShapeKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {

  // ── Symbolic shape range path ──────────────────────────────────────────
  // When enabled, collect cross-segment inputs, feed them to the shape
  // profiler, and (after warmup) use range-based hashing that ignores
  // dynamic dimensions.
  if (seg.exec.symbolicShapeEnabled && seg.exec.symbolicRangeData != nullptr) {
    auto* profile = static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData);

    // Collect cross-segment input arrays (same logic as standard path below)
    std::unordered_set<int> segOutputSlots;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
      }
    }

    std::vector<NDArray*> crossInputs;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
            crossInputs.push_back(externalInputs[extIdx]);
          }
        } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
          if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
            crossInputs.push_back(outputSlots_[srcIdx]);
          }
        }
      }
    }

    // Record observations during warmup
    if (!isWarmupComplete(profile)) {
      recordObservedShapes(profile, crossInputs.data(),
                           static_cast<int>(crossInputs.size()));
      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] observation %d/%d",
               seg.startSlot, seg.endSlot,
               getObservationCount(profile), getWarmupSteps(profile));
    }

    // After warmup, use range-based key
    if (isWarmupComplete(profile)) {
      LongType rangeKey = computeRangeBasedShapeKey(
          profile, crossInputs.data(), static_cast<int>(crossInputs.size()),
          seg.startSlot, seg.endSlot);

      // Mix op names, iArgs, and tArgs into the range-based key so different
      // plans with the same input shapes but different ops produce unique keys
      // in singleton backend caches (OpenVINO, OneDNN Graph). v2-cache-fix.
      auto mixRange = [&rangeKey](LongType val) {
        rangeKey ^= val;
        rangeKey *= 0x100000001b3ULL;
      };
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        if (!slot.ident.opName.empty()) {
          for (const char* p = slot.ident.opName.c_str(); *p != '\0'; p++) {
            mixRange(static_cast<LongType>(*p));
          }
        }
        mixRange(static_cast<LongType>(slot.args.numIArgs));
        for (int a = 0; a < slot.args.numIArgs; a++) {
          mixRange(static_cast<LongType>(slot.args.iArgs[a]));
        }
        mixRange(static_cast<LongType>(slot.args.numTArgs));
      }

      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] using range-based key=%lld (with-op-mix)",
               seg.startSlot, seg.endSlot, rangeKey);
      // Cache the key for subsequent calls (when shapesFrozen_ is enabled)
      seg.exec.cachedShapeKey = rangeKey;
      return rangeKey;
    }
    // Fall through to standard path during warmup
  }

  // ── Standard FNV-1a path ───────────────────────────────────────────────
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  auto mixArraySignature = [&](NDArray* arr) {
    if (arr == nullptr) return;

    const LongType* si = arr->shapeInfo();
    LongType rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
    }
    mix(static_cast<LongType>(arr->lengthOf()));
    mix(static_cast<LongType>(arr->dataType()));

    // Hash actual DATA VALUES for small inputs (≤32 elements).
    // This makes the shape key sensitive to value changes in shape tensors,
    // axis constants, broadcast targets, reshape dims, etc. — eliminating the
    // need for a hardcoded "value-dependent shape op" list.
    // Only small inputs are hashed to avoid GPU→CPU sync overhead on large tensors.
    // The sync is safe here because shape key computation runs BEFORE graph capture
    // (during warmup or shape-change detection), not during replay.
    LongType len = arr->lengthOf();
    if (len > 0 && len <= 32) {
      arr->syncToHost();
      auto dt = arr->dataType();
      for (LongType e = 0; e < len; e++) {
        if (dt == INT32 || dt == INT64 || dt == BOOL) {
          mix(arr->e<LongType>(e));
        } else {
          // Float values: cast to int64 bit pattern for hashing
          double v = arr->e<double>(e);
          LongType bits;
          std::memcpy(&bits, &v, sizeof(bits));
          mix(bits);
        }
      }
    }
  };

  mix(seg.startSlot);
  mix(seg.endSlot);

  // Mix op names so different plans with same slot indices + shapes don't collide
  // in singleton backend caches (e.g. OpenVINO, OneDNN Graph)
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    if (!slot.ident.opName.empty()) {
      for (const char* p = slot.ident.opName.c_str(); *p != '\0'; p++) {
        mix(static_cast<LongType>(*p));
      }
    }
    mix(static_cast<LongType>(slot.wiring.numInputs));
    mix(static_cast<LongType>(slot.wiring.numOutputs));
    mix(static_cast<LongType>(slot.args.numIArgs));
    // Mix actual iArg values (e.g. reshape target shape, axis indices)
    for (int a = 0; a < slot.args.numIArgs; a++) {
      mix(static_cast<LongType>(slot.args.iArgs[a]));
    }
    // Mix tArg count (float args like epsilon, scale)
    mix(static_cast<LongType>(slot.args.numTArgs));
  }

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
    }
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mixArraySignature(externalInputs[extIdx]);
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mixArraySignature(outputSlots_[srcIdx]);
        }
      }
    }
  }

  // Value-dependent-shape consumers can be fed by small internal subgraphs
  // (for example concat -> create) whose boundary inputs are external/cross-segment
  // but whose current values are invisible to the plain cross-input key above.
  // Walk those internal producer chains and mix their boundary array signatures
  // plus op structure so frozen replay sees internal shape-driving variance.
  auto mixShapeDriverChain = [&](auto&& self, int srcIdx,
                                 std::unordered_set<int>& visiting) -> void {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      mix(static_cast<LongType>(0xE1));
      mix(extIdx);
      if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
        mixArraySignature(externalInputs[extIdx]);
      }
      return;
    }

    int producerStep = findProducerStep(seg, slots_, srcIdx);
    if (producerStep < 0) {
      mix(static_cast<LongType>(0xC1));
      mix(srcIdx);
      if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
        mixArraySignature(outputSlots_[srcIdx]);
      }
      return;
    }

    if (visiting.count(producerStep) != 0) {
      mix(static_cast<LongType>(0x51));
      mix(producerStep);
      return;
    }

    visiting.insert(producerStep);
    NativeSlot& producer = slots_[producerStep];
    mix(static_cast<LongType>(0xA1));
    mix(producerStep);
    mix(static_cast<LongType>(resolveSegmentShapeTraits(producer)));
    mix(static_cast<LongType>(producer.wiring.numInputs));
    mix(static_cast<LongType>(producer.wiring.numOutputs));
    mix(static_cast<LongType>(producer.args.numIArgs));
    for (int a = 0; a < producer.args.numIArgs; a++) {
      mix(static_cast<LongType>(producer.args.iArgs[a]));
    }
    mix(static_cast<LongType>(producer.args.numTArgs));
    if (!producer.ident.opName.empty()) {
      for (const char* p = producer.ident.opName.c_str(); *p != '\0'; p++) {
        mix(static_cast<LongType>(*p));
      }
    }
    for (int i = 0; i < producer.wiring.numInputs; i++) {
      self(self, producer.wiring.inputSourceIndices[i], visiting);
    }
    visiting.erase(producerStep);
  };

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    const uint32_t traits = resolveSegmentShapeTraits(slot);
    if ((traits & sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) == 0) continue;

    for (int i = 0; i < slot.wiring.numInputs; i++) {
      const int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0 || segOutputSlots.find(srcIdx) == segOutputSlots.end()) continue;

      mix(static_cast<LongType>(0xD1));
      mix(s);
      mix(srcIdx);
      std::unordered_set<int> visiting;
      mixShapeDriverChain(mixShapeDriverChain, srcIdx, visiting);
    }
  }

  return key;
}

// ─── CPU Graph backend selection ────────────────────────────────────────────

GraphBackend* NativeDynamicShapePlan::getCpuGraphBackend() {
  if (cpuGraphBackendChecked_) return cpuGraphBackend_;
  cpuGraphBackendChecked_ = true;
  const auto mode = graphExecutionMode_;

  if (mode == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#ifdef SD_CUDA
  if (mode == GraphExecutionMode::GEM_TRITON ||
      mode == GraphExecutionMode::GEM_NVRTC_JIT ||
      mode == GraphExecutionMode::GEM_PTX_JIT ||
      mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
      mode == GraphExecutionMode::GEM_LEVELZERO ||
      mode == GraphExecutionMode::GEM_VULKAN ||
      mode == GraphExecutionMode::GEM_METAL ||
      mode == GraphExecutionMode::GEM_TPU ||
      mode == GraphExecutionMode::GEM_HEXAGON) {
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#ifdef SD_CUDA
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS);
#else
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS ||
                             mode == GraphExecutionMode::GEM_TRITON ||
                             mode == GraphExecutionMode::GEM_NVRTC_JIT ||
                             mode == GraphExecutionMode::GEM_PTX_JIT ||
                             mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
                             mode == GraphExecutionMode::GEM_LEVELZERO ||
                             mode == GraphExecutionMode::GEM_VULKAN ||
                             mode == GraphExecutionMode::GEM_METAL ||
                             mode == GraphExecutionMode::GEM_TPU ||
                             mode == GraphExecutionMode::GEM_HEXAGON);
#endif

#if HAVE_MLX
  if (mode == GraphExecutionMode::GEM_MLX || autoLikeMode) {
    auto& mlx = MlxGraphBackend::getInstance();
    if (mlx.isAvailable()) {
      cpuGraphBackend_ = &mlx;
      if (mode == GraphExecutionMode::GEM_MLX) {
        DSP_DIAG(BACKEND, "using MLX Apple Silicon backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using MLX Apple Silicon backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_MLX) {
      DSP_DIAG(BACKEND, "GEM_MLX requested but MLX not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_MLX) {
    DSP_DIAG(BACKEND, "GEM_MLX requested but HAVE_MLX=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#if HAVE_ONEDNN
  if (autoLikeMode) {
    auto& onednn = OneDnnGraphBackend::getInstance();
    if (onednn.isAvailable()) {
      cpuGraphBackend_ = &onednn;
      DSP_DIAG(BACKEND, "using oneDNN Graph backend");
      return cpuGraphBackend_;
    }
  }
#endif

#if HAVE_OPENVINO
  if (mode == GraphExecutionMode::GEM_OPENVINO || autoLikeMode) {
    auto& ov = OpenVinoGraphBackend::getInstance();
    if (ov.isAvailable()) {
      cpuGraphBackend_ = &ov;
      if (mode == GraphExecutionMode::GEM_OPENVINO) {
        DSP_DIAG(BACKEND, "using OpenVINO CPU backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using OpenVINO CPU backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_OPENVINO) {
      DSP_DIAG(BACKEND, "GEM_OPENVINO requested but OpenVINO not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_OPENVINO) {
    DSP_DIAG(BACKEND, "GEM_OPENVINO requested but HAVE_OPENVINO=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#if HAVE_ARMCOMPUTE
  if (autoLikeMode) {
    auto& acl = AclGraphBackend::getInstance();
    if (acl.isAvailable()) {
      cpuGraphBackend_ = &acl;
      DSP_DIAG(BACKEND, "using ARM ACL backend");
      return cpuGraphBackend_;
    }
  }
#endif

  if (mode == GraphExecutionMode::GEM_TVM) {
    DSP_DIAG(BACKEND, "GEM_TVM requested but TVM backend removed (use triton-cpu instead)");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }

#if HAVE_NNAPI
  if (mode == GraphExecutionMode::GEM_NNAPI || autoLikeMode) {
    auto& nnapi = NnapiGraphBackend::getInstance();
    if (nnapi.isAvailable()) {
      cpuGraphBackend_ = &nnapi;
      if (mode == GraphExecutionMode::GEM_NNAPI) {
        DSP_DIAG(BACKEND, "using Android NNAPI backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using Android NNAPI backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_NNAPI) {
      DSP_DIAG(BACKEND, "GEM_NNAPI requested but NNAPI not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_NNAPI) {
    DSP_DIAG(BACKEND, "GEM_NNAPI requested but HAVE_NNAPI=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#if HAVE_MLIR
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID || autoLikeMode) {
    auto& armHybrid = ArmHybridGraphBackend::getInstance();
    if (armHybrid.isAvailable()) {
      cpuGraphBackend_ = &armHybrid;
      if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
        DSP_DIAG(BACKEND, "using ARM Hybrid (MLIR CPU + Vulkan) backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using ARM Hybrid (MLIR CPU + Vulkan) backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
      DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but backend not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
    DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but this platform is not ARM Android/Linux");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  if (autoLikeMode) {
    auto& mlirBackend = MlirCpuGraphBackend::getInstance();
    if (mlirBackend.isAvailable()) {
      cpuGraphBackend_ = &mlirBackend;
      DSP_DIAG(BACKEND, "using MLIR CPU JIT backend");
      return cpuGraphBackend_;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
    DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but HAVE_MLIR=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  cpuGraphBackend_ = nullptr;
  return nullptr;
}

// ─── CPU Graph backend chain (prioritized list of all available backends) ────

const std::vector<GraphBackend*>& NativeDynamicShapePlan::getCpuGraphBackendChain() {
  if (cpuGraphBackendChainBuilt_) return cpuGraphBackendChain_;
  cpuGraphBackendChainBuilt_ = true;
  cpuGraphBackendChain_.clear();

  const auto mode = graphExecutionMode_;

  // If mode is explicitly non-CPU-graph, return empty chain
  // On CPU builds (no SD_CUDA), TRITON/NVRTC/PTX/HIP/etc. have no GPU backends,
  // so fall through to the CPU backend chain (oneDNN, OpenVINO, etc.) instead of
  // returning empty and forcing slot-by-slot.
  if (mode == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    return cpuGraphBackendChain_;
  }
#ifdef SD_CUDA
  if (mode == GraphExecutionMode::GEM_TRITON ||
      mode == GraphExecutionMode::GEM_NVRTC_JIT ||
      mode == GraphExecutionMode::GEM_PTX_JIT ||
      mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
      mode == GraphExecutionMode::GEM_LEVELZERO ||
      mode == GraphExecutionMode::GEM_VULKAN ||
      mode == GraphExecutionMode::GEM_METAL ||
      mode == GraphExecutionMode::GEM_TPU ||
      mode == GraphExecutionMode::GEM_HEXAGON) {
    return cpuGraphBackendChain_;
  }
#endif

#ifdef SD_CUDA
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS);
#else
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS ||
                             mode == GraphExecutionMode::GEM_TRITON ||
                             mode == GraphExecutionMode::GEM_NVRTC_JIT ||
                             mode == GraphExecutionMode::GEM_PTX_JIT ||
                             mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
                             mode == GraphExecutionMode::GEM_LEVELZERO ||
                             mode == GraphExecutionMode::GEM_VULKAN ||
                             mode == GraphExecutionMode::GEM_METAL ||
                             mode == GraphExecutionMode::GEM_TPU ||
                             mode == GraphExecutionMode::GEM_HEXAGON);
#endif

  // If a specific backend is forced, only return that one
  bool forcedMode = !autoLikeMode;

#if HAVE_MLX
  if (mode == GraphExecutionMode::GEM_MLX || autoLikeMode) {
    auto& mlx = MlxGraphBackend::getInstance();
    if (mlx.isAvailable()) {
      cpuGraphBackendChain_.push_back(&mlx);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_ONEDNN
  if (autoLikeMode) {
    auto& onednn = OneDnnGraphBackend::getInstance();
    if (onednn.isAvailable()) {
      cpuGraphBackendChain_.push_back(&onednn);
    }
  }
#endif

#if HAVE_OPENVINO
  if (mode == GraphExecutionMode::GEM_OPENVINO || autoLikeMode) {
    auto& ov = OpenVinoGraphBackend::getInstance();
    if (ov.isAvailable()) {
      cpuGraphBackendChain_.push_back(&ov);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_ARMCOMPUTE
  if (autoLikeMode) {
    auto& acl = AclGraphBackend::getInstance();
    if (acl.isAvailable()) {
      cpuGraphBackendChain_.push_back(&acl);
    }
  }
#endif

#if HAVE_NNAPI
  if (mode == GraphExecutionMode::GEM_NNAPI || autoLikeMode) {
    auto& nnapi = NnapiGraphBackend::getInstance();
    if (nnapi.isAvailable()) {
      cpuGraphBackendChain_.push_back(&nnapi);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_MLIR
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID || autoLikeMode) {
    auto& armHybrid = ArmHybridGraphBackend::getInstance();
    if (armHybrid.isAvailable()) {
      cpuGraphBackendChain_.push_back(&armHybrid);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

  if (autoLikeMode) {
    auto& mlirBackend = MlirCpuGraphBackend::getInstance();
    if (mlirBackend.isAvailable()) {
      cpuGraphBackendChain_.push_back(&mlirBackend);
    }
  }
#endif

  if (!cpuGraphBackendChain_.empty()) {
    DSP_DIAG(BACKEND, "CPU backend chain built: %d backends available", (int)cpuGraphBackendChain_.size());
    for (size_t i = 0; i < cpuGraphBackendChain_.size(); i++) {
      DSP_DIAG(BACKEND, "  chain[%d] = %s", (int)i, cpuGraphBackendChain_[i]->name());
    }
  }

  return cpuGraphBackendChain_;
}

// ─── Segment execution: CPU graph backend (with per-segment cascade) ────────

Status NativeDynamicShapePlan::executeSegmentWithCpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // If all backends have been exhausted for this segment, skip immediately
  if (seg.exec.compilationFailed) {
    DSP_DIAG_SEG(FALLBACK, seg.startSlot,
                 "executeSegmentWithCpuGraph: seg[%d-%d] skipped (compilationFailed=true, all backends exhausted)",
                 seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // If we already resolved a backend for this segment, use it directly
  if (seg.resolvedCpuBackend != nullptr) {
    return executeSegmentWithSpecificBackend(seg, seg.resolvedCpuBackend, externalArrays, numExt, stream);
  }

  // Cascade through the backend chain to find one that works
  const auto& chain = getCpuGraphBackendChain();
  if (chain.empty()) {
    DSP_DIAG_SEG(BACKEND, seg.startSlot,
                 "executeSegmentWithCpuGraph: no CPU graph backends available for seg[%d-%d]",
                 seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Warmup must happen before any backend tries to compile (needs output shapes)
  if (seg.exec.executionCount == 0) {
    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    DSP_DIAG(EXECUTE, "executeSegmentWithCpuGraph: warmup %s for seg[%d-%d], executionCount→%d",
             warmupStatus == Status::OK ? "OK" : "FAILED",
             seg.startSlot, seg.endSlot, seg.exec.executionCount);
    if (warmupStatus != Status::OK) {
      return warmupStatus;
    }
  }

  // Try each backend in priority order
  for (size_t i = 0; i < chain.size(); i++) {
    GraphBackend* backend = chain[i];
    const char* backendName = backend->name();

    if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
      DSP_DIAG(BACKEND, "cascade: backend=%s cannot fuse seg[%d-%d], trying next",
                backendName, seg.startSlot, seg.endSlot);
      continue;
    }

    // Attempt compile + validate + execute with this backend
    auto status = executeSegmentWithSpecificBackend(seg, backend, externalArrays, numExt, stream);
    if (status == Status::OK) {
      // Cache the resolved backend for future executions
      seg.resolvedCpuBackend = backend;
      DSP_DIAG(BACKEND, "cascade: seg[%d-%d] resolved to backend=%s (chain position %d/%d)",
                seg.startSlot, seg.endSlot, backendName, (int)i + 1, (int)chain.size());
      return Status::OK;
    }

    DSP_DIAG(BACKEND, "cascade: backend=%s failed for seg[%d-%d] (status=%d), trying next",
              backendName, seg.startSlot, seg.endSlot, static_cast<int>(status));
    // Reset compilationFailed so next backend gets a fresh try
    seg.exec.compilationFailed = false;
  }

  // ALL backends exhausted — mark as permanently failed
  seg.exec.compilationFailed = true;
  DSP_DIAG(FALLBACK, "cascade: ALL %d backends failed for seg[%d-%d], falling back to slot-by-slot",
            (int)chain.size(), seg.startSlot, seg.endSlot);
  return Status::KERNEL_FAILURE;
}

// ─── Execute segment with a specific backend (shared logic) ─────────────────

Status NativeDynamicShapePlan::executeSegmentWithSpecificBackend(
    GraphSegment& seg, GraphBackend* backend, NDArray** externalArrays, int numExt, void* stream) {

  const char* backendName = backend->name();

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  // EXCEPTION: segments with value-dependent ops must ALWAYS recompute the shape key
  // because input VALUES (hashed by computeSegmentShapeKey for small inputs ≤32 elements)
  // can change even when shapes are frozen. Without this guard, the cached key would
  // miss value changes in reshape targets, broadcast dims, etc., causing replay with
  // stale output shapes.
  //
  // REPLAY OPTIMIZATION: During stable replay (executionCount >= 3), skip shape key
  // computation entirely — even for hasValueDepOps segments. The shape key was
  // validated at capture time. Value-dependent inputs are handled by capture buffer
  // refresh, not by shape key changes. If a value change truly requires graph
  // invalidation, the createValueKey mechanism catches it. Skipping shape key here
  // eliminates N syncToHost calls per step (one per small INT/INT64 cross-segment
  // input array).
  LongType segShapeKey;
  bool needsCompile;
  bool isStableReplay = shapesFrozen_ && seg.exec.executionCount >= 3 &&
                         seg.exec.cachedShapeKey != 0;
  if (isStableReplay) {
    segShapeKey = seg.exec.cachedShapeKey;
    needsCompile = false;
  } else if (shapesFrozen_ && seg.exec.cachedShapeKey != 0 && !seg.hasValueDepOps) {
    segShapeKey = seg.exec.cachedShapeKey;
    needsCompile = false;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    needsCompile = (seg.exec.executionCount == 1) || (seg.shapeKey != segShapeKey);
  }

  if (needsCompile) {
    DSP_DIAG_SEG(COMPILE, seg.startSlot,
                 "seg[%d-%d] needs compile: %s (execCount=%d shapeKey=%lld->%lld backend=%s)",
                 seg.startSlot, seg.endSlot,
                 seg.exec.executionCount == 1 ? "first-compile" : "shape-key-changed",
                 seg.exec.executionCount,
                 static_cast<long long>(seg.shapeKey),
                 static_cast<long long>(segShapeKey),
                 backendName);
  }
  if (needsCompile) {
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithSpecificBackend: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  if (seg.exec.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    bool allCompiled = true;
    for (const auto& entry : audit) {
      if (!entry.wasCompiled) {
        allCompiled = false;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (!allCompiled) {
      DSP_DIAG(FALLBACK, "%s VALIDATION FAILURE: segment [%d-%d] has ops not covered by backend.",
                backendName, seg.startSlot, seg.endSlot);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    } else {
      DSP_DIAG_SEG(COMPILE, seg.startSlot,
                   "%s VALIDATION OK: seg[%d-%d] all %d ops compiled successfully",
                   backendName, seg.startSlot, seg.endSlot, (int)audit.size());
    }
  }

  seg.exec.cachedShapeKey = segShapeKey;
  seg.shapeKey = segShapeKey;
  tl_graphExecutionActive = true;
  DSP_DIAG(EXECUTE, "PRE-EXECUTE: seg[%d-%d] backend=%s shapeKey=%lld",
           seg.startSlot, seg.endSlot, backendName, (long long)segShapeKey);
  auto status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                         outputSlots_, totalOutputSlots_, stream);
  tl_graphExecutionActive = false;
  DSP_DIAG(EXECUTE, "POST-EXECUTE: seg[%d-%d] backend=%s status=%d",
           seg.startSlot, seg.endSlot, backendName, (int)status);

  DSP_DIAG(EXECUTE, "executeSegmentWithSpecificBackend: exec%d seg[%d-%d]: backend=%s status=%d(%s)",
            seg.exec.executionCount, seg.startSlot, seg.endSlot, backendName,
            static_cast<int>(status), statusName_seg(status));

  if (status == Status::OK) {
    seg.exec.executionCount++;
    totalGraphReplays_++;
  }

  return status;
}

// ─── Segment execution: slot-by-slot ─────────────────────────────────────────

// ─── Control flow helpers ────────────────────────────────────────────────────

namespace {

// Resolve an input for a control flow slot
inline NDArray* resolveCfInput(NativeSlot& slot, int inputIdx,
                               NDArray** outputSlots, int totalOutputSlots,
                               NDArray** externalInputs, int numExt) {
  if (inputIdx < 0 || inputIdx >= slot.wiring.numInputs) return nullptr;
  int srcIdx = slot.wiring.inputSourceIndices[inputIdx];
  if (srcIdx >= 0) {
    return (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
  } else {
    int extIdx = -(srcIdx + 1);
    return (extIdx < numExt) ? externalInputs[extIdx] : nullptr;
  }
}

// Check if any input from an output slot is dead.
// In TF-style control flow, if ANY input comes from a dead Switch branch,
// the op is on that dead branch and must be skipped entirely.
// External inputs (srcIdx < 0) don't participate in dead propagation.
inline bool anyInputDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx >= 0 && srcIdx < slotIsDeadSize && slotIsDead[srcIdx]) {
      return true;
    }
  }
  return false;
}

// Mark all outputs of a slot as dead
inline void markOutputsDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < slotIsDeadSize) slotIsDead[si] = true;
  }
}

// Forward input[0] to all outputs (identity operation for Enter/Exit/LoopCond/NextIteration)
inline void forwardInput(NativeDynamicShapePlan* plan, NativeSlot& slot, NDArray** outputSlots,
                         int totalOutputSlots, NDArray** externalInputs, int numExt,
                         const char* tag) {
  NDArray* input = resolveCfInput(slot, 0, outputSlots, totalOutputSlots, externalInputs, numExt);
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < totalOutputSlots) {
      plan->writeOutputSlot(si, input, tag);
    }
  }
}

#ifdef SD_CUDA
// Verify helper: log control flow slot output mutations
inline void verifyCfSlotWrite(int stepIdx, const char* cfType, const char* opName,
                               NDArray** outputSlots, int* outputSlotIndices,
                               int numOutputs, int totalOutputSlots) {
  if (!Environment::getInstance().tritonVerifyKernels()) return;
  for (int i = 0; i < numOutputs; i++) {
    int si = outputSlotIndices[i];
    if (si < 0 || si >= totalOutputSlots) continue;
    NDArray* out = outputSlots[si];
    if (out == nullptr) {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s (nullptr/dead)", si, cfType, opName);
    } else {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s dtype=%s len=%lld addr=%p",
                si, cfType, opName,
                DataTypeUtils::asString(out->dataType()).c_str(),
                (long long)out->lengthOf(), DSP_BUF(out));
    }
  }
}
#endif

}  // namespace

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_DIAG_SEG(EXECUTE, seg.startSlot,
               "executeSegmentSlotBySlot: ENTER seg[%d-%d] size=%d execCount=%d capturable=%d compilationFailed=%d",
               seg.startSlot, seg.endSlot, seg.endSlot - seg.startSlot + 1,
               seg.exec.executionCount, seg.isCapturable ? 1 : 0, seg.exec.compilationFailed ? 1 : 0);
  bool streamIsCapturing = false;
#ifdef SD_CUDA
  if (stream != nullptr) {
    cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
    streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
  }
#endif

  // Dead-slot flags are reset once per plan execution (in the main execute loop),
  // NOT per segment — dead flags from Switch in seg N must persist to affect
  // ops in seg N+1.

  int stepIdx = seg.startSlot;
  int loopIterations = 0;

  while (stepIdx <= seg.endSlot) {
    NativeSlot& slot = slots_[stepIdx];

    // ── Control flow dispatch ────────────────────────────────────────
    if (slot.cf.controlFlowType != CF_NONE) {
      // Dead propagation: if all inputs are dead and this is not a Merge, propagate dead
      if (slot.cf.controlFlowType != CF_MERGE && hasControlFlow_ && slotIsDead_ != nullptr) {
        if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
                        "slot %d (%s) DEAD: propagated from dead input (cf=%d)",
                        stepIdx, slot.ident.opName.c_str(), (int)slot.cf.controlFlowType);
          markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
          stepIdx++;
          continue;
        }
      }

      switch (slot.cf.controlFlowType) {
        case CF_SWITCH: {
          // Switch: input[0] = data, input[1] = predicate
          // If predicate is true: output[1] = data, output[0] is dead
          // If predicate is false: output[0] = data, output[1] is dead
          NDArray* data = resolveCfInput(slot, 0, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          NDArray* pred = resolveCfInput(slot, 1, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          bool predValue = false;
          if (pred != nullptr && !pred->isEmpty()) {
            predValue = pred->e<bool>(0);
          }
          int liveIdx = predValue ? 1 : 0;
          int deadIdx = predValue ? 0 : 1;
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              if (i == liveIdx) {
                writeOutputSlot(si, data, "cf-switch-live");
                if (slotIsDead_) slotIsDead_[si] = false;
              } else {
                writeOutputSlot(si, nullptr, "cf-switch-dead");
                if (slotIsDead_) slotIsDead_[si] = true;
              }
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "SWITCH", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_MERGE: {
          // Merge: select first non-dead, non-null input
          NDArray* selected = nullptr;
          for (int i = 0; i < slot.wiring.numInputs; i++) {
            int srcIdx = slot.wiring.inputSourceIndices[i];
            bool isDead = (srcIdx >= 0 && srcIdx < slotIsDeadSize_ && slotIsDead_ && slotIsDead_[srcIdx]);
            if (!isDead) {
              NDArray* inp = resolveCfInput(slot, i, outputSlots_, totalOutputSlots_, externalArrays, numExt);
              if (inp != nullptr) {
                selected = inp;
                break;
              }
            }
          }
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              writeOutputSlot(si, selected, "cf-merge");
              if (slotIsDead_) slotIsDead_[si] = (selected == nullptr);
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "MERGE", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_ENTER:
        case CF_EXIT:
        case CF_LOOP_COND:
          // Identity: forward input[0] to output[0]
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       slot.cf.controlFlowType == CF_ENTER ? "cf-enter"
                       : slot.cf.controlFlowType == CF_EXIT ? "cf-exit"
                       : "cf-loop-cond");
#ifdef SD_CUDA
          {
            const char* cfName = (slot.cf.controlFlowType == CF_ENTER) ? "ENTER" :
                                  (slot.cf.controlFlowType == CF_EXIT) ? "EXIT" : "LOOP_COND";
            verifyCfSlotWrite(stepIdx, cfName, slot.ident.opName.c_str(),
                              outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
          }
#endif
          break;

        case CF_NEXT_ITERATION: {
          // Forward input[0] to output[0], then jump back to Merge
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       "cf-next-iter");
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "NEXT_ITER", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif

          if (slot.cf.loopBackTarget >= 0 && slot.cf.loopBackTarget >= seg.startSlot) {
            loopIterations++;
            if (loopIterations >= MAX_LOOP_ITERATIONS) {
              DSP_DIAG(EXECUTE, "loop iteration limit (%d) reached at slot %d",
                        MAX_LOOP_ITERATIONS, stepIdx);
              return Status::KERNEL_FAILURE;
            }
            // Clear dead flags for loop body range
            if (slotIsDead_ && slot.cf.loopRegionIndex >= 0 && slot.cf.loopRegionIndex < numLoopRegions_) {
              LoopRegion& lr = loopRegions_[slot.cf.loopRegionIndex];
              for (int s = lr.mergeSlot; s <= lr.bodyEndSlot && s < numSlots_; s++) {
                NativeSlot& bodySlot = slots_[s];
                for (int oi = 0; oi < bodySlot.wiring.numOutputs; oi++) {
                  int si = bodySlot.wiring.outputSlotIndices[oi];
                  if (si >= 0 && si < slotIsDeadSize_) slotIsDead_[si] = false;
                }
              }
            }
            stepIdx = slot.cf.loopBackTarget;
            continue; // jump back to Merge, don't increment stepIdx
          }
          break;
        }

        default:
          break;
      }

      // Release schedule removed: arrays persist (one array per slot, never nullified)

      stepIdx++;
      continue;
    }

    // ── Dead propagation for regular ops in CF graphs ────────────────
    if (hasControlFlow_ && slotIsDead_ != nullptr) {
      if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
        markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
        stepIdx++;
        continue;
      }
    }

    // ── Batched GEMM dispatch ─────────────────────────────────────────
    // Strategy: the FIRST member in each group is the trigger.
    // When reached, it executes the entire batch and populates outputs for
    // ALL members. Non-first members are skipped (output already computed).
    // This ensures downstream ops between members see valid outputs.
#ifdef SD_CUDA
    if (!batchedGemmGroups_.empty() && stepIdx < (int)slotToBatchedGemmGroup_.size()) {
      int bgIdx = slotToBatchedGemmGroup_[stepIdx];
      if (bgIdx >= 0 && bgIdx < (int)batchedGemmGroups_.size()) {
        auto& bgGroup = batchedGemmGroups_[bgIdx];
        if (stepIdx == bgGroup.triggerSlot) {
          // This is the trigger (FIRST slot in group) — execute entire batch.
          // All members' inputs are guaranteed available (checked at detection time).
          cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
          Status batchStatus = executeBatchedGemmGroup(bgIdx, externalArrays, numExt, execStream);

          if (batchStatus == Status::OK) {
            // Release schedule removed: arrays persist (one array per slot)
            stepIdx++;
            continue;
          }
          // On failure, fall through to individual execution of this slot
          DSP_DIAG(FALLBACK, "batched GEMM group %d failed (status=%d), falling back to individual execution",
                    bgIdx, (int)batchStatus);
        } else {
          // Non-first member: output already computed by the trigger's batch call.
          // Release schedule removed: arrays persist (one array per slot)
          stepIdx++;
          continue;
        }
      }
    }
#endif

    // ── Normal op execution ──────────────────────────────────────────
    Status status;
    bool retriedAfterTrim = false;
executeSlot_retry:
    try {
      status = executeSlot(stepIdx, externalArrays, numExt, stream);
    } catch (const std::exception& e) {
      std::string msg = e.what();
#ifdef SD_CUDA
      if (!streamIsCapturing &&
          !retriedAfterTrim && (msg.find("cannot allocate") != std::string::npos ||
                                 msg.find("out of memory") != std::string::npos ||
                                 msg.find("Error code: [2]") != std::string::npos)) {
        retriedAfterTrim = true;
        DSP_DIAG_SLOT(MEMORY, stepIdx, "slot %d (%s) OOM, trimming pool and retrying...",
                  stepIdx, slots_[stepIdx].ident.opName.c_str());
        cudaGetLastError();
        if (stream) {
          cudaStream_t execStr = *static_cast<cudaStream_t*>(stream);
          cudaStreamSynchronize(execStr);
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(nullptr));
        {
          cudaMemPool_t pool = nullptr;
          int dev = 0;
          cudaGetDevice(&dev);
          if (cudaDeviceGetMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
            cudaMemPoolTrimTo(pool, 0);
            DSP_DIAG(MEMORY, "trimmed memory pool on device %d", dev);
          }
        }
        goto executeSlot_retry;
      }
#endif
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw exception: %s",
               stepIdx, slots_[stepIdx].ident.opName.c_str(), e.what());
      DSP_DIAG(FALLBACK, "%s", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    } catch (...) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw unknown exception",
               stepIdx, slots_[stepIdx].ident.opName.c_str());
      DSP_DIAG(FALLBACK, "%s", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    }
    // ── Diagnostic: per-slot CUDA error check on warmup execution ──────
    // On the first execution of each segment (warmup), synchronize the device
    // after every slot to catch latent CUDA kernel errors (error 700) at the
    // exact slot that caused them, rather than discovering them hundreds of
    // slots later during an unrelated cudaMallocAsync call.
    // This is expensive (blocks GPU pipeline) but essential for diagnosing
    // stale-pointer bugs in restored cached plan handles.
#ifdef SD_CUDA
    if (status == Status::OK && seg.exec.executionCount == 0 && !streamIsCapturing) {
      cudaError_t syncErr = cudaDeviceSynchronize();
      if (syncErr != cudaSuccess) {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                 "CUDA ERROR 700 DIAGNOSTIC: cudaDeviceSynchronize after slot %d (%s) "
                 "returned error %d (%s). This kernel accessed invalid GPU memory. "
                 "seg=[%d-%d] execCount=%d shapesFrozen=%d",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(),
                 static_cast<int>(syncErr), cudaGetErrorString(syncErr),
                 seg.startSlot, seg.endSlot, executeCount_, static_cast<int>(shapesFrozen_));
        sd_printf("%s\n", buf);
        // Log all inputs to the failing slot
        auto& faultSlot = slots_[stepIdx];
        for (int i = 0; i < faultSlot.wiring.numInputs; i++) {
          int srcIdx = faultSlot.wiring.inputSourceIndices[i];
          NDArray* inp = nullptr;
          if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
            inp = outputSlots_[srcIdx];
          } else if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx >= 0 && extIdx < numExt) inp = externalArrays[extIdx];
          }
          if (inp != nullptr && inp->dataBuffer() != nullptr) {
            sd_printf("  FAULT INPUT[%d] srcIdx=%d: shape=%s special=%p primary=%p "
                      "len=%lld db=%p closed=%d devId=%d\n",
                      i, srcIdx, ShapeUtils::shapeAsString(inp).c_str(),
                      inp->dataBuffer()->special(), inp->dataBuffer()->primary(),
                      (long long)inp->lengthOf(), (void*)inp->dataBuffer(),
                      inp->dataBuffer()->isClosed() ? 1 : 0,
                      inp->dataBuffer()->deviceId());
          } else {
            sd_printf("  FAULT INPUT[%d] srcIdx=%d: %s\n",
                      i, srcIdx, inp ? "db=null" : "null");
          }
        }
        // Log outputs of the failing slot
        for (int i = 0; i < faultSlot.wiring.numOutputs; i++) {
          int si = faultSlot.wiring.outputSlotIndices[i];
          NDArray* out = (si >= 0 && si < totalOutputSlots_) ? outputSlots_[si] : nullptr;
          if (out != nullptr && out->dataBuffer() != nullptr) {
            sd_printf("  FAULT OUTPUT[%d] slotIdx=%d: shape=%s special=%p len=%lld\n",
                      i, si, ShapeUtils::shapeAsString(out).c_str(),
                      out->dataBuffer()->special(), (long long)out->lengthOf());
          } else {
            sd_printf("  FAULT OUTPUT[%d] slotIdx=%d: %s\n", i, si, out ? "db=null" : "null");
          }
        }
        cudaGetLastError(); // clear sticky error
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(syncErr));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
        return Status::KERNEL_FAILURE;
      }
    }
#endif

    if (status != Status::OK) {
      char buf[1024];
      const char* existingMsg =
          sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
      if (existingMsg != nullptr && existingMsg[0] != '\0') {
        snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d: %s",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(),
                 static_cast<int>(status), existingMsg);
      } else {
        snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(), static_cast<int>(status));
      }
      DSP_DIAG(FALLBACK, "%s", buf);

      auto& failedSlot = slots_[stepIdx];
      for (int i = 0; i < failedSlot.wiring.numInputs; i++) {
        int srcIdx = failedSlot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          NDArray* inp = (srcIdx < totalOutputSlots_ ? outputSlots_[srcIdx] : nullptr);
          if (inp != nullptr) {
            // Protect rankOf() call — if shapeInfo is null, rankOf() would throw
            // and propagate out of this catch handler, causing cascading failures.
            try {
              DSP_DIAG(FALLBACK, "  input[%d] from outputSlot[%d]: rank=%lld shapeInfo=%p db=%p",
                        i, srcIdx, (long long)inp->rankOf(),
                        (void*)inp->shapeInfo(), (void*)inp->dataBuffer());
            } catch (...) {
              DSP_DIAG(FALLBACK, "  input[%d] from outputSlot[%d]: ptr=%p (shapeInfo INVALID)",
                        i, srcIdx, (void*)inp);
            }
          } else {
            DSP_DIAG(FALLBACK, "  input[%d] from outputSlot[%d]: null", i, srcIdx);
          }
        } else {
          DSP_DIAG(FALLBACK, "  input[%d] from external[%d]", i, -(srcIdx + 1));
        }
      }

      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(status));
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);

#ifdef SD_CUDA
      cudaGetLastError();
#endif
      return status;
    }

    // Classify ownership for all outputs produced by this slot.
    // This populates slotOwnership_[si] with SLOT_OWNED, VIEW_OF_SLOT,
    // VIEW_OF_WEIGHT, etc., and maintains viewRefCount on parent slots.
    // Runs in parallel with existing cleanup logic during Phase 1 integration.
    if (slotOwnership_ != nullptr) {
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        int si = slot.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          classifyAndUpdateOwnership(
              slotOwnership_[si], outputSlots_[si], si,
              externalArrays, numExt,
              outputSlots_, totalOutputSlots_,
              slotOwnership_);
        }
      }
    }

    // Record op for FunctionalReplayHandle capture
    if (seg.exec.replayHandle && seg.exec.replayHandle->getState() == ReplayState::CAPTURING) {
      auto* funcHandle = dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());
      if (funcHandle) funcHandle->recordOp(slot.ident.op, stepIdx);
    }

    // Release schedule removed: arrays persist (one array per slot, never nullified).
    // Same plan = same shapes. Arrays allocated on first execution, reused forever.

    stepIdx++;
  }

  if (!viewProducerDetectionDone_) {
    viewProducerDetectionDone_ = true;
    int viewCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotIsViewProducer_[i]) viewCount++;
    }
    DSP_DIAG(SHAPE, "view producer detection done: %d/%d output slots are view producers",
              viewCount, totalOutputSlots_);
  }

  seg.exec.executionCount++;
  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Emulated Graph Replay
// ═══════════════════════════════════════════════════════════════════════════════
//
// Executes ops slot-by-slot but emulates the full graph replay lifecycle:
//   executionCount == 0 (WARMUP): Record baseline shape key + address snapshot
//   executionCount == 1 (EMULATED_CAPTURE): Verify shape/address stability
//   executionCount >= 2 (EMULATED_REPLAY): Steady-state with stability tracking
//
// Emits DSP_DIAG_EMULATED_REPLAY diagnostics at every phase, reporting what a
// real CUDA graph replay backend would see. This lets users diagnose graph
// replay failures without needing actual CUDA graph capture.
// ═══════════════════════════════════════════════════════════════════════════════

LongType NativeDynamicShapePlan::computeSegmentInputAddrKeyPortable(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  // FNV-1a hash of buffer addresses for all segment inputs (external + cross-segment).
  // On CUDA, uses specialBuffer(); on CPU, uses primaryBuffer().
  // Address changes between executions indicate the graph would have stale pointers.
  LongType hash = 0xcbf29ce484222325ULL;
  auto mix = [&hash](uintptr_t val) {
    hash ^= val;
    hash *= 0x100000001b3ULL;
  };

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      NDArray* arr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt) arr = externalInputs[extIdx];
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        arr = outputSlots_[srcIdx];
      }
      if (arr != nullptr) {
#if defined(SD_CUDA)
        mix(reinterpret_cast<uintptr_t>(arr->specialBuffer()));
#else
        mix(reinterpret_cast<uintptr_t>(arr->buffer()));
#endif
      } else {
        mix(0);  // nullptr sentinel
      }
    }
  }
  return hash;
}

Status NativeDynamicShapePlan::executeSegmentEmulatedReplay(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  int segSize = seg.endSlot - seg.startSlot + 1;
  int execCount = seg.exec.executionCount;

  // ── Phase determination ─────────────────────────────────────────────────
  const char* phaseName;
  if (execCount == 0) {
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
    phaseName = "WARMUP";
  } else if (execCount == 1) {
    seg.exec.currentPhase = ExecutionPhase::COMPILING;  // "capture" equivalent
    phaseName = "EMULATED_CAPTURE";
  } else {
    seg.exec.currentPhase = ExecutionPhase::REPLAYING;
    phaseName = "EMULATED_REPLAY";
  }

  DSP_DIAG(EMULATED_REPLAY,
           "EMULATED seg[%d-%d] phase=%s execCount=%d slots=%d capturable=%d frozen=%d",
           seg.startSlot, seg.endSlot, phaseName, execCount, segSize,
           seg.isCapturable ? 1 : 0, shapesFrozen_ ? 1 : 0);

  // ── Gap 1: Fast path — skip key recomputation when stable ──────────────
  // When argTableStable was set on the previous execution (both shape and addr
  // keys matched), skip the expensive hash computations and go straight to
  // slot-by-slot execution. This eliminates shape key overhead (~5-10us per
  // segment) that real graph replay also avoids.
  bool fastPath = false;
  if (execCount >= 2 && seg.exec.argTableStable) {
    fastPath = true;
    DSP_DIAG(EMULATED_REPLAY,
             "  FAST PATH: argTableStable=true from previous step, skipping key recomputation");
  }

  LongType currentShapeKey = 0;
  LongType currentAddrKey = 0;

  if (!fastPath) {
    // ── Compute shape key ──────────────────────────────────────────────────
    currentShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    currentAddrKey = computeSegmentInputAddrKeyPortable(seg, externalArrays, numExt);
  }

  if (execCount == 0) {
    // ══════════════════════════════════════════════════════════════════════
    // WARMUP: baseline keys + fusion analysis + capture buffer sizing + DOT
    // ══════════════════════════════════════════════════════════════════════
    seg.exec.cachedShapeKey = currentShapeKey;
    seg.exec.capturedInputAddrKey = currentAddrKey;
    seg.exec.argTableStable = false;

    DSP_DIAG(EMULATED_REPLAY,
             "  WARMUP baseline: shapeKey=0x%llx addrKey=0x%llx",
             (long long)currentShapeKey, (long long)currentAddrKey);

    // ── Gap 3: Capture buffer sizing (byte-level) ────────────────────────
    int numPlaceholders = 0, numConstants = 0, numVariables = 0;
    size_t placeholderBytes = 0, constantBytes = 0, variableBytes = 0;
    // Track unique external indices to avoid double-counting
    std::unordered_set<int> seenExt;

    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (seenExt.count(extIdx)) continue;
          seenExt.insert(extIdx);

          int8_t srcType = slot.wiring.inputSourceTypes[i];
          size_t bytes = 0;
          if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
            bytes = externalArrays[extIdx]->lengthOf() * externalArrays[extIdx]->sizeOfT();
          }

          if (srcType == SOURCE_PLACEHOLDER) {
            numPlaceholders++;
            placeholderBytes += bytes;
          } else if (srcType == SOURCE_CONSTANT) {
            numConstants++;
            constantBytes += bytes;
          } else if (srcType == SOURCE_VARIABLE) {
            numVariables++;
            variableBytes += bytes;
          }
        }
      }
    }

    DSP_DIAG(EMULATED_REPLAY,
             "  capture buffers: %d placeholders (%zuKB staging needed), "
             "%d constants (%zuKB direct ref), %d variables (%zuKB direct ref if frozen)",
             numPlaceholders, placeholderBytes / 1024,
             numConstants, constantBytes / 1024,
             numVariables, variableBytes / 1024);

    // Per-placeholder detail for large inputs
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      seenExt.clear();
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (seenExt.count(extIdx)) continue;
            seenExt.insert(extIdx);
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER &&
                extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              size_t bytes = arr->lengthOf() * arr->sizeOfT();
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] PLACEHOLDER shape=[%s] dtype=%d bytes=%zu",
                       extIdx, ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType(), bytes);
            }
          }
        }
      }
    }

    // ── Gap 2: Kernel fusion analysis ────────────────────────────────────
    int numIdentity = 0, numViewOps = 0, numFusedChains = 0, numFusedTails = 0;
    int numInPlaceFused = 0, numDataDependent = 0, numControlFlow = 0;
    int numMatmul = 0, numElementwise = 0, numOther = 0;
    int totalFusedChainOps = 0;

    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      if (slot.flags.isIdentityOp)     numIdentity++;
      if (slot.flags.isViewCapableOp)  numViewOps++;
      if (slot.fusedChain.isFusedChainHead) { numFusedChains++; totalFusedChainOps += slot.fusedChain.fusedChainLength; }
      if (slot.fusedChain.isFusedChainTail) numFusedTails++;
      if (slot.flags.inPlaceFused)     numInPlaceFused++;
      if (slot.flags.isDataDependent)  numDataDependent++;
      if (slot.cf.controlFlowType != CF_NONE) numControlFlow++;

      // Classify by op name heuristic
      const auto& name = slot.ident.opName;
      if (name.find("matmul") != std::string::npos || name.find("mmul") != std::string::npos ||
          name.find("gemm") != std::string::npos || name.find("batched_gemm") != std::string::npos) {
        numMatmul++;
      } else if (slot.flags.isIdentityOp || slot.flags.isViewCapableOp || slot.fusedChain.isFusedChainTail) {
        // Already counted above — these are "free" ops
      } else {
        // Heuristic: ops with no iArgs, 1-2 inputs, and no data dependency are likely elementwise
        if (!slot.flags.isDataDependent && slot.wiring.numInputs <= 2 && slot.wiring.numOutputs == 1) {
          numElementwise++;
        } else {
          numOther++;
        }
      }
    }

    int eliminatedOps = numIdentity + numFusedTails;
    int effectiveOps = segSize - eliminatedOps;

    DSP_DIAG(EMULATED_REPLAY,
             "  fusion analysis: %d total ops, %d effective (-%d identity, -%d fused tails)",
             segSize, effectiveOps, numIdentity, numFusedTails);
    DSP_DIAG(EMULATED_REPLAY,
             "    matmul=%d elementwise=%d view=%d inPlaceFused=%d dataDep=%d controlFlow=%d other=%d",
             numMatmul, numElementwise, numViewOps, numInPlaceFused,
             numDataDependent, numControlFlow, numOther);
    if (numFusedChains > 0) {
      DSP_DIAG(EMULATED_REPLAY,
               "    fused chains: %d chains covering %d ops (avg %.1f ops/chain)",
               numFusedChains, totalFusedChainOps,
               numFusedChains > 0 ? (float)totalFusedChainOps / numFusedChains : 0.0f);
    }

    // Segment pattern classification
    const char* pattern = "MIXED";
    if (numDataDependent > 0)           pattern = "DATA_DEPENDENT (non-capturable)";
    else if (numMatmul > 0 && numElementwise > 0) pattern = "MATMUL_EPILOGUE (best for graph capture)";
    else if (numMatmul > 0)             pattern = "PURE_MATMUL (cuBLAS graph capture)";
    else if (numElementwise == effectiveOps) pattern = "PURE_ELEMENTWISE (best for kernel fusion)";
    else if (numViewOps == segSize)      pattern = "PURE_VIEW (zero compute, identity graph)";

    DSP_DIAG(EMULATED_REPLAY, "    segment pattern: %s", pattern);

    // ── Gap 4: DOT graph topology ────────────────────────────────────────
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      DSP_DIAG(EMULATED_REPLAY, "  DOT_BEGIN seg[%d-%d]", seg.startSlot, seg.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "  digraph segment_%d_%d {", seg.startSlot, seg.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "    rankdir=TB;");
      DSP_DIAG(EMULATED_REPLAY, "    node [shape=box, fontsize=10];");

      // External input nodes
      std::unordered_set<int> emittedExt;
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (emittedExt.count(extIdx)) continue;
            emittedExt.insert(extIdx);
            const char* srcLabel = "EXT";
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) srcLabel = "PH";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_CONSTANT) srcLabel = "CONST";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_VARIABLE) srcLabel = "VAR";
            DSP_DIAG(EMULATED_REPLAY,
                     "    ext_%d [label=\"%s[%d]\", shape=ellipse, style=filled, fillcolor=lightblue];",
                     extIdx, srcLabel, extIdx);
          }
        }
      }

      // Op nodes
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        const char* color = "white";
        if (slot.flags.isIdentityOp)         color = "gray90";
        else if (slot.fusedChain.isFusedChainHead) color = "lightyellow";
        else if (slot.fusedChain.isFusedChainTail) color = "lightyellow";
        else if (slot.flags.isViewCapableOp)  color = "honeydew";
        else if (slot.flags.isDataDependent)  color = "mistyrose";

        DSP_DIAG(EMULATED_REPLAY,
                 "    slot_%d [label=\"[%d] %s\", style=filled, fillcolor=%s];",
                 s, s, slot.ident.opName.empty() ? "?" : slot.ident.opName.c_str(), color);
      }

      // Edges
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            DSP_DIAG(EMULATED_REPLAY, "    ext_%d -> slot_%d;", extIdx, s);
          } else if (srcIdx >= 0) {
            // Find which slot produces this output
            for (int ps = seg.startSlot; ps < s; ps++) {
              NativeSlot& pslot = slots_[ps];
              for (int o = 0; o < pslot.wiring.numOutputs; o++) {
                if (pslot.wiring.outputSlotIndices[o] == srcIdx) {
                  DSP_DIAG(EMULATED_REPLAY, "    slot_%d -> slot_%d;", ps, s);
                  goto nextInput;
                }
              }
            }
            // Cross-segment input
            DSP_DIAG(EMULATED_REPLAY, "    cross_%d [label=\"slot[%d]\", shape=diamond];", srcIdx, srcIdx);
            DSP_DIAG(EMULATED_REPLAY, "    cross_%d -> slot_%d;", srcIdx, s);
            nextInput:;
          }
        }
      }

      DSP_DIAG(EMULATED_REPLAY, "  }");
      DSP_DIAG(EMULATED_REPLAY, "  DOT_END seg[%d-%d]", seg.startSlot, seg.endSlot);
    }

  } else if (!fastPath) {
    // ══════════════════════════════════════════════════════════════════════
    // POST-WARMUP: stability checks (not on fast path)
    // ══════════════════════════════════════════════════════════════════════
    bool shapeStable = (currentShapeKey == seg.exec.cachedShapeKey);
    bool addrStable = (currentAddrKey == seg.exec.capturedInputAddrKey);

    const char* shapeVerdict = shapeStable ? "STABLE" : "CHANGED";
    const char* addrVerdict = addrStable ? "STABLE" : "CHANGED";

    DSP_DIAG(EMULATED_REPLAY,
             "  stability: shape=%s (0x%llx vs cached 0x%llx) addr=%s (0x%llx vs cached 0x%llx)",
             shapeVerdict,
             (long long)currentShapeKey, (long long)seg.exec.cachedShapeKey,
             addrVerdict,
             (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey);

    if (!shapeStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** SHAPE KEY CHANGED: CUDA graph would INVALIDATE and re-capture. "
               "Identify which input shapes changed between executions.");

      // Detailed: find which external inputs changed shape
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] type=%d shape=[%s] dtype=%d",
                       extIdx, (int)slot.wiring.inputSourceTypes[i],
                       ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType());
            }
          }
        }
      }

      seg.exec.cachedShapeKey = currentShapeKey;
    }

    if (!addrStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** ADDRESS KEY CHANGED: capture buffer D2D copies needed. "
               "Placeholders with new addresses require staging buffer updates.");
      seg.exec.capturedInputAddrKey = currentAddrKey;
    }

    // Replay readiness assessment
    if (shapeStable && addrStable) {
      seg.exec.argTableStable = true;  // Enable fast path on next step
      DSP_DIAG(EMULATED_REPLAY,
               "  REPLAY READY: shapes and addresses stable — "
               "CUDA graph replay would succeed without re-capture. (fast path enabled)");
    } else {
      seg.exec.argTableStable = false;  // Disable fast path
      if (shapeStable && !addrStable) {
        DSP_DIAG(EMULATED_REPLAY,
                 "  REPLAY with D2D: shapes stable but addresses changed — "
                 "CUDA graph would replay after capture buffer D2D copies.");
      } else {
        DSP_DIAG(EMULATED_REPLAY,
                 "  RE-CAPTURE needed: shape change requires full graph re-capture.");
      }
    }
  }
  // else: fast path — no key computation, no stability check, just execute

  // ── Execute ops slot-by-slot ────────────────────────────────────────────
  auto tSlotStart = std::chrono::high_resolution_clock::now();

  auto status = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  auto tSlotEnd = std::chrono::high_resolution_clock::now();
  auto slotUs = std::chrono::duration_cast<std::chrono::microseconds>(tSlotEnd - tSlotStart).count();

  // Dispatch overhead estimate: ~15us per effective op (shape inference + dispatch)
  // Identity/fused-tail ops are skipped by executeSlot, so don't count them.
  int estimatedSkippedOps = 0;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    if (slots_[s].flags.isIdentityOp || slots_[s].fusedChain.isFusedChainTail) estimatedSkippedOps++;
  }
  int effectiveDispatchOps = segSize - estimatedSkippedOps;
  long long estimatedDispatchUs = effectiveDispatchOps * 15LL;

  DSP_DIAG(EMULATED_REPLAY,
           "  execution: %lldus total (%d ops, %d dispatched, %d skipped)%s",
           (long long)slotUs, segSize, effectiveDispatchOps, estimatedSkippedOps,
           fastPath ? " [FAST PATH]" : "");
  DSP_DIAG(EMULATED_REPLAY,
           "  overhead estimate: ~%lldus dispatch + ~%lldus compute = %lldus. "
           "Graph replay would save ~%lldus (%.0f%%)",
           estimatedDispatchUs,
           (long long)slotUs - estimatedDispatchUs,
           (long long)slotUs,
           estimatedDispatchUs,
           slotUs > 0 ? (100.0 * estimatedDispatchUs / slotUs) : 0.0);

  if (status != Status::OK) {
    seg.exec.argTableStable = false;  // Force stability re-check on next step
    DSP_DIAG(EMULATED_REPLAY,
             "  ** EXECUTION FAILED: status=%d — graph capture would also fail here",
             (int)status);
  }

  // Note: executeSegmentSlotBySlot already increments seg.exec.executionCount
  return status;
}

}  // namespace graph
}  // namespace sd
