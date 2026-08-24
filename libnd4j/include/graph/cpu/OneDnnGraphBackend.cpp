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

#include <config.h>

#if HAVE_ONEDNN

#include <graph/cpu/OneDnnGraphBackend.h>
#include <graph/cpu/OneDnnGraphEmitterCatalog.h>
#include <graph/DspDiagnostics.h>
#include <ops/declarable/platform/mkldnn/OnednnVersionProvider.h>
#include <system/Environment.h>

#include <algorithm>
#include <climits>
#include <mutex>
#include <thread>

namespace sd {
namespace graph {

// ─── Thread-local stream ────────────────────────────────────────────────────

static thread_local std::unique_ptr<dnnl::stream> tls_onednn_stream;

dnnl::stream& OneDnnGraphBackend::getThreadStream() {
  if (!tls_onednn_stream) {
    tls_onednn_stream = std::make_unique<dnnl::stream>(engine_);
  }
  return *tls_onednn_stream;
}

// ─── Singleton ──────────────────────────────────────────────────────────────

OneDnnGraphBackend& OneDnnGraphBackend::getInstance() {
  static OneDnnGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new OneDnnGraphBackend();
  });
  return *instance;
}

OneDnnGraphBackend::OneDnnGraphBackend()
    : engine_(dnnl::engine::kind::cpu, 0) {
  // Sync OMP thread count with Environment (controlled via -Domp.num.threads).
  // OneDNN with DNNL_CPU_RUNTIME=OMP uses omp_get_max_threads() at execution time.
  // KMP_BLOCKTIME/KMP_AFFINITY/GOMP_SPINCOUNT are already configured globally
  // by CoreConfig::initFromEnvironment() — no need to set them again here.
  int numThreads = sd::Environment::getInstance().maxMasterThreads();
  if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
  omp_set_num_threads(numThreads);

  DSP_DIAG(COMPILE, "OneDNN: configured %d OMP threads (blocktime/affinity set by CoreConfig)",
           numThreads);
}

OneDnnGraphBackend::~OneDnnGraphBackend() = default;

// ─── Availability ───────────────────────────────────────────────────────────

bool OneDnnGraphBackend::isAvailable() const {
  // -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
  static std::once_flag selfTestFlag;
  std::call_once(selfTestFlag, []() {
    try {
      dg::graph selfTest(dnnl::engine::kind::cpu);
      auto st_in0 = dg::logical_tensor(90000, dg::logical_tensor::data_type::f32,
                                         {1, 7, 1024}, dg::logical_tensor::layout_type::strided);
      auto st_in1 = dg::logical_tensor(90001, dg::logical_tensor::data_type::f32,
                                         {1024, 2048}, dg::logical_tensor::layout_type::strided);
      auto st_out = dg::logical_tensor(90002, dg::logical_tensor::data_type::f32,
                                         {1, 7, 2048}, dg::logical_tensor::layout_type::strided);
      dg::op st_mm(90003, dg::op::kind::MatMul, "selftest_matmul");
      st_mm.set_attr<bool>(dg::op::attr::transpose_a, false);
      st_mm.set_attr<bool>(dg::op::attr::transpose_b, false);
      st_mm.add_inputs({st_in0, st_in1});
      st_mm.add_outputs({st_out});
      selfTest.add_op(st_mm);
      selfTest.finalize();
      auto stParts = selfTest.get_partitions();
      DSP_DIAG(COMPILE, "OneDNN SELF-TEST: %d partitions, first supported=%d",
               static_cast<int>(stParts.size()),
               stParts.empty() ? -1 : (stParts[0].is_supported() ? 1 : 0));
    } catch (const std::exception& e) {
      DSP_DIAG(COMPILE, "OneDNN SELF-TEST: EXCEPTION: %s", e.what());
    }
  });
  return sd::ops::platforms::onednn::OnednnVersionProvider::hasGraphApi();
}

bool OneDnnGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY ||
         request.executionMode == GraphExecutionMode::GEM_ONEDNN;
}

int OneDnnGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_ONEDNN ? 1000 : 500;
}

// ─── Data type mapping ──────────────────────────────────────────────────────

dg::logical_tensor::data_type OneDnnGraphBackend::mapDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return dg::logical_tensor::data_type::f32;
    case DataType::BFLOAT16: return dg::logical_tensor::data_type::bf16;
    case DataType::HALF: return dg::logical_tensor::data_type::f16;
    case DataType::INT32: return dg::logical_tensor::data_type::s32;
    case DataType::INT8: return dg::logical_tensor::data_type::s8;
    case DataType::UINT8: return dg::logical_tensor::data_type::u8;
    case DataType::BOOL: return dg::logical_tensor::data_type::boolean;
    // oneDNN Graph API does NOT support INT64, FLOAT64, UINT16, UINT32, UINT64, etc.
    // Return undef so callers can detect and skip ops with unsupported types.
    default: return dg::logical_tensor::data_type::undef;
  }
}

// ─── Segment fusibility check ───────────────────────────────────────────────

bool OneDnnGraphBackend::canResolveSlot(const GraphBackendRequest& request,
                                        NativeSlot* slots, int slotIndex) {
  return slots != nullptr && slotIndex >= 0 && isResolvable(request) &&
         isAvailable() && findOneDnnGraphEmitter(slots[slotIndex]) != nullptr;
}

bool OneDnnGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) {
    DSP_DIAG(BACKEND, "OneDnnGraphBackend::canFuseSegment: oneDNN not available");
    return false;
  }

  int mappableOps = 0;
  int anchorOps = 0;
  int totalOps = end - start + 1;

  for (int i = start; i <= end; i++) {
    const auto* emitter = findOneDnnGraphEmitter(slots[i]);
    if (emitter != nullptr) {
      mappableOps++;
      if (emitter->anchor) anchorOps++;
    }
  }

  // Require at least one anchor op. A segment of pure elementwise ops
  // gets no benefit from oneDNN — its optimization is in FUSING elementwise
  // into anchors (matmul+relu, conv+bias+gelu, etc.), not running them standalone.
  if (anchorOps < 1) {
    DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: no anchor ops "
             "(mappable=%d totalOps=%d) — skipping",
             start, end, mappableOps, totalOps);
    return false;
  }

  // Accept: at least one anchor op exists, and enough mappable ops for fusion benefit.
  bool coverageOk = mappableOps >= MIN_MAPPABLE_OPS;
  DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: "
           "anchors=%d mappable=%d/%d canFuse=%s",
           start, end, anchorOps, mappableOps, totalOps,
           coverageOk ? "true" : "false");
  return coverageOk;
}

// ─── Thread-local native slot executor ─────────────────────────────────────

thread_local OneDnnGraphBackend::NativeSlotExecutor OneDnnGraphBackend::nativeExecutor_ = nullptr;

void OneDnnGraphBackend::setNativeSlotExecutor(NativeSlotExecutor executor) {
  nativeExecutor_ = std::move(executor);
}

void OneDnnGraphBackend::clearNativeSlotExecutor() {
  nativeExecutor_ = nullptr;
}

// ─── Graph building ─────────────────────────────────────────────────────────
//
// Supports two modes:
//   Pure-OneDNN:  All ops in [startSlot, endSlot] are mappable. One dg::graph
//                 covering all slots, partitioned + compiled by oneDNN.
//   Mixed:        Some ops are unmappable (e.g. gather, rope, stridedslice).
//                 We split the range into consecutive runs of:
//                   - "OneDNN islands": contiguous mappable ops compiled as sub-graphs
//                   - "native ranges": contiguous unmappable ops recorded for
//                     slot-by-slot execution via nativeExecutor_
//                 The executionSchedule vector records the interleaved order.
//
// Unmappable ops that appear between mappable ops are not added to any dg::graph.
// Their input tensors are treated as external inputs for the next OneDNN island
// (since native execution will write those outputs before the next island runs).

OneDnnGraphBackend::CompiledSegment OneDnnGraphBackend::buildGraph(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledSegment result;
  result.valid = false;

  // Helper: resolve an NDArray* from a wiring source index.
  auto resolveWiringArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx >= 0 && srcIdx < totalOutputSlots) return outputSlots[srcIdx];
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs) return externalInputs[extIdx];
    }
    return nullptr;
  };

  // Resolve and validate every slot through the emitter that owns its complete
  // oneDNN contract. Names are diagnostic only; descriptor identity selects the
  // lowering and the lowering decides exact dtype/layout/argument support.
  int totalOps = endSlot - startSlot + 1;
  std::vector<bool> slotMappable(static_cast<size_t>(totalOps), false);
  int mappableOps = 0;
  for (int s = startSlot; s <= endSlot; s++) {
    const auto* emitter = findOneDnnGraphEmitter(slots[s]);
    if (emitter == nullptr) continue;
    std::vector<NDArray*> inputs;
    std::vector<NDArray*> outputs;
    for (int input = 0; input < slots[s].wiring.numInputs; ++input) {
      inputs.push_back(resolveWiringArray(slots[s].wiring.inputSourceIndices[input]));
    }
    for (int output = 0; output < slots[s].wiring.numOutputs; ++output) {
      const int outputSlot = slots[s].wiring.outputSlotIndices[output];
      outputs.push_back(outputSlot >= 0 && outputSlot < totalOutputSlots
                            ? outputSlots[outputSlot]
                            : nullptr);
    }
    OneDnnLoweredOp validationOp(0, emitter->kind, slots[s].ident.opName);
    std::string rejectionReason;
    if (!emitter->lower({slots[s], inputs, outputs}, validationOp,
                        rejectionReason)) {
      DSP_DIAG(COMPILE,
               "OneDnnGraphBackend: slot %d op '%s' rejected by emitter: %s",
               s, slots[s].ident.opName.c_str(), rejectionReason.c_str());
      continue;
    }
    slotMappable[static_cast<size_t>(s - startSlot)] = true;
    ++mappableOps;
  }
  // Framework elementwise fusion is an execution unit, not independent slot
  // metadata.  A native tail intentionally skips because its head executes the
  // complete chain.  Splitting a chain between oneDNN and a native range would
  // therefore leave the tail output at its warmup value.  Keep every chain
  // atomic: lower all of its slots, or route its complete interval through the
  // ordered native executor.
  for (int s = startSlot; s <= endSlot; ++s) {
    const FusedChain& chain = slots[s].fusedChain;
    if (!chain.isFusedChainHead || chain.fusedChainLength < 2) continue;

    bool completeAndMappable = true;
    for (int chainIndex = 0; chainIndex < chain.fusedChainLength; ++chainIndex) {
      const int chainSlot = chain.fusedChainSlots[chainIndex];
      if (chainSlot < startSlot || chainSlot > endSlot) {
        DSP_DIAG(COMPILE,
                 "OneDnnGraphBackend: fused chain head=%d crosses segment "
                 "[%d-%d] at slot=%d; rejecting partial-chain lowering",
                 s, startSlot, endSlot, chainSlot);
        return result;
      }
      if (!slotMappable[static_cast<size_t>(chainSlot - startSlot)]) {
        completeAndMappable = false;
        break;
      }
    }
    if (completeAndMappable) continue;

    DSP_DIAG(COMPILE,
             "OneDnnGraphBackend: fused chain head=%d length=%d is not fully "
             "lowerable; assigning the complete chain to native execution",
             s, chain.fusedChainLength);
    for (int chainIndex = 0; chainIndex < chain.fusedChainLength; ++chainIndex) {
      const int chainSlot = chain.fusedChainSlots[chainIndex];
      if (chainSlot >= startSlot && chainSlot <= endSlot) {
        slotMappable[static_cast<size_t>(chainSlot - startSlot)] = false;
      }
    }
  }

  mappableOps = static_cast<int>(std::count(
      slotMappable.begin(), slotMappable.end(), true));
  result.isMixedSegment = (mappableOps != totalOps);

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::buildGraph [%d-%d]: totalOps=%d mappable=%d mixed=%s",
           startSlot, endSlot, totalOps, mappableOps,
           result.isMixedSegment ? "true" : "false");

  // ── Identify sub-ranges: runs of mappable and unmappable ops ────────────
  // A "sub-range" is [first, last] inclusive plus whether it's native or OneDNN.
  struct SubRange {
    int first, last;
    bool isNative;  // true = unmappable ops for native execution
  };
  std::vector<SubRange> subRanges;
  {
    int cur = startSlot;
    while (cur <= endSlot) {
      bool curMappable = slotMappable[static_cast<size_t>(cur - startSlot)];
      int runEnd = cur;
      while (runEnd + 1 <= endSlot &&
             slotMappable[static_cast<size_t>(runEnd + 1 - startSlot)] == curMappable) {
        runEnd++;
      }
      subRanges.push_back({cur, runEnd, !curMappable});
      cur = runEnd + 1;
    }
  }

  // ── Compile each OneDNN sub-range into a separate dg::graph ─────────────
  // Each OneDNN island is independent: inputs from prior native ranges are treated
  // as external inputs with pre-known shapes.
  //
  // IMPORTANT: Use a single global tensorId counter across all islands.
  // Each island builds its own dg::graph but they all share result.tensorIdToSlotMap.
  // If each island reset tensorId=0, different islands would assign the same numeric
  // ID to different slots/externals, corrupting the tensorIdToSlotMap lookups used
  // at execution time to resolve NDArray* pointers from PartitionEntry tensor IDs.
  size_t globalTensorId = 0;

  for (auto& sr : subRanges) {
    if (sr.isNative) {
      // Record native range: will be executed by nativeExecutor_ at execution time
      int nativeIdx = static_cast<int>(result.nativeRanges.size());
      result.nativeRanges.push_back({sr.first, sr.last});
      result.executionSchedule.push_back({true, nativeIdx});

      // Audit: mark all slots in native range as natively handled
      for (int s = sr.first; s <= sr.last; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].ident.opName;
        entry.wasCompiled = false;
        entry.isNativeHandled = true;
        entry.reason = "unmappable op — executed natively via NativeSlotExecutor";
        result.compilationAudit.push_back(std::move(entry));
      }
      DSP_DIAG(COMPILE, "OneDnnGraphBackend::buildGraph: native range [%d-%d] (%d ops) added to schedule",
               sr.first, sr.last, sr.last - sr.first + 1);
      continue;
    }

    // ── Build one dg::graph for this OneDNN island ──────────────────────
    try {
      dg::graph g(dnnl::engine::kind::cpu);
      // Use globalTensorId (NOT a local reset-to-zero counter) so IDs are unique
      // across all islands within this segment. All islands share result.tensorIdToSlotMap,
      // so collisions would cause wrong NDArray* resolution at execute time.
      size_t& tensorId = globalTensorId;

      std::unordered_map<int, size_t> slotToTensorId;
      std::unordered_map<int, size_t> extToTensorId;
      std::unordered_map<size_t, dg::logical_tensor> logicalTensors;

      // Compile against the exact framework strides. The emitter has already
      // rejected unsupported views, so runtime binding must match this contract.
      auto makeLT = [](size_t id, dg::logical_tensor::data_type dtype,
                       NDArray* array) -> dg::logical_tensor {
        if (array == nullptr || dtype == dg::logical_tensor::data_type::undef) {
          THROW_EXCEPTION("OneDnnGraphBackend: missing exact tensor descriptor");
        }
        const int rank = array->rankOf();
        if (rank == 0) {
          return dg::logical_tensor(
              id, dtype, std::vector<int64_t>{},
              dg::logical_tensor::layout_type::strided);
        }
        std::vector<int64_t> dimensions(rank);
        std::vector<int64_t> strides(rank);
        for (int dimension = 0; dimension < rank; ++dimension) {
          dimensions[dimension] = array->sizeAt(dimension);
          strides[dimension] = array->strideAt(dimension);
        }
        return dg::logical_tensor(id, dtype, dimensions, strides);
      };

      auto getExternalInputTensor = [&](int extIdx) -> dg::logical_tensor {
        auto it = extToTensorId.find(extIdx);
        if (it != extToTensorId.end()) return logicalTensors.at(it->second);

        NDArray* arr = externalInputs[extIdx];
        if (arr == nullptr) THROW_EXCEPTION("OneDnnGraphBackend: null external input");

        size_t id = tensorId++;
        auto dtype = mapDataType(arr->dataType());
        auto lt = makeLT(id, dtype, arr);
        logicalTensors.emplace(id, lt);
        extToTensorId[extIdx] = id;
        result.tensorIdToSlotMap[id] = -(extIdx + 1);
        return lt;
      };

      auto getSlotOutputTensor = [&](int slotIdx, NDArray* arr) -> dg::logical_tensor {
        auto it = slotToTensorId.find(slotIdx);
        if (it != slotToTensorId.end()) return logicalTensors.at(it->second);

        size_t id = tensorId++;
        auto dtype = mapDataType(arr != nullptr ? arr->dataType() : DataType::INHERIT);
        dg::logical_tensor lt = makeLT(id, dtype, arr);
        logicalTensors.emplace(id, lt);
        slotToTensorId[slotIdx] = id;
        result.tensorIdToSlotMap[id] = slotIdx;
        return lt;
      };

      int opsAdded = 0;
      // Map from dg::op ID → slot index, so we can recover which slots belong
      // to each partition after oneDNN partitions the graph.
      std::unordered_map<size_t, int> opIdToSlot;

      for (int s = sr.first; s <= sr.last; s++) {
        NativeSlot& slot = slots[s];
        const auto* emitter = findOneDnnGraphEmitter(slot);
        if (emitter == nullptr) return result;
        size_t opId = tensorId++;
        opIdToSlot[opId] = s;
        std::vector<NDArray*> frameworkInputs;
        std::vector<NDArray*> frameworkOutputs;
        for (int input = 0; input < slot.wiring.numInputs; ++input) {
          frameworkInputs.push_back(
              resolveWiringArray(slot.wiring.inputSourceIndices[input]));
        }
        for (int output = 0; output < slot.wiring.numOutputs; ++output) {
          const int outputSlot = slot.wiring.outputSlotIndices[output];
          frameworkOutputs.push_back(
              outputSlot >= 0 && outputSlot < totalOutputSlots
                  ? outputSlots[outputSlot]
                  : nullptr);
        }
        OneDnnLoweredOp lowered(opId, emitter->kind, slot.ident.opName);
        std::string rejectionReason;
        if (!emitter->lower({slot, frameworkInputs, frameworkOutputs}, lowered,
                            rejectionReason)) {
          DSP_DIAG(COMPILE,
                   "OneDnnGraphBackend: emitter changed admission for slot %d "
                   "op '%s': %s",
                   s, slot.ident.opName.c_str(), rejectionReason.c_str());
          return result;
        }

        // Wire inputs in the exact order declared by the emitter.
        std::vector<dg::logical_tensor> inputTensors;
        for (int frameworkInput : lowered.frameworkInputOrder) {
          if (frameworkInput < 0 || frameworkInput >= slot.wiring.numInputs) {
            return result;
          }
          int srcIdx = slot.wiring.inputSourceIndices[frameworkInput];
          if (srcIdx >= 0) {
            NDArray* arr = (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
            inputTensors.push_back(getSlotOutputTensor(srcIdx, arr));
          } else {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExternalInputs) {
              inputTensors.push_back(getExternalInputTensor(extIdx));
            }
          }
        }
        lowered.operation.add_inputs(inputTensors);

        // Wire outputs
        std::vector<dg::logical_tensor> outputTensors;
        for (int i = 0; i < slot.wiring.numOutputs; i++) {
          int outSlotIdx = slot.wiring.outputSlotIndices[i];
          NDArray* arr = (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots)
                             ? outputSlots[outSlotIdx] : nullptr;
          outputTensors.push_back(getSlotOutputTensor(outSlotIdx, arr));
        }
        lowered.operation.add_outputs(outputTensors);

        // Dump detailed logical tensor info for debugging partition support
        for (size_t ti = 0; ti < inputTensors.size(); ti++) {
          auto& lt = inputTensors[ti];
          auto dims = lt.get_dims();
          std::string shapeStr = "[";
          for (size_t d = 0; d < dims.size(); d++) {
            if (d > 0) shapeStr += ",";
            shapeStr += std::to_string(dims[d]);
          }
          shapeStr += "]";
          DSP_DIAG(COMPILE, "    input[%d] tensorId=%zu dtype=%d ndims=%d shape=%s",
                   static_cast<int>(ti), lt.get_id(),
                   static_cast<int>(lt.get_data_type()),
                   static_cast<int>(dims.size()), shapeStr.c_str());
        }
        for (size_t ti = 0; ti < outputTensors.size(); ti++) {
          auto& lt = outputTensors[ti];
          auto dims = lt.get_dims();
          std::string shapeStr = "[";
          for (size_t d = 0; d < dims.size(); d++) {
            if (d > 0) shapeStr += ",";
            shapeStr += std::to_string(dims[d]);
          }
          shapeStr += "]";
          DSP_DIAG(COMPILE, "    output[%d] tensorId=%zu dtype=%d ndims=%d shape=%s",
                   static_cast<int>(ti), lt.get_id(),
                   static_cast<int>(lt.get_data_type()),
                   static_cast<int>(dims.size()), shapeStr.c_str());
        }
        DSP_DIAG(COMPILE, "  slot %d op '%s' kind=%d numInputs=%d numOutputs=%d opId=%zu",
                 s, slot.ident.opName.c_str(), static_cast<int>(emitter->kind),
                 static_cast<int>(inputTensors.size()), static_cast<int>(outputTensors.size()),
                 opId);

        try {
          g.add_op(lowered.operation);
          opsAdded++;
        } catch (const std::exception& e) {
          DSP_DIAG(COMPILE, "OneDNN Graph: add_op failed for slot %d op '%s': %s",
                   s, slot.ident.opName.c_str(), e.what());
          return result;
        }

        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = true;
        result.compilationAudit.push_back(std::move(auditEntry));
      }

      if (opsAdded < 1) continue;  // Empty island — shouldn't happen

      g.finalize();

      auto partitions = g.get_partitions();
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: OneDNN island [%d-%d] → %d partitions (%d ops)",
               sr.first, sr.last, static_cast<int>(partitions.size()), opsAdded);
      if (partitions.empty()) {
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: no partitions for island [%d-%d]", sr.first, sr.last);
        continue;
      }

      // Every catalog-admitted operation must be accepted by oneDNN. An
      // unsupported partition means the catalog claim is wrong for this concrete
      // contract, so fail lowering before execution begins and let the resolver
      // choose the next backend. Never introduce an unplanned runtime fallback.

      struct PartitionSlotInfo {
        int minSlot, maxSlot;
        int partitionIdx;  // into partitions vector
        bool supported;
        // For supported partitions:
        std::vector<dg::logical_tensor> inputLTs, outputLTs;
      };
      std::vector<PartitionSlotInfo> partInfos;

      int supportedCount = 0;
      for (size_t partIdx = 0; partIdx < partitions.size(); partIdx++) {
        auto& partition = partitions[partIdx];
        PartitionSlotInfo info;
        info.supported = partition.is_supported();
        info.partitionIdx = -1;

        // Determine which slots this partition covers
        auto opIds = partition.get_ops();
        info.minSlot = INT_MAX;
        info.maxSlot = INT_MIN;
        for (auto opId : opIds) {
          auto slotIt = opIdToSlot.find(opId);
          if (slotIt != opIdToSlot.end()) {
            info.minSlot = std::min(info.minSlot, slotIt->second);
            info.maxSlot = std::max(info.maxSlot, slotIt->second);
          }
        }
        DSP_DIAG(COMPILE, "  partition[%d] supported=%d numOps=%d opIds=[%s] minSlot=%d maxSlot=%d",
                 static_cast<int>(partIdx), info.supported ? 1 : 0,
                 static_cast<int>(opIds.size()),
                 opIds.empty() ? "" : std::to_string(opIds[0]).c_str(),
                 info.minSlot, info.maxSlot);
        if (info.minSlot == INT_MAX) {
          DSP_DIAG(COMPILE, "  partition[%d] rejected: no slot mapping for any opId", static_cast<int>(partIdx));
          return result;
        }

        if (!info.supported) {
          DSP_DIAG(COMPILE,
                   "OneDnnGraphBackend: catalog-admitted island [%d-%d] produced "
                   "unsupported partition [%d-%d]",
                   sr.first, sr.last, info.minSlot, info.maxSlot);
          return result;
        }
        supportedCount++;
          auto inPorts = partition.get_input_ports();
          auto outPorts = partition.get_output_ports();
          for (auto& lt : inPorts) {
            auto it = logicalTensors.find(lt.get_id());
            if (it != logicalTensors.end()) info.inputLTs.push_back(it->second);
          }
          for (auto& lt : outPorts) {
            auto it = logicalTensors.find(lt.get_id());
            if (it != logicalTensors.end()) info.outputLTs.push_back(it->second);
          }

          try {
            CompiledSegment::PartitionEntry entry;
            entry.compiledPartition = partition.compile(info.inputLTs, info.outputLTs, engine_);
            entry.startSlot = info.minSlot;
            entry.endSlot = info.maxSlot;
            for (auto& lt : info.inputLTs) entry.inputTensorIds.push_back(lt.get_id());
            for (auto& lt : info.outputLTs) entry.outputTensorIds.push_back(lt.get_id());
            info.partitionIdx = static_cast<int>(result.partitions.size());
            result.partitions.push_back(std::move(entry));
          } catch (const std::exception& e) {
            DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] partition [%d-%d] compile failed: %s",
                     sr.first, sr.last, info.minSlot, info.maxSlot, e.what());
            return result;
          }

        partInfos.push_back(std::move(info));
      }

      // Sort by slot order so execution schedule is correct
      std::sort(partInfos.begin(), partInfos.end(),
                [](const PartitionSlotInfo& a, const PartitionSlotInfo& b) {
                  return a.minSlot < b.minSlot;
                });

      // Build the oneDNN portion of the interleaved schedule. Native ranges in
      // this schedule came only from slots absent from the exact emitter catalog.
      for (auto& info : partInfos) {
        if (info.supported && info.partitionIdx >= 0) {
          result.executionSchedule.push_back({false, info.partitionIdx});
        } else {
          return result;
        }
      }
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] compiled %d partitions",
               sr.first, sr.last, supportedCount);

    } catch (const std::exception& e) {
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] build failed: %s",
               sr.first, sr.last, e.what());
      return result;
    }
  }

  result.valid = !result.partitions.empty();
  if (result.isMixedSegment) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: mixed segment [%d-%d] compiled: "
             "%d partitions, %d native ranges, %d schedule steps, valid=%s",
             startSlot, endSlot,
             static_cast<int>(result.partitions.size()),
             static_cast<int>(result.nativeRanges.size()),
             static_cast<int>(result.executionSchedule.size()),
             result.valid ? "true" : "false");
  } else {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: pure-OneDNN segment [%d-%d] compiled: "
             "%d partitions, valid=%s",
             startSlot, endSlot,
             static_cast<int>(result.partitions.size()),
             result.valid ? "true" : "false");
  }

  return result;
}

// ─── Compile segment ────────────────────────────────────────────────────────

bool OneDnnGraphBackend::compileSegment(
    const GraphBackendRequest& request, GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots, LongType shapeKey,
    int totalSlots, int* requestedOutputSlotIndices,
    int numRequestedOutputs) {
  if (!compileSegment(seg, slots, externalInputs, numExternalInputs,
                      outputSlots, totalOutputSlots, shapeKey, totalSlots,
                      requestedOutputSlotIndices, numRequestedOutputs)) {
    return false;
  }
  if (request.executionMode != GraphExecutionMode::GEM_ONEDNN) return true;
  auto compiled = std::static_pointer_cast<CompiledSegment>(
      seg.compiledGraphBackendArtifact);
  std::lock_guard<std::mutex> executionLock(*compiled->executionMtx);
  if (compiled->nativeRanges.empty()) return true;
  DSP_DIAG(COMPILE,
           "OneDnnGraphBackend: strict mode rejected seg[%d-%d]: %d native ranges",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(compiled->nativeRanges.size()));
  seg.clearCompiledGraphBackendArtifact();
  return false;
}

bool OneDnnGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  if (seg.compiledGraphBackendArtifactOwner == this &&
      seg.compiledGraphBackendArtifactShapeKey == shapeKey &&
      seg.compiledGraphBackendArtifact) {
    auto existing = std::static_pointer_cast<CompiledSegment>(
        seg.compiledGraphBackendArtifact);
    std::lock_guard<std::mutex> registryLock(cacheMtx_);
    std::lock_guard<std::mutex> executionLock(*existing->executionMtx);
    if (existing->valid) {
      lastCompilationAudit_ = existing->compilationAudit;
      DSP_DIAG(JIT,
               "OneDnnGraphBackend::compileSegment [%d-%d]: segment artifact HIT "
               "(shapeKey=0x%llx)",
               seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);
      return true;
    }
  }

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: cache MISS, building graph (shapeKey=0x%llx)",
           seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);

  auto compiled = std::make_shared<CompiledSegment>(
      buildGraph(slots, seg.def.startSlot, seg.def.endSlot,
                 externalInputs, numExternalInputs,
                 outputSlots, totalOutputSlots));
  compiled->shapeKey = shapeKey;

  {
    std::lock_guard<std::mutex> registryLock(cacheMtx_);
    lastCompilationAudit_ = compiled->compilationAudit;
  }

  if (compiled->valid) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: SUCCESS partitions=%d",
             seg.def.startSlot, seg.def.endSlot, (int)compiled->partitions.size());
    {
      std::lock_guard<std::mutex> registryLock(cacheMtx_);
      compiledArtifacts_.erase(
          std::remove_if(compiledArtifacts_.begin(), compiledArtifacts_.end(),
                         [](const std::weak_ptr<CompiledSegment>& artifact) {
                           return artifact.expired();
                         }),
          compiledArtifacts_.end());
      compiledArtifacts_.push_back(compiled);
    }
    seg.compilationAudit = compiled->compilationAudit;
    seg.setCompiledGraphBackendArtifact(this, shapeKey, compiled);
    return true;
  }

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: FAILED",
           seg.def.startSlot, seg.def.endSlot);
  return false;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> OneDnnGraphBackend::getLastCompilationAudit() const {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  return lastCompilationAudit_;
}

// ─── Execute segment ────────────────────────────────────────────────────────

Status OneDnnGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  if (seg.compiledGraphBackendArtifactOwner != this ||
      !seg.compiledGraphBackendArtifact) {
    DSP_DIAG(EXECUTE,
             "OneDnnGraphBackend::executeSegment [%d-%d]: no owned artifact",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }
  auto compiledHandle = std::static_pointer_cast<CompiledSegment>(
      seg.compiledGraphBackendArtifact);
  std::lock_guard<std::mutex> executionLock(*compiledHandle->executionMtx);
  CompiledSegment* compiled = compiledHandle.get();
  if (!compiled->valid || compiled->shapeKey != seg.def.shapeKeyState.compiledShapeKey ||
      seg.compiledGraphBackendArtifactShapeKey != compiled->shapeKey) {
    DSP_DIAG(EXECUTE,
             "OneDnnGraphBackend::executeSegment [%d-%d]: stale artifact "
             "artifactKey=0x%llx compiledKey=0x%llx",
             seg.def.startSlot, seg.def.endSlot,
             (long long)compiled->shapeKey,
             (long long)seg.def.shapeKeyState.compiledShapeKey);
    return Status::KERNEL_FAILURE;
  }

  auto& strm = getThreadStream();

  // Helper: resolve NDArray from slot index
  auto resolveArray = [&](int slotIdx) -> NDArray* {
    if (slotIdx < 0) {
      int extIdx = -(slotIdx + 1);
      return (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
    }
    return (slotIdx < totalOutputSlots) ? outputSlots[slotIdx] : nullptr;
  };

  auto matchesCompiledDescriptor = [&](NDArray* array,
                                       const dg::logical_tensor& tensor) {
    if (array == nullptr || mapDataType(array->dataType()) != tensor.get_data_type()) {
      return false;
    }
    const auto dimensions = tensor.get_dims();
    if (dimensions.size() != static_cast<size_t>(array->rankOf())) return false;
    const auto strides = tensor.get_strides();
    if (!strides.empty() && strides.size() != dimensions.size()) return false;
    for (int dimension = 0; dimension < array->rankOf(); ++dimension) {
      if (dimensions[static_cast<size_t>(dimension)] != array->sizeAt(dimension)) {
        return false;
      }
      if (!strides.empty() &&
          strides[static_cast<size_t>(dimension)] != array->strideAt(dimension)) {
        return false;
      }
    }
    return true;
  };

  // Validate every oneDNN boundary before the schedule starts. Returning
  // BAD_GRAPH here is a pre-execution lowering miss, so the resolver may choose
  // another backend without risking partial execution.
  for (auto& part : compiled->partitions) {
    if (part.cachedInputTensors.empty()) {
      part.cachedInputTensors.resize(part.inputTensorIds.size());
      for (size_t index = 0; index < part.inputTensorIds.size(); ++index) {
        const size_t tensorId = part.inputTensorIds[index];
        if (compiled->tensorIdToSlotMap.find(tensorId) ==
            compiled->tensorIdToSlotMap.end()) {
          return Status::BAD_GRAPH;
        }
        part.cachedInputTensors[index].lt =
            part.compiledPartition.query_logical_tensor(tensorId);
      }
    }
    if (part.cachedOutputTensors.empty()) {
      part.cachedOutputTensors.resize(part.outputTensorIds.size());
      for (size_t index = 0; index < part.outputTensorIds.size(); ++index) {
        const size_t tensorId = part.outputTensorIds[index];
        if (compiled->tensorIdToSlotMap.find(tensorId) ==
            compiled->tensorIdToSlotMap.end()) {
          return Status::BAD_GRAPH;
        }
        part.cachedOutputTensors[index].lt =
            part.compiledPartition.query_logical_tensor(tensorId);
      }
    }
    for (size_t index = 0; index < part.inputTensorIds.size(); ++index) {
      const auto mapping =
          compiled->tensorIdToSlotMap.find(part.inputTensorIds[index]);
      if (mapping == compiled->tensorIdToSlotMap.end() ||
          !matchesCompiledDescriptor(resolveArray(mapping->second),
                                     part.cachedInputTensors[index].lt)) {
        return Status::BAD_GRAPH;
      }
    }
    for (size_t index = 0; index < part.outputTensorIds.size(); ++index) {
      const auto mapping =
          compiled->tensorIdToSlotMap.find(part.outputTensorIds[index]);
      if (mapping == compiled->tensorIdToSlotMap.end() ||
          !matchesCompiledDescriptor(resolveArray(mapping->second),
                                     part.cachedOutputTensors[index].lt)) {
        return Status::BAD_GRAPH;
      }
    }
  }

  if (!compiled->isMixedSegment) {
    // ── Pure-OneDNN path: execute all partitions in order ────────────────
    DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: pure-OneDNN, %d partitions",
             seg.def.startSlot, seg.def.endSlot, (int)compiled->partitions.size());

    for (auto& part : compiled->partitions) {
      if (part.cachedInputTensors.empty()) {
        part.cachedInputTensors.resize(part.inputTensorIds.size());
        for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
          size_t tid = part.inputTensorIds[i];
          if (compiled->tensorIdToSlotMap.find(tid) ==
              compiled->tensorIdToSlotMap.end()) {
            return Status::KERNEL_FAILURE;
          }
          part.cachedInputTensors[i].lt =
              part.compiledPartition.query_logical_tensor(tid);
        }
      }
      if (part.cachedOutputTensors.empty()) {
        part.cachedOutputTensors.resize(part.outputTensorIds.size());
        for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
          size_t tid = part.outputTensorIds[i];
          if (compiled->tensorIdToSlotMap.find(tid) ==
              compiled->tensorIdToSlotMap.end()) {
            return Status::KERNEL_FAILURE;
          }
          part.cachedOutputTensors[i].lt =
              part.compiledPartition.query_logical_tensor(tid);
        }
      }

      std::vector<dg::tensor> inputTensors;
      inputTensors.reserve(part.inputTensorIds.size());
      for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.inputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!matchesCompiledDescriptor(arr, part.cachedInputTensors[i].lt)) {
          return Status::KERNEL_FAILURE;
        }
        inputTensors.emplace_back(part.cachedInputTensors[i].lt, engine_, arr->buffer());
      }

      std::vector<dg::tensor> outputTensors;
      outputTensors.reserve(part.outputTensorIds.size());
      for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.outputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!matchesCompiledDescriptor(arr, part.cachedOutputTensors[i].lt)) {
          return Status::KERNEL_FAILURE;
        }
        outputTensors.emplace_back(part.cachedOutputTensors[i].lt, engine_, arr->buffer());
      }

      try {
        part.compiledPartition.execute(strm, inputTensors, outputTensors);
      } catch (const std::exception& e) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: partition execute failed: %s", e.what());
        return Status::KERNEL_FAILURE;
      }
    }

    strm.wait();
    return Status::OK;
  }

  // ── Mixed-segment path: interleave OneDNN partitions and native ranges ──
  DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: mixed segment, "
           "%d schedule steps (%d partitions, %d native ranges)",
           seg.def.startSlot, seg.def.endSlot,
           (int)compiled->executionSchedule.size(),
           (int)compiled->partitions.size(),
           (int)compiled->nativeRanges.size());

  if (compiled->nativeRanges.empty() == false && !nativeExecutor_) {
    DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: MISSING NativeSlotExecutor "
             "for mixed segment (%d native ranges). Call setNativeSlotExecutor() before execute.",
             seg.def.startSlot, seg.def.endSlot, (int)compiled->nativeRanges.size());
    return Status::KERNEL_FAILURE;
  }

  for (const auto& step : compiled->executionSchedule) {
    if (step.isNative) {
      // Execute native range via the plan's slot-by-slot executor
      const auto& nr = compiled->nativeRanges[step.index];
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend: NATIVE range [%d-%d] (step.index=%d)",
               nr.startSlot, nr.endSlot, step.index);
      auto nativeStatus = nativeExecutor_(nr.startSlot, nr.endSlot);
      if (nativeStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: native range [%d-%d] failed with status=%d",
                 nr.startSlot, nr.endSlot, static_cast<int>(nativeStatus));
        return nativeStatus;
      }
    } else {
      // Execute OneDNN partition
      auto& part = compiled->partitions[step.index];
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend: OneDNN partition (index=%d) for island [%d-%d]",
               step.index, part.startSlot, part.endSlot);

      if (part.cachedInputTensors.empty()) {
        part.cachedInputTensors.resize(part.inputTensorIds.size());
        for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
          size_t tid = part.inputTensorIds[i];
          if (compiled->tensorIdToSlotMap.find(tid) ==
              compiled->tensorIdToSlotMap.end()) {
            return Status::KERNEL_FAILURE;
          }
          part.cachedInputTensors[i].lt =
              part.compiledPartition.query_logical_tensor(tid);
        }
      }
      if (part.cachedOutputTensors.empty()) {
        part.cachedOutputTensors.resize(part.outputTensorIds.size());
        for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
          size_t tid = part.outputTensorIds[i];
          if (compiled->tensorIdToSlotMap.find(tid) ==
              compiled->tensorIdToSlotMap.end()) {
            return Status::KERNEL_FAILURE;
          }
          part.cachedOutputTensors[i].lt =
              part.compiledPartition.query_logical_tensor(tid);
        }
      }

      std::vector<dg::tensor> inputTensors;
      inputTensors.reserve(part.inputTensorIds.size());
      for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.inputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!matchesCompiledDescriptor(arr, part.cachedInputTensors[i].lt)) {
          return Status::KERNEL_FAILURE;
        }
        inputTensors.emplace_back(part.cachedInputTensors[i].lt, engine_, arr->buffer());
      }

      std::vector<dg::tensor> outputTensors;
      outputTensors.reserve(part.outputTensorIds.size());
      for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.outputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!matchesCompiledDescriptor(arr, part.cachedOutputTensors[i].lt)) {
          return Status::KERNEL_FAILURE;
        }
        outputTensors.emplace_back(part.cachedOutputTensors[i].lt, engine_, arr->buffer());
      }

      try {
        part.compiledPartition.execute(strm, inputTensors, outputTensors);
        // Flush completed OneDNN work before native ranges read the outputs
        strm.wait();
      } catch (const std::exception& e) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: mixed partition [%d-%d] execute failed: %s",
                 part.startSlot, part.endSlot, e.what());
        return Status::KERNEL_FAILURE;
      }
    }
  }

  return Status::OK;
}

// ─── Cache invalidation ─────────────────────────────────────────────────────

void OneDnnGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& weakArtifact : compiledArtifacts_) {
    if (auto artifact = weakArtifact.lock()) artifact->invalidate();
  }
  compiledArtifacts_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
