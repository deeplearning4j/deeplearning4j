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
#include <graph/DspDiagnostics.h>
#include <helpers/shape.h>
#include <ops/declarable/platform/mkldnn/OnednnVersionProvider.h>

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
  static OneDnnGraphBackend instance;
  return instance;
}

OneDnnGraphBackend::OneDnnGraphBackend()
    : engine_(dnnl::engine::kind::cpu, 0) {}

OneDnnGraphBackend::~OneDnnGraphBackend() = default;

// ─── Availability ───────────────────────────────────────────────────────────

bool OneDnnGraphBackend::isAvailable() const {
  return sd::ops::platforms::onednn::OnednnVersionProvider::hasGraphApi();
}

// ─── Op kind mapping ────────────────────────────────────────────────────────

dg::op::kind OneDnnGraphBackend::mapOpKind(const std::string& opName) {
  // Element-wise binary
  if (opName == "add" || opName == "Add") return dg::op::kind::Add;
  if (opName == "subtract" || opName == "Sub") return dg::op::kind::Subtract;
  if (opName == "multiply" || opName == "Mul") return dg::op::kind::Multiply;
  if (opName == "divide" || opName == "Div" || opName == "RealDiv") return dg::op::kind::Divide;
  if (opName == "minimum" || opName == "Min") return dg::op::kind::Minimum;
  if (opName == "maximum" || opName == "Max") return dg::op::kind::Maximum;

  // Element-wise unary / activations
  if (opName == "relu" || opName == "Relu") return dg::op::kind::ReLU;
  if (opName == "sigmoid" || opName == "Sigmoid") return dg::op::kind::Sigmoid;
  if (opName == "tanh" || opName == "Tanh") return dg::op::kind::Tanh;
  if (opName == "gelu" || opName == "Gelu") return dg::op::kind::GELU;
  if (opName == "elu" || opName == "Elu") return dg::op::kind::Elu;
  if (opName == "exp" || opName == "Exp") return dg::op::kind::Exp;
  if (opName == "log" || opName == "Log") return dg::op::kind::Log;
  if (opName == "abs" || opName == "Abs") return dg::op::kind::Abs;
  if (opName == "sqrt" || opName == "Sqrt") return dg::op::kind::Sqrt;
  if (opName == "square" || opName == "Square") return dg::op::kind::Square;
  if (opName == "pow" || opName == "Pow") return dg::op::kind::Pow;
  if (opName == "clamp" || opName == "ClipByValue" || opName == "clip_by_value") return dg::op::kind::Clamp;
  if (opName == "hardswish" || opName == "HardSwish") return dg::op::kind::HardSwish;
  if (opName == "hardsigmoid" || opName == "HardSigmoid") return dg::op::kind::HardSigmoid;
  if (opName == "mish" || opName == "Mish") return dg::op::kind::Mish;
  if (opName == "round" || opName == "Round") return dg::op::kind::Round;

  // Matrix operations
  if (opName == "matmul" || opName == "MatMul" || opName == "mmul") return dg::op::kind::MatMul;
  if (opName == "batch_matmul" || opName == "BatchMatMul") return dg::op::kind::MatMul;

  // Reduction
  if (opName == "reduce_sum" || opName == "ReduceSum") return dg::op::kind::ReduceSum;
  if (opName == "reduce_mean" || opName == "ReduceMean" || opName == "reduce_mean_bp") return dg::op::kind::ReduceMean;
  if (opName == "reduce_min" || opName == "ReduceMin") return dg::op::kind::ReduceMin;
  if (opName == "reduce_max" || opName == "ReduceMax") return dg::op::kind::ReduceMax;
  if (opName == "reduce_prod" || opName == "ReduceProd") return dg::op::kind::ReduceProd;

  // Normalization
  if (opName == "softmax" || opName == "Softmax") return dg::op::kind::SoftMax;
  if (opName == "log_softmax" || opName == "LogSoftmax") return dg::op::kind::LogSoftmax;
  if (opName == "layer_norm" || opName == "LayerNorm") return dg::op::kind::LayerNorm;
  if (opName == "batchnorm" || opName == "BatchNorm") return dg::op::kind::BatchNormInference;

  // Shape operations
  if (opName == "reshape" || opName == "Reshape" || opName == "reshape_no_copy" || opName == "ReshapeNoCopy") return dg::op::kind::StaticReshape;
  if (opName == "transpose" || opName == "Transpose" || opName == "permute" || opName == "Permute") return dg::op::kind::StaticTranspose;

  // Concat
  if (opName == "concat" || opName == "Concat" || opName == "concat_v2") return dg::op::kind::Concat;

  // Convolution (2D)
  if (opName == "conv2d" || opName == "Conv2D") return dg::op::kind::Convolution;

  // Type casting
  if (opName == "cast" || opName == "Cast") return dg::op::kind::TypeCast;

  // Not mappable to oneDNN graph
  return dg::op::kind::LastSymbol;
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
    default: return dg::logical_tensor::data_type::f32;
  }
}

// ─── Segment fusibility check ───────────────────────────────────────────────

// Ops that benefit from backend execution even as a single op (optimized kernels)
static bool isSingleOpWorthCompiling(const std::string& opName) {
  static const std::unordered_set<std::string> worthwhile = {
    "matmul", "Matmul", "MatMul",
    "softmax", "Softmax",
    "layer_norm", "LayerNorm",
    "batch_norm", "BatchNorm", "batchnorm",
    "conv2d", "Conv2d",
    "avgpool2d", "Avgpool2d",
    "maxpool2d", "Maxpool2d",
  };
  return worthwhile.count(opName) > 0;
}

bool OneDnnGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) {
    DSP_DIAG(BACKEND, "OneDnnGraphBackend::canFuseSegment: oneDNN not available");
    return false;
  }

  int mappableOps = 0;
  int totalOps = end - start + 1;

  for (int i = start; i <= end; i++) {
    auto kind = mapOpKind(slots[i].ident.opName);
    if (kind != dg::op::kind::LastSymbol) {
      mappableOps++;
    } else {
      DSP_DIAG(BACKEND, "OneDnnGraphBackend::canFuseSegment: unmappable op '%s' at slot %d",
               slots[i].ident.opName.c_str(), i);
    }
  }

  bool canFuse = mappableOps == totalOps && mappableOps >= 1;
  DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: mappable=%d/%d canFuse=%s",
           start, end, mappableOps, totalOps, canFuse ? "true" : "false");
  // Accept any segment where all ops are mappable — no minimum op count.
  return canFuse;
}

// ─── Graph building ─────────────────────────────────────────────────────────

OneDnnGraphBackend::CompiledSegment OneDnnGraphBackend::buildGraph(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledSegment result;
  result.valid = false;

  try {
    dg::graph g(dnnl::engine::kind::cpu);
    size_t tensorId = 0;

    // Track tensor IDs for each output slot so we can wire inputs to prior outputs
    // slotToTensorId[outputSlotIdx] = logical tensor ID for that slot's output
    std::unordered_map<int, size_t> slotToTensorId;

    // Track tensor IDs for external inputs
    // extToTensorId[extIdx] = logical tensor ID for that external input
    std::unordered_map<int, size_t> extToTensorId;

    // Track all logical tensors by ID for execution wiring
    std::unordered_map<size_t, dg::logical_tensor> logicalTensors;

    // Helper to get or create a logical tensor for an external input
    auto getExternalInputTensor = [&](int extIdx) -> dg::logical_tensor {
      auto it = extToTensorId.find(extIdx);
      if (it != extToTensorId.end()) {
        return logicalTensors.at(it->second);
      }

      NDArray* arr = externalInputs[extIdx];
      if (arr == nullptr) {
        THROW_EXCEPTION("OneDnnGraphBackend: null external input");
      }

      size_t id = tensorId++;
      auto dtype = mapDataType(arr->dataType());
      int rank = arr->rankOf();
      std::vector<int64_t> shape(rank);
      std::vector<int64_t> strides(rank);
      for (int d = 0; d < rank; d++) {
        shape[d] = arr->sizeAt(d);
        strides[d] = arr->strideAt(d);
      }

      auto lt = dg::logical_tensor(id, dtype, shape, strides);
      logicalTensors.emplace(id, lt);
      extToTensorId[extIdx] = id;
      result.tensorIdToSlotMap[id] = -(extIdx + 1);  // Negative = external
      return lt;
    };

    // Helper to get or create a logical tensor for a slot output
    auto getSlotOutputTensor = [&](int slotIdx, NDArray* arr) -> dg::logical_tensor {
      auto it = slotToTensorId.find(slotIdx);
      if (it != slotToTensorId.end()) {
        return logicalTensors.at(it->second);
      }

      size_t id = tensorId++;
      auto dtype = mapDataType(arr != nullptr ? arr->dataType() : DataType::FLOAT32);

      // For outputs from prior ops in the segment, shape might not be known yet
      // Use strided layout if we have the array
      dg::logical_tensor lt;
      if (arr != nullptr) {
        int rank = arr->rankOf();
        std::vector<int64_t> shape(rank);
        std::vector<int64_t> strides(rank);
        for (int d = 0; d < rank; d++) {
          shape[d] = arr->sizeAt(d);
          strides[d] = arr->strideAt(d);
        }
        lt = dg::logical_tensor(id, dtype, shape, strides);
      } else {
        lt = dg::logical_tensor(id, dtype, dg::logical_tensor::layout_type::strided);
      }

      logicalTensors.emplace(id, lt);
      slotToTensorId[slotIdx] = id;
      result.tensorIdToSlotMap[id] = slotIdx;  // Positive = output slot
      return lt;
    };

    int opsAdded = 0;

    for (int s = startSlot; s <= endSlot; s++) {
      NativeSlot& slot = slots[s];
      auto kind = mapOpKind(slot.ident.opName);
      if (kind == dg::op::kind::LastSymbol) {
        // Can't map this op — record as skipped in the audit
        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = false;
        auditEntry.reason = "unmappable op kind";
        result.compilationAudit.push_back(std::move(auditEntry));
        continue;
      }

      // Create the oneDNN graph op
      dg::op dgOp(tensorId++, kind, slot.ident.opName);

      // Set op-specific attributes
      if (kind == dg::op::kind::MatMul) {
        // Check iArgs for transpose flags
        bool transposeA = (slot.args.numIArgs > 0 && slot.args.iArgs[0] != 0);
        bool transposeB = (slot.args.numIArgs > 1 && slot.args.iArgs[1] != 0);
        dgOp.set_attr<bool>(dg::op::attr::transpose_a, transposeA);
        dgOp.set_attr<bool>(dg::op::attr::transpose_b, transposeB);
      } else if (kind == dg::op::kind::SoftMax || kind == dg::op::kind::LogSoftmax) {
        int64_t axis = -1;
        if (slot.args.numIArgs > 0) axis = static_cast<int64_t>(slot.args.iArgs[0]);
        dgOp.set_attr<int64_t>(dg::op::attr::axis, axis);
      } else if (kind == dg::op::kind::Concat) {
        int64_t axis = 0;
        if (slot.args.numIArgs > 0) axis = static_cast<int64_t>(slot.args.iArgs[0]);
        dgOp.set_attr<int64_t>(dg::op::attr::axis, axis);
      } else if (kind == dg::op::kind::Elu) {
        float alpha = 1.0f;
        if (slot.args.numTArgs > 0) alpha = static_cast<float>(slot.args.tArgs[0]);
        dgOp.set_attr<float>(dg::op::attr::alpha, alpha);
      } else if (kind == dg::op::kind::Clamp) {
        float minVal = -std::numeric_limits<float>::infinity();
        float maxVal = std::numeric_limits<float>::infinity();
        if (slot.args.numTArgs > 0) minVal = static_cast<float>(slot.args.tArgs[0]);
        if (slot.args.numTArgs > 1) maxVal = static_cast<float>(slot.args.tArgs[1]);
        dgOp.set_attr<float>(dg::op::attr::min, minVal);
        dgOp.set_attr<float>(dg::op::attr::max, maxVal);
      } else if (kind == dg::op::kind::StaticReshape) {
        // Reshape needs target shape from iArgs, skipping ordering marker
        // ND4J encodes iArgs as [-order, dim0, dim1, ...] where -99=C, -102=F
        if (slot.args.numIArgs > 0) {
          int startIdx = 0;
          if (slot.args.iArgs[0] < 0 && (slot.args.iArgs[0] == -99 || slot.args.iArgs[0] == -102)) {
            startIdx = 1;
          }
          std::vector<int64_t> shape(slot.args.numIArgs - startIdx);
          for (int i = startIdx; i < slot.args.numIArgs; i++) {
            shape[i - startIdx] = static_cast<int64_t>(slot.args.iArgs[i]);
          }
          dgOp.set_attr<std::vector<int64_t>>(dg::op::attr::shape, shape);
        }
      } else if (kind == dg::op::kind::StaticTranspose) {
        // Transpose needs permutation from iArgs
        std::vector<int64_t> order;
        if (slot.args.numIArgs > 0) {
          order.resize(slot.args.numIArgs);
          for (int i = 0; i < slot.args.numIArgs; i++) {
            order[i] = static_cast<int64_t>(slot.args.iArgs[i]);
          }
        } else {
          // No iArgs = simple transpose: reverse dimensions
          // Get rank from input source (prior slot output or external input)
          int rank = 0;
          if (slot.wiring.numInputs > 0) {
            int srcIdx = slot.wiring.inputSourceIndices[0];
            if (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots[srcIdx] != nullptr) {
              rank = outputSlots[srcIdx]->rankOf();
            } else if (srcIdx < 0) {
              int extIdx = -(srcIdx + 1);
              if (extIdx < numExternalInputs && externalInputs[extIdx] != nullptr) {
                rank = externalInputs[extIdx]->rankOf();
              }
            }
          }
          for (int d = rank - 1; d >= 0; d--) order.push_back(d);
        }
        if (!order.empty()) {
          dgOp.set_attr<std::vector<int64_t>>(dg::op::attr::order, order);
        }
      } else if (kind == dg::op::kind::ReduceSum || kind == dg::op::kind::ReduceMean ||
                 kind == dg::op::kind::ReduceMin || kind == dg::op::kind::ReduceMax ||
                 kind == dg::op::kind::ReduceProd) {
        // Reduction axes from iArgs
        if (slot.args.numIArgs > 0) {
          std::vector<int64_t> axes(slot.args.numIArgs);
          for (int i = 0; i < slot.args.numIArgs; i++) {
            axes[i] = static_cast<int64_t>(slot.args.iArgs[i]);
          }
          dgOp.set_attr<std::vector<int64_t>>(dg::op::attr::axes, axes);
        }
        dgOp.set_attr<bool>(dg::op::attr::keep_dims, false);
        // Check bArgs for keepDims
        if (slot.args.numBArgs > 0 && slot.args.bArgs[0]) {
          dgOp.set_attr<bool>(dg::op::attr::keep_dims, true);
        }
      }

      // Wire inputs
      std::vector<dg::logical_tensor> inputTensors;
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          // From a prior slot's output
          NDArray* arr = (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
          inputTensors.push_back(getSlotOutputTensor(srcIdx, arr));
        } else {
          // From external input
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs) {
            inputTensors.push_back(getExternalInputTensor(extIdx));
          }
        }
      }
      dgOp.add_inputs(inputTensors);

      // Wire outputs
      std::vector<dg::logical_tensor> outputTensors;
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int outSlotIdx = slot.wiring.outputSlotIndices[i];
        NDArray* arr = (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots)
                           ? outputSlots[outSlotIdx] : nullptr;
        outputTensors.push_back(getSlotOutputTensor(outSlotIdx, arr));
      }
      dgOp.add_outputs(outputTensors);

      try {
        g.add_op(dgOp);
        opsAdded++;
      } catch (const std::exception& e) {
        DSP_DIAG(COMPILE, "OneDNN Graph: add_op failed for slot %d op '%s': %s",
                 s, slot.ident.opName, e.what());
        return result;  // Return invalid result
      }

      // Record successful compilation in audit
      CompilationAuditEntry auditEntry;
      auditEntry.slotIndex = s;
      auditEntry.opName = slot.ident.opName;
      auditEntry.wasCompiled = true;
      result.compilationAudit.push_back(std::move(auditEntry));
    }

    if (opsAdded < 1) {
      // No ops mapped — nothing to compile
      return result;
    }

    g.finalize();

    auto partitions = g.get_partitions();
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: segment [%d-%d] finalized graph with %d ops → %d partitions",
              startSlot, endSlot, opsAdded, static_cast<int>(partitions.size()));
    if (partitions.empty()) {
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: no partitions found for segment [%d-%d]",
                startSlot, endSlot);
      return result;
    }

    // Compile each partition
    int supportedCount = 0, unsupportedCount = 0;
    for (auto& partition : partitions) {
      if (!partition.is_supported()) {
        unsupportedCount++;
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: partition %d unsupported (num_ops=%zu)",
                  supportedCount + unsupportedCount,
                  partition.get_ops_num());
        continue;
      }
      supportedCount++;

      auto inIds = partition.get_input_ports();
      auto outIds = partition.get_output_ports();

      std::vector<dg::logical_tensor> inputLTs, outputLTs;
      for (auto& lt : inIds) {
        auto it = logicalTensors.find(lt.get_id());
        if (it != logicalTensors.end()) {
          inputLTs.push_back(it->second);
        }
      }
      for (auto& lt : outIds) {
        auto it = logicalTensors.find(lt.get_id());
        if (it != logicalTensors.end()) {
          outputLTs.push_back(it->second);
        }
      }

      try {
        CompiledSegment::PartitionEntry entry;
        entry.compiledPartition = partition.compile(inputLTs, outputLTs, engine_);

        for (auto& lt : inputLTs) entry.inputTensorIds.push_back(lt.get_id());
        for (auto& lt : outputLTs) entry.outputTensorIds.push_back(lt.get_id());

        result.partitions.push_back(std::move(entry));
      } catch (const std::exception& e) {
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: partition compile failed: %s", e.what());
        return result;
      }
    }

    result.valid = !result.partitions.empty();
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: segment [%d-%d] partition summary: %d supported, %d unsupported, compiled=%d",
              startSlot, endSlot, supportedCount, unsupportedCount,
              static_cast<int>(result.partitions.size()));
    if (result.valid) {
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: compiled segment [%d-%d] into %d partitions (%d ops)",
                startSlot, endSlot,
                static_cast<int>(result.partitions.size()), opsAdded);
    }

  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: graph build failed: %s", e.what());
  }

  return result;
}

// ─── Compile segment ────────────────────────────────────────────────────────

bool OneDnnGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.valid) {
    DSP_DIAG(JIT, "OneDnnGraphBackend::compileSegment [%d-%d]: cache HIT (shapeKey=0x%llx)",
             seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);
    return true;  // Already compiled for this shape
  }

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: cache MISS, building graph (shapeKey=0x%llx)",
           seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);

  auto compiled = buildGraph(slots, seg.def.startSlot, seg.def.endSlot,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots);
  compiled.shapeKey = shapeKey;

  // Store compilation audit for validation
  lastCompilationAudit_ = compiled.compilationAudit;

  if (compiled.valid) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: SUCCESS partitions=%d",
             seg.def.startSlot, seg.def.endSlot, (int)compiled.partitions.size());
    cache_[key] = std::move(compiled);
    return true;
  }

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: FAILED", seg.def.startSlot, seg.def.endSlot);
  return false;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> OneDnnGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Execute segment ────────────────────────────────────────────────────────

Status OneDnnGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKey};

  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: no compiled segment found",
               seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: executing %d partitions",
           seg.def.startSlot, seg.def.endSlot, (int)compiled->partitions.size());

  auto& strm = getThreadStream();

  // Helper: resolve NDArray from slot index
  auto resolveArray = [&](int slotIdx) -> NDArray* {
    if (slotIdx < 0) {
      int extIdx = -(slotIdx + 1);
      return (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
    }
    return (slotIdx < totalOutputSlots) ? outputSlots[slotIdx] : nullptr;
  };

  // Execute each compiled partition
  for (auto& part : compiled->partitions) {
    // Initialize tensor cache on first execution
    if (part.cachedInputTensors.empty()) {
      part.cachedInputTensors.resize(part.inputTensorIds.size());
      for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
        size_t tid = part.inputTensorIds[i];
        auto slotIt = compiled->tensorIdToSlotMap.find(tid);
        if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        int rank = arr->rankOf();
        std::vector<int64_t> shape(rank), strides(rank);
        for (int d = 0; d < rank; d++) {
          shape[d] = arr->sizeAt(d);
          strides[d] = arr->strideAt(d);
        }
        part.cachedInputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
        part.cachedInputTensors[i].lastBuffer = arr->buffer();
      }
    }
    if (part.cachedOutputTensors.empty()) {
      part.cachedOutputTensors.resize(part.outputTensorIds.size());
      for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
        size_t tid = part.outputTensorIds[i];
        auto slotIt = compiled->tensorIdToSlotMap.find(tid);
        if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        int rank = arr->rankOf();
        std::vector<int64_t> shape(rank), strides(rank);
        for (int d = 0; d < rank; d++) {
          shape[d] = arr->sizeAt(d);
          strides[d] = arr->strideAt(d);
        }
        part.cachedOutputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
        part.cachedOutputTensors[i].lastBuffer = arr->buffer();
      }
    }

    // Build input tensors — reuse cached logical_tensors, only update buffer if changed
    std::vector<dg::tensor> inputTensors;
    inputTensors.reserve(part.inputTensorIds.size());
    for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
      auto slotIt = compiled->tensorIdToSlotMap.find(part.inputTensorIds[i]);
      NDArray* arr = resolveArray(slotIt->second);
      if (!arr) return Status::KERNEL_FAILURE;
      // Logical tensor is stable (same shape key = same shapes) — just wrap current buffer
      inputTensors.emplace_back(part.cachedInputTensors[i].lt, engine_, arr->buffer());
    }

    // Build output tensors — same optimization
    std::vector<dg::tensor> outputTensors;
    outputTensors.reserve(part.outputTensorIds.size());
    for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
      auto slotIt = compiled->tensorIdToSlotMap.find(part.outputTensorIds[i]);
      NDArray* arr = resolveArray(slotIt->second);
      if (!arr) return Status::KERNEL_FAILURE;
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

// ─── Cache invalidation ─────────────────────────────────────────────────────

void OneDnnGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
