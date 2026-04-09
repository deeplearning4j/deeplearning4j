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

#if HAVE_ARMCOMPUTE

#include <graph/cpu/AclGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <ops/declarable/platform/armcompute/ArmComputeVersionProvider.h>
#include <ops/declarable/platform/armcompute/armcomputeUtils.h>

namespace sd {
namespace graph {

// ─── Singleton ──────────────────────────────────────────────────────────────

AclGraphBackend& AclGraphBackend::getInstance() {
  static AclGraphBackend instance;
  return instance;
}

AclGraphBackend::AclGraphBackend() = default;
AclGraphBackend::~AclGraphBackend() = default;

// ─── Availability ───────────────────────────────────────────────────────────

bool AclGraphBackend::isAvailable() const {
  return sd::ops::platforms::ArmComputeVersionProvider::isAvailable();
}

// ─── Data type mapping ──────────────────────────────────────────────────────

arm_compute::DataType AclGraphBackend::mapDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return arm_compute::DataType::F32;
    case DataType::HALF: return arm_compute::DataType::F16;
    case DataType::BFLOAT16: return arm_compute::DataType::BFLOAT16;
    case DataType::INT32: return arm_compute::DataType::S32;
    case DataType::INT64: return arm_compute::DataType::S64;
    case DataType::INT8: return arm_compute::DataType::S8;
    case DataType::UINT8: return arm_compute::DataType::U8;
    case DataType::BOOL: return arm_compute::DataType::U8;
    default: return arm_compute::DataType::F32;
  }
}

// ─── TensorInfo from NDArray ────────────────────────────────────────────────

arm_compute::TensorInfo AclGraphBackend::getTensorInfo(NDArray* arr) {
  int rank = arr->rankOf();
  arm_compute::TensorShape shape;

  // ACL uses reversed dimension ordering (innermost first)
  for (int d = rank - 1; d >= 0; d--) {
    shape.set(rank - 1 - d, arr->sizeAt(d));
  }

  return arm_compute::TensorInfo(shape, 1, mapDataType(arr->dataType()));
}

// ─── Activation mapping ─────────────────────────────────────────────────────

arm_compute::ActivationLayerInfo::ActivationFunction AclGraphBackend::mapActivation(
    const std::string& opName) {
  if (opName == "relu" || opName == "Relu")
    return arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
  if (opName == "sigmoid" || opName == "Sigmoid")
    return arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
  if (opName == "tanh" || opName == "Tanh")
    return arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
  if (opName == "elu" || opName == "Elu")
    return arm_compute::ActivationLayerInfo::ActivationFunction::ELU;
  if (opName == "hardswish" || opName == "HardSwish")
    return arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH;
  if (opName == "abs" || opName == "Abs")
    return arm_compute::ActivationLayerInfo::ActivationFunction::ABS;
  if (opName == "sqrt" || opName == "Sqrt")
    return arm_compute::ActivationLayerInfo::ActivationFunction::SQRT;
  if (opName == "square" || opName == "Square")
    return arm_compute::ActivationLayerInfo::ActivationFunction::SQUARE;

  return arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY;
}

// ─── Fusion detection ───────────────────────────────────────────────────────

bool AclGraphBackend::canFuseActivation(NativeSlot* slots, int idx, int endSlot) {
  if (idx + 1 > endSlot) return false;

  auto actFn = mapActivation(slots[idx + 1].ident.opName);
  if (actFn == arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY) return false;

  // The activation must take the current op's output as its only input
  if (slots[idx + 1].wiring.numInputs != 1) return false;
  if (slots[idx + 1].wiring.numOutputs != 1) return false;

  // Input to activation must be this slot's output
  if (slots[idx].wiring.numOutputs < 1) return false;
  int outSlotIdx = slots[idx].wiring.outputSlotIndices[0];
  int actInputIdx = slots[idx + 1].wiring.inputSourceIndices[0];
  return (actInputIdx == outSlotIdx);
}

// ─── Segment fusibility check ───────────────────────────────────────────────

bool AclGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  // Check if we have at least one ACL-accelerable pattern
  int aclOps = 0;
  for (int i = start; i <= end; i++) {
    const auto& name = slots[i].ident.opName;
    // ACL supports these as individual functions
    if (name == "matmul" || name == "mmul" || name == "MatMul" ||
        name == "add" || name == "Add" ||
        name == "subtract" || name == "Sub" ||
        name == "multiply" || name == "Mul" ||
        name == "relu" || name == "Relu" ||
        name == "sigmoid" || name == "Sigmoid" ||
        name == "tanh" || name == "Tanh" ||
        name == "softmax" || name == "Softmax" ||
        name == "batchnorm" || name == "BatchNorm" ||
        name == "conv2d" || name == "Conv2D") {
      aclOps++;
    }
  }

  // Need at least 2 ACL-able ops with at least one fusible pattern
  return aclOps >= 2;
}

// ─── Build ACL functions ────────────────────────────────────────────────────

AclGraphBackend::AclFunctionGroup AclGraphBackend::buildFunctions(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  AclFunctionGroup result;
  result.valid = false;

  try {
    // Helper to get or create an ACL Tensor for a slot
    auto getOrCreateTensor = [&](int slotIdx, NDArray* arr) -> std::shared_ptr<arm_compute::Tensor> {
      if (slotIdx >= 0) {
        auto it = result.slotToTensor.find(slotIdx);
        if (it != result.slotToTensor.end()) return it->second;
      }

      auto tensor = std::make_shared<arm_compute::Tensor>();
      if (arr != nullptr) {
        auto info = getTensorInfo(arr);
        tensor->allocator()->init(info);
      }

      if (slotIdx >= 0) {
        result.slotToTensor[slotIdx] = tensor;
      }
      return tensor;
    };

    auto getExternalTensor = [&](int extIdx) -> std::shared_ptr<arm_compute::Tensor> {
      auto it = result.extToTensor.find(extIdx);
      if (it != result.extToTensor.end()) return it->second;

      NDArray* arr = (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
      auto tensor = std::make_shared<arm_compute::Tensor>();
      if (arr != nullptr) {
        auto info = getTensorInfo(arr);
        tensor->allocator()->init(info);
      }
      result.extToTensor[extIdx] = tensor;
      return tensor;
    };

    int functionsBuilt = 0;

    for (int s = startSlot; s <= endSlot; s++) {
      NativeSlot& slot = slots[s];

      // Get input arrays
      std::vector<NDArray*> inputArrays(slot.wiring.numInputs);
      std::vector<std::shared_ptr<arm_compute::Tensor>> inputTensors(slot.wiring.numInputs);
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          inputArrays[i] = (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
          inputTensors[i] = getOrCreateTensor(srcIdx, inputArrays[i]);
        } else {
          int extIdx = -(srcIdx + 1);
          inputArrays[i] = (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
          inputTensors[i] = getExternalTensor(extIdx);
        }
      }

      // Get output array
      int outSlotIdx = (slot.wiring.numOutputs > 0) ? slot.wiring.outputSlotIndices[0] : -1;
      NDArray* outArr = (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots)
                            ? outputSlots[outSlotIdx] : nullptr;
      auto outTensor = getOrCreateTensor(outSlotIdx, outArr);

      // Check for fusible activation pattern
      arm_compute::ActivationLayerInfo actInfo;
      if (canFuseActivation(slots, s, endSlot)) {
        auto actFn = mapActivation(slots[s + 1].ident.opName);
        float alpha = 0.0f, beta = 0.0f;
        if (slots[s + 1].args.numTArgs > 0) alpha = static_cast<float>(slots[s + 1].args.tArgs[0]);
        if (slots[s + 1].args.numTArgs > 1) beta = static_cast<float>(slots[s + 1].args.tArgs[1]);
        actInfo = arm_compute::ActivationLayerInfo(actFn, alpha, beta);

        // The output goes to the activation's output slot instead
        int actOutSlot = slots[s + 1].wiring.outputSlotIndices[0];
        NDArray* actOutArr = (actOutSlot >= 0 && actOutSlot < totalOutputSlots)
                                 ? outputSlots[actOutSlot] : nullptr;
        outTensor = getOrCreateTensor(actOutSlot, actOutArr);
      }

      FunctionEntry entry;
      bool built = false;

      if (slot.ident.opName == "matmul" || slot.ident.opName == "mmul" || slot.ident.opName == "MatMul") {
        if (slot.wiring.numInputs >= 2 && inputArrays[0] != nullptr && inputArrays[1] != nullptr) {
          auto* gemm = new arm_compute::NEGEMM();
          float alpha = 1.0f, beta = 0.0f;
          gemm->configure(inputTensors[0].get(), inputTensors[1].get(), nullptr,
                          outTensor.get(), alpha, beta, arm_compute::GEMMInfo());
          entry.function.reset(gemm);
          entry.tensors = {inputTensors[0], inputTensors[1], outTensor};
          built = true;
        }
      } else if (slot.ident.opName == "add" || slot.ident.opName == "Add") {
        if (slot.wiring.numInputs >= 2) {
          auto* addLayer = new arm_compute::NEArithmeticAddition();
          addLayer->configure(inputTensors[0].get(), inputTensors[1].get(),
                              outTensor.get(), arm_compute::ConvertPolicy::SATURATE);
          entry.function.reset(addLayer);
          entry.tensors = {inputTensors[0], inputTensors[1], outTensor};
          built = true;
        }
      } else if (slot.ident.opName == "multiply" || slot.ident.opName == "Mul") {
        if (slot.wiring.numInputs >= 2) {
          auto* mulLayer = new arm_compute::NEPixelWiseMultiplication();
          mulLayer->configure(inputTensors[0].get(), inputTensors[1].get(),
                              outTensor.get(), 1.0f, arm_compute::ConvertPolicy::SATURATE,
                              arm_compute::RoundingPolicy::TO_ZERO);
          entry.function.reset(mulLayer);
          entry.tensors = {inputTensors[0], inputTensors[1], outTensor};
          built = true;
        }
      } else if (slot.ident.opName == "softmax" || slot.ident.opName == "Softmax") {
        if (slot.wiring.numInputs >= 1) {
          auto* smLayer = new arm_compute::NESoftmaxLayer();
          float beta = 1.0f;
          smLayer->configure(inputTensors[0].get(), outTensor.get(), beta);
          entry.function.reset(smLayer);
          entry.tensors = {inputTensors[0], outTensor};
          built = true;
        }
      } else {
        // Check if it's a pure activation
        auto actFn = mapActivation(slot.ident.opName);
        if (actFn != arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY &&
            slot.wiring.numInputs >= 1) {
          float alpha = 0.0f, beta = 0.0f;
          if (slot.args.numTArgs > 0) alpha = static_cast<float>(slot.args.tArgs[0]);
          if (slot.args.numTArgs > 1) beta = static_cast<float>(slot.args.tArgs[1]);
          auto* actLayer = new arm_compute::NEActivationLayer();
          actLayer->configure(inputTensors[0].get(), outTensor.get(),
                              arm_compute::ActivationLayerInfo(actFn, alpha, beta));
          entry.function.reset(actLayer);
          entry.tensors = {inputTensors[0], outTensor};
          built = true;
        }
      }

      if (built) {
        result.functions.push_back(std::move(entry));
        functionsBuilt++;

        // Record successful compilation in audit
        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = true;
        result.compilationAudit.push_back(std::move(auditEntry));

        // Skip the fused activation if we detected one
        if (actInfo.enabled()) {
          // Record fused activation in audit
          CompilationAuditEntry fusedEntry;
          fusedEntry.slotIndex = s + 1;
          fusedEntry.opName = slots[s + 1].ident.opName;
          fusedEntry.wasCompiled = true;
          fusedEntry.reason = "fused with previous op";
          result.compilationAudit.push_back(std::move(fusedEntry));
          s++;  // Skip next slot (the activation)
        }
      } else {
        // Record skipped op in audit
        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = false;
        auditEntry.reason = "unsupported op";
        result.compilationAudit.push_back(std::move(auditEntry));
      }
    }

    result.valid = (functionsBuilt >= 2);
    if (result.valid) {
      DSP_DIAG(COMPILE, "AclGraphBackend: built %d functions for segment [%d-%d]",
                functionsBuilt, startSlot, endSlot);
    }

  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "AclGraphBackend: build failed: %s", e.what());
  }

  return result;
}

// ─── Compile segment ────────────────────────────────────────────────────────

bool AclGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.valid) {
    return true;
  }

  auto compiled = buildFunctions(slots, seg.startSlot, seg.endSlot,
                                 externalInputs, numExternalInputs,
                                 outputSlots, totalOutputSlots);
  compiled.shapeKey = shapeKey;

  // Store compilation audit for validation
  lastCompilationAudit_ = compiled.compilationAudit;

  if (compiled.valid) {
    cache_[key] = std::move(compiled);
    return true;
  }

  return false;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> AclGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Execute segment ────────────────────────────────────────────────────────

Status AclGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  AclFunctionGroup* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  // Import buffers from NDArrays into ACL tensors
  for (auto& [slotIdx, tensor] : compiled->slotToTensor) {
    if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx] != nullptr) {
      NDArray* arr = outputSlots[slotIdx];
      if (!arr->hasPaddedBuffer() && !tensor->info()->has_padding()) {
        tensor->allocator()->import_memory(arr->buffer());
      } else {
        // Need to allocate and copy
        tensor->allocator()->allocate();
        std::memcpy(tensor->buffer(), arr->buffer(), arr->lengthOf() * arr->sizeOfT());
      }
    }
  }

  for (auto& [extIdx, tensor] : compiled->extToTensor) {
    if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs[extIdx] != nullptr) {
      NDArray* arr = externalInputs[extIdx];
      if (!arr->hasPaddedBuffer() && !tensor->info()->has_padding()) {
        tensor->allocator()->import_memory(arr->buffer());
      } else {
        tensor->allocator()->allocate();
        std::memcpy(tensor->buffer(), arr->buffer(), arr->lengthOf() * arr->sizeOfT());
      }
    }
  }

  // Execute all functions in order
  for (auto& entry : compiled->functions) {
    entry.function->run();
  }

  // Copy results back if needed (when import_memory wasn't used)
  for (auto& [slotIdx, tensor] : compiled->slotToTensor) {
    if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx] != nullptr) {
      NDArray* arr = outputSlots[slotIdx];
      if (arr->hasPaddedBuffer() || tensor->info()->has_padding()) {
        std::memcpy(arr->buffer(), tensor->buffer(), arr->lengthOf() * arr->sizeOfT());
      }
    }
  }

  return Status::OK;
}

// ─── Cache invalidation ─────────────────────────────────────────────────────

void AclGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ARMCOMPUTE
