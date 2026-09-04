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
#include <helpers/shape.h>
#include <ops/declarable/platform/armcompute/ArmComputeVersionProvider.h>
#include <ops/declarable/platform/armcompute/armcomputeUtils.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

namespace sd {
namespace graph {

namespace {

bool isDenseCOrder(NDArray* array) {
  return array != nullptr && array->ordering() == 'c' &&
         shape::strideDescendingCAscendingF(array->shapeInfo());
}

bool isGatherName(const std::string& name) {
  return name == "gather" || name == "Gather";
}

bool isSubtractName(const std::string& name) {
  return name == "subtract" || name == "Subtract" || name == "sub" ||
         name == "Sub";
}

class StagedAclGatherFunction final : public arm_compute::IFunction {
 public:
  StagedAclGatherFunction(int sourceIndex, DataType sourceDataType,
                          std::vector<LongType> sourceDimensions,
                          LongType vocabularySize,
                          std::shared_ptr<arm_compute::Tensor> indicesTensor)
      : sourceIndex_(sourceIndex),
        sourceDataType_(sourceDataType),
        sourceDimensions_(std::move(sourceDimensions)),
        vocabularySize_(vocabularySize),
        indicesTensor_(std::move(indicesTensor)) {}

  void configure(const arm_compute::ITensor* input, arm_compute::ITensor* output,
                 int axis) {
    gather_.configure(input, indicesTensor_.get(), output, axis);
    indicesTensor_->allocator()->allocate();
  }

  int sourceIndex() const { return sourceIndex_; }

  bool stageIndices(NDArray* indices) {
    if (indices == nullptr || indices->dataType() != sourceDataType_ ||
        static_cast<size_t>(indices->rankOf()) != sourceDimensions_.size() ||
        indices->dataBuffer() == nullptr || !indices->dataBuffer()->isValid()) {
      return false;
    }
    for (int dimension = 0; dimension < indices->rankOf(); ++dimension) {
      if (indices->sizeAt(dimension) !=
          sourceDimensions_[static_cast<size_t>(dimension)]) {
        return false;
      }
    }

    indices->syncToHost();
    auto* destination = reinterpret_cast<int32_t*>(indicesTensor_->buffer());
    if (destination == nullptr) return false;
    for (LongType element = 0; element < indices->lengthOf(); ++element) {
      const LongType value = indices->e<LongType>(element);
      if (value < 0 || value >= vocabularySize_ ||
          value > static_cast<LongType>(std::numeric_limits<int32_t>::max())) {
        DSP_DIAG(EXECUTE,
                 "ACL_GATHER_INDEX_RANGE source_slot=%d element=%lld value=%lld "
                 "vocabulary=%lld",
                 sourceIndex_, static_cast<long long>(element),
                 static_cast<long long>(value),
                 static_cast<long long>(vocabularySize_));
        return false;
      }
      destination[element] = static_cast<int32_t>(value);
    }
    return true;
  }

  void run() override { gather_.run(); }

 private:
  int sourceIndex_;
  DataType sourceDataType_;
  std::vector<LongType> sourceDimensions_;
  LongType vocabularySize_;
  std::shared_ptr<arm_compute::Tensor> indicesTensor_;
  arm_compute::NEGather gather_;
};

class CompiledAclInt64SubtractFunction final : public arm_compute::IFunction {
 public:
  CompiledAclInt64SubtractFunction(
      int leftSource, int rightSource, int outputSlot,
      std::shared_ptr<arm_compute::Tensor> leftTensor,
      std::shared_ptr<arm_compute::Tensor> rightTensor,
      std::shared_ptr<arm_compute::Tensor> outputTensor)
      : leftSource_(leftSource),
        rightSource_(rightSource),
        outputSlot_(outputSlot),
        leftTensor_(std::move(leftTensor)),
        rightTensor_(std::move(rightTensor)),
        outputTensor_(std::move(outputTensor)) {}

  int leftSource() const { return leftSource_; }
  int rightSource() const { return rightSource_; }
  int outputSlot() const { return outputSlot_; }

  bool validateBindings(NDArray* left, NDArray* right, NDArray* output) const {
    auto validScalar = [](NDArray* array) {
      return array != nullptr && array->dataType() == DataType::INT64 &&
             array->lengthOf() == 1 && isDenseCOrder(array) &&
             array->dataBuffer() != nullptr && array->dataBuffer()->isValid();
    };
    if (!validScalar(left) || !validScalar(right) || !validScalar(output)) {
      return false;
    }
    left->syncToHost();
    right->syncToHost();
    const LongType leftValue = left->e<LongType>(0);
    const LongType rightValue = right->e<LongType>(0);
    return rightValue == 1 && leftTensor_->buffer() != nullptr &&
           rightTensor_->buffer() != nullptr && outputTensor_->buffer() != nullptr &&
           !((rightValue > 0 &&
              leftValue < std::numeric_limits<LongType>::min() + rightValue) ||
             (rightValue < 0 &&
              leftValue > std::numeric_limits<LongType>::max() + rightValue));
  }

  void run() override {
    const auto* left = reinterpret_cast<const LongType*>(leftTensor_->buffer());
    const auto* right = reinterpret_cast<const LongType*>(rightTensor_->buffer());
    auto* output = reinterpret_cast<LongType*>(outputTensor_->buffer());
    output[0] = left[0] - right[0];
  }

 private:
  int leftSource_;
  int rightSource_;
  int outputSlot_;
  std::shared_ptr<arm_compute::Tensor> leftTensor_;
  std::shared_ptr<arm_compute::Tensor> rightTensor_;
  std::shared_ptr<arm_compute::Tensor> outputTensor_;
};

}  // namespace

// ─── Singleton ──────────────────────────────────────────────────────────────

AclGraphBackend& AclGraphBackend::getInstance() {
  static AclGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new AclGraphBackend();
  });
  return *instance;
}

AclGraphBackend::AclGraphBackend() = default;
AclGraphBackend::~AclGraphBackend() { invalidateCache(); }

// ─── Availability ───────────────────────────────────────────────────────────

bool AclGraphBackend::isAvailable() const {
  return sd::ops::platforms::armcompute::ArmComputeVersionProvider::isAvailable();
}

bool AclGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_ARM_HYBRID ||
         request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY;
}

int AclGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_ARM_HYBRID ? 900 : 400;
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
  if (rank == 0) {
    // ACL has no empty-rank physical tensor descriptor. Preserve ND4J scalar
    // semantics while allocating/importing one physical element.
    shape.set(0, 1);
  } else {
    for (int d = rank - 1; d >= 0; d--) {
      shape.set(rank - 1 - d, arr->sizeAt(d));
    }
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

// ─── Segment fusibility check ───────────────────────────────────────────────

bool AclGraphBackend::isSupportedSlotContract(const NativeSlot& slot) {
  const auto& name = slot.ident.opName;
  const bool gather = isGatherName(name);
  const bool subtract = isSubtractName(name);
  const bool binary =
      name == "matmul" || name == "mmul" || name == "MatMul" ||
      name == "add" || name == "Add" || subtract ||
      name == "multiply" || name == "Mul";
  const bool unary = name == "softmax" || name == "Softmax" ||
                     mapActivation(name) !=
                         arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY;
  if (!gather && !binary && !unary) return false;

  const int requiredInputs = (binary || gather) ? 2 : 1;
  if (slot.wiring.numInputs != requiredInputs || slot.wiring.numOutputs != 1 ||
      slot.wiring.inputSourceIndices == nullptr ||
      slot.wiring.outputSlotIndices == nullptr) {
    return false;
  }
  if (subtract) {
    return slot.args.numIArgs == 0 && slot.args.numTArgs == 0 &&
           slot.args.numBArgs == 0 && slot.args.numDArgs == 0 &&
           slot.args.numSArgs == 0;
  }
  if (!gather) return true;

  return slot.args.numIArgs == 1 && slot.args.iArgs != nullptr &&
         slot.args.iArgs[0] == 0 && slot.args.numTArgs == 0 &&
         slot.args.numBArgs == 0 && slot.args.numDArgs == 0 &&
         slot.args.numSArgs == 0;
}

bool AclGraphBackend::canResolveSlot(const GraphBackendRequest& request,
                                     NativeSlot* slots, int slotIndex) {
  return isResolvable(request) && slots != nullptr && slotIndex >= 0 &&
         (isGatherName(slots[slotIndex].ident.opName) ||
          isSubtractName(slots[slotIndex].ident.opName)) &&
         isSupportedSlotContract(slots[slotIndex]);
}

bool AclGraphBackend::canResolveSegment(const GraphBackendRequest& request,
                                        NativeSlot* slots, int start, int end) {
  if (isResolvable(request) && slots != nullptr && start == end && start >= 0 &&
      (isGatherName(slots[start].ident.opName) ||
       isSubtractName(slots[start].ident.opName))) {
    return isSupportedSlotContract(slots[start]);
  }
  return canFuseSegment(slots, start, end);
}

bool AclGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable() || !slots || end < start) return false;

  // ACL buildFunctions is currently all-or-nothing at segment scope. Do
  // not claim a mixed segment merely because it contains two supported ops:
  // buildFunctions would leave unsupported slots uncovered and the audit layer
  // would throw instead of allowing the next backend/fallback to run.
  int supportedOps = 0;
  for (int i = start; i <= end; i++) {
    const auto& name = slots[i].ident.opName;
    if (!isSupportedSlotContract(slots[i])) {
      DSP_DIAG(BACKEND,
               "ACL admission rejected mixed seg[%d-%d]: unsupported op %s at slot %d",
               start, end, name.c_str(), i);
      return false;
    }
    supportedOps++;
  }

  return supportedOps >= 2;
}

// ─── Build ACL functions ────────────────────────────────────────────────────

std::shared_ptr<AclGraphBackend::AclFunctionGroup> AclGraphBackend::buildFunctions(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  auto result = std::make_shared<AclFunctionGroup>();
  result->startSlot = startSlot;
  result->endSlot = endSlot;

  std::unordered_set<int> segmentOutputIndices;
  for (int slotIndex = startSlot; slotIndex <= endSlot; ++slotIndex) {
    for (int output = 0; output < slots[slotIndex].wiring.numOutputs; ++output) {
      segmentOutputIndices.insert(
          slots[slotIndex].wiring.outputSlotIndices[output]);
    }
  }

  try {
    // Helper to get or create an ACL Tensor for a slot
    auto getOrCreateTensor = [&](int slotIdx, NDArray* arr) -> std::shared_ptr<arm_compute::Tensor> {
      if (slotIdx >= 0) {
        auto it = result->slotToTensor.find(slotIdx);
        if (it != result->slotToTensor.end()) return it->second;
      }

      auto tensor = std::make_shared<arm_compute::Tensor>();
      if (arr != nullptr) {
        auto info = getTensorInfo(arr);
        tensor->allocator()->init(info);
      }

      if (slotIdx >= 0) {
        result->slotToTensor[slotIdx] = tensor;
      }
      return tensor;
    };

    auto getExternalTensor = [&](int extIdx) -> std::shared_ptr<arm_compute::Tensor> {
      auto it = result->extToTensor.find(extIdx);
      if (it != result->extToTensor.end()) return it->second;

      NDArray* arr = (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
      auto tensor = std::make_shared<arm_compute::Tensor>();
      if (arr != nullptr) {
        auto info = getTensorInfo(arr);
        tensor->allocator()->init(info);
      }
      result->extToTensor[extIdx] = tensor;
      return tensor;
    };

    int functionsBuilt = 0;

    for (int s = startSlot; s <= endSlot; s++) {
      NativeSlot& slot = slots[s];
      const bool gatherSlot = isGatherName(slot.ident.opName);

      // Get input arrays
      std::vector<NDArray*> inputArrays(slot.wiring.numInputs);
      std::vector<std::shared_ptr<arm_compute::Tensor>> inputTensors(slot.wiring.numInputs);
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          inputArrays[i] = (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
          if (!(gatherSlot && i == 1)) {
            inputTensors[i] = getOrCreateTensor(srcIdx, inputArrays[i]);
          }
        } else {
          int extIdx = -(srcIdx + 1);
          inputArrays[i] = (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
          if (!(gatherSlot && i == 1)) {
            inputTensors[i] = getExternalTensor(extIdx);
          }
        }
      }

      // Get output array
      int outSlotIdx = (slot.wiring.numOutputs > 0) ? slot.wiring.outputSlotIndices[0] : -1;
      NDArray* outArr = (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots)
                            ? outputSlots[outSlotIdx] : nullptr;
      auto outTensor = getOrCreateTensor(outSlotIdx, outArr);
      if (outSlotIdx >= 0) result->producedSlots.insert(outSlotIdx);

      AclFunctionGroup::FunctionEntry entry;
      bool built = false;

      if (gatherSlot) {
        NDArray* table = inputArrays[0];
        NDArray* indices = inputArrays[1];
        const bool supportedTableType =
            table != nullptr &&
            (table->dataType() == DataType::FLOAT32 ||
             table->dataType() == DataType::HALF ||
             table->dataType() == DataType::BFLOAT16);
        bool shapeContract =
            supportedTableType && indices != nullptr && outArr != nullptr &&
            table->rankOf() == 2 &&
            (indices->rankOf() == 1 || indices->rankOf() == 2) &&
            (indices->dataType() == DataType::INT32 ||
             indices->dataType() == DataType::INT64) &&
            outArr->dataType() == table->dataType() &&
            outArr->rankOf() == indices->rankOf() + 1 &&
            isDenseCOrder(table) && isDenseCOrder(outArr);
        if (shapeContract) {
          for (int dimension = 0; dimension < indices->rankOf(); ++dimension) {
            shapeContract &= outArr->sizeAt(dimension) == indices->sizeAt(dimension);
          }
          shapeContract &=
              outArr->sizeAt(outArr->rankOf() - 1) == table->sizeAt(1);
        }

        const int indexSource = slot.wiring.inputSourceIndices[1];
        if (shapeContract &&
            (indexSource < 0 || segmentOutputIndices.count(indexSource) == 0)) {
          arm_compute::TensorShape indexShape;
          indexShape.set_num_dimensions(indices->rankOf());
          std::vector<LongType> indexDimensions(
              static_cast<size_t>(indices->rankOf()));
          for (int dimension = 0; dimension < indices->rankOf(); ++dimension) {
            indexDimensions[static_cast<size_t>(dimension)] =
                indices->sizeAt(dimension);
            indexShape[indices->rankOf() - 1 - dimension] =
                indices->sizeAt(dimension);
          }
          auto indexTensor = std::make_shared<arm_compute::Tensor>();
          arm_compute::TensorInfo indexInfo(
              indexShape, 1, arm_compute::DataType::S32);
          indexTensor->allocator()->init(indexInfo);

          const int armAxis = table->rankOf() - 1;
          const auto validation = arm_compute::NEGather::validate(
              inputTensors[0]->info(), indexTensor->info(), outTensor->info(),
              armAxis);
          if (validation) {
            auto* gather = new StagedAclGatherFunction(
                indexSource, indices->dataType(), std::move(indexDimensions),
                table->sizeAt(0), indexTensor);
            gather->configure(inputTensors[0].get(), outTensor.get(), armAxis);
            entry.function.reset(gather);
            entry.tensors = {inputTensors[0], indexTensor, outTensor};
            built = true;
            DSP_DIAG(COMPILE,
                     "ACL_GATHER_LOWERING slot=%d axis=0 table=[%lld,%lld] "
                     "lookups_rank=%d lookups=%lld index_binding=S32",
                     s, static_cast<long long>(table->sizeAt(0)),
                     static_cast<long long>(table->sizeAt(1)),
                     indices->rankOf(),
                     static_cast<long long>(indices->lengthOf()));
          } else {
            DSP_DIAG(COMPILE,
                     "ACL gather validation rejected slot %d: %s", s,
                     validation.error_description().c_str());
          }
        } else {
          DSP_DIAG(COMPILE,
                   "ACL gather contract rejected slot %d: table=%p indices=%p "
                   "output=%p internal_indices=%d",
                   s, static_cast<void*>(table), static_cast<void*>(indices),
                   static_cast<void*>(outArr),
                   indexSource >= 0 && segmentOutputIndices.count(indexSource) != 0);
        }
      } else if (isSubtractName(slot.ident.opName)) {
        NDArray* left = inputArrays[0];
        NDArray* right = inputArrays[1];
        bool shapeContract =
            left != nullptr && right != nullptr && outArr != nullptr &&
            left->dataType() == DataType::INT64 &&
            right->dataType() == DataType::INT64 &&
            outArr->dataType() == DataType::INT64 && left->lengthOf() == 1 &&
            right->lengthOf() == 1 && outArr->lengthOf() == 1 &&
            left->rankOf() == right->rankOf() &&
            left->rankOf() == outArr->rankOf() && isDenseCOrder(left) &&
            isDenseCOrder(right) && isDenseCOrder(outArr);
        if (shapeContract) {
          for (int dimension = 0; dimension < left->rankOf(); ++dimension) {
            shapeContract &= left->sizeAt(dimension) == right->sizeAt(dimension) &&
                             left->sizeAt(dimension) == outArr->sizeAt(dimension);
          }
        }
        if (shapeContract) {
          right->syncToHost();
          shapeContract = right->e<LongType>(0) == 1;
        }

        if (shapeContract) {
          auto* subtract = new CompiledAclInt64SubtractFunction(
              slot.wiring.inputSourceIndices[0],
              slot.wiring.inputSourceIndices[1], outSlotIdx, inputTensors[0],
              inputTensors[1], outTensor);
          entry.function.reset(subtract);
          entry.tensors = {inputTensors[0], inputTensors[1], outTensor};
          built = true;
          DSP_DIAG(COMPILE,
                   "ACL_INT64_SUBTRACT_LOWERING slot=%d elements=1 rhs=1",
                   s);
        } else {
          DSP_DIAG(COMPILE,
                   "ACL INT64 subtract contract rejected slot %d: left=%p "
                   "right=%p output=%p",
                   s, static_cast<void*>(left), static_cast<void*>(right),
                   static_cast<void*>(outArr));
        }
      } else if (slot.ident.opName == "matmul" || slot.ident.opName == "mmul" || slot.ident.opName == "MatMul") {
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
        result->functions.push_back(std::move(entry));
        functionsBuilt++;

        // Record successful compilation in audit
        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = true;
        result->compilationAudit.push_back(std::move(auditEntry));
      } else {
        // Record skipped op in audit
        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = false;
        auditEntry.reason = "unsupported op";
        result->compilationAudit.push_back(std::move(auditEntry));
      }
    }

    const int expectedFunctions = endSlot - startSlot + 1;
    const bool completeAudit =
        static_cast<int>(result->compilationAudit.size()) == expectedFunctions &&
        std::all_of(result->compilationAudit.begin(),
                    result->compilationAudit.end(),
                    [](const CompilationAuditEntry& entry) {
                      return entry.wasCompiled || entry.isNativeHandled;
                    });
    result->valid = functionsBuilt == expectedFunctions && completeAudit;
    if (result->valid) {
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

  // Serialize build/publication with global invalidation. The same lock order
  // (registry, then group) is used by invalidateCache().
  std::lock_guard<std::mutex> registryLock(cacheMtx_);

  if (seg.compiledGraphBackendArtifactOwner == this &&
      seg.compiledGraphBackendArtifactShapeKey == shapeKey &&
      seg.compiledGraphBackendArtifact) {
    auto existing = std::static_pointer_cast<AclFunctionGroup>(
        seg.compiledGraphBackendArtifact);
    std::lock_guard<std::mutex> executionLock(existing->executionMtx);
    if (existing->valid && existing->startSlot == seg.def.startSlot &&
        existing->endSlot == seg.def.endSlot) {
      lastCompilationAudit_ = existing->compilationAudit;
      return true;
    }
  }

  auto compiled = buildFunctions(slots, seg.def.startSlot, seg.def.endSlot,
                                 externalInputs, numExternalInputs,
                                 outputSlots, totalOutputSlots);
  compiled->shapeKey = shapeKey;

  // Store compilation audit for validation
  lastCompilationAudit_ = compiled->compilationAudit;

  if (compiled->valid) {
    compiledArtifacts_.erase(
        std::remove_if(compiledArtifacts_.begin(), compiledArtifacts_.end(),
                       [](const std::weak_ptr<AclFunctionGroup>& artifact) {
                         return artifact.expired();
                       }),
        compiledArtifacts_.end());
    compiledArtifacts_.push_back(compiled);
    seg.compilationAudit = compiled->compilationAudit;
    seg.setCompiledGraphBackendArtifact(this, shapeKey, compiled);
    return true;
  }

  return false;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> AclGraphBackend::getLastCompilationAudit() const {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  return lastCompilationAudit_;
}

// ─── Execute segment ────────────────────────────────────────────────────────

Status AclGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  auto fail = [&](const std::string& reason) {
    const std::string message =
        reason + " [ACL segment " + std::to_string(seg.def.startSlot) + "-" +
        std::to_string(seg.def.endSlot) + ", status=KERNEL_FAILURE (50)]";
    safeSetErrorContext(static_cast<int>(Status::KERNEL_FAILURE), message.c_str());
    return Status::KERNEL_FAILURE;
  };

  if (seg.compiledGraphBackendArtifactOwner != this ||
      !seg.compiledGraphBackendArtifact) {
    return fail("ACL execution has no segment-owned compiled artifact");
  }
  auto compiled = std::static_pointer_cast<AclFunctionGroup>(
      seg.compiledGraphBackendArtifact);
  std::lock_guard<std::mutex> executionLock(compiled->executionMtx);
  if (!compiled->valid || compiled->startSlot != seg.def.startSlot ||
      compiled->endSlot != seg.def.endSlot ||
      compiled->shapeKey != seg.def.shapeKeyState.compiledShapeKey) {
    return fail(
        "ACL compiled artifact is invalid or stale: artifactSlots=" +
        std::to_string(compiled->startSlot) + "-" +
        std::to_string(compiled->endSlot) + ", artifactShapeKey=" +
        std::to_string(compiled->shapeKey) + ", planShapeKey=" +
        std::to_string(seg.def.shapeKeyState.compiledShapeKey));
  }

  auto bindTensor = [](NDArray* array,
                       const std::shared_ptr<arm_compute::Tensor>& tensor,
                       bool& staged) {
    staged = false;
    if (array == nullptr || tensor == nullptr || array->dataBuffer() == nullptr ||
        !array->dataBuffer()->isValid() || !isDenseCOrder(array)) {
      return false;
    }
    array->syncToHost();
    if (tensor->allocator()->is_allocated()) tensor->allocator()->free();
    if (!array->hasPaddedBuffer() && !tensor->info()->has_padding() &&
        isDenseCOrder(array)) {
      return static_cast<bool>(
          tensor->allocator()->import_memory(array->buffer()));
    }
    tensor->allocator()->allocate();
    if (array->rankOf() == 0) {
      std::memcpy(tensor->buffer(), array->buffer(), array->sizeOfT());
    } else {
      sd::ops::platforms::copyToTensor(*array, *tensor);
    }
    staged = true;
    return true;
  };

  std::unordered_set<int> stagedProducedSlots;

  // Import buffers from NDArrays into ACL tensors
  for (auto& [slotIdx, tensor] : compiled->slotToTensor) {
    if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx] != nullptr) {
      NDArray* arr = outputSlots[slotIdx];
      bool staged = false;
      if (!bindTensor(arr, tensor, staged)) {
        return fail("ACL output-slot tensor binding failed: slot=" +
                    std::to_string(slotIdx));
      }
      if (staged && compiled->producedSlots.count(slotIdx) != 0) {
        stagedProducedSlots.insert(slotIdx);
      }
    }
  }

  for (auto& [extIdx, tensor] : compiled->extToTensor) {
    if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs[extIdx] != nullptr) {
      NDArray* arr = externalInputs[extIdx];
      bool staged = false;
      if (!bindTensor(arr, tensor, staged)) {
        return fail("ACL external-input tensor binding failed: externalIndex=" +
                    std::to_string(extIdx));
      }
    }
  }

  auto resolveSourceArray = [&](int sourceIndex) -> NDArray* {
    if (sourceIndex < 0) {
      const int externalIndex = -(sourceIndex + 1);
      return externalIndex >= 0 && externalIndex < numExternalInputs
                 ? externalInputs[externalIndex]
                 : nullptr;
    }
    return sourceIndex < totalOutputSlots ? outputSlots[sourceIndex] : nullptr;
  };

  // Execute all functions in order
  for (auto& entry : compiled->functions) {
    if (auto* gather =
            dynamic_cast<StagedAclGatherFunction*>(entry.function.get())) {
      NDArray* indices = resolveSourceArray(gather->sourceIndex());
      if (!gather->stageIndices(indices)) {
        DSP_DIAG(EXECUTE,
                 "ACL_GATHER_STAGING_FAILED seg[%d-%d] source_slot=%d",
                 seg.def.startSlot, seg.def.endSlot, gather->sourceIndex());
        return fail("ACL gather index staging failed: sourceSlot=" +
                    std::to_string(gather->sourceIndex()));
      }
    } else if (auto* subtract = dynamic_cast<CompiledAclInt64SubtractFunction*>(
                   entry.function.get())) {
      NDArray* left = resolveSourceArray(subtract->leftSource());
      NDArray* right = resolveSourceArray(subtract->rightSource());
      NDArray* output = subtract->outputSlot() >= 0 &&
                                subtract->outputSlot() < totalOutputSlots
                            ? outputSlots[subtract->outputSlot()]
                            : nullptr;
      if (!subtract->validateBindings(left, right, output)) {
        DSP_DIAG(EXECUTE,
                 "ACL_INT64_SUBTRACT_BINDING_FAILED seg[%d-%d] left=%d "
                 "right=%d output=%d",
                 seg.def.startSlot, seg.def.endSlot, subtract->leftSource(),
                 subtract->rightSource(), subtract->outputSlot());
        return fail(
            "ACL INT64 subtract binding validation failed: left=" +
            std::to_string(subtract->leftSource()) + ", right=" +
            std::to_string(subtract->rightSource()) + ", output=" +
            std::to_string(subtract->outputSlot()));
      }
    }
    entry.function->run();
  }

  // Copy results back if needed (when import_memory wasn't used)
  for (int slotIdx : compiled->producedSlots) {
    auto tensorIt = compiled->slotToTensor.find(slotIdx);
    if (tensorIt != compiled->slotToTensor.end() && slotIdx >= 0 &&
        slotIdx < totalOutputSlots && outputSlots[slotIdx] != nullptr) {
      NDArray* arr = outputSlots[slotIdx];
      auto& tensor = tensorIt->second;
      if (stagedProducedSlots.count(slotIdx) != 0) {
        if (arr->rankOf() == 0) {
          std::memcpy(arr->buffer(), tensor->buffer(), arr->sizeOfT());
        } else {
          sd::ops::platforms::copyFromTensor(*tensor, *arr);
        }
      }
      arr->tickWriteHost();
    }
  }

  return Status::OK;
}

// ─── Cache invalidation ─────────────────────────────────────────────────────

void AclGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& weakArtifact : compiledArtifacts_) {
    if (auto artifact = weakArtifact.lock()) artifact->invalidate();
  }
  compiledArtifacts_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ARMCOMPUTE
