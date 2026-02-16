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

#include <graph/NativePlanCompiler.h>
#include <graph/generated/graph_generated.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <queue>
#include <unordered_set>

// Bring in FlatBuffer-generated types from the ::graph namespace
using namespace ::graph;

namespace sd {
namespace graph {

// ─── Op classification ─────────────────────────────────────────────────────

bool NativePlanCompiler::isDataDependentOp(const std::string& opName) {
  static const std::unordered_set<std::string> DATA_DEPENDENT_OPS = {
      "Where", "unique", "non_max_suppression", "non_max_suppression_v3"
  };
  return DATA_DEPENDENT_OPS.count(opName) > 0;
}

bool NativePlanCompiler::isFullyWritingOp(const std::string& opName) {
  static const std::unordered_set<std::string> FULLY_WRITING_OPS = {
      "add", "subtract", "multiply", "divide", "multiply_no_nan",
      "matmul", "mmul", "batched_gemm",
      "relu", "sigmoid", "tanh", "softmax", "log_softmax", "gelu", "silu", "swish", "mish",
      "exp", "log", "abs", "neg", "square", "sqrt", "rsqrt", "reciprocal",
      "sin", "cos", "ceil", "floor", "round",
      "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
      "reduce_sum_bp", "reduce_mean_bp",
      "concat", "stack", "unstack", "split", "slice",
      "reshape", "permute", "transpose", "expand_dims", "squeeze",
      "cast", "identity", "assign",
      "layer_norm", "batch_norm", "normalize_moments",
      "conv2d", "maxpool2d", "avgpool2d",
      "gather", "gather_nd", "scatter_update",
      "tile", "repeat", "broadcast_to",
      "zeros_like", "ones_like", "fill",
      "pow", "maximum", "minimum", "mod", "floormod",
      "greater", "less", "greater_equal", "less_equal", "equals", "not_equals",
      "logical_and", "logical_or", "logical_not",
      "clip_by_value",
  };
  return FULLY_WRITING_OPS.count(opName) > 0;
}

bool NativePlanCompiler::isValueDependentShapeOp(const std::string& opName) {
  static const std::unordered_set<std::string> VALUE_DEPENDENT_OPS = {
      "reshape", "squeeze", "expand_dims",
      "slice", "strided_slice", "gather", "gather_nd",
      "tile", "repeat", "pad", "fill",
      "range", "linspace",
      "shape_of", "size_at", "rank",
  };
  return VALUE_DEPENDENT_OPS.count(opName) > 0;
}

// ─── Compile FlatGraph → NativeDynamicShapePlan ─────────────────────────────

NativeDynamicShapePlan* NativePlanCompiler::compile(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {

  if (!graph) return nullptr;

  auto* nodes = graph->nodes();
  auto* flatVars = graph->variables();
  if (!nodes || nodes->size() == 0) return nullptr;

  // ── Step 1: Build variable type maps ──────────────────────────────────────
  std::unordered_set<std::string> constants;
  std::unordered_set<std::string> placeholders;
  std::unordered_set<std::string> variableNames;

  if (flatVars) {
    for (unsigned int i = 0; i < flatVars->size(); i++) {
      auto* fv = flatVars->Get(i);
      if (!fv || !fv->name()) continue;
      std::string name = fv->name()->str();
      auto vtype = fv->variabletype();
      switch (vtype) {
        case VarType_CONSTANT: constants.insert(name); break;
        case VarType_PLACEHOLDER: placeholders.insert(name); break;
        case VarType_VARIABLE: variableNames.insert(name); break;
        default: break;
      }
    }
  }

  // ── Step 2: Filter to actual execution ops ────────────────────────────────
  std::vector<const FlatNode*> opNodes;
  for (unsigned int i = 0; i < nodes->size(); i++) {
    auto* node = nodes->Get(i);
    if (!node) continue;
    auto opType = node->opType();
    // Skip VARIABLE and LOGIC types
    if (opType == OpType_VARIABLE || opType == OpType_LOGIC) continue;
    opNodes.push_back(node);
  }

  if (opNodes.empty()) return nullptr;

  int numSteps = static_cast<int>(opNodes.size());

  // ── Step 3: Build external input index map ────────────────────────────────
  std::vector<std::string> externalInputKeys;
  std::unordered_map<std::string, int> externalIndexMap;

  auto addExternal = [&](const std::string& name) -> int {
    auto it = externalIndexMap.find(name);
    if (it != externalIndexMap.end()) return it->second;
    int idx = static_cast<int>(externalInputKeys.size());
    externalInputKeys.push_back(name);
    externalIndexMap[name] = idx;
    return idx;
  };

  // Pre-register all constants, variables, placeholders
  for (auto& name : constants) addExternal(name);
  for (auto& name : variableNames) addExternal(name);
  for (auto& name : placeholders) addExternal(name);

  // ── Step 4: Assign output slot indices ────────────────────────────────────
  std::unordered_map<std::string, int> varToOutputSlot;
  int totalOutputSlots = 0;

  for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
    auto* node = opNodes[stepIdx];
    auto* outputNames = node->outputNames();
    if (outputNames) {
      for (unsigned int i = 0; i < outputNames->size(); i++) {
        std::string name = outputNames->Get(i)->str();
        varToOutputSlot[name] = totalOutputSlots;
        totalOutputSlots++;
      }
    } else {
      // Single unnamed output
      std::string name = node->name() ? node->name()->str() :
                          ("node_" + std::to_string(node->id()));
      varToOutputSlot[name] = totalOutputSlots;
      totalOutputSlots++;
    }
  }

  // ── Step 5: Pre-build lookup maps for O(1) input resolution ─────────────
  // Maps node ID → FlatNode* for fast producer lookup (avoids O(N) per input)
  std::unordered_map<int, const FlatNode*> nodeById;
  for (unsigned int n = 0; n < nodes->size(); n++) {
    auto* srcNode = nodes->Get(n);
    if (srcNode) nodeById[srcNode->id()] = srcNode;
  }
  // Maps variable ID (first element of IntPair) → FlatVariable* for fast var lookup
  std::unordered_map<int, const ::graph::FlatVariable*> varById;
  if (flatVars) {
    for (unsigned int v = 0; v < flatVars->size(); v++) {
      auto* fv = flatVars->Get(v);
      if (fv && fv->id()) varById[fv->id()->first()] = fv;
    }
  }

  // ── Step 6: Build slots ───────────────────────────────────────────────────
  auto* plan = new NativeDynamicShapePlan();
  plan->numSlots_ = numSteps;
  plan->totalOutputSlots_ = totalOutputSlots;
  plan->numExternalInputs_ = static_cast<int>(externalInputKeys.size());
  plan->slots_ = new NativeSlot[numSteps];

  std::vector<int> slotLastConsumerStep(totalOutputSlots, -1);
  std::vector<int> slotProducerStep(totalOutputSlots, -1);

  for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
    auto* node = opNodes[stepIdx];
    NativeSlot& slot = plan->slots_[stepIdx];

    // Op identification
    slot.opHash = node->opNum();
    slot.opName = node->opName() ? node->opName()->str() :
                  (node->name() ? node->name()->str() : "unknown");
    slot.isCustomOp = (node->opType() == OpType_CUSTOM);

    // Resolve op
    slot.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.opHash);
    if (!slot.op && !slot.opName.empty()) {
      slot.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.opName.c_str());
    }
    if (!slot.op) {
      sd_printf("NativePlanCompiler: cannot resolve op hash=%lld name=%s\n",
                slot.opHash, slot.opName.c_str());
      delete plan;
      return nullptr;
    }

    // Classify op
    bool isDataDep = isDataDependentOp(slot.opName);
    slot.isDataDependent = isDataDep;
    slot.needsZeroedOutput = !isFullyWritingOp(slot.opName) || isDataDep;
    slot.outputShapeDependsOnInputValues = isValueDependentShapeOp(slot.opName) || isDataDep;
    slot.isIdentityOp = (slot.opName == "identity");

    // Build input wiring from inputPaired
    auto* inputPaired = node->inputPaired();
    int numInputs = inputPaired ? inputPaired->size() : 0;
    slot.numInputs = numInputs;
    slot.inputSourceIndices = new int[numInputs];
    slot.inputSourceTypes = new int8_t[numInputs];

    bool hasIntLong = false;
    for (int i = 0; i < numInputs; i++) {
      auto* pair = inputPaired->Get(i);
      int nodeId = pair->first();
      int outIdx = pair->second();

      // Build the variable name for this input using pre-built maps (O(1) lookup)
      std::string inputName;
      bool found = false;

      // Look up producing node by ID (O(1) via nodeById map)
      auto nodeIt = nodeById.find(nodeId);
      if (nodeIt != nodeById.end()) {
        auto* srcNode = nodeIt->second;
        auto* srcOutputNames = srcNode->outputNames();
        if (srcOutputNames && outIdx < static_cast<int>(srcOutputNames->size())) {
          inputName = srcOutputNames->Get(outIdx)->str();
        } else if (srcNode->name()) {
          inputName = srcNode->name()->str();
          if (outIdx > 0) inputName += ":" + std::to_string(outIdx);
        }
        found = true;
      }

      // Look up variable by ID (O(1) via varById map)
      if (!found) {
        auto varIt = varById.find(nodeId);
        if (varIt != varById.end()) {
          auto* fv = varIt->second;
          if (fv->name()) inputName = fv->name()->str();
          found = true;
        }
      }

      // Look up in output slot map or external
      auto slotIt = varToOutputSlot.find(inputName);
      if (slotIt != varToOutputSlot.end()) {
        slot.inputSourceIndices[i] = slotIt->second;
        slot.inputSourceTypes[i] = SOURCE_OP_OUTPUT;
        if (stepIdx > slotLastConsumerStep[slotIt->second]) {
          slotLastConsumerStep[slotIt->second] = stepIdx;
        }
      } else {
        int extIdx = addExternal(inputName);
        slot.inputSourceIndices[i] = -(extIdx + 1);
        if (constants.count(inputName)) {
          slot.inputSourceTypes[i] = SOURCE_CONSTANT;
        } else if (variableNames.count(inputName)) {
          slot.inputSourceTypes[i] = SOURCE_VARIABLE;
        } else {
          slot.inputSourceTypes[i] = SOURCE_PLACEHOLDER;
        }
      }
    }
    slot.needsIntLongSync = hasIntLong || isDataDep;

    // Build output wiring
    auto* outputNames = node->outputNames();
    int numOutputs = outputNames ? outputNames->size() : 1;
    slot.numOutputs = numOutputs;
    slot.outputSlotIndices = new int[numOutputs];

    for (int i = 0; i < numOutputs; i++) {
      std::string outName;
      if (outputNames && i < static_cast<int>(outputNames->size())) {
        outName = outputNames->Get(i)->str();
      } else {
        outName = node->name() ? node->name()->str() : ("node_" + std::to_string(node->id()));
      }
      auto it = varToOutputSlot.find(outName);
      slot.outputSlotIndices[i] = (it != varToOutputSlot.end()) ? it->second : -1;
      if (it != varToOutputSlot.end()) {
        slotProducerStep[it->second] = stepIdx;
      }
    }

    // Freeze arguments from FlatNode
    auto* extraInteger = node->extraInteger();
    if (extraInteger && extraInteger->size() > 0) {
      slot.numIArgs = extraInteger->size();
      slot.iArgs = new LongType[slot.numIArgs];
      for (int i = 0; i < slot.numIArgs; i++) {
        slot.iArgs[i] = extraInteger->Get(i);
      }
    }

    auto* extraParams = node->extraParams();
    if (extraParams && extraParams->size() > 0) {
      slot.numTArgs = extraParams->size();
      slot.tArgs = new double[slot.numTArgs];
      for (int i = 0; i < slot.numTArgs; i++) {
        slot.tArgs[i] = extraParams->Get(i);
      }
    }

    auto* extraBools = node->extraBools();
    if (extraBools && extraBools->size() > 0) {
      slot.numBArgs = extraBools->size();
      slot.bArgs = new bool[slot.numBArgs];
      for (int i = 0; i < slot.numBArgs; i++) {
        slot.bArgs[i] = extraBools->Get(i);
      }
    }

    auto* extraTypes = node->extraTypes();
    if (extraTypes && extraTypes->size() > 0) {
      slot.numDArgs = extraTypes->size();
      slot.dArgs = new DataType[slot.numDArgs];
      for (int i = 0; i < slot.numDArgs; i++) {
        // Convert FlatBuffer DType to native DataType
        slot.dArgs[i] = static_cast<DataType>(extraTypes->Get(i));
      }
    }

    slot.targetDeviceId = node->device();
  }

  // ── Step 6: Build release schedule ────────────────────────────────────────
  // Identify requested output slots
  std::unordered_set<int> finalOutputSlots;
  plan->numRequestedOutputs_ = static_cast<int>(requestedOutputs.size());
  plan->requestedOutputSlotIndices_ = new int[plan->numRequestedOutputs_];
  for (int i = 0; i < plan->numRequestedOutputs_; i++) {
    auto it = varToOutputSlot.find(requestedOutputs[i]);
    if (it != varToOutputSlot.end()) {
      plan->requestedOutputSlotIndices_[i] = it->second;
      finalOutputSlots.insert(it->second);
      slotLastConsumerStep[it->second] = INT32_MAX;
    } else {
      plan->requestedOutputSlotIndices_[i] = -1;
    }
  }

  plan->releaseAtStep_ = new int*[numSteps];
  plan->releaseAtStepCounts_ = new int[numSteps];

  for (int step = 0; step < numSteps; step++) {
    std::vector<int> toRelease;
    for (int slotIdx = 0; slotIdx < totalOutputSlots; slotIdx++) {
      if (slotLastConsumerStep[slotIdx] == step && !finalOutputSlots.count(slotIdx)) {
        toRelease.push_back(slotIdx);
      }
    }
    plan->releaseAtStepCounts_[step] = static_cast<int>(toRelease.size());
    if (!toRelease.empty()) {
      plan->releaseAtStep_[step] = new int[toRelease.size()];
      std::copy(toRelease.begin(), toRelease.end(), plan->releaseAtStep_[step]);
    } else {
      plan->releaseAtStep_[step] = nullptr;
    }
  }

  // ── Step 7: Allocate execution state ──────────────────────────────────────
  plan->outputSlots_ = new NDArray*[totalOutputSlots];
  std::memset(plan->outputSlots_, 0, sizeof(NDArray*) * totalOutputSlots);

  plan->slotArrayCache_ = new NDArray*[totalOutputSlots];
  std::memset(plan->slotArrayCache_, 0, sizeof(NDArray*) * totalOutputSlots);

  plan->slotIsViewProducer_ = new bool[totalOutputSlots];
  std::memset(plan->slotIsViewProducer_, 0, sizeof(bool) * totalOutputSlots);

  plan->contextPool_ = new Context*[numSteps];
  for (int i = 0; i < numSteps; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // ── Shape static analysis (same as fromSerializedPlan) ──────────────────
  {
    std::vector<int> outputSlotToStepIndex(totalOutputSlots, -1);
    for (int s = 0; s < numSteps; s++) {
      NativeSlot& slot = plan->slots_[s];
      for (int i = 0; i < slot.numOutputs; i++) {
        int si = slot.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots) {
          outputSlotToStepIndex[si] = s;
        }
      }
    }

    int staticCount = 0, dynamicCount = 0;
    for (int s = 0; s < numSteps; s++) {
      NativeSlot& slot = plan->slots_[s];
      slot.shapeStatic = true;

      if (slot.isDataDependent || slot.outputShapeDependsOnInputValues) {
        slot.shapeStatic = false;
        dynamicCount++;
        continue;
      }

      for (int i = 0; i < slot.numInputs; i++) {
        int srcIdx = slot.inputSourceIndices[i];
        if (srcIdx < 0) {
          if (slot.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            slot.shapeStatic = false;
            break;
          }
        } else {
          if (srcIdx < totalOutputSlots) {
            int producerStep = outputSlotToStepIndex[srcIdx];
            if (producerStep >= 0 && !plan->slots_[producerStep].shapeStatic) {
              slot.shapeStatic = false;
              break;
            }
          }
        }
      }

      if (slot.shapeStatic) staticCount++;
      else dynamicCount++;
    }

    sd_printf("NativePlanCompiler: shape analysis: %d static, %d dynamic out of %d slots\n",
              staticCount, dynamicCount, numSteps);
  }

  // Build CUDA graph segments
  plan->buildSegments();

  return plan;
}

}  // namespace graph
}  // namespace sd
