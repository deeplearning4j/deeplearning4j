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

#include <array/ByteOrderUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <graph/NativePlanCompiler.h>
#include <graph/PlanDefinition.h>
#include <graph/ExecutionState.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspSegmentOutputUtils.h>
#include <graph/GraphBackendResolver.h>
#include <graph/Node.h>
#include <graph/generated/graph_generated.h>
#include <helpers/helper_hash.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <queue>
#include <unordered_set>

// Bring in FlatBuffer-generated types from the ::graph namespace
using namespace ::graph;

namespace sd {
namespace graph {

int legacyOpTypeForFlatOp(::graph::OpType opType, int numInputs) noexcept {
  switch (opType) {
    case ::graph::OpType_TRANSFORM_FLOAT:
      return numInputs > 1 ? LEGACY_PAIRWISE_TRANSFORM
                           : LEGACY_TRANSFORM_FLOAT;
    case ::graph::OpType_TRANSFORM_SAME:
      return numInputs > 1 ? LEGACY_PAIRWISE_TRANSFORM
                           : LEGACY_TRANSFORM_SAME;
    case ::graph::OpType_TRANSFORM_STRICT:
      return numInputs > 1 ? LEGACY_PAIRWISE_TRANSFORM
                           : LEGACY_TRANSFORM_STRICT;
    case ::graph::OpType_TRANSFORM_ANY:
      return numInputs > 1 ? LEGACY_PAIRWISE_TRANSFORM
                           : LEGACY_TRANSFORM_ANY;
    case ::graph::OpType_TRANSFORM_BOOL:
      return LEGACY_TRANSFORM_BOOL;
    case ::graph::OpType_PAIRWISE:
      return LEGACY_PAIRWISE_TRANSFORM;
    case ::graph::OpType_PAIRWISE_BOOL:
      return LEGACY_PAIRWISE_BOOL;
    case ::graph::OpType_SCALAR:
      return LEGACY_SCALAR;
    case ::graph::OpType_SCALAR_BOOL:
      return LEGACY_SCALAR_BOOL;
    case ::graph::OpType_REDUCE_FLOAT:
      return LEGACY_REDUCE_FLOAT;
    case ::graph::OpType_REDUCE_SAME:
      return LEGACY_REDUCE_SAME;
    case ::graph::OpType_REDUCE_BOOL:
      return LEGACY_REDUCE_BOOL;
    case ::graph::OpType_REDUCE_LONG:
      return LEGACY_REDUCE_LONG;
    case ::graph::OpType_REDUCE_3:
      return LEGACY_REDUCE3;
    case ::graph::OpType_SUMMARYSTATS:
      return LEGACY_STATS;
    case ::graph::OpType_INDEX_REDUCE:
      return LEGACY_INDEX_REDUCE;
    case ::graph::OpType_BROADCAST:
      return LEGACY_BROADCAST;
    case ::graph::OpType_BROADCAST_BOOL:
      return LEGACY_BROADCAST_BOOL;
    case ::graph::OpType_RANDOM:
      return LEGACY_RANDOM;
    default:
      return LEGACY_NOT_SET;
  }
}

namespace {
std::string normalizeOpName(const std::string& opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}

bool flatScalarAsDouble(const ::graph::FlatArray* flatScalar, double* value,
                        std::string* errorMessage) {
  if (flatScalar == nullptr || value == nullptr) {
    if (errorMessage != nullptr) *errorMessage = "FlatNode scalar is null";
    return false;
  }
  auto* buffer = flatScalar->buffer();
  if (buffer == nullptr) {
    if (errorMessage != nullptr) *errorMessage = "FlatNode scalar has no data buffer";
    return false;
  }

  try {
    const auto dataType = sd::DataTypeUtils::fromFlatDataType(flatScalar->dtype());
    const auto elementSize = sd::DataTypeUtils::sizeOf(dataType);
    if (elementSize == 0 || buffer->size() < elementSize) {
      if (errorMessage != nullptr) {
        *errorMessage = "FlatNode scalar buffer is smaller than its declared data type";
      }
      return false;
    }
    sd::DataTypeConversions<double>::convertType(
        value, const_cast<int8_t*>(buffer->data()), dataType,
        sd::ByteOrderUtils::fromFlatByteOrder(flatScalar->byteOrder()), 1);
    return true;
  } catch (const std::exception& error) {
    if (errorMessage != nullptr) *errorMessage = error.what();
    return false;
  }
}
}  // namespace

// Op classification is owned by each operation's OpDescriptor.

static bool hasOpTrait(sd::ops::DeclarableOp* op, uint64_t trait) {
  if (op && op->getOpDescriptor()) {
    return op->getOpDescriptor()->hasAnyTrait(trait);
  }
  return false;
}

// ─── Compile FlatGraph → NativeDynamicShapePlan ─────────────────────────────

NativeDynamicShapePlan* NativePlanCompiler::compile(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs,
    GraphExecutionMode mode,
    std::string* errorMessage,
    const NativePlanCompileOptions& compileOptions) {

  if (errorMessage != nullptr) errorMessage->clear();
  auto fail = [&](const std::string& message) -> NativeDynamicShapePlan* {
    if (errorMessage != nullptr) *errorMessage = message;
    DSP_DIAG(COMPILE, "NativePlanCompiler::compile: %s", message.c_str());
    return nullptr;
  };

  if (!graph) return fail("FlatGraph is null");

  auto* nodes = graph->nodes();
  auto* flatVars = graph->variables();
  if (!nodes || nodes->size() == 0) return fail("FlatGraph has no nodes");

  DSP_DIAG(COMPILE, "NativePlanCompiler::compile: ENTER nodes=%d vars=%d requestedOutputs=%d",
           (int)nodes->size(), flatVars ? (int)flatVars->size() : 0, (int)requestedOutputs.size());

  // ── Step 1: Build variable type and exact identity maps ───────────────────
  std::unordered_set<std::string> constants;
  std::unordered_set<std::string> placeholders;
  std::unordered_set<std::string> variableNames;

  auto pairKey = [](int first, int second) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(first)) << 32U) |
           static_cast<uint32_t>(second);
  };
  std::unordered_map<uint64_t, const ::graph::FlatVariable*> variableByPair;

  if (flatVars) {
    for (unsigned int i = 0; i < flatVars->size(); i++) {
      auto* fv = flatVars->Get(i);
      if (!fv || !fv->name()) continue;
      std::string name = fv->name()->str();
      if (fv->id()) {
        variableByPair[pairKey(fv->id()->first(), fv->id()->second())] = fv;
      }
      auto vtype = fv->variabletype();
      switch (vtype) {
        case VarType_CONSTANT: constants.insert(name); break;
        case VarType_PLACEHOLDER: placeholders.insert(name); break;
        case VarType_VARIABLE: variableNames.insert(name); break;
        default: break;
      }
    }
  }

  // ── Step 2: Index executable nodes and their exact outputs ────────────────
  std::vector<const FlatNode*> serializedOpNodes;
  std::unordered_map<int, const FlatNode*> nodeById;
  std::unordered_map<const FlatNode*, int> serializedIndex;
  for (unsigned int i = 0; i < nodes->size(); i++) {
    auto* node = nodes->Get(i);
    if (!node) continue;
    auto opType = node->opType();
    // Skip VARIABLE and LOGIC types
    if (opType == OpType_VARIABLE || opType == OpType_LOGIC) continue;
    if (nodeById.find(node->id()) != nodeById.end()) {
      return fail("duplicate executable node id: " + std::to_string(node->id()));
    }
    serializedIndex[node] = static_cast<int>(serializedOpNodes.size());
    nodeById[node->id()] = node;
    serializedOpNodes.push_back(node);
  }

  if (serializedOpNodes.empty()) {
    return fail("FlatGraph has no executable operation nodes");
  }

  auto outputCountFor = [](const FlatNode* node) -> int {
    auto* names = node->outputNames();
    return names != nullptr && names->size() > 0
               ? static_cast<int>(names->size())
               : 1;
  };
  auto outputNameFor = [&](const FlatNode* node, int outputIndex) -> std::string {
    auto* names = node->outputNames();
    if (names != nullptr && outputIndex >= 0 &&
        outputIndex < static_cast<int>(names->size()) &&
        names->Get(outputIndex) != nullptr) {
      return names->Get(outputIndex)->str();
    }
    auto variableIt = variableByPair.find(pairKey(node->id(), outputIndex));
    if (variableIt != variableByPair.end() && variableIt->second->name()) {
      return variableIt->second->name()->str();
    }
    std::string name = node->name() ? node->name()->str()
                                    : ("node_" + std::to_string(node->id()));
    if (outputIndex > 0) name += ":" + std::to_string(outputIndex);
    return name;
  };

  std::unordered_map<uint64_t, const FlatNode*> producerByPair;
  std::unordered_map<std::string, const FlatNode*> producerByOutputName;
  std::unordered_map<const FlatNode*, std::vector<std::string>> nodeOutputNames;
  for (const auto* node : serializedOpNodes) {
    auto& outputNames = nodeOutputNames[node];
    const int outputCount = outputCountFor(node);
    outputNames.reserve(outputCount);
    for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
      const auto name = outputNameFor(node, outputIndex);
      if (name.empty()) {
        return fail("empty output variable name for node id " +
                    std::to_string(node->id()));
      }
      if (producerByOutputName.find(name) != producerByOutputName.end()) {
        return fail("duplicate output variable name: " + name);
      }
      const auto key = pairKey(node->id(), outputIndex);
      if (producerByPair.find(key) != producerByPair.end()) {
        return fail("duplicate executable output identity (" +
                    std::to_string(node->id()) + "," +
                    std::to_string(outputIndex) + ")");
      }
      outputNames.push_back(name);
      producerByOutputName[name] = node;
      producerByPair[key] = node;
    }
  }

  struct InputBinding {
    bool valid = false;
    int sourceId = 0;
    int outputIndex = 0;
    std::string name;
    const FlatNode* producer = nullptr;
    const ::graph::FlatVariable* variable = nullptr;
  };

  auto inputCountFor = [](const FlatNode* node) -> int {
    auto* paired = node->inputPaired();
    if (paired != nullptr && paired->size() > 0) {
      return static_cast<int>(paired->size());
    }
    auto* legacy = node->input();
    return legacy != nullptr ? static_cast<int>(legacy->size()) : 0;
  };

  auto resolveInput = [&](const FlatNode* node, int inputIndex) -> InputBinding {
    InputBinding binding;
    auto* paired = node->inputPaired();
    if (paired != nullptr && paired->size() > 0) {
      if (inputIndex < 0 || inputIndex >= static_cast<int>(paired->size()) ||
          paired->Get(inputIndex) == nullptr) {
        return binding;
      }
      binding.sourceId = paired->Get(inputIndex)->first();
      binding.outputIndex = paired->Get(inputIndex)->second();
    } else {
      auto* legacy = node->input();
      if (legacy == nullptr || inputIndex < 0 ||
          inputIndex >= static_cast<int>(legacy->size())) {
        return binding;
      }
      binding.sourceId = legacy->Get(inputIndex);
      binding.outputIndex = 0;
    }

    const auto key = pairKey(binding.sourceId, binding.outputIndex);
    auto variableIt = variableByPair.find(key);
    if (variableIt != variableByPair.end()) {
      binding.variable = variableIt->second;
      // PLACEHOLDER/CONSTANT/VARIABLE identities are external even when their
      // first ID collides with an executable node ID. ARRAY identities describe
      // operation outputs and therefore defer to an exact producer when present.
      if (binding.variable->variabletype() != VarType_ARRAY) {
        binding.name = binding.variable->name() ? binding.variable->name()->str() : "";
        binding.valid = !binding.name.empty();
        return binding;
      }
    }

    auto producerIt = producerByPair.find(key);
    if (producerIt != producerByPair.end()) {
      binding.producer = producerIt->second;
      const auto& names = nodeOutputNames.at(binding.producer);
      if (binding.outputIndex >= 0 &&
          binding.outputIndex < static_cast<int>(names.size())) {
        binding.name = names[binding.outputIndex];
        binding.valid = true;
        return binding;
      }
    }

    if (binding.variable != nullptr && binding.variable->name()) {
      binding.name = binding.variable->name()->str();
      binding.valid = !binding.name.empty();
    }
    return binding;
  };

  // ── Step 3: Retain only operations needed by the requested outputs ────────
  std::unordered_set<const FlatNode*> reachable;
  std::vector<const FlatNode*> pending;
  if (requestedOutputs.empty()) {
    pending = serializedOpNodes;
  } else {
    pending.reserve(requestedOutputs.size());
    for (const auto& output : requestedOutputs) {
      auto producerIt = producerByOutputName.find(output);
      if (producerIt == producerByOutputName.end()) {
        return fail("requested output is not produced by the graph: " + output);
      }
      pending.push_back(producerIt->second);
    }
  }
  while (!pending.empty()) {
    const FlatNode* node = pending.back();
    pending.pop_back();
    if (!reachable.insert(node).second) continue;
    const int inputCount = inputCountFor(node);
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
      const auto binding = resolveInput(node, inputIndex);
      if (!binding.valid) {
        return fail("cannot resolve input " + std::to_string(inputIndex) +
                    " of node id " + std::to_string(node->id()));
      }
      if (binding.producer != nullptr && binding.producer != node) {
        pending.push_back(binding.producer);
      }
    }
  }

  // Stable Kahn ordering preserves serialized order for independent operations
  // while guaranteeing that every ordinary producer precedes its consumers.
  std::vector<int> indegree(serializedOpNodes.size(), 0);
  std::vector<std::vector<int>> consumers(serializedOpNodes.size());
  for (const auto* node : serializedOpNodes) {
    if (!reachable.count(node)) continue;
    const int consumerIndex = serializedIndex.at(node);
    std::unordered_set<int> uniqueProducers;
    const int inputCount = inputCountFor(node);
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
      const auto binding = resolveInput(node, inputIndex);
      if (binding.producer == nullptr || binding.producer == node ||
          !reachable.count(binding.producer)) {
        continue;
      }
      const int producerIndex = serializedIndex.at(binding.producer);
      if (uniqueProducers.insert(producerIndex).second) {
        indegree[consumerIndex]++;
        consumers[producerIndex].push_back(consumerIndex);
      }
    }
  }

  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int index = 0; index < static_cast<int>(serializedOpNodes.size()); index++) {
    if (reachable.count(serializedOpNodes[index]) && indegree[index] == 0) {
      ready.push(index);
    }
  }
  std::vector<const FlatNode*> opNodes;
  opNodes.reserve(reachable.size());
  while (!ready.empty()) {
    const int index = ready.top();
    ready.pop();
    opNodes.push_back(serializedOpNodes[index]);
    for (const int consumer : consumers[index]) {
      if (--indegree[consumer] == 0) ready.push(consumer);
    }
  }
  if (opNodes.size() != reachable.size()) {
    bool hasControlFlowCycle = false;
    for (int index = 0; index < static_cast<int>(serializedOpNodes.size()); index++) {
      if (!reachable.count(serializedOpNodes[index]) || indegree[index] == 0) continue;
      const auto* node = serializedOpNodes[index];
      const auto name = normalizeOpName(
          node->opName() ? node->opName()->str()
                         : (node->name() ? node->name()->str() : ""));
      if (name == "switch" || name == "merge" || name == "enter" ||
          name == "exit" || name == "next_iteration" || name == "loop_cond") {
        hasControlFlowCycle = true;
        break;
      }
    }
    if (!hasControlFlowCycle) {
      return fail("cycle detected while topologically ordering requested graph operations");
    }
    // Control-flow loop backedges are intentionally cyclic. Keep the acyclic
    // prefix and append the remaining loop operations in serialized order.
    std::unordered_set<const FlatNode*> alreadyOrdered(opNodes.begin(), opNodes.end());
    for (const auto* node : serializedOpNodes) {
      if (reachable.count(node) && !alreadyOrdered.count(node)) opNodes.push_back(node);
    }
  }

  const int numSteps = static_cast<int>(opNodes.size());

  // ── Step 4: Build external input index map on first reachable use ─────────
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

  // ── Step 5: Assign output slot indices ────────────────────────────────────
  std::unordered_map<std::string, int> varToOutputSlot;
  int totalOutputSlots = 0;

  for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
    auto* node = opNodes[stepIdx];
    for (const auto& name : nodeOutputNames.at(node)) {
      varToOutputSlot[name] = totalOutputSlots;
      totalOutputSlots++;
    }
  }

  // ── Step 6: Build slots ───────────────────────────────────────────────────
  auto* plan = new NativeDynamicShapePlan();
  plan->setGraphExecutionMode(mode);
  plan->setRuntimeCompilationAllowed(
      compileOptions.runtimeCompilationAllowed);
  plan->setRuntimeArtifactDirectory(
      compileOptions.runtimeArtifactDirectory);
  plan->setDeviceCompilationCacheDirectory(
      compileOptions.deviceCompilationCacheDirectory);
  plan->setDeviceCompilationCacheModelKey(
      compileOptions.deviceCompilationCacheModelKey);
  plan->numSlots_ = numSteps;
  plan->totalOutputSlots_ = totalOutputSlots;
  plan->dirtySlotGenerations_.resize(totalOutputSlots, 0);
  plan->slots_ = new NativeSlot[numSteps];

  std::vector<int> slotLastConsumerStep(totalOutputSlots, -1);
  std::vector<int> slotProducerStep(totalOutputSlots, -1);

  for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
    auto* node = opNodes[stepIdx];
    NativeSlot& slot = plan->slots_[stepIdx];

    // Op identification — resolve by name first since FlatGraph hashes may differ.
    LongType serializedOpHash = node->opNum();
    slot.ident.opHash = serializedOpHash;
    slot.ident.opName = node->opName() ? node->opName()->str() :
                  (node->name() ? node->name()->str() : "unknown");
    const int numInputs = inputCountFor(node);
    slot.flags.isCustomOp = (node->opType() == OpType_CUSTOM);
    slot.legacy.legacyOpType =
        legacyOpTypeForFlatOp(node->opType(), numInputs);
    slot.legacy.legacyOpNum = slot.legacy.legacyOpType != LEGACY_NOT_SET
                                  ? static_cast<int>(node->opNum())
                                  : -1;

    // Early control flow detection — CF ops are not registered as declarable ops
    // in OpRegistrator, so we must identify them before op resolution.
    {
      auto normalized = normalizeOpName(slot.ident.opName);
      slot.cf.controlFlowType = CF_NONE;
      slot.cf.loopBackTarget = -1;
      slot.cf.loopRegionIndex = -1;
      if (normalized == "switch") {
        slot.cf.controlFlowType = CF_SWITCH;
      } else if (normalized == "merge") {
        slot.cf.controlFlowType = CF_MERGE;
      } else if (normalized == "enter") {
        slot.cf.controlFlowType = CF_ENTER;
      } else if (normalized == "exit") {
        slot.cf.controlFlowType = CF_EXIT;
      } else if (normalized == "next_iteration") {
        slot.cf.controlFlowType = CF_NEXT_ITERATION;
      } else if (normalized == "loop_cond") {
        slot.cf.controlFlowType = CF_LOOP_COND;
      }
    }

    // Resolve op — legacy FlatGraph families carry an enum op number, not a
    // declarable-op hash. Construct their wrapper explicitly and retain it for
    // the lifetime of the compiled plan. Control-flow nodes are data routing.
    slot.ident.op = nullptr;
    if (slot.cf.controlFlowType == CF_NONE &&
        slot.legacy.legacyOpType != LEGACY_NOT_SET) {
      auto legacyFlatOpType = node->opType();
      if (slot.legacy.legacyOpType == LEGACY_PAIRWISE_TRANSFORM) {
        legacyFlatOpType = OpType_PAIRWISE;
      }
      try {
        slot.ident.op = Node::buildOpByType(
            legacyFlatOpType, numInputs, 0, 0,
            slot.legacy.legacyOpNum, nullptr);
      } catch (...) {
        slot.ident.op = nullptr;
      }
      if (!slot.ident.op) {
        delete plan;
        return fail("cannot construct legacy operation '" + slot.ident.opName +
                    "' (flat op type " +
                    std::to_string(static_cast<int>(node->opType())) +
                    ", op number " +
                    std::to_string(slot.legacy.legacyOpNum) + ")");
      }
      plan->ownedLegacyOps_.push_back(slot.ident.op);
      slot.ident.opHash = slot.ident.op->getOpDescriptor()->getHash();
    } else if (slot.cf.controlFlowType == CF_NONE) {
      if (!slot.ident.opName.empty()) {
        slot.ident.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.ident.opName.c_str());
      }
      // Fallback to serialized hash for graphs that carry native hashes already.
      if (!slot.ident.op) {
        slot.ident.op = sd::ops::OpRegistrator::getInstance().getOperation(serializedOpHash);
      }
      if (!slot.ident.op) {
        DSP_DIAG(COMPILE, "NativePlanCompiler: cannot resolve op hash=%lld name=%s",
                  serializedOpHash, slot.ident.opName.c_str());
        delete plan;
        return fail("cannot resolve operation '" + slot.ident.opName +
                    "' (serialized hash " +
                    std::to_string(serializedOpHash) + ")");
      }
      slot.ident.opHash = slot.ident.op->getOpDescriptor()->getHash();
    } else {
      // CF ops don't need an op pointer or hash — they're pure data routing.
      slot.ident.opHash = 0;
      DSP_DIAG(COMPILE, "slot %d: control flow op '%s' (type=%d)",
                stepIdx, slot.ident.opName.c_str(), (int)slot.cf.controlFlowType);
    }

    slot.disableInPlaceFusion();
    slot.fusedChain.isFusedChainHead = false;
    slot.fusedChain.fusedChainLength = 0;
    slot.fusedChain.isFusedChainTail = false;
    std::memset(slot.fusedChain.fusedChainOpCodes, 0, sizeof(slot.fusedChain.fusedChainOpCodes));
    std::memset(slot.fusedChain.fusedChainSlots, 0, sizeof(slot.fusedChain.fusedChainSlots));
    std::fill(std::begin(slot.fusedChain.fusedChainSecondaryInputSources), std::end(slot.fusedChain.fusedChainSecondaryInputSources), INT32_MIN);

    // Build input wiring from exact paired identities or legacy node IDs.

    // Copy intrinsic classification from the resolved operation. The descriptor
    // is the single source of truth; opName remains diagnostics-only.
    if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
      slot.opTraits_ = slot.ident.op->getOpDescriptor()->getTraits64();
    }
    // A ternary-elementwise op with exactly 3 inputs (e.g. select cond?x:y) has a fixed
    // broadcast output shape — not data-dependent / dynamic-output. Resolve by TRAIT + arity,
    // NEVER by hardcoded op name (trait handling must stay general across all such ops).
    if (slot.hasOpTrait(sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE) && numInputs == 3) {
      slot.clearOpTrait(sd::ops::OP_TRAIT_DATA_DEPENDENT);
      slot.clearOpTrait(sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
    }
    slot.wiring.numInputs = numInputs;
    slot.wiring.inputSourceIndices = new int[numInputs];
    slot.wiring.inputSourceTypes = new int8_t[numInputs];

    bool hasIntLong = false;
    for (int i = 0; i < numInputs; i++) {
      const auto binding = resolveInput(node, i);
      if (!binding.valid) {
        delete plan;
        return fail("cannot resolve input " + std::to_string(i) +
                    " of node id " + std::to_string(node->id()));
      }
      if (!hasIntLong && binding.variable != nullptr) {
        auto dt = binding.variable->dtype();
        if (dt == DType_INT32 || dt == DType_INT64) hasIntLong = true;
      }

      auto slotIt = binding.producer != nullptr && binding.producer != node
                        ? varToOutputSlot.find(binding.name)
                        : varToOutputSlot.end();
      if (slotIt != varToOutputSlot.end()) {
        slot.wiring.inputSourceIndices[i] = slotIt->second;
        slot.wiring.inputSourceTypes[i] = SOURCE_OP_OUTPUT;
        if (stepIdx > slotLastConsumerStep[slotIt->second]) {
          slotLastConsumerStep[slotIt->second] = stepIdx;
        }
      } else {
        if (binding.producer != nullptr && binding.producer != node) {
          delete plan;
          return fail("reachable producer output was not assigned a slot: " +
                      binding.name);
        }
        int extIdx = addExternal(binding.name);
        slot.wiring.inputSourceIndices[i] = -(extIdx + 1);
        if (constants.count(binding.name)) {
          slot.wiring.inputSourceTypes[i] = SOURCE_CONSTANT;
        } else if (variableNames.count(binding.name)) {
          slot.wiring.inputSourceTypes[i] = SOURCE_VARIABLE;
        } else {
          slot.wiring.inputSourceTypes[i] = SOURCE_PLACEHOLDER;
        }
        // Check external input NDArray dtype for INT/LONG detection
        if (!hasIntLong) {
          auto varIt2 = variables.find(binding.name);
          if (varIt2 != variables.end() && varIt2->second != nullptr) {
            auto dt = varIt2->second->dataType();
            if (dt == INT32 || dt == INT64) {
              hasIntLong = true;
            }
          }
        }
      }
    }
    slot.flags.needsIntLongSync = hasIntLong || slot.isDataDependent();
    // VALUE_DEPENDENT_SHAPE trait: ops whose output shapes depend on input VALUES
    // (not just shapes). Drives shapeStatic=false and shape key value-hashing.
    // DATA_DEPENDENT is orthogonal: it means the op's result depends on data,
    // not that the output shape depends on data. argmax/argmin are data-dependent
    // but their output shapes are determined by input shapes + axis iArgs.
    // Stored as a boolean because it can be cleared per-instance below.
    slot.flags.outputShapeDependsOnInputValues =
        slot.hasOpTrait(sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE);

    // Build output wiring from the same exact names used for dependency binding.
    const auto& outputNames = nodeOutputNames.at(node);
    int numOutputs = static_cast<int>(outputNames.size());
    slot.wiring.numOutputs = numOutputs;
    slot.wiring.outputSlotIndices = new int[numOutputs];

    for (int i = 0; i < numOutputs; i++) {
      const std::string& outName = outputNames[i];
      auto it = varToOutputSlot.find(outName);
      slot.wiring.outputSlotIndices[i] = (it != varToOutputSlot.end()) ? it->second : -1;
      if (it != varToOutputSlot.end()) {
        slotProducerStep[it->second] = stepIdx;
      }
    }

    // Freeze arguments from FlatNode
    // For reduce ops, dimensions are stored in the 'dimensions' field, not 'extraInteger'
    auto* dimensions = node->dimensions();
    if (dimensions && dimensions->size() > 0) {
      slot.args.numIArgs = dimensions->size();
      slot.args.iArgs = new LongType[slot.args.numIArgs];
      for (int i = 0; i < slot.args.numIArgs; i++) {
        slot.args.iArgs[i] = dimensions->Get(i);
      }
    } else {
      auto* extraInteger = node->extraInteger();
      if (extraInteger && extraInteger->size() > 0) {
        int numToUse = extraInteger->size();
        // Structural argument metadata belongs to the resolved operation.
        slot.flags.structuralIArgCount =
            slot.ident.op != nullptr
                ? slot.ident.op->getOpDescriptor()->getNumberOfStructuralIArgs()
                : -1;

        // Cap iArgs to structural-only when data comes from input tensors.
        // Ops like strided_slice have structural iArgs (masks, flags) followed by
        // data iArgs (begin/end/strides) — cap so the op reads data from inputs.
        if (slot.flags.structuralIArgCount >= 0 && numToUse > slot.flags.structuralIArgCount
            && slot.wiring.numInputs > 1) {
          DSP_DIAG(COMPILE, "Capping iArgs for op %s from %d to %d (structural only, data from inputs)",
                   slot.ident.opName.c_str(), numToUse, slot.flags.structuralIArgCount);
          numToUse = slot.flags.structuralIArgCount;
        }
        slot.args.numIArgs = numToUse;
        slot.args.iArgs = new LongType[slot.args.numIArgs];
        for (int i = 0; i < slot.args.numIArgs; i++) {
          slot.args.iArgs[i] = extraInteger->Get(i);
        }
      }
    }

    auto* extraParams = node->extraParams();
    auto* flatScalar = node->scalar();
    const int scalarArgCount = flatScalar == nullptr ? 0 : 1;
    const int extraParamCount =
        extraParams == nullptr ? 0 : static_cast<int>(extraParams->size());
    if (scalarArgCount + extraParamCount > 0) {
      slot.args.numTArgs = scalarArgCount + extraParamCount;
      slot.args.tArgs = new double[slot.args.numTArgs];
      if (flatScalar != nullptr) {
        std::string scalarError;
        if (!flatScalarAsDouble(flatScalar, &slot.args.tArgs[0], &scalarError)) {
          delete plan;
          return fail("cannot decode serialized scalar for node id " +
                      std::to_string(node->id()) + ": " + scalarError);
        }
      }
      for (int i = 0; i < extraParamCount; i++) {
        slot.args.tArgs[scalarArgCount + i] = extraParams->Get(i);
      }
    }

    auto* extraBools = node->extraBools();
    if (extraBools && extraBools->size() > 0) {
      slot.args.numBArgs = extraBools->size();
      slot.args.bArgs = new bool[slot.args.numBArgs];
      for (int i = 0; i < slot.args.numBArgs; i++) {
        slot.args.bArgs[i] = extraBools->Get(i);
      }
    }

    auto* extraTypes = node->extraTypes();
    if (extraTypes && extraTypes->size() > 0) {
      slot.args.numDArgs = extraTypes->size();
      slot.args.dArgs = new DataType[slot.args.numDArgs];
      for (int i = 0; i < slot.args.numDArgs; i++) {
        // Convert FlatBuffer DType to native DataType
        slot.args.dArgs[i] = static_cast<DataType>(extraTypes->Get(i));
      }
    }

    auto* extraStrings = node->extraStrings();
    if (extraStrings && extraStrings->size() > 0) {
      slot.args.numSArgs = extraStrings->size();
      slot.args.sArgs = new std::string[slot.args.numSArgs];
      for (int i = 0; i < slot.args.numSArgs; i++) {
        slot.args.sArgs[i] = extraStrings->Get(i)->str();
      }
    }

    slot.targetDeviceId = node->device();

    // Resolve the intrinsic value-dependent-shape trait for this concrete
    // invocation using descriptor traits, operand arity, and frozen arguments.
    // Operation names are diagnostic-only and never participate in semantics.
    if (slot.flags.outputShapeDependsOnInputValues) {
      const bool hasNoRuntimeInputs = slot.wiring.numInputs == 0;
      const bool argumentShapedView =
          slot.isViewCapableOp() && slot.wiring.numInputs <= 1 &&
          slot.args.numIArgs > 0;
      if (hasNoRuntimeInputs || argumentShapedView) {
        slot.flags.outputShapeDependsOnInputValues = false;
      }
    }
    if (!slot.flags.outputShapeDependsOnInputValues) {
      const bool tensorAxisConcat =
          slot.hasOpTrait(sd::ops::OP_TRAIT_CONCAT) &&
          slot.args.numBArgs > 0 && slot.args.bArgs[0];
      const bool tensorControlledView =
          slot.isViewCapableOp() && slot.isDataDependent() &&
          slot.wiring.numInputs > 1 && slot.args.numIArgs == 0;
      const bool tensorControlledReduction =
          slot.hasOpTrait(sd::ops::OP_TRAIT_REDUCTION) &&
          slot.isDataDependent() && slot.wiring.numInputs > 1;
      slot.flags.outputShapeDependsOnInputValues =
          tensorAxisConcat || tensorControlledView ||
          tensorControlledReduction;
    }

    // A genuinely dynamic output extent is necessarily value-dependent and needs
    // fresh shape inference/output allocation on each execution. Store that fact
    // in the canonical per-slot flag in addition to isDynamicShape; the latter also
    // propagates to downstream slots and therefore is too broad for deciding which
    // individual gap actions require full shape-aware execution.
    if (slot.hasDynamicOutputSize()) {
      slot.flags.outputShapeDependsOnInputValues = true;
      slot.markDynamicShape();
    }
  }

  plan->numExternalInputs_ = static_cast<int>(externalInputKeys.size());
  plan->externalInputNames_ = externalInputKeys;

  // ── Step 5b: Buffer aliasing — mark unary elementwise ops for in-place execution.
  // When an op is UNARY_ELEMENTWISE | FULLY_WRITING and its single op-output input
  // has only one consumer (this op), the op can write directly into its input buffer
  // instead of allocating a separate output buffer. This reduces peak memory.
  {
    // Compute consumer count per output slot
    std::vector<int> slotConsumerCount(totalOutputSlots, 0);
    for (int s = 0; s < numSteps; s++) {
      auto& sl = plan->slots_[s];
      for (int i = 0; i < sl.wiring.numInputs; i++) {
        int srcIdx = sl.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots) {
          slotConsumerCount[srcIdx]++;
        }
      }
    }

    // Build requested output slot set so we never mark in-place when the source
    // slot is a user-requested output. Without this, chains like mmul→add→sigmoid
    // all share the same buffer and the final op's values overwrite intermediates.
    std::unordered_set<int> requestedOutputSlotSet;
    for (size_t ri = 0; ri < requestedOutputs.size(); ri++) {
      auto it = varToOutputSlot.find(requestedOutputs[ri]);
      if (it != varToOutputSlot.end()) {
        requestedOutputSlotSet.insert(it->second);
      }
    }

    int aliasCount = 0;
    for (int s = 0; s < numSteps; s++) {
      auto& sl = plan->slots_[s];
      // Must be unary elementwise and fully writing
      if (!sl.hasOpTrait(sd::ops::OP_TRAIT_UNARY_ELEMENTWISE) ||
          !sl.hasOpTrait(sd::ops::OP_TRAIT_FULLY_WRITING)) continue;
      // Must have exactly 1 input
      if (sl.wiring.numInputs != 1) continue;
      // Input must come from another op's output (not external)
      int srcSlot = sl.wiring.inputSourceIndices[0];
      if (srcSlot < 0 || srcSlot >= totalOutputSlots) continue;
      // Multi-GPU: in-place aliasing reuses the INPUT slot's buffer as this op's output.
      // Output slot IDs and plan-step IDs are different domains for multi-output ops;
      // resolve the producer step before comparing device placement.
      const int producerStep = slotProducerStep[srcSlot];
      if (producerStep < 0 || producerStep >= numSteps) continue;
      const auto& sourceProducer = plan->slots_[producerStep];
      if (sl.targetDeviceId != sourceProducer.targetDeviceId) continue;
      // The in-place output becomes another publication of the source NDArray.
      // A view/identity producer can publish borrowed external storage or mint a
      // fresh wrapper on every execution, so its pointer is not stable enough to
      // become a frozen in-place output. Besides violating the frozen-slot
      // contract, treating that borrowed array as a plan-owned output can make
      // teardown release memory owned by the caller.
      if (sourceProducer.aliasesInput() || sourceProducer.frozenConstantSlot()) continue;
      // Input slot must have only this op as consumer (safe to overwrite)
      if (slotConsumerCount[srcSlot] != 1) continue;
      // Skip if this slot is already marked (e.g., from fusion pass)
      if (sl.isInPlaceFused()) continue;
      // Source slot is a user-requested output — must preserve its value
      if (requestedOutputSlotSet.count(srcSlot)) continue;
      // Skip ops that change dtype. The runtime in-place path validates dtype
      // match (line ~4130), but skipping at compile time avoids wasted attempts.
      // CAST, COMPARISON, and LOGICAL ops all produce a different dtype than input.
      if (sl.hasOpTrait(sd::ops::OP_TRAIT_CAST) ||
          sl.hasOpTrait(sd::ops::OP_TRAIT_COMPARISON) ||
          sl.hasOpTrait(sd::ops::OP_TRAIT_LOGICAL)) continue;

      // The input buffer will be reused as the output. Extend the input slot's
      // lifetime to match the output slot's lifetime so the release schedule
      // doesn't free the shared buffer while the output is still live.
      int outSlot = (sl.wiring.numOutputs > 0) ? sl.wiring.outputSlotIndices[0] : -1;
      if (outSlot >= 0 && outSlot < totalOutputSlots) {
        int outLastConsumer = slotLastConsumerStep[outSlot];
        if (outLastConsumer > slotLastConsumerStep[srcSlot]) {
          slotLastConsumerStep[srcSlot] = outLastConsumer;
        }
      }

      sl.enableInPlaceFusion(0);
      aliasCount++;
    }

    if (aliasCount > 0) {
      DSP_DIAG(COMPILE, "Buffer aliasing: marked %d unary elementwise ops for in-place execution", aliasCount);
    }

    // Post-pass: disable ALL in-place fused ops (including those set by FusionPass)
    // when their source slot is a requested output. The FusionPass doesn't have
    // access to requestedOutputs, so we must clean up here at compile time.
    // Without this, chains like mmul→add→sigmoid share one buffer and the final
    // op overwrites intermediate values that outputAll() needs to return.
    int disabledForOutput = 0;
    for (int s = 0; s < numSteps; s++) {
      auto& sl = plan->slots_[s];
      int srcSlot = sl.inPlaceSourceSlot();
      if (srcSlot < 0) continue;
      if (requestedOutputSlotSet.count(srcSlot)) {
        sl.disableInPlaceFusion();
        disabledForOutput++;
      }
    }
    if (disabledForOutput > 0) {
      DSP_DIAG(COMPILE, "Buffer aliasing: disabled %d in-place ops whose source is a requested output", disabledForOutput);
    }
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
      delete plan;
      return fail("requested output has no valid producer/output slot: " +
                  requestedOutputs[i]);
    }
  }

  // Build external input classification. externalInputIsVariable_ is the
  // replay/staging class (PLACEHOLDER inputs need refresh before graph replay).
  // externalInputIsPlaceholder_ is the lifecycle class (placeholders are not
  // protected model weights).
  plan->externalInputIsVariable_.resize(plan->numExternalInputs_, false);
  plan->externalInputIsPlaceholder_.resize(plan->numExternalInputs_, false);
  for (int s = 0; s < numSteps; s++) {
    auto& slot = plan->slots_[s];
    const bool inPlaceOnnxMha =
        slot.ident.op != nullptr && slot.ident.op->getOpName() != nullptr
            && *slot.ident.op->getOpName() == "onnx_multi_head_attention"
            && slot.wiring.numInputs >= 7;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < plan->numExternalInputs_) {
          if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            plan->externalInputIsVariable_[extIdx] = true;
            plan->externalInputIsPlaceholder_[extIdx] = true;
          }
          // Seven-input ONNX MHA writes past_key/past_value in-place when
          // cache_position is present. Those canonical external buffers are
          // both inputs and persistent state: staging them would strand the
          // captured write in a plan-owned copy and the next decode would read
          // stale KV. Classify them before the first prefill execution.
          if (inPlaceOnnxMha && (i == 4 || i == 5)) {
            plan->externalInputIsVariable_[extIdx] = true;
            plan->externalInputIsPlaceholder_[extIdx] = false;
          }
          // NOTE: SOURCE_VARIABLE inputs (trainable weights) are NOT marked
          // variable here. During inference (generation), weights are constants —
          // they never change between decode steps. Marking all 297 weights as
          // variable would cause computeSlotVariableDependency() to mark ALL
          // decoder slots as variable-dependent, preventing detectFrozenConstants()
          // from freezing shape/range ops, and causing unnecessary D2D staging
          // copies in pre-replay sync that corrupt CUDA graph baked-in addresses.
          //
          // For training (calculateGradients), the Java side calls
          // markPlanExternalInputVariable() for weights that the optimizer updates.
          // This is the correct entry point — it handles invalidation properly.
        }
      }
    }
  }

  // Compute transitive variable dependency for the frozen fast-path gate.
  plan->computeSlotVariableDependency();

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

  // ── Step 6b: Persist slot liveness data for buffer coloring ─────────────
  {
    auto* liveness = new SlotLivenessData();
    liveness->totalOutputSlots = totalOutputSlots;
    liveness->producerStep = new int[totalOutputSlots];
    liveness->lastConsumerStep = new int[totalOutputSlots];
    std::copy(slotProducerStep.begin(), slotProducerStep.end(), liveness->producerStep);
    std::copy(slotLastConsumerStep.begin(), slotLastConsumerStep.end(), liveness->lastConsumerStep);
    plan->slotLiveness_ = liveness;
    DSP_DIAG(COMPILE, "Persisted slot liveness data: %d output slots", totalOutputSlots);
  }

  // ── Step 7: Allocate execution state ──────────────────────────────────────
  plan->outputSlots_ = new NDArray*[totalOutputSlots];
  std::memset(plan->outputSlots_, 0, sizeof(NDArray*) * totalOutputSlots);

  // slotIsViewProducer_ replaced by slots_[i].slotPhase.isViewProducer (value-initialized with slots_)

  plan->slotOwnership_ = new SlotBufferInfo[totalOutputSlots]();  // value-initialized to UNSET

  plan->contextPool_ = new Context*[numSteps];
  for (int i = 0; i < numSteps; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // ── Control flow detection and loop region setup ────────────────────────
  plan->hasControlFlow_ = false;
  plan->loopRegions_ = nullptr;
  plan->numLoopRegions_ = 0;
  plan->cfLoopBackStep_ = -1;
  for (int s = 0; s < numSteps; s++) {
    if (plan->slots_[s].cf.controlFlowType != CF_NONE) {
      plan->hasControlFlow_ = true;
      break;
    }
  }

  // Allocate dead-slot tracking for control flow
  plan->slotIsDeadSize_ = totalOutputSlots;
  plan->slotIsDead_ = new bool[plan->slotIsDeadSize_];
  std::memset(plan->slotIsDead_, 0, sizeof(bool) * plan->slotIsDeadSize_);

  if (plan->hasControlFlow_) {
    DSP_DIAG(COMPILE, "control flow detected in FlatGraph-compiled plan");

    // Build loop regions: find NextIteration slots that loop back to Merge slots.
    // For each NextIteration, find the Merge it targets by scanning backward.
    std::vector<LoopRegion> regions;
    for (int s = 0; s < numSteps; s++) {
      NativeSlot& slot = plan->slots_[s];
      if (slot.cf.controlFlowType == CF_NEXT_ITERATION) {
        // Find the Merge this NextIteration feeds: look at output wiring.
        // NextIteration output feeds into a Merge's input. Find the Merge by
        // scanning for a Merge whose inputSourceIndices references our output slot.
        int nextIterOutputSlot = (slot.wiring.numOutputs > 0) ? slot.wiring.outputSlotIndices[0] : -1;
        int mergeSlotIdx = -1;
        if (nextIterOutputSlot >= 0) {
          for (int m = 0; m < numSteps; m++) {
            if (plan->slots_[m].cf.controlFlowType == CF_MERGE) {
              for (int inp = 0; inp < plan->slots_[m].wiring.numInputs; inp++) {
                if (plan->slots_[m].wiring.inputSourceIndices[inp] == nextIterOutputSlot) {
                  mergeSlotIdx = m;
                  break;
                }
              }
              if (mergeSlotIdx >= 0) break;
            }
          }
        }

        if (mergeSlotIdx >= 0) {
          slot.cf.loopBackTarget = mergeSlotIdx;
          slot.cf.loopRegionIndex = static_cast<int>(regions.size());

          LoopRegion lr;
          lr.mergeSlot = mergeSlotIdx;
          lr.nextIterSlot = s;
          lr.bodyStartSlot = mergeSlotIdx + 1;
          lr.bodyEndSlot = s;
          // Find Switch and Exit in this region
          lr.switchSlot = -1;
          lr.exitSlot = -1;
          for (int r = mergeSlotIdx; r <= s; r++) {
            if (plan->slots_[r].cf.controlFlowType == CF_SWITCH && lr.switchSlot < 0)
              lr.switchSlot = r;
            if (plan->slots_[r].cf.controlFlowType == CF_EXIT && lr.exitSlot < 0)
              lr.exitSlot = r;
          }
          regions.push_back(lr);
        }
      }
    }

    if (!regions.empty()) {
      plan->numLoopRegions_ = static_cast<int>(regions.size());
      plan->loopRegions_ = new LoopRegion[plan->numLoopRegions_];
      std::copy(regions.begin(), regions.end(), plan->loopRegions_);
      DSP_DIAG(COMPILE, "built %d loop region(s)", plan->numLoopRegions_);
    }
  }

  // ── Shape static analysis (same as fromSerializedPlan) ──────────────────
  {
    std::vector<int> outputSlotToStepIndex(totalOutputSlots, -1);
    for (int s = 0; s < numSteps; s++) {
      NativeSlot& slot = plan->slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < totalOutputSlots) {
          outputSlotToStepIndex[si] = s;
        }
      }
    }

    int staticCount = 0, dynamicCount = 0;
    for (int s = 0; s < numSteps; s++) {
      NativeSlot& slot = plan->slots_[s];
      slot.shapeCache.shapeStatic = true;

      if (slot.hasValueDependentShape()) {
        slot.shapeCache.shapeStatic = false;
        dynamicCount++;
        continue;
      }

      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            slot.shapeCache.shapeStatic = false;
            break;
          }
        } else {
          if (srcIdx < totalOutputSlots) {
            int producerStep = outputSlotToStepIndex[srcIdx];
            if (producerStep >= 0 && !plan->slots_[producerStep].shapeCache.shapeStatic) {
              slot.shapeCache.shapeStatic = false;
              break;
            }
          }
        }
      }

      if (slot.shapeCache.shapeStatic) staticCount++;
      else dynamicCount++;
    }

    DSP_DIAG(SHAPE, "shape analysis: %d static, %d dynamic out of %d slots",
              staticCount, dynamicCount, numSteps);
  }

  // Build CUDA graph segments
  plan->buildSegments();

  {
    const auto request = plan->makeGraphBackendRequest();
    const auto calibrationOutputs =
        dsp::collectPrecommitCalibrationOutputSlots(
            request, plan->getGraphBackendCandidates(), plan->slots_,
            plan->numSlots_, plan->totalOutputSlots_);
    const int disabledForCalibration =
        dsp::disableInPlaceConsumersOfSlots(
            plan->slots_, plan->numSlots_, calibrationOutputs);
    if (disabledForCalibration > 0) {
      DSP_DIAG(
          FUSION,
          "compiler calibration: preserved %zu backend outputs by disabling "
          "%d in-place consumers",
          calibrationOutputs.size(), disabledForCalibration);
    }
  }

  // ── Build shared immutable PlanDefinition ───────────────────────────────
  {
    auto builder = PlanDefinition::Builder();
    builder.setNumSlots(plan->numSlots_)
           .setTotalOutputSlots(plan->totalOutputSlots_)
           .setNumExternalInputs(plan->numExternalInputs_)
           .setNumRequestedOutputs(plan->numRequestedOutputs_)
           .setRequestedOutputSlotIndices(plan->requestedOutputSlotIndices_,
                                          plan->numRequestedOutputs_)
           .setExternalInputNames(plan->externalInputNames_)
           .setExternalInputIsVariable(plan->externalInputIsVariable_)
           .setHasControlFlow(plan->hasControlFlow_)
           .setNumLoopRegions(plan->numLoopRegions_)
           .setBackendPriority(plan->backendPriority_);
    plan->planDef_ = builder.build();
    DSP_DIAG(COMPILE, "PlanDefinition created: %d slots, %d outputs, %d ext inputs, refCount=%d",
             plan->planDef_->numSlots(), plan->planDef_->totalOutputSlots(),
             plan->planDef_->numExternalInputs(), plan->planDef_->refCount());
  }

  // ── Create per-instance ExecutionState ──────────────────────────────────
  plan->execState_ = new ExecutionState(plan->totalOutputSlots_);
  DSP_DIAG(COMPILE, "ExecutionState created: %d output slots", plan->totalOutputSlots_);

  // Diagnostic: count how many slots need zeroed output and compilation summary
  if (DSP_DIAG_ENABLED(COMPILE) || DSP_DIAG_ENABLED(SHAPE)) {
    int needsZero = 0, skipZero = 0, customOps = 0, cfOps = 0;
    int dataDep = 0, valueDep = 0, identityOps = 0, fusedChains = 0;
    for (int i = 0; i < plan->numSlots_; i++) {
      if (plan->slots_[i].needsZeroedOutput()) needsZero++;
      else skipZero++;
      if (plan->slots_[i].flags.isCustomOp) customOps++;
      if (plan->slots_[i].cf.controlFlowType != CF_NONE) cfOps++;
      if (plan->slots_[i].isDataDependent()) dataDep++;
      if (plan->slots_[i].flags.outputShapeDependsOnInputValues) valueDep++;
      if (plan->slots_[i].isIdentityOp()) identityOps++;
      if (plan->slots_[i].fusedChain.isFusedChainHead) fusedChains++;
    }
    DSP_DIAG(SHAPE, "%d/%d slots need zeroed output (%d can skip nullify)",
              needsZero, plan->numSlots_, skipZero);
    DSP_DIAG(COMPILE, "NativePlanCompiler::compile: DONE %d slots, %d outputSlots, %d ext inputs, "
             "%d segments, %d custom, %d CF, %d dataDep, %d valueDep, %d identity, %d fusedChains",
             plan->numSlots_, plan->totalOutputSlots_, plan->numExternalInputs_,
             (int)plan->segments_.size(), customOps, cfOps, dataDep, valueDep,
             identityOps, fusedChains);
  }

  return plan;
}

}  // namespace graph
}  // namespace sd
