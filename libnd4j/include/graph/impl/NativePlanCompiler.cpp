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
#include <graph/PlanDefinition.h>
#include <graph/ExecutionState.h>
#include <graph/DspDiagnostics.h>
#include <graph/generated/graph_generated.h>
#include <helpers/helper_hash.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/OpTraitTable.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <queue>
#include <unordered_set>

// Bring in FlatBuffer-generated types from the ::graph namespace
using namespace ::graph;

namespace sd {
namespace graph {

namespace {
std::string normalizeOpName(const std::string& opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}
}  // namespace

// Op classification is driven by OpDescriptor traits (see OpTraitTable.h).
// For ops not in OpRegistrator (e.g., control flow), fall back to name lookup.

static bool hasOpTrait(sd::ops::DeclarableOp* op, uint32_t trait) {
  if (op && op->getOpDescriptor()) {
    return op->getOpDescriptor()->hasAnyTrait(trait);
  }
  return false;
}

// ─── Compile FlatGraph → NativeDynamicShapePlan ─────────────────────────────

NativeDynamicShapePlan* NativePlanCompiler::compile(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {

  // Ensure all op traits are populated from the centralized table
  sd::ops::initOpTraits();

  if (!graph) return nullptr;

  auto* nodes = graph->nodes();
  auto* flatVars = graph->variables();
  if (!nodes || nodes->size() == 0) return nullptr;

  DSP_DIAG(COMPILE, "NativePlanCompiler::compile: ENTER nodes=%d vars=%d requestedOutputs=%d",
           (int)nodes->size(), flatVars ? (int)flatVars->size() : 0, (int)requestedOutputs.size());

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
  plan->dirtySlotGenerations_.resize(totalOutputSlots, 0);
  plan->numExternalInputs_ = static_cast<int>(externalInputKeys.size());
  plan->externalInputNames_ = externalInputKeys;
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
    slot.flags.isCustomOp = (node->opType() == OpType_CUSTOM);

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

    // Resolve op — skip for control flow ops (they're handled by CF dispatch,
    // not as declarable ops, and may not be registered in OpRegistrator).
    slot.ident.op = nullptr;
    if (slot.cf.controlFlowType == CF_NONE) {
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
        return nullptr;
      }
      slot.ident.opHash = sd::ops::HashHelper::getInstance().getLongHash(slot.ident.opName);
    } else {
      // CF ops don't need an op pointer or hash — they're pure data routing.
      slot.ident.opHash = 0;
      DSP_DIAG(COMPILE, "slot %d: control flow op '%s' (type=%d)",
                stepIdx, slot.ident.opName.c_str(), (int)slot.cf.controlFlowType);
    }

    slot.flags.inPlaceFused = false;
    slot.flags.inPlaceFusedInputIdx = -1;
    slot.fusedChain.isFusedChainHead = false;
    slot.fusedChain.fusedChainLength = 0;
    slot.fusedChain.isFusedChainTail = false;
    std::memset(slot.fusedChain.fusedChainOpCodes, 0, sizeof(slot.fusedChain.fusedChainOpCodes));
    std::memset(slot.fusedChain.fusedChainSlots, 0, sizeof(slot.fusedChain.fusedChainSlots));
    std::fill(std::begin(slot.fusedChain.fusedChainSecondaryInputSources), std::end(slot.fusedChain.fusedChainSecondaryInputSources), INT32_MIN);

    // Build input wiring from inputPaired
    auto* inputPaired = node->inputPaired();
    int numInputs = inputPaired ? inputPaired->size() : 0;

    // Classify op via OpDescriptor traits (set by OpTraitTable at init time).
    // Where with 3 inputs (condition, x, y) is element-wise select with static output shape.
    // Only Where with 1 input (coordinate extraction) has data-dependent variable-length output.
    bool isDataDep = hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_DATA_DEPENDENT);
    if (isDataDep && normalizeOpName(slot.ident.opName) == "where" && numInputs == 3) {
      isDataDep = false;
    }
    slot.flags.isDataDependent = isDataDep;
    slot.flags.isIdentityOp = hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_IDENTITY);
    slot.flags.isViewCapableOp = hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_VIEW_PRODUCING);
    // Output-zeroing classification (computed once, read at runtime).
    bool aliasesInput  = slot.flags.isViewCapableOp || slot.flags.isIdentityOp;
    bool fullyWrites   = hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_FULLY_WRITING);
    // scatter_nd is a true partial writer; scatter_nd_update has FULLY_WRITING
    // because it does output->assign(input). Only suppress for partial without FULLY_WRITING.
    bool partialWriter = (hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_SCATTER_ND) ||
                          hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_SCATTER_ND_UPDATE))
                         && !fullyWrites;
    slot.flags.isFullyWriting    = fullyWrites && !slot.flags.isDataDependent && !partialWriter;
    slot.flags.needsZeroedOutput = !aliasesInput && !slot.flags.isFullyWriting;
    slot.wiring.numInputs = numInputs;
    slot.wiring.inputSourceIndices = new int[numInputs];
    slot.wiring.inputSourceTypes = new int8_t[numInputs];

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
          // Check FlatVariable dtype for INT/LONG detection
          if (!hasIntLong) {
            auto dt = fv->dtype();
            if (dt == DType_INT32 || dt == DType_INT64) {
              hasIntLong = true;
            }
          }
          found = true;
        }
      }

      // Look up in output slot map or external
      auto slotIt = varToOutputSlot.find(inputName);
      // If not found by exact name, try with baseName (strip :N suffix) — mirrors Java compiler logic
      if (slotIt == varToOutputSlot.end()) {
        auto colonPos = inputName.rfind(':');
        if (colonPos != std::string::npos) {
          std::string baseName = inputName.substr(0, colonPos);
          slotIt = varToOutputSlot.find(baseName);
        }
      }
      // Guard against self-reference: if resolved slot is one of THIS step's own outputs, treat as external
      bool isSelfRef = false;
      if (slotIt != varToOutputSlot.end()) {
        auto* myOutputNames = node->outputNames();
        if (myOutputNames) {
          for (unsigned int oi = 0; oi < myOutputNames->size(); oi++) {
            std::string myOut = myOutputNames->Get(oi)->str();
            auto myIt = varToOutputSlot.find(myOut);
            if (myIt != varToOutputSlot.end() && myIt->second == slotIt->second) {
              isSelfRef = true;
              DSP_DIAG(COMPILE, "NativePlanCompiler: self-reference at step %d (op %s): input '%s' -> slot %d == own output '%s' slot %d",
                       stepIdx, slot.ident.opName.c_str(), inputName.c_str(), slotIt->second, myOut.c_str(), myIt->second);
              break;
            }
          }
        }
        // Also check: if the input resolves to a slot produced by THIS step (same stepIdx), it's self-referential
        if (!isSelfRef) {
          int resolvedSlot = slotIt->second;
          if (resolvedSlot >= 0 && resolvedSlot < (int)slotProducerStep.size() && slotProducerStep[resolvedSlot] == stepIdx) {
            isSelfRef = true;
            DSP_DIAG(COMPILE, "NativePlanCompiler: self-reference (producer match) at step %d (op %s): input '%s' -> slot %d produced at same step",
                     stepIdx, slot.ident.opName.c_str(), inputName.c_str(), resolvedSlot);
          }
        }
      }
      if (slotIt != varToOutputSlot.end() && !isSelfRef) {
        slot.wiring.inputSourceIndices[i] = slotIt->second;
        slot.wiring.inputSourceTypes[i] = SOURCE_OP_OUTPUT;
        if (stepIdx > slotLastConsumerStep[slotIt->second]) {
          slotLastConsumerStep[slotIt->second] = stepIdx;
        }
      } else {
        int extIdx = addExternal(inputName);
        slot.wiring.inputSourceIndices[i] = -(extIdx + 1);
        if (constants.count(inputName)) {
          slot.wiring.inputSourceTypes[i] = SOURCE_CONSTANT;
        } else if (variableNames.count(inputName)) {
          slot.wiring.inputSourceTypes[i] = SOURCE_VARIABLE;
        } else {
          slot.wiring.inputSourceTypes[i] = SOURCE_PLACEHOLDER;
        }
        // Check external input NDArray dtype for INT/LONG detection
        if (!hasIntLong) {
          auto varIt2 = variables.find(inputName);
          if (varIt2 != variables.end() && varIt2->second != nullptr) {
            auto dt = varIt2->second->dataType();
            if (dt == INT32 || dt == INT64) {
              hasIntLong = true;
            }
          }
        }
      }
    }
    slot.flags.needsIntLongSync = hasIntLong || isDataDep;
    // VALUE_DEPENDENT_SHAPE trait: ops whose output shapes depend on input VALUES
    // (not just shapes). Drives shapeStatic=false and shape key value-hashing.
    bool isValueDepShape = hasOpTrait(slot.ident.op, sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE);
    slot.flags.outputShapeDependsOnInputValues = isValueDepShape || isDataDep;

    // Build output wiring
    auto* outputNames = node->outputNames();
    int numOutputs = outputNames ? outputNames->size() : 1;
    slot.wiring.numOutputs = numOutputs;
    slot.wiring.outputSlotIndices = new int[numOutputs];

    for (int i = 0; i < numOutputs; i++) {
      std::string outName;
      if (outputNames && i < static_cast<int>(outputNames->size())) {
        outName = outputNames->Get(i)->str();
      } else {
        outName = node->name() ? node->name()->str() : ("node_" + std::to_string(node->id()));
      }
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
        // Set structural iArg count from centralized OpTraitTable
        slot.flags.structuralIArgCount = sd::ops::getStructuralIArgCount(slot.ident.opName);

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
    if (extraParams && extraParams->size() > 0) {
      slot.args.numTArgs = extraParams->size();
      slot.args.tArgs = new double[slot.args.numTArgs];
      for (int i = 0; i < slot.args.numTArgs; i++) {
        slot.args.tArgs[i] = extraParams->Get(i);
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

  // Build externalInputIsVariable_: only PLACEHOLDER inputs need forced H2D
  // before graph replay (dynamic inputs that Java updates each step).
  // SOURCE_VARIABLE (model weights) are device-authoritative after initial load.
  plan->externalInputIsVariable_.resize(plan->numExternalInputs_, false);
  for (int s = 0; s < numSteps; s++) {
    auto& slot = plan->slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < plan->numExternalInputs_ &&
            slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
          plan->externalInputIsVariable_[extIdx] = true;
        }
      }
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

  plan->slotIsViewProducer_ = new bool[totalOutputSlots];
  std::memset(plan->slotIsViewProducer_, 0, sizeof(bool) * totalOutputSlots);

  plan->slotOwnership_ = new SlotBufferInfo[totalOutputSlots]();  // value-initialized to UNSET

  plan->contextPool_ = new Context*[numSteps];
  for (int i = 0; i < numSteps; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // ── Control flow detection and loop region setup ────────────────────────
  plan->hasControlFlow_ = false;
  plan->loopRegions_ = nullptr;
  plan->numLoopRegions_ = 0;
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

      if (slot.flags.isDataDependent || slot.flags.outputShapeDependsOnInputValues) {
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
      if (plan->slots_[i].flags.needsZeroedOutput) needsZero++;
      else skipZero++;
      if (plan->slots_[i].flags.isCustomOp) customOps++;
      if (plan->slots_[i].cf.controlFlowType != CF_NONE) cfOps++;
      if (plan->slots_[i].flags.isDataDependent) dataDep++;
      if (plan->slots_[i].flags.outputShapeDependsOnInputValues) valueDep++;
      if (plan->slots_[i].flags.isIdentityOp) identityOps++;
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
