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

//
// TritonIRBuilder — Segment analysis pipeline:
//   profileSegment, matchPatterns, classifyAndAnalyze, analyzeSegment,
//   classifySegment, selectTileConfig
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/DspDiagnostics.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/shape.h>
#include <system/Environment.h>

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

using namespace ir_builder_internal;

// Maximum number of direct function arguments before switching to indirect
// argument passing via a pointer array.
static constexpr int TRITON_DIRECT_ARG_LIMIT = 200;

// ─── Pass 1: Segment Profiling ──────────────────────────────────────────────

SegmentProfile TritonIRBuilder::profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                                NDArray** outputSlots, int totalOutputSlots) {
  SegmentProfile profile;
  int segSize = endSlot - startSlot + 1;
  profile.totalOps = segSize;
  profile.nodes.resize(segSize);

  // Build slotIndex → localIndex map and outputSlot → producer local index map
  std::unordered_map<int, int> slotToLocal;
  std::unordered_map<int, int> outputSlotToProducer;  // output slot idx → local index that produces it

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    slotToLocal[absSlot] = i;

    auto& node = profile.nodes[i];
    node.slotIndex = absSlot;
    node.localIndex = i;
    node.opName = slots[absSlot].opName;
    node.category = getOpCategory(slots[absSlot].opName);
    node.hasExternalInput = false;

    // Register outputs produced by this node
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSlotToProducer[slots[absSlot].outputSlotIndices[o]] = i;
    }

    // Populate output shape from DSP's pre-calculated cache or live outputSlots
    if (slots[absSlot].numOutputs > 0) {
      int outIdx = slots[absSlot].outputSlotIndices[0];

      // Priority 1: Use NativeSlot's cached shape info (pre-calculated by DSP)
      if (slots[absSlot].shapeCacheValid() && !slots[absSlot].cachedOutputShapes.empty()) {
        const LongType* shapeInfo = slots[absSlot].cachedOutputShapes[0];
        if (shapeInfo) {
          LongType rank = shape::rank(shapeInfo);
          node.outputShape.resize(rank);
          for (int d = 0; d < rank; d++) {
            node.outputShape[d] = shapeInfo[d + 1];
          }
          node.hasOutputShape = true;
        }
      }

      // Priority 2: Fall back to live outputSlots array
      if (!node.hasOutputShape && outputSlots && outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        auto& arr = *outputSlots[outIdx];
        node.outputShape.resize(arr.rankOf());
        for (int d = 0; d < arr.rankOf(); d++) {
          node.outputShape[d] = arr.sizeAt(d);
        }
        node.outputDtype = arr.dataType();
        node.hasOutputShape = true;
      }
    }

    // Count categories
    int catIdx = static_cast<int>(node.category);
    if (catIdx >= 0 && catIdx < 16) profile.categoryCounts[catIdx]++;
  }

  // Build dataflow edges and consumer lists
  std::unordered_set<int> externalInputSet;
  std::unordered_map<int, std::vector<int>> outputToConsumers;  // output slot → list of consuming local indices

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    auto& node = profile.nodes[i];

    for (int inp = 0; inp < slots[absSlot].numInputs; inp++) {
      int srcIdx = slots[absSlot].inputSourceIndices[inp];

      if (srcIdx < 0) {
        // External input
        node.inputLocalIndices.push_back(-1);
        node.hasExternalInput = true;
        externalInputSet.insert(srcIdx);
      } else {
        // Check if this source is produced within the segment
        auto producerIt = outputSlotToProducer.find(srcIdx);
        if (producerIt != outputSlotToProducer.end()) {
          int producerLocal = producerIt->second;
          node.inputLocalIndices.push_back(producerLocal);
          outputToConsumers[srcIdx].push_back(i);
        } else {
          // Pre-segment output — treat as external
          node.inputLocalIndices.push_back(-1);
          node.hasExternalInput = true;
          externalInputSet.insert(srcIdx);
        }
      }
    }
  }

  // Fill consumer lists
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      int outIdx = slots[absSlot].outputSlotIndices[o];
      auto it = outputToConsumers.find(outIdx);
      if (it != outputToConsumers.end()) {
        for (int consumer : it->second) {
          profile.nodes[i].consumerLocalIndices.push_back(consumer);
        }
      }
    }
  }

  // Count unique outputs (produced within segment)
  std::unordered_set<int> outputSet;
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSet.insert(slots[absSlot].outputSlotIndices[o]);
    }
  }

  profile.numUniqueExternalInputs = static_cast<int>(externalInputSet.size());
  profile.numUniqueOutputs = static_cast<int>(outputSet.size());

  // Set summary flags from category counts
  profile.hasMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)] > 0;
  profile.hasReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)] > 0;
  profile.hasNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)] > 0;
  profile.hasFusedAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)] > 0;
  profile.hasShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)] > 0;
  profile.hasDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)] > 0;

  return profile;
}

// ─── Pass 2: Pattern Detection ──────────────────────────────────────────────

namespace {

// --- Concrete pattern detectors (file-local) ---

class FusedAttentionOpDetector : public PatternDetector {
 public:
  const char* name() const override { return "FusedAttentionOp"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasFusedAttention) return results;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::FUSED_ATTENTION) {
        PatternMatch m;
        m.type = PatternMatch::FUSED_ATTENTION_OP;
        m.priority = 100;
        m.localIndices.push_back(node.localIndex);
        m.description = "fused attention op at slot " + std::to_string(node.slotIndex);
        results.push_back(m);
      }
    }
    return results;
  }
};

class AttentionPatternDetector : public PatternDetector {
 public:
  const char* name() const override { return "AttentionQKV"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    int matmulCount = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
    if (matmulCount < 2 || (!profile.hasNormalization && !profile.hasReduction)) return results;

    // Find pairs: matmul → (elementwise chain) → softmax/reduction → (elementwise chain) → matmul
    std::vector<int> matmulLocals;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::MATMUL) {
        matmulLocals.push_back(node.localIndex);
      }
    }

    for (size_t mi = 0; mi + 1 < matmulLocals.size(); mi++) {
      int firstMatmul = matmulLocals[mi];
      int secondMatmul = matmulLocals[mi + 1];
      // Check for softmax/reduction between the two matmuls
      bool hasSoftmaxBetween = false;
      for (int j = firstMatmul + 1; j < secondMatmul; j++) {
        auto cat = profile.nodes[j].category;
        if (cat == TritonOpCategory::NORMALIZATION || cat == TritonOpCategory::REDUCTION) {
          hasSoftmaxBetween = true;
          break;
        }
      }
      if (hasSoftmaxBetween) {
        PatternMatch m;
        m.type = PatternMatch::ATTENTION_QKV;
        m.priority = 90;
        for (int j = firstMatmul; j <= secondMatmul; j++) {
          m.localIndices.push_back(j);
        }
        m.description = "attention pattern: matmul[" + std::to_string(profile.nodes[firstMatmul].slotIndex) +
                         "] → softmax → matmul[" + std::to_string(profile.nodes[secondMatmul].slotIndex) + "]";
        results.push_back(m);
      }
    }
    return results;
  }
};

class FFNBlockDetector : public PatternDetector {
 public:
  const char* name() const override { return "FFNBlock"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    int matmulCount = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
    if (matmulCount < 2) return results;

    // Find pairs: matmul → activation (elementwise) → matmul (with no reduction/norm between)
    std::vector<int> matmulLocals;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::MATMUL) {
        matmulLocals.push_back(node.localIndex);
      }
    }

    for (size_t mi = 0; mi + 1 < matmulLocals.size(); mi++) {
      int firstMatmul = matmulLocals[mi];
      int secondMatmul = matmulLocals[mi + 1];
      bool hasActivation = false;
      bool hasHeavyweight = false;
      for (int j = firstMatmul + 1; j < secondMatmul; j++) {
        auto cat = profile.nodes[j].category;
        if (TritonIRBuilder::isElementwiseCompatible(cat)) hasActivation = true;
        if (cat == TritonOpCategory::REDUCTION || cat == TritonOpCategory::NORMALIZATION) {
          hasHeavyweight = true;
        }
      }
      if (hasActivation && !hasHeavyweight) {
        PatternMatch m;
        m.type = PatternMatch::FFN_BLOCK;
        m.priority = 85;
        for (int j = firstMatmul; j <= secondMatmul; j++) {
          m.localIndices.push_back(j);
        }
        m.description = "FFN block: matmul[" + std::to_string(profile.nodes[firstMatmul].slotIndex) +
                         "] → activation → matmul[" + std::to_string(profile.nodes[secondMatmul].slotIndex) + "]";
        results.push_back(m);
      }
    }
    return results;
  }
};

class DecomposedSoftmaxDetector : public PatternDetector {
 public:
  const char* name() const override { return "DecomposedSoftmax"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasReduction) return results;

    // Mirror FusionPass Pass 6: reduce_max → sub → exp → reduce_sum → div
    for (int i = 0; i < profile.totalOps; i++) {
      if (profile.nodes[i].opName != "reduce_max" && profile.nodes[i].opName != "ReduceMax") continue;

      int absI = startSlot + i;
      if (slots[absI].numOutputs != 1) continue;
      int out0 = slots[absI].outputSlotIndices[0];

      // Find sub consuming reduce_max output
      int subLocal = -1;
      for (int j = i + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "subtract" && name != "Sub") continue;
        int absJ = startSlot + j;
        for (int k = 0; k < slots[absJ].numInputs; k++) {
          if (slots[absJ].inputSourceIndices[k] == out0) { subLocal = j; break; }
        }
        if (subLocal >= 0) break;
      }
      if (subLocal < 0) continue;

      int outSub = slots[startSlot + subLocal].outputSlotIndices[0];
      // Find exp consuming sub output
      int expLocal = -1;
      for (int j = subLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "exp" && name != "Exp") continue;
        int absJ = startSlot + j;
        if (slots[absJ].numInputs >= 1 && slots[absJ].inputSourceIndices[0] == outSub) {
          expLocal = j; break;
        }
      }
      if (expLocal < 0) continue;

      int outExp = slots[startSlot + expLocal].outputSlotIndices[0];
      // Find reduce_sum consuming exp output
      int sumLocal = -1;
      for (int j = expLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "reduce_sum" && name != "ReduceSum") continue;
        int absJ = startSlot + j;
        if (slots[absJ].numInputs >= 1 && slots[absJ].inputSourceIndices[0] == outExp) {
          sumLocal = j; break;
        }
      }
      if (sumLocal < 0) continue;

      int outSum = slots[startSlot + sumLocal].outputSlotIndices[0];
      // Find div consuming exp and sum outputs
      int divLocal = -1;
      for (int j = sumLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "divide" && name != "Div" && name != "RealDiv") continue;
        int absJ = startSlot + j;
        bool hasExp = false, hasSum = false;
        for (int k = 0; k < slots[absJ].numInputs; k++) {
          if (slots[absJ].inputSourceIndices[k] == outExp) hasExp = true;
          if (slots[absJ].inputSourceIndices[k] == outSum) hasSum = true;
        }
        if (hasExp && hasSum) { divLocal = j; break; }
      }
      if (divLocal < 0) continue;

      PatternMatch m;
      m.type = PatternMatch::SOFTMAX_DECOMPOSED;
      m.priority = 80;
      m.localIndices = {i, subLocal, expLocal, sumLocal, divLocal};
      m.description = "decomposed softmax: reduce_max[" + std::to_string(startSlot + i) +
                       "] → sub → exp → reduce_sum → div[" + std::to_string(startSlot + divLocal) + "]";
      results.push_back(m);
    }
    return results;
  }
};

class MatmulEpilogueDetector : public PatternDetector {
 public:
  const char* name() const override { return "MatmulEpilogue"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasMatmul) return results;

    for (int i = 0; i < profile.totalOps; i++) {
      if (profile.nodes[i].category != TritonOpCategory::MATMUL) continue;
      // BFS forward through elementwise-compatible consumers
      std::vector<int> epilogueOps = {i};
      for (int j = i + 1; j < profile.totalOps; j++) {
        if (profile.nodes[j].category == TritonOpCategory::MATMUL) break;
        if (TritonIRBuilder::isElementwiseCompatible(profile.nodes[j].category)) {
          epilogueOps.push_back(j);
        } else {
          break;  // Stop at non-elementwise
        }
      }
      if (epilogueOps.size() > 1) {
        PatternMatch m;
        m.type = PatternMatch::MATMUL_EPILOGUE;
        m.priority = 70;
        m.localIndices = epilogueOps;
        m.description = "matmul epilogue: matmul[" + std::to_string(startSlot + i) +
                         "] + " + std::to_string(epilogueOps.size() - 1) + " elementwise ops";
        results.push_back(m);
      }
    }
    return results;
  }
};

class ElementwisePatternDetector : public PatternDetector {
 public:
  const char* name() const override { return "PureElementwise"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    bool allElementwise = true;
    for (auto& node : profile.nodes) {
      if (!TritonIRBuilder::isElementwiseCompatible(node.category)) {
        allElementwise = false;
        break;
      }
    }
    if (allElementwise && profile.totalOps > 0) {
      PatternMatch m;
      m.type = PatternMatch::PURE_ELEMENTWISE;
      m.priority = 10;
      for (int i = 0; i < profile.totalOps; i++) m.localIndices.push_back(i);
      m.description = "pure elementwise chain (" + std::to_string(profile.totalOps) + " ops)";
      results.push_back(m);
    }
    return results;
  }
};

class MegaSegmentDetector : public PatternDetector {
 public:
  const char* name() const override { return "MegaSegment"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (profile.totalOps <= 50) return results;

    // Count heavyweight categories present
    int heavyweightCount = 0;
    if (profile.hasMatmul) heavyweightCount++;
    if (profile.hasReduction) heavyweightCount++;
    if (profile.hasNormalization) heavyweightCount++;
    if (profile.hasFusedAttention) heavyweightCount++;
    if (profile.hasShapeManip) heavyweightCount++;
    if (profile.hasDataMovement) heavyweightCount++;

    if (heavyweightCount >= 2) {
      PatternMatch m;
      m.type = PatternMatch::MIXED_MEGA_SEGMENT;
      m.priority = 5;
      for (int i = 0; i < profile.totalOps; i++) m.localIndices.push_back(i);
      m.description = "mixed mega-segment (" + std::to_string(profile.totalOps) + " ops, " +
                       std::to_string(heavyweightCount) + " heavyweight categories)";
      results.push_back(m);
    }
    return results;
  }
};

// Registry of pattern detectors.
// NOTE: libnd4j is compiled with -fno-threadsafe-statics, so function-local
// static initialization is not synchronized. Keep this registry as a fixed
// global object with immutable detector pointers to avoid racey lazy init.
class PatternRegistry {
 public:
  PatternRegistry()
      : detectors_{&fusedAttentionOpDetector_,
                   &attentionPatternDetector_,
                   &ffnBlockDetector_,
                   &decomposedSoftmaxDetector_,
                   &matmulEpilogueDetector_,
                   &elementwisePatternDetector_,
                   &megaSegmentDetector_} {}

  const std::array<PatternDetector*, 7>& detectors() const { return detectors_; }

 private:
  FusedAttentionOpDetector fusedAttentionOpDetector_;
  AttentionPatternDetector attentionPatternDetector_;
  FFNBlockDetector ffnBlockDetector_;
  DecomposedSoftmaxDetector decomposedSoftmaxDetector_;
  MatmulEpilogueDetector matmulEpilogueDetector_;
  ElementwisePatternDetector elementwisePatternDetector_;
  MegaSegmentDetector megaSegmentDetector_;
  std::array<PatternDetector*, 7> detectors_;
};

PatternRegistry gPatternRegistry;

}  // anonymous namespace

MatchedPatterns TritonIRBuilder::matchPatterns(const SegmentProfile& profile,
                                                NativeSlot* slots, int startSlot) {
  MatchedPatterns matched;
  const auto& detectors = gPatternRegistry.detectors();
  auto& env = Environment::getInstance();
  const bool collectAllMatches = env.tritonLogAllPatterns() || env.tritonVerbose();
  constexpr int MAX_PATTERN_PRIORITY = 100;  // FUSED_ATTENTION_OP

  int bestPriority = std::numeric_limits<int>::min();
  for (auto* detector : detectors) {
    auto hits = detector->detect(profile, slots, startSlot);

    if (collectAllMatches) {
      for (auto& hit : hits) {
        if (hit.priority > bestPriority) bestPriority = hit.priority;
        matched.matches.push_back(std::move(hit));
      }
    } else {
      for (const auto& hit : hits) {
        if (hit.priority > bestPriority) {
          bestPriority = hit.priority;
          matched.matches.clear();
          matched.matches.push_back(hit);
        }
      }
      // Cannot beat max priority; skip lower-priority detector passes.
      if (bestPriority >= MAX_PATTERN_PRIORITY) break;
    }
  }

  if (collectAllMatches) {
    // Sort by priority (descending)
    std::sort(matched.matches.begin(), matched.matches.end(),
              [](const PatternMatch& a, const PatternMatch& b) { return a.priority > b.priority; });
  }

  return matched;
}

// ─── Pass 3: Classify and Analyze ───────────────────────────────────────────

SegmentAnalysis TritonIRBuilder::classifyAndAnalyze(const SegmentProfile& profile,
                                                     const MatchedPatterns& patterns,
                                                     NativeSlot* slots, int startSlot, int endSlot,
                                                     int totalSlots,
                                                     NDArray** externalInputs, int numExternalInputs,
                                                     NDArray** outputSlots, int totalOutputSlots,
                                                     int* requestedOutputSlotIndices,
                                                     int numRequestedOutputs) {
  SegmentAnalysis analysis;

  // Fill category counts from profile
  analysis.numElementwise = profile.categoryCounts[static_cast<int>(TritonOpCategory::BINARY_ELEMENTWISE)] +
                             profile.categoryCounts[static_cast<int>(TritonOpCategory::UNARY_ELEMENTWISE)];
  analysis.numMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
  analysis.numReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)];
  analysis.numNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)];
  analysis.numAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)];
  analysis.numShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)];
  analysis.numDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)];
  analysis.numConstGen = profile.categoryCounts[static_cast<int>(TritonOpCategory::CONSTANT_GENERATION)];
  analysis.numIdentity = profile.categoryCounts[static_cast<int>(TritonOpCategory::IDENTITY)];
  analysis.numCast = profile.categoryCounts[static_cast<int>(TritonOpCategory::CAST)];

  // Count unique input/output args (same logic as buildModule, but no MLIR)
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenInputs;
  int inputArgCount = 0;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);

      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
          inputArgCount++;
        }
      } else if (!internalSlotOutputs.count(srcIdx)) {
        if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
          inputArgCount++;
        }
      }
    }
  }

  // Compute externally-visible outputs: only these need kernel args.
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  int outputArgCount = 0;
  int skippedInternalOutputs = 0;
  std::unordered_set<int> seenOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      if (seenOutputs.count(outIdx)) continue;  // Deduplicate
      seenOutputs.insert(outIdx);
      if (!externalOutputs.count(outIdx)) {
        skippedInternalOutputs++;
        continue;  // Purely internal — SSA forwarded, no kernel arg needed
      }
      outputArgCount++;
    }
  }
  if (skippedInternalOutputs > 0) {
    DSP_DIAG(JIT, "TritonIRBuilder::classifyAndAnalyze: eliminated %d internal intermediate outputs "
              "(keeping %d externally-visible output args)",
              skippedInternalOutputs, outputArgCount);
  }

  analysis.totalInputArgs = inputArgCount;
  analysis.totalOutputArgs = outputArgCount;
  analysis.totalArgs = inputArgCount + outputArgCount + 1;  // +1 for n_elements

  // Map best pattern type to SegmentKernelPattern
  const PatternMatch* best = patterns.bestMatch();
  if (best) {
    switch (best->type) {
      case PatternMatch::FUSED_ATTENTION_OP:
        analysis.pattern = SegmentKernelPattern::FUSED_ATTENTION;
        break;
      case PatternMatch::ATTENTION_QKV:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::FFN_BLOCK:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::SOFTMAX_DECOMPOSED:
        analysis.pattern = SegmentKernelPattern::NORMALIZATION;
        break;
      case PatternMatch::MATMUL_EPILOGUE:
        analysis.pattern = SegmentKernelPattern::MATMUL_EPILOGUE;
        break;
      case PatternMatch::PURE_MATMUL:
        analysis.pattern = SegmentKernelPattern::MATMUL_2D;
        break;
      case PatternMatch::PURE_REDUCTION:
        analysis.pattern = SegmentKernelPattern::REDUCTION_1D;
        break;
      case PatternMatch::PURE_NORMALIZATION:
        analysis.pattern = SegmentKernelPattern::NORMALIZATION;
        break;
      case PatternMatch::MIXED_MEGA_SEGMENT:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::PURE_ELEMENTWISE:
      default:
        analysis.pattern = SegmentKernelPattern::ELEMENTWISE_1D;
        break;
    }
  } else {
    // No pattern matched — check if all ops are elementwise-compatible
    bool allEw = true;
    for (auto& node : profile.nodes) {
      if (!isElementwiseCompatible(node.category)) { allEw = false; break; }
    }
    analysis.pattern = allEw ? SegmentKernelPattern::ELEMENTWISE_1D
                             : SegmentKernelPattern::WHOLE_GRAPH;
  }

  // Validate feasibility
  {
    analysis.canCompile = true;

    // Check for UNSUPPORTED ops — any UNSUPPORTED op in the segment means
    // we can't compile it as a Triton kernel, must fall back to native execution.
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::UNSUPPORTED) {
        analysis.canCompile = false;
        analysis.failureReason = "segment contains UNSUPPORTED op '" + node.opName +
                                 "' (slot " + std::to_string(node.slotIndex) +
                                 ") — no Triton emitter available, falling back to native";
        DSP_DIAG(JIT, "TritonIRBuilder::classifyAndAnalyze: %s", analysis.failureReason.c_str());
        break;
      }
    }

    if (analysis.canCompile && analysis.totalArgs > TRITON_DIRECT_ARG_LIMIT) {
      DSP_DIAG(JIT, "TritonIRBuilder::classifyAndAnalyze: segment will use indirect arg passing "
                "(%d args > %d direct limit)", analysis.totalArgs, TRITON_DIRECT_ARG_LIMIT);
    }
  }

  return analysis;
}

// ─── Combined analysis entry point ──────────────────────────────────────────

SegmentAnalysis TritonIRBuilder::analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                                 int totalSlots,
                                                 NDArray** externalInputs, int numExternalInputs,
                                                 NDArray** outputSlots, int totalOutputSlots,
                                                 int* requestedOutputSlotIndices,
                                                 int numRequestedOutputs) {
  auto profile = profileSegment(slots, startSlot, endSlot, outputSlots, totalOutputSlots);
  auto matched = matchPatterns(profile, slots, startSlot);

  // Log diagnostics
  DSP_DIAG(JIT, "TritonIRBuilder::analyzeSegment [%d-%d]: %d ops, %d ext inputs, %d outputs",
            startSlot, endSlot, profile.totalOps, profile.numUniqueExternalInputs, profile.numUniqueOutputs);
  DSP_DIAG(JIT, "  categories: elem=%d matmul=%d reduce=%d norm=%d attn=%d shape=%d data=%d const=%d id=%d cast=%d",
            profile.categoryCounts[0] + profile.categoryCounts[1],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::CONSTANT_GENERATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::IDENTITY)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::CAST)]);

  auto& env = Environment::getInstance();
  const bool logAllPatterns = env.tritonLogAllPatterns() || env.tritonVerbose();
  if (logAllPatterns) {
    for (auto& m : matched.matches) {
      DSP_DIAG(JIT, "  pattern: %s (priority=%d, %d ops)",
                m.description.c_str(), m.priority, static_cast<int>(m.localIndices.size()));
    }
  } else if (!matched.matches.empty()) {
    const auto* best = matched.bestMatch();
    if (best != nullptr) {
      DSP_DIAG(JIT, "  pattern(best): %s (priority=%d, %d ops)%s",
                best->description.c_str(), best->priority,
                static_cast<int>(best->localIndices.size()),
                matched.matches.size() > 1 ? " [set ND4J_TRITON_LOG_ALL_PATTERNS=1 for full list]" : "");
    }
  } else {
    DSP_DIAG(JIT, "  pattern(best): none%s", "");
  }

  auto analysis = classifyAndAnalyze(profile, matched, slots, startSlot, endSlot,
                                      totalSlots, externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots,
                                      requestedOutputSlotIndices, numRequestedOutputs);

  DSP_DIAG(JIT, "  result: pattern=%d, %d inputs, %d outputs, %d total args, canCompile=%d%s",
            static_cast<int>(analysis.pattern), analysis.totalInputArgs, analysis.totalOutputArgs,
            analysis.totalArgs, analysis.canCompile,
            analysis.canCompile ? "" : (", reason: " + analysis.failureReason).c_str());

  return analysis;
}

// ─── classifySegment — now delegates to 3-pass pipeline ─────────────────────

SegmentKernelPattern TritonIRBuilder::classifySegment(NativeSlot* slots, int startSlot, int endSlot) {
  auto profile = profileSegment(slots, startSlot, endSlot);
  auto matched = matchPatterns(profile, slots, startSlot);
  auto* best = matched.bestMatch();

  if (!best) {
    // Fallback: check if all ops are elementwise-compatible
    bool allEw = true;
    for (auto& node : profile.nodes) {
      if (!isElementwiseCompatible(node.category)) { allEw = false; break; }
    }
    return allEw ? SegmentKernelPattern::ELEMENTWISE_1D : SegmentKernelPattern::WHOLE_GRAPH;
  }

  switch (best->type) {
    case PatternMatch::FUSED_ATTENTION_OP:
      return SegmentKernelPattern::FUSED_ATTENTION;
    case PatternMatch::ATTENTION_QKV:
    case PatternMatch::FFN_BLOCK:
    case PatternMatch::MIXED_MEGA_SEGMENT:
      return SegmentKernelPattern::WHOLE_GRAPH;
    case PatternMatch::SOFTMAX_DECOMPOSED:
      return SegmentKernelPattern::NORMALIZATION;
    case PatternMatch::MATMUL_EPILOGUE:
      return SegmentKernelPattern::MATMUL_EPILOGUE;
    case PatternMatch::PURE_MATMUL:
      return SegmentKernelPattern::MATMUL_2D;
    case PatternMatch::PURE_REDUCTION:
      return SegmentKernelPattern::REDUCTION_1D;
    case PatternMatch::PURE_NORMALIZATION:
      return SegmentKernelPattern::NORMALIZATION;
    case PatternMatch::PURE_ELEMENTWISE:
    default:
      return SegmentKernelPattern::ELEMENTWISE_1D;
  }
}

// ─── Tile configuration ─────────────────────────────────────────────────────

void TritonIRBuilder::selectTileConfig(const std::vector<TritonOpCategory>& categories,
                                       const std::vector<std::vector<LongType>>& shapes,
                                       int& blockSize, int& numWarps, int& numStages) {
  bool hasMatmul = false;
  bool hasReduction = false;
  bool hasFusedAttention = false;
  bool hasNormalization = false;

  // Compute total output length for dynamic dim functions
  LongType maxOutputLen = 0;
  for (auto& shape : shapes) {
    LongType len = 1;
    for (auto d : shape) len *= d;
    if (len > maxOutputLen) maxOutputLen = len;
  }

  for (auto cat : categories) {
    if (cat == TritonOpCategory::MATMUL) hasMatmul = true;
    if (cat == TritonOpCategory::REDUCTION) hasReduction = true;
    if (cat == TritonOpCategory::NORMALIZATION) hasNormalization = true;
    if (cat == TritonOpCategory::FUSED_ATTENTION) hasFusedAttention = true;
  }

  if (hasFusedAttention) {
    LongType numTads = maxOutputLen > 0 ? maxOutputLen : 1;
    LongType tadLen = 64;
    for (auto& shape : shapes) {
      if (shape.size() >= 2) { tadLen = shape.back(); break; }
    }
    dim3 dims = getSoftmaxDims(numTads, tadLen);
    blockSize = 64;
    // 4 warps empirically optimal for decode attention (86.91 tok/s vs 69.56 at 2 warps)
    numWarps = 4;
    // numStages=1 empirically optimal for decode: no multi-buffering overhead
    // for the small tile loads in single-token decode (blockM=1).
    numStages = 1;
  } else if (hasMatmul) {
    // Auto-tune matmul tile config based on problem size.
    // Extract M, N, K from the largest matmul output shape.
    int approxM = 0, approxN = 0;
    for (size_t i = 0; i < categories.size(); i++) {
      if (categories[i] == TritonOpCategory::MATMUL && shapes[i].size() >= 2) {
        int m = static_cast<int>(shapes[i][shapes[i].size() - 2]);
        int n = static_cast<int>(shapes[i][shapes[i].size() - 1]);
        if (m * n > approxM * approxN) { approxM = m; approxN = n; }
      }
    }

    if (approxM <= 16) {
      // Small M (decode): fewer warps, smaller tiles to reduce overhead
      blockSize = 64;
      numWarps = 2;
      numStages = 2;
    } else if (approxM <= 64) {
      // Medium M: balanced config
      blockSize = 128;
      numWarps = 4;
      numStages = 3;
    } else {
      // Large M (prefill): maximize throughput with large tiles
      blockSize = 128;
      numWarps = 8;
      numStages = 4;
    }
  } else if (hasReduction || hasNormalization) {
    int xLength = static_cast<int>(std::min(maxOutputLen, static_cast<LongType>(INT_MAX)));
    int p = 64;
    while (p < xLength) p <<= 1;
    blockSize = std::min(p, 4096);
    numWarps = std::max(1, blockSize / 32);
    numStages = 2;
  } else {
    try {
      dim3 dims = getLaunchDims("pairwiseTransforms");
      blockSize = static_cast<int>(dims.y);
      numWarps = std::max(1, blockSize / 32);
    } catch (...) {
      blockSize = 1024;
      numWarps = 4;
    }
    numStages = 3;
  }

  // Ensure blockSize is power of 2 (Triton requirement for efficient tiling)
  if (blockSize > 0 && (blockSize & (blockSize - 1)) != 0) {
    int p = 1;
    while (p < blockSize) p <<= 1;
    blockSize = p;
  }

  // Clamp to reasonable Triton tile range
  blockSize = std::max(64, std::min(blockSize, 4096));
  numWarps = std::max(1, std::min(numWarps, 16));
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
