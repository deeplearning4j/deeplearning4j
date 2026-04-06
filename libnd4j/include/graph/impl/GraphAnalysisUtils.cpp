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

#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX

#include <graph/GraphAnalysisUtils.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

SegmentProfile GraphAnalysisUtils::profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                                   NDArray** outputSlots, int totalOutputSlots) {
  SegmentProfile profile;
  int segSize = endSlot - startSlot + 1;
  profile.totalOps = segSize;
  profile.nodes.resize(segSize);

  std::unordered_map<int, int> slotToLocal;
  std::unordered_map<int, int> outputSlotToProducer;

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    slotToLocal[absSlot] = i;

    auto& node = profile.nodes[i];
    node.slotIndex = absSlot;
    node.localIndex = i;
    node.opName = slots[absSlot].opName;

    const auto& table = getOpCategoryTable();
    auto catIt = table.find(node.opName);
    node.category = (catIt != table.end()) ? catIt->second : TritonOpCategory::UNSUPPORTED;
    node.hasExternalInput = false;

    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSlotToProducer[slots[absSlot].outputSlotIndices[o]] = i;
    }

    if (slots[absSlot].numOutputs > 0) {
      int outIdx = slots[absSlot].outputSlotIndices[0];
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

    int catIdx = static_cast<int>(node.category);
    if (catIdx >= 0 && catIdx < 16) profile.categoryCounts[catIdx]++;
  }

  std::unordered_set<int> externalInputSet;
  std::unordered_map<int, std::vector<int>> outputToConsumers;

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    auto& node = profile.nodes[i];

    for (int inp = 0; inp < slots[absSlot].numInputs; inp++) {
      int srcIdx = slots[absSlot].inputSourceIndices[inp];
      if (srcIdx < 0) {
        node.inputLocalIndices.push_back(-1);
        node.hasExternalInput = true;
        externalInputSet.insert(srcIdx);
      } else {
        auto producerIt = outputSlotToProducer.find(srcIdx);
        if (producerIt != outputSlotToProducer.end()) {
          node.inputLocalIndices.push_back(producerIt->second);
          outputToConsumers[srcIdx].push_back(i);
        } else {
          node.inputLocalIndices.push_back(-1);
          node.hasExternalInput = true;
          externalInputSet.insert(srcIdx);
        }
      }
    }
  }

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

  std::unordered_set<int> outputSet;
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSet.insert(slots[absSlot].outputSlotIndices[o]);
    }
  }

  profile.numUniqueExternalInputs = static_cast<int>(externalInputSet.size());
  profile.numUniqueOutputs = static_cast<int>(outputSet.size());

  profile.hasMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)] > 0;
  profile.hasReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)] > 0;
  profile.hasNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)] > 0;
  profile.hasFusedAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)] > 0;
  profile.hasShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)] > 0;
  profile.hasDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)] > 0;

  return profile;
}

std::unordered_set<int> GraphAnalysisUtils::computeExternallyVisibleOutputs(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots) {

  std::unordered_set<int> segmentOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      segmentOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> externalOutputs;
  for (int outIdx : segmentOutputs) {
    bool consumedOutside = false;
    for (int i = endSlot + 1; i < totalSlots && !consumedOutside; i++) {
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        if (slots[i].inputSourceIndices[inp] == outIdx) {
          consumedOutside = true;
          break;
        }
      }
    }
    if (consumedOutside) {
      externalOutputs.insert(outIdx);
    }
  }

  // All outputs from the last slot are always externally visible
  for (int o = 0; o < slots[endSlot].numOutputs; o++) {
    externalOutputs.insert(slots[endSlot].outputSlotIndices[o]);
  }

  return externalOutputs;
}

}  // namespace graph
}  // namespace sd

#endif  // defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
