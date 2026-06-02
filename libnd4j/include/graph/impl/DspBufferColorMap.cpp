/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#include <graph/DspBufferColorMap.h>
#include <graph/DspBufferPool.h>
#include <graph/DspDiagnostics.h>
#include <array/DataTypeUtils.h>
#include <helpers/ShapeUtils.h>
#include <execution/LaunchContext.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// ─── SlotLivenessData validation ─────────────────────────────────────────────

void SlotLivenessData::validate() const {
  if (totalOutputSlots <= 0) {
    THROW_EXCEPTION("SlotLivenessData::validate(): totalOutputSlots <= 0");
  }
  if (producerStep == nullptr || lastConsumerStep == nullptr) {
    THROW_EXCEPTION("SlotLivenessData::validate(): null producerStep or lastConsumerStep");
  }
  for (int i = 0; i < totalOutputSlots; i++) {
    if (producerStep[i] >= 0 && lastConsumerStep[i] >= 0) {
      if (producerStep[i] > lastConsumerStep[i]) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "SlotLivenessData::validate(): slot %d has producerStep=%d > lastConsumerStep=%d",
                 i, producerStep[i], lastConsumerStep[i]);
        THROW_EXCEPTION(msg);
      }
    }
  }
}

// ─── DspBufferColorMap ───────────────────────────────────────────────────────

DspBufferColorMap::DspBufferColorMap() {}

DspBufferColorMap::~DspBufferColorMap() {
  delete[] colorOf_;
  colorOf_ = nullptr;
}

// ─── compute() ───────────────────────────────────────────────────────────────

void DspBufferColorMap::compute(
    const SlotLivenessData& liveness,
    NDArray** outputSlots,
    int totalOutputSlots,
    const SlotBufferInfo* ownership,
    const std::unordered_set<int>& requestedOutputSlots) {

  DSP_DIAG(MEMORY, "COLORING_COMPUTE_START: totalSlots=%d", totalOutputSlots);

  // Validate inputs
  if (liveness.totalOutputSlots != totalOutputSlots) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "DspBufferColorMap::compute(): liveness.totalOutputSlots=%d != totalOutputSlots=%d",
             liveness.totalOutputSlots, totalOutputSlots);
    THROW_EXCEPTION(msg);
  }

  liveness.validate();

  // Reset any previous computation
  reset();

  totalOutputSlots_ = totalOutputSlots;
  colorOf_ = new int[totalOutputSlots];
  std::memset(colorOf_, -1, sizeof(int) * totalOutputSlots);  // -1 = uncolored

  // Step 1: Determine eligibility
  // A slot is eligible for coloring if:
  //   - It is SLOT_OWNED (not a view, not a weight, not workspace)
  //   - It has no VIEW_OF_SLOT children (viewRefCount == 0)
  //   - It is NOT a requested output slot
  //   - It has valid liveness data (producerStep >= 0 and lastConsumerStep >= 0)
  //   - It has a non-null output array

  std::vector<bool> eligible(totalOutputSlots, false);
  int numViews = 0, numViewParents = 0, numOutputs = 0, numNoLiveness = 0, numNull = 0;

  for (int i = 0; i < totalOutputSlots; i++) {
    if (outputSlots[i] == nullptr) { numNull++; continue; }
    if (ownership[i].ownership != BufferOwnership::SLOT_OWNED) { numViews++; continue; }
    if (ownership[i].viewRefCount > 0) { numViewParents++; continue; }
    if (requestedOutputSlots.count(i) > 0) { numOutputs++; continue; }
    if (liveness.producerStep[i] < 0 || liveness.lastConsumerStep[i] < 0) { numNoLiveness++; continue; }
    eligible[i] = true;
  }

  int numEligible = 0;
  for (int i = 0; i < totalOutputSlots; i++) {
    if (eligible[i]) numEligible++;
  }

  DSP_DIAG(MEMORY, "COLORING_ELIGIBILITY: eligible=%d (null=%d views=%d viewParents=%d "
           "outputs=%d noLiveness=%d) of total=%d",
           numEligible, numNull, numViews, numViewParents, numOutputs, numNoLiveness,
           totalOutputSlots);

  if (numEligible == 0) {
    computed_ = true;
    return;  // Nothing to color
  }

  // Step 2: Compute view-extended liveness
  int* effectiveLastConsumer = new int[totalOutputSlots];
  std::memcpy(effectiveLastConsumer, liveness.lastConsumerStep, sizeof(int) * totalOutputSlots);
  computeViewExtendedLiveness(ownership, totalOutputSlots,
                              liveness.producerStep, liveness.lastConsumerStep,
                              effectiveLastConsumer);

  // Step 3: Group eligible slots by (shape signature, dtype, device)
  // Shape signature = product of dimensions (element count) since we're matching buffers
  struct ShapeKey {
    LongType numElements;
    DataType dtype;
    int deviceId;

    bool operator==(const ShapeKey& o) const {
      return numElements == o.numElements && dtype == o.dtype && deviceId == o.deviceId;
    }
  };
  struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const {
      size_t h = std::hash<LongType>{}(k.numElements);
      h ^= std::hash<int>{}(static_cast<int>(k.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(k.deviceId) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::unordered_map<ShapeKey, std::vector<int>, ShapeKeyHash> groups;

  for (int i = 0; i < totalOutputSlots; i++) {
    if (!eligible[i]) continue;

    auto* arr = outputSlots[i];
    ShapeKey key;
    key.numElements = arr->lengthOf();
    key.dtype = arr->dataType();
    key.deviceId = ownership[i].deviceId;
    groups[key].push_back(i);
  }

  int largestGroup = 0;
  for (auto& kv : groups) {
    if (static_cast<int>(kv.second.size()) > largestGroup) {
      largestGroup = static_cast<int>(kv.second.size());
    }
  }

  DSP_DIAG(MEMORY, "COLORING_SHAPE_GROUPS: numGroups=%d largestGroup=%d",
           static_cast<int>(groups.size()), largestGroup);

  // Step 4: Greedy interval coloring within each shape group
  // For each group, sort slots by producerStep (ascending).
  // Greedily assign each slot to the first color whose last member's
  // effectiveLastConsumer < this slot's producerStep.

  int nextColor = 0;
  bytesBefore_ = 0;

  for (auto& kv : groups) {
    auto& slots = kv.second;

    if (slots.size() <= 1) {
      // Single-slot groups can't share, skip
      for (int s : slots) {
        bytesBefore_ += outputSlots[s]->lengthOf() * DataTypeUtils::sizeOfElement(outputSlots[s]->dataType());
      }
      continue;
    }

    // Sort by producer step
    std::sort(slots.begin(), slots.end(), [&](int a, int b) {
      return liveness.producerStep[a] < liveness.producerStep[b];
    });

    // Track the last consumer step for each color in this group
    struct ColorEnd {
      int colorId;
      int lastConsumer;  // effectiveLastConsumer of the last assigned slot
    };
    std::vector<ColorEnd> colorEnds;

    for (int slotIdx : slots) {
      int producer = liveness.producerStep[slotIdx];

      // Try to reuse an existing color whose last member is dead before this slot
      int bestColor = -1;
      int bestLastConsumer = -1;
      for (int c = 0; c < static_cast<int>(colorEnds.size()); c++) {
        if (colorEnds[c].lastConsumer < producer) {
          // This color is available — pick the one with the latest lastConsumer
          // (best-fit: minimize wasted "gap" between colors)
          if (bestColor == -1 || colorEnds[c].lastConsumer > bestLastConsumer) {
            bestColor = c;
            bestLastConsumer = colorEnds[c].lastConsumer;
          }
        }
      }

      if (bestColor >= 0) {
        // Reuse existing color
        colorOf_[slotIdx] = colorEnds[bestColor].colorId;
        colorEnds[bestColor].lastConsumer = effectiveLastConsumer[slotIdx];
      } else {
        // Need a new color
        int newColorId = nextColor++;
        colorOf_[slotIdx] = newColorId;
        colorEnds.push_back({newColorId, effectiveLastConsumer[slotIdx]});
      }

      size_t slotBytes = outputSlots[slotIdx]->lengthOf() *
                         DataTypeUtils::sizeOfElement(outputSlots[slotIdx]->dataType());
      bytesBefore_ += slotBytes;
    }
  }

  // Also count bytes for uncolored eligible slots
  for (int i = 0; i < totalOutputSlots; i++) {
    if (eligible[i] && colorOf_[i] == -1) {
      bytesBefore_ += outputSlots[i]->lengthOf() *
                      DataTypeUtils::sizeOfElement(outputSlots[i]->dataType());
    }
  }

  // Step 5: Build color info
  numColors_ = nextColor;
  numColoredSlots_ = 0;
  colorInfos_.resize(numColors_);
  for (int c = 0; c < numColors_; c++) {
    colorInfos_[c].masterSlotIdx = -1;
    colorInfos_[c].memberCount = 0;
    colorInfos_[c].bufferBytes = 0;
    colorInfos_[c].sharedBuffer = nullptr;
  }

  for (int i = 0; i < totalOutputSlots; i++) {
    int c = colorOf_[i];
    if (c < 0) continue;
    numColoredSlots_++;
    colorInfos_[c].memberCount++;
    if (colorInfos_[c].masterSlotIdx == -1) {
      colorInfos_[c].masterSlotIdx = i;
      colorInfos_[c].bufferBytes = outputSlots[i]->lengthOf() *
                                    DataTypeUtils::sizeOfElement(outputSlots[i]->dataType());
    }
  }

  // Remove colors with only 1 member (nothing to share)
  for (int i = 0; i < totalOutputSlots; i++) {
    int c = colorOf_[i];
    if (c >= 0 && colorInfos_[c].memberCount <= 1) {
      colorOf_[i] = -1;
      numColoredSlots_--;
    }
  }

  // Recount after pruning singles
  int actualColors = 0;
  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount > 1) actualColors++;
  }

  // Compute bytes after (one buffer per color with >1 member, plus all uncolored)
  bytesAfter_ = 0;
  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount > 1) {
      bytesAfter_ += colorInfos_[c].bufferBytes;
    }
  }
  for (int i = 0; i < totalOutputSlots; i++) {
    if (eligible[i] && colorOf_[i] == -1) {
      bytesAfter_ += outputSlots[i]->lengthOf() *
                     DataTypeUtils::sizeOfElement(outputSlots[i]->dataType());
    }
  }

  bytesSaved_ = (bytesBefore_ > bytesAfter_) ? (bytesBefore_ - bytesAfter_) : 0;

  // Validate no overlaps within each color
  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount <= 1) continue;
    std::vector<int> members;
    for (int i = 0; i < totalOutputSlots; i++) {
      if (colorOf_[i] == c) members.push_back(i);
    }
    for (size_t a = 0; a < members.size(); a++) {
      for (size_t b = a + 1; b < members.size(); b++) {
        assertNoOverlap(members[a], members[b],
                        liveness.producerStep, effectiveLastConsumer, c);
      }
    }
  }

  delete[] effectiveLastConsumer;

  computed_ = true;

  DSP_DIAG(MEMORY, "COLORING_COMPUTE_DONE: %d slots -> %d colors (effective=%d), saving %zuMB",
           numColoredSlots_, numColors_, actualColors,
           bytesSaved_ / (1024 * 1024));
}

// ─── apply() ─────────────────────────────────────────────────────────────────

int DspBufferColorMap::apply(
    NDArray** outputSlots,
    SlotBufferInfo* ownership,
    std::unordered_set<NDArray*>& planOwnedArrays,
    DspBufferPool& pool) {

  if (!computed_) {
    THROW_EXCEPTION("DspBufferColorMap::apply(): compute() not called yet");
  }
  if (applied_) {
    THROW_EXCEPTION("DspBufferColorMap::apply(): already applied");
  }

  int consolidated = 0;

  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount <= 1) continue;

    int masterIdx = colorInfos_[c].masterSlotIdx;
    if (masterIdx < 0 || outputSlots[masterIdx] == nullptr) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "DspBufferColorMap::apply(): master slot %d for color %d is null",
               masterIdx, c);
      THROW_EXCEPTION(msg);
    }

    // The master keeps its own buffer — record it
    auto* masterDb = outputSlots[masterIdx]->dataBuffer();
    colorInfos_[c].sharedBuffer = masterDb;

    // For all non-master members: create new NDArray wrapping the master's buffer
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (colorOf_[i] != c || i == masterIdx) continue;

      auto* oldArr = outputSlots[i];
      if (oldArr == nullptr) continue;

      // Create new NDArray wrapping the master's DataBuffer with this slot's shape
      auto* shapeInfo = const_cast<LongType*>(oldArr->shapeInfo());
      auto* newArr = new NDArray(masterDb, shapeInfo, LaunchContext::defaultContext(), 0);

      DSP_DIAG(MEMORY, "COLORING_APPLY_SLOT: slot=%d color=%d masterSlot=%d buffer=%p",
               i, c, masterIdx, (void*)masterDb);

      // Validate shape match
      if (newArr->dataType() != oldArr->dataType() || newArr->lengthOf() != oldArr->lengthOf()) {
        delete newArr;
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "DspBufferColorMap::apply(): shape mismatch slot %d vs master %d", i, masterIdx);
        THROW_EXCEPTION(msg);
      }

      // Replace in plan
      planOwnedArrays.erase(oldArr);
      delete oldArr;
      outputSlots[i] = newArr;
      planOwnedArrays.insert(newArr);

      // Update ownership to reflect shared buffer
      ownership[i].dataBuffer = masterDb;

      consolidated++;
    }
  }

  applied_ = true;

  DSP_DIAG(MEMORY, "COLORING_APPLY_DONE: consolidated=%d, saved %zuMB",
           consolidated, bytesSaved_ / (1024 * 1024));

  return consolidated;
}

// ─── eject() ─────────────────────────────────────────────────────────────────

int DspBufferColorMap::eject(
    NDArray** outputSlots,
    SlotBufferInfo* ownership,
    std::unordered_set<NDArray*>& planOwnedArrays,
    DspBufferPool& pool) {

  if (!applied_) {
    return 0;  // Nothing to eject
  }

  int restored = 0;

  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount <= 1) continue;
    int masterIdx = colorInfos_[c].masterSlotIdx;

    // For each non-master member, give it a fresh dedicated buffer
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (colorOf_[i] != c || i == masterIdx) continue;

      auto* oldArr = outputSlots[i];
      if (oldArr == nullptr) continue;

      LongType numElements = oldArr->lengthOf();
      DataType dtype = oldArr->dataType();

      // Acquire fresh buffer from pool
      DataBuffer* freshDb = pool.acquire(numElements, dtype);

      // Copy current data from shared buffer to new dedicated buffer
      // This preserves correctness mid-execution
      auto* sharedDb = oldArr->dataBuffer();
      if (sharedDb != nullptr && freshDb != nullptr) {
        size_t bytes = numElements * DataTypeUtils::sizeOfElement(dtype);
        // Copy on host side
        if (sharedDb->primary() != nullptr && freshDb->primary() != nullptr) {
          std::memcpy(freshDb->primary(), sharedDb->primary(), bytes);
        }
#ifdef SD_CUDA
        // Copy on device side
        if (sharedDb->special() != nullptr && freshDb->special() != nullptr) {
          auto stream = LaunchContext::defaultContext()->getCudaStream();
          cudaMemcpyAsync(freshDb->special(), sharedDb->special(), bytes,
                          cudaMemcpyDeviceToDevice, *stream);
        }
#endif
      }

      // Create new NDArray with dedicated buffer
      auto* shapeInfo = const_cast<LongType*>(oldArr->shapeInfo());
      auto* newArr = new NDArray(freshDb, shapeInfo, LaunchContext::defaultContext(), 0);

      // Replace in plan
      planOwnedArrays.erase(oldArr);
      delete oldArr;
      outputSlots[i] = newArr;
      planOwnedArrays.insert(newArr);

      // Update ownership
      ownership[i].dataBuffer = freshDb;

      restored++;
    }
  }

  applied_ = false;

  // Clear color info shared buffer refs
  for (auto& ci : colorInfos_) {
    ci.sharedBuffer = nullptr;
  }

  DSP_DIAG(MEMORY, "COLORING_EJECTED: restored %d slots", restored);

  return restored;
}

// ─── validate() ──────────────────────────────────────────────────────────────

void DspBufferColorMap::validate(
    NDArray** outputSlots,
    const SlotBufferInfo* ownership,
    int totalOutputSlots) const {

  if (!applied_) return;  // Nothing to validate if not applied

  if (totalOutputSlots != totalOutputSlots_) {
    THROW_EXCEPTION("DspBufferColorMap::validate(): totalOutputSlots mismatch");
  }

  // Check 1: Same-color slots must share the same DataBuffer address
  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount <= 1) continue;
    int masterIdx = colorInfos_[c].masterSlotIdx;
    if (masterIdx < 0 || outputSlots[masterIdx] == nullptr) continue;

    DataBuffer* expectedDb = outputSlots[masterIdx]->dataBuffer();

    for (int i = 0; i < totalOutputSlots; i++) {
      if (colorOf_[i] != c || i == masterIdx) continue;
      if (outputSlots[i] == nullptr) continue;

      DataBuffer* actualDb = outputSlots[i]->dataBuffer();
      if (actualDb != expectedDb) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "DspBufferColorMap::validate(): color %d slot %d has DataBuffer %p "
                 "but master slot %d has %p",
                 c, i, (void*)actualDb, masterIdx, (void*)expectedDb);
        THROW_EXCEPTION(msg);
      }
    }
  }

  // Check 2: VIEW_OF_SLOT parents must NOT be colored
  for (int i = 0; i < totalOutputSlots; i++) {
    if (ownership[i].viewRefCount > 0 && colorOf_[i] >= 0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "DspBufferColorMap::validate(): slot %d has viewRefCount=%d but is colored (color %d)",
               i, ownership[i].viewRefCount, colorOf_[i]);
      THROW_EXCEPTION(msg);
    }
  }

  // Check 3: Uncolored SLOT_OWNED slots should retain their own buffer
  // (not sharing with any colored slot)
  std::unordered_set<DataBuffer*> coloredBuffers;
  for (int i = 0; i < totalOutputSlots; i++) {
    if (colorOf_[i] >= 0 && outputSlots[i] != nullptr) {
      coloredBuffers.insert(outputSlots[i]->dataBuffer());
    }
  }

  for (int i = 0; i < totalOutputSlots; i++) {
    if (colorOf_[i] >= 0) continue;  // colored — skip
    if (outputSlots[i] == nullptr) continue;
    if (ownership[i].ownership != BufferOwnership::SLOT_OWNED) continue;

    DataBuffer* db = outputSlots[i]->dataBuffer();
    if (coloredBuffers.count(db) > 0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "DspBufferColorMap::validate(): uncolored SLOT_OWNED slot %d shares "
               "DataBuffer %p with a colored slot",
               i, (void*)db);
      THROW_EXCEPTION(msg);
    }
  }

  DSP_DIAG(MEMORY, "COLORING_VALIDATE_OK: all invariants passed");
}

// ─── Introspection ───────────────────────────────────────────────────────────

int DspBufferColorMap::colorOf(int slotIdx) const {
  if (colorOf_ == nullptr) return -1;
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) {
    char msg[128];
    snprintf(msg, sizeof(msg), "DspBufferColorMap::colorOf(): slotIdx %d out of range [0, %d)",
             slotIdx, totalOutputSlots_);
    THROW_EXCEPTION(msg);
  }
  return colorOf_[slotIdx];
}

int DspBufferColorMap::colorMemberCount(int colorId) const {
  if (colorId < 0 || colorId >= numColors_) return 0;
  return colorInfos_[colorId].memberCount;
}

size_t DspBufferColorMap::colorBufferBytes(int colorId) const {
  if (colorId < 0 || colorId >= numColors_) return 0;
  return colorInfos_[colorId].bufferBytes;
}

DspBufferColorMap::ColoringReport DspBufferColorMap::report() const {
  ColoringReport r;
  r.totalSlots = totalOutputSlots_;
  r.coloredSlots = numColoredSlots_;
  r.uncoloredSlots = totalOutputSlots_ - numColoredSlots_;
  r.numColors = numColors_;
  r.bytesBefore = bytesBefore_;
  r.bytesAfter = bytesAfter_;
  r.bytesSaved = bytesSaved_;

  // Count shape groups and largest
  r.numShapeGroups = 0;
  r.largestGroupSize = 0;
  for (int c = 0; c < numColors_; c++) {
    if (colorInfos_[c].memberCount > 1) {
      r.numShapeGroups++;
      if (colorInfos_[c].memberCount > r.largestGroupSize) {
        r.largestGroupSize = colorInfos_[c].memberCount;
      }
    }
  }

  return r;
}

// ─── reset() ─────────────────────────────────────────────────────────────────

void DspBufferColorMap::reset() {
  if (applied_) {
    DSP_DIAG(MEMORY, "COLORING_RESET: WARNING — reset() called while still applied, "
             "call eject() first to avoid buffer leaks");
  }

  int oldColors = numColors_;
  delete[] colorOf_;
  colorOf_ = nullptr;
  totalOutputSlots_ = 0;
  numColors_ = 0;
  numColoredSlots_ = 0;
  bytesBefore_ = 0;
  bytesAfter_ = 0;
  bytesSaved_ = 0;
  computed_ = false;
  applied_ = false;
  colorInfos_.clear();

  DSP_DIAG(MEMORY, "COLORING_RESET: cleared %d colors", oldColors);
}

// ─── View-extended liveness ──────────────────────────────────────────────────

void DspBufferColorMap::computeViewExtendedLiveness(
    const SlotBufferInfo* ownership,
    int totalOutputSlots,
    const int* producerStep,
    const int* lastConsumerStep,
    int* effectiveLastConsumer) const {

  // For each VIEW_OF_SLOT slot, extend its parent's effective lastConsumer
  // to at least cover the view's lastConsumerStep.
  for (int i = 0; i < totalOutputSlots; i++) {
    if (ownership[i].ownership == BufferOwnership::VIEW_OF_SLOT) {
      int parent = ownership[i].parentSlotIdx;
      if (parent >= 0 && parent < totalOutputSlots) {
        if (lastConsumerStep[i] > effectiveLastConsumer[parent]) {
          effectiveLastConsumer[parent] = lastConsumerStep[i];
        }
      }
    }
  }
}

// ─── Overlap assertion ───────────────────────────────────────────────────────

void DspBufferColorMap::assertNoOverlap(
    int slotA, int slotB,
    const int* producerStep,
    const int* effectiveLastConsumer,
    int color) const {

  int startA = producerStep[slotA];
  int endA = effectiveLastConsumer[slotA];
  int startB = producerStep[slotB];
  int endB = effectiveLastConsumer[slotB];

  // Overlap: [startA, endA] ∩ [startB, endB] ≠ ∅
  // i.e., startA <= endB && startB <= endA
  if (startA <= endB && startB <= endA) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "DspBufferColorMap: overlap in color %d: slot %d [%d,%d] vs slot %d [%d,%d]",
             color, slotA, startA, endA, slotB, startB, endB);
    THROW_EXCEPTION(msg);
  }
}

}  // namespace graph
}  // namespace sd
