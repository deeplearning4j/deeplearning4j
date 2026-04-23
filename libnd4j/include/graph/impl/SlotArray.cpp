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

#include <graph/SlotArray.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <cassert>

namespace sd {
namespace graph {

SlotArray::SlotArray(int totalSlots) : totalSlots_(totalSlots) {
  assert(totalSlots > 0 && "SlotArray: totalSlots must be positive");

  entries_ = new SlotEntry[totalSlots_];
  rawArrays_ = new NDArray*[totalSlots_];
  std::memset(rawArrays_, 0, sizeof(NDArray*) * totalSlots_);
}

SlotArray::~SlotArray() {
  delete[] entries_;
  entries_ = nullptr;
  delete[] rawArrays_;
  rawArrays_ = nullptr;
  totalSlots_ = 0;
}

void SlotArray::setArray(int idx, NDArray* arr, BufferOwnership ownershipType, int parentSlotIdx) {
  assert(idx >= 0 && idx < totalSlots_ && "SlotArray::setArray: index out of bounds");

  auto& entry = entries_[idx];

  // If replacing an existing VIEW_OF_SLOT, decrement parent's viewRefCount
  if (entry.ownership.ownership == BufferOwnership::VIEW_OF_SLOT &&
      entry.ownership.parentSlotIdx >= 0 &&
      entry.ownership.parentSlotIdx < totalSlots_) {
    entries_[entry.ownership.parentSlotIdx].ownership.removeViewRef();
  }

  // Install the array
  entry.array = arr;
  rawArrays_[idx] = arr;

  // Set ownership
  entry.ownership.ownership = ownershipType;
  entry.ownership.parentSlotIdx = parentSlotIdx;
  entry.ownership.dataBuffer = (arr != nullptr) ? arr->dataBuffer() : nullptr;

  // If this is a VIEW_OF_SLOT, increment parent's viewRefCount
  if (ownershipType == BufferOwnership::VIEW_OF_SLOT &&
      parentSlotIdx >= 0 && parentSlotIdx < totalSlots_) {
    entries_[parentSlotIdx].ownership.addViewRef();
  }

  // Bump generation
  entry.bumpGeneration();
}

void SlotArray::setState(int idx, SlotLifecycleState newState) {
  assert(idx >= 0 && idx < totalSlots_ && "SlotArray::setState: index out of bounds");
  entries_[idx].state = newState;
}

void SlotArray::setViewProducer(int idx, bool isView) {
  assert(idx >= 0 && idx < totalSlots_ && "SlotArray::setViewProducer: index out of bounds");
  entries_[idx].isViewProducer = isView;
}

void SlotArray::addProtectedWeight(DataBuffer* db) {
  if (db != nullptr) {
    protectedWeightBuffers_.insert(db);
  }
}

bool SlotArray::isProtectedWeight(DataBuffer* db) const {
  return protectedWeightBuffers_.count(db) > 0;
}

void SlotArray::resetAll(bool keepWeights) {
  for (int i = 0; i < totalSlots_; i++) {
    auto& entry = entries_[i];

    if (keepWeights && (entry.ownership.ownership == BufferOwnership::WEIGHT ||
                        entry.ownership.ownership == BufferOwnership::VIEW_OF_WEIGHT)) {
      continue;
    }

    entry.array = nullptr;
    rawArrays_[i] = nullptr;
    entry.ownership.reset();
    entry.state = SlotLifecycleState::WARMUP;
    entry.isViewProducer = false;
    entry.generation = 0;
    for (int j = 0; j < SLOT_MAX_UNTRACKED_OUTPUTS; j++) {
      entry.untrackedOutputs[j] = nullptr;
    }
  }
}

void SlotArray::demoteAllFrozen() {
  for (int i = 0; i < totalSlots_; i++) {
    if (entries_[i].state >= SlotLifecycleState::FROZEN) {
      entries_[i].state = SlotLifecycleState::SHAPE_CACHED;
    }
  }
}

void SlotArray::bindToCurrentThread() {
  ownerThread_ = std::this_thread::get_id();
  bound_.store(true, std::memory_order_release);
}

void SlotArray::assertBoundThread() const {
  if (bound_.load(std::memory_order_acquire)) {
    assert(std::this_thread::get_id() == ownerThread_ &&
           "SlotArray accessed from wrong thread");
  }
}

int SlotArray::countByOwnership(BufferOwnership type) const {
  int count = 0;
  for (int i = 0; i < totalSlots_; i++) {
    if (entries_[i].ownership.ownership == type) {
      count++;
    }
  }
  return count;
}

int SlotArray::countByState(SlotLifecycleState state) const {
  int count = 0;
  for (int i = 0; i < totalSlots_; i++) {
    if (entries_[i].state == state) {
      count++;
    }
  }
  return count;
}

}  // namespace graph
}  // namespace sd
