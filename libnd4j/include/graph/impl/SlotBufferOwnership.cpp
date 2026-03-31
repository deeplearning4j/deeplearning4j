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

#include <graph/SlotBufferOwnership.h>
#include <graph/DspDiagnostics.h>
#include <array/NDArray.h>

namespace sd {
namespace graph {

BufferOwnership classifyBufferOwnership(
    DataBuffer* outBuffer,
    int slotIdx,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int* parentSlotIdxOut) {

  if (outBuffer == nullptr) {
    return BufferOwnership::UNSET;
  }

  if (parentSlotIdxOut != nullptr) {
    *parentSlotIdxOut = -1;
  }

  // Check if this buffer belongs to an external input (weight/constant/placeholder)
  for (int i = 0; i < numExternalInputs; i++) {
    if (externalInputs[i] != nullptr && externalInputs[i]->dataBuffer() == outBuffer) {
      return BufferOwnership::VIEW_OF_WEIGHT;
    }
  }

  // Check if this buffer belongs to another output slot (view of another slot's output)
  for (int i = 0; i < totalOutputSlots; i++) {
    if (i == slotIdx) continue;  // skip self
    if (outputSlots[i] != nullptr && outputSlots[i]->dataBuffer() == outBuffer) {
      if (parentSlotIdxOut != nullptr) {
        *parentSlotIdxOut = i;
      }
      return BufferOwnership::VIEW_OF_SLOT;
    }
  }

  // Buffer is unique to this slot — slot owns it
  return BufferOwnership::SLOT_OWNED;
}

bool validateOwnershipConsistency(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots) {

  if (ownership == nullptr || outputSlots == nullptr) return true;

  bool valid = true;
  for (int i = 0; i < totalSlots; i++) {
    const auto& info = ownership[i];

    // VIEW_OF_SLOT must have a valid parent
    if (info.ownership == BufferOwnership::VIEW_OF_SLOT) {
      if (info.parentSlotIdx < 0 || info.parentSlotIdx >= totalSlots) {
        DSP_DIAG(MEMORY, "OWNERSHIP_INVALID: slot %d is VIEW_OF_SLOT but parentSlotIdx=%d (totalSlots=%d)",
                 i, info.parentSlotIdx, totalSlots);
        valid = false;
      }
    }

    // SLOT_OWNED should not have a parent
    if (info.ownership == BufferOwnership::SLOT_OWNED && info.parentSlotIdx >= 0) {
      DSP_DIAG(MEMORY, "OWNERSHIP_INVALID: slot %d is SLOT_OWNED but has parentSlotIdx=%d",
               i, info.parentSlotIdx);
      valid = false;
    }

    // If the output slot exists, the dataBuffer should match
    if (outputSlots[i] != nullptr && info.dataBuffer != nullptr) {
      if (outputSlots[i]->dataBuffer() != info.dataBuffer) {
        DSP_DIAG(MEMORY, "OWNERSHIP_STALE: slot %d dataBuffer mismatch (tracked=%p, actual=%p)",
                 i, info.dataBuffer, outputSlots[i]->dataBuffer());
        valid = false;
      }
    }
  }

  return valid;
}

const char* bufferOwnershipName(BufferOwnership ownership) {
  switch (ownership) {
    case BufferOwnership::UNSET:          return "UNSET";
    case BufferOwnership::SLOT_OWNED:     return "SLOT_OWNED";
    case BufferOwnership::VIEW_OF_SLOT:   return "VIEW_OF_SLOT";
    case BufferOwnership::VIEW_OF_WEIGHT: return "VIEW_OF_WEIGHT";
    case BufferOwnership::WEIGHT:         return "WEIGHT";
    case BufferOwnership::CAPTURE_BUFFER: return "CAPTURE_BUFFER";
    case BufferOwnership::WORKSPACE:      return "WORKSPACE";
    default:                              return "UNKNOWN";
  }
}

void classifyAndUpdateOwnership(
    SlotBufferInfo& info,
    NDArray* outArray,
    int slotIdx,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    SlotBufferInfo* ownershipArray) {

  // 1. Null output or null dataBuffer → UNSET
  if (outArray == nullptr || outArray->dataBuffer() == nullptr) {
    info.ownership = BufferOwnership::UNSET;
    info.dataBuffer = nullptr;
    return;
  }

  DataBuffer* outBuffer = outArray->dataBuffer();

  // 2. Check external inputs — if buffer matches, it's a view of a weight/constant
  for (int i = 0; i < numExternalInputs; i++) {
    if (externalInputs[i] != nullptr && externalInputs[i]->dataBuffer() == outBuffer) {
      info.ownership = BufferOwnership::VIEW_OF_WEIGHT;
      info.dataBuffer = outBuffer;
      return;
    }
  }

  // 3. Check other output slots — if buffer matches another slot, it's a view of that slot
  for (int i = 0; i < totalOutputSlots; i++) {
    if (i == slotIdx) continue;
    if (outputSlots[i] != nullptr && outputSlots[i]->dataBuffer() == outBuffer) {
      info.ownership = BufferOwnership::VIEW_OF_SLOT;
      info.parentSlotIdx = i;
      info.dataBuffer = outBuffer;
      if (ownershipArray != nullptr && ownershipArray[i].ownership == BufferOwnership::SLOT_OWNED) {
        ownershipArray[i].addViewRef();
      }
      DSP_DIAG(MEMORY, "OWNERSHIP_CLASSIFY: slot %d → %s (parentSlot=%d)",
               slotIdx, bufferOwnershipName(info.ownership), info.parentSlotIdx);
      return;
    }
  }

  // 4. Buffer is unique to this slot — slot owns it
  info.ownership = BufferOwnership::SLOT_OWNED;
  info.parentSlotIdx = -1;
  info.dataBuffer = outBuffer;

  DSP_DIAG(MEMORY, "OWNERSHIP_CLASSIFY: slot %d → %s",
           slotIdx, bufferOwnershipName(info.ownership));
}

}  // namespace graph
}  // namespace sd
