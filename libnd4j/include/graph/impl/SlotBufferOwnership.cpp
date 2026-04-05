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
#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

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

// ═══════════════════════════════════════════════════════════════════════════════
// BufferPointerSnapshot
// ═══════════════════════════════════════════════════════════════════════════════

void BufferPointerSnapshot::capture(NDArray** outputSlots, int numSlots,
                                     NDArray** externalInputs, int numExt) {
  clear();
  totalSlots = numSlots;
  numExternalInputs = numExt;

#ifdef SD_CUDA
  int currentDevice = -1;
  cudaGetDevice(&currentDevice);
  capturedDeviceId = currentDevice;
#else
  capturedDeviceId = 0;
#endif

  if (numSlots > 0) {
    slotGpuAddresses = new void*[numSlots];
    slotDataBuffers = new DataBuffer*[numSlots];
    slotDeviceIds = new int[numSlots];
    for (int i = 0; i < numSlots; i++) {
      if (outputSlots[i] != nullptr) {
        slotGpuAddresses[i] = outputSlots[i]->specialBuffer();
        slotDataBuffers[i] = outputSlots[i]->dataBuffer();
        slotDeviceIds[i] = outputSlots[i]->dataBuffer() != nullptr
            ? outputSlots[i]->dataBuffer()->deviceId() : -1;
      } else {
        slotGpuAddresses[i] = nullptr;
        slotDataBuffers[i] = nullptr;
        slotDeviceIds[i] = -1;
      }
    }
  }

  if (numExt > 0) {
    extGpuAddresses = new void*[numExt];
    extDataBuffers = new DataBuffer*[numExt];
    extDeviceIds = new int[numExt];
    for (int i = 0; i < numExt; i++) {
      if (externalInputs[i] != nullptr) {
        extGpuAddresses[i] = externalInputs[i]->specialBuffer();
        extDataBuffers[i] = externalInputs[i]->dataBuffer();
        extDeviceIds[i] = externalInputs[i]->dataBuffer() != nullptr
            ? externalInputs[i]->dataBuffer()->deviceId() : -1;
      } else {
        extGpuAddresses[i] = nullptr;
        extDataBuffers[i] = nullptr;
        extDeviceIds[i] = -1;
      }
    }
  }

  valid = true;
}

bool BufferPointerSnapshot::validate(NDArray** outputSlots, int numSlots,
                                      NDArray** externalInputs, int numExt,
                                      char* errMsg, int errMsgLen) const {
  if (!valid) return true;  // No snapshot to compare against

  // Check slot pointers
  int checkSlots = std::min(totalSlots, numSlots);
  for (int i = 0; i < checkSlots; i++) {
    if (slotDataBuffers[i] == nullptr) continue;  // Was null at snapshot time

    if (outputSlots[i] == nullptr) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: slot %d was non-null at freeze (db=%p gpu=%p) "
               "but is NULL now — buffer was freed or slot was cleared",
               i, (void*)slotDataBuffers[i], slotGpuAddresses[i]);
      return false;
    }

    DataBuffer* currentDb = outputSlots[i]->dataBuffer();
    if (currentDb != slotDataBuffers[i]) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: slot %d DataBuffer replaced during frozen execution "
               "(snapshot=%p current=%p) — ownership violated",
               i, (void*)slotDataBuffers[i], (void*)currentDb);
      return false;
    }

    if (currentDb != nullptr && currentDb->isClosed()) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: slot %d DataBuffer %p is CLOSED during frozen execution "
               "— use-after-free will occur on next access",
               i, (void*)currentDb);
      return false;
    }

    void* currentGpu = outputSlots[i]->specialBuffer();
    if (slotGpuAddresses[i] != nullptr && currentGpu != slotGpuAddresses[i]) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: slot %d GPU address changed during frozen execution "
               "(snapshot=%p current=%p) — pointer drift, CUDA graph replay will use stale address",
               i, slotGpuAddresses[i], currentGpu);
      return false;
    }

    // Device ID check: detect buffer migration during frozen execution
    if (slotDeviceIds != nullptr && slotDeviceIds[i] >= 0 && currentDb != nullptr) {
      int currentDevId = currentDb->deviceId();
      if (currentDevId != slotDeviceIds[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d buffer migrated from device %d to device %d "
                 "during frozen execution — CUDA graph captured on device %d will access wrong memory",
                 i, slotDeviceIds[i], currentDevId, capturedDeviceId);
        return false;
      }
    }
  }

  // Check external input pointers
  int checkExt = std::min(numExternalInputs, numExt);
  for (int i = 0; i < checkExt; i++) {
    if (extDataBuffers[i] == nullptr) continue;

    if (externalInputs[i] == nullptr) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: external input %d was non-null at freeze but is NULL now "
               "— weight/constant was freed",
               i);
      return false;
    }

    DataBuffer* currentDb = externalInputs[i]->dataBuffer();
    if (currentDb != nullptr && currentDb->isClosed()) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: external input %d DataBuffer %p is CLOSED during frozen "
               "execution — model weight/constant was freed",
               i, (void*)currentDb);
      return false;
    }

    void* currentGpu = externalInputs[i]->specialBuffer();
    if (extGpuAddresses[i] != nullptr && currentGpu != extGpuAddresses[i]) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: external input %d GPU address changed during frozen "
               "execution (snapshot=%p current=%p) — weight buffer migrated or reallocated",
               i, extGpuAddresses[i], currentGpu);
      return false;
    }

    // Device ID check: detect weight migration during frozen execution
    if (extDeviceIds != nullptr && extDeviceIds[i] >= 0 && currentDb != nullptr) {
      int currentDevId = currentDb->deviceId();
      if (currentDevId != extDeviceIds[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d buffer migrated from device %d to device %d "
                 "during frozen execution — frozen refs should have prevented migration",
                 i, extDeviceIds[i], currentDevId);
        return false;
      }
    }
  }

  // Execution device check: validate current device matches capture device
#ifdef SD_CUDA
  if (capturedDeviceId >= 0) {
    int currentDevice = -1;
    cudaGetDevice(&currentDevice);
    if (currentDevice != capturedDeviceId) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: execution device changed from %d (at capture) to %d "
               "— CUDA graphs captured on device %d cannot replay on device %d",
               capturedDeviceId, currentDevice, capturedDeviceId, currentDevice);
      return false;
    }
  }
#endif

  return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase-Aware Lifecycle Validation
// ═══════════════════════════════════════════════════════════════════════════════

bool validateLifecycleForPhase(
    int planPhase,
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers,
    const BufferPointerSnapshot* snapshot,
    char* errMsg, int errMsgLen) {

  // ── Level 0: SLOT_BY_SLOT — ownership consistency ──
  if (ownership != nullptr && outputSlots != nullptr) {
    for (int i = 0; i < totalSlots; i++) {
      const auto& info = ownership[i];
      if (info.ownership == BufferOwnership::UNSET) continue;

      // VIEW_OF_SLOT must have valid parent
      if (info.ownership == BufferOwnership::VIEW_OF_SLOT) {
        if (info.parentSlotIdx < 0 || info.parentSlotIdx >= totalSlots) {
          snprintf(errMsg, errMsgLen,
                   "LIFECYCLE_ERROR: slot %d is VIEW_OF_SLOT but parentSlotIdx=%d is invalid "
                   "(totalSlots=%d)",
                   i, info.parentSlotIdx, totalSlots);
          return false;
        }
        // Parent must still exist and own a buffer
        if (outputSlots[info.parentSlotIdx] == nullptr) {
          snprintf(errMsg, errMsgLen,
                   "LIFECYCLE_ERROR: slot %d is VIEW_OF_SLOT(parent=%d) but parent slot "
                   "is NULL — parent was freed while view still alive (dangling view)",
                   i, info.parentSlotIdx);
          return false;
        }
      }

      // DataBuffer identity check: tracked buffer must match actual.
      // If this fires, a slot's output array was replaced without updating
      // ownership. Find the allocation site and add re-classification there.
      if (info.dataBuffer != nullptr && outputSlots[i] != nullptr) {
        DataBuffer* actualDb = outputSlots[i]->dataBuffer();
        if (actualDb != info.dataBuffer) {
          snprintf(errMsg, errMsgLen,
                   "LIFECYCLE_ERROR: slot %d ownership tracks DataBuffer %p but actual is %p "
                   "— stale ownership, slot was replaced without re-classification",
                   i, (void*)info.dataBuffer, (void*)actualDb);
          return false;
        }
      }
    }
  }

  // ── Level 1: SHAPES_FROZEN — no closed DataBuffers ──
  if (planPhase >= 1) {  // SHAPES_FROZEN
    // Check output slots for closed buffers
    if (outputSlots != nullptr) {
      for (int i = 0; i < totalSlots; i++) {
        if (outputSlots[i] == nullptr) continue;
        DataBuffer* db = outputSlots[i]->dataBuffer();
        if (db != nullptr && db->isClosed()) {
          snprintf(errMsg, errMsgLen,
                   "LIFECYCLE_ERROR: slot %d has CLOSED DataBuffer %p during SHAPES_FROZEN+ "
                   "phase — buffer was freed while shapes assumed stable",
                   i, (void*)db);
          return false;
        }
      }
    }

    // Check external inputs for closed buffers
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] == nullptr) continue;
      DataBuffer* db = externalInputs[i]->dataBuffer();
      if (db != nullptr && db->isClosed()) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d has CLOSED DataBuffer %p during "
                 "SHAPES_FROZEN+ phase — weight/constant was freed",
                 i, (void*)db);
        return false;
      }
    }

    // Check that no protected weight buffer was closed
    for (auto* db : protectedWeightBuffers) {
      if (db != nullptr && db->isClosed()) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: protected weight DataBuffer %p is CLOSED during "
                 "SHAPES_FROZEN+ phase — model weight was freed while plan active",
                 (void*)db);
        return false;
      }
    }
  }

  // ── Level 2: POINTERS_STABLE — buffer addresses match snapshot ──
  if (planPhase >= 2 && snapshot != nullptr) {  // POINTERS_STABLE
    if (!snapshot->validate(outputSlots, totalSlots,
                            externalInputs, numExternalInputs,
                            errMsg, errMsgLen)) {
      return false;  // errMsg already populated by snapshot->validate()
    }
  }

  return true;
}

}  // namespace graph
}  // namespace sd
