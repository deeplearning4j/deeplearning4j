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
#include <graph/DspDeviceDispatch.h>

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

  // Only scan earlier slots — scanning future slots risks stale alias inversion.
  for (int i = 0; i < slotIdx && i < totalOutputSlots; i++) {
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
    case BufferOwnership::WORKSPACE:      return "WORKSPACE";
    default:                              return "UNKNOWN";
  }
}

void resetSlotBufferOwnership(
    SlotBufferInfo* ownershipArray, int totalSlots, int slotIdx) {
  if (ownershipArray == nullptr || slotIdx < 0 || slotIdx >= totalSlots) return;

  auto& info = ownershipArray[slotIdx];
  if (info.ownership == BufferOwnership::VIEW_OF_SLOT &&
      info.parentSlotIdx >= 0 && info.parentSlotIdx < totalSlots) {
    auto& parent = ownershipArray[info.parentSlotIdx];
    if (parent.viewRefCount > 0) {
      parent.removeViewRef();
    }
  }
  info.reset();
}

void classifyAndUpdateOwnership(
    SlotBufferInfo& info,
    NDArray* outArray,
    int slotIdx,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    SlotBufferInfo* ownershipArray) {

  // Remember previous ownership for change detection, then detach any prior
  // parent relation before installing the new classification.
  BufferOwnership prevOwnership = info.ownership;
  DataBuffer* prevDataBuffer = info.dataBuffer;
  int prevParentSlotIdx = info.parentSlotIdx;
  if (ownershipArray != nullptr) {
    resetSlotBufferOwnership(ownershipArray, totalOutputSlots, slotIdx);
  } else {
    info.reset();
  }

  // 1. Null output or null dataBuffer → UNSET
  if (outArray == nullptr || outArray->dataBuffer() == nullptr) {
    return;
  }

  DataBuffer* outBuffer = outArray->dataBuffer();

  // 2. Check external inputs — if buffer matches, it's a view of a weight/constant
  //    (or a view of a placeholder; pruneTransientViewSlots uses
  //    protectedWeightBuffers to distinguish the two cases).
  for (int i = 0; i < numExternalInputs; i++) {
    if (externalInputs[i] != nullptr && externalInputs[i]->dataBuffer() == outBuffer) {
      info.ownership = BufferOwnership::VIEW_OF_WEIGHT;
      info.dataBuffer = outBuffer;
      DSP_DIAG(MEMORY, "OWNERSHIP_CLASSIFY: slot %d → %s (extIdx=%d)",
               slotIdx, bufferOwnershipName(info.ownership), i);
      return;
    }
  }

  // 3. Check earlier output slots only. Later slots may still contain stale
  // wrappers from the previous execution and must never become parents of the
  // slot currently being classified.
  for (int i = 0; i < slotIdx && i < totalOutputSlots; i++) {
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

  // Ownership change detection — log transitions that indicate buffer lifecycle events
  if (prevOwnership != BufferOwnership::UNSET && prevOwnership != info.ownership) {
    DSP_DIAG(MEMORY,
             "OWNERSHIP_CHANGE: slot %d %s → %s (prevDb=%p newDb=%p prevParent=%d newParent=%d)",
             slotIdx, bufferOwnershipName(prevOwnership), bufferOwnershipName(info.ownership),
             (void*)prevDataBuffer, (void*)info.dataBuffer,
             prevParentSlotIdx, info.parentSlotIdx);
  } else if (prevDataBuffer != nullptr && prevDataBuffer != info.dataBuffer) {
    DSP_DIAG(MEMORY,
             "OWNERSHIP_BUFFER_SWAP: slot %d ownership=%s unchanged but DataBuffer changed "
             "(prev=%p new=%p) — buffer replaced in-place",
             slotIdx, bufferOwnershipName(info.ownership),
             (void*)prevDataBuffer, (void*)info.dataBuffer);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BufferPointerSnapshot
// ═══════════════════════════════════════════════════════════════════════════════

void BufferPointerSnapshot::capture(NDArray** outputSlots, int numSlots,
                                     NDArray** externalInputs, int numExt) {
  clear();
  totalSlots = numSlots;
  numExternalInputs = numExt;

  capturedDeviceId = dspGetCurrentDevice();
  if (capturedDeviceId < 0) capturedDeviceId = 0;  // CPU default

  if (numSlots > 0) {
    slotGpuAddresses = new void*[numSlots];
    slotDataBuffers = new DataBuffer*[numSlots];
    slotDeviceIds = new int[numSlots];
    slotPrimaryAddresses = new void*[numSlots];
    slotShapeInfoAddresses = new LongType*[numSlots];
    slotNDArrayIdentity = new NDArray*[numSlots];
    slotBufferOffsets = new LongType[numSlots];
    slotLengths = new LongType[numSlots];
    slotActualityFlags = new uint8_t[numSlots];
    slotOrderings = new char[numSlots];
    for (int i = 0; i < numSlots; i++) {
      if (outputSlots[i] != nullptr) {
        DataBuffer* db = outputSlots[i]->dataBuffer();
        slotGpuAddresses[i] = outputSlots[i]->specialBuffer();
        slotDataBuffers[i] = db;
        slotDeviceIds[i] = db != nullptr ? db->deviceId() : -1;
        slotPrimaryAddresses[i] = db != nullptr ? db->primary() : nullptr;
        slotShapeInfoAddresses[i] = outputSlots[i]->specialShapeInfo();
        slotNDArrayIdentity[i] = outputSlots[i];
        slotBufferOffsets[i] = outputSlots[i]->offset();
        slotLengths[i] = outputSlots[i]->lengthOf();
        uint8_t flags = 0;
        if (db != nullptr) {
          if (db->isPrimaryActual()) flags |= 1;
          if (db->isSpecialActual()) flags |= 2;
        }
        slotActualityFlags[i] = flags;
        slotOrderings[i] = outputSlots[i]->ordering();
      } else {
        slotGpuAddresses[i] = nullptr;
        slotDataBuffers[i] = nullptr;
        slotDeviceIds[i] = -1;
        slotPrimaryAddresses[i] = nullptr;
        slotShapeInfoAddresses[i] = nullptr;
        slotNDArrayIdentity[i] = nullptr;
        slotBufferOffsets[i] = 0;
        slotLengths[i] = 0;
        slotActualityFlags[i] = 0;
        slotOrderings[i] = 0;
      }
    }
  }

  if (numExt > 0) {
    extGpuAddresses = new void*[numExt];
    extDataBuffers = new DataBuffer*[numExt];
    extDeviceIds = new int[numExt];
    extPrimaryAddresses = new void*[numExt];
    extShapeInfoAddresses = new LongType*[numExt];
    extNDArrayIdentity = new NDArray*[numExt];
    extBufferOffsets = new LongType[numExt];
    extLengths = new LongType[numExt];
    extActualityFlags = new uint8_t[numExt];
    extOrderings = new char[numExt];
    for (int i = 0; i < numExt; i++) {
      if (externalInputs[i] != nullptr) {
        DataBuffer* db = externalInputs[i]->dataBuffer();
        extGpuAddresses[i] = externalInputs[i]->specialBuffer();
        extDataBuffers[i] = db;
        extDeviceIds[i] = db != nullptr ? db->deviceId() : -1;
        extPrimaryAddresses[i] = db != nullptr ? db->primary() : nullptr;
        extShapeInfoAddresses[i] = externalInputs[i]->specialShapeInfo();
        extNDArrayIdentity[i] = externalInputs[i];
        extBufferOffsets[i] = externalInputs[i]->offset();
        extLengths[i] = externalInputs[i]->lengthOf();
        uint8_t flags = 0;
        if (db != nullptr) {
          if (db->isPrimaryActual()) flags |= 1;
          if (db->isSpecialActual()) flags |= 2;
        }
        extActualityFlags[i] = flags;
        extOrderings[i] = externalInputs[i]->ordering();
      } else {
        extGpuAddresses[i] = nullptr;
        extDataBuffers[i] = nullptr;
        extDeviceIds[i] = -1;
        extPrimaryAddresses[i] = nullptr;
        extShapeInfoAddresses[i] = nullptr;
        extNDArrayIdentity[i] = nullptr;
        extBufferOffsets[i] = 0;
        extLengths[i] = 0;
        extActualityFlags[i] = 0;
        extOrderings[i] = 0;
      }
    }
  }

  valid = true;
}

void BufferPointerSnapshot::pruneTransientViewSlots(
    const SlotBufferInfo* ownership,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers,
    NDArray** externalInputs, int numExternalInputs) {
  if (!valid || ownership == nullptr) return;
  if (slotDataBuffers == nullptr) return;

  // Build set of non-protected (placeholder) external input DataBuffers.
  // Used to catch slots whose DataBuffer matches a placeholder even when
  // ownership is SLOT_OWNED (e.g. reshape output that sometimes creates a
  // new buffer and sometimes reuses the placeholder's buffer).
  std::unordered_set<DataBuffer*> placeholderBuffers;
  if (externalInputs != nullptr) {
    for (int e = 0; e < numExternalInputs; e++) {
      if (externalInputs[e] == nullptr) {
        DSP_DIAG(MEMORY, "pruneTransientViewSlots: ext[%d] NDArray=null", e);
        continue;
      }
      DataBuffer* edb = externalInputs[e]->dataBuffer();
      bool isProtected = (edb != nullptr && protectedWeightBuffers.count(edb) > 0);
      DSP_DIAG(MEMORY, "pruneTransientViewSlots: ext[%d] NDArray=%p db=%p isProtected=%d",
               e, (void*)externalInputs[e], (void*)edb, isProtected ? 1 : 0);
      if (edb != nullptr && !isProtected) {
        placeholderBuffers.insert(edb);
      }
    }
  }

  DSP_DIAG(MEMORY,
           "pruneTransientViewSlots: totalSlots=%d numExtInputs=%d "
           "protectedWeightBuffers=%d placeholderBuffers=%d",
           totalSlots, numExternalInputs,
           static_cast<int>(protectedWeightBuffers.size()),
           static_cast<int>(placeholderBuffers.size()));

  int pruned = 0;
  for (int i = 0; i < totalSlots; i++) {
    DataBuffer* db = slotDataBuffers[i];
    if (db == nullptr) continue;

    bool isPlaceholderView = false;
    if (ownership[i].ownership == BufferOwnership::VIEW_OF_WEIGHT) {
      // Classic path: classified as view of external input, check if non-protected
      isPlaceholderView = (protectedWeightBuffers.count(db) == 0);
    } else if (!placeholderBuffers.empty() && placeholderBuffers.count(db) > 0) {
      // Ownership may be SLOT_OWNED but the DataBuffer is actually a
      // placeholder's buffer (reshape/view ops oscillate between SLOT_OWNED
      // and VIEW_OF_WEIGHT depending on execution timing).
      isPlaceholderView = true;
    }

    if (i == 0) {
      DSP_DIAG(MEMORY,
               "pruneTransientViewSlots: slot 0 db=%p ownership=%d "
               "isPlaceholderView=%d protectedCount=%d placeholderCount=%d",
               (void*)db, static_cast<int>(ownership[i].ownership),
               isPlaceholderView ? 1 : 0,
               static_cast<int>(protectedWeightBuffers.count(db)),
               static_cast<int>(placeholderBuffers.count(db)));
    }

    if (!isPlaceholderView) continue;

    // Transient placeholder view — null out so later snapshot->validate()
    // calls naturally skip this slot via the existing null-entry guards.
    if (slotGpuAddresses != nullptr) slotGpuAddresses[i] = nullptr;
    slotDataBuffers[i] = nullptr;
    if (slotDeviceIds != nullptr) slotDeviceIds[i] = -1;
    if (slotPrimaryAddresses != nullptr) slotPrimaryAddresses[i] = nullptr;
    if (slotShapeInfoAddresses != nullptr) slotShapeInfoAddresses[i] = nullptr;
    if (slotNDArrayIdentity != nullptr) slotNDArrayIdentity[i] = nullptr;
    if (slotBufferOffsets != nullptr) slotBufferOffsets[i] = 0;
    if (slotLengths != nullptr) slotLengths[i] = 0;
    if (slotActualityFlags != nullptr) slotActualityFlags[i] = 0;
    if (slotOrderings != nullptr) slotOrderings[i] = 0;
    ++pruned;
  }

  if (pruned > 0) {
    DSP_DIAG(MEMORY,
             "BufferPointerSnapshot::pruneTransientViewSlots: nullified %d "
             "slot(s) sharing DataBuffers with non-protected (placeholder) "
             "external inputs — those slots are refreshed by slot-exec each call",
             pruned);
  }
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

    // "is-closed" check lives in validateLifecycleForPhase() which has the
    // protectedWeightBuffers context needed to avoid false positives.

    void* currentGpu = outputSlots[i]->specialBuffer();
    if (slotGpuAddresses[i] != nullptr && currentGpu != slotGpuAddresses[i]) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: slot %d GPU address changed during frozen execution "
               "(snapshot=%p current=%p) — pointer drift, CUDA graph replay will use stale address",
               i, slotGpuAddresses[i], currentGpu);
      return false;
    }

    // Host (primary) buffer address check: detect host reallocation
    if (slotPrimaryAddresses != nullptr && slotPrimaryAddresses[i] != nullptr) {
      void* currentPrimary = currentDb != nullptr ? currentDb->primary() : nullptr;
      if (currentPrimary != slotPrimaryAddresses[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d primary (host) address changed during frozen execution "
                 "(snapshot=%p current=%p) — host buffer reallocated",
                 i, slotPrimaryAddresses[i], currentPrimary);
        return false;
      }
    }

    // Device-side shape info pointer check: detect shape re-registration
    if (slotShapeInfoAddresses != nullptr && slotShapeInfoAddresses[i] != nullptr) {
      LongType* currentShapeInfo = outputSlots[i]->specialShapeInfo();
      if (currentShapeInfo != slotShapeInfoAddresses[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d device shape info pointer changed during frozen execution "
                 "(snapshot=%p current=%p) — shape info re-registered",
                 i, (void*)slotShapeInfoAddresses[i], (void*)currentShapeInfo);
        return false;
      }
    }

    // NDArray wrapper identity check: detect wrapper replacement
    if (slotNDArrayIdentity != nullptr && slotNDArrayIdentity[i] != nullptr) {
      if (outputSlots[i] != slotNDArrayIdentity[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d NDArray wrapper identity changed during frozen execution "
                 "(snapshot=%p current=%p) — slot wrapper replaced",
                 i, (void*)slotNDArrayIdentity[i], (void*)outputSlots[i]);
        return false;
      }
    }

    // Buffer offset check: detect view base move
    if (slotBufferOffsets != nullptr) {
      LongType currentOffset = outputSlots[i]->offset();
      if (currentOffset != slotBufferOffsets[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d buffer offset changed during frozen execution "
                 "(snapshot=%lld current=%lld) — view base moved",
                 i, (long long)slotBufferOffsets[i], (long long)currentOffset);
        return false;
      }
    }

    // Length check: detect shape mutation (should be impossible under frozen shapes)
    if (slotLengths != nullptr) {
      LongType currentLength = outputSlots[i]->lengthOf();
      if (currentLength != slotLengths[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d lengthOf changed during frozen execution "
                 "(snapshot=%lld current=%lld) — shape mutation under frozen plan",
                 i, (long long)slotLengths[i], (long long)currentLength);
        return false;
      }
    }

    // Actuality flag check: flag only dangerous transitions (device lost
    // authoritative data, or host overwrote device data). Benign transitions
    // like 0x03→0x02 (host stale, device still authoritative) pass.
    if (slotActualityFlags != nullptr && currentDb != nullptr) {
      uint8_t currentFlags = 0;
      if (currentDb->isPrimaryActual()) currentFlags |= 1;
      if (currentDb->isSpecialActual()) currentFlags |= 2;
      bool snapSAct = (slotActualityFlags[i] & 2) != 0;
      bool curSAct = (currentFlags & 2) != 0;
      bool curPAct = (currentFlags & 1) != 0;
      if ((snapSAct && !curSAct) || (!curSAct && curPAct && snapSAct)) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d actuality flags changed during frozen execution "
                 "(snapshot=0x%02x current=0x%02x) — device data invalidated or host overwrote device",
                 i, (unsigned)slotActualityFlags[i], (unsigned)currentFlags);
        return false;
      }
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

    // Memory ordering check: detect layout flip (c↔f reorder)
    if (slotOrderings != nullptr && slotOrderings[i] != 0) {
      char currentOrdering = outputSlots[i]->ordering();
      if (currentOrdering != slotOrderings[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: slot %d memory ordering changed during frozen execution "
                 "(snapshot='%c' current='%c') — stride layout mutated under frozen plan",
                 i, slotOrderings[i], currentOrdering);
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

    // DataBuffer identity check: ext input DataBuffer must not be replaced
    if (currentDb != extDataBuffers[i]) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: external input %d DataBuffer replaced during frozen execution "
               "(snapshot=%p current=%p) — weight/constant wrapper swapped",
               i, (void*)extDataBuffers[i], (void*)currentDb);
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

    // Host (primary) buffer address check: detect host reallocation
    if (extPrimaryAddresses != nullptr && extPrimaryAddresses[i] != nullptr) {
      void* currentPrimary = currentDb != nullptr ? currentDb->primary() : nullptr;
      if (currentPrimary != extPrimaryAddresses[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d primary (host) address changed during frozen "
                 "execution (snapshot=%p current=%p) — weight host buffer reallocated",
                 i, extPrimaryAddresses[i], currentPrimary);
        return false;
      }
    }

    // Device-side shape info pointer check: detect shape re-registration
    if (extShapeInfoAddresses != nullptr && extShapeInfoAddresses[i] != nullptr) {
      LongType* currentShapeInfo = externalInputs[i]->specialShapeInfo();
      if (currentShapeInfo != extShapeInfoAddresses[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d device shape info pointer changed during frozen "
                 "execution (snapshot=%p current=%p) — shape info re-registered",
                 i, (void*)extShapeInfoAddresses[i], (void*)currentShapeInfo);
        return false;
      }
    }

    // NDArray wrapper identity check: detect wrapper replacement
    if (extNDArrayIdentity != nullptr && extNDArrayIdentity[i] != nullptr) {
      if (externalInputs[i] != extNDArrayIdentity[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d NDArray wrapper identity changed during frozen "
                 "execution (snapshot=%p current=%p) — ext input wrapper replaced",
                 i, (void*)extNDArrayIdentity[i], (void*)externalInputs[i]);
        return false;
      }
    }

    // Buffer offset check: detect view base move
    if (extBufferOffsets != nullptr) {
      LongType currentOffset = externalInputs[i]->offset();
      if (currentOffset != extBufferOffsets[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d buffer offset changed during frozen execution "
                 "(snapshot=%lld current=%lld) — view base moved",
                 i, (long long)extBufferOffsets[i], (long long)currentOffset);
        return false;
      }
    }

    // Length check: detect shape mutation (should be impossible under frozen shapes)
    if (extLengths != nullptr) {
      LongType currentLength = externalInputs[i]->lengthOf();
      if (currentLength != extLengths[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d lengthOf changed during frozen execution "
                 "(snapshot=%lld current=%lld) — shape mutation under frozen plan",
                 i, (long long)extLengths[i], (long long)currentLength);
        return false;
      }
    }

    // Actuality flag check: only flag the two dangerous transitions.
    // Matches detectStaleActualityTransitions(); benign transitions like
    // 0x03 -> 0x02 (host copy stale, device still authoritative) pass.
    if (extActualityFlags != nullptr && currentDb != nullptr) {
      uint8_t currentFlags = 0;
      if (currentDb->isPrimaryActual()) currentFlags |= 1;
      if (currentDb->isSpecialActual()) currentFlags |= 2;
      bool snapSAct = (extActualityFlags[i] & 2) != 0;
      bool curSAct = (currentFlags & 2) != 0;
      bool curPAct = (currentFlags & 1) != 0;
      if ((snapSAct && !curSAct) || (!curSAct && curPAct && snapSAct)) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d actuality flags changed during frozen execution "
                 "(snapshot=0x%02x current=0x%02x) — device data invalidated or host overwrote device",
                 i, (unsigned)extActualityFlags[i], (unsigned)currentFlags);
        return false;
      }
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

    // Memory ordering check: detect layout flip (c↔f reorder)
    if (extOrderings != nullptr && extOrderings[i] != 0) {
      char currentOrdering = externalInputs[i]->ordering();
      if (currentOrdering != extOrderings[i]) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: external input %d memory ordering changed during frozen execution "
                 "(snapshot='%c' current='%c') — stride layout mutated under frozen plan",
                 i, extOrderings[i], currentOrdering);
        return false;
      }
    }
  }

  // Execution device check: validate current device matches capture device
  if (capturedDeviceId >= 0) {
    int currentDevice = dspGetCurrentDevice();
    if (currentDevice >= 0 && currentDevice != capturedDeviceId) {
      snprintf(errMsg, errMsgLen,
               "LIFECYCLE_ERROR: execution device changed from %d (at capture) to %d "
               "— CUDA graphs captured on device %d cannot replay on device %d",
               capturedDeviceId, currentDevice, capturedDeviceId, currentDevice);
      return false;
    }
  }

  return true;
}

void BufferPointerSnapshot::detectStaleActualityTransitions(
    NDArray** outputSlots, int numSlots,
    NDArray** externalInputs, int numExt,
    int planPhase) const {
  if (!valid) return;
  if (planPhase < 2) return;  // Only check from REPLAYING (2) onward

  int checkSlots = std::min(totalSlots, numSlots);
  for (int i = 0; i < checkSlots; i++) {
    if (slotDataBuffers[i] == nullptr || outputSlots[i] == nullptr) continue;
    DataBuffer* currentDb = outputSlots[i]->dataBuffer();
    if (currentDb == nullptr) continue;

    uint8_t snapFlags = slotActualityFlags[i];
    uint8_t currentFlags = 0;
    if (currentDb->isPrimaryActual()) currentFlags |= 1;
    if (currentDb->isSpecialActual()) currentFlags |= 2;

    bool snapSAct = (snapFlags & 2) != 0;
    bool curSAct = (currentFlags & 2) != 0;
    bool curPAct = (currentFlags & 1) != 0;

    // sAct was set at capture → now cleared: device data was invalidated
    if (snapSAct && !curSAct) {
      DSP_THROW(MEMORY,
               "STALE_DATA: slot %d lost isSpecialActual during frozen phase %d "
               "(snapshot=0x%02x current=0x%02x) — device buffer was invalidated without a known writer",
               i, planPhase, (unsigned)snapFlags, (unsigned)currentFlags);
    }

    // pAct appeared without sAct: host wrote to a device-authoritative buffer
    if (!curSAct && curPAct && snapSAct) {
      DSP_THROW(MEMORY,
               "STALE_DATA: slot %d has pAct=1 sAct=0 during frozen phase %d "
               "(snapshot=0x%02x current=0x%02x) — host write overwrote device-authoritative data",
               i, planPhase, (unsigned)snapFlags, (unsigned)currentFlags);
    }
  }

  int checkExt = std::min(numExternalInputs, numExt);
  for (int i = 0; i < checkExt; i++) {
    if (extDataBuffers[i] == nullptr || externalInputs[i] == nullptr) continue;
    DataBuffer* currentDb = externalInputs[i]->dataBuffer();
    if (currentDb == nullptr) continue;

    uint8_t snapFlags = extActualityFlags[i];
    uint8_t currentFlags = 0;
    if (currentDb->isPrimaryActual()) currentFlags |= 1;
    if (currentDb->isSpecialActual()) currentFlags |= 2;

    bool snapSAct = (snapFlags & 2) != 0;
    bool curSAct = (currentFlags & 2) != 0;
    bool curPAct = (currentFlags & 1) != 0;

    if (snapSAct && !curSAct) {
      DSP_THROW(MEMORY,
               "STALE_DATA: ext input %d lost isSpecialActual during frozen phase %d "
               "(snapshot=0x%02x current=0x%02x) — device buffer was invalidated without a known writer",
               i, planPhase, (unsigned)snapFlags, (unsigned)currentFlags);
    }

    if (!curSAct && curPAct && snapSAct) {
      DSP_THROW(MEMORY,
               "STALE_DATA: ext input %d has pAct=1 sAct=0 during frozen phase %d "
               "(snapshot=0x%02x current=0x%02x) — host write overwrote device-authoritative data",
               i, planPhase, (unsigned)snapFlags, (unsigned)currentFlags);
    }
  }
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
          // A slot whose underlying buffer is NOT in protectedWeightBuffers
          // is not a true weight — it is either a view over a placeholder
          // external input, or a SLOT_OWNED buffer that tracked a transient
          // wrapper whose underlying memory was reclaimed (e.g., classifier
          // saw separate DataBuffer* wrappers for view ops that nonetheless
          // share the placeholder's memory).  In both cases the stale wrapper
          // will be refreshed by the imminent slot-exec re-execution of the
          // producing op.  Only protected weight/constant buffers must never
          // be closed during a frozen phase — that is a true use-after-free.
          if (protectedWeightBuffers.count(db) == 0) {
            continue;
          }
          snprintf(errMsg, errMsgLen,
                   "LIFECYCLE_ERROR: slot %d has CLOSED protected weight DataBuffer %p during "
                   "SHAPES_FROZEN+ phase — model weight was freed while plan active",
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

  // ── Level 2: REPLAYING — snapshot drift detection ──
  // Buffer identity/address checks only become authoritative once the plan has
  // observed stable pointers across replay-eligible segments. SHAPES_FROZEN
  // still allows legitimate buffer churn while the plan converges on stable
  // steady-state allocations.
  if (planPhase >= 2 && snapshot != nullptr) {  // REPLAYING
    if (!snapshot->validate(outputSlots, totalSlots,
                            externalInputs, numExternalInputs,
                            errMsg, errMsgLen)) {
      return false;  // errMsg already populated by snapshot->validate()
    }
  }

  // ── Level 2: REPLAYING — device sync state invariant ──
  // In frozen steady-state the plan skips prepareSpecialUse/registerSpecialUse,
  // so every SLOT_OWNED buffer must remain device-authoritative (sAct=true).
  // If pAct=1 && sAct=0, a host write poisoned the device state.
  // Only SLOT_OWNED buffers are checked — views/weights have external owners.
  // Host-only builds have no separate device-actuality domain.
  if (dspHasDeviceMemory() && planPhase >= 2 && ownership != nullptr && outputSlots != nullptr) {
    for (int i = 0; i < totalSlots; i++) {
      const auto& info = ownership[i];
      if (info.ownership != BufferOwnership::SLOT_OWNED) continue;
      if (outputSlots[i] == nullptr) continue;
      DataBuffer* db = outputSlots[i]->dataBuffer();
      if (db == nullptr || db->isClosed()) continue;
      if (db->getLenInBytes() == 0) continue;  // empty/placeholder — no sync state

      bool sAct = db->isSpecialActual();
      bool pAct = db->isPrimaryActual();

      // Trace every SLOT_OWNED slot's flags — one line per slot at MEMORY level.
      DSP_DIAG(MEMORY,
               "FROZEN_SYNC_STATE: slot %d db=%p len=%lld pAct=%d sAct=%d phase=%d",
               i, (void*)db, (long long)db->getLenInBytes(),
               (int)pAct, (int)sAct, planPhase);

      // Device-stale: pAct flipped to 1 but sAct=0 → host wrote over device.
      if (pAct && !sAct) {
        snprintf(errMsg, errMsgLen,
                 "LIFECYCLE_ERROR: SLOT_OWNED slot %d has pAct=1 sAct=0 at SHAPES_FROZEN+ "
                 "(db=%p len=%lld) — host wrote over device-authoritative buffer between "
                 "frozen executions. Next op will syncToDevice with stale host data.",
                 i, (void*)db, (long long)db->getLenInBytes());
        return false;
      }

      // Both zero: buffer was never written at all. Legitimate for slots that
      // prezeroSegmentOutputs cleared but weren't yet executed in this step —
      // do not fail, just trace. A bad sync-skip interaction would show up as
      // pAct=1/sAct=0 above, not here.
    }
  }

  return true;
}

bool validateViewRefCounts(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots) {

  if (ownership == nullptr || outputSlots == nullptr) return true;

  bool valid = true;

  for (int i = 0; i < totalSlots; i++) {
    const auto& info = ownership[i];
    if (info.ownership != BufferOwnership::SLOT_OWNED) continue;

    // Count actual VIEW_OF_SLOT children pointing at this slot
    int actualViewCount = 0;
    for (int j = 0; j < totalSlots; j++) {
      if (j == i) continue;
      if (ownership[j].ownership == BufferOwnership::VIEW_OF_SLOT &&
          ownership[j].parentSlotIdx == i) {
        actualViewCount++;
      }
    }

    if (actualViewCount != info.viewRefCount) {
      DSP_DIAG(MEMORY,
               "VIEW_REF_MISMATCH: slot %d (SLOT_OWNED) has viewRefCount=%d "
               "but %d VIEW_OF_SLOT children actually reference it "
               "— refcount leak or missed removeViewRef()",
               i, info.viewRefCount, actualViewCount);
      valid = false;
    }
  }

  // Check for VIEW_OF_SLOT entries whose parent is no longer SLOT_OWNED
  for (int i = 0; i < totalSlots; i++) {
    const auto& info = ownership[i];
    if (info.ownership != BufferOwnership::VIEW_OF_SLOT) continue;
    if (info.parentSlotIdx < 0 || info.parentSlotIdx >= totalSlots) {
      DSP_DIAG(MEMORY,
               "VIEW_DANGLING_PARENT: slot %d is VIEW_OF_SLOT with invalid "
               "parentSlotIdx=%d (totalSlots=%d)",
               i, info.parentSlotIdx, totalSlots);
      valid = false;
      continue;
    }
    if (ownership[info.parentSlotIdx].ownership != BufferOwnership::SLOT_OWNED) {
      DSP_DIAG(MEMORY,
               "VIEW_ORPHANED: slot %d is VIEW_OF_SLOT(parent=%d) but parent "
               "ownership is %s, not SLOT_OWNED — parent was re-classified "
               "without updating children",
               i, info.parentSlotIdx,
               bufferOwnershipName(ownership[info.parentSlotIdx].ownership));
      valid = false;
    }
  }

  return valid;
}

int detectStaleOwnership(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots) {

  if (ownership == nullptr || outputSlots == nullptr) return 0;

  int staleCount = 0;
  for (int i = 0; i < totalSlots; i++) {
    const auto& info = ownership[i];
    if (info.ownership == BufferOwnership::UNSET) continue;
    if (info.dataBuffer == nullptr) continue;

    if (outputSlots[i] == nullptr) {
      DSP_DIAG(MEMORY,
               "STALE_OWNERSHIP: slot %d ownership=%s tracks DataBuffer %p "
               "but outputSlot is NULL — slot was freed without clearing ownership",
               i, bufferOwnershipName(info.ownership), (void*)info.dataBuffer);
      staleCount++;
      continue;
    }

    DataBuffer* actualDb = outputSlots[i]->dataBuffer();
    if (actualDb != info.dataBuffer) {
      DSP_DIAG(MEMORY,
               "STALE_OWNERSHIP: slot %d ownership=%s tracks DataBuffer %p "
               "but actual is %p — buffer was replaced without re-classification",
               i, bufferOwnershipName(info.ownership),
               (void*)info.dataBuffer, (void*)actualDb);
      staleCount++;
    }
  }

  return staleCount;
}

int detectClosedBuffers(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers) {

  int closedCount = 0;

  // Check output slots. A closed buffer is a true use-after-free ONLY when it is a
  // protected weight/constant. A closed NON-protected buffer is a stale view or a
  // transient wrapper over a placeholder external input — including a view-op slot
  // whose ownership oscillated to SLOT_OWNED when its ext input was swapped+closed
  // between executions — and it is refreshed by the imminent slot-exec re-execution
  // of its producing op. Using protection (not ownership) keeps this consistent with
  // validateLifecycleForPhase's SHAPES_FROZEN rule and avoids false-positive UAFs on
  // stale ext-input views (e.g. squeeze/reshape of a placeholder in REPLAY_BLOCKED).
  if (outputSlots != nullptr) {
    for (int i = 0; i < totalSlots; i++) {
      if (outputSlots[i] == nullptr) continue;
      DataBuffer* db = outputSlots[i]->dataBuffer();
      if (db != nullptr && db->isClosed()) {
        if (protectedWeightBuffers.count(db) == 0) {
          continue;
        }
        const char* ownershipStr = (ownership != nullptr)
            ? bufferOwnershipName(ownership[i].ownership)
            : "unknown";
        DSP_DIAG(MEMORY,
                 "CLOSED_BUFFER: slot %d DataBuffer %p is CLOSED "
                 "(ownership=%s, len=%lld) — protected weight freed (use-after-free)",
                 i, (void*)db, ownershipStr,
                 (long long)db->getLenInBytes());
        closedCount++;
      }
    }
  }

  // Check external inputs
  if (externalInputs != nullptr) {
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] == nullptr) continue;
      DataBuffer* db = externalInputs[i]->dataBuffer();
      if (db != nullptr && db->isClosed()) {
        DSP_DIAG(MEMORY,
                 "CLOSED_BUFFER: external input %d DataBuffer %p is CLOSED "
                 "(len=%lld) — weight/constant was freed while plan active",
                 i, (void*)db, (long long)db->getLenInBytes());
        closedCount++;
      }
    }
  }

  return closedCount;
}

// =============================================================================
// Always-On Array Validity Assertions
// =============================================================================

const char* arrayInvalidReasonName(ArrayInvalidReason reason) {
  switch (reason) {
    case ArrayInvalidReason::VALID:              return "VALID";
    case ArrayInvalidReason::NULL_ARRAY:         return "NULL_ARRAY";
    case ArrayInvalidReason::NULL_SHAPE_INFO:    return "NULL_SHAPE_INFO";
    case ArrayInvalidReason::NULL_DATABUFFER:    return "NULL_DATABUFFER";
    case ArrayInvalidReason::DESTROYED_DATABUFFER: return "DESTROYED_DATABUFFER";
    case ArrayInvalidReason::CLOSED_DATABUFFER:  return "CLOSED_DATABUFFER";
    case ArrayInvalidReason::INVALID_DATABUFFER: return "INVALID_DATABUFFER";
    case ArrayInvalidReason::NULL_GPU_POINTER:   return "NULL_GPU_POINTER";
    default:                                     return "UNKNOWN";
  }
}

ArrayInvalidReason validateArrayForExecution(const NDArray* arr) {
  if (arr == nullptr)
    return ArrayInvalidReason::NULL_ARRAY;

  // NDArray methods lack const qualifiers. Cast once — all reads below are
  // logically const (no sync, no allocation, no mutation).
  NDArray* a = const_cast<NDArray*>(arr);

  // Check shape info validity FIRST using non-throwing accessor.
  // shapeInfo() throws when _shapeInfo is nullptr (destructed/stale NDArray),
  // and isEmpty() calls shapeInfo() internally — both must be avoided
  // until we know the shape info is valid.
  bool validShapeInfo = false;
  try {
    validShapeInfo = a->hasValidShapeInfo();
  } catch (const std::exception& e) {
    DSP_DIAG(MEMORY,
             "ARRAY_INVALID_SHAPE_CHECK_EXCEPTION: arr=%p exception=%s",
             (void*)arr, e.what());
    return ArrayInvalidReason::NULL_SHAPE_INFO;
  } catch (...) {
    DSP_DIAG(MEMORY,
             "ARRAY_INVALID_SHAPE_CHECK_EXCEPTION: arr=%p unknown exception",
             (void*)arr);
    return ArrayInvalidReason::NULL_SHAPE_INFO;
  }
  if (!validShapeInfo)
    return ArrayInvalidReason::NULL_SHAPE_INFO;

  // Empty arrays (scalar-empty, rank-0-empty, etc.) have no DataBuffer by design.
  // Nd4j.empty() creates arrays with null data — always valid.
  // Safe to call now because hasValidShapeInfo() passed.
  if (a->isEmpty())
    return ArrayInvalidReason::VALID;

  DataBuffer* db = a->dataBuffer();
  if (db == nullptr)
    return ArrayInvalidReason::NULL_DATABUFFER;

  // Check in order of severity: use-after-free > closed > corrupt
  if (db->isDestroyed())
    return ArrayInvalidReason::DESTROYED_DATABUFFER;

  if (db->isClosed())
    return ArrayInvalidReason::CLOSED_DATABUFFER;

  if (!db->isValid())
    return ArrayInvalidReason::INVALID_DATABUFFER;

  // On device backends, non-empty arrays must have a device buffer allocated.
  // special() is a plain pointer read — no sync, no side effects.
  if (sd::graph::dspHasDeviceMemory() && db->special() == nullptr && a->lengthOf() > 0)
    return ArrayInvalidReason::NULL_GPU_POINTER;

  return ArrayInvalidReason::VALID;
}

int validateSlotInputs(
    int slotIdx, const char* opName,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const int* inputSourceIndices, int numInputs,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen) {

  if (inputSourceIndices == nullptr || numInputs <= 0) return 0;

  int invalidCount = 0;
  bool firstError = true;

  for (int i = 0; i < numInputs; i++) {
    int srcIdx = inputSourceIndices[i];
    NDArray* input = nullptr;
    const char* sourceKind = "unknown";
    int resolvedIdx = -1;

    if (srcIdx >= 0) {
      // Slot-sourced input
      if (srcIdx < totalOutputSlots) {
        input = outputSlots[srcIdx];
      }
      sourceKind = "slot";
      resolvedIdx = srcIdx;
    } else {
      // External input (negative index encoding: extIdx = -(srcIdx + 1))
      int extIdx = -(srcIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs) {
        input = externalInputs[extIdx];
      }
      sourceKind = "ext";
      resolvedIdx = -(srcIdx + 1);
    }

    // Null inputs are allowed for some ops (optional inputs).
    // Only validate non-null inputs.
    if (input == nullptr) continue;

    ArrayInvalidReason reason = validateArrayForExecution(input);
    if (reason != ArrayInvalidReason::VALID) {
      invalidCount++;
      if (firstError && errMsg != nullptr && errMsgLen > 0) {
        firstError = false;
        DataBuffer* db = nullptr;
        if (reason != ArrayInvalidReason::NULL_ARRAY &&
            reason != ArrayInvalidReason::NULL_SHAPE_INFO) {
          db = input->dataBuffer();
        }
        snprintf(errMsg, errMsgLen,
                 "ARRAY_INVALID at slot %d input %d (op=%s): reason=%s "
                 "source=%s[%d] DataBuffer=%p "
                 "exec=%d phase=%d",
                 slotIdx, i, opName ? opName : "?",
                 arrayInvalidReasonName(reason),
                 sourceKind, resolvedIdx,
                 (void*)db,
                 executeCount, planPhase);
      }
      // Log every invalid input (not just the first) via DSP_DIAG
      DSP_DIAG(MEMORY,
               "ARRAY_INVALID_INPUT: slot=%d input=%d op=%s reason=%s "
               "source=%s[%d] exec=%d phase=%d",
               slotIdx, i, opName ? opName : "?",
               arrayInvalidReasonName(reason),
               sourceKind, resolvedIdx,
               executeCount, planPhase);
    }
  }

  return invalidCount;
}

int validateSlotOutputs(
    int slotIdx, const char* opName,
    NDArray** outputSlots, int totalOutputSlots,
    const int* outputSlotIndices, int numOutputs,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen) {

  if (outputSlotIndices == nullptr || numOutputs <= 0) return 0;

  int invalidCount = 0;
  bool firstError = true;

  for (int i = 0; i < numOutputs; i++) {
    int si = outputSlotIndices[i];
    if (si < 0 || si >= totalOutputSlots) continue;

    NDArray* output = outputSlots[si];
    // Null outputs are legitimate for some ops (conditional branches, dead slots).
    if (output == nullptr) continue;

    ArrayInvalidReason reason = validateArrayForExecution(output);
    if (reason != ArrayInvalidReason::VALID) {
      invalidCount++;
      if (firstError && errMsg != nullptr && errMsgLen > 0) {
        firstError = false;
        DataBuffer* db = nullptr;
        if (reason != ArrayInvalidReason::NULL_ARRAY &&
            reason != ArrayInvalidReason::NULL_SHAPE_INFO) {
          db = output->dataBuffer();
        }
        snprintf(errMsg, errMsgLen,
                 "ARRAY_INVALID at slot %d output %d (outputSlot=%d, op=%s): reason=%s "
                 "DataBuffer=%p "
                 "exec=%d phase=%d",
                 slotIdx, i, si, opName ? opName : "?",
                 arrayInvalidReasonName(reason),
                 (void*)db,
                 executeCount, planPhase);
      }
      DSP_DIAG(MEMORY,
               "ARRAY_INVALID_OUTPUT: slot=%d output=%d outputSlot=%d op=%s reason=%s "
               "exec=%d phase=%d",
               slotIdx, i, si, opName ? opName : "?",
               arrayInvalidReasonName(reason),
               executeCount, planPhase);
    }
  }

  return invalidCount;
}

int validateSlotRange(
    int startSlot, int endSlot,
    NDArray** outputSlots, int totalOutputSlots,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen) {

  if (outputSlots == nullptr) return 0;

  int invalidCount = 0;
  bool firstError = true;

  // Clamp range to valid bounds
  if (startSlot < 0) startSlot = 0;
  if (endSlot >= totalOutputSlots) endSlot = totalOutputSlots - 1;

  for (int i = startSlot; i <= endSlot; i++) {
    NDArray* arr = outputSlots[i];
    // Null slots are legitimate (dead branches, conditional outputs).
    if (arr == nullptr) continue;

    ArrayInvalidReason reason = validateArrayForExecution(arr);
    if (reason != ArrayInvalidReason::VALID) {
      invalidCount++;
      if (firstError && errMsg != nullptr && errMsgLen > 0) {
        firstError = false;
        DataBuffer* db = nullptr;
        if (reason != ArrayInvalidReason::NULL_ARRAY &&
            reason != ArrayInvalidReason::NULL_SHAPE_INFO) {
          db = arr->dataBuffer();
        }
        snprintf(errMsg, errMsgLen,
                 "ARRAY_INVALID at slot %d (range [%d-%d]): reason=%s "
                 "DataBuffer=%p "
                 "exec=%d phase=%d",
                 i, startSlot, endSlot,
                 arrayInvalidReasonName(reason),
                 (void*)db,
                 executeCount, planPhase);
      }
      DSP_DIAG(MEMORY,
               "ARRAY_INVALID_RANGE: slot=%d range=[%d-%d] reason=%s "
               "exec=%d phase=%d",
               i, startSlot, endSlot,
               arrayInvalidReasonName(reason),
               executeCount, planPhase);
    }
  }

  return invalidCount;
}

}  // namespace graph
}  // namespace sd
