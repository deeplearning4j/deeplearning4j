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

#include <graph/ExecutionState.h>
#include <graph/DspDiagnostics.h>
#include <system/common.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace sd {
namespace graph {

ExecutionState::ExecutionState(int totalOutputSlots)
    : totalOutputSlots_(totalOutputSlots),
      slotArrays_(nullptr),
      ownership_(nullptr) {
  if (totalOutputSlots > 0) {
    slotArrays_ = new NDArray*[totalOutputSlots]();  // zero-initialized
    ownership_ = new SlotBufferInfo[totalOutputSlots]();  // zero-initialized
  }
  DSP_DIAG(MEMORY, "ExecutionState: created with %d output slots", totalOutputSlots);
}

ExecutionState::~ExecutionState() {
  // Free slot arrays based on ownership
  if (slotArrays_ != nullptr && ownership_ != nullptr) {
    // Build dedup set to prevent double-free (views share DataBuffers)
    std::unordered_set<NDArray*> deleted;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotArrays_[i] != nullptr && deleted.insert(slotArrays_[i]).second) {
        auto& info = ownership_[i];
        if (info.canFree() || info.ownership == BufferOwnership::UNSET) {
          // SLOT_OWNED or unclassified — we own this array
          delete slotArrays_[i];
        } else if (info.isView()) {
          // View — delete the wrapper only (not the DataBuffer)
          // The NDArray destructor handles this correctly when isView() is true
          delete slotArrays_[i];
        }
        // WEIGHT, VIEW_OF_WEIGHT — do NOT delete
      }
    }
  }

  delete[] slotArrays_;
  delete[] ownership_;
  delete[] segmentStates_;

  // captureWorkspace_ is NOT owned by ExecutionState — it's a borrowed pointer
  // from the plan's capture workspace allocation (CudaMemoryPool or similar).

  DSP_DIAG(MEMORY, "ExecutionState: destroyed");
}

void ExecutionState::bindToCurrentThread() {
  auto currentId = std::this_thread::get_id();
  bool expected = false;
  if (bound_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    // First bind — set owner thread
    ownerThread_ = currentId;
    DSP_DIAG(EXECUTE, "ExecutionState: bound to thread");
  } else {
    // Already bound — verify same thread
    if (ownerThread_ != currentId) {
      std::ostringstream ss;
      ss << "ExecutionState bound to thread " << ownerThread_
         << " but called from thread " << currentId;
      THROW_EXCEPTION(ss.str().c_str());
    }
  }
}

void ExecutionState::assertBoundThread() const {
  if (!bound_.load(std::memory_order_relaxed)) return;  // Not yet bound — OK
  if (ownerThread_ != std::this_thread::get_id()) {
    std::ostringstream ss;
    ss << "ExecutionState bound to thread " << ownerThread_
       << " but called from thread " << std::this_thread::get_id();
    THROW_EXCEPTION(ss.str().c_str());
  }
}

void ExecutionState::freeSlotMemory(int slotIdx, void* stream) {
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) return;
  NDArray* arr = slotArrays_[slotIdx];
  if (arr == nullptr) return;

  auto& info = ownership_[slotIdx];
  if (!info.canFreeNow()) return;  // Not freeable or has live views

  auto* db = arr->dataBuffer();
  if (db != nullptr && !db->isConstant && !db->isClosed() &&
      db->special() != nullptr && db->_isOwnerSpecial &&
      !isProtectedWeight(db)) {
    db->freeGpuOnStream(stream);
  }

  // Decrement viewRefCount on parent if this was a VIEW_OF_SLOT
  // (handled by the caller in cleanup, not here — this only frees SLOT_OWNED)

  delete arr;
  slotArrays_[slotIdx] = nullptr;
  info.reset();
}

void ExecutionState::trimPoolIfNeeded() {
  // Pool trimming is only needed after OOM or significant free activity.
  // Steady-state (frozen shapes) should have zero churn.
  // This is called from allocateFailover() in CudaMemoryPool, not from here.
}

// ── Capture workspace ─────────────────────────────────────────────────

void ExecutionState::setCaptureWorkspace(void* buffer, size_t size) {
  captureWorkspace_ = buffer;
  captureWorkspaceSize_ = size;
  captureWorkspaceOffset_ = 0;
  DSP_DIAG(MEMORY, "ExecutionState: capture workspace set, %zu bytes", size);
}

void* ExecutionState::workspaceAlloc(size_t bytes, size_t align) {
  if (captureWorkspace_ == nullptr || captureWorkspaceSize_ == 0) return nullptr;
  // Align offset
  size_t aligned = (captureWorkspaceOffset_ + align - 1) & ~(align - 1);
  if (aligned + bytes > captureWorkspaceSize_) return nullptr;
  void* ptr = static_cast<char*>(captureWorkspace_) + aligned;
  captureWorkspaceOffset_ = aligned + bytes;
  return ptr;
}

// ── Segment state management ──────────────────────────────────────────

void ExecutionState::initSegmentStates(int numSegments) {
  delete[] segmentStates_;
  segmentStates_ = nullptr;
  numSegments_ = 0;
  if (numSegments > 0) {
    segmentStates_ = new SegmentExecState[numSegments]();
    numSegments_ = numSegments;
  }
  DSP_DIAG(MEMORY, "ExecutionState: initialized %d segment states", numSegments);
}

// Stream/device management: platform-specific implementations in
//   graph/cpu/ExecutionState.cpp     (CPU build)
//   graph/cuda/ExecutionState_cuda.cu (CUDA build)

}  // namespace graph
}  // namespace sd
