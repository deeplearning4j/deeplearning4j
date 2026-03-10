/* ******************************************************************************
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

// Batch-zero kernel for CUDA graph node reduction.
//
// During CUDA graph capture, each native fallback op gets its output array
// nullified individually, producing ~1000 cudaMemsetAsync graph nodes.
// This kernel replaces all of them with a single kernel launch that zeros
// all output buffers in parallel.
//
// Each thread block zeros one buffer using vectorized int4 writes.
// For typical VLM decode (hidden_dim=576, seq=1): ~2KB per buffer,
// ~1000 buffers total — the kernel completes in <10μs.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <helpers/DebugHelper.h>

namespace sd {
namespace graph {

// ── CUDA Kernel ──────────────────────────────────────────────────────────────

__global__ void batchZeroKernel(void** ptrs, int* sizes, int numBuffers) {
  int bid = blockIdx.x;
  if (bid >= numBuffers) return;

  char* ptr = (char*)ptrs[bid];
  int sz = sizes[bid];
  if (ptr == nullptr || sz <= 0) return;

  // Vectorized zero using int4 (16 bytes per write)
  int4* p4 = (int4*)ptr;
  int n4 = sz / 16;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    p4[i] = make_int4(0, 0, 0, 0);
  }

  // Handle remainder bytes
  int base = n4 * 16;
  for (int i = base + threadIdx.x; i < sz; i += blockDim.x) {
    ptr[i] = 0;
  }
}

// ── Host API ─────────────────────────────────────────────────────────────────

// Thread-local flag: when true, slot execution skips individual nullify()
// because a batch-zero kernel has already zeroed all output buffers.
thread_local bool tl_batchZeroActive = false;

// Registration mode: when true, nullify() still runs but also records each
// buffer into a thread-local list. Used during warmup to learn the exact
// set of buffers that need batch-zeroing during capture.
thread_local bool tl_batchZeroRegistering = false;
struct RegEntry { void* ptr; int bytes; };
thread_local std::vector<RegEntry> tl_batchZeroRegistered;

bool NativeDynamicShapePlan::isBatchZeroActive() {
  return tl_batchZeroActive;
}

bool NativeDynamicShapePlan::isBatchZeroRegistering() {
  return tl_batchZeroRegistering;
}

void NativeDynamicShapePlan::startBatchZeroRegistration() {
  tl_batchZeroRegistered.clear();
  tl_batchZeroRegistering = true;
  DSP_DIAG(MEMORY, "batch-zero registration STARTED");
}

void NativeDynamicShapePlan::registerBatchZeroBuffer(void* ptr, size_t bytes) {
  if (!tl_batchZeroRegistering || ptr == nullptr || bytes <= 0) return;
  // Avoid duplicates
  for (auto& entry : tl_batchZeroRegistered) {
    if (entry.ptr == ptr) return;
  }
  tl_batchZeroRegistered.push_back(RegEntry{ptr, static_cast<int>(bytes)});
}

void NativeDynamicShapePlan::finishBatchZeroRegistration() {
  tl_batchZeroRegistering = false;
  batchZeroEntries_.clear();
  batchZeroEntries_.reserve(tl_batchZeroRegistered.size());

  for (auto& r : tl_batchZeroRegistered) {
    batchZeroEntries_.push_back({r.ptr, r.bytes});
  }
  DSP_DIAG(MEMORY, "batch-zero registration FINISHED: %d buffers registered",
           static_cast<int>(batchZeroEntries_.size()));
  tl_batchZeroRegistered.clear();
}

void NativeDynamicShapePlan::collectBatchZeroTargets(const std::unordered_set<int>& gapSlots) {
  batchZeroEntries_.clear();

  // FALLBACK pre-scan approach: used when registration-based learning is not available
  // (e.g., capture retry at executionCount >= 3).
  //
  // WARNING: This approach collects ~143 EXTRA buffers for slots that don't actually
  // execute during the segment (identity ops, fused chains, etc.). This over-collection
  // can cause incorrect zeroing of buffers that should retain their values.
  // The preferred approach is registration-based learning (startBatchZeroRegistration +
  // finishBatchZeroRegistration) which observes exactly which buffers get nullified
  // during a warmup execution and records that exact set.
  //
  // Only zero output buffers for gap (native fallback) slots — NOT Triton sub-kernel
  // outputs. Triton sub-kernels fully overwrite their outputs; zeroing them would add
  // unnecessary work and potentially interfere with multi-phase kernel correctness.
  //
  // Must match the exact set of buffers that per-slot nullify() touches during capture.
  // Slots that return early without nullify must be skipped here:
  //   - frozenConstantSlot: early return, no execution
  //   - isIdentityOp: wires output=input, no allocation/nullify
  //   - isFusedChainTail: head already computed result, tail returns early
  //   - inPlaceFused: output IS the input, no separate buffer
  //   - view producers: share input's DataBuffer
  //   - isFusedChainHead: nullifies only the LAST chain slot's output, not head's own
  int skippedIdentity = 0, skippedTail = 0, skippedHead = 0, skippedView = 0;
  for (int s = 0; s < numSlots_; s++) {
    // Only include gap slots (slots NOT covered by any Triton sub-kernel)
    if (gapSlots.find(s) == gapSlots.end()) continue;

    auto& slot = slots_[s];

    // Skip frozen constants — they don't execute
    if (slot.frozenConstantSlot) continue;

    // Skip identity ops — they wire output=input, no nullify happens
    if (slot.isIdentityOp) { skippedIdentity++; continue; }

    // Skip fused chain tails — head already computed, tail returns early
    if (slot.isFusedChainTail) { skippedTail++; continue; }

    // Skip in-place fused — output IS the input
    if (slot.inPlaceFused) continue;

    // Fused chain heads only nullify the LAST chain slot's output,
    // not the head's own outputSlotIndices. Collect that specific buffer.
    if (slot.isFusedChainHead && slot.fusedChainLength > 0) {
      skippedHead++;
      int lastSlotIdx = slot.fusedChainSlots[slot.fusedChainLength - 1];
      if (lastSlotIdx >= 0 && lastSlotIdx < numSlots_) {
        int lastOutIdx = slots_[lastSlotIdx].outputSlotIndices[0];
        if (lastOutIdx >= 0 && lastOutIdx < totalOutputSlots_) {
          NDArray* cached = slotArrayCache_[lastOutIdx];
          if (cached != nullptr) {
            void* devPtr = cached->specialBuffer();
            if (devPtr != nullptr) {
              size_t bytes = cached->dataBuffer()->getLenInBytes();
              if (bytes > 0) {
                bool duplicate = false;
                for (auto& entry : batchZeroEntries_) {
                  if (entry.ptr == devPtr) { duplicate = true; break; }
                }
                if (!duplicate) {
                  batchZeroEntries_.push_back({devPtr, static_cast<int>(bytes)});
                  DSP_DIAG(MEMORY, "batchZero[%d]: fusedHead slot %d -> lastChain=%d outIdx=%d ptr=%p bytes=%d",
                              static_cast<int>(batchZeroEntries_.size()) - 1,
                              s, lastSlotIdx, lastOutIdx, devPtr, static_cast<int>(bytes));
                }
              }
            }
          }
        }
      }
      continue;  // Don't fall through to collect head's own outputs
    }

    for (int o = 0; o < slot.numOutputs; o++) {
      int outIdx = slot.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;

      // Skip view-producer output slots — they share the input's DataBuffer.
      // slotIsViewProducer_ is indexed by OUTPUT SLOT INDEX, not op slot index.
      if (slotIsViewProducer_[outIdx]) { skippedView++; continue; }

      NDArray* cached = slotArrayCache_[outIdx];
      if (cached == nullptr) continue;

      void* devPtr = cached->specialBuffer();
      if (devPtr == nullptr) continue;

      // Use getLenInBytes() from the DataBuffer — this is the FULL allocation size
      // (including any padding/overallocation). Must match setToZeroBuffers() semantics
      // which also zeros getLenInBytes(), not just lengthOf() * sizeOfT().
      size_t bytes = cached->dataBuffer()->getLenInBytes();
      if (bytes <= 0) continue;

      // Avoid duplicates (multiple ops may reference same output slot)
      bool duplicate = false;
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr == devPtr) { duplicate = true; break; }
      }
      if (!duplicate) {
        batchZeroEntries_.push_back({devPtr, static_cast<int>(bytes)});
        DSP_DIAG(MEMORY, "batchZero[%d]: slot %d output[%d] -> slotIdx=%d ptr=%p bytes=%d op=%s",
                    static_cast<int>(batchZeroEntries_.size()) - 1,
                    s, o, outIdx, devPtr, static_cast<int>(bytes),
                    slot.opName.c_str());
      }
    }
  }

  DSP_DIAG(MEMORY, "collectBatchZeroTargets: %d buffers to zero (gapSlots=%d, skipped: identity=%d tail=%d head=%d view=%d)",
           static_cast<int>(batchZeroEntries_.size()),
           static_cast<int>(gapSlots.size()),
           skippedIdentity, skippedTail, skippedHead, skippedView);
}

void NativeDynamicShapePlan::prepareBatchZeroDevice(cudaStream_t stream) {
  int count = static_cast<int>(batchZeroEntries_.size());
  if (count <= 0) return;

  // Free old device arrays if size changed
  if (batchZeroDevicePtrs_ != nullptr && batchZeroDeviceCount_ != count) {
    cudaFree(batchZeroDevicePtrs_);
    cudaFree(batchZeroDeviceSizes_);
    cudaFreeHost(batchZeroHostPtrs_);
    cudaFreeHost(batchZeroHostSizes_);
    batchZeroDevicePtrs_ = nullptr;
    batchZeroDeviceSizes_ = nullptr;
    batchZeroHostPtrs_ = nullptr;
    batchZeroHostSizes_ = nullptr;
  }

  // Allocate device-side arrays (persistent — used on every graph replay)
  if (batchZeroDevicePtrs_ == nullptr) {
    cudaMalloc(&batchZeroDevicePtrs_, count * sizeof(void*));
    cudaMalloc(&batchZeroDeviceSizes_, count * sizeof(int));
    cudaMallocHost(&batchZeroHostPtrs_, count * sizeof(void*));
    cudaMallocHost(&batchZeroHostSizes_, count * sizeof(int));
    batchZeroDeviceCount_ = count;
  }

  // Fill host arrays
  for (int i = 0; i < count; i++) {
    static_cast<void**>(batchZeroHostPtrs_)[i] = batchZeroEntries_[i].ptr;
    static_cast<int*>(batchZeroHostSizes_)[i] = batchZeroEntries_[i].bytes;
  }

  // Upload to device (this happens OUTSIDE graph capture, so it's a normal H2D copy)
  cudaMemcpyAsync(batchZeroDevicePtrs_, batchZeroHostPtrs_,
                   count * sizeof(void*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(batchZeroDeviceSizes_, batchZeroHostSizes_,
                   count * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}

void NativeDynamicShapePlan::launchBatchZero(cudaStream_t stream) {
  int count = static_cast<int>(batchZeroEntries_.size());
  if (count <= 0) return;

  if (sd::Environment::getInstance().dspBatchZeroKernel()) {
    // Single kernel mode: 1 graph node instead of N, ~797 fewer nodes.
    // Each thread block zeros one buffer using vectorized int4 writes.
    int threadsPerBlock = 256;
    batchZeroKernel<<<count, threadsPerBlock, 0, stream>>>(
        static_cast<void**>(batchZeroDevicePtrs_),
        static_cast<int*>(batchZeroDeviceSizes_),
        count);
    DSP_DIAG(MEMORY, "launchBatchZero: single kernel (%d buffers, %d blocks)", count, count);
  } else {
    // Per-buffer memset mode: N graph nodes but guaranteed same semantics as
    // per-slot nullify (same CUDA driver path for cudaMemsetAsync).
    for (int i = 0; i < count; i++) {
      if (batchZeroEntries_[i].ptr != nullptr && batchZeroEntries_[i].bytes > 0) {
        cudaMemsetAsync(batchZeroEntries_[i].ptr, 0, batchZeroEntries_[i].bytes, stream);
      }
    }
  }
}

void NativeDynamicShapePlan::setBatchZeroActive(bool active) {
  tl_batchZeroActive = active;
}

void NativeDynamicShapePlan::freeBatchZeroResources() {
  if (batchZeroDevicePtrs_) { cudaFree(batchZeroDevicePtrs_); batchZeroDevicePtrs_ = nullptr; }
  if (batchZeroDeviceSizes_) { cudaFree(batchZeroDeviceSizes_); batchZeroDeviceSizes_ = nullptr; }
  if (batchZeroHostPtrs_) { cudaFreeHost(batchZeroHostPtrs_); batchZeroHostPtrs_ = nullptr; }
  if (batchZeroHostSizes_) { cudaFreeHost(batchZeroHostSizes_); batchZeroHostSizes_ = nullptr; }
  batchZeroDeviceCount_ = 0;
  batchZeroEntries_.clear();
}

}  // namespace graph
}  // namespace sd
