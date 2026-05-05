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

// Batch-zero target collection + batched D2D copy kernel.
//
// collectBatchZeroTargets() walks segment slots and produces the set of
// output buffers that must be zeroed before each replay. The consumer is the
// pre-capture loop in NativeDynamicShapePlan_gpubackend.cpp which issues
// cudaMemsetAsync on each entry before beginning CUDA graph capture.
//
// batchD2DKernel replaces ~357 individual cudaMemcpyAsync D2D calls with a
// single kernel launch: each thread block copies one buffer using int4 reads.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/DebugHelper.h>

namespace sd {
namespace graph {

void NativeDynamicShapePlan::collectBatchZeroTargets(const std::unordered_set<int>& gapSlots) {
  batchZeroEntries_.clear();

  // Zero only gap (native fallback) slot outputs that actually need a
  // zero-before-write pass. This must match the generic replay prezero rules:
  // skip outputs that are fully overwritten, identity/view aliases, fused tails,
  // in-place writes, and frozen constants.
  int skippedIdentity = 0, skippedTail = 0, skippedHead = 0, skippedView = 0;
  int skippedNoZeroNeeded = 0, skippedFullyWriting = 0;
  for (int s = 0; s < numSlots_; s++) {
    // Only include gap slots (slots NOT covered by any Triton sub-kernel)
    if (gapSlots.find(s) == gapSlots.end()) continue;

    auto& slot = slots_[s];

    // Skip frozen constants — they don't execute
    if (slot.frozenConstantSlot()) {
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=frozen-constant",
                   s, slot.ident.opName.c_str());
      continue;
    }

    if (!slot.flags.needsZeroedOutput) {
      skippedNoZeroNeeded++;
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=no-zero-needed",
                   s, slot.ident.opName.c_str());
      continue;
    }

    // Skip identity ops — they wire output=input, no nullify happens
    if (slot.flags.isIdentityOp) {
      skippedIdentity++;
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=identity-op",
                   s, slot.ident.opName.c_str());
      continue;
    }

    // Skip fused chain tails — head already computed, tail returns early
    if (slot.fusedChain.isFusedChainTail) {
      skippedTail++;
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=fused-chain-tail",
                   s, slot.ident.opName.c_str());
      continue;
    }

    // Skip in-place fused — output IS the input
    if (slot.flags.inPlaceFused) {
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=in-place-fused",
                   s, slot.ident.opName.c_str());
      continue;
    }

    if (slot.state_ >= NativeSlot::SlotState::FROZEN && slot.flags.isFullyWriting) {
      skippedFullyWriting++;
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=fully-writing",
                   s, slot.ident.opName.c_str());
      continue;
    }

    // Fused chain heads only nullify the LAST chain slot's output,
    // not the head's own outputSlotIndices. Collect that specific buffer.
    if (slot.fusedChain.isFusedChainHead && slot.fusedChain.fusedChainLength > 0) {
      skippedHead++;
      int lastSlotIdx = slot.fusedChain.fusedChainSlots[slot.fusedChain.fusedChainLength - 1];
      if (lastSlotIdx >= 0 && lastSlotIdx < numSlots_) {
        int lastOutIdx = slots_[lastSlotIdx].wiring.outputSlotIndices[0];
        if (lastOutIdx >= 0 && lastOutIdx < totalOutputSlots_) {
          NDArray* cached = outputSlots_[lastOutIdx];
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
                  batchZeroEntries_.push_back({devPtr, static_cast<int>(bytes), lastOutIdx});
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

    // Skip view-capable ops — they share input's buffer, zeroing would corrupt data
    if (slot.flags.isViewCapableOp) {
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=view-capable",
                   s, slot.ident.opName.c_str());
      continue;
    }

    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;

      // Skip view-producer output slots — they share the input's DataBuffer.
      // slotIsViewProducer_ is indexed by OUTPUT SLOT INDEX, not op slot index.
      if (slotIsViewProducer_[outIdx]) {
        skippedView++;
        DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d outIdx=%d op=%s reason=view-producer",
                     s, outIdx, slot.ident.opName.c_str());
        continue;
      }

      NDArray* cached = outputSlots_[outIdx];
      if (cached == nullptr) continue;

      void* devPtr = cached->specialBuffer();
      if (devPtr == nullptr) continue;

      // Use getLenInBytes() (full allocation size) to match setToZeroBuffers().
      size_t bytes = cached->dataBuffer()->getLenInBytes();
      if (bytes <= 0) continue;

      // Avoid duplicates (multiple ops may reference same output slot)
      bool duplicate = false;
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr == devPtr) { duplicate = true; break; }
      }
      if (!duplicate) {
        batchZeroEntries_.push_back({devPtr, static_cast<int>(bytes), outIdx});
        DSP_DIAG(MEMORY, "batchZero[%d]: slot %d output[%d] -> slotIdx=%d ptr=%p bytes=%d op=%s",
                    static_cast<int>(batchZeroEntries_.size()) - 1,
                    s, o, outIdx, devPtr, static_cast<int>(bytes),
                    slot.ident.opName.c_str());
      }
    }
  }

  DSP_DIAG(MEMORY, "collectBatchZeroTargets: %d buffers to zero (gapSlots=%d, skipped: noZero=%d fullyWriting=%d identity=%d tail=%d head=%d view=%d)",
           static_cast<int>(batchZeroEntries_.size()),
           static_cast<int>(gapSlots.size()),
           skippedNoZeroNeeded, skippedFullyWriting,
           skippedIdentity, skippedTail, skippedHead, skippedView);
}

// ── Batch D2D copy kernel ──────────────────────────────────────────────────
// Single kernel launch replaces N individual cudaMemcpyAsync D2D calls.
// One block per buffer, vectorized int4 (16B) reads/writes.

SD_KERNEL void batchD2DKernel(void** srcPtrs, void** dstPtrs,
                              size_t* sizes, int numBuffers) {
  const int bid = blockIdx.x;
  if (bid >= numBuffers) return;

  const char* src = static_cast<const char*>(srcPtrs[bid]);
  char* dst = static_cast<char*>(dstPtrs[bid]);
  const size_t sz = sizes[bid];
  if (src == nullptr || dst == nullptr || sz == 0) return;

  // Vectorized bulk copy — int4 = 16 bytes per transaction.
  // Requires 16-byte alignment which cudaMalloc guarantees (256-byte aligned).
  const int4* s4 = reinterpret_cast<const int4*>(src);
  int4* d4 = reinterpret_cast<int4*>(dst);
  const int n4 = static_cast<int>(sz / 16);
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    d4[i] = s4[i];
  }

  // Remainder bytes (0-15) — byte-wise fallback
  const int base = n4 * 16;
  for (int i = base + threadIdx.x; i < static_cast<int>(sz); i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ── Batch memset kernel ──────────────────────────────────────────────────
// Replaces N individual cudaMemsetAsync calls (prezero) with a single kernel.
// Each thread block zeroes one buffer using vectorized int4 stores.
// Same grid geometry as batchD2DKernel.

SD_KERNEL void batchMemsetKernel(void** dstPtrs, size_t* sizes, int numBuffers) {
  const int bid = blockIdx.x;
  if (bid >= numBuffers) return;

  char* dst = static_cast<char*>(dstPtrs[bid]);
  const size_t sz = sizes[bid];
  if (dst == nullptr || sz == 0) return;

  // Vectorized bulk zero — int4 = 16 bytes per store
  int4* d4 = reinterpret_cast<int4*>(dst);
  const int4 zero4 = make_int4(0, 0, 0, 0);
  const int n4 = static_cast<int>(sz / 16);
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    d4[i] = zero4;
  }

  // Remainder bytes (0-15)
  const int base = n4 * 16;
  for (int i = base + threadIdx.x; i < static_cast<int>(sz); i += blockDim.x) {
    dst[i] = 0;
  }
}

// ── Host/device array management ────────────────────────────────────────

void NativeDynamicShapePlan::prepareBatchD2DDevice(int count, cudaStream_t stream) {
  if (count <= 0) return;

  if (batchD2DAllocated_ >= count) return;

  // Capacity insufficient — free old arrays and allocate at requested size
  freeBatchD2DResources();

  auto check = [](cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      sd_printf("prepareBatchD2DDevice %s failed: %s\n", what, cudaGetErrorString(err));
    }
  };

  int deviceId = sd::AffinityManager::currentDeviceId();
  batchD2DDeviceSrcPtrs_ = sd::memory::CudaMemoryPool::getInstance().allocate(count * sizeof(void*), deviceId, stream);
  batchD2DDeviceDstPtrs_ = sd::memory::CudaMemoryPool::getInstance().allocate(count * sizeof(void*), deviceId, stream);
  batchD2DDeviceSizes_   = sd::memory::CudaMemoryPool::getInstance().allocate(count * sizeof(size_t), deviceId, stream);
  if (!batchD2DDeviceSrcPtrs_) sd_printf("prepareBatchD2DDevice device src failed\n");
  if (!batchD2DDeviceDstPtrs_) sd_printf("prepareBatchD2DDevice device dst failed\n");
  if (!batchD2DDeviceSizes_)   sd_printf("prepareBatchD2DDevice device sizes failed\n");
  check(cudaMallocHost(&batchD2DHostSrcPtrs_, count * sizeof(void*)),  "pinned src");
  check(cudaMallocHost(&batchD2DHostDstPtrs_, count * sizeof(void*)),  "pinned dst");
  check(cudaMallocHost(&batchD2DHostSizes_,   count * sizeof(size_t)), "pinned sizes");
  batchD2DAllocated_ = count;
}

// ── Launch: batched D2D copy ────────────────────────────────────────────

void NativeDynamicShapePlan::launchBatchD2D(cudaStream_t stream) {
  if (batchD2DCount_ <= 0) return;

  // Source pointers change each step (Java rebinds externals) — upload them.
  // Destination pointers and sizes are static (uploaded once in prepare/setup),
  // so only srcPtrs needs a per-step H2D copy.
  cudaMemcpyAsync(batchD2DDeviceSrcPtrs_, batchD2DHostSrcPtrs_,
                  batchD2DCount_ * sizeof(void*), cudaMemcpyHostToDevice, stream);

  // One block per buffer, 256 threads per block.
  constexpr int kThreadsPerBlock = 256;
  batchD2DKernel<<<batchD2DCount_, kThreadsPerBlock, 0, stream>>>(
      static_cast<void**>(batchD2DDeviceSrcPtrs_),
      static_cast<void**>(batchD2DDeviceDstPtrs_),
      static_cast<size_t*>(batchD2DDeviceSizes_),
      batchD2DCount_);
  DSP_DIAG(EXECUTE, "launchBatchD2D: %d buffers in 1 kernel launch", batchD2DCount_);
}

// ── Launch: batched memset (prezero) ────────────────────────────────────
// Accepts raw arrays of destination pointers and sizes (host memory).
// Uploads to device pinned arrays, then launches batchMemsetKernel.
// Caller can pass batchZeroEntries_ data or prezero-collected data.

void NativeDynamicShapePlan::launchBatchMemset(cudaStream_t stream,
                                                void** dstPtrsHost,
                                                size_t* sizesHost,
                                                int count) {
  if (count <= 0) return;

  // Ensure device-side arrays are large enough
  prepareBatchD2DDevice(count, stream);

  // Copy caller-provided arrays into pinned host memory
  auto** pinnedDst = static_cast<void**>(batchD2DHostDstPtrs_);
  auto* pinnedSizes = static_cast<size_t*>(batchD2DHostSizes_);
  memcpy(pinnedDst, dstPtrsHost, count * sizeof(void*));
  memcpy(pinnedSizes, sizesHost, count * sizeof(size_t));

  // Upload to device
  cudaMemcpyAsync(batchD2DDeviceDstPtrs_, pinnedDst,
                  count * sizeof(void*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(batchD2DDeviceSizes_, pinnedSizes,
                  count * sizeof(size_t), cudaMemcpyHostToDevice, stream);

  constexpr int kThreadsPerBlock = 256;
  batchMemsetKernel<<<count, kThreadsPerBlock, 0, stream>>>(
      static_cast<void**>(batchD2DDeviceDstPtrs_),
      static_cast<size_t*>(batchD2DDeviceSizes_),
      count);
  DSP_DIAG(MEMORY, "launchBatchMemset: %d buffers zeroed in 1 kernel launch", count);
}

void NativeDynamicShapePlan::freeBatchD2DResources() {
  int deviceId = sd::AffinityManager::currentDeviceId();
  if (batchD2DDeviceSrcPtrs_) { sd::memory::CudaMemoryPool::getInstance().free(batchD2DDeviceSrcPtrs_, deviceId); batchD2DDeviceSrcPtrs_ = nullptr; }
  if (batchD2DDeviceDstPtrs_) { sd::memory::CudaMemoryPool::getInstance().free(batchD2DDeviceDstPtrs_, deviceId); batchD2DDeviceDstPtrs_ = nullptr; }
  if (batchD2DDeviceSizes_) { sd::memory::CudaMemoryPool::getInstance().free(batchD2DDeviceSizes_, deviceId); batchD2DDeviceSizes_ = nullptr; }
  if (batchD2DHostSrcPtrs_) { cudaFreeHost(batchD2DHostSrcPtrs_); batchD2DHostSrcPtrs_ = nullptr; }
  if (batchD2DHostDstPtrs_) { cudaFreeHost(batchD2DHostDstPtrs_); batchD2DHostDstPtrs_ = nullptr; }
  if (batchD2DHostSizes_) { cudaFreeHost(batchD2DHostSizes_); batchD2DHostSizes_ = nullptr; }
  batchD2DCount_ = 0;
  batchD2DAllocated_ = 0;
}

}  // namespace graph
}  // namespace sd
