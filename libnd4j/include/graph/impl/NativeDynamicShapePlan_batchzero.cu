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
#include <helpers/DebugHelper.h>

namespace sd {
namespace graph {

void NativeDynamicShapePlan::collectBatchZeroTargets(const std::unordered_set<int>& gapSlots) {
  batchZeroEntries_.clear();

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
    if (slot.frozenConstantSlot()) {
      DSP_DIAG_SEG(SEGMENT, s, "batchZero EXCLUDE slot=%d op=%s reason=frozen-constant",
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
        batchZeroEntries_.push_back({devPtr, static_cast<int>(bytes), outIdx});
        DSP_DIAG(MEMORY, "batchZero[%d]: slot %d output[%d] -> slotIdx=%d ptr=%p bytes=%d op=%s",
                    static_cast<int>(batchZeroEntries_.size()) - 1,
                    s, o, outIdx, devPtr, static_cast<int>(bytes),
                    slot.ident.opName.c_str());
      }
    }
  }

  DSP_DIAG(MEMORY, "collectBatchZeroTargets: %d buffers to zero (gapSlots=%d, skipped: identity=%d tail=%d head=%d view=%d)",
           static_cast<int>(batchZeroEntries_.size()),
           static_cast<int>(gapSlots.size()),
           skippedIdentity, skippedTail, skippedHead, skippedView);
}

// ── Batch D2D copy kernel ──────────────────────────────────────────────────
// Replaces ~357 individual cudaMemcpyAsync D2D calls with a single kernel.
// Each thread block copies one buffer using vectorized int4 reads/writes.

__global__ void batchD2DKernel(void** srcPtrs, void** dstPtrs, size_t* sizes, int numBuffers) {
  int bid = blockIdx.x;
  if (bid >= numBuffers) return;

  const char* src = (const char*)srcPtrs[bid];
  char* dst = (char*)dstPtrs[bid];
  size_t sz = sizes[bid];
  if (src == nullptr || dst == nullptr || sz == 0) return;

  // Vectorized copy using int4 (16 bytes per read/write)
  const int4* s4 = (const int4*)src;
  int4* d4 = (int4*)dst;
  int n4 = sz / 16;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    d4[i] = s4[i];
  }

  // Handle remainder bytes
  int base = n4 * 16;
  for (int i = base + threadIdx.x; i < (int)sz; i += blockDim.x) {
    dst[i] = src[i];
  }
}

void NativeDynamicShapePlan::prepareBatchD2DDevice(int count, cudaStream_t stream) {
  if (count <= 0) return;

  // Reuse existing arrays if capacity is sufficient; reallocate if too small
  if (batchD2DAllocated_ < count) {
    freeBatchD2DResources();
    cudaMalloc(&batchD2DDeviceSrcPtrs_, count * sizeof(void*));
    cudaMalloc(&batchD2DDeviceDstPtrs_, count * sizeof(void*));
    cudaMalloc(&batchD2DDeviceSizes_, count * sizeof(size_t));
    cudaMallocHost(&batchD2DHostSrcPtrs_, count * sizeof(void*));
    cudaMallocHost(&batchD2DHostDstPtrs_, count * sizeof(void*));
    cudaMallocHost(&batchD2DHostSizes_, count * sizeof(size_t));
    batchD2DAllocated_ = count;
  }
}

void NativeDynamicShapePlan::launchBatchD2D(cudaStream_t stream) {
  if (batchD2DCount_ <= 0) return;

  // Upload src pointers (updated each step) to device
  cudaMemcpyAsync(batchD2DDeviceSrcPtrs_, batchD2DHostSrcPtrs_,
                  batchD2DCount_ * sizeof(void*), cudaMemcpyHostToDevice, stream);

  // Launch single kernel: one block per buffer, 256 threads per block
  int threadsPerBlock = 256;
  batchD2DKernel<<<batchD2DCount_, threadsPerBlock, 0, stream>>>(
      static_cast<void**>(batchD2DDeviceSrcPtrs_),
      static_cast<void**>(batchD2DDeviceDstPtrs_),
      static_cast<size_t*>(batchD2DDeviceSizes_),
      batchD2DCount_);
  DSP_DIAG(EXECUTE, "launchBatchD2D: single kernel (%d buffers, %d blocks)",
           batchD2DCount_, batchD2DCount_);
}

void NativeDynamicShapePlan::freeBatchD2DResources() {
  if (batchD2DDeviceSrcPtrs_) { cudaFree(batchD2DDeviceSrcPtrs_); batchD2DDeviceSrcPtrs_ = nullptr; }
  if (batchD2DDeviceDstPtrs_) { cudaFree(batchD2DDeviceDstPtrs_); batchD2DDeviceDstPtrs_ = nullptr; }
  if (batchD2DDeviceSizes_) { cudaFree(batchD2DDeviceSizes_); batchD2DDeviceSizes_ = nullptr; }
  if (batchD2DHostSrcPtrs_) { cudaFreeHost(batchD2DHostSrcPtrs_); batchD2DHostSrcPtrs_ = nullptr; }
  if (batchD2DHostDstPtrs_) { cudaFreeHost(batchD2DHostDstPtrs_); batchD2DHostDstPtrs_ = nullptr; }
  if (batchD2DHostSizes_) { cudaFreeHost(batchD2DHostSizes_); batchD2DHostSizes_ = nullptr; }
  batchD2DCount_ = 0;
  batchD2DAllocated_ = 0;
}

}  // namespace graph
}  // namespace sd
