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

#include <ops/declarable/helpers/kv_scatter.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <execution/cuda/LaunchDims.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CUDA kernel: copies present[b, h, lastPos, :] -> output[b, h, cachePos, :]
 *
 * Grid: one block per (batch, head) slice
 * Block: 256 threads, grid-stride over dim
 */
template <typename T>
__global__ void kvScatterKernel(const T* __restrict__ present,
                                 T* __restrict__ output,
                                 const LongType batch,
                                 const LongType heads,
                                 const LongType srcSeqLen,
                                 const LongType dstSeqLen,
                                 const LongType dim,
                                 const LongType lastPos,
                                 const LongType cachePos) {
    // Each block handles one (batch, head) slice
    auto slice = blockIdx.x;
    auto b = slice / heads;
    auto h = slice % heads;

    auto srcOffset = b * heads * srcSeqLen * dim + h * srcSeqLen * dim + lastPos * dim;
    auto dstOffset = b * heads * dstSeqLen * dim + h * dstSeqLen * dim + cachePos * dim;

    // Grid-stride loop over dim
    for (LongType d = threadIdx.x; d < dim; d += blockDim.x) {
        output[dstOffset + d] = present[srcOffset + d];
    }
}

template <typename T>
static void kvScatterCudaLauncher(const cudaStream_t* stream,
                                   const void* vPresent, void* vOutput,
                                   LongType batch, LongType heads,
                                   LongType srcSeqLen, LongType dstSeqLen,
                                   LongType dim, LongType lastPos,
                                   LongType cachePos) {
    auto present = reinterpret_cast<const T*>(vPresent);
    auto output = reinterpret_cast<T*>(vOutput);

    auto numSlices = batch * heads;
    dim3 launchDims = getLaunchDims("kv_scatter");

    kvScatterKernel<T><<<numSlices, launchDims.y, launchDims.z, *stream>>>(
        present, output, batch, heads, srcSeqLen, dstSeqLen, dim, lastPos, cachePos);
    DebugHelper::checkGlobalErrorCode("kvScatter kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvScatterCudaLauncher, (const cudaStream_t* stream, const void* vPresent, void* vOutput, LongType batch, LongType heads, LongType srcSeqLen, LongType dstSeqLen, LongType dim, LongType lastPos, LongType cachePos), SD_FLOAT_TYPES);

void kvScatter(NDArray* present, NDArray* output,
               LongType cachePos, LaunchContext* context) {
    auto stream = context->getCudaStream();

    auto batch = present->sizeAt(0);
    auto heads = present->sizeAt(1);
    auto srcSeqLen = present->sizeAt(2);
    auto dstSeqLen = output->sizeAt(2);
    auto dim = present->sizeAt(3);
    auto lastPos = srcSeqLen - 1;

    NDArray::prepareSpecialUse({output}, {present});

    BUILD_SINGLE_SELECTOR(present->dataType(), kvScatterCudaLauncher,
                          (stream, present->specialBuffer(), output->specialBuffer(),
                           batch, heads, srcSeqLen, dstSeqLen, dim, lastPos, cachePos),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {present});
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batched KV scatter: single kernel launch for all mappings
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Batched kernel: each block handles one (mapping, head) pair.
 * Grid: total_slices = sum(entries[i].heads) across all entries
 * Block: 256 threads, grid-stride over dim
 */
template <typename T>
__global__ void kvScatterBatchedKernel(const KvScatterEntry* __restrict__ entries,
                                        const int* __restrict__ entrySliceOffsets,
                                        int numEntries,
                                        int totalSlices) {
    int globalSlice = blockIdx.x;
    if (globalSlice >= totalSlices) return;

    // Binary search for the entry this block belongs to
    int lo = 0, hi = numEntries - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (entrySliceOffsets[mid + 1] <= globalSlice) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int entryIdx = lo;
    const KvScatterEntry& e = entries[entryIdx];

    int localSlice = globalSlice - entrySliceOffsets[entryIdx];
    auto h = static_cast<LongType>(localSlice);

    auto srcOffset = h * e.srcSeqLen * e.dim + e.lastPos * e.dim;
    auto dstOffset = h * e.dstSeqLen * e.dim + e.cachePos * e.dim;

    auto* src = reinterpret_cast<const T*>(e.srcPtr);
    auto* dst = reinterpret_cast<T*>(e.dstPtr);

    for (LongType d = threadIdx.x; d < e.dim; d += blockDim.x) {
        dst[dstOffset + d] = src[srcOffset + d];
    }
}

template <typename T>
static void kvScatterBatchedCudaLauncher(const cudaStream_t* stream,
                                          const KvScatterEntry* entries, int numEntries) {
    // Compute prefix sums of slice counts (on host, then copy to device)
    std::vector<int> offsets(numEntries + 1);
    offsets[0] = 0;
    for (int i = 0; i < numEntries; i++) {
        offsets[i + 1] = offsets[i] + static_cast<int>(entries[i].heads);
    }
    int totalSlices = offsets[numEntries];

    // Allocate device memory for entries and offsets
    KvScatterEntry* dEntries = nullptr;
    int* dOffsets = nullptr;
    cudaMallocAsync(&dEntries, numEntries * sizeof(KvScatterEntry), *stream);
    cudaMallocAsync(&dOffsets, (numEntries + 1) * sizeof(int), *stream);
    cudaMemcpyAsync(dEntries, entries, numEntries * sizeof(KvScatterEntry),
                    cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(dOffsets, offsets.data(), (numEntries + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("kv_scatter");
    kvScatterBatchedKernel<T><<<totalSlices, launchDims.y, launchDims.z, *stream>>>(
        dEntries, dOffsets, numEntries, totalSlices);

    cudaFreeAsync(dEntries, *stream);
    cudaFreeAsync(dOffsets, *stream);

    DebugHelper::checkGlobalErrorCode("kvScatterBatched kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvScatterBatchedCudaLauncher, (const cudaStream_t* stream, const KvScatterEntry* entries, int numEntries), SD_FLOAT_TYPES);

void kvScatterBatched(const KvScatterEntry* entries, int numEntries,
                       DataType dtype, LaunchContext* context) {
    if (numEntries == 0) return;

    auto stream = context->getCudaStream();

    BUILD_SINGLE_SELECTOR(dtype, kvScatterBatchedCudaLauncher,
                          (stream, entries, numEntries),
                          SD_FLOAT_TYPES);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dynamic-position batched KV scatter: cachePos read from device-side int64 ptr
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Batched dynamic-position kernel: reads cachePos from a shared device pointer.
 *
 * Grid: total_slices = sum(entries[i].heads) across all entries
 * Block: 256 threads, grid-stride over dim
 *
 * The position value (*kvPosPtr) is read once per block from the shared device
 * scalar. Because the kernel dereferences the pointer at runtime, CUDA graph
 * capture bakes the pointer ADDRESS (stable) into the graph — not the value.
 * Between replays the caller updates *kvPosPtr via cudaMemcpyAsync and replays
 * the same graph with the new position.
 */
template <typename T>
__global__ void kvScatterDynBatchedKernel(const KvScatterDynEntry* __restrict__ entries,
                                           const int* __restrict__ entrySliceOffsets,
                                           int numEntries,
                                           int totalSlices) {
    int globalSlice = blockIdx.x;
    if (globalSlice >= totalSlices) return;

    // Binary search for the entry this block belongs to
    int lo = 0, hi = numEntries - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (entrySliceOffsets[mid + 1] <= globalSlice) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int entryIdx = lo;
    const KvScatterDynEntry& e = entries[entryIdx];

    // Read cachePos dynamically from device pointer (address is stable across replays)
    LongType cachePos = *e.kvPosPtr;

    int localSlice = globalSlice - entrySliceOffsets[entryIdx];
    auto h = static_cast<LongType>(localSlice);

    auto srcOffset = h * e.srcSeqLen * e.dim + e.lastPos * e.dim;
    auto dstOffset = h * e.dstSeqLen * e.dim + cachePos * e.dim;

    auto* src = reinterpret_cast<const T*>(e.srcPtr);
    auto* dst = reinterpret_cast<T*>(e.dstPtr);

    for (LongType d = threadIdx.x; d < e.dim; d += blockDim.x) {
        dst[dstOffset + d] = src[srcOffset + d];
    }
}

template <typename T>
static void kvScatterDynBatchedCudaLauncher(const cudaStream_t* stream,
                                             const KvScatterDynEntry* entries,
                                             int numEntries) {
    // Compute prefix sums of slice counts (on host, then copy to device)
    std::vector<int> offsets(numEntries + 1);
    offsets[0] = 0;
    for (int i = 0; i < numEntries; i++) {
        offsets[i + 1] = offsets[i] + static_cast<int>(entries[i].heads);
    }
    int totalSlices = offsets[numEntries];

    // Allocate device memory for entries and offsets
    KvScatterDynEntry* dEntries = nullptr;
    int* dOffsets = nullptr;
    cudaMallocAsync(&dEntries, numEntries * sizeof(KvScatterDynEntry), *stream);
    cudaMallocAsync(&dOffsets, (numEntries + 1) * sizeof(int), *stream);
    cudaMemcpyAsync(dEntries, entries, numEntries * sizeof(KvScatterDynEntry),
                    cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(dOffsets, offsets.data(), (numEntries + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("kv_scatter");
    kvScatterDynBatchedKernel<T><<<totalSlices, launchDims.y, launchDims.z, *stream>>>(
        dEntries, dOffsets, numEntries, totalSlices);

    cudaFreeAsync(dEntries, *stream);
    cudaFreeAsync(dOffsets, *stream);

    DebugHelper::checkGlobalErrorCode("kvScatterDynBatched kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvScatterDynBatchedCudaLauncher, (const cudaStream_t* stream, const KvScatterDynEntry* entries, int numEntries), SD_FLOAT_TYPES);

void kvScatterDynBatched(const KvScatterDynEntry* entries, int numEntries,
                          DataType dtype, LaunchContext* context) {
    if (numEntries == 0) return;

    auto stream = context->getCudaStream();

    BUILD_SINGLE_SELECTOR(dtype, kvScatterDynBatchedCudaLauncher,
                          (stream, entries, numEntries),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
