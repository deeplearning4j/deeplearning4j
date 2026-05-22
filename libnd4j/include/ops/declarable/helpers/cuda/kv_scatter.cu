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
#include <memory/cuda/CudaMemoryPool.h>
#include <cuda_runtime.h>
#include <string>

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

// Pre-allocated device buffers for static-position KV scatter — avoids
// cudaMallocAsync/cudaFreeAsync per decode step. Grow-only pattern.
static thread_local KvScatterEntry* tl_staticEntries = nullptr;
static thread_local int* tl_staticOffsets = nullptr;
static thread_local int tl_staticCachedCapacity = 0;

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

    // Reuse pre-allocated device buffers (grow-only)
    if (numEntries > tl_staticCachedCapacity) {
        int deviceId = sd::AffinityManager::currentDeviceId();
        if (tl_staticEntries != nullptr) sd::memory::CudaMemoryPool::getInstance().free(tl_staticEntries, deviceId, *stream);
        if (tl_staticOffsets != nullptr) sd::memory::CudaMemoryPool::getInstance().free(tl_staticOffsets, deviceId, *stream);
        tl_staticEntries = reinterpret_cast<KvScatterEntry*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(numEntries * sizeof(KvScatterEntry), deviceId, *stream));
        tl_staticOffsets = reinterpret_cast<int*>(
            sd::memory::CudaMemoryPool::getInstance().allocate((numEntries + 1) * sizeof(int), deviceId, *stream));
        tl_staticCachedCapacity = numEntries;
    }

    cudaMemcpyAsync(tl_staticEntries, entries, numEntries * sizeof(KvScatterEntry),
                    cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(tl_staticOffsets, offsets.data(), (numEntries + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("kv_scatter");
    kvScatterBatchedKernel<T><<<totalSlices, launchDims.y, launchDims.z, *stream>>>(
        tl_staticEntries, tl_staticOffsets, numEntries, totalSlices);

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

// Pre-allocated device buffers for KV scatter — avoids cudaMallocAsync/cudaFreeAsync
// per decode step. Grow-only: reallocated when numEntries exceeds cached capacity.
static thread_local KvScatterDynEntry* tl_dEntries = nullptr;
static thread_local int* tl_dOffsets = nullptr;
static thread_local int tl_cachedCapacity = 0;

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

    // Reuse pre-allocated device buffers (grow-only)
    if (numEntries > tl_cachedCapacity) {
        int deviceId = sd::AffinityManager::currentDeviceId();
        if (tl_dEntries != nullptr) sd::memory::CudaMemoryPool::getInstance().free(tl_dEntries, deviceId, *stream);
        if (tl_dOffsets != nullptr) sd::memory::CudaMemoryPool::getInstance().free(tl_dOffsets, deviceId, *stream);
        tl_dEntries = reinterpret_cast<KvScatterDynEntry*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(numEntries * sizeof(KvScatterDynEntry), deviceId, *stream));
        tl_dOffsets = reinterpret_cast<int*>(
            sd::memory::CudaMemoryPool::getInstance().allocate((numEntries + 1) * sizeof(int), deviceId, *stream));
        tl_cachedCapacity = numEntries;
    }

    cudaMemcpyAsync(tl_dEntries, entries, numEntries * sizeof(KvScatterDynEntry),
                    cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(tl_dOffsets, offsets.data(), (numEntries + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("kv_scatter");
    kvScatterDynBatchedKernel<T><<<totalSlices, launchDims.y, launchDims.z, *stream>>>(
        tl_dEntries, tl_dOffsets, numEntries, totalSlices);

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

// ═══════════════════════════════════════════════════════════════════════════════
// In-place KV write: write newKv at device-side cache_position into pastKv
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * CUDA kernel: writes newKv[b, s, h, d] -> pastKv[b, h, cachePos+s, d]
 *
 * newKv layout:  BSHD [batch, seqKV, numKvHeads, headDim]
 * pastKv layout: BHSD [batch, numKvHeads, maxSeqLen, headDim]
 * cachePosPtr:   device pointer to int64 scalar (current cache position)
 *
 * Grid: one block per (batch, seqKV, head) triple
 * Block: 256 threads, grid-stride over headDim
 *
 * Because cachePosPtr is dereferenced at kernel runtime (not baked as a
 * literal), this kernel is CUDA graph compatible: the pointer ADDRESS is
 * captured, and the VALUE at that address can change between replays.
 */
template <typename T>
SD_KERNEL void kvInPlaceWriteKernel(const T* __restrict__ newKv,
                                     T* __restrict__ pastKv,
                                     const LongType* __restrict__ cachePosPtr,
                                     LongType batch,
                                     LongType seqKV,
                                     LongType numKvHeads,
                                     LongType headDim,
                                     LongType pastMaxSeqLen) {
    // Read cache position from device memory (updated between graph replays)
    LongType cachePos = *cachePosPtr;

    // Each block handles one (b, s, h) triple
    LongType slice = blockIdx.x;
    LongType bsh = slice;
    LongType h = bsh % numKvHeads;
    bsh /= numKvHeads;
    LongType s = bsh % seqKV;
    LongType b = bsh / seqKV;

    // newKv is BSHD: offset = b * seqKV * numKvHeads * headDim + s * numKvHeads * headDim + h * headDim
    LongType srcOffset = b * seqKV * numKvHeads * headDim
                       + s * numKvHeads * headDim
                       + h * headDim;

    // pastKv is BHSD: offset = b * numKvHeads * pastMaxSeqLen * headDim + h * pastMaxSeqLen * headDim + (cachePos+s) * headDim
    LongType dstOffset = b * numKvHeads * pastMaxSeqLen * headDim
                       + h * pastMaxSeqLen * headDim
                       + (cachePos + s) * headDim;

    // Grid-stride loop over headDim
    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        pastKv[dstOffset + d] = newKv[srcOffset + d];
    }
}

template <typename T>
static void kvInPlaceWriteCudaLauncher(const cudaStream_t* stream,
                                        const void* vNewKv, void* vPastKv,
                                        const void* cachePosPtr,
                                        LongType batch, LongType seqKV,
                                        LongType numKvHeads, LongType headDim,
                                        LongType pastMaxSeqLen) {
    auto newKv = reinterpret_cast<const T*>(vNewKv);
    auto pastKv = reinterpret_cast<T*>(vPastKv);
    auto posPtr = reinterpret_cast<const LongType*>(cachePosPtr);

    auto numSlices = batch * seqKV * numKvHeads;
    dim3 launchDims = getLaunchDims("kv_scatter");

    kvInPlaceWriteKernel<T><<<numSlices, launchDims.y, launchDims.z, *stream>>>(
        newKv, pastKv, posPtr, batch, seqKV, numKvHeads, headDim, pastMaxSeqLen);
    DebugHelper::checkGlobalErrorCode("kvInPlaceWrite kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvInPlaceWriteCudaLauncher,
    (const cudaStream_t* stream, const void* vNewKv, void* vPastKv,
     const void* cachePosPtr, LongType batch, LongType seqKV,
     LongType numKvHeads, LongType headDim, LongType pastMaxSeqLen),
    SD_FLOAT_TYPES);

void kvInPlaceWrite(NDArray* pastKv, NDArray* newKv,
                     const void* cachePosPtr, LaunchContext* context) {
    auto stream = context->getCudaStream();

    // Auto-cast newKv to match cache dtype (e.g. FLOAT→HALF after FusedRoPE promotion).
    // Cache dtype is authoritative — it's the persistent storage format.
    NDArray* castBuf = nullptr;
    if (pastKv->dataType() != newKv->dataType()) {
        castBuf = newKv->cast(pastKv->dataType());
        newKv = castBuf;
    }

    // newKv is BSHD [batch, seqKV, numKvHeads, headDim]
    auto batch = newKv->sizeAt(0);
    auto seqKV = newKv->sizeAt(1);
    auto numKvHeads = newKv->sizeAt(2);
    auto headDim = newKv->sizeAt(3);

    // pastKv is BHSD [batch, numKvHeads, maxSeqLen, headDim]
    auto pastMaxSeqLen = pastKv->sizeAt(2);

    NDArray::prepareSpecialUse({pastKv}, {newKv});

    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWriteCudaLauncher,
                          (stream, newKv->specialBuffer(), pastKv->specialBuffer(),
                           cachePosPtr, batch, seqKV, numKvHeads, headDim, pastMaxSeqLen),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({pastKv}, {newKv});

    delete castBuf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BSHD→BSHD in-place KV write (for dot_product_attention_v2)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * CUDA kernel: writes newKv[b, s, h, d] → cache[b, cachePos+s, h, d]
 *
 * Both arrays are BSHD layout. For rank-3 BSF, treat as BSHD with heads=1
 * and dim=features.
 *
 * cachePosPtr is dereferenced at kernel runtime (not baked as a literal),
 * making this kernel CUDA graph compatible.
 */
template <typename T>
SD_KERNEL void kvInPlaceWriteBSHDKernel(const T* __restrict__ newKv,
                                         T* __restrict__ cache,
                                         const LongType* __restrict__ cachePosPtr,
                                         LongType batch,
                                         LongType seqKV,
                                         LongType innerSize,
                                         LongType cacheMaxSeqLen) {
    LongType cachePos = *cachePosPtr;

    // Each block handles one (b, s) pair; grid-stride over innerSize (heads*dim or features)
    LongType slice = blockIdx.x;
    LongType s = slice % seqKV;
    LongType b = slice / seqKV;

    // BSHD/BSF: contiguous innerSize elements per (b, s) position
    LongType srcOffset = b * seqKV * innerSize + s * innerSize;
    LongType dstOffset = b * cacheMaxSeqLen * innerSize + (cachePos + s) * innerSize;

    for (LongType i = threadIdx.x; i < innerSize; i += blockDim.x) {
        cache[dstOffset + i] = newKv[srcOffset + i];
    }
}

template <typename T>
static void kvInPlaceWriteBSHDCudaLauncher(const cudaStream_t* stream,
                                            const void* vNewKv, void* vCache,
                                            const void* cachePosPtr,
                                            LongType batch, LongType seqKV,
                                            LongType innerSize, LongType cacheMaxSeqLen) {
    auto newKv = reinterpret_cast<const T*>(vNewKv);
    auto cache = reinterpret_cast<T*>(vCache);
    auto posPtr = reinterpret_cast<const LongType*>(cachePosPtr);

    auto numSlices = batch * seqKV;
    dim3 launchDims = getLaunchDims("kv_scatter");

    kvInPlaceWriteBSHDKernel<T><<<numSlices, launchDims.y, launchDims.z, *stream>>>(
        newKv, cache, posPtr, batch, seqKV, innerSize, cacheMaxSeqLen);
    DebugHelper::checkGlobalErrorCode("kvInPlaceWriteBSHD kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvInPlaceWriteBSHDCudaLauncher,
    (const cudaStream_t* stream, const void* vNewKv, void* vCache,
     const void* cachePosPtr, LongType batch, LongType seqKV,
     LongType innerSize, LongType cacheMaxSeqLen),
    SD_FLOAT_TYPES);

void kvInPlaceWriteBSHD(NDArray* cache, NDArray* newKv,
                         const void* cachePosPtr, LaunchContext* context) {
    auto stream = context->getCudaStream();

    // Auto-cast newKv to match cache dtype (e.g. FLOAT→HALF after FusedRoPE promotion).
    // Cache dtype is authoritative — it's the persistent storage format.
    NDArray* castBuf = nullptr;
    if (cache->dataType() != newKv->dataType()) {
        castBuf = newKv->cast(cache->dataType());
        newKv = castBuf;
    }

    auto batch = newKv->sizeAt(0);
    auto seqKV = newKv->sizeAt(1);
    // innerSize = everything after the seq dimension (heads*dim for rank4, features for rank3)
    LongType innerSize = 1;
    for (int d = 2; d < newKv->rankOf(); d++) {
        innerSize *= newKv->sizeAt(d);
    }

    auto cacheMaxSeqLen = cache->sizeAt(1);

    NDArray::prepareSpecialUse({cache}, {newKv});

    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWriteBSHDCudaLauncher,
                          (stream, newKv->specialBuffer(), cache->specialBuffer(),
                           cachePosPtr, batch, seqKV, innerSize, cacheMaxSeqLen),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({cache}, {newKv});

    delete castBuf;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
