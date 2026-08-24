/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/kv_scatter.h>
#include <execution/Threads.h>
#include <string>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void kvScatter_(NDArray* present, NDArray* output,
                          LongType cachePos, LaunchContext* context) {
  auto lastPos = present->sizeAt(2) - 1;
  auto batch = present->sizeAt(0);
  auto heads = present->sizeAt(1);
  auto dim = present->sizeAt(3);

  auto srcSeqLen = present->sizeAt(2);
  auto dstSeqLen = output->sizeAt(2);

  const T* __restrict srcBuf = present->bufferAsT<T>();
  T* __restrict dstBuf = output->bufferAsT<T>();

  // Total number of (batch, head) slices to process — parallelize over these
  auto numSlices = batch * heads;

  auto func = PRAGMA_THREADS_FOR {
    for (auto slice = start; slice < stop; slice++) {
      auto b = slice / heads;
      auto h = slice % heads;

      auto srcOffset = b * heads * srcSeqLen * dim + h * srcSeqLen * dim + lastPos * dim;
      auto dstOffset = b * heads * dstSeqLen * dim + h * dstSeqLen * dim + cachePos * dim;

      const T* __restrict src = srcBuf + srcOffset;
      T* __restrict dst = dstBuf + dstOffset;

      PRAGMA_OMP_SIMD
      for (LongType d = 0; d < dim; d++) {
        dst[d] = src[d];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, numSlices);
}

void kvScatter(NDArray* present, NDArray* output,
               LongType cachePos, LaunchContext* context) {
  BUILD_SINGLE_SELECTOR(present->dataType(), kvScatter_, (present, output, cachePos, context), SD_FLOAT_TYPES);
}

template <typename T>
static void kvScatterBatched_(const KvScatterEntry* entries, int numEntries) {
  // Compute total slices across all entries for a single parallel_for
  int totalSlices = 0;
  std::vector<int> sliceOffsets(numEntries + 1);
  sliceOffsets[0] = 0;
  for (int i = 0; i < numEntries; i++) {
    sliceOffsets[i + 1] = sliceOffsets[i] + static_cast<int>(entries[i].heads);
  }
  totalSlices = sliceOffsets[numEntries];

  auto func = PRAGMA_THREADS_FOR {
    for (auto globalSlice = start; globalSlice < stop; globalSlice++) {
      // Binary search for the entry this slice belongs to
      int lo = 0, hi = numEntries - 1;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (sliceOffsets[mid + 1] <= globalSlice) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      const KvScatterEntry& e = entries[lo];
      auto h = static_cast<LongType>(globalSlice - sliceOffsets[lo]);

      auto srcOffset = h * e.srcSeqLen * e.dim + e.lastPos * e.dim;
      auto dstOffset = h * e.dstSeqLen * e.dim + e.cachePos * e.dim;

      const T* __restrict src = reinterpret_cast<const T*>(e.srcPtr) + srcOffset;
      T* __restrict dst = reinterpret_cast<T*>(e.dstPtr) + dstOffset;

      PRAGMA_OMP_SIMD
      for (LongType d = 0; d < e.dim; d++) {
        dst[d] = src[d];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, totalSlices);
}

void kvScatterBatched(const KvScatterEntry* entries, int numEntries,
                       DataType dtype, LaunchContext* context) {
  if (numEntries == 0) return;
  BUILD_SINGLE_SELECTOR(dtype, kvScatterBatched_, (entries, numEntries), SD_FLOAT_TYPES);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dynamic-position batched KV scatter (CPU)
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T>
static void kvScatterDynBatched_(const KvScatterDynEntry* entries, int numEntries) {
  // Compute total slices across all entries for a single parallel_for
  std::vector<int> sliceOffsets(numEntries + 1);
  sliceOffsets[0] = 0;
  for (int i = 0; i < numEntries; i++) {
    sliceOffsets[i + 1] = sliceOffsets[i] + static_cast<int>(entries[i].heads);
  }
  int totalSlices = sliceOffsets[numEntries];

  // Read cachePos from the shared host pointer (same pointer as used on device side)
  // All entries share the same kvPosPtr pointing to the plan's position scalar.
  LongType cachePos = (numEntries > 0 && entries[0].kvPosPtr != nullptr)
                      ? *entries[0].kvPosPtr
                      : 0;

  auto func = PRAGMA_THREADS_FOR {
    for (auto globalSlice = start; globalSlice < stop; globalSlice++) {
      // Binary search for the entry this slice belongs to
      int lo = 0, hi = numEntries - 1;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (sliceOffsets[mid + 1] <= globalSlice) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      const KvScatterDynEntry& e = entries[lo];
      auto h = static_cast<LongType>(globalSlice - sliceOffsets[lo]);

      auto srcOffset = h * e.srcSeqLen * e.dim + e.lastPos * e.dim;
      auto dstOffset = h * e.dstSeqLen * e.dim + cachePos * e.dim;

      const T* __restrict src = reinterpret_cast<const T*>(e.srcPtr) + srcOffset;
      T* __restrict dst = reinterpret_cast<T*>(e.dstPtr) + dstOffset;

      PRAGMA_OMP_SIMD
      for (LongType d = 0; d < e.dim; d++) {
        dst[d] = src[d];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, totalSlices);
}

void kvScatterDynBatched(const KvScatterDynEntry* entries, int numEntries,
                          DataType dtype, LaunchContext* context) {
  if (numEntries == 0) return;
  BUILD_SINGLE_SELECTOR(dtype, kvScatterDynBatched_, (entries, numEntries), SD_FLOAT_TYPES);
}

// ═══════════════════════════════════════════════════════════════════════════════
// In-place KV write (CPU): write newKv at cache_position into pastKv
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T>
static void kvInPlaceWrite_(NDArray* pastKv, NDArray* newKv,
                             const void* cachePosPtr, LaunchContext* context) {
    // newKv is BSHD [batch, seqKV, numKvHeads, headDim]
    auto batch = newKv->sizeAt(0);
    auto seqKV = newKv->sizeAt(1);
    auto numKvHeads = newKv->sizeAt(2);
    auto headDim = newKv->sizeAt(3);

    // pastKv is BHSD [batch, numKvHeads, maxSeqLen, headDim]
    auto pastMaxSeqLen = pastKv->sizeAt(2);

    // Read cache position from host pointer
    LongType cachePos = *reinterpret_cast<const LongType*>(cachePosPtr);

    const T* __restrict srcBuf = newKv->bufferAsT<T>();
    T* __restrict dstBuf = pastKv->bufferAsT<T>();

    // Total number of (batch, seqKV, head) slices to process
    auto numSlices = batch * seqKV * numKvHeads;

    auto func = PRAGMA_THREADS_FOR {
        for (auto slice = start; slice < stop; slice++) {
            auto bsh = slice;
            auto h = bsh % numKvHeads;
            bsh /= numKvHeads;
            auto s = bsh % seqKV;
            auto b = bsh / seqKV;

            // newKv BSHD offset
            auto srcOffset = b * seqKV * numKvHeads * headDim
                           + s * numKvHeads * headDim
                           + h * headDim;

            // pastKv BHSD offset
            auto dstOffset = b * numKvHeads * pastMaxSeqLen * headDim
                           + h * pastMaxSeqLen * headDim
                           + (cachePos + s) * headDim;

            const T* __restrict src = srcBuf + srcOffset;
            T* __restrict dst = dstBuf + dstOffset;

            PRAGMA_OMP_SIMD
            for (LongType d = 0; d < headDim; d++) {
                dst[d] = src[d];
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, numSlices);
}

void kvInPlaceWrite(NDArray* pastKv, NDArray* newKv,
                     const void* cachePosPtr, LaunchContext* context) {
    LongType cachePos = *reinterpret_cast<const LongType*>(cachePosPtr);
    if (cachePos < 0 || cachePos + newKv->sizeAt(1) > pastKv->sizeAt(2)) {
        THROW_EXCEPTION("kvInPlaceWrite: cache_position is outside the fixed KV capacity");
    }
    // Auto-cast newKv to match cache dtype (e.g. FLOAT→HALF after FusedRoPE promotion).
    // Cache dtype is authoritative — it's the persistent storage format.
    NDArray* castBuf = nullptr;
    if (pastKv->dataType() != newKv->dataType()) {
        castBuf = newKv->cast(pastKv->dataType());
        newKv = castBuf;
    }
    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWrite_,
                          (pastKv, newKv, cachePosPtr, context), SD_FLOAT_TYPES);
    delete castBuf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BSHD→BSHD in-place KV write (for dot_product_attention_v2)
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T>
static void kvInPlaceWriteBSHD_(NDArray* cache, NDArray* newKv,
                                 const void* cachePosPtr, LaunchContext* context) {
    auto batch = newKv->sizeAt(0);
    auto seqKV = newKv->sizeAt(1);
    bool rank4 = newKv->rankOf() == 4;
    LongType outerDim = newKv->sizeAt(2);
    LongType innerDim = rank4 ? newKv->sizeAt(3) : 1;
    LongType innerSize = outerDim * innerDim;
    auto cacheMaxSeqLen = cache->sizeAt(1);

    LongType srcStride0 = newKv->strideAt(0);
    LongType srcStride1 = newKv->strideAt(1);
    LongType srcStride2 = newKv->strideAt(2);
    LongType srcStride3 = rank4 ? newKv->strideAt(3) : 0;
    LongType dstStride0 = cache->strideAt(0);
    LongType dstStride1 = cache->strideAt(1);
    LongType dstStride2 = cache->strideAt(2);
    LongType dstStride3 = rank4 ? cache->strideAt(3) : 0;

    LongType cachePos = *reinterpret_cast<const LongType*>(cachePosPtr);

    const T* __restrict srcBuf = newKv->bufferAsT<T>();
    T* __restrict dstBuf = cache->bufferAsT<T>();

    auto numSlices = batch * seqKV;

    auto func = PRAGMA_THREADS_FOR {
        for (auto slice = start; slice < stop; slice++) {
            auto s = slice % seqKV;
            auto b = slice / seqKV;
            auto dstSeq = cachePos + s;
            if (dstSeq < 0 || dstSeq >= cacheMaxSeqLen) continue;

            PRAGMA_OMP_SIMD
            for (LongType i = 0; i < innerSize; i++) {
                LongType outer = i / innerDim;
                LongType inner = i - outer * innerDim;
                LongType srcOffset = b * srcStride0 + s * srcStride1
                                    + outer * srcStride2 + inner * srcStride3;
                LongType dstOffset = b * dstStride0 + dstSeq * dstStride1
                                    + outer * dstStride2 + inner * dstStride3;
                dstBuf[dstOffset] = srcBuf[srcOffset];
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, numSlices);
}

void kvInPlaceWriteBSHD(NDArray* cache, NDArray* newKv,
                         const void* cachePosPtr, LaunchContext* context) {
    // Auto-cast newKv to match cache dtype (e.g. FLOAT→HALF after FusedRoPE promotion).
    // Cache dtype is authoritative — it's the persistent storage format.
    NDArray* castBuf = nullptr;
    if (cache->dataType() != newKv->dataType()) {
        castBuf = newKv->cast(cache->dataType());
        newKv = castBuf;
    }
    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWriteBSHD_,
                          (cache, newKv, cachePosPtr, context), SD_FLOAT_TYPES);
    delete castBuf;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
