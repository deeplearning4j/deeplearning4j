/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBND4J_HELPERS_CUDA_ARGMAX_ROW_SCAN_CUH
#define LIBND4J_HELPERS_CUDA_ARGMAX_ROW_SCAN_CUH

#include <array/DataTypeUtils.h>
#include <system/common.h>

#include <cstdint>

namespace sd {
namespace ops {
namespace helpers {

// 16-byte loads each thread keeps in flight per iteration of the vectorized row
// scan: a ~250K-entry logits row is bound by memory-level parallelism, not
// bandwidth (one scalar load per iteration took ~110 us for a 1 MB FP32 row on
// GB10, 8 float4 loads ~30 us).
static constexpr int kArgmaxRowLoadsInFlight = 8;

/**
 * One thread's share of a block-cooperative argmax over a logits row, under the
 * CPU argmax contract (cpu/token_sample.cpp: maxVal = -inf, maxIdx = 0, strict >).
 *
 * The thread visits its elements in ascending index order and replaces its
 * candidate only on a strictly larger value, so it ends with its lowest-index
 * maximum; the caller's block merge must apply the same rule (larger value,
 * then smaller index, absent never wins). The candidate starts ABSENT
 * (value -inf, index vocabSize): NaN never compares larger, so NaN is never
 * selected, and a row with no value above -inf leaves every candidate absent,
 * which the caller maps to index 0.
 *
 * A unit-stride row of 4-byte elements whose start is 16-byte aligned is read
 * as float4 groups q = i * (blockDim * L) + u * blockDim + t (ascending per
 * thread); the remainder, and every other layout, is read element by element.
 * The alignment test is on the row pointer itself, so it is uniform across the
 * block. With trackNan the thread also reports whether it saw a NaN.
 */
template <typename T, typename V>
SD_DEVICE inline void argmaxScanRow(const T* row, LongType vocabSize, LongType elemStride,
                                    V& localMax, LongType& localIdx, bool trackNan, bool& sawNan) {
  localMax = -DataTypeUtils::infOrMax<V>();
  localIdx = vocabSize;  // ABSENT
  auto visit = [&](V val, LongType v) {
    if (trackNan && val != val) sawNan = true;
    if (val > localMax) {
      localMax = val;
      localIdx = v;
    }
  };
  LongType scalarStart = 0;
  if constexpr (sizeof(T) == 4) {
    if (elemStride == 1 && (reinterpret_cast<uintptr_t>(row) & 15) == 0) {
      constexpr int L = kArgmaxRowLoadsInFlight;
      const auto groups = reinterpret_cast<const float4*>(row);
      const LongType groupCount = vocabSize / 4;
      const LongType span = static_cast<LongType>(blockDim.x) * L;
      for (LongType first = 0; first < groupCount; first += span) {
        float4 loaded[L];
#pragma unroll
        for (int u = 0; u < L; ++u) {
          const LongType q = first + static_cast<LongType>(u) * blockDim.x + threadIdx.x;
          if (q < groupCount) loaded[u] = groups[q];
        }
#pragma unroll
        for (int u = 0; u < L; ++u) {
          const LongType q = first + static_cast<LongType>(u) * blockDim.x + threadIdx.x;
          if (q < groupCount) {
            const T* lanes = reinterpret_cast<const T*>(&loaded[u]);
            visit(static_cast<V>(lanes[0]), 4 * q);
            visit(static_cast<V>(lanes[1]), 4 * q + 1);
            visit(static_cast<V>(lanes[2]), 4 * q + 2);
            visit(static_cast<V>(lanes[3]), 4 * q + 3);
          }
        }
      }
      scalarStart = groupCount * 4;
    }
  }
  for (LongType v = scalarStart + threadIdx.x; v < vocabSize; v += blockDim.x)
    visit(static_cast<V>(row[v * elemStride]), v);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_CUDA_ARGMAX_ROW_SCAN_CUH
