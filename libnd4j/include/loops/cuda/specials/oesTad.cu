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

//
// @author raver119@gmail.com
//
#include <ops/specials_cuda.h>

// Odd-even transposition sort of every TAD: a block per TAD (grid-stride over the TADs), its threads stride over the
// pairs of a round, and a TAD of n elements takes n rounds. Lengths, counts and positions are 64-bit: an array of 2^31
// elements or more has TADs to count in 64 bits.

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL SD_INLINE void execOesTadKernelKey(void *vx, sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                                   sd::LongType *dimension, long long int dimensionLength, sd::LongType const *tadShapeInfo,
                                   sd::LongType const *tadOffsets, bool descending) {
  auto x = static_cast<X *>(vx);
  auto y = static_cast<Y *>(vy);

  __shared__ sd::LongType xLength;
  __shared__ sd::LongType xTadLength;
  __shared__ sd::LongType numTads;
  __shared__ sd::LongType tadRank;
  __shared__ sd::LongType *tadShape;
  __shared__ sd::LongType *tadStride;

  if (threadIdx.x == 0) {
    xLength = shape::length(xShapeInfo);
    xTadLength = shape::length(tadShapeInfo);
    numTads = xLength / xTadLength;

    // Cache shape information
    tadRank = shape::rank(tadShapeInfo);
    tadShape = shape::shapeOf(tadShapeInfo);
    tadStride = shape::stride(tadShapeInfo);
  }
  __syncthreads();

  for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    auto dx = x + tadOffsets[r];
    auto dy = y + tadOffsets[r];

    // this is general loop, we go uncached
    sd::LongType iterations = xTadLength;

    for (sd::LongType i = 0; i < iterations; i++) {
      if (i % 2 == 0) {
        for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
          auto top = 2 * tid + 1;
          if (top < xTadLength) {
            sd::LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
            sd::LongType t0Offset, t1Offset;

            INDEX2COORDS(top - 1, tadRank, tadShape, t0Coords);
            COORDS2INDEX(tadRank, tadStride, t0Coords, t0Offset);
            INDEX2COORDS(top, tadRank, tadShape, t1Coords);
            COORDS2INDEX(tadRank, tadStride, t1Coords, t1Offset);

            if (descending ? (dx[t0Offset] < dx[t1Offset]) : (dx[t0Offset] > dx[t1Offset])) {
              X dt0 = dx[t0Offset];
              dx[t0Offset] = dx[t1Offset];
              dx[t1Offset] = dt0;

              Y dy0 = dy[t0Offset];
              dy[t0Offset] = dy[t1Offset];
              dy[t1Offset] = dy0;
            }
          }
        }
      } else {
        for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
          auto top = 2 * tid + 2;
          if (top < xTadLength) {
            sd::LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
            sd::LongType t0Offset, t1Offset;

            INDEX2COORDS(top - 1, tadRank, tadShape, t0Coords);
            COORDS2INDEX(tadRank, tadStride, t0Coords, t0Offset);
            INDEX2COORDS(top, tadRank, tadShape, t1Coords);
            COORDS2INDEX(tadRank, tadStride, t1Coords, t1Offset);

            if (descending ? (dx[t0Offset] < dx[t1Offset]) : (dx[t0Offset] > dx[t1Offset])) {
              X dt0 = dx[t0Offset];
              dx[t0Offset] = dx[t1Offset];
              dx[t1Offset] = dt0;

              Y dy0 = dy[t0Offset];
              dy[t0Offset] = dy[t1Offset];
              dy[t1Offset] = dy0;
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// sharedBytes is the dynamic shared memory the launch gave the block: a TAD that fits in it is sorted there
template <typename T>
SD_KERNEL SD_INLINE void execOesTadKernel(void *vx, sd::LongType const *xShapeInfo, sd::LongType *dimension,
                                sd::LongType dimensionLength,
                                sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, bool descending,
                                sd::LongType sharedBytes) {
  auto x = static_cast<T *>(vx);

  __shared__ sd::LongType xLength;
  __shared__ sd::LongType xTadLength;
  __shared__ sd::LongType numTads;
  __shared__ T *shmem;
  __shared__ bool cached;
  __shared__ sd::LongType tadRank;
  __shared__ sd::LongType *tadShape;
  __shared__ sd::LongType *tadStride;

  if (threadIdx.x == 0) {
    xLength = shape::length(xShapeInfo);
    xTadLength = shape::length(tadShapeInfo);
    numTads = xLength / xTadLength;

    extern __shared__ unsigned char shrd[];
    shmem = (T *)shrd;

    cached = xTadLength <= (sharedBytes / static_cast<sd::LongType>(sizeof(T)));

    // Cache shape information
    tadRank = shape::rank(tadShapeInfo);
    tadShape = shape::shapeOf(tadShapeInfo);
    tadStride = shape::stride(tadShapeInfo);
  }
  __syncthreads();

  for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    auto dx = x + tadOffsets[r];

    // this is general loop, we go uncached
    sd::LongType iterations = xTadLength;
    if (cached) {
      for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
        sd::LongType xCoords[SD_MAX_RANK];
        sd::LongType xOffset;
        INDEX2COORDS(tid, tadRank, tadShape, xCoords);
        COORDS2INDEX(tadRank, tadStride, xCoords, xOffset);
        shmem[tid] = dx[xOffset];
      }

      __syncthreads();
      dx = shmem;
    }

    for (sd::LongType i = 0; i < iterations; i++) {
      if (i % 2 == 0) {
        for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
          auto top = 2 * tid + 1;
          if (top < xTadLength) {
            sd::LongType t0Offset, t1Offset;
            if (cached) {
              // shmem is loaded linearly [0..n-1]; use linear indices directly
              t0Offset = top - 1;
              t1Offset = top;
            } else {
              sd::LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
              INDEX2COORDS(top - 1, tadRank, tadShape, t0Coords);
              COORDS2INDEX(tadRank, tadStride, t0Coords, t0Offset);
              INDEX2COORDS(top, tadRank, tadShape, t1Coords);
              COORDS2INDEX(tadRank, tadStride, t1Coords, t1Offset);
            }

            if (descending ? (dx[t0Offset] < dx[t1Offset]) : (dx[t0Offset] > dx[t1Offset])) {
              T dt0 = dx[t0Offset];
              dx[t0Offset] = dx[t1Offset];
              dx[t1Offset] = dt0;
            }
          }
        }
      } else {
        for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
          auto top = 2 * tid + 2;
          if (top < xTadLength) {
            sd::LongType t0Offset, t1Offset;
            if (cached) {
              // shmem is loaded linearly [0..n-1]; use linear indices directly
              t0Offset = top - 1;
              t1Offset = top;
            } else {
              sd::LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
              INDEX2COORDS(top - 1, tadRank, tadShape, t0Coords);
              COORDS2INDEX(tadRank, tadStride, t0Coords, t0Offset);
              INDEX2COORDS(top, tadRank, tadShape, t1Coords);
              COORDS2INDEX(tadRank, tadStride, t1Coords, t1Offset);
            }

            if (descending ? (dx[t0Offset] < dx[t1Offset]) : (dx[t0Offset] > dx[t1Offset])) {
              T dt0 = dx[t0Offset];
              dx[t0Offset] = dx[t1Offset];
              dx[t1Offset] = dt0;
            }
          }
        }
      }
      __syncthreads();
    }

    if (cached) {
      dx = x + tadOffsets[r];
      for (sd::LongType tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
        sd::LongType xCoords[SD_MAX_RANK];
        sd::LongType xOffset;
        INDEX2COORDS(tid, tadRank, tadShape, xCoords);
        COORDS2INDEX(tadRank, tadStride, xCoords, xOffset);
        dx[xOffset] = shmem[tid];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void oesTadGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                           sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType const *tadShapeInfo,
                           sd::LongType const *tadOffsets, bool descending) {
  // threads-first dims (getSortTadLarge): x = threads per block, y = blocks, z = dynamic shared bytes, which the
  // kernel uses to hold a TAD
  execOesTadKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      vx, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending,
      static_cast<sd::LongType>(launchDims.z));

  sd::DebugHelper::checkErrorCode(stream, "execOesTadKernel failed");
}

template <typename X, typename Y>
SD_HOST void oesTadGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                              void *vy, sd::LongType const *yShapeInfo, sd::LongType *dimension,
                              sd::LongType dimensionLength,
                              sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, bool descending) {
  execOesTadKernelKey<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      vx, xShapeInfo, vy, yShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
  sd::DebugHelper::checkErrorCode(stream, "execOesTadKernelKey failed");
}

#ifdef SD_SPLIT_TYPE_INDEX
#define SD_COMMON_TYPES_FIRST SD_SPLIT_TYPE_LIST
#else
#define SD_COMMON_TYPES_FIRST SD_COMMON_TYPES
#endif
#if !defined(SD_SPLIT_TYPE_INDEX) || (COUNT_NARG(SD_COMMON_TYPES) > SD_SPLIT_TYPE_INDEX)
BUILD_SINGLE_TEMPLATE( void oesTadGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                       sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType const *tadShapeInfo,
                       sd::LongType const *tadOffsets, bool descending),
                      SD_COMMON_TYPES_FIRST);

BUILD_DOUBLE_TEMPLATE( void oesTadGenericKey,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
                       sd::LongType const *yShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                       sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, bool descending),
                      SD_COMMON_TYPES_FIRST, SD_COMMON_TYPES);
#endif
#ifdef SD_COMMON_TYPES_FIRST
#undef SD_COMMON_TYPES_FIRST
#endif
