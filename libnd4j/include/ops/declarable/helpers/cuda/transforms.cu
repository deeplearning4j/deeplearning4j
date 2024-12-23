/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void invertPermutationCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                            const LongType* zShapeInfo) {
  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType len, totalThreads;

  // Cache shape information in shared memory
  __shared__ int xRank, zRank;
  __shared__ LongType *xShape, *zShape;
  __shared__ LongType *xStride, *zStride;

  if (threadIdx.x == 0) {
    len = shape::length(xShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    // Cache ranks
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }

  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType xCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xOffset;
  LongType zOffset;

  for (LongType i = tid; i < len; i += totalThreads) {
    INDEX2COORDS(i, xRank, xShape, xCoords);
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);
    const LongType index = x[xOffset];
    INDEX2COORDS(index, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    z[zOffset] = i;
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void invertPermutationCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                                  const int sharedMemory, const cudaStream_t* stream, const void* vx,
                                                  const LongType* xShapeInfo, void* vz,
                                                  const LongType* zShapeInfo) {
  invertPermutationCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMemory, *stream>>>(vx, xShapeInfo, vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "invertPermutationCuda failed");

}

////////////////////////////////////////////////////////////////////////
void invertPermutation(LaunchContext* context, NDArray& input, NDArray& output) {
  dim3 invertPermuteDims = invertPermutationDims(input.lengthOf());
  PointersManager manager(context, "invertPermutation");

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_SINGLE_SELECTOR(input.dataType(), invertPermutationCudaLauncher,
                        (invertPermuteDims.x, invertPermuteDims.y, invertPermuteDims.z,context->getCudaStream(), input.specialBuffer(),
                            input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo()),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void traceCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                const LongType* zShapeInfo, const LongType diagLen) {
  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ T sharedMem[SD_CUDA_BLOCK_SIZE];
  __shared__ int xRank, zRank;  // xRank = zRank + 2
  __shared__ LongType xLen, zLen;

  // Cache shape information in shared memory
  __shared__ LongType *xShape, *zShape;
  __shared__ LongType *xStride, *zStride;

  if (threadIdx.x == 0) {
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);
    xLen = shape::length(xShapeInfo);
    zLen = shape::length(zShapeInfo);  // corresponds to number of matrices

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType m = blockIdx.x; m < zLen; m += gridDim.x) {  // one block per each element of z, that is per each matrix
    INDEX2COORDS(m, zRank, zShape, coords);
    LongType zOffset;
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    sharedMem[threadIdx.x] = 0;

    for (LongType i = threadIdx.x; i < diagLen; i += blockDim.x) {
      coords[zRank] = coords[zRank + 1] = i;
      LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      sharedMem[threadIdx.x] += x[xOffset];
    }

    __syncthreads();

    // aggregate sum
    for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
      if (threadIdx.x < activeThreads) sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
      __syncthreads();
    }

    if (threadIdx.x == 0) z[zOffset] = *sharedMem;
    __syncthreads();
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
static void traceCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                              const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, void* vz,
                              const LongType* zShapeInfo, const LongType diagLen) {
  traceCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, diagLen);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "traceCuda failed");

}

///////////////////////////////////////////////////////////////////
void trace(LaunchContext* context, NDArray& input, NDArray& output) {
  PointersManager manager(context, "trace");

  const LongType diagLen = input.sizeAt(-1) < input.sizeAt(-2) ? input.sizeAt(-1) : input.sizeAt(-2);
  const int threadsPerBlock = SD_CUDA_BLOCK_SIZE;
  const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = 1024;

  dim3 traceDims2 = traceDims(output.lengthOf());
  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_SINGLE_SELECTOR(input.dataType(), traceCudaLauncher,
                        (traceDims2.y, traceDims2.x, traceDims2.z, context->getCudaStream(), input.specialBuffer(),
                            input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), diagLen),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void triuBPCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                 const LongType* zShapeInfo, const int diag) {
  // x and z have same shapes
  const auto x = reinterpret_cast<const T*>(vx);  // gradO
  auto z = reinterpret_cast<T*>(vz);              // gradI

  __shared__ int rank, areSameOffsets;
  __shared__ LongType len, totalThreads;  // xLen = zLen

  // Cache shape information in shared memory
  __shared__ LongType *xShape, *zShape;
  __shared__ LongType *xStride, *zStride;
  __shared__ int xRank, zRank;

  if (threadIdx.x == 0) {
    areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    rank = shape::rank(xShapeInfo);
    len = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    // Cache ranks
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }

  __syncthreads();

  LongType coords[SD_MAX_RANK];

  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < len; i += totalThreads) {
    INDEX2COORDS(i, zRank, zShape, coords);

    sd::LongType zOffset;
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    if ((coords[rank - 2] + diag > coords[rank - 1]))  // row + diag > col
      z[zOffset] = 0;
    else {
      sd::LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      z[zOffset] = x[areSameOffsets ? zOffset : xOffset];
    }
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
static void triuBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, void* vz,
                               const LongType* zShapeInfo, const int diag) {
  triuBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, diag);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "triuBP failed");

}

///////////////////////////////////////////////////////////////////
void triuBP(LaunchContext* context, NDArray& input, NDArray& gradO, NDArray& gradI,
            const int diagonal) {
  const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  const int blocksPerGrid = (gradO.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(LongType) * gradO.rankOf() + 128;
  dim3 triuDims2 = triuDims(gradO.lengthOf(),gradO.rankOf());
  PointersManager manager(context, "triuBP");

  NDArray::prepareSpecialUse({&gradI}, {&gradO});
  BUILD_SINGLE_SELECTOR(gradI.dataType(), triuBPCudaLauncher,
                        (triuDims2.y, triuDims2.x, triuDims2.z, context->getCudaStream(), gradO.specialBuffer(),
                            gradO.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), diagonal),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({&gradI}, {&gradO});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void tileBPCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                 const LongType* zShapeInfo,
                                 LongType* globMem) {
  // x and z have same shapes
  const auto x = reinterpret_cast<const T*>(vx);  // gradO
  auto z = reinterpret_cast<T*>(vz);              // gradI

  __shared__ int xRank, zRank;                               // xRank >= zRank
  __shared__ LongType numOfXOffsets, zLen, totalThreads;     // xLen >= zLen

  // Cache shape information in shared memory
  __shared__ LongType *xShape, *zShape;
  __shared__ LongType *xStride, *zStride;

  if (threadIdx.x == 0) {
    xRank = shape::rank(zShapeInfo);
    zRank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
    numOfXOffsets = shape::length(xShapeInfo) / zLen;

    totalThreads = gridDim.x * blockDim.x;

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }

  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType memBuff[SD_MAX_RANK * 2];
  auto xOffsets = globMem + tid * numOfXOffsets;

  for (LongType i = tid; i < zLen; i += totalThreads) {
    LongType zCoords[SD_MAX_RANK];
    LongType zOffset;

    INDEX2COORDS(i, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    shape::outerArrayOffsets(xOffsets, i, xShapeInfo, zShapeInfo, memBuff, nullptr);

    z[zOffset] = x[xOffsets[0]];                      // first offset
    for (LongType j = 1; j < numOfXOffsets; ++j)      // rest offsets
      z[zOffset] += x[xOffsets[j]];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void tileBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, void* vz,
                               const LongType* zShapeInfo, LongType* globMem) {
  tileBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, globMem);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "tileBPCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////////
void tileBP(LaunchContext* context, NDArray gradO /*input*/, NDArray& gradI /*output*/,
            const std::vector<LongType> reps) {
  auto grad0Shape = gradO.getShapeAsVector();
  NDArray memBuff(
      'c', grad0Shape, INT64,
      context);  // empty auxiliary array for storing device memory which will be used in kernel calculations

  dim3 tileDims2 = tileDims(gradI.lengthOf(),gradI.rankOf());
  PointersManager manager(context, "tileBP");

  NDArray::prepareSpecialUse({&gradI}, {&gradO, &memBuff});
  BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBPCudaLauncher,
                        (tileDims2.y, tileDims2.x, tileDims2.z, context->getCudaStream(), gradO.specialBuffer(),
                            gradO.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(),
                            reinterpret_cast<sd::LongType*>(memBuff.specialBuffer())),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&gradI}, {&gradO, &memBuff});

  manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
void eye(LaunchContext* context, NDArray& output) { output.setIdentity(); }

}  // namespace helpers
}  // namespace ops
}  // namespace sd
