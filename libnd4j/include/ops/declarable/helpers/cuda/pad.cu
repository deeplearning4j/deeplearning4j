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
// x - input, y - paddings, z - output
template <typename X, typename Y>
SD_KERNEL static void padCuda(const int mode, const void* vx, const LongType* xShapeInfo, const void* vy,
                              const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                              const void* vPadVal) {
  const X padVal = *reinterpret_cast<const X*>(vPadVal);

  const auto x = reinterpret_cast<const X*>(vx);
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<X*>(vz);

  __shared__ int rank, rankMinusOne;
  __shared__ LongType zLen, totalThreads, *coords, *xShape, *zShape, shift1, shift2, yStride0;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType*>(shmem);
    zLen = shape::length(zShapeInfo);
    xShape = shape::shapeOf(const_cast<LongType*>(xShapeInfo));
    zShape = shape::shapeOf(const_cast<LongType*>(zShapeInfo));
    yStride0 = shape::stride(const_cast<LongType*>(yShapeInfo))[0];
    rank = shape::rank(xShapeInfo);
    zLen = shape::length(zShapeInfo);
    rankMinusOne = rank - 1;
    totalThreads = gridDim.x * blockDim.x;
    shift1 = mode == 1 ? 0 : 1;  // REFLECT : SYMMETRIC
    shift2 = mode == 1 ? 2 : 1;  // REFLECT : SYMMETRIC
  }

  __syncthreads();

  auto xzCoord = coords + threadIdx.x * rank;  // we use xzCoord storage both for x and z arrays

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (mode == 0) {  // CONSTANT case

    for (LongType i = tid; i < zLen; i += totalThreads) {
      INDEX2COORDS(i, rank, zShapeInfo, xzCoord);
      LongType zOffset;
      COORDS2INDEX(rank, zShape, xzCoord, zOffset);

      bool within = true;
      for (int j = rankMinusOne; j >= 0; --j) {
        if (xShape[j] == zShape[j]) continue;
        const auto left = y[shape::getIndexOffset(yStride0 * j, yShapeInfo)];
        if (xzCoord[j] < left || xzCoord[j] >= left + xShape[j]) {
          within = false;
          break;
        } else {
          xzCoord[j] = xzCoord[j] - left;
        }
      }

      if (within) {
        LongType xOffset;
        COORDS2INDEX(rank, xShape, xzCoord, xOffset);
        z[zOffset] = x[xOffset];
      } else {
        z[zOffset] = padVal;
      }
    }
  } else {  // REFLECT and SYMMETRIC cases

    for (LongType i = tid; i < zLen; i += totalThreads) {
      INDEX2COORDS(i, rank, zShapeInfo, xzCoord);
      LongType zOffset;
      COORDS2INDEX(rank, zShape, xzCoord, zOffset);

      for (int j = rankMinusOne; j >= 0; --j) {
        if (xShape[j] == zShape[j]) continue;
        xzCoord[j] = xzCoord[j] - y[shape::getIndexOffset(yStride0 * j, yShapeInfo)];  // are ready to fill middle (within input dimension range)
        if (xzCoord[j] < 0)
          xzCoord[j] = -xzCoord[j] - shift1;  // means fill from left
        else if (xzCoord[j] >= xShape[j])
          xzCoord[j] = 2 * xShape[j] - xzCoord[j] - shift2;  // means fill from right
      }

      LongType xOffset;
      COORDS2INDEX(rank, xShape, xzCoord, xOffset);
      z[zOffset] = x[xOffset];
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void padCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                            const cudaStream_t* stream, const int mode, const void* vx, const LongType* xShapeInfo,
                            const void* vy, const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                            const void* padVal) {
  padCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(mode, vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                        zShapeInfo, padVal);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "padCuda failed");

}

///////////////////////////////////////////////////////////////////
void pad(LaunchContext* context, const int mode, NDArray& input, NDArray& paddings, NDArray& output,
         NDArray& padValue) {
  PointersManager manager(context, "pad");

  NDArray::prepareSpecialUse({&output}, {&input, &paddings, &padValue});

  dim3 padLaunch = padDims(output.lengthOf(),output.rankOf());
  const auto xType = input.dataType();
  const auto yType = paddings.dataType();

  BUILD_DOUBLE_SELECTOR(
      xType, yType, padCudaLauncher,
      (padLaunch.y, padLaunch.x, padLaunch.z, context->getCudaStream(), mode, input.specialBuffer(),
          input.specialShapeInfo(), paddings.specialBuffer(), paddings.specialShapeInfo(), output.specialBuffer(),
          output.specialShapeInfo(), padValue.specialBuffer()),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&output}, {&input, &paddings, &padValue});
  manager.synchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mirrorPadLinearKernel(void const* vx, const LongType* xShape, void* vz,
                                            const LongType* zShape,
                                            LongType leftSide, LongType leftSideCorrected, LongType xLen, LongType len,
                                            LongType zLen) {
  __shared__ T const* x;
  __shared__ T* z;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T const*>(vx);
    z = reinterpret_cast<T*>(vz);
  }
  __syncthreads();
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i < zLen; i += step) {
    auto zIndex = shape::getIndexOffset(i, zShape);
    auto xIndex = shape::getIndexOffset(len - i, xShape);
    if (i < leftSide)  // left side
      xIndex = shape::getIndexOffset(leftSideCorrected - i, xShape);

    else if (i >= leftSide && i < leftSide + xLen)  // middle
      xIndex = shape::getIndexOffset(i - leftSide, xShape);

    if(zIndex >= 0 && xIndex >= 0 && zIndex < zLen && xIndex < xLen)
      z[zIndex] = x[xIndex];
  }
}

template <typename F, typename I>
static SD_KERNEL void mirrorPadKernel(void const* vx, const LongType* xShape, void* vz, const LongType* zShape,
                                      LongType outLen, void const* paddings, const LongType* paddingShape,
                                      int reflBorder) {
  __shared__ F const* x;
  __shared__ I const* pads;
  __shared__ F* z;
  __shared__ LongType zRank, rank;
  __shared__ LongType* xIdx;
  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    xIdx = reinterpret_cast<LongType*>(shmem);
    rank = shape::rank(xShape);

    x = reinterpret_cast<F const*>(vx);  //
    pads = reinterpret_cast<I const*>(paddings);
    z = reinterpret_cast<F*>(vz);
  }
  __syncthreads();
  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  LongType coords[2];

  for (LongType i = start; i < outLen; i += step) {
    auto xzCoord = xIdx + threadIdx.x * rank;
    INDEX2COORDS(i, rank, zShape, xzCoord);
    LongType outOffset;
    COORDS2INDEX(rank, shape::shapeOf(zShape), xzCoord, outOffset);
    for (LongType j = 0; j < rank; j++) {
      const LongType inLen = shape::sizeAt(xShape, j);
      coords[0] = j;
      coords[1] = 0;
      LongType padOffset;
      COORDS2INDEX(2, shape::shapeOf(paddingShape), coords, padOffset);  // padding already has rank 2
      const auto leftSide = pads[padOffset];
      const auto leftSideCorrected = leftSide - reflBorder;
      const LongType len = 2 * (inLen - 1) + leftSide + reflBorder;

      if (xzCoord[j] < leftSide)  // left side
        xzCoord[j] = leftSideCorrected - xzCoord[j];

      else if (xzCoord[j] >= leftSide && xzCoord[j] < leftSide + inLen)  // middle
        xzCoord[j] = xzCoord[j] - leftSide;

      else if (len > xzCoord[j])  // right side
        xzCoord[j] = len - xzCoord[j];
      else
        xzCoord[j] = xzCoord[j] - len;
    }

    LongType inOffset;
    COORDS2INDEX(rank, shape::shapeOf(xShape), xzCoord, inOffset);
    z[outOffset] = x[inOffset];
  }
}

template <typename F, typename I>
static void mirrorPad_(LaunchContext* context, NDArray& input, NDArray& paddings, NDArray& output,
                       const int mode) {
  // mode:  0 - REFLECT, else - SYMMETRIC
  const int reflBorder = (bool)mode ? 1 : 0;
  const LongType rank = input.rankOf();
  const LongType outLen = output.lengthOf();
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({&output}, {&input, &paddings});

  if (rank <= 1) {
    const LongType inLen = input.isScalar() ? 1 : input.lengthOf();
    const auto leftSide = paddings.e<LongType>(0);
    const auto leftSideCorrected = leftSide - reflBorder;
    const LongType len = 2 * (inLen - 1) + leftSide + reflBorder;

    dim3 mirrorPadLinearDims2 = mirrorPadLinearDims(len);
    mirrorPadLinearKernel<F><<<mirrorPadLinearDims2.y, mirrorPadLinearDims2.x, mirrorPadLinearDims2.z, *stream>>>(
        input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftSide,
        leftSideCorrected, inLen, len, outLen);
    DebugHelper::checkErrorCode(stream, "helpers::mirrorPadLinearKernel(...) failed");
  } else {
    dim3 mirrorPadDims = mirrorPadTad(output.lengthOf(),input.rankOf());
    mirrorPadKernel<F, I><<<mirrorPadDims.y, mirrorPadDims.x, mirrorPadDims.z, *stream>>>(
        input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), outLen,
        paddings.specialBuffer(), paddings.specialShapeInfo(), reflBorder);
    DebugHelper::checkErrorCode(stream, "helpers::mirrorPadKernel(...) failed");
  }
  NDArray::registerSpecialUse({&output}, {&input, &paddings});
}

void mirrorPad(LaunchContext* context, NDArray& input, NDArray& paddings, NDArray& output,
               const int mode) {
  BUILD_DOUBLE_SELECTOR(input.dataType(), paddings.dataType(), mirrorPad_, (context, input, paddings, output, mode),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
