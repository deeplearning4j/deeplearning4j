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
  __shared__ LongType zLen, totalThreads;
  __shared__ const LongType *xShape, *zShape, *xStride, *zStride;
  __shared__ LongType yStride0, shift1, shift2;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShapeInfo);
    rankMinusOne = rank - 1;
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);
    yStride0 = shape::stride(yShapeInfo)[0];
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
    shift1 = (mode == 1) ? 0 : 1;  // REFLECT : SYMMETRIC
    shift2 = (mode == 1) ? 2 : 1;  // REFLECT : SYMMETRIC
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = totalThreads;

  LongType xzCoord[SD_MAX_RANK];

  for (LongType i = start; i < zLen; i += step) {
    // Compute output coordinate and offset
    INDEX2COORDS(i, rank, zShape, xzCoord);
    LongType zOffset;
    COORDS2INDEX(rank, zStride, xzCoord, zOffset);

    bool within = true;

    for (int j = rankMinusOne; j >= 0; --j) {
      if (xShape[j] == zShape[j]) continue;

      LongType leftOffset;
      LongType leftCoords[] = {yStride0 * j};
      COORDS2INDEX(1, shape::stride(yShapeInfo), leftCoords, leftOffset);
      const auto left = y[leftOffset];

      if (xzCoord[j] < left || xzCoord[j] >= left + xShape[j]) {
        within = false;

        if (mode != 0) {  // REFLECT or SYMMETRIC
          xzCoord[j] = xzCoord[j] - left;

          if (xzCoord[j] < 0) {  // Left boundary
            xzCoord[j] = -xzCoord[j] - shift1;
          } else if (xzCoord[j] >= xShape[j]) {  // Right boundary
            xzCoord[j] = 2 * xShape[j] - xzCoord[j] - shift2;
          }
        }

        break;
      } else {
        xzCoord[j] -= left;
      }
    }

    if (within || mode != 0) {
      LongType xOffset;
      COORDS2INDEX(rank, xStride, xzCoord, xOffset);
      z[zOffset] = within ? x[xOffset] : x[xOffset];  // Handles REFLECT or SYMMETRIC
    } else {
      z[zOffset] = padVal;  // CONSTANT padding
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
  __shared__ LongType rankX, rankZ;
  __shared__ const LongType* shapeX;
  __shared__ const LongType* strideX;
  __shared__ const LongType* shapeZ;
  __shared__ const LongType* strideZ;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<T const*>(vx);
    z = reinterpret_cast<T*>(vz);

    rankX = shape::rank(xShape);
    rankZ = shape::rank(zShape);
    shapeX = shape::shapeOf(xShape);
    strideX = shape::stride(xShape);
    shapeZ = shape::shapeOf(zShape);
    strideZ = shape::stride(zShape);
  }
  __syncthreads();

  const auto start = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  LongType zCoords[SD_MAX_RANK];
  LongType xOffset, zOffset;

  for (LongType i = start; i < zLen; i += step) {
    // Compute coordinates and offset for the output
    INDEX2COORDS(i, rankZ, shapeZ, zCoords);
    COORDS2INDEX(rankZ, strideZ, zCoords, zOffset);

    // Adjust input offset based on the mirror padding logic
    if (i < leftSide) {  // Left side
      const LongType mirrorIndex = leftSideCorrected - i;
      COORDS2INDEX(rankX, strideX, &mirrorIndex, xOffset);
    } else if (i < leftSide + xLen) {  // Middle section
      const LongType middleIndex = i - leftSide;
      COORDS2INDEX(rankX, strideX, &middleIndex, xOffset);
    } else {  // Right side
      const LongType mirrorIndex = len - i;
      COORDS2INDEX(rankX, strideX, &mirrorIndex, xOffset);
    }

    // Assign value from input to output
    if (zOffset < zLen && xOffset < xLen) {
      z[zOffset] = x[xOffset];
    }
  }
}

template <typename F, typename I>
static SD_KERNEL void mirrorPadKernel(void const* vx,  const LongType* xShape, void* vz,  const LongType* zShape,
                                      LongType outLen, void const* paddings, const LongType* paddingShape,
                                      int reflBorder) {
  __shared__ F const* x;
  __shared__ I const* pads;
  __shared__ F* z;
  __shared__ LongType rank;
  __shared__ sd::LongType *zStride;
  __shared__ sd::LongType *xStride;
  __shared__  LongType* zShapeArr;
  __shared__  LongType* xShapeArr;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShape);
    zShapeArr = shape::shapeOf(zShape);
    zStride = shape::stride(zShape);
    xShapeArr = shape::shapeOf(xShape);
    xStride = shape::stride(xShape);

    x = reinterpret_cast<F const*>(vx);
    pads = reinterpret_cast<I const*>(paddings);
    z = reinterpret_cast<F*>(vz);
  }
  __syncthreads();

  const auto start = threadIdx.x + blockIdx.x * blockDim.x;
  const auto step = blockDim.x * gridDim.x;

  LongType xzCoord[SD_MAX_RANK];
  LongType coords[2];

  for (LongType i = start; i < outLen; i += step) {
    // Calculate output coordinate and offset
    INDEX2COORDS(i, rank, zShapeArr, xzCoord);
    LongType outOffset;
    COORDS2INDEX(rank, zStride, xzCoord, outOffset);

    // Adjust input coordinates based on mirror padding
    for (LongType j = 0; j < rank; ++j) {
      const auto inLen = shape::sizeAt(xShape, j);

      coords[0] = j;
      coords[1] = 0;

      LongType padOffset;
      COORDS2INDEX(2, shape::stride(paddingShape), coords, padOffset);
      const auto leftSide = pads[padOffset];
      const auto leftSideCorrected = leftSide - reflBorder;
      const auto len = 2 * (inLen - 1) + leftSide + reflBorder;

      if (xzCoord[j] < leftSide) {  // Left side
        xzCoord[j] = leftSideCorrected - xzCoord[j];
      } else if (xzCoord[j] < leftSide + inLen) {  // Middle
        xzCoord[j] = xzCoord[j] - leftSide;
      } else if (xzCoord[j] < len) {  // Right side
        xzCoord[j] = len - xzCoord[j];
      } else {  // Beyond the mirrored region
        xzCoord[j] = xzCoord[j] - len;
      }
    }

    // Calculate input offset and assign value
    LongType inOffset;
    COORDS2INDEX(rank, xStride, xzCoord, inOffset);
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
