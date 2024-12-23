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
//  @author raver119@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/roll.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void SD_DEVICE rollKernelLinearStage1Dev(const void *vx, const LongType *xShapeInfo, void *vz,
                                                const LongType *zShapeInfo, LongType fullLength,
                                                int actualShift) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType rank;
  __shared__ const LongType *xShape, *xStride, *zShape, *zStride;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  LongType xCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];

  for (LongType i = threadIdx.x + blockIdx.x * blockDim.x; i < actualShift; i += blockDim.x * gridDim.x) {
    LongType sourceIndex = fullLength - actualShift + i;

    LongType xOffsetA, xOffsetB, zOffsetA, zOffsetB;

    // Calculate input offsets
    INDEX2COORDS(i, rank, xShape, xCoords);
    COORDS2INDEX(rank, xStride, xCoords, xOffsetA);

    INDEX2COORDS(sourceIndex, rank, xShape, xCoords);
    COORDS2INDEX(rank, xStride, xCoords, xOffsetB);

    // Calculate output offsets
    INDEX2COORDS(i, rank, zShape, zCoords);
    COORDS2INDEX(rank, zStride, zCoords, zOffsetA);

    INDEX2COORDS(sourceIndex, rank, zShape, zCoords);
    COORDS2INDEX(rank, zStride, zCoords, zOffsetB);

    // Perform element swap
    auto eA = x[xOffsetA];
    auto eB = x[xOffsetB];

    z[zOffsetA] = eB;
    z[zOffsetB] = eA;
  }
}


template <typename T>
static void SD_KERNEL rollKernelLinearStage1(const void *vx, const LongType *xShapeInfo, void *vz,
                                             const LongType *zShapeInfo, LongType fullLength, int actualShift) {
  rollKernelLinearStage1Dev<T>(vx, xShapeInfo, vz, zShapeInfo, fullLength, actualShift);
}

template <typename T>
static void SD_KERNEL rollKernelLinearStage3(const void *vx, const LongType *xShapeInfo, void *vz,
                                             const LongType *zShapeInfo, LongType fullLength, int actualShift,
                                             int remainShift) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType rank;
  __shared__ const LongType *xShape, *xStride, *zShape, *zStride;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  LongType xCoordsA[SD_MAX_RANK];
  LongType xCoordsB[SD_MAX_RANK];
  LongType zCoordsA[SD_MAX_RANK];
  LongType zCoordsB[SD_MAX_RANK];

  for (LongType i = threadIdx.x + blockIdx.x * blockDim.x; i < actualShift; i += blockDim.x * gridDim.x) {
    LongType remainIdx = i + actualShift;
    LongType sourceIndex = remainIdx + remainShift;

    LongType xOffsetA, xOffsetB, zOffsetA, zOffsetB;

    // Calculate offsets for input and output
    INDEX2COORDS(remainIdx, rank, xShape, xCoordsA);
    COORDS2INDEX(rank, xStride, xCoordsA, xOffsetA);

    INDEX2COORDS(sourceIndex, rank, xShape, xCoordsB);
    COORDS2INDEX(rank, xStride, xCoordsB, xOffsetB);

    INDEX2COORDS(remainIdx, rank, zShape, zCoordsA);
    COORDS2INDEX(rank, zStride, zCoordsA, zOffsetA);

    INDEX2COORDS(sourceIndex, rank, zShape, zCoordsB);
    COORDS2INDEX(rank, zStride, zCoordsB, zOffsetB);

    // Swap the elements
    auto eA = x[xOffsetA];
    auto eB = x[xOffsetB];

    z[zOffsetA] = eB;
    z[zOffsetB] = eA;
  }
}


template <typename T>
static void SD_KERNEL rollKernelLinearStage3(const void *vx, const LongType *xShapeInfo, void *vz,
                                             const LongType *zShapeInfo, LongType fullLength, int actualShift,
                                             int remainShift) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType rank;
  __shared__ const LongType *xShape, *xStride, *zShape, *zStride;

  if (threadIdx.x == 0) {
    rank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  LongType xCoordsA[SD_MAX_RANK];
  LongType xCoordsB[SD_MAX_RANK];
  LongType zCoordsA[SD_MAX_RANK];
  LongType zCoordsB[SD_MAX_RANK];

  for (LongType i = threadIdx.x + blockIdx.x * blockDim.x; i < actualShift; i += blockDim.x * gridDim.x) {
    LongType remainIdx = i + actualShift;
    LongType sourceIndex = remainIdx + remainShift;

    LongType xOffsetA, xOffsetB, zOffsetA, zOffsetB;

    // Calculate offsets for input and output
    INDEX2COORDS(remainIdx, rank, xShape, xCoordsA);
    COORDS2INDEX(rank, xStride, xCoordsA, xOffsetA);

    INDEX2COORDS(sourceIndex, rank, xShape, xCoordsB);
    COORDS2INDEX(rank, xStride, xCoordsB, xOffsetB);

    INDEX2COORDS(remainIdx, rank, zShape, zCoordsA);
    COORDS2INDEX(rank, zStride, zCoordsA, zOffsetA);

    INDEX2COORDS(sourceIndex, rank, zShape, zCoordsB);
    COORDS2INDEX(rank, zStride, zCoordsB, zOffsetB);

    // Swap the elements
    auto eA = x[xOffsetA];
    auto eB = x[xOffsetB];

    z[zOffsetA] = eB;
    z[zOffsetB] = eA;
  }
}


template <typename T>
static void SD_DEVICE swapTadsKernel(void *vx, void *vz, const LongType *zShapeInfo, LongType tadLength) {
  auto x = reinterpret_cast<T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType rank, *zShape, *zStride;

  if (threadIdx.x == 0) {
    rank = shape::rank(zShapeInfo);
    zShape = shape::shapeOf(const_cast<LongType*>(zShapeInfo));
    zStride = shape::stride(const_cast<LongType*>(zShapeInfo));
  }
  __syncthreads();

  LongType zCoords[SD_MAX_RANK];

  for (LongType e = threadIdx.x + blockIdx.x * blockDim.x; e < tadLength; e += gridDim.x * blockDim.x) {
    INDEX2COORDS(e, rank, zShape, zCoords);

    LongType zOffset;
    COORDS2INDEX(rank, zStride, zCoords, zOffset);

    // Swap the elements
    auto eA = x[zOffset];
    auto eB = z[zOffset];

    x[zOffset] = eB;
    z[zOffset] = eA;
  }
}


template <typename T>
static void SD_KERNEL rollKernelFullAnyDimensionStage1(const void *vx, const LongType *xTadShapeInfo,
                                                       const LongType *xTadOffsets, void *vz,
                                                       const LongType *zTadShapeInfo,
                                                       const LongType *zTadOffsets, int numTads, LongType tadLength, int dim, LongType sizeAt,
                                                       int theShift) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  for (int e = blockIdx.x + theShift; e < sizeAt - theShift; e += gridDim.x) {
    int sourceIndex = dim * sizeAt + e - theShift;
    int targetIndex = dim * sizeAt + e;

    swapTadsKernel<T>(z + xTadOffsets[sourceIndex], z + xTadOffsets[targetIndex], zTadShapeInfo, tadLength);
  }
}

template <typename T>
static void SD_KERNEL rollKernelFullAnyDimensionStage2(void *vx, const LongType *xTadShapeInfo,
                                                       const LongType *xTadOffsets, void *vz,
                                                       const LongType *zTadShapeInfo,
                                                       const LongType *zTadOffsets, int numTads, LongType tadLength, int dim, LongType sizeAt,
                                                       int theShift) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  for (int e = blockIdx.x; e < theShift; e += gridDim.x) {
    int sourceIndex = dim * sizeAt + sizeAt - theShift + e;
    int targetIndex = dim * sizeAt + e;

    swapTadsKernel<T>(z + zTadOffsets[sourceIndex], z + zTadOffsets[targetIndex], zTadShapeInfo, tadLength);
  }
}

template <typename T>
static void rollFunctorFull_(NDArray *input, NDArray *output, std::vector<LongType> const &shifts,
                             std::vector<LongType> const &axes, bool inplace) {
  if (!inplace) output->assign(*input);

  for (size_t i = 0; i < axes.size(); i++) {
    int axe = axes[i];
    ResultSet listOfTensors = input->allTensorsAlongDimension({axe});
    ResultSet listOfOutTensors = output->allTensorsAlongDimension({axe});
    int fullLen = listOfTensors.size();
    int theShift = shifts[i];
    for (int k = 0; k < fullLen; k++) {
      rollFunctorLinear(output->getContext(), listOfTensors.at(k), listOfOutTensors.at(k), theShift, true);
    }
    }
  }


template <typename T>
static void rollFunctorLinear_(NDArray *input, NDArray *output, int shift, bool inplace) {
  if (!inplace) output->assign(*input);

  dim3 launchDims = getLaunchDims("roll");
  auto fullLen = input->lengthOf();
  int actualShift = shift;  // % fullLen; // shift already non-negative then
  if (actualShift < 0) {
    actualShift -= fullLen * (actualShift / fullLen - 1);
  } else
    actualShift %= fullLen;

  if (actualShift) {
    int shiftCount = fullLen / actualShift - 1;
    int remainShift = fullLen % actualShift;

    // stage 1) swap last actualShift elements with first ones.
    rollKernelLinearStage1<T><<<launchDims.y, launchDims.x, launchDims.z, *(output->getContext()->getCudaStream())>>>(
        output->specialBuffer(), output->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
        fullLen, actualShift);
    sd::DebugHelper::checkErrorCode(output->getContext()->getCudaStream(), "rollKernelLinearStage1 failed");

    // stage 2) swap swapped actualShift elements with rest remainShiftCount times.
    rollKernelLinearStage2<T><<<launchDims.y, launchDims.x, launchDims.z, *(output->getContext()->getCudaStream())>>>(
        output->specialBuffer(), output->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
        fullLen, actualShift, shiftCount);
    sd::DebugHelper::checkErrorCode(output->getContext()->getCudaStream(), "rollKernelLinearStage2 failed");
    // FIXME: no parallelism here :(
    // stage 3) swap remainer of items.
    if (remainShift && shiftCount)
      rollKernelLinearStage3<T><<<launchDims.y,launchDims.x,launchDims.z, *(output->getContext()->getCudaStream())>>>(
          output->specialBuffer(), output->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
          fullLen, actualShift, remainShift);
    sd::DebugHelper::checkErrorCode(output->getContext()->getCudaStream(), "rollKernelLinearStage3 failed");

  }
}

void rollFunctorFull(LaunchContext *context, NDArray *input, NDArray *output, std::vector<LongType> const &shifts,
                     std::vector<LongType> const &axes, bool inplace) {
  input->syncToDevice();

  BUILD_SINGLE_SELECTOR(input->dataType(), rollFunctorFull_, (input, output, shifts, axes, inplace), SD_COMMON_TYPES);

  output->tickWriteDevice();
}

void rollFunctorLinear(LaunchContext *context, NDArray *input, NDArray *output, int shift, bool inplace) {
  input->syncToDevice();

  BUILD_SINGLE_SELECTOR(input->dataType(), rollFunctorLinear_, (input, output, shift, inplace), SD_COMMON_TYPES);

  output->tickWriteDevice();
}

BUILD_SINGLE_TEMPLATE(template void rollFunctorLinear_, (NDArray * input, NDArray *output, int shift, bool inplace),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void rollFunctorFull_,
                      (NDArray * input, NDArray *output, std::vector<sd::LongType> const &shifts, std::vector<sd::LongType> const &axes,
                       bool inplace),
                      SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
