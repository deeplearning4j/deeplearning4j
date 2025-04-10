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

  // Cache shape information for x buffer
  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  // Cache shape information for z buffer
  __shared__ sd::LongType zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    // Cache x shape information
    xRank = shape::rank(xShapeInfo);
    xShapePtr = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    // Cache z shape information
    zRank = shape::rank(zShapeInfo);
    zShapePtr = shape::shapeOf(zShapeInfo);
    zStridePtr = shape::stride(zShapeInfo);
  }
  __syncthreads();

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  LongType xCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xOffsetA;
  LongType xOffsetB;
  LongType zOffsetA;
  LongType zOffsetB;

  for (LongType i = tid; i < actualShift; i += blockDim.x * gridDim.x) {
    int sourceIndex = fullLength - actualShift + i;

    INDEX2COORDS(i, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffsetA);
    INDEX2COORDS(sourceIndex, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffsetB);

    INDEX2COORDS(i, zRank, zShapePtr, zCoords);
    COORDS2INDEX(zRank, zStridePtr, zCoords, zOffsetA);
    INDEX2COORDS(sourceIndex, zRank, zShapePtr, zCoords);
    COORDS2INDEX(zRank, zStridePtr, zCoords, zOffsetB);

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
static void SD_KERNEL rollKernelLinearStage2(const void *vx, const LongType *xShapeInfo, void *vz,
                                             const LongType *zShapeInfo, LongType fullLength, int actualShift,
                                             int shiftCount) {
 auto x = reinterpret_cast<const T *>(vx);
 auto z = reinterpret_cast<T *>(vz);

 // Cache shape information for x buffer
 __shared__ sd::LongType xRank;
 __shared__ const sd::LongType* xShapePtr;
 __shared__ const sd::LongType* xStridePtr;

 // Cache shape information for z buffer
 __shared__ sd::LongType zRank;
 __shared__ const sd::LongType* zShapePtr;
 __shared__ const sd::LongType* zStridePtr;

 if (threadIdx.x == 0) {
   // Cache x shape information
   xRank = shape::rank(xShapeInfo);
   xShapePtr = shape::shapeOf(xShapeInfo);
   xStridePtr = shape::stride(xShapeInfo);

   // Cache z shape information
   zRank = shape::rank(zShapeInfo);
   zShapePtr = shape::shapeOf(zShapeInfo);
   zStridePtr = shape::stride(zShapeInfo);
 }
 __syncthreads();

 auto tid = threadIdx.x + blockIdx.x * blockDim.x;

 LongType xCoords[SD_MAX_RANK];
 LongType zCoords[SD_MAX_RANK];
 LongType xOffsetA;
 LongType xOffsetB;
 LongType zOffsetA;
 LongType zOffsetB;

 for (int count = 1; count < shiftCount; ++count) {
   for (int i = tid; i < actualShift; i += blockDim.x * gridDim.x) {
     int destinationIndex = fullLength - (count + 1) * actualShift + i;
     int sourceIndex = fullLength - count * actualShift + i;

     INDEX2COORDS(destinationIndex, xRank, xShapePtr, xCoords);
     COORDS2INDEX(xRank, xStridePtr, xCoords, xOffsetA);
     INDEX2COORDS(sourceIndex, xRank, xShapePtr, xCoords);
     COORDS2INDEX(xRank, xStridePtr, xCoords, xOffsetB);

     INDEX2COORDS(destinationIndex, zRank, zShapePtr, zCoords);
     COORDS2INDEX(zRank, zStridePtr, zCoords, zOffsetA);
     INDEX2COORDS(sourceIndex, zRank, zShapePtr, zCoords);
     COORDS2INDEX(zRank, zStridePtr, zCoords, zOffsetB);

     auto eA = x[xOffsetB];
     auto eB = x[xOffsetA];

     z[zOffsetA] = eA;
     z[zOffsetB] = eB;
   }

   __syncthreads();
 }
}
template <typename T>
static void SD_KERNEL rollKernelLinearStage3(const void *vx, const LongType *xShapeInfo, void *vz,
                                             const LongType *zShapeInfo, LongType fullLength, int actualShift,
                                             int remainShift) {
 auto x = reinterpret_cast<const T *>(vx);
 auto z = reinterpret_cast<T *>(vz);

 // Cache shape information for x buffer
 __shared__ sd::LongType xRank;
 __shared__ const sd::LongType* xShapePtr;
 __shared__ const sd::LongType* xStridePtr;

 // Cache shape information for z buffer
 __shared__ sd::LongType zRank;
 __shared__ const sd::LongType* zShapePtr;
 __shared__ const sd::LongType* zStridePtr;

 if (threadIdx.x == 0) {
   // Cache x shape information
   xRank = shape::rank(xShapeInfo);
   xShapePtr = shape::shapeOf(xShapeInfo);
   xStridePtr = shape::stride(xShapeInfo);

   // Cache z shape information
   zRank = shape::rank(zShapeInfo);
   zShapePtr = shape::shapeOf(zShapeInfo);
   zStridePtr = shape::stride(zShapeInfo);
 }
 __syncthreads();

 auto tid = threadIdx.x + blockIdx.x * blockDim.x;

 for (int i = tid; i < actualShift; i += blockDim.x * gridDim.x) {
   int remainIdx = i + actualShift;
   int sourceIndex = remainIdx + remainShift;

   LongType xCoordsA[SD_MAX_RANK];
   LongType xCoordsB[SD_MAX_RANK];
   LongType zCoordsA[SD_MAX_RANK];
   LongType zCoordsB[SD_MAX_RANK];
   LongType xOffsetA;
   LongType xOffsetB;
   LongType zOffsetA;
   LongType zOffsetB;

   INDEX2COORDS(remainIdx, xRank, xShapePtr, xCoordsA);
   COORDS2INDEX(xRank, xStridePtr, xCoordsA, xOffsetA);
   INDEX2COORDS(sourceIndex, xRank, xShapePtr, xCoordsB);
   COORDS2INDEX(xRank, xStridePtr, xCoordsB, xOffsetB);

   INDEX2COORDS(remainIdx, zRank, zShapePtr, zCoordsA);
   COORDS2INDEX(zRank, zStridePtr, zCoordsA, zOffsetA);
   INDEX2COORDS(sourceIndex, zRank, zShapePtr, zCoordsB);
   COORDS2INDEX(zRank, zStridePtr, zCoordsB, zOffsetB);

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

 // Cache shape information for z buffer
 __shared__ sd::LongType zRank;
 __shared__ const sd::LongType* zShapePtr;
 __shared__ const sd::LongType* zStridePtr;

 if (threadIdx.x == 0) {
   // Cache z shape information
   zRank = shape::rank(zShapeInfo);
   zShapePtr = shape::shapeOf(zShapeInfo);
   zStridePtr = shape::stride(zShapeInfo);
 }
 __syncthreads();

 auto tid = threadIdx.x + blockIdx.x * blockDim.x;

 for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
   LongType zCoords[SD_MAX_RANK];
   LongType zOffset;

   INDEX2COORDS(e, zRank, zShapePtr, zCoords);
   COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);

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
 if (!inplace) output->assign(input);

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
 if (!inplace) output->assign(input);

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
