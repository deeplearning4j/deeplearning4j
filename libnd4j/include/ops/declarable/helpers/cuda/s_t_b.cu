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
// Created by raver119 on 19.01.18.
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/s_t_b.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void batchToSpaceCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const LongType cropBottom,
                                       const LongType cropLeft) {
  // input [bS, H * blockSize, W * blockSize, iC]
  // output [bS, H * blockSize - cropBottom - cropTop, W * blockSize - cropLeft - cropRight, iC]

  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ LongType rank, *sharedMem;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
  }
  __syncthreads();

  LongType* coords = sharedMem + threadIdx.x * rank;

  const LongType i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= zLen) return;

  INDEX2COORDS(i, rank, zShapeInfo, coords);

  LongType zOffset;
  COORDS2INDEX(rank, shape::shapeOf(zShapeInfo), coords, zOffset);

  coords[1] += cropBottom;
  coords[2] += cropLeft;

  LongType xOffset;
  COORDS2INDEX(rank, shape::shapeOf(xShapeInfo), coords, xOffset);

  z[zOffset] = x[xOffset];
}
///////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpaceCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                     void* vz, const LongType* zShapeInfo, const LongType cropBottom,
                                     const LongType cropLeft) {
  batchToSpaceCuda<T>
      <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, cropBottom, cropLeft);
}
BUILD_SINGLE_TEMPLATE(template void batchToSpaceCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                       const sd::LongType* zShapeInfo, const sd::LongType cropBottom, const sd::LongType cropLeft),
                      SD_COMMON_TYPES);

///////////////////////////////////////////////////////////////////
void batchToSpace(sd::LaunchContext* context, NDArray input, NDArray& output,
                  const sd::LongType cropBottom, const sd::LongType cropTop, const sd::LongType cropLeft,
                  const sd::LongType cropRight, const sd::LongType blockSize) {
  // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
  // oH = H - cropTop  - cropBottom
  // oW = W - cropLeft - cropRight

  std::vector<sd::LongType> rearrShape =  {blockSize, blockSize, output.sizeAt(0), input.sizeAt(1), input.sizeAt(2), input.sizeAt(3)};
  NDArray inputRearranged0 = input.reshape(
      input.ordering(), rearrShape,false);
  inputRearranged0.permutei({2, 3, 0, 4, 1, 5}, false, false);

  if (input.lengthOf() == output.lengthOf()) {
    output.assign(inputRearranged0);
  } else {
    std::vector<sd::LongType> outputShape =  {output.sizeAt(0), input.sizeAt(1) * blockSize, input.sizeAt(2) * blockSize, input.sizeAt(3)};
    NDArray inputRearranged1 = inputRearranged0.reshape(
        input.ordering(),
        outputShape);

    const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(LongType) * output.rankOf() + 128;

    PointersManager manager(context, "batchToSpace");

    NDArray::prepareSpecialUse({&output}, {&inputRearranged1});
    BUILD_SINGLE_SELECTOR(
        input.dataType(), batchToSpaceCudaLauncher,
        (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), inputRearranged1.specialBuffer(),
         inputRearranged1.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), cropBottom, cropLeft),
        SD_COMMON_TYPES);
    NDArray::registerSpecialUse({&output}, {&inputRearranged1});

    manager.synchronize();
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL static void batchToSpaceNDCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                         const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                         const LongType numOfSpatialDims) {
  // 4D example, numOfSpatialDims = 2
  // input [bS, H * blockShape[0], W * blockShape[1], iC]
  // output [bS, H * blockShape[0] - cropBottom - cropTop, W * blockShape[1] - cropLeft - cropRight, iC]

  // if (cropTop = cropBottom = cropRight = cropLeft = 0) shapes are the same
  // else:
  // oH -> [cropBottom, iH - cropTop]
  // oW -> [cropLeft,   iH - cropRight]
  // xLen >= zLen

  const auto x = reinterpret_cast<const X*>(vx);
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<X*>(vz);

  __shared__ LongType rank, *sharedMem;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
  }

  __syncthreads();

  LongType* coords = sharedMem + threadIdx.x * rank;

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {
    INDEX2COORDS(i, rank, zShapeInfo, coords);

    LongType zOffset;
    COORDS2INDEX(rank, shape::shapeOf(zShapeInfo), coords, zOffset);

    // evaluate spatial coordinates for x
    for (LongType j = 1; j <= numOfSpatialDims; ++j) {
      const LongType yOffset = (j - 1) * yShapeInfo[3];  // yRank = 2, calculate offset manually
      coords[j] += y[yOffset];                       // add crop left
    }

    LongType xOffset;
    COORDS2INDEX(rank, shape::shapeOf(xShapeInfo), coords, xOffset);

    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void batchToSpaceNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                       const void* vy, const LongType* yShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const LongType numOfSpatialDims) {
  batchToSpaceNDCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                                   zShapeInfo, numOfSpatialDims);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "batchToSpaceNDCuda failed");

}
BUILD_DOUBLE_TEMPLATE(template void batchToSpaceNDCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, const void* vy,
                       const sd::LongType* yShapeInfo, void* vz, const sd::LongType* zShapeInfo,
                       const sd::LongType numOfSpatialDims),
                      SD_COMMON_TYPES, SD_INTEGER_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpaceND(sd::LaunchContext* context, NDArray& input, NDArray& blockShape, NDArray& crop,
                    NDArray& output) {
  // 4D example, numOfSpatialDims = 2 - two spatial dimensions
  // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  -
  // cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

  const LongType rank = input.rankOf();
  const LongType numOfSpatialDims = blockShape.sizeAt(0);

  //*** construct reshaping std::vector for first reshape of input array ***//

  std::vector<LongType> temp(numOfSpatialDims + rank);

  int i;
  for (i = 0; i < numOfSpatialDims; ++i) temp[i] = blockShape.e<LongType>(i);
  temp[i++] = output.sizeAt(0);
  for (int j = 1; j < rank; ++i, ++j) temp[i] = input.sizeAt(j);

  NDArray inputRearranged0 = input.reshape(input.ordering(), temp);

  //*** construct permuting std::vector for permutation of input array ***//

  temp[0] = numOfSpatialDims;

  for (i = 1; i <= numOfSpatialDims; ++i) {
    temp[2 * i - 1] = numOfSpatialDims + i;
    temp[2 * i] = i - 1;
  }
  for (i = 2 * numOfSpatialDims + 1; i < temp.size(); ++i) temp[i] = i;

  inputRearranged0.permutei(temp, 0, false);

  if (input.lengthOf() == output.lengthOf()) {
    output.assign(inputRearranged0);
  } else {
    //*** construct reshaping std::vector for second reshape of input array ***//

    temp.resize(rank);

    temp[0] = output.sizeAt(0);

    for (i = 1; i < rank; ++i)
      temp[i] = (i <= numOfSpatialDims) ? input.sizeAt(i) * blockShape.e<LongType>(i - 1) : input.sizeAt(i);

    NDArray inputRearranged1 = inputRearranged0.reshape(input.ordering(), temp);

    dim3 launchDims = batchToSpaceNdLaunch(output.lengthOf(),output.rankOf());

    PointersManager manager(context, "batchToSpaceND");

    NDArray::prepareSpecialUse({&output}, {&inputRearranged1, &crop});
    BUILD_DOUBLE_SELECTOR(
        input.dataType(), crop.dataType(), batchToSpaceNDCudaLauncher,
        (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), inputRearranged1.specialBuffer(),
         inputRearranged1.specialShapeInfo(), crop.specialBuffer(), crop.specialShapeInfo(), output.specialBuffer(),
         output.specialShapeInfo(), numOfSpatialDims),
        SD_COMMON_TYPES, SD_INTEGER_TYPES);
    NDArray::registerSpecialUse({&output}, {&inputRearranged1, &crop});

    manager.synchronize();
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void spaceToBatchCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const LongType padBottom,
                                       const LongType padTop, const LongType padLeft,
                                       const LongType padRight) {
  // input [bS, H * blockSize - padBottom - padTop, W * blockSize - padLeft - padRight, iC]
  // output [bs, H * blockSize, W * blockSize, iC]

  // if (padTop = padBottom = padRight = padLeft = 0) shapes are the same
  // else:
  // iH -> [padBottom, oH - padTop]
  // iW -> [padLeft,   oW - padRight]
  // zLen > xLen

  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ LongType rank, *sharedMem;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
  }
  __syncthreads();

  LongType* coords = sharedMem + threadIdx.x * rank;

  const LongType i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= zLen) return;

  INDEX2COORDS(i, rank, zShapeInfo, coords);

  LongType zOffset;
  COORDS2INDEX(rank, shape::shapeOf(zShapeInfo), coords, zOffset);

  if (coords[1] >= padBottom && coords[1] < zShapeInfo[2] - padTop && coords[2] >= padLeft &&
      coords[2] < zShapeInfo[3] - padRight) {
    coords[1] -= padBottom;
    coords[2] -= padLeft;

    LongType xOffset;
    COORDS2INDEX(rank, shape::shapeOf(xShapeInfo), coords, xOffset);

    z[zOffset] = x[xOffset];
  } else
    z[zOffset] = 0.f;
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatchCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                     void* vz, const LongType* zShapeInfo, const LongType padBottom,
                                     const LongType padTop, const LongType padLeft, const LongType padRight) {
  spaceToBatchCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, padBottom,
                                                                              padTop, padLeft, padRight);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "spaceToBatchCudaLauncher failed");

}
BUILD_SINGLE_TEMPLATE(template void spaceToBatchCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                       const sd::LongType* zShapeInfo, const sd::LongType padBottom, const sd::LongType padTop,
                       const sd::LongType padLeft, const sd::LongType padRight),
                      SD_COMMON_TYPES);

///////////////////////////////////////////////////////////////////
void spaceToBatch(LaunchContext* context, NDArray& input, NDArray& output, const LongType padBottom,
                  const LongType padTop, const LongType padLeft, const LongType padRight,
                  const LongType blockSize) {
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW +
  // padLeft + padRight)/blockSize, iC]

  std::vector<sd::LongType> outputShape = {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), input.sizeAt(3)};
  NDArray outputRearranged0 = output.reshape(
      output.ordering(), outputShape,
      false);
  outputRearranged0.permutei({2, 3, 0, 4, 1, 5}, false, false);

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(input);
  } else {
    std::vector<sd::LongType> outReArrShape =  {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, input.sizeAt(3)};
    NDArray outputRearranged1 = outputRearranged0.reshape(
        output.ordering(),
        outReArrShape, false);


    dim3 launchDims = spaceToBatchLaunch(output.lengthOf(),output.rankOf());

    PointersManager manager(context, "spaceToBatch");

    NDArray::prepareSpecialUse({&outputRearranged1}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchCudaLauncher,
                          (launchDims.y,launchDims.x,launchDims.z, context->getCudaStream(), input.specialBuffer(),
                           input.specialShapeInfo(), outputRearranged1.specialBuffer(),
                           outputRearranged1.specialShapeInfo(), padBottom, padTop, padLeft, padRight),
                          SD_COMMON_TYPES);
    NDArray::registerSpecialUse({&outputRearranged1}, {&input});

    manager.synchronize();

    if (output.specialBuffer() != outputRearranged1.specialBuffer()) outputRearranged0.assign(outputRearranged1);
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL static void spaceToBatchNDCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                         const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                         const LongType numOfSpatialDims) {
  // x - input, y - padding, z - output

  // 4D example
  // input [bS, H * blockShape[0] - padBottom - padTop, W * blockShape[1] - padLeft - padRight, iC]
  // output [bS, H * blockShape[0], W * blockShape[1], iC]

  // if (padTop = padBottom = padRight = padLeft = 0) shapes are the same
  // else:
  // iH -> [padBottom, oH - padTop]
  // iW -> [padLeft,   oW - padRight]
  // zLen > xLen

  const auto x = reinterpret_cast<const X*>(vx);
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<X*>(vz);

  __shared__ LongType rank, *sharedMem;  // xRank = zRank, yRank = 2;
  __shared__ LongType zLen, totalThreads;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }

  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;

  for (LongType i = blockDim.x * blockIdx.x + threadIdx.x; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, rank, zShapeInfo, coords);

    LongType zOffset;
    COORDS2INDEX(rank, shape::shapeOf(zShapeInfo), coords, zOffset);

    bool within = true;

    for (LongType j = 1; j <= numOfSpatialDims; ++j) {
      // yRank = 2, calculate offset manually
      const auto yOffset = (j - 1) * yShapeInfo[3];
      const auto padLeft = y[yOffset];
      const auto padRight = y[yOffset + yShapeInfo[4]];

      within &=
          (coords[j] >= padLeft && coords[j] < shape::shapeOf(const_cast<LongType*>(zShapeInfo))[j] - padRight);

      if (!within) break;

      coords[j] -= padLeft;  // get coordinates for x
    }

    LongType xOffset;
    COORDS2INDEX(rank, shape::shapeOf(xShapeInfo), coords, xOffset);

    if (within)
      z[zOffset] = x[xOffset];
    else
      z[zOffset] = 0.f;
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void spaceToBatchNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                       const void* vy, const LongType* yShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const LongType numOfSpatialDims) {
  spaceToBatchNDCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                                   zShapeInfo, numOfSpatialDims);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "spaceToBatchNDCuda failed");

}
BUILD_DOUBLE_TEMPLATE(template void spaceToBatchNDCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, const void* vy,
                       const sd::LongType* yShapeInfo, void* vz, const sd::LongType* zShapeInfo,
                       const sd::LongType numOfSpatialDims),
                      SD_COMMON_TYPES, SD_INTEGER_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatchND(LaunchContext* context, NDArray& input, NDArray& blockShape, NDArray& padding,
                    NDArray& output) {
  // 4D example with two spatial dimensions
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom +
  // padTop)/blockShape[0], (iW + padLeft + padRight)/blockShape[1], iC]

  const LongType rank = input.rankOf();

  const LongType numOfSpatialDims = blockShape.sizeAt(0);

  //*** construct reshaping std::vector for first reshape of output array ***//
  std::vector<LongType> temp(numOfSpatialDims + rank);

  int i;
  for (i = 0; i < numOfSpatialDims; ++i) temp[i] = blockShape.e<LongType>(i);
  temp[i++] = input.sizeAt(0);
  for (int j = 1; j < rank; ++i, ++j) temp[i] = output.sizeAt(j);

  NDArray outputRearranged0 = output.reshape(output.ordering(), temp, false);

  //*** construct permuting std::vector for permutation of output array ***//

  temp[0] = numOfSpatialDims;

  for (i = 1; i <= numOfSpatialDims; ++i) {
    temp[2 * i - 1] = numOfSpatialDims + i;
    temp[2 * i] = i - 1;
  }
  for (i = 2 * numOfSpatialDims + 1; i < temp.size(); ++i) temp[i] = i;

  outputRearranged0.permutei(temp, false, false);

  // ****** //

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(input);
  } else {
    //*** construct reshaping std::vector for second reshape of output array ***//
    temp.resize(rank);

    temp[0] = input.sizeAt(0);

    for (i = 1; i < rank; ++i)
      temp[i] = (i <= numOfSpatialDims) ? output.sizeAt(i) * blockShape.e<LongType>(i - 1) : output.sizeAt(i);

    NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), temp, false);

    dim3 launchDims = spaceToBatchNdLaunch(output.lengthOf(),output.rankOf());
    PointersManager manager(context, "spaceToBatchND");

    NDArray::prepareSpecialUse({&outputRearranged1}, {&input, &padding});
    BUILD_DOUBLE_SELECTOR(input.dataType(), padding.dataType(), spaceToBatchNDCudaLauncher,
                          (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), input.specialBuffer(),
                           input.specialShapeInfo(), padding.specialBuffer(), padding.specialShapeInfo(),
                           outputRearranged1.specialBuffer(), outputRearranged1.specialShapeInfo(), numOfSpatialDims),
                          SD_COMMON_TYPES, SD_INTEGER_TYPES);
    NDArray::registerSpecialUse({&outputRearranged1}, {&input, &padding});

    manager.synchronize();

    if (output.specialBuffer() != outputRearranged1.specialBuffer()) outputRearranged0.assign(outputRearranged1);
  }
}



}  // namespace helpers
}  // namespace ops
}  // namespace sd
