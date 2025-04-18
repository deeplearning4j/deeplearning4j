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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author raver119@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/s_t_b.h>
#if NOT_EXCLUDED(OP_space_to_batch)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpace_(NDArray& input, NDArray& output, const sd::LongType cropBottom,
                          const sd::LongType cropTop, const sd::LongType cropLeft, const sd::LongType cropRight) {
  // input [bS, H * blockSize, W * blockSize, iC]
  // output [bS, H * blockSize - cropBottom - cropTop, W * blockSize - cropLeft - cropRight, iC]

  // if (cropTop = cropBottom = cropRight = cropLeft = 0) shapes are the same
  // else:
  // oH -> [cropBottom, iH - cropTop]
  // oW -> [cropLeft,   iH - cropRight]
  // xLen > zLen

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const int rank = 4;

  const sd::LongType* xShapeInfo = input.shapeInfo();
  const sd::LongType* zShapeInfo = output.shapeInfo();

  const sd::LongType bS = xShapeInfo[1];
  const sd::LongType iH = xShapeInfo[2];
  const sd::LongType iW = xShapeInfo[3];
  const sd::LongType iC = xShapeInfo[4];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto h = start_y; h < stop_y; h += inc_y) {
        for (auto w = start_z; w < stop_z; w += inc_z) {
          for (sd::LongType c = 0; c < iC; ++c) {
            const sd::LongType xOffset = b * xShapeInfo[5] + h * xShapeInfo[6] + w * xShapeInfo[7] + c * xShapeInfo[8];
            const sd::LongType zOffset = b * zShapeInfo[5] + (h - cropBottom) * zShapeInfo[6] +
                                         (w - cropLeft) * zShapeInfo[7] + c * zShapeInfo[8];

            z[zOffset] = x[xOffset];
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, cropBottom, iH - cropTop, 1, cropLeft, iW - cropRight, 1);
}

BUILD_SINGLE_TEMPLATE(template void batchToSpace_,
                      (NDArray& input, NDArray& output, const sd::LongType cropBottom, const sd::LongType cropTop,
                       const sd::LongType cropLeft, const sd::LongType cropRight),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpace(sd::LaunchContext* context, NDArray input, NDArray& output, const sd::LongType cropBottom,
                  const sd::LongType cropTop, const sd::LongType cropLeft, const sd::LongType cropRight,
                  const sd::LongType blockSize) {
  // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
  // oH = H - cropTop  - cropBottom
  // oW = W - cropLeft - cropRight

  std::vector<sd::LongType> shape =  {blockSize, blockSize, output.sizeAt(0), input.sizeAt(1), input.sizeAt(2), input.sizeAt(3)};
  NDArray inputRearranged0 = input.reshape(
      input.ordering(),shape);
  inputRearranged0.permutei({2, 3, 0, 4, 1, 5}, false, false);

  if (input.lengthOf() == output.lengthOf())
    output.assign(&inputRearranged0);
  else {
    std::vector<sd::LongType> temp = {output.sizeAt(0), input.sizeAt(1) * blockSize, input.sizeAt(2) * blockSize, input.sizeAt(3)};
    NDArray inputRearranged1 = inputRearranged0.reshape(
        input.ordering(),
        temp);
    BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpace_,
                          (inputRearranged1, output, cropBottom, cropTop, cropLeft, cropRight), SD_COMMON_TYPES);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpaceND_(NDArray& input, NDArray& crop, NDArray& output,
                            const LongType numOfSpatialDims) {
  // input [bS, H * blockShape[0], W * blockShape[1], iC]
  // output [bS, H * blockShape[0] - cropBottom - cropTop, W * blockShape[1] - cropLeft - cropRight, iC]

  // if (cropTop = cropBottom = cropRight = cropLeft = 0) shapes are the same
  // else:
  // oH -> [cropBottom, iH - cropTop]
  // oW -> [cropLeft,   iH - cropRight]
  // xLen >= zLen

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const sd::LongType rank = input.rankOf();
  const sd::LongType zLen = output.lengthOf();

  // loop through input array
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, rank, shape::shapeOf(output.shapeInfo()), zCoords);

      memcpy(xCoords, zCoords, rank * sizeof(sd::LongType));

      // evaluate spatial coordinates for x
      for (sd::LongType j = 1; j <= numOfSpatialDims; ++j)
        xCoords[j] += crop.e<sd::LongType>(j - 1, 0);  // add crop left

      sd::LongType zOffset, xOffset;
      COORDS2INDEX(rank, shape::stride(output.shapeInfo()), zCoords, zOffset);
      COORDS2INDEX(rank, shape::stride(input.shapeInfo()), xCoords, xOffset);

      z[zOffset] = x[xOffset];
    }
  };

  samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void batchToSpaceND_,
                      (NDArray& input, NDArray& crop, NDArray& output, const sd::LongType numOfSpatialDims),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpaceND(sd::LaunchContext* context, NDArray& input, NDArray& blockShape, NDArray& crop,
                    NDArray& output){
  // 4D example, numOfSpatialDims = 2 - two spatial dimensions
  // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  -
  // cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

  const sd::LongType rank = input.rankOf();
  const sd::LongType numOfSpatialDims = blockShape.sizeAt(0);

  //*** construct reshaping std::vector for first reshape of input array ***//

  std::vector<sd::LongType> temp(numOfSpatialDims + rank);

  sd::LongType i;
  for (i = 0; i < numOfSpatialDims; ++i) temp[i] = blockShape.e<sd::LongType>(i);
  temp[i++] = output.sizeAt(0);
  for (sd::LongType j = 1; j < rank; ++i, ++j) temp[i] = input.sizeAt(j);

  NDArray inputRearranged0 = input.reshape(input.ordering(), temp);

  //*** construct permuting std::vector for permutation of input array ***//

  temp[0] = numOfSpatialDims;

  for (i = 1; i <= numOfSpatialDims; ++i) {
    temp[2 * i - 1] = numOfSpatialDims + i;
    temp[2 * i] = i - 1;
  }
  for (i = 2 * numOfSpatialDims + 1; i < static_cast<sd::LongType>(temp.size()); ++i) temp[i] = i;

  inputRearranged0.permutei(temp, false, false);

  if (input.lengthOf() == output.lengthOf()) {
    output.assign(&inputRearranged0);
  } else {
    //*** construct reshaping std::vector for second reshape of input array ***//

    temp.resize(rank);

    temp[0] = output.sizeAt(0);

    for (i = 1; i < rank; ++i)
      temp[i] = (i <= numOfSpatialDims) ? input.sizeAt(i) * blockShape.e<sd::LongType>(i - 1) : input.sizeAt(i);

    NDArray inputRearranged1 = inputRearranged0.reshape(input.ordering(), temp);

    BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpaceND_, (inputRearranged1, crop, output, numOfSpatialDims),
                          SD_COMMON_TYPES);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatch_(NDArray& input, NDArray& output, const sd::LongType padBottom,
                          const sd::LongType padTop, const sd::LongType padLeft, const sd::LongType padRight) {
  // input [bS, H * blockSize - padBottom - padTop, W * blockSize - padLeft - padRight, iC]
  // output [bS, H * blockSize, W * blockSize, iC]

  // if (padTop = padBottom = padRight = padLeft = 0) shapes are the same
  // else:
  // iH -> [padBottom, oH - padTop]
  // iW -> [padLeft,   oW - padRight]
  // zLen > xLen

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const int rank = 4;

  const sd::LongType* xShapeInfo = input.shapeInfo();
  const sd::LongType* zShapeInfo = output.shapeInfo();

  const sd::LongType bS = zShapeInfo[1];
  const sd::LongType oH = zShapeInfo[2];
  const sd::LongType oW = zShapeInfo[3];
  const sd::LongType iC = zShapeInfo[4];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto h = start_y; h < stop_y; h += inc_y) {
        for (sd::LongType w = 0; w < oW; ++w) {
          for (sd::LongType c = 0; c < iC; ++c) {
            const sd::LongType zOffset = b * zShapeInfo[5] + h * zShapeInfo[6] + w * zShapeInfo[7] + c * zShapeInfo[8];

            if (h >= padBottom && h < oH - padTop && w >= padLeft && w < oW - padRight) {
              const sd::LongType xOffset = b * xShapeInfo[5] + (h - padBottom) * xShapeInfo[6] +
                                           (w - padLeft) * xShapeInfo[7] + c * xShapeInfo[8];
              z[zOffset] = x[xOffset];
            } else
              z[zOffset] = 0.f;
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, 0, oH, 1);
}

BUILD_SINGLE_TEMPLATE(template void spaceToBatch_,
                      (NDArray& input, NDArray& output, const sd::LongType padBottom, const sd::LongType padTop,
                       const sd::LongType padLeft, const sd::LongType padRight),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatch(sd::LaunchContext* context, NDArray& input, NDArray& output, const sd::LongType padBottom,
                  const sd::LongType padTop, const sd::LongType padLeft, const sd::LongType padRight,
                  const sd::LongType blockSize) {
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW +
  // padLeft + padRight)/blockSize, iC]

  std::vector<sd::LongType> shape1 = {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), output.sizeAt(3)};
  NDArray outputRearranged0 = output.reshape(
      output.ordering(), shape1,
      false);
  outputRearranged0.permutei({2, 3, 0, 4, 1, 5}, false, false);

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(&input);
  } else {
    std::vector<sd::LongType> shape2 = {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, output.sizeAt(3)};
    NDArray outputRearranged1 = outputRearranged0.reshape(
        output.ordering(),
        shape2, false);
    BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatch_,
                          (input, outputRearranged1, padBottom, padTop, padLeft, padRight), SD_COMMON_TYPES);

    if (output.buffer() != outputRearranged1.buffer()) outputRearranged0.assign(&outputRearranged1);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatchND_(NDArray& input, NDArray& padding, NDArray& output,
                            const LongType numOfSpatialDims) {
  // 4D example
  // input [bS, H * blockShape[0] - padBottom - padTop, W * blockShape[1] - padLeft - padRight, iC]
  // output [bS, H * blockShape[0], W * blockShape[1], iC]

  // if (padTop = padBottom = padRight = padLeft = 0) shapes are the same
  // else:
  // iH -> [padBottom, oH - padTop]
  // iW -> [padLeft,   oW - padRight]
  // zLen > xLen

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const int rank = input.rankOf();
  const sd::LongType zLen = output.lengthOf();

  // loop through output array
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK];

    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, rank, shape::shapeOf(output.shapeInfo()), zCoords);

      sd::LongType zOffset;
      COORDS2INDEX(rank, shape::stride(output.shapeInfo()), zCoords, zOffset);

      memcpy(xCoords, zCoords, rank * sizeof(LongType));

      bool within = true;

      for (sd::LongType j = 1; j <= numOfSpatialDims; ++j) {
        const auto padLeft = padding.e<sd::LongType>(j - 1, 0);
        const auto padRight = padding.e<sd::LongType>(j - 1, 1);

        within &= zCoords[j] >= padLeft && zCoords[j] < output.sizeAt(j) - padRight;

        if (!within) break;

        xCoords[j] = zCoords[j] - padLeft;  // get coordinates for x
      }

      if (within) {
        sd::LongType xOffset;
        COORDS2INDEX(rank, shape::stride(input.shapeInfo()), xCoords, xOffset);
        z[zOffset] = x[xOffset];
      } else {
        z[zOffset] = 0.f;
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void spaceToBatchND_,
                      (NDArray& input, NDArray& padding, NDArray& output,
                       const sd::LongType numOfSpatialDims),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatchND(sd::LaunchContext* context, NDArray& input, NDArray& blockShape, NDArray& padding,
                    NDArray& output) {
  // 4D example with two spatial dimensions
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom +
  // padTop)/blockShape[0], (iW + padLeft + padRight)/blockShape[1], iC]

  const sd::LongType rank = input.rankOf();

  const sd::LongType numOfSpatialDims = blockShape.sizeAt(0);

  //*** construct reshaping std::vector for first reshape of output array ***//
  std::vector<sd::LongType> temp(numOfSpatialDims + rank);

  int i;
  for (i = 0; i < numOfSpatialDims; ++i) temp[i] = blockShape.e<sd::LongType>(i);
  temp[i++] = input.sizeAt(0);
  for (int j = 1; j < rank; ++i, ++j) temp[i] = output.sizeAt(j);

  NDArray outputRearranged0 = output.reshape(output.ordering(), temp, false);

  //*** construct permuting std::vector for permutation of output array ***//

  temp[0] = numOfSpatialDims;

  for (i = 1; i <= numOfSpatialDims; ++i) {
    temp[2 * i - 1] = numOfSpatialDims + i;
    temp[2 * i] = i - 1;
  }
  for (i = 2 * numOfSpatialDims + 1; i < static_cast<int>(temp.size()); ++i) temp[i] = i;

  outputRearranged0.permutei(temp, false, false);

  // ****** //

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(&input);
  } else {
    //*** construct reshaping std::vector for second reshape of output array ***//
    temp.resize(rank);

    temp[0] = input.sizeAt(0);

    for (i = 1; i < rank; ++i)
      temp[i] = (i <= numOfSpatialDims) ? output.sizeAt(i) * blockShape.e<sd::LongType>(i - 1) : output.sizeAt(i);

    NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), temp, false);

    BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchND_, (input, padding, outputRearranged1, numOfSpatialDims),
                          SD_COMMON_TYPES);

    if (output.buffer() != outputRearranged1.buffer()) outputRearranged0.assign(&outputRearranged1);
  }
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif