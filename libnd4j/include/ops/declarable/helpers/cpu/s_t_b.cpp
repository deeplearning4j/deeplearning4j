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
static void batchToSpace_(const NDArray& input, NDArray& output, const sd::Unsigned cropBottom,
                          const sd::Unsigned cropTop, const sd::Unsigned cropLeft, const sd::Unsigned cropRight) {
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

  const sd::Unsigned bS = xShapeInfo[1];
  const sd::Unsigned iH = xShapeInfo[2];
  const sd::Unsigned iW = xShapeInfo[3];
  const sd::Unsigned iC = xShapeInfo[4];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto h = start_y; h < stop_y; h += inc_y) {
        for (auto w = start_z; w < stop_z; w += inc_z) {
          for (sd::Unsigned c = 0; c < iC; ++c) {
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
                      (const NDArray& input, NDArray& output, const sd::Unsigned cropBottom, const sd::Unsigned cropTop,
                       const sd::Unsigned cropLeft, const sd::Unsigned cropRight),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpace(sd::LaunchContext* context, const NDArray& input, NDArray& output, const sd::Unsigned cropBottom,
                  const sd::Unsigned cropTop, const sd::Unsigned cropLeft, const sd::Unsigned cropRight,
                  const sd::Unsigned blockSize) {
  // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
  // oH = H - cropTop  - cropBottom
  // oW = W - cropLeft - cropRight

  NDArray inputRearranged0 = input.reshape(
      input.ordering(), {blockSize, blockSize, output.sizeAt(0), input.sizeAt(1), input.sizeAt(2), input.sizeAt(3)});
  inputRearranged0.permutei({2, 3, 0, 4, 1, 5});

  if (input.lengthOf() == output.lengthOf())
    output.assign(inputRearranged0);
  else {
    NDArray inputRearranged1 = inputRearranged0.reshape(
        input.ordering(),
        {output.sizeAt(0), input.sizeAt(1) * blockSize, input.sizeAt(2) * blockSize, input.sizeAt(3)});
    BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpace_,
                          (inputRearranged1, output, cropBottom, cropTop, cropLeft, cropRight), SD_COMMON_TYPES);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpaceND_(const NDArray& input, const NDArray& crop, NDArray& output,
                            const sd::Unsigned numOfSpatialDims) {
  // input [bS, H * blockShape[0], W * blockShape[1], iC]
  // output [bS, H * blockShape[0] - cropBottom - cropTop, W * blockShape[1] - cropLeft - cropRight, iC]

  // if (cropTop = cropBottom = cropRight = cropLeft = 0) shapes are the same
  // else:
  // oH -> [cropBottom, iH - cropTop]
  // oW -> [cropLeft,   iH - cropRight]
  // xLen >= zLen

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const int rank = input.rankOf();
  const sd::LongType zLen = output.lengthOf();

  // loop through input array
  auto func = PRAGMA_THREADS_FOR {
    int zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, output.shapeInfo(), zCoords);

      memcpy(xCoords, zCoords, rank * sizeof(int));

      // evaluate spatial coordinates for x
      for (sd::Unsigned j = 1; j <= numOfSpatialDims; ++j)
        xCoords[j] += crop.e<sd::Unsigned>(j - 1, 0);  // add crop left

      const auto zOffset = shape::getOffset(output.shapeInfo(), zCoords);
      const auto xOffset = shape::getOffset(input.shapeInfo(), xCoords);

      z[zOffset] = x[xOffset];
    }
  };

  samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void batchToSpaceND_,
                      (const NDArray& input, const NDArray& crop, NDArray& output, const sd::Unsigned numOfSpatialDims),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpaceND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape, const NDArray& crop,
                    NDArray& output) {
  // 4D example, numOfSpatialDims = 2 - two spatial dimensions
  // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  -
  // cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

  const sd::Unsigned rank = input.rankOf();
  const sd::Unsigned numOfSpatialDims = blockShape.sizeAt(0);

  //*** construct reshaping std::vector for first reshape of input array ***//

  std::vector<sd::LongType> temp(numOfSpatialDims + rank);

  sd::Unsigned i;
  for (i = 0; i < numOfSpatialDims; ++i) temp[i] = blockShape.e<sd::LongType>(i);
  temp[i++] = output.sizeAt(0);
  for (sd::Unsigned j = 1; j < rank; ++i, ++j) temp[i] = input.sizeAt(j);

  NDArray inputRearranged0 = input.reshape(input.ordering(), temp);

  //*** construct permuting std::vector for permutation of input array ***//

  temp[0] = numOfSpatialDims;

  for (i = 1; i <= numOfSpatialDims; ++i) {
    temp[2 * i - 1] = numOfSpatialDims + i;
    temp[2 * i] = i - 1;
  }
  for (i = 2 * numOfSpatialDims + 1; i < static_cast<sd::Unsigned>(temp.size()); ++i) temp[i] = i;

  inputRearranged0.permutei(temp);

  if (input.lengthOf() == output.lengthOf()) {
    output.assign(inputRearranged0);
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
static void spaceToBatch_(const NDArray& input, NDArray& output, const sd::Unsigned padBottom,
                          const sd::Unsigned padTop, const sd::Unsigned padLeft, const sd::Unsigned padRight) {
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

  const sd::Unsigned bS = zShapeInfo[1];
  const sd::Unsigned oH = zShapeInfo[2];
  const sd::Unsigned oW = zShapeInfo[3];
  const sd::Unsigned iC = zShapeInfo[4];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto h = start_y; h < stop_y; h += inc_y) {
        for (sd::Unsigned w = 0; w < oW; ++w) {
          for (sd::Unsigned c = 0; c < iC; ++c) {
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
                      (const NDArray& input, NDArray& output, const sd::Unsigned padBottom, const sd::Unsigned padTop,
                       const sd::Unsigned padLeft, const sd::Unsigned padRight),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatch(sd::LaunchContext* context, const NDArray& input, NDArray& output, const sd::Unsigned padBottom,
                  const sd::Unsigned padTop, const sd::Unsigned padLeft, const sd::Unsigned padRight,
                  const sd::Unsigned blockSize) {
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW +
  // padLeft + padRight)/blockSize, iC]

  NDArray outputRearranged0 = output.reshape(
      output.ordering(), {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), output.sizeAt(3)},
      false);
  outputRearranged0.permutei({2, 3, 0, 4, 1, 5});

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(input);
  } else {
    NDArray outputRearranged1 = outputRearranged0.reshape(
        output.ordering(),
        {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, output.sizeAt(3)}, false);
    BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatch_,
                          (input, outputRearranged1, padBottom, padTop, padLeft, padRight), SD_COMMON_TYPES);

    if (output.buffer() != outputRearranged1.buffer()) outputRearranged0.assign(outputRearranged1);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatchND_(const NDArray& input, const NDArray& padding, NDArray& output,
                            const sd::Unsigned numOfSpatialDims) {
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
    int zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, output.shapeInfo(), zCoords);

      const auto zOffset = shape::getOffset(output.shapeInfo(), zCoords);

      memcpy(xCoords, zCoords, rank * sizeof(int));

      bool within = true;

      for (sd::Unsigned j = 1; j <= numOfSpatialDims; ++j) {
        const auto padLeft = padding.e<sd::Unsigned>(j - 1, 0);
        const auto padRight = padding.e<sd::Unsigned>(j - 1, 1);

        within &= zCoords[j] >= padLeft && zCoords[j] < output.sizeAt(j) - padRight;

        if (!within) break;

        xCoords[j] = zCoords[j] - padLeft;  // get coordinates for x
      }

      if (within)
        z[zOffset] = x[shape::getOffset(input.shapeInfo(), xCoords)];
      else
        z[zOffset] = 0.f;
    }
  };

  samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void spaceToBatchND_,
                      (const NDArray& input, const NDArray& padding, NDArray& output,
                       const sd::Unsigned numOfSpatialDims),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatchND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape, const NDArray& padding,
                    NDArray& output) {
  // 4D example with two spatial dimensions
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom +
  // padTop)/blockShape[0], (iW + padLeft + padRight)/blockShape[1], iC]

  const sd::Unsigned rank = input.rankOf();

  const sd::Unsigned numOfSpatialDims = blockShape.sizeAt(0);

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
  for (i = 2 * numOfSpatialDims + 1; i < temp.size(); ++i) temp[i] = i;

  outputRearranged0.permutei(temp);

  // ****** //

  if (input.lengthOf() == output.lengthOf()) {
    outputRearranged0.assign(input);
  } else {
    //*** construct reshaping std::vector for second reshape of output array ***//
    temp.resize(rank);

    temp[0] = input.sizeAt(0);

    for (i = 1; i < rank; ++i)
      temp[i] = (i <= numOfSpatialDims) ? output.sizeAt(i) * blockShape.e<sd::LongType>(i - 1) : output.sizeAt(i);

    NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), temp, false);

    BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchND_, (input, padding, outputRearranged1, numOfSpatialDims),
                          SD_COMMON_TYPES);

    if (output.buffer() != outputRearranged1.buffer()) outputRearranged0.assign(outputRearranged1);
  }
}

/*
    template <int N, bool B2S>
    struct SpaceToBatchHelper {
        template <typename T>
        static void run(T *ptrSpace, const sd::LongType *space_shape, const sd::LongType *space_strides, const
sd::LongType *block_shape, const sd::LongType *pad_start, const sd::LongType *block_offsets, T *ptrBatch, const
sd::LongType *batch_shape, const sd::LongType *batch_strides) { for (int batch_pos = 0; batch_pos < batch_shape[0];
++batch_pos) { const int space_pos = batch_pos * block_shape[0] + block_offsets[0] - pad_start[0]; if (space_pos >= 0 &&
space_pos < space_shape[0]) { SpaceToBatchHelper<N - 1, B2S>::run(ptrSpace + space_pos * space_strides[0], space_shape +
1, space_strides + 1, block_shape + 1, pad_start + 1, block_offsets + 1, ptrBatch, batch_shape + 1, batch_strides + 1);
                } else {
                    if (!B2S)
                        for (int i = 0; i < batch_strides[0]; i++)
                            ptrBatch[i] = (T) 0.f;
                }

                ptrBatch += batch_strides[0];
            }
        }
    };

    template <bool B2S>
    struct SpaceToBatchHelper<0, B2S> {
        template <typename T>
        static void run(T *ptrSpace, const sd::LongType *space_shape, const sd::LongType *space_strides, const
sd::LongType *block_shape, const sd::LongType *pad_start, const sd::LongType *block_offsets, T *ptrBatch, const
sd::LongType *batch_shape, const sd::LongType *batch_strides) { int str = batch_strides[-1]; for (int i = 0; i < str;
i++) if (B2S) ptrSpace[i] = ptrBatch[i]; else ptrBatch[i] = ptrSpace[i];
        }
    };

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(sd::LaunchContext * context, void *vptrSpace, const sd::LongType *space_shape, const sd::LongType
*space_strides, const sd::LongType *block_shape, const sd::LongType *pad_start, const sd::LongType *block_offsets, void
*vptrBatch, const sd::LongType *batch_shape, const sd::LongType *batch_strides) { auto ptrSpace = reinterpret_cast<T
*>(vptrSpace); auto ptrBatch = reinterpret_cast<T *>(vptrBatch); SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace,
space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };

    sd::Status _spaceToBatch(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output,
std::vector<sd::LongType> &internal_input_shape, std::vector<sd::LongType> &internal_output_shape, sd::LongType
*block_shape, sd::LongType *paddings) { auto in = input->reshape('c', internal_input_shape); auto out =
output->reshape('c', internal_output_shape); switch (internal_block_dims) { case 1: _prepare<1, false>(context, &in,
&out, block_shape, paddings); break; case 2: _prepare<2, false>(context, &in, &out, block_shape, paddings); break; case
3: _prepare<3, false>(context, &in, &out, block_shape, paddings); break; case 4: _prepare<4, false>(context, &in, &out,
block_shape, paddings); break; default: { return Logger::logKernelFailureMsg("SpaceToBatch: Wrong number of
internal_block_dims");
            }
        }

        return sd::Status::OK;
    }

    sd::Status _batchToSpace(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output,
std::vector<sd::LongType> &internal_input_shape, std::vector<sd::LongType> &internal_output_shape, sd::LongType
*block_shape, sd::LongType *crops) { auto in = input->reshape('c', internal_input_shape); auto out =
output->reshape('c', internal_output_shape); switch (internal_block_dims) { case 1: _prepare<1, true>(context, &in,
&out, block_shape, crops); break; case 2: _prepare<2, true>(context, &in, &out, block_shape, crops); break; case 3:
                _prepare<3, true>(context, &in, &out, block_shape, crops);
                break;
            case 4:
                _prepare<4, true>(context, &in, &out, block_shape, crops);
                break;
            default: {
                return Logger::logKernelFailureMsg("BatchToSpace: Wrong number of internal_block_dims");
            }
        }

        return sd::Status::OK;
    }

#define STB_DIM (0, 1),\
                (1, 2),\
                (2, 3),\
                (3, 4)

#define STB_BOOL (0, false),\
                 (1, true)

    BUILD_TRIPLE_TEMPLATE(template void _execute, (sd::LaunchContext * context, void *ptrSpace, const sd::LongType
*space_shape, const sd::LongType *space_strides, const sd::LongType *block_shape, const sd::LongType *pad_start, const
sd::LongType *block_offsets, void *ptrBatch, const sd::LongType *batch_shape, const sd::LongType *batch_strides),
SD_COMMON_TYPES, STB_DIM, STB_BOOL);

#undef STB_BOOL
#undef STB_DIM
*/

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif