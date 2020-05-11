/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/s_t_b.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpace_(const NDArray& input, NDArray& output, const uint cropBottom, const uint cropTop, const uint cropLeft, const uint cropRight) {

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

    const Nd4jLong* xShapeInfo = input.shapeInfo();
    const Nd4jLong* zShapeInfo = output.shapeInfo();

    const uint bS = xShapeInfo[1];
    const uint iH = xShapeInfo[2];
    const uint iW = xShapeInfo[3];
    const uint iC = xShapeInfo[4];

    // loop through output array
    auto func = PRAGMA_THREADS_FOR_3D {
        for (auto b = start_x; b < stop_x; b += inc_x) {
            for (auto h = start_y; h < stop_y; h += inc_y) {
                for (auto w = start_z; w < stop_z; w += inc_z) {
                    for (uint c = 0; c < iC; ++c) {
                        const Nd4jLong xOffset = b * xShapeInfo[5] + h * xShapeInfo[6] + w * xShapeInfo[7] + c * xShapeInfo[8];
                        const Nd4jLong zOffset = b * zShapeInfo[5] + (h - cropBottom) * zShapeInfo[6] + (w - cropLeft) * zShapeInfo[7] + c * zShapeInfo[8];

                        z[zOffset] = x[xOffset];
                    }
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, bS, 1, cropBottom, iH - cropTop, 1, cropLeft, iW - cropRight, 1);
}

BUILD_SINGLE_TEMPLATE(template void batchToSpace_, (const NDArray& input, NDArray& output, const uint cropBottom, const uint cropTop, const uint cropLeft, const uint cropRight), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpace(sd::LaunchContext* context, const NDArray& input, NDArray& output, const uint cropBottom, const uint cropTop, const uint cropLeft, const uint cropRight, const uint blockSize) {

    // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
    // oH = H - cropTop  - cropBottom
    // oW = W - cropLeft - cropRight

    NDArray inputRearranged0 = input.reshape(input.ordering(), {blockSize, blockSize, output.sizeAt(0), input.sizeAt(1), input.sizeAt(2), input.sizeAt(3)});
    inputRearranged0.permutei({2, 3,0, 4,1, 5});

    if(input.lengthOf() == output.lengthOf())
        output.assign(inputRearranged0);
    else {
        NDArray inputRearranged1 = inputRearranged0.reshape(input.ordering(), {output.sizeAt(0), input.sizeAt(1) * blockSize, input.sizeAt(2) * blockSize, input.sizeAt(3)});
        BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpace_, (inputRearranged1, output, cropBottom, cropTop, cropLeft, cropRight), LIBND4J_TYPES);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchToSpaceND_(const NDArray& input, const NDArray& crop, NDArray& output, const uint numOfSpatialDims) {

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
    const Nd4jLong zLen = output.lengthOf();

    // loop through input array
    auto func = PRAGMA_THREADS_FOR {

        int zCoords[MAX_RANK], xCoords[MAX_RANK];

        for (auto i = start; i < stop; i++) {

            shape::index2coordsCPU(start, i, output.shapeInfo(), zCoords);

            memcpy(xCoords, zCoords, rank * sizeof(int));

            // evaluate spatial coordinates for x
            for (uint j = 1; j <= numOfSpatialDims; ++j)
                xCoords[j] += crop.e<uint>(j - 1, 0);       // add crop left

            const auto zOffset = shape::getOffset(output.shapeInfo(), zCoords);
            const auto xOffset = shape::getOffset(input.shapeInfo(), xCoords);

            z[zOffset] = x[xOffset];
        }
    };

    samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void batchToSpaceND_, (const NDArray& input, const NDArray& crop, NDArray& output, const uint numOfSpatialDims), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpaceND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape, const NDArray& crop, NDArray& output) {

    // 4D example, numOfSpatialDims = 2 - two spatial dimensions
    // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  - cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

    const uint rank = input.rankOf();
    const uint numOfSpatialDims = blockShape.sizeAt(0);

    //*** construct reshaping std::vector for first reshape of input array ***//

    std::vector<Nd4jLong> temp(numOfSpatialDims + rank);

    uint i;
    for(i = 0; i < numOfSpatialDims; ++i)
        temp[i] = blockShape.e<Nd4jLong>(i);
    temp[i++] = output.sizeAt(0);
    for(uint j = 1; j < rank; ++i, ++j)
        temp[i] = input.sizeAt(j);

    NDArray inputRearranged0 = input.reshape(input.ordering(), temp);

    //*** construct permuting std::vector for permutation of input array ***//

    temp[0] = numOfSpatialDims;

    for(i = 1; i <= numOfSpatialDims; ++i) {
        temp[2*i - 1] = numOfSpatialDims + i;
        temp[2*i]     = i - 1;
    }
    for(i = 2 * numOfSpatialDims + 1; i < static_cast<uint>(temp.size()); ++i)
        temp[i] = i;

    inputRearranged0.permutei(temp);


    if(input.lengthOf() == output.lengthOf()) {
        output.assign(inputRearranged0);
    }
    else {
        //*** construct reshaping std::vector for second reshape of input array ***//

        temp.resize(rank);

        temp[0] = output.sizeAt(0);

        for(i = 1; i < rank; ++i)
            temp[i] = (i <= numOfSpatialDims) ? input.sizeAt(i) * blockShape.e<Nd4jLong>(i - 1) : input.sizeAt(i);

        NDArray inputRearranged1 = inputRearranged0.reshape(input.ordering(), temp);

        BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpaceND_, (inputRearranged1, crop, output, numOfSpatialDims), LIBND4J_TYPES);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatch_(const NDArray& input, NDArray& output, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight) {

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

    const Nd4jLong* xShapeInfo = input.shapeInfo();
    const Nd4jLong* zShapeInfo = output.shapeInfo();

    const uint bS = zShapeInfo[1];
    const uint oH = zShapeInfo[2];
    const uint oW = zShapeInfo[3];
    const uint iC = zShapeInfo[4];

    // loop through output array
    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b += inc_x) {
            for (auto h = start_y; h < stop_y; h += inc_y) {
                for (uint w = 0; w < oW; ++w) {
                    for (uint c = 0; c < iC; ++c) {

                        const Nd4jLong zOffset = b * zShapeInfo[5] + h * zShapeInfo[6] + w * zShapeInfo[7] + c * zShapeInfo[8];

                        if (h >= padBottom && h < oH - padTop && w >= padLeft && w < oW - padRight) {
                            const Nd4jLong xOffset = b * xShapeInfo[5] + (h - padBottom) * xShapeInfo[6] + (w - padLeft) * xShapeInfo[7] + c * xShapeInfo[8];
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

BUILD_SINGLE_TEMPLATE(template void spaceToBatch_, (const NDArray& input, NDArray& output, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatch(sd::LaunchContext* context, const NDArray& input, NDArray& output, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight, const uint blockSize) {

    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW + padLeft + padRight)/blockSize, iC]

    NDArray outputRearranged0 = output.reshape(output.ordering(), {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), output.sizeAt(3)}, false);
    outputRearranged0.permutei({2, 3,0, 4,1, 5});

    if(input.lengthOf() == output.lengthOf()) {
        outputRearranged0.assign(input);
    }
    else {
        NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, output.sizeAt(3)}, false);
        BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatch_, (input, outputRearranged1, padBottom, padTop, padLeft, padRight), LIBND4J_TYPES);

        if(output.buffer() != outputRearranged1.buffer())
            outputRearranged0.assign(outputRearranged1);
    }
}



















//////////////////////////////////////////////////////////////////////////
template <typename T>
static void spaceToBatchND_(const NDArray& input, const NDArray& padding, NDArray& output, const uint numOfSpatialDims) {

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
    const Nd4jLong zLen = output.lengthOf();

    // loop through output array
    auto func = PRAGMA_THREADS_FOR {

        int zCoords[MAX_RANK], xCoords[MAX_RANK];

        for (auto i = start; i < stop; i++) {

            shape::index2coordsCPU(start, i, output.shapeInfo(), zCoords);

            const auto zOffset = shape::getOffset(output.shapeInfo(), zCoords);

            memcpy(xCoords, zCoords, rank * sizeof(int));

            bool within = true;

            for (uint j = 1; j <= numOfSpatialDims; ++j) {

                const auto padLeft = padding.e<uint>(j - 1, 0);
                const auto padRight = padding.e<uint>(j - 1, 1);

                within &= zCoords[j] >= padLeft && zCoords[j] < output.sizeAt(j) - padRight;

                if (!within)
                    break;

                xCoords[j] = zCoords[j] - padLeft;       // get coordinates for x
            }

            if (within)
                z[zOffset] = x[shape::getOffset(input.shapeInfo(), xCoords)];
            else
                z[zOffset] = 0.f;
        }
    };

    samediff::Threads::parallel_tad(func, 0, zLen);
}

BUILD_SINGLE_TEMPLATE(template void spaceToBatchND_, (const NDArray& input, const NDArray& padding, NDArray& output, const uint numOfSpatialDims), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void spaceToBatchND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape, const NDArray& padding, NDArray& output ) {

    // 4D example with two spatial dimensions
    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom + padTop)/blockShape[0], (iW + padLeft + padRight)/blockShape[1], iC]

    const uint rank = input.rankOf();

    const uint numOfSpatialDims = blockShape.sizeAt(0);

    //*** construct reshaping std::vector for first reshape of output array ***//
    std::vector<Nd4jLong> temp(numOfSpatialDims + rank);

    int i;
    for(i = 0; i < numOfSpatialDims; ++i)
        temp[i] = blockShape.e<Nd4jLong>(i);
    temp[i++] = input.sizeAt(0);
    for(int j = 1; j < rank; ++i, ++j)
        temp[i] = output.sizeAt(j);

    NDArray outputRearranged0 = output.reshape(output.ordering(), temp, false);

    //*** construct permuting std::vector for permutation of output array ***//

    temp[0] = numOfSpatialDims;

    for(i = 1; i <= numOfSpatialDims; ++i) {
        temp[2*i - 1] = numOfSpatialDims + i;
        temp[2*i]     = i - 1;
    }
    for(i = 2 * numOfSpatialDims + 1; i < temp.size(); ++i)
        temp[i] = i;

    outputRearranged0.permutei(temp);

    // ****** //

    if(input.lengthOf() == output.lengthOf()) {
        outputRearranged0.assign(input);
    }
    else {

        //*** construct reshaping std::vector for second reshape of output array ***//
        temp.resize(rank);

        temp[0] = input.sizeAt(0);

        for(i = 1; i < rank; ++i)
            temp[i] = (i <= numOfSpatialDims) ? output.sizeAt(i) * blockShape.e<Nd4jLong>(i - 1) : output.sizeAt(i);

        NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), temp, false);

        BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchND_, (input, padding, outputRearranged1, numOfSpatialDims), LIBND4J_TYPES);

        if(output.buffer() != outputRearranged1.buffer())
            outputRearranged0.assign(outputRearranged1);
    }
}


/*
    template <int N, bool B2S>
    struct SpaceToBatchHelper {
        template <typename T>
        static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
            for (int batch_pos = 0; batch_pos < batch_shape[0]; ++batch_pos) {
                const int space_pos = batch_pos * block_shape[0] + block_offsets[0] - pad_start[0];
                if (space_pos >= 0 && space_pos < space_shape[0]) {
                    SpaceToBatchHelper<N - 1, B2S>::run(ptrSpace + space_pos * space_strides[0], space_shape + 1, space_strides + 1, block_shape + 1, pad_start + 1, block_offsets + 1, ptrBatch, batch_shape + 1, batch_strides + 1);
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
        static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
            int str = batch_strides[-1];
            for (int i = 0; i < str; i++)
                if (B2S)
                    ptrSpace[i] = ptrBatch[i];
                else
                    ptrBatch[i] = ptrSpace[i];
        }
    };

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(sd::LaunchContext * context, void *vptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *vptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
        auto ptrSpace = reinterpret_cast<T *>(vptrSpace);
        auto ptrBatch = reinterpret_cast<T *>(vptrBatch);
        SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace, space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };

    Nd4jStatus _spaceToBatch(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *paddings) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<1, false>(context, &in, &out, block_shape, paddings);
                break;
            case 2:
                _prepare<2, false>(context, &in, &out, block_shape, paddings);
                break;
            case 3:
                _prepare<3, false>(context, &in, &out, block_shape, paddings);
                break;
            case 4:
                _prepare<4, false>(context, &in, &out, block_shape, paddings);
                break;
            default: {
                return Status::THROW("SpaceToBatch: Wrong number of internal_block_dims");
            }
        }

        return Status::OK();
    }

    Nd4jStatus _batchToSpace(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *crops) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<1, true>(context, &in, &out, block_shape, crops);
                break;
            case 2:
                _prepare<2, true>(context, &in, &out, block_shape, crops);
                break;
            case 3:
                _prepare<3, true>(context, &in, &out, block_shape, crops);
                break;
            case 4:
                _prepare<4, true>(context, &in, &out, block_shape, crops);
                break;
            default: {
                return Status::THROW("BatchToSpace: Wrong number of internal_block_dims");
            }
        }

        return Status::OK();
    }

#define STB_DIM (0, 1),\
                (1, 2),\
                (2, 3),\
                (3, 4)

#define STB_BOOL (0, false),\
                 (1, true)

    BUILD_TRIPLE_TEMPLATE(template void _execute, (sd::LaunchContext * context, void *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides), LIBND4J_TYPES, STB_DIM, STB_BOOL);

#undef STB_BOOL
#undef STB_DIM
*/

}
}
}