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
// Created by raver119 on 19.01.18.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/s_t_b.h>
#include <helpers/PointersManager.h>

namespace sd    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void batchToSpaceCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint cropBottom, const uint cropLeft) {

    // input [bS, H * blockSize, W * blockSize, iC]
    // output [bS, H * blockSize - cropBottom - cropTop, W * blockSize - cropLeft - cropRight, iC]

    // if (cropTop = cropBottom = cropRight = cropLeft = 0) shapes are the same
    // else:
    // oH -> [cropBottom, iH - cropTop]
    // oW -> [cropLeft,   iH - cropRight]
    // xLen >= zLen

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ int rank, *sharedMem;
    __shared__ Nd4jLong zLen;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= zLen)
        return;

    shape::index2coords(i, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    coords[1] += cropBottom;
    coords[2] += cropLeft;

    const auto xOffset = shape::getOffset(xShapeInfo, coords);

    z[zOffset] = x[xOffset];

}

///////////////////////////////////////////////////////////////////
template<typename T>
static void batchToSpaceCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint cropBottom, const uint cropLeft) {

    batchToSpaceCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, cropBottom, cropLeft);
}
BUILD_SINGLE_TEMPLATE(template void batchToSpaceCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint cropBottom, const uint cropLeft), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
void batchToSpace(sd::LaunchContext* context, const NDArray& input, NDArray& output, const uint cropBottom, const uint cropTop, const uint cropLeft, const uint cropRight, const uint blockSize) {

    // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
    // oH = H - cropTop  - cropBottom
    // oW = W - cropLeft - cropRight

    NDArray inputRearranged0 = input.reshape(input.ordering(), {blockSize, blockSize, output.sizeAt(0), input.sizeAt(1), input.sizeAt(2), input.sizeAt(3)});
    inputRearranged0.permutei({2, 3,0, 4,1, 5});

    if(input.lengthOf() == output.lengthOf()) {

        output.assign(inputRearranged0);
    }
    else {

        NDArray inputRearranged1 = inputRearranged0.reshape(input.ordering(), {output.sizeAt(0), input.sizeAt(1) * blockSize, input.sizeAt(2) * blockSize, input.sizeAt(3)});

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(int) * output.rankOf() + 128;

        PointersManager manager(context, "batchToSpace");

        NDArray::prepareSpecialUse({&output}, {&inputRearranged1});
        BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpaceCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), inputRearranged1.getSpecialBuffer(), inputRearranged1.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), cropBottom, cropLeft), LIBND4J_TYPES);
        NDArray::registerSpecialUse({&output}, {&inputRearranged1});

        manager.synchronize();
    }
}



///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void batchToSpaceNDCuda(const void* vx, const Nd4jLong* xShapeInfo,
                                          const void* vy, const Nd4jLong* yShapeInfo,
                                                void* vz, const Nd4jLong* zShapeInfo,
                                          const uint numOfSpatialDims) {

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

    __shared__ int rank, *sharedMem;
    __shared__ Nd4jLong zLen;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, zShapeInfo, coords);

        const auto zOffset = shape::getOffset(zShapeInfo, coords);

        // evaluate spatial coordinates for x
        for(uint j = 1; j <= numOfSpatialDims; ++j) {
            const auto yOffset  = (j - 1) * yShapeInfo[3];  // yRank = 2, calculate offset manually
            coords[j] += y[yOffset];                        // add crop left
        }

        const auto xOffset = shape::getOffset(xShapeInfo, coords);

        z[zOffset] = x[xOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename X,typename Y>
static void batchToSpaceNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint numOfSpatialDims) {

    batchToSpaceNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, numOfSpatialDims);
}
BUILD_DOUBLE_TEMPLATE(template void batchToSpaceNDCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint numOfSpatialDims), LIBND4J_TYPES, INTEGER_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchToSpaceND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape, const NDArray& crop, NDArray& output) {

    // 4D example, numOfSpatialDims = 2 - two spatial dimensions
    // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  - cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

    const uint rank = input.rankOf();
    const uint numOfSpatialDims = blockShape.sizeAt(0);

    //*** construct reshaping std::vector for first reshape of input array ***//

    std::vector<Nd4jLong> temp(numOfSpatialDims + rank);

    int i;
    for(i = 0; i < numOfSpatialDims; ++i)
        temp[i] = blockShape.e<Nd4jLong>(i);
    temp[i++] = output.sizeAt(0);
    for(int j = 1; j < rank; ++i, ++j)
        temp[i] = input.sizeAt(j);

    NDArray inputRearranged0 = input.reshape(input.ordering(), temp);

    //*** construct permuting std::vector for permutation of input array ***//

    temp[0] = numOfSpatialDims;

    for(i = 1; i <= numOfSpatialDims; ++i) {
        temp[2*i - 1] = numOfSpatialDims + i;
        temp[2*i]     = i - 1;
    }
    for(i = 2 * numOfSpatialDims + 1; i < temp.size(); ++i)
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

        const int threadsPerBlock = MAX_NUM_THREADS / 4;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(int) * output.rankOf() + 128;

        PointersManager manager(context, "batchToSpaceND");

        NDArray::prepareSpecialUse({&output}, {&inputRearranged1, &crop});
        BUILD_DOUBLE_SELECTOR(input.dataType(), crop.dataType(), batchToSpaceNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), inputRearranged1.getSpecialBuffer(), inputRearranged1.getSpecialShapeInfo(), crop.getSpecialBuffer(), crop.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), numOfSpatialDims), LIBND4J_TYPES, INTEGER_TYPES);
        NDArray::registerSpecialUse({&output}, {&inputRearranged1, &crop});

        manager.synchronize();
    }
}



///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void spaceToBatchCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight) {

    // input [bS, H * blockSize - padBottom - padTop, W * blockSize - padLeft - padRight, iC]
    // output [bs, H * blockSize, W * blockSize, iC]

    // if (padTop = padBottom = padRight = padLeft = 0) shapes are the same
    // else:
    // iH -> [padBottom, oH - padTop]
    // iW -> [padLeft,   oW - padRight]
    // zLen > xLen

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ int rank, *sharedMem;
    __shared__ Nd4jLong zLen;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= zLen)
        return;

    shape::index2coords(i, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    if(coords[1] >= padBottom && coords[1] < zShapeInfo[2] - padTop && coords[2] >= padLeft && coords[2] < zShapeInfo[3] - padRight) {

        coords[1] -= padBottom;
        coords[2] -= padLeft;

        const auto xOffset = shape::getOffset(xShapeInfo, coords);

        z[zOffset] = x[xOffset];
    }
    else
        z[zOffset] = 0.f;
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void spaceToBatchCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight) {

    spaceToBatchCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, padBottom, padTop, padLeft, padRight);
}
BUILD_SINGLE_TEMPLATE(template void spaceToBatchCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
void spaceToBatch(sd::LaunchContext* context, const NDArray& input, NDArray& output, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight, const uint blockSize) {

    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW + padLeft + padRight)/blockSize, iC]

    NDArray outputRearranged0 = output.reshape(output.ordering(), {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), input.sizeAt(3)}, false);
    outputRearranged0.permutei({2, 3,0, 4,1, 5});

    if(input.lengthOf() == output.lengthOf()) {

        outputRearranged0.assign(input);
    }
    else {

        NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, input.sizeAt(3)}, false);

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(int) * output.rankOf() + 128;

        PointersManager manager(context, "spaceToBatch");

        NDArray::prepareSpecialUse({&outputRearranged1}, {&input});
        BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), outputRearranged1.specialBuffer(), outputRearranged1.specialShapeInfo(), padBottom, padTop, padLeft, padRight), LIBND4J_TYPES);
        NDArray::registerSpecialUse({&outputRearranged1}, {&input});

        manager.synchronize();

        if(output.getSpecialBuffer() != outputRearranged1.getSpecialBuffer())
            outputRearranged0.assign(outputRearranged1);
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void spaceToBatchNDCuda(const void* vx, const Nd4jLong* xShapeInfo,
                                          const void* vy, const Nd4jLong* yShapeInfo,
                                                void* vz, const Nd4jLong* zShapeInfo,
                                          const uint numOfSpatialDims) {

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

    __shared__ int rank, *sharedMem;    // xRank = zRank, yRank = 2;
    __shared__ Nd4jLong zLen, totalThreads;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, coords);

        const auto zOffset = shape::getOffset(zShapeInfo, coords);

        bool within = true;

        for(uint j = 1; j <= numOfSpatialDims; ++j) {

            // yRank = 2, calculate offset manually
            const auto yOffset  = (j - 1) * yShapeInfo[3];
            const auto padLeft  = y[yOffset];
            const auto padRight = y[yOffset + yShapeInfo[4]];

            within &= (coords[j] >= padLeft && coords[j] < shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo))[j] - padRight);

            if(!within)
                break;

            coords[j] -= padLeft;       // get coordinates for x
        }

        if(within)
            z[zOffset] = x[shape::getOffset(xShapeInfo, coords)];
        else
            z[zOffset] = 0.f;
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void spaceToBatchNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint numOfSpatialDims) {

    spaceToBatchNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, numOfSpatialDims);
}
BUILD_DOUBLE_TEMPLATE(template void spaceToBatchNDCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint numOfSpatialDims), LIBND4J_TYPES, INTEGER_TYPES);

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

        const int threadsPerBlock = MAX_NUM_THREADS / 4;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(int) * output.rankOf() + 128;

        PointersManager manager(context, "spaceToBatchND");

        NDArray::prepareSpecialUse({&outputRearranged1}, {&input, &padding});
        BUILD_DOUBLE_SELECTOR(input.dataType(), padding.dataType(), spaceToBatchNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), padding.getSpecialBuffer(), padding.getSpecialShapeInfo(), outputRearranged1.specialBuffer(), outputRearranged1.specialShapeInfo(), numOfSpatialDims), LIBND4J_TYPES, INTEGER_TYPES);
        NDArray::registerSpecialUse({&outputRearranged1}, {&input, &padding});

        manager.synchronize();

        if(output.getSpecialBuffer() != outputRearranged1.getSpecialBuffer())
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

    Nd4jStatus _batchToSpace(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *crops) {

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