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
#include <PointersManager.h>

namespace nd4j    {
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
    // xLen > zLen

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ int rank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= zLen)
        return;

    shape::index2coords(rank, zShapeInfo + 1, i, zLen, coords);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    coords[1] += cropBottom;
    coords[2] += cropLeft;

    const auto xOffset = shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank);

    z[zOffset] = x[xOffset];

}

///////////////////////////////////////////////////////////////////
template<typename T>
static void batchToSpaceCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint cropBottom, const uint cropLeft) {

    batchToSpaceCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, cropBottom, cropLeft);
}
BUILD_SINGLE_TEMPLATE(template void batchToSpaceCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint cropBottom, const uint cropLeft), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
void batchToSpace(nd4j::LaunchContext* context, const NDArray& input, NDArray& output, const uint cropBottom, const uint cropTop, const uint cropLeft, const uint cropRight, const uint blockSize) {

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
        const int sharedMem = threadsPerBlock * sizeof(Nd4jLong) * output.rankOf() + 128;

        PointersManager manager(context, "batchToSpace");

        NDArray::prepareSpecialUse({&output}, {&inputRearranged1});
        BUILD_SINGLE_SELECTOR(input.dataType(), batchToSpaceCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), inputRearranged1.getSpecialBuffer(), inputRearranged1.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), cropBottom, cropLeft), LIBND4J_TYPES);
        NDArray::registerSpecialUse({&output}, {&inputRearranged1});

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

    __shared__ int rank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        rank  = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= zLen)
        return;

    shape::index2coords(rank, zShapeInfo + 1, i, zLen, coords);

    const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

    if(coords[1] >= padBottom && coords[1] < zShapeInfo[2] - padTop && coords[2] >= padLeft && coords[2] < zShapeInfo[3] - padRight) {

        coords[1] -= padBottom;
        coords[2] -= padLeft;

        const auto xOffset = shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank);

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
void spaceToBatch(nd4j::LaunchContext* context, const NDArray& input, NDArray& output, const uint padBottom, const uint padTop, const uint padLeft, const uint padRight, const uint blockSize) {

    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW + padLeft + padRight)/blockSize, iC]

    NDArray outputRearranged0 = output.reshape(output.ordering(), {blockSize, blockSize, input.sizeAt(0), output.sizeAt(1), output.sizeAt(2), input.sizeAt(3)});
    outputRearranged0.permutei({2, 3,0, 4,1, 5});

    if(input.lengthOf() == output.lengthOf()) {

        outputRearranged0.assign(input);
    }
    else {

        NDArray outputRearranged1 = outputRearranged0.reshape(output.ordering(), {input.sizeAt(0), output.sizeAt(1) * blockSize, output.sizeAt(2) * blockSize, input.sizeAt(3)});

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(Nd4jLong) * output.rankOf() + 128;

        PointersManager manager(context, "spaceToBatch");

        NDArray::prepareSpecialUse({&outputRearranged1}, {&input});
        BUILD_SINGLE_SELECTOR(input.dataType(), spaceToBatchCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), outputRearranged1.specialBuffer(), outputRearranged1.specialShapeInfo(), padBottom, padTop, padLeft, padRight), LIBND4J_TYPES);
        NDArray::registerSpecialUse({&outputRearranged1}, {&input});

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
    void _execute(nd4j::LaunchContext * context, void *vptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *vptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
        auto ptrSpace = reinterpret_cast<T *>(vptrSpace);
        auto ptrBatch = reinterpret_cast<T *>(vptrBatch);
        SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace, space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };

    Nd4jStatus _batchToSpace(nd4j::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *crops) {

        return Status::OK();
    }

#define STB_DIM (0, 1),\
                (1, 2),\
                (2, 3),\
                (3, 4)

#define STB_BOOL (0, false),\
                 (1, true)

    BUILD_TRIPLE_TEMPLATE(template void _execute, (nd4j::LaunchContext * context, void *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides), LIBND4J_TYPES, STB_DIM, STB_BOOL);

#undef STB_BOOL
#undef STB_DIM
*/

}
}
}