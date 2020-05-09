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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/scatter.h>
#include <numeric>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>

namespace sd    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
// x - indices, y - contains number of bad indices, z - input/output
template<typename X>
__global__ static void checkIndicesCuda(const void *vx, const Nd4jLong *xShapeInfo, Nd4jLong* y, const Nd4jLong *zShapeInfo, const int axis) {

    const auto x = reinterpret_cast<const X*>(vx);

    __shared__ int xRank, *coords, xLastDim;
    __shared__ Nd4jLong xLen, numOfBadIndxPerBlock;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);

        xRank = shape::rank(xShapeInfo);
        xLen  = shape::length(xShapeInfo);

        numOfBadIndxPerBlock = 0;
    }
    __syncthreads();

    auto xCoords = coords + threadIdx.x * xRank;

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, xShapeInfo, xCoords);

        const Nd4jLong currentInd = x[shape::getOffset(xShapeInfo, xCoords)];

        if(currentInd >= shape::sizeAt(zShapeInfo, axis == -1 ? xCoords[xRank-1] : axis)) {
            printf("checkIndices cuda: out of range element %lld at index %lld \n", currentInd,  i);
            sd::math::atomics::nd4j_atomicAdd<Nd4jLong>(&numOfBadIndxPerBlock, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && numOfBadIndxPerBlock != 0)
        sd::math::atomics::nd4j_atomicAdd<Nd4jLong>(y, numOfBadIndxPerBlock);
}

///////////////////////////////////////////////////////////////////
template<typename X>
static void checkIndicesCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                              const void *vx, const Nd4jLong *xShapeInfo, Nd4jLong* y, const Nd4jLong *zShapeInfo, const int axis) {

    checkIndicesCuda<X><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, y, zShapeInfo, axis);
}


///////////////////////////////////////////////////////////////////
Nd4jLong checkIndices(sd::LaunchContext *context, const NDArray& indices, const NDArray& output, const int axis) {

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (indices.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(int) * indices.rankOf() + 256;

    const auto xType = indices.dataType();

    PointersManager manager(context, "scatterNDcheckIndices");

    // scalar, initial value = 0
    NDArray numOfBadIndx(sd::DataType::INT64, context, true);

    NDArray::prepareSpecialUse({&numOfBadIndx}, {&indices});
    BUILD_SINGLE_SELECTOR(xType, checkIndicesCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices.specialBuffer(), indices.specialShapeInfo(), reinterpret_cast<Nd4jLong*>(numOfBadIndx.specialBuffer()), output.specialShapeInfo(), axis), INDEXING_TYPES);
    NDArray::registerSpecialUse({&numOfBadIndx}, {&indices});

    manager.synchronize();

    return numOfBadIndx.t<Nd4jLong>(0);
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template<typename X, typename Y>
__global__ static void scatterLockCuda(const int opCode,
                                        const void *vx, const Nd4jLong *xShapeInfo,
                                        const void *vy, const Nd4jLong *yShapeInfo,
                                              void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ int xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim, *coords;
    __shared__ Nd4jLong xLen, zLen;
    __shared__ bool is1Dcase, xySameStride;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);

        xLen = shape::length(xShapeInfo);
        zLen = shape::length(zShapeInfo);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

        is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) && (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) && (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

        if(is1Dcase)
            xySameStride = shape::stride(xShapeInfo)[xNonUnitDim] = shape::stride(yShapeInfo)[yNonUnitDim];
    }
    __syncthreads();


    Nd4jLong yOffset, zOffset;
    int zFirstCoord, *yCoords, *zCoords;

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {

        if(!is1Dcase) {

            yCoords = coords + threadIdx.x * (yRank + zRank);
            zCoords = yCoords + yRank;
            shape::index2coords(i, zShapeInfo, zCoords);
        }

        for (Nd4jLong j = 0; j < xLen; ++j) {

            if(is1Dcase) {

                yOffset = j * shape::stride(yShapeInfo)[yNonUnitDim];
                zFirstCoord = x[xySameStride ? yOffset : j * shape::stride(xShapeInfo)[xNonUnitDim]];

                if(i != zFirstCoord)
                    continue;

                zOffset = i * shape::stride(zShapeInfo)[zNonUnitDim];
            }

            else {

                shape::index2coords(j, xShapeInfo, yCoords);                 // first xRank coordinates in yCoords are the same for y and x

                zFirstCoord = x[shape::getOffset(xShapeInfo, yCoords)];

                if(zCoords[0] != zFirstCoord)
                    continue;

                for (uint k = 0; k < yRank - xRank; ++k)
                    yCoords[xRank + k] = zCoords[k + 1];

                yOffset = shape::getOffset(yShapeInfo, yCoords);
                zOffset = shape::getOffset(zShapeInfo, zCoords);
            }

            switch (opCode) {
                case pairwise::Add:
                    z[zOffset] += y[yOffset];
                    break;
                case pairwise::Subtract:
                    z[zOffset] -= y[yOffset];
                    break;
                case pairwise::Multiply:
                    z[zOffset] *= y[yOffset];
                    break;
                case pairwise::Divide:
                    z[zOffset] /= y[yOffset];
                    break;
                case pairwise::ReverseSubtract:
                    z[zOffset] = y[yOffset] - z[zOffset];
                    break;
                case pairwise::ReverseDivide:
                    z[zOffset] = y[yOffset] / z[zOffset];
                    break;
                case pairwise::CopyPws:
                    z[zOffset] = y[yOffset];
                    break;
                case pairwise::MaxPairwise:
                    if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                case pairwise::MinPairwise:
                    if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template<typename X, typename Y>
__global__ static void scatterCuda(const int opCode,
                                   const void *vx, const Nd4jLong *xShapeInfo,
                                   const void *vy, const Nd4jLong *yShapeInfo,
                                         void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ int xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim, *coords;
    __shared__ Nd4jLong yLen;
    __shared__ bool is1Dcase, xySameStride;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);

        yLen = shape::length(yShapeInfo);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

        is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) && (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) && (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

        if(is1Dcase)
            xySameStride = shape::stride(xShapeInfo)[xNonUnitDim] = shape::stride(yShapeInfo)[yNonUnitDim];
    }
    __syncthreads();


    Nd4jLong xOffset, yOffset, zOffset;
    int *yCoords, *zCoords;

    if(!is1Dcase) {
        yCoords = coords + threadIdx.x * (yRank + zRank);
        zCoords = yCoords + yRank;
    }

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {

        if(is1Dcase) {

            yOffset = i * shape::stride(yShapeInfo)[yNonUnitDim];
            zOffset = x[xySameStride ? yOffset : i * shape::stride(xShapeInfo)[xNonUnitDim]] * shape::stride(zShapeInfo)[zNonUnitDim];
        }
        else {
            shape::index2coords(i, yShapeInfo, yCoords);

            yOffset = shape::getOffset(yShapeInfo, yCoords);
            xOffset = shape::getOffset(xShapeInfo, yCoords);                // first xRank coordinates in yCoords are the same for y and x -> for (uint j = 0; j < xRank; ++j) xCoords[j] = yCoords[j];

            zCoords[0] = x[xOffset];

            for (uint j = 0; j < yRank - xRank; ++j)
                zCoords[j + 1] = yCoords[xRank + j];

            zOffset = shape::getOffset(zShapeInfo, zCoords);
        }

        switch (opCode) {
            case pairwise::Add:
                z[zOffset] += y[yOffset];
                break;
            case pairwise::Subtract:
                z[zOffset] -= y[yOffset];
                break;
            case pairwise::Multiply:
                z[zOffset] *= y[yOffset];
                break;
            case pairwise::Divide:
                z[zOffset] /= y[yOffset];
                break;
            case pairwise::ReverseSubtract:
                z[zOffset] = y[yOffset] - z[zOffset];
                break;
            case pairwise::ReverseDivide:
                z[zOffset] = y[yOffset] / z[zOffset];
                break;
            case pairwise::CopyPws:
                z[zOffset] = y[yOffset];
                break;
            case pairwise::MaxPairwise:
                if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                break;
            case pairwise::MinPairwise:
                if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                break;
            default:
                continue;
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void scatterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const int opCode,
                                const void *vx, const Nd4jLong *xShapeInfo,
                                const void *vy, const Nd4jLong *yShapeInfo,
                                      void *vz, const Nd4jLong *zShapeInfo,
                                const bool lock) {

    if(lock)
        scatterLockCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
    else
        scatterCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}


///////////////////////////////////////////////////////////////////
void scatter(sd::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const auto xType = indices.dataType();
    const auto yType = updates.dataType();

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = ((lock ? output.lengthOf() : updates.lengthOf()) + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = sizeof(int) * threadsPerBlock * (updates.rankOf() + output.rankOf()) + 256;

    PointersManager manager(context, "scatter");

    NDArray::prepareSpecialUse({&output}, {&updates, &indices});
    BUILD_DOUBLE_SELECTOR(xType, yType, scatterCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), op, indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    NDArray::registerSpecialUse({&output}, {&updates, &indices});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template<typename X, typename Y>
__global__ static void scatterNDLockCuda(const int opCode,
                                        const void *vx, const Nd4jLong *xShapeInfo,
                                        const void *vy, const Nd4jLong *yShapeInfo,
                                              void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ int xRank, yRank, zRank, biggerXYRank, xLastDim, *coords, xNonUnitDim, yNonUnitDim, zNonUnitDim;
    __shared__ Nd4jLong zLen, len;
    __shared__ bool is1Dcase;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);
        xLastDim = shape::sizeAt(xShapeInfo, -1);

        biggerXYRank = xRank > yRank ? xRank : yRank;

        xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

        is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) && (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) && (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

        len  = is1Dcase ?  shape::length(xShapeInfo) : shape::length(xShapeInfo) / xLastDim;
        zLen = shape::length(zShapeInfo);
    }
    __syncthreads();

    Nd4jLong yOffset, zOffset, xOffset;
    int *yCoords, *zCoords;

    if(!is1Dcase) {
        yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
        zCoords = yCoords + biggerXYRank;
    }

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {

        if(!is1Dcase)
            shape::index2coords(i, zShapeInfo, zCoords);

        for (Nd4jLong j = 0; j < len; ++j) {        // if !is1Dcase then we loop through first xRank-1 dimensions of x, that is we exclude last x dimension

            if(is1Dcase) {

                if(x[j * shape::stride(xShapeInfo)[xNonUnitDim]] != i)
                    continue;

                yOffset = j * shape::stride(yShapeInfo)[yNonUnitDim];
                zOffset = i * shape::stride(zShapeInfo)[zNonUnitDim];
            }
            else {

                shape::index2coords(j, xRank-1, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), yCoords);        // first xRank-1 coordinates in yCoords are the same for y and x

                // first iteration
                yCoords[xRank - 1] = 0;
                xOffset = shape::getOffset(xShapeInfo, yCoords);
                if(zCoords[0] != x[xOffset])
                    continue;

                // rest iterations
                bool matched = true;
                for (uint k = 1; k < xLastDim; ++k) {
                    yCoords[xRank - 1] = k;
                    xOffset += shape::stride(xShapeInfo)[xRank-1];
                    if(zCoords[k] != x[xOffset]) {
                        matched = false;
                        break;
                    }
                }

                if(!matched)
                    continue;

                for (uint k = xLastDim; k < zRank; ++k)
                    yCoords[yRank - zRank + k] = zCoords[k];

                yOffset = shape::getOffset(yShapeInfo, yCoords);
                zOffset = shape::getOffset(zShapeInfo, zCoords);
            }

            switch (opCode) {
                case pairwise::Add:
                    z[zOffset] += y[yOffset];
                    break;
                case pairwise::Subtract:
                    z[zOffset] -= y[yOffset];
                    break;
                case pairwise::Multiply:
                    z[zOffset] *= y[yOffset];
                    break;
                case pairwise::Divide:
                    z[zOffset] /= y[yOffset];
                    break;
                case pairwise::ReverseSubtract:
                    z[zOffset] = y[yOffset] - z[zOffset];
                    break;
                case pairwise::ReverseDivide:
                    z[zOffset] = y[yOffset] / z[zOffset];
                    break;
                case pairwise::CopyPws:
                    z[zOffset] = y[yOffset];
                    break;
                case pairwise::MaxPairwise:
                    if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                case pairwise::MinPairwise:
                    if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template<typename X, typename Y>
__global__ static void scatterNDCuda(const int opCode,
                                     const void *vx, const Nd4jLong *xShapeInfo,
                                     const void *vy, const Nd4jLong *yShapeInfo,
                                           void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ int xRank, yRank, zRank, biggerXYRank, xLastDim, *coords, xNonUnitDim, yNonUnitDim, zNonUnitDim;
    __shared__ Nd4jLong yLen;
    __shared__ bool is1Dcase;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);

        yLen  = shape::length(yShapeInfo);
        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);
        xLastDim = shape::sizeAt(xShapeInfo, -1);

        biggerXYRank = xRank > yRank ? xRank : yRank;

        xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

        is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) && (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) && (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));
    }
    __syncthreads();

    Nd4jLong yOffset, zOffset;
    int *yCoords, *zCoords;

    if(!is1Dcase) {
        yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
        zCoords = yCoords + biggerXYRank;
    }

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {

        if(is1Dcase) {

            yOffset = i * shape::stride(yShapeInfo)[zNonUnitDim];
            zOffset = x[i * shape::stride(xShapeInfo)[xNonUnitDim]] * shape::stride(zShapeInfo)[zNonUnitDim];
        }
        else {

            shape::index2coords(i, yShapeInfo, yCoords);

            yOffset = shape::getOffset(yShapeInfo, yCoords);

            if(yRank >= xRank)
                zCoords[xLastDim] = yCoords[xRank - 1];                // saving y coordinate, since it might be changed in next instructions

            for (uint j = 0; j < xLastDim; ++j) {                      // first xRank-1 coordinates in yCoords are the same for y and x
                yCoords[xRank - 1] = j;
                zCoords[j] = x[shape::getOffset(xShapeInfo, yCoords)];
            }

            for (uint j = xLastDim + 1; j < zRank; ++j)
                zCoords[j] = yCoords[yRank - zRank + j];

            zOffset = shape::getOffset(zShapeInfo, zCoords);
        }

        switch (opCode) {
            case pairwise::Add:
                z[zOffset] += y[yOffset];
                break;
            case pairwise::Subtract:
                z[zOffset] -= y[yOffset];
                break;
            case pairwise::Multiply:
                z[zOffset] *= y[yOffset];
                break;
            case pairwise::Divide:
                z[zOffset] /= y[yOffset];
                break;
            case pairwise::ReverseSubtract:
                z[zOffset] = y[yOffset] - z[zOffset];
                break;
            case pairwise::ReverseDivide:
                z[zOffset] = y[yOffset] / z[zOffset];
                break;
            case pairwise::CopyPws:
                z[zOffset] = y[yOffset];
                break;
            case pairwise::MaxPairwise:
                if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                break;
            case pairwise::MinPairwise:
                if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                break;
            default:
                continue;
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void scatterNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                  const int opCode,
                                  const void *vx, const Nd4jLong *xShapeInfo,
                                  const void *vy, const Nd4jLong *yShapeInfo,
                                        void *vz, const Nd4jLong *zShapeInfo,
                                  const bool lock) {

    if(lock)
        scatterNDLockCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
    else
        scatterNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterND(sd::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const int xRank = indices.rankOf();
    const int yRank = updates.rankOf();
    const int zRank = output.rankOf();

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = ((lock ? output.lengthOf() : updates.lengthOf()) + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(int) * ((yRank > xRank ? yRank : xRank) + zRank) + 256;

    const auto xType = indices.dataType();
    const auto yType = updates.dataType();

    PointersManager manager(context, "scatterND");

    NDArray::prepareSpecialUse({&output}, {&updates, &indices});
    BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), op, indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    NDArray::registerSpecialUse({&output}, {&updates, &indices});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Z>
__global__ void scatterForLossCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                         void *vy, const Nd4jLong *yShapeInfo,
                                         void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
          auto y = reinterpret_cast<Z*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ Nd4jLong xLen;
    __shared__ int xRank, *sharedMem;   // xRank = zRank, yRank = xRank + 1

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        xLen  = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
    }
    __syncthreads();

    const auto xInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(xInd >= xLen)
        return;

    auto coords = sharedMem + threadIdx.x * (xRank + 1);

    shape::index2coords(xInd, xShapeInfo, coords);

    // y last coordinate
    coords[xRank] = x[shape::getOffset(xShapeInfo, coords)];

    const auto yOffset = shape::getOffset(yShapeInfo, coords);

    if(z == nullptr) { // gradient calculation
        y[yOffset] -= 1.f;
    }
    else {
        z[shape::getOffset(zShapeInfo, coords)] = y[yOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Z>
static void scatterForLossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong* xShapeInfo, void *vy, const Nd4jLong* yShapeInfo, void *vz, const Nd4jLong* zShapeInfo) {

    scatterForLossCuda<X, Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterForLoss(sd::LaunchContext* context, const NDArray& indices, NDArray& updates, NDArray& output, const bool calcGrad) {
    // shapes of indices and output must be the same
    // shape of indices should be the same as updates shape with last dimension excluded, for example if updates is {a,b,c} then indices should be {a,b}

    PointersManager manager(context, "scatterForLoss");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (indices.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = updates.rankOf() * sizeof(int) * threadsPerBlock  + 128;

    if(calcGrad) {
        NDArray::prepareSpecialUse({&updates}, {&indices});
        BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), nullptr, nullptr), INDEXING_TYPES, FLOAT_TYPES);
        NDArray::registerSpecialUse({&updates}, {&indices});
    }
    else {
        NDArray::prepareSpecialUse({&output}, {&indices, &updates});
        BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo()), INDEXING_TYPES, FLOAT_TYPES);
        NDArray::registerSpecialUse({&output}, {&indices, &updates});
    }

    manager.synchronize();
}

}
}
}


/*

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void scatterLockCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                    const int opCode,
                                    const void* vx, const Nd4jLong *xShapeInfo,
                                    const void* vy, const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                                          void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets,
                                    const Nd4jLong xLen, const Nd4jLong yTadLen, const Nd4jLong zTadLen) {

    scatterLockCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yTadShapeInfo, yOffsets, vz, zTadShapeInfo, zOffsets, xLen, yTadLen, zTadLen);
}


///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template<typename X, typename Y>
__global__ static void scatterLockCuda(const int opCode,
                                       const void* vx, const Nd4jLong *xShapeInfo,
                                       const void* vy, const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                                             void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets,
                                       const Nd4jLong xLen, const Nd4jLong yTadLen, const Nd4jLong zTadLen) {



 const int xRank = indices.rankOf();

        std::vector<int> zTadDims = ShapeUtils::evalDimsToExclude(output.rankOf(), {0});

        int sizeOfUpdDims = xRank;
        if(output.rankOf() == updates.rankOf() && indices.isVector())
            sizeOfUpdDims = 1;

        std::vector<int> yTadDims(sizeOfUpdDims);
        std::iota(yTadDims.begin(), yTadDims.end(), 0);

        auto packY = sd::ConstantTadHelper::getInstance()->tadForDimensions(updates.shapeInfo(), ShapeUtils::evalDimsToExclude(updates.rankOf(), yTadDims));
        auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output.shapeInfo(), zTadDims);

        const Nd4jLong zTadLen = shape::length(packZ.primaryShapeInfo());
        const Nd4jLong yTadLen = shape::length(packY.primaryShapeInfo());

        const auto threadsPerBlock = sd::math::nd4j_max<int>(32, sd::math::nd4j_min<int>(zTadLen, 1024));
        const auto blocksPerGrid = indices.lengthOf();

        const auto xType = indices.dataType();
        const auto yType = updates.dataType();

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterLockCudaLauncher, (blocksPerGrid, threadsPerBlock, 1024, context->getCudaStream(), op, indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(), packY.specialShapeInfo(), packY.specialOffsets(), output.specialBuffer(), packZ.specialShapeInfo(), packZ.specialOffsets(), indices.lengthOf(), yTadLen, zTadLen), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);



    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ bool vectorCase;
    if(threadIdx.x == 0)
        vectorCase = yTadLen == xLen && shape::rank(xShapeInfo) <= 1;
    __syncthreads();

    for (int e = 0; e < xLen; e++) {

        const Nd4jLong zIndex = x[shape::getIndexOffset(e, xShapeInfo)];
        const bool isOwner = zIndex < gridDim.x ? blockIdx.x == zIndex : blockIdx.x == zIndex % gridDim.x;

        if (!isOwner)
            continue;

        if(vectorCase) { // means z_rank = 1 and might be yTadLen != zTadLen in this case

            if(threadIdx.x != 0)
                continue;

            const auto yOffset = shape::getIndexOffset(e,      yTadShapeInfo);
            const auto zOffset = shape::getIndexOffset(zIndex, zTadShapeInfo);

            switch (opCode) {
                case pairwise::Add:
                    z[zOffset] += y[yOffset];
                    break;
                case pairwise::Subtract:
                    z[zOffset] -= y[yOffset];
                    break;
                case pairwise::Multiply:
                    z[zOffset] *= y[yOffset];
                    break;
                case pairwise::Divide:
                    z[zOffset] /= y[yOffset];
                    break;
                case pairwise::ReverseSubtract:
                    z[zOffset] = y[yOffset] - z[zOffset];
                    break;
                case pairwise::ReverseDivide:
                    z[zOffset] = y[yOffset] / z[zOffset];
                    break;
                case pairwise::CopyPws:
                    z[zOffset] = y[yOffset];
                    break;
                case pairwise::MaxPairwise:
                    if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                case pairwise::MinPairwise:
                    if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
        else {      // yTadLen == zTadLen in this case

            const Y* yTad = y + yOffsets[e];
                  Y* zTad = z + zOffsets[zIndex];

            for (Nd4jLong i = threadIdx.x; i < zTadLen; i += blockDim.x) {

                const auto yOffset = shape::getIndexOffset(i, yTadShapeInfo);
                const auto zOffset = shape::getIndexOffset(i, zTadShapeInfo);

                switch (opCode) {
                    case pairwise::Add:
                        zTad[zOffset] += yTad[yOffset];
                        break;
                    case pairwise::Subtract:
                        zTad[zOffset] -= yTad[yOffset];
                        break;
                    case pairwise::Multiply:
                        zTad[zOffset] *= yTad[yOffset];
                        break;
                    case pairwise::Divide:
                        zTad[zOffset] /= yTad[yOffset];
                        break;
                    case pairwise::ReverseSubtract:
                        zTad[zOffset] = yTad[yOffset] - zTad[zOffset];
                        break;
                    case pairwise::ReverseDivide:
                        zTad[zOffset] = yTad[yOffset] / zTad[zOffset];
                        break;
                    case pairwise::CopyPws:
                        zTad[zOffset] = yTad[yOffset];
                        break;
                    case pairwise::MaxPairwise:
                        if(zTad[zOffset] < yTad[yOffset]) zTad[zOffset] = yTad[yOffset];
                        break;
                    case pairwise::MinPairwise:
                        if(zTad[zOffset] > yTad[yOffset]) zTad[zOffset] = yTad[yOffset];
                        break;
                    default:
                        continue;
                }
            }
        }
    }
}

            template<typename T, bool locking>
            __global__ static void scatterCuda(const int opCode, const int numOfSubArrs,
                                                     void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
                                                     void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
                                                     const int* indexes, unsigned int arrLenX, unsigned int arrLenY) {

                __shared__ T *x, *y;

                if (locking) {

                    for (int e = 0; e < numOfSubArrs; e++) {

                        const auto xIndex = indexes[e];
                        const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

                        if (!isOwner)
                            continue;

                        if (threadIdx.x == 0) {
                            x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
                            y = reinterpret_cast<T *>(vy) + yOffsets[e];
                        }
                        __syncthreads();

                        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

                            const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
                            const auto yOffset = shape::getIndexOffset(i, yShapeInfo);

                            switch (opCode) {
                                case pairwise::Add:
                                    x[xOffset] += y[yOffset];
                                    break;
                                case pairwise::Subtract:
                                    x[xOffset] -= y[yOffset];
                                    break;
                                case pairwise::Multiply:
                                    x[xOffset] *= y[yOffset];
                                    break;
                                case pairwise::Divide:
                                    x[xOffset] /= y[yOffset];
                                    break;
                                case pairwise::ReverseSubtract:
                                    x[xOffset] = y[yOffset] - x[xOffset];
                                    break;
                                case pairwise::ReverseDivide:
                                    x[xOffset] = y[yOffset] / x[xOffset];
                                    break;
                                case pairwise::CopyPws:
                                    x[xOffset] = y[yOffset];
                                    break;
                                default:
                                    continue;
                            }
                        }
                        __syncthreads();
                    }
                } else {
                    for (int e = blockIdx.x; e < numOfSubArrs; e+= gridDim.x) {

                        if (threadIdx.x == 0) {
                            const auto xIndex = indexes[e];
                            x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
                            y = reinterpret_cast<T *>(vy) + yOffsets[e];
                        }
                        __syncthreads();

                        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {
                            const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
                            const auto yOffset = shape::getIndexOffset(i, yShapeInfo);

                            switch (opCode) {
                                case pairwise::Add:
                                    x[xOffset] += y[yOffset];
                                    break;
                                case pairwise::Subtract:
                                    x[xOffset] -= y[yOffset];
                                    break;
                                case pairwise::Multiply:
                                    x[xOffset] *= y[yOffset];
                                    break;
                                case pairwise::Divide:
                                    x[xOffset] /= y[yOffset];
                                    break;
                                case pairwise::ReverseSubtract:
                                    x[xOffset] = y[yOffset] - x[xOffset];
                                    break;
                                case pairwise::ReverseDivide:
                                    x[xOffset] = y[yOffset] / x[xOffset];
                                    break;
                                case pairwise::CopyPws:
                                    x[xOffset] = y[yOffset];
                                    break;
                                default:
                                    continue;
                            }
                        }
                        __syncthreads();
                    }
                }
            }


            template <typename T>
            void scatter_(sd::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {
                std::vector<int> dims = {0};
                auto inverted = ShapeUtils::evalDimsToExclude(output.rankOf(), dims);

                auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(output.shapeInfo(), inverted);
                auto packY = sd::ConstantTadHelper::getInstance()->tadForDimensions(updates.shapeInfo(), inverted);

                auto psX = packX.specialShapeInfo();
                auto psY = packY.specialShapeInfo();

                PointersManager manager(context, "scatter");

                auto poX = packX.specialOffsets();
                auto poY = packY.specialOffsets();

                NDArray::prepareSpecialUse({&output}, {&updates, &indices});

                unsigned int tadLengthX = shape::length(packX.primaryShapeInfo());
                unsigned int tadLengthY = shape::length(packY.primaryShapeInfo());
                if (tadLengthX != tadLengthY)
                    throw std::runtime_error("scatter: Lengths of TADs must be equal");

                auto blockSize = sd::math::nd4j_max<int>(32, sd::math::nd4j_min<int>(tadLengthX, 1024));

                if (lock)
                    scatterCuda<T, true><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.specialBuffer(), psX, poX, updates.specialBuffer(), psY, poY, reinterpret_cast<int *>(indices.specialBuffer()), tadLengthX, tadLengthY);
                else
                    scatterCuda<T, false><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.specialBuffer(), psX, poX, updates.specialBuffer(), psY, poY, reinterpret_cast<int *>(indices.specialBuffer()), tadLengthX, tadLengthY);

                 NDArray::registerSpecialUse({&output}, {&updates, &indices});
                manager.synchronize();
            }



///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template<typename X, typename Y>
__global__ static void scatterNDLockCuda(const int opCode,
                                         const void* vx, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                         const void* vy, const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                                               void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets,
                                         const Nd4jLong *zShapeInfo,
                                         const Nd4jLong numOfXTads, const Nd4jLong numOfZTads, const Nd4jLong yTadLen) {



---------------------------------------------------------------------------
const int xLastDim = indices.sizeAt(-1);

        // y_tad and z_tad have the same shape
        std::vector<int> yTadDims(zRank - xLastDim), zTadDims(zRank - xLastDim);
        for (int j = 0, i = zTadDims.size() - 1; i >=0 ; --i, ++j) {
            yTadDims[i] = yRank - 1 - j;
            zTadDims[i] = zRank - 1 - j;
        }

        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(indices.shapeInfo(), {xRank - 1});
        auto packY = sd::ConstantTadHelper::getInstance()->tadForDimensions(updates.shapeInfo(), yTadDims);
        auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output.shapeInfo(), zTadDims);

        const int threadsPerBlock = MAX_NUM_THREADS / 4;
        const int blocksPerGrid = packZ.numberOfTads();
        const int sharedMem = 8 * threadsPerBlock * xLastDim + 128;
---------------------------------------------------------------------------

    // zTadLen == yTadLen if numOfZTads > 1, in opposite case z and y are vectors
    // numOfXTads == numOfYTads if numOfZTads > 1, in opposite case z and y are vectors

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ Nd4jLong *zTadCoords;
    __shared__ int xLastDim;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        zTadCoords = reinterpret_cast<Nd4jLong*>(shmem);
        xLastDim = xTadShapeInfo[1];   // xTad has rank = 1 always
    }
    __syncthreads();

    Nd4jLong* zTadCoordsPerThread = zTadCoords + threadIdx.x * xLastDim;

    for (Nd4jLong i = 0; i < numOfXTads; ++i) {

        const X* xTad = x + xOffsets[i];

        for (uint k = 0; k < xLastDim; ++k)
            zTadCoordsPerThread[k] = xTad[shape::getIndexOffset(k, xTadShapeInfo)];

        const auto zTadIndex = shape::coords2index(xLastDim, zShapeInfo + 1, zTadCoordsPerThread);

        const bool isOwner = zTadIndex < gridDim.x ? blockIdx.x == zTadIndex : blockIdx.x == zTadIndex % gridDim.x;

        if(!isOwner)
            continue;

        if(numOfZTads == 1) {     // yTadLen == numOfXTads in this case

            if(threadIdx.x != 0)
                continue;

            const auto yOffset = shape::getIndexOffset(i,         yTadShapeInfo);
            const auto zOffset = shape::getIndexOffset(zTadIndex, zTadShapeInfo);

            switch (opCode) {
                case pairwise::Add:
                    z[zOffset] += y[yOffset];
                    break;
                case pairwise::Subtract:
                    z[zOffset] -= y[yOffset];
                    break;
                case pairwise::Multiply:
                    z[zOffset] *= y[yOffset];
                    break;
                case pairwise::Divide:
                    z[zOffset] /= y[yOffset];
                    break;
                case pairwise::ReverseSubtract:
                    z[zOffset] = y[yOffset] - z[zOffset];
                    break;
                case pairwise::ReverseDivide:
                    z[zOffset] = y[yOffset] / z[zOffset];
                    break;
                case pairwise::CopyPws:
                    z[zOffset] = y[yOffset];
                    break;
                case pairwise::MaxPairwise:
                    if(z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                case pairwise::MinPairwise:
                    if(z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
        else {
            const auto yTad = y + yOffsets[i];
            const auto zTad = z + zOffsets[zTadIndex];

            for (Nd4jLong j = threadIdx.x; j < yTadLen; j += blockDim.x) {

                const auto yOffset = shape::getIndexOffset(j, yTadShapeInfo);
                const auto zOffset = shape::getIndexOffset(j, zTadShapeInfo);

                switch (opCode) {
                    case pairwise::Add:
                        zTad[zOffset] += yTad[yOffset];
                        break;
                    case pairwise::Subtract:
                        zTad[zOffset] -= yTad[yOffset];
                        break;
                    case pairwise::Multiply:
                        zTad[zOffset] *= yTad[yOffset];
                        break;
                    case pairwise::Divide:
                        zTad[zOffset] /= yTad[yOffset];
                        break;
                    case pairwise::ReverseSubtract:
                        zTad[zOffset] = yTad[yOffset] - zTad[zOffset];
                        break;
                    case pairwise::ReverseDivide:
                        zTad[zOffset] = yTad[yOffset] / zTad[zOffset];
                        break;
                    case pairwise::CopyPws:
                        zTad[zOffset] = yTad[yOffset];
                        break;
                    case pairwise::MaxPairwise:
                        if(zTad[zOffset] < yTad[yOffset]) zTad[zOffset] = yTad[yOffset];
                        break;
                    case pairwise::MinPairwise:
                        if(zTad[zOffset] > yTad[yOffset]) zTad[zOffset] = yTad[yOffset];
                        break;
                    default:
                        continue;
                }
            }
        }
    }
}

*/
        // PointersManager manager(&context, "NativeOps::concat");
        // PointersManager::printDevContentOnDev<int>(vx, 2);
        // PointersManager::printDevContentOnDev<Nd4jLong>(xShapeInfo, 8);
        // PointersManager::printDevContentOnDev<float>(vy, 8);
        // PointersManager::printDevContentOnDev<Nd4jLong>(yShapeInfo, 8);
        // PointersManager::printDevContentOnDev<Nd4jLong>(zShapeInfo, 8);

        // manager.printDevContentOnHost<int>(indices.specialBuffer(), indices.lengthOf());
        // manager.printDevContentOnHost<Nd4jLong>(indices.specialShapeInfo(), shape::shapeInfoLength(indices.rankOf()));
        // manager.printDevContentOnHost<float>(updates.specialBuffer(), updates.lengthOf());
        // manager.printDevContentOnHost<Nd4jLong>(updates.specialShapeInfo(), shape::shapeInfoLength(updates.rankOf()));
        // manager.printDevContentOnHost<Nd4jLong>(output.specialShapeInfo(), shape::shapeInfoLength(output.rankOf()));
        // printf("!!!!!!!\n");
        // manager.printDevContentOnHost<Nd4jLong>(packX.specialShapeInfo(), 2*shape::rank(packX.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packX.specialOffsets(), packX.numberOfTads());
        // manager.printDevContentOnHost<Nd4jLong>(packY.specialShapeInfo(), 2*shape::rank(packY.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packY.specialOffsets(), packY.numberOfTads());
        // manager.printDevContentOnHost<Nd4jLong>(packZ.specialShapeInfo(), 2*shape::rank(packZ.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packZ.specialOffsets(), packZ.numberOfTads());
        // printf("dddddddd\n");
        // shape::printShapeInfoLinear(packY.primaryShapeInfo());