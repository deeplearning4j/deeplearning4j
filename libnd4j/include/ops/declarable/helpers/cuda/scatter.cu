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
#include <TAD.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

            // template<typename T, bool locking>
            // __global__ static void scatterCuda(const int opCode, const int numOfSubArrs,
            //                                          void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
            //                                          void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
            //                                          const int* indexes, unsigned int arrLenX, unsigned int arrLenY) {

            //     __shared__ T *x, *y;

            //     if (locking) {

            //         for (int e = 0; e < numOfSubArrs; e++) {

            //             const auto xIndex = indexes[e];
            //             const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

            //             if (!isOwner)
            //                 continue;

            //             if (threadIdx.x == 0) {
            //                 x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
            //                 y = reinterpret_cast<T *>(vy) + yOffsets[e];
            //             }
            //             __syncthreads();

            //             for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

            //                 const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
            //                 const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

            //                 switch (opCode) {
            //                     case pairwise::Add:
            //                         x[xOffset] += y[yOffset];
            //                         break;
            //                     case pairwise::Subtract:
            //                         x[xOffset] -= y[yOffset];
            //                         break;
            //                     case pairwise::Multiply:
            //                         x[xOffset] *= y[yOffset];
            //                         break;
            //                     case pairwise::Divide:
            //                         x[xOffset] /= y[yOffset];
            //                         break;
            //                     case pairwise::ReverseSubtract:
            //                         x[xOffset] = y[yOffset] - x[xOffset];
            //                         break;
            //                     case pairwise::ReverseDivide:
            //                         x[xOffset] = y[yOffset] / x[xOffset];
            //                         break;
            //                     case pairwise::CopyPws:
            //                         x[xOffset] = y[yOffset];
            //                         break;
            //                     default:
            //                         continue;
            //                 }
            //             }
            //             __syncthreads();
            //         }
            //     } else {
            //         for (int e = blockIdx.x; e < numOfSubArrs; e+= gridDim.x) {

            //             if (threadIdx.x == 0) {
            //                 const auto xIndex = indexes[e];
            //                 x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
            //                 y = reinterpret_cast<T *>(vy) + yOffsets[e];
            //             }
            //             __syncthreads();

            //             for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {
            //                 const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
            //                 const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

            //                 switch (opCode) {
            //                     case pairwise::Add:
            //                         x[xOffset] += y[yOffset];
            //                         break;
            //                     case pairwise::Subtract:
            //                         x[xOffset] -= y[yOffset];
            //                         break;
            //                     case pairwise::Multiply:
            //                         x[xOffset] *= y[yOffset];
            //                         break;
            //                     case pairwise::Divide:
            //                         x[xOffset] /= y[yOffset];
            //                         break;
            //                     case pairwise::ReverseSubtract:
            //                         x[xOffset] = y[yOffset] - x[xOffset];
            //                         break;
            //                     case pairwise::ReverseDivide:
            //                         x[xOffset] = y[yOffset] / x[xOffset];
            //                         break;
            //                     case pairwise::CopyPws:
            //                         x[xOffset] = y[yOffset];
            //                         break;
            //                     default:
            //                         continue;
            //                 }
            //             }
            //             __syncthreads();
            //         }
            //     }
            // }


            // template <typename T>
            // void scatter_(nd4j::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {
            //     std::vector<int> dims = {0};
            //     auto inverted = ShapeUtils::evalDimsToExclude(output.rankOf(), dims);

            //     auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), inverted);
            //     auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(updates.getShapeInfo(), inverted);

            //     auto psX = packX.specialShapeInfo();
            //     auto psY = packY.specialShapeInfo();

            //     PointersManager manager(context, "scatter");

            //     auto poX = packX.specialOffsets();
            //     auto poY = packY.specialOffsets();

            //     NDArray::prepareSpecialUse({&output}, {&updates, &indices});

            //     unsigned int tadLengthX = shape::length(packX.primaryShapeInfo());
            //     unsigned int tadLengthY = shape::length(packY.primaryShapeInfo());
            //     if (tadLengthX != tadLengthY)
            //         throw std::runtime_error("scatter: Lengths of TADs must be equal");

            //     auto blockSize = nd4j::math::nd4j_max<int>(32, nd4j::math::nd4j_min<int>(tadLengthX, 1024));

            //     if (lock)
            //         scatterCuda<T, true><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.getSpecialBuffer(), psX, poX, updates.getSpecialBuffer(), psY, poY, reinterpret_cast<int *>(indices.getSpecialBuffer()), tadLengthX, tadLengthY);
            //     else
            //         scatterCuda<T, false><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.getSpecialBuffer(), psX, poX, updates.getSpecialBuffer(), psY, poY, reinterpret_cast<int *>(indices.getSpecialBuffer()), tadLengthX, tadLengthY);

            //      NDArray::registerSpecialUse({&output}, {&updates, &indices});
            //     manager.synchronize();
            // }

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template<typename X, typename Y>
__global__ static void scatterLockCuda(const int opCode,
                                       const void* vx, const Nd4jLong *xShapeInfo,
                                       const void* vy, const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                                             void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets,
                                       const Nd4jLong xLen, const Nd4jLong yTadLen, const Nd4jLong zTadLen) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ bool vectorCase;
    if(threadIdx.x == 0)
        vectorCase = yTadLen == xLen && shape::rank(xShapeInfo) == 1;
    __syncthreads();

    for (int e = 0; e < xLen; e++) {

        const Nd4jLong zIndex = x[shape::getIndexOffset(e, xShapeInfo, xLen)];
        const bool isOwner = zIndex < gridDim.x ? blockIdx.x == zIndex : blockIdx.x == zIndex % gridDim.x;

        if (!isOwner)
            continue;

        if(vectorCase) { // means z_rank = 1 and might be yTadLen != zTadLen in this case

            if(threadIdx.x != 0)
                continue;

            const auto yOffset = shape::getIndexOffset(e,      yTadShapeInfo, yTadLen);
            const auto zOffset = shape::getIndexOffset(zIndex, zTadShapeInfo, zTadLen);

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

                const auto yOffset = shape::getIndexOffset(i, yTadShapeInfo, zTadLen);
                const auto zOffset = shape::getIndexOffset(i, zTadShapeInfo, zTadLen);

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
__global__ static void scatterCuda(const int opCode,
                                   const void *vx, const Nd4jLong *xShapeInfo,
                                   const void *vy, const Nd4jLong *yShapeInfo,
                                         void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Y*>(vz);

    __shared__ int xRank, yRank, zRank;
    __shared__ Nd4jLong yLen, totalThreads, *coord;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coord = reinterpret_cast<Nd4jLong*>(shmem);
        yLen = shape::length(yShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);
    }
    __syncthreads();

    auto xCoord = coord + threadIdx.x * (xRank + yRank + zRank);
    auto yCoord = xCoord + xRank;
    auto zCoord = yCoord + yRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < yLen; i += totalThreads) {

        shape::index2coords(yRank, shape::shapeOf(const_cast<Nd4jLong*>(yShapeInfo)), i, yLen, yCoord);

        for (uint j = 0; j < xRank; ++j)
            xCoord[j] = yCoord[j];

        const auto xOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), xCoord, xRank);
        zCoord[0] = x[xOffset];

        for (uint j = 0; j < yRank - xRank; ++j)
            zCoord[j + 1] = yCoord[xRank + j];

        const auto yOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(yShapeInfo)), shape::stride(const_cast<Nd4jLong*>(yShapeInfo)), yCoord, yRank);
        const auto zOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), shape::stride(const_cast<Nd4jLong*>(zShapeInfo)), zCoord, zRank);

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
                                      void *vz, const Nd4jLong *zShapeInfo) {

    scatterCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}


///////////////////////////////////////////////////////////////////
void scatter(nd4j::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    PointersManager manager(context, "scatter");

    NDArray::prepareSpecialUse({&output}, {&updates, &indices});

    if(lock) {

        const int xRank = indices.rankOf();

        std::vector<int> zTadDims = ShapeUtils::evalDimsToExclude(output.rankOf(), {0});
        std::vector<int> yTadDims(xRank);
        std::iota(yTadDims.begin(), yTadDims.end(), xRank == 1 ? 0 : xRank);

        auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(updates.getShapeInfo(), yTadDims);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), zTadDims);

        const Nd4jLong zTadLen = shape::length(packZ.primaryShapeInfo());
        const Nd4jLong yTadLen = shape::length(packY.primaryShapeInfo());

        const auto threadsPerBlock = nd4j::math::nd4j_max<int>(32, nd4j::math::nd4j_min<int>(zTadLen, 1024));
        const auto blocksPerGrid = indices.lengthOf();

        const auto xType = indices.dataType();
        const auto yType = updates.dataType();

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterLockCudaLauncher, (blocksPerGrid, threadsPerBlock, 1024, context->getCudaStream(), op, indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), updates.getSpecialBuffer(), packY.specialShapeInfo(), packY.specialOffsets(), output.getSpecialBuffer(), packZ.specialShapeInfo(), packZ.specialOffsets(), indices.lengthOf(), yTadLen, zTadLen), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    }
    else {

        const int threadsPerBlock = MAX_NUM_THREADS / 8;
        const int blocksPerGrid = (updates.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = 8 * threadsPerBlock * (indices.rankOf() + updates.rankOf() + output.rankOf()) + 128;

        const auto xType = indices.dataType();
        const auto yType = updates.dataType();

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), op, indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), updates.getSpecialBuffer(), updates.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    }

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
            zTadCoordsPerThread[k] = xTad[shape::getIndexOffset(k, xTadShapeInfo, xLastDim)];

        const auto zTadIndex = shape::coords2index(xLastDim, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), zTadCoordsPerThread);

        const bool isOwner = zTadIndex < gridDim.x ? blockIdx.x == zTadIndex : blockIdx.x == zTadIndex % gridDim.x;

        if(!isOwner)
            continue;

        if(numOfZTads == 1) {     // yTadLen == numOfXTads in this case

            if(threadIdx.x != 0)
                continue;

            const auto yOffset = shape::getIndexOffset(i,         yTadShapeInfo, yTadLen);
            const auto zOffset = shape::getIndexOffset(zTadIndex, zTadShapeInfo, yTadLen);

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

                const auto yOffset = shape::getIndexOffset(j, yTadShapeInfo, yTadLen);
                const auto zOffset = shape::getIndexOffset(j, zTadShapeInfo, yTadLen);

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

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void scatterNDLockCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                      const int opCode,
                                      const void* vx, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                      const void* vy, const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                                            void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets,
                                      const Nd4jLong *zShapeInfo,
                                      const Nd4jLong numOfXTads, const Nd4jLong numOfZTads, const Nd4jLong zTadLen) {

    scatterNDLockCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode,
                                                                                   vx, xTadShapeInfo, xOffsets,
                                                                                   vy, yTadShapeInfo, yOffsets,
                                                                                   vz, zTadShapeInfo, zOffsets,
                                                                                   zShapeInfo,
                                                                                   numOfXTads, numOfZTads, zTadLen);
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

    __shared__ int xRank, yRank, zRank, xLastDim;
    __shared__ Nd4jLong yLen, totalThreads, *coord;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        coord = reinterpret_cast<Nd4jLong*>(shmem);
        yLen = shape::length(yShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);
        xLastDim = xShapeInfo[xRank];
    }
    __syncthreads();

    auto xCoord = coord + threadIdx.x * (xRank + yRank + zRank);
    auto yCoord = xCoord + xRank;
    auto zCoord = yCoord + yRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < yLen; i += totalThreads) {

        shape::index2coords(yRank, shape::shapeOf(const_cast<Nd4jLong*>(yShapeInfo)), i, yLen, yCoord);

        for (uint j = 0; j < xRank - 1; ++j)
            xCoord[j] = yCoord[j];

        for (uint j = 0; j < xLastDim; ++j) {
            xCoord[xRank - 1] = j;
            const auto xOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), xCoord, xRank);
            zCoord[j] = x[xOffset];
        }

        for (uint j = xLastDim; j < zRank; ++j)
            zCoord[j] = yCoord[yRank - zRank + j];

        const auto yOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(yShapeInfo)), shape::stride(const_cast<Nd4jLong*>(yShapeInfo)), yCoord, yRank);
        const auto zOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), shape::stride(const_cast<Nd4jLong*>(zShapeInfo)), zCoord, zRank);

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
                                        void *vz, const Nd4jLong *zShapeInfo) {

    scatterNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterND(nd4j::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const int xRank = indices.rankOf();
    const int yRank = updates.rankOf();
    const int zRank = output.rankOf();

    PointersManager manager(context, "scatterND");

    NDArray::prepareSpecialUse({&output}, {&updates, &indices});

    if(lock) {

        const int xLastDim = indices.sizeAt(-1);

        // y_tad and z_tad have the same shape
        std::vector<int> yTadDims(zRank - xLastDim), zTadDims(zRank - xLastDim);
        for (int j = 0, i = zTadDims.size() - 1; i >=0 ; --i, ++j) {
            yTadDims[i] = yRank - 1 - j;
            zTadDims[i] = zRank - 1 - j;
        }

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(indices.getShapeInfo(), {xRank - 1});
        auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(updates.getShapeInfo(), yTadDims);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), zTadDims);

        const int threadsPerBlock = MAX_NUM_THREADS / 4;
        const int blocksPerGrid = packZ.numberOfTads();
        const int sharedMem = 8 * threadsPerBlock * xLastDim + 128;

        const auto xType = indices.dataType();
        const auto yType = updates.dataType();

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDLockCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), op, indices.getSpecialBuffer(), packX.specialShapeInfo(), packX.specialOffsets(), updates.getSpecialBuffer(), packY.specialShapeInfo(), packY.specialOffsets(), output.getSpecialBuffer(), packZ.specialShapeInfo(), packZ.specialOffsets(), output.getSpecialShapeInfo(), packX.numberOfTads(), packZ.numberOfTads(), shape::length(packY.primaryShapeInfo())), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    }
    else {

        const int threadsPerBlock = MAX_NUM_THREADS / 8;
        const int blocksPerGrid = (updates.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = 8 * threadsPerBlock * (xRank + yRank + zRank) + 128;

        const auto xType = indices.dataType();
        const auto yType = updates.dataType();

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), op, indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), updates.getSpecialBuffer(), updates.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), INDEXING_TYPES, GENERIC_NUMERIC_TYPES);
    }

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

    __shared__ Nd4jLong xLen, *sharedMem;
    __shared__ int xRank;   // xRank = zRank, yRank = xRank + 1

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xLen  = shape::length(xShapeInfo);
        xRank = shape::rank(xShapeInfo);
    }
    __syncthreads();

    const auto xInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(xInd >= xLen)
        return;

    auto coords = sharedMem + threadIdx.x * (xRank + 1);

    shape::index2coords(xRank, xShapeInfo + 1, xInd, xLen, coords);

    // y last coordinate
    coords[xRank] = x[shape::getOffset(0, xShapeInfo + 1, xShapeInfo + xRank + 1, coords, xRank)];

    const auto yOffset = shape::getOffset(0, yShapeInfo + 1, yShapeInfo + xRank + 2, coords, xRank + 1);

    if(z == nullptr) { // gradient calculation
        y[yOffset] -= 1.f;
    }
    else {
        z[shape::getOffset(0, zShapeInfo + 1, zShapeInfo + xRank + 1, coords, xRank)] = y[yOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Z>
static void scatterForLossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong* xShapeInfo, void *vy, const Nd4jLong* yShapeInfo, void *vz, const Nd4jLong* zShapeInfo) {

    scatterForLossCuda<X, Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterForLoss(nd4j::LaunchContext* context, const NDArray& indices, NDArray& updates, NDArray& output, const bool calcGrad) {
    // shapes of indices and output must be the same
    // shape of indices should be the same as updates shape with last dimension excluded, for example if updates is {a,b,c} then indices should be {a,b}

    PointersManager manager(context, "scatterForLoss");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (indices.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = updates.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    if(calcGrad) {
        NDArray::prepareSpecialUse({&updates}, {&indices});
        BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), nullptr, nullptr), INDEXING_TYPES, FLOAT_TYPES);
        NDArray::registerSpecialUse({&updates}, {&indices});
    }
    else {
        NDArray::prepareSpecialUse({&output}, {&indices, &updates});
        BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), updates.getSpecialBuffer(), updates.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo()), INDEXING_TYPES, FLOAT_TYPES);
        NDArray::registerSpecialUse({&output}, {&indices, &updates});
    }

    manager.synchronize();
}

}
}
}

        // PointersManager manager(&context, "NativeOps::concat");
        // PointersManager::printDevContentOnDev<int>(vx, 2);
        // PointersManager::printDevContentOnDev<Nd4jLong>(xShapeInfo, 8);
        // PointersManager::printDevContentOnDev<float>(vy, 8);
        // PointersManager::printDevContentOnDev<Nd4jLong>(yShapeInfo, 8);
        // PointersManager::printDevContentOnDev<Nd4jLong>(zShapeInfo, 8);

        // manager.printDevContentOnHost<int>(indices.getSpecialBuffer(), indices.lengthOf());
        // manager.printDevContentOnHost<Nd4jLong>(indices.getSpecialShapeInfo(), shape::shapeInfoLength(indices.rankOf()));
        // manager.printDevContentOnHost<float>(updates.getSpecialBuffer(), updates.lengthOf());
        // manager.printDevContentOnHost<Nd4jLong>(updates.getSpecialShapeInfo(), shape::shapeInfoLength(updates.rankOf()));
        // manager.printDevContentOnHost<Nd4jLong>(output.getSpecialShapeInfo(), shape::shapeInfoLength(output.rankOf()));
        // printf("!!!!!!!\n");
        // manager.printDevContentOnHost<Nd4jLong>(packX.specialShapeInfo(), 2*shape::rank(packX.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packX.specialOffsets(), packX.numberOfTads());
        // manager.printDevContentOnHost<Nd4jLong>(packY.specialShapeInfo(), 2*shape::rank(packY.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packY.specialOffsets(), packY.numberOfTads());
        // manager.printDevContentOnHost<Nd4jLong>(packZ.specialShapeInfo(), 2*shape::rank(packZ.primaryShapeInfo()) + 4);
        // manager.printDevContentOnHost<Nd4jLong>(packZ.specialOffsets(), packZ.numberOfTads());
        // printf("dddddddd\n");
        // shape::printShapeInfoLinear(packY.primaryShapeInfo());