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
//

// #include <exceptions/cuda_exception.h>
#include <TrueBroadcastHelper.h>
#include <PointersManager.h>
#include <execution/LaunchContext.h>
#include <specials.h>
#include <logger.h>
#include <ops/ops.h>
// #include <cuda_runtime.h>
// #include <cuda.h>

using namespace simdOps;

namespace nd4j    {
namespace helpers {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
__global__ static void trueBroadcastCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int xRank, yRank, zRank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto xCoords = sharedMem + threadIdx.x * (xRank + yRank + zRank);
    auto yCoords = xCoords + xRank;
    auto zCoords = yCoords + yRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, zCoords);

        for(int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

            if(ix >= 0)
                if(xShapeInfo[ix + 1] == zShapeInfo[iz + 1])
                    xCoords[ix--] = zCoords[iz];
                else
                    xCoords[ix--] = 0;

            if(iy >= 0)
                if(yShapeInfo[iy + 1] == zShapeInfo[iz + 1])
                    yCoords[iy--] = zCoords[iz];
                else
                    yCoords[iy--] = 0;
        }

        const auto xOffset = shape::getOffset(xShapeInfo, xCoords);
        const auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        const auto yOffset = shape::getOffset(yShapeInfo, yCoords);

        z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template <typename OpType>
void TrueBroadcastHelper<X,Y,Z>::execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    trueBroadcastCuda<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void TrueBroadcastHelper<X,Y,Z>::exec(const nd4j::broadcast::Ops opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {

    dim3 launchDims;

    launchDims.x = MAX_NUM_THREADS / 8;   // threadsPerBlock
    launchDims.y = (zArr.lengthOf() + launchDims.x - 1) / launchDims.x;  // blocksPerGrid
    launchDims.z = sizeof(Nd4jLong) * launchDims.x * (xArr.rankOf() + yArr.rankOf() + zArr.rankOf()) + 128; // sharedMem

    PointersManager manager(xArr.getContext(), "TrueBroadcastHelper<X,Y,Z>::exec");

    NDArray::prepareSpecialUse({&zArr}, {&xArr, &yArr});

    DISPATCH_BY_OPNUM_TTT(execLauncher, PARAMS(launchDims, xArr.getContext()->getCudaStream(), xArr.getSpecialBuffer(), xArr.getSpecialShapeInfo(), yArr.getSpecialBuffer(), yArr.getSpecialShapeInfo(), zArr.specialBuffer(), zArr.specialShapeInfo()), OPS_A(BROADCAST_OPS));

    NDArray::registerSpecialUse({&zArr}, {&xArr, &yArr});

    manager.synchronize();
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__global__ static void trueBroadcastBoolCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const X*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int xRank, yRank, zRank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto xCoords = sharedMem + threadIdx.x * (xRank + yRank + zRank);
    auto yCoords = xCoords + xRank;
    auto zCoords = yCoords + yRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, zCoords);

        for(int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

            if(ix >= 0)
                if(xShapeInfo[ix + 1] == zShapeInfo[iz + 1])
                    xCoords[ix--] = zCoords[iz];
                else
                    xCoords[ix--] = 0;

            if(iy >= 0)
                if(yShapeInfo[iy + 1] == zShapeInfo[iz + 1])
                    yCoords[iy--] = zCoords[iz];
                else
                    yCoords[iy--] = 0;
        }

        const auto xOffset = shape::getOffset(xShapeInfo, xCoords);
        const auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        const auto yOffset = shape::getOffset(yShapeInfo, yCoords);

        z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
template <typename OpType>
void TrueBroadcastBoolHelper<X,Z>::execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    trueBroadcastBoolCuda<X,Z,OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
void TrueBroadcastBoolHelper<X,Y>::exec(const nd4j::broadcast::BoolOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {

    dim3 launchDims;
    launchDims.x = MAX_NUM_THREADS / 8;   // threadsPerBlock
    launchDims.y = (zArr.lengthOf() + launchDims.x - 1) / launchDims.x;  // blocksPerGrid
    launchDims.z = sizeof(Nd4jLong) * launchDims.x * (xArr.rankOf() + yArr.rankOf() + zArr.rankOf()) + 128; // sharedMem

    PointersManager manager(xArr.getContext(), "TrueBroadcastBoolHelper<X,Y>::exec");

    NDArray::prepareSpecialUse({&zArr}, {&xArr, &yArr});

    DISPATCH_BY_OPNUM_TT(execLauncher, PARAMS(launchDims, xArr.getContext()->getCudaStream(), xArr.getSpecialBuffer(), xArr.getSpecialShapeInfo(), yArr.getSpecialBuffer(), yArr.getSpecialShapeInfo(), zArr.specialBuffer(), zArr.specialShapeInfo()), OPS_A(BROADCAST_BOOL_OPS));

    NDArray::registerSpecialUse({&zArr}, {&xArr, &yArr});

    manager.synchronize();
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ static void trueBroadcastIntCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const X*>(vy);
          auto z = reinterpret_cast<X*>(vz);

    __shared__ int xRank, yRank, zRank;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xRank = shape::rank(xShapeInfo);
        yRank = shape::rank(yShapeInfo);
        zRank = shape::rank(zShapeInfo);

        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }
    __syncthreads();

    auto xCoords = sharedMem + threadIdx.x * (xRank + yRank + zRank);
    auto yCoords = xCoords + xRank;
    auto zCoords = yCoords + yRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, zCoords);

        for(int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

            if(ix >= 0)
                if(xShapeInfo[ix + 1] == zShapeInfo[iz + 1])
                    xCoords[ix--] = zCoords[iz];
                else
                    xCoords[ix--] = 0;

            if(iy >= 0)
                if(yShapeInfo[iy + 1] == zShapeInfo[iz + 1])
                    yCoords[iy--] = zCoords[iz];
                else
                    yCoords[iy--] = 0;
        }

        const auto xOffset = shape::getOffset(xShapeInfo, xCoords);
        const auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        const auto yOffset = shape::getOffset(yShapeInfo, yCoords);

        z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X>
template <typename OpType>
void TrueBroadcastIntHelper<X>::execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    trueBroadcastIntCuda<X,OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
template<typename X>
void TrueBroadcastIntHelper<X>::exec(const nd4j::broadcast::IntOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {

    dim3 launchDims;
    launchDims.x = MAX_NUM_THREADS / 8;   // threadsPerBlock
    launchDims.y = (zArr.lengthOf() + launchDims.x - 1) / launchDims.x;  // blocksPerGrid
    launchDims.z = sizeof(Nd4jLong) * launchDims.x * (xArr.rankOf() + yArr.rankOf() + zArr.rankOf()) + 128; // sharedMem

    PointersManager manager(xArr.getContext(), "TrueBroadcastIntHelper<X>::exec");

    NDArray::prepareSpecialUse({&zArr}, {&xArr, &yArr});

    DISPATCH_BY_OPNUM_T(execLauncher, PARAMS(launchDims, xArr.getContext()->getCudaStream(), xArr.getSpecialBuffer(), xArr.getSpecialShapeInfo(), yArr.getSpecialBuffer(), yArr.getSpecialShapeInfo(), zArr.specialBuffer(), zArr.specialShapeInfo()), OPS_A(BROADCAST_INT_OPS));

    NDArray::registerSpecialUse({&zArr}, {&xArr, &yArr});

    manager.synchronize();
}



BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_0);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_1);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_2);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_3);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_4);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_5);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_6);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_7);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_8);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_9);

BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastBoolHelper, , LIBND4J_TYPES, BOOL_TYPES);

BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastIntHelper, , INTEGER_TYPES);

}
}