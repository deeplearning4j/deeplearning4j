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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void concatCuda(const int numOfArrs, void* pVx,  void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

    __shared__ int arrIdx, blocksPerArr;
    __shared__ T *x, *z;
    __shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLen, arrLenPerBlock, start, end;

    if (threadIdx.x == 0) {

        blocksPerArr = (gridDim.x + numOfArrs - 1) / numOfArrs;     // ceil
        arrIdx = blockIdx.x / blocksPerArr;

        x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[arrIdx]);
        z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[arrIdx]);
        xShapeInfo = reinterpret_cast<Nd4jLong**>(pxShapeInfo)[arrIdx];
        zShapeInfo = reinterpret_cast<Nd4jLong**>(pzShapeInfo)[arrIdx];
        arrLen = shape::length(xShapeInfo);

        arrLenPerBlock = (arrLen + blocksPerArr - 1) / blocksPerArr;  // ceil

        start = (blockIdx.x % blocksPerArr) * arrLenPerBlock;
        end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);
    }

    __syncthreads();

    for (Nd4jLong i = start + threadIdx.x; i < end; i += blockDim.x)
        z[shape::getIndexOffset(i, zShapeInfo, arrLen)] = x[shape::getIndexOffset(i, xShapeInfo, arrLen)];
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void concatCudaLauncher(const int numOfArrs, const cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

    concatCuda<T><<<512, 256, 1024, *stream>>>(numOfArrs, pVx, pxShapeInfo, pVz, pzShapeInfo);
}
BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher,  (const int numOfArrs, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
// x - input, y - paddings, z - output
template<typename X, typename Y>
__global__ static void padCuda(const int mode,
                               const void *vx, const Nd4jLong *xShapeInfo,
                               const void *vy, const Nd4jLong *yShapeInfo,
                                     void *vz, const Nd4jLong *zShapeInfo,
                               const void *vPadVal) {

    const X padVal = *reinterpret_cast<const X*>(vPadVal);

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<X*>(vz);

    __shared__ int rank, rankMinusOne;
    __shared__ Nd4jLong zLen, yLen, totalThreads, *coords, *xShape, *zShape, *xStride, *zStride, shift1, shift2, yStride0;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        coords    = reinterpret_cast<Nd4jLong*>(shmem);
        zLen     = shape::length(zShapeInfo);
        xShape   = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
        zShape   = shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo));
        xStride  = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
        zStride  = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));
        yStride0 = shape::stride(const_cast<Nd4jLong*>(yShapeInfo))[0];
        rank     = shape::rank(xShapeInfo);
        zLen     = shape::length(zShapeInfo);
        yLen     = 2 * rank;
        rankMinusOne = rank - 1;
        totalThreads = gridDim.x * blockDim.x;
        shift1 = mode == 1 ? 0 : 1;         // REFLECT : SYMMETRIC
        shift2 = mode == 1 ? 2 : 1;         // REFLECT : SYMMETRIC
    }

    __syncthreads();

    auto xzCoord = coords + threadIdx.x * rank;       // we use xzCoord storage both for x and z arrays

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(mode == 0) { // CONSTANT case

        for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

            shape::index2coords(rank, zShape, i, zLen, xzCoord);
            const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);

            bool within = true;
            for(int j = rankMinusOne; j >= 0; --j) {
                if(xShape[j] == zShape[j]) continue;
                const auto left = y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];
                if(xzCoord[j] < left || xzCoord[j] >= left + xShape[j]) {within = false; break;}
                else                                                    {xzCoord[j] = xzCoord[j] - left;}
            }

            if(within)
                z[zOffset] = x[shape::getOffset(0, xShape, xStride, xzCoord, rank)];
            else
                z[zOffset] = padVal;
        }
    }
    else {  // REFLECT and SYMMETRIC cases

        for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

            shape::index2coords(rank, zShape, i, zLen, xzCoord);
            const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);

            for(int j = rankMinusOne; j >= 0; --j) {

                if(xShape[j] == zShape[j]) continue;
                xzCoord[j] = xzCoord[j] - y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];    // are ready to fill middle (within input dimension range)
                if(xzCoord[j] < 0)               xzCoord[j] = -xzCoord[j] - shift1;                // means fill from left
                else if(xzCoord[j] >= xShape[j]) xzCoord[j] = 2 * xShape[j] - xzCoord[j] - shift2; // means fill from right
            }

            const auto xOffset = shape::getOffset(0, xShape, xStride, xzCoord, rank);
            z[zOffset] = x[xOffset];
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void padCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const int mode,
                                const void *vx, const Nd4jLong *xShapeInfo,
                                const void *vy, const Nd4jLong *yShapeInfo,
                                      void *vz, const Nd4jLong *zShapeInfo,
                                const void* padVal) {

    padCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(mode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, padVal);
}
BUILD_DOUBLE_TEMPLATE(template void padCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const int mode, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const void* vPadVal), LIBND4J_TYPES, INTEGER_TYPES);

///////////////////////////////////////////////////////////////////
void pad(nd4j::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, const NDArray& padValue) {

    PointersManager manager(context, "pad");

    NDArray::prepareSpecialUse({&output}, {&input, &paddings, &padValue});

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = 8 * threadsPerBlock * output.rankOf() + 128;

    const auto xType = input.dataType();
    const auto yType = paddings.dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, padCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), mode, input.getSpecialBuffer(), input.getSpecialShapeInfo(), paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), padValue.getSpecialBuffer()), LIBND4J_TYPES, INTEGER_TYPES);

    NDArray::registerSpecialUse({&output}, {&input, &paddings, &padValue});
    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void invertPermutationCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo) {

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong len, totalThreads;

    if (threadIdx.x == 0) {

        len  = shape::length(xShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < len; i += totalThreads) {

        const auto xOffset = shape::getIndexOffset(i, xShapeInfo, len);
        const Nd4jLong index = x[xOffset];
        const auto zOffset = shape::getIndexOffset(index, zShapeInfo, len);
        z[zOffset] = i;
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void invertPermutationCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                                   const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo) {

    invertPermutationCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo);
}
BUILD_SINGLE_TEMPLATE(template void invertPermutationCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void invertPermutation(nd4j::LaunchContext* context, const NDArray& input, NDArray& output) {

    const int threadsPerBlock = MAX_NUM_THREADS;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "invertPermutation");

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), invertPermutationCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), LIBND4J_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void traceCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint diagLen) {

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ T* sharedMem;
    __shared__ int xRank, zRank;        // xRank = zRank + 2
    __shared__ Nd4jLong xLen, zLen, *coordsMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<T*>(shmem);
        coordsMem = reinterpret_cast<Nd4jLong*>(shmem + blockDim.x * sizeof(T));

        xRank = shape::rank(xShapeInfo);
        zRank = shape::rank(zShapeInfo);
        xLen = shape::length(xShapeInfo);
        zLen = shape::length(zShapeInfo);   // corresponds to number of matrices

    }
    __syncthreads();

    Nd4jLong* coords = coordsMem + threadIdx.x * xRank;

    for (uint m = blockIdx.x; m < zLen; m += gridDim.x) {   // one block per each element of z, that is per each matrix

        shape::index2coords(zRank, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), m, zLen, coords);
        const auto zOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), shape::stride(const_cast<Nd4jLong*>(zShapeInfo)), coords, zRank);

        sharedMem[threadIdx.x] = 0;

          for (uint i = threadIdx.x; i < diagLen; i += blockDim.x) {

            coords[zRank] = coords[zRank + 1] = i;
            const auto xOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), coords, xRank);
            sharedMem[threadIdx.x] += x[xOffset];
          }

          __syncthreads();

        // aggregate sum
        for (Nd4jLong activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
            if (threadIdx.x < activeThreads)
                sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            z[zOffset] = *sharedMem;
    }

}

///////////////////////////////////////////////////////////////////
template<typename T>
static void traceCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                              const void *vx, const Nd4jLong *xShapeInfo,
                                    void *vz, const Nd4jLong *zShapeInfo,
                                    const uint diagLen) {

    traceCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, diagLen);
}
BUILD_SINGLE_TEMPLATE(template void traceCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const uint diagLen), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
void trace(nd4j::LaunchContext* context, const NDArray& input, NDArray& output) {

    PointersManager manager(context, "trace");

    const uint diagLen = input.sizeAt(-1) < input.sizeAt(-2) ? input.sizeAt(-1) : input.sizeAt(-2);
    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * (sizeof(Nd4jLong) * input.rankOf() + input.sizeOfT()) + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), traceCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), diagLen), LIBND4J_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void triuBPCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int diag) {

    // x and z have same shapes
    const auto x = reinterpret_cast<const T*>(vx);  // gradO
          auto z = reinterpret_cast<T*>(vz);        // gradI

    __shared__ int rank, areSameOffsets;                // xRank = zRank
    __shared__ Nd4jLong len, totalThreads, *sharedMem;  // xLen = zLen

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);
        areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        rank = shape::rank(xShapeInfo);
        len  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < len; i += totalThreads) {

        shape::index2coords(rank, zShapeInfo + 1, i, len, coords);

        const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + rank + 1, coords, rank);

        if((coords[rank - 2] + diag > coords[rank - 1]))    // row + diag > col
            z[zOffset] = 0;
        else
            z[zOffset] = x[areSameOffsets ? zOffset : shape::getOffset(0, xShapeInfo + 1, xShapeInfo + rank + 1, coords, rank)];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void triuBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int diag) {

    triuBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, diag);
}
BUILD_SINGLE_TEMPLATE(template void triuBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int diag), LIBND4J_TYPES);

///////////////////////////////////////////////////////////////////
void triuBP(nd4j::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradO.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(Nd4jLong) * gradO.rankOf() + 128;

    PointersManager manager(context, "triuBP");

    NDArray::prepareSpecialUse({&gradI}, {&gradO});
    BUILD_SINGLE_SELECTOR(gradI.dataType(), triuBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), diagonal), LIBND4J_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&gradO});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void tileBPCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, Nd4jLong* globMem) {

    // x and z have same shapes
    const auto x = reinterpret_cast<const T*>(vx);  // gradO
          auto z = reinterpret_cast<T*>(vz);        // gradI

    __shared__ int xRank, zRank;                // xRank >= zRank
    __shared__ Nd4jLong numOfXOffsets, zLen, totalThreads, *sharedMem;  // xLen >= zLen

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xRank = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        numOfXOffsets = shape::length(xShapeInfo) / zLen;

        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto memBuff  = sharedMem + threadIdx.x * 2 * xRank;
    auto xOffsets = globMem + tid * numOfXOffsets;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        const auto zOffset = shape::getIndexOffset(i, zShapeInfo, zLen);

        shape::outerArrayOffsets(xOffsets, i, xShapeInfo, zShapeInfo, memBuff);

        z[zOffset] = x[xOffsets[0]];                    // first offset
        for (Nd4jLong j = 1; j < numOfXOffsets; ++j)    // rest offsets
            z[zOffset] += x[xOffsets[j]];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void tileBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, Nd4jLong* globMem) {

    tileBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, globMem);
}
BUILD_SINGLE_TEMPLATE(template void tileBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, Nd4jLong* globMem), FLOAT_TYPES);


//////////////////////////////////////////////////////////////////////////
void tileBP(nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    NDArray memBuff('c', gradO.getShapeAsVector(), nd4j::DataType::INT64, context);        // empty auxiliary array for storing device memory which will be used in kernel calculations

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(Nd4jLong) * 2 * gradO.rankOf() + 128;

    PointersManager manager(context, "tileBP");

    NDArray::prepareSpecialUse({&gradI}, {&gradO, &memBuff});
    BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), reinterpret_cast<Nd4jLong*>(memBuff.specialBuffer())), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&gradO, &memBuff});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void scatterUpdateCuda(const int opCode, const int numOfInd,
                                              void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
                                              void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
                                              const int* indexes) {

    __shared__ T *x, *y;
    __shared__ Nd4jLong arrLenX, arrLenY;

    for (int e = 0; e < numOfInd; e++ ) {

        const auto xIndex = indexes[e];
        const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

        if (!isOwner)
            continue;

        if (threadIdx.x == 0) {
            x = reinterpret_cast<T*>(vx) + xOffsets[xIndex];
            y = reinterpret_cast<T*>(vy) + yOffsets[e];
            arrLenX = shape::length(xShapeInfo);
            arrLenY = shape::length(yShapeInfo);
        }

        __syncthreads();

        if (arrLenX != arrLenY)
            return;

        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

            const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
            const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

            switch (opCode) {
                case 0:
                    x[xOffset] += y[yOffset];
                    break;
                case 1:
                    x[xOffset] -= y[yOffset];
                    break;
                case 2:
                    x[xOffset] *= y[yOffset];
                    break;
                case 3:
                    x[xOffset] /= y[yOffset];
                    break;
                case 4:
                    x[xOffset] = y[yOffset] - x[xOffset];
                    break;
                case 5:
                    x[xOffset] = y[yOffset] / x[xOffset];
                    break;
                case 6:
                    x[xOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
        __syncthreads();
    }
}

template<typename T>
__host__ static void scatterUpdateCudaLauncher(const cudaStream_t* stream, const int opCode, const int numOfInd, void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets, void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets, const int* indexes) {

    scatterUpdateCuda<T><<<512, 256, MAX_NUM_THREADS, *stream>>>(opCode, numOfInd, vx, xShapeInfo, xOffsets, vy, yShapeInfo, yOffsets, indexes);
}


//////////////////////////////////////////////////////////////////////////
void scatterUpdate(nd4j::LaunchContext* context, NDArray& input, NDArray& updates, const std::vector<int>* intArgs) {

    const int opCode    = (*intArgs)[0];
    const int numOfDims = (*intArgs)[1];
    const int numOfInd  = (*intArgs)[2 + numOfDims];

    std::vector<int> tadDimensions(numOfDims);
    for (int e = 2; e < 2 + numOfDims; e++)
        tadDimensions[e-2] = (*intArgs)[e];

    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), tadDimensions);
    auto packY = ConstantTadHelper::getInstance()->tadForDimensions(updates.getShapeInfo(), tadDimensions);

    NDArray indices(const_cast<int*>(intArgs->data()) + numOfDims + 3, 'c', {numOfInd}, nd4j::DataType::INT32, context);

    PointersManager manager(context, "scatterUpdate");

    NDArray::prepareSpecialUse({&input}, {&input, &updates, &indices});
    BUILD_SINGLE_SELECTOR(input.dataType(), scatterUpdateCudaLauncher, (context->getCudaStream(), opCode, numOfInd, input.specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), updates.specialBuffer(), packY.platformShapeInfo(), packY.platformOffsets(), reinterpret_cast<int*>(indices.getSpecialBuffer())), LIBND4J_TYPES);
    NDArray::registerSpecialUse({&input}, {&input, &updates, &indices});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
// x - input, y - indices, z - output
template<typename X, typename Y>
__global__ static void gatherNDCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                    const void *vy, const Nd4jLong *yShapeInfo,
                                          void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<X*>(vz);

    __shared__ int xRank, yRank, zRank, maxRank, yLastDim;
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        xRank   = shape::rank(xShapeInfo);
        yRank   = shape::rank(yShapeInfo);
        zRank   = shape::rank(zShapeInfo);
        maxRank = nd4j::math::nd4j_max<int>(yRank, nd4j::math::nd4j_max<int>(xRank, zRank));

        zLen     = shape::length(zShapeInfo);
        yLastDim = yShapeInfo[yRank];

        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    auto coord = sharedMem + threadIdx.x * maxRank;

    Nd4jLong *zCoordStart, *xCoordStart;

    if(yLastDim == xRank) {
        zCoordStart = coord;
        xCoordStart = coord;
    }
    if(zRank >= xRank) {
        zCoordStart = coord;
        xCoordStart = coord + zRank - xRank;
    }
    else {
        zCoordStart = coord + xRank - zRank;
        xCoordStart = coord;
    }

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(zRank, zShapeInfo + 1, i, zLen, zCoordStart);

        const auto zOffset = shape::getOffset(0, zShapeInfo + 1, zShapeInfo + zRank + 1, zCoordStart, zRank);

        // last y coordinate
        int coordToRestore;
        if(yLastDim != xRank)
            coordToRestore = static_cast<int>(zCoordStart[yRank - 1]);

        zCoordStart[yRank - 1] = 0; // last y coordinate
        const auto yOffset = shape::getOffset(0, yShapeInfo + 1, yShapeInfo + yRank + 1, zCoordStart, yRank);

        //restore z coordinate
        if(yLastDim != xRank)
            zCoordStart[yRank - 1] = coordToRestore;

        // construct coordinates for x
        for(uint j = 0; j < yLastDim; ++j)
            xCoordStart[j] = y[yOffset + j * yShapeInfo[2 * yRank]];   // last stride

        const auto xOffset = shape::getOffset(0, xShapeInfo + 1, xShapeInfo + xRank + 1, xCoordStart, xRank);

        z[zOffset] = x[xOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void gatherNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                 const void *vx, const Nd4jLong *xShapeInfo,
                                 const void *vy, const Nd4jLong *yShapeInfo,
                                       void *vz, const Nd4jLong *zShapeInfo) {

    gatherNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}
BUILD_DOUBLE_TEMPLATE(template void gatherNDCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo), LIBND4J_TYPES, INTEGER_TYPES);

///////////////////////////////////////////////////////////////////
void gatherND(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {

    const int maxRank = nd4j::math::nd4j_max<int>(indices.rankOf(), nd4j::math::nd4j_max<int>(input.rankOf(), output.rankOf()));

    const int threadsPerBlock = MAX_NUM_THREADS;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = 8 * threadsPerBlock * maxRank + 128;

    const auto xType = input.dataType();
    const auto yType = indices.dataType();

    PointersManager manager(context, "gatherND");

    NDArray::prepareSpecialUse({&output}, {&input, &indices});
    BUILD_DOUBLE_SELECTOR(xType, yType, gatherNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), LIBND4J_TYPES, INTEGER_TYPES);
    NDArray::registerSpecialUse({&output}, {&input, &indices});

    manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
// x - input, y - gradO, z - gradI
template<typename X, typename Z>
__global__ static void clipByNormBPWholeArrCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, void* vreducBuff, const Z clipNormVal) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= shape::length(zShapeInfo))
        return;

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Z*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    auto reducBuff = reinterpret_cast<Z*>(vreducBuff);
    uint* count    = reinterpret_cast<uint*>(vreducBuff) + 16384;

    __shared__ Z* shMem;
    __shared__ Nd4jLong len;
    __shared__ bool amIinLastBlock;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        shMem = reinterpret_cast<Z*>(shmem);

        len = shape::length(zShapeInfo);   // xLen = yLen = zLen
    }
    __syncthreads();

    // fill shared memory with array elements
    const auto xVal = x[shape::getIndexOffset(tid, xShapeInfo, len)];
    const auto yVal = y[shape::getIndexOffset(tid, yShapeInfo, len)];

    shMem[2*threadIdx.x]     = static_cast<Z>(xVal * xVal);   // for norm
    shMem[2*threadIdx.x + 1] = static_cast<Z>(xVal * yVal);   // for input * gradO

    __syncthreads();

    // accumulate sum per block
    for (int activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {

        if (threadIdx.x < activeThreads && tid + activeThreads < len) {

            shMem[2*threadIdx.x]     += shMem[2*(threadIdx.x + activeThreads)];
            shMem[2*threadIdx.x + 1] += shMem[2*(threadIdx.x + activeThreads) + 1];
        }
        __syncthreads();
    }

    // store accumulated sums in reduction buffer (reducBuff)
    if (threadIdx.x == 0) {

        reducBuff[2*blockIdx.x]     = shMem[0];
        reducBuff[2*blockIdx.x + 1] = shMem[1];

        __threadfence();

        amIinLastBlock = gridDim.x == 1 || (atomicInc(count, gridDim.x) == gridDim.x - 1);
    }
    __syncthreads();

    // shared memory of last block is used for final summation of values stored in reduction buffer
    if (amIinLastBlock) {

        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {

            shMem[2*threadIdx.x]     = (i == threadIdx.x ) ? reducBuff[2*i]     : reducBuff[2*i]     + shMem[2*threadIdx.x];
            shMem[2*threadIdx.x + 1] = (i == threadIdx.x ) ? reducBuff[2*i + 1] : reducBuff[2*i + 1] + shMem[2*threadIdx.x + 1];
        }
        __syncthreads();

        // accumulate sum
        for (int activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {

            if (threadIdx.x < activeThreads && threadIdx.x + activeThreads < gridDim.x) {
                shMem[2*threadIdx.x]     += shMem[2*(threadIdx.x + activeThreads)];
                shMem[2*threadIdx.x + 1] += shMem[2*(threadIdx.x + activeThreads) + 1];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {

            reducBuff[0] = math::nd4j_sqrt<Z,Z>(shMem[0]);
            reducBuff[1] = shMem[1];
            count = 0;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// x - input, y - gradO, z - gradI
template<typename X, typename Z>
__global__ static void clipByNormBPCalcGradCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, void* vreducBuff, const Z clipNormVal) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    const Nd4jLong len = shape::length(zShapeInfo);     // xLen = yLen = zLen

    if(tid >= len)
        return;

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Z*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ Z norm, sumOfProd;

    if (threadIdx.x == 0) {

        norm = reinterpret_cast<Z*>(vreducBuff)[0];
        sumOfProd = reinterpret_cast<Z*>(vreducBuff)[1];
    }
    __syncthreads();

    const auto yOffset = shape::getIndexOffset(tid, yShapeInfo, len);
    const auto zOffset = shape::getIndexOffset(tid, zShapeInfo, len);

   if(norm > clipNormVal) {

        const auto xOffset = shape::getIndexOffset(tid, xShapeInfo, len);

        const Z factor1 = static_cast<Z>(1) / norm;             // 1 / norm
        const Z factor2 = factor1 / (norm * norm);              // 1 / (norm * norm * norm)

        z[zOffset] = clipNormVal * (factor1 * y[yOffset] - factor2 * sumOfProd * x[xOffset]);
    }
    else {
        z[zOffset] = y[yOffset];
    }
}

//////////////////////////////////////////////////////////////////////////
// x - input, y - gradO, z - gradI
template<typename X, typename Z>
__global__ static void clipByNormBPTadsCuda(const void* vx, const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets, const void* vy, const Nd4jLong* yTadShapeInfo, const Nd4jLong* yTadOffsets, void* vz, const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets, const Z clipNormVal) {

    const auto x = reinterpret_cast<const X*>(vx);
    const auto y = reinterpret_cast<const Z*>(vy);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ Z* shMem;
    __shared__ Nd4jLong tadLen;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        shMem = reinterpret_cast<Z*>(shmem);
        tadLen = shape::length(zTadShapeInfo);                  // xTadLen = yTadLen = zTadLen
    }
    __syncthreads();

    const auto* xTad = x + xTadOffsets[blockIdx.x];
    const auto* yTad = y + yTadOffsets[blockIdx.x];
          auto* zTad = z + zTadOffsets[blockIdx.x];

    // *** FIRST STAGE - ACCUMULATE REQUIRED SUMS *** //

    Z norm = 0;
    Z sumOfProd = 0;

    for (uint i = threadIdx.x; i < tadLen; i += blockDim.x) {

        const auto xOffset = shape::getIndexOffset(i, xTadShapeInfo, tadLen);
        const auto yOffset = shape::getIndexOffset(i, yTadShapeInfo, tadLen);

        shMem[2*threadIdx.x]     = static_cast<Z>(xTad[xOffset] * xTad[xOffset]);   // for norm
        shMem[2*threadIdx.x + 1] = static_cast<Z>(xTad[xOffset] * yTad[yOffset]);   // for input * gradO

        __syncthreads();

        // accumulate sum per block
        for (uint activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {

            if (threadIdx.x < activeThreads && i + activeThreads < tadLen) {

                shMem[2*threadIdx.x]     += shMem[2*(threadIdx.x + activeThreads)];
                shMem[2*threadIdx.x + 1] += shMem[2*(threadIdx.x + activeThreads) + 1];
            }
            __syncthreads();
        }

        norm      += shMem[0];
        sumOfProd += shMem[1];
    }

    // *** SECOND STAGE - GRADIENT CALCULATION *** //

    norm = math::nd4j_sqrt<Z,Z>(norm);

    for (uint i = threadIdx.x; i < tadLen; i += blockDim.x) {

        const auto yOffset = shape::getIndexOffset(i, yTadShapeInfo, tadLen);
        const auto zOffset = shape::getIndexOffset(i, zTadShapeInfo, tadLen);

        if(norm > clipNormVal) {

            const auto xOffset = shape::getIndexOffset(i, xTadShapeInfo, tadLen);

            const Z factor1 = static_cast<Z>(1) / norm;             // 1 / norm
            const Z factor2 = factor1 / (norm * norm);              // 1 / (norm * norm * norm)

            zTad[zOffset] = clipNormVal * (factor1 * yTad[yOffset] - factor2 * sumOfProd * xTad[xOffset]);
        }
        else {
            zTad[zOffset] = yTad[yOffset];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
static void clipByNormBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                    const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                    const void* vy, const Nd4jLong* yShapeInfo, const Nd4jLong* yTadOffsets,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets,
                                    void* vreducBuff, const double clipNormVal) {

    if(xTadOffsets == nullptr) {  // means whole array
        clipByNormBPWholeArrCuda<X,Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vreducBuff, static_cast<Z>(clipNormVal));
        clipByNormBPCalcGradCuda<X,Z><<<blocksPerGrid, threadsPerBlock, 256,       *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, vreducBuff, static_cast<Z>(clipNormVal));
    }
    else                        // means tads using
        clipByNormBPTadsCuda<X,Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, xTadOffsets, vy, yShapeInfo, yTadOffsets, vz, zShapeInfo, zTadOffsets, static_cast<Z>(clipNormVal));
}
BUILD_DOUBLE_TEMPLATE(template void clipByNormBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const Nd4jLong* xTadOffsets, const void *vy, const Nd4jLong *yShapeInfo, const Nd4jLong* yTadOffsets, void *vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets, void* vreducBuff, const double clipNormVal), LIBND4J_TYPES, FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBP(nd4j::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {

    PointersManager manager(context, "clipByNormBP");

    const double clipNormVal = clipNorm.e<double>(0);

    const auto xType = input.dataType();
    const auto zType = gradI.dataType();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int sharedMem = threadsPerBlock * 2 * input.sizeOfT() + 128;

    NDArray::prepareSpecialUse({&gradI}, {&input, &gradO});


    if(dimensions.empty() || dimensions.size() == input.rankOf()) {  // means whole array

        const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        BUILD_DOUBLE_SELECTOR(xType, zType, clipByNormBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), nullptr, gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), nullptr, gradI.getSpecialBuffer(), gradI.getSpecialShapeInfo(), nullptr, context->getReductionPointer(), clipNormVal), LIBND4J_TYPES, FLOAT_TYPES);
    }
    else {  // means tads using

        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
        auto packY = ConstantTadHelper::getInstance()->tadForDimensions(gradO.getShapeInfo(), dimensions);
        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(gradI.getShapeInfo(), dimensions);

        const int blocksPerGrid = packX.numberOfTads();
        BUILD_DOUBLE_SELECTOR(xType, zType, clipByNormBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), gradO.getSpecialBuffer(), packY.platformShapeInfo(), packY.platformOffsets(), gradI.getSpecialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), nullptr, clipNormVal), LIBND4J_TYPES, FLOAT_TYPES);
    }

    NDArray::registerSpecialUse({&gradI}, {&input, &gradO});

    manager.synchronize();
}






































    template <typename T>
    static __global__ void swapShuffleKernel(T* input, Nd4jLong* shape, Nd4jLong firstDim, Nd4jLong len, nd4j::graph::RandomGenerator* rng) {
        auto tid = blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (int i = firstDim - 1 - tid - threadIdx.x; i > 0; i -= step) {
            int r = rng->relativeInt(i) % i;
            if (i != r) {
                T e0 = input[shape::getIndexOffset(i, shape, len)];
                T e1 = input[shape::getIndexOffset(r, shape, len)];
                //math::nd4j_swap<T>(input(i), input(r));
                input[shape::getIndexOffset(i, shape, len)] = e1;
                input[shape::getIndexOffset(r, shape, len)] = e0;
            }
        }
    }
    template <typename T>
    static __global__ void fillShuffleKernel(T* input, Nd4jLong* inputShape, T* output, Nd4jLong* outputShape, Nd4jLong firstDim, Nd4jLong len, int* indices, nd4j::graph::RandomGenerator* rng) {

//        PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->tadThreshold())
        auto tid = blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for(int i = firstDim - 1 - tid - threadIdx.x; i > 0; i -= step) {
            int r = rng->relativeInt(i) % i;
            output[shape::getIndexOffset(i, outputShape, len)] = input[shape::getIndexOffset(indices[r], inputShape, len)];
            if(i != r) {
                output[shape::getIndexOffset(r, outputShape, len)] = input[shape::getIndexOffset(indices[i], inputShape, len)];
//                output.p(r, input.e<T>(indices[i]));
//                math::nd4j_swap<int>(indices[i], indices[r]);
                atomicExch(&indices[i], indices[r]);
            }
        }

    }
    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void randomShuffle_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace) {

        // check edge cases first
        int temp;
        const int firstDim = input.sizeAt(0);
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({&output}, {&input});
        if(input.lengthOf() == 1 || firstDim == 1) {
            if(!isInplace)
                output.assign(input);
        }
        else if (input.isVector() || shape::isLikeVector(input.getShapeInfo(), temp)) {

            // apply Fisher-Yates shuffle
            nd4j::graph::RandomGenerator* dRandom = nullptr;
            cudaMalloc(&dRandom, sizeof(nd4j::graph::RandomGenerator));
            cudaMemcpy(dRandom, &rng, sizeof(nd4j::graph::RandomGenerator), cudaMemcpyHostToDevice);
            T* inputBuf = reinterpret_cast<T*>(input.specialBuffer());
            if(isInplace) {
                swapShuffleKernel<T><<<128, 256, 1024, *stream>>>(inputBuf, input.specialShapeInfo(), firstDim, input.lengthOf(), dRandom);
            }
            else {
                std::vector<int> indices(firstDim);
                std::iota(indices.begin(), indices.end(), 0);
                cudaMemcpy(output.specialBuffer(), input.specialBuffer(), sizeof(T), cudaMemcpyDeviceToDevice);
                //output.p<T>(Nd4jLong(0), input.e<T>(0));
                PointersManager pointersManager(context, "helper::randomShuffle_");
                int* indicesDev = reinterpret_cast<int*>(pointersManager.replicatePointer(indices.data(), indices.size() * sizeof(int)));
                T* outputBuf = reinterpret_cast<T*>(output.specialBuffer());
                fillShuffleKernel<T><<<128, 256, 1024, *stream>>>(inputBuf, input.specialShapeInfo(), outputBuf, output.specialShapeInfo(), firstDim, input.lengthOf(), indicesDev, dRandom);
                pointersManager.synchronize();
            }
//            rng.rewindH(firstDim - 1);
            cudaFree(dRandom);
        }
        else {

            // evaluate sub-arrays list of input array through all dimensions excluding first one
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});
            auto subArrsListIn = input.allTensorsAlongDimension(dimensions);

            // apply Fisher-Yates shuffle
            if(isInplace) {
                PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->elementwiseThreshold())
                for(int i = firstDim - 1; i > 0; --i) {
                    int r = rng.relativeInt(i) % i;

                    if(i != r)
                        subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
                }
            }
            else {
                // evaluate sub-arrays list of output array through all dimensions excluding first one
                auto subArrsListOut = output.allTensorsAlongDimension(dimensions);
                std::vector<int> indices(firstDim);
                std::iota(indices.begin(), indices.end(), 0);
                bool isZeroShuffled = false;
                PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->tadThreshold())
                for(int i = firstDim - 1; i > 0; --i) {
                    int r = rng.relativeInt(i) % i;
                    subArrsListOut->at(i)->assign(subArrsListIn->at(indices[r]));
                    if(r == 0)
                        isZeroShuffled = true;

                    if(i != r) {
                        subArrsListOut->at(r)->assign(subArrsListIn->at(indices[i]));
                        math::nd4j_swap<int>(indices[i], indices[r]);
                    }
                }
                if(!isZeroShuffled)
                    subArrsListOut->at(0)->assign(subArrsListIn->at(0));
                delete subArrsListOut;
            }
            rng.rewindH(firstDim-1);
            delete subArrsListIn;
        }
        NDArray::registerSpecialUse({&output}, {&input});

    }

    void randomShuffle(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace), LIBND4J_TYPES);



//////////////////////////////////////////////////////////////////////////
void eye(nd4j::LaunchContext * context, NDArray& output) {

    output.setIdentity();
}

    //////////////////////////////////////////////////////////////////////////
    template <typename T, typename Z>
    static __global__ void global_mergeMaxIndex_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<Z*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T mVal = -DataTypeUtils::max<T>();
            Z mIdx(0);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                auto val = x[shape::getIndexOffset(e, xShape, length)];;
                if (mVal < val)
                    mIdx = static_cast<Z>(e);
            }
            __syncthreads();

            output[shape::getIndexOffset(e, outputShape, length)] = mIdx;
        }
    }

    template <typename T, typename Z>
    static void mergeMaxIndex_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeMaxIndex");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeMaxIndex_<T,Z><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }

    void mergeMaxIndex(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void mergeMaxIndex_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES, INTEGER_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeMax_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T mVal = -DataTypeUtils::max<T>();

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);
                auto val = x[shape::getIndexOffset(e, xShape, length)];;
                if (mVal < val)
                    mVal = val;
            }
            __syncthreads();

            output[shape::getIndexOffset(e, outputShape, length)] = mVal;
        }
    }

    template<typename T>
    static void mergeMax_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeMax");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeMax_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeMax_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeMax(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeAvg_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T sum(0.0f);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                sum += x[shape::getIndexOffset(e, xShape, length)];
            }

            output[shape::getIndexOffset(e, outputShape, length)] = sum / numArrays;
        }
    }

    template<typename T>
    static void mergeAvg_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeAvg");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeAvg_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAvg(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void global_mergeAdd_(void **inArrs, void **inShapes, const int numArrays, void *voutput, Nd4jLong *outputShape, Nd4jLong length) {
        auto output = reinterpret_cast<T*>(voutput);

        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            T sum(0.0f);

            for (int i = 0; i < numArrays; i++) {
                auto x = reinterpret_cast<T*>(inArrs[i]);
                auto xShape = reinterpret_cast<Nd4jLong *>(inShapes[i]);

                sum += x[shape::getIndexOffset(e, xShape, length)];
            }

            output[shape::getIndexOffset(e, outputShape, length)] = sum;
        }
    }

    template<typename T>
    static void mergeAdd_(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        std::vector<void *> inBuffers(inArrs.size());
        std::vector<void *> inShapes(inArrs.size());

        for (int e = 0; e < inArrs.size(); e++) {
            inBuffers[e] = inArrs[e]->getSpecialBuffer();
            inShapes[e] = inArrs[e]->getSpecialShapeInfo();
        }

        PointersManager manager(context, "mergeAdd");

        auto pInBuffers = reinterpret_cast<void **>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void *)));
        auto pInShapes = reinterpret_cast<void **>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void *)));
        auto length = output.lengthOf();

        global_mergeAdd_<T><<<512, 512, 512, *context->getCudaStream()>>>(pInBuffers, pInShapes, (int) inArrs.size(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), length);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    void mergeAdd(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), LIBND4J_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void clipByNormInplaceKernel(Nd4jLong numOfSubArrs, T* inputBuffer, Nd4jLong* shape, Nd4jLong* inputOffsets, T* norm2Buf, Nd4jLong* norm2shape, T clipNorm) {
        for (int arr = blockIdx.x; arr < numOfSubArrs; arr += gridDim.x) {
            __shared__ T* z;
            __shared__ Nd4jLong len;
            if (threadIdx.x == 0) {
                len = shape::length(shape);
                z = inputBuffer + inputOffsets[arr];
            }
            __syncthreads();
            for (int j = threadIdx.x; j < len; j+= blockDim.x) {
                auto xIndex = shape::getIndexOffset(j, shape, len);

                if(norm2Buf[arr] > clipNorm)
                z[xIndex] *= clipNorm / norm2Buf[arr]; // case with ews = 1 and ordering is 'c'
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void clipByNormKernel(Nd4jLong numOfSubArrs, T* inputBuffer, Nd4jLong* shape, Nd4jLong* inputOffsets, T* outputBuffer, Nd4jLong* outputShape, Nd4jLong* outputOffsets, T* norm2Buf, Nd4jLong* norm2shape, T clipNorm) {
        for (Nd4jLong arr = blockIdx.x; arr < numOfSubArrs; arr += gridDim.x) {
            __shared__ T* x, *z;
            __shared__ Nd4jLong lenX, lenZ;
            __shared__ T norm2;

            if (threadIdx.x == 0) {
                lenX = shape::length(shape);
                x = inputBuffer + inputOffsets[arr];
                z = outputBuffer + outputOffsets[arr];
                lenZ = shape::length(outputShape);
                norm2 = norm2Buf[shape::getIndexOffset(arr, norm2shape, numOfSubArrs)];
                //printf("%d: %lf (vs %lf) %lld %lld\n", arr, norm2, clipNorm, lenX, lenZ);
            }
            __syncthreads();
            for (Nd4jLong j = threadIdx.x; j < lenZ; j+= blockDim.x) {
                auto xIndex = shape::getIndexOffset(j, shape, lenX);
                auto zIndex = shape::getIndexOffset(j, outputShape, lenZ);
                if(norm2 > clipNorm) {
                    z[zIndex] = x[xIndex] * clipNorm / norm2; // case with ews = 1 and ordering is 'c'
                } else {
                    z[zIndex] = x[xIndex];
                }
                //printf("%lld: %lf %lf\n", j, z[zIndex], x[xIndex]);
            }
            __syncthreads();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNorm_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, NDArray const& clipNormA, const bool isInplace) {
        const int rank = input.rankOf();
        auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions);
        clipNormA.syncToHost();
        //norm2.printBuffer("Norm2");
        T const clipNorm = clipNormA.e<T>(0);
        //clipNormA.printBuffer("ClipNorm");
        auto stream = context->getCudaStream();
        if (isInplace) {
            if(norm2.lengthOf() == 1) {
                norm2.syncToHost();
                T norm2Val = norm2.e<T>(0);
                if(norm2Val > clipNorm)
                    input *= clipNorm / norm2Val;
            }
            else {

                std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
                const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
                //auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimsToExclude);
                T* inputBuffer = reinterpret_cast<T*>(input.specialBuffer());
                T* norm2buf = reinterpret_cast<T*>(norm2.specialBuffer());

                clipByNormInplaceKernel<T><<<256, 512, 1024, *stream>>>(numOfSubArrs, inputBuffer, packX.specialShapeInfo(), packX.specialOffsets(), norm2buf, norm2.specialShapeInfo(), clipNorm);
            }
        }
        else {

            if(norm2.lengthOf() == 1) {
                norm2.syncToHost();
                T norm2Val = norm2.e<T>(0);

                if(norm2Val > clipNorm)
                    output.assign( input * (clipNorm / norm2Val));
                else
                    output.assign( input );
            }
            else {

                std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
                const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
                auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimensions);
                T* inputBuffer = reinterpret_cast<T*>(input.specialBuffer());
                T* norm2buf = reinterpret_cast<T*>(norm2.specialBuffer());
                T* outputBuffer = reinterpret_cast<T*>(output.specialBuffer());

                clipByNormKernel<T><<<256, 512, 1024, *stream>>>(numOfSubArrs, inputBuffer, packX.specialShapeInfo(), packX.specialOffsets(), outputBuffer, packZ.specialShapeInfo(), packZ.specialOffsets(), norm2buf, norm2.specialShapeInfo(), clipNorm);
            }
        }
    }

    void clipByNorm(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNorm_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

    template <typename T>
    static void clipByGlobalNorm_(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {

    }

    void clipByGlobalNorm(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (context, inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByAveraged_(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        auto cn = clipNorm.e<T>(0);
        if (dimensions.size() == 0) {
            // all-reduce
            T n2 = input.reduceNumber(reduce::Norm2).e<T>(0) / input.lengthOf();
            if (n2 <= cn) {
                if (!isInplace)
                    output.assign(input);
            }
            else {
                const T factor = cn / n2;
                //auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
                //input.applyLambda<T>(lambda, &output);
                output.assign(input * factor);
            }
        }
        else {
            // along dimension
            auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions, false);
            if (!isInplace)
                output.assign(input);
            auto tads = output.allTensorsAlongDimension(dimensions);
            auto outTads = output.allTensorsAlongDimension(dimensions);
            // TODO: make this CUDA-compliant somehow
            for (int e = 0; e < tads->size(); e++) {
                T n2 = norm2.e<T>(e) / tads->at(e)->lengthOf();
                const T factor = cn / n2;
                if (n2 > cn) {
                    //auto lambda = LAMBDA_T(_x, factor) {return _x * factor;};
                    tads->at(e)->applyScalar(scalar::Multiply, factor, outTads->at(e));//applyLambda<T>(lambda, &output);
                }
            }
            delete tads;
            delete outTads;
        }
    }

    void clipByAveraged(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

/*
    if (d1 > params[1])
    return params[1];
    else if (d1 < params[0])
    return params[0];
    else return d1;
*/
    template <typename T>
    static void __global__ clipByValueKernel(void* input, Nd4jLong* inputShape, void* output, Nd4jLong* outputShape, double leftBound, double rightBound) {
        __shared__ T* outputBuf;
        __shared__ T* inputBuf;
        __shared__ Nd4jLong length;
        __shared__ bool linearBuffers;
        if (threadIdx.x == 0) {
            outputBuf = reinterpret_cast<T *>(output);
            inputBuf = reinterpret_cast<T *>(input);
            length = shape::length(inputShape);
            linearBuffers = shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape) && shape::elementWiseStride(inputShape) == 1;
        }
        __syncthreads();
        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            if (linearBuffers) {
                if (inputBuf[e] > rightBound) outputBuf[e] = (T) rightBound;
                else if (inputBuf[e] < leftBound) outputBuf[e] = (T) leftBound;
                else outputBuf[e] = inputBuf[e];
            }
            else {
                auto inputOffset = shape::getIndexOffset(e, inputShape, length);
                auto outputOffset = shape::getIndexOffset(e, outputShape, length);
                if (inputBuf[inputOffset] > rightBound) outputBuf[outputOffset] = (T) rightBound;
                else if (inputBuf[inputOffset] < leftBound) outputBuf[outputOffset] = (T) leftBound;
                else outputBuf[outputOffset] = inputBuf[outputOffset];
            }
        }
    }

    template <typename T>
    static void clipByValue_(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        auto stream = context->getCudaStream();
        if (!input.isActualOnDeviceSide())
            input.syncToDevice();
        NDArray::prepareSpecialUse({&output}, {&input});
        clipByValueKernel<T><<<256, 512, 8192, *stream>>>(input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftBound, rightBound);
        NDArray::registerSpecialUse({&output}, {&input});
    }

    void clipByValue(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void mirrorPadLinearKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong leftSide, Nd4jLong leftSideCorrected, Nd4jLong xLen, Nd4jLong len, Nd4jLong zLen) {

        __shared__ T const* x;
        __shared__ T* z;
        if (threadIdx.x == 0) {
            x = reinterpret_cast<T const*>(vx);
            z = reinterpret_cast<T*>(vz);
        }
        __syncthreads();
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for(int i = start; i < zLen; i+= step) {
            auto zIndex = shape::getIndexOffset(i, zShape, zLen);
            auto xIndex = shape::getIndexOffset(len - i, xShape, xLen);

            if (i < leftSide)                                   // left side
                xIndex = shape::getIndexOffset(leftSideCorrected - i, xShape, xLen);

            else if(i >= leftSide && i < leftSide + xLen)       // middle
                xIndex = shape::getIndexOffset(i - leftSide, xShape, xLen);

//            else                                                // right side
//                z[i] = x[len - i];
            z[zIndex] = x[xIndex];
        }

    }

    template <typename F, typename I>
    static __global__ void mirrorPadKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong outLen, void const* paddings, Nd4jLong* paddingShape, int reflBorder) {

        __shared__ F const* x;
        __shared__ I const* pads;
        __shared__ F* z;
        __shared__ Nd4jLong zRank, rank;
        __shared__ Nd4jLong* xShapeOf, *xStrideOf, *padsShapeOf, *padsStrideOf;
        __shared__ Nd4jLong* zShapeOf, *zStrideOf;
        __shared__ Nd4jLong* xIdx;
        if (threadIdx.x == 0) {
            extern __shared__ unsigned char shmem[];
            xIdx    = reinterpret_cast<Nd4jLong*>(shmem);
            rank = shape::rank(xShape);

            x = reinterpret_cast<F const*>(vx);//
            pads = reinterpret_cast<I const*>(paddings);
            z = reinterpret_cast<F*>(vz);
            xShapeOf = shape::shapeOf(xShape);
            xStrideOf = shape::stride(xShape);
            zShapeOf = shape::shapeOf(zShape);
            zRank = shape::rank(zShape);
            zStrideOf = shape::stride(zShape);
            padsShapeOf = shape::shapeOf(paddingShape);
            padsStrideOf = shape::stride(paddingShape);
        }
        __syncthreads();
        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

            for(Nd4jLong i = start; i < outLen; i+= step) {
                auto xzCoord = xIdx + threadIdx.x * rank;
                //auto zxCoord = xIdx + (threadIdx.x + threadIdx.x % 2 + 1) * rank;

                shape::index2coords(rank, zShapeOf, i, xzCoord);
                auto outOffset = shape::getOffset(0, zShapeOf, zStrideOf, xzCoord, rank);
//                auto intStep = blockDim.y * gridDim.y;
                for(int j = 0; j < rank; j++) {

                    const Nd4jLong inLen         = shape::sizeAt(xShape, j);
                    Nd4jLong coords[2] = {j, 0};
                    auto padOffset = shape::getOffset(0, padsShapeOf, padsStrideOf, coords, 2); // padding already has rank 2
                    const auto leftSide          = pads[padOffset];
                    const auto leftSideCorrected = leftSide - reflBorder;
                    const Nd4jLong len           = 2 * (inLen - 1) + leftSide + reflBorder;

                    if(xzCoord[j] < leftSide)                                        // left side
                        xzCoord[j] = leftSideCorrected - xzCoord[j];

                    else if(xzCoord[j] >= leftSide && xzCoord[j] < leftSide + inLen)  // middle
                        xzCoord[j] = xzCoord[j] - leftSide;

                    else if (len > xzCoord[j])                                                           // right side
                        xzCoord[j] = len - xzCoord[j];
                    else
                        xzCoord[j] = xzCoord[j] - len;
                }

                auto inOffset  = shape::getOffset(0, xShapeOf, xStrideOf,  xzCoord,  rank);
                z[outOffset] = x[inOffset];
            }
    }

    template<typename F, typename I>
    static void mirrorPad_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        // mode:  0 - REFLECT, else - SYMMETRIC
        const int reflBorder = (bool)mode ? 1 : 0;
        const int rank        = input.rankOf();
        const Nd4jLong outLen = output.lengthOf();
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({&output}, {&input, &paddings});

        if(rank <= 1) {

            const Nd4jLong inLen         = input.lengthOf();
            const auto leftSide          = paddings.e<Nd4jLong>(0);
            const auto leftSideCorrected = leftSide - reflBorder;
            const Nd4jLong len           = 2*(inLen-1) + leftSide + reflBorder;

            mirrorPadLinearKernel<F><<<256, 512, 256, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftSide, leftSideCorrected, inLen, len, outLen);
            nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadLinearKernel(...) failed");
        }
        else {
            mirrorPadKernel<F, I><<<256, 256, 8192, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), outLen, paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), reflBorder);
            nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadKernel(...) failed");
        }
        NDArray::registerSpecialUse({&output}, {&input, &paddings});
    }

    void mirrorPad(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_DOUBLE_SELECTOR(input.dataType(), paddings.dataType(), mirrorPad_, (context, input, paddings, output, mode), LIBND4J_TYPES, INTEGER_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void mirrorPad_, (nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES, INTEGER_TYPES);

//////////////////////////////////////////////////////////////////////////
void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

    const int numOfArrs = inArrs.size();
    for(int i = 0; i < numOfArrs; ++i)
        if(!inArrs[i]->isActualOnDeviceSide()) inArrs[i]->syncToDevice();

    const int rank  = inArrs[0]->rankOf();
    const int rank2 = 2*rank;
    std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

    // take into account indices for first array
    indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

    // loop through the rest of input arrays
    for(int i = 1; i < numOfArrs; ++i) {
        indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
        indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
    }

    std::vector<NDArray*> outSubArrs(numOfArrs);
    for(int i = 0; i < numOfArrs; ++i)
        outSubArrs[i] = new NDArray(output(indices[i], true));

    // prepare arrays of pointers on buffers and shapes
    std::vector<void*>     hOutBuffers(numOfArrs), hInBuffers(numOfArrs);
    std::vector<Nd4jLong*> hOutShapeInfo(numOfArrs), hInShapeInfo(numOfArrs);
    for(int i = 0; i < numOfArrs; ++i) {
        hOutBuffers[i]   = outSubArrs[i]->getSpecialBuffer();
        hInBuffers[i]    =     inArrs[i]->getSpecialBuffer();
        hOutShapeInfo[i] = outSubArrs[i]->getSpecialShapeInfo();
        hInShapeInfo[i]  =     inArrs[i]->getSpecialShapeInfo();
    }

    // allocate and copy all buffers and shapes arrays to global memory
    PointersManager manager(context, "helpers::concat");
    void* dOutBuffers	= manager.replicatePointer(hOutBuffers.data(),   hOutBuffers.size() * sizeof(void*));
    void* dInBuffers	= manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
    void* dInShapeInfo  = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));
    void* dOutShapeInfo = manager.replicatePointer(hOutShapeInfo.data(), hOutShapeInfo.size() * sizeof(Nd4jLong*));

    BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher, (numOfArrs, context->getCudaStream(), dInBuffers, dInShapeInfo, dOutBuffers, dOutShapeInfo), LIBND4J_TYPES);

    manager.synchronize();

    for(int i = 0; i < numOfArrs; ++i)
        delete outSubArrs[i];

    for(int i = 0; i < numOfArrs; ++i)
        inArrs[i]->tickReadHost();

    output.tickWriteDevice();
}

    template <typename X, typename Y>
    static _CUDA_G void scatterSimpleKernel(void *vx, Nd4jLong *xTadShape, Nd4jLong *xTadOffsets, Nd4jLong xLength, Nd4jLong numTads, void *vi, Nd4jLong *iShapeInfo, Nd4jLong iLength, void *vu, Nd4jLong *uShapeInfo, Nd4jLong uLength) {
        auto u = reinterpret_cast<X*>(vu);
        auto indices = reinterpret_cast<Y*>(vi);

        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i = tid; i < iLength; i += blockDim.x * gridDim.x) {
            auto x = reinterpret_cast<X*>(vx) + xTadOffsets[i];
            auto idx = indices[shape::getIndexOffset(i, iShapeInfo, iLength)];

            x[shape::getIndexOffset(idx, xTadShape, xLength)] = u[shape::getIndexOffset(i, uShapeInfo, uLength)];
        }
    }


    template <typename X, typename Y>
    void scatterSimple_(nd4j::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

        auto dims = ShapeUtils::evalDimsToExclude(input.rankOf(), dimensions);
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dims);

        auto xLength = shape::length(packX.primaryShapeInfo());
        auto iLength = indices.lengthOf();
        auto uLength = updates.lengthOf();

        scatterSimpleKernel<X,Y><<<256, 256, 1024, *context->getCudaStream()>>>(input.getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), xLength, packX.numberOfTads(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), iLength, updates.getSpecialBuffer(), updates.getSpecialShapeInfo(), uLength);
    }


    void scatterSimple(nd4j::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {
        auto xType = input.dataType();
        auto yType = indices.dataType();

        if (opId != 6)
            throw std::runtime_error("scatterSimple: only copy op is supported");

        NDArray::prepareSpecialUse({&input}, {&updates, &indices});

        BUILD_DOUBLE_SELECTOR(xType, yType, scatterSimple_, (context, opId, input, updates, indices, dimensions), LIBND4J_TYPES, INTEGER_TYPES);

        NDArray::registerSpecialUse({&input}, {&updates, &indices});
    }




}
}
}

