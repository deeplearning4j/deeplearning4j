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
        __syncthreads();
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
BUILD_DOUBLE_TEMPLATE(template void clipByNormBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const Nd4jLong* xTadOffsets, const void *vy, const Nd4jLong *yShapeInfo, const Nd4jLong* yTadOffsets, void *vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets, void* vreducBuff, const double clipNormVal), FLOAT_TYPES, FLOAT_TYPES);

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
        BUILD_DOUBLE_SELECTOR(xType, zType, clipByNormBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), nullptr, gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), nullptr, gradI.getSpecialBuffer(), gradI.getSpecialShapeInfo(), nullptr, context->getReductionPointer(), clipNormVal), FLOAT_TYPES, FLOAT_TYPES);
    }
    else {  // means tads using

        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimensions);
        auto packY = ConstantTadHelper::getInstance()->tadForDimensions(gradO.getShapeInfo(), dimensions);
        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(gradI.getShapeInfo(), dimensions);

        const int blocksPerGrid = packX.numberOfTads();
        BUILD_DOUBLE_SELECTOR(xType, zType, clipByNormBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), gradO.getSpecialBuffer(), packY.platformShapeInfo(), packY.platformOffsets(), gradI.getSpecialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), nullptr, clipNormVal), FLOAT_TYPES, FLOAT_TYPES);
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
    void clipByGlobalNorm_(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        NDArray globalNorm = NDArrayFactory::create<T>(0, inputs[0]->getContext()); //sqrt(sum([l2norm(t)**2 for t in t_list]))

        for (auto i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto l2norm = input->reduceNumber(reduce::Norm2);
            globalNorm += l2norm * l2norm;
        }

        globalNorm.applyTransform(transform::Sqrt, nullptr, nullptr);// = nd4j::math::nd4j_sqrt(globalNorm);
        outputs[inputs.size()]->p(0, globalNorm);
        globalNorm.syncToHost();
        const T factor = clipNorm / globalNorm.e<T>(0);

        for (size_t e = 0; e < inputs.size(); e++) {
            // all-reduce
            auto input = inputs[e];
            auto output = outputs[e];

            if (globalNorm.e<double>(0) <= clipNorm) {
                output->assign(input);
            }
            else {

                auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
                input->applyLambda(lambda, output);
            }
        }
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
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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

}
}
}

