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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#include <loops/reduce_same.h>
#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>
#include <types/types.h>

using namespace simdOps;


template <typename X, typename OpType>
__device__ void reduceSimpleGeneric(void *x,  Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    void *z, Nd4jLong *zShapeInfo,
                                    int *dimension, int dimensionLength,
                                    void *reductionBuffer, 
                                    Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceSameFunction<X>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }

    __syncthreads();

    functions::reduce::ReduceSameFunction<X>::template transformCudaXD<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
}

// template <typename X, typename Z, typename OpType>
// __device__ void reduceScalarGeneric(
//         void *x,
//         Nd4jLong *xShapeInfo,
//         void *extraParams,
//         void *z,
//         Nd4jLong *zShapeInfo,
//         int *dimension,
//         int dimensionLength,
//         void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

//     __shared__ UnifiedSharedMemory *manager;

//     if (threadIdx.x == 0) {
//         extern __shared__ unsigned char shmem[];
//         manager = new(shmem) UnifiedSharedMemory((int *) shmem);
//         manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceSameFunction<X,Z>), sizeof(shape::TAD), 0);
//     }
//     __syncthreads();

//     functions::reduce::ReduceSameFunction<X, Z>::template execScalarCuda<OpType>(
//             x,
//             xShapeInfo,
//             extraParams,
//             z,
//             zShapeInfo,
//             reductionBuffer,
//             manager,
//             tadOnlyShapeInfo);
// };

//     template <typename X, typename Z, typename OpType>
//     __global__ void simpleScalar(
//         void *x,
//         Nd4jLong *xShapeInfo,
//         void *extraParams,
//         void *z,
//         Nd4jLong *zShapeInfo,
//         int *dimension,
//         int dimensionLength,
//         void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {
//             reduceScalarGeneric<X, Z, OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo);
//         }

// // reduceScalar
// DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float, INPUT(float *x, Nd4jLong *xShapeInfo, float *extraParams, float *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
// DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, double, INPUT(double *x, Nd4jLong *xShapeInfo, double *extraParams, double *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
// DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float16, INPUT(float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, float16 *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))

  // template <typename X, typename Z, typename OpType>
  // __global__ void simpleReduce(
  //   void *x,
  //   Nd4jLong *xShapeInfo,
  //   void *extraParams,
  //   void *z,
  //   Nd4jLong *zShapeInfo,
  //   int *dimension,
  //   int dimensionLength,
  //   void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
  //     reduceSimpleGeneric<X, Z, OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
  // }



// // reduceXD
// DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float, INPUT(float *x, Nd4jLong *xShape, float *extraParams, float *z, Nd4jLong *zShape, int *dimension, int dimensionLength, float *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
// DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, double, INPUT(double *x, Nd4jLong *xShape, double *extraParams, double *z, Nd4jLong *zShape, int *dimension, int dimensionLength, double *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
// DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float16, INPUT(float16 *x, Nd4jLong *xShape, float16 *extraParams, float16 *z, Nd4jLong *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))


namespace functions {
namespace reduce    {

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ReduceSameFunction<X>::aggregatePartials(void *vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams) {
    
    // start the shared memory loop on the next power of 2 less
    // than the block size.  If block size is not a power of 2,
    // accumulate the intermediate sums in the remainder range.
    
    auto sPartials = static_cast<X*>(vsPartials);
    auto extraParams = static_cast<X*>(vextraParams);

    Nd4jLong floorPow2 = numItems;

    if (floorPow2 & (floorPow2 - 1)) {
        
        while (floorPow2 & (floorPow2 - 1)) 
            floorPow2 &= floorPow2 - 1;
        
        if (tid >= floorPow2) 
            sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);

        __syncthreads();
    }

    for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
          if (tid < activeThreads && tid + activeThreads < numItems) 
            sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
          
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ReduceSameFunction<X>::transformCudaXD( void *vx, Nd4jLong *xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong *zShapeInfo,
                                                        int *dimension,  int dimensionLength,
                                                        void *vreductionBuffer, 
                                                        UnifiedSharedMemory *manager,
                                                        Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<X*>(vx);
    auto z = reinterpret_cast<X*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);
    auto reductionBuffer = reinterpret_cast<X*>(vreductionBuffer);

    if (OpType::requiresSpecialAccumulation) {
        OpType::execSpecialCuda(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
        return;
    }

    //shared memory space for storing intermediate results
    __shared__ X* sPartials;

    //  __shared__ shape::TAD *tad;
    __shared__ int tadLength;
    __shared__ int tadRank;
    __shared__ int numTads;
    __shared__ Nd4jLong *tadShape;
    __shared__ Nd4jLong *tadStride;
    
    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<X*>(shmem);
        tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadRank = shape::rank(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        tadShape = shape::shapeOf(tadOnlyShapeInfo);
        tadStride = shape::stride(tadOnlyShapeInfo);
    }
    __syncthreads();
    
    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        
        Nd4jLong tadOffsetForBlock = tadOffsets[r];
        sPartials[threadIdx.x] = OpType::startingValue(x + tadOffsetForBlock);

          for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
            
            auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo, tadLength);
            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[xOffset], extraParams), extraParams);
          }
          __syncthreads();

          // aggregate. do NOT reduce for elements > tadLength
          aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

          __syncthreads();

          if (threadIdx.x == 0)
            z[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
    }
}


      // template <typename X, typename Z>
      // template<typename OpType>
      // __host__ void ReduceSameFunction<X,Z>::intermediateXD(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShape, void *extraParams, void *z, Nd4jLong *zShape, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
      //   simpleReduce<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets);
      // }

      //       template <typename X, typename Z>
      //       template<typename OpType>
      //       __host__ void ReduceSameFunction<X,Z>::intermediateScalar(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {
      //           simpleScalar<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo);
      //       }

      // template <typename X, typename Y>
      //       _CUDA_H void ReduceSameFunction<X,Y>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {
      //           DISPATCH_BY_OPNUM_TT(intermediateScalar, PARAMS(launchDims, stream, x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_FLOAT_OPS));

      //   nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarFloat(...) failed");
      //       }



      //       template <typename X, typename Y>
      //       _CUDA_H void ReduceSameFunction<X, Y>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, void *x, Nd4jLong *xShape, void *extraParams, void *z, Nd4jLong *zShape, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
      //           DISPATCH_BY_OPNUM_TT(intermediateXD, PARAMS(launchDims, stream, x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_FLOAT_OPS));

      //           DEBUG_KERNEL(stream, opNum);
      //       }

      //       template <typename T>
      //       __device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize) {
      //           int sPartialsLength = sMemSize / sizeof(T);
      //           T *sPartialsDeref = (T *) *sPartials;
      //           for (int i = 0; i < sPartialsLength; i++) {
      //               sPartialsDeref[i] = extraParams[0];
      //           }
      //       }

      //       template <typename X, typename Z>
      //       template <typename OpType>
      // __device__ void ReduceSameFunction<X,Z>::execScalarCuda(
      //   void *vx,
      //   Nd4jLong *xShapeInfo,
      //   void *vextraParams,
      //   void *vz,
      //   Nd4jLong *zShapeInfo,
      //   void *vreductionBuffer,
      //   UnifiedSharedMemory *manager,
      //   Nd4jLong *tadOnlyShapeInfo) {

      //           auto x = reinterpret_cast<X*>(vx);
      //           auto z = reinterpret_cast<X*>(vz);
      //           auto extraParams = reinterpret_cast<X*>(vextraParams);
      //           auto reductionBuffer = reinterpret_cast<X*>(vreductionBuffer);

      //   int elementWiseStride = shape::elementWiseStride(xShapeInfo);

      //   auto n = shape::length(xShapeInfo);

      //   auto tid = blockDim.x * blockIdx.x + threadIdx.x;

      //   //shared memory space for storing intermediate results
      //   Z *sPartials = reinterpret_cast<X*>(manager->getSharedReductionBuffer());

      //   sPartials[threadIdx.x] = OpType::startingValue(x);

      //   if (elementWiseStride >= 1) {
      //     for (int i = tid; i < n; i += (blockDim.x * gridDim.x)) {
      //       sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[i * elementWiseStride], extraParams), extraParams);
      //     }
      //   }
      //   else {
      //       __shared__ int rank;
      //       __shared__ Nd4jLong *xShape;
      //       __shared__ Nd4jLong *xStride;
      //       if (threadIdx.x == 0) {
      //                   rank = shape::rank(xShapeInfo);
      //                   xShape = shape::shapeOf(xShapeInfo);
      //                   xStride = shape::stride(xShapeInfo);
      //       }
      //       __syncthreads();

      //     Nd4jLong ind2sub[MAX_RANK];

      //     for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
      //       shape::ind2subC(rank, xShape, i, n, ind2sub);

      //       auto offset = shape::getOffset(0, xShape, xStride, ind2sub, rank);
      //       sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[offset], extraParams), extraParams);
      //     }
      //   }

      //   __syncthreads();
      //   aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), extraParams);


      //   __syncthreads();

      //   if (gridDim.x > 1) {
      //     unsigned int *tc = (unsigned int *)reductionBuffer;
      //     __shared__ bool amLast;
      //     tid = threadIdx.x;
      //     if (threadIdx.x == 0) {
      //       reductionBuffer[blockIdx.x] = sPartials[0];//this->postProcess(sPartials[0],n,extraParams);
      //     }
      //     __threadfence();
      //     __syncthreads();

      //     if (threadIdx.x == 0) {
      //       unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
      //       amLast = (ticket == gridDim.x - 1);
      //     }

      //     __syncthreads();

      //     if (amLast) {
      //       tc[16384] = 0;

      //       sPartials[threadIdx.x] = OpType::startingValue(x);

      //       for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      //         sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraParams);
      //       }
      //       __syncthreads();



      //       aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraParams);

      //       __syncthreads();
      //       if (threadIdx.x == 0) {
      //         z[0] = OpType::postProcess(sPartials[0], n, extraParams);
      //       }
      //     }
      //   }
      //   else {
      //     if (threadIdx.x == 0) {
      //       unsigned int *tc = (unsigned *)reductionBuffer;
      //       tc[16384] = 0;
      //       z[0] = OpType::postProcess(sPartials[0], n, extraParams);
      //     }
      //   }
      // }




            

        
        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ReduceSameFunction, , LIBND4J_TYPES);
}
}