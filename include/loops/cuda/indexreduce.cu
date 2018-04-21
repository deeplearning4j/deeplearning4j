//
// Created by raver on 4/9/2018.
//

#include <Environment.h>
#include "../indexreduce.h"
#include <op_boilerplate.h>
#include <helpers/DebugHelper.h>

#include "../legacy_ops.h"


template <typename T>
static __device__ void indexReduceGeneric(
        const int op,
        T *dx,
        int *xShapeInfo, int xRank,
        T *extraParams,
        T *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::indexreduce::IndexReduce<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

    functions::indexreduce::IndexReduce<T>::transform(
            op,
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot,
            allocationBuffer,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo,
            tadOffsets);
}

__global__ void indexReduceDouble(
        int op,
        double *dx,
        int *xShapeInfo, int xRank,
        double *extraParams,
        double *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<double>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void indexReduceFloat(
        int op,
        float *dx,
        int *xShapeInfo, int xRank,
        float *extraParams,
        float *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot,  int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<float>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void indexReduceHalf(
        int op,
        float16 *dx,
        int *xShapeInfo, int xRank,
        float16 *extraParams,
        float16 *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot,  int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<float16>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}


namespace functions {
    namespace indexreduce {

        template <>
        _CUDA_H void IndexReduce<float>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int opNum, float *dx, int *xShapeInfo, int xRank, float *extraParams, float *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

            indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			nullptr, 0,
			nullptr,
			1,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            checkCudaErrors(cudaStreamSynchronize(*stream));
            nd4j::DebugHelper::checkErrorCode(stream, "execIndexReduceScalarFloat(...) failed");
        }

        template <>
        _CUDA_H void IndexReduce<double>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int opNum, double *dx, int *xShapeInfo, int xRank, double *extraParams, double *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

            indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			nullptr, 0,
			nullptr,
			1,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            nd4j::DebugHelper::checkErrorCode(stream, "execIndexReduceScalarDouble(...) failed");
        }

        template <>
        _CUDA_H void IndexReduce<float16>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int opNum, float16 *dx, int *xShapeInfo, int xRank, float16 *extraParams, float16 *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
            
            indexReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			nullptr, 0,
			nullptr,
			1,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            nd4j::DebugHelper::checkErrorCode(stream, "execIndexReduceScalarHalf(...) failed");
        }


        template <>
        _CUDA_H void IndexReduce<float>::executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int opNum, float *dx, int *xShapeInfo, int xRank, float *extraParams, float *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

            indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void IndexReduce<double>::executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int opNum, double *dx, int *xShapeInfo, int xRank, double *extraParams, double *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

            indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void IndexReduce<float16>::executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int opNum, float16 *dx, int *xShapeInfo, int xRank, float16 *extraParams, float16 *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
            
            indexReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, xRank,
			extraParams,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength,
			1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


        // This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
        template<typename T>
        struct SharedIndexValue {
            // Ensure that we won't compile any un-specialized types
            __device__ T * getPointer() {
                extern __device__ void error(void);
                error();
                return 0;
            }
        };

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

        template<>
        struct SharedIndexValue<float> {
            __device__ IndexValue<float> * getPointer() {
                extern __shared__ IndexValue<float> s_int2[];
                return s_int2;
            }
        };
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

        template<>
        struct SharedIndexValue<double> {
            __device__ IndexValue<double> * getPointer() {
                extern __shared__ IndexValue<double> s_int6[];
                return s_int6;
            }
        };

        template <typename T>
        template <typename OpType>
        __device__ void IndexReduce<T>::aggregatePartials(IndexValue<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.
            IndexValue<T> *sPartials = *sPartialsRef;
            int floorPow2 = blockDim.x;

            if (floorPow2 & (floorPow2 - 1)) {
                while ( floorPow2 & (floorPow2 - 1) ) {
                    floorPow2 &= floorPow2 - 1;
                }

                if (tid >= floorPow2) {
                    IndexValue<T> prev = sPartials[tid - floorPow2];
                    IndexValue<T> curr = sPartials[tid];
                    sPartials[tid - floorPow2] = OpType::update(prev,curr,extraParams);
                }
                __syncthreads();
            }

            for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads && tid + activeThreads < numElements) {
                    IndexValue<T> curr = sPartials[tid];
                    IndexValue<T> next = sPartials[tid + activeThreads];
                    sPartials[tid] = OpType::update(curr,next,extraParams);
                }
                __syncthreads();
            }
        }

        template <typename T>
        __device__ void IndexReduce<T>::transform(
                const int opNum,
                T *x,
                int *xShapeInfo,
                T *extraParams,
                T *result,
                int *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationBuffer,
                T *reductionBuffer,
                UnifiedSharedMemory *manager,
                int *tadShapeInfo,
                Nd4jIndex *tadOffset) {
            DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, manager, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
        }


        template <typename T>
        template <typename OpType>
        __device__ void IndexReduce<T>::transform(
                T *dx,
                int *xShapeInfo,
                T *extraParams,
                T *result,
                int *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationBuffer,
                T *reductionBuffer,
                UnifiedSharedMemory *manager,
                int *tadOnlyShapeInfo,
                Nd4jIndex *tadOffsets){
            /**
             * Gpu information for the problem
             */
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ volatile int resultScalar;

            //shared memory space for storing intermediate results
            IndexValue<T> *sPartials;


            sPartials = (IndexValue<T> *)manager->getSharedReductionBuffer(); //holder.getPointer();
//		T startingVal = OpType::startingValue(dx);


            //	IndexValue <T> val = {startingVal, threadIdx.x};
            sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

            //length for the tad
            __shared__ volatile Nd4jIndex xLength;

            __shared__ volatile Nd4jIndex resultLength;



            //only compute the tad indexes once
            IndexValue <T> reduction = OpType::startingIndexValue(dx);

            if (threadIdx.x == 0) {
                if (resultShapeInfo != nullptr)
                    resultLength = shape::length(resultShapeInfo);
                else resultLength = 1;

                if (dimensionLength == 1) {
                    if (dimension == nullptr || dimension[0] == MAX_DIMENSION)
                        resultScalar = 1;
                    else
                        resultScalar = 0;
                }
                else
                    resultScalar = 0;

                if (resultLength == 1)
                    resultScalar = 1;

                //	xElementWiseStride = shape::elementWiseStride(xShapeInfo);

                xLength = shape::length(xShapeInfo);
            }
            __syncthreads();

            if (!resultScalar) {

                __shared__ Nd4jIndex tadLength;
                __shared__ int tadEWS;
                __shared__ int tadRank;
                __shared__ int numTads;
                __shared__ int *tadShape;
                __shared__ int *tadStride;
                __shared__ char tadOrder;

                if (threadIdx.x == 0) {
                    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                    tadRank = shape::rank(tadOnlyShapeInfo);
                    numTads = shape::length(xShapeInfo) / tadLength;

                    tadShape = shape::shapeOf(tadOnlyShapeInfo);
                    tadStride = shape::stride(tadOnlyShapeInfo);
                    tadOrder = shape::order(tadOnlyShapeInfo);
                }
                __syncthreads();

                if (dimensionLength > 1 || tadEWS < 1) {
                    int xCoord[MAX_RANK];

                    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
                        Nd4jIndex tadOffsetForBlock = tadOffsets[r];

                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for(int i = threadIdx.x;i < tadLength; i += blockDim.x) {
                            shape::ind2subC(tadRank,tadShape, i, xCoord);

                            Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                            IndexValue<T> comp {dx[xOffset], i};

                            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], comp, extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[r] = (T) sPartials[threadIdx.x].index;
                        }
                    }
                } else {

                    for(int i = blockIdx.x; i < numTads; i+= gridDim.x) {
                        Nd4jIndex tadOffsetForBlock = tadOffsets[i];

                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for (int x = threadIdx.x; x < tadLength; x+= blockDim.x) {
                            IndexValue<T> comp {dx[tadOffsetForBlock + x * tadEWS], x};
                            sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], comp, extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[i] = (T)  sPartials[threadIdx.x].index; //postProcess(sPartials[0],tadLength ,extraParams);
                        }
                    }
                }
            }


                //reduce to 1 result
            else if (resultScalar) {
                Nd4jIndex n = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);

                if(xElementWiseStride >= 1) {
                    for(Nd4jIndex i = tid;i < n; i += (blockDim.x * gridDim.x)) {
                        IndexValue <T> indexVal = {dx[i * xElementWiseStride], i};
                        reduction = OpType::update(reduction, indexVal, extraParams);
                    }
                } else {
                    int rank = shape::rank(xShapeInfo);
                    int ind2sub[MAX_RANK];

                    for(Nd4jIndex i = tid;i < n; i += blockDim.x * gridDim.x) {
                        shape::ind2subC(rank,shape::shapeOf(xShapeInfo),i,ind2sub);

                        Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),ind2sub,rank);
                        IndexValue <T> indexVal = {dx[offset], i};
                        reduction = OpType::update(reduction, indexVal, extraParams);
                    }
                }


                sPartials[threadIdx.x] = reduction;
                __syncthreads();

                aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, (int) n),extraParams);
                __syncthreads();

                if (gridDim.x > 1) {
                    __shared__ bool amLast;
                    unsigned int *tc = (unsigned int *) reductionBuffer;
                    int rank = shape::rank(xShapeInfo);
                    tid = threadIdx.x;
                    if (threadIdx.x == 0) {
                        IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;
                        pBuffer[blockIdx.x] = {sPartials[0].value, sPartials[0].index};
                    }
                    __threadfence();
                    __syncthreads();

                    if (tid==0) {
                        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                        amLast = (ticket == gridDim.x-1);
                    }

                    __syncthreads();

                    if (amLast) {
                        tc[16384] = 0;
                        IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;


                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for (Nd4jIndex i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
                            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], pBuffer[i], extraParams);
                        }



                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x),extraParams);

                        __syncthreads();
                        if (tid == 0) {
                            result[0] = (T)  sPartials[0].index;
                        }
                    }
                } else {
                    if (tid == 0) {
                        unsigned int *tc = (unsigned *) reductionBuffer;
                        tc[16384] = 0;
                        result[0] = (T) sPartials[0].index;
                    }
                }
            }
        }
    }
}



