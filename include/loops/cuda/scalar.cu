//
// @author raver119@gmail.com
//

#ifndef SCALAR_CU
#define SCALAR_CU

#include "../scalar.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <op_boilerplate.h>
#include <helpers/TAD.h>
#include <types/float16.h>



template <typename T, typename OpType>
__device__ void scalarAlongDimensionGeneric(T *x,
                                            int *xShapeInfo,
                                            T *extraParams,
                                            T *z,
                                            int *zShapeInfo,
                                            T *scalars,
                                            int *dimension,
                                            int dimensionLength,
                                            int *tadShapeInfo,
                                            Nd4jIndex *tadOffsets,
                                            int *tadShapeInfoZ,
                                            Nd4jIndex *tadOffsetsZ) {

    functions::scalar::ScalarTransform<T>::template transformCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

template <typename T, typename OpClass>
__device__ void scalarSimpleGeneric(
        Nd4jIndex n,
        T dx,
        T *dy,
        int incy, T *params,
        T *result,int resultStride, int *allocationBuffer) {

    functions::scalar::ScalarTransform<T>::template transformCuda<OpClass>(
            n,
            dx,
            dy,
            incy,
            params,
            result,
            resultStride,
            allocationBuffer,
            NULL);
}

/*
// LEGACY KERNELS,
template <typename T>
__device__ void scalarGenericIndexes(
        int opNum,
        Nd4jIndex n,
        T dx,
        T *dy,
        T *params,
        T *result,int *indexes, int *allocationBuffer) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);
    }
    __syncthreads();

    functions::scalar::ScalarTransform<T>::transform(
            opNum,
            n,
            dx,
            dy,
            params,
            result,
            indexes,
            allocationBuffer,
            manager);
}

__global__ void scalarDoubleIndexes(
        int opNum,
        Nd4jIndex n,
        double dx,
        double *dy,
        double *params,
        double *result,int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<double>(opNum,
                                 n,
                                 dx,
                                 dy,
                                 params,
                                 result,
                                 indexes, allocationBuffer);
}

__global__ void scalarFloatIndexes(
        int opNum,
        Nd4jIndex n,
        float dx,
        float *dy,
        float *params,
        float *result,
        int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<float>(opNum,
                                n,
                                dx,
                                dy,
                                params,
                                result,
                                indexes, allocationBuffer);
}
*/

template <typename T, typename OpClass>
__device__ void scalarSimpleGeneric(
        T dx,
        T *dy,
        int *xShapeInfo,
        T *params,
        T *result,
        int *resultShapeInfo,
        int *allocationBuffer) {

    functions::scalar::ScalarTransform<T>::template transformCuda<OpClass>(
            dx,
            dy,
            xShapeInfo,
            params,
            result,
            resultShapeInfo,
            allocationBuffer,
            NULL);
}



// ScalarOp Along Dimension kernels
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float, INPUT(float *x, int *xShapeInfo, float *extraParams, float *z, int *zShapeInfo, float *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, double, INPUT(double *x, int *xShapeInfo, double *extraParams, double *z, int *zShapeInfo, double *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *extraParams, float16 *z, int *zShapeInfo, float16 *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

// scalar shape
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float, INPUT(float dx, float *dy, int *xShapeInfo, float *params, float *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, double, INPUT(double dx, double *dy, int *xShapeInfo, double *params, double *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float16, INPUT(float16 dx, float16 *dy, int *xShapeInfo, float16 *params, float16 *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))

// scalar strided
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float, INPUT(Nd4jIndex n, float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, double, INPUT(Nd4jIndex n, double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float16, INPUT(Nd4jIndex n, float16 dx, float16 *dy, int incy, float16 *params, float16 *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))


namespace functions {
    namespace scalar {

    template<>
    void ScalarTransform<float>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, int xStride, float *result, int resultStride, float scalar, float *extraParams, Nd4jIndex n) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("F13 opNum:[%i]\n", opNum);

		int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(scalarSimpleStrided, float, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))
    }


    template<>
    void ScalarTransform<float16>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, int xStride, float16 *result, int resultStride, float16 scalar, float16 *extraParams, Nd4jIndex n) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("H13 opNum:[%i]\n", opNum);

		int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(scalarSimpleStrided, float16, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))
    }


    template<>
    void ScalarTransform<double>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, int xStride, double *result, int resultStride, double scalar, double *extraParams, Nd4jIndex n) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("D13 opNum:[%i]\n", opNum);

		int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(scalarSimpleStrided, double, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))
    }


    template<>
    void ScalarTransform<float16>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, int *xShapeInfo, float16 *result, int *resultShapeInfo, float16 scalar, float16 *extraParams) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("H14 opNum:[%i]\n", opNum);

		int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

        DISPATCH_SIMPLE(scalarSimpleShaped, float16, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    }

    template<>
    void ScalarTransform<float>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, int *xShapeInfo, float *result, int *resultShapeInfo, float scalar, float *extraParams) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("F14 opNum:[%i]\n", opNum);

        int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

        DISPATCH_SIMPLE(scalarSimpleShaped, float, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    }

    template<>
    void ScalarTransform<double>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, int *xShapeInfo, double *result, int *resultShapeInfo, double scalar, double *extraParams) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("D14 opNum:[%i]\n", opNum);

		int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

        DISPATCH_SIMPLE(scalarSimpleShaped, double, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    }

    template<>
    void ScalarTransform<double>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, double *x, int *xShapeInfo, double *z, int *zShapeInfo, double *scalars, double *extraParams, int *dimension, int dimensionLength) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
        Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
        int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
        Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

        DISPATCH_SIMPLE(scalarAlongDimension, double, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    }

    template<>
    void ScalarTransform<float>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, float *x, int *xShapeInfo, float *z, int *zShapeInfo, float *scalars, float *extraParams, int *dimension, int dimensionLength) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
        Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
        int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
        Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

        DISPATCH_SIMPLE(scalarAlongDimension, float, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    }

    template<>
    void ScalarTransform<float16>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, float16 *scalars, float16 *extraParams, int *dimension, int dimensionLength) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
        Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
        int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
        Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

        DISPATCH_SIMPLE(scalarAlongDimension, float16, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    }


        /**
     * Cuda implementation of transform
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    template<typename T>
    template<typename OpType>
    __device__ void ScalarTransform<T>::transform(
            Nd4jIndex n,
            T scalar,
            T *dy,
            T *params,
            T *result,
            int *indexes,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
        int totalThreads = gridDim.x * blockDim.x;
        int tid = threadIdx.x;
        Nd4jIndex i = blockIdx.x * blockDim.x + tid;

        /* equal, positive, non-unit increments. */
        for (; i < n; i+= totalThreads) {
            result[indexes[i]] = OpType::op(dy[indexes[i]],scalar, params);
        }
    }


    /**
     * Cuda implementation of transform
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
     template<typename T>
     template<typename OpType>
     __device__ void ScalarTransform<T>::transformCuda(
            T scalar,
            T *dy,
            int *shapeInfo,
            T *params,
            T *result,
            int *resultShapeInfo,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {

        int *xShape = shape::shapeOf(shapeInfo);
        int *xStride = shape::stride(shapeInfo);
        char xOrder = shape::order(shapeInfo);
        int xRank = shape::rank(shapeInfo);
        int xOffset = shape::offset(shapeInfo);
        int xElementWiseStride = shape::elementWiseStride(shapeInfo);
        int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
        int *zShape = shape::shapeOf(resultShapeInfo);
        int *zStride = shape::stride(resultShapeInfo);
        int zRank = shape::rank(resultShapeInfo);

        int totalThreads = gridDim.x * blockDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ int length;
        if(threadIdx.x == 0)
            length = shape::length(shapeInfo);
        __syncthreads();


        if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == shape::order(resultShapeInfo)) {
            transformCuda<OpType>(
                    length,
                    scalar,
                    dy,
                    xElementWiseStride,
                    params,
                    result,resultElementWiseStride, allocationBuffer, manager);
        }
        else {
            int xIdx[MAX_RANK];

            for (Nd4jIndex i = tid; i < length; i+= totalThreads) {
                shape::ind2sub(xRank, xShape, i,xIdx);
                int xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
                int resultOffset = shape::getOffset(0, zShape, zStride, xIdx, zRank);
                result[resultOffset] = OpType::op(dy[xOffset2],scalar, params);
            }
        }
    }
/**
  * ScalarOp along dimension
**/
    template<typename T>
    template<typename OpType>
    void __device__ ScalarTransform<T>::transformCuda(T *x,
                                  int *xShapeInfo,
                                  T *extraParams,
                                  T *z,
                                  int *zShapeInfo,
                                  T *scalars,
                                  int *dimension,
                                  int dimensionLength,
                                  int *tadShapeInfo,
                                  Nd4jIndex *tadOffsets,
                                  int *tadShapeInfoZ,
                                  Nd4jIndex *tadOffsetsZ) {


                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeInfo;
                    tadOffsetsZ = tadOffsets;
                }

                // tad preparation
                int tadEWS = shape::elementWiseStride(tadShapeInfo);
                int zEWS = shape::elementWiseStride(tadShapeInfo);
                int tadRank = shape::rank(tadShapeInfo);
                int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int numTads =shape::length(xShapeInfo) / tadLength;

                // main loop, rolling over tads
                for (int r = blockIdx.x; r < numTads; r+=gridDim.x) {
                    Nd4jIndex offset = tadOffsets[r];
                    Nd4jIndex offsetZ = tadOffsetsZ[r];
                    T scalar = scalars[r];

                    if (tadEWS >= 1 && zEWS >= 1) {
                        T *oZ = z + offsetZ;
                        T *oX = x + offset;

                       for (int f = threadIdx.x; f < tadLength; f+= blockDim.x) {
                            oZ[f] = OpType::op(oX[f], scalar, extraParams);
                        }
                    } else {
                        // ind2sub loop
                        printf("Super-bad loop visited. Shouldn't ever happen\n");
                    }
                }
    }
    /**
     *
     * @param n
     * @param idx
     * @param dx
     * @param dy
     * @param incy
     * @param params
     * @param result
     * @param blockSize
     */
        template<typename T>
        template<typename OpType>
        __device__ void ScalarTransform<T>::transformCuda(
            Nd4jIndex n,
            T dx,
            T *dy,
            int incy,
            T *params,
            T *result,
            int resultStride,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {

        int totalThreads = gridDim.x * blockDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        Nd4jIndex i = tid;
        if(incy == 1 && resultStride == 1) {
            for (; i < n; i += totalThreads) {
                result[i] = OpType::op(dy[i],dx, params);
            }
        }
        else {
            for (; i < n; i += totalThreads) {
                result[i * resultStride] = OpType::op(dy[i * incy],dx, params);
            }
        }
    }

/*
        static inline __device__ void transformCuda(
            const int opNum,
            T scalar,
            T *dy,
            int *shapeInfo,
            T *params,
            T *result,
            int *resultShapeInfo,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(scalar, dy, shapeInfo, params, result, resultShapeInfo, allocationBuffer, manager), SCALAR_OPS);
                    }


        static inline __device__ void transform(
            const int opNum,
            Nd4jIndex n,
            T scalar,
            T *dy,
            T *params,
            T *result,
            int *indexes,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transform, PARAMS(n, scalar, dy, params, result, indexes, allocationBuffer, manager), SCALAR_OPS);
        }


        static inline __device__ void transformCuda(
            const int opNum,
            Nd4jIndex n,
            T dx,
            T *dy,
            int incy,
            T *params,
            T *result,
            int resultStride,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer, manager), SCALAR_OPS);
        }
        */

        BUILD_CALL_1(template __device__ void ScalarTransform<float>::transformCuda, float, (float, float*, int *, float*, float*, int*, int*, UnifiedSharedMemory *), SCALAR_OPS)
        BUILD_CALL_1(template __device__ void ScalarTransform<float16>::transformCuda, float16, (float16, float16*, int *, float16*, float16*, int*, int*, UnifiedSharedMemory *), SCALAR_OPS)
        BUILD_CALL_1(template __device__ void ScalarTransform<double>::transformCuda, double, (double, double*, int *, double*, double*, int*, int*, UnifiedSharedMemory *), SCALAR_OPS)
    }
}



#endif // SCALAR_CU