/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
#include <dll.h>
#include <helpers/sharedmem.h>
#include <helpers/shape.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <pairwise_util.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif

#include <TAD.h>

#include "legacy_ops.h"

namespace functions {
    namespace broadcast {

/**
 * Broadcast operation
 * for broadcasting a smaller tensor
 * along long a bigger one.
 */
        template<typename T>
        class Broadcast {
        public:

#ifdef __CUDACC__

            template<typename OpType>
			static __inline__ __device__ void transformCuda(
			T *x,
			int *xShapeInfo,
			T *y,
			int *yShapeInfo,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {

		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ int tadLength;
      __shared__ int tadEWS;
      __shared__ int tadRank;
      __shared__ int numTads;
      __shared__ int *tadShape;
      __shared__ int *tadStride;
      __shared__ int yEWS;
      __shared__ int zEWS;
      __shared__ int zRank;
      __shared__ int *zShape;
      __shared__ int *zStride;
      __shared__ int yRank;
      __shared__ int *yShape;
      __shared__ int *yStride;
      if (threadIdx.x == 0) {
        if (tadOnlyShapeInfoZ == nullptr) {
            tadOnlyShapeInfoZ = tadOnlyShapeInfo;
            tadOffsetsZ = tadOffsets;
        }

   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        yEWS = shape::elementWiseStride(yShapeInfo);
      	zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);

        if (tadEWS < 1 || zEWS < 1 || yEWS < 1 || dimensionLength > 1) {
            tadRank = shape::rank(tadOnlyShapeInfo);
            tadShape = shape::shapeOf(tadOnlyShapeInfo);
      	    tadStride = shape::stride(tadOnlyShapeInfo);
      	    zRank = shape::rank(tadOnlyShapeInfoZ);
      	    zShape = shape::shapeOf(tadOnlyShapeInfoZ);
      	    zStride = shape::stride(tadOnlyShapeInfoZ);
      	    yRank = shape::rank(yShapeInfo);
      	    yShape = shape::shapeOf(yShapeInfo);
      	    yStride = shape::stride(yShapeInfo);
        }
      }
      __syncthreads();

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {


            __shared__ Nd4jIndex tadOffsetForBlock;
            __shared__ Nd4jIndex tadOffsetForBlockZ;
            __shared__ T *rR;
            __shared__ T *rX;
            if (threadIdx.x == 0) {
                tadOffsetForBlockZ = tadOffsetsZ[r];
                if (result != x)
                    tadOffsetForBlock = tadOffsets[r];
                else
                    tadOffsetForBlock = tadOffsetForBlockZ;

                rR = result + tadOffsetForBlockZ;
                rX = x + tadOffsetForBlock;
            }
            __syncthreads();


            if(tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {
            	if (tadEWS == 1 && yEWS == 1 && zEWS == 1) {
                	for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i] = OpType::op(rX[i], y[i]);
                	}
                } else {
					for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS]);
                	}
                }
            }
            else {
                int xCoord[MAX_RANK];
                int yCoord[MAX_RANK];
                int zCoord[MAX_RANK];

                for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {

                    if (shape::order(tadOnlyShapeInfo) == 'c') {
                        shape::ind2subC(tadRank,tadShape, i, xCoord);
                        shape::ind2subC(yRank, yShape, i, yCoord);
                    } else {
                        shape::ind2sub(tadRank,tadShape, i, xCoord);
                        shape::ind2sub(yRank, yShape, i, yCoord);
                    }

                    if (shape::order(tadOnlyShapeInfoZ) == 'c')
                        shape::ind2subC(zRank,zShape, i, zCoord);
                    else
                        shape::ind2sub(zRank,zShape, i, zCoord);

                    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    Nd4jIndex zOffset = shape::getOffset(tadOffsetForBlockZ, zShape, zStride, zCoord, zRank);
                    Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                    result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            }
		}
	}

#endif

            static void exec(const int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               result,
                                               resultShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset,
                                               tadShapeInfoZ,
                                               tadOffsetZ), BROADCAST_OPS);
            }

            /**
             * CPU execution
             * @param x the input
             * @param xShapeInfo the x shape information
             * @param y the y data
             * @param yShapeInfo the y shape information
             * @param result the result
             * @param resultShapeInfo the result shape information
             * @param dimension the dimension to broadcast along long
             * @param dimensionLength the length of the dimension buffer
             */
            template<typename OpType>
            static void exec(T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ) {


                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                int *tadShapeShapeInfo = tadShapeInfo;
                Nd4jIndex *tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    tadShapeShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }

                //int *resultStride = shape::stride(tadShapeShapeInfo);
                int tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int yStride = shape::elementWiseStride(yShapeInfo);
                int tads = shape::length(xShapeInfo) / tadLength;

                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeShapeInfo;
                    tadOffsetZ = tadOffsets;
                }

                int zEWS = shape::elementWiseStride(tadShapeInfoZ);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(_threads) if (_threads > 1) proc_bind(AFFINITY) default(shared)
                for (int i = 0; i < tads; i++) {
                    Nd4jIndex offset = tadOffsets[i];
                    Nd4jIndex offsetZ = tadOffsetZ[i];
//                    printf("Tad: [%i]; Offset: [%lld]; OffsetZ: [%lld];\n", i, offset, offsetZ);


                    if (tadEWS > 0 && yStride > 0 && zEWS > 0 && dimensionLength == 1) {
                        T *oRes = result + offsetZ;
                        T *oX = x + offset;

                        if (tadEWS == 1 && yStride == 1 && zEWS == 1) {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f] = OpType::op(oX[f], y[f]);
                            }
                        } else {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f * zEWS] = OpType::op(oX[f * tadEWS], y[f * yStride]);
                            }
                        }
                    }
                    else {
                        int *zShape = shape::shapeOf(tadShapeInfoZ);
                        int *zStride = shape::stride(tadShapeInfoZ);
                        int zRank = shape::rank(tadShapeInfoZ);

                        int *xShape = shape::shapeOf(tadShapeShapeInfo);
                        int *xStride = shape::stride(tadShapeShapeInfo);
                        int xRank = shape::rank(tadShapeShapeInfo);

                        int *yShape = shape::shapeOf(yShapeInfo);
                        int *yStride = shape::stride(yShapeInfo);
                        int yRank = shape::rank(yShapeInfo);

                        int xCoord[MAX_RANK];
                        int yCoord[MAX_RANK];
                        int zCoord[MAX_RANK];


// TODO: cover this codebranch with tests
// all this stuff already happens within thread
                        for (int f = 0; f < tadLength; f++) {
                            if (shape::order(tadShapeShapeInfo) == 'c') {
                                shape::ind2subC(xRank, xShape, f, xCoord);
                                shape::ind2subC(yRank, yShape, f, yCoord);
                            } else {
                                shape::ind2sub(xRank, xShape, f, xCoord);
                                shape::ind2sub(yRank, yShape, f, yCoord);
                            }

                            if (shape::order(tadShapeInfoZ) == 'c')
                                shape::ind2subC(zRank, zShape, f, zCoord);
                            else
                                shape::ind2sub(zRank, zShape, f, zCoord);

                            Nd4jIndex xOffset = shape::getOffset(offset, xShape, xStride, xCoord, xRank);
                            Nd4jIndex zOffset = shape::getOffset(offsetZ, zShape, zStride, zCoord, zRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                            result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                        }
                    }
                }

                if (tad != nullptr)
                    delete tad;
            }
        };
    }
}

#ifdef __CUDACC__

/**
 * Meant to be called from an external interface
 * and the driver api
 * @param opNum the op number to execute
 * @param x the input data
 * @param xShapeInfo the x shape info for input
 * @param y the y to broadcast
 * @param yShapeInfo the shape information of the broadcast info
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result buffer
 * @param dimension the dimension(s) to do broadcast along long
 * @param dimensionLength the length of the dimension buffer
 * @param gpuInformation the gpu information such as blockdim,griddim and shared
 * memory size
 */
template <typename T, typename OpClass>
__device__ void broadcastSimpleGeneric(
		T *x,
		int *xShapeInfo,
		T *y,
		int *yShapeInfo,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {


	functions::broadcast::Broadcast<T>::template transformCuda<OpClass>(
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			NULL,
			tadOnlyShapeInfo,
			tadOffsets,
			tadOnlyShapeInfoZ,
			tadOffsetsZ);
}

// broadcast kernel call
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float, INPUT(float *x, int *xShapeInfo, float *y, int *yShapeInfo, float *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, double, INPUT(double *x, int *xShapeInfo, double *y, int *yShapeInfo, double *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *y, int *yShapeInfo, float16 *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

#endif



#endif /* BROADCASTING_H_ */
