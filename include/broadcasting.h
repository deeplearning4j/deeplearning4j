/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
#include <dll.h>
#include <sharedmem.h>
#include <shape.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <pairwise_util.h>
#include <ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif

#define BROADCAST_OPS \
       (0, simdOps::Add), \
       (1, simdOps::Subtract), \
       (2, simdOps::Multiply), \
       (3, simdOps::Divide), \
       (4, simdOps::ReverseDivide), \
       (5, simdOps::ReverseSubtract), \
       (6, simdOps::Copy)

  
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
			int dimensionLength, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ) {

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
      __shared__ int yStride;
      __shared__ int zEWS;
      __shared__ int zRank;
      __shared__ int *zShape;
      __shared__ int *zStride;
      if (threadIdx.x == 0) {
        if (tadOnlyShapeInfoZ == nullptr) {
            tadOnlyShapeInfoZ = tadOnlyShapeInfo;
            tadOffsetsZ = tadOffsets;
        }

   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        yStride = shape::elementWiseStride(yShapeInfo);
      	zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);

        if (tadEWS < 1 || zEWS < 1) {
            tadRank = shape::rank(tadOnlyShapeInfo);
            zRank = shape::rank(tadOnlyShapeInfoZ);
            tadShape = shape::shapeOf(tadOnlyShapeInfo);
      	    tadStride = shape::stride(tadOnlyShapeInfo);
      	    zShape = shape::shapeOf(tadOnlyShapeInfoZ);
      	    zStride = shape::stride(tadOnlyShapeInfoZ);
        }
      }
      __syncthreads();

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {


            __shared__ int tadOffsetForBlock;
            __shared__ int tadOffsetForBlockZ;
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


            if(tadEWS > 0 && zEWS > 0) {
            	if (tadEWS == 1 && yStride == 1 && zEWS == 1) {
                	for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i] = OpType::op(rX[i], y[i]);
                	}
                } else {
					for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yStride]);
                	}
                }
            }
            else {
                int xCoord[MAX_RANK];
                int zCoord[MAX_RANK];

                for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    shape::ind2subC(tadRank,tadShape, i, xCoord);
                    shape::ind2subC(zRank,zShape, i, zCoord);
                    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    Nd4jIndex zOffset = shape::getOffset(tadOffsetForBlockZ, zShape, zStride, zCoord, zRank);
                    result[zOffset] = OpType::op(x[xOffset], y[i * yStride]);
                }
            }
		}
	}

#endif

			static void exec(const int opNum, T *x,
				int *xShapeInfo,
				T *y,
				int *yShapeInfo,
				T *result,
				int *dimension,
				int dimensionLength, int *tadShapeInfo, int *tadOffset, int *tadShapeInfoZ, int *tadOffsetZ) {
                                DISPATCH_BY_OPNUM(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, result, dimension, dimensionLength, tadShapeInfo, tadOffset, tadShapeInfoZ, tadOffsetZ), BROADCAST_OPS);
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
							 int *dimension,
							 int dimensionLength, int *tadShapeInfo, int *tadOffset, int *tadShapeInfoZ, int *tadOffsetZ) {

				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				int *tadShapeShapeInfo =  tadShapeInfo;
				int *tadOffsets = tadOffset;
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
				int tads =shape::length(xShapeInfo) / tadLength;

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
						int offset = tadOffsets[i];
                        int offsetZ = tadOffsetZ[i];


						if (tadEWS > 0 && yStride > 0 && zEWS > 0) {


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
						} else {
							int *zShape = shape::shapeOf(tadShapeInfoZ);
							int *zStride = shape::stride(tadShapeInfoZ);
							int *xShape = shape::shapeOf(tadShapeShapeInfo);
							int *xStride = shape::stride(tadShapeShapeInfo);
							int zRank = shape::rank(tadShapeInfoZ);
							int tadRank = shape::rank(tadShapeShapeInfo);

                            int xCoord[MAX_RANK];
                            int zCoord[MAX_RANK];

// all this stuff already happens within thread
							for (int f = 0; f < tadLength; f++) {

                                shape::ind2subC(tadRank,xShape, i, xCoord);
                                shape::ind2subC(zRank,zShape, i, zCoord);
                                Nd4jIndex xOffset = shape::getOffset(offset, xShape, xStride, xCoord, tadRank);
                                Nd4jIndex zOffset = shape::getOffset(offsetZ, zShape, zStride, zCoord, zRank);
                                result[zOffset] = OpType::op(x[xOffset], y[i * yStride]);
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
		int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ) {


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
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float, INPUT(float *x, int *xShapeInfo, float *y, int *yShapeInfo, float *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, double, INPUT(double *x, int *xShapeInfo, double *y, int *yShapeInfo, double *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *y, int *yShapeInfo, float16 *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

#endif



#endif /* BROADCASTING_H_ */
