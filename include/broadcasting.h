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
			int dimensionLength, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, int *tadOffsets) {

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
      if (threadIdx.x == 0) {
   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        tadRank = shape::rank(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;

        tadShape = shape::shapeOf(tadOnlyShapeInfo);
      	tadStride = shape::stride(tadOnlyShapeInfo);
      	yStride = shape::elementWiseStride(yShapeInfo);
      }
      __syncthreads();

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {

			int tadOffsetForBlock = tadOffsets[r];
            T *rR = result + tadOffsetForBlock;
            T *rX = x + tadOffsetForBlock;


            if(tadEWS > 0) {
            	if (tadEWS == 1 && yStride == 1) {
                	for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i] = OpType::op(rX[i], y[i]);
                	}
                } else {
					for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i * tadEWS] = OpType::op(rX[i * tadEWS], y[i * yStride]);
                	}
                }
            }
            else {
                int xCoord[MAX_RANK];
                for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    shape::ind2subC(tadRank,tadShape, i, xCoord);
                    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    result[xOffset] = OpType::op(x[xOffset], y[i * yStride]);
                }
            }
		}
	}


            
		static inline __device__ void transformCuda(const int opNum,
				T *x,
				int *xShapeInfo,
				T *y,
				int *yShapeInfo,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength, UnifiedSharedMemory *manager, int *tadShapeInfo, int *tadOffset) {

                                DISPATCH_BY_OPNUM(transformCuda, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension,  dimensionLength, manager, tadShapeInfo, tadOffset), BROADCAST_OPS);
			}
#endif

			static void exec(const int opNum, T *x,
				int *xShapeInfo,
				T *y,
				int *yShapeInfo,
				T *result,
				int *dimension,
				int dimensionLength, int *tadShapeInfo, int *tadOffset) {
                                DISPATCH_BY_OPNUM(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, result, dimension, dimensionLength, tadShapeInfo, tadOffset), BROADCAST_OPS);
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
							 int dimensionLength, int *tadShapeInfo, int *tadOffset) {

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

				int *xShape = shape::shapeOf(tadShapeShapeInfo);
				int *xStride = shape::stride(tadShapeShapeInfo);
				int *resultStride = shape::stride(tadShapeShapeInfo);
				int tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
				int tadRank = shape::rank(tadShapeShapeInfo);
				int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
				int yStride = shape::elementWiseStride(yShapeInfo);
				int tads =shape::length(xShapeInfo) / tadLength;

				int tadsPerThread = tads / 64;
				int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
				num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

				if (result == x) {
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1)
					for (int i = 0; i < tads; i++) {
						int offset = tadOffsets[i];

						if (tadEWS > 0 && yStride > 0) {
							T *oRes = result + offset;
							T *oX = x + offset;

							if (tadEWS == 1 && yStride == 1) {
#pragma omp simd
								for (int f = 0; f < tadLength; f++) {
									oRes[f] = OpType::op(oX[f], y[f]);
								}
							} else {
#pragma omp simd
								for (int f = 0; f < tadLength; f++) {
									oRes[f * tadEWS] = OpType::op(oX[f * tadEWS], y[f * yStride]);
								}
							}
						} else {
							int xCoord[MAX_RANK];

							for (int f = 0; f < tadLength; f++) {
								shape::ind2subC(tadRank,xShape, f, xCoord);
								Nd4jIndex xOffset = shape::getOffset(offset, xShape, xStride, xCoord, tadRank);
								result[xOffset] = OpType::op(x[xOffset], y[f * yStride]);
							}
						}
					}
				}
				else {

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1)
					for (int i = 0; i < tads; i++) {
						int offset = tadOffsets[i];
						T *xIter = x + offset;
						T *resultIter = result + offset;
						int shapeIter[MAX_RANK];
						int coord[MAX_RANK];
						int dim;
						int xStridesIter[MAX_RANK];
						int resultStridesIter[MAX_RANK];
						int rank = shape::rank(tadShapeShapeInfo);
						int vectorIdx = 0;
						if (PrepareTwoRawArrayIter<T>(rank,
													  xShape,
													  xIter,
													  xStride,
													  resultIter,
													  resultStride,
													  &rank,
													  shapeIter,
													  &xIter,
													  xStridesIter,
													  &resultIter,
													  resultStridesIter) >= 0) {
							ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
							{
								/* Process the innermost dimension */
								T val = OpType::op(xIter[0], y[vectorIdx]);
								resultIter[0] = val;
								vectorIdx += shape::elementWiseStride(yShapeInfo);
							}
							ND4J_RAW_ITER_TWO_NEXT(dim,
												   rank,
												   coord,
												   shapeIter,
												   xIter,
												   xStridesIter,
												   resultIter,
												   resultStridesIter);


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
template <typename T>
__device__ void broadcastGeneric(
		int opNum,
		T *x,
		int *xShapeInfo,
		int xRank,
		T *y,
		int *yShapeInfo,
		int yRank,
		T *result,
		int *resultShapeInfo,
		int zRank,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::broadcast::Broadcast<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

	functions::broadcast::Broadcast<T>::transformCuda(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			manager,
			tadOnlyShapeInfo,
			tadOffsets);
}

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
extern "C" __global__ void broadcastDouble(
		int opNum,
		double *x, int *xShapeInfo, int xRank,
		double *y, int *yShapeInfo, int yRank,
		double *result, int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets) {
	broadcastGeneric<double>(
			opNum,
			x,
			xShapeInfo, xRank,
			y,
			yShapeInfo, yRank,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength, tadOnlyShapeInfo, tadOffsets);

}


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
extern "C" __global__ void broadcastFloat(
		int opNum,
		float *x, int *xShapeInfo, int xRank,
		float *y, int *yShapeInfo, int yRank,
		float *result, int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets) {
	broadcastGeneric<float>(
			opNum,
			x,
			xShapeInfo, xRank,
			y,
			yShapeInfo, yRank,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength, tadOnlyShapeInfo, tadOffsets);

}


extern "C" __global__ void broadcastHalf(
		int opNum,
		float16 *x, int *xShapeInfo, int xRank,
		float16 *y, int *yShapeInfo, int yRank,
		float16 *result, int *resultShapeInfo, int zRank,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets) {
	broadcastGeneric<float16>(
			opNum,
			x,
			xShapeInfo, xRank,
			y,
			yShapeInfo, yRank,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength, tadOnlyShapeInfo, tadOffsets);

}

#endif



#endif /* BROADCASTING_H_ */
