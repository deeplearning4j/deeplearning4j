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
#include <op.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <pairwise_util.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif
namespace functions {
	namespace broadcast {

/**
 * Broadcast operation
 * for broadcasting a smaller tensor
 * along long a bigger one.
 */
		template<typename T>
		class Broadcast: public functions::ops::Op<T> {
		public:

			/**
             *
             * @param d1
             * @param d2
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __device__  __host__

#elif defined(__GNUC__)


#endif
			T op(T d1, T d2) = 0;
			/**
             *
             * @param d1
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __device__  __host__

#elif defined(__GNUC__)


#endif
			T op(T d1) = 0;

#ifdef __CUDACC__
			__inline__ __device__ void transformCuda(
			T *x,
			int *xShapeInfo,
			T *y,
			int *yShapeInfo,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength, UnifiedSharedMemory<T> *manager) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ shape::TAD *tad;
	  __shared__ int rank;
        if (threadIdx.x == 0) {
            tad = new(manager->getTADSpace()) shape::TAD(); //(xShapeInfo,dimension,dimensionLength)
            tad->setExternalBuffers((void *) manager);
            tad->init(xShapeInfo,dimension,dimensionLength);
            tad->createTadOnlyShapeInfo();
			rank = shape::rank(tad->tadOnlyShapeInfo);
        }
       __syncthreads();


        //int *xCoord = shape::cuMalloc(manager->getSharedCoordBuffer(), rank);
        int xCoord[MAX_RANK];


		for (int r = blockIdx.x; r < tad->numTads; r += gridDim.x) {
			if (threadIdx.x == 0)
            	tad->createOffsetForBlock(r);
            __syncthreads();

			int tadOffsetForBlock = tad->tadOffsetForBlock;

            for (int i = threadIdx.x; i < shape::length(tad->tadOnlyShapeInfo); i+= blockDim.x) {
				// now we need coords for both X, Y. Z is uses the same coord as X in this case
				// Y is always vector, however it might be stided
				shape::ind2subC(rank,tad->tadShape, i, xCoord);
                Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tad->tadShape, tad->tadStride, xCoord, rank);

				int yStride = shape::elementWiseStride(yShapeInfo);

				result[xOffset] = this->op(x[xOffset], y[i * yStride]);
            }
		}
	}
#endif

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
			virtual void exec(T *x,
							  int *xShapeInfo,
							  T *y,
							  int *yShapeInfo,
							  T *result,
							  int *dimension,
							  int dimensionLength) {
				shape::TAD tad(xShapeInfo,dimension,dimensionLength);
				tad.createTadOnlyShapeInfo();
				tad.createOffsets();
				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				int *tadShapeShapeInfo =  tad.tadOnlyShapeInfo;
				int tads = tad.numTads;
				int *xShape = shape::shapeOf(tadShapeShapeInfo);
				int *xStride = shape::stride(tadShapeShapeInfo);
				int *resultStride = shape::stride(tadShapeShapeInfo);

				if (result == x) {
#pragma omp  parallel  for
					for (int i = 0; i < tads; i++) {
						int offset = tad.tadOffsets[i];
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
								T val = this->op(xIter[0], y[vectorIdx]);
								// printf("TAD %d x %f and y %f with vector idx %d and result %f\n",i,xIter[0],y[vectorIdx],vectorIdx,val);
								xIter[0] = val;
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
				else {

#pragma omp  parallel  for
					for (int i = 0; i < tads; i++) {
						int offset = tad.tadOffsets[i];
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
								T val = this->op(xIter[0], y[vectorIdx]);
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




			}

			virtual inline
#ifdef __CUDACC__
			__host__ __device__
#endif
			void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
				//no extra params aggregation needs to happen
			}
#ifdef __CUDACC__
			inline __host__ __device__
#elif defined(__GNUC__)

#endif
			virtual ~Broadcast() {
			}
#ifdef __CUDACC__
			inline __host__ __device__
#elif defined(__GNUC__)

#endif
			Broadcast() {
			}

		};

		namespace ops {
			template<typename T>
			class Add: public  functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d1 + d2;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~Add() {
				}
			};

			template<typename T>
			class Copy: public virtual functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d2;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~Copy() {
				}

			};

			template<typename T>
			class Divide: public virtual functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d1 / d2;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~Divide() {
				}

			};

			template<typename T>
			class Multiply: public virtual functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d1 * d2;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~Multiply() {
				}

			};

			template<typename T>
			class ReverseDivide: public virtual functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d2 / d1;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~ReverseDivide() {
				}

			};

			template<typename T>
			class ReverseSubtract: public virtual functions::broadcast::Broadcast<T> {
			public:


				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d2 - d1;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~ReverseSubtract() {
				}

			};

			template<typename T>
			class Subtract: public virtual functions::broadcast::Broadcast<T> {
			public:

				/**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T d2) {
					return d1 - d2;
				}
				/**
                 *
                 * @param d1
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~Subtract() {
				}

			};
		}

		template<typename T>
		class BroadcastOpFactory {
		public:

#ifdef __CUDACC__
			__host__ __device__
#endif
			BroadcastOpFactory() {
			}


			/**
             * creates an operation
             * @param op the op number to create:
             * 0: Add
             * 1: Subtract
             * 2: Multiply
             * 3: Divide
             * 4: ReverseDivide
             * 5: Reverse Subtract
             * 6: Copy
             * @return the broadcast operation
             */
#ifdef __CUDACC__
			__inline__ __device__
            Broadcast<T> * getOp(int op, unsigned char *buffer) {
#else
			Broadcast<T> * getOp(int op) {
#endif
				if (op == 0) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::Add<T>();
#else
					return new functions::broadcast::ops::Add<T>();
#endif
				} else if (op == 1) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::Subtract<T>();
#else
					return new functions::broadcast::ops::Subtract<T>();
#endif
				} else if (op == 2) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::Multiply<T>();
#else
					return new  functions::broadcast::ops::Multiply<T>();
#endif
				} else if (op == 3) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::Divide<T>();
#else
					return new functions::broadcast::ops::Divide<T>();
#endif
				} else if (op == 4) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::ReverseDivide<T>();
#else
					return new functions::broadcast::ops::ReverseDivide<T>();
#endif
				} else if (op == 5) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::ReverseSubtract<T>();
#else
					return new functions::broadcast::ops::ReverseSubtract<T>();
#endif
				} else if (op == 6) {
#ifdef __CUDACC__
					return new(buffer) functions::broadcast::ops::Copy<T>();
#else
					return new functions::broadcast::ops::Copy<T>();
#endif
				}

				return nullptr;

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
		int dimensionLength) {

	__shared__ functions::broadcast::Broadcast<T> *op;
	__shared__ functions::broadcast::BroadcastOpFactory<T> *newOpFactory;

	__shared__ UnifiedSharedMemory<T> *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory<T>();
	    manager->init(sizeof(UnifiedSharedMemory<T>), sizeof(functions::broadcast::BroadcastOpFactory<T>), sizeof(functions::broadcast::Broadcast<T>), sizeof(shape::TAD));
	    manager->setXSpace(xRank);
	    manager->setYSpace(yRank);
	    manager->setZSpace(zRank);
	    manager->setTADSpace(dimensionLength);
    }
    __syncthreads();

	__shared__ int *ptrSharedXShapeInfo;
	__shared__ int *ptrSharedYShapeInfo;
    __shared__ int *ptrSharedZShapeInfo;

	if (xShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(xShapeInfo, manager->getXShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedXShapeInfo = manager->getXShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedXShapeInfo = nullptr;

    if (yShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(yShapeInfo, manager->getYShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedYShapeInfo = manager->getYShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedYShapeInfo = nullptr;

    if (resultShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(resultShapeInfo, manager->getZShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedZShapeInfo = manager->getZShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedZShapeInfo = nullptr;

	if(threadIdx.x == 0) {
		newOpFactory =  new(manager->getFactorySpace()) functions::broadcast::BroadcastOpFactory<T>();
		op = newOpFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();


	op->transformCuda(
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength, manager);
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
		int dimensionLength) {
	broadcastGeneric<double>(
			opNum,
			x,
			xShapeInfo, xRank,
			y,
			yShapeInfo, yRank,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength);

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
		int dimensionLength) {
	broadcastGeneric<float>(
			opNum,
			x,
			xShapeInfo, xRank,
			y,
			yShapeInfo, yRank,
			result,
			resultShapeInfo, zRank,
			dimension,
			dimensionLength);

}

#endif



#endif /* BROADCASTING_H_ */
