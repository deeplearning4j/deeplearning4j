/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_
#ifdef __JNI__
#include <jni.h>
#endif
#include <op.h>
#include <omp.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <shape.h>
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace functions {
	namespace pairwise_transforms {
#define MIN 1e-12

/**
 * Transforms involving 2 arrays
 */
		template<typename T>
		class PairWiseTransform : public virtual functions::ops::Op<T> {
		protected:
			bool requiresSpecial = false;
		public:
			virtual
#ifdef __CUDACC__
			inline __host__ __device__
#elif defined(__GNUC__)

#endif
			T op(T d1, T d2, T *params) = 0;

			virtual
#ifdef __CUDACC__
			inline __host__ __device__
#elif defined(__GNUC__)

#endif
			T op(T d1, T *params) = 0;

#ifdef __CUDACC__
			/**
	 *
	 */
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			Nd4jIndex n,
			int *indexes,int *allocationPointer) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				indexes,
				indexes, allocationPointer);
	}

	/**
	 *
	 */
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *indexes,
			int *yIndexes,
			int *resultIndexes,int *allocationPointer) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Nd4jIndex i = tid;
		Nd4jIndex n = shape::length(xShapeBuffer);
		for (; i < n; i += totalThreads) {
			result[resultIndexes[i]] = op(dx[indexes[i]],y[yIndexes[i]], extraParams);
		}
	}


	/**
	 *
	 */
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *indexes,
			int *yIndexes,int *allocationPointer) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				yIndexes,
				indexes, allocationPointer);
	}

	/**
	 *
	 */
	virtual __inline__ __device__ void transformCuda(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Nd4jIndex i = tid;


		int *xShape = shape::shapeOf(xShapeBuffer);
		int *yShape = shape::shapeOf(yShapeBuffer);
		int *resultShape = shape::shapeOf(resultShapeBuffer);

		int *xStride = shape::stride(xShapeBuffer);
		int *yStride = shape::stride(yShapeBuffer);
		int *resultStride = shape::stride(resultShapeBuffer);

		int xRank = shape::rank(xShapeBuffer);
		int yRank = shape::rank(yShapeBuffer);
		int resultRank = shape::rank(resultShapeBuffer);

		int xOffset = shape::offset(xShapeBuffer);
		int yOffset = shape::offset(yShapeBuffer);
		int resultOffset = shape::offset(resultShapeBuffer);


		char xOrder = shape::order(xShapeBuffer);
		char yOrder = shape::order(yShapeBuffer);
		char resultOrder = shape::order(resultShapeBuffer);

		int xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
		int yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
		int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);



		Nd4jIndex n = shape::length(xShapeBuffer);
		if(xElementWiseStride >= 1 && yElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == yOrder && resultOrder == xOrder) {
			transformCuda(
					n,
					dx,
					y,
					xElementWiseStride,
					yElementWiseStride,
					extraParams,
					result,
					resultElementWiseStride, allocationPointer);
		}

		else {

			long allocSize = sizeof(int) * (xRank + yRank + resultRank);
			int *tB = shape::cuMalloc(allocationPointer, allocSize);

			int *xCoord = tB;
			int *yCoord = tB + xRank;
			int *resultCoord = yCoord + yRank;


			if (dx == result) {
				for (i = tid; i < n; i += totalThreads) {
					shape::ind2subC(xRank,xShape, i, xCoord);
					shape::ind2subC(yRank,yShape, i, yCoord);

					Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
					Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
					result[xOffset] = op(dx[xOffset], y[yOffset], extraParams);
				}
			} else {
				for (; i < n; i += totalThreads) {
					shape::ind2subC(xRank,xShape, i, xCoord);
					shape::ind2subC(yRank,yShape, i, yCoord);
					shape::ind2subC(resultRank,resultShape, i, resultCoord);

					Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
					Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
					Nd4jIndex resultOffset = shape::getOffset(0, resultShape, resultShape, resultCoord, resultRank);
					result[resultOffset] = op(dx[xOffset], y[yOffset], extraParams);
				}
			}

			if (tid * allocSize > PREALLOC_SIZE - allocSize) {
                free(tB);
            }
		}




	}

	/**
	 *
	 * @param n
	 * @param xOffset
	 * @param yOffset
	 * @param resultOffset
	 * @param dx
	 * @param dy
	 * @param incx
	 * @param incy
	 * @param params
	 * @param result
	 * @param incz
	 * @param blockSize
	 */
	virtual __inline__ __device__ void transformCuda(
			Nd4jIndex n,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result,
			int incz,int *allocationPointer) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Nd4jIndex i = tid;

		if (incy == 0) {
#pragma unroll
				for (; i < n; i++) {
					result[i * incz] = op(dx[i * incx], params);
				}
		} else if ((incx == incy) && (incx > 0)) {
			/* equal, positive, increments */
			if (incx == 1) {
				/* both increments equal to 1 */
#pragma unroll
				for (; i < n; i += totalThreads) {
					result[i * incz] = op(dx[i * incx], dy[i * incy],
							params);
				}
			} else {
				/* equal, positive, non-unit increments. */
#pragma unroll
				for (; i < n; i += totalThreads) {
					result[i * incz] = op(dx[i * incx], dy[i * incy],
							params);
				}
			}
		} else {
			/* unequal or nonpositive increments */
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * incz] = op(dx[i * incx], dy[i * incy],
						params);
			}
		}
	}

#endif
		public:


			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
			virtual void exec(
					T *dx,
					int *xShapeBuffer,
					T *y,
					int *yShapeBuffer,
					T *result,
					int *resultShapeBuffer,
					T *extraParams,
					int *indexes,
					int *yIndexes) {
				exec(dx,
					 xShapeBuffer,
					 y,
					 yShapeBuffer,
					 result,
					 resultShapeBuffer,
					 extraParams,
					 indexes,
					 yIndexes,
					 indexes);
			}


			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
			virtual void exec(
					T *dx,
					int *xShapeBuffer,
					T *y,
					int *yShapeBuffer,
					T *result,
					int *resultShapeBuffer,
					T *extraParams,
					int *indexes,
					int *yIndexes,
					int *resultIndexes) {
				Nd4jIndex n = shape::length(xShapeBuffer);
#pragma omp parallel for
				for (Nd4jIndex i = 0; i < n; i++) {
					result[resultIndexes[i]] = op(dx[indexes[i]], y[yIndexes[i]], extraParams);

				}
			}





			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
                 * @param indexes which indexes to copy
             */
			virtual void exec(
					T *dx,
					int *xShapeBuffer,
					T *y,
					int *yShapeBuffer,
					T *result,
					int *resultShapeBuffer,
					T *extraParams,
					int *indexes) {
				Nd4jIndex n = shape::length(xShapeBuffer);
#pragma omp parallel for
				for (Nd4jIndex i = 0; i < n; i++) {
					result[indexes[i]] = op(dx[indexes[i]],y[indexes[i]], extraParams);

				}

			}

			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             */
			virtual void execSpecial(
					T *dx,
					int *xShapeBuffer,
					T *y,
					int *yShapeBuffer,
					T *result,
					int *resultShapeBuffer,
					T *extraParams) = 0;
			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             */
			virtual void exec(
					T *dx,
					int *xShapeBuffer,
					T *y,
					int *yShapeBuffer,
					T *result,
					int *resultShapeBuffer,
					T *extraParams) {
				Nd4jIndex n = shape::length(xShapeBuffer);
				int xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
				int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);

				bool sameShape = shape::shapeEquals(shape::rank(xShapeBuffer), shape::shapeOf(xShapeBuffer),
													shape::rank(yShapeBuffer), shape::shapeOf(yShapeBuffer));
				//ignore everything else
				if (this->requiresSpecial) {
					this->execSpecial(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
					return;
				}


				if (xElementWiseStride >= 1 &&
					yElementWiseStride >= 1 &&
					resultElementWiseStride >= 1 &&
					shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
					shape::order(resultShapeBuffer) == shape::order(xShapeBuffer) &&
					sameShape) {
					exec(dx,
						 xElementWiseStride,
						 y,
						 yElementWiseStride,
						 result,
						 resultElementWiseStride,
						 extraParams,
						 n);
				}
					//not same shape
				else if (!sameShape && shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
						 shape::order(resultShapeBuffer) == shape::order(xShapeBuffer) && xElementWiseStride >= 1 &&
						 yElementWiseStride >= 1 &&
						 resultElementWiseStride >= 1) {
					exec(dx,
						 xElementWiseStride,
						 y,
						 yElementWiseStride,
						 result,
						 resultElementWiseStride,
						 extraParams,
						 shape::length(yShapeBuffer));
				}

				else if (sameShape) {
					int rank = shape::rank(xShapeBuffer);
					int *xShape = shape::shapeOf(xShapeBuffer);

					int *xStride = shape::stride(xShapeBuffer);
					int *yStride = shape::stride(yShapeBuffer);
					int *resultStride = shape::stride(resultShapeBuffer);

					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];
					int yStridesIter[MAX_RANK];
					int resultStridesIter[MAX_RANK];
					if (PrepareThreeRawArrayIter<T>(rank,
													xShape,
													dx,
													xStride,
													y,
													yStride,
													result,
													resultStride,
													rank,
													shapeIter,
													&dx,
													xStridesIter,
													&y,
													yStridesIter,
													&result,
													resultStridesIter) >= 0) {
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
						{
							/* Process the innermost dimension */
							T *xIter = dx;
							T *yIter = y;
							T *resultIter = result;
							resultIter[0] = op(xIter[0], yIter[0], extraParams);
						}
						ND4J_RAW_ITER_THREE_NEXT(dim,
												 rank,
												 coord,
												 shapeIter,
												 dx,
												 xStridesIter,
												 y,
												 yStridesIter,
												 result,
												 resultStridesIter);
					}
					else {
						printf("Unable to prepare array\n");
					}

				}

				else {
					Nd4jIndex len = shape::length(xShapeBuffer);
					int xRank = shape::rank(xShapeBuffer);
					int yRank = shape::rank(yShapeBuffer);
					int resultRank = shape::rank(resultShapeBuffer);
					int *xCoord = new int[xRank];
					int *yCoord = new int[yRank];
					int *resultCoord = new int[resultRank];

					int *xShape = shape::shapeOf(xShapeBuffer);
					int *xStride = shape::stride(xShapeBuffer);

					int *yShape = shape::shapeOf(yShapeBuffer);
					int *yStride = shape::stride(yShapeBuffer);

					int *resultShape = shape::shapeOf(resultShapeBuffer);
					if(dx == result) {
						for (Nd4jIndex i = 0; i < len; i++) {
							shape::ind2subC(xRank,xShape, i, xCoord);
							shape::ind2subC(yRank,yShape, i, yCoord);
							shape::ind2subC(resultRank,resultShape, i, resultCoord);

							Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
							Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
							result[xOffset] = op(dx[xOffset], y[yOffset], extraParams);

						}
					}
					else {
						for (Nd4jIndex i = 0; i < len; i++) {
							shape::ind2subC(xRank,xShape, i, xCoord);
							shape::ind2subC(yRank,yShape, i, yCoord);
							shape::ind2subC(resultRank,resultShape, i, resultCoord);

							Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
							Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
							Nd4jIndex resultOffset = shape::getOffset(0, resultShape, resultShape, resultCoord, resultRank);
							result[resultOffset] = op(dx[xOffset], y[yOffset], extraParams);

						}
					}


					delete[] xCoord;
					delete[] yCoord;
					delete []resultCoord;
				}
			}


			/**
             * CPU operation execution
             * @param dx the input data
             * @param xStride the stride to iterate over
             * the x input
             * @param y the y data
             * @param yStride the stride to iterate
             * over the y buffer
             * @param result the buffer
             * to store the result in
             * @param resultStride the stride for the buffer
             * @param extraParams the extra parameters for the transform
             * @param n the length of the input
             */
			virtual void exec(T *dx, int xStride, T *y, int yStride, T *result,
							  int resultStride, T *extraParams,  Nd4jIndex n) {
				if (xStride == 1 && yStride == 1 && resultStride == 1) {
					if(n < 8000) {
						for (Nd4jIndex i = 0; i < n; i++) {
							result[i] = op(dx[i], y[i], extraParams);
						}
					}
					else {
#pragma omp parallel for
						for (Nd4jIndex i = 0; i < n; i++) {
							result[i] = op(dx[i], y[i], extraParams);
						}
					}



				}

				else {
					if(n < 8000) {
						for (Nd4jIndex i = 0; i < n; i++) {
							result[i * resultStride] = op(dx[i * xStride],
														  y[i * yStride], extraParams);
						}
					}
					else {
#pragma omp parallel for
						for (Nd4jIndex i = 0; i < n; i++) {
							result[i * resultStride] = op(dx[i * xStride],
														  y[i * yStride], extraParams);
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
			virtual ~PairWiseTransform() {
			}
#ifdef __CUDACC__
			inline __host__ __device__
#elif defined(__GNUC__)


#endif
			PairWiseTransform() {
			}

		};

		namespace ops {
/**
 * x + y
 */
			template<typename T>
			class Add: public virtual PairWiseTransform<T> {
			public:

				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {
					//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 + d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Add() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Add() {
				}
			};

/**
 * Copy y to x
 */
			template<typename T>
			class Copy: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					(void)d1;
					(void)params;
					return d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					(void)params;
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Copy() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Copy() {
				}
			};

/**
 * Divide x / y
 */
			template<typename T>
			class Divide: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x inputCopy
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 / d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Divide() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Divide() {
				}
			};



/**
 *Set x to y
 */
			template<typename T>
			class Set: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Set() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Set() {
				}
			};


/**
 * Whether 2 elements in an array
 * are epsilion equal
 */
			template<typename T>
			class Epsilon: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					T diff = d1 - d2;
					T absDiff = abs(diff);
					if (absDiff > MIN)
						return 1;
					return 0;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Epsilon() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Epsilon() {
				}
			};

/**
 * x == y (binary result)
 */
			template<typename T>
			class EqualTo: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 == d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~EqualTo() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				EqualTo() {
				}
			};

/**
 * x == y (binary result)
 */
			template<typename T>
			class NotEqualTo: public virtual PairWiseTransform<T> {
			public:

				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 != d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~NotEqualTo() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				NotEqualTo() {
				}
			};



/**
 * Whether x > y
 */
			template<typename T>
			class GreaterThanOrEqual: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 >= d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~GreaterThanOrEqual() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				GreaterThanOrEqual() {
				}
			};


/**
 * Whether x > y
 */
			template<typename T>
			class GreaterThan: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 > d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~GreaterThan() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				GreaterThan() {
				}
			};

/**
 * Whether x < y
 */
			template<typename T>
			class LessThan: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 < d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~LessThan() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				LessThan() {
				}
			};

/**
 * Whether x < y
 */
			template<typename T>
			class LessThanOrEqual: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 <= d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~LessThanOrEqual() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				LessThanOrEqual() {
				}
			};

/**
 * x * y
 */
			template<typename T>
			class Multiply: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 * d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}

#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Multiply() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Multiply() {
				}
			};

/**
 * y / x
 */
			template<typename T>
			class ReverseDivide: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d2 / d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~ReverseDivide() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				ReverseDivide() {
				}
			};

/**
 * y - x
 */
			template<typename T>
			class ReverseSubtraction: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d2 - d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~ReverseSubtraction() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				ReverseSubtraction() {
				}
			};

/**
 * x - y
 */
			template<typename T>
			class Subtract: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return d1 - d2;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Subtract() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Subtract() {
				}
			};


/**
 * x - y
 */
			template<typename T>
			class Max: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return nd4j::math::nd4j_max<T>(d1,d2);
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Max() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Max() {
				}
			};



/**
 * x - y
 */
			template<typename T>
			class Min: public virtual PairWiseTransform<T> {
			public:
				/**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
				virtual void execSpecial(
						T *dx,
						int *xShapeBuffer,
						T *y,
						int *yShapeBuffer,
						T *result,
						int *resultShapeBuffer,
						T *extraParams) {//no-op
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T d2, T *params) {
					return nd4j::math::nd4j_min(d1,d2);
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)

#endif
				T op(T d1, T *params) {
					return d1;
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				virtual ~Min() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#elif defined(__GNUC__)


#endif
				Min() {
				}
			};


		}

/**
 * Creates pair wise operations.
 */
		template<typename T>
		class PairWiseTransformOpFactory {
		public:



#ifdef __CUDACC__
			__host__ __device__
#endif
			PairWiseTransformOpFactory() {
			}

			/**
             * Create an operation
             * @param op the op number
             * 0: Add
             * 1: Copy
             * 2: Divie
             * 3: equal to
             * 4: greater than
             * 5: less than
             * 6: multiply
             * 7: reverse divide
             * 8 reverse subtract
             * 9: subtract
             * @return the operation based on the op number
             */
#ifdef __CUDACC__
			__inline__ __host__ __device__
#endif
			PairWiseTransform<T> *getOp(int op) {
				if (op == 0)
					return new pairwise_transforms::ops::Add<T>();
				else if (op == 1)
					return new pairwise_transforms::ops::Copy<T>();
				else if (op == 2)
					return new pairwise_transforms::ops::Divide<T>();
				else if (op == 3)
					return new pairwise_transforms::ops::EqualTo<T>();
				else if (op == 4)
					return new pairwise_transforms::ops::GreaterThan<T>();
				else if (op == 5)
					return new pairwise_transforms::ops::LessThan<T>();
				else if (op == 6)
					return new pairwise_transforms::ops::Multiply<T>();
				if (op == 7)
					return new pairwise_transforms::ops::ReverseDivide<T>();
				if (op == 8)
					return new pairwise_transforms::ops::ReverseSubtraction<T>();
				if (op == 9)
					return new pairwise_transforms::ops::Subtract<T>();
				if (op == 10)
					return new pairwise_transforms::ops::Epsilon<T>();
				if(op == 11)
					return new pairwise_transforms::ops::GreaterThanOrEqual<T>();
				if(op == 12)
					return new pairwise_transforms::ops::LessThanOrEqual<T>();
				if(op == 13)
					return new pairwise_transforms::ops::Max<T>();
				if(op == 14)
					return new pairwise_transforms::ops::Min<T>();
				if(op == 15)
					return new pairwise_transforms::ops::NotEqualTo<T>();
				if(op == 16)
					return new pairwise_transforms::ops::Set<T>();


				return NULL;
			}



		};
	}
}

#ifdef __CUDACC__

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo, int *allocationPointer) {
	__shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
	__shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0) {
		newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
		op = newOpFactory->getOp(opNum);
	}
	__syncthreads();

	op->transformCuda(dx,xShapeInfo,dy,yShapeInfo,result,resultShapeInfo,params, allocationPointer);

	__syncthreads();
	if(threadIdx.x == 0) {
		delete op;
		delete newOpFactory;
	}

}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformDouble(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo, int *allocationPointer) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo, allocationPointer);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformFloat(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo, int *allocationPointer) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo, allocationPointer);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer) {
	__shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
	__shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0) {
		newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
		op = newOpFactory->getOp(opNum);
	}
	__syncthreads();

	op->transform(
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			result,
			resultShapeInfo,
			params,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer);

	__syncthreads();
	if(threadIdx.x == 0) {
		delete op;
		delete newOpFactory;
	}

}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
__global__ void pairWiseTransformDoubleIndex(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
__global__ void pairWiseTransformFloatIndex(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo,
		int *yShapeInfo,
		int *resultShapeInfo,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo,
			yShapeInfo,
			resultShapeInfo,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer);
}

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template<typename T>
__device__ void pairWiseTransformStridedGeneric(
		int opNum,
		Nd4jIndex n,
		T *dx,
		T *dy,
		int incx,
		int incy,
		T *params,
		T *result,
		int incz, int *allocationPointer) {
	__shared__ functions::pairwise_transforms::PairWiseTransform<T> *op;
	__shared__ functions::pairwise_transforms::PairWiseTransformOpFactory<T> *newOpFactory;
	if (threadIdx.x == 0) {
		newOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
		op = newOpFactory->getOp(opNum);
	}
	__syncthreads();

	op->transformCuda(n, dx, dy, incx, incy, params, result, incz, allocationPointer);

	__syncthreads();
	if (threadIdx.x == 0) {
		delete op;
		delete newOpFactory;
	}

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
__global__ void pairWiseTransformStridedDouble(
		int opNum,
		Nd4jIndex n,
		double *dx,
		double *dy,
		int incx,
		int incy,
		double *params,
		double *result,
		int incz, int *allocationPointer) {
	pairWiseTransformStridedGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer);
}
/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
__global__ void pairWiseTransformStridedFloat(
		int opNum,
		Nd4jIndex n,
		float *dx,
		float *dy,
		int incx,
		int incy,
		float *params,
		float *result,
		int incz, int *allocationPointer) {
	pairWiseTransformStridedGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer);
}



#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
