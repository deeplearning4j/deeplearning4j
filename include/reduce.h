#include <dll.h>
//#include <string>
#include <sharedmem.h>
#include <stdio.h>
#include <shape.h>
#include <op.h>
#include <omp.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <nd4jmalloc.h>
#include <pairwise_util.h>
#pragma once
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif

//an op for the kernel
namespace functions {
	namespace reduce {

/**
 * A reduce function
 * reduces a vector down to
 * a subset of itself
 * via aggregating member
 * elements.
 */
		template<typename T>
		class ReduceFunction: public functions::ops::Op<T> {
		protected:
			int extraParamsLength = 0;
			int indexBased = 1;
		public:
			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			int getIndexBased() {
				return indexBased;
			}


			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			ReduceFunction<T> ** extraParamsFunctions() = 0;
			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			int getExtraParamsLength() {
				return extraParamsLength;
			}

			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			T * createExtraParams() {
				T *ret = (T *) malloc(sizeof(T) * this->getExtraParamsLength());
				return ret;
			}


#ifdef __CUDACC__
			virtual __host__ __device__
	T * generateExtraParamsCuda(T *input,int *shapeInfo) {
		return NULL;
	}
#endif

			/**
             * Merge the 2 inputs
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			T merge(T old, T opOutput, T *extraParams) = 0;

			/**
             * Op with 1 parameter
             * @param d1
             * @param extraParams
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			T op(T d1, T *extraParams) = 0;

			//calculate an update of the reduce operation
			/**
             * Op with 2 parameters
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			T update(T old, T opOutput, T *extraParams) = 0;
#ifdef __CUDACC__


			__inline__ __device__ void execScalarCuda(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *allocationBuffer, T *reductionBuffer) {
		int elementWiseStride = shape::elementWiseStride(xShapeInfo);

		int n = shape::length(xShapeInfo);

		if (threadIdx.x == 0)
			printf("Starting scalarReduce block: [%i]\n", blockIdx.x);

		int tid = blockDim.x * blockIdx.x + threadIdx.x;

		//shared memory space for storing intermediate results
		SharedMemory <T> val;
		volatile T *sPartials = val.getPointer();
		int numElements = blockDim.x;
		T init = this->startingValue(dx);
		for(int i = threadIdx.x; i < blockDim.x; i+= blockDim.x)
			sPartials[i] = init;
		__syncthreads();

		if(elementWiseStride >= 1) {
			if(elementWiseStride == 1) {
#pragma unroll
				for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
					sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],this->op(dx[i],extraParams),extraParams);
				}
			}
			else {
#pragma unroll
				for(int i = elementWiseStride * tid;i < n; i += (blockDim.x * gridDim.x * elementWiseStride)) {
					sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],this->op(dx[i * elementWiseStride],extraParams),extraParams);
				}
			}
		}
		else {
			int rank = shape::rank(xShapeInfo);
			long allocSize = sizeof(int) * rank;
			int *ind2sub = shape::cuMalloc(allocationBuffer, allocSize);
#pragma unroll
			for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
				shape::ind2sub(rank,shape::shapeOf(xShapeInfo),i,ind2sub);
				sPartials[tid] = this->update(sPartials[tid],this->op(dx[i],extraParams),extraParams);
				__syncthreads();
			}

			if (tid * allocSize > PREALLOC_SIZE - allocSize) {
                free(ind2sub);
            }
		}

				__syncthreads();
			T **sPartialsRef = (T **) &sPartials;
			aggregatePartials(sPartialsRef, threadIdx.x, numElements,extraParams);


			__syncthreads();
			// if we have blocks > 1, we'll do blockwise reduce
			if (gridDim.x > 1) {
				/*
			 	at this point we have everything stored at sPartials[0]
				 so, we just need to store our partials result
				*/
				int rank = shape::rank(xShapeInfo);
				if (threadIdx.x == 0) {
					reductionBuffer[blockIdx.x] = this->postProcess(sPartials[0],n,extraParams);

					// we mark this block as finished
					allocationBuffer[tid * rank] = 119120121;
				}

				if (blockIdx.x == 0) {

					// now we must ensure, that all other blocks are finished
					__shared__ int finisher;
					clock_t *globalClock;
					if (tid == 0)
							finisher = 0;
					__syncthreads();

					while (finisher != gridDim.x) {
						if (tid == 0)
							finisher = 0;
						__syncthreads();

						if (threadIdx.x < gridDim.x) {
							if (allocationBuffer[threadIdx.x * blockDim.x * rank] == 119120121) {
								atomicAdd(&finisher, 1);
								sPartials[threadIdx.x] =  reductionBuffer[threadIdx.x];
							}
						}
						__syncthreads();

						if (finisher != gridDim.x) {
							// TODO: sleep here
							clock_t start = clock();
							clock_t now;
							for (;;) {
  								now = clock();
  								clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
  								if (cycles >= 10000) {
    								break;
  								}
							}

							*globalClock = now;
						}
					}

					__syncthreads();

					T **sPartialsRef = (T **) &sPartials;
					aggregatePartials(sPartialsRef, threadIdx.x, gridDim.x,extraParams);

					__syncthreads();
					// write result for this block to global mem
					if (tid == 0) {
						result[0] = this->postProcess(sPartials[0],n,extraParams);
					}
				}
			} else {
				__syncthreads();
				// write result for this block to global mem
				if (tid == 0) {
					result[0] = this->postProcess(sPartials[0],n,extraParams);
				}
			}
	}
	/**
	 * Kernel invocation for reduce
	 * @param n the length of the buffer
	 * @param dx the input
	 * @param xShapeInfo the shape information for the input
	 * @param extraParams extra parameters (starting value,..)
	 * @param result the result buffer
	 * @param resultShapeInfo the shapeinformation for the result buffer
	 * @param gpuInformation the gpu information (shared memory allocated,..)
	 * @param dimension the dimension to do reduce along long
	 * @param dimensionLength the length of the dimension buffer
	 * @param postProcessOrNot whether to reduce or not
	 */
	__inline__ __device__ virtual void transformCuda(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot,
			int *allocationBuffer,T *reductionBuffer) {

		/**
		 * Gpu information for the problem
		 */
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		__shared__ volatile int resultScalar;

		__shared__ int xElementWiseStride;

		//shared memory space for storing intermediate results
		SharedMemory <T> val;
		volatile T *sPartials = val.getPointer();
		int numElements = blockDim.x;
		T init = this->startingValue(dx);
		for (int i = threadIdx.x; i < numElements; i += blockDim.x)
			sPartials[i] = init;
		__syncthreads();

		//length for the tad

		__shared__ int resultLength;



		//only compute the tad indexes once
		T reduction = this->startingValue(dx);
		if (threadIdx.x == 0) {
			if (resultShapeInfo != NULL)
				resultLength = shape::length(resultShapeInfo);
			else resultLength = 1;

			if (dimensionLength == 1) {
				if (dimension == NULL || dimension[0] == MAX_DIMENSION)
					resultScalar = 1;
				else
					resultScalar = 0;
			}
			else
				resultScalar = 0;

			if (resultLength == 1)
				resultScalar = 1;
			/**
			 * The element wise stride belong longs to a reduction index.
			 * When used out of order, we can get rid of the data
			 * dependencies and rely on using the max dimension
			 * specified for stride instead.
			 * Say we take the sum(0,1) along long arr
			 * we can use arr.stride(1) as a representation
			 * along long which to iterate.
			 */


			int *xStride = shape::stride(xShapeInfo);
			char xOrder = shape::order(xShapeInfo);
			/*
				int *xShape = shape::shapeOf(xShapeInfo);
				int *xStride = shape::stride(xShapeInfo);
				char xOrder = shape::order(xShapeInfo);
				int n = shape::length(xShapeInfo);
				int xRank = shape::rank(xShapeInfo);
				int xOffset = shape::offset(xShapeInfo);
			 */
			//				int

			if (dimension != NULL && (dimension[0] != MAX_DIMENSION && dimensionLength == 1)) {
				xElementWiseStride =  xStride[dimension[0]];//shape::computeElementWiseStride(xRank,xShape,xStride,xOrder == 'f');
			} else {
				xElementWiseStride = shape::elementWiseStride(xShapeInfo);
			}


			//printf("Order is: [%c], stride is: xElementStride: [%i], passed strides are: [%i], dimension: [%i]\n", xOrder, xElementWiseStride, xStride[0], dimension[0]);

		}
		__syncthreads();





		if (!resultScalar) {
			if(dimensionLength > 1) {
				__shared__ int numOnes;
				__shared__ bool squeezed;
				__shared__ bool newSqueezeDimensions;
				__shared__ int *inputShapeInfo;
				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem

				__shared__ int *tadShapeShapeInfo;

				if(threadIdx.x == 0) {
					inputShapeInfo = xShapeInfo;
				}

				__syncthreads();

				int *shape = shape::shapeOf(inputShapeInfo);
				int *stride = shape::stride(inputShapeInfo);
				int wholeRank = shape::rank(inputShapeInfo);

				if(threadIdx.x == 0) {
					numOnes = 0;
					for(int i = 0; i < wholeRank; i++) {
						if(shape[i] == 1)
							numOnes++;
					}

					//squeeze the dimensions
					if(numOnes > 0) {
						squeezed = false;
						newSqueezeDimensions = false;
						inputShapeInfo = shape::squeezeDimensions(
							inputShapeInfo,
							dimension,
							dimensionLength,
							&squeezed,
							&newSqueezeDimensions,
							wholeRank,
							numOnes);
					}
				}

				__syncthreads();

				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				if(threadIdx.x == 0)
					tadShapeShapeInfo = shape::shapeInfoOnlyShapeAndStride(inputShapeInfo,dimension,dimensionLength,false);
				__syncthreads();

				int *xShape = shape::shapeOf(tadShapeShapeInfo);
				int *xStride = shape::stride(tadShapeShapeInfo);
				int tadLength = shape::length(tadShapeShapeInfo);
				int rank = shape::rank(tadShapeShapeInfo);
#pragma unroll
				for(int i = tid; i < resultLength; i+= gridDim.x * blockDim.x) {
					int offset = shape::tadOffset(i,inputShapeInfo,dimension,dimensionLength);
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int rankIter = rank;
					int xStridesIter[MAX_RANK];
					T *xPointer = dx + offset;
					T start = this->startingValue(xPointer);
					if(PrepareOneRawArrayIter<T>(rankIter,
							xShape,
							xPointer,
							xStride,
							&rankIter,
							shapeIter,
							&xPointer,
							xStridesIter) >= 0) {
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							/* Process the innermost dimension */
							start = update(start,op(xPointer[0],extraParams),extraParams);
						} ND4J_RAW_ITER_ONE_NEXT(dim,
								rank,
								coord,
								shapeIter,
								xPointer,
								xStridesIter);
						start = postProcess(start,tadLength,extraParams);
					}
					else {
						printf("Unable to prepare array\n");
					}

					result[i] = start;
				}

				__syncthreads();
				if (threadIdx.x == 0) {
					free(tadShapeShapeInfo);

					if(newSqueezeDimensions) {
						free(dimension);
					}

					if(numOnes > 0) {
						free(inputShapeInfo);
					}
				}
			}
			else {

				int resultLength = shape::length(resultShapeInfo);

				/**
				 * The element wise stride belong longs to a reduction index.
				 * When used out of order, we can get rid of the data
				 * dependencies and rely on using the max dimension
				 * specified for stride instead.
				 * Say we take the sum(0,1) along long arr
				 * we can use arr.stride(1) as a representation
				 * along long which to iterate.
				 */
				int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
				int tadLength = elementsPerReductionIndex;
				int xLength = shape::length(xShapeInfo);
				int i = 0,j = 0;

#pragma unroll
				for(i = tid; i < resultLength; i+= blockDim.x * gridDim.x) {
					int offsetForTad = shape::tadOffset(tid, xShapeInfo, dimension, dimensionLength);
					sPartials[tid] = op(dx[offsetForTad], extraParams);

					for(j = 1; j < elementsPerReductionIndex; j++) {
						sPartials[tid] =  update(sPartials[tid],op(dx[offsetForTad + xElementWiseStride * j], extraParams), extraParams);
					}

					result[i] = postProcess(sPartials[tid],tadLength,extraParams);
				}



			}


		}
		else {
			this->execScalarCuda(
					dx,
					xShapeInfo,
					extraParams,
					result,
					resultShapeInfo,
					allocationBuffer, reductionBuffer);
		}
	}

	/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	__device__ virtual void aggregatePartials(T **sPartialsRef, int tid, int numItems,T *extraParams) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		T *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
			while (floorPow2 & (floorPow2 - 1)) {
				floorPow2 &= floorPow2 - 1;
			}
			if (tid >= floorPow2) {
				sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
			}
			__syncthreads();
		}


#pragma unroll
		for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numItems) {
				sPartials[tid] = update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
			}
			__syncthreads();
		}

	}
#endif
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			T postProcess(T reduction, Nd4jIndex n, T *extraParams)  {
				return reduction;
			}

#ifdef __CUDACC__
			__inline__ __host__
#endif
			T aggregateBuffer(int n,T *buffer,T *extraParams) {

				T ret = buffer[0];
#pragma omp for
				for(int i = 1; i < n; i++) {
					ret = update(ret,buffer[i],extraParams);
				}

				return ret;
			}

			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			~ReduceFunction() {
			}
#ifdef __CUDACC__
			__host__ __device__
#endif
			ReduceFunction() {
			}





			/**
             * CPU implementation
             * @param x the input data
             * @param xShapeInfo the shape information for
             * the input data
             * @param extraParams the extra parameters for the problem
             * @param result the result buffer
             * @param resultShapeInfo the shape information
             */
#ifdef __CUDACC__
			__host__ __device__
#endif
			void exec(T *x,
					  int *xShapeInfo,
					  T *extraParams,
					  T *result,
					  int *resultShapeInfo) {
				T startingVal = this->execScalar(x,xShapeInfo,extraParams);
				result[0] = startingVal;

			}



			/**
             * Reduce down to 1 number
             * @param x the input
             * @param xShapeInfo the shape information
             * for the input
             * @param extraParams the extra params
             * @return
             */
#ifdef __CUDACC__
			__host__
#endif
			T execScalar(const T *x,int xElementWiseStride,Nd4jIndex length,T *extraParams) {
				T startingVal = this->startingValue(x);
				if (xElementWiseStride == 1) {
					if(length < 8000) {
						T local = this->startingValue(x);
#pragma omp simd
						for(Nd4jIndex i = 0; i < length; i++) {
							T curr = op(x[i], extraParams);
							local = update(local, curr, extraParams);

						}
						local = postProcess(local, length,extraParams);

						return local;
					}

					else {
						T finalVal = startingVal;
						BlockInformation info(length);

#pragma omp parallel
						{
							T local = this->startingValue(x);
							for(int i = omp_get_thread_num(); i < info.chunks; i+= info.threads) {
								Nd4jIndex newOffset = (i * info.items);
								const T *chunk = x + newOffset;
								Nd4jIndex itemsToLoop = info.items;
								if(newOffset >= length) {
									break;
								}

								//handle modulo case
								if(newOffset + info.items >= length) {
									itemsToLoop = length - newOffset;
								}

								for (Nd4jIndex j = 0; j < itemsToLoop; j++) {
									T curr = op(chunk[j], extraParams);
									local = update(local, curr, extraParams);
								}

							}

#pragma omp critical
							{
								finalVal = update(finalVal,local,extraParams);

							}
						}


						finalVal = postProcess(finalVal, length,extraParams);
						return finalVal;

					}

				}

				else {
					if(length < 8000) {
						T local = this->startingValue(x);
#pragma omp simd
						for(Nd4jIndex i = 0; i < length; i++) {
							T curr = op(x[i *xElementWiseStride], extraParams);
							local = update(local, curr, extraParams);

						}

						local = postProcess(local, length,extraParams);

						return local;
					}

					T finalVal = startingVal;
					BlockInformation info(length);

#pragma omp parallel
					{
						T local = this->startingValue(x);
						for(int i = omp_get_thread_num(); i < info.chunks; i+= info.threads) {
							Nd4jIndex newOffset = (i * info.items) * xElementWiseStride;
							const T *chunk = x + newOffset;
							Nd4jIndex itemsToLoop = info.items;


							for (Nd4jIndex i = 0; i < itemsToLoop; i++) {
								T curr = op(chunk[i * xElementWiseStride], extraParams);
								local = update(local, curr, extraParams);
							}


						}

#pragma omp critical
						{
							finalVal = update(finalVal,local,extraParams);

						}
					}


					finalVal = postProcess(finalVal, length,extraParams);
					return finalVal;

				}

			}



			/**
             * Reduce down to 1 number
             * @param x the input
             * @param xShapeInfo the shape information
             * for the input
             * @param extraParams the extra params
             * @return
             */
#ifdef __CUDACC__
			__host__
#endif
			T execScalar(T *x, int *xShapeInfo,T *extraParams) {
				const Nd4jIndex length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				if(xElementWiseStride >= 1) {
					return execScalar(x, xElementWiseStride, length, extraParams);
				}
				else {
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];

					int *xShape = shape::shapeOf(xShapeInfo);
					int *xStride = shape::stride(xShapeInfo);
					T start = this->startingValue(x);
					int rank = shape::rank(xShapeInfo);

					if(PrepareOneRawArrayIter<T>(rank,
												 xShape,
												 x,
												 xStride,
												 &rank,
												 shapeIter,
												 &x,
												 xStridesIter) >= 0) {

						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							/* Process the innermost dimension */
							const T *xIter = x;
							start = update(start,op(xIter[0],extraParams),extraParams);
						} ND4J_RAW_ITER_ONE_NEXT(dim,
												 rank,
												 coord,
												 shapeIter,
												 x,
												 xStridesIter);
						start = postProcess(start,shape::length(xShapeInfo),extraParams);
					}
					else {
						printf("Unable to prepare array\n");
					}

					return start;


				}

			}

			/**
             * Execute on the cpu
             * @param x the input data
             * @param xShapeInfo the shape information for x
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to perform
             * the reduce along long
             * @param dimensionLength the length of the dimension buffer
             */
			virtual
#ifdef __CUDACC__
			__host__
#endif
			void exec(T *x,
					  int *xShapeInfo,
					  T *extraParams,
					  T *result,
					  int *resultShapeInfoBuffer,
					  int *dimension,
					  int dimensionLength) {
				int resultLength = shape::length(resultShapeInfoBuffer);

				if(resultLength == 1 || dimensionLength == shape::rank(xShapeInfo)) {
					result[0] = execScalar(x,xShapeInfo,extraParams);
					return;
				}

				if(dimensionLength > 1) {
					int numOnes = 0;
					int *shape = shape::shapeOf(xShapeInfo);
					int wholeRank = shape::rank(xShapeInfo);
					bool squeezed = false;
					bool newSqueezeDimensions = false;

					for (int i = 0; i < wholeRank; i++) {
						if (shape[i] == 1)
							numOnes++;
					}

					//squeeze the dimensions
					if (numOnes > 0) {
						xShapeInfo = shape::squeezeDimensions(
								xShapeInfo,
								dimension,
								dimensionLength,
								&squeezed,
								&newSqueezeDimensions,
								wholeRank,
								numOnes);
					}

					//decompose in to several sub tads after
					//moving all dimensions (in sorted order)
					//to the back.
					//permuted version of the x shape info for setting up the tad problem
					int *tadShapeShapeInfo = shape::shapeInfoOnlyShapeAndStride(xShapeInfo,dimension,dimensionLength,false);
					int *xShape = shape::shapeOf(tadShapeShapeInfo);
					int *xStride = shape::stride(tadShapeShapeInfo);
					int tadLength = shape::length(tadShapeShapeInfo);
					int rank = shape::rank(tadShapeShapeInfo);
#pragma omp  parallel  for
					for(int i = 0; i < resultLength; i++) {
						int offset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);


						int shapeIter[MAX_RANK];
						int coord[MAX_RANK];
						int dim;
						int rankIter = rank;
						int xStridesIter[MAX_RANK];
						T *xPointer = x + offset;

						T start = this->startingValue(xPointer);
						if(PrepareOneRawArrayIter<T>(rankIter,
													 xShape,
													 xPointer,
													 xStride,
													 &rankIter,
													 shapeIter,
													 &xPointer,
													 xStridesIter) >= 0) {
							ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
								/* Process the innermost dimension */
								start = update(start,op(xPointer[0],extraParams),extraParams);
							} ND4J_RAW_ITER_ONE_NEXT(dim,
													 rank,
													 coord,
													 shapeIter,
													 xPointer,
													 xStridesIter);
							start = postProcess(start,tadLength,extraParams);
						}
						else {
							printf("Unable to prepare array\n");
						}

						result[i] = start;

					}


					//	free(tadShapeShapeInfo);


					if (newSqueezeDimensions) {
						free(dimension);
					}

					//	if (numOnes > 0) {
					//		free(xShapeInfo);
					//	}
				}

				else {
					if(shape::order(xShapeInfo) == 'f') {
						int tadElementWiseStride = shape::reductionIndexElementWiseStride(xShapeInfo, dimension, dimensionLength);
						int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
#pragma omp parallel for
						for(int i = 0;  i < resultLength; i++) {
							int baseOffset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);
							T currResult = op(x[baseOffset],extraParams);
							for(int j = 1; j < tadLength; j++) {
								currResult = update(currResult,op(x[baseOffset + j * tadElementWiseStride],extraParams),extraParams);
							}

							result[i] = postProcess(currResult,tadLength,extraParams);
						}

					}
					else {
						int tadElementWiseStride = shape::reductionIndexElementWiseStride(xShapeInfo, dimension, dimensionLength);
						int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
#pragma omp parallel for
						for(int i = 0;  i < resultLength; i++) {
							int baseOffset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);
							T currResult = op(x[baseOffset],extraParams);
							result[i] = currResult;
							for(int j = 1; j < tadLength; j++) {
								currResult = op(x[baseOffset + j * tadElementWiseStride],extraParams);
								result[i] = update(result[i],currResult,extraParams);
							}

							result[i] = postProcess(result[i],tadLength,extraParams);
						}

					}

				}
			}

			virtual inline
#ifdef __CUDACC__
			__host__ __device__
#endif
			void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
				// no extra params aggregation needs to happen
			}

			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			T startingValue(const T *input) = 0;




		};

#ifdef __CUDACC__
		/**
 *
 * @param extraParams
 * @param sPartials
 * @param sMemSize
 */
template<typename T>
__device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize) {
	int sPartialsLength = sMemSize / sizeof(T);
	T *sPartialsDeref = (T *) *sPartials;
	for (int i = 0; i < sPartialsLength; i++) {
		sPartialsDeref[i] = extraParams[0];
	}
}

#endif

		namespace ops {
/**
 * Summation operation
 */
			template<typename T>
			class Sum: public virtual functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return (T) 0.0;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput + old;
				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return opOutput + old;
				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Sum() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Sum() {
				}
			};

/**
 * The product operation
 */
			template<typename T>
			class Prod: public virtual functions::reduce::ReduceFunction<T> {
			public:

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput * old;
				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return opOutput * old;
				}
				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction;
				}

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return 1.0;
				}


				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				~Prod() {
				}
#ifdef __CUDACC__
				__host__ __device__
#endif
				Prod() {
				}
			};

/**
 * Mean operation
 */
			template<typename T>
			class Mean: public virtual functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return 0.0;
				}


				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput + old;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return opOutput + old;
				}
				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction / (T) n;
				}

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				~Mean() {
				}
#ifdef __CUDACC__
				__host__ __device__
#endif
				Mean() {
				}
			};


/**
 * Max reduction
 */
			template<typename T>
			class Max: public virtual functions::reduce::ReduceFunction<T> {
			public:

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return nd4j::math::nd4j_max<T>(old, opOutput);
				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return nd4j::math::nd4j_max<T>(opOutput, old);
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction;
				}


				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					return input[0];
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Max() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Max() {
					this->indexBased = 1;
				}
			};

/**
 * Min operation
 */
			template<typename T>
			class Min: public virtual functions::reduce::ReduceFunction<T> {
			public:

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}


				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return nd4j::math::nd4j_min<T>(old, opOutput);
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return nd4j::math::nd4j_min<T>(opOutput, old);
				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction;
				}

				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					return input[0];
				}


				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Min() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Min() {
					this->indexBased = 1;
				}
			};

/**
 * Norm1 of a buffer
 */
			template<typename T>
			class Norm1: public virtual functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return 0.0;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput + old;

				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return opOutput + old;

				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return nd4j::math::nd4j_abs<T>(d1);
				}

				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return reduction;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Norm1() {}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Norm1() {}
			};

/**
 * Norm2 of an array
 */
			template<typename T>
			class Norm2: public virtual functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return 0.0;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}


				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput + old;

				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return opOutput + old;

				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1 * d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return nd4j::math::nd4j_sqrt<T>(reduction);
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Norm2() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Norm2() {
				}
			};

/**
 * Norm max of an array
 */
			template<typename T>
			class NormMax: public virtual functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) {
					(void)input;
					return 0.0;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}


				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return opOutput + old;

				}

				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(old),
												   nd4j::math::nd4j_abs<T>(opOutput));

				}
				virtual
#ifdef __CUDACC__
				__host__  __device__

#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					return d1;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(reduction),
												   nd4j::math::nd4j_abs<T>(reduction));
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~NormMax() {
				}

#ifdef __CUDACC__
				inline __host__ __device__
#endif
				NormMax() {
				}
			};

			template<typename T>
			class Variance: public  functions::reduce::ReduceFunction<T> {
			public:
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(const T *input) override {
					(void)input;
					return 0.0;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				ReduceFunction<T> ** extraParamsFunctions() {
					return NULL;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T merge(T old, T opOutput, T *extraParams) override {
					return old + opOutput;

				}
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T update(T old, T opOutput, T *extraParams) override {
					return old + opOutput;

				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__


#elif defined(__GNUC__)


#endif
				T op(T d1, T *extraParams) override {
					T mean = extraParams[0];
					T ret = d1 - mean;
					return ret * ret;
				}

				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					T bias = extraParams[1];
					return (reduction - (nd4j::math::nd4j_pow<T>(bias, 2.0) / (T) n))
						   / (T) (n - 1.0);
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~Variance() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				Variance() {
					this->extraParamsLength = 2;
				}
			};

/**
 * Standard deviation of a buffer
 */
			template<typename T>
			class StandardDeviation: public virtual Variance<T> {
			public:


				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				T postProcess(T reduction, Nd4jIndex n,T *extraParams) override {
					T ret = Variance<T>::postProcess(reduction,n,extraParams);
					T sqrtRet = nd4j::math::nd4j_sqrt<T>(ret);
					return sqrtRet;
				}

				virtual
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				~StandardDeviation() {
				}
#ifdef __CUDACC__
				inline __host__ __device__
#endif
				StandardDeviation() : Variance<T>() {
				}
			};



		}

		template<typename T>
		class ReduceOpFactory: public virtual functions::ops::OpFactory<T> {

		public:
#ifdef __CUDACC__
			__device__ __host__
#endif
			ReduceOpFactory() {
			}

			/**
             * Create an operation given an op number
             * @param op the operation number
             * 0: mean
             * 1: sum
             * 2: bias
             * 3: max
             * 4: min
             * 5: norm1
             * 6: norm2
             * 7: normmaxc
             * 8: prod
             * 9: std
             * 10: variance
             * @return
             */
#ifdef __CUDACC__
			__inline__ __device__ __host__
#endif

			virtual functions::reduce::ReduceFunction<T> * create(int op) {
				if (op == 0)
					return new functions::reduce::ops::Mean<T>();
				else if (op == 1)
					return new functions::reduce::ops::Sum<T>();
				else if (op == 3)
					return new functions::reduce::ops::Max<T>();
				else if (op == 4)
					return new functions::reduce::ops::Min<T>();
				else if (op == 5)
					return new functions::reduce::ops::Norm1<T>();
				else if (op == 6)
					return new functions::reduce::ops::Norm2<T>();
				else if (op == 7)
					return new functions::reduce::ops::NormMax<T>();
				else if (op == 8)
					return new functions::reduce::ops::Prod<T>();
				else if (op == 9)
					return new functions::reduce::ops::StandardDeviation<T>();
				else if (op == 10)
					return new functions::reduce::ops::Variance<T>();

				return NULL;
			}


#ifdef __CUDACC__
			__device__ __host__
#endif

			virtual ~ReduceOpFactory() {
			}
		};

	}

}


#ifdef __CUDACC__
/**
 * Interface for the c and driver api
 * @param op the operation number
 * @param n the length of the problem
 * @param dx  the input information
 * @param xShapeInfo the shape information
 * @param extraParams the extra parameters
 * @param result the result data
 * @param resultShapeInfo the result shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__global__ void reduceGenericGlobal(
		int op,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		int *allocationBuffer, T *reductionBuffer) {

	__shared__ functions::reduce::ReduceFunction<T> *reduceFunctionToInvoke;
	__shared__ functions::reduce::ReduceOpFactory<T> *newOpFactory;

	if(threadIdx.x == 0)
		newOpFactory =  new functions::reduce::ReduceOpFactory<T>();
	__syncthreads();

	if(threadIdx.x == 0)
		reduceFunctionToInvoke = newOpFactory->create(op);
	__syncthreads();
	reduceFunctionToInvoke->transformCuda(
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationBuffer, reductionBuffer);

	__syncthreads();
	if(threadIdx.x == 0) {
		delete reduceFunctionToInvoke;
		delete newOpFactory;
	}

}

/**
 * Interface for the c and driver api
 * @param op the operation number
 * @param n the length of the problem
 * @param dx  the input information
 * @param xShapeInfo the shape information
 * @param extraParams the extra parameters
 * @param result the result data
 * @param resultShapeInfo the result shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__device__ void reduceGeneric(
		int op,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		int *allocationBuffer, T *reductionBuffer) {
	__shared__ functions::reduce::ReduceFunction<T> *reduceFunctionToInvoke;
	__shared__ functions::reduce::ReduceOpFactory<T> *newOpFactory;

	if(threadIdx.x == 0)
		newOpFactory =  new functions::reduce::ReduceOpFactory<T>();
	__syncthreads();

	if(threadIdx.x == 0)
		reduceFunctionToInvoke = newOpFactory->create(op);
	__syncthreads();
	reduceFunctionToInvoke->transformCuda(
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationBuffer, reductionBuffer);

	__syncthreads();
	if(threadIdx.x == 0) {
		delete reduceFunctionToInvoke;
		delete newOpFactory;
	}

}

/**
 * Interface for the c and driver api
 * @param op the operation number
 * @param n the length of the problem
 * @param dx  the input information
 * @param xShapeInfo the shape information
 * @param extraParams the extra parameters
 * @param result the result data
 * @param resultShapeInfo the result shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void reduceDouble(
		int op,
		double *dx,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		int *allocationBuffer, double *reductionBuffer) {
	reduceGeneric<double>(
			op,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationBuffer, reductionBuffer);

}

/**
 * Interface for the c and driver api
 * @param op the operation number
 * @param n the length of the problem
 * @param dx  the input information
 * @param xShapeInfo the shape information
 * @param extraParams the extra parameters
 * @param result the result data
 * @param resultShapeInfo the result shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void reduceFloat(
		int op,
		float *dx,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		int *allocationBuffer,
		float *reductionBuffer
		) {
	reduceGeneric<float>(
			op,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationBuffer, reductionBuffer);
}



#endif

