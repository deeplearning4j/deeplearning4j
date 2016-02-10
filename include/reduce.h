#include <string>
#include <sharedmem.h>
#include <stdio.h>
#include <shape.h>
#include <op.h>
#include <omp.h>
#include <templatemath.h>
#include <helper_cuda.h>
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
            virtual
#ifdef __CUDACC__
            __host__
#endif
            T * generateExtraParams(T *input,int *shapeInfo) {
                T *ret = createExtraParams();
                ReduceFunction<T> **functions = this->extraParamsFunctions();
                for(int i = 0; i < getExtraParamsLength(); i++) {
                    ReduceFunction<T> *r = functions[i];
                    //assume the param ordering is composed
                    //of the previous parameters
                    //the most prominent example being
                    //variance and std which requires
                    //mean followed by bias
                    T val = r->execScalar(input,shapeInfo,ret);
                    ret[i] = val;
                    delete functions[i];
                }

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
            __always_inline

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
            __always_inline

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
            __always_inline

#endif
            T update(T old, T opOutput, T *extraParams) = 0;
#ifdef __CUDACC__



            /**
	 * Kernel invocation for reduce
	 * @param n the length of the buffer
	 * @param dx the input
	 * @param xShapeInfo the shape information for the input
	 * @param extraParams extra parameters (starting value,..)
	 * @param result the result buffer
	 * @param resultShapeInfo the shapeinformation for the result buffer
	 * @param gpuInformation the gpu information (shared memory allocated,..)
	 * @param dimension the dimension to do reduce along
	 * @param dimensionLength the length of the dimension buffer
	 * @param postProcessOrNot whether to reduce or not
	 */
	__inline__ __device__ virtual void transform(
			int n,
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *gpuInformation,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot) {

		/**
		 * Gpu information for the problem
		 */
		int tid = threadIdx.x;

		__shared__ volatile int resultScalar;

		__shared__ int xElementWiseStride;
		__shared__ int xOffset;

		//shared memory space for storing intermediate results
		SharedMemory <T> val;
		volatile T *sPartials = val.getPointer();
		int numElements = gpuInformation[2] / sizeof(T);
		T init = this->startingValue(dx);
		for (int i = tid; i < numElements; i += blockDim.x)
			sPartials[i] = init;
		__syncthreads();

		//length for the tad
		__shared__ int xLength;

		__shared__ int resultLength;

		__shared__ int elementsPerTad;


		//only compute the tad indexes once
		__shared__ shape::TADPermuteInfo xTadInfo;

		__syncthreads();

		T reduction = this->startingValue(dx);
		if (tid == 0) {
			resultLength = shape::length(resultShapeInfo);
			if (dimensionLength == 1) {
				if (dimension[0] == shape::MAX_DIMENSION)
					resultScalar = 1;
				else
					resultScalar = 0;
			}
			else
				resultScalar = 0;

			if (resultLength == 1)
				resultScalar = 1;
			xOffset = shape::offset(xShapeInfo);
			/**
			 * The element wise stride belongs to a reduction index.
			 * When used out of order, we can get rid of the data
			 * dependencies and rely on using the max dimension
			 * specified for stride instead.
			 * Say we take the sum(0,1) along arr
			 * we can use arr.stride(1) as a representation
			 * along which to iterate.
			 */
			int tadElementWiseStride = dimensionLength > 1 ? shape::stride(xShapeInfo)[dimensionLength - 1] : shape::computeElementWiseStride(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),shape::order(xShapeInfo) == 'f',dimension,dimensionLength);

			xElementWiseStride = tadElementWiseStride;
			xLength = shape::length(xShapeInfo);
			elementsPerTad = xLength / resultLength;
	}
		__syncthreads();

		if (!resultScalar) {
			if(tid == 0) {
				xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
			}
			__syncthreads();

			int resultLength = shape::length(resultShapeInfo);
			if(tid >= resultLength)
				return;

			/**
			 * The element wise stride belongs to a reduction index.
			 * When used out of order, we can get rid of the data
			 * dependencies and rely on using the max dimension
			 * specified for stride instead.
			 * Say we take the sum(0,1) along arr
			 * we can use arr.stride(1) as a representation
			 * along which to iterate.
			 */
			int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
			int tadLength = xTadInfo.tensorShapeProd;
			int xLength = shape::length(xShapeInfo);
			int i = 0,j = 0;
#pragma unroll
			for(i = tid; i < resultLength; i+= blockDim.x * gridDim.x) {
				int offsetForTad = shape::offset(i, xShapeInfo, dimension,dimensionLength, xTadInfo);
				sPartials[tid] = op(dx[offsetForTad], extraParams);
				__syncthreads();
				for(j = 1; j < elementsPerReductionIndex; j++) {
					sPartials[tid] =  update(sPartials[tid],op(dx[offsetForTad + xElementWiseStride * j], extraParams), extraParams);
					__syncthreads();
				}

				result[i] = postProcess(sPartials[tid],tadLength,extraParams);
			}


			if(tid == 0) {
				shape::freePermuteInfo(xTadInfo);
			}

		}
		else {

			T curr;
			if (resultScalar) {
				if(blockIdx.x >= resultLength)
					return;


				T *realExtraParams;
				if(tid == 0) {
					realExtraParams = extraParams;

				}

				__syncthreads();
				/**
				 * Need to look closer
				 * at how to handle the params
				 * wrt each offset at each tad
				 *
				 * An idea would be to calculate and
				 * save the tad offset we could use
				 * for the statistics and extra params.
				 *
				 * Another option would be to have a shared variable
				 * for handling the computation of different
				 * extra parameters wrt
				 * the offsets of each dimension.
				 *
				 *
				 */
				unsigned int i = blockIdx.x * xElementWiseStride + tid;
				unsigned int gridSize = blockDim.x * gridDim.x * xElementWiseStride;
				if(xOffset == 0) {
					if(!this->indexBased) {
#pragma unroll
						while (i < n) {
							curr = op(dx[i],realExtraParams);
							reduction = update(reduction,curr, realExtraParams);
							i += gridSize;
						}
					}
					else {
#pragma unroll
						while (i < n) {
							int tadIndex = shape::tadIndexForLinear(i,elementsPerTad);
							if(tadIndex == 0) {
								curr = op(dx[i],realExtraParams);
								reduction = curr;
							}
							else {
								curr = op(dx[i],realExtraParams);
								reduction = update(reduction,curr, realExtraParams);
							}

							i += gridSize;
						}
					}

				}
				else {
					// we reduce multiple elements per thread.  The number is determined by the
					// number of active thread blocks (via gridDim).  More blocks will result
					// in a larger gridSize and therefore fewer elements per thread
					if(this->indexBased)
#pragma unroll
						while (xOffset + i < n) {
							curr = op(dx[xOffset + i],realExtraParams);
							reduction = update(reduction,curr, realExtraParams);
							i += gridSize;
						}
					else {
#pragma unroll
						while (xOffset + i < n) {
							int tadIndex = shape::tadIndexForLinear(i,elementsPerTad);
							if(tadIndex == 0) {
								curr = op(dx[xOffset + i],realExtraParams);
							}
							else {
								curr = op(dx[xOffset + i],realExtraParams);
								reduction = update(reduction,curr, realExtraParams);
							}

							i += gridSize;
						}
					}

				}

				// each thread puts its local sum into shared memory
				sPartials[tid] = reduction;
				__syncthreads();
				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials(sPartialsRef, tid, numElements,realExtraParams);

				// write result for this block to global mem
				if (tid == 0) {
					if(postProcessOrNot)
						result[blockIdx.x] = this->postProcess(sPartials[0],n,realExtraParams);
					else
						result[blockIdx.x] = sPartials[0];
					if(extraParamsLength >= 1)
						delete[] realExtraParams;
				}



			}


		}

	}

	/**
	 * This implements a collapsing tad reduction
	 * based on different dimensions.
	 *
	 * The reason we need this is because of the fact that
	 * there are certain dimension combinations (usually > 1)
	 * that don't have an element wise stride.
	 *
	 * A way to bypass this problem is to expand the problem
	 * in to a 1 dimension reduction problem
	 * and then collapsing the results in to the equivalent
	 * shape of the multi dimension problem.
	 *
	 * An example problem would be an array of:
	 * linspace(1,24,24).reshape(2,2,3,2)
	 *
	 * The tad for reduction:
	 * 2,3 doesn't have an element wise stride.
	 *
	 * However, the tad for reduction:
	 * 3 does
	 *
	 * What we can exploit here is the ability
	 * to reshape problems of multiple dimensions
	 *
	 * in to equivalent expanded problems based on smaller tads
	 * eg:
	 * multiple reductions for each dimension along dimension 3
	 * followed by collapsing the problem in to an equivalent state
	 * as if we had specified 2,3 for the dimensions instead.
	 *
	 * This gives us a way of executing an element wise stride based
	 * algorithm  that is executable on the gpu.
	 *
	 * For the GPU, we force each block to process a  tad
	 * at the singular dimension level. Eg: dimension 3
	 *
	 * So for example along dimension 3 of the 2,2,3,2
	 * array we have 12 tensors along dimension.
	 *
	 * We then map those 12 tads to a reduction index.
	 *
	 * A reduction index is the equivalent value
	 * in teh result as if we had specified the reduction dimensions
	 * to be 2,3 instead.
	 *
	 * For example, if we have 12 tads for dimension 3
	 * we will only have 4 for dimensions 2,3
	 *
	 * The goal will be then to generate the equivalent results
	 * using dimension 3 but collapsing the results according to
	 * the dimension 2,3 space (remember: the reason we are doing this mapping
	 * is because we are trying to map the multi dimensional problem on to
	 * a problem that allows us to solve it via element wise stride)
	 *
	 *
	 * An example mapping relative to a gpu block is as follows:
	 * ([[[[  1.,   2.],
	 [  3.,   4.],
	 [  5.,   6.]],

	 [[  7.,   8.],
	 [  9.,  10.],
	 [ 11.,  12.]]],


	 [[[ 13.,  14.],
	 [ 15.,  16.],
	 [ 17.,  18.]],

	 [[ 19.,  20.],
	 [ 21.,  22.],
	 [ 23.,  24.]]]])



	 * Along dimension 3 we will have tads of length 2
	 * and 4 reduction indexes we need to map for the
	 * 2,3 dimension problem.
	 *
	 *
	 * The first reduction index will map to the first 3 tads of length 2
	 * The next reduction index will map to the next 3, etc.
	 *
	 * We then process a reduction index per block on the gpu.
	 * If any gpu block index is > the number of
	 * reduction indexes we skip it.
	 *
	 * Note here we did this implementation because of
	 * race conditions on the block and shared memory.
	 *
	 * This way of mapping allows us to avoid race conditions.
	 *
	 * @param data the data to process
	 * @param result the result vector
	 * @param initialValue the initial value for the reductino
	 * @param elementsPerTad the elements per tad
	 * for the expanded tad (eg: the one being collapsed from)
	 * @param numTads the number of tads for the final result
	 * @param n the number of elements in the buffer total
	 * @param elementWiseStride the element wise stride
	 * we use for the singular dimensions for each tad
	 * @param numOriginalTads the number of original tads for the expanded version (eg: we are doing
	 * reduction mapping a single dimension problem that allows for an element wise stride on to a multi
	 * index problem)
	 * @param sharedMemorySize the shared memory size we specified for launching the kernel - this is used for figuring out
	 * how many elements are possible for the shared memory buffer for initializing the values to be default
	 * @param xShapeInfo the shape information for the buffer - for more information on this see tad.h
	 * @param dimension the dimension for the problem on the smaller scale (eg: the expanded version of the problem)
	 * @param dimensionLength the length of the number of dimensions
	 *
	 */
	__device__ virtual void collapseTad(
			T *data,
			T *result,
			T *extraParams,
			int numOriginalTads,
			int sharedMemorySize,
			int *xShapeInfo,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength) {
		SharedMemory <T> val;
		//number of tads for the reduced solution
		int numTads = shape::tensorsAlongDimension(xShapeInfo, dimension, dimensionLength);

		volatile T *sPartials = val.getPointer();
		int tid = threadIdx.x;
		//initialize the values
		int numItems = sharedMemorySize / sizeof(T);
		T initialShapredValue = this->startingValue(data);
		for (int i = tid; i < numItems; i += blockDim.x) {
			sPartials[i] = initialShapredValue;
		}
		__syncthreads();

		//each block processes a reduction index
		//don't bother iterating on this block if it goes over the number of tads

		__shared__ shape::TADPermuteInfo xTadInfo;

		if (tid == 0) {
			xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
		}

		__syncthreads();

		/**
		 * Reverse engineer which tads belong to a particular
		 * reduction index.
		 *
		 * Each tad should be handled by a thread.
		 *
		 * Combine them all in the block at the end.
		 *
		 *
		 */

		//number of tads per reduce index
		__shared__ int tadsPerReduceIndex2;
		if (tid == 0) {
			tadsPerReduceIndex2 = shape::tadsPerReduceIndex(numTads, numOriginalTads);
		}

		__syncthreads();

		//each thread does a tad
		if (tid >= numTads || blockIdx.x >= tadsPerReduceIndex2)
			return;

		/**
		 * Need to ensure we stay in bounds on each block -
		 * we need to compute the proper tads for each block and
		 * do bounds checking on each thread.
		 *
		 * This is to ensure that each thread processes
		 * a unique tad at most once.
		 *
		 *
		 */
		/**
		 * NEXT PART HERE
		 */

		/**
		 * Now WRT the thread id
		 * we want to iterate through a tad
		 * on each thread using the element wise stride
		 * and num elements per tad to compute a reduce
		 * for the tad. We then reduce in shared memory
		 * setting the item in the shared memory space
		 * and aggregate all of thh partial results
		 * on thread 0 aggregating the final results
		 * on the block resulting in one global write.
		 */
		//compute the offset for the tad for this thread
		//iterating via element wise stride
		//note here blockidx.x + tid is the tad we want
		int tadForThread = tid + blockIdx.x * tadsPerReduceIndex2;
		int offsetForBlock = shape::offset(tadForThread, xShapeInfo, dimension,dimensionLength, xTadInfo);
#pragma unroll
		for (int i = 0; i < tadsPerReduceIndex2; offsetForBlock += shape::elementWiseStride(xShapeInfo), i++) {
			sPartials[tid] = update(sPartials[tid], op(data[offsetForBlock], extraParams), extraParams);
			__syncthreads();
		}

		if (tid == 0 && blockIdx.x < numTads) {
			//start at 1 so we don't count the first entry twice
#pragma unroll
			for (int i = 1; i < numTads; i++) {
				sPartials[0] = update(sPartials[0], sPartials[i], extraParams);
				__syncthreads();
			}

			result[blockIdx.x] = sPartials[0];
			shape::freePermuteInfo(xTadInfo);
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
            __always_inline

#endif
            T postProcess(T reduction, int n, T *extraParams)  {
                return reduction;
            }

#ifdef __CUDACC__
            __inline__ __host__
#endif
            T aggregateBuffer(int n,T *buffer,T *extraParams) {

                T ret = buffer[0];
#pragma omp simd
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
                T startingVal = this->startingValue(x);
                const int length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
                if (xElementWiseStride == 1 && resultElementWiseStride == 1) {

                    int i;
#pragma omp parallel private (i)
                    {
#pragma omp simd
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            T curr = op(x[i], extraParams);
                            startingVal = update(startingVal, curr, extraParams);

                        }
                    }

                    T finalVal = postProcess(startingVal, length,extraParams);
                    result[0] = finalVal;
                } else {

                    int i;

#pragma omp parallel private(i)
                    {
#pragma omp simd
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            startingVal = update(startingVal,
                                                 op(x[i * xElementWiseStride], extraParams),
                                                 extraParams);
                        }

                        result[0] = postProcess(startingVal, length, extraParams);

                    }
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
            T execScalar(T *x,int xElementWiseStride,int length,T *extraParams) {
                T startingVal = this->startingValue(x);
                int i;
                if (xElementWiseStride == 1) {

                    T finalVal = startingVal;
#pragma omp parallel shared(finalVal)
                    {
                        T local = startingVal;

#pragma omp for nowait
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            T curr = op(x[i], extraParams);
                            local = update(local, curr, extraParams);
                        }

#pragma omp critical
                        {
                            finalVal = update(finalVal,local,extraParams);
                        }


                    }

                    finalVal = postProcess(finalVal, length,extraParams);
                    return finalVal;



                }

                else {
                    T finalVal = startingVal;
#pragma omp parallel shared(finalVal)
                    {
                        T local = startingVal;

#pragma omp for nowait
                        for (i = omp_get_thread_num(); i < length; i+= omp_get_num_threads()) {
                            T curr = op(x[i * xElementWiseStride], extraParams);
                            local = update(local, curr, extraParams);
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
                const int length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                return execScalar(x,xElementWiseStride,length,extraParams);

            }

            /**
             * Execute on the cpu
             * @param x the input data
             * @param xShapeInfo the shape information for x
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to perform
             * the reduce along
             * @param dimensionLength the length of the dimension buffer
             */
            virtual
#ifdef __CUDACC__
            __host__
#endif
            void exec(T *x, int *xShapeInfo,
                      T *extraParams,
                      T *result,
                      int *resultShapeInfoBuffer,
                      int *dimension,
                      int dimensionLength) {

                if(shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar(x,xShapeInfo,extraParams);
                    return;
                }

                if(dimensionLength > 1) {
                    shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo,dimension, dimensionLength);
                    const int resultLength = shape::length(resultShapeInfoBuffer);
                    /**
                     * The element wise stride belongs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along arr
                     * we can use arr.stride(1) as a representation
                     * along which to iterate.
                     */
                    int tadElementWiseStride = dimensionLength > 1 ? shape::stride(xShapeInfo)[dimensionLength - 1] : shape::computeElementWiseStride(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),shape::order(xShapeInfo) == 'f',dimension,dimensionLength);
                    const int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
                    int tadLength = tadPermuteInfo.tensorShapeProd;

                    int i,j;
#pragma omp parallel for  private(i,j)
                    for(i = omp_get_thread_num(); i < resultLength; i+= omp_get_num_threads()) {
                        int offset = shape::offset(i,xShapeInfo,dimension,dimensionLength,tadPermuteInfo);
                        if(dimensionLength > 1) {
                            offset = i;
                        }
                        else if(dimensionLength == 1) {
                            if(tadElementWiseStride == 1)
                                offset = i * tadLength;
                            else
                                offset = i;
                        }

                        result[i] = op(x[offset], extraParams);
#pragma omp simd
                        for(j = 1; j < elementsPerReductionIndex; j++) {
                            result[i] =  update(result[i],op(x[offset + tadElementWiseStride * j], extraParams), extraParams);
                        }

                        result[i] = postProcess(result[i],tadLength,extraParams);

                    }

                    shape::freePermuteInfo(tadPermuteInfo);
                }

                else {
                    int tadElementWiseStride = shape::tadElementWiseStride(xShapeInfo, dimension, dimensionLength);
                    int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                    const int resultLength = shape::length(resultShapeInfoBuffer);
                    int i, baseOffset;
#pragma omp parallel private(i,baseOffset)
                    {
                        int nThreads = omp_get_num_threads();
                        for(i = omp_get_thread_num(),baseOffset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);
                            i < resultLength; i+= nThreads,baseOffset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength)) {
                            T currResult = op(x[baseOffset],extraParams);
                            result[i] = currResult;
                            for(int j = 1; j < tadLength; j++) {
                                currResult = op(x[baseOffset + j * tadElementWiseStride],extraParams);
                                result[i] = update(result[i],currResult,extraParams);
                            }

                            if(i < resultLength)
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
            T startingValue(T *input) = 0;




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
                T startingValue(T *input) {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("sum");
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("prod");
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput * old;
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput * old;
                }
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
                    return reduction;
                }

                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T startingValue(T *input) {
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
                T startingValue(T *input) {
                    return 0.0;
                }
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() override {
                    return std::string("mean");
                }

                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
 * Bias operation for statistics
 */
            template<typename T>
            class Bias: public virtual functions::reduce::ReduceFunction<T> {
            public:
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T startingValue(T *input) {
                    return 0.0;
                }
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                ReduceFunction<T> ** extraParamsFunctions() {
                    ReduceFunction<T> ** ret = (ReduceFunction<T> **) malloc(sizeof(ReduceFunction<T> **));
                    ret[0] = new Mean<T>();
                    return ret;
                }
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() override {
                    return std::string("bias");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;
                }


                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    T mean = extraParams[0];
                    T curr = (d1 - mean);
                    return curr;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
                    return reduction;
                }

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#endif
                ~Bias() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#endif
                Bias() {
                    this->extraParamsLength = 1;
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("max");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return nd4j::math::nd4j_max<T>(old, opOutput);
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return nd4j::math::nd4j_max<T>(opOutput, old);
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
                    return reduction;
                }


                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T startingValue(T *input) {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("min");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return nd4j::math::nd4j_min<T>(old, opOutput);
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return nd4j::math::nd4j_min<T>(opOutput, old);
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
                    return reduction;
                }

                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T startingValue(T *input) {
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
                T startingValue(T *input) {
                    return 0.0;
                }
                virtual
#ifdef __CUDACC__
                __host__

#endif
                std::string name() override {
                    return std::string("norm1");
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;

                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;

                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return nd4j::math::nd4j_abs<T>(d1);
                }

                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                T startingValue(T *input) {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("norm2");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;

                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;

                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1 * d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                T startingValue(T *input) {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("normmax");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return opOutput + old;

                }

                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return nd4j::math::nd4j_max<T>(nd4j::math::nd4j_abs<T>(old),
                                                   nd4j::math::nd4j_abs<T>(opOutput));

                }
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *extraParams) override {
                    return d1;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                T startingValue(T *input) {
                    return 0.0;
                }
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                ReduceFunction<T> ** extraParamsFunctions() {
                    ReduceFunction<T> **ret = (ReduceFunction<T> **) malloc(sizeof(ReduceFunction<T>) * 2);
                    Mean<T> *mean = new Mean<T>();
                    Bias<T> *bias = new Bias<T>();
                    ret[0] = mean;
                    ret[1] = bias;
                    return ret;
                }
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() override {
                    return std::string("var");
                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T merge(T old, T opOutput, T *extraParams) override {
                    return old + opOutput;

                }
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T update(T old, T opOutput, T *extraParams) override {
                    return old + opOutput;

                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__


#elif defined(__GNUC__)
                __always_inline

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
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                inline __host__

#endif
                std::string name() override {
                    return std::string("std");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T postProcess(T reduction, int n,T *extraParams) override {
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
                else if (op == 2)
                    return new functions::reduce::ops::Bias<T>();
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
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__global__ void reduceGenericGlobal(
		int op,
		int n,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {

	__shared__ functions::reduce::ReduceFunction<T> *reduceFunctionToInvoke;
	__shared__ functions::reduce::ReduceOpFactory<T> *newOpFactory;

	if(threadIdx.x == 0)
		newOpFactory =  new functions::reduce::ReduceOpFactory<T>();
	__syncthreads();

	if(threadIdx.x == 0)
		reduceFunctionToInvoke = newOpFactory->create(op);
	__syncthreads();
	reduceFunctionToInvoke->transform(
			n,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,
			postProcessOrNot);
	if(threadIdx.x == 0) {
		delete  reduceFunctionToInvoke;
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
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__device__ void reduceGeneric(
		int op,
		int n,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {
	__shared__ functions::reduce::ReduceFunction<T> *reduceFunctionToInvoke;
	__shared__ functions::reduce::ReduceOpFactory<T> *newOpFactory;

	if(threadIdx.x == 0)
		newOpFactory =  new functions::reduce::ReduceOpFactory<T>();
	__syncthreads();

	if(threadIdx.x == 0)
		reduceFunctionToInvoke = newOpFactory->create(op);
	__syncthreads();
	reduceFunctionToInvoke->transform(
			n,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,
			postProcessOrNot);
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
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void reduceDouble(
		int op,
		int n,
		double *dx,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {
	reduceGeneric<double>(
			op,
			n,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,
			postProcessOrNot);

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
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void reduceFloat(
		int op,
		int n,
		float *dx,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {
	reduceGeneric<float>(
			op,
			n,
			dx,
			xShapeInfo
			,extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,
			postProcessOrNot);
}



#endif

