/*
 * summarystatsreduce.h
 *
 *  Created on: Jan 19, 2016
 *      Author: agibsonccc
 */

#ifndef SUMMARYSTATSREDUCE_H_
#define SUMMARYSTATSREDUCE_H_
#include <templatemath.h>

#include <shape.h>
#include <op.h>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif

namespace functions {
    namespace summarystats {

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other
// statistical properties encountered so far.
        template <typename T>
        class SummaryStatsData {

        public:
            T n;
            T min;
            T max;
            T mean;
            T M2;
            T M3;
            T M4;
            T bias;
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData() {
                initialize();
            }

            // initialize to the identity element

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void initialize() {
                n = mean = M2 = M3 = M4 = bias =  0;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void initWithValue(T val) {
                n = 1;
                min = val;
                max = val;
                mean = val;
                M2 = 0;
                M3 = 0;
                M4 = 0;
                bias = 0;
            }
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setValues(SummaryStatsData<T> *target) {
                n = target->n;
                min = target->min;
                max = target->max;
                mean = target->mean;
                M2 = target->M2;
                M3 = target->M3;
                M4 = target->M4;
                bias = target->bias;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T variance()   {
                if(n <= 1)
                    return 0.0;
                return M2 / (n - 1);
            }
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            T varianceBiasCorrected()   {
                if(n <= 1)
                    return 0.0;
                //  result = (accum - (FastMath.pow(bias, 2.0) / n())) / (n() - 1.0);

                return (M2 - nd4j_pow<T>(skewness(),2.0) / n) / n - 1.0;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T variance_n() {
                if(n <= 1)
                    return 0.0;
                return M2 / n;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T skewness()   { return nd4j::math::nd4j_sqrt<int>(n) * M3 / nd4j::math::nd4j_pow(M2, (T) 1.5); }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T kurtosis()   { return n * M4 / (M2 * M2); }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getM2() const {
                return M2;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setM2(T m2) {
                M2 = m2;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getM3() const {
                return M3;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setM3(T m3) {
                M3 = m3;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getM4() const {
                return M4;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setM4(T m4) {
                M4 = m4;
            }

#ifdef __CUDACC__
            __inline__ __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getMax() const {
                return max;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setMax(T max) {
                this->max = max;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getMean() const {
                return mean;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setMean(T mean) {
                this->mean = mean;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getMin() const {
                return min;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setMin(T min) {
                this->min = min;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getN() const {
                return n;
            }

#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            void setN(T n) {
                this->n = n;
            }
        };






#ifdef __CUDACC__
        // This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template<typename T>
struct SharedSummaryStatsData {
	// Ensure that we won't compile any un-specialized types
	__device__ T * getPointer() {
		extern __device__ void error(void);
		error();
		return 0;
	}
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedSummaryStatsData<float> {
	__device__ SummaryStatsData<float> * getPointer() {
		extern __shared__ SummaryStatsData<float> s_int2[];
		return s_int2;
	}
};
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedSummaryStatsData<double> {
	__device__ SummaryStatsData<double> * getPointer() {
		extern __shared__ SummaryStatsData<double> s_int6[];
		return s_int6;
	}
};
#endif

/**
 * Standard deviation or variance 1 pass
 */
        template<typename T>
        class SummaryStatsReduce: public  functions::ops::Op<T> {
        protected:
            bool biasCorrected = true;

        public:
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)


#endif
            SummaryStatsReduce(bool biasCorrected) {
                this->biasCorrected = biasCorrected;
            }
            /**
             *
             * @param val
             * @param extraParams
             * @return
             */
            //an op for the kernel
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData<T> op(SummaryStatsData<T> val, T *extraParams) {
                return val;
            }

            /**
             *
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
            //calculate an update of the reduce operation
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData<T> update(SummaryStatsData<T> x, SummaryStatsData<T> y,
                                       T *extraParams) {
                if(x.n == 0 && y.n > 0)
                    return y;
                else if(x.n > 0 && y.n == 0)
                    return x;
                SummaryStatsData<T> result;
                T n  = x.n + y.n;
                T n2 = n  * n;
                T n3 = n2 * n;


                T delta  = y.mean - x.mean;
                T delta2 = delta  * delta;
                T delta3 = delta2 * delta;
                T delta4 = delta3 * delta;

                //Basic number of samples (n), min, and max
                result.n   = n;
                result.min = nd4j::math::nd4j_min(x.min, y.min);
                result.max = nd4j::math::nd4j_max(x.max, y.max);

                result.mean = x.mean + delta * y.n / n;

                result.M2  = x.M2 + y.M2;
                result.M2 += delta2 * x.n * y.n / n;

                result.M3  = x.M3 + y.M3;
                result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
                result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

                result.M4  = x.M4 + y.M4;
                result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
                result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
                result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

                return result;
            }

            /**
             *
             * @param f1
             * @param f2
             * @param extraParams
             * @return
             */
            //invoked when combining two kernels
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData<T> merge(SummaryStatsData<T> f1, SummaryStatsData<T> f2, T *extraParams) = 0;

            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            T getValue(SummaryStatsData<T> val) = 0;

            /**
             *
             * @param reduction
             * @param n
             * @param xOffset
             * @param dx
             * @param incx
             * @param extraParams
             * @param result
             * @return
             */
            //post process result (for things like means etc)
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData<T> postProcess(SummaryStatsData<T> reduction, int n, int xOffset,
                                            T *dx, int incx, T *extraParams, T *result) = 0;

            /**
             *
             * @param d1
             * @param d2
             * @param extraParams
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __host__  __device__

#elif defined(__GNUC__)
            

#endif
            SummaryStatsData<T> op(SummaryStatsData<T> d1, SummaryStatsData<T> d2, T *extraParams) = 0;

#ifdef __CUDACC__
            /**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(SummaryStatsData<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		SummaryStatsData<T> *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
#pragma unroll
			while ( floorPow2 & (floorPow2 - 1) ) {
				floorPow2 &= floorPow2 - 1;
			}

			if (tid >= floorPow2) {
				SummaryStatsData<T> prev = sPartials[tid - floorPow2];
				SummaryStatsData<T> curr = sPartials[tid];
				sPartials[tid - floorPow2] = update(prev,curr,extraParams);
			}
			__syncthreads();
		}

#pragma unroll
		for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numElements) {
				SummaryStatsData<T> curr = sPartials[tid];
				SummaryStatsData<T> next = sPartials[tid + activeThreads];
				sPartials[tid] = update(curr,next,extraParams);
			}
			__syncthreads();
		}

	}

	/**
	 * @param n n is the number of
	 *        elements to loop through
	 * @param dx the data to operate on
	 * @param xVectorInfo the meta data for the vector:
	 *                              0 is the offset
	 *                              1 is the increment/stride
	 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
	 *                              3 is the element wise stride for the buffer
	 *                              4 is the number of elements it takes to get to the next row/column/tensor
	 * @param gpuInformation
	 *                              0 is the block size
	 *                              1 is the grid size
	 *                              2 is the shared memory size
	 * @param problemDefinition
	 *                          0 is the number of elements per vector
	 *                          1 is the number of vectors
	 */
	__inline__ __device__ void transform(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot) {


		/**
		 * Gpu information for the problem
		 */
		int tid = threadIdx.x;
		__shared__ volatile int resultScalar;

		__shared__ int xElementWiseStride;
		__shared__ int reductionIndexesPerBlock;

		int numElements = gridDim.x;
		//shared memory space for storing intermediate results
		SummaryStatsData<T> *sPartials;
		functions::summarystats::SharedSummaryStatsData<T> holder;

		sPartials = holder.getPointer();
		T startingVal = this->startingValue(dx);
#pragma unroll
		for (int i = tid; i < numElements; i += blockDim.x) {
			SummaryStatsData<T> val;
			val.initWithValue(startingVal);
			val.n = 0;
			sPartials[i] = val;
		}
		__syncthreads();

		//length for the tad
		__shared__ volatile int xLength;

		__shared__ volatile int resultLength;

		__shared__ int tensorsForDimension;

		__shared__ int elementsPerTad;

		//only compute the tad indexes once
		__shared__
		shape::TADPermuteInfo xTadInfo;

		SummaryStatsData <T> reduction;
		reduction.initWithValue(0.0);
		reduction.n = 0;
		if (tid == 0) {
			tensorsForDimension = shape::tensorsAlongDimension(xShapeInfo, dimension, dimensionLength);
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
			xElementWiseStride = shape::elementWiseStride(xShapeInfo);
			xLength = shape::length(xShapeInfo);
			elementsPerTad = xLength / resultLength;

			if (gridDim.x >= resultLength) {
				reductionIndexesPerBlock = 1;
			}
			else {
				reductionIndexesPerBlock = resultLength / gridDim.x;
			}


		}
		__syncthreads();
		if (!resultScalar && dimensionLength > 1) {
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
			int tadElementWiseStride = shape::stride(xShapeInfo)[dimensionLength - 1];
			int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
			int xLength = shape::length(xShapeInfo);
			int i = 0,j = 0;
#pragma unroll
			for(i = tid; i < resultLength; i+= gridDim.x * blockDim.x) {
				SummaryStatsData <T> indexVal;
				indexVal.initWithValue(dx[i]);
				sPartials[tid] = op(indexVal, extraParams);
				__syncthreads();
				for(j = 1; j < elementsPerReductionIndex; j++) {
					SummaryStatsData <T> indexVal2;
					indexVal2.initWithValue(dx[i + tadElementWiseStride * j]);
					sPartials[tid] =  update(sPartials[tid],op(indexVal2, extraParams), extraParams);
					__syncthreads();
				}

				result[i] = getValue(sPartials[tid]);
			}


			if(tid == 0) {
				shape::freePermuteInfo(xTadInfo);
			}

		}
		else if (resultScalar) {
			if(blockIdx.x >= resultLength && tid < numElements)
				return;

			unsigned int i = blockIdx.x * xElementWiseStride + tid;
			unsigned int gridSize = blockDim.x * gridDim.x * xElementWiseStride;
			int n = shape::length(xShapeInfo);
			// we reduce multiple elements per thread.  The number is determined by the
			// number of active thread blocks (via gridDim).  More blocks will result
			// in a larger gridSize and therefore fewer elements per thread
#pragma unroll
			while (i < n) {
				SummaryStatsData <T> indexVal;
				indexVal.initWithValue(dx[i]);
				reduction = update(reduction, indexVal, extraParams);
				i += gridSize;
			}

			// each thread puts its local sum into shared memory
			if(tid < numElements && reduction.n > 0)
				sPartials[tid] = reduction;
			__syncthreads();
			if(tid < numElements && reduction.n > 0)
				aggregatePartials(&sPartials, tid,numElements ,extraParams);

			// write result for this block to global mem
			if (tid == 0) {
				reduction = sPartials[0];
				result[blockIdx.x] = getValue(reduction);
			}
		}

		else if (!resultScalar) {
			__shared__ int *tadShapeBuffer;
			if(tid == 0) {
				xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
			}
			__syncthreads();

			if(tid == 0) {
				tadShapeBuffer = shape::shapeBuffer(xTadInfo.tensorShapeLength,xTadInfo.tensorShape);
			}

			__syncthreads();




			if (reductionIndexesPerBlock * blockIdx.x >= resultLength)
				return;

			int tadsPerReductionIndex = tensorsForDimension / resultLength;
			//minimum number of threads needed for each reduction index
			int tadsNeeded = reductionIndexesPerBlock * tadsPerReductionIndex;

			if(tid >= tadsNeeded)
				return;
			else {
				//process each tad
				//tad wrt the thread
				int currTad = tid + (blockIdx.x * reductionIndexesPerBlock);
				int offsetForTad = shape::offset(currTad, xShapeInfo,dimension, dimensionLength, xTadInfo);

				//update the reduction for the thread for the current tad
				//note here that we compute the offset and then accumulate in shared memory
				if(xElementWiseStride > 1)
#pragma unroll
					for (int element = 0; element < elementsPerTad; element++, offsetForTad += xElementWiseStride) {
						SummaryStatsData <T> indexVal;
						indexVal.initWithValue(dx[offsetForTad]);
						SummaryStatsData<T> opOutput = op(indexVal,extraParams);
						sPartials[tid] = update(sPartials[tid], opOutput, extraParams);
						__syncthreads();
					}
				else {
#pragma unroll
					for (int element = 0; element < elementsPerTad; element++, offsetForTad++) {
						SummaryStatsData <T> indexVal;
						indexVal.initWithValue(dx[offsetForTad]);
						SummaryStatsData<T> opOutput = op(indexVal,extraParams);
						sPartials[tid] = update(sPartials[tid], opOutput, extraParams);
						__syncthreads();
					}
				}




			}

			//first thread for a reduction index
			if (tid % tadsPerReductionIndex == 0 && tadsPerReductionIndex > 1) {
				/**
				 * Each reduction index is handled by k tads
				 * which need to be combined in each thread.
				 *
				 * Since the TADS to be combined
				 * are to be next to each other
				 * we can assume that
				 * the items in shared memory
				 * can be combined and collapsed
				 * in to the first thread's
				 * entry.
				 *
				 * This follows a similar pattern
				 * for global block wise reduction
				 * and computing parallel sums
				 * in other reduction implementations.
				 *
				 */
#pragma unroll
				for (int i = 1; i < tadsPerReductionIndex; i++) {
					sPartials[tid] = update(sPartials[tid], sPartials[tid + i], extraParams);
					__syncthreads();
				}
			}

			__syncthreads();

			//after all the threads are done processing each first item in shared memory
			//should correspond to the final value for the particular reduction index
			//that was set for this block.
			if (tid == 0) {
				for (int i = 0; i < reductionIndexesPerBlock; i++) {
					int reductionIndexToProcess = i + blockIdx.x * reductionIndexesPerBlock;
					result[reductionIndexToProcess] = getValue(sPartials[i]);
				}


				free(tadShapeBuffer);
				shape::freePermuteInfo(xTadInfo);

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
	 * 2,3 doesn't have an element wise stride
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
	 * in the result as if we had specified the reduction dimensions
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
	__device__ void collapseTad(
			T *data,
			T *result,
			T *extraParams,
			int elementsPerTad, int numTads, int n, int elementWiseStride,
			int numOriginalTads, int sharedMemorySize,
			int *xShapeInfo, int *dimension, int dimensionLength) {
		//shared memory space for storing intermediate results
		SummaryStatsData <T> *sPartials;
		SharedSummaryStatsData <T> holder;

		sPartials = holder.getPointer();

		int tid = threadIdx.x;
		//intialize te values
		int numItems = sharedMemorySize / sizeof(T);
#pragma unroll
		for (int i = tid; i < numItems; i += blockDim.x) {
			SummaryStatsData <T> valInit;
			valInit.initWithValue(0.0);
			sPartials[i] = valInit;
		}
		__syncthreads();

		//each block processes a reduction index
		if (blockIdx.x >= numTads)
			return;

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
		int tadsPerReduceIndex2 = shape::tadsPerReduceIndex(numTads, numOriginalTads);
		//each thread does a tad
		if (tid >= tadsPerReduceIndex2)
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
		for (int i = 0; i < elementsPerTad; offsetForBlock += elementWiseStride, i++) {
			SummaryStatsData <T> opApply;
			opApply.initWithValue(data[offsetForBlock]);
			sPartials[tid] = update(sPartials[tid], op(opApply, extraParams), extraParams);
			__syncthreads();
		}

		if (tid == 0 && blockIdx.x < numTads) {
			//start at 1 so we don't count the first entry twice
#pragma unroll
			for (int i = 1; i < numTads; i++) {
				sPartials[0] = update(sPartials[0], sPartials[i], extraParams);
				__syncthreads();
			}

			result[blockIdx.x] = getValue(sPartials[0]);
			shape::freePermuteInfo(xTadInfo);
		}
	}

#endif

            /**
             * CPU interface
             * @param x the input
             * @param xShapeInfo the shape information for input
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfo the shape information
             * for result
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)
            
#endif
            void exec(T *x,
                      int *xShapeInfo,
                      T *extraParams,
                      T *result,
                      int *resultShapeInfo) {
                result[0] = this->execScalar(x,xShapeInfo,extraParams);
            }


            /**
             * CPU interface
             * @param x the input
             * @param xShapeInfo the shape information for input
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfo the shape information
             * for result
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)
            
#endif
            T execScalar(T *x,
                         int *xShapeInfo,
                         T *extraParams) {
                SummaryStatsData<T> startingIndex;
                startingIndex.initialize();
                int length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                if (xElementWiseStride == 1) {
#pragma omp parallel for shared(startingIndex)
                    for (int i = 0; i < length; i++) {
                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[i]);
#pragma omp critical
                        {
                            startingIndex = update(startingIndex, curr,
                                                   extraParams);
                        }

                    }

                    T finalVal = this->getValue(startingIndex);
                    return finalVal;
                } else {

#pragma omp parallel for shared(startingIndex)
                    for (int i = 0; i < length; i++) {
                        SummaryStatsData<T> curr;
                        curr.initWithValue(x[i]);
#pragma omp critical
                        {
                            startingIndex = update(startingIndex, curr,
                                                   extraParams);
                        }

                    }

                    T finalVal = this->getValue(startingIndex);
                    return finalVal;
                }


            }


            /**
             * Dimension wise execution for CPU
             * @param x the input
             * @param xShapeInfo the shape information
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to execute along
             * @param dimensionLength the length of the dimension
             */
            virtual
#ifdef __CUDACC__
            inline __host__

#elif defined(__GNUC__)
            
#endif
            void exec(T *x,
                      int *xShapeInfo,
                      T *extraParams,
                      T *result,
                      int *resultShapeInfoBuffer,
                      int *dimension, int dimensionLength) {
                if (shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar(x, xShapeInfo, extraParams);
                    return;
                }

                shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo,
                                                                      dimension, dimensionLength);
                int resultLength = shape::length(resultShapeInfoBuffer);
                /**
                 * The element wise stride belongs to a reduction index.
                 * When used out of order, we can get rid of the data
                 * dependencies and rely on using the max dimension
                 * specified for stride instead.
                 * Say we take the sum(0,1) along arr
                 * we can use arr.stride(1) as a representation
                 * along which to iterate.
                 */
                int tadElementWiseStride = dimensionLength > 1 ? shape::stride(xShapeInfo)[dimensionLength - 1]
                                                               : shape::computeElementWiseStride(
                                shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), shape::stride(xShapeInfo),
                                shape::order(xShapeInfo) == 'f', dimension, dimensionLength);
                int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
                SummaryStatsData<T> *currStartingValue = (SummaryStatsData<T> *) malloc(
                        sizeof(SummaryStatsData<T>) * resultLength);
#pragma omp parallel for
                for (int i = 0; i < resultLength; i++) {
                    currStartingValue[i].initialize();
                }


                if (dimensionLength > 1) {
                    shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
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
                    int tadElementWiseStride = shape::reductionIndexElementWiseStride(xShapeInfo, dimension,dimensionLength);
                    const int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
                    int tadLength = tadPermuteInfo.tensorShapeProd;

#pragma omp  parallel  for
                    for (int i = 0; i < resultLength; i++) {
                        int offset = i + tadElementWiseStride  * tadLength;
                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[offset]);
                        currStartingValue[i] = op(comp, extraParams);
#pragma omp simd
                        for (int j = 1; j < elementsPerReductionIndex; j++) {
                            SummaryStatsData<T> comp2;
                            comp2.initWithValue(x[offset + tadElementWiseStride * j]);
                            currStartingValue[i] = update(currStartingValue[i], comp2, extraParams);
                        }

                        result[i] = getValue(currStartingValue[i]);


                    }


                    shape::freePermuteInfo(tadPermuteInfo);
                }

                else {
                    int tadElementWiseStride = shape::tadElementWiseStride(xShapeInfo, dimension, dimensionLength);
                    int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                    const int resultLength = shape::length(resultShapeInfoBuffer);
#pragma omp parallel for
                    for (int i = 0; i < resultLength; i++) {
                        int offset = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength);
                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[offset]);
                        currStartingValue[i] = op(comp, extraParams);
                        for (int j = 1; j < tadLength; j++) {
                            SummaryStatsData<T> comp2;
                            comp2.initWithValue(x[offset + tadElementWiseStride * j]);
                            currStartingValue[i] = update(currStartingValue[i], comp2, extraParams);
                        }

                        result[i] = getValue(currStartingValue[i]);
                    }


                }

                free(currStartingValue);
            }

            virtual
#ifdef __CUDACC__
            __host__ __device__
#endif
            T startingValue(T *input) {
                return 0;
            }

            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                //no extra params aggregation needs to happen
            }

#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)
            
#endif
            virtual ~SummaryStatsReduce() {
            }
#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)
            
#endif
            SummaryStatsReduce() {
            }

        };

        namespace ops {
/**
 * var(x)
 */
            template<typename T>
            class Variance: public  functions::summarystats::SummaryStatsReduce<T> {
            public:
                /**
                 * Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("variance");
                }
                /**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> op(
                        functions::summarystats::SummaryStatsData<T> val, T *extraParams) override {
                    return val;
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                T getValue(SummaryStatsData<T> val) {
                    return val.variance();
                }


                /**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
                //invoked when combining two kernels
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> merge(
                        functions::summarystats::SummaryStatsData<T> f1,
                        functions::summarystats::SummaryStatsData<T> f2, T *extraParams) override {
                    return this->update(f1,f2,extraParams);
                }

                /**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
                //post process result (for things like means etc)
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> postProcess(
                        functions::summarystats::SummaryStatsData<T> reduction, int n, int xOffset,
                        T *dx, int incx, T *extraParams, T *result) override {
                    return reduction;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,
                                       functions::summarystats::SummaryStatsData<T> d2, T *extraParams) override {
                    return d1;
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)
                
#endif
                virtual ~Variance() {
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)
                
#endif
                Variance() {
                }

            };
/**
 * std(x)
 */
            template<typename T>
            class StandardDeviation: public  functions::summarystats::SummaryStatsReduce<T> {
            public:
                /**
                 * Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("std");
                }

                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                T getValue(SummaryStatsData<T> val) {
                    return nd4j::math::nd4j_sqrt(val.variance());
                }

                /**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
                //an op for the kernel
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> op(
                        functions::summarystats::SummaryStatsData<T> val, T *extraParams) override {
                    return val;
                }


                /**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
                //invoked when combining two kernels
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> merge(
                        functions::summarystats::SummaryStatsData<T> f1,
                        functions::summarystats::SummaryStatsData<T> f2, T *extraParams) override {
                    return this->update(f1,f2,extraParams);
                }

                /**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
                //post process result (for things like means etc)
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                functions::summarystats::SummaryStatsData<T> postProcess(
                        functions::summarystats::SummaryStatsData<T> reduction, int n, int xOffset,
                        T *dx, int incx, T *extraParams, T *result) override {
                    return reduction;
                }

                /**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                

#endif
                SummaryStatsData<T> op(functions::summarystats::SummaryStatsData<T> d1,
                                       functions::summarystats::SummaryStatsData<T> d2, T *extraParams) override {
                    return d1;
                }

#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)
                
#endif
                virtual ~StandardDeviation() {
                }
#ifdef __CUDACC__
                __host__ __device__
#elif defined(__GNUC__)
                
#endif
                StandardDeviation() {
                }
            };
        }

        template<typename T>
        class SummaryStatsReduceOpFactory {
        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            SummaryStatsReduceOpFactory() {
            }


#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            functions::summarystats::SummaryStatsReduce<T> * getOp(int op) {
                if (op == 0) {
                    return new functions::summarystats::ops::Variance<T>();
                } else if (op == 1) {
                    return new functions::summarystats::ops::StandardDeviation<T>();

                }
                return NULL;
            }
        };
    }


}


#ifdef __CUDACC__
/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
template <typename T>
__device__ void summaryStatsReduceGeneric(
		int op,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	__shared__ functions::summarystats::SummaryStatsReduce<T> *indexReduce;
	__shared__ functions::summarystats::SummaryStatsReduceOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory = new functions::summarystats::SummaryStatsReduceOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0)
		indexReduce = newOpFactory->getOp(op);
	__syncthreads();
	indexReduce->transform(dx,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,postProcessOrNot);
	if(threadIdx.x == 0) {
		free(indexReduce);
		free(newOpFactory);
	}
}

/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
extern "C" __global__ void summaryStatsReduceDouble(
		int op,
		double *dx,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {
	summaryStatsReduceGeneric<double>(
			op,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot);

}

/**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
extern "C" __global__ void summaryStatsReduceFloat(
		int op,
		float *dx,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot) {
	summaryStatsReduceGeneric<float>(
			op,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot);

}



#endif






#endif /* SUMMARYSTATSREDUCE_H_ */
