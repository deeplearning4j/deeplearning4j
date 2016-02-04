/*
 * indexreduce.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include <shape.h>
#include <op.h>
#include <omp.h>
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif

#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#ifdef __JNI__
#include <jni.h>
#endif


namespace functions {
namespace indexreduce {
template<typename T>
struct IndexValue {
	T value;
	int index;
};

#ifdef __CUDACC__
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
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<float> {
	__device__ IndexValue<float> * getPointer() {
		extern __shared__ IndexValue<float> s_int2[];
		return s_int2;
	}
};
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<double> {
	__device__ IndexValue<double> * getPointer() {
		extern __shared__ IndexValue<double> s_int6[];
		return s_int6;
	}
};
#endif

template<typename T>
class IndexReduce: public  functions::ops::Op<T> {

public:
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
	__always_inline

#endif
	IndexValue<T> op(IndexValue<T> val, T *extraParams) = 0;

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
	__always_inline

#endif
	IndexValue<T> update(IndexValue<T> old, IndexValue<T> opOutput,
			T *extraParams) = 0;

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
	__always_inline

#endif
	IndexValue<T> merge(IndexValue<T> f1, IndexValue<T> f2, T *extraParams) = 0;

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
	__always_inline

#endif
	IndexValue<T> postProcess(IndexValue<T> reduction, int n, int xOffset,
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
	__always_inline

#endif
	IndexValue<T> op(IndexValue<T> d1, IndexValue<T> d2, T *extraParams) = 0;

#ifdef __CUDACC__
	/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(IndexValue<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
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
				sPartials[tid - floorPow2] = update(prev,curr,extraParams);
			}
			__syncthreads();
		}

#pragma unroll
		for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numElements) {
				IndexValue<T> curr = sPartials[tid];
				IndexValue<T> next = sPartials[tid + activeThreads];
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
			int n, T *dx,
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
		__shared__ int reductionIndexesPerBlock;

		int numElements = gpuInformation[2] / sizeof(IndexValue <T>);
		//shared memory space for storing intermediate results
		IndexValue<T> *sPartials;
		functions::indexreduce::SharedIndexValue<T> holder;

		sPartials = holder.getPointer();
		T startingVal = this->startingValue(dx);

#pragma unroll
		for (int i = tid; i < numElements; i += blockDim.x) {
			IndexValue <T> val = {startingVal, i};
			sPartials[i] = val;
		}
		__syncthreads();

		//length for the tad
		__shared__ volatile int xLength;

		__shared__ volatile int resultLength;

		__shared__ int tensorsForDimension;

		__shared__ int elementsPerTad;

		//only compute the tad indexes once
		__shared__ shape::TADPermuteInfo xTadInfo;

		IndexValue <T> reduction = {startingVal, 0};
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
			xOffset = shape::offset(xShapeInfo);
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
			int tadLength = xTadInfo.tensorShapeProd;
			int xLength = shape::length(xShapeInfo);
			int i = 0,j = 0;
#pragma unroll
			for(i = tid; i < resultLength; i++) {
				IndexValue<T> comp2;
				comp2.value = dx[i];
				comp2.index = i % tadLength;
				sPartials[tid] = op(comp2, extraParams);
				__syncthreads();
				for(j = 1; j < elementsPerReductionIndex; j++) {
					IndexValue<T> comp;
					comp.value = dx[i + tadElementWiseStride * j];
					comp.index =  i + tadElementWiseStride * j % tadLength;
					sPartials[tid] =  update(sPartials[tid],op(comp, extraParams), extraParams);
					__syncthreads();
				}

				result[i] = sPartials[tid].index;
			}


			if(tid == 0) {
				shape::freePermuteInfo(xTadInfo);
			}

		}


		//reduce to 1 result
		else if (resultScalar) {
			//don't need any more blocks than the result length
			if(blockIdx.x >= resultLength)
				return;

			unsigned int i = blockIdx.x * xElementWiseStride + tid;
			unsigned int gridSize = blockDim.x * gridDim.x * xElementWiseStride;
			if(xOffset == 0) {
				// we reduce multiple elements per thread.  The number is determined by the
				// number of active thread blocks (via gridDim).  More blocks will result
				// in a larger gridSize and therefore fewer elements per thread
#pragma unroll
				while (i < n) {
					int currIdx = i;
					IndexValue <T> indexVal = {dx[i], currIdx};
					reduction = update(reduction, indexVal, extraParams);
					i += gridSize;
				}
			}
			else {
				// we reduce multiple elements per thread.  The number is determined by the
				// number of active thread blocks (via gridDim).  More blocks will result
				// in a larger gridSize and therefore fewer elements per thread
				while (xOffset + i < n) {
					int currIdx = xOffset + i;
					IndexValue <T> indexVal = {dx[xOffset + i], currIdx};
					reduction = update(reduction, indexVal, extraParams);
					i += gridSize;
				}
			}

			// each thread puts its local sum into shared memory
			sPartials[tid] = reduction;
			__syncthreads();

			aggregatePartials(&sPartials, tid,numElements ,extraParams);

			// write result for this block to global mem
			if (tid == 0) {
				result[blockIdx.x] = sPartials[0].index;
			}
		}

		//multi dimensional
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
				int offsetForTad = shape::offset(currTad, xShapeInfo, dimensionLength, xTadInfo);

				//update the reduction for the thread for the current tad
				//note here that we compute the offset and then accumulate in shared memory
				if(xElementWiseStride > 1)
#pragma unroll
					for (int element = 0; element < elementsPerTad; element++, offsetForTad += xElementWiseStride) {
						IndexValue <T> indexVal = {dx[offsetForTad], element};
						sPartials[tid] = update(sPartials[tid], indexVal, extraParams);
						__syncthreads();
					}
				else {
#pragma unroll
					for (int element = 0; element < elementsPerTad; element++, offsetForTad++) {
						IndexValue <T> indexVal = {dx[offsetForTad], element};
						sPartials[tid] = update(sPartials[tid], indexVal, extraParams);
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
					result[reductionIndexToProcess] = sPartials[i].index;
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
		IndexValue <T> *sPartials;
		SharedIndexValue <T> holder;

		sPartials = holder.getPointer();

		int tid = threadIdx.x;
		//intialize te values
		int numItems = sharedMemorySize / sizeof(T);
#pragma unroll
		for (int i = tid; i < numItems; i += blockDim.x) {
			IndexValue <T> valInit = {extraParams[0], 0};
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
		int offsetForBlock = shape::offset(tadForThread, xShapeInfo, dimensionLength, xTadInfo);
#pragma unroll
		for (int i = 0; i < elementsPerTad; offsetForBlock += elementWiseStride, i++) {
			IndexValue <T> opApply = {data[offsetForBlock], offsetForBlock};
			sPartials[tid] = update(sPartials[tid],opApply, extraParams);
			__syncthreads();
		}

		if (tid == 0 && blockIdx.x < numTads) {
			//start at 1 so we don't count the first entry twice
#pragma unroll
			for (int i = 1; i < numTads; i++) {
				sPartials[0] = update(sPartials[0], sPartials[i], extraParams);
				__syncthreads();
			}

			result[blockIdx.x] = sPartials[0].index;
			shape::freePermuteInfo(xTadInfo);
		}
	}

#endif
	/**
	 * CPU operations
	 * @param x the input data
	 * @param xShapeInfo the shape information for the input data
	 * @param extraParams the extra parameters
	 * @param result the result data
	 * @param resultShapeInfo the shpae information
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline
#endif
	T execScalar(T *x,
			int *xShapeInfo,
			T *extraParams) {
		T startingVal = this->startingValue(x);
		IndexValue<T> startingIndex;
		startingIndex.value = startingVal;
		startingIndex.index = 0;
		int length = shape::length(xShapeInfo);
		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		if (xElementWiseStride == 1) {
#pragma omp simd
			for (int i = 0; i < length; i++) {
				IndexValue<T> curr;
				curr.value = x[i];
				curr.index = i;
				startingIndex = update(startingIndex, curr,
						extraParams);

			}

			return startingIndex.index;
		} else {

#pragma omp simd
			for (int i = 0; i < length; i++) {
				IndexValue<T> curr;
				curr.value = x[i * xElementWiseStride];
				curr.index = i;
				startingIndex = update(startingIndex, curr,
						extraParams);
			}
			return  startingIndex.index;

		}


	}


	/**
	 * CPU operations
	 * @param x the input data
	 * @param xShapeInfo the shape information for the input data
	 * @param extraParams the extra parameters
	 * @param result the result data
	 * @param resultShapeInfo the shpae information
	 */
	virtual
#ifdef __CUDACC__
	inline __host__  __device__

#elif defined(__GNUC__)
	__always_inline
#endif
	void exec(T *x,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo) {
		T startingVal = this->startingValue(x);
		IndexValue<T> startingIndex;
		startingIndex.value = startingVal;
		startingIndex.index = 0;
		int length = shape::length(xShapeInfo);
		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
		if (xElementWiseStride == 1 && resultElementWiseStride == 1) {
#pragma omp simd
			for (int i = 0; i < length; i++) {
				IndexValue<T> curr;
				curr.value = x[i];
				curr.index = i;
				startingIndex = update(startingIndex, curr,
						extraParams);

			}

			result[0] = startingIndex.index;
		} else {

#pragma omp simd
			for (int i = 0; i < length; i++) {
				IndexValue<T> curr;
				curr.value = x[i * xElementWiseStride];
				curr.index = i;
				startingIndex = update(startingIndex, curr,
						extraParams);
			}
			result[0] = startingIndex.index;

		}


	}

	/**
	 * The dimension wise
	 * CPU implementation
	 * @param x the input data
	 * @param xShapeInfo the x shape information
	 * @param extraParams the extra parameters for the reduce
	 * @param result the result buffer
	 * @param resultShapeInfoBuffer the shape information
	 * @param dimension the dimension to do reduce along
	 * @param dimensionLength the length of the dimension
	 * buffer
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#elif defined(__GNUC__)
	__always_inline
#endif
	void exec(T *x,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfoBuffer,
			int *dimension, int dimensionLength) {
		if(shape::isScalar(resultShapeInfoBuffer)) {
			exec(x,xShapeInfo,extraParams,result,resultShapeInfoBuffer);
			return;
		}

		shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo,dimension, dimensionLength);
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
		IndexValue<T> *startingIndex = new IndexValue<T>[resultLength];

		int tadElementWiseStride = dimensionLength > 1 ? shape::stride(xShapeInfo)[dimensionLength - 1] : shape::computeElementWiseStride(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),shape::order(xShapeInfo) == 'f',dimension,dimensionLength);
		int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
		int tadLength = tadPermuteInfo.tensorShapeProd;

#pragma omp simd
		for (int i = 0; i < resultLength; i++) {
			IndexValue<T> val;
			val.value = this->startingValue(x);
			val.index = 0;
			startingIndex[i] = val;
		}

		int i = 0,j = 0;
#pragma omp parallel private(j,i)
		for(i = omp_get_thread_num(); i < resultLength; i++) {
			int offset = dimensionLength > 1 ? i : tadLength * i;
			startingIndex[i].value = x[offset];
			startingIndex[i].index = offset % tadLength;
			for(j = 1; j < elementsPerReductionIndex; j++) {
				IndexValue<T> comp2;
				comp2.value = x[offset + tadElementWiseStride * j];
				comp2.index = (offset + tadElementWiseStride * j) % tadLength;
				startingIndex[i] = update(startingIndex[i],comp2, extraParams);
				result[i] = startingIndex[i].index;
			}


		}

		shape::freePermuteInfo(tadPermuteInfo);
        delete[] startingIndex;
	}

	virtual
#ifdef __CUDACC__
	__host__ __device__
#endif
	T startingValue(T *input) = 0;


#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~IndexReduce() {
	}
#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	IndexReduce() {
	}

};

namespace ops {

/**
 * Find the max index
 */
template<typename T>
class IMax: public  functions::indexreduce::IndexReduce<T> {
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
		return std::string("imax");
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> op(
			functions::indexreduce::IndexValue<T> val, T *extraParams) override {
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> update(
			functions::indexreduce::IndexValue<T> old,
			functions::indexreduce::IndexValue<T> opOutput, T *extraParams) override {
		if (opOutput.value > old.value)
			return opOutput;
		return old;
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> merge(
			functions::indexreduce::IndexValue<T> f1,
			functions::indexreduce::IndexValue<T> f2, T *extraParams) override {
		if (f1.value > f2.value)
			return f2;
		return f1;
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> postProcess(
			functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
			T *dx, int incx, T *extraParams, T *result) override {
		return reduction;
	}
	virtual
#ifdef __CUDACC__
	__host__ __device__
#endif
	T startingValue(T *input) {
		return MIN_FLOAT;
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
	__always_inline

#endif
	IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
			functions::indexreduce::IndexValue<T> d2, T *extraParams) override {
		return d1;
	}
#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~IMax() {
	}
#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	IMax() {
	}

};

/**
 * Find the min index
 */
template<typename T>
class IMin: public  functions::indexreduce::IndexReduce<T> {
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
		return std::string("imin");
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> op(
			functions::indexreduce::IndexValue<T> val, T *extraParams) override {
		return val;
	}
	virtual
#ifdef __CUDACC__
	__host__ __device__
#endif
	T startingValue(T *input) {
		return MAX_FLOAT;
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> update(
			functions::indexreduce::IndexValue<T> old,
			functions::indexreduce::IndexValue<T> opOutput, T *extraParams) override {
		if (opOutput.value < old.value) {
			return opOutput;
		}


		return old;
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> merge(
			functions::indexreduce::IndexValue<T> f1,
			functions::indexreduce::IndexValue<T> f2, T *extraParams) override {
		if (f1.value < f2.value)
			return f2;
		return f1;
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
	__always_inline

#endif
	functions::indexreduce::IndexValue<T> postProcess(
			functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
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
	__always_inline

#endif
	IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
			functions::indexreduce::IndexValue<T> d2, T *extraParams) override {
		return d1;
	}

#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	virtual ~IMin() {
	}
#ifdef __CUDACC__
	__host__ __device__
#elif defined(__GNUC__)
	__always_inline
#endif
	IMin() {
	}
};
}

template<typename T>
class IndexReduceOpFactory {
public:

#ifdef __CUDACC__
	__host__ __device__
#endif
	IndexReduceOpFactory() {
	}


#ifdef __CUDACC__
	__inline__ __host__ __device__
#endif
	functions::indexreduce::IndexReduce<T> * getOp(int op) {
		if (op == 0) {
			return new functions::indexreduce::ops::IMax<T>();
		} else if (op == 1) {
			return new functions::indexreduce::ops::IMin<T>();

		}
		return NULL;
	}
};
}


}


#ifdef __CUDACC__

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__device__ void indexReduceGeneric(
		int op,
		int n,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	__shared__ functions::indexreduce::IndexReduce<T> *indexReduce;
	__shared__ functions::indexreduce::IndexReduceOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory = new functions::indexreduce::IndexReduceOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0)
		indexReduce = newOpFactory->getOp(op);
	__syncthreads();
	indexReduce->transform(n,dx,xShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);
	if(threadIdx.x == 0) {
		free(indexReduce);
		free(newOpFactory);
	}
}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void indexReduceDouble(int op,int n, double *dx, int *xShapeInfo, double *extraParams, double *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	indexReduceGeneric<double>(op,n,dx,xShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);

}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
extern "C" __global__ void indexReduceFloat(int op,int n, float *dx, int *xShapeInfo, float *extraParams, float *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	indexReduceGeneric<float>(op,n,dx,xShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);

}



#endif

#endif /* INDEXREDUCE_H_ */

