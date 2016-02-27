#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "reduce_common.h"
#include "sharedmem.h"
#include "postprocess.h"
#include "semaphore.h"

//an op for the kernel
template<typename T>
__device__ T op(T d1,T *extraParams);

//calculate an update of the reduce operation
template<typename T>
__device__ T update(T old,T opOutput,T *extraParams);
//invoked when combining two kernels
template<typename T>
__device__ T merge(T f1, T f2,T *extraParams);

//post process result (for things like means etc)



template<typename T>
__device__ T op(T d1,T d2,T *extraParams);


template <>  double merge<double>(double  opOutput,double other,double *extraParams);
template <> double update<double>(double old,double opOutput,double *extraParams);
template <> double op<double>(double d1,double *extraParams);


template <> float merge<float>(float old,float opOutput,float *extraParams);
template <>float update<float>(float old,float opOutput,float *extraParams);
template <> float op<float>(float d1,float *extraParams);





template<typename T>
__device__ void aggregatePartials(T **sPartialsRef,int tid,T *extraParams) {
	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	T *sPartials = *sPartialsRef;
	int floorPow2 = blockDim.x;

	if (floorPow2 & (floorPow2 - 1)) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if (tid >= floorPow2) {
			sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],sPartials[tid],extraParams);
		}
		__syncthreads();
	}

	for (int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1) {
		if (tid < activeThreads) {
			sPartials[tid] = update(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		__syncthreads();
	}

}




template<typename T>
__device__ T doBlock(int n,T *sPartials,T *dx,int xOffset,int incx,T *extraParams) {
	T sum = extraParams[0];
	for (int i = 0; i < n; i++) {
		int currIdx = xOffset + i * incx;
		T curr = dx[currIdx];
		sum = update(sum,op(curr,extraParams),extraParams);
	}

	return sum;
}





template <typename T>
__device__ void initializeShared(T *extraParams,T** sPartials,int sMemSize) {
	int sPartialsLength = sMemSize / sizeof(T);
	T *sPartialsDeref = (T *) *sPartials;
	for(int i = 0; i < sPartialsLength; i++) {
		sPartialsDeref[i] = extraParams[0];
	}
}



extern "C"
__global__ void printShapeBuffer(int n,int *buff) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	if(i < n) {
		printf("Buff item %d is %d\n",i,buff[i]);
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
template<typename T>
__device__ void transform(
		int n
		,T *dx
		,int *xShapeInfo
		,T *extraParams
		,T *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {

	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;


	__shared__ volatile int resultScalar;


	__shared__ int xElementWiseStride;
	__shared__ int xOffset;


	//shared memory space for storing intermediate results
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	int numElements = gpuInformation[2] / sizeof(T);
	for (int i = tid; i < numElements; i += blockDim.x)
		sPartials[i] = extraParams[0];
	__syncthreads();



	//starting index for tad
	__shared__ volatile int currentBlockOffset;
	//ending index for tad
	__shared__ volatile int endingOffset;
	//length for the tad
	__shared__ volatile int xLength;

	__shared__ volatile int resultLength;

	__shared__ volatile int tadsForBlock;

	__shared__ volatile int elementsPerThread;

	__shared__ volatile int elementsPerTad;

	__shared__ volatile int tensorsForDimension;

	//only compute the tad indexes once
	__shared__ TADPermuteInfo xTadInfo;

	__shared__ volatile int tadsPerThread;

	__shared__ volatile int reductionIndexesPerBlock;

	T reduction = extraParams[0];
	if(tid == 0) {
		tensorsForDimension = tensorsAlongDimension(xShapeInfo,dimension,dimensionLength);
		resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));
		if(dimensionLength == 1) {
			if(dimension[0] == MAX_DIMENSION)
				resultScalar = 1;
			else
				resultScalar = 0;
		}
		else
			resultScalar = 0;

		if(resultLength == 1)
			resultScalar = 1;
		xOffset = offset(xShapeInfo);
		xElementWiseStride = elementWiseStride(xShapeInfo);
		xLength = length(xShapeInfo);
		elementsPerTad = xLength / resultLength;

		if(gridDim.x >= resultLength) {
			reductionIndexesPerBlock = 1;
		}
		else {
			reductionIndexesPerBlock = resultLength / gridDim.x;
		}

		if(blockDim.x < tensorsForDimension) {
			tadsPerThread = tensorsForDimension / blockDim.x;
		}
		//number of threads > tads
		else {
			tadsPerThread = 1;
		}


		xTadInfo = tadInfo(xShapeInfo,dimension,dimensionLength);


	}
	__syncthreads();



	T curr;
	if(resultScalar) {
		unsigned int i =    blockIdx.x   *  xElementWiseStride + tid;
		unsigned int gridSize = blockDim.x * gridDim.x *  xElementWiseStride;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (xOffset + i  < n)	{
			curr = dx[xOffset + i];
			reduction = update(reduction,op(curr,extraParams),extraParams);
			i += gridSize;
		}

		// each thread puts its local sum into shared memory
		sPartials[tid] = reduction;
		__syncthreads();

		T ** sPartialsRef = (T **) &sPartials;
		aggregatePartials(sPartialsRef,tid,extraParams);



		// write result for this block to global mem
		if (tid == 0) {
			if(postProcessOrNot) {
				result[blockIdx.x] = update(sPartials[0],result[blockIdx.x],extraParams);
			}
			else {
				result[blockIdx.x] = sPartials[0];

			}
		}
	}

	else if(!resultScalar) {
		if(reductionIndexesPerBlock * blockIdx.x >= resultLength)
			return;


		int tadsPerReductionIndex = tensorsForDimension / resultLength;
		//notice here that we do reduction indexes per block
		//so as far as threads / reduction index are concerned, this is only for the particular block
		int threadsPerReductionIndex = blockDim.x / reductionIndexesPerBlock;

		//minimum number of threads needed for each reduction index
		int tadsNeeded = reductionIndexesPerBlock * tadsPerReductionIndex;

		//for each reduction index in the block
		//get the current reduction index to solve for
		int reductionIndexToProcess = blockIdx.x  * reductionIndexesPerBlock;
		//don't need all threads
		if(tid >= tadsNeeded)
			return;
		else {
			//process each tad
			//tad wrt the thread
			int currTad = tid + (blockIdx.x  * reductionIndexesPerBlock);
			int offsetForTad = offset(currTad,xShapeInfo,dimension,dimensionLength,xTadInfo);

			//update the reduction for the thread for the current tad
			//note here that we compute the offset and then accumulate in shared memory
			for(int element = 0; element < elementsPerTad; element++,offsetForTad += xElementWiseStride) {
				sPartials[tid] = update(sPartials[tid],dx[offsetForTad],extraParams);
				__syncthreads();
			}

		}


		//first thread for a reduction index
		if(tid % tadsPerReductionIndex == 0 && tadsPerReductionIndex > 1) {
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
			for(int i = 1; i < tadsPerReductionIndex; i++) {
				sPartials[tid] = update(sPartials[tid],sPartials[tid + i],extraParams);
				__syncthreads();
			}
		}



		__syncthreads();

		//after all the threads are done processing each first item in shared memory
		//should correspond to the final value for the particular reduction index
		//that was set for this block.
		if(tid == 0) {
			for(int i = 0; i < reductionIndexesPerBlock; i++) {
				int reductionIndexToProcess = i + blockIdx.x *  reductionIndexesPerBlock;
				result[reductionIndexToProcess] = sPartials[i];
			}

			freePermuteInfo(xTadInfo);

		}

	}

}




//kernels defined to allow lookup by name: notice extern c
extern "C"
__global__ void transform_double(
		int n
		,double *dx
		,int *xShapeInfo
		,double *extraParams
		,double *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	transform<double>(
			n,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,postProcessOrNot);
}


extern "C"
__global__ void transform_float(
		int n
		,float *dx
		,int *xShapeInfo
		,float *extraParams
		,float *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	transform<float>(n,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,dimensionLength,postProcessOrNot);
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
template <typename T>
__device__ void collapseTad(
		T *data
		,T *result
		,T *extraParams,
		int numOriginalTads,
		int sharedMemorySize,
		int *xShapeInfo
		,int *resultShapeInfo
		,int *dimension,int dimensionLength) {
	SharedMemory<T> val;
	//number of tads for the reduced solution
	int numTads = tensorsAlongDimension(xShapeInfo,dimension,dimensionLength);

	volatile T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	//initialize the values
	int numItems = sharedMemorySize / sizeof(T);

	for (int i = tid; i < numItems; i += blockDim.x) {
		sPartials[i] = extraParams[0];
	}
	__syncthreads();



	//each block processes a reduction index
	//don't bother iterating on this block if it goes over the number of tads


	__shared__ TADPermuteInfo xTadInfo;
	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
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
	if(tid == 0) {
		tadsPerReduceIndex2 = tadsPerReduceIndex(numTads,numOriginalTads);
	}

	__syncthreads();


	//each thread does a tad
	if(tid >= numTads || blockIdx.x >= tadsPerReduceIndex2)
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
	int offsetForBlock = offset(tadForThread,xShapeInfo,dimension,dimensionLength,xTadInfo);
	for(int i = 0; i < tadsPerReduceIndex2; offsetForBlock += elementWiseStride(xShapeInfo),i++) {
		sPartials[tid] = update(sPartials[tid],op(data[offsetForBlock],extraParams),extraParams);
		printf("TAD %d and tid %d processing value %f with element wise stride %d and block %d and tads per reduce index %d\n"
				,tadForThread,tid,data[offsetForBlock],elementWiseStride(xShapeInfo),blockIdx.x,tadsPerReduceIndex2);
		__syncthreads();
	}



	if(tid == 0 && blockIdx.x < numTads) {
		//start at 1 so we don't count the first entry twice
		for(int i = 1; i < numTads; i++) {
			sPartials[0] = update(sPartials[0],sPartials[i],extraParams);
			__syncthreads();
		}

		result[blockIdx.x] = sPartials[0];
		freePermuteInfo(xTadInfo);
	}
}



extern "C"
__global__ void collapseTad_float(
		float *data
		,float *result
		,float *extraParams
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo,int *resultShapeInfo
		,int *dimension,int dimensionLength) {
	/**
	 * 		T *data
		,T *result
		,T *extraParams,
		int numOriginalTads,
		int sharedMemorySize,
		int *xShapeInfo,int *resultShapeInfo
		,int *dimension,int dimensionLength
	 */
	collapseTad<float>(
			data,
			result,
			extraParams,
			numOriginalTads,
			sharedMemorySize,
			xShapeInfo,
			resultShapeInfo,
			dimension,
			dimensionLength);

}

extern "C"
__global__ void collapseTad_double(
		double *data
		,double *result
		,double *extraParams
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo,int *resultShapeInfo
		,int *dimension,int dimensionLength) {
	collapseTad<double>(
			data,
			result,
			extraParams,
			numOriginalTads,
			sharedMemorySize,
			xShapeInfo,
			resultShapeInfo,
			dimension,
			dimensionLength);

}





