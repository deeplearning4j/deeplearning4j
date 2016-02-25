#include <reduce_common.h>
#include <sharedmem.h>
#include <postprocess.h>
//an op for the kernel
template<typename T>
__device__ T op(T d1,T d2,T *extraParams);

//calculate an update of the reduce operation
template<typename T>
__device__ T update(T old,T opOutput,T *extraParams);




template<typename T>
__device__ T merge(T old,T opOutput,T *extraParams);

template <typename T>
__device__ T doBlock(
		int n,
		T *sPartials,
		T *dx,
		int xOffset,
		int incx,
		T *dy,
		int yOffset,
		int incy,
		T *extraParams) {
	T reduce = extraParams[0];
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;
	for (int i = start; i < n; i += totalThreads) {
		int currIdx = xOffset + i * incx;
		int currYIdx = yOffset + i * incy;
		T curr = dx[currIdx];
		T currY = dy[currYIdx];
		reduce = update(reduce,op(curr,currY,extraParams),extraParams);
	}

	return reduce;
}


template<typename T>
__global__ void doReduce(
		T *dx
		,T *extraParams
		,int n
		,int incx
		,int xOffset,
		T *dy,
		int incy,
		int yOffset,
		T *result,
		int resultOffset) {

	SharedMemory<T> val;
	T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	T reduce = doBlock(n,sPartials,dx,xOffset,incx,dy,yOffset,incy,extraParams);
	sPartials[tid] = reduce;
	__syncthreads();

	aggregatePartials(sPartials,tid,extraParams);

	if (tid == 0) {
		result[resultOffset] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}
}

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


/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
template<typename T>
__device__ void transform(
		int n
		,T *dx
		,int *xShapeInfo,
		T *dy,
		int *yShapeInfo
		,T *extraParams
		,T *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	int nIsPow2 = (n % 2 == 0);
	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;


	__shared__ volatile int resultScalar;


	__shared__ int *xShape;
	__shared__ int xRank;
	__shared__ int xElementWiseStride;
	__shared__ int xOffset;


	__shared__ int *yShape;
	__shared__ int yRank;
	__shared__ int yElementWiseStride;
	__shared__ int yOffset;



	//shared memory space for storing intermediate results
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	int numElements = gpuInformation[2] / sizeof(T);
	for (int i = tid; i < numElements; i += blockDim.x)
		sPartials[i] = extraParams[0];
	__syncthreads();


	sPartials[tid] = extraParams[0];
	sPartials[(1 + tid) * 2] = extraParams[0];
	__syncthreads();


	//starting index for tad
	__shared__ volatile int currentYBlockOffset;
	//ending index for tad
	__shared__ volatile int endingYOffset;
	//length for the tad
	__shared__ volatile int yLength;




	//starting index for tad
	__shared__ volatile int currentBlockOffset;
	//ending index for tad
	__shared__ volatile int endingOffset;
	//length for the tad
	__shared__ volatile int xLength;



	__shared__ volatile int resultLength;

	__shared__ volatile int tadsForBlock;

	__shared__ volatile int elementsPerThread;


	//only compute the tad indexes once
	__shared__ TADPermuteInfo xTadInfo;
	__shared__ TADPermuteInfo yTadInfo;
	__shared__ TADPermuteInfo resultTadInfo;

	int valueOffset,valueYOffset;

	__shared__ T startValue;


	T reduction = extraParams[0];
	if(tid == 0) {
		if(dimensionLength == 1) {
			if(dimension[0] == MAX_DIMENSION)
				resultScalar = 1;
			else
				resultScalar = 0;
		}
		else
			resultScalar = 0;
		resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));
		xOffset = offset(xShapeInfo);
		xElementWiseStride = elementWiseStride(xShapeInfo);

		yElementWiseStride = elementWiseStride(yShapeInfo);
		yOffset = offset(yShapeInfo);
	}


	__syncthreads();

	T curr,currY;

	if(resultScalar) {
		int blockSize = gpuInformation[0];

		unsigned int i = xOffset +   blockIdx.x   *  xElementWiseStride + tid;
		unsigned int j = yOffset +   blockIdx.x   *  yElementWiseStride + tid;
		unsigned int gridSize = blockDim.x * gridDim.x *  xElementWiseStride;
		unsigned int gridSizeY = blockDim.x * gridDim.x *  yElementWiseStride;


		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i * xElementWiseStride < xLength && j * yElementWiseStride < yLength)	{
			curr = dx[i];
			currY = dy[j];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			i += gridSize;
			j += gridSizeY;
		}


		// each thread puts its local sum into shared memory
		sPartials[tid] = reduction;
		__syncthreads();

		T ** sPartialsRef = (T **) &sPartials;
		aggregatePartials(sPartialsRef,tid,extraParams);

		// write result for this block to global mem
		if (tid == 0) {
			if(postProcessOrNot)
				result[blockIdx.x] = postProcess(sPartials[0],xLength,xOffset,dx, xElementWiseStride,extraParams,result);
			else {
				result[blockIdx.x] = sPartials[0];
			}
		}
	}

	else if(!resultScalar) {
		if(tid == 0) {
			xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
			yTadInfo  = tadInfo(yShapeInfo,dimension,dimensionLength);
			resultTadInfo = tadInfo(resultShapeInfo,dimension,dimensionLength);


			resultScalar = isScalar(resultShapeInfo);
			currentBlockOffset = offset(blockIdx.x, xShapeInfo,dimension,dimensionLength,xTadInfo);
			endingOffset = offset(blockIdx.x + 1 ,xShapeInfo,dimension,dimensionLength,xTadInfo);
			resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));

			//initialize x
			xShape = shape(xShapeInfo);
			xRank = rank(xShapeInfo);
			xOffset = offset(xShapeInfo);
			xElementWiseStride = elementWiseStride(xShapeInfo);


			//initialize y
			yShape = shape(yShapeInfo);
			yRank = rank(yShapeInfo);
			yOffset = offset(yShapeInfo);
			yElementWiseStride = elementWiseStride(yShapeInfo);


			currentYBlockOffset = offset(blockIdx.x, yShapeInfo,dimension,dimensionLength,yTadInfo);
			endingYOffset = offset(blockIdx.x + 1 ,yShapeInfo,dimension,dimensionLength,yTadInfo);


			//reduction on whole buffer
			if(resultScalar)
				xLength = n;

			else
				xLength = prod(xTadInfo.tensorShape,xTadInfo.tensorShapeLength);

			valueOffset = tadOffset(xShapeInfo,currentBlockOffset);
			double tads = tensorsAlongDimension(xRank,prod(xShape,xRank),xShape,dimension,dimensionLength);
			if(gpuInformation[0] >= MAX_NUM_THREADS && tads > gpuInformation[0])
				tadsForBlock = tadsPerBlock(gpuInformation[0],tads);
			else
				tadsForBlock = 1;
			if(tadsForBlock < 1)
				tadsForBlock = 1;
			//set a constant start value
			startValue = reduction;
			//when the number of elements per tad is greater than grid size, we need to compute partial
			//reductions when initializing
			if(xLength > gpuInformation[1])
				elementsPerThread = xLength / gpuInformation[1];
			else
				elementsPerThread = 1;
		}

		__syncthreads();

		//number of tads per block to process
		for(int i = 0; i < tadsForBlock; i++) {
			int tadIndex = tadForBlockIndex(gpuInformation[0],blockIdx.x,i);
			int blockOffset = offset(tadIndex, xShapeInfo,dimension,dimensionLength,xTadInfo);
			int blockYOffset = offset(tadIndex, yShapeInfo,dimension,dimensionLength,yTadInfo);

			//concurrently load all elements in to shared memory
			if(elementsPerThread > 1) {
				for(int i = 0; i < elementsPerThread; i++) {
					if(i > 0) {
						valueOffset = blockOffset  +(tid * i * xElementWiseStride);
						valueYOffset = blockYOffset + (tid * i * yElementWiseStride);
						//break at the end
						if(valueOffset >= n)
							break;
						T val = dx[valueOffset];
						T yVal = dy[valueYOffset];
						sPartials[tid] = update(sPartials[tid],op(val,yVal,extraParams),extraParams);
					}

					else {
						valueOffset = blockOffset  +(tid * i * xElementWiseStride);
						valueYOffset = blockYOffset + (tid * i * yElementWiseStride);

						//break at the end
						if(valueOffset >= n)
							break;
						T val = dx[valueOffset];
						T yVal = dy[valueYOffset];
						printf("Comparing value x %f and y %f\n",val,yVal);
						sPartials[tid] = val;
						sPartials[(1 + tid)* 2] = yVal;
					}



				}
			}
			else {
				int blockOffset = currentBlockOffset;
				int yBlockOffset = currentYBlockOffset;
				valueOffset = blockOffset  + tid * xElementWiseStride;
				valueYOffset = yBlockOffset + tid * yElementWiseStride;
				T val = dx[valueOffset];
				T val2 = dy[valueYOffset];
				sPartials[tid] = val;
				sPartials[(1 + tid) * 2] = val2;
			}

			__syncthreads();

			//do reduction in shared memory only on the first thread
			if(tid == 0) {
				curr = startValue;
				for(int j = 0; j < xLength; j++) {
					curr = update(curr,op(sPartials[j],sPartials[(1 + j) * 2],extraParams),extraParams);
				}

				if(postProcessOrNot) {
					result[tadIndex] = postProcess(curr,xLength,xOffset,dx, xElementWiseStride,extraParams,result);
				}
				else {
					result[tadIndex] = curr;
				}
			}
		}


	}



	if(resultScalar && tid == 0) {
		freePermuteInfo(xTadInfo);
		freePermuteInfo(yTadInfo);
		freePermuteInfo(resultTadInfo);
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

extern "C"
__global__ void transform_double(
		int n
		,double *dx
		,int *xShapeInfo,
		double *dy,
		int *yShapeInfo
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
			dy,
			yShapeInfo,
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
		,int *xShapeInfo,
		float *dy,
		int *yShapeInfo
		,float *extraParams
		,float *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	transform<float>(
			n,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,postProcessOrNot);

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
		T *data,
		T *y
		,T *result
		,T *extraParams
		,int elementsPerTad
		,int numTads
		,int n
		,int elementWiseStride,
		int yElementWiseStride
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo,
		int *yShapeInfo
		,int *dimension,int dimensionLength) {
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	//intialize the values
	int numItems = sharedMemorySize / sizeof(T);

	for (int i = tid; i < numItems; i += blockDim.x) {
		sPartials[i] = extraParams[0];
	}
	__syncthreads();

	//each block processes a reduction index
	if(blockIdx.x >= numTads)
		return;


	__shared__ TADPermuteInfo xTadInfo;
	__shared__ TADPermuteInfo yTadInfo;
	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
		yTadInfo  = tadInfo(yShapeInfo,dimension,dimensionLength);
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
	int tadsPerReduceIndex2 = tadsPerReduceIndex(numTads,numOriginalTads);
	//each thread does a tad
	if(tid >= tadsPerReduceIndex2)
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
	int offsetForBlockY = offset(tadForThread,yShapeInfo,dimension,dimensionLength,yTadInfo);

	for(int i = 0; i < elementsPerTad; offsetForBlock += elementWiseStride,offsetForBlockY += yElementWiseStride,i++) {
		sPartials[tid] = update(sPartials[tid],op(data[offsetForBlock],y[offsetForBlockY],extraParams),extraParams);
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
		freePermuteInfo(yTadInfo);
	}
}

extern "C"
__global__ void collapseTad_float(
		float *data,
		float *y
		,float *result
		,float *extraParams
		,int elementsPerTad
		,int numTads
		,int n
		,int elementWiseStride,
		int yElementWiseStride
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo,
		int *yShapeInfo
		,int *dimension,int dimensionLength) {
	collapseTad<float>(
				data,
				y,
				result
				,extraParams
				,elementsPerTad,
				numTads,
				n,
				elementWiseStride,
				yElementWiseStride,
				numOriginalTads,
				sharedMemorySize,
				xShapeInfo,
				yShapeInfo,
				dimension,
				dimensionLength);

}

extern "C"
__global__ void collapseTad_double(
		double *data,
		double *y
		,double *result
		,double *extraParams
		,int elementsPerTad
		,int numTads
		,int n
		,int elementWiseStride,
		int yElementWiseStride
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo,
		int *yShapeInfo
		,int *dimension,int dimensionLength) {
	collapseTad<double>(
			data,
			y,
			result
			,extraParams
			,elementsPerTad,
			numTads,
			n,
			elementWiseStride,
			yElementWiseStride,
			numOriginalTads,
			sharedMemorySize,
			xShapeInfo,
			yShapeInfo,
			dimension,
			dimensionLength);

}


