#include <sharedmem.h>
#include <postprocess.h>
#include <shape.h>
#include <reduce3.h>

namespace functions {
namespace reduce3 {


template<typename T>
/**
 *
 */
class BaseReduce3 : public virtual Reduce3<T> {

public:

	/**
	 * Aggregate shared memory
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(T **sPartialsRef, int tid, T *extraParams) {
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

		for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads) {
				sPartials[tid] = update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
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
	virtual __device__ void transform(
			int n, T *dx, int *xShapeInfo,
			T *dy,
			int *yShapeInfo, T *extraParams, T *result,
			int *resultShapeInfo, int *gpuInformation,
			int *dimension,
			int dimensionLength, int postProcessOrNot) {
		/**
		 * Gpu information for the problem
		 */
		int tid = threadIdx.x;


		__shared__ volatile int resultScalar;


		__shared__ int *xShape;
		__shared__ int xRank;
		__shared__ int xElementWiseStride;
		__shared__ int xOffset;

		__shared__ int yElementWiseStride;
		__shared__ int yOffset;



		//shared memory space for storing intermediate results
		SharedMemory <T> val;
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
		__shared__
		shape::TADPermuteInfo xTadInfo;
		__shared__
		shape::TADPermuteInfo yTadInfo;
		__shared__
		shape::TADPermuteInfo resultTadInfo;

		int valueOffset, valueYOffset;

		__shared__
		T startValue;


		T reduction = extraParams[0];
		if (tid == 0) {
			if (dimensionLength == 1) {
				if (dimension[0] == shape::MAX_DIMENSION)
					resultScalar = 1;
				else
					resultScalar = 0;
			}
			else
				resultScalar = 0;
			resultLength = shape::prod(shape::shapeOf(resultShapeInfo), shape::rank(resultShapeInfo));
			xOffset = shape::offset(xShapeInfo);
			xElementWiseStride = shape::elementWiseStride(xShapeInfo);

			yElementWiseStride = shape::elementWiseStride(yShapeInfo);
			yOffset = shape::offset(yShapeInfo);
		}


		__syncthreads();

		T curr, currY;

		if (resultScalar) {

			unsigned int i = xOffset + blockIdx.x * xElementWiseStride + tid;
			unsigned int j = yOffset + blockIdx.x * yElementWiseStride + tid;
			unsigned int gridSize = blockDim.x * gridDim.x * xElementWiseStride;
			unsigned int gridSizeY = blockDim.x * gridDim.x * yElementWiseStride;


			// we reduce multiple elements per thread.  The number is determined by the
			// number of active thread blocks (via gridDim).  More blocks will result
			// in a larger gridSize and therefore fewer elements per thread
			while (i * xElementWiseStride < xLength && j * yElementWiseStride < yLength) {
				curr = dx[i];
				currY = dy[j];
				reduction = update(reduction, op(curr, currY, extraParams), extraParams);
				i += gridSize;
				j += gridSizeY;
			}


			// each thread puts its local sum into shared memory
			sPartials[tid] = reduction;
			__syncthreads();

			T **sPartialsRef = (T **) &sPartials;
			aggregatePartials(sPartialsRef, tid, extraParams);

			// write result for this block to global mem
			if (tid == 0) {
				if (postProcessOrNot)
					result[blockIdx.x] = postProcess(sPartials[0], xLength, xOffset, dx, xElementWiseStride,
							extraParams, result);
				else {
					result[blockIdx.x] = sPartials[0];
				}
			}
		}

		else if (!resultScalar) {
			if (tid == 0) {
				xTadInfo = shape::tadInfo(xShapeInfo, dimension, dimensionLength);
				yTadInfo = shape::tadInfo(yShapeInfo, dimension, dimensionLength);
				resultTadInfo = shape::tadInfo(resultShapeInfo, dimension, dimensionLength);


				resultScalar = shape::isScalar(resultShapeInfo);
				currentBlockOffset = offset(blockIdx.x, xShapeInfo, dimension, dimensionLength, xTadInfo);
				endingOffset = offset(blockIdx.x + 1, xShapeInfo, dimension, dimensionLength, xTadInfo);
				resultLength = shape::prod(shape::shapeOf(resultShapeInfo), shape::rank(resultShapeInfo));

				//initialize x
				xShape = shape::shapeOf(xShapeInfo);
				xRank = shape::rank(xShapeInfo);
				xOffset = shape::offset(xShapeInfo);
				xElementWiseStride = shape::elementWiseStride(xShapeInfo);



				yOffset = shape::offset(yShapeInfo);
				yElementWiseStride = shape::elementWiseStride(yShapeInfo);


				currentYBlockOffset = offset(blockIdx.x, yShapeInfo, dimension, dimensionLength, yTadInfo);
				endingYOffset = offset(blockIdx.x + 1, yShapeInfo, dimension, dimensionLength, yTadInfo);


				//reduction on whole buffer
				if (resultScalar)
					xLength = n;

				else
					xLength = shape::prod(xTadInfo.tensorShape, xTadInfo.tensorShapeLength);

				valueOffset = shape::tadOffset(xShapeInfo, currentBlockOffset);
				double tads = shape::tensorsAlongDimension(xRank, shape::prod(xShape, xRank), xShape, dimension,
						dimensionLength);
				if (gpuInformation[0] >= shape::MAX_NUM_THREADS && tads > gpuInformation[0])
					tadsForBlock = shape::tadsPerBlock(gpuInformation[0], tads);
				else
					tadsForBlock = 1;
				if (tadsForBlock < 1)
					tadsForBlock = 1;
				//set a constant start value
				startValue = reduction;
				//when the number of elements per tad is greater than grid size, we need to compute partial
				//reductions when initializing
				if (xLength > gpuInformation[1])
					elementsPerThread = xLength / gpuInformation[1];
				else
					elementsPerThread = 1;
			}

			__syncthreads();

			//number of tads per block to process
			for (int i = 0; i < tadsForBlock; i++) {
				int tadIndex = shape::tadForBlockIndex(gpuInformation[0], blockIdx.x, i);
				int blockOffset = offset(tadIndex, xShapeInfo, dimension, dimensionLength, xTadInfo);
				int blockYOffset = offset(tadIndex, yShapeInfo, dimension, dimensionLength, yTadInfo);

				//concurrently load all elements in to shared memory
				if (elementsPerThread > 1) {
					for (int i = 0; i < elementsPerThread; i++) {
						if (i > 0) {
							valueOffset = blockOffset + (tid * i * xElementWiseStride);
							valueYOffset = blockYOffset + (tid * i * yElementWiseStride);
							//break at the end
							if (valueOffset >= n)
								break;
							T val = dx[valueOffset];
							T yVal = dy[valueYOffset];
							sPartials[tid] = update(sPartials[tid], op(val, yVal, extraParams), extraParams);
						}

						else {
							valueOffset = blockOffset + (tid * i * xElementWiseStride);
							valueYOffset = blockYOffset + (tid * i * yElementWiseStride);

							//break at the end
							if (valueOffset >= n)
								break;
							T val = dx[valueOffset];
							T yVal = dy[valueYOffset];
							printf("Comparing value x %f and y %f\n", val, yVal);
							sPartials[tid] = val;
							sPartials[(1 + tid) * 2] = yVal;
						}


					}
				}
				else {
					int blockOffset = currentBlockOffset;
					int yBlockOffset = currentYBlockOffset;
					valueOffset = blockOffset + tid * xElementWiseStride;
					valueYOffset = yBlockOffset + tid * yElementWiseStride;
					T val = dx[valueOffset];
					T val2 = dy[valueYOffset];
					sPartials[tid] = val;
					sPartials[(1 + tid) * 2] = val2;
				}

				__syncthreads();

				//do reduction in shared memory only on the first thread
				if (tid == 0) {
					curr = startValue;
					for (int j = 0; j < xLength; j++) {
						curr = update(curr, op(sPartials[j], sPartials[(1 + j) * 2], extraParams), extraParams);
					}

					if (postProcessOrNot) {
						result[tadIndex] = postProcess(curr, xLength, xOffset, dx, xElementWiseStride,
								extraParams, result);
					}
					else {
						result[tadIndex] = curr;
					}
				}
			}


		}


		if (resultScalar && tid == 0) {
			shape::freePermuteInfo(xTadInfo);
			shape::freePermuteInfo(yTadInfo);
			shape::freePermuteInfo(resultTadInfo);
		}


	}

};

}
}

