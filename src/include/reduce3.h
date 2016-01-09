/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_
#include <op.h>
#include <templatemath.h>
namespace functions {
namespace reduce3 {

template<typename T>
class Reduce3: public virtual functions::ops::Op<T> {

public:

	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T postProcess(T reduction, int n, int xOffset, T *dx, int incx,
			T *extraParams, T *result) = 0;

	/**
	 *
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//an op for the kernel
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *extraParams) = 0;

	//calculate an update of the reduce operation
	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T update(T old, T opOutput, T *extraParams) = 0;

	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T merge(T old, T opOutput, T *extraParams) = 0;

#ifdef __CUDACC__
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
				currentBlockOffset = offset(blockIdx.x, xShapeInfo, dimensionLength, xTadInfo);
				endingOffset = offset(blockIdx.x + 1, xShapeInfo, dimensionLength, xTadInfo);
				resultLength = shape::prod(shape::shapeOf(resultShapeInfo), shape::rank(resultShapeInfo));

				//initialize x
				xShape = shape::shapeOf(xShapeInfo);
				xRank = shape::rank(xShapeInfo);
				xOffset = shape::offset(xShapeInfo);
				xElementWiseStride = shape::elementWiseStride(xShapeInfo);

				yOffset = shape::offset(yShapeInfo);
				yElementWiseStride = shape::elementWiseStride(yShapeInfo);

				currentYBlockOffset = offset(blockIdx.x, yShapeInfo, dimensionLength, yTadInfo);
				endingYOffset = offset(blockIdx.x + 1, yShapeInfo, dimensionLength, yTadInfo);

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
				int blockOffset = offset(tadIndex, xShapeInfo, dimensionLength, xTadInfo);
				int blockYOffset = offset(tadIndex, yShapeInfo, dimensionLength, yTadInfo);

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
#endif

	void exec(T *x, int *xShapeInfo, T *extraParams, T *y, int *yShapeInfo,
			T *result, int *resultShapeInfo) {
		T startingVal = extraParams[0];
		int length = shape::length(xShapeInfo);
		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
		int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
		if (xElementWiseStride == 1 && resultElementWiseStride == 1) {
#pragma omp simd
			for (int i = 0; i < length; i++) {
				startingVal = update(startingVal, op(x[i], y[i], extraParams),
						extraParams);
			}

			result[0] = postProcess(startingVal, length,
					shape::offset(xShapeInfo), x,
					shape::elementWiseStride(xShapeInfo), extraParams, result);

		} else {
#pragma omp simd

			for (int i = 0; i < length; i++) {
				startingVal = update(startingVal,
						op(x[i * xElementWiseStride], y[i * yElementWiseStride],
								extraParams), extraParams);
			}

			result[0] = postProcess(startingVal, length,
					shape::offset(xShapeInfo), x,
					shape::elementWiseStride(xShapeInfo), extraParams, result);

		}

	}

	void exec(T *x, int *xShapeInfo, T *extraParams, T *y, int *yShapeInfo,
			T *result, int *resultShapeInfoBuffer, int *dimension,
			int dimensionLength) {
		shape::TADPermuteInfo tadPermuteInfo = shape::tadInfo(xShapeInfo,
				dimension, dimensionLength);
		int resultLength = shape::length(resultShapeInfoBuffer);
		int tadElementWiseStride = shape::computeElementWiseStride(
				tadPermuteInfo.xRank, tadPermuteInfo.permutedShape,
				tadPermuteInfo.permutedStrides,
				shape::order(xShapeInfo) == 'f');
		int tadLength = tadPermuteInfo.tensorShapeProd;
#pragma omp simd
		for (int i = 0; i < shape::length(xShapeInfo); i++) {
			int reductionIndex = shape::reductionIndexForLinear(i,
					tadElementWiseStride, tadLength, resultLength,
					resultLength);
			result[reductionIndex] = update(result[reductionIndex],
					op(x[i], y[i], extraParams), extraParams);
		}
#pragma omp simd
		for (int i = 0; i < resultLength; i++) {
			result[i] = postProcess(result[i], tadLength,
					shape::offset(xShapeInfo), x,
					shape::elementWiseStride(xShapeInfo), extraParams, result);
		}
	}

#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual ~Reduce3() {
	}

};

namespace ops {
template<typename T>
class CosineSimilarity: public virtual Reduce3<T> {
#ifdef __CUDACC__
	__host__ __device__
#endif
	inline T postProcess(T reduction, int n, int xOffset, T *dx, int incx,
			T *extraParams, T *result) {
		return reduction / (extraParams[1] * extraParams[2]);
	}
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//an op for the kernel
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *extraParams) {
		return d1 * d2;
	}

	//calculate an update of the reduce operation
	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T update(T old, T opOutput, T *extraParams) {
		return old + opOutput;
	}

	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T merge(T old, T opOutput, T *extraParams) {
		return update(old, opOutput, extraParams);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("cosinesimilarity_strided");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual ~CosineSimilarity() {
	}
};

template<typename T>
class EuclideanDistance: public virtual Reduce3<T> {
#ifdef __CUDACC__
	__host__ __device__
#endif
	inline T postProcess(T reduction, int n, int xOffset, T *dx, int incx,
			T *extraParams, T *result) {
		return nd4j::math::nd4j_sqrt<T>(reduction);
	}
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//an op for the kernel
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *extraParams) {
		return d1 - d2;
	}

	//calculate an update of the reduce operation
	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T update(T old, T opOutput, T *extraParams) {
		T squared = nd4j::math::nd4j_pow(opOutput, 2.0);
		return squared + old;
	}

	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T merge(T old, T opOutput, T *extraParams) {
		return update(old, opOutput, extraParams);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("euclidean_strided");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual ~EuclideanDistance() {
	}
};

template<typename T>
class ManhattanDistance: public virtual Reduce3<T> {
#ifdef __CUDACC__
	__host__ __device__
#endif
	inline T postProcess(T reduction, int n, int xOffset, T *dx, int incx,
			T *extraParams, T *result) {
		return reduction / extraParams[0] / extraParams[1];
	}
	/**
	 *
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//an op for the kernel
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T op(T d1, T d2, T *extraParams) {
		return d1 - d2;
	}

	//calculate an update of the reduce operation
	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T update(T old, T opOutput, T *extraParams) {
		return nd4j::math::nd4j_pow<T>(old, 2) + opOutput;
	}

	/**
	 *
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual
#ifdef __CUDACC__
	__host__  __device__

#endif
	inline T merge(T old, T opOutput, T *extraParams) {
		return update(old, opOutput, extraParams);
	}

	/** Name of the op
	 * @return the name of the operation
	 */
	virtual
#ifdef __CUDACC__
	inline __host__

#endif
	std::string name() {
		return std::string("manhattan_strided");
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	virtual ~ManhattanDistance() {
	}
};

}

template<typename T>
class Reduce3OpFactory {
public:
	Reduce3OpFactory() {
	}

#ifdef __CUDACC__
	__host__
#endif
	Reduce3<T> * getOp(std::string name) {
		return getOp(name.c_str());
	}
#ifdef __CUDACC__
	__host__ __device__
#endif
	Reduce3<T> * getOp(char * name) {
		if (functions::ops::strcmp(name,"manhattan_strided"))
			return (functions::reduce3::ops::ManhattanDistance<T> *) malloc(sizeof(functions::reduce3::ops::ManhattanDistance<T>));
		else if (functions::ops::strcmp(name,"euclidean_strided"))
			return (functions::reduce3::ops::EuclideanDistance<T> *) malloc(sizeof(functions::reduce3::ops::EuclideanDistance<T>));
		else if (functions::ops::strcmp(name,"cosinesimilarity_strided"))
			return (functions::reduce3::ops::CosineSimilarity<T> *) malloc(sizeof(functions::reduce3::ops::CosineSimilarity<T>));
		return NULL;
	}
};

}
}

#ifdef __CUDACC__
__constant__ functions::reduce3::Reduce3OpFactory<double> *reduce3OpFactory;
__constant__ functions::reduce3::Reduce3OpFactory<float> *reduce3OpFactoryFloat;
extern "C" __global__ void reduce3Double(
		char *name,
		int n, double *dx, int *xShapeInfo,
		double *dy,
		int *yShapeInfo, double *extraParams, double *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	functions::reduce3::Reduce3<double> * op = reduce3OpFactory->getOp(name);
	op->transform(n,dx,xShapeInfo,dy,yShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);
	free(op);

}
extern "C" __global__ void reduce3Float(
		char *name,
		int n, float *dx, int *xShapeInfo,
		float *dy,
		int *yShapeInfo, float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot) {
	functions::reduce3::Reduce3<float> * op = reduce3OpFactoryFloat->getOp(name);
	op->transform(n,dx,xShapeInfo,dy,yShapeInfo,extraParams,result,resultShapeInfo,gpuInformation,dimension,dimensionLength,postProcessOrNot);
	free(op);
}

#endif



#endif /* REDUCE3_H_ */
