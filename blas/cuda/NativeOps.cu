#include "../NativeOps.h"
#include <cuda.h>
#include <cuda_launch_config.h>

#include <buffer.h>
#include <shape.h>

#include <cublas_v2.h>
#include <reduce3.h>
#include <reduce.h>
#include <indexreduce.h>
#include <pairwise_transform.h>
#include <transform.h>
#include <scalar.h>
#include <broadcasting.h>
#include <summarystatsreduce.h>
#include <thread>
#include <map>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <pointercast.h>
#include <stdio.h>

cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[28];
int blockLimit = 128;
int maxThreads = 512;
bool debug = true;
bool verbose = true;

template <typename T>
dim3 getOptimalDimensions(Nd4jIndex n,cudaFuncAttributes attributes, cudaDeviceProp properties) {

	// we can combine the two to compute a block size
	int num_threads = block_size_with_maximum_potential_occupancy(attributes, properties);

	// no real sense launching more threads, then number of elements we have
	if (num_threads > n) num_threads = n;

	if (maxThreads > 0 && num_threads > maxThreads) num_threads = maxThreads;

	// compute the number of blocks of size num_threads to launch
	int num_blocks = n / num_threads;

	// check for partial block at the end

	if (num_blocks > blockLimit) num_blocks = blockLimit;

	if (num_blocks < 4 && n > 128) {
		num_blocks = 4;
		num_threads = n / num_blocks;
	}

	if (num_threads >= 768) {
		num_blocks = num_blocks * 2;
		num_threads = num_threads / 2;
	}

	if(n % num_threads && num_blocks < blockLimit) ++num_blocks;
    //(num_threads * sizeof(T)) + attributes.sharedSizeBytes);
	return dim3(num_blocks,num_threads, 3000);
}

int getBaseMemorySize(int xRank, int yRank, int zRank) {
	int memory_limit = 1024;

	if (xRank == 0) xRank = 2;

	memory_limit += (xRank * 2 + 4) * 3; // we reserve memory for xShape + T1/T2 shapes
	memory_limit += yRank == 0 ? 0 : (yRank * 2 + 4);
	memory_limit += zRank == 0 ? 0 : (zRank * 2 + 4);
	memory_limit += xRank * 6;
	memory_limit += MAX_RANK; // special case, needed roughtly in one pase

	return memory_limit;
}

int getDeviceBlockThreshold(int deviceId) {
	int ccMinor = deviceProperties[deviceId].minor;
	int ccMajor = deviceProperties[deviceId].major;

	int blockThreshold;

	if (ccMajor >= 5)
		blockThreshold = 32;
	else if (ccMajor == 3)
		blockThreshold = 16;
	else if (ccMajor < 3)
		blockThreshold = 8;

	return blockThreshold;
}

dim3 getBasicLaunchParams(int deviceId, long problemLength, int sharedMemoryPerThread) {
	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int blockThreshold = getDeviceBlockThreshold(deviceId);

	int num_threads = problemLength / (countMP * blockThreshold);
	num_threads = 64;

	int num_blocks = 64;

	int memory_limit = (sharedMemoryPerThread * num_threads) + getBaseMemorySize(1,0,0);

	dim3 launchDims = dim3(num_blocks, num_threads, memory_limit);

	if (debug)
		printf("Preliminary basic launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i]\n", num_blocks, num_threads, memory_limit);


	return launchDims;
}

int getDeviceSharedThreshold(int deviceId) {
	int ccMinor = deviceProperties[deviceId].minor;
	int ccMajor = deviceProperties[deviceId].major;

	// please note threshold isn't multiple of 32, and that's NOT a mistake

	int shmemThreshold;
	if (ccMajor == 5 && ccMinor == 2)
		shmemThreshold = 96000;
	else if (ccMajor == 5)
		shmemThreshold = 64000;
	else if (ccMajor == 3 && ccMinor == 7)
		shmemThreshold = 112000;
	else shmemThreshold = 48000;

	return shmemThreshold;
}


dim3 getBetterDimensions(int deviceId, int numTads, int tadLength, int xRank, int yRank, int zRank, int dimensionLength, int elementSize, int reduction) {

	int num_threads = nd4j::math::nd4j_min<int>(tadLength, maxThreads);



	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;
	int warpSize = deviceProperties[deviceId].warpSize;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	// round num_threads to nearest warpSize
	num_threads -= num_threads % warpSize;


	// since we use shared memory as fast memory for some cases - we need to count that in
	int memory_limit = getBaseMemorySize(xRank, yRank, zRank);
	int memory_floor = memory_limit;
	int effective_block_limit =  countMP * blockThreshold;

	// at this moment we've stored all required information for things. time to count in reduction multipliers
	int reduction_per_block = 0;
	bool found = false;
	if (reduction > 0)
		while (!found) {
			reduction_per_block = (num_threads * elementSize * reduction);
			if (memory_limit + reduction_per_block < 5000) {
				memory_limit += reduction_per_block;
				found = true;
			} else {
				if (num_threads >= 128) {
					num_threads -= 32;
				} else {
					memory_limit += reduction_per_block;
					found = true;
				}
			}
		}

	// at this moment we know total memory used per block, and we also know per-mp limit.
	int max_active_blocks = shmemThreshold / memory_limit;

	// we don't want to spawn more blocks, that gpu can actually handle without queue
	int num_blocks = nd4j::math::nd4j_min<int>(numTads, effective_block_limit);
	num_blocks = nd4j::math::nd4j_min<int>(num_blocks, max_active_blocks);
	num_blocks = nd4j::math::nd4j_min<int>(num_blocks, blockLimit);
    num_blocks = num_blocks - (num_blocks % countMP);
	num_blocks = nd4j::math::nd4j_max<int>(num_blocks, 1);

	int targetBlocksPerMP = num_blocks / countMP;

	// now we know desired number of blocks wrt to shared memory. So, now we should take in account number of threads per SM
	if (targetBlocksPerMP * num_threads > 2048) {
		while (targetBlocksPerMP * num_threads > 2048) {
			num_threads -= 32;
			if (num_threads <= 96)
				break;
		}

		memory_limit = memory_floor + (num_threads * elementSize * reduction);
	}




	if (debug)
		printf("Preliminary reduce launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], reduction_per_block: [%i], blocksPerMP: [%i]\n", num_blocks, num_threads, memory_limit, reduction_per_block, targetBlocksPerMP);

	return dim3(num_blocks,num_threads, memory_limit);
}


dim3 getFlatLaunchParams(int deviceId, int *xShapeInfo, int *yShapeInfo) {
	int xRank = shape::rank(xShapeInfo);
	int yRank = yShapeInfo == nullptr ? 0 : shape::rank(yShapeInfo);
	int zRank = 0;

	int memory_limit = getBaseMemorySize(xRank, yRank, zRank);

	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	int xLength = shape::length(xShapeInfo);



	int memory_floor = memory_limit;
	int effective_block_limit =  countMP * blockThreshold;

	int num_blocks = nd4j::math::nd4j_min<int>(blockLimit, effective_block_limit);
	num_blocks = nd4j::math::nd4j_max<int>(num_blocks, 1);
	int num_threads = 1024;

	int targetBlocksPerMP = num_blocks / countMP;

	// now we know desired number of blocks wrt to shared memory. So, now we should take in account number of threads per SM
	if (targetBlocksPerMP * num_threads > 2048) {
		while (targetBlocksPerMP * num_threads > 2048) {
			num_threads -= 32;
			if (num_threads <= 96)
				break;
		}
	}

	dim3 launchDims = dim3(num_blocks, num_threads, memory_limit);

	if (debug)
		printf("Preliminary scalar launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], blocksPerMP: [%i], problemLength: [%i]\n", num_blocks, num_threads, memory_limit, targetBlocksPerMP, xLength);


	return launchDims;
}

dim3 getReduceLaunchParams(int deviceId, int *xShapeInfo, int *yShapeInfo, int *zShapeInfo, int dimensionLength, int elementSize, int reductionSize) {

	int tadLength;
	int numTads;
	if (zShapeInfo != nullptr) {
		tadLength = shape::length(xShapeInfo) / shape::length(zShapeInfo);
		numTads = shape::length(xShapeInfo) / tadLength;

		if (tadLength == 1) {
			printf("xLength: [%i], zLength: [%i]\n", shape::length(xShapeInfo), shape::length(zShapeInfo));
		}
	} else{
		// we have special case - reduction along all dimensions
		tadLength = 2048;
		numTads = shape::length(xShapeInfo) / tadLength;
	}

	int xRank = shape::rank(xShapeInfo);
	int yRank = yShapeInfo == nullptr ? 0 : shape::rank(yShapeInfo);
	int zRank = zShapeInfo == nullptr ? 0 : shape::rank(zShapeInfo);

	return getBetterDimensions(deviceId, numTads, tadLength, xRank, yRank, zRank, dimensionLength, elementSize, reductionSize);
}

/**
 * Returns optimal launch parameters
 * given the extra pointers passed in.
 * The extra pointer should be
 * the host pointer for the shape information
 * associated with the data.
 * From there it is used to obtain the length
 * from which we can derive the optimal launch parameters.
 *
 */
template <typename T>
dim3 getOptimalLaunchParameters(Nd4jPointer *extraPointers, cudaFuncAttributes attributes, cudaDeviceProp properties) {
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	Nd4jIndex n = shape::length(hostXShapeInfo);

	dim3 launchDims = getOptimalDimensions<T>(n,attributes, properties);

	if (debug)
		printf("Params: gridSize: [%i], blockSize: [%i], shMem: [%i], problemLength: [%i], totalThreads:[%i]\n", launchDims.x, launchDims.y, launchDims.z, n, (launchDims.x * launchDims.y));

	return launchDims;
}


nd4j::buffer::Buffer<int> * createScalarBuffer(cudaStream_t stream) {
	int *scalarShapeInfo = shape::createScalarShapeInfo();
	nd4j::buffer::Buffer<int> *buff = nd4j::buffer::createBuffer(scalarShapeInfo,shape::shapeInfoLength(2), stream);
	nd4j::buffer::copyDataToGpu(&buff, stream);
	return buff;
}


class ScalarShapeInformation {
private:
	nd4j::buffer::Buffer<int> *scalarDimension;
	nd4j::buffer::Buffer<int> *scalarShapeInfo;
	std::thread::id threadId;

public:
	ScalarShapeInformation(cudaStream_t stream) {
		int *scalarDimensionBuff = (int *) malloc(sizeof(int));
		scalarDimensionBuff[0] = MAX_DIMENSION;
		scalarDimension = nd4j::buffer::createBuffer(scalarDimensionBuff,1, stream);
		scalarShapeInfo = createScalarBuffer(stream);
		threadId = std::this_thread::get_id();

	}
	~ScalarShapeInformation() {
		nd4j::buffer::freeBuffer(&scalarShapeInfo);
		nd4j::buffer::freeBuffer(&scalarDimension);
	}


	int *getShapeInfoHostPointer() {
		return scalarShapeInfo->data;
	}

	int * getShapeInfoGpuPointer() {
		return scalarShapeInfo->gData;
	}

	int * getDimensionHostPointer() {
		return scalarDimension->data;
	}

	int  * getDimensionGpuPointer() {
		return scalarDimension->gData;
	}

};





template <typename T>
class ScalarInfo {
	nd4j::buffer::Buffer<T> *scalarData;
	ScalarShapeInformation *shapeInfo;
	T finalResult;
	cudaStream_t streamRef;
public:
	ScalarInfo(cudaStream_t stream) {
		T *scalarResult = (T*)malloc(sizeof(T));
		shapeInfo = new ScalarShapeInformation(stream);
		scalarData = nd4j::buffer::createBuffer(scalarResult,1, stream);
		streamRef = stream;
		nd4j::buffer::copyDataToGpu(&scalarData, stream);
	}

	T getFinalResultFromDevice() {
		nd4j::buffer::copyDataFromGpu(&scalarData, streamRef);
		return scalarData->data[0];
	}

	/**
	 * Get the device shape information
	 * representing a scalar
	 */
	 int *getDeviceShapeInfo() {
		return shapeInfo->getShapeInfoGpuPointer();
	}

	/**
	 * Get the result pointers
	 */
	 T *getDevicePointer() {
		 return scalarData->gData;
	 }

	 /**
	  * Get the infinite dimension device pointer
	  */
	 int *getDimensionDevicePointer() {
		 return shapeInfo->getDimensionGpuPointer();
	 }

	 ~ScalarInfo() {
		 nd4j::buffer::freeBuffer(&scalarData);
		 delete shapeInfo;
	 }
};


/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D1 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[27], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(double), 2);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	double result = resultPointer[0];
	return result;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension, int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D2 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, dimensionLength, sizeof(double), 2);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execBroadcastDouble(Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer dimension, int dimensionLength){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D3 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[26], deviceProperties[(int) extraPointers[2]]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer, yShapeInfoPointer,  dimensionLength, sizeof(double), 0);

	broadcastDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			yPointer,
			yShapeInfoPointer, shape::rank(yShapeInfoPointer),
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}



/**
 *
 * @param opNum
 * @param dx
 * @param xStride
 * @param y
 * @param yStride
 * @param result
 * @param resultStride
 * @param extraParams
 * @param n
 */
void   NativeOps::execPairwiseTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		int xStride,
		Nd4jPointer y,
		int yStride,
		Nd4jPointer result,
		int resultStride,
		Nd4jPointer extraParams, Nd4jIndex n) {
	double *xPointer = reinterpret_cast<double *>(dx);
	double *yPointer = reinterpret_cast<double *>(y);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D4 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[25], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>> (
			opNum,
			n,
			xPointer,
			yPointer,
			xStride,
			yStride,
			extraParamsPointer,
			resultPointer,
			resultStride, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 * @param xIndexes
 * @param yIndexes
 * @param resultIndexes
 */
void NativeOps::execPairwiseTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer xIndexes,
		Nd4jPointer yIndexes,
		Nd4jPointer resultIndexes) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *yIndexesPointer = reinterpret_cast<int *>(yIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D5 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[24], deviceProperties[(int) extraPointers[2]]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformDoubleIndex <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			yShapeInfoPointer, shape::rank(yShapeInfoPointer),
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			xIndexesPointer,
			yIndexesPointer,
			resultIndexesPointer, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void NativeOps::execPairwiseTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer  xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer  yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer  resultShapeInfo,
		Nd4jPointer extraParams) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D6 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[23], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	pairWiseTransformDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
			yShapeInfoPointer,  shape::rank(yShapeInfoPointer),
			resultShapeInfoPointer,  shape::rank(resultShapeInfoPointer), allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D7 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, 1, sizeof(double), 1);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			nullptr,
			1,
			1,
			allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceDouble(
		Nd4jPointer *extraPointers
		,int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer dimension,
		int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D8 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, dimensionLength, sizeof(double), 1);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1,
			allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
double NativeOps::execReduceScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D9 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(double), 1);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,
			allocPointer, reductionPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	double result = resultPointer[0];
	return result;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduce3Double(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D10 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, resultShapeInfoPointer, 1, sizeof(double), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Double<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
			nullptr,
			1,
			1, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 */
double   NativeOps::execReduce3ScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	if (debug && verbose)
		printf("D11 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, nullptr, 1, sizeof(double), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Double<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
			nullptr,
			1,
			1, allocationPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	double result  = resultPointer[0];
	return result;
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Double(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension,
		int dimensionLength){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D12 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, resultShapeInfoPointer, dimensionLength, sizeof(double), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Double<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
			dimensionPointer,
			dimensionLength,
			1, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}
/**
 *
 * @param opNum
 * @param x
 * @param xStride
 * @param result
 * @param resultStride
 * @param scalar
 * @param extraParams
 * @param n
 */
void   NativeOps::execScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		int xStride,
		Nd4jPointer result,
		int resultStride,
		double scalar,
		Nd4jPointer extraParams,
		Nd4jIndex n) {
	double *xPointer = reinterpret_cast<double *>(x);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D13 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[20], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			xPointer,
			xStride,
			extraParamsPointer,
			resultPointer,resultStride, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param scalar
 * @param extraParams
 * @param n
 */
void NativeOps::execScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		double scalar,
		Nd4jPointer extraParams){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D14 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[19], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			scalar,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,resultShapeInfoPointer, shape::rank(resultShapeInfoPointer), allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param scalar
 * @param extraParams
 * @param n
 * @param xIndexes
 * @param resultIndexes
 */
void NativeOps::execScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		double scalar,
		Nd4jPointer extraParams,
		Nd4jIndex n,
		Nd4jPointer xIndexes,
		Nd4jPointer resultIndexes){
	double *xPointer = reinterpret_cast<double *>(x);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D15 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[18], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], (int *) resultShapeInfo);

	scalarDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			xPointer,
			extraParamsPointer,
			resultPointer,
			resultIndexesPointer, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execSummaryStatsScalarDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,bool biasCorrected){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D16 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(double), 8);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	double result = resultPointer[0];
	return result;

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execSummaryStatsDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,bool biasCorrected) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	if (debug && verbose)
		printf("D17 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, 1, sizeof(double), 8);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension, int dimensionLength,bool biasCorrected){
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D18 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, dimensionLength, sizeof(double), 8);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1,biasCorrected, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param dx
 * @param xStride
 * @param result
 * @param resultStride
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		int xStride,
		Nd4jPointer result,
		int resultStride,
		Nd4jPointer extraParams,
		Nd4jIndex n) {
	double *xPointer = reinterpret_cast<double *>(dx);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D19 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[16], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	transformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			xPointer,
			xStride,
			extraParamsPointer,
			resultPointer,resultStride, allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams){
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *resultShapeInfoPointer =  reinterpret_cast<int *>(resultShapeInfo);
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("D20 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	double *specialPointer = reinterpret_cast<double *>(extraPointers[6]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	int *dimension = (int *) allocPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	double * special = (double *) maxShapeBuffer + 8;

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			transformDouble<<< 1, nd4j::math::nd4j_min<int>(512, length), launchDims.z, *stream >>> (
					opNum,
							xPointer,
							xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
							extraParamsPointer,
							resultPointer, resultShapeInfoPointer,  shape::rank(resultShapeInfoPointer), allocPointer, reductionPointer);
		} else {
			// going for blockwise specials
			//float *xpf = reinterpret_cast<float *>(dx);

			int *shape = shape::shapeOf(hostXShapeInfo);
			//printf("Rows num: %i\n", shape[0]);
			switch (opNum) {
				case 40: // LogSoftMax
				case 39: // SoftMax Derivative
				case 38: {// softmax
					prepareShapeBuffer << < 1, 1, 128, *stream >> > (dimension, maxDimension, maxShapeBuffer, shape[0]);

					//checkCudaErrors(cudaStreamSynchronize(*stream));

					// max 3
					execReduceDouble(extraPointers, 3, dx, xShapeInfo, extraParams, (Nd4jPointer) special,
									(Nd4jPointer) maxShapeBuffer, (Nd4jPointer) maxDimension, 1);

					// sub 1
					execBroadcastDouble(extraPointers, 1, dx, xShapeInfo, (Nd4jPointer) special,
									   (Nd4jPointer) maxShapeBuffer, dx, xShapeInfo, (Nd4jPointer) dimension, 1);

					// exp 3
					execTransformDouble(extraPointers, 3, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					//sum 1
					execReduceDouble(extraPointers, 1, dx, xShapeInfo, extraParams, (Nd4jPointer) special,
									(Nd4jPointer) maxShapeBuffer, (Nd4jPointer) maxDimension, 1);

					// divide 3
					execBroadcastDouble(extraPointers, 3, dx, xShapeInfo, (Nd4jPointer) special,
									   (Nd4jPointer) maxShapeBuffer, dx, xShapeInfo, (Nd4jPointer) dimension, 1);

					// log 3
					if (opNum == 40)
						execTransformDouble(extraPointers, 5, dx, xShapeInfo, dx, xShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformDouble(extraPointers, 42, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					break;
				}
				case 41: {
					// IsMax along all dimensions
					if (extraParamsPointer == nullptr) {
						int maxIdx = (int) execIndexReduceScalarDouble(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxDouble<<< 8, 96, 0, *stream >>>(resultPointer, shape::length(hostXShapeInfo), targetIdx);
					} else {
						// going for dimension-based IsMax
						execIndexReduceDouble(extraPointers,0, dx, xShapeInfo, extraParams, result, resultShapeInfo, (Nd4jPointer) dimension, 1);
					}
					break;
				}
				default: {
					printf("Bad case for transformFloat\n");
					break;
				}
			}
		}
	} else {
		transformDouble<<<launchDims.x, launchDims.y, launchDims.z, *stream>>> (
				opNum,
						xPointer,
						xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
						extraParamsPointer,
						resultPointer, resultShapeInfoPointer, shape::rank(resultShapeInfoPointer), allocPointer, reductionPointer);
	}
	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer xIndexes,
		Nd4jPointer resultIndexes) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D21 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[14], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], (int *) resultShapeInfo);

	transformDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultIndexesPointer, allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
float   NativeOps::execIndexReduceScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	if (debug && verbose)
		printf("F1 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[(int) extraPointers[2]]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(float), 2);

	indexReduceFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension,
		int dimensionLength){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F2 opNum:[%i]\n", opNum);

	// dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[(int) extraPointers[2]]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, dimensionLength, sizeof(float), 2);

	indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execBroadcastFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer dimension, int dimensionLength){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F3 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[12], deviceProperties[(int) extraPointers[2]]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer, yShapeInfoPointer, 1, sizeof(float), 0);

	broadcastFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			yPointer,
			yShapeInfoPointer, shape::rank(yShapeInfoPointer),
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}



/**
 *
 * @param opNum
 * @param dx
 * @param xStride
 * @param y
 * @param yStride
 * @param result
 * @param resultStride
 * @param extraParams
 * @param n
 */
void   NativeOps::execPairwiseTransformFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		int xStride,
		Nd4jPointer y,
		int yStride,
		Nd4jPointer result,
		int resultStride,
		Nd4jPointer extraParams, Nd4jIndex n){
	float *xPointer = reinterpret_cast<float *>(dx);
	float *yPointer = reinterpret_cast<float *>(y);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F4 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[11], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(float), 0);

	pairWiseTransformStridedFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			n,
			xPointer,
			yPointer,
			xStride,
			yStride,
			extraParamsPointer,
			resultPointer,
			resultStride, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 * @param xIndexes
 * @param yIndexes
 * @param resultIndexes
 */
void NativeOps::execPairwiseTransformFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer xIndexes,
		Nd4jPointer yIndexes,
		Nd4jPointer resultIndexes){
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *yIndexesPointer = reinterpret_cast<int *>(yIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F5 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[10], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, resultShapeInfoPointer, 1, sizeof(float), 0);

	pairWiseTransformFloatIndex<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			yShapeInfoPointer, shape::rank(yShapeInfoPointer),
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			xIndexesPointer,
			yIndexesPointer,
			resultIndexesPointer, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void NativeOps::execPairwiseTransformFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer  xShapeInfo,
		Nd4jPointer y,
		Nd4jPointer  yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer  resultShapeInfo,
		Nd4jPointer extraParams){
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F6 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[9], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer,  yShapeInfoPointer, 1, sizeof(float), 0);
	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	pairWiseTransformFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			yShapeInfoPointer, shape::rank(yShapeInfoPointer),
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer), allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F7 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, 1, sizeof(float), 1);

	reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			nullptr,
			1,
			1,
			allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer dimension,int dimensionLength){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F8 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);


	// DO NOT REMOVE COMMENTS OR CODE BELOW.
	// raver119@gmail.com

//	shape::TAD *tad = new shape::TAD();
//	tad->init(xShapeInfoPointer, dimensionPointer, dimensionLength);
//	tad->setOutputBuffer(allocPointer);
//	tad->createTadOnlyShapeInfo();

//	shape::printShapeInfo(tad->tadOnlyShapeInfo);

// dim3 getBetterDimensions(int deviceId, int numTads, int tadLength, int xRank, int yRank, int zRank, int dimensionLength, int elementSize, int reduction)

	dim3 temp = getReduceLaunchParams((int) extraPointers[2], xShapeInfoPointer, nullptr, resultShapeInfoPointer, dimensionLength, sizeof(float), 1);

	reduceFloat<<<temp.x,temp.y,temp.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1,
			allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

	//delete tad;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
float NativeOps::execReduceScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F9 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 temp = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(float), 1);

	reduceFloat<<< temp.x,temp.y, temp.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,
			allocPointer,
			reductionPointer
	);


	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduce3Float(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F10 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, resultShapeInfoPointer, 1, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Float<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
			nullptr,
			1,
			1, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 */
float   NativeOps::execReduce3ScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F11 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, nullptr, 1, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Float<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
			nullptr,
			1,
			1, allocationPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	double result  = resultPointer[0];
	return result;

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Float(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParamsVals,
		Nd4jPointer y,
		Nd4jPointer yShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension,
		int dimensionLength){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	if (debug && verbose)
		printf("F12 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer, resultShapeInfoPointer, dimensionLength, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], yShapeInfoPointer);
	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length((int *) extraPointers[0]), 16);

	reduce3Float<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
			dimensionPointer,
			dimensionLength,
			1, allocationPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param x
 * @param xStride
 * @param result
 * @param resultStride
 * @param scalar
 * @param extraParams
 * @param n
 */
void   NativeOps::execScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		int xStride,
		Nd4jPointer result,
		int resultStride,
		double scalar,
		Nd4jPointer extraParams,
		Nd4jIndex n){
	float *xPointer = reinterpret_cast<float *>(x);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	if (debug && verbose)
		printf("F13 opNum:[%i]\n", opNum);

	dim3 temp = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	scalarFloat<<<temp.x,temp.y,temp.z, *stream>>>(
			opNum,
			n,
			scalar,
			xPointer,
			xStride,
			extraParamsPointer,
			resultPointer,resultStride, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param scalar
 * @param extraParams
 * @param n
 */
void NativeOps::execScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		float scalar,
		Nd4jPointer extraParams){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	Nd4jIndex n = shape::length(hostXShapeInfo);

	if (debug && verbose)
		printf("F14 opNum:[%i]\n", opNum);

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 temp = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	scalarFloat<<<temp.x, temp.y,temp.z, *stream>>>(
			opNum,
			scalar,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,resultShapeInfoPointer, shape::rank(resultShapeInfoPointer), allocPointer );

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param scalar
 * @param extraParams
 * @param n
 * @param xIndexes
 * @param resultIndexes
 */
void NativeOps::execScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		double scalar,
		Nd4jPointer extraParams,
		Nd4jPointer xIndexes,
		Nd4jPointer resultIndexes){
	float *xPointer = reinterpret_cast<float *>(x);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	Nd4jIndex n = shape::length(hostShapeInfo);

	if (debug && verbose)
		printf("F15 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[4], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], (int *) resultShapeInfo);

	scalarFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			xPointer,
			extraParamsPointer,
			resultPointer,
			resultIndexesPointer, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
float   NativeOps::execSummaryStatsScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,bool biasCorrected){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F16 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, nullptr, 1, sizeof(float), 8);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execSummaryStatsFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,bool biasCorrected){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F17 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, 1, sizeof(float), 8);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer x,
		Nd4jPointer xShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfoBuffer,
		Nd4jPointer dimension,
		int dimensionLength,bool biasCorrected){
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F18 opNum:[%i]\n", opNum);

	//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr, resultShapeInfoPointer, dimensionLength, sizeof(float), 8);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer, shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer, shape::rank(resultShapeInfoPointer),
			dimensionPointer,
			dimensionLength,
			1,biasCorrected, allocationPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}
/**
 *
 * @param opNum
 * @param dx
 * @param xStride
 * @param result
 * @param resultStride
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		int xStride,
		Nd4jPointer result,
		int resultStride,
		Nd4jPointer extraParams,
		Nd4jIndex n) {
	float *xPointer = reinterpret_cast<float *>(dx);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F19 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[2], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	transformFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			xPointer,
			xStride,
			extraParamsPointer,
			resultPointer,resultStride, allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformFloat(Nd4jPointer *extraPointers,int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams) {
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("F20 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	float *specialPointer = reinterpret_cast<float *>(extraPointers[6]);

	int *dimension = (int *) allocPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float * special = (float *) maxShapeBuffer + 8;

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], resultShapeInfoPointer);

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(length, 256);
			transformFloat <<< 1, block, launchDims.z + (block * sizeof(float) * 4), *stream >> > (
					opNum,
					xPointer,
					xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
					extraParamsPointer,
					resultPointer, resultShapeInfoPointer,  shape::rank(resultShapeInfoPointer),  allocPointer, reductionPointer);
		} else {
			// going for blockwise specials
			//float *xpf = reinterpret_cast<float *>(dx);

			int *shape = shape::shapeOf(hostXShapeInfo);
			switch (opNum) {
				case 40: // LogSoftMax
				case 39: // SoftMax Derivative
				case 38: {// softmax
					prepareShapeBuffer << < 1, 1, 128, *stream >> > (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					shape::printShapeInfo(maxShapeBuffer);

					// max 3
					execReduceFloat(extraPointers, 3, dx, xShapeInfo, extraParams, (Nd4jPointer) special,
									(Nd4jPointer) maxShapeBuffer, (Nd4jPointer) maxDimension, 1);

					// sub 1
					execBroadcastFloat(extraPointers, 1, dx, xShapeInfo, (Nd4jPointer) special,
									   (Nd4jPointer) maxShapeBuffer, dx, xShapeInfo, (Nd4jPointer) dimension, 1);

					// exp 3
					execTransformFloat(extraPointers, 3, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					//sum 1
					execReduceFloat(extraPointers, 1, dx, xShapeInfo, extraParams, (Nd4jPointer) special,
									(Nd4jPointer) maxShapeBuffer, (Nd4jPointer) maxDimension, 1);

					// divide 3
					execBroadcastFloat(extraPointers, 3, dx, xShapeInfo, (Nd4jPointer) special,
									   (Nd4jPointer) maxShapeBuffer, dx, xShapeInfo, (Nd4jPointer) dimension, 1);

					// log 3
					if (opNum == 40)
						execTransformFloat(extraPointers, 5, dx, xShapeInfo, dx, xShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformFloat(extraPointers, 42, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					break;
				}
				case 41: {
					// IsMax along all dimensions
					if (extraParamsPointer == nullptr) {
						int maxIdx = (int) execIndexReduceScalarFloat(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxFloat<<< 256, 256, 0, *stream >>>(resultPointer, shape::length(hostXShapeInfo), targetIdx);
					} else {
						// going for dimension-based IsMax
					}
					break;
				}
				default: {
					printf("Bad case for transformFloat\n");
					break;
				}
			}
		}
	} else {
		transformFloat <<<launchDims.x, launchDims.y, launchDims.z, *stream>>> (
				opNum,
				xPointer,
				xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
				extraParamsPointer,
				resultPointer, resultShapeInfoPointer,  shape::rank(resultShapeInfoPointer), allocPointer, reductionPointer);
	}

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}

/**
 *
 * @param opNum
 * @param dx
 * @param xShapeInfo
 * @param result
 * @param resultShapeInfo
 * @param extraParams
 * @param n
 */
void   NativeOps::execTransformFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		Nd4jPointer dx,
		Nd4jPointer xShapeInfo,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer extraParams,
		Nd4jPointer xIndexes,
		Nd4jPointer resultIndexes) {
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("F21 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[0], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams((int) extraPointers[2], (int *) extraPointers[0], nullptr);

	transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,  shape::rank(xShapeInfoPointer),
			extraParamsPointer,
			resultPointer,
			resultIndexesPointer, allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));


}


template <typename T>
__device__ void flattenKernelGeneric(int dOffset,
					char order,
					T *result,
					int *resultShapeInfo,
					T *input,
					int *inputShapeInfo, int *allocationPointer) {

	__shared__ UnifiedSharedMemory<T> *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory<T>();
		manager->init(sizeof(UnifiedSharedMemory<T>), 4, 4, sizeof(shape::TAD));
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int *zShape = shape::shapeOf(resultShapeInfo);
	int *zStride = shape::stride(resultShapeInfo);


	int *yShape = shape::shapeOf(inputShapeInfo);
	int *yStride = shape::stride(inputShapeInfo);
	char yOrder = shape::order(inputShapeInfo);

	int len = shape::length(inputShapeInfo);

	int resultEWS = shape::elementWiseStride(resultShapeInfo);
	int inputEWS = shape::elementWiseStride(inputShapeInfo);
	if (yOrder == order) {
		if (resultEWS >= 1 && inputEWS >= 1) {
			for (int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				result[i * resultEWS + dOffset] = input[i * inputEWS];
			}
		} else {

			int rank = shape::rank(inputShapeInfo);
			/*
			long allocSize = sizeof(int) * rank;
			int *coord = shape::cuMalloc(allocationPointer, allocSize, manager);
			 */
			int coord[MAX_RANK];

			if(order == 'f') {
				for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
					shape::ind2sub(rank,yShape,i,coord);
					int offset = shape::getOffset(0,yShape,yStride,coord,rank);
					result[i + dOffset] = input[offset];
				}
			}
			else {
				for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
					shape::ind2subC(rank,yShape,i,coord);
					int offset = shape::getOffset(0,yShape,yStride,coord,rank);
					result[i + dOffset] = input[offset];
				}
			}
/*
			if (rank > MAX_COORD && tid * allocSize > PREALLOC_SIZE - allocSize) {
				free(coord);
			}
			*/
		}
	} else {
		int rank = shape::rank(inputShapeInfo);
		/*
		long allocSize = sizeof(int) * rank;
		int *coord = shape::cuMalloc(allocationPointer, allocSize, manager);
		 */
		int coord[MAX_RANK];

		if(order == 'f') {
			for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				shape::ind2sub(rank,yShape,i,coord);
				int offset = shape::getOffset(0,yShape,yStride,coord,rank);
				result[i+dOffset] = input[offset];
			}
		}
		else {
			for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				shape::ind2subC(rank,yShape,i,coord);
				int offset = shape::getOffset(0,yShape,yStride,coord,rank);
				result[i+dOffset] = input[offset];
			}
		}
		/*
		if (rank > MAX_COORD && tid * allocSize > PREALLOC_SIZE - allocSize) {
			free(coord);
		}*/
	}

}

extern "C" __global__ void flattenKernelDouble(int offset,
											  char order,
											  double *result,
											  int *resultShapeInfo,
											  double *input,
											  int *inputShapeInfo, int *allocationPointer) {
	flattenKernelGeneric<double>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocationPointer);
}

extern "C" __global__ void flattenKernelFloat(int offset,
											  char order,
											  float *result,
											  int *resultShapeInfo,
											  float *input,
											  int *inputShapeInfo, int *allocationPointer) {

	flattenKernelGeneric<float>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocationPointer);
}

/**
 * Append an input array
 * to the end of a flat array
 * in a particular order
 * @param offset the offset of the array to start at
 * @param order the order
 * @param result the result array
 * @param resultShapeInfo the shape info for te array
 * @param input the input for the array
 * @param inputShapeInfo the shape information for that array
 */
void NativeOps::flattenFloat(
		Nd4jPointer *extraPointers,
		int offset,
		char order,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer input,
		Nd4jPointer inputShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(result);
	int *xShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *yPointer = reinterpret_cast<float *>(input);
	int *yShapeInfoPointer = reinterpret_cast<int *>(inputShapeInfo);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);

	if (debug && verbose)
		printf("F22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length(yShapeInfoPointer), 2);

	flattenKernelFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, xPointer, xShapeInfoPointer, yPointer, yShapeInfoPointer, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}
/**
 * Append an input array
 * to the end of a flat array
 * in a particular order
 * @param offset the offset of the array to start at
 * @param order the order
 * @param result the result array
 * @param resultShapeInfo the shape info for te array
 * @param input the input for the array
 * @param inputShapeInfo the shape information for that array
 */
void NativeOps::flattenDouble(
		Nd4jPointer *extraPointers,
		int offset,
		char order,
		Nd4jPointer result,
		Nd4jPointer resultShapeInfo,
		Nd4jPointer input,
		Nd4jPointer inputShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(result);
	int *xShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *yPointer = reinterpret_cast<double *>(input);
	int *yShapeInfoPointer = reinterpret_cast<int *>(inputShapeInfo);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D30 opNum:[7]\n");

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams((int) extraPointers[2], shape::length(yShapeInfoPointer), 2);

	flattenKernelDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, xPointer, xShapeInfoPointer, yPointer, yShapeInfoPointer, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::initializeDevicesAndFunctions() {
	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);
	deviceProperties = new cudaDeviceProp[devCnt];
	for (int i = 0; i < devCnt; i++) {
		cudaSetDevice(i);
		cudaGetDeviceProperties(&deviceProperties[i], i);

		cudaDeviceSetLimit(cudaLimitStackSize, 10000);
		cudaDeviceSetLimit(cudaLimitMallocHeapSize , 10000);
	}

	cudaSetDevice(0);

	cudaFuncGetAttributes(&funcAttributes[0], (void *)transformFloatIndexes);

	void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocationPointer, float *reductionPointer) = transformFloat;
	cudaFuncGetAttributes(&funcAttributes[1], transformFloatPointer1);

	void (*transformFloatPointer2)(int opNum, Nd4jIndex n, float *dy, int incy, float *params, float *result,int resultStride, int *allocationPointer, float *reductionPointer) = transformFloat;
	cudaFuncGetAttributes(&funcAttributes[2], transformFloatPointer2);

	cudaFuncGetAttributes(&funcAttributes[3], (void *)summaryStatsReduceFloat);

	cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

	void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarFloat;
	cudaFuncGetAttributes(&funcAttributes[5], scalarFloatPointer1);

	void (*scalarFloatPointer2)(int opNum, Nd4jIndex n,float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocPointer) = scalarFloat;
	cudaFuncGetAttributes(&funcAttributes[6], scalarFloatPointer2);

	cudaFuncGetAttributes(&funcAttributes[7], reduce3Float);

	cudaFuncGetAttributes(&funcAttributes[8], reduceFloat);

	cudaFuncGetAttributes(&funcAttributes[9], pairWiseTransformFloat);

	cudaFuncGetAttributes(&funcAttributes[10], pairWiseTransformFloatIndex);

	cudaFuncGetAttributes(&funcAttributes[11], pairWiseTransformStridedFloat);

	cudaFuncGetAttributes(&funcAttributes[12], broadcastFloat);

	cudaFuncGetAttributes(&funcAttributes[13], indexReduceFloat);

	///////////////////////////////////////// Doubles are separate, just in case of...

	cudaFuncGetAttributes(&funcAttributes[14], transformDoubleIndexes);

	void (*transformDoublePointer1)(int opNum, double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocationPointer, double *reductionPointer) = transformDouble;
	cudaFuncGetAttributes(&funcAttributes[15], transformDoublePointer1);

	void (*transformDoublePointer2)(int opNum, Nd4jIndex n, double *dy, int incy, double *params, double *result,int resultStride, int *allocationPointer, double *reductionPointer) = transformDouble;
	cudaFuncGetAttributes(&funcAttributes[16], transformDoublePointer2);

	cudaFuncGetAttributes(&funcAttributes[17], summaryStatsReduceDouble);

	cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

	void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarDouble;
	cudaFuncGetAttributes(&funcAttributes[19], scalarDoublePointer1);


	void (*scalarDoublePointer2)(int opNum, Nd4jIndex n,double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocPointer) = scalarDouble;
	cudaFuncGetAttributes(&funcAttributes[20], scalarDoublePointer2);

	cudaFuncGetAttributes(&funcAttributes[21], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[22], reduceDouble);

	cudaFuncGetAttributes(&funcAttributes[23], pairWiseTransformDouble);

	cudaFuncGetAttributes(&funcAttributes[24], pairWiseTransformDoubleIndex);

	cudaFuncGetAttributes(&funcAttributes[25], pairWiseTransformStridedDouble);

	cudaFuncGetAttributes(&funcAttributes[26], broadcastDouble);

	cudaFuncGetAttributes(&funcAttributes[27], indexReduceDouble);
}


/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocHost(long memorySize, int flags) {
	Nd4jPointer pointer;
	cudaError_t res = cudaHostAlloc((void **)&pointer, memorySize, cudaHostAllocMapped |cudaHostAllocPortable );
	if (res != 0)
		pointer = 0L;
	return pointer;
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocDevice(long memorySize, Nd4jPointer ptrToDeviceId, int flags) {
	Nd4jPointer pointer;
	cudaError_t res = cudaMalloc((void **)&pointer, memorySize);
	if (res != 0)
		pointer = 0L;
	return pointer;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
Nd4jPointer NativeOps::freeHost(Nd4jPointer pointer) {
	cudaError_t res = cudaFreeHost((void *) pointer);
	if (res != 0)
		pointer = 0L;
	return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
Nd4jPointer NativeOps::freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId) {
	cudaError_t res = cudaFree((void *)pointer);
	if (res != 0)
		pointer = 0L;
	return 1L;
}


Nd4jPointer NativeOps::createContext() {
	return 0L;
}

Nd4jPointer NativeOps::createStream() {
	Nd4jPointer nativeStream = 0;
	cudaError_t result = cudaStreamCreate((cudaStream_t *) &nativeStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return nativeStream;
}

Nd4jPointer NativeOps::createEvent() {
	Nd4jPointer nativeEvent= 0;
	cudaError_t result = cudaEventCreateWithFlags((cudaEvent_t *) &nativeEvent, cudaEventBlockingSync | cudaEventDisableTiming);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return nativeEvent;
}

Nd4jPointer NativeOps::createBlasHandle() {
	Nd4jPointer nativeHandle= 0;
	cublasStatus_t result = cublasCreate((cublasHandle_t *) &nativeHandle);
	if (result != 0)
		return 0L;
	else return nativeHandle;
}

Nd4jPointer NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaEventRecord(*pEvent, *pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

Nd4jPointer NativeOps::setBlasStream(Nd4jPointer handle, Nd4jPointer stream) {
	cublasHandle_t *pHandle = reinterpret_cast<cublasHandle_t *>(&handle);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cublasStatus_t result = cublasSetStream(*pHandle, *pStream);
	if (result != 0)
		return 0L;
	else return 1L;
}

Nd4jPointer NativeOps::setDevice(Nd4jPointer ptrToDeviceId) {
	int deviceId = (int) ptrToDeviceId;
	cudaError_t result = cudaSetDevice(deviceId);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

long NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
	int device = (int) ptrToDeviceId;

	if (device >= 0) {
		setDevice(ptrToDeviceId);
	}
	size_t memFree = 0;
	size_t memTotal = 0;

	cudaMemGetInfo(&memFree, &memTotal);

	return (long) memFree;
}

Nd4jPointer NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, long size, int flags, Nd4jPointer reserved) {

	return memcpyAsync(dst, src, size, flags, reserved);
}

Nd4jPointer NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, long size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaMemcpyKind 	kind;

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*pStream));

	switch (flags) {
		case 0: {
				kind = cudaMemcpyHostToHost;
			}
			break;
		case 1: {
				kind = cudaMemcpyHostToDevice;
			}
			break;
		case 2: {
				kind = cudaMemcpyDeviceToHost;
			}
		case 3: {
			kind = cudaMemcpyDeviceToDevice;
		}
			break;
	}

	cudaError_t result = cudaMemcpyAsync((void *) dst, (const void *) src, (size_t) size, kind, *pStream);
	checkCudaErrors(result);
	if (result != 0) {
		printf("Failed on [%lu] -> [%lu], size: [%i], direction: [%i]\n", src, dst, size, flags );
		return 0L;
	}
	else return 1;
}

Nd4jPointer NativeOps::memset(Nd4jPointer dst, int value, long size, int flags, Nd4jPointer reserved) {
	//cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t result = cudaMemset((void *) dst, value, (size_t) size);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

Nd4jPointer NativeOps::memsetAsync(Nd4jPointer dst, int value, long size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t result = cudaMemsetAsync((void *) dst, value, (size_t) size, *pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

Nd4jPointer NativeOps::destroyEvent(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaError_t result = cudaEventDestroy(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

Nd4jPointer NativeOps::streamSynchronize(Nd4jPointer stream) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaStreamSynchronize(*pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1L;
}

Nd4jPointer NativeOps::eventSynchronize(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);

	cudaError_t result = cudaEventSynchronize(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1L;
}

Nd4jPointer NativeOps::getAvailableDevices() {
	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);
	return (Nd4jPointer) devCnt;
}

void NativeOps::enableDebugMode(bool reallyEnable) {
	debug = reallyEnable;
}

void NativeOps::setGridLimit(int gridSize) {
	blockLimit = gridSize;
}


int NativeOps::ompGetNumThreads() {
	return maxThreads;
}

void NativeOps::setOmpNumThreads(int threads) {
	maxThreads = threads;
}
