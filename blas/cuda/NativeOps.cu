
int tad_threshold = 1;
int element_threshold = 32;

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
#include <stdlib.h>
#include <type_conversions.h>
#include <op_boilerplate.h>
#include <grid.h>
#include <aggregates.h>
//#include <sys/time.h>



cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[64];
int blockLimit = 128;
int maxThreads = 512;
bool debug = false;
bool verbose = true;
bool allowedP2P = false;
bool supportedP2P = false;
#ifdef __EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif

int minThreads = 32;

__constant__ char deviceConstantMemory[49152];

typedef struct {
    long streamId;
    long callId;
} __syncInfo;

typedef __syncInfo SyncInfo;

void CUDART_CB syncCallback(cudaStream_t stream, cudaError_t status, void *data){
    SyncInfo *sync = (SyncInfo *) data;

    printf("Finished stream: [%i], kernel call: [%i]\n", sync->streamId, sync->callId);
}

int getDeviceId(Nd4jPointer ptrToDeviceId) {
    return (int)(Nd4jIndex)ptrToDeviceId;
}

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

int getBaseMemorySize(int xRank, cudaFuncAttributes funcAttr) {
	int memory_limit = 256; //funcAttr.sharedSizeBytes;

	// TODO: remove this later
	memory_limit += sizeof(UnifiedSharedMemory) + 32; // sizeof(shape::TAD) + (xRank * 4 * 4)
/*
	if (xRank == 0) xRank = 2;

	memory_limit += (xRank * 2 + 4) * 3 * 4; // we reserve memory for xShape + T1/T2 shapes
	memory_limit += yRank == 0 ? 0 : (yRank * 2 + 4) * 4;
	memory_limit += zRank == 0 ? 0 : (zRank * 2 + 4) * 4;
	memory_limit += (xRank * 4) * 6;
	memory_limit += MAX_RANK * 4; // special case, needed roughtly in one pase
*/
	return memory_limit;
}

int getDeviceBlockThreshold(int deviceId) {
	int ccMinor = deviceProperties[deviceId].minor;
	int ccMajor = deviceProperties[deviceId].major;

	int blockThreshold = 8;

	if (ccMajor >= 5)
		blockThreshold = 32;
	else if (ccMajor == 3)
		blockThreshold = 16;
	else if (ccMajor < 3)
		blockThreshold = 8;

	return blockThreshold;
}

dim3 getBasicLaunchParams(int deviceId, long problemLength, int sharedMemoryPerThread, cudaFuncAttributes funcAttr) {
	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int blockThreshold = getDeviceBlockThreshold(deviceId);

	int num_threads = problemLength / (countMP * blockThreshold);
    num_threads = nd4j::math::nd4j_min<int>(num_threads, maxThreads);
    num_threads = nd4j::math::nd4j_max<int>(num_threads, 64);
    num_threads = nd4j::math::nd4j_max<int>(num_threads, minThreads);

	int num_blocks = nd4j::math::nd4j_max<int>(problemLength / num_threads, 1);
    num_blocks = nd4j::math::nd4j_min<int>(num_blocks, blockLimit);

	int memory_limit = (sharedMemoryPerThread * num_threads) + getBaseMemorySize(1, funcAttr);

	dim3 launchDims = dim3(num_blocks, num_threads, memory_limit);

	if (debug && verbose)
		printf("Preliminary basic launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i]\n", num_blocks, num_threads, memory_limit);


	return launchDims;
}

int getDeviceSharedThreshold(int deviceId) {
	int ccMinor = deviceProperties[deviceId].minor;
	int ccMajor = deviceProperties[deviceId].major;

	// please note threshold isn't multiple of 32, and that's NOT a mistake

	int shmemThreshold;
	if (ccMajor == 6 && ccMinor == 0)
		shmemThreshold = 65536;
	else if (ccMajor == 6 && ccMinor == 1)
		shmemThreshold = 98304 / 0.3;
	else if (ccMajor == 5 && ccMinor == 2)
		shmemThreshold = 98304;
	else if (ccMajor == 5)
		shmemThreshold = 65536;
	else if (ccMajor == 3 && ccMinor == 7)
		shmemThreshold = 114688;
	else shmemThreshold = 49152;

	return shmemThreshold;
}


dim3 getBetterDimensions(int deviceId, int numTads, int tadLength, int xRank, cudaFuncAttributes funcAttr, int dimensionLength, int elementSize, int reduction) {

	int num_threads = nd4j::math::nd4j_min<int>(tadLength, maxThreads);



	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;
	int warpSize = deviceProperties[deviceId].warpSize;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	// round num_threads to nearest warpSize
	num_threads -= num_threads % warpSize;

	num_threads = nd4j::math::nd4j_max<int>(1, num_threads);
    if (num_threads < warpSize && tadLength < warpSize)
        num_threads = tadLength;

	// since we use shared memory as fast memory for some cases - we need to count that in
	int memory_limit = getBaseMemorySize(xRank, funcAttr);
	int memory_floor = memory_limit;
	int effective_block_limit =  countMP * blockThreshold;

	int num_blocks =  numTads; //nd4j::math::nd4j_min<int>(numTads, effective_block_limit);

	int desiredShared = shmemThreshold / nd4j::math::nd4j_max<int>((num_blocks / countMP), 1);

	if (debug && verbose)
		printf("Launch context: numBlocks: [%i], numThreads: [%i], countMap: [%i], shmemThreshold: [%i], desiredShared: [%i], elementSize: [%i]\n", num_blocks, num_threads, countMP, shmemThreshold, desiredShared, elementSize);

	// at this moment we've stored all required information for things. time to count in reduction multipliers
	int reduction_per_block = 0;
	bool found = false;
	if (reduction > 0)
		while (!found) {
			reduction_per_block = (num_threads * elementSize * reduction);
			if (memory_limit + reduction_per_block < desiredShared) {
				memory_limit += reduction_per_block;
				found = true;
			} else {
				if (num_threads > minThreads) {
					num_threads -= 32;
				} else {
					memory_limit += reduction_per_block;
					found = true;
				}
			}
		}

	// at this moment we know total memory used per block, and we also know per-mp limit.
	int max_active_blocks = shmemThreshold / nd4j::math::nd4j_max<int>(memory_limit, 1);

	if (debug && verbose)
		printf("MAB: [%i], memory_floor: [%i], memory_limit: [%i], reductionPerBlock: [%i]\n", max_active_blocks, memory_floor, memory_limit, reduction_per_block);

	// we don't want to spawn more blocks, that gpu can actually handle without queue

	//num_blocks = nd4j::math::nd4j_min<int>(num_blocks, max_active_blocks);
	num_blocks = nd4j::math::nd4j_min<int>(num_blocks, blockLimit);

//	if (num_blocks > countMP)
//    	num_blocks = num_blocks - (num_blocks % countMP);

	num_blocks = nd4j::math::nd4j_max<int>(num_blocks, 1);

	int targetBlocksPerMP = num_blocks / countMP;

	// now we know desired number of blocks wrt to shared memory. So, now we should take in account number of threads per SM
	if (targetBlocksPerMP * num_threads > 2048) {
		while (targetBlocksPerMP * num_threads > 2048) {
			if (num_threads <= minThreads)
				break;

			num_threads -= 32;
		}

		reduction_per_block = (num_threads * elementSize * reduction);
		memory_limit = memory_floor + reduction_per_block;
	}




	if (debug && verbose)
		printf("Preliminary reduce launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], reduction_per_block: [%i], blocksPerMP: [%i]\n", num_blocks, num_threads, memory_limit, reduction_per_block, targetBlocksPerMP);

	return dim3(num_blocks,num_threads, memory_limit);
}


dim3 getFlatLaunchParams(int deviceId, int *xShapeInfo, int *yShapeInfo, cudaFuncAttributes funcAttr) {
	int xRank = shape::rank(xShapeInfo);
	int yRank = yShapeInfo == nullptr ? 0 : shape::rank(yShapeInfo);
	int zRank = 0;

	int memory_limit = getBaseMemorySize(xRank, funcAttr);

	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	int xLength = shape::length(xShapeInfo);
	int effective_block_limit =  countMP * blockThreshold;

	// for flat calls we just want as much concurrent blocks, as possible, and we're not tied to TAD here
	int num_threads = xLength / effective_block_limit;
	if (num_threads < minThreads)
		num_threads = minThreads;

	num_threads = num_threads - (num_threads % 32);

	int memory_floor = memory_limit;

	int num_blocks = xLength / num_threads;
	num_blocks = nd4j::math::nd4j_min<int>(num_blocks, blockLimit);
//	num_blocks = nd4j::math::nd4j_min<int>(num_blocks, effective_block_limit);
	num_blocks = nd4j::math::nd4j_max<int>(num_blocks, 1);

	int targetBlocksPerMP = num_blocks / countMP;

	// now we know desired number of blocks wrt to shared memory. So, now we should take in account number of threads per SM
	if (targetBlocksPerMP * num_threads > 2048 && num_threads >= 128) {
		while (targetBlocksPerMP * num_threads > 2048) {
			if (num_threads <= minThreads)
				break;
			num_threads -= 32;
		}
	}

    if (xLength / num_threads > blockLimit)
        num_blocks *= 2;

	dim3 launchDims = dim3(num_blocks, num_threads, memory_limit);

	if (debug && verbose)
		printf("Preliminary scalar launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], blocksPerMP: [%i], problemLength: [%i], effectiveBlockLimit: [%i]\n", num_blocks, num_threads, memory_limit, targetBlocksPerMP, xLength, effective_block_limit);


	return launchDims;
}

dim3 getReduceLaunchParams(int deviceId, int *xShapeInfo, int *tadShapeInfo, cudaFuncAttributes funcAttr, int dimensionLength, int elementSize, int reductionSize) {

	int tadLength = 0;
	int numTads = 0;
	if (tadShapeInfo != nullptr) {
		tadLength = shape::length(tadShapeInfo);
		numTads = shape::length(xShapeInfo) / tadLength;

		if (tadLength == 1) {
			if (debug && verbose)
				printf("A xLength: [%i], zLength: [%i]\n", shape::length(xShapeInfo), shape::length(tadShapeInfo));
		}
	} else{
		// we have special case - reduction along all dimensions
		tadLength = nd4j::math::nd4j_min<int>(shape::length(xShapeInfo), 768);
		numTads = shape::length(xShapeInfo) / tadLength;
	}

	int xRank = shape::rank(xShapeInfo);
	int zRank = tadShapeInfo == nullptr ? 0 : shape::rank(tadShapeInfo);

	dim3 launchDims = getBetterDimensions(deviceId, numTads, tadLength, xRank, funcAttr, dimensionLength, elementSize, reductionSize);

	if ((debug && verbose ) ) { //|| launchDims.x == 1
		printf("Reduce LaunchParams: xLength: [%i], numTads: [%i], tadLength: [%i], launchDims.x: [%i], launchDims.y: [%i], launchDims.z: [%i]\n", shape::length(xShapeInfo), numTads, tadLength, launchDims.x, launchDims.y, launchDims.z);
	}

	return launchDims;
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

	if (debug && verbose)
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
		double *x,
		int *xShapeInfo,
		double *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D1 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[27], deviceProperties[getDeviceId(extraPointers[2])]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], 1, sizeof(double), 2);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension, int dimensionLength) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D2 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], dimensionLength, sizeof(double), 2);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		double *x,
		int *xShapeInfo,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo,
		int *dimension, int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	int *deviceTADOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);


	if (debug && verbose)
		printf("D3 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[26], deviceProperties[getDeviceId(extraPointers[2])]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[26],  dimensionLength, sizeof(double), 2);

    DISPATCH_SIMPLE(broadcastSimple, double, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ), OPS_A(BROADCAST_OPS))

/*
	broadcastDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			y,
			yShapeInfo, shape::rank(hostYShapeInfo),
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);
*/
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
		double *dx,
		int xStride,
		double *y,
		int yStride,
		double *result,
		int resultStride,
		double *extraParams, Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("D4 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[25], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[25]);

	pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>> (
			opNum,
			n,
			dx,
			y,
			xStride,
			yStride,
			extraParams,
			result,
			resultStride, allocationPointer, deviceTADShapeInfo);

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
		double *dx,
		int *xShapeInfo,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo,
		double *extraParams,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("D5 opNum:[%i]\n", opNum);


	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[24]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformDoubleIndex <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
			opNum,
			dx,
			y,
			extraParams,
			result,
			xShapeInfo, shape::rank(hostXShapeInfo),
			yShapeInfo, shape::rank(hostYShapeInfo),
			resultShapeInfo, shape::rank(hostZShapeInfo),
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, deviceTADShapeInfo);

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
		double *dx,
		int *xShapeInfo,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo,
		double *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D6 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[23]);

	pairWiseTransformDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			dx,
			y,
			extraParams,
			result,
			xShapeInfo,  shape::rank(hostXShapeInfo),
			yShapeInfo,  shape::rank(hostYShapeInfo),
			resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);

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
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D7 opNum:[%i]\n", opNum);


	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], 1, sizeof(double), 1);

    DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))
/*
	reduceScalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			nullptr,
			1,
			reductionPointer, deviceTADShapeInfo);
*/
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
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D8 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[getDeviceId(extraPointers[2])]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);



	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(double), 2);

        DISPATCH_SIMPLE(reduceSimpleGeneric1D, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))

        /*
		reduceDouble1D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
*/
	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[33], dimensionLength, sizeof(double), 2);

        DISPATCH_SIMPLE(reduceSimpleGeneric3D, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
        /*
		reduceDouble6D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
*/
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], dimensionLength, sizeof(double), 2);

        DISPATCH_SIMPLE(reduceSimpleGenericXD, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
        /*
		reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
         */
	}

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
		double *x,
		int *xShapeInfo,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("D9 opNum:[%i]\n", opNum);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], 1, sizeof(double), 1);

    DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

    /*
	reduceScalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			extraParams,
			resultPointer,
			nullptr,
			nullptr,
			1,
			reductionPointer, deviceTADShapeInfo);
*/
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
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D10 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[21]);

	reduce3Double<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			nullptr,
			1,
			1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *y,
		int *yShapeInfo){
	if (debug && verbose)
		printf("D11 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

    double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[21]);

	reduce3ScalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo,
					y,
					yShapeInfo,
					extraParams,
					resultPointer,
					nullptr,
					nullptr,
					1,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Double(
		Nd4jPointer *extraPointers,
		int opNum,
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D12 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo, resultShapeInfo, dimensionLength, sizeof(double), 2);
	//dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[21]);

	reduce3Double<<<1,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		double *x,
		int xStride,
		double *result,
		int resultStride,
		double scalar,
		double *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D13 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[20], deviceProperties[getDeviceId(extraPointers[2])]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[20]);

    DISPATCH_SIMPLE(scalarSimpleStrided, double, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))

    /*
	scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			xStride,
			extraParams,
			result,resultStride, allocPointer);
*/
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
		double *x,
		int *xShapeInfo,
		double *result,
		int *resultShapeInfo,
		double scalar,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (debug && verbose)
		printf("D14 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[19], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[19]);

    DISPATCH_SIMPLE(scalarSimpleShaped, double, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))

    /*
	scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			scalar,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer);*/

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
		double *x,
		int *xShapeInfo,
		double *result,
		int *resultShapeInfo,
		double scalar,
		double *extraParams,
		Nd4jIndex n,
		int *xIndexes,
		int *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (debug && verbose)
		printf("D15 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[18], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[18]);

	scalarDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			extraParams,
			result,
			resultIndexes, allocPointer);

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
		double *x,
		int *xShapeInfo,
		double *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D16 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[getDeviceId(extraPointers[2])]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[17], 1, sizeof(double), 8);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,bool biasCorrected) {
	if (debug && verbose)
		printf("D17 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[getDeviceId(extraPointers[2])]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[17], 1, sizeof(double), 8);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsDouble(
		Nd4jPointer *extraPointers,
		int opNum,
		double *x,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension, int dimensionLength,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("D18 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[17], dimensionLength, sizeof(double), 8);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		double *dx,
		int xStride,
		double *z,
		int zStride,
		double *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("D19 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[16]);

    DISPATCH_SIMPLE(transformStrided, double, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

    /*
	transformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			dx,
			xStride,
			extraParams,
			result,resultStride, allocPointer, reductionPointer);
    */

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
		double *dx,
		int *xShapeInfo,
		double *result,
		int *resultShapeInfo,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (debug && verbose)
		printf("D20 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);
    int *maskedAllocPointer = allocPointer;

	// special pointer for special buffer for special ops
	double *specialPointer = reinterpret_cast<double *>(extraPointers[6]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	double * special = (double *) maxShapeBuffer + (MAX_RANK * 2 + 4);

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(256, length);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(double) * 4);

            DISPATCH_SIMPLE(transformShaped, double, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))


            /*
			transformDouble<<< 1, block,launchDims.z + (block * sizeof(double) * 8), *stream >>> (
					opNum,
							dx,
							xShapeInfo,  shape::rank(hostXShapeInfo),
							extraParams,
							result, resultShapeInfo,  shape::rank(hostZShapeInfo), allocPointer, reductionPointer);*/
		} else {
			// going for blockwise specials
			//float *xpf = reinterpret_cast<float *>(dx);

			int *shape = shape::shapeOf(hostXShapeInfo);
			//printf("Rows num: %i\n", shape[0]);
			switch (opNum) {
				case 40: // LogSoftMax
				case 39: // SoftMax Derivative
				case 38: {// softmax
					Nd4jPointer tempPointers[16];
					tempPointers[0] = extraPointers[0];
					tempPointers[1] = extraPointers[1];
					tempPointers[2] = extraPointers[2];
					tempPointers[3] = extraPointers[3];
					tempPointers[4] = extraPointers[4];
					tempPointers[5] = extraPointers[5];
					tempPointers[6] = extraPointers[6];
					tempPointers[7] = extraPointers[7];
					tempPointers[8] = extraPointers[8];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
					tempPointers[12] = extraPointers[12];
					tempPointers[13] = extraPointers[13];
					tempPointers[14] = extraPointers[14];
					tempPointers[15] = extraPointers[15];


					int maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					int *hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);
					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					prepareShapeBuffer << < 1, 1, 128, *stream >> > (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceDouble(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// sub 1
					execBroadcastDouble(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

					// exp 3
					execTransformDouble(extraPointers, 3, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceDouble(tempPointers, 1, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// divide 3
					execBroadcastDouble(tempPointers, 3, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

					// log 3
					if (opNum == 40)
						execTransformDouble(extraPointers, 5, dx, xShapeInfo, dx, xShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformDouble(extraPointers, 42, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions
					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					} else {
						//extraParams == nullptr || (shape::isVector(hostXShapeInfo))
						//if (shape::isVector(hostXShapeInfo) && extraParams[1] == 1) {
						//	scalarCheat = true;
						//}
					}

					if (scalarCheat) {
						//printf("Going for scalar IsMax\n");
						int maxIdx = (int) execIndexReduceScalarDouble(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxDouble<<< 1, 128, 0, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);
					} else {
						// going for dimension-based IsMax
						//printf("Going for dimension-based IsMax\n");

						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
						int *tadMaxOffsets = reinterpret_cast<int *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<double *>(extraPointers[17]);

						// we call for IMax on specified dimension
						execIndexReduceDouble(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, 1);

						if (debug)
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxDouble<<<blockLimit, 16, funcAttributes[37].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, 1, tadMaxOffsets );


                        checkCudaErrors(cudaStreamSynchronize(*stream));


					}
					break;
				}
				default: {
					printf("Bad case for transformDouble\n");
					break;
				}
			}
		}
	} else {
        if (opNum == 37 || opNum == 36) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 768;
        }

        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc((void **)&maskedAllocPointer, length * launchDims.x * sizeof(double));
        }

        DISPATCH_SIMPLE(transformShaped, double, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (debug || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

        if (opNum == 48) {
            cudaFree((void *)maskedAllocPointer);
        }
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
		double *dx,
		int *xShapeInfo,
		double *result,
		int *resultShapeInfo,
		double *extraParams,
		int *xIndexes,
		int *resultIndexes) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (debug && verbose)
		printf("D21 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[14], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[14]);

	transformDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultIndexes, allocPointer, reductionPointer);

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
		float *x,
		int *xShapeInfo,
		float *extraParams){
	if (debug && verbose)
		printf("F1 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[getDeviceId(extraPointers[2])]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float), 2);

	if (debug && verbose && launchDims.x == 1)
		printf("AF1 opNum:[%i]\n", opNum);

	indexReduceFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}


float   NativeOps::execIndexReduceScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams){
	if (debug && verbose)
		printf("H1 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[getDeviceId(extraPointers[2])]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float16), 2);

	if (debug && verbose && launchDims.x == 1)
		printf("AH1 opNum:[%i]\n", opNum);

	indexReduceHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					resultPointer,
					nullptr, 0,
					nullptr,
					1,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = (float) resultPointer[0];
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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F2 opNum:[%i]\n", opNum);

	// dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float), 2);

	if (verbose && launchDims.x == 1)
		printf("AF2 opNum:[%i]\n", opNum);

	indexReduceFloat<<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}

void   NativeOps::execIndexReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H2 opNum:[%i]\n", opNum);

	// dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float16), 2);

	if (verbose && launchDims.x == 1)
		printf("AH2 opNum:[%i]\n", opNum);

	indexReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultShapeInfo, shape::rank(hostZShapeInfo),
					dimension,
					dimensionLength,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		float *x,
		int *xShapeInfo,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo,
		int *dimension, int dimensionLength){
/*
    cudaEvent_t start;
    cudaEventCreateWithFlags(&start, cudaEventDisableTiming);

    timespec tsX;
    timespec tsY;
    clock_gettime(CLOCK_REALTIME, &tsX);
*/
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	int *deviceTADOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);



	if (debug && verbose)
		printf("F3 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[12], deviceProperties[getDeviceId(extraPointers[2])]);


	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float), 0);

    DISPATCH_SIMPLE(broadcastSimple, float, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ), OPS_A(BROADCAST_OPS))

/*
    SyncInfo *info = new SyncInfo();
    info->streamId = 32;
    info->callId = 1234567890;

    timespec ts1;
    timespec ts2;
    clock_gettime(CLOCK_REALTIME, &ts1);
*/
    /*
	broadcastFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			y,
			yShapeInfo, shape::rank(hostYShapeInfo),
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);
     */
/*
    clock_gettime(CLOCK_REALTIME, &ts2);

//    cudaEventRecord(start, 0);

//    cudaStreamAddCallback(*stream, syncCallback, (void*)info, 0);
*/
	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
/*
    clock_gettime(CLOCK_REALTIME, &tsY);

    printf("Execution time: %i\n", (ts2.tv_nsec - ts1.tv_nsec));
    printf("Overall time: %i\n", (tsY.tv_nsec - tsX.tv_nsec));
    printf("Callback setup time: %i\n", (tsY.tv_nsec - ts2.tv_nsec));
    printf("-------------------------------------\n");
    */
}


void   NativeOps::execBroadcastHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *y,
		int *yShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		int *dimension, int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	int *deviceTADOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);


	if (debug && verbose)
		printf("H3 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[12], deviceProperties[getDeviceId(extraPointers[2])]);


	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float16), 0);

    DISPATCH_SIMPLE(broadcastSimple, float16, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ), OPS_A(BROADCAST_OPS))

    /*
	broadcastHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					y,
					yShapeInfo, shape::rank(hostYShapeInfo),
					result,
					resultShapeInfo, shape::rank(hostZShapeInfo),
					dimension,
					dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);
     */

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
		float *dx,
		int xStride,
		float *y,
		int yStride,
		float *result,
		int resultStride,
		float *extraParams, Nd4jIndex n){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F4 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[11], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], nullptr, (int *) extraPointers[7], 1, sizeof(float), 0);
	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[11]);

	if (verbose && launchDims.x == 1)
		printf("AF4 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

	pairWiseTransformStridedFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			n,
			dx,
			y,
			xStride,
			yStride,
			extraParams,
			result,
			resultStride, allocationPointer, deviceTADShapeInfo);

	if (debug) {
        checkCudaErrors(cudaStreamSynchronize(*stream));
    }
}

void   NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		int xStride,
		float16 *y,
		int yStride,
		float16 *result,
		int resultStride,
		float16 *extraParams, Nd4jIndex n){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H4 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[11], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], nullptr, (int *) extraPointers[7], 1, sizeof(float), 0);
	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[11]);

	if (verbose && launchDims.x == 1)
		printf("AH4 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

	pairWiseTransformStridedHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
					n,
					dx,
					y,
					xStride,
					yStride,
					extraParams,
					result,
					resultStride, allocationPointer, deviceTADShapeInfo);

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
		float *dx,
		int *xShapeInfo,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo,
		float *extraParams,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F5 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[10], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[10], 1, sizeof(float), 0);

	if (verbose && launchDims.x == 1)
		printf("AF5 opNum:[%i]\n", opNum);

	pairWiseTransformFloatIndex<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			dx,
			y,
			extraParams,
			result,
			xShapeInfo, shape::rank(hostXShapeInfo),
			yShapeInfo, shape::rank(hostYShapeInfo),
			resultShapeInfo, shape::rank(hostZShapeInfo),
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, deviceTADShapeInfo);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		int *xShapeInfo,
		float16 *y,
		int *yShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		float16 *extraParams,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H5 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[10], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[10], 1, sizeof(float16), 0);

	if (verbose && launchDims.x == 1)
		printf("AH5 opNum:[%i]\n", opNum);

	pairWiseTransformHalfIndex<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
					dx,
					y,
					extraParams,
					result,
					xShapeInfo, shape::rank(hostXShapeInfo),
					yShapeInfo, shape::rank(hostYShapeInfo),
					resultShapeInfo, shape::rank(hostZShapeInfo),
					xIndexes,
					yIndexes,
					resultIndexes, allocationPointer, deviceTADShapeInfo);

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
		float *dx,
		int *xShapeInfo,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo,
		float *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F6 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[9], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], resultShapeInfo,  yShapeInfo, 1, sizeof(float), 0);
	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[9]);

	if (verbose && launchDims.x == 1) {
		printf("AF6 opNum:[%i], launchDims.x: [%i], launchDims.y: [%i]\n", opNum, launchDims.x, launchDims.y);
		shape::printShapeInfoLinear(hostXShapeInfo);
	}

	pairWiseTransformFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			dx,
			y,
			extraParams,
			result,
			xShapeInfo, shape::rank(hostXShapeInfo),
			yShapeInfo, shape::rank(hostYShapeInfo),
			resultShapeInfo, shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		int *xShapeInfo,
		float16 *y,
		int *yShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		float16 *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H6 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[9], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], resultShapeInfo,  yShapeInfo, 1, sizeof(float), 0);
	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[9]);

	if (verbose && launchDims.x == 1) {
		printf("HF6 opNum:[%i], launchDims.x: [%i], launchDims.y: [%i]\n", opNum, launchDims.x, launchDims.y);
		shape::printShapeInfoLinear(hostXShapeInfo);
	}

	pairWiseTransformHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
					dx,
					y,
					extraParams,
					result,
					xShapeInfo, shape::rank(hostXShapeInfo),
					yShapeInfo, shape::rank(hostYShapeInfo),
					resultShapeInfo, shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);

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
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F7 opNum:[%i]\n", opNum);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], 1, sizeof(float), 1);

	if (verbose && launchDims.x == 1)
		printf("AF7 opNum:[%i]\n", opNum);

    DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

    /*
	reduceScalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			nullptr,
			1,
			reductionPointer, deviceTADShapeInfo);
     */

	checkCudaErrors(cudaStreamSynchronize(*stream));
}

void   NativeOps::execReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H7 opNum:[%i]\n", opNum);

	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], 1, sizeof(float16), 1);

	if (verbose && launchDims.x == 1)
		printf("AH7 opNum:[%i]\n", opNum);

    DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

    /*
	reduceScalarHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo,
					extraParams,
					result,
					resultShapeInfo,
					nullptr,
					1,
					reductionPointer, deviceTADShapeInfo);
*/
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
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F8 opNum:[%i]\n", opNum);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);


	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float), 1);

	if (verbose && launchDims.x == 1)
		printf("AF8 opNum:[%i]\n", opNum);

	if (dimensionLength == 1) {

        DISPATCH_SIMPLE(reduceSimpleGeneric1D, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
        // DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, int *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
		/*reduceFloat1D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
						opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);*/
	} else if (shape::rank(hostTADShapeInfo) <= 3) {

        DISPATCH_SIMPLE(reduceSimpleGeneric3D, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
        /*
		reduceFloat6D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
						opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);*/
	} else {

        DISPATCH_SIMPLE(reduceSimpleGenericXD, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))

        /*
		reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
						opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
         */
	}



	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

	//delete tad;
}

void   NativeOps::execReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H8 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[getDeviceId(extraPointers[2])]);


	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float16), 1);

	if (verbose && launchDims.x == 1)
		printf("AH8 opNum:[%i]\n", opNum);

	if (dimensionLength == 1) {

        DISPATCH_SIMPLE(reduceSimpleGeneric1D, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))

        /*
		reduceHalf1D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);*/
	} else if (shape::rank(hostTADShapeInfo) <= 3) {

        DISPATCH_SIMPLE(reduceSimpleGeneric3D, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))

        /*
		reduceHalf6D<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);*/
	} else {

        DISPATCH_SIMPLE(reduceSimpleGenericXD, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
        /*
		reduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
				opNum,
						x,
						xShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
         */
	}



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
		float *x,
		int *xShapeInfo,
		float *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F9 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[getDeviceId(extraPointers[2])]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[8], 1, sizeof(float), 1);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[8]);

	if (verbose && launchDims.x == 1)
		printf("AF9 opNum:[%i]\n", opNum);

    //printf("Launch params: {x: %i, y: %i, z: %i}\n", launchDims.x,launchDims.y, launchDims.z);

	//printf("reduceScalarFloat is going to start...\n");

    DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

    /*
	reduceScalarFloat<<< launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			extraParams,
			resultPointer,
			nullptr,
			nullptr,
			1,
			reductionPointer, deviceTADShapeInfo
	);
*/
	//printf("kernel fired...\n");


	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}

float NativeOps::execReduceScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H9 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[getDeviceId(extraPointers[2])]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[8], 1, sizeof(float), 1);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 2, funcAttributes[8]);

	if (verbose && launchDims.x == 1)
		printf("AH9 opNum:[%i]\n", opNum);


    DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

    /*
	reduceScalarHalf<<< launchDims.x,launchDims.y, launchDims.z, *stream>>>(
					opNum,
					x,
					xShapeInfo,
					extraParams,
					resultPointer,
					nullptr,
					nullptr,
					1,
					reductionPointer, deviceTADShapeInfo

	);
*/

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = (float) resultPointer[0];
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
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F10 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo, resultShapeInfo, 1, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AF10 opNum:[%i]\n", opNum);

	reduce3ScalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void   NativeOps::execReduce3Half(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		int *yShapeInfo,
		float16 *result,
		int *resultShapeInfo){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H10 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo, resultShapeInfo, 1, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AH10 opNum:[%i]\n", opNum);

	reduce3ScalarHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo,
					y,
					yShapeInfo,
					extraParams,
					result,
					resultShapeInfo,
					nullptr,
					1,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *y,
		int *yShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F11 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[getDeviceId(extraPointers[2])]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo, nullptr, 1, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 32, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AF11 opNum:[%i]\n", opNum);

	reduce3ScalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			extraParams,
			resultPointer,
			nullptr,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result  = resultPointer[0];
	return result;
}

float   NativeOps::execReduce3ScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		int *yShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H11 opNum:[%i]\n", opNum);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AH11 opNum:[%i]\n", opNum);

	reduce3ScalarHalf<<<launchDims.x,launchDims.y,launchDims.z + 2048, *stream>>>(
			opNum,
					x,
					xShapeInfo,
					y,
					yShapeInfo,
					extraParams,
					resultPointer,
					nullptr,
					nullptr,
					1,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result  = (float) resultPointer[0];
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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Float(
		Nd4jPointer *extraPointers,
		int opNum,
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[getDeviceId(extraPointers[2])]);

	if (debug && verbose)
		printf("F12 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo, resultShapeInfo, dimensionLength, sizeof(float), 2);
	//dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), (int *) extraPointers[0], yShapeInfo);
	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AF12 opNum:[%i]\n", opNum);
	if (shape::isScalar(hostZShapeInfo) || dimension == nullptr) {
		reduce3ScalarFloat << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (
				opNum,
						x,
						xShapeInfo,
						y,
						yShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else {
		reduce3Float << < 1, launchDims.y, launchDims.z, *stream >> > (
				opNum,
						x,
						xShapeInfo,
						y,
						yShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets);
	}

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void   NativeOps::execReduce3Half(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		int *yShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[getDeviceId(extraPointers[2])]);

	if (debug && verbose)
		printf("H12 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[7]);

	if (verbose && launchDims.x == 1)
		printf("AH12 opNum:[%i]\n", opNum);

	if (shape::isScalar(hostZShapeInfo) || dimension == nullptr) {
		reduce3ScalarHalf<< < launchDims.x, launchDims.y, launchDims.z, *stream >> > (
				opNum,
						x,
						xShapeInfo,
						y,
						yShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else {
		reduce3Half<< < 1, launchDims.y, launchDims.z, *stream >> > (
				opNum,
						x,
						xShapeInfo,
						y,
						yShapeInfo,
						extraParams,
						result,
						resultShapeInfo,
						dimension,
						dimensionLength,
						1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets);
	}

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
		float *x,
		int xStride,
		float *result,
		int resultStride,
		float scalar,
		float *extraParams,
		Nd4jIndex n){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	if (debug && verbose)
		printf("F13 opNum:[%i]\n", opNum);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

	if (verbose && launchDims.x == 1)
		printf("AF13 opNum:[%i]\n", opNum);

    DISPATCH_SIMPLE(scalarSimpleStrided, float, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))

    /*
	scalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			xStride,
			extraParams,
			result,resultStride, allocPointer);
*/
	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


void   NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        int xStride,
        float16 *result,
        int resultStride,
        float scalar,
        float16 *extraParams,
        Nd4jIndex n){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

    if (debug && verbose)
        printf("F13 opNum:[%i]\n", opNum);

    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

    if (verbose && launchDims.x == 1)
        printf("AF13 opNum:[%i]\n", opNum);

    DISPATCH_SIMPLE(scalarSimpleStrided, float16, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))

    /*
	scalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			xStride,
			extraParams,
			result,resultStride, allocPointer);
*/
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
		float *x,
		int *xShapeInfo,
		float *result,
		int *resultShapeInfo,
		float scalar,
		float *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	Nd4jIndex n = shape::length(hostXShapeInfo);

	if (debug && verbose)
		printf("F14 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[5]);

	if (verbose && launchDims.x == 1)
		printf("AF14 opNum:[%i], xLength:[%i]\n", opNum, shape::length(hostXShapeInfo));

    DISPATCH_SIMPLE(scalarSimpleShaped, float, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))

    /*
	scalarFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
			opNum,
			scalar,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer );
*/
	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		float scalarF,
		float16 *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	Nd4jIndex n = shape::length(hostXShapeInfo);

	if (debug && verbose)
		printf("H14 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[5]);

    float16 scalar = (float16) scalarF;

	if (verbose && launchDims.x == 1)
		printf("AH14 opNum:[%i], xLength:[%i]\n", opNum, shape::length(hostXShapeInfo));

    DISPATCH_SIMPLE(scalarSimpleShaped, float16, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    /*
	scalarHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
			opNum,
					scalar,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					result,resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer );
*/
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
		float *x,
		int *xShapeInfo,
		float *result,
		int *resultShapeInfo,
		double scalar,
		float *extraParams,
		int *xIndexes,
		int *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	Nd4jIndex n = shape::length(hostXShapeInfo);

	if (debug && verbose)
		printf("F15 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[4], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[4]);

	if (verbose && launchDims.x == 1)
		printf("AF15 opNum:[%i]\n", opNum);

	scalarFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			extraParams,
			result,
			resultIndexes, allocPointer);

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
		float *x,
		int *xShapeInfo,
		float *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("F16 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], 1, sizeof(float), 8);

	if (verbose && launchDims.x == 1)
		printf("AF16 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = resultPointer[0];
	return result;
}


float   NativeOps::execSummaryStatsScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (debug && verbose)
		printf("H16 opNum:[%i]\n", opNum);

//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], 1, sizeof(float16), 8);

	if (verbose && launchDims.x == 1)
		printf("AH16 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					resultPointer,
					nullptr, 0,
					nullptr,
					1,
					1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	checkCudaErrors(cudaStreamSynchronize(*stream));

	float result = (float) resultPointer[0];
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
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F17 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], 1, sizeof(float), 8);

	if (verbose && launchDims.x == 1)
		printf("AF17 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			nullptr,
			1,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


void   NativeOps::execSummaryStatsHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H17 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], 1, sizeof(float16), 8);

	if (verbose && launchDims.x == 1)
		printf("AH17 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultShapeInfo, shape::rank(hostZShapeInfo),
					nullptr,
					1,
					1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		float *x,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("F18 opNum:[%i]\n", opNum);

	//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], dimensionLength, sizeof(float), 8);

	if (verbose && launchDims.x == 1)
		printf("AF18 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));

}


void   NativeOps::execSummaryStatsHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	int *deviceTADOffsets = reinterpret_cast<int *>(extraPointers[11]);

	if (debug && verbose)
		printf("H18 opNum:[%i]\n", opNum);

	//	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[3], dimensionLength, sizeof(float16), 8);

	if (verbose && launchDims.x == 1)
		printf("AH18 opNum:[%i]\n", opNum);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultShapeInfo, shape::rank(hostZShapeInfo),
					dimension,
					dimensionLength,
					1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		float *dx,
		int xStride,
		float *z,
		int zStride,
		float *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("F19 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[2], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (verbose && launchDims.x == 1)
		printf("AF19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

    DISPATCH_SIMPLE(transformStrided, float, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
/*
	transformFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			dx,
			xStride,
			extraParams,
			result,resultStride, allocPointer, reductionPointer);
*/
	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


void   NativeOps::execTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		int xStride,
		float16 *z,
		int zStride,
		float16 *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("H19 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[2], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (verbose && launchDims.x == 1)
		printf("AH19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

    DISPATCH_SIMPLE(transformStrided, float16, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
/*
	transformHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					n,
					dx,
					xStride,
					extraParams,
					result,resultStride, allocPointer, reductionPointer);
*/
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
		float *dx,
		int *xShapeInfo,
		float *result,
		int *resultShapeInfo,
		float *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (debug && verbose)
		printf("F20 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	float *specialPointer = reinterpret_cast<float *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float * special = (float *) maxShapeBuffer + (MAX_RANK * 2 + 4);

    int *maskedAllocPointer = allocPointer;

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	if (verbose && launchDims.x == 1)
		printf("AF20 opNum:[%i]\n", opNum);

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(length, 256);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(float) * 4);

            DISPATCH_SIMPLE(transformShaped, float, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

            /*
			transformFloat <<< 1, block, launchDims.z + (block * sizeof(float) * 4), *stream >> > (
					opNum,
					dx,
					xShapeInfo,  shape::rank(hostXShapeInfo),
					extraParams,
					result, resultShapeInfo,  shape::rank(hostZShapeInfo),  allocPointer, reductionPointer);*/
		} else {
			// going for blockwise specials
			//float *xpf = reinterpret_cast<float *>(dx);

			int *shape = shape::shapeOf(hostXShapeInfo);
			switch (opNum) {
				case 40: // LogSoftMax
				case 39: // SoftMax Derivative
				case 38: {// softmax
					Nd4jPointer tempPointers[16];
					tempPointers[0] = extraPointers[0];
					tempPointers[1] = extraPointers[1];
					tempPointers[2] = extraPointers[2];
					tempPointers[3] = extraPointers[3];
					tempPointers[4] = extraPointers[4];
					tempPointers[5] = extraPointers[5];
					tempPointers[6] = extraPointers[6];
					tempPointers[7] = extraPointers[7];
					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
					tempPointers[12] = extraPointers[12];
					tempPointers[13] = extraPointers[13];
					tempPointers[14] = extraPointers[14];
					tempPointers[15] = extraPointers[15];


					int maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					int *hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);

					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					prepareShapeBuffer <<< 1, 1, 128, *stream >>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceFloat(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastFloat(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// exp 3
					execTransformFloat(extraPointers, 3, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceFloat(tempPointers, 1, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// divide 3
					execBroadcastFloat(tempPointers, 3, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// log 3
					if (opNum == 40)
						execTransformFloat(extraPointers, 5, dx, xShapeInfo, dx, xShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformFloat(extraPointers, 42, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions

//					int *dimensionHostPointer = reinterpret_cast<int *> (extraPointers[16]);

					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					} else {
					/*	//extraParams == nullptr || (shape::isVector(hostXShapeInfo))
						if (shape::isVector(hostXShapeInfo) && dimensionHostPointer[0] == 1) {
							scalarCheat = true;
						}*/
					}

					if (scalarCheat) {
						//printf("Going for scalar IsMax\n");
						int maxIdx = (int) execIndexReduceScalarFloat(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxFloat<<< 1, 128, 1536, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);

                        checkCudaErrors(cudaStreamSynchronize(*stream));
					} else {
						// going for dimension-based IsMax
						//printf("Going for dimension-based IsMax\n");

						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
						int *tadMaxOffsets = reinterpret_cast<int *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float *>(extraPointers[17]);

						// we call for IMax on specified dimension
						execIndexReduceFloat(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, 1);

						if (debug)
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxFloat<<<blockLimit, 16, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, 1, tadMaxOffsets );


						checkCudaErrors(cudaStreamSynchronize(*stream));

					}
					break;
				}
				default: {
					printf("Bad case for transformFloat\n");
					break;
				}
			}
		}
//	} else if (opNum == 48) {
        //

    } else {
        if (opNum == 37 || opNum == 36) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 384;
        }

        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc((void **) &maskedAllocPointer, length * launchDims.x * sizeof(float));
        }

        DISPATCH_SIMPLE(transformShaped, float,
                        PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo,
                               shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (debug || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

        if (opNum == 48) {
            cudaFree((void *) maskedAllocPointer);
        }
    }

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void   NativeOps::execTransformHalf(Nd4jPointer *extraPointers,int opNum,
									 float16 *dx,
									 int *xShapeInfo,
									 float16 *result,
									 int *resultShapeInfo,
									 float16 *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (debug && verbose)
		printf("H20 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);
    int *maskedAllocPointer = allocPointer;

	// special pointer for special buffer for special ops
	float16 *specialPointer = reinterpret_cast<float16 *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float16 * special = (float16 *) maxShapeBuffer + (MAX_RANK * 2 + 4);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	if (verbose && launchDims.x == 1)
		printf("AH20 opNum:[%i]\n", opNum);

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(length, 256);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(float16) * 4);

            DISPATCH_SIMPLE(transformShaped, float16, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

		} else {
			// going for blockwise specials
			//float *xpf = reinterpret_cast<float *>(dx);

			int *shape = shape::shapeOf(hostXShapeInfo);
			switch (opNum) {
				case 40: // LogSoftMax
				case 39: // SoftMax Derivative
				case 38: {// softmax
					Nd4jPointer tempPointers[16];
					tempPointers[0] = extraPointers[0];
					tempPointers[1] = extraPointers[1];
					tempPointers[2] = extraPointers[2];
					tempPointers[3] = extraPointers[3];
					tempPointers[4] = extraPointers[4];
					tempPointers[5] = extraPointers[5];
					tempPointers[6] = extraPointers[6];
					tempPointers[7] = extraPointers[7];
					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
					tempPointers[12] = extraPointers[12];
					tempPointers[13] = extraPointers[13];
					tempPointers[14] = extraPointers[14];
					tempPointers[15] = extraPointers[15];


					int maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					int *hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);

					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					prepareShapeBuffer <<< 1, 1, 128, *stream >>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceHalf(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastHalf(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// exp 3
					execTransformHalf(extraPointers, 3, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceHalf(tempPointers, 1, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// divide 3
					execBroadcastHalf(tempPointers, 3, dx, xShapeInfo, special,
									   maxShapeBuffer, dx, xShapeInfo, dimension, 1);

                    if (opNum == 40) {
                        if (debug)
                            checkCudaErrors(cudaStreamSynchronize(*stream));

                        execTransformHalf(tempPointers, 47, dx, xShapeInfo, dx, xShapeInfo, extraParams);
                    }

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// log 3
					if (opNum == 40)
						execTransformHalf(extraPointers, 5, dx, xShapeInfo, dx, xShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformHalf(extraPointers, 42, dx, xShapeInfo, dx, xShapeInfo, extraParams);

					if (debug)
						checkCudaErrors(cudaStreamSynchronize(*stream));

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions

			//		int *dimensionHostPointer = reinterpret_cast<int *> (extraPointers[16]);

					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					} else {
						/*	//extraParams == nullptr || (shape::isVector(hostXShapeInfo))
                            if (shape::isVector(hostXShapeInfo) && dimensionHostPointer[0] == 1) {
                                scalarCheat = true;
                            }*/
					}

					if (scalarCheat) {
						//printf("Going for scalar IsMax\n");
						int maxIdx = (int) execIndexReduceScalarHalf(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxHalf<<< 1, 128, 1536, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);
					} else {
						// going for dimension-based IsMax
						//printf("Going for dimension-based IsMax\n");

						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
						int *tadMaxOffsets = reinterpret_cast<int *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float16 *>(extraPointers[17]);

						// we call for IMax on specified dimension
						execIndexReduceHalf(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, 1);

						if (debug)
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxHalf<<<blockLimit, 16, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, 1, tadMaxOffsets );


                        checkCudaErrors(cudaStreamSynchronize(*stream));

					}
					break;
				}
				default: {
					printf("Bad case for transformHalf\n");
					break;
				}
			}
		}
	} else {
        if (opNum == 37 || opNum == 36) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 384;
        }

        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc((void **)&maskedAllocPointer, length * launchDims.x * sizeof(float16));
        }

        DISPATCH_SIMPLE(transformShaped, float16, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (debug || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

        if (opNum == 48) {
            cudaFree((void *)maskedAllocPointer);
        }
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
		float *dx,
		int *xShapeInfo,
		float *result,
		int *resultShapeInfo,
		float *extraParams,
		int *xIndexes,
		int *resultIndexes) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("F21 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[0], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[0]);

	if (verbose && launchDims.x == 1)
		printf("AF21 opNum:[%i]\n", opNum);

	transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo,  shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultIndexes, allocPointer, reductionPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));


}


void   NativeOps::execTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		int *xShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		float16 *extraParams,
		int *xIndexes,
		int *resultIndexes) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (debug && verbose)
		printf("H21 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[0]);

	if (verbose && launchDims.x == 1)
		printf("AH21 opNum:[%i]\n", opNum);

	transformHalfIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					dx,
					xShapeInfo,  shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultIndexes, allocPointer, reductionPointer);

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

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 4, 4, sizeof(shape::TAD), 2);
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
	flattenKernelGeneric<double>(
			offset,
			order, result,
			resultShapeInfo,
			input,
			inputShapeInfo,
			allocationPointer);
}

extern "C" __global__ void flattenKernelFloat(int offset,
											  char order,
											  float *result,
											  int *resultShapeInfo,
											  float *input,
											  int *inputShapeInfo, int *allocationPointer) {

	flattenKernelGeneric<float>(
			offset,
			order,
			result,
			resultShapeInfo,
			input,
			inputShapeInfo,
			allocationPointer);
}

extern "C" __global__ void flattenKernelHalf(int offset,
											  char order,
											  float16 *result,
											  int *resultShapeInfo,
											  float16 *input,
											  int *inputShapeInfo, int *allocationPointer) {

	flattenKernelGeneric<float16>(
			offset,
			order,
			result,
			resultShapeInfo,
			input,
			inputShapeInfo,
			allocationPointer);
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
		float *result,
		int *resultShapeInfo,
		float *input,
		int *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[getDeviceId(extraPointers[2])]);

	if (debug && verbose)
		printf("F22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (verbose && launchDims.x == 1)
		printf("AF222 opNum:[7]\n");

	flattenKernelFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::flattenHalf(
		Nd4jPointer *extraPointers,
		int offset,
		char order,
		float16 *result,
		int *resultShapeInfo,
		float16 *input,
		int *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);

	if (debug && verbose)
		printf("H22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (verbose && launchDims.x == 1)
		printf("AH222 opNum:[7]\n");

	flattenKernelHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

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
		double *result,
		int *resultShapeInfo,
		double *input,
		int *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);

	if (debug && verbose)
		printf("D30 opNum:[7]\n");

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[getDeviceId(extraPointers[2])]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo),  2, funcAttributes[34]);

	flattenKernelDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::checkP2P() {
	int curDevice = 0;

	cudaGetDevice(&curDevice);

	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);

	if (curDevice < 0 && curDevice > devCnt)
		curDevice = 0;

	bool tempSupport = true;

	if (devCnt > 1) {
		for (int x = 0; x < devCnt; x++) {

			for (int y = 0; y < devCnt; y++) {
				if (x == y)
					continue;

				int canAccess = 0;
				cudaSetDevice(x);

				cudaDeviceCanAccessPeer(&canAccess, x , y);

				if (!canAccess)
					tempSupport = false;
			}
		}

		supportedP2P = tempSupport;

		cudaSetDevice(curDevice);
	} else {
		// if we have only 1 device - we say that we support P2P, since all data will be on 1 device
		supportedP2P = true;
	}
}

void NativeOps::enableP2P(bool enable) {
    if (enable == allowedP2P)
        return;

    int curDevice = 0;

    cudaGetDevice(&curDevice);

    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);

	if (curDevice < 0 && curDevice > devCnt)
		curDevice = 0;

    if (devCnt > 1) {
        for (int x = 0; x < devCnt; x++) {

            for (int y = 0; y < devCnt; y++) {
                if (x == y)
                    continue;

                int canAccess = 0;
                cudaSetDevice(x);

                cudaDeviceCanAccessPeer(&canAccess, x , y);

                if (canAccess) {
                    if (enable) {
                        cudaDeviceEnablePeerAccess(y, 0);
                    } else {
                        cudaDeviceDisablePeerAccess(y);
                    }
                } else {
					if (verbose) printf("Peer access [%i] -> [%i] isn't possible\n", x, y);
				}
            }
        }

        cudaSetDevice(curDevice);
    }

    allowedP2P = enable;

    cudaSetDevice(curDevice);
}

bool NativeOps::isP2PAvailable() {
	// always TRUE for cpu backend
	return supportedP2P;
}


void NativeOps::initializeDevicesAndFunctions() {
	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);
	deviceProperties = new cudaDeviceProp[devCnt];
	for (int i = 0; i < devCnt; i++) {
		cudaSetDevice(i);
		cudaGetDeviceProperties(&deviceProperties[i], i);

		cudaDeviceSetLimit(cudaLimitStackSize, 4096);
	}

	cudaSetDevice(0);

	checkP2P();

	if (supportedP2P && devCnt > 1)
    	enableP2P(allowedP2P);

	cudaFuncGetAttributes(&funcAttributes[0], (void *)transformFloatIndexes);

	//void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[1], transformFloatIndexes);

	//void (*transformFloatPointer2)(int opNum, Nd4jIndex n, float *dy, int incy, float *params, float *result,int resultStride, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[2], transformFloatIndexes);

	cudaFuncGetAttributes(&funcAttributes[3], (void *)summaryStatsReduceFloat);

	cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

//	void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarFloat;
	cudaFuncGetAttributes(&funcAttributes[5], scalarFloatIndexes);

//	void (*scalarFloatPointer2)(int opNum, Nd4jIndex n,float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocPointer) = scalarFloat;
	cudaFuncGetAttributes(&funcAttributes[6], scalarFloatIndexes);

	cudaFuncGetAttributes(&funcAttributes[7], reduce3Float);

	cudaFuncGetAttributes(&funcAttributes[8], reduceSimpleGenericXD_0_float);
//	printf("reduceFloat regs: [%i], static shmem: [%i]\n", funcAttributes[8].numRegs, funcAttributes[8].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[28], reduceSimpleGeneric1D_0_float); // 1D
//	printf("reduceFloat1D regs: [%i], static shmem: [%i]\n", funcAttributes[28].numRegs, funcAttributes[28].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[29], reduceSimpleGeneric3D_0_float); // 6D
//	printf("reduceFloat6D regs: [%i], static shmem: [%i]\n", funcAttributes[29].numRegs, funcAttributes[29].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[30], flattenKernelFloat);

	cudaFuncGetAttributes(&funcAttributes[31], concatKernelFloat);

	cudaFuncGetAttributes(&funcAttributes[9], pairWiseTransformFloat);

	cudaFuncGetAttributes(&funcAttributes[10], pairWiseTransformFloatIndex);

	cudaFuncGetAttributes(&funcAttributes[11], pairWiseTransformStridedFloat);

	cudaFuncGetAttributes(&funcAttributes[12], broadcastSimple_0_float);

	cudaFuncGetAttributes(&funcAttributes[13], indexReduceFloat);

	///////////////////////////////////////// Doubles are separate, just in case of...

	cudaFuncGetAttributes(&funcAttributes[14], transformDoubleIndexes);

//	void (*transformDoublePointer1)(int opNum, double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocationPointer, double *reductionPointer) = transformDouble;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[15], transformDoubleIndexes);

	//void (*transformDoublePointer2)(int opNum, Nd4jIndex n, double *dy, int incy, double *params, double *result,int resultStride, int *allocationPointer, double *reductionPointer) = transformDouble;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[16], transformDoubleIndexes);

	cudaFuncGetAttributes(&funcAttributes[17], summaryStatsReduceDouble);

	cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

	//void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarDouble;
	cudaFuncGetAttributes(&funcAttributes[19], scalarDoubleIndexes);


	//void (*scalarDoublePointer2)(int opNum, Nd4jIndex n,double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocPointer) = scalarDouble;
	cudaFuncGetAttributes(&funcAttributes[20], scalarDoubleIndexes);

	cudaFuncGetAttributes(&funcAttributes[21], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[22], reduceSimpleGenericXD_0_double);

	cudaFuncGetAttributes(&funcAttributes[23], pairWiseTransformDouble);

	cudaFuncGetAttributes(&funcAttributes[24], pairWiseTransformDoubleIndex);

	cudaFuncGetAttributes(&funcAttributes[25], pairWiseTransformStridedDouble);

	cudaFuncGetAttributes(&funcAttributes[26], broadcastSimple_0_double);

	cudaFuncGetAttributes(&funcAttributes[27], indexReduceDouble);

	cudaFuncGetAttributes(&funcAttributes[32], reduceSimpleGeneric1D_0_double); // 1D

	cudaFuncGetAttributes(&funcAttributes[33], reduceSimpleGeneric3D_0_double); // 6D

	cudaFuncGetAttributes(&funcAttributes[34], flattenKernelDouble);

	cudaFuncGetAttributes(&funcAttributes[35], concatKernelDouble);

	cudaFuncGetAttributes(&funcAttributes[36], fillDimensionalIsMaxFloat);

	cudaFuncGetAttributes(&funcAttributes[37], fillDimensionalIsMaxDouble);


	cudaFuncGetAttributes(&funcAttributes[38], concatKernelScalarFloat);

	cudaFuncGetAttributes(&funcAttributes[39], concatKernelScalarDouble);

	cudaFuncGetAttributes(&funcAttributes[40], concatKernelVStackFloat);

	cudaFuncGetAttributes(&funcAttributes[41], concatKernelVStackDouble);

	cudaFuncGetAttributes(&funcAttributes[42], concatKernelHStackFloat);

	cudaFuncGetAttributes(&funcAttributes[43], concatKernelHStackDouble);

    /////////////////////////

    cudaFuncGetAttributes(&funcAttributes[44], averagingKernelHalf);

    cudaFuncGetAttributes(&funcAttributes[45], averagingKernelFloat);

    cudaFuncGetAttributes(&funcAttributes[46], averagingKernelDouble);


    //

    cudaFuncGetAttributes(&funcAttributes[47], scalarAlongDimension_0_float);
    cudaFuncGetAttributes(&funcAttributes[48], scalarAlongDimension_0_float16);
    cudaFuncGetAttributes(&funcAttributes[48], scalarAlongDimension_0_double);
}


/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocHost(Nd4jIndex memorySize, int flags) {
	Nd4jPointer pointer;
	// cudaHostAllocMapped |cudaHostAllocPortable
	cudaError_t res = cudaHostAlloc((void **)&pointer, memorySize, cudaHostAllocDefault);
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
Nd4jPointer NativeOps::mallocDevice(Nd4jIndex memorySize, Nd4jPointer ptrToDeviceId, int flags) {
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
int NativeOps::freeHost(Nd4jPointer pointer) {
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
int NativeOps::freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId) {
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
	cudaError_t result = cudaEventCreateWithFlags((cudaEvent_t *) &nativeEvent, cudaEventDisableTiming);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return nativeEvent;
}

Nd4jPointer NativeOps::createBlasHandle() {
	Nd4jPointer nativeHandle= 0;
	cublasStatus_t result = cublasCreate((cublasHandle_t *) &nativeHandle);
	if (result != 0) {
        printf("cuBLAS errorCode: [%i]\n", result);
		return 0L;
    }
	else return nativeHandle;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaEventRecord(*pEvent, *pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

int NativeOps::setBlasStream(Nd4jPointer handle, Nd4jPointer stream) {
	cublasHandle_t *pHandle = reinterpret_cast<cublasHandle_t *>(&handle);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cublasStatus_t result = cublasSetStream(*pHandle, *pStream);
	if (result != 0)
		return 0L;
	else return 1L;
}

int NativeOps::setDevice(Nd4jPointer ptrToDeviceId) {
	int deviceId = getDeviceId(ptrToDeviceId);
	cudaError_t result = cudaSetDevice(deviceId);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

Nd4jIndex NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);

	if (device >= 0) {
		setDevice(ptrToDeviceId);
	}
	size_t memFree = 0;
	size_t memTotal = 0;

	cudaMemGetInfo(&memFree, &memTotal);

	return (Nd4jIndex) memFree;
}

Nd4jIndex NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);

	if (device >= 0) {
		setDevice(ptrToDeviceId);
	}
	size_t memFree = 0;
	size_t memTotal = 0;

	cudaMemGetInfo(&memFree, &memTotal);

	return (Nd4jIndex) memTotal;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {

	return memcpyAsync(dst, src, size, flags, reserved);
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
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
		default: {

			printf("UNDEFINED MEMCPY!\n");
			break;
		}
	}

	cudaError_t result = cudaMemcpyAsync((void *) dst, (const void *) src, (size_t) size, kind, *pStream);
	if (result != 0) {
        checkCudaErrors(result);
		printf("Failed on [%lu] -> [%lu], size: [%i], direction: [%i], result: [%i]\n", src, dst, size, flags, (int) result );
        fflush(stdout);
        fflush(stderr);
		return 0L;
	}
	else return 1;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jIndex size, int flags, Nd4jPointer reserved) {
	//cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t result = cudaMemset((void *) dst, value, (size_t) size);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jIndex size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t result = cudaMemsetAsync((void *) dst, value, (size_t) size, *pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

int NativeOps::destroyEvent(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaError_t result = cudaEventDestroy(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1;
}

int NativeOps::streamSynchronize(Nd4jPointer stream) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaStreamSynchronize(*pStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1L;
}

int NativeOps::eventSynchronize(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);

	cudaError_t result = cudaEventSynchronize(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return 1L;
}

int NativeOps::getAvailableDevices() {
	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);
	return devCnt;
}

void NativeOps::enableDebugMode(bool reallyEnable) {
	debug = reallyEnable;
}

void NativeOps::setGridLimit(int gridSize) {
	if (gridSize > 8192)
		gridSize = 8192;
	if (gridSize < 1)
		gridSize = 1;
	blockLimit = gridSize;
}

int NativeOps::ompGetMaxThreads() {
	return maxThreads;
}

int NativeOps::ompGetNumThreads() {
	return maxThreads;
}

void NativeOps::setOmpNumThreads(int threads) {
	if (threads > 1024)
		threads = 1024;
	if (threads < 32)
		threads = 32;
	maxThreads = threads;
}

void NativeOps::enableVerboseMode(bool reallyEnable) {
	verbose = reallyEnable;
}

int NativeOps::getDeviceMajor(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);
	return deviceProperties[device].major;
}

int NativeOps::getDeviceMinor(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);
	return deviceProperties[device].minor;
}


const char * NativeOps::getDeviceName(Nd4jPointer ptrToDeviceId) {
    int device = getDeviceId(ptrToDeviceId);

    return deviceProperties[device].name;
}

/**
  * Concatneate multi array of the same shape together
  * along a particular dimension
  */
 void NativeOps::concatFloat(
		Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float *result,
        int *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int **hostShapePointers = reinterpret_cast<int **>(extraPointers[9]);

	// numArrays will be used as number of TADs, so each block process 1 input

	int smem = 0;
	bool isVstack = false;
	bool isScalar = true;
	bool isHstack = false;

	for (int i = 0; i < numArrays; i++) {
		if (!shape::isScalar(hostShapePointers[i])) {
			isScalar = false;
			break;
		}
	}

	if (!isScalar && dimension == 0 && shape::rank(hostXShapeInfo) == 2 && shape::order(hostXShapeInfo) == 'c' ) {
		isVstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c') {
				isVstack = false;
				break;
			}
		}
	}

	if (!isScalar && !isVstack && dimension == 1 && shape::isVector(hostXShapeInfo)) {
		isHstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0) {
				isHstack = false;
				break;
			}
		}
	}

	if (isScalar) {
		if (debug && verbose)
			printf("Going scalar concat\n");

		smem = funcAttributes[38].sharedSizeBytes;
		concatKernelScalarFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (debug && verbose)
			printf("Going VStack concat\n");

		smem = funcAttributes[40].sharedSizeBytes;
		concatKernelVStackFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (debug && verbose)
			printf("Going HStack concat\n");
		smem = funcAttributes[42].sharedSizeBytes;

		concatKernelHStackFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (debug && verbose)
			printf("Going generic concat\n");

		smem = nd4j::math::nd4j_max<int>(funcAttributes[31].sharedSizeBytes + 768, 1280);

		concatKernelFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	}
	if (debug && verbose)
		printf("sharedMemory requested for concatFloat: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}



void NativeOps::concatHalf(
		Nd4jPointer *extraPointers,
		int dimension,
		int numArrays,
		Nd4jPointer *data,
		Nd4jPointer *inputShapeInfo,
		float16 *result,
		int *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int **hostShapePointers = reinterpret_cast<int **>(extraPointers[9]);

	// numArrays will be used as number of TADs, so each block process 1 input

	int smem = 0;
	bool isVstack = false;
	bool isScalar = true;
	bool isHstack = false;

	for (int i = 0; i < numArrays; i++) {
		if (!shape::isScalar(hostShapePointers[i])) {
			isScalar = false;
			break;
		}
	}

	if (!isScalar && dimension == 0 && shape::rank(hostXShapeInfo) == 2 && shape::order(hostXShapeInfo) == 'c' ) {
		isVstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c') {
				isVstack = false;
				break;
			}
		}
	}

	if (!isScalar && !isVstack && dimension == 1 && shape::isVector(hostXShapeInfo)) {
		isHstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0) {
				isHstack = false;
				break;
			}
		}
	}

	if (isScalar) {
		if (debug && verbose)
			printf("Going scalar concat\n");

		smem = funcAttributes[38].sharedSizeBytes;
		concatKernelScalarHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (debug && verbose)
			printf("Going VStack concat\n");

		smem = funcAttributes[40].sharedSizeBytes;
		concatKernelVStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (debug && verbose)
			printf("Going HStack concat\n");
		smem = funcAttributes[42].sharedSizeBytes;

		concatKernelHStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (debug && verbose)
			printf("Going generic concat\n");

		smem = nd4j::math::nd4j_max<int>(funcAttributes[31].sharedSizeBytes + 768, 1280);

		concatKernelHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	}
	if (debug && verbose)
		printf("sharedMemory requested for concatHalf: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}


/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::concatDouble(
		Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        double *result,
        int *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int **hostShapePointers = reinterpret_cast<int **>(extraPointers[9]);

	// numArrays will be used as number of TADs, so each block process 1 input

	int smem = 0;
	bool isVstack = false;
	bool isScalar = true;
	bool isHstack = false;

	for (int i = 0; i < numArrays; i++) {
		if (!shape::isScalar(hostShapePointers[i])) {
			isScalar = false;
			break;
		}
	}

	if (!isScalar && dimension == 0 && shape::rank(hostXShapeInfo) == 2 && shape::order(hostXShapeInfo) == 'c' ) {
		isVstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c') {
				isVstack = false;
				break;
			}
		}
	}

	if (!isScalar && !isVstack && dimension == 1 && shape::isVector(hostXShapeInfo)) {
		isHstack = true;
		for (int i = 0; i < numArrays; i++) {
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0) {
				isHstack = false;
				break;
			}
		}
	}

	if (isScalar) {
		if (debug && verbose)
			printf("Going scalar concat\n");

		smem = funcAttributes[39].sharedSizeBytes;
		concatKernelScalarDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (debug && verbose)
			printf("Going VStack concat\n");

		smem = funcAttributes[41].sharedSizeBytes;
		concatKernelVStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (debug && verbose)
			printf("Going HStack concat\n");
		smem = funcAttributes[43].sharedSizeBytes;

		concatKernelHStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (debug && verbose)
			printf("Going generic concat\n");

		smem = nd4j::math::nd4j_max<int>(funcAttributes[35].sharedSizeBytes + 768, 1280);

		concatKernelDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	}
	if (debug && verbose)
		printf("sharedMemory requested for concatFloat: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 * This method saves
 */
void NativeOps::tadOnlyShapeInfo(int *xShapeInfo, int *dimension, int dimensionLength, int *target, int *offsets) {
	shape::TAD *tad = new shape::TAD();
	tad->init(xShapeInfo, dimension, dimensionLength);
	//tad->setOutputBuffer(target);
	tad->createTadOnlyShapeInfo();
	tad->createOffsets();


	std::memcpy((void *) target, tad->tadOnlyShapeInfo, (tad->tadOnlyShapeInfo[0] * 2 + 4) * sizeof(int));
	std::memcpy((void *) offsets, tad->tadOffsets, tad->numTads * sizeof(int));
/*
	shape::printShapeInfoLinear(hostXShapeInfo);
	shape::printShapeInfoLinear(tad->tadOnlyShapeInfo);
	shape::printShapeInfoLinear(target);
*/
	delete tad;
}

int NativeOps::memcpyConstantAsync(Nd4jIndex dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
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
	//cudaError_t result = cudaMemcpyAsync((void *) dst, (const void *) src, (size_t) size, kind, *pStream);
	cudaError_t result = cudaMemcpyToSymbolAsync(deviceConstantMemory, (const void *) src, size, dst, kind, *pStream);
	checkCudaErrors(result);
	if (result != 0) {
		printf("Symbol failed on [%lu] -> [%lu], size: [%i], direction: [%i]\n", src, dst, size, flags );
		return 0L;
	}
	else return 1;
}

Nd4jPointer NativeOps::getConstantSpace() {
	Nd4jPointer dConstAddr;
	cudaError_t result = cudaGetSymbolAddress((void **)&dConstAddr, deviceConstantMemory);

	return dConstAddr;
}

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    pullRowsKernelHalf<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, float *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelFloat<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, double *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelDouble<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	if (debug)
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 **x = reinterpret_cast<float16 **>(dx);

    if (debug && verbose)
        printf("averageHalf called\n");

    dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float16), funcAttributes[44]);

    averagingKernelHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float **x = reinterpret_cast<float **>(dx);

    if (debug && verbose)
        printf("averageFloat called\n");

    dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float), funcAttributes[45]);

    averagingKernelFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    double **x = reinterpret_cast<double **>(dx);

    if (debug && verbose)
        printf("averageDouble called\n");

    dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(double), funcAttributes[46]);

    averagingKernelDouble<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::shuffleDouble(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleKernelDouble<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::shuffleFloat(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleKernelFloat<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::shuffleHalf(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 **x = reinterpret_cast<float16 **>(dx);
    float16 **z = reinterpret_cast<float16 **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleKernelHalf<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedFloat<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

	if (opTypeA == 2) {
		if (opTypeB == 0) {
			DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
		}
	}

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedDouble<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedHalf<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalA, scalB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned) {
    // no-op

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

/*
 metaPredicateReduceFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *tadShapeInfo, int *tadOffsets, float *reductionBuffer, float *extraA, float *extraB, float scalarA, float scalarB) {
 */

//    metaPredicateReduceFloat<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, nullptr, extraA, extraB, scalarA, scalarB, scalarReturned);

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}



void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

	if (opTypeA == 2) {
		if (opTypeB == 0) {
			DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalA, scalB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
		}
	}
    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

bool NativeOps::isExperimentalEnabled() {
    return experimentalSupport;
}

void NativeOps::setOmpMinThreads(int threads) {
    minThreads = nd4j::math::nd4j_max<int>(32, threads);
    minThreads = nd4j::math::nd4j_min<int>(maxThreads, minThreads);
}

int NativeOps::getDevice() {
    int curDevice = -1;

    cudaGetDevice(&curDevice);

    return curDevice;
}

void NativeOps::setElementThreshold(int num) {
    // TODO: to be implemented
}

void NativeOps::setTADThreshold(int num) {
    // TODO: to be implemented
}

void NativeOps::execScalarFloat(Nd4jPointer *extraPointers,int opNum,
					 float *x,
					 int *xShapeInfo,
					 float *z,
					 int *zShapeInfo,
					 float *scalars,
					 float *extraParams,
					 int *dimension,
					 int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);


    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *hostTadShapeInfo = reinterpret_cast<int *>(extraPointers[9]);

    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[11]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);

    //dim3 launchDims = dim3(512, 32, 512);
    dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]),hostXShapeInfo, hostTadShapeInfo, funcAttributes[47] ,dimensionLength, sizeof(float), 0);

//    printf("ProblemLength: %i; tadLength: %i; .x: %i; .y: %i\n", shape::length(hostXShapeInfo), shape::length(hostTadShapeInfo), launchDims.x, launchDims.y);
//    fflush(stdout);

    DISPATCH_SIMPLE(scalarAlongDimension, float, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                double *x,
                                int *xShapeInfo,
                                double *z,
                                int *zShapeInfo,
                                double *scalars,
                                double *extraParams,
                                int *dimension,
                                int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    dim3 launchDims = dim3(256, 256, 1024);

    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[11]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);

    DISPATCH_SIMPLE(scalarAlongDimension, double, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execScalarHalf(Nd4jPointer *extraPointers,int opNum,
                                float16 *x,
                                int *xShapeInfo,
                                float16 *z,
                                int *zShapeInfo,
                                float16 *scalars,
                                float16 *extraParams,
                                int *dimension,
                                int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    dim3 launchDims = dim3(256, 256, 1024);

    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[11]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[13]);

    DISPATCH_SIMPLE(scalarAlongDimension, float16, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execAggregateFloat(Nd4jPointer *extraPointers,int opNum,
                                   float **arguments,
                                   int numArguments,
                                   int **shapes,
                                   int numShapes,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   float *realArguments,
                                   int numRealArguments) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: proper launch dims required here
    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateSimple, float, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::execAggregateDouble(Nd4jPointer *extraPointers,int opNum,
                                   double **arguments,
                                   int numArguments,
                                   int **shapes,
                                   int numShapes,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   double *realArguments,
                                   int numRealArguments) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: proper launch dims required here
    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateSimple, double, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execAggregateHalf(Nd4jPointer *extraPointers,int opNum,
                                   float16 **arguments,
                                   int numArguments,
                                   int **shapes,
                                   int numShapes,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   float16 *realArguments,
                                   int numRealArguments) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: proper launch dims required here
    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateSimple, float16, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execAggregateBatchFloat(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: fix this, we want something better then fixed number of threads per block
    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

    //printf("Launch params: .X: %i; .Y: %i; .Z: %i\n", numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateBatchSimple, float, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execAggregateBatchDouble(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: fix this, we want something better then fixed number of threads per block
    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

    //printf("Launch params: .X: %i; .Y: %i; .Z: %i\n", numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateBatchSimple, double, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execAggregateBatchHalf(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    // TODO: fix this, we want something better then fixed number of threads per block
    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

    //printf("Launch params: .X: %i; .Y: %i; .Z: %i\n", numBlocks, numThreads, shmem);

    DISPATCH_SIMPLE(aggregateBatchSimple, float16, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

    if (debug)
        checkCudaErrors(cudaStreamSynchronize(*stream));
}