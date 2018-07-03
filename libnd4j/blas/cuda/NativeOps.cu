
#include "../NativeOps.h"
#include <cuda.h>
#include <cuda_launch_config.h>

#include <buffer.h>
#include <helpers/shape.h>
#include "../Environment.h"
#include <helpers/TAD.h>

#include <ops/specials.h>
#include <loops/reduce3.h>
#include <loops/reduce.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>
#include <loops/scalar.h>
#include <loops/broadcasting.h>
#include <loops/summarystatsreduce.h>
#include <loops/random.h>

//#include <thread>
#include <map>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <pointercast.h>
#include <stdio.h>
#include <stdlib.h>
#include <loops/type_conversions.h>
#include <op_boilerplate.h>
#include <loops/grid_shaped.h>
#include <loops/grid_strided.h>
#include <loops/aggregates.h>
#include <helpers/threshold.h>
#include <ShapeList.h>
#include <Context.h>
#include <ops/specials_cuda.h>

// FIXME: we need cuda-specific implementations
#include <helpers/logger.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <graph/VariablesSet.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>



//#include <sys/time.h>

// b40c only available for gcc :(
#ifdef  __clang__
// do nothing
#elif __GNUC__
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>

#include <b40c/radix_sort/enactor.cuh>
#endif

#include <curand.h>
#include <Status.h>
#include <helpers/DebugHelper.h>

using namespace nd4j;

#include <loops/special_kernels.h>

cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[64];
int blockLimit = 128;
int maxThreads = 512;
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


// this method isn't used, left here for legacy and caution purposes
// TLDR: don't use this way, it sucks
void CUDART_CB syncCallback(cudaStream_t stream, cudaError_t status, void *data){
    SyncInfo *sync = reinterpret_cast<SyncInfo *>(data);

    printf("Finished stream: [%i], kernel call: [%i]\n", sync->streamId, sync->callId);
}

// this method just does type conversion in fancy way
int getDeviceId(Nd4jPointer ptrToDeviceId) {
    return (int)(Nd4jLong)ptrToDeviceId;
}

template <typename T>
dim3 getOptimalDimensions(Nd4jLong n,cudaFuncAttributes attributes, cudaDeviceProp properties) {

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

/*
 * Basic CUDA constants here: number of blocks per MP
 */
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("Preliminary basic launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i]\n", num_blocks, num_threads, memory_limit);


	return launchDims;
}

/*
 * This message returns shared memory threshold value. default overflow ratio is 0.3
 */
int getDeviceSharedThreshold(int deviceId) {
	int ccMinor = deviceProperties[deviceId].minor;
	int ccMajor = deviceProperties[deviceId].major;

	// please note threshold isn't multiple of 32, and that's NOT a mistake

	int shmemThreshold;
	if (ccMajor == 6 && ccMinor == 0)
		shmemThreshold = 65536;
	else if (ccMajor == 6 && ccMinor == 1)
		shmemThreshold = 49152;
	else if (ccMajor == 5 && ccMinor == 2)
		shmemThreshold = 98304;
	else if (ccMajor == 5)
		shmemThreshold = 65536;
	else if (ccMajor == 3 && ccMinor == 7)
		shmemThreshold = 114688;
	else shmemThreshold = 49152;

	return shmemThreshold / 0.3;
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
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




	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("Preliminary reduce launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], reduction_per_block: [%i], blocksPerMP: [%i]\n", num_blocks, num_threads, memory_limit, reduction_per_block, targetBlocksPerMP);

	return dim3(num_blocks,num_threads, memory_limit);
}

/*
 * This method returns kernel launch param for linear memory access
 */
dim3 getFlatLaunchParams(int deviceId, Nd4jLong *xShapeInfo, Nd4jLong *yShapeInfo, cudaFuncAttributes funcAttr) {
	auto xRank = shape::rank(xShapeInfo);
	auto yRank = yShapeInfo == nullptr ? 0 : shape::rank(yShapeInfo);
	auto zRank = 0;

	int memory_limit = getBaseMemorySize(xRank, funcAttr);

	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	auto xLength = shape::length(xShapeInfo);
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("Preliminary scalar launch params: gridSize: [%i], blockSize: [%i], base shmem: [%i], blocksPerMP: [%i], problemLength: [%i], effectiveBlockLimit: [%i]\n", num_blocks, num_threads, memory_limit, targetBlocksPerMP, xLength, effective_block_limit);


	return launchDims;
}

/**
 * This method returns kernel launch params with TAD-based memory access
 *
 * @param deviceId
 * @param xShapeInfo
 * @param tadShapeInfo
 * @param funcAttr
 * @param dimensionLength
 * @param elementSize
 * @param reductionSize
 * @return
 */
dim3 getReduceLaunchParams(int deviceId, Nd4jLong *xShapeInfo, Nd4jLong *tadShapeInfo, cudaFuncAttributes funcAttr, int dimensionLength, int elementSize, int reductionSize) {

	Nd4jLong tadLength = 0;
	Nd4jLong numTads = 0;
	if (tadShapeInfo != nullptr) {
		tadLength = shape::length(tadShapeInfo);
		numTads = shape::length(xShapeInfo) / tadLength;

		if (tadLength == 1) {
			if (nd4j::Environment::getInstance()->isDebugAndVerbose())
				printf("A xLength: [%i], zLength: [%i]\n", shape::length(xShapeInfo), shape::length(tadShapeInfo));
		}
	} else{
		// we have special case - reduction along all dimensions
		tadLength = nd4j::math::nd4j_min<int>(shape::length(xShapeInfo), 768);
		numTads = shape::length(xShapeInfo) / tadLength;
	}

	auto xRank = shape::rank(xShapeInfo);
	int zRank = tadShapeInfo == nullptr ? 0 : shape::rank(tadShapeInfo);

	dim3 launchDims = getBetterDimensions(deviceId, numTads, tadLength, xRank, funcAttr, dimensionLength, elementSize, reductionSize);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose()) { //|| launchDims.x == 1
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
	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto n = shape::length(hostXShapeInfo);

	dim3 launchDims = getOptimalDimensions<T>(n,attributes, properties);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("Params: gridSize: [%i], blockSize: [%i], shMem: [%i], problemLength: [%i], totalThreads:[%i]\n", launchDims.x, launchDims.y, launchDims.z, n, (launchDims.x * launchDims.y));

	return launchDims;
}

nd4j::buffer::Buffer<Nd4jLong> * createScalarBuffer(cudaStream_t stream) {
	Nd4jLong *scalarShapeInfo = shape::createScalarShapeInfo();
	nd4j::buffer::Buffer<Nd4jLong> *buff = nd4j::buffer::createBuffer(scalarShapeInfo,shape::shapeInfoLength(2), stream);
	nd4j::buffer::copyDataToGpu(&buff, stream);
	return buff;
}


class ScalarShapeInformation {
private:
	nd4j::buffer::Buffer<Nd4jLong> *scalarDimension;
	nd4j::buffer::Buffer<Nd4jLong> *scalarShapeInfo;
//	std::thread::id threadId;

public:
	ScalarShapeInformation(cudaStream_t stream) {
		auto scalarDimensionBuff = reinterpret_cast<Nd4jLong *>(malloc(sizeof(Nd4jLong)));

		CHECK_ALLOC(scalarDimensionBuff, "Failed to allocate ShapeInfoBuffer");	

		scalarDimensionBuff[0] = MAX_DIMENSION;
		scalarDimension = nd4j::buffer::createBuffer(scalarDimensionBuff,1, stream);
		scalarShapeInfo = createScalarBuffer(stream);
//		threadId = std::this_thread::get_id();

	}
	~ScalarShapeInformation() {
		nd4j::buffer::freeBuffer(&scalarShapeInfo);
		nd4j::buffer::freeBuffer(&scalarDimension);
	}


	Nd4jLong *getShapeInfoHostPointer() {
		return scalarShapeInfo->data;
	}

	Nd4jLong * getShapeInfoGpuPointer() {
		return scalarShapeInfo->gData;
	}

	Nd4jLong * getDimensionHostPointer() {
		return scalarDimension->data;
	}

	Nd4jLong  * getDimensionGpuPointer() {
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
		T *scalarResult = reinterpret_cast<T*>(malloc(sizeof(T)));

		CHECK_ALLOC(scalarResult, "Failed to allocate new scalar buffer");

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
	 Nd4jLong *getDeviceShapeInfo() {
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
	  Nd4jLong *getDimensionDevicePointer() {
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
		Nd4jLong *xShapeInfo,
		double *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	Nd4jLong *hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	Nd4jLong *hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	Nd4jLong *deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	Nd4jLong *deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D1 opNum:[%i]\n", opNum);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], 1, sizeof(double), 3);

	functions::indexreduce::IndexReduce<double>::executeIndexReduceScalar(launchDims, stream, opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension, int dimensionLength) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	Nd4jLong *hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	Nd4jLong *hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	Nd4jLong *hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	Nd4jLong *deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	Nd4jLong *deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D2 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], dimensionLength, sizeof(double), 3);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	functions::indexreduce::IndexReduce<double>::executeIndexReduce(launchDims, stream,
			opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
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
		Nd4jLong *xShapeInfo,
		double *y,
		Nd4jLong *yShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension, int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
	auto deviceTADShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto deviceTADOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);


	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[26],  dimensionLength, sizeof(double), 2);

	functions::broadcast::Broadcast<double>::executeBroadcast(launchDims, stream, opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);
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
		Nd4jLong xStride,
		double *y,
		Nd4jLong yStride,
		double *result,
		Nd4jLong resultStride,
		double *extraParams, Nd4jLong n) {

    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<double>::execudaCudaStrided(launchDims, extraPointers, opNum, dx, xStride, y, yStride, result, resultStride, extraParams, n);
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
		Nd4jLong *xShapeInfo,
		double *y,
		Nd4jLong *yShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *yIndexes,
		Nd4jLong *resultIndexes) {
	///
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
		Nd4jLong *xShapeInfo,
		double *y,
		Nd4jLong *yShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double *extraParams) {

	dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<double>::execudaCudaShaped(launchDims, extraPointers, opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams);;
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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D7 opNum:[%i]\n", opNum);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	if (opNum == 19) {
		execReduceDouble(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo);
	}

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], 1, sizeof(double), 1);

	// this macro builds bunch of IF/ELSE selectors for kernel launch

	functions::reduce::ReduceFunction<double>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceDouble(...) failed");
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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D8 opNum:[%i]\n", opNum);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	if (opNum == 19) {
		execReduceDouble(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
	}

	/**
	 * We have separate kernels, optimized for different number of dimensions for reductions
	 */
	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<double>::execReduceXD(launchDims, stream, opNum, 1, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[33], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<double>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<double>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	}

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
		Nd4jLong *xShapeInfo,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D9 opNum:[%i]\n", opNum);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	//dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], 1, sizeof(double), 1);
    dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[22]);

	// for LogExpSum op we need to know max value, and store it
	if (opNum == 19) {
		double tmp = execReduceScalarDouble(extraPointers, 3, x, xShapeInfo, extraParams);
		extraParams = resultPointer;
	};

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

	functions::reduce::ReduceFunction<double>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarDouble(...) failed");

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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *y,
		Nd4jLong *yShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D10 opNum:[%i]\n", opNum);

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
			1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *y,
		Nd4jLong *yShapeInfo){
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D11 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	auto resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	
	auto allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

    double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

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
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	// since this method should return scalar value - we should block on this call
    nd4j::DebugHelper::checkErrorCode(stream, "execReduce3ScalarDouble(...) failed");

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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *y,
		Nd4jLong *yShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D12 opNum:[%i]\n", opNum);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	Nd4jLong *yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

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
			1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong xStride,
		double *result,
		Nd4jLong resultStride,
		double scalar,
		double *extraParams,
		Nd4jLong n) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[20]);

    functions::scalar::ScalarTransform<double>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, scalar, extraParams, n);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double scalar,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[19]);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleShaped, double, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<double>::executeCudaShaped(launchDims, extraPointers, opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double scalar,
		double *extraParams,
		Nd4jLong n,
		Nd4jLong *xIndexes,
		Nd4jLong *resultIndexes){


    printf("Unsupported operation: scalarIndices\n");
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
		Nd4jLong *xShapeInfo,
		double *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(double), 8);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	return functions::summarystats::SummaryStatsReduce<double>::execSummaryStatsReduceScalar(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, biasCorrected);
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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,bool biasCorrected) {
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D17 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(double), 8);

	// we have to limit grid size here, due to limited nature of reduction/allocation pointers
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<double>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, biasCorrected);
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
		Nd4jLong *xShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension, int dimensionLength,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], dimensionLength, sizeof(double), 8);

	// we're limiting maximum grid size for summaryStats ops
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<double>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, biasCorrected);
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
		Nd4jLong xStride,
		double *z,
		Nd4jLong zStride,
		double *extraParams,
		Nd4jLong n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D19 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[16]);

	functions::transform::Transform<double>::executeTransformStrided(launchDims, stream, opNum, n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer);
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
		Nd4jLong *xShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D20 opNum:[%i]\n", opNum);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);
    int *maskedAllocPointer = allocPointer;

	// special pointer for special buffer for special ops
	double *specialPointer = reinterpret_cast<double *>(extraPointers[6]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	auto dimension = reinterpret_cast<int *>(specialPointer);
	int *maxDimension = dimension + 1;
	auto maxShapeBuffer = reinterpret_cast<Nd4jLong *>(maxDimension + 1);
	double * special = reinterpret_cast<double *>(maxShapeBuffer + (MAX_RANK * 2 + 4));


    auto devTadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
    auto devTadOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);


    /**
     * ops between 38 and 41 are special ops:
     * SoftMax, LogSoftMax, SoftMaxDerivative, IsMax
     * On cuda we execute them as
     */
	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			/*
			 * For vector cases of everything, but IsMax (41) we go for single-kernel calls
			 */
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(256, length);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(double) * 4);
			
			functions::transform::Transform<double>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);
		} else {
			// going for blockwise specials
			// we'll do some pointers mangling here, and execute kernels one by one

			auto shape = shape::shapeOf(hostXShapeInfo);
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


					Nd4jLong maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					auto hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);
					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					// TODO: we could get rid of this one eventually
					prepareShapeBuffer <<<1, 1, 128, *stream>>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					DEBUG_KERNEL(stream, opNum);

					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceDouble(tempPointers, 3, dx, xShapeInfo, extraParams, special, maxShapeBuffer, maxDimension, 1);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// sub 1
					execBroadcastDouble(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					// exp 3
					execTransformDouble(extraPointers, 3, result, resultShapeInfo, result, resultShapeInfo, extraParams);

					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceDouble(tempPointers, 1, result, resultShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// divide 3
					execBroadcastDouble(tempPointers, 3, result, resultShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					// log 3
					if (opNum == 40)
						execTransformDouble(extraPointers, 5, result, resultShapeInfo, result, resultShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformDouble(extraPointers, 42, result, resultShapeInfo, result, resultShapeInfo, extraParams);

                    nd4j::DebugHelper::checkErrorCode(stream, "SoftMax failed failed");

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions
					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					}

					if (scalarCheat) {
						/**
						 * In case of vector-input for IsMax, it just turns into IndexReduce call + further filler call
						 */
						int maxIdx = (int) execIndexReduceScalarDouble(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxDouble<<< 1, 128, 0, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);
					} else {
						auto tadMaxShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
                        auto tadMaxOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<double *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceDouble(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						DEBUG_KERNEL(stream, opNum);

						// at this point, all IMax indexes are gathered, and we execute filler
						fillDimensionalIsMaxDouble<<<blockLimit, 64, funcAttributes[37].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );

                        nd4j::DebugHelper::checkErrorCode(stream, "Legacy IsMax(...) failed");
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
		// for Im2Col & Col2Im we enforce higher dimensionality
		// TODO: investigate this on high-end gpus
        if (opNum == 37 || opNum == 36 || opNum == 71) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 512 * sizeof(double);
        } else if (opNum == 70) {
            // we'll be using shared memory to speed up reverse

            launchDims.z += launchDims.y * sizeof(double);
        }

		// Histogram op requires additional memory chunk
		// FIXME: make this one to use cache
        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc(reinterpret_cast<void **>(&maskedAllocPointer), length * launchDims.x * sizeof(double));
        }


        if (opNum == 71) {
            launchDims.z += 512 * sizeof(double);
        }

		functions::transform::Transform<double>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);

        // we need guaranteed sync here, due to temp memory release
        if (opNum == 48)
            nd4j::DebugHelper::checkErrorCode(stream, "execTransformShaped(...) failed");

		// release Histogram memory
        if (opNum == 48) {
            cudaFree(reinterpret_cast<void *>(maskedAllocPointer));
        }
	}

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		double *result,
		Nd4jLong *resultShapeInfo,
		double *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *resultIndexes) {
    //
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
		Nd4jLong *xShapeInfo,
		float *extraParams){
	if (nd4j::Environment::getInstance()->isDebug())
		printf("F1 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float), 4);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose() && launchDims.x == 1)
		printf("AF1 opNum:[%i]\n", opNum);

	functions::indexreduce::IndexReduce<float>::executeIndexReduceScalar(launchDims, stream, opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

    nd4j::DebugHelper::checkErrorCode(stream, "execIndexReduceScalarFloat(...) failed");

	float result = resultPointer[0];
	return result;
}


float   NativeOps::execIndexReduceScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams){
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H1 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float16), 8);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose() && launchDims.x == 1)
		printf("AH1 opNum:[%i]\n", opNum);

	functions::indexreduce::IndexReduce<float16>::executeIndexReduceScalar(launchDims, stream, opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			resultPointer,
			nullptr, 0,
			nullptr,
			1,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);


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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F2 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float), 4);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF2 opNum:[%i]\n", opNum);

	functions::indexreduce::IndexReduce<float>::executeIndexReduce(launchDims, stream, opNum,
			x,
			xShapeInfo, shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultShapeInfo, shape::rank(hostZShapeInfo),
			dimension,
			dimensionLength,
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
}

void   NativeOps::execIndexReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H2 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float16), 8);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH2 opNum:[%i]\n", opNum);

	functions::indexreduce::IndexReduce<float16>::executeIndexReduce(launchDims, stream, opNum,
					x,
					xShapeInfo, shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultShapeInfo, shape::rank(hostZShapeInfo),
					dimension,
					dimensionLength,
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
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
		Nd4jLong *xShapeInfo,
		float *y,
		Nd4jLong *yShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension, int dimensionLength){
/*
    cudaEvent_t start;
    cudaEventCreateWithFlags(&start, cudaEventDisableTiming);

    timespec tsX;
    timespec tsY;
    clock_gettime(CLOCK_REALTIME, &tsX);
*/
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
	auto deviceTADShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto deviceTADOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float), 0);

	functions::broadcast::Broadcast<float>::executeBroadcast(launchDims, stream, opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);

	DEBUG_KERNEL(stream, opNum);


}


void   NativeOps::execBroadcastHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *y,
		Nd4jLong *yShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension, int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
	auto deviceTADShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto deviceTADOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);


	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float16), 0);

	functions::broadcast::Broadcast<float16>::executeBroadcast(launchDims, stream, opNum, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ);
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
		Nd4jLong xStride,
		float *y,
		Nd4jLong yStride,
		float *result,
		Nd4jLong resultStride,
		float *extraParams, Nd4jLong n){
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float>::execudaCudaStrided(launchDims, extraPointers, opNum, dx, xStride, y, yStride, result, resultStride, extraParams, n);
}

void   NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		Nd4jLong xStride,
		float16 *y,
		Nd4jLong yStride,
		float16 *result,
		Nd4jLong resultStride,
		float16 *extraParams, Nd4jLong n){
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float16>::execudaCudaStrided(launchDims, extraPointers, opNum, dx, xStride, y, yStride, result, resultStride, extraParams, n);
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
		Nd4jLong *xShapeInfo,
		float *y,
		Nd4jLong *yShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *yIndexes,
		Nd4jLong *resultIndexes){
    ///
}


void NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *y,
		Nd4jLong *yShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		float16 *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *yIndexes,
		Nd4jLong *resultIndexes){

    ///
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
		Nd4jLong *xShapeInfo,
		float *y,
		Nd4jLong *yShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float *extraParams){
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float>::execudaCudaShaped(launchDims, extraPointers, opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams);;

}

void NativeOps::execPairwiseTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *y,
		Nd4jLong *yShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		float16 *extraParams){
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float16>::execudaCudaShaped(launchDims, extraPointers, opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams);;

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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F7 opNum:[%i]\n", opNum);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], 1, sizeof(float), 1);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF7 opNum:[%i]\n", opNum);

	if (opNum == 19) {
		execReduceFloat(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo);
	}

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))
	functions::reduce::ReduceFunction<float>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceFloat(...) failed");
}

void   NativeOps::execReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H7 opNum:[%i]\n", opNum);

	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], 1, sizeof(float16), 1);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH7 opNum:[%i]\n", opNum);

	if (opNum == 19) {
		execReduceHalf(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo);
	}

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))
	
	functions::reduce::ReduceFunction<float16>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceHalf(...) failed");
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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F8 opNum:[%i]\n", opNum);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

//	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float), 1);

	if (opNum == 19) {
		execReduceFloat(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
	}

	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float>::execReduceXD(launchDims, stream, opNum, 1, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[33], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	}
}

void   NativeOps::execReduceHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H8 opNum:[%i]\n", opNum);

	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float16), 1);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH8 opNum:[%i]\n", opNum);

	if (opNum == 19) {
		execReduceHalf(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
	}

	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float16>::execReduceXD(launchDims, stream, opNum, 1, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[33], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float16>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], dimensionLength, sizeof(double), 2);

		functions::reduce::ReduceFunction<float16>::execReduceXD(launchDims, stream, opNum, shape::rank(hostTADShapeInfo), x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);
	}
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
		Nd4jLong *xShapeInfo,
		float *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F9 opNum:[%i]\n", opNum);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[8]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF9 opNum:[%i]\n", opNum);

	// for LogExpSum op we need to know max value, and store it
	if (opNum == 19) {
		float tmp = execReduceScalarFloat(extraPointers, 3, x, xShapeInfo, extraParams);
		extraParams = resultPointer;
	};

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))
	functions::reduce::ReduceFunction<float>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo);

	// blocking this one
    nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarFloat(...) failed");

	float result = resultPointer[0];
	return result;
}

float NativeOps::execReduceScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H9 opNum:[%i]\n", opNum);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 2, funcAttributes[8]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH9 opNum:[%i]\n", opNum);

	// for LogExpSum op we need to know max value, and store it
	if (opNum == 19) {
		float tmp = execReduceScalarHalf(extraPointers, 3, x, xShapeInfo, extraParams);
		extraParams = resultPointer;
	};

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))
	functions::reduce::ReduceFunction<float16>::execReduceScalar(launchDims, stream, opNum, x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo);

	// blocking call
    nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarHalf(...) failed");

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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *y,
		Nd4jLong *yShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F10 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	DEBUG_KERNEL(stream, opNum);
}

void   NativeOps::execReduce3Half(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		Nd4jLong *yShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H10 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *y,
		Nd4jLong *yShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F11 opNum:[%i]\n", opNum);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 32, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
			1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	// blocking call
    nd4j::DebugHelper::checkErrorCode(stream, "execReduce3ScalarFloat(...) failed");

	float result  = resultPointer[0];
	return result;
}

float   NativeOps::execReduce3ScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		Nd4jLong *yShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H11 opNum:[%i]\n", opNum);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
					1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);

	// blocking call
    nd4j::DebugHelper::checkErrorCode(stream, "execReduce3ScalarHalf(...) failed");

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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *y,
		Nd4jLong *yShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F12 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 16, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
						1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);
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
						1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);
	}

	DEBUG_KERNEL(stream, opNum);
}

void   NativeOps::execReduce3Half(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *y,
		Nd4jLong *yShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

    auto yDeviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto yDeviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H12 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostXShapeInfo), 8, funcAttributes[7]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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
						1, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);
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
						1, allocationPointer, deviceTADShapeInfo, deviceTADOffsets, yDeviceTADShapeInfo, yDeviceTADOffsets);
	}

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong xStride,
		float *result,
		Nd4jLong resultStride,
		float scalar,
		float *extraParams,
		Nd4jLong n){

        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

        int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        functions::scalar::ScalarTransform<float>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, scalar, extraParams, n);

	DEBUG_KERNEL(stream, opNum);
}


void   NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong xStride,
        float16 *result,
        Nd4jLong resultStride,
        float scalar,
        float16 *extraParams,
        Nd4jLong n){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleStrided, float16, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))
    float16 sc = (float16) scalar;
    functions::scalar::ScalarTransform<float16>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, sc, extraParams, n);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float scalar,
		float *extraParams){
	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	Nd4jLong n = shape::length(hostXShapeInfo);

//	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
//		printf("F14 opNum:[%i]\n", opNum);

	//dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[getDeviceId(extraPointers[2])]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[5]);

	//if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
	//	printf("AF14 opNum:[%i], xLength:[%i]\n", opNum, shape::length(hostXShapeInfo));

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleShaped, float, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    functions::scalar::ScalarTransform<float>::executeCudaShaped(launchDims, extraPointers, opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		float scalarF,
		float16 *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto n = shape::length(hostXShapeInfo);

	//if (nd4j::Environment::getInstance()->isDebugAndVerbose())
	//	printf("H14 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[5]);

    float16 scalar = (float16) scalarF;

    //if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
    //		printf("AH14 opNum:[%i], xLength:[%i]\n", opNum, shape::length(hostXShapeInfo));

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleShaped, float16, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<float16>::executeCudaShaped(launchDims, extraPointers, opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float scalar,
		float *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto n = shape::length(hostXShapeInfo);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F15 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[4]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF15 opNum:[%i]\n", opNum);

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		float *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

    auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(float), 8);

	// we limit grid size for SummaryStats calls
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	return functions::summarystats::SummaryStatsReduce<float>::execSummaryStatsReduceScalar(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, biasCorrected);
}


float   NativeOps::execSummaryStatsScalarHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

    Nd4jLong *deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(float16), 8);

    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    return (float) functions::summarystats::SummaryStatsReduce<float16>::execSummaryStatsReduceScalar(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, biasCorrected);
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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(float), 8);

	// limiting number of blocks in grid, to match buffer memory size
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<float>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, biasCorrected);
}


void   NativeOps::execSummaryStatsHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(float16), 8);

	// as everywhere else, we limit maximal number of blocks for SummaryStats calls
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<float16>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, biasCorrected);
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
		Nd4jLong *xShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], dimensionLength, sizeof(float), 8);

	// as everywhere else, we limit maximal number of blocks for SummaryStats calls
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<float>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, biasCorrected);

}


void   NativeOps::execSummaryStatsHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *x,
		Nd4jLong *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		bool biasCorrected) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], dimensionLength, sizeof(float16), 8);

	// as everywhere else, we limit maximal number of blocks for SummaryStats calls
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

    functions::summarystats::SummaryStatsReduce<float16>::execSummaryStatsReduce(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, biasCorrected);

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
		Nd4jLong xStride,
		float *z,
		Nd4jLong zStride,
		float *extraParams,
		Nd4jLong n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F19 opNum:[%i]\n", opNum);


	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

	functions::transform::Transform<float>::executeTransformStrided(launchDims, stream, opNum, n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer);
}


void   NativeOps::execTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		Nd4jLong xStride,
		float16 *z,
		Nd4jLong zStride,
		float16 *extraParams,
		Nd4jLong n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H19 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));


	functions::transform::Transform<float16>::executeTransformStrided(launchDims, stream, opNum, n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer);
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
		Nd4jLong *xShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F20 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	float *specialPointer = reinterpret_cast<float *>(extraPointers[6]);

	int *dimension = reinterpret_cast<int *>(specialPointer);
	int *maxDimension = dimension + 1;
	auto maxShapeBuffer = reinterpret_cast<Nd4jLong *>(maxDimension + 1);
	float * special = reinterpret_cast<float *> (maxShapeBuffer + (MAX_RANK * 2 + 4));

    int *maskedAllocPointer = allocPointer;

    auto devTadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
    Nd4jLong *devTadOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);


    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF20 opNum:[%i]\n", opNum);

	// simple trick to get workaround over reductions into scalar
	// that's special ops: SoftMax, SoftMaxDerivative, LogSoftMax, IsMax
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(length, 256);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(float) * 4);

			functions::transform::Transform<float>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);

		} else {
			// going for blockwise specials

			auto shape = shape::shapeOf(hostXShapeInfo);
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


					Nd4jLong maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					auto hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);

					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					prepareShapeBuffer <<< 1, 1, 128, *stream >>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					DEBUG_KERNEL(stream, opNum);

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceFloat(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					DEBUG_KERNEL(stream, opNum);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastFloat(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					DEBUG_KERNEL(stream, opNum);

					// exp 3
					execTransformFloat(extraPointers, 3, result, resultShapeInfo, result, resultShapeInfo, extraParams);

					DEBUG_KERNEL(stream, opNum);


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceFloat(tempPointers, 1, result, resultShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					DEBUG_KERNEL(stream, opNum);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// divide 3
					execBroadcastFloat(tempPointers, 3, result, resultShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					DEBUG_KERNEL(stream, opNum);

					// log 3
					if (opNum == 40)
						execTransformFloat(extraPointers, 5, result, resultShapeInfo, result, resultShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformFloat(extraPointers, 42, result, resultShapeInfo, result, resultShapeInfo, extraParams);


                    nd4j::DebugHelper::checkErrorCode(stream, "SoftMaxFloat(...) failed");

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions
					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					}

					if (scalarCheat) {
						// if that's 1D input - we'll just go for single dim IMax op call + filler
						int maxIdx = (int) execIndexReduceScalarFloat(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxFloat<<< 1, 128, 1536, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);

                        nd4j::DebugHelper::checkErrorCode(stream, "Legacy IsMax(...) failed");
					} else {
						// going for dimension-based IsMax
						auto tadMaxShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
                        auto tadMaxOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);
						auto dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceFloat(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						DEBUG_KERNEL(stream, opNum);

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxFloat<<<blockLimit, 64, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );

                        nd4j::DebugHelper::checkErrorCode(stream, "Legacy IsMax(...) failed");

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
		// we're enforcing larger grids for Col2Im & Im2Col
		// TODO: for high-end gpus we might use higher values here
        if (opNum == 37 || opNum == 36 || opNum == 71) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 512 * sizeof(float);
        } else if (opNum == 70) {
			// we'll be using shared memory to speed up reverse

			launchDims.z += launchDims.y * sizeof(float);
		}

		// histogram op requies additional memory chunk :(
        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc(reinterpret_cast<void **>(&maskedAllocPointer), length * launchDims.x * sizeof(float));
        }

		if (opNum == 71) {
			launchDims.z += 512 * sizeof(float);
		}
/*
		DISPATCH_SIMPLE(transformShaped, float,
                        PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo,
                               shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))
*/

		functions::transform::Transform<float>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);

        // we need guaranteed sync here, due to temp memory release
        if (opNum == 48)
            nd4j::DebugHelper::checkErrorCode(stream, "Legacy HistogramFloat(...) failed");

		// release memory chunk
        if (opNum == 48) {
            cudaFree(reinterpret_cast<void *>(maskedAllocPointer));
        }
    }

	DEBUG_KERNEL(stream, opNum);
}

void   NativeOps::execTransformHalf(Nd4jPointer *extraPointers,int opNum,
									 float16 *dx,
									 Nd4jLong *xShapeInfo,
									 float16 *result,
									 Nd4jLong *resultShapeInfo,
									 float16 *extraParams) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H20 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);
    int *maskedAllocPointer = allocPointer;

	float16 *specialPointer = reinterpret_cast<float16 *>(extraPointers[6]);

	int *dimension = reinterpret_cast<int *>(specialPointer);
	int *maxDimension = dimension + 1;
	auto maxShapeBuffer = reinterpret_cast<Nd4jLong *>(maxDimension + 1);
	float16 * special = reinterpret_cast<float16 *>(maxShapeBuffer + (MAX_RANK * 2 + 4));

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH20 opNum:[%i]\n", opNum);

    auto devTadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
    auto devTadOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);


    // simple trick to get workaround over reductions into scalar
	// SoftMax, SoftMaxDerivative, LogSoftMax, IsMax
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			auto length = shape::length(hostXShapeInfo);
			auto block = nd4j::math::nd4j_min<Nd4jLong>(length, 256);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(float16) * 4);

			functions::transform::Transform<float16>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);
		} else {
			// going for blockwise specials

			auto shape = shape::shapeOf(hostXShapeInfo);
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


					Nd4jLong maxShape[2] = {shape::shapeOf(hostXShapeInfo)[0], 1};
					auto hostMaxShapeBuffer = shape::shapeBuffer(2, maxShape);

					tempPointers[7] = (Nd4jPointer) hostMaxShapeBuffer;
					tempPointers[8] = (Nd4jPointer) hostMaxShapeBuffer;

					// FIXME: fix this
					prepareShapeBuffer <<< 1, 1, 128, *stream >>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					DEBUG_KERNEL(stream, opNum);

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceHalf(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					DEBUG_KERNEL(stream, opNum);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastHalf(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					DEBUG_KERNEL(stream, opNum);

					// exp 3
					execTransformHalf(extraPointers, 3, result, resultShapeInfo, result, resultShapeInfo, extraParams);

					DEBUG_KERNEL(stream, opNum);


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceHalf(tempPointers, 1, result, resultShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					DEBUG_KERNEL(stream, opNum);

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// divide 3
					execBroadcastHalf(tempPointers, 3, result, resultShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

                    if (opNum == 40) {
						DEBUG_KERNEL(stream, opNum);

                        execTransformHalf(tempPointers, 47, result, resultShapeInfo, result, resultShapeInfo, extraParams);
                    }

					DEBUG_KERNEL(stream, opNum);

					// log 3
					if (opNum == 40)
						execTransformHalf(extraPointers, 5, result, resultShapeInfo, result, resultShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformHalf(extraPointers, 42, result, resultShapeInfo, result, resultShapeInfo, extraParams);


                    nd4j::DebugHelper::checkErrorCode(stream, "Legacy SoftMaxHalf(...) failed");

					delete hostMaxShapeBuffer;

					break;
				}
				case 41: {
					// IsMax along all dimensions

					bool scalarCheat = false;
					if (extraParams == nullptr) {
						scalarCheat = true;
					}

					if (scalarCheat) {
						// 1D input, aka vector
						int maxIdx = (int) execIndexReduceScalarHalf(extraPointers, 0, dx, xShapeInfo, extraParams);
						int targetIdx = 0;

						if (shape::order(hostXShapeInfo) == 'c' || shape::order(hostXShapeInfo) == 'f' && maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1] >= shape::length(hostXShapeInfo))
							targetIdx = maxIdx;
						else
							targetIdx = maxIdx * shape::stride(hostXShapeInfo)[shape::rank(hostXShapeInfo) - 1];

						fillIsMaxHalf<<< 1, 128, 1536, *stream >>>(result, shape::length(hostXShapeInfo), targetIdx);
					} else {
						// going for dimension-based IsMax
						auto tadMaxShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[10]);
                        auto tadMaxOffsets = reinterpret_cast<Nd4jLong *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float16 *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceHalf(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						DEBUG_KERNEL(stream, opNum);

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxHalf<<<blockLimit, 64, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );


                        nd4j::DebugHelper::checkErrorCode(stream, "Legacy IsMaxHalf(...) failed");

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
		// Im2Col & Col2Im enforced grids
        if (opNum == 37 || opNum == 36 || opNum == 71) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 512 * sizeof(float16);
        } else if (opNum == 70) {
            // we'll be using shared memory to speed up reverse

            launchDims.z += launchDims.y * sizeof(float16);
        }

		// Histogram op requires additional memory chunk
        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc(reinterpret_cast<void **>(&maskedAllocPointer), length * launchDims.x * sizeof(float16));
        }

        if (opNum == 71) {
            launchDims.z += 512 * sizeof(float16);
        }

		functions::transform::Transform<float16>::executeTransformShaped(launchDims, stream, opNum, dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets);

        // we need guaranteed sync here, due to temp memory release
        if (opNum == 48)
            nd4j::DebugHelper::checkErrorCode(stream, "Legacy HistogramHalf(...) failed");

		// release that histogram memory chunk
        if (opNum == 48) {
            cudaFree(reinterpret_cast<void *>(maskedAllocPointer));
        }
	}

	DEBUG_KERNEL(stream, opNum);
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
		Nd4jLong *xShapeInfo,
		float *result,
		Nd4jLong *resultShapeInfo,
		float *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *resultIndexes) {
    ///
}


void   NativeOps::execTransformHalf(
		Nd4jPointer *extraPointers,
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		float16 *extraParams,
		Nd4jLong *xIndexes,
		Nd4jLong *resultIndexes) {
    ///
}


template <typename T>
__device__ void flattenKernelGeneric(int dOffset,
					char order,
					T *result,
					Nd4jLong *resultShapeInfo,
					T *input,
					Nd4jLong *inputShapeInfo, int *allocationPointer) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory(reinterpret_cast<int *>(shmem));
		manager->init(sizeof(UnifiedSharedMemory), 4, 4, sizeof(shape::TAD), 2);
	}
	__syncthreads();

	Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;

	auto zShape = shape::shapeOf(resultShapeInfo);
	auto zStride = shape::stride(resultShapeInfo);


	auto yShape = shape::shapeOf(inputShapeInfo);
	auto yStride = shape::stride(inputShapeInfo);
	auto yOrder = shape::order(inputShapeInfo);

	auto len = shape::length(inputShapeInfo);

	auto resultEWS = shape::elementWiseStride(resultShapeInfo);
	auto inputEWS = shape::elementWiseStride(inputShapeInfo);
	if (yOrder == order) {
		if (resultEWS >= 1 && inputEWS >= 1) {
			for (int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				result[i * resultEWS + dOffset] = input[i * inputEWS];
			}
		} else {

			auto rank = shape::rank(inputShapeInfo);
			Nd4jLong coord[MAX_RANK];

			if(order == 'f') {
				for(auto i = tid; i < len; i+= gridDim.x * blockDim.x) {
					shape::ind2sub(rank,yShape,i,coord);
					auto offset = shape::getOffset(0,yShape,yStride,coord,rank);
					result[i + dOffset] = input[offset];
				}
			}
			else {
				for(auto i = tid; i < len; i+= gridDim.x * blockDim.x) {
					shape::ind2subC(rank,yShape,i,coord);
					auto offset = shape::getOffset(0,yShape,yStride,coord,rank);
					result[i + dOffset] = input[offset];
				}
			}
		}
	} else {
		int rank = shape::rank(inputShapeInfo);
		Nd4jLong coord[MAX_RANK];

		if(order == 'f') {
			for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				shape::ind2sub(rank,yShape,i,coord);
				auto offset = shape::getOffset(0,yShape,yStride,coord,rank);
				result[i+dOffset] = input[offset];
			}
		}
		else {
			for(int i = tid; i < len; i+= gridDim.x * blockDim.x) {
				shape::ind2subC(rank,yShape,i,coord);
				auto offset = shape::getOffset(0,yShape,yStride,coord,rank);
				result[i+dOffset] = input[offset];
			}
		}
	}

}

extern "C" __global__ void flattenKernelDouble(int offset,
											  char order,
											  double *result,
											  Nd4jLong *resultShapeInfo,
											  double *input,
											  Nd4jLong *inputShapeInfo, int *allocationPointer) {
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
											  Nd4jLong *resultShapeInfo,
											  float *input,
											  Nd4jLong *inputShapeInfo, int *allocationPointer) {

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
											  Nd4jLong *resultShapeInfo,
											  float16 *input,
											  Nd4jLong *inputShapeInfo, int *allocationPointer) {

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
		Nd4jLong *resultShapeInfo,
		float *input,
		Nd4jLong *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF222 opNum:[7]\n");

	flattenKernelFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	DEBUG_KERNEL(stream, -1);
}


void NativeOps::flattenHalf(
		Nd4jPointer *extraPointers,
		int offset,
		char order,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		float16 *input,
		Nd4jLong *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH222 opNum:[7]\n");

	flattenKernelHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	DEBUG_KERNEL(stream, -1);
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
		Nd4jLong *resultShapeInfo,
		double *input,
		Nd4jLong *inputShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D30 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo),  2, funcAttributes[34]);

	flattenKernelDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	DEBUG_KERNEL(stream, -1);
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

				if (!canAccess) {
                    tempSupport = false;
                    break;
                }
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
					if (nd4j::Environment::getInstance()->isVerbose()) printf("Peer access [%i] -> [%i] isn't possible\n", x, y);
				}
            }
        }

        cudaSetDevice(curDevice);
    }

    allowedP2P = enable;

    cudaSetDevice(curDevice);
}

bool NativeOps::isP2PAvailable() {
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

	// enabling p2p gpu access if it's supported
	if (supportedP2P && devCnt > 1)
    	enableP2P(allowedP2P);

	//cudaFuncGetAttributes(&funcAttributes[0], (void *)transformFloatIndexes);

	//void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    //cudaFuncGetAttributes(&funcAttributes[1], transformFloatIndexes);

	//void (*transformFloatPointer2)(int opNum, Nd4jLong n, float *dy, int incy, float *params, float *result,int resultStride, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    //cudaFuncGetAttributes(&funcAttributes[2], transformFloatIndexes);

	//cudaFuncGetAttributes(&funcAttributes[3], (void *)functions::summarystats::summaryStatsReduceFloat);

	//cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

//	void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarFloat;
//	cudaFuncGetAttributes(&funcAttributes[5], scalarFloatIndexes);

//	void (*scalarFloatPointer2)(int opNum, Nd4jLong n,float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocPointer) = scalarFloat;
//	cudaFuncGetAttributes(&funcAttributes[6], scalarFloatIndexes);

	cudaFuncGetAttributes(&funcAttributes[7], reduce3Float);

	cudaFuncGetAttributes(&funcAttributes[8], reduce3Float);
//	printf("reduceFloat regs: [%i], static shmem: [%i]\n", funcAttributes[8].numRegs, funcAttributes[8].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[28], reduce3Float); // 1D
//	printf("reduceFloat1D regs: [%i], static shmem: [%i]\n", funcAttributes[28].numRegs, funcAttributes[28].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[29], reduce3Float); // 6D
//	printf("reduceFloat6D regs: [%i], static shmem: [%i]\n", funcAttributes[29].numRegs, funcAttributes[29].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[30], flattenKernelFloat);

	cudaFuncGetAttributes(&funcAttributes[31], concatKernelFloat);

//	cudaFuncGetAttributes(&funcAttributes[9], pairWiseTransformFloat);

//  cudaFuncGetAttributes(&funcAttributes[10], pairWiseTransformFloatIndex);

//	cudaFuncGetAttributes(&funcAttributes[11], pairWiseTransformStridedFloat);

	cudaFuncGetAttributes(&funcAttributes[12], reduce3Float);

	cudaFuncGetAttributes(&funcAttributes[13], reduce3Float);

	///////////////////////////////////////// Doubles are separate, just in case of...

	//cudaFuncGetAttributes(&funcAttributes[14], transformDoubleIndexes);

//	void (*transformDoublePointer1)(int opNum, double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocationPointer, double *reductionPointer) = transformDouble;
	// FIXME
    //cudaFuncGetAttributes(&funcAttributes[15], transformDoubleIndexes);

	//void (*transformDoublePointer2)(int opNum, Nd4jLong n, double *dy, int incy, double *params, double *result,int resultStride, int *allocationPointer, double *reductionPointer) = transformDouble;
	// FIXME
    //cudaFuncGetAttributes(&funcAttributes[16], transformDoubleIndexes);

	//cudaFuncGetAttributes(&funcAttributes[17], functions::summarystats::summaryStatsReduceDouble);

//	cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

	//void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarDouble;
//	cudaFuncGetAttributes(&funcAttributes[19], scalarDoubleIndexes);


	//void (*scalarDoublePointer2)(int opNum, Nd4jLong n,double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocPointer) = scalarDouble;
//	cudaFuncGetAttributes(&funcAttributes[20], scalarDoubleIndexes);

	cudaFuncGetAttributes(&funcAttributes[21], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[22], reduce3Float);

//	cudaFuncGetAttributes(&funcAttributes[23], pairWiseTransformDouble);

//	cudaFuncGetAttributes(&funcAttributes[24], pairWiseTransformDoubleIndex);

//	cudaFuncGetAttributes(&funcAttributes[25], pairWiseTransformStridedDouble);

	cudaFuncGetAttributes(&funcAttributes[26], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[27], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[32], reduce3Float); // 1D

	cudaFuncGetAttributes(&funcAttributes[33], reduce3Float); // 6D

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

    //cudaFuncGetAttributes(&funcAttributes[47], scalarAlongDimension_0_float);
    //cudaFuncGetAttributes(&funcAttributes[48], scalarAlongDimension_0_float16);
    //cudaFuncGetAttributes(&funcAttributes[48], scalarAlongDimension_0_double);
}

void NativeOps::initializeFunctions(Nd4jPointer *functions) {
    nd4j::BlasHelper::getInstance()->initializeDeviceFunctions(functions);
	/*
	this->cublasSgemv = (CublasSgemv)functions[0];
    this->cublasDgemv = (CublasDgemv)functions[1];
    this->cublasHgemm = (CublasHgemm)functions[2];
    this->cublasSgemm = (CublasSgemm)functions[3];
    this->cublasDgemm = (CublasDgemm)functions[4];
    this->cublasSgemmEx = (CublasSgemmEx)functions[5];
    this->cublasHgemmBatched = (CublasHgemmBatched)functions[6];
    this->cublasSgemmBatched = (CublasSgemmBatched)functions[7];
    this->cublasDgemmBatched = (CublasDgemmBatched)functions[8];
	*/
}


/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocHost(Nd4jLong memorySize, int flags) {
	Nd4jPointer pointer;
	// cudaHostAllocMapped |cudaHostAllocPortable
	cudaError_t res = cudaHostAlloc(reinterpret_cast<void **>(&pointer), memorySize, cudaHostAllocDefault);
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
Nd4jPointer NativeOps::mallocDevice(Nd4jLong memorySize, Nd4jPointer ptrToDeviceId, int flags) {
	Nd4jPointer pointer;
	cudaError_t res = cudaMalloc(reinterpret_cast<void **>(&pointer), memorySize);
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
	cudaError_t res = cudaFreeHost(reinterpret_cast<void *>(pointer));
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
	cudaError_t res = cudaFree(reinterpret_cast<void *>(pointer));
	if (res != 0)
		pointer = 0L;
	return 1L;
}


Nd4jPointer NativeOps::createContext() {
	return 0L;
}

Nd4jPointer NativeOps::createStream() {
	Nd4jPointer nativeStream = (Nd4jPointer) malloc(sizeof(cudaStream_t));

	CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream");

	cudaError_t result = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaStreamCreate(...) failed");

	return nativeStream;
}

Nd4jPointer NativeOps::createEvent() {
	Nd4jPointer nativeEvent= (Nd4jPointer) malloc(sizeof(cudaEvent_t));

	CHECK_ALLOC(nativeEvent, "Failed to allocate new CUDA event buffer");

	cudaError_t result = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&nativeEvent), cudaEventDisableTiming);
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaEventCreateWithFlags(...) failed");


	return nativeEvent;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaEventRecord(*pEvent, *pStream);
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaEventRecord(...) failed");

	return 1;
}

int NativeOps::setDevice(Nd4jPointer ptrToDeviceId) {
	int deviceId = getDeviceId(ptrToDeviceId);
	cudaError_t result = cudaSetDevice(deviceId);
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaSetDevice(...) failed");

	return 1;
}

Nd4jLong NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);
	int orig = -1;

	cudaGetDevice(&orig);

	if (device >= 0 && device != orig) {
		cudaSetDevice(device);
	}

	size_t memFree = 0;
	size_t memTotal = 0;

	cudaMemGetInfo(&memFree, &memTotal);

	if (device >= 0 && device != orig) {
		cudaSetDevice(orig);
	}

	return (Nd4jLong) memFree;
}

Nd4jLong NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
	int device = getDeviceId(ptrToDeviceId);
	int orig = -1;

	cudaGetDevice(&orig);

	if (device >= 0 && device != orig) {
		cudaSetDevice(device);
	}
	size_t memFree = 0;
	size_t memTotal = 0;

	cudaMemGetInfo(&memFree, &memTotal);

	if (device >= 0 && device != orig) {
		cudaSetDevice(orig);
	}

	return (Nd4jLong) memTotal;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {

	return memcpyAsync(dst, src, size, flags, reserved);
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaMemcpyKind 	kind;

	DEBUG_KERNEL(pStream, 0);

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

	cudaError_t result = cudaMemcpyAsync(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)), static_cast<size_t>(size), kind, *pStream);
	if (result != 0) {
        checkCudaErrors(result);
		printf("Failed on [%lu] -> [%lu], size: [%i], direction: [%i], result: [%i]\n", src, dst, size, flags, static_cast<int>(result));
        fflush(stdout);
        fflush(stderr);
        throw std::runtime_error("cudaMemcpyAsync(...) failed");
		//return 0L;
	}

	return 1;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaError_t result = cudaMemset(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size));
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaMemset(...) failed");

	return 1;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t result = cudaMemsetAsync(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size), *pStream);
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaMemsetAsync(...) failed");

	return 1;
}

int NativeOps::destroyEvent(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaError_t result = cudaEventDestroy(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
		throw std::runtime_error("cudaEvenDestroy(...) failed");

	return 1;
}

int NativeOps::streamSynchronize(Nd4jPointer stream) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t result = cudaStreamSynchronize(*pStream);
	checkCudaErrors(result);
	if (result != 0)
        throw std::runtime_error("cudaStreamSynchronize(...) failed");

	return 1L;
}

int NativeOps::eventSynchronize(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);

	cudaError_t result = cudaEventSynchronize(*pEvent);
	checkCudaErrors(result);
	if (result != 0)
        throw std::runtime_error("cudaEventSynchronize(...) failed");

	return 1L;
}

int NativeOps::getAvailableDevices() {
	int devCnt = 0;
	cudaGetDeviceCount(&devCnt);
	return devCnt;
}

void NativeOps::enableDebugMode(bool reallyEnable) {
	nd4j::Environment::getInstance()->setDebug(reallyEnable);
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
	nd4j::Environment::getInstance()->setVerbose(reallyEnable);
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
        Nd4jLong *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostShapePointers = reinterpret_cast<Nd4jLong **>(extraPointers[9]);

	// numArrays will be used as number of TADs, so each block process 1 input

	int smem = 8192;
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
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0 ||
				shape::order(hostShapePointers[i]) != 'c') {
				isVstack = false;
				break;
			}
		}
	}

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
		Nd4jLong length0 = shape::length(hostShapePointers[0]);
        isVstack = true;
        for (int i = 0; i < numArrays; i++) {
            if (shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c' || length0 != shape::length(hostShapePointers[i])) {
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
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going scalar concat\n");

		concatKernelScalarFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		concatKernelVStackFloat<<< 128, 512, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");

		concatKernelHStackFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

        auto devZTadShape = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
		auto devZOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

		concatKernelFloat<<< 512, 512, 8192, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]), devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatFloat: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


    nd4j::DebugHelper::checkErrorCode(stream, "Legacy ConcatFloat(...) failed");
}



void NativeOps::concatHalf(
		Nd4jPointer *extraPointers,
		int dimension,
		int numArrays,
		Nd4jPointer *data,
		Nd4jPointer *inputShapeInfo,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostShapePointers = reinterpret_cast<Nd4jLong **>(extraPointers[9]);

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

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
        Nd4jLong length0 = shape::length(hostShapePointers[0]);
        isVstack = true;
        for (int i = 0; i < numArrays; i++) {
            if (shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c' || length0 != shape::length(hostShapePointers[i])) {
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
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going scalar concat\n");

		smem = funcAttributes[38].sharedSizeBytes;
		concatKernelScalarHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		smem = funcAttributes[40].sharedSizeBytes;
		concatKernelVStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");
		smem = funcAttributes[42].sharedSizeBytes;

		concatKernelHStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

		//smem = nd4j::math::nd4j_max<int>(funcAttributes[31].sharedSizeBytes + 768, 1280);

        auto devZTadShape = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
		auto devZOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

		concatKernelHalf<<< 512, 128, 4096, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]), devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatHalf: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


    nd4j::DebugHelper::checkErrorCode(stream, "ConcatHalf(...) failed");
}


void NativeOps::specialConcatFloat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

}


void NativeOps::specialConcatHalf(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float16>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);
}
/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::specialConcatDouble(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<double>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

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
        Nd4jLong *resultShapeInfo,
		Nd4jPointer *tadPointers,
		Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	auto hostShapePointers = reinterpret_cast<Nd4jLong **>(extraPointers[9]);

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

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
        Nd4jLong length0 = shape::length(hostShapePointers[0]);
        isVstack = true;
        for (int i = 0; i < numArrays; i++) {
            if (shape::elementWiseStride(hostShapePointers[i]) <= 0 || shape::order(hostShapePointers[i]) != 'c' || length0 != shape::length(hostShapePointers[i])) {
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
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going scalar concat\n");

		smem = funcAttributes[39].sharedSizeBytes;
		concatKernelScalarDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		smem = funcAttributes[41].sharedSizeBytes;
		concatKernelVStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");
		smem = funcAttributes[43].sharedSizeBytes;

		concatKernelHStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]));
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

        auto devZTadShape = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
        auto devZOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

		concatKernelDouble<<< 512, 128, 4096, *stream>>> (dimension, numArrays, reinterpret_cast<Nd4jPointer *>(data[0]), reinterpret_cast<Nd4jPointer *>(inputShapeInfo[0]), result, resultShapeInfo, reinterpret_cast<Nd4jPointer *>(tadPointers[0]), reinterpret_cast<Nd4jPointer *>(offsetPointers[0]), devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatDouble: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


    nd4j::DebugHelper::checkErrorCode(stream, "ConcatDouble(...) failed");
}

/**
 * This method saves
 */
void NativeOps::tadOnlyShapeInfo(Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *target, Nd4jLong *offsets) {
	shape::TAD tad;
	tad.init(xShapeInfo, dimension, dimensionLength);
	//tad->setOutputBuffer(target);
	tad.createTadOnlyShapeInfo();
	tad.createOffsets();


	std::memcpy(reinterpret_cast<void *>(target), tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
	std::memcpy(reinterpret_cast<void *>(offsets), tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));
}

int NativeOps::memcpyConstantAsync(Nd4jLong dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaMemcpyKind 	kind;

	DEBUG_KERNEL(pStream, -1);

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
	cudaError_t result = cudaMemcpyToSymbolAsync(deviceConstantMemory, const_cast<const void *>(src), size, dst, kind, *pStream);
	checkCudaErrors(result);
	if (result != 0)
        throw std::runtime_error("cudaMemcpyToSymbolAsync(...) failed");

	return 1;
}

Nd4jPointer NativeOps::getConstantSpace() {
	Nd4jPointer dConstAddr;
	cudaError_t result = cudaGetSymbolAddress(reinterpret_cast<void **>(&dConstAddr), deviceConstantMemory);

	if (result != 0)
        throw std::runtime_error("cudaGetSymbolAddress(...) failed");

	return dConstAddr;
}

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, float16 *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    pullRowsKernelHalf<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	DEBUG_KERNEL(stream, -1);
}


void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, float *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelFloat<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	DEBUG_KERNEL(stream, -1);
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, double *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelDouble<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	DEBUG_KERNEL(stream, -1);
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jLong length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

    float16 **x = reinterpret_cast<float16 **>(dx);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("averageHalf called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float16), funcAttributes[44]);

		averagingKernelHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

        nd4j::DebugHelper::checkErrorCode(stream, "AverageHalf(...) failed");
	} else {
        nd4j::SpecialMethods<float16>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jLong length, bool propagate) {
	cudaStream_t * stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float **x = reinterpret_cast<float **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("averageFloat called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float), funcAttributes[45]);

		averagingKernelFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

        nd4j::DebugHelper::checkErrorCode(stream, "AverageFloat(...) failed");
	} else {
		// launching on host memory
        nd4j::SpecialMethods<float>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jLong length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

    double **x = reinterpret_cast<double **>(dx);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("averageDouble called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(double), funcAttributes[46]);

		averagingKernelDouble << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (x, dz, n, length, propagate);

        nd4j::DebugHelper::checkErrorCode(stream, "AverageDouble(...) failed");
	} else {
        nd4j::SpecialMethods<double>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::accumulateHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jLong length) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float16 **x = reinterpret_cast<float16 **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateHalf called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float16), funcAttributes[44]);

		accumulateKernelHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length);

        nd4j::DebugHelper::checkErrorCode(stream, "AccumulateHalf(...) failed");
	} else {
        nd4j::SpecialMethods<float16>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::accumulateFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jLong length) {
	cudaStream_t * stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float **x = reinterpret_cast<float **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateFloat called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float), funcAttributes[45]);

        accumulateKernelFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length);

        nd4j::DebugHelper::checkErrorCode(stream, "AccumulateFloat(...) failed");
	} else {
		// launching on host memory
        nd4j::SpecialMethods<float>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::accumulateDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jLong length) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	double **x = reinterpret_cast<double **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateDouble called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(double), funcAttributes[46]);

		accumulateKernelDouble << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (x, dz, n, length);

        nd4j::DebugHelper::checkErrorCode(stream, "AccumulateDouble(...) failed");
	} else {
        nd4j::SpecialMethods<double>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::shuffleDouble(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);


    shuffleKernelDouble<<<32, 128, 2048, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

	DEBUG_KERNEL(stream, 0);
}

void NativeOps::shuffleFloat(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    shuffleKernelFloat<<<32, 128, 2048, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

	DEBUG_KERNEL(stream, 0);
}

void NativeOps::shuffleHalf(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 **x = reinterpret_cast<float16 **>(dx);
    float16 **z = reinterpret_cast<float16 **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    shuffleKernelHalf<<<32, 128, 2048, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

	DEBUG_KERNEL(stream, 0);
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong xStride, float *dy, Nd4jLong yStride, float *dz, Nd4jLong zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    functions::grid::GRIDStrided<float>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong xStride, double *dy, Nd4jLong yStride, double *dz, Nd4jLong zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    functions::grid::GRIDStrided<double>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong xStride, float16 *dy, Nd4jLong yStride, float16 *dz, Nd4jLong zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

    functions::grid::GRIDStrided<float16>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}


void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned) {
    // no-op

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

/*
 metaPredicateReduceFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *tadShapeInfo, int *tadOffsets, float *reductionBuffer, float *extraA, float *extraB, float scalarA, float scalarB) {
 */

//    metaPredicateReduceFloat<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, nullptr, extraA, extraB, scalarA, scalarB, scalarReturned);

	DEBUG_KERNEL(stream, opNumA);
}



void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong *xShapeInfo, double *dy, Nd4jLong *yShapeInfo, double *dz, Nd4jLong *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    functions::grid::GRIDShaped<double>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong *xShapeInfo, float16 *dy, Nd4jLong *yShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

	// we have to converf float -> fp16 prior to kernel call
    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

	functions::grid::GRIDShaped<float16>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    functions::grid::GRIDShaped<float>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
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
    // this is no-op for CUDA
}

void NativeOps::setTADThreshold(int num) {
    // this is no-op for CUDA
}

void NativeOps::execScalarFloat(Nd4jPointer *extraPointers,int opNum,
					 float *x,
					 Nd4jLong *xShapeInfo,
					 float *z,
					 Nd4jLong *zShapeInfo,
					 float *scalars,
					 float *extraParams,
					 int *dimension,
					 int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto hostTadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);

    //dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]),hostXShapeInfo, hostTadShapeInfo, funcAttributes[47] ,dimensionLength, sizeof(float), 0);
    dim3 launchDims = dim3(256, 256, 1024);

    functions::scalar::ScalarTransform<float>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                double *x,
                                Nd4jLong *xShapeInfo,
                                double *z,
                                Nd4jLong *zShapeInfo,
                                double *scalars,
                                double *extraParams,
                                int *dimension,
                                int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    dim3 launchDims = dim3(256, 256, 1024);

	functions::scalar::ScalarTransform<double>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execScalarHalf(Nd4jPointer *extraPointers,int opNum,
                                float16 *x,
                                Nd4jLong *xShapeInfo,
                                float16 *z,
                                Nd4jLong *zShapeInfo,
                                float16 *scalars,
                                float16 *extraParams,
                                int *dimension,
                                int dimensionLength) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    dim3 launchDims = dim3(256, 256, 1024);

    functions::scalar::ScalarTransform<float16>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execAggregateFloat(Nd4jPointer *extraPointers,int opNum,
                                   float **arguments,
                                   int numArguments,
                                   Nd4jLong **shapes,
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateSimple, float, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    nd4j::DebugHelper::checkErrorCode(stream, "execAggregateFloat(...) failed");
}


void NativeOps::execAggregateDouble(Nd4jPointer *extraPointers,int opNum,
                                   double **arguments,
                                   int numArguments,
                                   Nd4jLong **shapes,
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateSimple, double, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    nd4j::DebugHelper::checkErrorCode(stream, "execAggregateDouble(...) failed");
}

void NativeOps::execAggregateHalf(Nd4jPointer *extraPointers,int opNum,
                                   float16 **arguments,
                                   int numArguments,
                                   Nd4jLong **shapes,
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateSimple, float16, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    nd4j::DebugHelper::checkErrorCode(stream, "execAggregateHalf(...) failed");
}

void NativeOps::execAggregateBatchFloat(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateBatchSimple, float, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execAggregateBatchDouble(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateBatchSimple, double, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execAggregateBatchHalf(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateBatchSimple, float16, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaSingle(launchDims, extraPointers, opNum, stateHost, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, Nd4jLong *xShapeBuffer, float *y, Nd4jLong *yShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaTriple(launchDims, extraPointers, opNum, stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, Nd4jLong *xShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaDouble(launchDims, extraPointers, opNum, stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaSingle(launchDims, extraPointers, opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *y, Nd4jLong *yShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaTriple(launchDims, extraPointers, opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaDouble(launchDims, extraPointers, opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaSingle(launchDims, extraPointers, opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *y, Nd4jLong *yShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaTriple(launchDims, extraPointers, opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaDouble(launchDims, extraPointers, opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}


Nd4jPointer NativeOps::initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {

    unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we don't synchronize at random initialization, it's safe to go unsync here
	// cudaStreamSynchronize(*stream);

    auto ptrDev = reinterpret_cast<unsigned long long *>(ptrToBuffer);
    auto buffer = new nd4j::random::RandomBuffer(seed, bufferSize, reinterpret_cast<uint64_t *>(ptrHost), reinterpret_cast<uint64_t *>(ptrDev));
    buffer->propagateToDevice(buffer, *stream);

    nd4j::DebugHelper::checkErrorCode(stream, "initRandom(...) failed A");

	// we generate sequence in the host memory
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

	// and copy it to gpu
    cudaMemcpyAsync(ptrDev, ptrHost, bufferSize * 8, cudaMemcpyHostToDevice, *stream);
    nd4j::DebugHelper::checkErrorCode(stream, "initRandom(...) failed B");

    return buffer;
}


void NativeOps::destroyRandom(Nd4jPointer ptrBuffer) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrBuffer);

    // FIXME: it's bad thing, but we can't know in advance, which stream(s) where using this generator in practice
    cudaDeviceSynchronize();

    delete buffer;
}

void NativeOps::refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    cudaStreamSynchronize(*stream);

    uint64_t *ptrDev = buffer->getDeviceBuffer();

	// update rng state
    buffer->setSeed(seed);
    buffer->setOffset(0);
    buffer->propagateToDevice(buffer, *stream);

	// refresh buffer on host size
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

	// copy back to gpu
    cudaMemcpyAsync(ptrDev, ptrHost, buffer->getSize() * 8, cudaMemcpyHostToDevice, *stream);
}

void NativeOps::reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    cudaStreamSynchronize(*stream);

	// update rng state
    buffer->reSeed(seed);
    buffer->setOffset(0);
    buffer->propagateToDevice(buffer, *stream);
}


/**
 *
 * @param npyArray
 * @return
 */
Nd4jPointer NativeOps::shapeBufferForNumpy(Nd4jPointer npyArray) {
    /*
	cnpy::NpyArray *arrPointer = reinterpret_cast<cnpy::NpyArray *>(npyArray);
	int *shapeBuffer = shape::shapeBufferOfNpy(*arrPointer);
	return reinterpret_cast<Nd4jPointer>(shapeBuffer);
     */
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int *shape = new unsigned int[arr.shape.size()];
    for(int i = 0; i < arr.shape.size(); i++) {
        shape[i] = arr.shape[i];
    }

    auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(),
                                               shape,
                                               arr.fortranOrder);
    delete[] shape;
    return reinterpret_cast<Nd4jPointer>(shapeBuffer);
}


/**
 *
 * @param npyArray
 * @return
 */
Nd4jPointer NativeOps::dataPointForNumpy(Nd4jPointer npyArray) {
    char *buff = reinterpret_cast<char *>(npyArray);
    //printf("Pointer contents %s\n",buff);
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    cnpy::NpyArray *arrPointer = &arr;
    char *data = arrPointer->data;
    if(arrPointer->wordSize == sizeof(float)) {
        float *floatData = reinterpret_cast<float *>(data);
        return reinterpret_cast<Nd4jPointer>(floatData);
    }
    else if(arrPointer->wordSize == sizeof(double)) {
        double *doubleData = reinterpret_cast<double *>(data);
        return reinterpret_cast<Nd4jPointer >(doubleData);
    }

    return reinterpret_cast<Nd4jPointer >(0);
}

/**
 * Load a numpy array from a file
 * and return it as an Nd4jPointer
 * @param path
 * @return
 */
Nd4jPointer NativeOps::numpyFromFile(std::string path) {
    /*cnpy::NpyArray arr = cnpy::npyLoad(path);
    return reinterpret_cast<Nd4jPointer >(&arr);
     */
	char *numpyBuffer = cnpy::loadFile(path.data());
	return reinterpret_cast<Nd4jPointer >(numpyBuffer);
}

void NativeOps::releaseNumpy(Nd4jPointer npyArray) {
    free(reinterpret_cast<void *>(npyArray));
}


/**
    * Return the length of a shape buffer
    * based on the pointer
    * @param buffer  the buffer pointer to check
    * @return
    */
int NativeOps::lengthForShapeBufferPointer(Nd4jPointer buffer) {
    auto shapeBuffer = reinterpret_cast<Nd4jLong *>(buffer);
    return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
  * Get the element size for a numpy array
  * @param npyArray  the numpy array's address
  * to get the length for
  * @return
  */
int NativeOps::elementSizeForNpyArray(Nd4jPointer npyArray) {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    cnpy::NpyArray *arrPointer = &arr;
    int size = arrPointer->wordSize;

    return size;
    /*
    cnpy::NpyArray *arr = reinterpret_cast<cnpy::NpyArray *>(npyArray);
    return arr->wordSize;
     */
}

/**
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */

Nd4jPointer NativeOps::pointerForAddress(Nd4jLong address) {
	return reinterpret_cast<Nd4jPointer >(address);
}

void NativeOps::tearDouble(Nd4jPointer *extras, double *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelDouble<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    nd4j::DebugHelper::checkErrorCode(stream, "tearDouble(...) failed");
}

void NativeOps::tearFloat(Nd4jPointer *extras, float *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelFloat<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    nd4j::DebugHelper::checkErrorCode(stream, "tearFloat(...) failed");
}

void NativeOps::tearHalf(Nd4jPointer *extras, float16 *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelHalf<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    nd4j::DebugHelper::checkErrorCode(stream, "tearHalf(...) failed");
}


void prescanArrayRecursive(Nd4jPointer *extras, int *z, int *x, int numElements, int level) {

    auto stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
    auto g_scanBlockSums = reinterpret_cast<int **>(&extras[2]);

    int blockSize = 512; // max size of the thread blocks
    int numBlocks = nd4j::math::nd4j_max<int>(1, static_cast<int>(ceil(static_cast<float>(numElements) / (2.f * blockSize))));
    int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (nd4j::isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = nd4j::floorPow2(numElements);

    int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    int numEltsLastBlock =
            numElements - (numBlocks-1) * numEltsPerBlock;
    int numThreadsLastBlock = nd4j::math::nd4j_max<int>(1, numEltsLastBlock / 2);
    int np2LastBlock = 0;
    int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);

        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    int extraSpace = numEltsPerBlock / NUM_BANKS;
    int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3 threads(numThreads, 1, 1);
    dim3 gridOnes(1, 1, 1);
    dim3 threadsOnes(numThreadsLastBlock, 1, 1);

    if (sharedMemSize < 2048)
        sharedMemSize = 2048;

    if (sharedMemLastBlock < 2048)
        sharedMemLastBlock = 2048;

    // execute the scan
    if (numBlocks > 1) {
        nd4j::prescanLauncher<true, false>(grid, threads, sharedMemSize, stream, z, x, g_scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            nd4j::prescanLauncher<true, true>(gridOnes, threadsOnes, sharedMemLastBlock, stream, z, x, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be sdded to each block to
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(extras, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        nd4j::uniformAdd<<<grid, threads, 1024, *stream>>>(z, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);

        if (np2LastBlock) {
            nd4j::uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(z, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        nd4j::prescanLauncher<false, false>(grid, threads, sharedMemSize, stream, z, x, 0, numThreads * 2, 0, 0);
    } else {
        nd4j::prescanLauncher<false, true>(grid, threads, sharedMemSize, stream, z, x, 0, numElements, 0, 0);
    }
}


void NativeOps::encodeThresholdP1Float(Nd4jPointer *extras, float *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::encoderKernelP1Float<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP1Float(...) failed");
}


void NativeOps::encodeThresholdP1Double(Nd4jPointer *extras, double *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::encoderKernelP1Double<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP1Double(...) failed");
}


void NativeOps::encodeThresholdP1Half(Nd4jPointer *extras, float16 *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP1Half<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP1Half(...) failed");
}

void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jLong N, int *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    //encoderKernelP2Float<<<numBlocks, blockSize , 1024 * sizeof(float), *stream>>>(dx, N, dz);

    // it
    prescanArrayRecursive(extraPointers, dz, dx + 1, (int) N, 0);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP2Int(...) failed");
}

void NativeOps::encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jLong N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::encoderKernelP3Float<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP3Float(...) failed");
}

void NativeOps::encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jLong N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::encoderKernelP3Double<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP3Double(...) failed");
}


void NativeOps::encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jLong N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::encoderKernelP3Half<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP3Half(...) failed");
}


void NativeOps::decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::decoderKernelFloat<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeThresholdFloat(...) failed");
}

void NativeOps::decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::decoderKernelDouble<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeThresholdDouble(...) failed");
}

void NativeOps::decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    nd4j::decoderKernelHalf<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeThresholdHalf(...) failed");
}


void NativeOps::execReduce3AllDouble(Nd4jPointer *extraPointers,
									 int opNum,
									 double *x,
									 Nd4jLong *xInfo,
									 double *extraParamsVals,
									 double *y,
									 Nd4jLong *yInfo,
									 double *result,
									 Nd4jLong *resultShapeInfoBuffer,
									 int *dimension,
									 int dimensionLength,
									 Nd4jLong *xTadShapeInfo,
                                     Nd4jLong *xOffsets,
									 Nd4jLong *yTadShapeInfo,
                                     Nd4jLong *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
    auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);


    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("D119 opNum:[%i]\n", opNum);

    int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

    dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[7], dimensionLength, sizeof(double), 2);

    if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
        printf("AD119 opNum:[%i]\n", opNum);

    reduce3AllDouble<<<launchDims.x, 512, (512 * 8 * 2 + 512), *stream>>>(
            opNum,
                    x,
                    xInfo,
                    y,
                    yInfo,
                    extraParamsVals,
                    result,
                    resultShapeInfoBuffer,
                    dimension,
                    dimensionLength,
                    1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execReduce3AllFloat(Nd4jPointer *extraPointers,
									int opNum,
									float *x,
									Nd4jLong *xInfo,
									float *extraParamsVals,
									float *y,
									Nd4jLong *yInfo,
									float *result,
									Nd4jLong *resultShapeInfoBuffer,
									int *dimension,
									int dimensionLength,
									Nd4jLong *xTadShapeInfo,
                                    Nd4jLong *xOffsets,
									Nd4jLong *yTadShapeInfo,
                                    Nd4jLong *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
    auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);


    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("F119 opNum:[%i]\n", opNum);

    int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

    dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[7], dimensionLength, sizeof(float), 2);

    if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
        printf("AF119 opNum:[%i]\n", opNum);

    reduce3AllFloat<<<launchDims.x, 512, (512 * 4 * 2 + 512), *stream>>>(
                opNum,
                        x,
                        xInfo,
                        y,
                        yInfo,
                        extraParamsVals,
                        result,
                        resultShapeInfoBuffer,
                        dimension,
                        dimensionLength,
                        1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execReduce3AllHalf(Nd4jPointer *extraPointers,
								   int opNum,
								   float16 *x,
								   Nd4jLong *xInfo,
								   float16 *extraParamsVals,
								   float16 *y,
								   Nd4jLong *yInfo,
								   float16 *result,
								   Nd4jLong *resultShapeInfoBuffer,
								   int *dimension,
								   int dimensionLength,
								   Nd4jLong *xTadShapeInfo,
                                   Nd4jLong *xOffsets,
								   Nd4jLong *yTadShapeInfo,
                                   Nd4jLong *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
    auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);


    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("H119 opNum:[%i]\n", opNum);

    int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
    float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

    dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[7], dimensionLength, sizeof(float16), 2);

    if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
        printf("AH119 opNum:[%i]\n", opNum);

    reduce3AllHalf<<<launchDims.x, 512, (512 * 2 * 2 + 512), *stream>>>(
            opNum,
                    x,
                    xInfo,
                    y,
                    yInfo,
                    extraParamsVals,
                    result,
                    resultShapeInfoBuffer,
                    dimension,
                    dimensionLength,
                    1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::sortFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[     1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    auto xLength = shape::length(hostXShapeInfo);
    auto xEWS = shape::elementWiseStride(hostXShapeInfo);

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
        int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
        int numBlocks = xLength / numThreads;
        if (xLength % numThreads > 0 || numBlocks == 0)
            numBlocks++;

        for (int k = 2; k <= xLength; k = 2*k) {
            for (int j = k >> 1; j > 0; j = j >> 1) {
                cudaBitonicSortFloat<<<numBlocks, numThreads, 512, *stream>>>(x, xShapeInfo, j, k, xLength, descending);
            }
        }
    } else {

#ifdef  __clang__
        if (1 > 0) {
#elif __GNUC__
        if ((xLength > 1024 * 1024 * 10) && xEWS == 1) {
            b40c::radix_sort::Enactor enactor;

            b40c::util::DoubleBuffer<float> sort_storage(x);

            enactor.Sort(sort_storage, xLength);

            // fire reverse op
            if (descending)
                execTransformFloat(extraPointers, 70, x, xShapeInfo, x, xShapeInfo, nullptr);
        } else {
#else
        if (1 > 0) {
#endif
            int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
            int numBlocks = xLength / numThreads;
            if (xLength % numThreads > 0 || numBlocks == 0)
                numBlocks++;

            numBlocks = nd4j::math::nd4j_min<int>(512, numBlocks);

            int max = 2, dg = 0;
            while (max < xLength) {
                max <<= 1;
                dg++;
            }
            max <<= 1;


            for (int window = 2; window < max; window<<=1) {
                int n = window;
                int rev = 0;
                do{
                    int half = n >> 1;
                    cudaSortFloat<<<numBlocks, numThreads, numThreads * 2 * sizeof(float), *stream>>>(x, xShapeInfo, n, xLength, rev, descending);
                    n>>=1;
                    rev = 1;
                } while(n > 1);
            }
        }
    }

    nd4j::DebugHelper::checkErrorCode(stream, "sortFloat(...) failed");
}


void NativeOps::sortDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    auto xLength = shape::length(hostXShapeInfo);
    auto xEWS = shape::elementWiseStride(hostXShapeInfo);

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
        int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
        int numBlocks = xLength / numThreads;
        if (xLength % numThreads > 0 || numBlocks == 0)
            numBlocks++;

        for (int k = 2; k <= xLength; k = 2*k) {
            for (int j = k >> 1; j > 0; j = j >> 1) {
                cudaBitonicSortDouble<<<numBlocks, numThreads, 512, *stream>>>(x, xShapeInfo, j, k, xLength, descending);
            }
        }
    } else {
#ifdef  __clang__
        if (1 > 0) {
#elif __GNUC__
        if ((xLength > 1024 * 1024 * 10) && xEWS == 1) {
            b40c::radix_sort::Enactor enactor;

            b40c::util::DoubleBuffer<double> sort_storage(x);

            enactor.Sort(sort_storage, xLength);

            // fire reverse op
            if (descending)
                execTransformDouble(extraPointers, 70, x, xShapeInfo, x, xShapeInfo, nullptr);
        } else {
#else
        if ( 1 > 0) {
#endif
            int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
            int numBlocks = xLength / numThreads;
            if (xLength % numThreads > 0 || numBlocks == 0)
                numBlocks++;

            numBlocks = nd4j::math::nd4j_min<int>(512, numBlocks);

            int max = 2, dg = 0;
            while (max < xLength) {
                max <<= 1;
                dg++;
            }
            max <<= 1;


            for (int window = 2; window < max; window<<=1) {
                int n = window;
                int rev = 0;
                do{
                    int half = n >> 1;
                    cudaSortDouble<<<numBlocks, numThreads, numThreads * 2 * sizeof(double), *stream>>>(x, xShapeInfo, n, xLength, rev, descending);
                    n>>=1;
                    rev = 1;
                } while(n > 1);
            }
        }
    }

    nd4j::DebugHelper::checkErrorCode(stream, "sortDouble(...) failed");
}


void NativeOps::sortHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    int xLength = shape::length(hostXShapeInfo);

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0)) {
        int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
        int numBlocks = xLength / numThreads;
        if (xLength % numThreads > 0 || numBlocks == 0)
            numBlocks++;

        for (int k = 2; k <= xLength; k = 2*k) {
            for (int j = k >> 1; j > 0; j = j >> 1) {
                cudaBitonicSortHalf<<<numBlocks, numThreads, 512, *stream>>>(x, xShapeInfo, j, k, xLength, descending);
            }
        }
    } else {
        // half is incompatible with radix, so only bitonic here

        int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
        int numBlocks = xLength / numThreads;
        if (xLength % numThreads > 0 || numBlocks == 0)
            numBlocks++;

        numBlocks = nd4j::math::nd4j_min<int>(512, numBlocks);

        int max = 2, dg = 0;
        while (max < xLength) {
            max <<= 1;
            dg++;
        }
        max <<= 1;


        for (int window = 2; window < max; window<<=1) {
            int n = window;
            int rev = 0;
            do{
                int half = n >> 1;
                cudaSortHalf<<<numBlocks, numThreads, numThreads * 2 * sizeof(float16), *stream>>>(x, xShapeInfo, n, xLength, rev, descending);
                n>>=1;
                rev = 1;
            } while(n > 1);
        }
    }

    nd4j::DebugHelper::checkErrorCode(stream, "sortHalf(...) failed");
}

void NativeOps::sortTadFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaSortTadFloat<<<512, 512, 1088 * sizeof(float), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    nd4j::DebugHelper::checkErrorCode(stream, "sortTadFloat(...) failed");
}

void NativeOps::sortTadHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaSortTadHalf<<<512, 512, 1088 * sizeof(float16), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    nd4j::DebugHelper::checkErrorCode(stream, "sortTadHalf(...) failed");
}

void NativeOps::sortTadDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaSortTadDouble<<<512, 512, 1088 * sizeof(double), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    nd4j::DebugHelper::checkErrorCode(stream, "sortTadDouble(...) failed");
}

void NativeOps::sortCooIndicesFloat(Nd4jPointer *extraPointers, Nd4jLong *indices, float *values, Nd4jLong length, int rank) {
	throw std::runtime_error("Not implemented yet");
}

void NativeOps::sortCooIndicesDouble(Nd4jPointer *extraPointers, Nd4jLong *indices, double *values, Nd4jLong length, int rank) {
	throw std::runtime_error("Not implemented yet");
}

void NativeOps::sortCooIndicesHalf(Nd4jPointer *extraPointers, Nd4jLong *indices, float16 *values, Nd4jLong length, int rank) {
	throw std::runtime_error("Not implemented yet");
}


Nd4jLong NativeOps::encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto *hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapFloat<<<512, 512, 512 * 2 * sizeof(float) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeBitmapFloat(...) failed");

    Nd4jLong result = (Nd4jLong) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

Nd4jLong NativeOps::encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapDouble<<<512, 512, 512 * 2 * sizeof(double) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "encodeBitmapDouble(...) failed");

    Nd4jLong result = (Nd4jLong) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

Nd4jLong NativeOps::encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapHalf<<<512, 512, (512 * sizeof(float16)) + (512 * sizeof(int)) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    nd4j::DebugHelper::checkErrorCode(stream, "execBitmapHalf(...) failed");

    Nd4jLong result = (Nd4jLong) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

void NativeOps::decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaDecodeBitmapFloat<<<512, 512, 512 * sizeof(float) + 384, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeBitmapFloat(...) failed");
}


void NativeOps::decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaDecodeBitmapDouble<<<512, 512, 512 * sizeof(double) + 384, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeBitmapDouble(...) failed");
}


void NativeOps::decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

    cudaDecodeBitmapHalf<<<512, 512, 512 * sizeof(float16) + 384, *stream>>>(dx, N, dz);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeBitmapDouble(...) failed");
}

Nd4jLong* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length) {
	return nullptr;
}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jLong* ptrMap, Nd4jLong length) {

}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer protoBufferPointer) {
	return nullptr;
}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, const char *fileName) {
	return nullptr;
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphHalf(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}


nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphDouble(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}
		


const char* NativeOps::getAllCustomOps() {
	return nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}


template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    nd4j::graph::VariableSpace<T> varSpace;
    Context<T> block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = reinterpret_cast<Nd4jLong *>(inputShapes[e]);
        auto buffer_ = reinterpret_cast<T *>(inputBuffers[e]);
        auto array = new nd4j::NDArray<T>(buffer_, shape_);
        array->triggerAllocationFlag(false, false);

        // block should contain references to proper variable
        varSpace.putVariable(1, e, array);
        block.pickInput(1, e);

        inShapes.push_back(shape_);
    }

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    if (varSpace.workspace() != nullptr)
        shapeList->detach();

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}


template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    nd4j::graph::Context<T> block(1);
	nd4j::ShapeList inShapes;

	for (int e = 0; e < numIArgs; e++)
		block.getIArguments()->push_back(iArgs[e]);

	for (int e = 0; e < numTArgs; e++)
		block.getTArguments()->push_back(tArgs[e]);

	for (int e = 0; e < numInputShapes; e++)
		inShapes.push_back(static_cast<Nd4jLong *>(inputShapes[e]));

	auto shapeList = op->calculateOutputShape(&inShapes, block);

	return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

	return _calculateOutputShapes<float>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

	return _calculateOutputShapes<float16>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

	return _calculateOutputShapes<double>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
static FORCEINLINE Nd4jStatus realExec(nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
	if (op == nullptr)
		nd4j_printf("Can't find requested operation: [%lld]\n", hash);

	// we're using the same fake nodeId everywhere here

	std::vector<nd4j::NDArray<T>*> inputs(numInputs);
	std::vector<nd4j::NDArray<T>*> outputs(numOutputs);
	std::vector<T> ttArgs(numTArgs);
	std::vector<Nd4jLong> iiArgs(numIArgs);

	// filling block now with inputs
	for (int e = 0; e < numInputs; e++) {
		auto buffer = reinterpret_cast<T *>(inputBuffers[e]);
		auto shape = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

		inputs[e] = new nd4j::NDArray<T>(buffer, shape);
	}

	// if not inplace - transferring output arrays

	if (!isInplace)
		for (int e = 0; e < numOutputs; e++) {
			auto buffer = reinterpret_cast<T *>(outputBuffers[e]);

			// we want to keep original output shape intact
			auto shape = shape::copyShape(reinterpret_cast<Nd4jLong *>(outputShapes[e]));

			auto array = new nd4j::NDArray<T>(buffer, shape);
			outputs[e] = array;

			// and we want to release shape copy once we're done
			array->triggerAllocationFlag(false, true);
		}

	for (int e = 0; e < numIArgs; e++)
		iiArgs[e] = iArgs[e];


	for (int e = 0; e < numTArgs; e++)
		ttArgs[e] = tArgs[e];


	// hypothetically at this point we have everything filled
	auto result = op->execute(inputs, outputs, ttArgs, iiArgs, isInplace);
	//auto result = op->execute(inputs, ttArgs, iiArgs, isInplace);


	if (!isInplace)
		for (int e = 0; e < numOutputs; e++) {
			//shape::printShapeInfoLinear("JVM output shape", (int *) outputShapes[e]);
			//shape::printShapeInfoLinear("C++ output shape", (int *) outputs[e]->shapeInfo());
			//outputs[e]->printIndexedBuffer("C++ raw output");
			//outputs[e]->printBuffer("C++ indexed output");

			if (outputs[e]->ordering() != shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])))
				outputs[e]->streamline(shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])));
		}

/*
    if (!isInplace) {
        if (result->size() != numOutputs) {
            return ND4J_STATUS_BAD_OUTPUT;
        }

        for (int e = 0; e < numOutputs; e++) {
            auto buffer = (T *) outputBuffers[e];
            auto shape = (int *) outputShapes[e];
            nd4j::NDArray<T> tmp(buffer, shape);

            if (tmp.lengthOf() != result->at(e)->lengthOf()) {
                nd4j_printf("Provided output array for [%s] has length of %i, but actual result has length of %i\n", op->getOpName()->c_str(), tmp.lengthOf(), result->at(e)->lengthOf());
                return ND4J_STATUS_BAD_OUTPUT;
            }

            tmp.assign(result->at(e));
        }
    } else {
        // if op is inplace, our ResultSet holds pointers
        result->purge();
    }


    delete result;

*/

	for (auto v: inputs)
		delete v;

	for (auto v: outputs)
		delete v;

	return Status::OK();
}


int NativeOps::execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

	return realExec<float>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

	return realExec<double>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

	return realExec<float16>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::registerGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<float>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

int NativeOps::registerGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<double>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

int NativeOps::registerGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<float16>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

template <typename T>
static VariablesSet<T>* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	auto graph = nd4j::graph::GraphHolder::getInstance()->pullGraph<T>(graphId);
	auto varSpace = graph->getVariableSpace()->clone();

	std::vector<nd4j::NDArray<T> *> handles;

	for (int e = 0; e < numInputs; e++) {
		auto idx = inputIndices[e];

		// we'll delete this array later, together with cloned VariableSpace
		auto array = new nd4j::NDArray<T>(reinterpret_cast<T *>(inputBuffers[e]), reinterpret_cast<Nd4jLong *>(inputShapes[e]));
		handles.emplace_back(array);

		if (varSpace->hasVariable(idx)) {
			auto var = varSpace->getVariable(idx);
			if (var->hasNDArray())
				delete var->getNDArray();

			var->setNDArray(array);
		} else
			varSpace->putVariable(idx, array);
	}

	auto result = nd4j::graph::GraphExecutioner<T>::execute(graph, varSpace);
	auto varSet = new nd4j::graph::VariablesSet<T>(result);

	if (result == ND4J_STATUS_OK) {
		// pull back results, and provide them
		auto outputs = graph->fetchOutputs();
		for (int e = 0; e < outputs->size(); e++) {
			// we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
			std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

			auto var = varSpace->getVariable(varId);

			varSet->push_back(var->clone());
		}

		delete outputs;
	}

	delete varSpace;

	return varSet;
}

VariablesSet<float>* NativeOps::executeStoredGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<float>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<float16>* NativeOps::executeStoredGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<float16>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<double>* NativeOps::executeStoredGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<double>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

int NativeOps::unregisterGraph(Nd4jPointer *extraPointers, Nd4jLong graphId) {

	nd4j::graph::GraphHolder::getInstance()->dropGraphAny(graphId);

	return ND4J_STATUS_OK;
}

void NativeOps::deletePointerArray(Nd4jPointer pointer) {
    Nd4jPointer *ptr = reinterpret_cast<Nd4jPointer *>(pointer);
    delete[] ptr;
}

void NativeOps::deleteIntArray(Nd4jPointer pointer) {
	auto ptr = reinterpret_cast<int *>(pointer);
	delete[] ptr;
}

void NativeOps::deleteLongArray(Nd4jPointer pointer) {
	auto ptr = reinterpret_cast<Nd4jLong *>(pointer);
	delete[] ptr;
}

template <typename T>
static void deleteVariablesSetT(Nd4jPointer pointer) {
	nd4j::graph::VariablesSet<T>* ptr = reinterpret_cast<nd4j::graph::VariablesSet<T>*>(pointer);
	delete ptr;
}

void NativeOps::deleteVariablesSetFloat(Nd4jPointer pointer) {
	deleteVariablesSetT<float>(pointer);
}

void NativeOps::deleteVariablesSetHalf(Nd4jPointer pointer) {
	deleteVariablesSetT<float16>(pointer);
}

void NativeOps::deleteVariablesSetDouble(Nd4jPointer pointer) {
	deleteVariablesSetT<double>(pointer);
}

void NativeOps::deleteShapeList(Nd4jPointer shapeList) {
    nd4j::ShapeList* list = reinterpret_cast<nd4j::ShapeList*>(shapeList);

    list->destroy();
    delete list;
}

const char* NativeOps::getAllOperations() {
    return nd4j::OpTracker::getInstance()->exportOperations();
}

Nd4jPointer NativeOps::getGraphStateHalf(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float16>(id);
}

Nd4jPointer NativeOps::getGraphStateFloat(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float>(id);
}

Nd4jPointer NativeOps::getGraphStateDouble(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<double>(id);
}

void NativeOps::deleteGraphStateHalf(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<float16> *>(state);
    delete stateP;
}

void NativeOps::deleteGraphStateFloat(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<float> *>(state);
    delete stateP;
}

void NativeOps::deleteGraphStateDouble(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<double> *>(state);
    delete stateP;
}

template <typename T>
Nd4jStatus execCustomOpWithScope(Nd4jPointer *extraPointers, nd4j::graph::GraphState<T> *state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    /**
     * That's basically exec, with VariableSpace provided in GraphState:
     * depending on operation (i.e. while of if), different logic executors could be used
     */

    auto graph = state->graph();
    auto varSpace = state->variableSpace();

    // Node is dynamically created, and has nothing beyond it: only inputs and outputs
    // this node has id of 0, and inputs are
    nd4j::graph::Node<T> node(OpType_LOGIC, opHash, 0);

    // mapping inputs
    for (int e = 0; e < numInputs; e++) {
        auto buffer = reinterpret_cast<T *>(inputBuffers[e]);
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        auto array = new nd4j::NDArray<T>(buffer, shapeInfo, varSpace->workspace());

        // now we just put array to VarSpace
        varSpace->putVariable(0, e, array);
        node.pickInput(0, e);
    }

    // mapping scopes
    for (int e = 0; e < numScopes; e++) {
        // we should check scope existence in GraphState/Graph
        int scopeId = (int) scopes[e];
        if (!state->hasScope(scopeId)) {
            // nd4j_printf("execCustomOpWithScope: referenced scope [%i] doesn't exist\n", scopeId);
            return Status::THROW();
        }
        node.pickInput(scopeId, 0);
    }

    auto result = LogicExecutor<T>::processNode(graph, &node);
    if (result != Status::OK())
        return result;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = reinterpret_cast<T *>(outputBuffers[e]);
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(outputShapes[e]);

        nd4j::NDArray<T> array(buffer, shapeInfo, varSpace->workspace());

        // now we just put array to VarSpace to the same ID
        //varSpace->putVariable(0, e, array);

        auto t = varSpace->getVariable(0, e)->getNDArray();
        array.assign(t);
    }

    // removing input variables
    for (int e = 0; e < numInputs; e++) {
        varSpace->dropVariable(0, e);
    }


    // after some bla-bla-bla we should have Graph and Node for current op
    return Status::OK();
}

Nd4jStatus NativeOps::execCustomOpWithScopeHalf(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float16>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<float16> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeFloat(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<float> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeDouble(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<double>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<double> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

void NativeOps::deleteResultWrapper(Nd4jPointer ptr) {
	// just 0 room for compiler s@!t
	auto p = reinterpret_cast<nd4j::graph::ResultWrapper *>(ptr);
	delete p;
}


int NativeOps::estimateThresholdFloat(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
	throw std::runtime_error("estimateThresholdFloat: Not implemented yet");
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jLong N, int dstType, Nd4jPointer z) {
 	auto dx = reinterpret_cast<void *>(x);
	auto dz = reinterpret_cast<void *>(z);

    if (srcType == ND4J_FLOAT8) {
        if (dstType == ND4J_FLOAT8) {
            // convertKernel<double, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<nd4j::float8, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //convertKernel<nd4j::int8, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: eventually we might want to add it
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int8, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: still might want to add
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<nd4j::uint8, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<float16, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<float16, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<float16, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<float16, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<float16, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<float16, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: .... ^^^
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<float16, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<float16, double>(extras, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            //nd4j::convertToThreshold<float16>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT16) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO...
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<nd4j::int16, double>(extras, dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<float, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<float, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<float, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<float, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<float, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<float, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<float, double>(extras, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            //nd4j::convertToThreshold<float>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGenericCuda<double, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<double, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<double, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<double, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<double, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<double, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<double, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //
        } else if (dstType == ND4J_THRESHOLD) {
            //nd4j::convertToThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_THRESHOLD) {
        if (dstType == ND4J_FLOAT16) {
            //nd4j::convertFromThreshold<float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT32) {
            //nd4j::convertFromThreshold<float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //nd4j::convertFromThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}