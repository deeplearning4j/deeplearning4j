
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
    SyncInfo *sync = (SyncInfo *) data;

    printf("Finished stream: [%i], kernel call: [%i]\n", sync->streamId, sync->callId);
}

// this method just does type conversion in fancy way
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
dim3 getReduceLaunchParams(int deviceId, int *xShapeInfo, int *tadShapeInfo, cudaFuncAttributes funcAttr, int dimensionLength, int elementSize, int reductionSize) {

	int tadLength = 0;
	int numTads = 0;
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

	int xRank = shape::rank(xShapeInfo);
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
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	Nd4jIndex n = shape::length(hostXShapeInfo);

	dim3 launchDims = getOptimalDimensions<T>(n,attributes, properties);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
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
//	std::thread::id threadId;

public:
	ScalarShapeInformation(cudaStream_t stream) {
		int *scalarDimensionBuff = (int *) malloc(sizeof(int));
		scalarDimensionBuff[0] = MAX_DIMENSION;
		scalarDimension = nd4j::buffer::createBuffer(scalarDimensionBuff,1, stream);
		scalarShapeInfo = createScalarBuffer(stream);
//		threadId = std::this_thread::get_id();

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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D1 opNum:[%i]\n", opNum);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], 1, sizeof(double), 3);

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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D2 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[27], dimensionLength, sizeof(double), 3);

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

	if (nd4j::Environment::getInstance()->isDebug())
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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *deviceTADOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);


	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[26],  dimensionLength, sizeof(double), 2);

	// this macro builds bunch of IF/ELSE selectors for kernel launch

    DISPATCH_SIMPLE(broadcastSimple, double, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ), OPS_A(BROADCAST_OPS))

	if (nd4j::Environment::getInstance()->isDebug())
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
		int *xShapeInfo,
		double *y,
		int *yShapeInfo,
		double *result,
		int *resultShapeInfo,
		double *extraParams,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes) {
	/*
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
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

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
    */
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
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D7 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	if (opNum == 19) {
		execReduceDouble(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo);
	}

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], 1, sizeof(double), 1);

	// this macro builds bunch of IF/ELSE selectors for kernel launch

    DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D8 opNum:[%i]\n", opNum);

	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	if (opNum == 19) {
		execReduceDouble(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
        //checkCudaErrors(cudaStreamSynchronize(*stream));
	}

	/**
	 * We have separate kernels, optimized for different number of dimensions for reductions
	 */
	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(double), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric1D, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))

	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[33], dimensionLength, sizeof(double), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric3D, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[22], dimensionLength, sizeof(double), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGenericXD, double, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	}

	if (nd4j::Environment::getInstance()->isDebug())
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
    DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D11 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

    double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D12 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[20]);

    functions::scalar::ScalarTransform<double>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, scalar, extraParams, n);

	if (nd4j::Environment::getInstance()->isDebug())
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

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[19]);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleShaped, double, PARAMS(scalar, x, xShapeInfo, extraParams, result, resultShapeInfo, allocPointer), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<double>::executeCudaShaped(launchDims, extraPointers, opNum, x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);

	if (nd4j::Environment::getInstance()->isDebug())
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


    printf("Unsupported operation: scalarIndices\n");
    /*
}
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D15 opNum:[%i]\n", opNum);

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

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
    */
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,bool biasCorrected) {
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D17 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
		int xStride,
		double *z,
		int zStride,
		double *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D19 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[16]);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(transformStrided, double, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D20 opNum:[%i]\n", opNum);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

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


    int *devTadShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
    Nd4jIndex *devTadOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);


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

			// this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(transformShaped, double, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))
		} else {
			// going for blockwise specials
			// we'll do some pointers mangling here, and execute kernels one by one

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

					// TODO: we could get rid of this one eventually
					prepareShapeBuffer <<<1, 1, 128, *stream>>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (nd4j::Environment::getInstance()->isDebug())
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

                    checkCudaErrors(cudaStreamSynchronize(*stream));

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
						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
                        Nd4jIndex *tadMaxOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<double *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceDouble(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						if (nd4j::Environment::getInstance()->isDebug())
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute filler
						fillDimensionalIsMaxDouble<<<blockLimit, 64, funcAttributes[37].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );

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
            cudaMalloc((void **)&maskedAllocPointer, length * launchDims.x * sizeof(double));
        }

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(transformShaped, double, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (nd4j::Environment::getInstance()->isDebug() || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

		// release Histogram memory
        if (opNum == 48) {
            cudaFree((void *)maskedAllocPointer);
        }
	}
	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D21 opNum:[%i]\n", opNum);

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

	if (nd4j::Environment::getInstance()->isDebug())
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
	if (nd4j::Environment::getInstance()->isDebug())
		printf("F1 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float), 4);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose() && launchDims.x == 1)
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

	// once again - since we return scalar value in this method, we should block this kernel launch
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
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H1 opNum:[%i]\n", opNum);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], 1, sizeof(float16), 8);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose() && launchDims.x == 1)
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

	// blocking for scalar output
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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F2 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float), 4);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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

	if (nd4j::Environment::getInstance()->isDebug())
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

	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H2 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[13], dimensionLength, sizeof(float16), 8);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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

	if (nd4j::Environment::getInstance()->isDebug())
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *deviceTADOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float), 0);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
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
	if (nd4j::Environment::getInstance()->isDebug())
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
	int *deviceTADShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *deviceTADOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);


	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H3 opNum:[%i]\n", opNum);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[12], 1, sizeof(float16), 0);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(broadcastSimple, float16, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, deviceTADShapeInfo, deviceTADOffsets, deviceTADShapeInfoZ, deviceTADOffsetsZ), OPS_A(BROADCAST_OPS))

	if (nd4j::Environment::getInstance()->isDebug())
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
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float>::execudaCudaStrided(launchDims, extraPointers, opNum, dx, xStride, y, yStride, result, resultStride, extraParams, n);
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
		int *xShapeInfo,
		float *y,
		int *yShapeInfo,
		float *result,
		int *resultShapeInfo,
		float *extraParams,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes){
    /*
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F5 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[10], 1, sizeof(float), 0);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
    */
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

    /*
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostYShapeInfo = reinterpret_cast<int *>(extraPointers[7]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H5 opNum:[%i]\n", opNum);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[10], 1, sizeof(float16), 0);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
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

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
    */
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
    dim3 launchDims(512, 512, 2048);

    functions::pairwise_transforms::PairWiseTransform<float>::execudaCudaShaped(launchDims, extraPointers, opNum, dx, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, extraParams);;

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
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

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
    DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

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
    DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F8 opNum:[%i]\n", opNum);

	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

//	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float), 1);

	if (opNum == 19) {
		execReduceFloat(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
	}

	// we call different kernels optimized for different number of dimensions in TAD
	if (dimensionLength == 1) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(float), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric1D, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	} else if (shape::rank(hostTADShapeInfo) <= 3) {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(float), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric3D, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	} else {
        dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[32], dimensionLength, sizeof(float), 2);

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGenericXD, float, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	}

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H8 opNum:[%i]\n", opNum);

	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[8], dimensionLength, sizeof(float16), 1);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH8 opNum:[%i]\n", opNum);

	if (opNum == 19) {
		execReduceHalf(extraPointers, 3, x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength);
	}

	// calling different kernels, depending on number of dimensions in TAD
	if (dimensionLength == 1) {

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric1D, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	} else if (shape::rank(hostTADShapeInfo) <= 3) {

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGeneric3D, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	} else {

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(reduceSimpleGenericXD, float16, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionPointer, deviceTADShapeInfo, deviceTADOffsets), OPS_A(REDUCE_OPS))
	}

	if (nd4j::Environment::getInstance()->isDebug())
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
float NativeOps::execReduceScalarFloat(
		Nd4jPointer *extraPointers,
		int opNum,
		float *x,
		int *xShapeInfo,
		float *extraParams){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);

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
    DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

	// blocking this one
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
    DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, resultPointer, nullptr, nullptr,1 , reductionPointer, deviceTADShapeInfo), OPS_A(REDUCE_OPS))

	// blocking call
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

    int *yDeviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[12]);
	Nd4jIndex *yDeviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);

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

	if (nd4j::Environment::getInstance()->isDebug())
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

	    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        functions::scalar::ScalarTransform<float>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, scalar, extraParams, n);

	    if (nd4j::Environment::getInstance()->isDebug())
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

    dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[6]);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarSimpleStrided, float16, PARAMS(n, scalar, x, xStride, extraParams, result, resultStride, allocPointer), OPS_A(SCALAR_OPS))
    float16 sc = (float16) scalar;
    functions::scalar::ScalarTransform<float16>::executeCudaStrided(launchDims, extraPointers, opNum, x, xStride, result, resultStride, sc, extraParams, n);

    if (nd4j::Environment::getInstance()->isDebug())
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
	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	Nd4jIndex n = shape::length(hostXShapeInfo);

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

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebug())
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
		float scalar,
		float *extraParams,
		int *xIndexes,
		int *resultIndexes){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	Nd4jIndex n = shape::length(hostXShapeInfo);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F15 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[4]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF15 opNum:[%i]\n", opNum);
/*
	scalarFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			n,
			scalar,
			x,
			extraParams,
			result,
			resultIndexes, allocPointer);
*/
	if (nd4j::Environment::getInstance()->isDebug())
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

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

    Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

	dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostTADShapeInfo, funcAttributes[1], 1, sizeof(float), 8);

	// we limit grid size for SummaryStats calls
    launchDims.x = nd4j::math::nd4j_min<int>(512, launchDims.x);

	return functions::summarystats::SummaryStatsReduce<float>::execSummaryStatsReduceScalar(launchDims, extraPointers, opNum, x, xShapeInfo, extraParams, biasCorrected);
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

	float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

    Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
		int *xShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,bool biasCorrected){
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
	int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);

	int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);
	int *deviceTADShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
	Nd4jIndex *deviceTADOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

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
		int xStride,
		float *z,
		int zStride,
		float *extraParams,
		Nd4jIndex n) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F19 opNum:[%i]\n", opNum);


	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(transformStrided, float, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H19 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[2]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH19 opNum:[%i], xLength: [%i]\n", opNum, shape::length(hostXShapeInfo));

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(transformStrided, float16, PARAMS(n, dx, xStride, extraParams, z, zStride, allocPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F20 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	float *specialPointer = reinterpret_cast<float *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float * special = (float *) maxShapeBuffer + (MAX_RANK * 2 + 4);

    int *maskedAllocPointer = allocPointer;

    int *devTadShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
    Nd4jIndex *devTadOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);


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

			// this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(transformShaped, float, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))

		} else {
			// going for blockwise specials

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

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceFloat(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastFloat(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// exp 3
					execTransformFloat(extraPointers, 3, result, resultShapeInfo, result, resultShapeInfo, extraParams);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceFloat(tempPointers, 1, result, resultShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];

					// divide 3
					execBroadcastFloat(tempPointers, 3, result, resultShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// log 3
					if (opNum == 40)
						execTransformFloat(extraPointers, 5, result, resultShapeInfo, result, resultShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformFloat(extraPointers, 42, result, resultShapeInfo, result, resultShapeInfo, extraParams);


					checkCudaErrors(cudaStreamSynchronize(*stream));

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

                        checkCudaErrors(cudaStreamSynchronize(*stream));
					} else {
						// going for dimension-based IsMax
						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
                        Nd4jIndex *tadMaxOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceFloat(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						if (nd4j::Environment::getInstance()->isDebug())
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxFloat<<<blockLimit, 64, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );


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
    } else {
		// we're enforcing larger grids for Col2Im & Im2Col
		// TODO: for high-end gpus we might use higher values here
        if (opNum == 37 || opNum == 36) {
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
            cudaMalloc((void **) &maskedAllocPointer, length * launchDims.x * sizeof(float));
        }

		DISPATCH_SIMPLE(transformShaped, float,
                        PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo,
                               shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (nd4j::Environment::getInstance()->isDebug() || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

		// release memory chunk
        if (opNum == 48) {
            cudaFree((void *) maskedAllocPointer);
        }
    }

    if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H20 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);
    int *maskedAllocPointer = allocPointer;

	float16 *specialPointer = reinterpret_cast<float16 *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float16 * special = (float16 *) maxShapeBuffer + (MAX_RANK * 2 + 4);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, hostZShapeInfo, funcAttributes[1]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH20 opNum:[%i]\n", opNum);

    int *devTadShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
    Nd4jIndex *devTadOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);


    // simple trick to get workaround over reductions into scalar
	// SoftMax, SoftMaxDerivative, LogSoftMax, IsMax
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			int length = shape::length(hostXShapeInfo);
			int block = nd4j::math::nd4j_min<int>(length, 256);

            launchDims.x = 1;
            launchDims.y = block;
            launchDims.z += (block * sizeof(float16) * 4);

			// this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(transformShaped, float16, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), allocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))

		} else {
			// going for blockwise specials

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

					// FIXME: fix this
					prepareShapeBuffer <<< 1, 1, 128, *stream >>> (dimension, maxDimension, maxShapeBuffer, shape[0]);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					//shape::printShapeInfo(maxShapeBuffer);
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					// max 3
					execReduceHalf(tempPointers, 3, dx, xShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					tempPointers[8] = extraPointers[8];
					tempPointers[9] = extraPointers[9];
					tempPointers[10] = extraPointers[10];
					tempPointers[11] = extraPointers[11];
                    tempPointers[12] = extraPointers[10];
                    tempPointers[13] = extraPointers[11];


					// sub 1
					execBroadcastHalf(tempPointers, 1, dx, xShapeInfo, special,
									   maxShapeBuffer, result, resultShapeInfo, dimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// exp 3
					execTransformHalf(extraPointers, 3, result, resultShapeInfo, result, resultShapeInfo, extraParams);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));


					tempPointers[8] = tempPointers[7];
					tempPointers[9] = extraPointers[12];
					tempPointers[10] = extraPointers[13];
					tempPointers[11] = extraPointers[14];

					//sum 1
					execReduceHalf(tempPointers, 1, result, resultShapeInfo, extraParams, special,
									maxShapeBuffer, maxDimension, 1);

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

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
                        if (nd4j::Environment::getInstance()->isDebug())
                            checkCudaErrors(cudaStreamSynchronize(*stream));

                        execTransformHalf(tempPointers, 47, result, resultShapeInfo, result, resultShapeInfo, extraParams);
                    }

					if (nd4j::Environment::getInstance()->isDebug())
						checkCudaErrors(cudaStreamSynchronize(*stream));

					// log 3
					if (opNum == 40)
						execTransformHalf(extraPointers, 5, result, resultShapeInfo, result, resultShapeInfo, extraParams);
					else if (opNum == 39)
						execTransformHalf(extraPointers, 42, result, resultShapeInfo, result, resultShapeInfo, extraParams);


					checkCudaErrors(cudaStreamSynchronize(*stream));

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
						int *tadMaxShapeInfo = reinterpret_cast<int *> (extraPointers[10]);
                        Nd4jIndex *tadMaxOffsets = reinterpret_cast<Nd4jIndex *> (extraPointers[11]);
						int *dimension = reinterpret_cast<int *> (extraPointers[15]);
                        special = reinterpret_cast<float16 *>(extraPointers[17]);
                        int dimensionLength = getDeviceId(extraPointers[18]);

						// we call for IMax on specified dimension
						execIndexReduceHalf(extraPointers, 0, dx, xShapeInfo, extraParams, special, hostYShapeInfo, dimension, dimensionLength);

						if (nd4j::Environment::getInstance()->isDebug())
							checkCudaErrors(cudaStreamSynchronize(*stream));

						// at this point, all IMax indexes are gathered, and we execute
						fillDimensionalIsMaxHalf<<<blockLimit, 64, funcAttributes[36].sharedSizeBytes, *stream>>>(special, hostYShapeInfo, result, resultShapeInfo, tadMaxShapeInfo, dimension, dimensionLength, tadMaxOffsets );


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
		// Im2Col & Col2Im enforced grids
        if (opNum == 37 || opNum == 36) {
            launchDims.x = 512;
            launchDims.y = 512;
            launchDims.z += 512 * sizeof(float16);
        } else if (opNum == 70) {
            // we'll be using shared memory to speed up reverse

            launchDims.z += launchDims.y * sizeof(float);
        }

		// Histogram op requires additional memory chunk
        if (opNum == 48) {
            int length = shape::length(hostZShapeInfo);
            cudaMalloc((void **)&maskedAllocPointer, length * launchDims.x * sizeof(float16));
        }

		// this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(transformShaped, float16, PARAMS(dx, xShapeInfo, shape::rank(hostXShapeInfo), extraParams, result, resultShapeInfo, shape::rank(hostZShapeInfo), maskedAllocPointer, reductionPointer, devTadShapeInfo, devTadOffsets), OPS_A(TRANSFORM_OPS))


        // we need guaranteed sync here, due to temp memory release
        if (nd4j::Environment::getInstance()->isDebug() || opNum == 48)
            checkCudaErrors(cudaStreamSynchronize(*stream));

		// release that histogram memory chunk
        if (opNum == 48) {
            cudaFree((void *)maskedAllocPointer);
        }
	}

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F21 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[0]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF21 opNum:[%i]\n", opNum);

	transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			dx,
			xShapeInfo,  shape::rank(hostXShapeInfo),
			extraParams,
			result,
			resultIndexes, allocPointer, reductionPointer);

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H21 opNum:[%i]\n", opNum);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

	dim3 launchDims = getFlatLaunchParams(getDeviceId(extraPointers[2]), hostXShapeInfo, nullptr, funcAttributes[0]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH21 opNum:[%i]\n", opNum);

	transformHalfIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
					dx,
					xShapeInfo,  shape::rank(hostXShapeInfo),
					extraParams,
					result,
					resultIndexes, allocPointer, reductionPointer);

	if (nd4j::Environment::getInstance()->isDebug())
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
		}
	} else {
		int rank = shape::rank(inputShapeInfo);
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF222 opNum:[7]\n");

	flattenKernelFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("H22 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo), 2, funcAttributes[30]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AH222 opNum:[7]\n");

	flattenKernelHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	if (nd4j::Environment::getInstance()->isDebug())
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

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("D30 opNum:[7]\n");

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hostYShapeInfo),  2, funcAttributes[34]);

	flattenKernelDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(offset, order, result, resultShapeInfo, input, inputShapeInfo, allocPointer);

	if (nd4j::Environment::getInstance()->isDebug())
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

	cudaFuncGetAttributes(&funcAttributes[0], (void *)transformFloatIndexes);

	//void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[1], transformFloatIndexes);

	//void (*transformFloatPointer2)(int opNum, Nd4jIndex n, float *dy, int incy, float *params, float *result,int resultStride, int *allocationPointer, float *reductionPointer) = transformFloat;
	// FIXME
    cudaFuncGetAttributes(&funcAttributes[2], transformFloatIndexes);

	//cudaFuncGetAttributes(&funcAttributes[3], (void *)functions::summarystats::summaryStatsReduceFloat);

	//cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

//	void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo, int xRank, float *params, float *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarFloat;
//	cudaFuncGetAttributes(&funcAttributes[5], scalarFloatIndexes);

//	void (*scalarFloatPointer2)(int opNum, Nd4jIndex n,float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocPointer) = scalarFloat;
//	cudaFuncGetAttributes(&funcAttributes[6], scalarFloatIndexes);

	cudaFuncGetAttributes(&funcAttributes[7], reduce3Float);

	cudaFuncGetAttributes(&funcAttributes[8], reduceSimpleGenericXD_0_float);
//	printf("reduceFloat regs: [%i], static shmem: [%i]\n", funcAttributes[8].numRegs, funcAttributes[8].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[28], reduceSimpleGeneric1D_0_float); // 1D
//	printf("reduceFloat1D regs: [%i], static shmem: [%i]\n", funcAttributes[28].numRegs, funcAttributes[28].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[29], reduceSimpleGeneric3D_0_float); // 6D
//	printf("reduceFloat6D regs: [%i], static shmem: [%i]\n", funcAttributes[29].numRegs, funcAttributes[29].sharedSizeBytes);

	cudaFuncGetAttributes(&funcAttributes[30], flattenKernelFloat);

	cudaFuncGetAttributes(&funcAttributes[31], concatKernelFloat);

//	cudaFuncGetAttributes(&funcAttributes[9], pairWiseTransformFloat);

//  cudaFuncGetAttributes(&funcAttributes[10], pairWiseTransformFloatIndex);

//	cudaFuncGetAttributes(&funcAttributes[11], pairWiseTransformStridedFloat);

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

	//cudaFuncGetAttributes(&funcAttributes[17], functions::summarystats::summaryStatsReduceDouble);

//	cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

	//void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo, int xRank, double *params, double *result,int *resultShapeInfo, int zRank, int *allocPointer) = scalarDouble;
//	cudaFuncGetAttributes(&funcAttributes[19], scalarDoubleIndexes);


	//void (*scalarDoublePointer2)(int opNum, Nd4jIndex n,double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocPointer) = scalarDouble;
//	cudaFuncGetAttributes(&funcAttributes[20], scalarDoubleIndexes);

	cudaFuncGetAttributes(&funcAttributes[21], reduce3Double);

	cudaFuncGetAttributes(&funcAttributes[22], reduceSimpleGenericXD_0_double);

//	cudaFuncGetAttributes(&funcAttributes[23], pairWiseTransformDouble);

//	cudaFuncGetAttributes(&funcAttributes[24], pairWiseTransformDoubleIndex);

//	cudaFuncGetAttributes(&funcAttributes[25], pairWiseTransformStridedDouble);

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
	Nd4jPointer nativeStream = (Nd4jPointer) malloc(sizeof(cudaStream_t));
	cudaError_t result = cudaStreamCreate((cudaStream_t *) &nativeStream);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return nativeStream;
}

Nd4jPointer NativeOps::createEvent() {
	Nd4jPointer nativeEvent= (Nd4jPointer) malloc(sizeof(cudaEvent_t));
	cudaError_t result = cudaEventCreateWithFlags((cudaEvent_t *) &nativeEvent, cudaEventDisableTiming);
	checkCudaErrors(result);
	if (result != 0)
		return 0L;
	else return nativeEvent;
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

	return (Nd4jIndex) memFree;
}

Nd4jIndex NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
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

	return (Nd4jIndex) memTotal;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {

	return memcpyAsync(dst, src, size, flags, reserved);
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaMemcpyKind 	kind;

	if (nd4j::Environment::getInstance()->isDebug())
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
			if (!shape::isVector(hostShapePointers[i]) || shape::elementWiseStride(hostShapePointers[i]) <= 0 ||
				shape::order(hostShapePointers[i]) != 'c') {
				isVstack = false;
				break;
			}
		}
	}

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
		Nd4jIndex length0 = shape::length(hostShapePointers[0]);
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
		concatKernelScalarFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		smem = funcAttributes[40].sharedSizeBytes;
		concatKernelVStackFloat<<< 128, 512, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");
		smem = funcAttributes[42].sharedSizeBytes;

		concatKernelHStackFloat<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

		//smem = nd4j::math::nd4j_max<int>(funcAttributes[31].sharedSizeBytes + 768, 1280);

        int *devZTadShape = reinterpret_cast<int *>(extraPointers[10]);
		Nd4jIndex *devZOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

		concatKernelFloat<<< 2048, 128, funcAttributes[31].sharedSizeBytes , *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0], devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatFloat: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


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

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
        Nd4jIndex length0 = shape::length(hostShapePointers[0]);
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
		concatKernelScalarHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		smem = funcAttributes[40].sharedSizeBytes;
		concatKernelVStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");
		smem = funcAttributes[42].sharedSizeBytes;

		concatKernelHStackHalf<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

		//smem = nd4j::math::nd4j_max<int>(funcAttributes[31].sharedSizeBytes + 768, 1280);

        int *devZTadShape = reinterpret_cast<int *>(extraPointers[10]);
		Nd4jIndex *devZOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

		concatKernelHalf<<< 2048, 128, funcAttributes[31].sharedSizeBytes, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0], devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatHalf: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


	checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::specialConcatFloat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float *result,
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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

    // let's try to fit N-dimensional vstack
    if (!isVstack && !isScalar && dimension == 0 && shape::order(hostXShapeInfo) == 'c') {
        Nd4jIndex length0 = shape::length(hostShapePointers[0]);
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
		concatKernelScalarDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isVstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going VStack concat\n");

		smem = funcAttributes[41].sharedSizeBytes;
		concatKernelVStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else if (isHstack) {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going HStack concat\n");
		smem = funcAttributes[43].sharedSizeBytes;

		concatKernelHStackDouble<<< 128, 128, smem, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0]);
	} else {
		if (nd4j::Environment::getInstance()->isDebugAndVerbose())
			printf("Going generic concat\n");

        int *devZTadShape = reinterpret_cast<int *>(extraPointers[10]);
        Nd4jIndex *devZOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);

		concatKernelDouble<<< 2048, 128, funcAttributes[35].sharedSizeBytes, *stream>>> (dimension, numArrays, (Nd4jPointer *) data[0], (Nd4jPointer *) inputShapeInfo[0], result, resultShapeInfo, (Nd4jPointer *) tadPointers[0], (Nd4jPointer *) offsetPointers[0], devZTadShape, devZOffsets);
	}
	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("sharedMemory requested for concatDouble: [%i], registers: [%i]\n", smem, funcAttributes[31].numRegs);


	checkCudaErrors(cudaStreamSynchronize(*stream));
}

/**
 * This method saves
 */
void NativeOps::tadOnlyShapeInfo(int *xShapeInfo, int *dimension, int dimensionLength, int *target, Nd4jIndex *offsets) {
	shape::TAD *tad = new shape::TAD();
	tad->init(xShapeInfo, dimension, dimensionLength);
	//tad->setOutputBuffer(target);
	tad->createTadOnlyShapeInfo();
	tad->createOffsets();


	std::memcpy((void *) target, tad->tadOnlyShapeInfo, (tad->tadOnlyShapeInfo[0] * 2 + 4) * sizeof(int));
	std::memcpy((void *) offsets, tad->tadOffsets, tad->numTads * sizeof(Nd4jIndex));
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

	if (nd4j::Environment::getInstance()->isDebug())
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

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    pullRowsKernelHalf<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, float *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelFloat<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, double *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	pullRowsKernelDouble<<<64, 256, 1024, *stream>>>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);

	if (nd4j::Environment::getInstance()->isDebug())
		checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

    float16 **x = reinterpret_cast<float16 **>(dx);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("averageHalf called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float16), funcAttributes[44]);

		averagingKernelHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
        nd4j::SpecialMethods<float16>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length, bool propagate) {
	cudaStream_t * stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float **x = reinterpret_cast<float **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("averageFloat called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float), funcAttributes[45]);

		averagingKernelFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length, propagate);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
		// launching on host memory
        nd4j::SpecialMethods<float>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length, bool propagate) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

    double **x = reinterpret_cast<double **>(dx);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("averageDouble called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(double), funcAttributes[46]);

		averagingKernelDouble << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (x, dz, n, length, propagate);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
        nd4j::SpecialMethods<double>::averageGeneric(x, dz, n, length, propagate);
	}
}

void NativeOps::accumulateHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float16 **x = reinterpret_cast<float16 **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateHalf called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float16), funcAttributes[44]);

		accumulateKernelHalf<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
        nd4j::SpecialMethods<float16>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::accumulateFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length) {
	cudaStream_t * stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	float **x = reinterpret_cast<float **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateFloat called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(float), funcAttributes[45]);

        accumulateKernelFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, dz, n, length);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
		// launching on host memory
        nd4j::SpecialMethods<float>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::accumulateDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length) {
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	double **x = reinterpret_cast<double **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateDouble called\n");

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims = getBasicLaunchParams(getDeviceId(extras[2]), length, sizeof(double), funcAttributes[46]);

		accumulateKernelDouble << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (x, dz, n, length);

		checkCudaErrors(cudaStreamSynchronize(*stream));
	} else {
        nd4j::SpecialMethods<double>::accumulateGeneric(x, dz, n, length);
	}
}

void NativeOps::shuffleDouble(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    Nd4jIndex **tadOffset = reinterpret_cast<Nd4jIndex **>(tadOffsets);

    shuffleKernelDouble<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::shuffleFloat(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    Nd4jIndex **tadOffset = reinterpret_cast<Nd4jIndex **>(tadOffsets);

    shuffleKernelFloat<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::shuffleHalf(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    float16 **x = reinterpret_cast<float16 **>(dx);
    float16 **z = reinterpret_cast<float16 **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    Nd4jIndex **tadOffset = reinterpret_cast<Nd4jIndex **>(tadOffsets);

    shuffleKernelHalf<<<32, 128, 1024, *stream>>>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedFloat<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    /*
	if (opTypeA == 2) {
		if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
		}
	}
*/
    functions::grid::GRIDStrided<float>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);



    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedDouble<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
/*
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
*/
    functions::grid::GRIDStrided<double>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

//    metaPredicateStridedHalf<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

    /*
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalA, scalB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
    */

    functions::grid::GRIDStrided<float16>::execMetaPredicateStrided(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned) {
    // no-op

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

/*
 metaPredicateReduceFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *tadShapeInfo, int *tadOffsets, float *reductionBuffer, float *extraA, float *extraB, float scalarA, float scalarB) {
 */

//    metaPredicateReduceFloat<<<256, 256, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, nullptr, extraA, extraB, scalarA, scalarB, scalarReturned);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}



void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    /*
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
     */

    functions::grid::GRIDShaped<double>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

	// we have to converf float -> fp16 prior to kernel call
    float16 scalA = (float16) scalarA;
    float16 scalB = (float16) scalarB;

    /*
	if (opTypeA == 2) {
		if (opTypeB == 0) {
			DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalA, scalB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
		}
	}
     */

    functions::grid::GRIDShaped<float16>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);


    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op;

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    /*
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
    */

    functions::grid::GRIDShaped<float>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

    if (nd4j::Environment::getInstance()->isDebug())
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
    // this is no-op for CUDA
}

void NativeOps::setTADThreshold(int num) {
    // this is no-op for CUDA
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

    //dim3 launchDims = getReduceLaunchParams(getDeviceId(extraPointers[2]),hostXShapeInfo, hostTadShapeInfo, funcAttributes[47] ,dimensionLength, sizeof(float), 0);
    dim3 launchDims = dim3(256, 256, 1024);

	// this macro builds bunch of IF/ELSE selectors for kernel launc	h
    //DISPATCH_SIMPLE(scalarAlongDimension, float, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<float>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

    if (nd4j::Environment::getInstance()->isDebug())
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

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarAlongDimension, double, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<double>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

    if (nd4j::Environment::getInstance()->isDebug())
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

    /*
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[10]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[11]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[12]);
    Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[13]);
*/
	// this macro builds bunch of IF/ELSE selectors for kernel launch
    //DISPATCH_SIMPLE(scalarAlongDimension, float16, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

    functions::scalar::ScalarTransform<float16>::executeCudaAlongDimension(launchDims, extraPointers, opNum, x, xShapeInfo, z, zShapeInfo, scalars, extraParams, dimension, dimensionLength);

    if (nd4j::Environment::getInstance()->isDebug())
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
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

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);

	// this macro builds bunch of IF/ELSE selectors for kernel launch
    DISPATCH_SIMPLE(aggregateSimple, float16, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

    checkCudaErrors(cudaStreamSynchronize(*stream));
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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *z, int *zShapeBuffer, float *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaSingle(launchDims, extraPointers, opNum, stateHost, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, int *xShapeBuffer, float *y, int *yShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments) {

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaTriple(launchDims, extraPointers, opNum, stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, int *xShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float)) );

    functions::random::RandomFunction<float>::executeCudaDouble(launchDims, extraPointers, opNum, stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *z, int *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaSingle(launchDims, extraPointers, opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, int *xShapeBuffer, double *y, int *yShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaTriple(launchDims, extraPointers, opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, int *xShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(double)));

    functions::random::RandomFunction<double>::executeCudaDouble(launchDims, extraPointers, opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *z, int *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaSingle(launchDims, extraPointers, opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *y, int *yShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaTriple(launchDims, extraPointers, opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = dim3(512, 512, sizeof(nd4j::random::RandomBuffer) + (560 * sizeof(float16)));

    functions::random::RandomFunction<float16>::executeCudaDouble(launchDims, extraPointers, opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}


Nd4jPointer NativeOps::initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {

    unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we don't synchronize at random initialization, it's safe to go unsync here
	// cudaStreamSynchronize(*stream);

    unsigned long long *ptrDev = reinterpret_cast<unsigned long long *>(ptrToBuffer);
    nd4j::random::RandomBuffer *buffer = new nd4j::random::RandomBuffer(seed, bufferSize, (uint64_t *) ptrHost, (uint64_t *) ptrDev);
    buffer->propagateToDevice(buffer, *stream);

    checkCudaErrors(cudaStreamSynchronize(*stream));

	// we generate sequence in the host memory
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

	// and copy it to gpu
    cudaMemcpyAsync(ptrDev, ptrHost, bufferSize * 8, cudaMemcpyHostToDevice, *stream);
	checkCudaErrors(cudaStreamSynchronize(*stream));

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

    int *shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(),
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
    free((void *) npyArray);
}


/**
    * Return the length of a shape buffer
    * based on the pointer
    * @param buffer  the buffer pointer to check
    * @return
    */
int NativeOps::lengthForShapeBufferPointer(Nd4jPointer buffer) {
    int *shapeBuffer = reinterpret_cast<int *>(buffer);
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

Nd4jPointer NativeOps::pointerForAddress(Nd4jIndex address) {
	return reinterpret_cast<Nd4jPointer >(address);
}

void NativeOps::tearDouble(Nd4jPointer *extras, double *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelDouble<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::tearFloat(Nd4jPointer *extras, float *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelFloat<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::tearHalf(Nd4jPointer *extras, float16 *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    tearKernelHalf<<<512, 512, 512, *stream>>>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::encodeThresholdP1Float(Nd4jPointer *extras, float *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP1Float<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::encodeThresholdP1Double(Nd4jPointer *extras, double *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP1Double<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::encodeThresholdP1Half(Nd4jPointer *extras, float16 *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP1Half<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz, threshold);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jIndex N, int *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    //encoderKernelP2Float<<<numBlocks, blockSize , 1024 * sizeof(float), *stream>>>(dx, N, dz);

    // it
    prescanArrayRecursive(extraPointers, dz, dx + 1, (int) N, 0);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jIndex N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP3Float<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jIndex N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP3Double<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jIndex N, int *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    encoderKernelP3Half<<<numBlocks, blockSize , 4096, *stream>>>(dx, offsets, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    decoderKernelFloat<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    decoderKernelDouble<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz){
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);

    decoderKernelHalf<<<numBlocks, blockSize , 1024, *stream>>>(dx, N, dz);
    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::execReduce3AllDouble(Nd4jPointer *extraPointers,
									 int opNum,
									 double *x,
									 int *xInfo,
									 double *extraParamsVals,
									 double *y,
									 int *yInfo,
									 double *result,
									 int *resultShapeInfoBuffer,
									 int *dimension,
									 int dimensionLength,
									 int *xTadShapeInfo,
                                     Nd4jIndex *xOffsets,
									 int *yTadShapeInfo,
                                     Nd4jIndex *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);
    int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);


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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execReduce3AllFloat(Nd4jPointer *extraPointers,
									int opNum,
									float *x,
									int *xInfo,
									float *extraParamsVals,
									float *y,
									int *yInfo,
									float *result,
									int *resultShapeInfoBuffer,
									int *dimension,
									int dimensionLength,
									int *xTadShapeInfo,
                                    Nd4jIndex *xOffsets,
									int *yTadShapeInfo,
                                    Nd4jIndex *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);
    int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);


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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::execReduce3AllHalf(Nd4jPointer *extraPointers,
								   int opNum,
								   float16 *x,
								   int *xInfo,
								   float16 *extraParamsVals,
								   float16 *y,
								   int *yInfo,
								   float16 *result,
								   int *resultShapeInfoBuffer,
								   int *dimension,
								   int dimensionLength,
								   int *xTadShapeInfo,
                                   Nd4jIndex *xOffsets,
								   int *yTadShapeInfo,
                                   Nd4jIndex *yOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *hostZShapeInfo = reinterpret_cast<int *>(extraPointers[8]);
    int *hostTADShapeInfo = reinterpret_cast<int *>(extraPointers[9]);


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

    if (nd4j::Environment::getInstance()->isDebug())
        checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::sortFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[     1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int xLength = shape::length(hostXShapeInfo);
    int xEWS = shape::elementWiseStride(hostXShapeInfo);

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

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::sortDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int xLength = shape::length(hostXShapeInfo);
    int xEWS = shape::elementWiseStride(hostXShapeInfo);

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

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::sortHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, bool descending) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

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

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::sortTadFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaSortTadFloat<<<512, 512, 1088 * sizeof(float), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::sortTadHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaSortTadHalf<<<512, 512, 1088 * sizeof(float16), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::sortTadDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    // to be implemented
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaSortTadDouble<<<512, 512, 1088 * sizeof(double), *stream>>>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

void NativeOps::sortCooIndicesFloat(Nd4jPointer *extraPointers, int *indices, float *values, Nd4jIndex length, int rank) {

}

void NativeOps::sortCooIndicesDouble(Nd4jPointer *extraPointers, int *indices, double *values, Nd4jIndex length, int rank) {

}

void NativeOps::sortCooIndicesHalf(Nd4jPointer *extraPointers, int *indices, float16 *values, Nd4jIndex length, int rank) {

}


Nd4jIndex NativeOps::encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapFloat<<<512, 512, 512 * 2 * sizeof(float) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    Nd4jIndex result = (Nd4jIndex) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

Nd4jIndex NativeOps::encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapDouble<<<512, 512, 512 * 2 * sizeof(double) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    Nd4jIndex result = (Nd4jIndex) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

Nd4jIndex NativeOps::encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jIndex N, int *dz, float threshold) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);

    cudaEncodeBitmapHalf<<<512, 512, (512 * sizeof(float16)) + (512 * sizeof(int)) + 384, *stream>>>(dx, N, dz, resultPointer, reductionPointer, threshold);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    Nd4jIndex result = (Nd4jIndex) resultPointer[0];
    resultPointer[0] = 0;

    return result;
}

void NativeOps::decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaDecodeBitmapFloat<<<512, 512, 512 * sizeof(float) + 384, *stream>>>(dx, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaDecodeBitmapDouble<<<512, 512, 512 * sizeof(double) + 384, *stream>>>(dx, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}


void NativeOps::decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz) {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);

    cudaDecodeBitmapHalf<<<512, 512, 512 * sizeof(float16) + 384, *stream>>>(dx, N, dz);

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

Nd4jIndex* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jIndex length) {
	return nullptr;
}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jIndex* ptrMap, Nd4jIndex length) {

}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer protoBufferPointer) {
	return nullptr;
}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, const char *fileName) {
	return nullptr;
}

Nd4jPointer NativeOps::executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}

Nd4jPointer NativeOps::executeFlatGraphHalf(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}

	
Nd4jPointer NativeOps::executeFlatGraphDouble(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
	return nullptr;
}
		


const char* NativeOps::getAllCustomOps() {
	return nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}


template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    nd4j::graph::VariableSpace<T> varSpace;
    Context<T> block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = (int *) inputShapes[e];
        auto buffer_ = (T *) inputBuffers[e];
        auto array = new nd4j::NDArray<T>(buffer_, shape_);
        array->triggerAllocationFlag(false, false);

        // block should contain references to proper variable
        varSpace.putVariable(1, e, array);
        block.pickInput(1, e);

        inShapes.push_back(shape_);
    }

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}


template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    nd4j::graph::Context<T> block(1);
	nd4j::ShapeList inShapes;

	for (int e = 0; e < numIArgs; e++)
		block.getIArguments()->push_back(iArgs[e]);

	for (int e = 0; e < numTArgs; e++)
		block.getTArguments()->push_back(tArgs[e]);

	for (int e = 0; e < numInputShapes; e++)
		inShapes.push_back((int *) inputShapes[e]);

	auto shapeList = op->calculateOutputShape(&inShapes, block);

	return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, int *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

	return _calculateOutputShapes<float>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, int *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

	return _calculateOutputShapes<float16>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, int *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

	return _calculateOutputShapes<double>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
static FORCEINLINE Nd4jStatus realExec(nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, T* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
	if (op == nullptr)
		nd4j_printf("Can't find requested operation: [%lld]\n", hash);

	// we're using the same fake nodeId everywhere here

	std::vector<nd4j::NDArray<T>*> inputs(numInputs);
    std::vector<nd4j::NDArray<T>*> outputs;
	std::vector<T> ttArgs(numTArgs);
	std::vector<int> iiArgs(numIArgs);

	// filling block now
	for (int e = 0; e < numInputs; e++) {
		auto buffer = (T *) inputBuffers[e];
		auto shape = (int *) inputShapes[e];

		// auto var = new Variable<T>(new NDArray<T>(buffer, shape));
		// block.getVariables()->emplace_back(var);
		auto array = new nd4j::NDArray<T>(buffer, shape);
		//array->setSpecialBuffers( (T *) inputBuffers[e + numInputs],  (int *) inputShapes[e + numInputs]);

		inputs[e] = array;
	}

	for (int e = 0; e < numIArgs; e++)
		iiArgs[e] = iArgs[e];


	for (int e = 0; e < numTArgs; e++)
		ttArgs[e] = tArgs[e];


	// hypothetically at this point we have everything filled
	auto result = op->execute(inputs, ttArgs, iiArgs, isInplace);

	if (result->status() != ND4J_STATUS_OK)
		return result->status();

    if (!isInplace) {
        if (result->size() != numOutputs) {
            return ND4J_STATUS_BAD_OUTPUT;
        }

        for (int e = 0; e < numOutputs; e++) {
            auto buffer = (T *) outputBuffers[e];
            auto shape = (int *) outputShapes[e];
            nd4j::NDArray <T> tmp(buffer, shape);

            tmp.assign(result->at(e));
        }
    }

	delete result;


	for (auto ptr: inputs)
		delete ptr;


	return ND4J_STATUS_OK;
}


int NativeOps::execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

	return realExec<float>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

	return realExec<double>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

	return realExec<float16>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::registerGraphFloat(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<float>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

int NativeOps::registerGraphDouble(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<double>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

int NativeOps::registerGraphHalf(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
	auto graph = nd4j::graph::GraphExecutioner<float16>::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}

template <typename T>
static VariablesSet<T>* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	auto graph = nd4j::graph::GraphHolder::getInstance()->pullGraph<T>(graphId);
	auto varSpace = graph->getVariableSpace()->clone();

	std::vector<nd4j::NDArray<T> *> handles;

	for (int e = 0; e < numInputs; e++) {
		auto idx = inputIndices[e];

		// we'll delete this array later, together with cloned VariableSpace
		auto array = new nd4j::NDArray<T>((T *) inputBuffers[e], (int *) inputShapes[e]);
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

VariablesSet<float>* NativeOps::executeStoredGraphFloat(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<float>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<float16>* NativeOps::executeStoredGraphHalf(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<float16>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<double>* NativeOps::executeStoredGraphDouble(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT<double>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

int NativeOps::unregisterGraph(Nd4jPointer *extraPointers, Nd4jIndex graphId) {

	nd4j::graph::GraphHolder::getInstance()->dropGraphAny(graphId);

	return ND4J_STATUS_OK;
}

void NativeOps::deletePointerArray(Nd4jPointer pointer) {
    Nd4jPointer *ptr = reinterpret_cast<Nd4jPointer *>(pointer);
    delete[] ptr;
}

void NativeOps::deleteIntArray(Nd4jPointer pointer) {
	int *ptr = reinterpret_cast<int *>(pointer);
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
