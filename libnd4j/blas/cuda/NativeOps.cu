/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


#include "NativeOpExecutioner.h"
#include "../NativeOps.h"
#include <cuda_launch_config.h>

#include <buffer.h>


#include <loops/transform_any.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_long.h>
#include <helpers/threshold.h>
#include <ops/specials_cuda.h>
#include <helpers/DebugHelper.h>

#include <exceptions/datatype_exception.h>
#include <helpers/CudaLaunchHelper.h>
// FIXME: we need cuda-specific implementations
#include <GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <ops/declarable/CustomOperations.h>
#include <PointersManager.h>


//#include <sys/time.h>

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
#ifdef __ND4J_EXPERIMENTAL__
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

    //printf("Finished stream: [%i], kernel call: [%i]\n", sync->streamId, sync->callId);
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
dim3 getFlatLaunchParams(int deviceId, Nd4jLong *dXShapeInfo, Nd4jLong *dYShapeInfo, cudaFuncAttributes funcAttr) {
	auto xRank = shape::rank(dXShapeInfo);
	auto yRank = dYShapeInfo == nullptr ? 0 : shape::rank(dYShapeInfo);
	auto zRank = 0;

	int memory_limit = getBaseMemorySize(xRank, funcAttr);

	int countMP = deviceProperties[deviceId].multiProcessorCount;
	int regPerBlock = deviceProperties[deviceId].regsPerBlock;

	int blockThreshold = getDeviceBlockThreshold(deviceId);
	int shmemThreshold = getDeviceSharedThreshold(deviceId);

	auto xLength = shape::length(dXShapeInfo);
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
 * @param dXShapeInfo
 * @param tadShapeInfo
 * @param funcAttr
 * @param dimensionLength
 * @param elementSize
 * @param reductionSize
 * @return
 */
dim3 getReduceLaunchParams(int deviceId, Nd4jLong *dXShapeInfo, Nd4jLong *tadShapeInfo, cudaFuncAttributes funcAttr, int dimensionLength, int elementSize, int reductionSize) {

	Nd4jLong tadLength = 0;
	Nd4jLong numTads = 0;
	if (tadShapeInfo != nullptr) {
		tadLength = shape::length(tadShapeInfo);
		numTads = shape::length(dXShapeInfo) / tadLength;

		if (tadLength == 1) {
			if (nd4j::Environment::getInstance()->isDebugAndVerbose())
				printf("A xLength: [%i], zLength: [%i]\n", shape::length(dXShapeInfo), shape::length(tadShapeInfo));
		}
	} else{
		// we have special case - reduction along all dimensions
		tadLength = nd4j::math::nd4j_min<int>(shape::length(dXShapeInfo), 768);
		numTads = shape::length(dXShapeInfo) / tadLength;
	}

	auto xRank = shape::rank(dXShapeInfo);
	int zRank = tadShapeInfo == nullptr ? 0 : shape::rank(tadShapeInfo);

	dim3 launchDims = getBetterDimensions(deviceId, numTads, tadLength, xRank, funcAttr, dimensionLength, elementSize, reductionSize);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose()) { //|| launchDims.dX == 1
		printf("Reduce LaunchParams: xLength: [%i], numTads: [%i], tadLength: [%i], launchDims.dX: [%i], launchDims.dY: [%i], launchDims.dZ: [%i]\n", shape::length(dXShapeInfo), numTads, tadLength, launchDims.x, launchDims.y, launchDims.z);
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
dim3 getOptimalLaunchParameters(const Nd4jLong *hXShapeInfo, cudaFuncAttributes attributes, cudaDeviceProp properties) {
	
	auto n = shape::length(hXShapeInfo);

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

		CHECK_ALLOC(scalarDimensionBuff, "Failed to allocate ShapeInfoBuffer", sizeof(Nd4jLong));

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

		CHECK_ALLOC(scalarResult, "Failed to allocate new scalar buffer", sizeof(T));

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
	 * Get the dZ pointers
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

NativeOps::NativeOps() {
//
}

void NativeOps::execPairwiseTransform( Nd4jPointer *extraPointers,
        								int opNum,
        								void *hX, Nd4jLong *hXShapeInfo,
        								void *dX, Nd4jLong *dXShapeInfo,
        								void *hY, Nd4jLong *hYShapeInfo,
        								void *dY, Nd4jLong *dYShapeInfo,
        								void *hZ, Nd4jLong *hZShapeInfo,
        								void *dZ, Nd4jLong *dZShapeInfo,
        								void *extraParams) {

	NativeOpExecutioner::execPairwiseTransform(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execPairwiseTransformBool(Nd4jPointer *extraPointers,
        								int opNum,
        								void *hX, Nd4jLong *hXShapeInfo,
        								void *dX, Nd4jLong *dXShapeInfo,
        								void *hY, Nd4jLong *hYShapeInfo,
        								void *dY, Nd4jLong *dYShapeInfo,
        								void *hZ, Nd4jLong *hZShapeInfo,
        								void *dZ, Nd4jLong *dZShapeInfo,
        								void *extraParams) {

	NativeOpExecutioner::execPairwiseBoolTransform(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execSummaryStatsScalar(Nd4jPointer *extraPointers,
                                       int opNum,
                                       void *hX, Nd4jLong *hXShapeInfo,
                                       void *dX, Nd4jLong *dXShapeInfo,
                                       void *extraParams,
                                       void *hZ, Nd4jLong *hZShapeInfo,
                                       void *dZ, Nd4jLong *dZShapeInfo,
                                       bool biasCorrected) {
	
	NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, biasCorrected);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execBroadcastBool(Nd4jPointer *extraPointers,
        						int opNum,
        						void *hX, Nd4jLong *hXShapeInfo,
        						void *dX, Nd4jLong *dXShapeInfo,
        						void *hY, Nd4jLong *hYShapeInfo,
        						void *dY, Nd4jLong *dYShapeInfo,
        						void *hZ, Nd4jLong *hZShapeInfo,
        						void *dZ, Nd4jLong *dZShapeInfo,
        						void *hDimension, Nd4jLong *hDimensionShape,
		void *dDimension, Nd4jLong *dDimensionShape) {

	Nd4jLong *tadOnlyShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    Nd4jLong *tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    Nd4jLong *tadOnlyShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    Nd4jLong *tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);

	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	auto hTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto dTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto dTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
	auto dTADShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto dTADOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);
    NativeOpExecutioner::execBroadcastBool(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);

}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param dY
 * @param dYShapeInfo
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execBroadcast(
		Nd4jPointer *extraPointers,
		int opNum,
		void *hX, Nd4jLong *hXShapeInfo,
		void *dX, Nd4jLong *dXShapeInfo,
		void *hY, Nd4jLong *hYShapeInfo,
		void *dY, Nd4jLong *dYShapeInfo,
		void *hZ, Nd4jLong *hZShapeInfo,
		void *dZ, Nd4jLong *dZShapeInfo,
		void *hDimension, Nd4jLong *hDimensionShape,
		void *dDimension, Nd4jLong *dDimensionShape) {
/*
    cudaEvent_t start;
    cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
    timespec tsX;
    timespec tsY;
    clock_gettime(CLOCK_REALTIME, &tsX);
*/
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	auto hTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto dTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
	auto dTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
	auto dTADShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
	auto dTADOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

	auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
	auto yType = nd4j::ArrayOptions::dataType(hYShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F3 opNum:[%i]\n", opNum);

	Nd4jLong *tadOnlyShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    Nd4jLong *tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    Nd4jLong *tadOnlyShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    Nd4jLong *tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);

    NativeOpExecutioner::execBroadcast(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
}


/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceFloat(Nd4jPointer *extraPointers,
							int opNum,
							void *hX, Nd4jLong *hXShapeInfo,
							void *dX, Nd4jLong *dXShapeInfo,
							void *extraParams,
							void *hZ, Nd4jLong *hZShapeInfo,
							void *dZ, Nd4jLong *dZShapeInfo) {

	NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceSame(Nd4jPointer *extraPointers,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo) {


}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceSame(Nd4jPointer *extraPointers,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
							   void *hDimension, Nd4jLong *hDimensionShape,
							   void *dDimension, Nd4jLong *dDimensionShape) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExecutioner::execReduceSame(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceLong(Nd4jPointer *extraPointers,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
							   void *hDimension, Nd4jLong *hDimensionShape,
							   void *dDimension, Nd4jLong *dDimensionShape) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExecutioner::execReduceLong(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void   NativeOps::execReduceLong(Nd4jPointer *extraPointers,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto dTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("LF7 opNum:[%i]\n", opNum);

    auto reductionPointer = reinterpret_cast<void *>(extraPointers[4]);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::INT64)
        throw datatype_exception::build("NativeOps::execReduceLong wrong Z data type", nd4j::DataType::INT64, zType);

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction, ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, extraParams, dZ, dZShapeInfo, nullptr, 1, reductionPointer, dTADShapeInfo), LIBND4J_TYPES, LONG_TYPES);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceLong(...) failed");
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceBool(Nd4jPointer *extraPointers,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
							   void *hDimension, Nd4jLong *hDimensionShape,
							   void *dDimension, Nd4jLong *dDimensionShape) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

	NativeOpExecutioner::execReduceBool(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void   NativeOps::execReduceBool(Nd4jPointer *extraPointers,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo) {

    auto stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
	auto dTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        printf("BF7 opNum:[%i]\n", opNum);

    auto reductionPointer = reinterpret_cast<void *>(extraPointers[4]);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

    if (zType != nd4j::DataType::BOOL)
        throw std::runtime_error("NativeOps::execReduceBool requires Z operand to have BOOL type");

    auto xLength = shape::length(hXShapeInfo);
    auto blockWidth = 256;
    auto numBlocks = CudaLaunchHelper::getReductionBlocks(xLength, blockWidth);
    dim3 launchDims(numBlocks, blockWidth, 32768);

    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction, ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, extraParams, dZ, dZShapeInfo, nullptr, 1, reductionPointer, dTADShapeInfo), LIBND4J_TYPES, BOOL_TYPES);

    nd4j::DebugHelper::checkErrorCode(stream, "execReduceBool(...) failed");
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
////////////////////////////////////////////////////////////////////////
void NativeOps::execIndexReduce(Nd4jPointer *extraPointers,
								 int opNum,
								 void *hX, Nd4jLong *hXShapeInfo,
        						 void *dX, Nd4jLong *dXShapeInfo,
        						 void *extraParams,
        						 void *hZ, Nd4jLong *hZShapeInfo,
        						 void *dZ, Nd4jLong *dZShapeInfo,
								 void *hDimension, Nd4jLong *hDimensionShape,
		void *dDimension, Nd4jLong *dDimensionShape) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	Nd4jLong *tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    Nd4jLong *tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExecutioner::execIndexReduce(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
////////////////////////////////////////////////////////////////////////
void NativeOps::execReduceFloat(Nd4jPointer *extraPointers,
								int opNum,
								void *hX, Nd4jLong *hXShapeInfo,
        						void *dX, Nd4jLong *dXShapeInfo,
        						void *extraParams,
        						void *hZ, Nd4jLong *hZShapeInfo,
								void *dZ, Nd4jLong *dZShapeInfo,
								void *hDimension, Nd4jLong *hDimensionShape,
		void *dDimension, Nd4jLong *dDimensionShape) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExecutioner::execReduceFloat(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 */
////////////////////////////////////////////////////////////////////////
void NativeOps::execIndexReduceScalar(
		Nd4jPointer *extraPointers,
		int opNum,
		void *hX, Nd4jLong *hXShapeInfo,
        void *dX, Nd4jLong *dXShapeInfo,
        void *extraParams,
        void *hZ, Nd4jLong *hZShapeInfo,
		void *dZ, Nd4jLong *dZShapeInfo){

	NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execTransformSame(Nd4jPointer *extraPointers,int opNum,
                                   void *hX, Nd4jLong *hXShapeInfo,
                                   void *dX, Nd4jLong *dXShapeInfo,
                                   void *hZ, Nd4jLong *hZShapeInfo,
                                   void *dZ, Nd4jLong *dZShapeInfo,
                                   void *extraParams) {

    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    NativeOpExecutioner::execTransformSame(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execTransformBool(Nd4jPointer *extraPointers,int opNum,
								  void *hX, Nd4jLong *hXShapeInfo,
								  void *dX, Nd4jLong *dXShapeInfo,
								  void *hZ, Nd4jLong *hZShapeInfo,
								  void *dZ, Nd4jLong *dZShapeInfo,
								  void *extraParams) {

	auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    NativeOpExecutioner::execTransformBool(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execTransformAny(Nd4jPointer *extraPointers,int opNum,
								  void *hX, Nd4jLong *hXShapeInfo,
								  void *dX, Nd4jLong *dXShapeInfo,
								  void *hZ, Nd4jLong *hZShapeInfo,
								  void *dZ, Nd4jLong *dZShapeInfo,
								  void *extraParams) {

	 NativeOpExecutioner::execTransformAny(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execTransformStrict(Nd4jPointer *extraPointers,int opNum,
                                  void *hX, Nd4jLong *hXShapeInfo,
                                  void *dX, Nd4jLong *dXShapeInfo,
                                  void *hZ, Nd4jLong *hZShapeInfo,
                                  void *dZ, Nd4jLong *dZShapeInfo,
                                  void *extraParams) {

    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    NativeOpExecutioner::execTransformStrict(nullptr,opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    void *extraParams) {

    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    NativeOpExecutioner::execTransformFloat(nullptr, opNum, hX, hXShapeInfo, dZ, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraParams, tadShapeInfo, tadOffsets);
}


/**
 * Append an input array
 * to the end of a flat array
 * in a particular order
 * @param offset the offset of the array to start at
 * @param order the order
 * @param dZ the dZ array
 * @param dZShapeInfo the shape info for te array
 * @param input the input for the array
 * @param inputShapeInfo the shape information for that array
 */
void NativeOps::flatten(Nd4jPointer *extraPointers,
						int offset,
						char order,
						void *hZ, Nd4jLong *hZShapeInfo,
						void *dZ, Nd4jLong *dZShapeInfo,
						void *hInput, Nd4jLong *hInputShapeInfo,
						void *dInput, Nd4jLong *dInputShapeInfo) {
	
	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
	auto hYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("F22 opNum:[7]\n");

	// int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	dim3 launchDims = getBasicLaunchParams(getDeviceId(extraPointers[2]), shape::length(hYShapeInfo), 2, funcAttributes[30]);

	if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
		printf("AF222 opNum:[7]\n");
	
	auto type = nd4j::ArrayOptions::dataType(hInputShapeInfo);    
    BUILD_SINGLE_SELECTOR(type, flattenKernelGeneric, (launchDims, stream, extraPointers, offset, order, dZ, dZShapeInfo, dInput, dInputShapeInfo), LIBND4J_TYPES);

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
		for (int dX = 0; dX < devCnt; dX++) {

			for (int dY = 0; dY < devCnt; dY++) {
				if (dX == dY)
					continue;

				int canAccess = 0;
				cudaSetDevice(dX);

				cudaDeviceCanAccessPeer(&canAccess, dX , dY);

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
        for (int dX = 0; dX < devCnt; dX++) {

            for (int dY = 0; dY < devCnt; dY++) {
                if (dX == dY)
                    continue;

                int canAccess = 0;
                cudaSetDevice(dX);

                cudaDeviceCanAccessPeer(&canAccess, dX , dY);

                if (canAccess) {
                    if (enable) {
                        cudaDeviceEnablePeerAccess(dY, 0);
                    } else {
                        cudaDeviceDisablePeerAccess(dY);
                    }
                } else {
					if (nd4j::Environment::getInstance()->isVerbose()) printf("Peer access [%i] -> [%i] isn't possible\n", dX, dY);
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
Nd4jPointer NativeOps::mallocDevice(Nd4jLong memorySize, int deviceId, int flags) {
	Nd4jPointer pointer;
	auto res = cudaMalloc(reinterpret_cast<void **>(&pointer), memorySize);
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
int NativeOps::freeDevice(Nd4jPointer pointer, int deviceId) {
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

	CHECK_ALLOC(nativeStream, "Failed to allocate memory for new CUDA stream", sizeof(cudaStream_t));

	cudaError_t dZ = cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&nativeStream));
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaStreamCreate(...) failed");

	return nativeStream;
}

Nd4jPointer NativeOps::createEvent() {
	Nd4jPointer nativeEvent= (Nd4jPointer) malloc(sizeof(cudaEvent_t));

	CHECK_ALLOC(nativeEvent, "Failed to allocate new CUDA event buffer", sizeof(cudaEvent_t));

	cudaError_t dZ = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&nativeEvent), cudaEventDisableTiming);
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaEventCreateWithFlags(...) failed");


	return nativeEvent;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t dZ = cudaEventRecord(*pEvent, *pStream);
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaEventRecord(...) failed");

	return 1;
}

int NativeOps::setDevice(int deviceId) {
	auto dZ = cudaSetDevice(deviceId);
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaSetDevice(...) failed");

	return 1;
}

Nd4jLong NativeOps::getDeviceFreeMemory() {
    size_t memFree = 0;
    size_t memTotal = 0;

    cudaMemGetInfo(&memFree, &memTotal);

    return (Nd4jLong) memFree;
}

Nd4jLong NativeOps::getDeviceFreeMemory(int device) {
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

Nd4jLong NativeOps::getDeviceTotalMemory(int device) {
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

	cudaError_t dZ = cudaMemcpyAsync(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)), static_cast<size_t>(size), kind, *pStream);
	if (dZ != 0) {
        checkCudaErrors(dZ);
		printf("Failed on [%lu] -> [%lu], size: [%i], direction: [%i], dZ: [%i]\n", src, dst, size, flags, static_cast<int>(dZ));
        fflush(stdout);
        fflush(stderr);
        throw std::runtime_error("cudaMemcpyAsync(...) failed");
		//return 0L;
	}

	return 1;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaError_t dZ = cudaMemset(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size));
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaMemset(...) failed");

	return 1;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&reserved);

	cudaError_t dZ = cudaMemsetAsync(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size), *pStream);
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaMemsetAsync(...) failed");

	return 1;
}

int NativeOps::destroyEvent(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);
	cudaError_t dZ = cudaEventDestroy(*pEvent);
	checkCudaErrors(dZ);
	if (dZ != 0)
		throw std::runtime_error("cudaEvenDestroy(...) failed");

	return 1;
}

int NativeOps::streamSynchronize(Nd4jPointer stream) {
	cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(&stream);

	cudaError_t dZ = cudaStreamSynchronize(*pStream);
	checkCudaErrors(dZ);
	if (dZ != 0)
        throw std::runtime_error("cudaStreamSynchronize(...) failed");

	return 1L;
}

int NativeOps::eventSynchronize(Nd4jPointer event) {
	cudaEvent_t *pEvent = reinterpret_cast<cudaEvent_t *>(&event);

	cudaError_t dZ = cudaEventSynchronize(*pEvent);
	checkCudaErrors(dZ);
	if (dZ != 0)
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

int NativeOps::getDeviceMajor(int device) {
	return deviceProperties[device].major;
}

int NativeOps::getDeviceMinor(int device) {
	return deviceProperties[device].minor;
}


const char * NativeOps::getDeviceName(int device) {
    return deviceProperties[device].name;
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void concatCuda(const int numOfArrs, void* pVx,  void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

    __shared__ int arrIdx, blocksPerArr;
    __shared__ T *x, *z;
    __shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLen, arrLenZ, arrLenPerBlock, start, end;

    if (threadIdx.x == 0) {

        blocksPerArr = (gridDim.x - gridDim.x % numOfArrs) / numOfArrs;     // floor
        arrIdx = blockIdx.x / blocksPerArr;
        if (arrIdx >= numOfArrs)
            arrIdx = numOfArrs - 1;
        x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[arrIdx]);
        z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[arrIdx]);
        xShapeInfo = reinterpret_cast<Nd4jLong**>(pxShapeInfo)[arrIdx];
        zShapeInfo = reinterpret_cast<Nd4jLong**>(pzShapeInfo)[arrIdx];

        arrLen = shape::length(xShapeInfo);
        arrLenZ = shape::length(zShapeInfo);
        arrLenPerBlock = (arrLen + blocksPerArr - arrLen % blocksPerArr) / blocksPerArr;  // ceil

        start = arrLenPerBlock * (blockIdx.x % blocksPerArr);
        end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);
    }

    __syncthreads();
    for (Nd4jLong i = threadIdx.x + start; i < end; i += blockDim.x) {
        auto zOffset = shape::getIndexOffset(i, zShapeInfo, arrLenZ);
        auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLen);
        //printf("z[%i][%lld] = x[%i][%lld]\n", arrIdx, zOffset, arrIdx, xOffset);
        z[zOffset] = x[xOffset];
    }
}
template<typename T>
__host__ static void concatCudaLauncher(const int numOfArrs, cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo) {
    //int blocks = numOfArrs * 16; // >> 1 << 2);
    //nd4j_printf("gridDim.x is %i\n", blocks);
    //if (blocks > 8192)
    //    blocks = 8192; // restrict grid dims to 8K max
    concatCuda<T><<<numOfArrs, 128, 512, *stream>>>(numOfArrs, pVx, pxShapeInfo, pVz, pzShapeInfo);
    nd4j::DebugHelper::checkErrorCode(stream, "concat(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher, (const int numOfArrs, cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);

static void
specialBufferAndShapeWithOffset(void* vZ, Nd4jLong* hZShapeInfo, Nd4jLong* dZShapeInfo, std::vector<Nd4jLong> const& idx, void*& outBuffer, Nd4jLong*& outShape) {
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);
    const int rank = shape::rank(hZShapeInfo);
    Nd4jLong* newShape = new Nd4jLong[shape::shapeInfoLength(rank)];
    //ALLOCATE(newShape, nullptr, , Nd4jLong)
    auto shapeSize = shape::shapeInfoByteLength(rank);
    memcpy(newShape, hZShapeInfo, shapeSize);

    auto shapeOf = shape::shapeOf(newShape);
    auto stridesOf = shape::stride(newShape);

    Nd4jLong offset(0), subArrLen(1);
    int n(2), first, last, stride;

    for (int d = rank - 1; d >= 0; --d) {

        if (idx[n * d] != idx[n * d + 1]) {
            auto axeDim = shape::sizeAt(hZShapeInfo, d);
            first  = idx[n * d]     >= 0 ? idx[n * d]     : idx[n * d]     + axeDim + 1;
            last   = idx[n * d + 1] >= 0 ? idx[n * d + 1] : idx[n * d + 1] + axeDim + 1;
            stride = 1;

            shapeOf[d] = (last - first + stride - 1) / stride;      // ceil (last - first) / stride;
            offset += first * stridesOf[d];

            if(shapeOf[d] != 1)
                stridesOf[d] *= stride;
        }

        subArrLen *= shapeOf[d];
    }

    // check if there is possibility to set ews = 1
    shape::calcEws(newShape, subArrLen);

    //makeBothBuffersActual();
    outBuffer = (void*)((int8_t*)vZ + offset * DataTypeUtils::sizeOfElement(zType));
    cudaError_t err = cudaMalloc(&outShape, shapeSize);
    if (err != 0) {
        printf("Cannot allocate memory with error %d\n", err);
        throw std::runtime_error("Cannot allocate memory for shape");
    }
    cudaMemcpy(outShape, newShape, shapeSize, cudaMemcpyHostToDevice);
    delete [] newShape;
}

/**
  * Concatneate multi array of the same shape together
  * along a particular dimension
  */
 void NativeOps::concat(
		Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data, Nd4jPointer *inputShapeInfo,
		Nd4jPointer *ddata, Nd4jPointer *dinputShapeInfo,
		void *hZ, Nd4jLong *hZShapeInfo,
        void *dZ, Nd4jLong *dZShapeInfo,
		Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    auto stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    auto hXShapeInfo = hZShapeInfo;
    auto hShapePointers = reinterpret_cast<Nd4jLong **>(inputShapeInfo);
    auto dShapePointers = reinterpret_cast<Nd4jLong **>(dinputShapeInfo);
    // numArrays will be used as number of TADs, so each block process 1 input
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);
    auto axis = dimension;

    const int rank  = shape::rank(reinterpret_cast<Nd4jLong*>(inputShapeInfo[0]));
    const int rank2 = 2 * rank;
    std::vector<std::vector<Nd4jLong>> indices(numArrays, std::vector<Nd4jLong>(rank2 == 0?2:rank2,0));

    // take into account indices for first array
    auto axisSize = shape::sizeAt(reinterpret_cast<Nd4jLong*>(inputShapeInfo[0]), axis);
//    nd4j_printf("Set up indices...", "");
//    nd4j_printf("\n\n\tElement 0 at %i is setting\n", 2 * axis + 1);
    indices[0][2 * axis + 1] = axisSize;
//    nd4j_printf("\n\n\tElement 0 at %i was set\n", 2 * axis + 1);
    // loop through the rest of input arrays
    for(int i = 1; i < numArrays; ++i) {
//        nd4j_printf("\tIteration %i:\n", i);
        indices[i][2 * axis]     = indices[i - 1][2 * axis + 1];                                // index start from
//        nd4j_printf("\n\n\tindices[%i][%i] was set\n", i, 2 * axis);
        indices[i][2 * axis + 1] = indices[i - 1][2 * axis + 1] + shape::sizeAt(reinterpret_cast<Nd4jLong*>(inputShapeInfo[i]), axis);      // index end with (excluding)
//        nd4j_printf("\tindices[%i][%i] was set\n", i, 2 * axis + 1);
    }
//    nd4j_printf(" done\n", "");
//    nd4j_printf("Pack output shapes and buffers...", "");

    std::vector<void*> outSubArrsBuffs(numArrays);
    std::vector<Nd4jLong*> outSubArrsShapes(numArrays);
    for(int i = 0; i < numArrays; ++i) {
        specialBufferAndShapeWithOffset(dZ, hZShapeInfo, dZShapeInfo, indices[i], outSubArrsBuffs[i], outSubArrsShapes[i]);
    }
//    nd4j_printf(" done\n", "");

//    nd4j_printf("Prepare device pointers...", "");
    // prepare arrays of pointers on buffers and shapes
    std::vector<void*>     hOutBuffers(numArrays), hInBuffers(numArrays);
    std::vector<Nd4jLong*> hOutShapeInfo(numArrays), hInShapeInfo(numArrays);
    for(int i = 0; i < numArrays; ++i) {
        hOutBuffers[i]   = outSubArrsBuffs[i];
        hInBuffers[i]    = ddata[i];//->getSpecialBuffer();
        hOutShapeInfo[i] = outSubArrsShapes[i];
        hInShapeInfo[i]  = (Nd4jLong*)(dShapePointers[i]);//->getSpecialShapeInfo();
//        nd4j_printf("X_%i shape ptr: %p; data ptr: %p;\n", i, hInShapeInfo[i], hInBuffers[i]);
    }
//    nd4j_printf(" done\n", "");
    LaunchContext context(stream);
    // allocate and copy all buffers and shapes arrays to global memory
    PointersManager manager(&context, "NativeOps::concat");
    void* dOutBuffers	= manager.replicatePointer(hOutBuffers.data(),   hOutBuffers.size() * sizeof(void*));
    void* dInBuffers	= manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
    void* dInShapeInfo  = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));
    void* dOutShapeInfo = manager.replicatePointer(hOutShapeInfo.data(), hOutShapeInfo.size() * sizeof(Nd4jLong*));

//    nd4j_printf("Concat itself run...", "");

    BUILD_SINGLE_SELECTOR(zType, concatCudaLauncher, (numArrays, stream, dInBuffers, dInShapeInfo, dOutBuffers, dOutShapeInfo), LIBND4J_TYPES);
    manager.synchronize();
//    nd4j_printf(" done\n", "");

//    nd4j_printf("Postprocessing...", "");

//    cudaError_t res = cudaStreamSynchronize(*stream);
//    checkCudaErrors(res);
//    nd4j::DebugHelper::checkErrorCode(stream, "Legacy ConcatFloat(...) failed");
//    nd4j_printf(" done\n", "");
//    nd4j_printf("Free up rest...", "");
    cudaError_t err;
    for(int i = 0; i < numArrays; ++i) {
        err = cudaFree(outSubArrsShapes[i]);
        if (err != 0) {
            printf("Error %d occured when shape %i was deallocating.\n", err, i);
            throw std::runtime_error("Cannot deallocate memory for shapes.");
        }
    }
//    nd4j_printf(" done\n", "");
//    nd4j_printf("All done!!!\n", "");
}



void NativeOps::specialConcat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        void *dZ,
        Nd4jLong *dZShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            dZ,
            dZShapeInfo);

}


/**
 * This method saves
 */
void NativeOps::tadOnlyShapeInfo(Nd4jLong *dXShapeInfo, int *dimension, int dimensionLength, Nd4jLong *target, Nd4jLong *offsets) {
	shape::TAD tad;
	tad.init(dXShapeInfo, dimension, dimensionLength);
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
	//cudaError_t dZ = cudaMemcpyAsync((void *) dst, (const void *) src, (size_t) size, kind, *pStream);
	cudaError_t dZ = cudaMemcpyToSymbolAsync(deviceConstantMemory, const_cast<const void *>(src), size, dst, kind, *pStream);
	checkCudaErrors(dZ);
	if (dZ != 0)
        throw std::runtime_error("cudaMemcpyToSymbolAsync(...) failed");

	return 1;
}

Nd4jPointer NativeOps::getConstantSpace() {
	Nd4jPointer dConstAddr;
	cudaError_t dZ = cudaGetSymbolAddress(reinterpret_cast<void **>(&dConstAddr), deviceConstantMemory);

	if (dZ != 0)
        throw std::runtime_error("cudaGetSymbolAddress(...) failed");

	return dConstAddr;
}

void NativeOps::pullRows(Nd4jPointer *extraPointers,
						 void *x, Nd4jLong *xShapeInfo,
						 void *dX, Nd4jLong *dXShapeInfo,
						 void *z, Nd4jLong *zShapeInfo,
						 void *dZ, Nd4jLong *dZShapeInfo,
						 Nd4jLong n,
						 Nd4jLong *indexes,
						 Nd4jLong *tadShapeInfo,
						 Nd4jLong *tadOffsets,
						 Nd4jLong *zTadShapeInfo,
						 Nd4jLong *zTadOffsets) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);	
	dim3 launchDims(64, 256, 1024);
	auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, pullRowsKernelGeneric, (launchDims, stream, dX, dZ, n, indexes, tadShapeInfo, tadOffsets,  zTadShapeInfo,  zTadOffsets), LIBND4J_TYPES);
	
	DEBUG_KERNEL(stream, -1);
}


void NativeOps::average(Nd4jPointer *extras,
						Nd4jPointer *x, Nd4jLong *xShapeInfo,
						Nd4jPointer *dx, Nd4jLong *dXShapeInfo,
						void *z, Nd4jLong *zShapeInfo,
						void *dz, Nd4jLong *dzShapeInfo,
						int n,
						Nd4jLong length,
						bool propagate) {

	cudaStream_t * stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	auto dX = reinterpret_cast<void **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("averageFloat called\n");

	auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
	// launching on gpu
	if (mode == 0) {
		dim3 launchDims(256, 256, 4096);
    	BUILD_SINGLE_SELECTOR(xType, averagingKernelGeneric, (launchDims, stream, dX, dz, n, length, propagate), LIBND4J_TYPES);		    	
        nd4j::DebugHelper::checkErrorCode(stream, "AverageFloat(...) failed");
	} else {
		// launching on host memory
        BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::averageGeneric(x, z, zShapeInfo, n, length, propagate), LIBND4J_TYPES);
	}
}

void NativeOps::accumulate(Nd4jPointer *extras,
						   Nd4jPointer *x, Nd4jLong *xShapeInfo,
						   Nd4jPointer *dx, Nd4jLong *dXShapeInfo,
						   void *z, Nd4jLong *zShapeInfo,
						   void *dz, Nd4jLong *dzShapeInfo,
						   int n,
						   Nd4jLong length) {
	
	auto stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
	int mode = getDeviceId(extras[3]);

	auto dX = reinterpret_cast<void **>(dx);

	if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		printf("accumulateFloat called\n");
	auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

	// launching on gpu
	if (mode == 0) {
		dim3 launchDims(n, 256, 16384);
        BUILD_SINGLE_SELECTOR(xType, accumulateKernelGeneric, (launchDims, stream, dX, dz, n,length), LIBND4J_TYPES);
        nd4j::DebugHelper::checkErrorCode(stream, "AccumulateFloat(...) failed");
	} else {
		// launching on host memory        
        BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::accumulateGeneric(x, z, zShapeInfo, n, length), LIBND4J_TYPES);
	}
}


void NativeOps::shuffle(Nd4jPointer *extras,
						Nd4jPointer *x, Nd4jPointer *xShapeInfo,
						Nd4jPointer *dx, Nd4jPointer *dXShapeInfo,
						Nd4jPointer *z, Nd4jPointer *zShapeInfo,
						Nd4jPointer *dz, Nd4jPointer *dZShapeInfo,
						int N,
						int *shuffleMap,
						Nd4jPointer *tadShapeInfo,
						Nd4jPointer *tadOffsets) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    auto dX = reinterpret_cast<void **>(dx);
    auto dZ = reinterpret_cast<void **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto dxShape = reinterpret_cast<Nd4jLong **>(dXShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    auto xType = nd4j::ArrayOptions::dataType(xShape[0]);
    dim3 launchDims(256, 512, 8192);
    BUILD_SINGLE_SELECTOR(xType, shuffleKernelGeneric, (launchDims, stream, dX, dxShape, dZ, N, shuffleMap,  tadOnlyShapeInfo, tadOffset), LIBND4J_TYPES);

    nd4j::DebugHelper::checkErrorCode(stream, "shuffle(...) failed");
}

/*
void NativeOps::execMetaPredicateShape(Nd4jPointer *extras, 
	                                  const int opTypeA, 
	                                  const int opNumA, 
	                                  const int opTypeB, 
	                                  const int opNumB, 
	                                  Nd4jLong N, 
	                                  void *hX, Nd4jLong *hXShapeInfo,
                                      void *dX, Nd4jLong *dXShapeInfo,
                                      void *hY, Nd4jLong *hYShapeInfo,
                                      void *dY, Nd4jLong *dYShapeInfo,
                                      void *hZ, Nd4jLong *hZShapeInfo,
                                      void *dZ, Nd4jLong *dZShapeInfo,
	                                  void *extraA, 
	                                  void *extraB, 
	                                  double scalarA, 
	                                  double scalarB) {
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, functions::grid::GRIDShaped, ::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraA, extraB, scalarA, scalarB), LIBND4J_TYPES);
    // functions::grid::GRIDShaped<float>::execMetaPredicateShaped(stream, extras, opTypeA, opNumA, opTypeB, opNumB, N, dX, dXShapeInfo, dy, dYShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

	DEBUG_KERNEL(stream, opNumA);
}
*/

bool NativeOps::isExperimentalEnabled() {
    return nd4j::Environment::getInstance()->isExperimentalBuild();
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

////////////////////////////////////////////////////////////////////////
void NativeOps::execSummaryStats(Nd4jPointer *extraPointers,
                                 int opNum,
                                 void *hX, Nd4jLong *hXShapeInfo,
                                 void *dX, Nd4jLong *dXShapeInfo,
                                 void *extraParams,
                                 void *hZ, Nd4jLong *hZShapeInfo,
                                 void *dZ, Nd4jLong *dZShapeInfo,
                                 bool biasCorrected) {

	NativeOpExecutioner::execSummaryStats(nullptr,opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, biasCorrected);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execSummaryStats(Nd4jPointer *extraPointers,
                                 int opNum,
                                 void *hX, Nd4jLong *hXShapeInfo,
                                 void *dX, Nd4jLong *dXShapeInfo,
                                 void *extraParams,
                                 void *hZ, Nd4jLong *hZShapeInfo,
                                 void *dZ, Nd4jLong *dZShapeInfo,
								 void *hDimension, Nd4jLong *hDimensionShape, void *dDimension, Nd4jLong *dDimensionShape,
                                 bool biasCorrected,
								 Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	NativeOpExecutioner::execSummaryStats(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduce3(Nd4jPointer *extraPointers,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo) {

	NativeOpExecutioner::execReduce3(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduce3(Nd4jPointer *extraPointers,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
							void *hDimension, Nd4jLong *hDimensionShape, void *dDimension, Nd4jLong *dDimensionShape,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	// if (extraPointers == nullptr || extraPointers[2] == 0)
 //        NativeOpExecutioner::execReduce3(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
 //    else {
 //        // going tad-ways
 //        auto tadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[0]);
 //        auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

 //        NativeOpExecutioner::execReduce3TAD(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
 //    }

    auto tadLength = shape::length(tadOnlyShapeInfo);
    auto yLength = shape::length(hYShapeInfo);

    if (tadLength != yLength)
        NativeOpExecutioner::execReduce3(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
    else
        NativeOpExecutioner::execReduce3TAD(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, yTadOffsets);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduce3Scalar(Nd4jPointer *extraPointers,int opNum,
                                  void *hX, Nd4jLong *hXShapeInfo,
                                  void *dX, Nd4jLong *dXShapeInfo,
                                  void *extraParams,
                                  void *hY, Nd4jLong *hYShapeInfo,
                                  void *dY, Nd4jLong *dYShapeInfo,
                                  void *hZ, Nd4jLong *hZShapeInfo,
                                  void *dZ, Nd4jLong *dZShapeInfo) {

	NativeOpExecutioner::execReduce3Scalar(nullptr, opNum,hX,hXShapeInfo,dX, dXShapeInfo,extraParams,hY,hYShapeInfo,dY,dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execScalarBool(Nd4jPointer *extraPointers,
							int opNum,
							void *hX, Nd4jLong *hXShapeInfo,
							void *dX, Nd4jLong *dXShapeInfo,
							void *hZ, Nd4jLong *hZShapeInfo,
							void *dZ, Nd4jLong *dZShapeInfo,
							void *hScalar, Nd4jLong *hScalarShapeInfo,
							void *dScalar, Nd4jLong *dScalarShapeInfo,
							void *extraParams) {
	
	NativeOpExecutioner::execScalarBool(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalar, hScalarShapeInfo, dScalar, dScalarShapeInfo, extraParams);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execScalarBool(Nd4jPointer *extraPointers,
						   int opNum,
						   void *hX, Nd4jLong *hXShapeInfo,
						   void *dX, Nd4jLong *dXShapeInfo,
						   void *hZ, Nd4jLong *hZShapeInfo,
						   void *dZ, Nd4jLong *dZShapeInfo,
						   void *hScalars, Nd4jLong *hScalarShapeInfo,
						   void *dScalars, Nd4jLong *dScalarShapeInfo,
						   void *extraParams,
							   void *hDimension, Nd4jLong *hDimensionShape, void *dDimension, Nd4jLong *dDimensionShape,
                           Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                           Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

	NativeOpExecutioner::execScalarBool(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalars, hScalarShapeInfo, dScalars, dScalarShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execScalar(Nd4jPointer *extraPointers,
						int opNum,
						void *hX, Nd4jLong *hXShapeInfo,
						void *dX, Nd4jLong *dXShapeInfo,
						void *hZ, Nd4jLong *hZShapeInfo,
						void *dZ, Nd4jLong *dZShapeInfo,
						void *hScalar, Nd4jLong *hScalarShapeInfo,
						void *dScalar, Nd4jLong *dScalarShapeInfo,
						void *extraParams) {
	
	NativeOpExecutioner::execScalar(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalar, hScalarShapeInfo, dScalar, dScalarShapeInfo, extraParams);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execScalar(Nd4jPointer *extraPointers,
					 int opNum,
					 void *hX, Nd4jLong *hXShapeInfo,
                     void *dX, Nd4jLong *dXShapeInfo,
                     void *hZ, Nd4jLong *hZShapeInfo,
                     void *dZ, Nd4jLong *dZShapeInfo,
                     void *hScalars, Nd4jLong *hScalarShapeInfo,
                     void *dScalars, Nd4jLong *dScalarShapeInfo,
					 void *extraParams,
						   void *hDimension, Nd4jLong *hDimensionShape,
						   void *dDimension, Nd4jLong *dDimensionShape,
                     Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                     Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    auto yType = nd4j::ArrayOptions::dataType(hScalarShapeInfo);
    auto zType = nd4j::ArrayOptions::dataType(hZShapeInfo);

	if (yType != xType && yType != nd4j::DataType::BOOL && !this->isExperimentalEnabled())
		throw nd4j::datatype_exception::build("NativeOps::execScalar both operands must have same data type", xType, yType);

	dim3 launchDims(256, 256, 16384);

#ifdef __ND4J_EXPERIMENTAL__
    BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform, ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES, LIBND4J_TYPES);
#else
	BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform, ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), LIBND4J_TYPES);
#endif

	DEBUG_KERNEL(stream, opNum);
}

void NativeOps::execAggregate(Nd4jPointer *extraPointers,
								   int opNum,
                                   void **arguments,
                                   int numArguments,
                                   Nd4jLong **shapes,
                                   int numShapes,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   void *realArguments,
                                   int numRealArguments,
                                   nd4j::DataType dtype) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    dim3 launchDims = dim3(numBlocks, numThreads, shmem);
	
    BUILD_SINGLE_SELECTOR(dtype, functions::aggregate::AggregatedFunction, ::aggregateKernelGeneric(launchDims, stream, opNum, arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), FLOAT_TYPES);
    nd4j::DebugHelper::checkErrorCode(stream, "execAggregateFloat(...) failed");
}

void NativeOps::execAggregateBatch(Nd4jPointer *extraPointers, 
									int numAggregates, int opNum, 
									int maxArgs, int maxShapes, 
									int maxIntArrays, int maxIntArraySize, 
									int maxIdx, int maxReals,  
									void *ptrToArguments, nd4j::DataType dtype) {
    // not implemented yet
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int numBlocks = getDeviceId(extraPointers[2]);
    int numThreads = getDeviceId(extraPointers[3]);
    int shmem = getDeviceId(extraPointers[4]);

    dim3 launchDims = dim3(numAggregates, numThreads, shmem);

	BUILD_SINGLE_SELECTOR(dtype, functions::aggregate::AggregatedFunction, ::aggregateBatchKernelGeneric(launchDims, stream, opNum, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), FLOAT_TYPES);

	DEBUG_KERNEL(stream, opNum);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execRandom(Nd4jPointer *extraPointers,
						  int opNum,
                          Nd4jPointer stateHost,
                          void *hZ, Nd4jLong *hZShapeInfo,
                          void *dZ, Nd4jLong *dZShapeInfo,
                          void *extraArguments) {

    NativeOpExecutioner::execRandom(nullptr, opNum, extraPointers, hZ, hZShapeInfo, dZ, dZShapeInfo, extraArguments);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execRandom(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost,
						   void *hX, Nd4jLong *hXShapeInfo, 
						   void *dX, Nd4jLong *dXShapeInfo, 
						   void *hZ, Nd4jLong *hZShapeInfo, 
						   void *dZ, Nd4jLong *dZShapeInfo, 
						   void *extraArguments) {
    
    NativeOpExecutioner::execRandom(nullptr, opNum, extraPointers, hX, hXShapeInfo, dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraArguments);
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execRandom(Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost,
							void *hX, Nd4jLong *hXShapeInfo,
							void *dX, Nd4jLong *dXShapeInfo,
							void *hY, Nd4jLong *hYShapeInfo,
							void *dY, Nd4jLong *dYShapeInfo,
							void *hZ, Nd4jLong *hZShapeInfo,
							void *dZ, Nd4jLong *dZShapeInfo,
							void *extraArguments) {

    NativeOpExecutioner::execRandom(nullptr, opNum, extraPointers, hX, hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, extraArguments);
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
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */

Nd4jPointer NativeOps::pointerForAddress(Nd4jLong address) {
	return reinterpret_cast<Nd4jPointer >(address);
}

void NativeOps::tear(Nd4jPointer *extras,
					 void *x, Nd4jLong *xShapeInfo,
					 void *dX, Nd4jLong *dXShapeInfo,
					 Nd4jPointer *targets,
					 Nd4jLong *zShapeInfo,
					 Nd4jLong *tadShapeInfo,
					 Nd4jLong *tadOffsets) {
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
    dim3 launchDims(512, 512, 512);   
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, tearKernelGeneric, (launchDims, stream, dX, dXShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets), LIBND4J_TYPES);

    nd4j::DebugHelper::checkErrorCode(stream, "tearFloat(...) failed");
}


void prescanArrayRecursive(Nd4jPointer *extras, int *dZ, int *dX, int numElements, int level) {

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
        nd4j::prescanLauncher<true, false>(grid, threads, sharedMemSize, stream, dZ, dX, g_scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            nd4j::prescanLauncher<true, true>(gridOnes, threadsOnes, sharedMemLastBlock, stream, dZ, dX, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be sdded to each block to
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(extras, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        nd4j::uniformAdd<<<grid, threads, 1024, *stream>>>(dZ, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);

        if (np2LastBlock) {
            nd4j::uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(dZ, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        nd4j::prescanLauncher<false, false>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numThreads * 2, 0, 0);
    } else {
        nd4j::prescanLauncher<false, true>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numElements, 0, 0);
    }

    nd4j::DebugHelper::checkErrorCode(stream, "prescanArray(...) failed");
}


void NativeOps::encodeThresholdP1(Nd4jPointer *extras, void *dx, Nd4jLong *hXShapeInfo, Nd4jLong N, int *dz, float threshold) {
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);
    
    dim3 launchDims(numBlocks, blockSize, 1024);
    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, encoderKernelP1Generic, (launchDims, stream, dx, N, dz, threshold), LIBND4J_TYPES);        

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP1Float(...) failed");
}



void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jLong N, int *dz) {
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    //encoderKernelP2Float<<<numBlocks, blockSize , 1024 * sizeof(float), *stream>>>(dx, N, dz);    
    prescanArrayRecursive(extraPointers, dz, dx + 1, (int) N, 0);
    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP2Int(...) failed");
}

void NativeOps::encodeThresholdP3(Nd4jPointer *extraPointers, void *dx, Nd4jLong *hXShapeInfo, int *offsets, Nd4jLong N, int *dz){
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int blockSize = 1024;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);
    
    dim3 launchDims(numBlocks, blockSize, 4096);
    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, encoderKernelP3Generic, (launchDims, stream, dx, offsets, N, dz), LIBND4J_TYPES);    

    nd4j::DebugHelper::checkErrorCode(stream, "encodeThresholdP3Float(...) failed");
}

void NativeOps::decodeThreshold(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, void *dz, Nd4jLong *zShapeInfo){
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    // we probably want to have smaller blocks here, memory writes are misaligned anyway
    int blockSize = 128;
    int numBlocks = N / blockSize + (N % blockSize ? 1 : 0);
    
    dim3 launchDims(numBlocks, blockSize, 1024);
    auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);
    BUILD_SINGLE_SELECTOR(zType, decoderKernelGeneric, (launchDims, stream, dx, N, dz), LIBND4J_TYPES);    

    nd4j::DebugHelper::checkErrorCode(stream, "decodeThresholdFloat(...) failed");
}

////////////////////////////////////////////////////////////////////////
void NativeOps::execReduce3All(Nd4jPointer *extraPointers,
									int opNum,
									void *hX, Nd4jLong *hXShapeInfo,
                            		void *dX, Nd4jLong *dXShapeInfo,
                            		void *extraParamsVals,
									void *hY, Nd4jLong *hYShapeInfo,
                            		void *dY, Nd4jLong *dYShapeInfo,
                            		void *hZ, Nd4jLong *hZShapeInfo,
                            		void *dZ, Nd4jLong *dZShapeInfo,
							   		void *hDimension, Nd4jLong *hDimensionShape,
							   		void *dDimension, Nd4jLong *dDimensionShape,
									Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
									Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {
	auto dimension = reinterpret_cast<int *>(dDimension);
	int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    NativeOpExecutioner::execReduce3All(nullptr, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}


void NativeOps::sort(Nd4jPointer *extraPointers,
					 void *x, Nd4jLong *xShapeInfo,
					 void *dX, Nd4jLong *dXShapeInfo,
					 bool descending) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[     1]);

    auto xLength = shape::length(xShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);


    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
        int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
        int numBlocks = xLength / numThreads;
        if (xLength % numThreads > 0 || numBlocks == 0)
            numBlocks++;

        dim3 launchDims(numBlocks, numThreads, 32768);

        for (int k = 2; k <= xLength; k = 2*k) {
            for (int j = k >> 1; j > 0; j = j >> 1) {
				BUILD_SINGLE_SELECTOR(xType, bitonicSortStepGeneric, (launchDims, stream, dX, dXShapeInfo, j, k, xLength, descending), LIBND4J_TYPES);
			}
        }
    } else {
    	int numThreads = nd4j::math::nd4j_min<int>(512, xLength);
    	int numBlocks = xLength / numThreads;
    	if (xLength % numThreads > 0 || numBlocks == 0)
    		numBlocks++;

    	numBlocks = nd4j::math::nd4j_min<int>(512, numBlocks);
    	dim3 launchDims(numBlocks, numThreads, 32768);

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
    			BUILD_SINGLE_SELECTOR(xType, bitonicArbitraryStepGeneric, (launchDims, stream, dX, dXShapeInfo, n, xLength, rev, descending), LIBND4J_TYPES);
    			n>>=1;
    			rev = 1;
    		} while(n > 1);
    	}
    }

    nd4j::DebugHelper::checkErrorCode(stream, "sort(...) failed");
}


void NativeOps::sortTad(Nd4jPointer *extraPointers,
						void *x, Nd4jLong *xShapeInfo,
						void *dX, Nd4jLong *dXShapeInfo,
						int *dimension,
						int dimensionLength,
						Nd4jLong *tadShapeInfo,
						Nd4jLong *tadOffsets,
						bool descending) {
    // to be implemented
    auto stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);

    dim3 launchDims(tadPack.numberOfTads(), 1024, 33768);

	auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, oesTadGeneric, (launchDims, stream, dX, dXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending), LIBND4J_TYPES);                     
    
    nd4j::DebugHelper::checkErrorCode(stream, "sortTadFloat(...) failed");
}

void NativeOps::sortCooIndices(Nd4jPointer *extraPointers, Nd4jLong *indices, void *values, Nd4jLong length, int rank) {
	throw std::runtime_error("sortCooIndices:: Not implemented yet");
}


Nd4jLong NativeOps::encodeBitmap(Nd4jPointer *extraPointers, 
								void *dx, Nd4jLong *hXShapeInfo,
								Nd4jLong N, 
								int *dz, 
								float threshold) {
    
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);
    int *resultPointer = reinterpret_cast<int *>(extraPointers[2]);
    int *reductionPointer = reinterpret_cast<int *>(extraPointers[3]);
        
    dim3 launchDims(512, 512, 32768);
    auto xType = nd4j::ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, cudaEncodeBitmapGeneric, (launchDims, stream, dx, N, dz, resultPointer, reductionPointer, threshold), LIBND4J_TYPES);     

    nd4j::DebugHelper::checkErrorCode(stream, "encodeBitmapFloat(...) failed");

    Nd4jLong dZ = (Nd4jLong) resultPointer[0];
    resultPointer[0] = 0;

    return dZ;
}


void NativeOps::decodeBitmap(Nd4jPointer *extraPointers, 
							void *dx,
							Nd4jLong N, 
							void *dz, Nd4jLong *zShapeInfo) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);        
    dim3 launchDims(512, 512, 16384);
    auto xType = nd4j::ArrayOptions::dataType(zShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, cudaDecodeBitmapGeneric, (launchDims, stream, dx, N, dz), LIBND4J_TYPES);

    nd4j::DebugHelper::checkErrorCode(stream, "decodeBitmapFloat(...) failed");
}

Nd4jLong* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length) {
	return nullptr;
}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jLong* ptrMap, Nd4jLong length) {

}


nd4j::graph::ResultWrapper* NativeOps::executeFlatGraph(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
}


const char* NativeOps::getAllCustomOps() {
	return nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}


nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool *bArgs, int numBArgs) {
    nd4j::graph::VariableSpace varSpace;
    Context block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

	for (int e = 0; e < numBArgs; e++)
		block.getBArguments()->push_back(bArgs[e]);

	for (int e = 0; e < numInputShapes; e++) {
		auto shape_ = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

		// we shouldn't copy buffer if that's empty array
		void *buffer_ = nd4j::ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

		auto array = new nd4j::NDArray(buffer_, shape_);
		array->triggerAllocationFlag(false);

		// block should contain references to proper variable
		varSpace.putVariable(1, e, array);
		block.pickInput(1, e);

		inShapes.push_back(shape_);
	}

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    if (varSpace.launchContext()->getWorkspace() != nullptr)
        shapeList->detach();

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapes(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool *bArgs, int numBArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs);
}

nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp* op, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    Context block(1);
	nd4j::ShapeList inShapes;

	for (int e = 0; e < numIArgs; e++)
		block.getIArguments()->push_back(iArgs[e]);

	for (int e = 0; e < numTArgs; e++)
		block.getTArguments()->push_back(tArgs[e]);

	for (int e = 0; e < numInputShapes; e++)
		inShapes.push_back(reinterpret_cast<Nd4jLong *>(inputShapes[e]));

	auto shapeList = op->calculateOutputShape(&inShapes, block);

	return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapes(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);

	return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}


static FORCEINLINE Nd4jStatus realExec(nd4j::ops::DeclarableOp* op, Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool* bArgs, int numBArgs, bool isInplace) {
	if (op == nullptr)
		nd4j_printf("Can't find requested operation: [%lld]\n", hash);

	// we're using the same fake nodeId everywhere here

	std::vector<nd4j::NDArray*> inputs(numInputs);
	std::vector<nd4j::NDArray*> outputs(numOutputs);
	std::vector<double> ttArgs(numTArgs);
	std::vector<bool> bbArgs(numBArgs);
	std::vector<Nd4jLong> iiArgs(numIArgs);

	// filling block now with inputs
	for (int e = 0; e < numInputs; e++) {
		auto shape = reinterpret_cast<Nd4jLong *>(inputShapes[e]);
		void *buffer = nd4j::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

		inputs[e] = new nd4j::NDArray(buffer, shape);
	}

	// if not inplace - transferring output arrays

	if (!isInplace)
		for (int e = 0; e < numOutputs; e++) {
			// we want to keep original output shape intact
			auto shape = shape::copyShape(reinterpret_cast<Nd4jLong *>(outputShapes[e]));
			void *buffer = nd4j::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : outputBuffers[e];

			// FIXME: revisit this.
			bool canNullify = true;
			for (int i = 0; i < numInputs; i++) {
				void *ibuffer = nd4j::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[i];
				if (ibuffer == buffer) {
					canNullify = false;
					break;
				}
			}

			if (canNullify)
				memset((uint8_t *) buffer, '\0', shape::length(shape) * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape)));

			auto array = new nd4j::NDArray(buffer, shape);
			outputs[e] = array;

			// and we want to release shape copy once we're done
			array->triggerAllocationFlag(false);
		}

	for (int e = 0; e < numIArgs; e++)
		iiArgs[e] = iArgs[e];

	for (int e = 0; e < numTArgs; e++)
		ttArgs[e] = tArgs[e];

    for (int e = 0; e < numBArgs; e++)
        bbArgs[e] = bArgs[e];


	// hypothetically at this point we have everything filled
	auto dZ = op->execute(inputs, outputs, ttArgs, iiArgs, bbArgs, isInplace);
	//auto dZ = op->execute(inputs, ttArgs, iiArgs, isInplace);


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
        if (dZ->size() != numOutputs) {
            return ND4J_STATUS_BAD_OUTPUT;
        }

        for (int e = 0; e < numOutputs; e++) {
            auto buffer = (T *) outputBuffers[e];
            auto shape = (int *) outputShapes[e];
            nd4j::NDArray<T> tmp(buffer, shape);

            if (tmp.lengthOf() != dZ->at(e)->lengthOf()) {
                nd4j_printf("Provided output array for [%s] has length of %i, but actual dZ has length of %i\n", op->getOpName()->c_str(), tmp.lengthOf(), dZ->at(e)->lengthOf());
                return ND4J_STATUS_BAD_OUTPUT;
            }

            tmp.assign(dZ->at(e));
        }
    } else {
        // if op is inplace, our ResultSet holds pointers
        dZ->purge();
    }


    delete dZ;

*/

	for (auto v: inputs)
		delete v;

	for (auto v: outputs)
		delete v;

	return Status::OK();
}


int NativeOps::execCustomOp(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool* bArgs, int numBArgs, bool isInplace) {
	auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);

	return realExec(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs, isInplace);
}

int NativeOps::execCustomOp(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer opContext) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);
    auto context = reinterpret_cast<Context*>(opContext);

    return op->execute(context);
}

int NativeOps::registerGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
	
	auto graph = nd4j::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

	nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

	return ND4J_STATUS_OK;
}


static VariablesSet* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	auto graph = nd4j::graph::GraphHolder::getInstance()->pullGraph(graphId);
	auto varSpace = graph->getVariableSpace()->clone();

	std::vector<nd4j::NDArray*> handles;

	for (int e = 0; e < numInputs; e++) {
		auto idx = inputIndices[e];

		// we'll delete this array later, together with cloned VariableSpace
		auto array = new nd4j::NDArray(inputBuffers[e], reinterpret_cast<Nd4jLong *>(inputShapes[e]));
		handles.emplace_back(array);

		if (varSpace->hasVariable(idx)) {
			auto var = varSpace->getVariable(idx);
			if (var->hasNDArray())
				delete var->getNDArray();

			var->setNDArray(array);
		} else
			varSpace->putVariable(idx, array);
	}

	auto dZ = nd4j::graph::GraphExecutioner::execute(graph, varSpace);
	auto varSet = new nd4j::graph::VariablesSet(dZ);

	if (dZ == ND4J_STATUS_OK) {
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

VariablesSet* NativeOps::executeStoredGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
	return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
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
	nd4j::graph::VariablesSet* ptr = reinterpret_cast<nd4j::graph::VariablesSet*>(pointer);
	delete ptr;
}

void NativeOps::deleteVariablesSet(Nd4jPointer pointer) {
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

Nd4jPointer NativeOps::getGraphState(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState(id);
}


void NativeOps::deleteGraphState(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState*>(state);
    delete stateP;
}


Nd4jStatus execCustomOpWithScope(Nd4jPointer *extraPointers, nd4j::graph::GraphState *state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    /**
     * That's basically exec, with VariableSpace provided in GraphState:
     * depending on operation (i.e. while of if), different logic executors could be used
     */

    auto graph = state->graph();
    auto varSpace = state->variableSpace();

    // Node is dynamically created, and has nothing beyond it: only inputs and outputs
    // this node has id of 0, and inputs are
    Node node(OpType_LOGIC, opHash, 0);

    // mapping inputs
    for (int e = 0; e < numInputs; e++) {
        auto buffer = inputBuffers[e];
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        auto array = new nd4j::NDArray(buffer, shapeInfo, varSpace->launchContext());

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

    auto dZ = LogicExecutor::processNode(graph, &node);
    if (dZ != Status::OK())
        return dZ;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = outputBuffers[e];
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(outputShapes[e]);

        NDArray array(buffer, shapeInfo, varSpace->launchContext());

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

           
Nd4jStatus NativeOps::execCustomOpWithScope(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    
    return execCustomOpWithScope(extraPointers, reinterpret_cast<nd4j::graph::GraphState*>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

void NativeOps::deleteResultWrapper(Nd4jPointer ptr) {
	// just 0 room for compiler s@!t
	auto p = reinterpret_cast<nd4j::graph::ResultWrapper *>(ptr);
	delete p;
}

int NativeOps::estimateThreshold(Nd4jPointer *extraPointers, Nd4jPointer dX, Nd4jLong *dXShapeInfo, int N, float threshold) {
	throw std::runtime_error("estimateThreshold: Not implemented yet");
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer dX, long N, int dstType, Nd4jPointer dZ);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer dX, Nd4jLong N, int dstType, Nd4jPointer dZ) {
 	auto dx = reinterpret_cast<void *>(dX);
	auto dz = reinterpret_cast<void *>(dZ);

    if (srcType == ND4J_FLOAT8) {
        if (dstType == ND4J_FLOAT8) {
            // convertKernel<double, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::uint8>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::int16>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, nd4j::uint16>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::float8, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            //nd4j::TypeCast::convertGenericCuda<nd4j::int8, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //convertKernel<nd4j::int8, nd4j::int8>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<int8_t, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<int8_t, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<int8_t, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<int8_t, uint16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: eventually we might want to add it
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<int8_t, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<int8_t, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
            //nd4j::TypeCast::convertGenericCuda<uint8_t, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, int8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, uint16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: still might want to add
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<uint8_t, double>(extras, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
            //nd4j::TypeCast::convertGenericCuda<float16, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<float16, int8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<float16, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<float16, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<float16, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<float16, uint16_t>(extras, dx, N, dz);
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
            //nd4j::TypeCast::convertGenericCuda<int16_t, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<int16_t, int8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<int16_t, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<int16_t, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<int16_t, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<int16_t, uint16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO...
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGenericCuda<int16_t, float>(extras, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGenericCuda<int16_t, double>(extras, dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
            //nd4j::TypeCast::convertGenericCuda<float, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<float, int8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<float, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<float, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<float, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<float, uint16_t>(extras, dx, N, dz);
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
            //nd4j::TypeCast::convertGenericCuda<double, nd4j::float8>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGenericCuda<double, int8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGenericCuda<double, uint8_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGenericCuda<double, float16>(extras, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGenericCuda<double, int16_t>(extras, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGenericCuda<double, uint16_t>(extras, dx, N, dz);
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

Nd4jPointer NativeOps::createUtf8String(Nd4jPointer *extraPointers, const char *string, int length) {
    auto u = new nd4j::utf8string(string, length);
    return reinterpret_cast<Nd4jPointer>(u);
}

void NativeOps::deleteUtf8String(Nd4jPointer *extraPointers, Nd4jPointer ptr) {
    delete(reinterpret_cast<nd4j::utf8string*>(ptr));
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void scatterUpdateCuda(const int opCode, const int numOfSubArrs,
										      void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
										      void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
										      const int* indexes) {

    __shared__ T *x, *y;
    __shared__ Nd4jLong arrLenX, arrLenY;

    for (int e = 0; e < numOfSubArrs; e++ ) {

        const auto xIndex = indexes[e];
        const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

        if (!isOwner)
            continue;

        if (threadIdx.x == 0) {
            x = reinterpret_cast<T*>(vx) + xOffsets[xIndex];
            y = reinterpret_cast<T*>(vy) + yOffsets[e];
            arrLenX = shape::length(xShapeInfo);
            arrLenY = shape::length(yShapeInfo);
        }

        __syncthreads();

        if (arrLenX != arrLenY)
            return;

        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

            const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
            const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

            switch (opCode) {
                case 0:
                    x[xOffset] += y[yOffset];
                    break;
                case 1:
                    x[xOffset] -= y[yOffset];
                    break;
                case 2:
                    x[xOffset] *= y[yOffset];
                    break;
                case 3:
                    x[xOffset] /= y[yOffset];
                    break;
                case 4:
                    x[xOffset] = y[yOffset] - x[xOffset];
                    break;
                case 5:
                    x[xOffset] = y[yOffset] / x[xOffset];
                    break;
                case 6:
                    x[xOffset] = y[yOffset];
                    break;
                default:
                    continue;
            }
        }
        __syncthreads();
    }
}

template<typename T>
__host__ static void scatterUpdateCudaLauncher(const cudaStream_t* stream, const int opCode, const int numOfSubArrs, void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets, void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets, const int* indexes) {

    scatterUpdateCuda<T><<<512, 256, MAX_NUM_THREADS, *stream>>>(opCode, numOfSubArrs, vx, xShapeInfo, xOffsets, vy, yShapeInfo, yOffsets, indexes);
}


//////////////////////////////////////////////////////////////////////////
void NativeOps::scatterUpdate(Nd4jPointer *extraPointers, int opCode, int numOfSubArrs,
                      			void* hX, Nd4jLong* hXShapeInfo, Nd4jLong* hXOffsets,
                      			void* dX, Nd4jLong* dXShapeInfo, Nd4jLong* dXOffsets,
                      			void* hY, Nd4jLong* hYShapeInfo, Nd4jLong* hYOffsets,
                      			void* dY, Nd4jLong* dYShapeInfo, Nd4jLong* dYOffsets,
                      			int* hIindexes, int* dIndexes) {

	auto stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	nd4j::DataType type = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(type, scatterUpdateCudaLauncher, (stream, opCode, numOfSubArrs, dX, dXShapeInfo, dXOffsets, dY, dYShapeInfo, dYOffsets, dIndexes), LIBND4J_TYPES);
    nd4j::DebugHelper::checkErrorCode(stream, "scatterUpdate(...) failed");
}

void NativeOps::inspectArray(Nd4jPointer *extraPointers, Nd4jPointer buffer, Nd4jLong *shapeInfo, Nd4jPointer specialBuffer, Nd4jLong *specialShapeInfo, Nd4jPointer debugInfo) {
    auto p = reinterpret_cast<nd4j::DebugInfo*>(debugInfo);
    NDArray array(buffer, shapeInfo, nullptr);
    nd4j::DebugHelper::retrieveDebugStatistics(p, &array);
}

void __global__ tryPointerKernel(void* p, int len) {
    auto buf = reinterpret_cast<int8_t*>(p);
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int b;
    if (tid < len)
        atomicAdd(&b, buf[tid]);

    __syncthreads();

    if (threadIdx.x ==0 && blockIdx.x == 0)
        printf("Pointer check complete: %i\n", b);
}

void NativeOps::tryPointer(Nd4jPointer extra, Nd4jPointer p, int len) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    tryPointerKernel<<<256, 512, len+64, stream>>>(p, len);
    auto e = cudaStreamSynchronize(stream);

    if (e != 0)
        throw std::runtime_error("tryPointer failed");

    cudaStreamDestroy(stream);
}
