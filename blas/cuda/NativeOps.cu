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
bool debug = false;

template <typename T>
dim3 getOptimalDimensions(Nd4jIndex n,cudaFuncAttributes attributes, cudaDeviceProp properties) {

	// we can combine the two to compute a block size
	int num_threads = block_size_with_maximum_potential_occupancy(attributes, properties);

	// no real sense launching more threads, then number of elements we have
	if (num_threads > n) num_threads = n;

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

	return dim3(num_blocks,num_threads, (num_threads * sizeof(T)) + attributes.sharedSizeBytes);
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[27], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z * 4, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
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


	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[27], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z * 2, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[26], deviceProperties[(int) extraPointers[2]]);

	broadcastDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[25], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>> (
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[24], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformDoubleIndex <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer,
			yShapeInfoPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[23], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer,
			yShapeInfoPointer,
			resultShapeInfoPointer, allocationPointer);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[22], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			nullptr,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[21], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[20], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[19], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			scalar,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,resultShapeInfoPointer, allocPointer);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[18], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[16], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	double *specialPointer = reinterpret_cast<double *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	double * special = (double *) maxShapeBuffer + 8;

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			transformDouble<<< 1, launchDims.y, launchDims.z * 3, *stream >> > (
					opNum,
							xPointer,
							xShapeInfoPointer,
							extraParamsPointer,
							resultPointer, resultShapeInfoPointer, allocPointer, reductionPointer);
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

						fillIsMaxDouble<<< 256, 256, 0, *stream >>>(resultPointer, shape::length(hostXShapeInfo), targetIdx);
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
		transformDouble<<<launchDims.x, launchDims.y, launchDims.z * 3, *stream>>> (
				opNum,
						xPointer,
						xShapeInfoPointer,
						extraParamsPointer,
						resultPointer, resultShapeInfoPointer, allocPointer, reductionPointer);
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[14], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

	transformDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[(int) extraPointers[2]]);

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	indexReduceFloat<<<launchDims.x,launchDims.y, launchDims.z * 2, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[13], deviceProperties[(int) extraPointers[2]]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z * 2, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[12], deviceProperties[(int) extraPointers[2]]);

	broadcastFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			yPointer,
			yShapeInfoPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[11], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformStridedFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[10], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformFloatIndex<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer,
			yShapeInfoPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[9], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	pairWiseTransformFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			yPointer,
			extraParamsPointer,
			resultPointer,
			xShapeInfoPointer,
			yShapeInfoPointer,
			resultShapeInfoPointer, allocationPointer);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	reduceFloat<<< launchDims.x,launchDims.y, launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer
			,extraParamsPointer,
			resultPointer,
			nullptr,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[6], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	scalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

	scalarFloat<<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(
			opNum,
			scalar,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,resultShapeInfoPointer, allocPointer );

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[4], deviceProperties[(int) extraPointers[2]]);
	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);
	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			nullptr,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

	int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z * 10, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
			extraParamsPointer,
			resultPointer,
			resultShapeInfoPointer,
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[2], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	// special pointer for special buffer for special ops
	float *specialPointer = reinterpret_cast<float *>(extraPointers[6]);

	int *dimension = (int *) specialPointer;
	int *maxDimension = dimension + 1;
	int *maxShapeBuffer = (int *) maxDimension + 1;
	float * special = (float *) maxShapeBuffer + 8;

	// simple trick to get workaround over reductions into scalar
	if (opNum >= 38 && opNum <= 41) {
		if (shape::isVector(hostXShapeInfo) && opNum != 41) {
			// if that's vector, we just go directly to op in 1 block
			transformFloat <<< 1, launchDims.y, launchDims.z * 3, *stream >> > (
					opNum,
					xPointer,
					xShapeInfoPointer,
					extraParamsPointer,
					resultPointer, resultShapeInfoPointer, allocPointer, reductionPointer);
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
						execIndexReduceFloat(extraPointers,0, dx, xShapeInfo, extraParams, result, resultShapeInfo, (Nd4jPointer) dimension, 1);
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
		transformFloat <<<launchDims.x, launchDims.y, launchDims.z * 3, *stream>>> (
				opNum,
				xPointer,
				xShapeInfoPointer,
				extraParamsPointer,
				resultPointer, resultShapeInfoPointer, allocPointer, reductionPointer);
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[0], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);
	float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

	transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			opNum,
			xPointer,
			xShapeInfoPointer,
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
			long allocSize = sizeof(int) * rank;
			int *coord = shape::cuMalloc(allocationPointer, allocSize);

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

			if (tid * allocSize > PREALLOC_SIZE - allocSize) {
				free(coord);
			}
		}
	} else {
		int rank = shape::rank(inputShapeInfo);
		long allocSize = sizeof(int) * rank;
		int *coord = shape::cuMalloc(allocationPointer, allocSize);
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
		if (tid * allocSize > PREALLOC_SIZE - allocSize) {
			free(coord);
		}
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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);

	int *allocPointer = reinterpret_cast<int *>(extraPointers[3]);

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

	void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, float *params, float *result,int *resultShapeInfo, int *allocationPointer, float *reductionPointer) = transformFloat;
	cudaFuncGetAttributes(&funcAttributes[1], transformFloatPointer1);

	void (*transformFloatPointer2)(int opNum, Nd4jIndex n, float *dy, int incy, float *params, float *result,int resultStride, int *allocationPointer, float *reductionPointer) = transformFloat;
	cudaFuncGetAttributes(&funcAttributes[2], transformFloatPointer2);

	cudaFuncGetAttributes(&funcAttributes[3], (void *)summaryStatsReduceFloat);

	cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

	void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo,float *params, float *result,int *resultShapeInfo, int *allocPointer) = scalarFloat;
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

	void (*transformDoublePointer1)(int opNum, double *dy, int *shapeInfo, double *params, double *result,int *resultShapeInfo, int *allocationPointer, double *reductionPointer) = transformDouble;
	cudaFuncGetAttributes(&funcAttributes[15], transformDoublePointer1);

	void (*transformDoublePointer2)(int opNum, Nd4jIndex n, double *dy, int incy, double *params, double *result,int resultStride, int *allocationPointer, double *reductionPointer) = transformDouble;
	cudaFuncGetAttributes(&funcAttributes[16], transformDoublePointer2);

	cudaFuncGetAttributes(&funcAttributes[17], summaryStatsReduceDouble);

	cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

	void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo,double *params, double *result,int *resultShapeInfo, int *allocPointer) = scalarDouble;
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


int NativeOps::ompGetNumThreads() {
	return blockLimit;
}

void NativeOps::setOmpNumThreads(int threads) {
	printf("Setting max grid size to [%i]\n", threads);
	//blockLimit = threads;
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
