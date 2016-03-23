#include "../NativeOps.h"
#include <cuda.h>
#include <cuda_launch_config.h>

#include <buffer.h>
#include <shape.h>

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


template <typename T>
dim3 getOptimalDimensions(int n,cudaFuncAttributes attributes, cudaDeviceProp properties) {

    // we can combine the two to compute a block size
    int num_threads = block_size_with_maximum_potential_occupancy(attributes, properties);

    // no real sense launching more threads, then number of elements we have
    if (num_threads > n) num_threads = n;

    // compute the number of blocks of size num_threads to launch
    int num_blocks = n / num_threads;

    // check for partial block at the end
    if(n % num_threads) ++num_blocks;
    return dim3(num_blocks,num_threads, (num_threads * sizeof(T)) + (attributes.sharedSizeBytes < 1024 ? 1024 : attributes.sharedSizeBytes));
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
    int n = shape::length(hostXShapeInfo);

    dim3 launchDims = getOptimalDimensions<T>(n,attributes, properties);

    //printf("Params: gridSize: [%i], blockSize: [%i], shMem: [%i], problemLength: [%i], totalThreads:[%i]\n", launchDims.x, launchDims.y, launchDims.z, n, (launchDims.x * launchDims.y));

    return launchDims;

    //return dim3(5,1024,24500);
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
        scalarDimensionBuff[0] = shape::MAX_DIMENSION;
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

    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>(*stream);
    indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result =  scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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

    indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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
        Nd4jPointer extraParams, int n) {
    double *xPointer = reinterpret_cast<double *>(dx);
    double *yPointer = reinterpret_cast<double *>(y);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[25], deviceProperties[(int) extraPointers[2]]);

    pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>> (
                    opNum,
                    n,
                    xPointer,
                    yPointer,
                    xStride,
                    yStride,
                    extraParamsPointer,
                    resultPointer,
                    resultStride);

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
                    resultIndexesPointer);

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

    pairWiseTransformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    yPointer,
                    extraParamsPointer,
                    resultPointer,
                    xShapeInfoPointer,
                    yShapeInfoPointer,
                    resultShapeInfoPointer);

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



    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>(*stream);

    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarInfo;


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

    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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

    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>(*stream);

    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result =  scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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

    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>(*stream);

    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarInfo;
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

    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>(*stream);

    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result  = scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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

    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
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
                    1);

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
        int n) {
    double *xPointer = reinterpret_cast<double *>(x);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[20], deviceProperties[(int) extraPointers[2]]);

    scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    n,
                    scalar,
                    xPointer,
                    xStride,
                    extraParamsPointer,
                    resultPointer);

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

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[19], deviceProperties[(int) extraPointers[2]]);

    scalarDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    scalar,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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
        int n,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes){
    double *xPointer = reinterpret_cast<double *>(x);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[18], deviceProperties[(int) extraPointers[2]]);

    scalarDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    n,
                    scalar,
                    xPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultIndexesPointer);

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
    ScalarInfo<double> *scalarShapeInformation = new ScalarInfo<double>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[17], deviceProperties[(int) extraPointers[2]]);

    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    scalarShapeInformation->getDevicePointer(),
                    scalarShapeInformation->getDeviceShapeInfo(),
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result = scalarShapeInformation->getFinalResultFromDevice();
    delete scalarShapeInformation;
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
    ScalarInfo<double> *scalarShapeInformation = new ScalarInfo<double>(*stream);

    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarShapeInformation;
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

    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1,biasCorrected);

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
        int n) {
    double *xPointer = reinterpret_cast<double *>(dx);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[16], deviceProperties[(int) extraPointers[2]]);

    transformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    n,
                    xPointer,
                    xStride,
                    extraParamsPointer,
                    resultPointer);

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

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[15], deviceProperties[(int) extraPointers[2]]);

    transformDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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

    transformDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultIndexesPointer);

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
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>(*stream);

    indexReduceFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    float result =  scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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

    indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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
        Nd4jPointer extraParams, int n){
    float *xPointer = reinterpret_cast<float *>(dx);
    float *yPointer = reinterpret_cast<float *>(y);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[11], deviceProperties[(int) extraPointers[2]]);

    pairWiseTransformStridedFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    n,
                    xPointer,
                    yPointer,
                    xStride,
                    yStride,
                    extraParamsPointer,
                    resultPointer,
                    resultStride);

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
                    resultIndexesPointer);

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

    pairWiseTransformFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                    xPointer,
                    yPointer,
                    extraParamsPointer,
                    resultPointer,
                    xShapeInfoPointer,
                    yShapeInfoPointer,
                    resultShapeInfoPointer);

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
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

    reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarInfo;
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

    reduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                   opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[8], deviceProperties[(int) extraPointers[2]]);

    reduceFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result =  scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarInfo;

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
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[7], deviceProperties[(int) extraPointers[2]]);

    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    double result  = scalarInfo->getFinalResultFromDevice();
    delete scalarInfo;
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

    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
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
                    1);

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
        int n){
    float *xPointer = reinterpret_cast<float *>(x);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[6], deviceProperties[(int) extraPointers[2]]);

    scalarFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    n,
                    scalar,
                    xPointer,
                    xStride,
                    extraParamsPointer,
                    resultPointer);

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
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int n = shape::length(hostXShapeInfo);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[5], deviceProperties[(int) extraPointers[2]]);

    scalarFloat<<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    scalar,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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
    int n = shape::length(hostShapeInfo);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[4], deviceProperties[(int) extraPointers[2]]);

    scalarFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    n,
                    scalar,
                    xPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultIndexesPointer);

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

    ScalarInfo<float> *scalarShapeInformation = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    xPointer,
                    xShapeInfoPointer,
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    float result = scalarShapeInformation->getFinalResultFromDevice();
    delete scalarShapeInformation;
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
    ScalarInfo<float> *scalarShapeInformation = new ScalarInfo<float>(*stream);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[3], deviceProperties[(int) extraPointers[2]]);

    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);

    checkCudaErrors(cudaStreamSynchronize(*stream));

    delete scalarShapeInformation;
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

    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1,biasCorrected);

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
        int n) {
    float *xPointer = reinterpret_cast<float *>(dx);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[2], deviceProperties[(int) extraPointers[2]]);

    transformFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
            n,
            xPointer,
            xStride,
            extraParamsPointer,
            resultPointer);

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

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

    dim3 launchDims = getOptimalLaunchParameters<float>(&extraPointers[0], funcAttributes[1], deviceProperties[(int) extraPointers[2]]);

    transformFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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

    transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultIndexesPointer);

    checkCudaErrors(cudaStreamSynchronize(*stream));


}


void NativeOps::initializeDevicesAndFunctions() {
    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);
    deviceProperties = new cudaDeviceProp[devCnt];
    for (int i = 0; i < devCnt; i++) {
        cudaGetDeviceProperties(&deviceProperties[i], i);
    }

    cudaFuncGetAttributes(&funcAttributes[0], (void *)transformFloatIndexes);

    void (*transformFloatPointer1)(int opNum, float *dy,int *shapeInfo, float *params, float *result) = transformFloat;
    cudaFuncGetAttributes(&funcAttributes[1], transformFloatPointer1);

    void (*transformFloatPointer2)(int opNum, int n, float *dy, int incy, float *params, float *result) = transformFloat;
    cudaFuncGetAttributes(&funcAttributes[2], transformFloatPointer2);

    cudaFuncGetAttributes(&funcAttributes[3], (void *)summaryStatsReduceFloat);

    cudaFuncGetAttributes(&funcAttributes[4], (void *)scalarFloatIndexes);

    void (*scalarFloatPointer1)(int opNum, float dx,float *dy, int *shapeInfo,float *params, float *result) = scalarFloat;
    cudaFuncGetAttributes(&funcAttributes[5], scalarFloatPointer1);

    void (*scalarFloatPointer2)(int opNum, int n,float dx, float *dy, int incy, float *params, float *result) = scalarFloat;
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

    void (*transformDoublePointer1)(int opNum, double *dy, int *shapeInfo, double *params, double *result) = transformDouble;
    cudaFuncGetAttributes(&funcAttributes[15], transformDoublePointer1);

    void (*transformDoublePointer2)(int opNum, int n, double *dy, int incy, double *params, double *result) = transformDouble;
    cudaFuncGetAttributes(&funcAttributes[16], transformDoublePointer2);

    cudaFuncGetAttributes(&funcAttributes[17], summaryStatsReduceDouble);

    cudaFuncGetAttributes(&funcAttributes[18], scalarDoubleIndexes);

    void (*scalarDoublePointer1)(int opNum, double dx,double *dy, int *shapeInfo,double *params, double *result) = scalarDouble;
    cudaFuncGetAttributes(&funcAttributes[19], scalarDoublePointer1);


    void (*scalarDoublePointer2)(int opNum, int n,double dx, double *dy, int incy, double *params, double *result) = scalarDouble;
    cudaFuncGetAttributes(&funcAttributes[20], scalarDoublePointer2);

    cudaFuncGetAttributes(&funcAttributes[21], reduce3Double);

    cudaFuncGetAttributes(&funcAttributes[22], reduceDouble);

    cudaFuncGetAttributes(&funcAttributes[23], pairWiseTransformDouble);

    cudaFuncGetAttributes(&funcAttributes[24], pairWiseTransformDoubleIndex);

    cudaFuncGetAttributes(&funcAttributes[25], pairWiseTransformStridedDouble);

    cudaFuncGetAttributes(&funcAttributes[26], broadcastDouble);

    cudaFuncGetAttributes(&funcAttributes[27], indexReduceDouble);
}