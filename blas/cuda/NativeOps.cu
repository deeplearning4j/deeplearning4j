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

template <typename T>
dim3 getOptimalDimensions(int n,cudaFuncAttributes attributes) {
    // next, get the cudaDeviceProp object corresponding to the current device
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // we can combine the two to compute a block size
    int num_threads = block_size_with_maximum_potential_occupancy(attributes, properties);

    // compute the number of blocks of size num_threads to launch
    int num_blocks = n / num_threads;

    // check for partial block at the end
    if(n % num_threads) ++num_blocks;

    return dim3(num_blocks,num_threads,num_threads * sizeof(T));
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
dim3 getOptimalLaunchParameters(Nd4jPointer *extraPointers) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, indexReduceDouble);
    int *hostXShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int n = shape::length(hostXShapeInfo);
    dim3 launchDims = getOptimalDimensions<T>(n,attributes);
    return launchDims;
}

nd4j::buffer::Buffer<int> * createScalarBuffer() {
    int *scalarShapeInfo = shape::createScalarShapeInfo();
    nd4j::buffer::Buffer<int> *buff = nd4j::buffer::createBuffer(scalarShapeInfo,shape::shapeInfoLength(2));
    nd4j::buffer::copyDataToGpu(&buff);
    return buff;
}


class ScalarShapeInformation {
private:
    nd4j::buffer::Buffer<int> *scalarDimension;
    nd4j::buffer::Buffer<int> *scalarShapeInfo;
    std::thread::id threadId;

public:
    ScalarShapeInformation() {
        int *scalarDimensionBuff = (int *) malloc(sizeof(int));
        scalarDimensionBuff[0] = shape::MAX_DIMENSION;
        scalarDimension = nd4j::buffer::createBuffer(scalarDimensionBuff,1);
        scalarShapeInfo = createScalarBuffer();
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
#ifdef R__WIN32
    static thread_local ScalarShapeInformation shapeInfo;
#else
    static  ScalarShapeInformation shapeInfo;

#endif
    T finalResult;
public:
    ScalarInfo() {
        T *scalarResult = (T*)malloc(sizeof(T));
        scalarData = nd4j::buffer::createBuffer(scalarResult,1);
        nd4j::buffer::copyDataToGpu(&scalarData);
    }

    T getFinalResultFromDevice() {
        nd4j::buffer::copyDataFromGpu(&scalarData);
        return scalarData->data[0];
    }

    /**
     * Get the device shape information
     * representing a scalar
     */
    int *getDeviceShapeInfo() {
        return shapeInfo.getShapeInfoGpuPointer();
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
        return shapeInfo.getDimensionGpuPointer();
    }

    ~ScalarInfo() {
        nd4j::buffer::freeBuffer(&scalarData);
    }
};

ScalarShapeInformation ScalarInfo<double>::shapeInfo;
ScalarShapeInformation ScalarInfo<float>::shapeInfo;

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>();

    indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);
    cudaDeviceSynchronize();

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);

    indexReduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);


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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);

    broadcastDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength);

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y,launchDims.z>>>
                                                               (
                                                                       opNum,
                                                                               n,
                                                                               xPointer,
                                                                               yPointer,
                                                                               xStride,
                                                                               yStride,
                                                                               extraParamsPointer,
                                                                               resultPointer,
                                                                               resultStride);
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    pairWiseTransformDoubleIndex <<<launchDims.x, launchDims.y, launchDims.z>>>(
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    pairWiseTransformDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    yPointer,
                    extraParamsPointer,
                    resultPointer,
                    xShapeInfoPointer,
                    yShapeInfoPointer,
                    resultShapeInfoPointer);


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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>();

    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>();
    reduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>();
    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarInfo = new ScalarInfo<double>();
    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    reduce3Double<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    scalarDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    n,
                    scalar,
                    xPointer,
                    xStride,
                    extraParamsPointer,
                    resultPointer);
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    scalarDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    scalar,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    scalarDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum, n,scalar,xPointer,extraParamsPointer,resultPointer,resultIndexesPointer);


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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarShapeInformation = new ScalarInfo<double>();
    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            scalarShapeInformation->getDevicePointer(),
            scalarShapeInformation->getDeviceShapeInfo(),
            scalarShapeInformation->getDimensionDevicePointer(),
            1,
            1,biasCorrected);
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    ScalarInfo<double> *scalarShapeInformation = new ScalarInfo<double>();
    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1,biasCorrected);

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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    transformDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,n,xPointer,xStride,extraParamsPointer,resultPointer);
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    transformDouble<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer);
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
    dim3 launchDims = getOptimalLaunchParameters<double>(extraPointers);
    transformDoubleIndexes<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultIndexesPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>();

    indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);
    cudaDeviceSynchronize();

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    indexReduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);

    broadcastFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    yPointer,
                    yShapeInfoPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength);
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    pairWiseTransformStridedFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,n,xPointer,yPointer,xStride,yStride,extraParamsPointer,resultPointer,resultStride);
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    pairWiseTransformFloatIndex<<<launchDims.x,launchDims.y,launchDims.z>>>(
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    pairWiseTransformFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    yPointer,
                    extraParamsPointer,
                    resultPointer,
                    xShapeInfoPointer,
                    yShapeInfoPointer,
                    resultShapeInfoPointer);
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>();
    reduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    reduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1);


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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>();
    reduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer
                    ,extraParamsPointer,
                    scalarInfo->getDevicePointer(),
                    scalarInfo->getDeviceShapeInfo(),
                    scalarInfo->getDimensionDevicePointer(),
                    1,
                    1);
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>();
    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
    ScalarInfo<float> *scalarInfo = new ScalarInfo<float>();
    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    reduce3Float<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    scalarFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    n,
                    scalar,
                    xPointer,
                    xStride,
                    extraParamsPointer,
                    resultPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    scalarFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    scalar,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    int *hostShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int n = shape::length(hostShapeInfo);
    scalarFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    n,
                    scalar,
                    xPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultIndexesPointer);

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
    ScalarInfo<float> *scalarShapeInformation = new ScalarInfo<float>();
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            xPointer,
            xShapeInfoPointer,
            scalarShapeInformation->getDimensionDevicePointer(),
            1,
            1,biasCorrected);
    cudaDeviceSynchronize();
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    ScalarInfo<float> *scalarShapeInformation = new ScalarInfo<float>();
    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    scalarShapeInformation->getDimensionDevicePointer(),
                    1,
                    1,biasCorrected);
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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(
            opNum,
                    xPointer,
                    xShapeInfoPointer,
                    extraParamsPointer,
                    resultPointer,
                    resultShapeInfoPointer,
                    dimensionPointer,
                    dimensionLength,
                    1,biasCorrected);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    transformFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,n,xPointer,xStride,extraParamsPointer,resultPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    transformFloat<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer);

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
    dim3 launchDims = getOptimalLaunchParameters<float>(extraPointers);
    transformFloatIndexes<<<launchDims.x,launchDims.y,launchDims.z>>>(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultIndexesPointer);


}
