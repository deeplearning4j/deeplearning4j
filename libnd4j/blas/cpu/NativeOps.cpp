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

//
// Created by agibsonccc on 2/21/16.
//

#define __STDC_CONSTANT_MACROS

#include "../NativeOps.h"
#include "../NativeOpExcutioner.h"
#include "../NDArray.h"
#include "../GraphExecutioner.h"
#include <graph/GraphHolder.h>
#include <templatemath.h>
#include <types/float8.h>
#include <loops/type_conversions.h>
#include <loops/aggregates.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <pointercast.h>
#include <pairwise_util.h>
#include <types/types.h>


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#else
#include <io.h>
#include <helpers/mman.h>
#endif
#include <sys/types.h>

#include <ops/declarable/CustomOperations.h>
#include <errno.h>


char *name;
bool nameSet = false;


#ifdef __EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif

#include <ops/specials.h>
#include "../Environment.h"
#include <TAD.h>
#include <ops/declarable/OpRegistrator.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>

using namespace nd4j;

void NativeOps::setElementThreshold(int num) {
    if (num > 0)
        nd4j::Environment::getInstance()->setElementwiseThreshold(num);
}

void NativeOps::setTADThreshold(int num) {
    if (num > 0)
        nd4j::Environment::getInstance()->setTadThreshold(num);
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
Nd4jLong NativeOps::execIndexReduceScalar(Nd4jPointer *extraPointers,
                                                int opNum,
                                                void *x,
                                                Nd4jLong *xShapeInfo,
                                                void *extraParams) {
    return NativeOpExcutioner::execIndexReduceScalar(opNum, x, xShapeInfo, extraParams);

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
void  NativeOps::execIndexReduce(Nd4jPointer *extraPointers,int opNum,
                                        void *x,
                                        Nd4jLong *xShapeInfo,
                                        void *extraParams,
                                        Nd4jLong *result,
                                        Nd4jLong *resultShapeInfo,
                                        int *dimension,
                                        int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExcutioner::execIndexReduce(opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            tadShapeInfo,
            tadOffsets);
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
void NativeOps::execBroadcast(Nd4jPointer *extraPointers,
                                      int opNum,
                                      void *x,
                                      Nd4jLong *xShapeInfo,
                                      void *y,
                                      Nd4jLong *yShapeInfo,
                                      void *result,
                                      Nd4jLong *resultShape,
                                      int *dimension, int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);
    NativeOpExcutioner::execBroadcast(
            opNum,
            x,
            xShapeInfo,
            y,
            yShapeInfo,
            result, resultShape,
            dimension,
            dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
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
void NativeOps::execPairwiseTransform(
        Nd4jPointer *extraPointers,
        int opNum,
        void *dx,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        void *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *yIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner::execPairwiseTransform(
            opNum,
            dx,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            xIndexes,
            yIndexes,
            resultIndexes);
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
void NativeOps::execPairwiseTransform(
        Nd4jPointer *extraPointers,
        int opNum,
        void *dx,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        void *extraParams) {
    NativeOpExcutioner::execPairwiseTransform(
            opNum,
            dx,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            extraParams);
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
void NativeOps::execReduce(
        Nd4jPointer *extraPointers,
        int opNum,
        void *x,
        Nd4jLong *xShapeInfo,
        void *extraParams,
        void *result,
        Nd4jLong *resultShapeInfo) {
    NativeOpExcutioner::execReduceScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo);

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
void NativeOps::execReduce(Nd4jPointer *extraPointers,
                                   int opNum,
                                   void *x,
                                   Nd4jLong *xShapeInfo,
                                   void *extraParams,
                                   void *result,
                                   Nd4jLong *resultShapeInfo,
                                   int *dimension,
                                   int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    NativeOpExcutioner::execReduce(opNum,
                                           x,
                                           xShapeInfo,
                                           extraParams,
                                           result,
                                           resultShapeInfo,
                                           dimension,
                                           dimensionLength,
                                           tadShapeInfo,
                                           tadOffsets);
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
void NativeOps::execReduce3(Nd4jPointer *extraPointers,
                                    int opNum,
                                    void *x,
                                    Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    void *y,
                                    Nd4jLong *yShapeInfo,
                                    void *result,
                                    Nd4jLong *resultShapeInfo) {
    NativeOpExcutioner::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo);
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
void NativeOps::execReduce3Scalar(Nd4jPointer *extraPointers,int opNum,
                                            void *x,
                                            Nd4jLong *xShapeInfo,
                                            void *extraParams,
                                            void *y,
                                            Nd4jLong *yShapeInfo,
                                            void *z,
                                            Nd4jLong *zShapeInfo) {
    NativeOpExcutioner::execReduce3Scalar(opNum,x,xShapeInfo,extraParams,y,yShapeInfo, z, zShapeInfo);
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
void NativeOps::execReduce3(Nd4jPointer *extraPointers,
                                    int opNum,
                                    void *x,
                                    Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    void *y,
                                    Nd4jLong *yShapeInfo,
                                    void *result,
                                    Nd4jLong *resultShapeInfo,
                                    int *dimension,
                                    int dimensionLength) {

    if (extraPointers == nullptr || extraPointers[2] == 0) {
        NativeOpExcutioner::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength);
    } else {
        // going tad-way
        auto tadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[0]);
        auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

        NativeOpExcutioner::execReduce3TAD(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
    }

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
void NativeOps::execScalar(
        Nd4jPointer *extraPointers,
        int opNum,
        void *x,
        Nd4jLong *xShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        void *scalar,
        Nd4jLong *scalarShapeInfo,
        void *extraParams) {
    NativeOpExcutioner::execScalar(
            opNum,
            x,
            xShapeInfo,
            result,
            resultShapeInfo,
            scalar,
            scalarShapeInfo,
            extraParams);
}


/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double NativeOps::execSummaryStatsScalar(Nd4jPointer *extraPointers,
        int opNum,
        void *x,
        Nd4jLong *xShapeInfo,
        void *extraParams,
        bool biasCorrected) {
    return NativeOpExcutioner::execSummaryStatsScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            biasCorrected);
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
void NativeOps::execSummaryStats(Nd4jPointer *extraPointers,
                                         int opNum,
                                         void *x,
                                         Nd4jLong *xShapeInfo,
                                         void *extraParams,
                                         void *result,
                                         Nd4jLong *resultShapeInfo,
                                         bool biasCorrected) {
    NativeOpExcutioner::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            biasCorrected);
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
void NativeOps::execSummaryStats(Nd4jPointer *extraPointers,
                                         int opNum,
                                         void *x,
                                         Nd4jLong *xShapeInfo,
                                         void *extraParams,
                                         void *result,
                                         Nd4jLong *resultShapeInfo,
                                         int *dimension,
                                         int dimensionLength,
                                         bool biasCorrected) {
    NativeOpExcutioner::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            biasCorrected);

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
void NativeOps::execTransform(
        Nd4jPointer *extraPointers,
        int opNum,
        void *dx,
        Nd4jLong *xShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        void *extraParams) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExcutioner::execTransform(
            opNum,
            dx,
            xShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            tadShapeInfo,
            tadOffsets);
}

void NativeOps::execReduce3All(Nd4jPointer *extraPointers,
                                     int opNum,
                                     void *x,
                                     Nd4jLong *xInfo,
                                     void *extraParamsVals,
                                     void *y,
                                     Nd4jLong *yInfo,
                                     void *result,
                                     Nd4jLong *resultShapeInfoBuffer,
                                     int *dimension,
                                     int dimensionLength,
                                     Nd4jLong *xTadShapeInfo,
                                     Nd4jLong *xOffsets,
                                     Nd4jLong *yTadShapeInfo,
                                     Nd4jLong *yOffsets) {

    NativeOpExcutioner::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}


template <typename T>
void flattenGeneric(Nd4jPointer *extraPointers,
                    int offset,
                    char order,
                    void *vresult,
                    Nd4jLong *resultShapeInfo,
                    void *vinput,
                    Nd4jLong *inputShapeInfo) {

    auto result = reinterpret_cast<T *>(vresult);
    auto input = reinterpret_cast<T *>(vinput);

    int numOnes = 0;
    auto shape = shape::shapeOf(inputShapeInfo);
    int wholeRank = shape::rank(inputShapeInfo);
    for(int i = 0; i < wholeRank; i++) {
        if(shape[i] == 1)
            numOnes++;
    }



    //start at the given offset
    result += offset;
    char inputOrder = shape::order(inputShapeInfo);
    auto len = shape::length(inputShapeInfo);
    auto resultEleStride = shape::elementWiseStride(resultShapeInfo);
    auto inputEleStride = shape::elementWiseStride(inputShapeInfo);
    Nd4jLong numTads, stride;
    int dimension, dimensionLength;
    int rank = shape::rank(inputShapeInfo);
    auto xStride = shape::stride(inputShapeInfo);
    auto xShape = shape::shapeOf(inputShapeInfo);

    dimensionLength = 1;
    if(order == 'f') {
        dimension = 0;
    }
    else {
        dimension = rank - 1;
    }
    stride  = xStride[dimension];
    // numTads is product of length of all dimensions excluding
    // the one we do the tad on
    numTads = 1;
    for (int i = 0; i < rank; i++) {
        if (i != dimension)
            numTads *= xShape[i];
    }

    if (inputOrder == order) {
        if (resultEleStride == 1 && inputEleStride == 1) {
            memcpy(result, input, len* sizeof(T));
        }
        else if (resultEleStride >= 1 && inputEleStride >= 1) {
            if (len < ELEMENT_THRESHOLD) {
#pragma omp simd
                for (int i = 0; i < len; i++) {
                    result[i * resultEleStride] = input[i * inputEleStride];
                }
            }
            else {
#pragma omp parallel for simd
                for (int i = 0; i < len; i++) {
                    result[i * resultEleStride] = input[i * inputEleStride];
                }
            }
        }
        else {
            int idx = 0;
            Nd4jLong coord[MAX_RANK];

            // FIXME: result[idx++] is bad idea, because of possible negative EWS
            if(order == 'f') {
                for(int i = 0; i < len; i++) {
                    shape::ind2sub(rank, xShape, i, coord);
                    auto offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }
            else {
                for(int i = 0; i < len; i++) {
                    shape::ind2subC(rank, xShape, i, coord);
                    auto offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }
        }
    }
    else {
        int rank = shape::rank(inputShapeInfo);
        auto xShape = shape::shapeOf(inputShapeInfo);
        auto tadShape = xShape[dimension];
        shape::TAD tad(inputShapeInfo,&dimension,dimensionLength);
        tad.createTadOnlyShapeInfo();
#pragma omp  parallel for schedule(guided) default(shared)
        for(int i = 0; i < numTads; i++) {

            Nd4jLong resultOffset;

            if (order == 'f') {
                // 1. get c ordering coordinates
                auto cIndexCoordinates = new Nd4jLong[rank - 1];
                int divisor = 1;
                for (int dim = rank - 1; dim > 0; dim--) {
                    cIndexCoordinates[dim - 1] = (i / divisor) % xShape[dim];
                    divisor *= xShape[dim];
                }


                // 2. convert to f ordering index
                int fIndex = 0;
                int multiplier = 1;
                for (int dim = 1; dim <= rank - 1; dim++) {
                    fIndex += cIndexCoordinates[dim - 1] * multiplier;
                    multiplier *= xShape[dim];
                }

                resultOffset = fIndex * tadShape;
                delete[] cIndexCoordinates;

            }
            else {
                resultOffset = i *  tadShape;
            }

            auto tadOffset = tad.tadOffset(i);
            for( int j = 0; j < tadShape; j++) {

                // TAD are returned in C ordering always
                result[resultOffset + j] = input[tadOffset + j * stride];

            }
        }
    }
}


/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::concat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        Nd4jPointer *tadPointers,
        Nd4jPointer *offsetPointers) {
    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, nd4j::SpecialMethods, ::concatCpuGeneric(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo), LIBND4J_TYPES);
}

/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::specialConcat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        Nd4jPointer *tadPointers,
        Nd4jPointer *offsetPointers) {

    auto zType = nd4j::ArrayOptions::dataType(resultShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, nd4j::SpecialMethods, ::concatCpuGeneric(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo), LIBND4J_TYPES);
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
void NativeOps::flatten(
        Nd4jPointer *extraPointers,
        int offset,
        char order,
        void *result,
        Nd4jLong *resultShapeInfo,
        void *input,
        Nd4jLong *inputShapeInfo) {
    auto xType = nd4j::ArrayOptions::dataType(inputShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, flattenGeneric, (extraPointers, offset, order, result, resultShapeInfo, input, inputShapeInfo), LIBND4J_TYPES);
}

/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void NativeOps::initializeDevicesAndFunctions() {

}

void NativeOps::initializeFunctions(Nd4jPointer *functions) {
    nd4j::BlasHelper::getInstance()->initializeFunctions(functions);
}

/**
       * This method acquires memory chunk of requested size on host side
       *
       * @param pointer pointer that'll be used for allocation
       * @param memorySize memory size, in bytes
       * @param flags optional parameter
       */
Nd4jPointer NativeOps::mallocHost(Nd4jLong memorySize, int flags) {
    Nd4jPointer pointer = (Nd4jPointer) malloc(memorySize);
    if (pointer == 0)
        return 0L;
    return pointer;
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocDevice(Nd4jLong memorySize, Nd4jPointer ptrToDeviceId, int flags) {
    // not supported
    return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int NativeOps::freeHost(Nd4jPointer pointer) {
    free(reinterpret_cast<void *>(pointer));
    return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int NativeOps::freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId) {
    // not supported
    return 0L;
}


/**
 * Returns the maximum number open mp threads
 */
int NativeOps::ompGetMaxThreads() {
    return omp_get_max_threads();
}

/**
 * Returns the number open mp threads
 */
int NativeOps::ompGetNumThreads() {
    return omp_get_num_threads();
}

/**
 * Sets the number of openmp threads
 */
void NativeOps::setOmpNumThreads(int threads) {
    omp_set_num_threads(threads);

}

Nd4jPointer NativeOps::createContext() {
    return 0L;
}

Nd4jPointer NativeOps::createStream() {
    return 0L;
}

Nd4jPointer NativeOps::createEvent() {
    return 0L;
}

int NativeOps::getDeviceMajor(Nd4jPointer ptrToDeviceId) {
    return 0;
}

int NativeOps::getDeviceMinor(Nd4jPointer ptrToDeviceId) {
    return 0;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
    return 0L;
}

int NativeOps::setDevice(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

Nd4jLong NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

Nd4jLong NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jLong size,  int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::destroyEvent(Nd4jPointer event) {
    return 0L;
}

int NativeOps::streamSynchronize(Nd4jPointer stream) {
    return 0L;
}

int NativeOps::eventSynchronize(Nd4jPointer event) {
    return 0L;
}

int NativeOps::getAvailableDevices() {
    return 0L;
}

void NativeOps::enableDebugMode(bool reallyEnable) {
    nd4j::Environment::getInstance()->setDebug(reallyEnable);
}

void NativeOps::enableVerboseMode(bool reallyEnable) {
    nd4j::Environment::getInstance()->setVerbose(reallyEnable);
}

void NativeOps::setGridLimit(int gridSize) {
    // no-op
}

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
    // no-op
    return 0L;
}

Nd4jPointer NativeOps::getConstantSpace() {
    // no-op
    return 0L;
}

template<typename T>
void pullRowsGeneric(void *vx,
                     Nd4jLong *xShapeInfo,
                     void *vz,
                     Nd4jLong *zShapeInfo,
                     const int n,
                     Nd4jLong *indexes,
                     Nd4jLong *tadShapeInfo,
                     Nd4jLong *tadOffsets,
                     Nd4jLong *zTadShapeInfo,
                     Nd4jLong *zTadOffsets) {
    auto x = reinterpret_cast<T *>(vx);
    auto z = reinterpret_cast<T *>(vz);

    const auto xEWS = shape::elementWiseStride(tadShapeInfo);
    const auto zEWS = shape::elementWiseStride(zTadShapeInfo);
    const auto tadLength = shape::length(tadShapeInfo);

    int elementsPerThread = n / TAD_THRESHOLD;
    int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
    _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(_threads) if (n > 1) schedule(guided) default(shared)
    for (int idx = 0; idx < n; idx++) {
        auto xTadOffsetForBlock = tadOffsets[indexes[idx]];
        auto zTadOffsetForBlock = zTadOffsets[idx];

        auto rX = x + xTadOffsetForBlock;
        auto rZ = z + zTadOffsetForBlock;

        if (xEWS == 1 && zEWS == 1) {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i] = rX[i];
            }
        } else if (xEWS >= 1 && zEWS >= 1) {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        } else {
            auto zShape = shape::shapeOf(zTadShapeInfo);
            auto zStride = shape::stride(zTadShapeInfo);
            auto xShape = shape::shapeOf(tadShapeInfo);
            auto xStride = shape::stride(tadShapeInfo);
            auto zRank = shape::rank(zTadShapeInfo);
            auto tadRank = shape::rank(tadShapeInfo);

            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            for (int i = 0; i < tadLength; i++) {
                shape::ind2subC(tadRank,xShape, i, xCoord);
                shape::ind2subC(zRank,zShape, i, zCoord);

                auto xOffset = shape::getOffset(xTadOffsetForBlock, xShape, xStride, xCoord, tadRank);
                auto zOffset = shape::getOffset(zTadOffsetForBlock, zShape, zStride, zCoord, zRank);
                z[zOffset] = x[xOffset];
            }
        }
    }
}

void NativeOps::pullRows(Nd4jPointer *extraPointers,
        void *x,
        Nd4jLong *xShapeInfo,
        void *z,
        Nd4jLong *zShapeInfo,
        Nd4jLong n,
        Nd4jLong *indexes,
        Nd4jLong *tadShapeInfo,
        Nd4jLong *tadOffsets,
        Nd4jLong *zTadShapeInfo,
        Nd4jLong *zTadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric, (x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets), LIBND4J_TYPES);
}

template<typename T>
void tearGeneric(void *vx,
        Nd4jLong *xShapeInfo,
        Nd4jPointer *targets,
        Nd4jLong *zShapeInfo,
        Nd4jLong *tadShapeInfo,
        Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<T *>(vx);

    const auto tadLength = shape::length(tadShapeInfo);
    auto tadEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(zShapeInfo);
    auto tadRank = shape::rank(tadShapeInfo);
    auto zRank = shape::rank(zShapeInfo);
    auto tadShape = shape::shapeOf(tadShapeInfo);
    auto tadStride = shape::stride(tadShapeInfo);
    auto zShape = shape::shapeOf(zShapeInfo);
    auto zStride = shape::stride(zShapeInfo);
    auto numTads = shape::length(xShapeInfo) / tadLength;

#pragma omp parallel for schedule(guided) default(shared)
    for (Nd4jLong i = 0; i < numTads; i++) {
        auto z = reinterpret_cast<T *>(targets[i]);
        auto s = x + tadOffsets[i];

        if (zEWS == 1 && tadEWS == 1) {
#pragma omp simd
            for (Nd4jLong j = 0; j < tadLength; j++) {
                z[j] = s[j];
            }
        } else if (zEWS > 0 && tadEWS > 0) {
#pragma omp simd
            for (Nd4jLong j = 0; j < tadLength; j++) {
                z[j * zEWS] = s[j * tadEWS];
            }
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            for (Nd4jLong j = 0; j < tadLength; j++) {
                shape::ind2sub(tadRank,tadShape, j, xCoord);
                shape::ind2sub(zRank, zShape, j, zCoord);

                auto xOffset = shape::getOffset(0, tadShape, tadStride, xCoord, tadRank);
                auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                z[zOffset] = s[xOffset];
            }
        }
    }
}

void NativeOps::tear(Nd4jPointer *extraPointers,
        void *x,
        Nd4jLong *xShapeInfo,
        Nd4jPointer *targets,
        Nd4jLong *zShapeInfo,
        Nd4jLong *tadShapeInfo,
        Nd4jLong *tadOffsets) {
    auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric, (x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
}


void NativeOps::average(Nd4jPointer *extras,
        Nd4jPointer *dx,
        void *dz,
        Nd4jLong *zShapeInfo,
        int n,
        Nd4jLong length,
        bool propagate) {
    auto xType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::averageGeneric(dx, dz, zShapeInfo, n, length, propagate), LIBND4J_TYPES);
}

void NativeOps::accumulate(Nd4jPointer *extras,
        Nd4jPointer *dx,
        void *dz,
        Nd4jLong *zShapeInfo,
        int n,
        Nd4jLong length) {

    auto xType = nd4j::ArrayOptions::dataType(zShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::accumulateGeneric(dx, dz, zShapeInfo, n, length), LIBND4J_TYPES);
}

void NativeOps::enableP2P(bool enable) {
    // no-op
}

void NativeOps::encodeThresholdP1Half(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Float(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Double(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}


void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jLong N, int *dz) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz){
    // TODO: to be implemented
}

bool NativeOps::isP2PAvailable() {
    // always TRUE for cpu backend
    return true;
}

void NativeOps::checkP2P() {
    // no-op
}

template<typename T>
void shuffleGeneric(void **dx, Nd4jLong **xShapeInfo, void **dz, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {

    auto dX = reinterpret_cast<T **>(dx);
    auto dZ = reinterpret_cast<T **>(dz);

#pragma omp parallel for if (N > 1) default(shared)
    for (int f = 0; f < N; f++) {
        auto x = reinterpret_cast<T *>(dX[f]);
        //auto z = reinterpret_cast<T *>(dZ[f]);

        auto tadOffset = reinterpret_cast<Nd4jLong *>(tadOffsets[f]);


        const auto tadLength = shape::length(tadOnlyShapeInfo[f]);
        auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
        auto tadRank = shape::rank(tadOnlyShapeInfo[f]);
        auto numTads = shape::length(xShapeInfo[f]) / tadLength;

        auto tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
        auto tadStride = shape::stride(tadOnlyShapeInfo[f]);

        // TODO: omp *probably* has no sense here, since 99% of uses for this method will be inside DataSet. but worth a check

        for (Nd4jLong r = 0; r < numTads; r++) {
            if (shuffleMap[r] < 0)
                continue;

            auto oldOffset = tadOffset[r];
            auto newOffset = tadOffset[shuffleMap[r]];

            auto rX = x + oldOffset;
            auto rY = x + newOffset;

            if (tadEWS == 1) {

#pragma omp simd
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    nd4j::math::nd4j_swap<T>(rX[i], rY[i]);
                }

            } else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                // ind2sub branch
#pragma omp parallel for schedule(guided) if (N == 1 && tadLength > 512) private(xCoord, yCoord)
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    shape::ind2subC(tadRank,tadShape, i, xCoord);
                    shape::ind2subC(tadRank,tadShape, i, yCoord);

                    auto xOffset = shape::getOffset(oldOffset, tadShape, tadStride, xCoord, tadRank);
                    auto yOffset = shape::getOffset(newOffset, tadShape, tadStride, yCoord, tadRank);

                    nd4j::math::nd4j_swap<T>(x[xOffset], x[yOffset]);
                }

            }

        }

    }
}

void NativeOps::shuffle(Nd4jPointer *extras,
                              Nd4jPointer *dx,
                              Nd4jPointer *xShapeInfo,
                              Nd4jPointer *dz,
                              Nd4jPointer *zShapeInfo,
                              int N,
                              int *shuffleMap,
                              Nd4jPointer *tadShapeInfo,
                              Nd4jPointer *tadOffsets) {
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    auto xType = nd4j::ArrayOptions::dataType(xShape[0]);

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (dx, xShape, dz, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset), LIBND4J_TYPES);
}

void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras,
                                             const int opTypeA,
                                             const int opNumA,
                                             const int opTypeB,
                                             const int opNumB,
                                             float *dx,
                                             Nd4jLong *xShapeInfo,
                                             float *dy,
                                             Nd4jLong *yShapeInfo,
                                             float *dz,
                                             Nd4jLong *zShapeInfo,
                                             int *dimension,
                                             int dimensionLength,
                                             Nd4jLong *tadShapeInfo,
                                             Nd4jLong *tadOffsets,
                                             float *extraA,
                                             float *extraB,
                                             float scalarA,
                                             float scalarB,
                                             bool scalarReturned) {
    // no-op
}

bool NativeOps::isExperimentalEnabled() {
    return experimentalSupport;
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::setOmpMinThreads(int threads) {
    // TODO: to be implemented
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong xStride, float *dy, Nd4jLong yStride, float *dz, Nd4jLong zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong *xShapeInfo, double *dy, Nd4jLong *yShapeInfo, double *dz, Nd4jLong *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong xStride, double *dy, Nd4jLong yStride, double *dz, Nd4jLong zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong *xShapeInfo, float16 *dy, Nd4jLong *yShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong xStride, float16 *dy, Nd4jLong yStride, float16 *dz, Nd4jLong zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op
}

int NativeOps::getDevice() {
    return 0;
}

void NativeOps::execScalar(Nd4jPointer *extraPointers,
                                 int opNum,
                                 void *x,
                                 Nd4jLong *xShapeInfo,
                                 void *z,
                                 Nd4jLong *zShapeInfo,
                                 void *scalars,
                                 Nd4jLong *scalarShapeInfo,
                                 void *extraParams,
                                 int *dimension,
                                 int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);

    NativeOpExcutioner::execScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            scalars,
            scalarShapeInfo,
            dimension,
            dimensionLength,
            tadShapeInfo,
            tadOffsets,
            tadShapeInfoZ,
            tadOffsetsZ);
}

const char * NativeOps::getDeviceName(Nd4jPointer ptrToDeviceId) {
    if (!nameSet) {
        name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

        CHECK_ALLOC(name, "Failed to allocate new string buffer");

        std::memset(name, 0, 256 * sizeof(char));
        nameSet = true;

        // TODO: provide proper CPU model name here
        sprintf(name, "x86-compatible CPU");
    }


    return name;
}


void NativeOps::execAggregateFloat(Nd4jPointer *extraPointers,int opNum,
                                   float **arguments,
                                   int numArguments,
                                   Nd4jLong **shapeArguments,
                                   int numShapeArguments,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   float *realArguments,
                                   int numRealArguments) {
/*
    NativeOpExcutioner<float>::execAggregate(opNum,
                                             arguments,
                                             numArguments,
                                             shapeArguments,
                                             numShapeArguments,
                                             indexArguments,
                                             numIndexArguments,
                                             intArrays,
                                             numIntArrays,
                                             realArguments,
                                             numRealArguments);
    */
}

void NativeOps::execAggregateDouble(Nd4jPointer *extraPointers,int opNum,
                                    double **arguments,
                                    int numArguments,
                                    Nd4jLong **shapeArguments,
                                    int numShapeArguments,
                                    int *indexArguments,
                                    int numIndexArguments,
                                    int **intArrays,
                                    int numIntArrays,
                                    double *realArguments,
                                    int numRealArguments) {
/*
    NativeOpExcutioner<double>::execAggregate(opNum,
                                              arguments,
                                              numArguments,
                                              shapeArguments,
                                              numShapeArguments,
                                              indexArguments,
                                              numIndexArguments,
                                              intArrays,
                                              numIntArrays,
                                              realArguments,
                                              numRealArguments);
    */
}

void NativeOps::execAggregateHalf(Nd4jPointer *extraPointers,int opNum,
                                  float16 **arguments,
                                  int numArguments,
                                  Nd4jLong **shapeArguments,
                                  int numShapeArguments,
                                  int *indexArguments,
                                  int numIndexArguments,
                                  int **intArrays,
                                  int numIntArrays,
                                  float16 *realArguments,
                                  int numRealArguments) {

    // TODO: add this at some point
    //NativeOpExcutioner<float16>::execAggregate(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
}



void NativeOps::execAggregateBatchFloat(Nd4jPointer *extraPointers,
                                        int numAggregates,
                                        int opNum,
                                        int maxArgs,
                                        int maxShapes,
                                        int maxIntArrays,
                                        int maxIntArraySize,
                                        int maxIdx,
                                        int maxReals,
                                        void *ptrToArguments) {

    //nd4j_printf("numAggregates: [%i]; opNum: [%i]; maxArgs: [%i]; maxShapes: [%i]; maxIntArrays: [%i]; maxIntArraySize: [%i]; maxIdx: [%i]; maxReals: [%i];\n", numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<float> helper(ptrToArguments,
                                       numAggregates,
                                       maxArgs,
                                       maxShapes,
                                       maxIntArrays,
                                       maxIntArraySize,
                                       maxIdx,
                                       maxReals);

    // special case here, we prefer spread arrangement here, all threads are detached from each other
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(close) default(shared)
    for (int i = 0; i < numAggregates; i++) {
        auto intArrays = new int *[maxIntArrays];

        auto arguments = helper.getArguments(i);
        auto shapes = helper.getShapeArguments(i);
        auto idxArg = helper.getIndexArguments(i);
        auto realArg = helper.getRealArguments(i);

        for (int e = 0; e < maxIntArrays; e++) {
            intArrays[e] = helper.getIntArrayArguments(i, e);
        }

        execAggregateFloat(extraPointers,
                           opNum,
                           arguments,
                           helper.getNumArguments(i),
                           shapes,
                           helper.getNumShapeArguments(i),
                           idxArg,
                           helper.getNumIndexArguments(i),
                           reinterpret_cast<int **>(intArrays),
                           helper.getNumIntArrayArguments(i),
                           realArg,
                           helper.getNumRealArguments(i));

        delete [] intArrays;
    }
}


void NativeOps::execAggregateBatchDouble(Nd4jPointer *extraPointers,
                                         int numAggregates,
                                         int opNum,
                                         int maxArgs,
                                         int maxShapes,
                                         int maxIntArrays,
                                         int maxIntArraySize,
                                         int maxIdx,
                                         int maxReals,
                                         void *ptrToArguments) {

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<double> helper(ptrToArguments,
                                        numAggregates,
                                        maxArgs,
                                        maxShapes,
                                        maxIntArrays,
                                        maxIntArraySize,
                                        maxIdx,
                                        maxReals);

    // special case here, we prefer spread arrangement here, all threads are detached from each other
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(spread) default(shared)
    for (int i = 0; i < numAggregates; i++) {
        auto intArrays = new int *[maxIntArrays];

        auto arguments = helper.getArguments(i);
        auto shapes = helper.getShapeArguments(i);
        auto idxArg = helper.getIndexArguments(i);
        auto realArg = helper.getRealArguments(i);

        for (int e = 0; e < maxIntArrays; e++) {
            intArrays[e] = helper.getIntArrayArguments(i, e);
        }

        execAggregateDouble(extraPointers,
                            opNum,
                            arguments,
                            helper.getNumArguments(i),
                            shapes,
                            helper.getNumShapeArguments(i),
                            idxArg,
                            helper.getNumIndexArguments(i),
                            intArrays,
                            helper.getNumIntArrayArguments(i),
                            realArg,
                            helper.getNumRealArguments(i));

        delete [] intArrays;
    }


}

void NativeOps::execAggregateBatchHalf(Nd4jPointer *extraPointers,
                                       int numAggregates,
                                       int opNum,
                                       int maxArgs,
                                       int maxShapes,
                                       int maxIntArrays,
                                       int maxIntArraySize,
                                       int maxIdx,
                                       int maxReals,
                                       void *ptrToArguments) {
    // TODO: add support for fp16
}



void NativeOps::execRandom(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 void *z,
                                 Nd4jLong *zShapeBuffer,
                                 void *extraArguments) {
    NativeOpExcutioner::execRandom(opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandom(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 void *x,
                                 Nd4jLong *xShapeBuffer,
                                 void *y,
                                 Nd4jLong *yShapeBuffer,
                                 void *z,
                                 Nd4jLong *zShapeBuffer,
                                 void *extraArguments) {
    NativeOpExcutioner::execRandom(opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandom(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 void *x,
                                 Nd4jLong *xShapeBuffer,
                                 void *z,
                                 Nd4jLong *zShapeBuffer,
                                 void *extraArguments) {
    NativeOpExcutioner::execRandom(opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}

Nd4jPointer NativeOps::initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {
    auto ptrBuf = reinterpret_cast<long *>(ptrToBuffer);
    auto buffer = new nd4j::random::RandomBuffer(seed, bufferSize, reinterpret_cast<uint64_t *>(ptrBuf));

    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

    return (Nd4jPointer) buffer;
}

void NativeOps::refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->setSeed(seed);
    buffer->setOffset(0);
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();
}

void NativeOps::reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->reSeed(seed);
}


void NativeOps::destroyRandom(Nd4jPointer ptrBuffer) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(ptrBuffer);
    delete buffer;
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

void NativeOps::sort(Nd4jPointer *extraPointers,
        void *x,
        Nd4jLong *xShapeInfo,
        bool descending) {
    NativeOpExcutioner::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortTad(Nd4jPointer *extraPointers,
            void *x,
            Nd4jLong *xShapeInfo,
            int *dimension,
            int dimensionLength,
            Nd4jLong *tadShapeInfo,
            Nd4jLong *tadOffsets,
            bool descending) {
    NativeOpExcutioner::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortCooIndices(Nd4jPointer *extraPointers,
        Nd4jLong *indices,
        void *values,
        Nd4jLong length,
        int rank) {
    NativeOpExcutioner::execSortCooIndices(indices, values, length, rank);
}

Nd4jLong NativeOps::encodeBitmap(Nd4jPointer *extraPointers, void *dx, Nd4jLong *xShapeInfo, Nd4jLong N, int *dz, float threshold) {
    return NativeOpExcutioner::encodeBitmap(dx, xShapeInfo, N, dz, threshold);
}

void NativeOps::decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz) {
    //NativeOpExcutioner<float16>::decodeBitmap(dx, N, dz);
}


Nd4jLong* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length) {
    auto result = new Nd4jLong[2];errno = 0;

#if defined(_WIN32) || defined(_WIN64)
    _mmap(result, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);// checking for failed fopen
    if (fd < 0) {
        nd4j_printf("Errno: %i\n", errno);
        throw std::runtime_error("Failed to open file for MMAP");
    }
    void * ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// check for failed allocation
    if (ptr == MAP_FAILED)
        return nullptr;

    result[0] = (Nd4jLong) ptr;
    result[1] = fd;

#endif

    return result;

}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jLong *ptrMap, Nd4jLong length) {
    munmap((Nd4jPointer) ptrMap[0], length);
#if defined(_WIN32) || defined(_WIN64)
    CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
    close((int) ptrMap[1]);
#endif

    delete[] ptrMap;
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraph(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
}

const char* NativeOps::getAllCustomOps() {
    return nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}

template <typename T>
FORCEINLINE int estimateThresholdGeneric(Nd4jPointer *extraPointers, Nd4jPointer x, int N, T threshold) {
    auto buffer = reinterpret_cast<T *>(x);

    int span = (N / 6) + 8;
    int cnt = 0;

#pragma omp parallel reduction(+:cnt)
    {
        int tid = omp_get_thread_num();
        int start = span * tid;
        int stop = span * (tid + 1);
        if (stop > N)
            stop = N;

#pragma omp simd
        for (int e = start; e < stop; e++) {
            auto v = nd4j::math::nd4j_abs<T>(buffer[e]);
            if (v >= threshold)
                cnt++;
        }
    }

    return cnt;
}

int NativeOps::estimateThresholdFloat(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<float>(extraPointers, x, N, threshold);
}

int NativeOps::estimateThresholdDouble(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<double>(extraPointers, x, N, threshold);
}

int NativeOps::estimateThresholdHalf(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<float16>(extraPointers, x, N, threshold);
}

void NativeOps::deleteShapeList(Nd4jPointer shapeList) {
    auto list = reinterpret_cast<nd4j::ShapeList*>(shapeList);

    list->destroy();
    delete list;
}

nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    nd4j::graph::VariableSpace varSpace;
    Context block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        // we shouldn't copy buffer if that's empty array
        void *buffer_ = nd4j::ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

        auto array = new nd4j::NDArray(buffer_, shape_);
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

nd4j::ShapeList* NativeOps::calculateOutputShapes(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp *op, Nd4jPointer* inputShapes, int numInputShapes, double *tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
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

Nd4jStatus realExec(nd4j::ops::DeclarableOp* op, Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    if (op == nullptr)
        nd4j_printf("Can't find requested operation: [%lld]\n", hash);

    // we're using the same fake nodeId everywhere here

    std::vector<nd4j::NDArray*> inputs(numInputs);
    std::vector<nd4j::NDArray*> outputs(numOutputs);
    std::vector<double> ttArgs(numTArgs);
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

            auto array = new nd4j::NDArray(buffer, shape);
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


int NativeOps::execCustomOp(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(hash);

    return realExec(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::registerGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

static VariablesSet* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    auto graph = nd4j::graph::GraphHolder::getInstance()->cloneGraph(graphId);
    auto varSpace = graph->getVariableSpace();

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

    auto result = nd4j::graph::GraphExecutioner::execute(graph, varSpace);
    auto varSet = new nd4j::graph::VariablesSet(result);

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

    delete graph;

    return varSet;
}

nd4j::graph::VariablesSet* NativeOps::executeStoredGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return nullptr;
}

int NativeOps::unregisterGraph(Nd4jPointer *extraPointers, Nd4jLong graphId) {

    nd4j::graph::GraphHolder::getInstance()->dropGraphAny(graphId);

    return nd4j::Status::OK();
}

void NativeOps::deletePointerArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<Nd4jPointer *>(pointer);
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
    auto ptr = reinterpret_cast<nd4j::graph::VariablesSet*>(pointer);
    delete ptr;
}

void NativeOps::deleteVariablesSet(Nd4jPointer pointer) {
    deleteVariablesSetT<double>(pointer);
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

        auto array = new nd4j::NDArray(buffer, shapeInfo, varSpace->workspace());

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

    auto result = LogicExecutor::processNode(graph, &node);
    if (result != Status::OK())
        return result;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = outputBuffers[e];
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(outputShapes[e]);

        NDArray array(buffer, shapeInfo, varSpace->workspace());

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
    return nd4j::Status::OK(); //execCustomOpWithScope<double>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<double> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

void NativeOps::deleteResultWrapper(Nd4jPointer ptr) {
    // just 0 room for compiler s@!t
    auto p = reinterpret_cast<nd4j::graph::ResultWrapper *>(ptr);
    delete p;
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
            // convertGeneric<double, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //nd4j::TypeCast::convertGeneric<nd4j::float8, double>(nullptr, dx, N, dz);
        } else {
            //nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            //nd4j::TypeCast::convertGeneric<nd4j::int8, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //convertGeneric<nd4j::int8, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<int8_t, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<int8_t, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<int8_t, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<int8_t, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: eventually we might want to add it
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<int8_t, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<int8_t, double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
        //    nd4j::TypeCast::convertGeneric<uint8_t, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<uint8_t, int8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<uint8_t, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<uint8_t, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<uint8_t, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<uint8_t, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: still might want to add
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<uint8_t, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<uint8_t, double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
        //    nd4j::TypeCast::convertGeneric<float16, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<float16, int8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<float16, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<float16, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<float16, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<float16, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: .... ^^^
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<float16, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<float16, double>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<float16>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT16) {
        if (dstType == ND4J_FLOAT8) {
         //   nd4j::TypeCast::convertGeneric<int16_t, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<int16_t, int8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<int16_t, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<int16_t, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            //nd4j::TypeCast::convertGeneric<int16_t, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<int16_t, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO...
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<int16_t, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<int16_t, double>(nullptr, dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
        //    nd4j::TypeCast::convertGeneric<float, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<float, int8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<float, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<float, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<float, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<float, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<float, double>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<float>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {
        if (dstType == ND4J_FLOAT8) {
         //   nd4j::TypeCast::convertGeneric<double, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<double, int8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<double, uint8_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<double, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<double, int16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<double, uint16_t>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<double, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_THRESHOLD) {
        if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertFromThreshold<float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertFromThreshold<float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertFromThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}


BUILD_SINGLE_TEMPLATE(template void flattenGeneric,(Nd4jPointer*, int, char, void*, Nd4jLong*, void*, Nd4jLong*), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void pullRowsGeneric, (void *, Nd4jLong*, void*, Nd4jLong*, const int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void tearGeneric, (void *, Nd4jLong*, Nd4jPointer*, Nd4jLong*, Nd4jLong*, Nd4jLong*), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void shuffleGeneric, (void**, Nd4jLong**, void**, Nd4jLong**, int, int*, Nd4jLong**, Nd4jLong**), LIBND4J_TYPES);


