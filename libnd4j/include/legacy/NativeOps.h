/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#ifndef NATIVEOPERATIONS_NATIVEOPS_H
#define NATIVEOPERATIONS_NATIVEOPS_H



#include <cnpy/cnpy.h>
#include <types/float16.h>


#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/InteropDataBuffer.h>
#include <array/ArrayOptions.h>
#include <array/DataTypeUtils.h>
#include <array/ShapeList.h>
#include <array/TadPack.h>
#include <graph/GraphState.h>
#include <graph/ResultWrapper.h>
#include <graph/VariablesSet.h>
#include <graph/execution/LogicExecutor.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/DebugInfo.h>
#include <memory/MemoryCounter.h>

typedef sd::InteropDataBuffer OpaqueDataBuffer;

extern "C" {


SD_LIB_EXPORT void saveNpy(std::string fname, const OpaqueDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
                           std::string mode = "w");

/**
 * Copy n elements from the buffer from the src
 * buffer to the target buffer
 * @param target the target buffer
 * @param n the number of elements
 * @param from the src buffer
 * @param fromOffset the starting offset for the source
 * @param targetOffset the starting offset for the target
 */
SD_LIB_EXPORT void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset);
/**
 * Print the device buffer
 * @param buffer
 */
SD_LIB_EXPORT void printDeviceBuffer(OpaqueDataBuffer *buffer);

/**
 * This function returns last error code stored,
 * @return non-zero if something bad happened
 */
SD_LIB_EXPORT int lastErrorCode();

/**
 * This function returns last error message, if last error code > 0
 * @return
 */
SD_LIB_EXPORT const char* lastErrorMessage();


/**
 *
 * @param p
 * @param len
 */
SD_LIB_EXPORT void tryPointer(sd::Pointer extra, sd::Pointer p, int len);

/**
 *
 * @param num
 */
SD_LIB_EXPORT void setElementThreshold(int num);

/**
 *
 * @param num
 */
SD_LIB_EXPORT void setTADThreshold(int num);

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
SD_LIB_EXPORT void execIndexReduceScalar(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                         sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                         void* extraParams, OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                         sd::LongType const* dZShapeInfo);

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
SD_LIB_EXPORT void execIndexReduce(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                   sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                   OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                   sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                   sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

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
SD_LIB_EXPORT void execBroadcast(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                 sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                 OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                 sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                 sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                 OpaqueDataBuffer* dbDimension, sd::LongType const* hDimensionShape,
                                 sd::LongType const* dDimensionShape);

SD_LIB_EXPORT void execBroadcastBool(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                     sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                     OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                     sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                     sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                     void* extraParams, OpaqueDataBuffer* dbDimension,
                                     sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

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
SD_LIB_EXPORT void execPairwiseTransform(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                         sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                         OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                         sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                         sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                         void* extraParams);

SD_LIB_EXPORT void execPairwiseTransformBool(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                             sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                             OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                             sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                             sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                             void* extraParams);

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
SD_LIB_EXPORT void execReduceFloat(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                   sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                   OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                   sd::LongType const* dZShapeInfo);

SD_LIB_EXPORT void execReduceSame(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                  OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                  sd::LongType const* dZShapeInfo);

SD_LIB_EXPORT void execReduceBool(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                  OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                  sd::LongType const* dZShapeInfo);

SD_LIB_EXPORT void execReduceLong(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                  OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                  sd::LongType const* dZShapeInfo);

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
SD_LIB_EXPORT void execReduceFloat2(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                    sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                    OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                    sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                    sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

SD_LIB_EXPORT void execReduceSame2(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                   sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                   OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                   sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                   sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

SD_LIB_EXPORT void execReduceBool2(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                   sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                   OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                   sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                   sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

SD_LIB_EXPORT void execReduceLong2(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                   sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                   OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                   sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                   sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape);

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
SD_LIB_EXPORT void execReduce3(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                               sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParamsVals,
                               OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo, sd::LongType const* dYShapeInfo,
                               OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo);

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 */
SD_LIB_EXPORT void execReduce3Scalar(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                     sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                     void* extraParamsVals, OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                     sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                     sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo);
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
SD_LIB_EXPORT void execReduce3Tad(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                  void* extraParamsVals, OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                  sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                  sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                  OpaqueDataBuffer* dbDimension, sd::LongType const* hDimensionShape,
                                  sd::LongType const* dDimensionShape, sd::LongType const* tadOnlyShapeInfo,
                                  sd::LongType const* tadOffsets, sd::LongType const* yTadOnlyShapeInfo,
                                  sd::LongType const* yTadOffsets);

SD_LIB_EXPORT void execReduce3All(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                  void* extraParamsVals, OpaqueDataBuffer* dbY, sd::LongType const* hYShapeInfo,
                                  sd::LongType const* dYShapeInfo, OpaqueDataBuffer* dbZ,
                                  sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                                  OpaqueDataBuffer* dbDimension, sd::LongType const* hDimensionShape,
                                  sd::LongType const* dDimensionShape, sd::LongType const* xTadShapeInfo,
                                  sd::LongType const* xOffsets, sd::LongType const* yTadShapeInfo,
                                  sd::LongType const* yOffsets);

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
SD_LIB_EXPORT void execScalar(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                              sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, OpaqueDataBuffer* dbZ,
                              sd::LongType const* hZShapeInfo, sd::LongType const* dZShapeInfo,
                              OpaqueDataBuffer* dbScalar, sd::LongType const* hSscalarShapeInfo,
                              sd::LongType const* dSscalarShapeInfo, void* extraParams);

SD_LIB_EXPORT void execScalarBool(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                  sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                  OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                  sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbScalar,
                                  sd::LongType const* hSscalarShapeInfo, sd::LongType const* dSscalarShapeInfo,
                                  void* extraParams);

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
SD_LIB_EXPORT void execSummaryStatsScalar(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                          sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                          void* extraParams, OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                          sd::LongType const* dZShapeInfo, bool biasCorrected);
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
SD_LIB_EXPORT void execSummaryStats(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                    sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo, void* extraParams,
                                    OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                    sd::LongType const* dZShapeInfo, bool biasCorrected);
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
SD_LIB_EXPORT void execSummaryStatsTad(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                       sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                       void* extraParams, OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                       sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbDimension,
                                       sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape,
                                       bool biasCorrected, sd::LongType const* tadShapeInfo,
                                       sd::LongType const* tadOffsets);

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
SD_LIB_EXPORT void execTransformFloat(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                      sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                      OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                      sd::LongType const* dZShapeInfo, void* extraParams);

SD_LIB_EXPORT void execTransformSame(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                     sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                     OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                     sd::LongType const* dZShapeInfo, void* extraParams);

SD_LIB_EXPORT void execTransformBool(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                     sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                     OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                     sd::LongType const* dZShapeInfo, void* extraParams);

SD_LIB_EXPORT void execTransformAny(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                    sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                    OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                    sd::LongType const* dZShapeInfo, void* extraParams);

SD_LIB_EXPORT void execTransformStrict(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                       sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                       OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                       sd::LongType const* dZShapeInfo, void* extraParams);

/**
 *
 * @param extraPointers
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param z
 * @param zShapeInfo
 * @param scalars
 * @param extraParams
 * @param dimension
 * @param dimensionLength
 */
SD_LIB_EXPORT void execScalarTad(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                 sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                 OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                 sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbScalars,
                                 sd::LongType const* hScalarShapeInfo, sd::LongType const* dScalarShapeInfo,
                                 void* extraParams, OpaqueDataBuffer* dbDimension, sd::LongType const* hDimensionShape,
                                 sd::LongType const* dDimensionShape, sd::LongType const* tadShapeInfo,
                                 sd::LongType const* tadOffsets, sd::LongType const* tadShapeInfoZ,
                                 sd::LongType const* tadOffsetsZ);

SD_LIB_EXPORT void execScalarBoolTad(sd::Pointer* extraPointers, int opNum, OpaqueDataBuffer* dbX,
                                     sd::LongType const* hXShapeInfo, sd::LongType const* dXShapeInfo,
                                     OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeInfo,
                                     sd::LongType const* dZShapeInfo, OpaqueDataBuffer* dbScalars,
                                     sd::LongType const* hScalarShapeInfo, sd::LongType const* dScalarShapeInfo,
                                     void* extraParams, OpaqueDataBuffer* dbDimension,
                                     sd::LongType const* hDimensionShape, sd::LongType const* dDimensionShape,
                                     sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                     sd::LongType const* tadShapeInfoZ, sd::LongType const* tadOffsetsZ);

SD_LIB_EXPORT void specialConcat(sd::Pointer* extraPointers, int dimension, int numArrays, sd::Pointer* data,
                                 sd::Pointer* inputShapeInfo, void* result, sd::LongType const* resultShapeInfo,
                                 sd::Pointer* tadPointers, sd::Pointer* offsetPointers);

/**
 * This method implementation exists only for cuda.
 * The other backends should have dummy method for JNI compatibility reasons.
 */
SD_LIB_EXPORT void initializeDevicesAndFunctions();

SD_LIB_EXPORT void initializeFunctions(sd::Pointer* functions);

/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
SD_LIB_EXPORT sd::Pointer mallocHost(sd::LongType memorySize, int flags);

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
SD_LIB_EXPORT sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int flags);

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
SD_LIB_EXPORT int freeHost(sd::Pointer pointer);

/**
 * This method releases previously allocated memory space on device
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
SD_LIB_EXPORT int freeDevice(sd::Pointer pointer, int deviceId);

/**
 *
 * @return
 */
SD_LIB_EXPORT int ompGetMaxThreads();

/**
 *
 * @return
 */
SD_LIB_EXPORT int ompGetNumThreads();

/**
 *
 * @param threads
 */
SD_LIB_EXPORT void setOmpNumThreads(int threads);

/**
 *
 * @param threads
 */
SD_LIB_EXPORT void setOmpMinThreads(int threads);

SD_LIB_EXPORT bool isBlasVersionMatches(int major, int minor, int build);

/**
 *
 * @return
 */
SD_LIB_EXPORT sd::Pointer createContext();

/**
 *
 * @return
 */
SD_LIB_EXPORT sd::Pointer createStream();

/**
 *
 * @return
 */
SD_LIB_EXPORT sd::Pointer createEvent();

/**
 *
 * @param event
 * @param stream
 * @return
 */
SD_LIB_EXPORT int registerEvent(sd::Pointer event, sd::Pointer stream);

/**
 *
 * @param event
 * @return
 */
SD_LIB_EXPORT int destroyEvent(sd::Pointer event);

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT int setDevice(int deviceId);

/**
 *
 * @return
 */
SD_LIB_EXPORT int getDevice();

/**
 *
 * @param stream
 * @return
 */
SD_LIB_EXPORT int streamSynchronize(sd::Pointer stream);

/**
 *
 * @param event
 * @return
 */
SD_LIB_EXPORT int eventSynchronize(sd::Pointer event);

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT sd::LongType getDeviceFreeMemory(int deviceId);

/**
 * Returns amount of free memory for current device
 * @return
 */
SD_LIB_EXPORT sd::LongType getDeviceFreeMemoryDefault();

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT sd::LongType getDeviceTotalMemory(int deviceId);

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT int getDeviceMajor(int deviceId);

/**
 * This method returns amount of cached memory
 * @param deviceId
 * @return
 */
SD_LIB_EXPORT sd::LongType getCachedMemory(int deviceId);

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT int getDeviceMinor(int deviceId);

/**
 *
 * @param ptrToDeviceId
 * @return
 */
SD_LIB_EXPORT const char* getDeviceName(int deviceId);

/**
 *
 * @param dst
 * @param src
 * @param size
 * @param flags
 * @param reserved
 * @return
 */
SD_LIB_EXPORT int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved);

/**
 *
 * @param dst
 * @param src
 * @param size
 * @param flags
 * @param reserved
 * @return
 */
SD_LIB_EXPORT int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved);

/**
 *
 * @param dst
 * @param value
 * @param size
 * @param flags
 * @param reserved
 * @return
 */
SD_LIB_EXPORT int memsetSync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved);

/**
 *
 * @param dst
 * @param value
 * @param size
 * @param flags
 * @param reserved
 * @return
 */
SD_LIB_EXPORT int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved);

/**
 *
 * @param dst
 * @param src
 * @param size
 * @param flags
 * @param reserved
 * @return
 */
SD_LIB_EXPORT int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size, int flags,
                                      sd::Pointer reserved);

/**
 *
 * @return
 */
SD_LIB_EXPORT sd::Pointer getConstantSpace();

/**
 *
 * @return
 */
SD_LIB_EXPORT int getAvailableDevices();

/**
 *
 * @param reallyEnable
 */
SD_LIB_EXPORT void enableDebugMode(bool reallyEnable);

/**
 *
 * @param reallyEnable
 */
SD_LIB_EXPORT void enableVerboseMode(bool reallyEnable);

/**
 *
 * @param gridSize
 */
SD_LIB_EXPORT void setGridLimit(int gridSize);

typedef sd::TadPack OpaqueTadPack;

/**
 *
 * @param xShapeInfo
 * @param dimension
 * @param dimensionLength
 * @param targetBuffer
 * @param offsetsBuffer
 */
SD_LIB_EXPORT OpaqueTadPack* tadOnlyShapeInfo(sd::LongType const* xShapeInfo, int* dimension, int dimensionLength);

SD_LIB_EXPORT sd::LongType const* getPrimaryShapeInfo(OpaqueTadPack* pack);
SD_LIB_EXPORT sd::LongType const* getPrimaryOffsets(OpaqueTadPack* pack);
SD_LIB_EXPORT sd::LongType const* getSpecialShapeInfo(OpaqueTadPack* pack);
SD_LIB_EXPORT sd::LongType const* getSpecialOffsets(OpaqueTadPack* pack);
SD_LIB_EXPORT sd::LongType getNumberOfTads(OpaqueTadPack* pack);
SD_LIB_EXPORT int getShapeInfoLength(OpaqueTadPack* pack);

SD_LIB_EXPORT void deleteTadPack(OpaqueTadPack* ptr);

/*
 * PullRow special op
 */

/**
 *
 * @param extraPointers
 * @param x
 * @param xShapeInfo
 * @param z
 * @param zShapeInfo
 * @param n
 * @param indexes
 * @param tadShapeInfo
 * @param tadOffsets
 * @param zTadShapeInfo
 * @param zTadOffsets
 */
SD_LIB_EXPORT void pullRows(sd::Pointer* extraPointers, OpaqueDataBuffer* dbX, sd::LongType const* xShapeInfo,
                            sd::LongType const* dxShapeInfo, OpaqueDataBuffer* dbZ, sd::LongType const* zShapeInfo,
                            sd::LongType const* dzShapeInfo, sd::LongType n, sd::LongType* indexes,
                            sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                            sd::LongType const* zTadShapeInfo, sd::LongType const* zTadOffsets);

/**
 *
 * @param extras
 * @param dx
 * @param dz
 * @param n
 * @param length
 * @param propagate
 */
SD_LIB_EXPORT void average(sd::Pointer* extras, sd::Pointer* x, sd::LongType const* xShapeInfo, sd::Pointer* dx,
                           sd::LongType const* dxShapeInfo, void* z, sd::LongType const* zShapeInfo, void* dz,
                           sd::LongType const* dzShapeInfo, int n, sd::LongType length, bool propagate);

SD_LIB_EXPORT void accumulate(sd::Pointer* extras, sd::Pointer* x, sd::LongType const* xShapeInfo, sd::Pointer* dx,
                              sd::LongType const* dxShapeInfo, void* z, sd::LongType const* zShapeInfo, void* dz,
                              sd::LongType const* dzShapeInfo, int n, sd::LongType length);

/**
 * P2P enabler
 */
/**
 *
 * @param enable
 */
SD_LIB_EXPORT void enableP2P(bool enable);

/**
 *
 */
SD_LIB_EXPORT void checkP2P();

/**
 *
 * @return
 */
SD_LIB_EXPORT bool isP2PAvailable();

/**
 * Shuffle methods
 */

/**
 *
 * @param extras
 * @param dx
 * @param xShapeInfo
 * @param dz
 * @param zShapeInfo
 * @param N
 * @param shuffleMap
 * @param tadShapeInfo
 * @param tadOffsets
 */
SD_LIB_EXPORT void shuffle(sd::Pointer* extras, sd::Pointer* x, sd::Pointer* xShapeInfo, sd::Pointer* dx,
                           sd::Pointer* dxShapeInfo, sd::Pointer* z, sd::Pointer* zShapeInfo, sd::Pointer* dz,
                           sd::Pointer* dzShapeInfo, int N, int* shuffleMap, sd::Pointer* tadShapeInfo,
                           sd::Pointer* tadOffsets);

/**
 * Type Conversions
 */

/**
 *
 * @param extras
 * @param srcType
 * @param x
 * @param N
 * @param dstType
 * @param z
 */
SD_LIB_EXPORT void convertTypes(sd::Pointer* extras, int srcType, sd::Pointer x, sd::LongType N, int dstType,
                                sd::Pointer z);

/**
 *
 * @return
 */
SD_LIB_EXPORT bool isExperimentalEnabled();

/**
 * Aggregate
 */

/**
 *
 * @param extraPointers
 * @param opNum
 * @param arguments
 * @param numArguments
 * @param shapeArguments
 * @param numShapeArguments
 * @param indexArguments
 * @param numIndexArguments
 * @param intArrays
 * @param numIntArrays
 * @param realArguments
 * @param numRealArguments
 */
SD_LIB_EXPORT void execAggregate(sd::Pointer* extraPointers, int opNum, void** arguments, int numArguments,
                                 sd::LongType** shapeArguments, int numShapeArguments, int* indexArguments,
                                 int numIndexArguments, int** intArrays, int numIntArrays, void* realArguments,
                                 int numRealArguments, sd::DataType dtype);

SD_LIB_EXPORT void batchExecutor(sd::Pointer* extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                                 int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void* ptrToArguments,
                                 sd::DataType dtype);

SD_LIB_EXPORT void execAggregateBatch(sd::Pointer* extraPointers, int numAggregates, int opNum, int maxArgs,
                                      int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                                      void* ptrToArguments, sd::DataType dtype);

/**
 * Random operations
 */

/**
 *
 * @param extraPointers
 * @param opNum
 * @param state
 * @param z
 * @param zShapeBuffer
 * @param extraArguments
 */
SD_LIB_EXPORT void execRandom(sd::Pointer* extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer* dbZ,
                              sd::LongType const* hZShapeBuffer, sd::LongType const* dZShapeBuffer,
                              void* extraArguments);

/**
 *
 * @param extraPointers
 * @param opNum
 * @param state
 * @param x
 * @param xShapeBuffer
 * @param y
 * @param yShapeBuffer
 * @param z
 * @param zShapeBuffer
 * @param extraArguments
 */
SD_LIB_EXPORT void execRandom3(sd::Pointer* extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer* dbX,
                               sd::LongType const* hXShapeBuffer, sd::LongType const* dXShapeBuffer,
                               OpaqueDataBuffer* dbY, sd::LongType const* hYShapeBuffer,
                               sd::LongType const* dYShapeBuffer, OpaqueDataBuffer* dbZ,
                               sd::LongType const* hZShapeBuffer, sd::LongType const* dZShapeBuffer,
                               void* extraArguments);

/**
 *
 * @param extraPointers
 * @param opNum
 * @param state
 * @param x
 * @param xShapeBuffer
 * @param z
 * @param zShapeBuffer
 * @param extraArguments
 */
SD_LIB_EXPORT void execRandom2(sd::Pointer* extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer* dbX,
                               sd::LongType const* hXShapeBuffer, sd::LongType const* dXShapeBuffer,
                               OpaqueDataBuffer* dbZ, sd::LongType const* hZShapeBuffer,
                               sd::LongType const* dZShapeBuffer, void* extraArguments);

/**
 *
 * @param extraPointers
 * @param seed
 * @param bufferSize
 * @param ptrToBuffer
 * @return
 */
SD_LIB_EXPORT sd::Pointer initRandom(sd::Pointer* extraPointers, long seed, long bufferSize, sd::Pointer ptrToBuffer);

/**
 *
 * @param extraPointers
 * @param seed
 * @param ptrRandom
 */
SD_LIB_EXPORT void refreshBuffer(sd::Pointer* extraPointers, long seed, sd::Pointer ptrRandom);

/**
 *
 * @param extraPointers
 * @param seed
 * @param ptrRandom
 */
SD_LIB_EXPORT void reSeedBuffer(sd::Pointer* extraPointers, long seed, sd::Pointer ptrRandom);

/**
 *
 * @param ptrRandom
 */
SD_LIB_EXPORT void destroyRandom(sd::Pointer ptrRandom);
}

/**
 *
 * @param data
 * @param shapeBuffer
 * @param wordSize
 * @param headerSize
 * @return
 */

template <typename T>
static sd::Pointer _numpyHeaderForNd4j(sd::Pointer data, const sd::Pointer shapeBuffer, sd::LongType wordSize,
                                       sd::LongType* headerSize) {
  sd::LongType const* shapeBufferCast = reinterpret_cast<const sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  const sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  char* ret = new char[npHeader.size() + 1];
  int count = 0;
  for (int i = 0; i < npHeader.size(); i++) {
    ret[count] = npHeader[i];
    count++;
  }

  ret[count] = '\0';
  count++;

  *headerSize = count;
  return reinterpret_cast<sd::Pointer>(ret);
}

extern "C" {

static long lengthInBytes(OpaqueDataBuffer *buffer) {
  return buffer->dataBuffer()->getLenInBytes();
}

static sd::Pointer numpyHeaderForNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize,
                                      sd::LongType* headerSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderForNd4j, (data, shapeBuffer, wordSize, headerSize), SD_COMMON_TYPES);
}

/**
 * Load numpy from a header
 * based on the cnpy parse from header method.
 * @param data the header data to parse
 * @return a pointer to a numpy cnpy:NpyArray struct
 */
static sd::Pointer loadNpyFromHeader(sd::Pointer data) {
  char* header = reinterpret_cast<char*>(data);

  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(header);
  cnpy::NpyArray* ret = new cnpy::NpyArray();
  int totalLengthOfShape = 1;
  for (int i = 0; i < arr.shape.size(); i++) {
    totalLengthOfShape *= arr.shape[i];
  }

  ret->data = arr.data;
  ret->wordSize = arr.wordSize;
  ret->shape = arr.shape;
  return reinterpret_cast<sd::Pointer>(ret);
}
}

/**
 * Create a numpy array from an nd4j
 * array
 * @param data a pointer to the data
 * @param shapeBuffer  the shapebuffer for the nd4j array
 * @param wordSize  the word size (4 for float, 8 for doubles)
 * @return a pointer to a numpy array
 */

template <typename T>
static sd::Pointer _numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>( npShape, rank, wordSize);
  char* dataChar = reinterpret_cast<char*>(data);
  char* npHeaderData = npHeader.data();
  char* ret = new char[(wordSize * length) + npHeader.size()];
  char* cursorStart = ret + npHeader.size();
  std::memcpy(ret, npHeaderData,
              npHeader.size());
  std::memcpy(cursorStart, dataChar,length  * wordSize);
  sd::Pointer rettPointer = reinterpret_cast<sd::Pointer>(ret);
  return rettPointer;
}
template<typename T>
static long _numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  sd::LongType wordSize = opaqueDataBuffer->dataBuffer()->getLenInBytes() / opaqueDataBuffer->dataBuffer()->getNumElements();
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  return ret;
}

template<typename  T>
 static long _numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  return ret;
}


extern "C" {

 static long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLengthWordSize, (shapeBuffer, wordSize), SD_COMMON_TYPES);

}

 static long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLength, (opaqueDataBuffer, shapeBuffer), SD_COMMON_TYPES);

}



 static sd::Pointer numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyFromNd4j, (data, shapeBuffer, wordSize), SD_COMMON_TYPES);
}

/**
 *
 * @param npyArray
 * @return
 */
SD_LIB_EXPORT sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray);

/**
 * Get the shape buffer from a
 * numpy array.
 * **Warning** this allocates memory
 * @param npyArray
 * @return
 */
static sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  auto shape = new unsigned int[arr.shape.size()];
  for (unsigned int i = 0; i < arr.shape.size(); i++) {
    shape[i] = arr.shape[i];
  }

  auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
  delete[] shape;
  return reinterpret_cast<sd::Pointer>(shapeBuffer);
}

/**
 *
 * @param npyArray
 * @return
 */
static sd::Pointer dataPointForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arr.data);
  return dataToPrint;
}

/**
 *
 * @param npyArray
 * @return
 */
static sd::Pointer dataPointForNumpyStruct(sd::Pointer npyArrayStruct) {
  cnpy::NpyArray* arrPointer = reinterpret_cast<cnpy::NpyArray*>(npyArrayStruct);
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arrPointer->data);
  return reinterpret_cast<sd::Pointer>(dataToPrint);
}

/**
 *
 * @param npyArray
 * @param fromFile
 * @return
 */
static sd::Pointer dataPointForNumpy(sd::Pointer npyArray) {
  char* npyArrayBuffer = reinterpret_cast<char*>(npyArray);
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(npyArrayBuffer);
  return dataPointForNumpyStruct(reinterpret_cast<sd::Pointer>(&arr));
}

/**
 * Load a numpy array from a file
 * and return it as an sd::Pointer
 * @param path
 * @return
 */
static sd::Pointer numpyFromFile(std::string path) {
  char* numpyBuffer = cnpy::loadFile(path.data());
  return reinterpret_cast<sd::Pointer>(numpyBuffer);
}

////// NPZ //////

static void* mapFromNpzFile(std::string path) {
  cnpy::npz_t* mapPtr = new cnpy::npz_t();
  cnpy::npz_t map = cnpy::npzLoad(path);
  mapPtr->insert(map.begin(), map.end());
  return reinterpret_cast<void*>(mapPtr);
}

static int getNumNpyArraysInMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  int n = arrays->size();
  return n;
}

static const char* getNpyArrayNameFromMap(void* map, int index, char* nameBuffer) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      size_t len_of_str = strlen(it->first.c_str());
      memcpy(nameBuffer, it->first.c_str(), len_of_str);
    }
  }
  throw std::runtime_error("No array at index.");
}

static void* getNpyArrayFromMap(void* map, int index) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  cnpy::NpyArray* arr = new cnpy::NpyArray();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      *arr = it->second;
      return arr;
    }
  }
  throw std::runtime_error("No array at index.");
}

SD_LIB_EXPORT int dataTypeFromNpyHeader(void* header);

static void* getNpyArrayData(void* npArray) {
  cnpy::NpyArray* npyArray2 = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return reinterpret_cast<void*>(npyArray2->data);
}

static int getNpyArrayRank(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int rank = arr->shape.size();
  return rank;
}

static sd::LongType* getNpyArrayShape(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int ndim = arr->shape.size();
  sd::LongType* shape = new sd::LongType[ndim];
  for (int i = 0; i < ndim; i++) {
    shape[i] = arr->shape.at(i);
  }
  return shape;
}

static char getNpyArrayOrder(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return (arr->fortranOrder) ? 'f' : 'c';
}

static int getNpyArrayElemSize(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return arr->wordSize;
}

static void deleteNPArrayStruct(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  delete arr;
}

static void deleteNPArrayMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  delete arrays;
}
//////

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
static int elementSizeForNpyArray(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  // arrPointer->destruct();
  return size;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
static int elementSizeForNpyArrayHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

static void releaseNumpy(sd::Pointer npyArray) { free(reinterpret_cast<void*>(npyArray)); }

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer pointer to check
 * @return
 */
SD_LIB_EXPORT int lengthForShapeBufferPointer(sd::Pointer buffer);

/**
 * The pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the pointer for the given address
 */

SD_LIB_EXPORT sd::Pointer pointerForAddress(sd::LongType address);

/**
 * This method takes single N-dimensional tensor, and copies its TADs to target arrays
 *
 * @param x
 * @param xShapeInfo
 * @param targets
 * @param zShapeInfo
 * @return
 */
SD_LIB_EXPORT void tear(sd::Pointer* extraPointers, OpaqueDataBuffer* dbX, sd::LongType const* xShapeInfo,
                        sd::LongType const* dxShapeInfo, sd::Pointer* targets, sd::LongType const* zShapeInfo,
                        sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets);

SD_LIB_EXPORT void sort(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                        sd::LongType const* dxShapeInfo, bool descending);

SD_LIB_EXPORT void sortByKey(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                             sd::LongType const* dxShapeInfo, void* y, sd::LongType const* yShapeInfo, void* dy,
                             sd::LongType const* dyShapeInfo, bool descending);

SD_LIB_EXPORT void sortByValue(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                               sd::LongType const* dxShapeInfo, void* y, sd::LongType const* yShapeInfo, void* dy,
                               sd::LongType const* dyShapeInfo, bool descending);

SD_LIB_EXPORT void sortTad(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                           sd::LongType const* dxShapeInfo, int* dimension, int dimensionLength,
                           sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, bool descending);

SD_LIB_EXPORT void sortTadByKey(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                                sd::LongType const* dxShapeInfo, void* y, sd::LongType const* yShapeInfo, void* dy,
                                sd::LongType const* dyShapeInfo, int* dimension, int dimensionLength, bool descending);

SD_LIB_EXPORT void sortTadByValue(sd::Pointer* extraPointers, void* x, sd::LongType const* xShapeInfo, void* dx,
                                  sd::LongType const* dxShapeInfo, void* y, sd::LongType const* yShapeInfo, void* dy,
                                  sd::LongType const* dyShapeInfo, int* dimension, int dimensionLength,
                                  bool descending);

// special sort impl for sorting out COO indices and values
SD_LIB_EXPORT void sortCooIndices(sd::Pointer* extraPointers, sd::LongType* indices, void* x, sd::LongType length,
                                  const sd::LongType* xShapeInfo);

/**
 *
 * @param extraPointers     not used
 * @param indices           DataBuffer containing COO indices for a sparse matrix that is to be raveled/flattened
 * @param flatIndices       DataBuffer where the raveled/flattened indices are to be written to
 * @param length            number of non-zero entries (length of flatIndices)
 * @param fullShapeBuffer   DataBuffer with ShapeInfo for the full matrix to be flattened
 * @param mode              clipMode determines the strategy to use if some of the the passed COO indices does
 *                          not fit into the shape determined by fullShapeBuffer
 *                              0   throw an exception (default)
 *                              1   wrap around shape
 *                              2   clip to shape
 */
SD_LIB_EXPORT void ravelMultiIndex(sd::Pointer* extraPointers, sd::LongType* indices, sd::LongType* flatIndices,
                                   sd::LongType length, sd::LongType* shapeInfo, int mode);

/**
 *
 * @param extraPointers     not used
 * @param indices           DataBuffer where the unraveled COO indices are to be written
 * @param flatIndices       DataBuffer containing the raveled/flattened indices to be unravel
 * @param length            number of non-zero entries (length of flatIndices)
 * @param fullShapeBuffer   DataBuffer with ShapeInfo for the full matrix to be unraveled
 */
SD_LIB_EXPORT void unravelIndex(sd::Pointer* extraPointers, sd::LongType* indices, sd::LongType* flatIndices,
                                sd::LongType length, sd::LongType* shapeInfo);

SD_LIB_EXPORT sd::LongType* mmapFile(sd::Pointer* extraPointers, const char* fileName, sd::LongType length);

SD_LIB_EXPORT void munmapFile(sd::Pointer* extraPointers, sd::LongType* ptrMap, sd::LongType length);

typedef sd::graph::ResultWrapper OpaqueResultWrapper;

// flatbuffers execution
SD_LIB_EXPORT OpaqueResultWrapper* executeFlatGraph(sd::Pointer* extraPointers, sd::Pointer flatBufferPointer);

SD_LIB_EXPORT sd::LongType getResultWrapperSize(OpaqueResultWrapper* ptr);
SD_LIB_EXPORT sd::Pointer getResultWrapperPointer(OpaqueResultWrapper* ptr);

SD_LIB_EXPORT const char* getAllCustomOps();

SD_LIB_EXPORT const char* getAllOperations();

// customOp executioner
SD_LIB_EXPORT sd::Status execCustomOp(sd::Pointer* extraPointers, sd::LongType hash, sd::Pointer* inputBuffers,
                                      sd::Pointer* inputShapes, int numInputs, sd::Pointer* outputBuffers,
                                      sd::Pointer* outputShapes, int numOutputs, double* tArgs, int numTArgs,
                                      sd::LongType* iArgs, int numIArgs, bool* bArgs, int numBArgs, bool isInplace);
SD_LIB_EXPORT sd::Status execCustomOp2(sd::Pointer* extraPointers, sd::LongType hash, sd::Pointer opContext);

typedef sd::ShapeList OpaqueShapeList;
typedef sd::graph::Context OpaqueContext;

SD_LIB_EXPORT OpaqueShapeList* calculateOutputShapes(sd::Pointer* extraPointers, sd::LongType hash,
                                                     sd::Pointer* inputShapes, int numInputShapes, double* tArgs,
                                                     int numTArgs, sd::LongType* iArgs, int numIArgs);
SD_LIB_EXPORT OpaqueShapeList* calculateOutputShapes2(sd::Pointer* extraPointers, sd::LongType hash,
                                                      sd::Pointer* inputBuffers, sd::Pointer* inputShapes,
                                                      int numInputShapes, double* tArgs, int numTArgs,
                                                      sd::LongType* iArgs, int numIArgs, bool* bArgs, int numBArgs,
                                                      int* dArgs, int numDArgs);
#ifdef __NEC__
SD_LIB_EXPORT OpaqueShapeList* calculateOutputShapesFromContext(OpaqueContext* ctx, sd::LongType hash);
SD_LIB_EXPORT int calculateOutputShapesAndFill(OpaqueContext *ctx, sd::LongType hash, void **handleState, int outBufferSizeInBytes, sd::LongType *outConcatenatedShapesBuffer);
SD_LIB_EXPORT void setGraphContextArgs(OpaqueContext *ctx, int numArr, sd::Pointer* inputArrDataShapePairs, int numIArgs, sd::LongType* iArgsPtr,
                                      int numDArgs, int *dArgsPtr, int numTArgs, double *tArgsPtr, int numBArgs, bool *bArgsPtr);
#endif
SD_LIB_EXPORT sd::LongType getShapeListSize(OpaqueShapeList* list);
SD_LIB_EXPORT sd::LongType const* getShape(OpaqueShapeList* list, sd::LongType i);

SD_LIB_EXPORT void deleteShapeList(sd::Pointer shapeList);

SD_LIB_EXPORT sd::Status registerGraph(sd::Pointer* extraPointers, sd::LongType graphId, sd::Pointer flatBufferPointer);

typedef sd::graph::VariablesSet OpaqueVariablesSet;
typedef sd::graph::Variable OpaqueVariable;

SD_LIB_EXPORT OpaqueVariablesSet* executeStoredGraph(sd::Pointer* extraPointers, sd::LongType graphId,
                                                     sd::Pointer* inputBuffers, sd::Pointer* inputShapes,
                                                     int* inputIndices, int numInputs);

SD_LIB_EXPORT sd::LongType getVariablesSetSize(OpaqueVariablesSet* set);
SD_LIB_EXPORT sd::Status getVariablesSetStatus(OpaqueVariablesSet* set);
SD_LIB_EXPORT OpaqueVariable* getVariable(OpaqueVariablesSet* set, sd::LongType i);
SD_LIB_EXPORT int getVariableId(OpaqueVariable* variable);
SD_LIB_EXPORT int getVariableIndex(OpaqueVariable* variable);
SD_LIB_EXPORT const char* getVariableName(OpaqueVariable* variable);
SD_LIB_EXPORT sd::LongType const* getVariableShape(OpaqueVariable* variable);
SD_LIB_EXPORT void* getVariableBuffer(OpaqueVariable* variable);

SD_LIB_EXPORT sd::Status unregisterGraph(sd::Pointer* extraPointers, sd::LongType graphId);

SD_LIB_EXPORT void deleteCharArray(sd::Pointer pointer);
SD_LIB_EXPORT void deleteIntArray(sd::Pointer pointer);
SD_LIB_EXPORT void deleteLongArray(sd::Pointer pointer);
SD_LIB_EXPORT void deletePointerArray(sd::Pointer pointer);

SD_LIB_EXPORT void deleteVariablesSet(OpaqueVariablesSet* pointer);

// GraphState creation
SD_LIB_EXPORT sd::Pointer getGraphState(sd::LongType id);

SD_LIB_EXPORT void deleteGraphState(sd::Pointer state);

SD_LIB_EXPORT void deleteResultWrapper(sd::Pointer ptr);

SD_LIB_EXPORT int estimateThreshold(sd::Pointer* extraPointers, sd::Pointer x, sd::LongType const* xShapeInfo, int N,
                                    float threshold);

// this method executes op that requires scope to be present: if/while/cond/whatever
SD_LIB_EXPORT sd::Status execCustomOpWithScope(sd::Pointer* extraPointers, sd::Pointer state, sd::LongType opHash,
                                               sd::LongType* scopes, int numScopes, sd::Pointer* inputBuffers,
                                               sd::Pointer* inputShapes, int numInputs, sd::Pointer* outputBuffers,
                                               sd::Pointer* outputShapes, int numOutputs);

// void fillUtf8String(sd::Pointer *extraPointers, const char **string, int numStrings, sd::Pointer buffer);
SD_LIB_EXPORT sd::Pointer createUtf8String(sd::Pointer* extraPointers, const char* string, int length);
SD_LIB_EXPORT sd::LongType getUtf8StringLength(sd::Pointer* extraPointers, sd::Pointer ptr);
SD_LIB_EXPORT char* getUtf8StringBuffer(sd::Pointer* extraPointers, sd::Pointer ptr);
SD_LIB_EXPORT void deleteUtf8String(sd::Pointer* extraPointers, sd::Pointer ptr);

SD_LIB_EXPORT void scatterUpdate(sd::Pointer* extraPointers, int opCode, int numOfSubArrs, void* hX,
                                 sd::LongType const* hXShapeInfo, sd::LongType const* hXOffsets, void* dX,
                                 sd::LongType const* dXShapeInfo, sd::LongType const* dXOffsets, void* hY,
                                 sd::LongType const* hYShapeInfo, sd::LongType const* hYOffsets, void* dY,
                                 sd::LongType const* dYShapeInfo, sd::LongType const* dYOffsets, void* hIindexes,
                                 sd::LongType const* hIndicesShapeInfo, void* dIindexes,
                                 sd::LongType const* dIndicesShapeInfo);

SD_LIB_EXPORT void inspectArray(sd::Pointer* extraPointers, sd::Pointer buffer, sd::LongType* shapeInfo,
                                sd::Pointer specialBuffer, sd::LongType* specialShapeInfo, sd::Pointer debugInfo);

typedef sd::ConstantDataBuffer OpaqueConstantDataBuffer;
typedef sd::ConstantShapeBuffer OpaqueConstantShapeBuffer;

SD_LIB_EXPORT OpaqueConstantShapeBuffer* shapeBuffer(int rank, sd::LongType* shape, sd::LongType* strides,
                                                     sd::DataType dtype, char order, sd::LongType ews, bool empty);
SD_LIB_EXPORT OpaqueConstantShapeBuffer* shapeBufferEx(int rank, sd::LongType* shape, sd::LongType* strides,
                                                       sd::DataType dtype, char order, sd::LongType ews,
                                                       sd::LongType extras);

SD_LIB_EXPORT OpaqueConstantDataBuffer* constantBufferLong(sd::DataType dtype, sd::LongType const* data, int length);
SD_LIB_EXPORT OpaqueConstantDataBuffer* constantBufferDouble(sd::DataType dtype, double* data, int length);
SD_LIB_EXPORT OpaqueConstantDataBuffer* constantBuffer(sd::DataType dtype, sd::ConstantDescriptor* descriptor);

SD_LIB_EXPORT sd::Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer* dbf);
SD_LIB_EXPORT sd::Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer* dbf);
SD_LIB_EXPORT sd::LongType getConstantDataBufferLength(OpaqueConstantDataBuffer* dbf);

SD_LIB_EXPORT sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer* dbf);
SD_LIB_EXPORT sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer* dbf);

SD_LIB_EXPORT void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer* ptr);
SD_LIB_EXPORT void deleteConstantDataBuffer(OpaqueConstantDataBuffer* ptr);

typedef sd::graph::RandomGenerator OpaqueRandomGenerator;

SD_LIB_EXPORT OpaqueContext* createGraphContext(int nodeId);
SD_LIB_EXPORT OpaqueRandomGenerator* getGraphContextRandomGenerator(OpaqueContext* ptr);
SD_LIB_EXPORT void ctxAllowHelpers(OpaqueContext* ptr, bool reallyAllow);
SD_LIB_EXPORT void ctxShapeFunctionOverride(OpaqueContext* ptr, bool reallyOverride);
SD_LIB_EXPORT void ctxSetExecutionMode(OpaqueContext* ptr, int execMode);
SD_LIB_EXPORT void ctxPurge(OpaqueContext* ptr);
SD_LIB_EXPORT void markGraphContextInplace(OpaqueContext* ptr, bool reallyInplace);
SD_LIB_EXPORT void setGraphContextCudaContext(OpaqueContext* ptr, void* stream, void* reductionPointer,
                                              void* allocationPointer);



SD_LIB_EXPORT void setGraphContextInputArray(OpaqueContext* ptr, int index, void* buffer, void* shapeInfo,
                                             void* specialBuffer, void* specialShapeInfo);
SD_LIB_EXPORT void setGraphContextOutputArray(OpaqueContext* ptr, int index, void* buffer, void* shapeInfo,
                                              void* specialBuffer, void* specialShapeInfo);
SD_LIB_EXPORT void setGraphContextInputBuffer(OpaqueContext* ptr, int index, OpaqueDataBuffer* buffer, void* shapeInfo,
                                              void* specialShapeInfo);
SD_LIB_EXPORT void setGraphContextOutputBuffer(OpaqueContext* ptr, int index, OpaqueDataBuffer* buffer, void* shapeInfo,
                                               void* specialShapeInfo);

SD_LIB_EXPORT void setGraphContextInputArrays(OpaqueContext* ptr, int numArrays, sd::Pointer * buffer, sd::Pointer * shapeInfo,
                                              sd::Pointer * specialBuffer, sd::Pointer * specialShapeInfo);
SD_LIB_EXPORT void setGraphContextOutputArrays(OpaqueContext* ptr, int numArrays, sd::Pointer * buffer, sd::Pointer * shapeInfo,
                                               sd::Pointer * specialBuffer, sd::Pointer * specialShapeInfo);
SD_LIB_EXPORT void setGraphContextInputBuffers(OpaqueContext* ptr, int numArrays, OpaqueDataBuffer** buffer, sd::Pointer * shapeInfo,
                                               sd::Pointer * specialShapeInfo);
SD_LIB_EXPORT void setGraphContextOutputBuffers(OpaqueContext* ptr, int numArrays, OpaqueDataBuffer** buffer, sd::Pointer * shapeInfo,
                                                sd::Pointer * specialShapeInfo);

SD_LIB_EXPORT void setShapeBuffer(sd::LongType *inputShapeData,sd::DataType dt,sd::LongType *bufferToSet,char order = 'c',int elementWiseStride = 1,bool isEmpty = false);

SD_LIB_EXPORT void setGraphContextDArguments(OpaqueContext* ptr, int* arguments, int numberOfArguments);
SD_LIB_EXPORT void setGraphContextTArguments(OpaqueContext* ptr, double* arguments, int numberOfArguments);
SD_LIB_EXPORT void setGraphContextIArguments(OpaqueContext* ptr, sd::LongType* arguments, int numberOfArguments);
SD_LIB_EXPORT void setGraphContextBArguments(OpaqueContext* ptr, bool* arguments, int numberOfArguments);
SD_LIB_EXPORT void deleteGraphContext(OpaqueContext* ptr);

SD_LIB_EXPORT OpaqueRandomGenerator* createRandomGenerator(sd::LongType rootSeed = 0, sd::LongType nodeSeed = 0);
SD_LIB_EXPORT sd::LongType getRandomGeneratorRootState(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT sd::LongType getRandomGeneratorNodeState(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT void setRandomGeneratorStates(OpaqueRandomGenerator* ptr, sd::LongType rootSeed = 0,
                                            sd::LongType nodeSeed = 0);
SD_LIB_EXPORT float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator* ptr, sd::LongType index);
SD_LIB_EXPORT double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator* ptr, sd::LongType index);
SD_LIB_EXPORT int getRandomGeneratorRelativeInt(OpaqueRandomGenerator* ptr, sd::LongType index);
SD_LIB_EXPORT sd::LongType getRandomGeneratorRelativeLong(OpaqueRandomGenerator* ptr, sd::LongType index);
SD_LIB_EXPORT float getRandomGeneratorNextFloat(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT double getRandomGeneratorNextDouble(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT int getRandomGeneratorNextInt(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT sd::LongType getRandomGeneratorNextLong(OpaqueRandomGenerator* ptr);
SD_LIB_EXPORT void deleteRandomGenerator(OpaqueRandomGenerator* ptr);

typedef sd::LaunchContext OpaqueLaunchContext;

SD_LIB_EXPORT OpaqueLaunchContext* defaultLaunchContext();
SD_LIB_EXPORT sd::Pointer lcScalarPointer(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcReductionPointer(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcAllocationPointer(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcExecutionStream(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcCopyStream(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcBlasHandle(OpaqueLaunchContext* lc);
SD_LIB_EXPORT sd::Pointer lcSolverHandle(OpaqueLaunchContext* lc);

SD_LIB_EXPORT OpaqueDataBuffer* allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth);
SD_LIB_EXPORT OpaqueDataBuffer* dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth);
SD_LIB_EXPORT OpaqueDataBuffer* dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary,
                                                           sd::Pointer special);
SD_LIB_EXPORT int dbUseCount(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT OpaqueDataBuffer* dbCreateView(OpaqueDataBuffer* dataBuffer, sd::LongType length, sd::LongType offset);
SD_LIB_EXPORT sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT sd::Pointer dbSpecialBuffer(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbExpandBuffer(OpaqueDataBuffer* dataBuffer, sd::LongType elements);
SD_LIB_EXPORT void dbAllocatePrimaryBuffer(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbAllocateSpecialBuffer(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbSetPrimaryBuffer(OpaqueDataBuffer* dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes);
SD_LIB_EXPORT void dbSetSpecialBuffer(OpaqueDataBuffer* dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes);
SD_LIB_EXPORT void dbSyncToSpecial(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbSyncToPrimary(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT int dbLocality(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT int dbDeviceId(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbSetDeviceId(OpaqueDataBuffer* dataBuffer, int deviceId);
SD_LIB_EXPORT void dbTickHostRead(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbTickHostWrite(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbTickDeviceRead(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbTickDeviceWrite(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbClose(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void deleteDataBuffer(OpaqueDataBuffer* dataBuffer);
SD_LIB_EXPORT void dbExpand(OpaqueDataBuffer* dataBuffer, sd::LongType elements);

SD_LIB_EXPORT int binaryLevel();
SD_LIB_EXPORT int optimalLevel();

SD_LIB_EXPORT bool isMinimalRequirementsMet();
SD_LIB_EXPORT bool isOptimalRequirementsMet();

SD_LIB_EXPORT void setVedaDeviceLibFolder(std::string path);

}

#endif  // NATIVEOPERATIONS_NATIVEOPS_H
