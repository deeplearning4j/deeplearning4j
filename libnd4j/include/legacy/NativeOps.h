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

#ifndef NATIVEOPS_H
#define NATIVEOPS_H

#include <array/ArrayOptions.hXX>
#include <array/DataTypeUtils.h>
#include <array/ShapeList.h>
#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/InteropDataBuffer.h>
#include <array/TadPack.h>
#include <cnpy/cnpy.h>
#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
// Windows-specific backtrace implementation
#else
#include <execinfo.h>
#include <unistd.h>
// Unix-style backtrace implementation
#endif
#include <graph/GraphState.h>
#include <graph/ResultWrapper.h>
#include <graph/VariablesSet.h>
#include <graph/execution/LogicExecutor.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/DebugInfo.h>
#include <memory/MemoryCounter.h>
#include <ops/declarable/OpRegistrator.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <types/float16.h>

typedef sd::InteropDataBuffer  OpaqueDataBuffer;
typedef sd::ops::OpExecTrace ExecTrace;
typedef sd::ShapeList OpaqueShapeList;
typedef Context OpaqueContext;
typedef sd::NDArray* OpaqueNDArray;
typedef sd::NDArray** OpaqueNDArrayArr;
typedef sd::LaunchContext* OpaqueLaunchContext;
typedef RandomGenerator* OpaqueRandomGenerator;
typedef sd::graph::ResultWrapper OpaqueResultWrapper;
typedef sd::graph::VariablesSet OpaqueVariablesSet;
typedef sd::graph::Variable OpaqueVariable;
typedef sd::TadPack OpaqueTadPack;

typedef sd::ConstantDataBuffer* OpaqueConstantDataBuffer;
typedef sd::ConstantShapeBuffer* OpaqueConstantShapeBuffer;




SD_LIB_EXPORT const char* getAllCustomOps();
SD_LIB_EXPORT OpaqueRandomGenerator createRandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed);

SD_LIB_EXPORT OpaqueContext *createGraphContext(int nodeId);
SD_LIB_EXPORT void setGraphContextCudaContext(OpaqueContext *ptr, void *stream, void *reductionPointer,
                                              void *allocationPointer);
SD_LIB_EXPORT OpaqueRandomGenerator getGraphContextRandomGenerator(OpaqueContext *ptr);

SD_LIB_EXPORT void shuffle(sd::Pointer *extras,
                           OpaqueNDArrayArr x,
                           OpaqueNDArrayArr z,
                           int N,
                           OpaqueNDArray dimension,
                           OpaqueNDArray shuffleMap);





SD_LIB_EXPORT void pullRows(sd::Pointer *extraPointers,
                            OpaqueNDArray x,
                            OpaqueNDArray z,
                            sd::LongType n,
                            OpaqueNDArray indexes,
                            sd::LongType dimension);

SD_LIB_EXPORT std::vector<ExecTrace*> * listOpTraces();
SD_LIB_EXPORT char *opName(void *execTrace);
SD_LIB_EXPORT std::vector<bool> * bArgs(void *execTrace);
SD_LIB_EXPORT std::vector<std::string> * sArgs(void *execTrace);
SD_LIB_EXPORT std::vector<double> * tArgs(void *execTrace);
SD_LIB_EXPORT std::vector<sd::LongType> * iArgs(void *execTrace);
SD_LIB_EXPORT std::vector<int> * dArgs(void *execTrace);
SD_LIB_EXPORT std::vector<const sd::LongType *> *inputShapeBuffers(void *execTrace);
SD_LIB_EXPORT std::vector<const sd::LongType *> *outputShapeBuffers(void *execTrace);
SD_LIB_EXPORT void deleteNDArray(OpaqueNDArray array);

SD_LIB_EXPORT sd::LongType getOpaqueNDArrayOffset(OpaqueNDArray array) ;


SD_LIB_EXPORT const sd::LongType* getOpaqueNDArrayShapeInfo(OpaqueNDArray array);

SD_LIB_EXPORT void* getOpaqueNDArrayBuffer(OpaqueNDArray array);

SD_LIB_EXPORT void* getOpaqueNDArraySpecialBuffer(OpaqueNDArray array);

SD_LIB_EXPORT OpaqueNDArray createOpaqueNDArray(OpaqueDataBuffer *shapeInfo,
                                                OpaqueDataBuffer *buffer,
                                                OpaqueDataBuffer *specialBuffer,
                                                sd::LongType offset);


SD_LIB_EXPORT  sd::Pointer loadNpyFromHeader(sd::Pointer data);
SD_LIB_EXPORT void saveNpy(std::string fname, const OpaqueDataBuffer  *data, const unsigned int *shape, const unsigned int ndims,
                           std::string mode);

SD_LIB_EXPORT void inspectArray(sd::Pointer *extraPointers, sd::Pointer buffer, sd::LongType *shapeInfo, sd::Pointer specialBuffer,
                                sd::LongType *specialShapeInfo, sd::Pointer debugInfo);

SD_LIB_EXPORT OpaqueResultWrapper* executeFlatGraph(sd::Pointer* extraPointers, sd::Pointer flatBufferPointer);

SD_LIB_EXPORT OpaqueVariablesSet *executeStoredGraph(sd::Pointer *extraPointers,
                                                     sd::LongType graphId,
                                                     sd::Pointer *inputBuffers,
                                                     sd::Pointer *inputShapes,
                                                     int *inputIndices, int numInputs);
SD_LIB_EXPORT sd::LongType const *getPrimaryShapeInfo(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getPrimaryOffsets(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getSpecialShapeInfo(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getSpecialOffsets(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType getNumberOfTads(OpaqueTadPack *pack);
SD_LIB_EXPORT int getShapeInfoLength(OpaqueTadPack *pack);

/**
 * Get the stack trace for a TadPack as a string.
 * Returns the allocation stack trace if functrace is enabled, empty string otherwise.
 * This is useful for debugging TAD cache lifecycle issues.
 *
 * @param pack The TadPack to get the stack trace from
 * @return C-string containing the formatted stack trace (caller must NOT free this)
 */
SD_LIB_EXPORT const char* getTadPackStackTrace(OpaqueTadPack *pack);

SD_LIB_EXPORT OpaqueTadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength);
SD_LIB_EXPORT OpaqueConstantDataBuffer constantBufferLong(sd::DataType dtype, sd::LongType  *data, int length);
SD_LIB_EXPORT OpaqueConstantDataBuffer constantBufferDouble(sd::DataType dtype, double  *data, int length);
SD_LIB_EXPORT OpaqueConstantDataBuffer constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor);

SD_LIB_EXPORT const char *getDeviceName(int device);

SD_LIB_EXPORT void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost, OpaqueNDArray z, void *extraArguments);

SD_LIB_EXPORT void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost, OpaqueNDArray x, OpaqueNDArray z, void *extraArguments);

SD_LIB_EXPORT void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost,
                               OpaqueNDArray x,
                               OpaqueNDArray y, OpaqueNDArray z, void *extraArguments);

SD_LIB_EXPORT sd::LongType const *getShape(OpaqueShapeList *list, sd::LongType i);

SD_LIB_EXPORT OpaqueShapeList *calculateOutputShapes2(sd::Pointer *extraPointers, sd::LongType hash, OpaqueContext *context);
SD_LIB_EXPORT sd::LongType getShapeListSize(OpaqueShapeList *list);

SD_LIB_EXPORT void dbPrintAllocationTrace(OpaqueDataBuffer *db) ;
SD_LIB_EXPORT int numIntermediateResults(OpaqueContext *contextPointer) ;
SD_LIB_EXPORT sd::LongType dbBufferLength(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void toggleOpTrace(bool opTrace) ;
SD_LIB_EXPORT void purgeOpTrace() ;
SD_LIB_EXPORT void printOpTrace() ;
SD_LIB_EXPORT void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) ;
SD_LIB_EXPORT int contextNumInputs(void *contextPointer) ;
SD_LIB_EXPORT int contextNumOutputs(void *contextPointer) ;
SD_LIB_EXPORT int numInputs(void *execTrace) ;
SD_LIB_EXPORT int numOutputs(void *execTrace) ;
SD_LIB_EXPORT int getDeviceId(sd::Pointer ptrToDeviceId) ;
SD_LIB_EXPORT int getDeviceBlockThreshold(int deviceId) ;
SD_LIB_EXPORT int getDeviceSharedThreshold(int deviceId) ;
SD_LIB_EXPORT void printDeviceBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) ;
SD_LIB_EXPORT void printDeviceBuffer(OpaqueDataBuffer *buffer) ;
SD_LIB_EXPORT void execPairwiseTransform(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams) ;
SD_LIB_EXPORT void execPairwiseTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execSummaryStatsScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, bool biasCorrected) ;
SD_LIB_EXPORT void execSummaryStatsTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z,
                                       OpaqueNDArray dimension, bool biasCorrected);
SD_LIB_EXPORT void execBroadcastBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execScalarBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams);
////////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT void execScalarBoolTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension);
SD_LIB_EXPORT void execScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams);
SD_LIB_EXPORT void execScalarTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension);
SD_LIB_EXPORT void execBroadcast(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execReduceFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execReduce3All(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension, void *extraParams);


SD_LIB_EXPORT void execReduceLong(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execReduceBool2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execReduceBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execIndexReduce(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execReduceFloat2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) ;
SD_LIB_EXPORT void execIndexReduceScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execTransformSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execTransformAny(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execTransformStrict(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void execTransformFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) ;
SD_LIB_EXPORT void checkP2P() ;
SD_LIB_EXPORT void enableP2P(bool enable) ;
SD_LIB_EXPORT bool isP2PAvailable() ;
SD_LIB_EXPORT void initializeDevicesAndFunctions() ;
SD_LIB_EXPORT void initializeFunctions(sd::Pointer *functions) ;

/**
 * Initialize the shape cache eagerly during early JVM startup.
 * This prevents race conditions during static initialization when multiple threads
 * try to create NDArrays concurrently before the shape cache is fully initialized.
 *
 * This should be called from Nd4j initialization before any class loading
 * that might create NDArrays (like DifferentialFunctionClassHolder).
 *
 * Safe to call multiple times - subsequent calls are no-ops.
 */
SD_LIB_EXPORT void initializeShapeCache() ;

/**
 * Initialize the TAD (Tensor-Along-Dimension) cache early to prevent race conditions.
 *
 * This forces initialization of DirectTadTrie in a controlled, single-threaded context.
 * This prevents race conditions during static initialization when multiple threads
 * try to create TAD packs concurrently before the TAD cache is fully initialized.
 *
 * This should be called from Nd4j initialization before any class loading
 * that might create TAD operations.
 *
 * Safe to call multiple times - subsequent calls are no-ops.
 */
SD_LIB_EXPORT void initializeTadCache() ;

SD_LIB_EXPORT sd::Pointer mallocHost(sd::LongType memorySize, int flags) ;
SD_LIB_EXPORT sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int flags) ;
SD_LIB_EXPORT int freeHost(sd::Pointer pointer) ;
SD_LIB_EXPORT int freeDevice(sd::Pointer pointer, int deviceId) ;
SD_LIB_EXPORT sd::Pointer createContext() ;
SD_LIB_EXPORT sd::Pointer createStream() ;
SD_LIB_EXPORT sd::Pointer createEvent() ;
SD_LIB_EXPORT int registerEvent(sd::Pointer event, sd::Pointer stream) ;
SD_LIB_EXPORT int setDevice(int deviceId) ;
SD_LIB_EXPORT sd::LongType getDeviceFreeMemoryDefault() ;
SD_LIB_EXPORT sd::LongType getDeviceFreeMemory(int device) ;
SD_LIB_EXPORT sd::LongType getDeviceTotalMemory(int device) ;
SD_LIB_EXPORT int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) ;
SD_LIB_EXPORT int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) ;
SD_LIB_EXPORT int memsetSync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) ;
SD_LIB_EXPORT int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) ;
SD_LIB_EXPORT int destroyEvent(sd::Pointer event) ;
SD_LIB_EXPORT int streamSynchronize(sd::Pointer stream) ;
SD_LIB_EXPORT int eventSynchronize(sd::Pointer event) ;
SD_LIB_EXPORT int getAvailableDevices() ;
SD_LIB_EXPORT void enableDebugMode(bool reallyEnable) ;
SD_LIB_EXPORT void setGridLimit(int gridSize) ;
SD_LIB_EXPORT int ompGetMaxThreads() ;
SD_LIB_EXPORT int ompGetNumThreads() ;
SD_LIB_EXPORT void setOmpNumThreads(int threads) ;
/**
 * Sets the number of threads used by OpenBLAS for BLAS operations.
 * This is separate from OMP threads and specifically controls OpenBLAS's internal threading.
 * Default should be 1 to prevent TLS corruption crashes in multi-threaded Java applications.
 * @param threads number of threads for OpenBLAS to use
 */
SD_LIB_EXPORT void setOpenBlasThreads(int threads) ;
SD_LIB_EXPORT void enableVerboseMode(bool reallyEnable) ;
SD_LIB_EXPORT int getDeviceMajor(int device) ;
SD_LIB_EXPORT int getDeviceMinor(int device) ;
SD_LIB_EXPORT int getShapeInfoLength(OpaqueTadPack *pack) ;
SD_LIB_EXPORT int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) ;
SD_LIB_EXPORT sd::Pointer getConstantSpace() ;
SD_LIB_EXPORT bool isExperimentalEnabled() ;
SD_LIB_EXPORT void setOmpMinThreads(int threads) ;
SD_LIB_EXPORT int getDevice() ;
SD_LIB_EXPORT void setElementThreshold(int num) ;
SD_LIB_EXPORT void setTADThreshold(int num) ;
SD_LIB_EXPORT void execReduceSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray  x,
                                  void *extraParams,OpaqueNDArray  z);
SD_LIB_EXPORT void execReduceSame2(sd::Pointer *extraPointers, int opNum,
                                   OpaqueNDArray x,void *extraParams,
                                   OpaqueNDArray z, OpaqueNDArray  dimension) ;
SD_LIB_EXPORT void execReduce3(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) ;
SD_LIB_EXPORT void execReduce3Scalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z);
SD_LIB_EXPORT void execReduce3Tad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension);
SD_LIB_EXPORT sd::Pointer initRandom(sd::Pointer *extraPointers, long seed, long bufferSize, sd::Pointer ptrToBuffer) ;
SD_LIB_EXPORT void destroyRandom(sd::Pointer ptrBuffer) ;
SD_LIB_EXPORT void refreshBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) ;
SD_LIB_EXPORT void reSeedBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) ;
SD_LIB_EXPORT int lengthForShapeBufferPointer(sd::Pointer buffer) ;
SD_LIB_EXPORT sd::Pointer pointerForAddress(sd::LongType address) ;
SD_LIB_EXPORT void prescanArrayRecursive(sd::Pointer *extras, int *dZ, int *dX, int numElements, int level) ;
SD_LIB_EXPORT void deleteNDArray(OpaqueNDArray array) ;
SD_LIB_EXPORT bool checkOpaqueNDArrayElementsNull(OpaqueNDArrayArr elements,int numElements);
SD_LIB_EXPORT sd::LongType getOpaqueNDArrayOffset(OpaqueNDArray array) ;
SD_LIB_EXPORT void* getOpaqueNDArrayBuffer(OpaqueNDArray array) ;
SD_LIB_EXPORT void* getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) ;
SD_LIB_EXPORT sd::LongType getShapeInfoLength(OpaqueNDArray array) ;
SD_LIB_EXPORT sd::LongType getOpaqueNDArrayLength(OpaqueNDArray array) ;
SD_LIB_EXPORT void sort(sd::Pointer *extraPointers, OpaqueNDArray x, bool descending) ;
SD_LIB_EXPORT void sortTad(sd::Pointer *extraPointers, OpaqueNDArray  x,
                           sd::LongType *dimension, sd::LongType dimensionLength,
                           sd::LongType *tadShapeInfo,  sd::LongType *tadOffsets, bool descending);
SD_LIB_EXPORT void sortByValue(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, bool descending);

SD_LIB_EXPORT void execReduceLong2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray  x,
                                   void *extraParams,
                                   OpaqueNDArray z, OpaqueNDArray dimension);

SD_LIB_EXPORT void sortTadByKey(sd::Pointer *extraPointers,
                                OpaqueNDArray x,
                                OpaqueNDArray y,
                                OpaqueNDArray dimension,
                                bool descending);

SD_LIB_EXPORT void sortTadByValue(sd::Pointer *extraPointers,
                                  OpaqueNDArray x,
                                  OpaqueNDArray y,
                                  OpaqueNDArray dimension,
                                  bool descending);
SD_LIB_EXPORT void munmapFile(sd::Pointer *extraPointers, sd::LongType *ptrMap, sd::LongType length) ;
SD_LIB_EXPORT sd::LongType* mmapFile(sd::Pointer* extraPointers, const char* fileName, sd::LongType length);

SD_LIB_EXPORT sd::LongType getResultWrapperSize(OpaqueResultWrapper *ptr) ;
SD_LIB_EXPORT sd::Pointer getResultWrapperPointer(OpaqueResultWrapper *ptr) ;
SD_LIB_EXPORT sd::LongType getShapeListSize(OpaqueShapeList *list) ;
SD_LIB_EXPORT sd::Status execCustomOp2(sd::Pointer *extraPointers, sd::LongType hash, OpaqueContext *opContext) ;
SD_LIB_EXPORT sd::Status registerGraph(sd::Pointer *extraPointers, sd::LongType graphId, sd::Pointer flatBufferPointer) ;
SD_LIB_EXPORT sd::LongType getVariablesSetSize(OpaqueVariablesSet *set) ;
SD_LIB_EXPORT sd::Status getVariablesSetStatus(OpaqueVariablesSet *set) ;
SD_LIB_EXPORT sd::LongType const *getVariableShape(OpaqueVariable *variable) ;
SD_LIB_EXPORT OpaqueVariable *getVariable(OpaqueVariablesSet *set, sd::LongType i);
SD_LIB_EXPORT int getVariableId(OpaqueVariable *variable) ;
SD_LIB_EXPORT int getVariableIndex(OpaqueVariable *variable) ;
SD_LIB_EXPORT void* getVariableBuffer(OpaqueVariable *variable) ;
SD_LIB_EXPORT const char*  getVariableName(OpaqueVariable *variable) ;
SD_LIB_EXPORT sd::Status unregisterGraph(sd::Pointer *extraPointers, sd::LongType graphId) ;
SD_LIB_EXPORT void deletePointerArray(sd::Pointer pointer) ;
SD_LIB_EXPORT void deleteCharArray(sd::Pointer pointer) ;
SD_LIB_EXPORT void deleteIntArray(sd::Pointer pointer) ;
SD_LIB_EXPORT void deleteLongArray(sd::Pointer pointer) ;
SD_LIB_EXPORT void deleteVariablesSet(OpaqueVariablesSet *pointer) ;
SD_LIB_EXPORT void deleteShapeList(sd::Pointer shapeList) ;
SD_LIB_EXPORT sd::Pointer getGraphState(sd::LongType id) ;
SD_LIB_EXPORT void deleteGraphState(sd::Pointer state) ;
SD_LIB_EXPORT void deleteResultWrapper(sd::Pointer ptr) ;
SD_LIB_EXPORT void convertTypes(sd::Pointer *extras, int srcType, sd::Pointer dX, sd::LongType N, int dstType, sd::Pointer dZ) ;
SD_LIB_EXPORT sd::Pointer createUtf8String(sd::Pointer *extraPointers, const char *string, int length) ;
SD_LIB_EXPORT sd::LongType getUtf8StringLength(sd::Pointer *extraPointers, sd::Pointer ptr) ;
SD_LIB_EXPORT void deleteUtf8String(sd::Pointer *extraPointers, sd::Pointer ptr) ;
SD_LIB_EXPORT void tryPointer(sd::Pointer extra, sd::Pointer p, int len) ;
SD_LIB_EXPORT void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) ;
SD_LIB_EXPORT void deleteConstantDataBuffer(OpaqueConstantDataBuffer *ptr) ;
SD_LIB_EXPORT void deleteTadPack(OpaqueTadPack *ptr) ;
SD_LIB_EXPORT bool isBlasVersionMatches(int major, int minor, int build) ;
SD_LIB_EXPORT sd::Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf) ;
SD_LIB_EXPORT sd::Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf) ;
SD_LIB_EXPORT sd::LongType getConstantDataBufferLength(OpaqueConstantDataBuffer dbf) ;
SD_LIB_EXPORT sd::LongType getConstantDataBufferSizeOf(OpaqueConstantDataBuffer dbf) ;
SD_LIB_EXPORT sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf) ;
SD_LIB_EXPORT sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf) ;

/**
 * Get the stack trace for a ConstantShapeBuffer as a string.
 * Returns the allocation stack trace if functrace is enabled, empty string otherwise.
 * This is useful for debugging shape buffer lifecycle issues.
 *
 * @param buffer The ConstantShapeBuffer to get the stack trace from
 * @return C-string containing the formatted stack trace (caller must NOT free this)
 */
SD_LIB_EXPORT const char* getConstantShapeBufferStackTrace(OpaqueConstantShapeBuffer buffer);

SD_LIB_EXPORT void markGraphContextInplace(OpaqueContext *ptr, bool reallyInplace) ;
SD_LIB_EXPORT OpaqueNDArray getOutputArrayNative(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT OpaqueNDArray getInputArrayNative(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT sd::LongType dataTypeNativeAt(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT bool bArgAtNative(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT sd::LongType iArgumentAtNative(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT sd::LongType numDNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT sd::LongType numBNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT sd::LongType numOutputsNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT sd::LongType numInputsNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT double tArgumentNative(OpaqueContext* ptr, int idx) ;
SD_LIB_EXPORT sd::LongType numTArgumentsNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT sd::LongType numIArgumentsNative(OpaqueContext* ptr) ;
SD_LIB_EXPORT void setGraphContextOutputArray(OpaqueContext* ptr, int index,OpaqueNDArray arr) ;
SD_LIB_EXPORT void setGraphContextInputArray(OpaqueContext* ptr,int index,OpaqueNDArray arr) ;
SD_LIB_EXPORT void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays, OpaqueNDArrayArr arr) ;
SD_LIB_EXPORT void setGraphContextInputArraysArr(OpaqueContext* ptr, int numArrays, OpaqueNDArrayArr arr) ;
SD_LIB_EXPORT void setGraphContextTArguments(OpaqueContext *ptr, double *arguments, int numberOfArguments) ;
SD_LIB_EXPORT void setGraphContextIArguments(OpaqueContext *ptr, sd::LongType *arguments, int numberOfArguments) ;
SD_LIB_EXPORT void setGraphContextBArguments(OpaqueContext *ptr, bool *arguments, int numberOfArguments) ;
SD_LIB_EXPORT void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) ;
SD_LIB_EXPORT void deleteGraphContext(OpaqueContext *ptr) ;
SD_LIB_EXPORT sd::LongType getRandomGeneratorRootState(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT sd::LongType getRandomGeneratorNodeState(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT void setRandomGeneratorStates(OpaqueRandomGenerator ptr, sd::LongType rootSeed, sd::LongType nodeSeed) ;
SD_LIB_EXPORT float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, sd::LongType index) ;
SD_LIB_EXPORT double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, sd::LongType index) ;
SD_LIB_EXPORT int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, sd::LongType index) ;
SD_LIB_EXPORT sd::LongType getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, sd::LongType index) ;
SD_LIB_EXPORT int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT sd::LongType getRandomGeneratorNextLong(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT void deleteRandomGenerator(OpaqueRandomGenerator ptr) ;
SD_LIB_EXPORT sd::LongType getCachedMemory(int deviceId) ;
SD_LIB_EXPORT sd::Pointer lcScalarPointer(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcReductionPointer(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcAllocationPointer(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcExecutionStream(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcCopyStream(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcBlasHandle(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT sd::Pointer lcSolverHandle(OpaqueLaunchContext lc) ;
SD_LIB_EXPORT void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) ;
SD_LIB_EXPORT void ctxPurge(OpaqueContext *ptr) ;
SD_LIB_EXPORT int binaryLevel() ;
SD_LIB_EXPORT int optimalLevel() ;
SD_LIB_EXPORT bool isMinimalRequirementsMet() ;
SD_LIB_EXPORT bool isOptimalRequirementsMet() ;
SD_LIB_EXPORT void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) ;
SD_LIB_EXPORT void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) ;
SD_LIB_EXPORT sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT sd::Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes) ;
SD_LIB_EXPORT void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes) ;
SD_LIB_EXPORT void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, sd::LongType elements) ;
SD_LIB_EXPORT int dbUseCount(OpaqueDataBuffer* dataBuffer) ;
SD_LIB_EXPORT void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbTickHostRead(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbExpand(OpaqueDataBuffer *dataBuffer, sd::LongType elements) ;
SD_LIB_EXPORT void dbClose(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT int dbDeviceId(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) ;
SD_LIB_EXPORT int dbLocality(OpaqueDataBuffer *dataBuffer) ;
SD_LIB_EXPORT OpaqueDataBuffer* dbCreateView(OpaqueDataBuffer* dataBuffer, sd::LongType length) ;
SD_LIB_EXPORT OpaqueDataBuffer* dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) ;
SD_LIB_EXPORT OpaqueDataBuffer* dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) ;
SD_LIB_EXPORT void setShapeBuffer(sd::LongType *inputShapeData,sd::DataType dt,sd::LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty,bool isView) ;
SD_LIB_EXPORT OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(sd::LongType *shapeInfo);
SD_LIB_EXPORT OpaqueConstantShapeBuffer shapeBuffer(int rank, sd::LongType* shape, sd::LongType* strides,
                                                    sd::DataType dtype, char order, sd::LongType ews, bool empty);
SD_LIB_EXPORT OpaqueConstantShapeBuffer shapeBufferEx(int rank, sd::LongType* shape, sd::LongType* strides,
                                                      sd::DataType dtype, char order, sd::LongType ews,
                                                      sd::LongType extras);

SD_LIB_EXPORT OpaqueDataBuffer *allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth);


SD_LIB_EXPORT  OpaqueLaunchContext defaultLaunchContext();

SD_LIB_EXPORT sd::Pointer lcScalarPointer(OpaqueLaunchContext* lc);

SD_LIB_EXPORT sd::Pointer lcReductionPointer(OpaqueLaunchContext* lc);

SD_LIB_EXPORT sd::Pointer lcAllocationPointer(OpaqueLaunchContext* lc);

SD_LIB_EXPORT sd::Pointer lcExecutionStream(OpaqueLaunchContext* lc);

SD_LIB_EXPORT sd::Pointer lcCopyStream(OpaqueLaunchContext* lc);

SD_LIB_EXPORT sd::Pointer lcBlasHandle(OpaqueLaunchContext* lc);
SD_LIB_EXPORT  long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize);
SD_LIB_EXPORT  long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer);
SD_LIB_EXPORT sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray);
SD_LIB_EXPORT  sd::Pointer numpyHeaderForNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize,
                                              sd::LongType* headerSize) ;
SD_LIB_EXPORT  sd::Pointer numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize);
SD_LIB_EXPORT  sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray);
SD_LIB_EXPORT  sd::Pointer dataPointForNumpyHeader(sd::Pointer npyArray);
SD_LIB_EXPORT  sd::Pointer dataPointForNumpyStruct(sd::Pointer npyArrayStruct);
SD_LIB_EXPORT  sd::Pointer dataPointForNumpy(sd::Pointer npyArray);
SD_LIB_EXPORT  sd::Pointer numpyFromFile(std::string path);
SD_LIB_EXPORT  void *mapFromNpzFile(std::string path);
SD_LIB_EXPORT  int getNumNpyArraysInMap(void *map);
SD_LIB_EXPORT  const char *getNpyArrayNameFromMap(void *map, int index, char *nameBuffer);
SD_LIB_EXPORT  void *getNpyArrayFromMap(void *map, int index);
SD_LIB_EXPORT  int dataTypeFromNpyHeader(void *header);
SD_LIB_EXPORT  void *getNpyArrayData(void *npArray);
SD_LIB_EXPORT  int getNpyArrayRank(void *npArray);
SD_LIB_EXPORT  sd::LongType *getNpyArrayShape(void *npArray);
SD_LIB_EXPORT  char getNpyArrayOrder(void *npArray);
SD_LIB_EXPORT  int getNpyArrayElemSize(void *npArray);
SD_LIB_EXPORT  void deleteNPArrayStruct(void *npArray);
SD_LIB_EXPORT long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize);
SD_LIB_EXPORT long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer);
SD_LIB_EXPORT  void deleteNPArrayMap(void *map);
SD_LIB_EXPORT  int elementSizeForNpyArray(sd::Pointer npyArray);
SD_LIB_EXPORT  int elementSizeForNpyArrayHeader(sd::Pointer npyArray);
SD_LIB_EXPORT  void releaseNumpy(sd::Pointer npyArray);
SD_LIB_EXPORT sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray) ;
SD_LIB_EXPORT int dataTypeFromNpyHeader(void* header);

SD_LIB_EXPORT std::vector<OpaqueDataBuffer *> intermediateResults(OpaqueContext *contextPointer);
SD_LIB_EXPORT std::vector<const sd::LongType *> intermediateResultsShapeInfo(OpaqueContext *contextPointer);
SD_LIB_EXPORT void setIntermediateResult(OpaqueContext* contextPointer, int index, OpaqueDataBuffer* buffer,
                                         OpaqueDataBuffer* shapeInfo, sd::LongType dataOffset);
SD_LIB_EXPORT void pushIntermediateResult(OpaqueContext* contextPointer, OpaqueDataBuffer* buffer,
                                          OpaqueDataBuffer* shapeInfo, sd::LongType offset);
SD_LIB_EXPORT OpaqueDataBuffer  * intermediateResultDataAt(int index, OpaqueContext *contextPointer);
SD_LIB_EXPORT const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer);
SD_LIB_EXPORT const char *lastErrorMessage();
SD_LIB_EXPORT int lastErrorCode();
SD_LIB_EXPORT void triggerLeakCheck();
SD_LIB_EXPORT void enableNDArrayTracking();
SD_LIB_EXPORT void disableNDArrayTracking();
SD_LIB_EXPORT void enableDataBufferTracking();
SD_LIB_EXPORT void disableDataBufferTracking();
SD_LIB_EXPORT void enableTADCacheTracking();
SD_LIB_EXPORT void disableTADCacheTracking();
SD_LIB_EXPORT void enableShapeCacheTracking();
SD_LIB_EXPORT void disableShapeCacheTracking();
SD_LIB_EXPORT void enableOpContextTracking();
SD_LIB_EXPORT void disableOpContextTracking();

/**
 * Set the current operation context for allocation tracking.
 * All allocations (NDArray, DataBuffer) made while an op context is set
 * will be tagged with the operation name for per-op leak analysis.
 * @param opName The name of the operation (e.g., "matmul", "add", "conv2d")
 *               Pass nullptr to clear the context.
 */
SD_LIB_EXPORT void setLifecycleOpContext(const char* opName);

/**
 * Clear the current operation context for allocation tracking.
 * Subsequent allocations will be tagged as "(unknown)" in leak reports.
 */
SD_LIB_EXPORT void clearLifecycleOpContext();

/**
 * Get the current operation context for allocation tracking.
 * @return The current operation name, or empty string if none is set
 */
SD_LIB_EXPORT const char* getLifecycleOpContext();

/**
 * Enable operation execution logging for crash detection.
 * When enabled (and SD_GCC_FUNCTRACE is defined), all operation executions
 * are logged to a file with full unified C++/Java stack traces.
 * The log file survives crashes and can be used for post-mortem debugging.
 *
 * Log files are located at: /tmp/nd4j_op_execution_<PID>.log
 * (or $SD_OP_LOG_DIR if set)
 *
 * NOTE: Only available when built with -Dlibnd4j.calltrace=ON
 */
SD_LIB_EXPORT void enableOpExecutionLogging();

/**
 * Disable operation execution logging.
 * No-op if SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void disableOpExecutionLogging();

/**
 * Check if operation execution logging is currently enabled.
 * @return true if logging is enabled, false otherwise
 */
SD_LIB_EXPORT bool isOpExecutionLoggingEnabled();

/**
 * Get the current operation execution log file path.
 * Returns empty string if logging is not enabled or not available.
 *
 * @return C-string containing the log file path (caller must NOT free this)
 */
SD_LIB_EXPORT const char* getOpExecutionLogPath();

/**
 * Get the current operation execution log contents as a string.
 * Useful for retrieving recent execution history for debugging.
 *
 * @param maxBytes Maximum bytes to read (0 = read entire file)
 * @param fromEnd If true, read from end of file (most recent entries)
 * @return C-string containing the log contents (caller must NOT free this)
 */
SD_LIB_EXPORT const char* getOpExecutionLogContents(size_t maxBytes, bool fromEnd);

/**
 * Force a flush of the operation execution log to disk.
 * The logger flushes after each operation by default, but this
 * can be called manually for explicit checkpointing.
 */
SD_LIB_EXPORT void dumpOpExecutionLog();

/**
 * Manually dump current state to the operation execution log.
 * Useful for checkpointing at specific points in code.
 *
 * @param message Optional message to include in the dump
 */
SD_LIB_EXPORT void dumpOpExecutionState(const char* message);

// ═══════════════════════════════════════════════════════════════
// Allocation Logging API (SD_GCC_FUNCTRACE only)
// Similar to OpExecutionLogging, but focuses on tracking NDArray
// and OpContext allocations for understanding memory growth patterns
// ═══════════════════════════════════════════════════════════════

/**
 * Get the current allocation log file path.
 * Allocation logging is always active in functrace builds (SD_GCC_FUNCTRACE).
 * Returns empty string if functrace is not enabled.
 *
 * Log file location: /tmp/nd4j_allocations_<PID>.log (configurable via SD_ALLOCATION_LOG_DIR)
 *
 * @return C-string containing the log file path (caller must NOT free this)
 */
SD_LIB_EXPORT const char* getAllocationLogPath();

/**
 * Set the current allocation context (operation name) for lifecycle tracking.
 * This is used to associate memory allocations with the operation that triggered them.
 * The context is thread-local, so each thread can have its own context.
 *
 * Call this before creating ops/arrays to tag allocations with the op name.
 * Call clearAllocationContext() when done.
 *
 * @param opName The operation name to associate with allocations (e.g., "Sum", "Mean", "Concat")
 */
SD_LIB_EXPORT void setAllocationContext(const char* opName);

/**
 * Clear the current allocation context for this thread.
 * Call this after op creation/execution to stop tagging allocations.
 */
SD_LIB_EXPORT void clearAllocationContext();

/**
 * Update the Java stack trace for an existing NDArray allocation record.
 * This is called from Java after creating an OpaqueNDArray to provide the full Java stack trace
 * captured before the JNI boundary. This gives much better context than capturing the stack
 * trace from within native code.
 *
 * @param array The OpaqueNDArray whose allocation record should be updated
 * @param javaStackTrace The full Java stack trace as a string
 */
SD_LIB_EXPORT void updateAllocationJavaStackTrace(OpaqueNDArray array, const char* javaStackTrace);

// ===============================
// Java-side Lifecycle Recording API
// These functions are called from Java to record allocation/deallocation events
// for lifecycle tracking. They delegate to the corresponding C++ lifecycle trackers.
// ===============================

/**
 * Record an NDArray allocation from Java side.
 * @param array The OpaqueNDArray that was allocated
 * @param size Size in bytes
 * @param dataType Data type of the array
 * @param isView Whether this is a view of another array
 */
SD_LIB_EXPORT void recordJavaNDArrayAllocation(OpaqueNDArray array, long size, int dataType, bool isView);

/**
 * Record an NDArray deallocation from Java side.
 * @param array The OpaqueNDArray being deallocated
 */
SD_LIB_EXPORT void recordJavaNDArrayDeallocation(OpaqueNDArray array);

/**
 * Record a DataBuffer allocation from Java side.
 * @param buffer The OpaqueDataBuffer that was allocated
 * @param size Size in bytes
 * @param dataType Data type of the buffer
 * @param isWorkspace Whether this buffer is from a workspace
 */
SD_LIB_EXPORT void recordJavaDataBufferAllocation(OpaqueDataBuffer *buffer, long size, int dataType, bool isWorkspace);

/**
 * Record a DataBuffer deallocation from Java side.
 * @param buffer The OpaqueDataBuffer being deallocated
 */
SD_LIB_EXPORT void recordJavaDataBufferDeallocation(OpaqueDataBuffer *buffer);

/**
 * Record an OpContext allocation from Java side.
 * @param context The OpaqueContext that was allocated
 * @param nodeId Node ID for the context
 * @param fastpathInSize Size of fastpath input arrays
 * @param fastpathOutSize Size of fastpath output arrays
 * @param intermediateResultsSize Size of intermediate results
 * @param handlesSize Size of handles
 * @param hasWorkspace Whether context has workspace
 * @param isFastPath Whether this is a fastpath context
 */
SD_LIB_EXPORT void recordJavaOpContextAllocation(OpaqueContext *context, int nodeId, long fastpathInSize, long fastpathOutSize, long intermediateResultsSize, long handlesSize, bool hasWorkspace, bool isFastPath);

/**
 * Record an OpContext deallocation from Java side.
 * @param context The OpaqueContext being deallocated
 */
SD_LIB_EXPORT void recordJavaOpContextDeallocation(OpaqueContext *context);

/**
 * Clear all cached TAD packs to prevent memory leaks during testing.
 * This frees all TadPack objects stored in the TAD cache.
 * NOTE: Will return early without action if setTADCacheShutdownInProgress(true) was called.
 */
SD_LIB_EXPORT void clearTADCache();

/**
 * Marks that shutdown is in progress.
 * CRITICAL: Call this early in JVM shutdown (e.g., from a shutdown hook)
 * to prevent SIGSEGV crashes during cache cleanup.
 *
 * During JVM/static destruction, memory allocators may have been destroyed,
 * leaving corrupted pointers in cached data structures. Setting this flag
 * causes clearTADCache() and similar functions to skip tree traversal,
 * letting the OS safely reclaim memory at process exit instead.
 *
 * @param inProgress true to mark shutdown in progress, false otherwise
 */
SD_LIB_EXPORT void setTADCacheShutdownInProgress(bool inProgress);

/**
 * Check if TAD cache shutdown is in progress.
 * @return true if shutdown is marked as in progress
 */
SD_LIB_EXPORT bool isTADCacheShutdownInProgress();

/**
 * Clears all cached shape buffers.
 * This frees all ConstantShapeBuffer objects stored in the shape cache.
 * Called during application shutdown to prevent memory leaks.
 */
SD_LIB_EXPORT void clearShapeCache();

/**
 * Get the total number of cached shape buffer entries.
 * @return Total number of cached shape buffers across all stripes
 */
SD_LIB_EXPORT sd::LongType getShapeCachedEntries();

/**
 * Get the total memory used by cached shape buffers in bytes.
 * @return Total memory used in bytes
 */
SD_LIB_EXPORT sd::LongType getShapeCachedBytes();

/**
 * Get the peak number of shape entries that were cached simultaneously.
 * @return Peak number of cached shape buffers
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedEntries();

/**
 * Get the peak memory usage by cached shape buffers in bytes.
 * @return Peak memory usage in bytes
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedBytes();

/**
 * Get the total number of cached TAD pack entries.
 * @return Total number of cached TAD packs across all stripes
 */
SD_LIB_EXPORT sd::LongType getTADCachedEntries();

/**
 * Get the total memory used by cached TAD packs in bytes.
 * This includes both shape_info and offset buffer sizes.
 * @return Total memory used in bytes
 */
SD_LIB_EXPORT sd::LongType getTADCachedBytes();

/**
 * Get the peak number of TAD pack entries that were cached simultaneously.
 * @return Peak number of cached TAD packs
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedEntries();

/**
 * Get the peak memory usage by cached TAD packs in bytes.
 * @return Peak memory usage in bytes
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedBytes();

/**
 * Get a string representation of the shape cache for debugging.
 * The returned string must be freed by the caller using freeString().
 *
 * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
 * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
 * @return String representation of the shape cache
 */
SD_LIB_EXPORT const char* getShapeCacheString(int maxDepth, int maxEntries);

/**
 * Get a string representation of the TAD cache for debugging.
 * The returned string must be freed by the caller using freeString().
 *
 * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
 * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
 * @return String representation of the TAD cache
 */
SD_LIB_EXPORT const char* getTADCacheString(int maxDepth, int maxEntries);

/**
 * Free a string returned by native code.
 * @param ptr String pointer to free
 */
SD_LIB_EXPORT void freeString(const char* ptr);

/**
 * Checks operation counter and automatically clears TAD/Shape caches periodically.
 * Called internally from operation execution entry points to prevent cache accumulation
 * during testing. Configurable via SD_CACHE_CLEANUP_INTERVAL and SD_AUTO_CACHE_CLEANUP
 * environment variables.
 *
 * Note: This function is available regardless of SD_GCC_FUNCTRACE build flag.
 * When SD_GCC_FUNCTRACE is disabled, it becomes a no-op stub.
 */
SD_LIB_EXPORT void checkAndCleanupCaches();

// Lifecycle tracking API
// NOTE: These functions are always declared but only fully functional with SD_GCC_FUNCTRACE.
// When SD_GCC_FUNCTRACE is not defined, stub implementations provide no-op behavior.

/**
 * Initializes lifecycle crash handlers to capture crash dumps.
 *
 * This must be called after JVM is fully initialized to ensure
 * the crash handler properly chains to JVM's hs_err generation.
 *
 * If called during library load (too early), the crash handler will capture
 * SIG_DFL instead of JVM's crash handler, preventing hs_err file generation.
 *
 * Safe to call multiple times - only initializes once.
 *
 * Recommended: Call from Java after NativeOpsHolder initialization.
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void initializeLifecycleCrashHandlers();

/**
 * Returns NDArray lifecycle statistics as a JSON string.
 * The returned string must be freed by the caller using freeString().
 *
 * JSON format:
 * {
 *   "total_allocations": <count>,
 *   "total_deallocations": <count>,
 *   "current_live": <count>,
 *   "peak_live": <count>,
 *   "current_bytes": <bytes>,
 *   "peak_bytes": <bytes>,
 *   "double_frees": <count>
 * }
 *
 * NOTE: Returns empty JSON "{}" when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT const char* getNDArrayLifecycleStats();

/**
 * Returns DataBuffer lifecycle statistics as a JSON string.
 * The returned string must be freed by the caller using freeString().
 *
 * JSON format:
 * {
 *   "primary": {
 *     "total_allocations": <count>,
 *     "total_deallocations": <count>,
 *     "current_live": <count>,
 *     "current_bytes": <bytes>,
 *     "peak_bytes": <bytes>
 *   },
 *   "special": {
 *     "total_allocations": <count>,
 *     "total_deallocations": <count>,
 *     "current_live": <count>,
 *     "current_bytes": <bytes>,
 *     "peak_bytes": <bytes>
 *   },
 *   "double_frees": <count>
 * }
 *
 * NOTE: Returns empty JSON "{}" when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT const char* getDataBufferLifecycleStats();

/**
 * Generates a flamegraph SVG file for NDArray allocations.
 * @param outputPath Path where the SVG file should be written
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateNDArrayAllocationFlamegraph(const char* outputPath);

/**
 * Generates a flamegraph SVG file for NDArray deallocations.
 * @param outputPath Path where the SVG file should be written
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateNDArrayDeallocationFlamegraph(const char* outputPath);

/**
 * Generates a flamegraph SVG file for DataBuffer allocations.
 * @param outputPath Path where the SVG file should be written
 * @param bufferType 0 = PRIMARY (host), 1 = SPECIAL (device)
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType);

/**
 * Generates a flamegraph SVG file for DataBuffer deallocations.
 * @param outputPath Path where the SVG file should be written
 * @param bufferType 0 = PRIMARY (host), 1 = SPECIAL (device)
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType);

/**
 * Generates a detailed leak report showing all currently live allocations.
 * @param outputPath Path where the report file should be written
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateLifecycleLeakReport(const char* outputPath);

/**
 * Generates a comprehensive leak source analysis report combining data from ALL lifecycle trackers.
 *
 * This function analyzes undeleted allocations across all 5 lifecycle trackers:
 * - NDArrayLifecycleTracker
 * - DataBufferLifecycleTracker
 * - TADCacheLifecycleTracker
 * - ShapeCacheLifecycleTracker
 * - OpContextLifecycleTracker
 *
 * For each allocation source (Java method or C++ function), the report shows:
 * - Total number of undeleted allocations
 * - Breakdown by object type (NDArray, DataBuffer, TAD, Shape, OpContext)
 * - Total bytes leaked
 * - Example stack traces (both Java and C++)
 *
 * Results are sorted by total leak count, making it easy to identify the top leak sources.
 *
 * @param outputDir Directory where report files should be written (e.g., "./leak_reports")
 *                  If NULL or empty, uses current directory.
 *
 * Output files generated:
 * - comprehensive_leak_report.txt - Detailed report with top 50 leak sources
 * - Console output shows top 20 leak sources summary
 *
 * Example usage from Java:
 *   Nd4j.getNativeOps().generateComprehensiveLeakAnalysis("./leak_reports");
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateComprehensiveLeakAnalysis(const char* outputDir);

/**
 * Generate temporal leak analysis report showing leak velocity over time windows.
 *
 * @param outputPath Path to output file
 * @param windowCount Number of time windows to analyze (default: 10)
 * @param windowDurationSec Duration of each window in seconds (default: 30.0)
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateNDArrayTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec);
SD_LIB_EXPORT void generateTADCacheTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec);

/**
 * Capture a snapshot of current leak state for differential analysis.
 *
 * @return Snapshot ID (use with generateSnapshotDiff)
 *
 * NOTE: Returns 0 when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT sd::LongType captureNDArrayLeakSnapshot();
SD_LIB_EXPORT sd::LongType captureTADCacheLeakSnapshot();

/**
 * Generate differential report comparing two snapshots.
 *
 * @param snapshot1 First snapshot ID
 * @param snapshot2 Second snapshot ID
 * @param outputPath Path to output file
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void generateNDArraySnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath);
SD_LIB_EXPORT void generateTADCacheSnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath);

/**
 * Clear all stored snapshots to free memory.
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void clearNDArraySnapshots();
SD_LIB_EXPORT void clearTADCacheSnapshots();

// ===============================
// DeallocatorService Lifecycle Tracking
// Records Java-side deallocation statistics from DeallocatorService
// to be merged with C++ lifecycle trackers in UnifiedMemoryReporter
// ===============================

/**
 * Records a snapshot of DeallocatorService statistics from Java.
 * Called by Java DeallocatorService to push its time-series tracking data.
 *
 * @param totalAllocations Total number of allocations tracked
 * @param totalDeallocations Total number of deallocations tracked
 * @param totalBytesAllocated Total bytes allocated
 * @param totalBytesDeallocated Total bytes deallocated
 * @param peakLiveCount Peak number of live objects observed
 * @param peakBytes Peak bytes in use observed
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void recordDeallocatorServiceSnapshot(
    sd::LongType totalAllocations, sd::LongType totalDeallocations,
    sd::LongType totalBytesAllocated, sd::LongType totalBytesDeallocated,
    sd::LongType peakLiveCount, sd::LongType peakBytes);

/**
 * Enables DeallocatorService lifecycle tracking on the C++ side.
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void enableDeallocatorServiceTracking();

/**
 * Disables DeallocatorService lifecycle tracking on the C++ side.
 *
 * NOTE: No-op when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT void disableDeallocatorServiceTracking();

/**
 * Checks if DeallocatorService tracking is enabled.
 *
 * @return true if tracking is enabled
 *
 * NOTE: Always returns false when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT bool isDeallocatorServiceTrackingEnabled();

/**
 * Gets the current live count from DeallocatorService tracker.
 *
 * @return current live count (allocations - deallocations)
 *
 * NOTE: Returns 0 when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceLiveCount();

/**
 * Gets the current bytes in use from DeallocatorService tracker.
 *
 * @return current bytes in use (allocated - deallocated)
 *
 * NOTE: Returns 0 when SD_GCC_FUNCTRACE is not defined.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceBytesInUse();

#endif // NATIVEOPS_H