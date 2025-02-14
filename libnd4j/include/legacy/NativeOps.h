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

#include <array/ArrayOptions.h>
#include <array/DataTypeUtils.h>
#include <array/ShapeList.h>
#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/InteropDataBuffer.h>
#include <array/TadPack.h>
#include <cnpy/cnpy.h>
#include <execinfo.h>
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
#include <unistd.h>
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
extern "C" {



//this is to ensure symbol is loaded and exported from this library instead when using LD_PRELOAD.
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_enter (void *this_fn,void *call_site);
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_exit  (void *this_fn,void *call_site);

}



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
SD_LIB_EXPORT void scatterUpdate(sd::Pointer *extraPointers, int opCode, OpaqueNDArray array, OpaqueNDArray indices, OpaqueNDArray updates, OpaqueNDArray axis);
SD_LIB_EXPORT sd::LongType const *getPrimaryShapeInfo(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getPrimaryOffsets(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getSpecialShapeInfo(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType const *getSpecialOffsets(OpaqueTadPack *pack);
SD_LIB_EXPORT sd::LongType getNumberOfTads(OpaqueTadPack *pack);
SD_LIB_EXPORT int getShapeInfoLength(OpaqueTadPack *pack);


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
SD_LIB_EXPORT void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) ;
SD_LIB_EXPORT void setGraphContextInputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) ;
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
SD_LIB_EXPORT sd::Pointer lcScalarPointer(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcReductionPointer(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcAllocationPointer(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcExecutionStream(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcCopyStream(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcBlasHandle(OpaqueLaunchContext *lc) ;
SD_LIB_EXPORT sd::Pointer lcSolverHandle(OpaqueLaunchContext *lc) ;
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



#endif // NATIVEOPS_H