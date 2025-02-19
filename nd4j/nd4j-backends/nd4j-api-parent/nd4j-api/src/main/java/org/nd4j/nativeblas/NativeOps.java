/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.nativeblas;

import org.bytedeco.javacpp.*;

import java.nio.IntBuffer;
import java.nio.LongBuffer;


/**
 * A common interface for misc operations
 * needed from c++
 *
 */
public interface NativeOps {


 OpaqueContext createGraphContext(int nodeId);

 void setGraphContextCudaContext(OpaqueContext ptr, Pointer stream, Pointer reductionPointer,
                                 Pointer allocationPointer);


 OpaqueRandomGenerator createRandomGenerator(long rootSeed,long nodeSeed);
 OpaqueRandomGenerator getGraphContextRandomGenerator(OpaqueContext ptr);

 void shuffle(PointerPointer extras,
              OpaqueNDArrayArr x,
              OpaqueNDArrayArr z,
              int N,
              OpaqueNDArray dimension,
              OpaqueNDArray shuffleMap);



 void pullRows(PointerPointer extraPointers,
               OpaqueNDArray x,
               OpaqueNDArray z,
               long n,
               OpaqueNDArray indexes,
              long dimension);


 PointerPointer listOpTraces();
 BytePointer opName(Pointer execTrace);
 BooleanPointer  bArgs(Pointer execTrace);
 PointerPointer sArgs(Pointer execTrace);
 DoublePointer tArgs(Pointer execTrace);
 LongPointer iArgs(Pointer execTrace);
 IntPointer dArgs(Pointer execTrace);
 PointerPointer inputShapeBuffers(Pointer execTrace);
 PointerPointer outputShapeBuffers(Pointer execTrace);

 LongPointer getOpaqueNDArrayShapeInfo(OpaqueNDArray array);




 OpaqueNDArray create(OpaqueDataBuffer shapeInfo,
                      OpaqueDataBuffer buffer,
                      OpaqueDataBuffer specialBuffer,
                      long offset);

 void saveNpy(String fname,  OpaqueDataBuffer  data, IntPointer shape, int ndims,
              String mode);


 OpaqueResultWrapper executeFlatGraph(PointerPointer extraPointers, Pointer flatBufferPointer);

 OpaqueVariablesSet executeStoredGraph(PointerPointer extraPointers,
                                        long graphId,
                                        PointerPointer inputBuffers,
                                        PointerPointer inputShapes,
                                       IntPointer inputIndices, int numInputs);
 LongPointer getShape(OpaqueShapeList list, long i);
 boolean checkOpaqueNDArrayElementsNull(OpaqueNDArrayArr elements,int numElements);
 OpaqueShapeList calculateOutputShapes2(PointerPointer extraPointers, long hash, OpaqueContext context);

 void dbPrintAllocationTrace(org.nd4j.nativeblas.OpaqueDataBuffer db);
 int numIntermediateResults(org.nd4j.nativeblas.OpaqueContext contextPointer);
 long dbBufferLength(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void toggleOpTrace(boolean opTrace);
 void purgeOpTrace();
 void printOpTrace();
 void copyBuffer(org.nd4j.nativeblas.OpaqueDataBuffer target, long n, org.nd4j.nativeblas.OpaqueDataBuffer from, long fromOffset, long targetOffset);
 int contextNumInputs(Pointer contextPointer);
 int contextNumOutputs(Pointer contextPointer);
 int numInputs(Pointer execTrace);
 int numOutputs(Pointer execTrace);
 int getDeviceId(Pointer ptrToDeviceId);
 int getDeviceBlockThreshold(int deviceId);
 int getDeviceSharedThreshold(int deviceId);
 void printDeviceBuffer(org.nd4j.nativeblas.OpaqueDataBuffer buffer, long offset);
 void printDeviceBuffer(org.nd4j.nativeblas.OpaqueDataBuffer buffer);
 void execPairwiseTransform(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y, org.nd4j.nativeblas.OpaqueNDArray z, Pointer extraParams);
 void execPairwiseTransformBool(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execSummaryStatsScalar(PointerPointer extraPointers,
                             int opNum,
                             org.nd4j.nativeblas.OpaqueNDArray x,
                             Pointer extraParams,
                             org.nd4j.nativeblas.OpaqueNDArray z,
                             boolean biasCorrected);

 void execSummaryStatsTad(PointerPointer extraPointers,
                          int opNum, OpaqueNDArray x,
                          Pointer extraParams,
                          OpaqueNDArray z,
                          OpaqueNDArray dimension,
                          boolean biasCorrected);

 void execRandom(PointerPointer extraPointers, int opNum, Pointer stateHost, OpaqueNDArray z, Pointer extraArguments);

 void execRandom2(PointerPointer extraPointers, int opNum,Pointer stateHost, OpaqueNDArray x,
                  OpaqueNDArray z, Pointer extraArguments);

 void execRandom3(PointerPointer extraPointers, int opNum,Pointer stateHost,
                  OpaqueNDArray x,
                  OpaqueNDArray y, OpaqueNDArray z, Pointer extraArguments);


 void execScalarBool(PointerPointer extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, Pointer extraParams);

 void execScalarBoolTad(PointerPointer extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, Pointer extraParams, OpaqueNDArray dimension);
 void execScalarTad(PointerPointer extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, Pointer extraParams, OpaqueNDArray dimension);
 void execScalar(PointerPointer extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, Pointer extraParams);

 void execBroadcastBool(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y, org.nd4j.nativeblas.OpaqueNDArray z, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execBroadcast(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y, org.nd4j.nativeblas.OpaqueNDArray z, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceFloat(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execReduceSame(PointerPointer extraPointers,
                     int opNum,
                     OpaqueNDArray x,
                     Pointer extraParams,
                     OpaqueNDArray z);
 void execReduceSame2(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceLong2(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceLong(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceBool2(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceBool(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execIndexReduce(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execReduceFloat2(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension);
 void execIndexReduceScalar(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execTransformSame(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execTransformBool(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execTransformAny(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execTransformStrict(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);
 void execTransformFloat(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray z);

 String getAllCustomOps();

 void inspectArray(PointerPointer  extraPointers, Pointer  buffer, LongPointer shapeInfo, Pointer specialBuffer,
                   LongPointer specialShapeInfo, Pointer  debugInfo);

 OpaqueConstantDataBuffer constantBufferDouble(int dtype, DoublePointer data, int length);

 OpaqueVariable getVariable(OpaqueVariablesSet  set, long i);


 LongPointer getVariableShape(OpaqueVariable variable);

 Pointer getVariableBuffer(OpaqueVariable variable);


 String getVariableName(OpaqueVariable variable);
 LongPointer getPrimaryShapeInfo(OpaqueTadPack pack);
 LongPointer getPrimaryOffsets(OpaqueTadPack pack);
 LongPointer getSpecialShapeInfo(OpaqueTadPack pack) ;
 LongPointer getSpecialOffsets(OpaqueTadPack pack);
 OpaqueTadPack tadOnlyShapeInfo(OpaqueDataBuffer hXShapeInfo, LongPointer dimension, long dimensionLength);
 void checkP2P();
 void enableP2P(boolean enable);
 boolean isP2PAvailable();
 void initializeDevicesAndFunctions();
 void initializeFunctions(PointerPointer functions);
 Pointer mallocHost(long memorySize, int flags);
 Pointer mallocDevice(long memorySize, int deviceId, int flags);
 int freeHost(Pointer pointer);
 int freeDevice(Pointer pointer, int deviceId);
 Pointer createContext();
 Pointer createStream();
 Pointer createEvent();
 int registerEvent(Pointer event, Pointer stream);
 int setDevice(int deviceId);
 OpaqueConstantDataBuffer constantBufferLong(int dtype, LongPointer data, int length);
 String getDeviceName(int device);
 long getDeviceFreeMemoryDefault();
 long getDeviceFreeMemory(int device);
 long getDeviceTotalMemory(int device);
 int memcpySync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);
 int memcpyAsync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);
 int memsetSync(Pointer dst, int value, long size, int flags, Pointer reserved);
 int memsetAsync(Pointer dst, int value, long size, int flags, Pointer reserved);
 int destroyEvent(Pointer event);
 int streamSynchronize(Pointer stream);
 int eventSynchronize(Pointer event);
 int getAvailableDevices();
 void enableDebugMode(boolean reallyEnable);
 void setGridLimit(int gridSize);
 int ompGetMaxThreads();
 int ompGetNumThreads();
 void setOmpNumThreads(int threads);
 void enableVerboseMode(boolean reallyEnable);
 int getDeviceMajor(int device);
 int getDeviceMinor(int device);
 long getNumberOfTads(OpaqueTadPack pack);
 int getShapeInfoLength(OpaqueTadPack pack);
 int memcpyConstantAsync(long dst, Pointer src, long size, int flags, Pointer reserved);
 Pointer getConstantSpace();
 boolean isExperimentalEnabled();
 void setOmpMinThreads(int threads);
 int getDevice();
 void setElementThreshold(int num);
 void setTADThreshold(int num);
 void execReduce3(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, Pointer extraParams, org.nd4j.nativeblas.OpaqueNDArray y, org.nd4j.nativeblas.OpaqueNDArray z);

 void execReduce3All(PointerPointer extraPointers, int opNum, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y,
                     org.nd4j.nativeblas.OpaqueNDArray z, org.nd4j.nativeblas.OpaqueNDArray dimension, Pointer extraParams);
 void execReduce3Scalar(PointerPointer extraPointers, int opNum, OpaqueNDArray x, Pointer extraParams, OpaqueNDArray y, OpaqueNDArray z);
 void execReduce3Tad(PointerPointer extraPointers, int opNum, OpaqueNDArray x, Pointer extraParams, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension);
 Pointer initRandom(PointerPointer extraPointers, long seed, long bufferSize, Pointer ptrToBuffer);
 void destroyRandom(Pointer ptrBuffer);
 void refreshBuffer(PointerPointer extraPointers, long seed, Pointer ptrRandom);
 void reSeedBuffer(PointerPointer extraPointers, long seed, Pointer ptrRandom);
 int lengthForShapeBufferPointer(Pointer buffer);
 Pointer pointerForAddress(long _address);
 void prescanArrayRecursive(PointerPointer extras, IntPointer dZ, IntPointer dX, int numElements, int level);
 void prescanArrayRecursive(PointerPointer extras, IntBuffer dZ, IntBuffer dX, int numElements, int level);
 void prescanArrayRecursive(PointerPointer extras, int[] dZ, int[] dX, int numElements, int level);
 void deleteNDArray(org.nd4j.nativeblas.OpaqueNDArray array);
 long getOpaqueNDArrayOffset(org.nd4j.nativeblas.OpaqueNDArray array);
 Pointer getOpaqueNDArrayBuffer(org.nd4j.nativeblas.OpaqueNDArray array);
 Pointer getOpaqueNDArraySpecialBuffer(org.nd4j.nativeblas.OpaqueNDArray array);
 long getShapeInfoLength(org.nd4j.nativeblas.OpaqueNDArray array);
 long getOpaqueNDArrayLength(org.nd4j.nativeblas.OpaqueNDArray array);
 void sort(PointerPointer extraPointers, org.nd4j.nativeblas.OpaqueNDArray x, boolean descending);
 void sortTad(PointerPointer extraPointers,  org.nd4j.nativeblas.OpaqueNDArray x,
             LongPointer dimension,  long dimensionLength,
             LongPointer tadShapeInfo,  LongPointer tadOffsets,  boolean descending);

 void sortByValue(PointerPointer extraPointers, org.nd4j.nativeblas.OpaqueNDArray x, org.nd4j.nativeblas.OpaqueNDArray y, boolean descending);
 void sortTadByKey(PointerPointer extraPointers,
                   OpaqueNDArray x,
                   OpaqueNDArray y,
                   OpaqueNDArray dimension,
                   boolean descending);

 void sortTadByValue(PointerPointer extraPointers,
                     OpaqueNDArray x,
                     OpaqueNDArray y,
                     OpaqueNDArray dimension,
                     boolean descending);
 void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);
 void munmapFile(PointerPointer extraPointers, LongBuffer ptrMap, long length);
 void munmapFile(PointerPointer extraPointers, long[] ptrMap, long length);
 long getResultWrapperSize(OpaqueResultWrapper ptr);
 Pointer getResultWrapperPointer(OpaqueResultWrapper ptr);
 long getShapeListSize(OpaqueShapeList list);
 int execCustomOp2(PointerPointer extraPointers, long hash, OpaqueContext opContext);
 int registerGraph(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);
 long getVariablesSetSize(OpaqueVariablesSet set);
 int getVariablesSetStatus(OpaqueVariablesSet set);
 int getVariableId(OpaqueVariable variable);
 int getVariableIndex(OpaqueVariable variable);
 int unregisterGraph(PointerPointer extraPointers, long graphId);
 void deletePointerArray(Pointer pointer);
 void deleteCharArray(Pointer pointer);
 void deleteIntArray(Pointer pointer);
 void deleteLongArray(Pointer pointer);
 void deleteVariablesSet(OpaqueVariablesSet pointer);
 void deleteShapeList(Pointer shapeList);
 Pointer getGraphState(long id);
 void deleteGraphState(Pointer state);
 void deleteResultWrapper(Pointer ptr);
 void convertTypes(PointerPointer extras, int srcType, Pointer dX, long N, int dstType, Pointer dZ);
 Pointer createUtf8String(PointerPointer extraPointers, String string, int length);
 Pointer createUtf8String(PointerPointer extraPointers, BytePointer string, int length);
 long getUtf8StringLength(PointerPointer extraPointers, Pointer ptr);
 void deleteUtf8String(PointerPointer extraPointers, Pointer ptr);
 void tryPointer(Pointer extra, Pointer p, int len);
 int dataTypeFromNpyHeader(Pointer header);
 void deleteConstantShapeBuffer(org.nd4j.nativeblas.OpaqueConstantShapeBuffer ptr);
 void deleteConstantDataBuffer(org.nd4j.nativeblas.OpaqueConstantDataBuffer ptr);
 void deleteTadPack(OpaqueTadPack ptr);
 boolean isBlasVersionMatches(int major, int minor, int build);
 Pointer getConstantDataBufferPrimary(org.nd4j.nativeblas.OpaqueConstantDataBuffer dbf);
 Pointer getConstantDataBufferSpecial(org.nd4j.nativeblas.OpaqueConstantDataBuffer dbf);
 long getConstantDataBufferLength(org.nd4j.nativeblas.OpaqueConstantDataBuffer dbf);
 long getConstantDataBufferSizeOf(org.nd4j.nativeblas.OpaqueConstantDataBuffer dbf);
 Pointer getConstantShapeBufferPrimary(org.nd4j.nativeblas.OpaqueConstantShapeBuffer dbf);
 Pointer getConstantShapeBufferSpecial(org.nd4j.nativeblas.OpaqueConstantShapeBuffer dbf);
 void markGraphContextInplace(OpaqueContext ptr, boolean reallyInplace);
 org.nd4j.nativeblas.OpaqueNDArray getOutputArrayNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 org.nd4j.nativeblas.OpaqueNDArray getInputArrayNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 long dataTypeNativeAt(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 boolean bArgAtNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 long iArgumentAtNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 long numDNative(org.nd4j.nativeblas.OpaqueContext ptr);
 long numBNative(org.nd4j.nativeblas.OpaqueContext ptr);
 long numOutputsNative(org.nd4j.nativeblas.OpaqueContext ptr);
 long numInputsNative(org.nd4j.nativeblas.OpaqueContext ptr);
 double tArgumentNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 long numTArgumentsNative(org.nd4j.nativeblas.OpaqueContext ptr);
 long numIArgumentsNative(org.nd4j.nativeblas.OpaqueContext ptr);
 void setGraphContextOutputArray(org.nd4j.nativeblas.OpaqueContext ptr, int index, org.nd4j.nativeblas.OpaqueNDArray arr);
 void setGraphContextInputArray(org.nd4j.nativeblas.OpaqueContext ptr, int index, org.nd4j.nativeblas.OpaqueNDArray arr);
 void setGraphContextOutputArraysArr(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, org.nd4j.nativeblas.OpaqueNDArrayArr arr);
 void setGraphContextInputArraysArr(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, org.nd4j.nativeblas.OpaqueNDArrayArr arr);
 void setGraphContextTArguments(org.nd4j.nativeblas.OpaqueContext ptr, DoublePointer arguments, int numberOfArguments);
 void setGraphContextTArguments(org.nd4j.nativeblas.OpaqueContext ptr, double[] arguments, int numberOfArguments);
 void setGraphContextIArguments(org.nd4j.nativeblas.OpaqueContext ptr, LongPointer arguments, int numberOfArguments);
 void setGraphContextIArguments(org.nd4j.nativeblas.OpaqueContext ptr, LongBuffer arguments, int numberOfArguments);
 void setGraphContextIArguments(org.nd4j.nativeblas.OpaqueContext ptr, long[] arguments, int numberOfArguments);
 void setGraphContextBArguments(org.nd4j.nativeblas.OpaqueContext ptr, BooleanPointer arguments, int numberOfArguments);
 void setGraphContextBArguments(org.nd4j.nativeblas.OpaqueContext ptr, boolean[] arguments, int numberOfArguments);
 void setGraphContextDArguments(org.nd4j.nativeblas.OpaqueContext ptr, IntPointer arguments, int numberOfArguments);
 void setGraphContextDArguments(org.nd4j.nativeblas.OpaqueContext ptr, IntBuffer arguments, int numberOfArguments);
 void setGraphContextDArguments(org.nd4j.nativeblas.OpaqueContext ptr, int[] arguments, int numberOfArguments);
 void deleteGraphContext(org.nd4j.nativeblas.OpaqueContext ptr);
 long getRandomGeneratorRootState(OpaqueRandomGenerator ptr);
 long getRandomGeneratorNodeState(OpaqueRandomGenerator ptr);
 void setRandomGeneratorStates(OpaqueRandomGenerator ptr, long rootSeed, long nodeSeed);
 float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, long index);
 double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, long index);
 int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, long index);
 long getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, long index);
 int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr);
 long getRandomGeneratorNextLong(OpaqueRandomGenerator ptr);
 float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr);
 double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr);
 void deleteRandomGenerator(OpaqueRandomGenerator ptr);
 Pointer shapeBufferForNumpy(Pointer npyArray);
 Pointer shapeBufferForNumpyHeader(Pointer npyArray);
 long numpyHeaderLength(OpaqueDataBuffer opaqueDataBuffer,Pointer shapeBuffer);

 long getCachedMemory(int deviceId);
 Pointer lcScalarPointer(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcReductionPointer(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcAllocationPointer(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcExecutionStream(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcCopyStream(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcBlasHandle(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 Pointer lcSolverHandle(org.nd4j.nativeblas.OpaqueLaunchContext lc);
 int lastErrorCode();
 void ctxShapeFunctionOverride(org.nd4j.nativeblas.OpaqueContext ptr, boolean reallyOverride);
 void ctxPurge(org.nd4j.nativeblas.OpaqueContext ptr);
 int binaryLevel();
 int optimalLevel();
 boolean isMinimalRequirementsMet();
 boolean isOptimalRequirementsMet();
 void ctxAllowHelpers(org.nd4j.nativeblas.OpaqueContext ptr, boolean reallyAllow);
 void ctxSetExecutionMode(org.nd4j.nativeblas.OpaqueContext ptr, int execMode);
 Pointer dbPrimaryBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 Pointer dbSpecialBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void deleteDataBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbSetPrimaryBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, Pointer primaryBuffer, long numBytes);
 void dbSetSpecialBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, Pointer specialBuffer, long numBytes);
 void dbAllocatePrimaryBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbAllocateSpecialBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbExpandBuffer(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, long elements);
 int dbUseCount(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbSyncToSpecial(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbSyncToPrimary(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickHostRead(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickHostWrite(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickDeviceRead(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickDeviceWrite(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbExpand(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, long elements);
 void dbClose(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 int dbDeviceId(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbSetDeviceId(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, int deviceId);
 int dbLocality(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 org.nd4j.nativeblas.OpaqueDataBuffer dbCreateView(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, long length);
 org.nd4j.nativeblas.OpaqueDataBuffer dbAllocateDataBuffer(long elements, int dataType, boolean allocateBoth);
 org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special);
 void setShapeBuffer(LongPointer inputShapeData, int dt, LongPointer bufferToSet, char order, int elementWiseStride, boolean isEmpty, boolean isView);

 org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(long[] shapeInfo);


 org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(int rank, LongPointer shape, LongPointer strides, int dtype, char order, long ews, long extras);

 org.nd4j.nativeblas.OpaqueDataBuffer allocateDataBuffer(long elements, int dataType, boolean allocateBoth);
 Pointer numpyHeaderForNd4j(Pointer data, Pointer shapeBuffer, long wordSize, LongPointer headerSize);
 Pointer numpyHeaderForNd4j(Pointer data, Pointer shapeBuffer, long wordSize, LongBuffer headerSize);
 Pointer numpyHeaderForNd4j(Pointer data, Pointer shapeBuffer, long wordSize, long[] headerSize);
 Pointer numpyFromNd4j(Pointer data, Pointer shapeBuffer, long wordSize);
 Pointer loadNpyFromHeader(Pointer data);
 Pointer numpyFromFile(BytePointer path);
 Pointer numpyFromFile(String path);
 Pointer mapFromNpzFile(BytePointer path);
 Pointer mapFromNpzFile(String path);
 int getNumNpyArraysInMap(Pointer map);
 LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);
 String getNpyArrayNameFromMap(Pointer map, int index, BytePointer nameBuffer);
 BytePointer getNpyArrayNameFromMap(Pointer map, int index, String nameBuffer);
 Pointer getNpyArrayFromMap(Pointer map, int index);
 Pointer getNpyArrayData(Pointer npArray);
 int getNpyArrayRank(Pointer npArray);
 LongPointer getNpyArrayShape(Pointer npArray);
 char getNpyArrayOrder(Pointer npArray);
 int getNpyArrayElemSize(Pointer npArray);
 void deleteNPArrayStruct(Pointer npArray);
 void deleteNPArrayMap(Pointer map);
 int elementSizeForNpyArray(Pointer npyArray);
 int elementSizeForNpyArrayHeader(Pointer npyArray);
 void releaseNumpy(Pointer npyArray);
 Pointer dataPointForNumpyHeader(Pointer npyArray);
 Pointer dataPointForNumpyStruct(Pointer npyArrayStruct);
 Pointer dataPointForNumpy(Pointer npyArray);
 PointerPointer intermediateResults(org.nd4j.nativeblas.OpaqueContext contextPointer);
 PointerPointer intermediateResultsShapeInfo(org.nd4j.nativeblas.OpaqueContext contextPointer);
 void setIntermediateResult(org.nd4j.nativeblas.OpaqueContext contextPointer, int index, org.nd4j.nativeblas.OpaqueDataBuffer buffer, org.nd4j.nativeblas.OpaqueDataBuffer shapeInfo, long dataOffset);
 void pushIntermediateResult(org.nd4j.nativeblas.OpaqueContext contextPointer, org.nd4j.nativeblas.OpaqueDataBuffer buffer, org.nd4j.nativeblas.OpaqueDataBuffer shapeInfo, long offset);
 org.nd4j.nativeblas.OpaqueDataBuffer intermediateResultDataAt(int index, org.nd4j.nativeblas.OpaqueContext contextPointer);
 LongPointer intermediateResultShapeInfoAt(int index, org.nd4j.nativeblas.OpaqueContext contextPointer);
 String lastErrorMessage();
 org.nd4j.nativeblas.OpaqueLaunchContext defaultLaunchContext();
 String buildInfo();
 boolean isFuncTrace();
}
