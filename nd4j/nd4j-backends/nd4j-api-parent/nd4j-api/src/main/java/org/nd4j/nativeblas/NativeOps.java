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

 /**
  * Get the stack trace for a TadPack as a string.
  * Returns the allocation stack trace if functrace is enabled, empty string otherwise.
  * This is useful for debugging TAD cache lifecycle issues.
  *
  * @param pack The TadPack to get the stack trace from
  * @return String containing the formatted stack trace
  */
 String getTadPackStackTrace(OpaqueTadPack pack);

 OpaqueTadPack tadOnlyShapeInfo(OpaqueDataBuffer hXShapeInfo, LongPointer dimension, long dimensionLength);
 void checkP2P();
 void enableP2P(boolean enable);
 boolean isP2PAvailable();
 void initializeDevicesAndFunctions();
 void initializeFunctions(PointerPointer functions);

 /**
  * Initialize the shape cache eagerly during early JVM startup.
  * This prevents race conditions during static initialization when multiple threads
  * try to create NDArrays concurrently before the shape cache is fully initialized.
  * <p>
  * This should be called from Nd4j initialization BEFORE any class loading
  * that might create NDArrays (like DifferentialFunctionClassHolder).
  * </p>
  * <p>
  * Safe to call multiple times - subsequent calls are no-ops.
  * </p>
  */
 void initializeShapeCache();

 /**
  * Initialize the TAD (Tensor-Along-Dimension) cache early to prevent race conditions.
  * <p>
  * This forces initialization of DirectTadTrie in a controlled, single-threaded context.
  * This prevents race conditions during static initialization when multiple threads
  * try to create TAD operations concurrently before the TAD cache is fully initialized.
  * </p>
  * <p>
  * This should be called from Nd4j initialization BEFORE any class loading
  * that might create TAD operations.
  * </p>
  * <p>
  * Safe to call multiple times - subsequent calls are no-ops.
  * </p>
  */
 void initializeTadCache();
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

 /**
  * Get the stack trace for a ConstantShapeBuffer as a string.
  * Returns the allocation stack trace if functrace is enabled, empty string otherwise.
  * This is useful for debugging shape buffer lifecycle issues.
  *
  * @param buffer The ConstantShapeBuffer to get the stack trace from
  * @return String containing the formatted stack trace
  */
 String getConstantShapeBufferStackTrace(org.nd4j.nativeblas.OpaqueConstantShapeBuffer buffer);

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

 /**
  * Triggers LeakSanitizer to perform a leak check immediately.
  * Only functional when libnd4j is built with ASAN/LSAN enabled.
  * No-op when built without sanitizers.
  */
 void triggerLeakCheck();

 /**
  * Initializes lifecycle crash handlers to capture crash dumps.
  * <p>
  * This must be called AFTER JVM is fully initialized to ensure
  * the crash handler properly chains to JVM's hs_err generation.
  * </p>
  * <p>
  * This fixes the signal handler installation race condition where crash handlers
  * were being installed during library load (too early), capturing SIG_DFL instead
  * of JVM's actual crash handler. This prevented hs_err file generation.
  * </p>
  * <p>
  * After this fix, crashes will generate BOTH:
  * <ul>
  * <li>sd_crash_*.log - Lifecycle tracker dump (NDArray allocations, etc.)</li>
  * <li>hs_err_pid*.log - JVM crash dump (full stack trace, thread info, etc.)</li>
  * </ul>
  * </p>
  * <p>
  * Safe to call multiple times - only initializes once.
  * </p>
  * <p>
  * NOTE: Only available when built with {@code -Dlibnd4j.calltrace=ON}. No-op otherwise.
  * </p>
  */
 void initializeLifecycleCrashHandlers();

 /**
  * Enables NDArray lifecycle tracking (allocation/deallocation monitoring).
  */
 void enableNDArrayTracking();

 /**
  * Disables NDArray lifecycle tracking.
  */
 void disableNDArrayTracking();

 /**
  * Enables DataBuffer lifecycle tracking.
  */
 void enableDataBufferTracking();

 /**
  * Disables DataBuffer lifecycle tracking.
  */
 void disableDataBufferTracking();

 /**
  * Enables TADCache lifecycle tracking.
  */
 void enableTADCacheTracking();

 /**
  * Disables TADCache lifecycle tracking.
  */
 void disableTADCacheTracking();

 /**
  * Enables ShapeCache lifecycle tracking.
  */
 void enableShapeCacheTracking();

 /**
  * Disables ShapeCache lifecycle tracking.
  */
 void disableShapeCacheTracking();

 /**
  * Enables OpContext lifecycle tracking.
  */
 void enableOpContextTracking();

 /**
  * Disables OpContext lifecycle tracking.
  */
 void disableOpContextTracking();

 /**
  * Enables operation execution logging for crash detection.
  * <p>
  * When enabled (and libnd4j is built with -Dlibnd4j.calltrace=ON), all operation executions
  * are logged to a file with full unified C++/Java stack traces. The log file survives crashes
  * and can be used for post-mortem debugging.
  * </p>
  * <p>
  * Log files are located at: {@code /tmp/nd4j_op_execution_<PID>.log}
  * (or {@code $SD_OP_LOG_DIR} if set)
  * </p>
  * <p>
  * NOTE: Only available when built with {@code -Dlibnd4j.calltrace=ON}. No-op otherwise.
  * </p>
  *
  * @see #disableOpExecutionLogging()
  * @see #getOpExecutionLogPath()
  * @see #getOpExecutionLogContents(long, boolean)
  */
 void enableOpExecutionLogging();

 /**
  * Disables operation execution logging.
  * <p>
  * No-op if SD_GCC_FUNCTRACE is not defined.
  * </p>
  *
  * @see #enableOpExecutionLogging()
  */
 void disableOpExecutionLogging();

 /**
  * Check if operation execution logging is currently enabled.
  *
  * @return true if logging is enabled, false otherwise
  */
 boolean isOpExecutionLoggingEnabled();

 /**
  * Get the current operation execution log file path.
  * <p>
  * Returns empty string if logging is not enabled or not available.
  * </p>
  *
  * @return String containing the log file path
  */
 String getOpExecutionLogPath();

 /**
  * Get the current operation execution log contents as a string.
  * <p>
  * Useful for retrieving recent execution history for debugging.
  * </p>
  *
  * @param maxBytes Maximum bytes to read (0 = read entire file)
  * @param fromEnd If true, read from end of file (most recent entries)
  * @return String containing the log contents (empty if logging not enabled)
  */
 String getOpExecutionLogContents(long maxBytes, boolean fromEnd);

 /**
  * Force a flush of the operation execution log to disk.
  * <p>
  * The logger flushes after each operation by default, but this
  * can be called manually for explicit checkpointing.
  * </p>
  */
 void dumpOpExecutionLog();

 /**
  * Manually dump current state to the operation execution log.
  * <p>
  * Useful for checkpointing at specific points in code.
  * </p>
  *
  * @param message Optional message to include in the dump (can be null)
  */
 void dumpOpExecutionState(String message);

 // ═══════════════════════════════════════════════════════════════
 // Allocation Logging API (SD_GCC_FUNCTRACE only)
 // Similar to OpExecutionLogging, but focuses on tracking NDArray
 // and OpContext allocations for understanding memory growth patterns
 // ═══════════════════════════════════════════════════════════════

 /**
  * Get the current allocation log file path.
  * <p>
  * Allocation logging is always active in functrace builds (SD_GCC_FUNCTRACE).
  * Returns empty string if functrace is not enabled.
  * </p>
  * <p>
  * Log file location: /tmp/nd4j_allocations_&lt;PID&gt;.log (configurable via SD_ALLOCATION_LOG_DIR)
  * </p>
  *
  * @return String containing the log file path
  */
 String getAllocationLogPath();

 /**
  * Generates a comprehensive leak source analysis report combining data from ALL lifecycle trackers.
  * <p>
  * This method analyzes undeleted allocations across all 5 lifecycle trackers:
  * <ul>
  *   <li>NDArrayLifecycleTracker</li>
  *   <li>DataBufferLifecycleTracker</li>
  *   <li>TADCacheLifecycleTracker</li>
  *   <li>ShapeCacheLifecycleTracker</li>
  *   <li>OpContextLifecycleTracker</li>
  * </ul>
  * <p>
  * For each allocation source (Java method or C++ function), the report shows:
  * <ul>
  *   <li>Total number of undeleted allocations</li>
  *   <li>Breakdown by object type (NDArray, DataBuffer, TAD, Shape, OpContext)</li>
  *   <li>Total bytes leaked</li>
  *   <li>Example stack traces (both Java and C++)</li>
  * </ul>
  * <p>
  * Results are sorted by total leak count, making it easy to identify the top leak sources.
  *
  * @param outputDir Directory where report files should be written (e.g., "./leak_reports").
  *                  If null or empty, uses current directory.
  *
  * <p>Output files generated:</p>
  * <ul>
  *   <li>comprehensive_leak_report.txt - Detailed report with top 50 leak sources</li>
  *   <li>Console output shows top 20 leak sources summary</li>
  * </ul>
  *
  * <p>Example usage:</p>
  * <pre>{@code
  * // Enable tracking
  * Nd4j.getNativeOps().enableNDArrayTracking();
  * Nd4j.getNativeOps().enableDataBufferTracking();
  * Nd4j.getNativeOps().enableTADCacheTracking();
  * Nd4j.getNativeOps().enableShapeCacheTracking();
  * Nd4j.getNativeOps().enableOpContextTracking();
  *
  * // Run workload
  * runYourWorkload();
  *
  * // Generate analysis
  * Nd4j.getNativeOps().generateComprehensiveLeakAnalysis("./leak_reports");
  * }</pre>
  *
  * @see #enableNDArrayTracking()
  * @see #enableDataBufferTracking()
  */
 void generateComprehensiveLeakAnalysis(String outputDir);

 /**
  * Clears all cached TAD packs to prevent memory leaks.
  * This is particularly useful when running memory leak tests or
  * during application shutdown to free accumulated cache memory.
  * NOTE: Will return early without action if setTADCacheShutdownInProgress(true) was called.
  */
 void clearTADCache();

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
 void setTADCacheShutdownInProgress(boolean inProgress);

 /**
  * Check if TAD cache shutdown is in progress.
  * @return true if shutdown is marked as in progress
  */
 boolean isTADCacheShutdownInProgress();

 /**
  * Clears all cached shape buffers to prevent memory leaks.
  * This is called during application shutdown to free accumulated cache memory.
  */
 void clearShapeCache();

 /**
  * Get the total number of cached shape buffer entries.
  *
  * @return Total number of cached shape buffers across all stripes
  */
 long getShapeCachedEntries();

 /**
  * Get the total memory used by cached shape buffers in bytes.
  *
  * @return Total memory used in bytes
  */
 long getShapeCachedBytes();

 /**
  * Get the peak number of shape entries that were cached simultaneously.
  *
  * @return Peak number of cached shape buffers
  */
 long getShapePeakCachedEntries();

 /**
  * Get the peak memory usage by cached shape buffers in bytes.
  *
  * @return Peak memory usage in bytes
  */
 long getShapePeakCachedBytes();

 /**
  * Get the total number of cached TAD pack entries.
  *
  * @return Total number of cached TAD packs across all stripes
  */
 long getTADCachedEntries();

 /**
  * Get the total memory used by cached TAD packs in bytes.
  * This includes both shape_info and offset buffer sizes.
  *
  * @return Total memory used in bytes
  */
 long getTADCachedBytes();

 /**
  * Get the peak number of TAD pack entries that were cached simultaneously.
  *
  * @return Peak number of cached TAD packs
  */
 long getTADPeakCachedEntries();

 /**
  * Get the peak memory usage by cached TAD packs in bytes.
  *
  * @return Peak memory usage in bytes
  */
 long getTADPeakCachedBytes();

 /**
  * Get a string representation of the shape cache for debugging.
  * Shows the trie structure with cached shape buffers.
  * The returned string must be freed using freeString().
  *
  * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
  * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
  * @return String representation of the shape cache
  */
 String getShapeCacheString(int maxDepth, int maxEntries);

 /**
  * Get a string representation of the TAD cache for debugging.
  * Shows the trie structure with cached TAD packs.
  * The returned string must be freed using freeString().
  *
  * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
  * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
  * @return String representation of the TAD cache
  */
 String getTADCacheString(int maxDepth, int maxEntries);

 /**
  * Check and cleanup caches at configurable intervals.
  * Called after operations to prevent cache accumulation.
  */
 void checkAndCleanupCaches();

 // NEW: Temporal leak analysis
 void generateNDArrayTemporalLeakReport(String outputPath, int windowCount, double windowDurationSec);
 void generateTADCacheTemporalLeakReport(String outputPath, int windowCount, double windowDurationSec);

 // NEW: Snapshot differential analysis
 long captureNDArrayLeakSnapshot();
 long captureTADCacheLeakSnapshot();
 void generateNDArraySnapshotDiff(long snapshot1, long snapshot2, String outputPath);
 void generateTADCacheSnapshotDiff(long snapshot1, long snapshot2, String outputPath);
 void clearNDArraySnapshots();
 void clearTADCacheSnapshots();

 // ===============================
 // DeallocatorService Lifecycle Tracking
 // Records Java-side deallocation statistics from DeallocatorService
 // to be merged with C++ lifecycle trackers in UnifiedMemoryReporter
 // ===============================

 /**
  * Records a snapshot of DeallocatorService statistics for integration with C++ lifecycle tracking.
  * <p>
  * This method is called by the Java DeallocatorService to push its time-series
  * tracking data to the C++ DeallocatorServiceLifecycleTracker, which is then
  * merged into the UnifiedMemoryReporter.
  * </p>
  * <p>
  * Only functional when built with SD_GCC_FUNCTRACE. No-op otherwise.
  * </p>
  *
  * @param totalAllocations Total number of allocations tracked
  * @param totalDeallocations Total number of deallocations tracked
  * @param totalBytesAllocated Total bytes allocated
  * @param totalBytesDeallocated Total bytes deallocated
  * @param peakLiveCount Peak number of live objects observed
  * @param peakBytes Peak bytes in use observed
  */
 void recordDeallocatorServiceSnapshot(long totalAllocations, long totalDeallocations,
                                       long totalBytesAllocated, long totalBytesDeallocated,
                                       long peakLiveCount, long peakBytes);

 /**
  * Enables DeallocatorService lifecycle tracking on the C++ side.
  * <p>
  * When enabled, the DeallocatorServiceLifecycleTracker will store snapshots
  * pushed from Java and include them in UnifiedMemoryReporter output.
  * </p>
  */
 void enableDeallocatorServiceTracking();

 /**
  * Disables DeallocatorService lifecycle tracking on the C++ side.
  */
 void disableDeallocatorServiceTracking();

 /**
  * Checks if DeallocatorService tracking is enabled.
  *
  * @return true if tracking is enabled
  */
 boolean isDeallocatorServiceTrackingEnabled();

 /**
  * Gets the current live count from DeallocatorService tracker.
  *
  * @return current live count (allocations - deallocations)
  */
 long getDeallocatorServiceLiveCount();

 /**
  * Gets the current bytes in use from DeallocatorService tracker.
  *
  * @return current bytes in use (allocated - deallocated)
  */
 long getDeallocatorServiceBytesInUse();

 /**
  * Free a string returned by native code.
  *
  * @param ptr String pointer to free
  */
 void freeString(String ptr);

 // ===============================
 // Allocation Context for Lifecycle Tracking
 // Allows Java code to tag allocations with operation names
 // ===============================

 /**
  * Set the current allocation context (operation name) for lifecycle tracking.
  * <p>
  * This allows tagging native memory allocations with the operation that triggered them,
  * providing much better granularity in leak reports. For example, instead of seeing
  * "720 leaks from data_buffer_alloc", you'll see "50 leaks from Sum", "30 leaks from Mean", etc.
  * </p>
  * <p>
  * The context is thread-local, so each thread can have its own context.
  * Call {@link #clearAllocationContext()} when the operation is complete.
  * </p>
  * <p>
  * Only functional when built with SD_GCC_FUNCTRACE. No-op otherwise.
  * </p>
  *
  * @param opName The operation name to associate with allocations (e.g., "Sum", "Mean", "Concat")
  */
 void setAllocationContext(String opName);

 /**
  * Clear the current allocation context for this thread.
  * <p>
  * Call this after op creation/execution to stop tagging allocations with the operation name.
  * </p>
  */
 void clearAllocationContext();

 /**
  * Update the Java stack trace for an existing NDArray allocation record.
  * <p>
  * This is called from Java after creating an OpaqueNDArray to provide the full Java stack trace
  * captured before the JNI boundary. This gives much better context than capturing the stack
  * trace from within native code.
  * </p>
  * <p>
  * Only functional when built with SD_GCC_FUNCTRACE. No-op otherwise.
  * </p>
  *
  * @param array The OpaqueNDArray whose allocation record should be updated
  * @param javaStackTrace The full Java stack trace as a string
  */
 void updateAllocationJavaStackTrace(OpaqueNDArray array, String javaStackTrace);
}
