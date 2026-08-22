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
import org.bytedeco.javacpp.annotation.NoDeallocator;

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




 @NoDeallocator
 OpaqueNDArray create(OpaqueDataBuffer shapeInfo,
                      OpaqueDataBuffer buffer,
                      OpaqueDataBuffer specialBuffer,
                      long offset);

 void saveNpy(String fname,  OpaqueDataBuffer  data, IntPointer shape, int ndims,
              String mode);


 LongPointer getShape(OpaqueShapeList list, long i);
 boolean checkOpaqueNDArrayElementsNull(OpaqueNDArrayArr elements,int numElements);
 OpaqueShapeList calculateOutputShapes2(PointerPointer extraPointers, long hash, OpaqueContext context);

 /**
  * Same as calculateOutputShapes2 but skips forceSyncToHost() on input arrays.
  * Use for ops whose shape function only reads shape info (not array values).
  * Avoids expensive CUDA D2H synchronization for shape-only ops.
  */
 OpaqueShapeList calculateOutputShapesNoSync(PointerPointer extraPointers, long hash, OpaqueContext context);

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

 /**
  * Look up op trait flags by op name. Returns the OP_TRAIT_* bitmask
  * matching {@code libnd4j/include/ops/declarable/OpDescriptor.h}, or 0 if
  * the op is unknown. Used by the Java session code to replace hardcoded
  * string sets with a trait query.
  */
 int getOpTraits(String opName);

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
  * Ask this backend to initialize optional function entry points from the native process.
  *
  * <p>The backend's native {@link #initializeFunctions(PointerPointer)} implementation owns
  * the concrete symbol set and the meaning of a null pointer table. This path avoids carrying
  * JavaCPP pointer and String objects across an embedded native-image/host-JVM boundary.</p>
  */
 default void initializeFunctionsFromProcessSymbols() {
     initializeFunctions(null);
 }

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

 /**
  * Check if CUDA memory pools are enabled.
  * CUDA memory pools (cudaMallocAsync/cudaFreeAsync) provide efficient memory reuse
  * without explicit caching. Only effective on CUDA 11.2+ with compatible devices.
  * @return true if memory pools are enabled and supported, false otherwise
  */
 boolean isMemoryPoolEnabled();

 /**
  * Enable or disable CUDA memory pools.
  * When disabled, falls back to regular cudaMalloc/cudaFree.
  * @param enabled true to enable memory pools, false to disable
  */
 void setMemoryPoolEnabled(boolean enabled);

 /**
  * Get memory pool statistics for a device.
  * @param deviceId The device to query
  * @param usedBytes Output pointer for currently used bytes in the pool
  * @param reservedBytes Output pointer for total reserved bytes in the pool
  */
 void getMemoryPoolStats(int deviceId, LongPointer usedBytes, LongPointer reservedBytes);

 /**
  * Trim unused memory from the pool.
  * Releases cached memory back to the system.
  * @param deviceId The device whose pool to trim
  */
 void trimMemoryPool(int deviceId);

 /**
  * Trim unused memory from the pool, syncing the specified stream.
  * Use when frees were issued on a specific execution stream.
  * @param deviceId The device whose pool to trim
  * @param stream The backend execution stream to synchronize before trimming
  */
 void trimMemoryPoolOnStream(int deviceId, org.bytedeco.javacpp.Pointer stream);

 /**
  * Get current pinned host memory usage from failover allocations.
  * @return bytes currently allocated in pinned host memory
  */
 long getPinnedHostBytesUsed();

 /**
  * Get pinned host memory limit for failover allocations.
  * @return max bytes, or 0 for unlimited
  */
 long getPinnedHostBytesLimit();

 /**
  * Set pinned host memory limit for failover allocations.
  * When exceeded, allocations will fail instead of consuming unbounded host memory.
  * Set to 0 for unlimited (default).
  * @param maxBytes maximum bytes allowed for pinned host failover
  */
 void setPinnedHostBytesLimit(long maxBytes);

 /**
  * Set the proactive soft-limit percentage for GPU memory pool allocation.
  * When a device's usage exceeds this threshold, new allocations are routed
  * to other devices via failover BEFORE individual allocations fail.
  * This prevents cumulative exhaustion from many small allocations
  * (e.g., DSP warmup intermediates) that individually succeed but
  * collectively fill the GPU past the watchdog's kill threshold.
  *
  * @param percent 0 = disabled (default), 1-100 = proactive failover threshold.
  *                Typical value: 70 (routes at 70%, before watchdog's 75% stop).
  */
 void setMemoryPoolSoftLimitPercent(int percent);

 /**
  * Get the current proactive soft-limit percentage for GPU memory pool allocation.
  * @return 0 if disabled, otherwise the threshold percentage (1-100).
  */
 int getMemoryPoolSoftLimitPercent();

 /**
  * Add a device to the failover exclusion list.
  * Excluded devices are skipped during memory failover allocation (Steps 2 and 2b).
  * Used to isolate subprocess memory so failover from one process doesn't
  * displace resident pages of another process on a different device.
  * @param deviceId The device to exclude from failover
  */
 void addExcludedFailoverDevice(int deviceId);

 /**
  * Remove a device from the failover exclusion list.
  * @param deviceId The device to un-exclude
  */
 void removeExcludedFailoverDevice(int deviceId);

 /**
  * Clear all device exclusions from the failover list.
  */
 void clearExcludedFailoverDevices();

 /**
  * Check if a device is excluded from failover allocation.
  * @param deviceId The device to check
  * @return true if the device is currently excluded
  */
 boolean isDeviceExcludedFromFailover(int deviceId);

 Pointer createContext();
 Pointer createStream();
 Pointer createEvent();
 int registerEvent(Pointer event, Pointer stream);
 int setDevice(int deviceId);
 void setAvailableDevices(IntPointer devices, int size);
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
 boolean isPeerAccessSupported(int srcDevice, int dstDevice);
 void enableDebugMode(boolean reallyEnable);
 void setGridLimit(int gridSize);
 int ompGetMaxThreads();
 int ompGetNumThreads();
 void setOmpNumThreads(int threads);
 /**
  * Sets the number of threads used by OpenBLAS for BLAS operations.
  * This is separate from OMP threads and specifically controls OpenBLAS's internal threading.
  * Default should be 1 to prevent TLS corruption crashes in multi-threaded Java applications.
  * @param threads number of threads for OpenBLAS to use
  */
 void setOpenBlasThreads(int threads);

 /**
  * Gets the number of threads OpenBLAS is configured to use.
  * @return number of threads, 0 if using default
  */
 int getOpenBlasThreads();

 /**
  * Check if BLAS call serialization is enabled.
  * When enabled, external BLAS calls are serialized to prevent OpenBLAS
  * TLS corruption and race conditions in multi-threaded environments.
  * @return true if serialization is enabled
  */
 boolean isSerializeBlasCalls();

 /**
  * Enable or disable BLAS call serialization.
  * When enabled, external BLAS calls are serialized to prevent OpenBLAS
  * TLS corruption and race conditions in multi-threaded environments.
  * Disable only if using a thread-safe BLAS implementation (e.g., MKL).
  * @param serialize true to enable serialization, false to disable
  */
 void setSerializeBlasCalls(boolean serialize);

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
 long getShapeListSize(OpaqueShapeList list);
 int execCustomOp2(PointerPointer extraPointers, long hash, OpaqueContext opContext);
 long getVariablesSetSize(OpaqueVariablesSet set);
 int getVariablesSetStatus(OpaqueVariablesSet set);
 int getVariableId(OpaqueVariable variable);
 int getVariableIndex(OpaqueVariable variable);
 void deletePointerArray(Pointer pointer);
 void deleteCharArray(Pointer pointer);
 void deleteIntArray(Pointer pointer);
 void deleteLongArray(Pointer pointer);
 void deleteVariablesSet(OpaqueVariablesSet pointer);
 void deleteShapeList(Pointer shapeList);
 Pointer getGraphState(long id);
 void deleteGraphState(Pointer state);
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
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueNDArray getOutputArrayNative(org.nd4j.nativeblas.OpaqueContext ptr, int idx);
 @NoDeallocator
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
 void setGraphContextSArgument(org.nd4j.nativeblas.OpaqueContext ptr, String argument, int index);
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
 void clearLastError();
 void ctxShapeFunctionOverride(org.nd4j.nativeblas.OpaqueContext ptr, boolean reallyOverride);
 void ctxPurge(org.nd4j.nativeblas.OpaqueContext ptr);
 void ctxPurgeNoSync(org.nd4j.nativeblas.OpaqueContext ptr);
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
 void dbForceSyncToPrimary(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbForceSyncToSpecial(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbMigrate(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbAsyncCrossDeviceCopy(org.nd4j.nativeblas.OpaqueDataBuffer dstBuffer,
                             org.nd4j.nativeblas.OpaqueDataBuffer srcBuffer,
                             org.bytedeco.javacpp.Pointer dstStream);
 void dbTickHostRead(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickHostWrite(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickDeviceRead(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbTickDeviceWrite(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbExpand(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, long elements);
 void dbClose(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbFreeBuffersOnly(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbFreeBuffersOnStream(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, org.bytedeco.javacpp.Pointer stream);
 boolean dbIsOwner(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbCloseGetDiagnostics(org.bytedeco.javacpp.LongPointer outStats);
 void dbCloseResetDiagnostics();
 int dbDeviceId(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbSetDeviceId(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, int deviceId);
 int dbLocality(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueDataBuffer dbCreateView(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, long length);
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueDataBuffer dbAllocateDataBuffer(long elements, int dataType, boolean allocateBoth);
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special);

 /**
  * Create an externalized data buffer that is ALREADY marked as constant.
  * This is critical for preventing race conditions where the Java GC can
  * finalize the buffer before setConstant() is called.
  *
  * The constant flag is set IN NATIVE CODE before returning to Java,
  * eliminating the race window between buffer creation and marking constant.
  *
  * Use this instead of dbCreateExternalDataBuffer() + setConstant() for constant buffers.
  *
  * @param elements Number of elements
  * @param dataType Data type integer
  * @param primary Primary (host) pointer
  * @param special Special (device) pointer
  * @return Buffer that is already marked as constant and will never be deallocated
  */
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueDataBuffer dbCreateConstantExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special);

 /**
  * Set the constant flag on an OpaqueDataBuffer.
  * Constant buffers (like shape info) should never be freed by the deallocator.
  * This propagates the flag from Java to the native InteropDataBuffer.
  * @param dataBuffer The buffer to mark
  * @param isConstant True to mark as constant (will not be freed)
  * @return true if the flag was successfully set, false if the buffer was invalid
  *         (already closed or freed by GC). If false is returned, the buffer has
  *         been deallocated and should not be used.
  */
 boolean dbSetConstant(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, boolean isConstant);

 /**
  * Check if a buffer is marked as constant.
  * @param dataBuffer The buffer to check
  * @return True if the buffer is marked as constant
  */
 boolean dbIsConstant(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);

 // =====================================================
 // DSP Lifecycle Gates — see libnd4j/include/graph/DspLifecycleContext.h
 //
 // These let InferenceSession.executeCustomOp (the slot-by-slot path) ask
 // the native DSP layer whether its intended mutation (resync, close, tick)
 // will race an in-flight DSP capture or replay. When it would, the mutation
 // is deferred to DSP's own reconciliation path.
 // =====================================================

 /** True if this DataBuffer has one or more frozen DSP plan references. */
 boolean dbIsFrozenPlanRegistered(OpaqueDataBuffer dataBuffer);

 /** True if `dataBuffer` is DSP-protected (frozen plan ref OR constant flag). */
 boolean dbDspProtects(OpaqueDataBuffer dataBuffer);

 /** True iff the current thread is inside a DSP graph capture session. */
 boolean dspIsCaptureActive();

 /** True iff the current thread is executing under a DSP-owned stream. */
 boolean dspIsReplayActive();

 /** True iff DSP owns the current thread's execution (capture OR replay). */
 boolean dspIsOwned();

 /**
  * True iff the slot-by-slot path should skip issuing a pre-exec
  * dbSyncToSpecial on this buffer (DSP owns the thread + buffer is protected).
  */
 boolean dspShouldSkipResync(OpaqueDataBuffer dataBuffer);

 /**
  * True iff a slot-by-slot close() on this buffer must be deferred to avoid
  * freeing memory DSP still references.
  */
 boolean dspShouldDeferClose(OpaqueDataBuffer dataBuffer);

 /**
  * Process-wide DspExecutionMode.
  *   0 = LEGACY_UNAWARE (all gates become no-ops; bisect regressions)
  *   1 = COEXIST_SAFE (default; slot-by-slot defers state mutations that
  *       would race a live DSP capture or replay)
  *   2 = STRICT_ISOLATED (same as COEXIST_SAFE plus hard refusal to touch
  *       DSP-protected buffers while DSP owns the thread — test-only)
  */
 int  dspGetExecutionMode();
 void dspSetExecutionMode(int mode);

 void setShapeBuffer(LongPointer inputShapeData, int dt, LongPointer bufferToSet, char order, int elementWiseStride, boolean isEmpty, boolean isView);

 org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(long[] shapeInfo);


 org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(int rank, LongPointer shape, LongPointer strides, int dtype, char order, long ews, long extras);

 @NoDeallocator
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
 @NoDeallocator
 org.nd4j.nativeblas.OpaqueDataBuffer intermediateResultDataAt(int index, org.nd4j.nativeblas.OpaqueContext contextPointer);
 LongPointer intermediateResultShapeInfoAt(int index, org.nd4j.nativeblas.OpaqueContext contextPointer);
 String lastErrorMessage();
 org.nd4j.nativeblas.OpaqueLaunchContext defaultLaunchContext();
 String buildInfo();

 /**
  * Single-line canonical form of {@link #buildInfo()} for cache fingerprinting.
  * buildInfo() is a multi-line human-readable report; line-oriented consumers
  * (key=value sidecar files such as the DSP plan cache .meta) truncate
  * multi-line values on read, so fingerprints must use this form.
  * Default throws for backend bindings generated before this API existed —
  * callers fall back to flattening {@link #buildInfo()} themselves
  * (see DspPlanDiskCache.getNativeBuildFingerprint()).
  */
 default String buildInfoFingerprint() {
     throw new UnsupportedOperationException("buildInfoFingerprint not implemented in this backend");
 }

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
  * Call this early in JVM shutdown (e.g., from a shutdown hook)
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
  * NOTE: Will return early without action if setShapeCacheShutdownInProgress(true) was called.
  */
 void clearShapeCache();

 /**
  * Marks that shutdown is in progress for the shape cache.
  * When set to true, clearShapeCache() becomes a no-op to avoid segfaults
  * during JVM shutdown when buffers may still have external references.
  *
  * @param inProgress true to mark shutdown in progress, false otherwise
  */
 void setShapeCacheShutdownInProgress(boolean inProgress);

 /**
  * Check if shape cache shutdown is in progress.
  *
  * @return true if setShapeCacheShutdownInProgress(true) was called
  */
 boolean isShapeCacheShutdownInProgress();

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


 void recordJavaNDArrayAllocation(OpaqueNDArray array, long size, int dataType, boolean isView);

 void recordJavaNDArrayDeallocation(OpaqueNDArray array);

 void recordJavaDataBufferAllocation(OpaqueDataBuffer buffer, long size, int dataType, boolean isWorkspace);

 void recordJavaDataBufferDeallocation(OpaqueDataBuffer buffer);

 void recordJavaOpContextAllocation(OpaqueContext context, int nodeId, long fastpathInSize, long fastpathOutSize, long intermediateResultsSize, long handlesSize, boolean hasWorkspace, boolean isFastPath);

 void recordJavaOpContextDeallocation(OpaqueContext context);

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

 // ===============================
 // Op Timing Tracker API
 // Low-overhead op execution timing with phase breakdown,
 // histograms, and export capabilities
 // ===============================

 /**
  * Enable op timing tracker.
  * @param enabled 1 to enable, 0 to disable
  * @param detailed 1 for phase-level timing, 0 for total-only
  */
 void setOpTimingEnabled(int enabled, int detailed);

 /**
  * Enable op timing with Chrome trace export.
  * @param detailed 1 for phase-level timing, 0 for total-only
  */
 void setOpTimingEnabledWithTrace(int detailed);

 /**
  * Check if op timing is enabled.
  * @return 1 if enabled, 0 if disabled
  */
 int isOpTimingEnabled();

 /**
  * Flush ring buffer to aggregated stats.
  * Call before reading any stats.
  */
 void flushOpTiming();

 /**
  * Print top N ops by total time to stdout.
  * @param topN number of ops to show
  */
 void printOpTimingStats(int topN);

 /**
  * Print phase breakdown for a specific op to stdout.
  * @param opName name of the op to analyze
  */
 void printOpTimingBreakdown(String opName);

 /**
  * Print timing histogram for a specific op to stdout.
  * @param opName name of the op to analyze
  */
 void printOpTimingHistogram(String opName);

 /**
  * Print per-thread timing statistics to stdout.
  */
 void printOpTimingThreadStats();

 /**
  * Reset all timing data.
  */
 void resetOpTiming();

 /**
  * Get number of unique ops tracked.
  * @return number of unique ops
  */
 int getOpTimingNumOps();

 /**
  * Get total number of op executions tracked.
  * @return total execution count
  */
 long getOpTimingTotalExecutions();

 /**
  * Export timing data to Chrome trace JSON format.
  * Visualizable in chrome://tracing or Perfetto.
  * @param filename output file path
  * @return 1 on success, 0 on failure
  */
 int exportOpTimingChromeTrace(String filename);

 /**
  * Export timing data to CSV format.
  * @param filename output file path
  * @return 1 on success, 0 on failure
  */
 int exportOpTimingCSV(String filename);

 // ========================
 // Workspace Management API
 // ========================

 /**
  * Create a native workspace for bump-allocated memory.
  * @param initialSize initial allocation in bytes
  * @return opaque workspace pointer
  */
 Pointer createNativeWorkspace(long initialSize);

 /**
  * Destroy a native workspace and free all memory.
  */
 void destroyNativeWorkspace(Pointer workspace);

 /**
  * Enter workspace scope - resets offsets for new allocation cycle.
  */
 void workspaceScopeIn(Pointer workspace);

 /**
  * Exit workspace scope - resets offsets, making memory reusable.
  */
 void workspaceScopeOut(Pointer workspace);

 /**
  * Attach a workspace to an op execution context.
  */
 void attachWorkspaceToContext(OpaqueContext ctx, Pointer workspace);

 /**
  * Detach workspace from context.
  */
 void detachWorkspaceFromContext(OpaqueContext ctx);

 /**
  * Get the current used size (offset) of the workspace.
  */
 long getWorkspaceCurrentOffset(Pointer workspace);

 /**
  * Get the total allocated size of the workspace.
  */
 long getWorkspaceAllocatedSize(Pointer workspace);

 // ========================
 // Multi-Backend Workspace API
 // ========================

 /**
  * Create a multi-backend workspace with device awareness.
  */
 Pointer createNativeMultiBackendWorkspace(long initialSize, int primaryDeviceType, int primaryDeviceIndex);

 /**
  * Destroy a multi-backend workspace.
  */
 void destroyNativeMultiBackendWorkspace(Pointer handle);

 /**
  * Allocate bytes from multi-backend workspace on primary device.
  */
 Pointer nativeMbwAllocateBytes(Pointer handle, long numBytes);

 /**
  * Multi-backend workspace scope in.
  */
 void nativeMbwScopeIn(Pointer handle);

 /**
  * Multi-backend workspace scope out.
  */
 void nativeMbwScopeOut(Pointer handle);

 /**
  * Transfer data between devices in multi-backend workspace.
  */
 void nativeMbwTransferTo(Pointer handle, int srcDeviceType, int srcDeviceIndex,
                          int dstDeviceType, int dstDeviceIndex);

 /**
  * Get coherence state for a device.
  */
 int nativeMbwGetCoherenceState(Pointer handle, int deviceType, int deviceIndex);

 /**
  * Get total allocated size across all devices.
  */
 long nativeMbwGetTotalAllocatedSize(Pointer handle);

 /**
  * Allocate bytes on a specific device in a multi-backend workspace.
  * @param handle multi-backend workspace handle
  * @param numBytes bytes to allocate
  * @param deviceType device type (0=CPU, 1=CUDA)
  * @param deviceIndex device index
  * @return pointer to allocated memory, or null on failure
  */
 default Pointer nativeMbwAllocateBytesOnDevice(Pointer handle, long numBytes, int deviceType, int deviceIndex) {
     throw new UnsupportedOperationException("nativeMbwAllocateBytesOnDevice not implemented in this backend");
 }

 /**
  * Sync a specific device (e.g., cudaDeviceSynchronize for CUDA devices).
  */
 default void nativeMbwSyncDevice(Pointer handle, int deviceType, int deviceIndex) {
     throw new UnsupportedOperationException("nativeMbwSyncDevice not implemented in this backend");
 }

 /**
  * Sync all devices in the workspace.
  */
 default void nativeMbwSyncAllDevices(Pointer handle) {
     throw new UnsupportedOperationException("nativeMbwSyncAllDevices not implemented in this backend");
 }

 /**
  * Get the allocated size on a specific device.
  */
 default long nativeMbwGetAllocatedSizeOnDevice(Pointer handle, int deviceType, int deviceIndex) {
     throw new UnsupportedOperationException("nativeMbwGetAllocatedSizeOnDevice not implemented in this backend");
 }

 /**
  * Get the current workspace offset on the primary device.
  */
 default long nativeMbwGetCurrentOffset(Pointer handle) {
     throw new UnsupportedOperationException("nativeMbwGetCurrentOffset not implemented in this backend");
 }

 // ─── Native Graph Executor (DynamicShapePlan) ────────────────────────────

 /**
  * Create a native plan cache. The returned handle owns the lifetimes of all
  * shape-keyed plan handles dispatched from it. Call {@link #freeNativePlanCache(Pointer)}
  * to release. Usually one cache per SameDiff instance.
  */
 default Pointer createNativePlanCache() {
     throw new UnsupportedOperationException("createNativePlanCache not implemented in this backend");
 }

 /**
  * Free a native plan cache, releasing all compiled plan handles it owns and
  * all associated GPU memory (capture buffers, workspaces, replay state).
  */
 default void freeNativePlanCache(Pointer cache) {
     throw new UnsupportedOperationException("freeNativePlanCache not implemented in this backend");
 }

 /**
  * Clear the cache's in-memory map of shape-keyed plan handles WITHOUT freeing
  * the cache itself. Typically called when the graph structure changes so stale
  * shape-keyed handles don't linger.
  */
 default void clearNativePlanCacheHandle(Pointer cache) {
     throw new UnsupportedOperationException("clearNativePlanCacheHandle not implemented in this backend");
 }

 /**
  * Dispatch a serialized DynamicShapePlan against a set of placeholder shapes.
  * Looks up (or compiles on miss) a handle keyed by the placeholder shape signature
  * and returns it. The returned handle is owned by the cache and must not be freed
  * by the caller; it remains valid until the cache is cleared or freed.
  *
  * @param cache            cache handle from {@link #createNativePlanCache()}
  * @param planBytes        pointer to the serialized plan bytes
  * @param planBytesLen     size of the serialized plan in bytes
  * @param outputNames      {@code char**} of null-terminated output variable names
  * @param numOutputs       number of output names
  * @param phShapeInfoPtrs  {@code LongType**} of placeholder shape-info pointers
  *                         (e.g., from ConstantShapeHelper), in the order declared by the plan
  * @param numPlaceholders  number of placeholder shape-info pointers
  * @return handle to the compiled plan for this shape signature (cache-owned), or null on failure
  */
 default Pointer dispatchNativePlan(Pointer cache,
                                    Pointer planBytes, long planBytesLen,
                                    Pointer outputNames, long numOutputs,
                                    Pointer phShapeInfoPtrs, long numPlaceholders,
                                    int graphExecutionMode, int newBorrower) {
     throw new UnsupportedOperationException("dispatchNativePlan not implemented in this backend");
 }

 /**
  * Unpin a plan handle, making it eligible for LRU eviction.
  * Must be called when a Java executor swaps to a different plan handle
  * or when the executor is closed. Paired with the automatic pinning
  * done by {@link #dispatchNativePlan}.
  *
  * @param cacheHandle cache from createNativePlanCache (non-null)
  * @param planHandle  plan handle from dispatchNativePlan (null is safe — no-op)
  */
 default void unpinNativePlan(Pointer cacheHandle, Pointer planHandle) {
     throw new UnsupportedOperationException("unpinNativePlan not implemented in this backend");
 }

 /**
  * Mark the native plan cache subsystem as shutting down.
  * When set, unpinPlan() and plan eviction skip plan deletion entirely —
  * the OS reclaims all memory on process exit.
  *
  * This prevents SIGSEGV from CUDA API calls (cudaStreamDestroy, cudaFree,
  * cudaGraphExecDestroy) racing with JVM shutdown tearing down the CUDA context.
  *
  * Must be called from the JVM shutdown hook BEFORE DeallocatorService shutdown.
  *
  * @param inProgress true to mark shutdown in progress, false otherwise
  */
 default void setPlanCacheShutdownInProgress(boolean inProgress) {
     // No-op default — backends that don't have native plan caches can ignore.
 }

 /**
  * Compile a serialized DynamicShapePlan into a native C++ executor.
  * @param serializedPlan pointer to the serialized plan bytes
  * @param planSize size of the serialized plan in bytes
  * @return opaque handle to the compiled plan, or null on failure
  * @deprecated Use {@link #dispatchNativePlan(Pointer, Pointer, long, PointerPointer, int, PointerPointer, int)}
  *             with a native plan cache instead — shape-keyed dispatch is required to avoid
  *             in-plan slot mutation across differing input shapes.
  */
 @Deprecated
 default Pointer compileDynamicShapePlan(Pointer serializedPlan, long planSize) {
     throw new UnsupportedOperationException("compileDynamicShapePlan not implemented in this backend");
 }

 /**
  * Execute a compiled native plan.
  * @param planHandle handle from compileDynamicShapePlan()
  * @param opContext OpaqueContext with inputs/outputs set via setGraphContextInputArray/setGraphContextOutputArray
  * @param stream backend execution stream pointer, or null for the backend default
  * @return 0 on success, non-zero on failure
  */
 default int executeDynamicShapePlan(Pointer planHandle,
                                     OpaqueContext opContext,
                                     Pointer stream) {
     throw new UnsupportedOperationException("executeDynamicShapePlan not implemented in this backend");
 }

 /**
  * Free a compiled native plan.
  * @param planHandle handle from compileDynamicShapePlan()
  */
 default void freeDynamicShapePlan(Pointer planHandle) {
     throw new UnsupportedOperationException("freeDynamicShapePlan not implemented in this backend");
 }

 /**
  * Clear shape caches in a compiled plan. Must be called on session reset.
  * @param planHandle handle from compileDynamicShapePlan()
  */
 default void clearDynamicShapePlanCaches(Pointer planHandle) {
     throw new UnsupportedOperationException("clearDynamicShapePlanCaches not implemented in this backend");
 }

 /**
  * Force-clear ALL shape caches unconditionally (including static slots).
  * @param planHandle handle from compileDynamicShapePlan()
  */
 default void clearAllDynamicShapePlanCachesForce(Pointer planHandle) {
     throw new UnsupportedOperationException("clearAllDynamicShapePlanCachesForce not implemented in this backend");
 }

 /**
  * Release all GPU memory held by intermediate computation results while keeping
  * the plan structure alive. Frees CUDA graph replay handles and associated
  * replay resources,
  * cuBLAS workspace, and non-weight output slot NDArrays. The plan enters a
  * "cold" state and will re-warm on the next execute() call.
  *
  * @param planHandle handle from compileDynamicShapePlan()
  * @return the number of intermediate NDArrays freed
  */
 default int releaseGpuIntermediates(Pointer planHandle) {
     throw new UnsupportedOperationException("releaseGpuIntermediates not implemented in this backend");
 }

/**
  * Get constant cache bytes for a specific device.
  * @param deviceId the device ID
  * @return number of bytes cached for constants
  */
 default long getConstantCacheBytes(int deviceId) { return 0L; }

 /**
  * Get TAD cache entry count.
  * @return number of cached TAD entries
  */
 default long getTadCacheEntries() { return 0L; }

 /**
  * Get TAD cache memory usage in bytes.
  * @return number of bytes used by TAD cache
  */
 default long getTadCacheBytes() { return 0L; }

 /**
  * Clear constant cache for all devices.
  */
 default void clearConstantCache() {}

 /**
  * Clear TAD cache.
  */
 default void clearTadCache() {}

 /**
  * Returns true if Triton backend support is compiled and available at runtime.
  */
 default boolean isTritonAvailable() {
     return false;
 }

 /**
  * Get the total number of Triton kernel launches since the backend was initialized.
  * Returns 0 if Triton is not available. Used by tests to verify Triton execution.
  */
 default long getTritonKernelLaunchCount() {
     return 0;
 }

 /**
  * Get the total number of Triton PTX cache hits since the backend was initialized.
  * Returns 0 if Triton is not available.
  */
 default long getTritonCacheHitCount() {
     return 0;
 }

 /**
  * Reset all Triton execution counters to zero.
  */
 default void resetTritonCounters() {}

 /**
  * Invalidate all cached Triton compiled kernels (frees CUmodule GPU memory).
  */
 default void invalidateTritonCache() {}

 /**
  * Export the Triton kernel disk cache to a shareable .tkcache bundle (STORED ZIP).
  * @param outputPath path to write the bundle file
  * @return number of kernels exported, or negative on error
  */
 default int exportTritonCacheBundle(String outputPath) { return -1; }

 /**
  * Import a .tkcache bundle into the Triton override directory.
  * Kernels from bundles take priority over on-disk cache.
  * @param bundlePath path to the bundle file
  * @param validateArch if true, reject bundles for incompatible GPU architectures
  * @return number of kernels imported, -1 on error, -2 on arch mismatch
  */
 default int importTritonCacheBundle(String bundlePath, boolean validateArch) { return -1; }

 /**
  * Read and return the manifest JSON from a .tkcache bundle without importing.
  * @param bundlePath path to the bundle file
  * @return JSON string with bundle manifest, or error JSON
  */
 default String inspectTritonCacheBundle(String bundlePath) {
     return "{\"error\": \"not implemented\"}";
 }

 /**
  * Load a model from an SDZ or SDNB file entirely in C++.
  * @param filePath path to the .sdz or .sdnb file
  * @return opaque handle to the loaded model, or null on failure
  */
 default Pointer loadModelFromFile(String filePath) {
     throw new UnsupportedOperationException("loadModelFromFile not implemented in this backend");
 }

 /**
  * Compile a loaded model into a native execution plan.
  * @param modelHandle handle from loadModelFromFile()
  * @param requestedOutputNames pointer to array of C strings (const char**)
  * @param numOutputs number of requested outputs
  * @return opaque plan handle, or null on failure
  */
 default Pointer compileModelPlan(Pointer modelHandle, Pointer requestedOutputNames, int numOutputs) {
     throw new UnsupportedOperationException("compileModelPlan not implemented in this backend");
 }

 /**
  * Convenience overload that converts String[] to native pointer array.
  */
 default Pointer compileModelPlan(Pointer modelHandle, String[] requestedOutputNames, int numOutputs) {
     org.bytedeco.javacpp.PointerPointer<org.bytedeco.javacpp.BytePointer> pp =
         new org.bytedeco.javacpp.PointerPointer<>(numOutputs);
     for (int i = 0; i < numOutputs; i++) {
         pp.put(i, new org.bytedeco.javacpp.BytePointer(requestedOutputNames[i]));
     }
     return compileModelPlan(modelHandle, (Pointer) pp, numOutputs);
 }

 /**
  * Free a loaded model.
  * @param modelHandle handle from loadModelFromFile()
  */
 default void freeLoadedModel(Pointer modelHandle) {
     throw new UnsupportedOperationException("freeLoadedModel not implemented in this backend");
 }

 /**
  * Get the number of external inputs required by a compiled plan.
  * @param planHandle handle from compileDynamicShapePlan()
  * @return number of external inputs, or -1 on error
  */
 default int getPlanNumExternalInputs(Pointer planHandle) {
     throw new UnsupportedOperationException("getPlanNumExternalInputs not implemented in this backend");
 }

 /**
  * Get the number of requested outputs in a compiled plan.
  * @param planHandle handle from compileDynamicShapePlan()
  * @return number of requested outputs, or -1 on error
  */
 default int getPlanNumRequestedOutputs(Pointer planHandle) {
     throw new UnsupportedOperationException("getPlanNumRequestedOutputs not implemented in this backend");
 }

 /**
  * Get the number of slots (ops) in a compiled plan.
  * @param planHandle handle from compileDynamicShapePlan()
  * @return number of slots, or -1 on error
  */
 default int getPlanNumSlots(Pointer planHandle) {
     throw new UnsupportedOperationException("getPlanNumSlots not implemented in this backend");
 }

 // ── External input introspection ─────────────────────────────────────────

 /** True if ext[extIdx] is classified as variable (participates in staging D2D). */
 default boolean getPlanIsExternalInputVariable(Pointer planHandle, int extIdx) {
     return false;
 }

 /** True if ext[extIdx] is classified as placeholder (forces H2D sync). */
 default boolean getPlanIsExternalInputPlaceholder(Pointer planHandle, int extIdx) {
     return false;
 }

 /** Count of external inputs currently classified as variable. */
 default int getPlanNumVariableExternalInputs(Pointer planHandle) {
     return 0;
 }

 /** Device address of the plan-owned staging buffer for ext[extIdx], or 0 if none. */
 default long getPlanStagingBufferAddress(Pointer planHandle, int extIdx) {
     return 0;
 }

 /** Device address the graph will read from for ext[extIdx] (staging or original). */
 default long getPlanEffectiveExternalAddress(Pointer planHandle, int extIdx) {
     return 0;
 }

 /** Number of variable ext inputs with allocated staging buffers. */
 default int getPlanNumStagingBuffers(Pointer planHandle) {
     return 0;
 }

 /** Current plan execution count. */
 default int getPlanExecuteCount(Pointer planHandle) {
     return -1;
 }

 /** Number of cached variable ext input indices (fast-path list). */
 default int getPlanNumCachedVariableExtIndices(Pointer planHandle) {
     return -1;
 }

 /** Get the i-th cached variable ext input index. Returns -1 if out of range. */
 default int getPlanCachedVariableExtIndex(Pointer planHandle, int i) {
     return -1;
 }

 /** Get the staging buffer NDArray for ext[extIdx]. Returns null if none. */
 default OpaqueNDArray getPlanStagingBufferArray(Pointer planHandle, int extIdx) {
     return null;
 }

 /**
  * Atomically copy staging buffer content for ext[extIdx] into dstBuffer.
  * Avoids the stale-pointer race of extracting specialBuffer() then copying separately.
  * Returns: 0=success, -1=no staging, -2=invalid staging, -3=copy failed.
  */
 default int copyPlanStagingToBuffer(Pointer planHandle, int extIdx, OpaqueDataBuffer dstBuffer) {
     return -1;
 }

 /** Device address of the last externalArrays[extIdx] passed to execute(). */
 default long getPlanLastExternalInputAddress(Pointer planHandle, int extIdx) {
     return 0;
 }

 /**
  * Enable or disable CUDA Graphs for a compiled plan.
  * @param planHandle handle from compileDynamicShapePlan()
  * @param enabled true to enable CUDA Graph capture/replay
  */
 default void setPlanCudaGraphsEnabled(Pointer planHandle, boolean enabled) {
     // No-op on backends that don't support CUDA Graphs
 }

 /**
  * Set the minimum segment size for CUDA graph capture.
  * Segments smaller than this are executed slot-by-slot. Default: 10.
  * @param planHandle handle from compileDynamicShapePlan()
  * @param minSize minimum number of slots (clamped to >=1)
  */
 default void setPlanMinCaptureSegmentSize(Pointer planHandle, int minSize) {
     // No-op on backends that don't support CUDA Graphs
 }

 /**
  * Set maximum segment size for CUDA graph capture to prevent OOM during capture.
  * @param planHandle handle from compileDynamicShapePlan()
  * @param maxSize maximum slots per capture segment (0 for unlimited)
  */
 default void setPlanMaxCaptureSegmentSize(Pointer planHandle, int maxSize) {
     // No-op on backends that don't support CUDA Graphs
 }

 /**
  * Enable/disable "shapes frozen" mode for a compiled plan.
  * When frozen, shape inference and cache clearing are skipped between executions.
  * Use during static KV decode where external input shapes are guaranteed constant.
  * @param planHandle handle from compileDynamicShapePlan()
  * @param frozen true to enable, false to disable
  */
 default void setPlanShapesFrozen(Pointer planHandle, boolean frozen) {
     // No-op by default
 }

  /**
   * Enable/disable shape-only dry-run mode for a compiled plan.
   * When enabled, executeSlot() runs all dispatch infrastructure (shape caching,
   * frozen detection, output allocation, segment dispatch) but skips op kernel execution.
   * Use to measure pure dispatch/infrastructure overhead separately from compute.
   * @param planHandle handle from compileDynamicShapePlan()
   * @param enabled true to enable shape-only mode, false to disable
   */
  default void setPlanShapeOnlyMode(Pointer planHandle, boolean enabled) {
      // No-op by default
  }

  /**
   * Enable/disable execution timing breakdown logging for a compiled plan.
   * @param planHandle handle from compileDynamicShapePlan()
   * @param enabled true to enable timing, false to disable
   */
  default void setPlanExecutionTimingEnabled(Pointer planHandle, boolean enabled) {
      // No-op by default
  }

  /**
   * Enable/disable trace logging for DSP execution decisions.
   * @param planHandle handle from compileDynamicShapePlan()
   * @param enabled true to enable trace, false to disable
   */
  default void setPlanTraceEnabled(Pointer planHandle, boolean enabled) {
      // No-op by default
  }

  /**
   * Set the JIT compilation mode for DSP segment execution.
   * @param planHandle handle from compileDynamicShapePlan()
   * @param mode 0 = GRAPH_ONLY (default), 1 = JIT_ONLY, 2 = GRAPH_PLUS_JIT
   */
  default void setPlanJitMode(Pointer planHandle, int mode) {
      // No-op on backends that don't support JIT compilation
  }

  /**
   * Set the graph execution mode for DSP execution.
   * Controls which backend is used for segment execution.
   * @param planHandle handle from compileDynamicShapePlan()
   * @param mode 0=AUTO, 1=SLOT_BY_SLOT, 2=CUDA_GRAPHS, 3=NVRTC_JIT, 4=PTX_JIT,
   *             5=TRITON, 6=MLX, 7=ARM_HYBRID, 8=NNAPI, 9=HIP_GRAPHS,
   *             10=LEVEL_ZERO, 11=VULKAN, 12=METAL, 13=TPU, 14=HEXAGON,
   *             15=OPENVINO, 16=TVM (deprecated), 17=EMULATED_REPLAY,
   *             18=SHAPE_INFERENCE_ONLY, 19=PORTABLE_REPLAY
   */
  default void setPlanGraphExecutionMode(Pointer planHandle, int mode) {
      // No-op on backends that don't support graph execution mode
  }

  /**
   * Set maximum sizes for specific output slots (KV cache pre-allocation).
   * When set, these slots will be pre-allocated at the specified maximum size,
   * keeping buffer addresses stable across all subsequent steps.
   * This enables CUDA graph capture for models with growing KV caches.
   * 
   * @param planHandle handle from compileDynamicShapePlan()
   * @param numSlots number of slot entries
   * @param slotIndicesBuffer pointer to int[] of output slot indices
   * @param maxSizesBuffer pointer to long[] of maximum sizes (in number of elements)
   */
  default void setPlanOutputSlotMaxSizes(Pointer planHandle, long numSlots,
                                          Pointer slotIndicesBuffer, Pointer maxSizesBuffer) {
      // No-op by default
  }

  /**
   * Typed overload: accepts primitive arrays directly, matching the native binding signatures.
   */
  default void setPlanOutputSlotMaxSizes(Pointer planHandle, long numSlots,
                                          int[] slotIndices, long[] maxSizes) {
      // No-op by default
  }

  /**
   * Configure plan-managed KV scatter for CUDA-graph-compatible decode loops.
   *
   * After this call the plan executes a batched KV scatter after each execute(),
   * updating static KV buffers at the current position and incrementing the
   * position scalar. Eliminates the Java-side scatterNewEntries() round-trip.
   *
   * @param planHandle          handle from compileDynamicShapePlan()
   * @param presentSlotIndices  int[] of output slot indices for present KV tensors
   * @param staticKvBufferPtrs  Pointer[] of NDArray native handles for static KV buffers
   * @param numPairs            number of (present, static) pairs
   * @param dtypeInt            DataType code of the KV tensors
   * @param heads               number of attention heads
   * @param srcSeqLen           present sequence length (typically 1 for decode)
   * @param dstSeqLen           static buffer sequence length (= maxKvLen)
   * @param dim                 head dimension
   * @param kvPositionPtr       LongPointer to device-accessible int64 position scalar
   */
  default void configurePlanKvScatter(Pointer planHandle,
                                       int[] presentSlotIndices,
                                       Pointer[] staticKvBufferPtrs,
                                       long numPairs,
                                       int dtypeInt,
                                       long heads,
                                       long srcSeqLen,
                                       long dstSeqLen,
                                       long dim,
                                       LongPointer kvPositionPtr) {
      // No-op by default
  }

  /**
   * Reset the KV cache position managed by the plan (e.g., after prefill).
   *
   * @param planHandle  handle from compileDynamicShapePlan()
   * @param position    new cache position value
   */
  default void resetPlanKvCachePosition(Pointer planHandle, long position) {
      // No-op by default
  }

  /**
   * Get the current KV cache position managed by the plan.
   *
   * @param planHandle  handle from compileDynamicShapePlan()
   * @return current position, or -1 if KV scatter not configured
   */
  default long getPlanKvCachePosition(Pointer planHandle) {
      return -1L;
  }

  /**
   * Get the number of graph segments in a compiled plan.
   * @param planHandle handle from compileDynamicShapePlan()
   * @return number of segments, or -1 on error
   */
  default int getPlanNumSegments(Pointer planHandle) {
     return -1;
 }

 /**
  * Get the number of segments captured as CUDA graphs.
  * @param planHandle handle from compileDynamicShapePlan()
  * @return number of captured segments
  */
 default int getPlanNumCapturedGraphSegments(Pointer planHandle) {
     return 0;
 }

 /**
  * Get the total number of CUDA graph replays across all segments.
  * @param planHandle handle from compileDynamicShapePlan()
  * @return total replay count
  */
 default int getPlanTotalGraphReplays(Pointer planHandle) {
     return 0;
 }

 /**
  * Validate that the captured CUDA graph covers all ops in the plan segment.
  * Must be called AFTER at least one execution with CUDA graphs enabled
  * and debug/verbose mode active (so the capture audit was collected).
  *
  * @param planHandle handle from compileDynamicShapePlan()
  * @return true if all ops contributed at least one CUDA graph node,
  *         false if any ops are host-only (their work won't replay)
  */
 default boolean validatePlanCapturedGraph(Pointer planHandle) {
     return true;  // Non-CUDA backends always "pass"
 }

 /**
  * Get the number of host-only ops detected during the last CUDA graph capture.
  * Host-only ops do work during capture but produce zero CUDA graph nodes,
  * meaning their results will be STALE on graph replay.
  *
  * @param planHandle handle from compileDynamicShapePlan()
  * @return count of host-only ops, or 0 if no audit data
  */
 default int getPlanNumHostOnlyOps(Pointer planHandle) {
     return 0;
 }

 /**
  * Get the names of host-only ops from the last CUDA graph capture audit,
  * as a pipe-delimited string (e.g. "shape_of|reduce_sum").
  *
  * @param planHandle handle from compileDynamicShapePlan()
  * @return pipe-delimited op names, or empty string if none
  */
 default String getPlanHostOnlyOpNames(Pointer planHandle) {
     return "";
 }

 /**
  * Print the full CUDA graph contents and capture audit to stderr.
  * Shows per-node details (kernel names, memcpy sizes) and per-op
  * CUDA node contribution. Flags host-only ops with warnings.
  *
  * @param planHandle handle from compileDynamicShapePlan()
  */
 default void printPlanCapturedGraphDebug(Pointer planHandle) {
     // No-op on non-CUDA backends
 }

  /**
   * Get detailed capture statistics as a formatted string.
   * Returns: "captured=N|oomRetrying=N|permFailed=N|nonCapt=N|tooSmall=N|addrUnstable=N"
   *
   * @param planHandle  Handle from compileDynamicShapePlan()
   * @return Thread-local static buffer with stats string
   */
  default String getPlanCaptureStats(Pointer planHandle) {
      return "";
  }

  // =============================================================================
  // Per-Segment Replay State
  // =============================================================================

  default int getPlanSegmentReplayState(Pointer planHandle, int segmentIdx) { return -1; }
  default int getPlanSegmentReplayCount(Pointer planHandle, int segmentIdx) { return 0; }
  default String getPlanSegmentBackendName(Pointer planHandle, int segmentIdx) { return ""; }
  default String getPlanSegmentStatisticsJson(Pointer planHandle, int segmentIdx) { return ""; }
  default int getPlanSegmentExecutionCount(Pointer planHandle, int segmentIdx) { return 0; }
  /**
   * Get the current execution phase for a segment.
   * Returns the ExecutionPhase ordinal: 0=WARMUP, 1=COMPILING, 2=COMPILED, 3=REPLAYING, 4=SLOT_BY_SLOT.
   * Returns -1 if the plan handle or segment index is invalid.
   */
  default int getPlanSegmentExecutionPhase(Pointer planHandle, int segmentIdx) { return -1; }

  /**
   * Get the plan-level phase.
   * Returns PlanPhase as int: 0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING
   */
  default int getPlanPhase(Pointer planHandle) { return -1; }

  /**
   * Get the replay schedule signature hash for a segment.
   * Returns the FNV-1a hash encoding of the ordered replay unit list.
   * Zero if the segment has no replay.
   */
  long getPlanReplaySignatureHash(Pointer planHandle, int segIdx);

  /**
   * Get the number of replay units for a segment after consolidation.
   * Returns the count of ordered replay units. Zero if no replay.
   */
  int getPlanReplayUnitCount(Pointer planHandle, int segIdx);

  /**
   * Get the number of GAP units in the segment's composite replay schedule.
   * Returns 0 if the segment has no composite schedule.
   */
  default int getPlanSegmentGapUnitCount(Pointer planHandle, int segIdx) { return 0; }

  /**
   * Get the number of TRITON_ISLAND units in the segment's composite replay schedule.
   * Returns 0 if the segment has no composite schedule.
   */
  default int getPlanSegmentIslandUnitCount(Pointer planHandle, int segIdx) { return 0; }

  /**
   * Get total gap slot count (sum of all gap unit ranges) in the segment.
   * Returns 0 if no gaps.
   */
  default int getPlanSegmentGapSlotCount(Pointer planHandle, int segIdx) { return 0; }

  /**
   * Check if a specific island's composite replay handle is ready.
   * @param islandIdx  Island index (0-based)
   * @return 1=ready, 0=not ready or null, -1=invalid index
   */
  default int getPlanSegmentIslandHandleReady(Pointer planHandle, int segIdx, int islandIdx) { return -1; }

  /**
   * Get the replay mode for this segment.
   * Returns: 0=NONE, 1=MONOLITHIC, 2=COMPOSITE, 3=SLOT_BY_SLOT
   */
  default int getPlanSegmentReplayMode(Pointer planHandle, int segIdx) { return 0; }

  /**
   * Check if the monolithic replay handle is ready.
   * Returns 1=ready, 0=not ready, -1=invalid.
   */
  default int getPlanSegmentMonolithicHandleReady(Pointer planHandle, int segIdx) { return -1; }

  /**
   * Get the execution count for a specific segment.
   * Returns -1 on error.
   */
  int getSegmentExecutionCount(Pointer planHandle, int segIdx);

  /**
   * Get the number of segments in the plan.
   * Returns -1 on error.
   */
  int getPlanSegmentCount(Pointer planHandle);

  /**
   * Check whether the plan's compilation has been sealed (first phaseCompile()
   * finished). Returns 1 if sealed, 0 if not, -1 if the handle is invalid.
   */
  default int isPlanCompilationSealed(Pointer planHandle) { return -1; }

  /**
   * Count of compileSegment() calls that occurred AFTER the plan was sealed.
   * Any value &gt; 0 indicates a mid-execution recompile — a correctness red
   * flag because it breaks freeze/capture invariants.
   * Returns -1 if the handle is invalid.
   */
  default long getPlanMidExecutionCompileCount(Pointer planHandle) { return -1L; }

  /**
   * Reset the mid-execution compile counter to zero. Used by tests that want
   * to bracket a section and assert zero recompiles happened inside it.
   */
  default void resetPlanMidExecutionCompileCount(Pointer planHandle) { /* no-op default */ }

  // ═══���═══════════════════════════════════════════════════════════════════════
  // FrozenPlan API — new unified execution entry point (Step 6 hierarchy)
  // ════════════════════════════���══════════════════════════════════════════════

  /**
   * Execute a FrozenPlan. This is the single entry point for DSP execution
   * in the new hierarchy. Internally handles warmup, compilation, capture,
   * and replay phases automatically.
   *
   * @param planHandle handle from dispatchNativePlan() (internally a FrozenPlan*)
   * @param opContext OpaqueContext with inputs/outputs set
   * @param stream backend execution stream pointer, or null for the backend default
   * @return 0 on success, non-zero on failure
   */
  default int executeFrozenPlan(Pointer planHandle,
                                OpaqueContext opContext,
                                Pointer stream) {
      // Delegates to existing executeDynamicShapePlan for backward compatibility
      return executeDynamicShapePlan(planHandle, opContext, stream);
  }

  /**
   * Check whether a FrozenPlan is sealed (all segments completed build phase
   * and are in steady-state replay).
   *
   * @param planHandle plan handle
   * @return 1 if sealed, 0 if still building, -1 if invalid handle
   */
  default int isFrozenPlanSealed(Pointer planHandle) {
      return isPlanCompilationSealed(planHandle);
  }

  /**
   * Get the build pass count for a FrozenPlan.
   * 0 = needs warmup, 1 = needs compile/capture, 2+ = sealed (replay only).
   *
   * @param planHandle plan handle
   * @return build pass count, or -1 if invalid handle
   */
  default int getFrozenPlanBuildPassCount(Pointer planHandle) { return -1; }

  /**
   * Get the segment executor phase for a specific segment in a FrozenPlan.
   * Returns: 0=BUILDING, 1=SEALED, 2=FAILED, -1=invalid.
   *
   * @param planHandle plan handle
   * @param segIdx segment index
   * @return phase code
   */
  default int getSegmentExecutorPhase(Pointer planHandle, int segIdx) { return -1; }

  // ════��══════════════════════════���═══════════════════════════════════════════

  /**
   * Check if all buffer pointers are stable (same addresses across executions).
   * Returns 1 if stable, 0 if not, -1 if invalid handle.
   */
  default int getPlanPointersStable(Pointer planHandle) { return -1; }

  /**
   * Get the number of executions since shapes were frozen.
   * Returns -1 if shapes are not frozen.
   */
  default int getPlanFrozenExecutionCount(Pointer planHandle) { return -1; }

  /**
   * Get the slot state for a specific slot.
   * Returns SlotState as int: 0=UNINITIALIZED, 1=WARMUP, 2=SHAPE_CACHED,
   *   3=COMPILED, 4=FROZEN, 5=FROZEN_CONSTANT. Returns -1 if invalid.
   */
  default int getPlanSlotState(Pointer planHandle, int slotIdx) { return -1; }

  /** Get the op name for a specific slot. */
  default String getPlanSlotOpName(Pointer planHandle, int slotIdx) { return ""; }

  /**
   * Get per-slot flags as a bitmask. See NativeOps.h for bit definitions.
   * bit 0: viewCapable, 1: dataDependent, 2: shapeDependsOnValues,
   * 3: identity, 4: inPlaceFused, 5: fusedChainHead, 6: fusedChainTail,
   * 7: needsZeroedOutput, 8: needsIntLongSync, 9: shapeStatic, 10: frozenConstant
   */
  default int getPlanSlotFlags(Pointer planHandle, int slotIdx) { return -1; }

  /**
   * Get the monotonic write-generation counter for a slot.
   * Incremented on kernel dispatch, prezero, nullify, fused chain emit, view install.
   * Returns -1 if invalid.
   */
  default int getPlanSlotGeneration(Pointer planHandle, int slotIdx) { return -1; }

  default boolean isPlanSegmentCapturable(Pointer planHandle, int segmentIdx) { return false; }
  default boolean isPlanSegmentCaptureFailed(Pointer planHandle, int segmentIdx) { return false; }

  // =============================================================================
  // Per-execution stats from last PlanExecutionContext
  // =============================================================================

  /** Segment stats from last execute(). Returns -1 if no context available. */
  default int getLastExecSegmentsWarmup(Pointer planHandle) { return -1; }
  default int getLastExecSegmentsCaptured(Pointer planHandle) { return -1; }
  default int getLastExecSegmentsReplayed(Pointer planHandle) { return -1; }
  default int getLastExecSegmentsSlotBySlot(Pointer planHandle) { return -1; }
  default int getLastExecSegmentsFailed(Pointer planHandle) { return -1; }
  default int getLastExecSegmentsTotal(Pointer planHandle) { return -1; }

  /** Sync level from last execute(): 0=NONE, 1=EVENT, 2=STREAM, 3=FULL_DEVICE, -1=unavailable. */
  default int getLastExecSyncLevel(Pointer planHandle) { return -1; }

  /** Stream sync count from last execute(). */
  default int getLastExecStreamSyncCount(Pointer planHandle) { return -1; }

  /** Consecutive executions where no variable ext input changed. */
  default int getLastExecConsecutiveUnchangedCount(Pointer planHandle) { return -1; }

  // =============================================================================
  // DspHandle API — Direct Plan Replay and Slot Inspection
  // =============================================================================

  /** Mark external input extIdx as variable (D2D staging buffer allocated). */
  default void markPlanExternalInputVariable(Pointer planHandle, int extIdx) {
    throw new UnsupportedOperationException("markPlanExternalInputVariable not implemented");
  }

  /** Mark external input extIdx as placeholder (variable + H2D sync forced). */
  default void markPlanExternalInputPlaceholder(Pointer planHandle, int extIdx) {
    throw new UnsupportedOperationException("markPlanExternalInputPlaceholder not implemented");
  }

  /**
   * Execute plan in steady-state mode (fast path).
   * Returns 0 on success, 8 if plan not in steady state, other = error.
   */
  default int executeSteadyStatePlan(Pointer planHandle, OpaqueContext opContext, Pointer stream) {
    return executeDynamicShapePlan(planHandle, opContext, stream);
  }

  /**
   * Get the OpaqueNDArray at the given output slot index.
   * Non-owning view — valid until next execute(). Returns null if out of range or empty.
   */
  default OpaqueNDArray getPlanSlotOutputArray(Pointer planHandle, int slotIdx) { return null; }

  /**
   * Get total number of output slots (all intermediate + final).
   * Returns -1 if handle is invalid.
   */
  default int getTotalPlanOutputSlots(Pointer planHandle) { return -1; }

  // =============================================================================
  // Per-Segment Pointer Tracking
  // =============================================================================

  default String getPlanSegmentTrackedPointers(Pointer planHandle, int segmentIdx) { return "[]"; }
  default int getPlanSegmentNumCaptureBuffers(Pointer planHandle, int segmentIdx) { return 0; }
  default String getPlanSegmentCaptureBuffersJson(Pointer planHandle, int segmentIdx) { return "[]"; }
  default int getPlanSegmentNumHostPointers(Pointer planHandle, int segmentIdx) { return 0; }

  // =============================================================================
  // Replay Cache Management
  // =============================================================================

  default boolean isReplayCacheEnabled() { return false; }
  default int getReplayCacheHits() { return 0; }
  default int getReplayCacheMisses() { return 0; }
  default void clearReplayCache() {}
  default String getReplayCacheDir() { return ""; }
  default String getReplayCacheDeviceStatsJson() { return "{}"; }
  default int getReplayCacheDeviceEntryCount(int deviceType, int deviceIndex) { return 0; }
  default void clearReplayCacheForDevice(int deviceType, int deviceIndex) {}
  default boolean migrateReplayCache(int fromType, int fromIdx, int toType, int toIdx) { return false; }
  default int pruneStaleReplayCacheDevices() { return 0; }
  default int loadReplayCacheForDevice(Pointer planHandle, int deviceType, int deviceIndex) { return 0; }
  default String getReplayCachedDevicesJson() { return "[]"; }

  // =============================================================================
  // Backend Plan Management
  // =============================================================================

  default String getPlanAvailableBackends(Pointer planHandle) { return "[]"; }
  default String getPlanSegmentCompiledBackend(Pointer planHandle, int segIdx) { return ""; }
  default String getPlanSegmentCompilationAudit(Pointer planHandle, int segIdx) { return "{}"; }
  default void invalidatePlanSegmentCache(Pointer planHandle, int segIdx) {}
  default void invalidatePlanBackendCaches(Pointer planHandle, String backendName) {}
  default String getPlanBackendCacheStats(Pointer planHandle) { return "{}"; }
  default void setPlanSegmentBackendOverride(Pointer planHandle, int segIdx, String backendName) {}
  default void setPlanBackendPriority(Pointer planHandle, String priorityList) {}

  /**
   * Get a JSON summary of all segments in the plan, including slot ranges,
   * execution counts, compilation status, and op counts.
   * Useful for testing and debugging segment structure after freeze transitions.
   * @param planHandle handle from compileDynamicShapePlan()
   * @return JSON array of segment info objects
   */
  default String getPlanSegmentsSummaryJson(Pointer planHandle) { return "[]"; }

  /**
   * Set DSP freeze merge segments flag at runtime.
   * When true, setShapesFrozen(true) merges all segments into one mega-segment.
   * When false (default), existing segment boundaries are preserved.
   * @param enable true to enable segment merging on freeze
   */
  default void setDspFreezeMergeSegments(boolean enable) {}

  /**
   * Set DSP freeze recompile flag at runtime.
   * When true, Triton recompilation is forced after freeze warmup.
   * When false (default), existing compilations are reused.
   * @param enable true to force recompilation
   */
  default void setDspFreezeRecompile(boolean enable) {}

  /**
   * Get current DSP freeze merge segments setting.
   */
  default boolean getDspFreezeMergeSegments() { return false; }

  /**
   * Get current DSP freeze recompile setting.
   */
  default boolean getDspFreezeRecompile() { return false; }

  /**
   * Export the CUDA graph visualization to Chrome trace format.
   * The output JSON file can be loaded in chrome://tracing for detailed timeline analysis.
   * Similar to PyTorch's torch.cuda.CUDAGraph.debug_dump() functionality.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   * @param outputPath Output file path (should end with .json)
   * @return true on success
   */
  default boolean exportPlanCudaGraphChromeTrace(Pointer planHandle, String outputPath) {
      return false;
  }

  /**
   * Export the CUDA graph visualization to HTML format.
   * Creates a standalone HTML file with interactive visualization.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   * @param outputPath Output HTML file path
   * @return true on success
   */
  default boolean exportPlanCudaGraphHtml(Pointer planHandle, String outputPath) {
      return false;
  }

  /**
   * Dump all CUDA graph debug files (DOT, JSON, HTML, nodes JSON).
   * Creates: {outputPath}.dot, {outputPath}.json, {outputPath}.html, {outputPath}_nodes.json
   * PyTorch-style debug dump similar to torch.cuda.CUDAGraph.debug_dump().
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   * @param outputPath Base path for output files (without extension)
   * @return true on success
   */
  default boolean debugDumpPlanCudaGraph(Pointer planHandle, String outputPath) {
      return false;
  }

  /**
   * Get the CUDA graph execution timeline as a JSON string in Chrome trace format.
   * For programmatic access to the timeline data.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   * @return JSON string, or empty string if no graph data
   */
  default String getPlanCudaGraphChromeTraceJson(Pointer planHandle) {
      return "";
  }

  /**
   * Clear the CUDA graph execution timeline history.
   * Useful to reset timing data between profiling sessions.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   */
  default void clearPlanCudaGraphTimeline(Pointer planHandle) {
      // No-op on non-CUDA backends
  }

  // ==========================================================================
  // NCCL Collective Communication Operations
  // ==========================================================================

  /**
   * Initialize an NCCL communicator for a group of GPUs (single-process).
   *
   * @param numRanks  number of ranks (GPUs) in the communicator
   * @param rankId    this process's rank (0-indexed)
   * @param deviceId  CUDA device ID for this rank
   * @return opaque handle to the NCCL communicator, or null on failure
   */
  default Pointer ncclCommInit(int numRanks, int rankId, int deviceId) {
      return null;
  }

  /**
   * Initialize an NCCL communicator from a unique ID (multi-process).
   *
   * @param numRanks  number of ranks
   * @param rankId    this process's rank
   * @param uniqueId  pointer to ncclUniqueId (128 bytes)
   * @return opaque handle to the NCCL communicator
   */
  default Pointer ncclCommInitWithId(int numRanks, int rankId, Pointer uniqueId) {
      return null;
  }

  /**
   * Generate a unique NCCL ID for multi-process initialization.
   *
   * @return pointer to a newly allocated ncclUniqueId
   */
  default Pointer ncclGetUniqueId() {
      return null;
  }

  /**
   * Destroy an NCCL communicator and release resources.
   *
   * @param commHandle handle from ncclCommInit
   */
  default void ncclCommDestroy(Pointer commHandle) {
  }

  /**
   * AllReduce: sum all tensors across ranks, result available on all ranks.
   *
   * @param commHandle   NCCL communicator handle
   * @param sendBuf      send buffer pointer (device memory)
   * @param recvBuf      receive buffer pointer (can be same as sendBuf for in-place)
   * @param numElements  number of elements
   * @param dataType     data type ordinal (sd.DataType)
   * @param reduceOp     0=SUM, 1=PROD, 2=MAX, 3=MIN, 4=AVG
   * @param stream       CUDA stream pointer (null for default)
   * @return 0 on success, non-zero on failure
   */
  default int ncclDoAllReduce(Pointer commHandle,
                               Pointer sendBuf, Pointer recvBuf,
                               long numElements, int dataType,
                               int reduceOp, Pointer stream) {
      return -1;
  }

  /**
   * AllGather: gather data from all ranks, result available on all ranks.
   *
   * @param commHandle   NCCL communicator handle
   * @param sendBuf      send buffer (this rank's data)
   * @param recvBuf      receive buffer (numRanks * sendCount elements)
   * @param sendCount    number of elements per rank
   * @param dataType     data type ordinal
   * @param stream       CUDA stream pointer
   * @return 0 on success, non-zero on failure
   */
  default int ncclDoAllGather(Pointer commHandle,
                               Pointer sendBuf, Pointer recvBuf,
                               long sendCount, int dataType,
                               Pointer stream) {
      return -1;
  }

  /**
   * ReduceScatter: reduce then scatter result across ranks.
   *
   * @param commHandle   NCCL communicator handle
   * @param sendBuf      send buffer
   * @param recvBuf      receive buffer (this rank's shard)
   * @param recvCount    number of elements per rank after scatter
   * @param dataType     data type ordinal
   * @param reduceOp     reduction operation
   * @param stream       CUDA stream pointer
   * @return 0 on success, non-zero on failure
   */
  default int ncclDoReduceScatter(Pointer commHandle,
                                   Pointer sendBuf, Pointer recvBuf,
                                   long recvCount, int dataType,
                                   int reduceOp, Pointer stream) {
      return -1;
  }

  /**
   * Send data to a specific peer rank.
   *
   * @param commHandle   NCCL communicator handle
   * @param sendBuf      send buffer
   * @param numElements  number of elements
   * @param dataType     data type ordinal
   * @param peerRank     destination rank
   * @param stream       CUDA stream pointer
   * @return 0 on success, non-zero on failure
   */
  default int ncclDoSend(Pointer commHandle,
                          Pointer sendBuf, long numElements,
                          int dataType, int peerRank, Pointer stream) {
      return -1;
  }

  /**
   * Receive data from a specific peer rank.
   *
   * @param commHandle   NCCL communicator handle
   * @param recvBuf      receive buffer
   * @param numElements  number of elements
   * @param dataType     data type ordinal
   * @param peerRank     source rank
   * @param stream       CUDA stream pointer
   * @return 0 on success, non-zero on failure
   */
  default int ncclDoRecv(Pointer commHandle,
                          Pointer recvBuf, long numElements,
                          int dataType, int peerRank, Pointer stream) {
      return -1;
  }

  /**
   * Begin a group of NCCL operations (for fusing multiple collectives).
   * @return 0 on success
   */
  default int ncclGroupStart() {
      return -1;
  }

  /**
   * End a group of NCCL operations.
   * @return 0 on success
   */
  default int ncclGroupEnd() {
      return -1;
  }

  // ─── DSP Diagnostics ─────────────────────────────────────────────────────

  /**
   * Set the enabled diagnostic categories (replaces current mask).
   * @param mask bitfield of DspDiagCategory values
   */
  default void dspDiagSetCategories(int mask) {
      throw new UnsupportedOperationException("DSP diagnostics categories are not implemented by " + getClass().getName());
  }

  /**
   * Enable additional diagnostic categories (OR with current mask).
   * @param mask bitfield of DspDiagCategory values to enable
   */
  default void dspDiagEnableCategories(int mask) {
      throw new UnsupportedOperationException("DSP diagnostics categories are not implemented by " + getClass().getName());
  }

  /**
   * Disable specific diagnostic categories (AND NOT with current mask).
   * @param mask bitfield of DspDiagCategory values to disable
   */
  default void dspDiagDisableCategories(int mask) {
      throw new UnsupportedOperationException("DSP diagnostics categories are not implemented by " + getClass().getName());
  }

  /**
   * Get the currently enabled diagnostic category mask.
   * @return bitfield of enabled DspDiagCategory values
   */
  default int dspDiagGetEnabledMask() {
      throw new UnsupportedOperationException("DSP diagnostics mask is not implemented by " + getClass().getName());
  }

  /**
   * Set the diagnostic detail level.
   * @param level 0=SUMMARY, 1=DETAILED, 2=FULL
   */
  default void dspDiagSetLevel(int level) {
      throw new UnsupportedOperationException("DSP diagnostics level is not implemented by " + getClass().getName());
  }

  /**
   * Get the active diagnostic detail level.
   * @return 0=SUMMARY, 1=DETAILED, 2=FULL
   */
  default int dspDiagGetLevel() {
      throw new UnsupportedOperationException("DSP diagnostics level is not implemented by " + getClass().getName());
  }

  /**
   * Set the file path for JSON diagnostic output.
   * @param path absolute file path, or null to disable
   */
  default void dspDiagSetJsonPath(String path) {
      throw new UnsupportedOperationException("DSP diagnostics output is not implemented by " + getClass().getName());
  }

  /**
   * Record a diagnostic event from Java code into the C++ ring buffer.
   * @param category DspDiagCategory bitfield value
   * @param slotId slot index or -1
   * @param segmentId segment index or -1
   * @param opName op name or null
   * @param timingUs timing in microseconds or 0
   * @param message event message
   */
  default void dspDiagRecordJavaEvent(int category, int slotId, int segmentId,
                                       String opName, long timingUs, String message) {
      throw new UnsupportedOperationException("DSP diagnostic events are not implemented by " + getClass().getName());
  }

  /**
   * Get the structured text plan report from the C++ diagnostics singleton.
   * @return formatted diagnostic report, or empty string if no data
   */
  default String dspDiagGetPlanReport() {
      throw new UnsupportedOperationException("DSP diagnostic reports are not implemented by " + getClass().getName());
  }

  /**
   * Get the JSON plan report from the C++ diagnostics singleton.
   * @return JSON string, or empty string if no data
   */
  default String dspDiagGetJsonReport() {
      throw new UnsupportedOperationException("DSP diagnostic reports are not implemented by " + getClass().getName());
  }

  /** Persist the current diagnostic ring to its configured JSON path. */
  void dspDiagFlushJson();

  /**
   * Clear all diagnostic events and stats.
   */
  default void dspDiagClear() {
      throw new UnsupportedOperationException("DSP diagnostic state is not implemented by " + getClass().getName());
  }

  /**
   * Get the total number of steps executed by the diagnostic singleton.
   */
  default int dspDiagGetStepCount() {
      throw new UnsupportedOperationException("DSP diagnostic counters are not implemented by " + getClass().getName());
  }

  /**
   * Get the total event count across all diagnostic categories.
   */
  default long dspDiagGetTotalEventCount() {
      throw new UnsupportedOperationException("DSP diagnostic counters are not implemented by " + getClass().getName());
  }

  /**
   * Get the event count for a specific diagnostic category by index (0..18).
   * @param categoryIndex  index into the category stats array
   * @return event count, 0 for invalid indices
   */
  default long dspDiagGetCategoryEventCount(int categoryIndex) {
      throw new UnsupportedOperationException("DSP diagnostic counters are not implemented by " + getClass().getName());
  }

  /**
   * Validate plan outputs after execution. Checks each output for NaN, Inf, all-zeros, or null.
   * Flags are bitmask: 1=NULL, 2=NaN, 4=Inf, 8=ALL_ZERO.
   *
   * @param planHandle  Handle from compileDynamicShapePlan()
   * @param flagsOut    int array of size numRequestedOutputs, filled with validation bitmask per output
   * @return number of outputs with validation issues
   */
  default int dspValidateOutputs(Pointer planHandle, int[] flagsOut) { return 0; }

  /**
   * Detect stale outputs by comparing current outputs against previous step norms.
   *
   * @param planHandle  Handle from compileDynamicShapePlan()
   * @param prevNorms   float array of size numRequestedOutputs (updated in-place with current norms)
   * @param staleOut    boolean array of size numRequestedOutputs, set to true for stale outputs
   * @param epsilon     threshold below which norm difference is considered stale
   * @return number of stale outputs detected
   */
  default int dspDetectStaleOutputs(Pointer planHandle, float[] prevNorms, boolean[] staleOut, float epsilon) { return 0; }

  // ─── DSP Cross-Stream Testing API ──────────────────────────────────────────

  /**
   * Create an independent CUDA stream for testing cross-stream sync behavior.
   * @return Opaque pointer to a cudaStream_t, or null on CPU builds
   */
  default Pointer dspCreateTestStream() {
    throw new UnsupportedOperationException("dspCreateTestStream not implemented");
  }

  /**
   * Destroy a test stream created by dspCreateTestStream().
   */
  default void dspDestroyTestStream(Pointer streamPtr) {
    throw new UnsupportedOperationException("dspDestroyTestStream not implemented");
  }

  /**
   * Write data from host to an external input's device buffer on the LC default stream.
   * Marks device as authoritative after write (isPrimaryActual() returns false).
   * @return 0 on success, -1 on error, -2 on CPU (no-op)
   */
  default int dspWriteDeviceBufferOnDefaultStream(
      Pointer planHandle, int extIdx, Pointer srcHost, long numBytes) {
    throw new UnsupportedOperationException("dspWriteDeviceBufferOnDefaultStream not implemented");
  }

  /**
   * Write data from host to an external input's device buffer on an explicit stream.
   * Creates the cross-stream hazard that performPreReplaySync must handle.
   * @return 0 on success, -1 on error, -2 on CPU (no-op)
   */
  default int dspWriteDeviceBufferOnExplicitStream(
      Pointer planHandle, int extIdx, Pointer srcHost, long numBytes, Pointer streamPtr) {
    throw new UnsupportedOperationException("dspWriteDeviceBufferOnExplicitStream not implemented");
  }

  /**
   * Synchronize a CUDA stream (cudaStreamSynchronize).
   * @param streamPtr  Pointer to stream, or null for default stream
   * @return 0 on success, CUDA error code on failure, -2 on CPU (no-op)
   */
  default int dspSyncStream(Pointer streamPtr) {
    throw new UnsupportedOperationException("dspSyncStream not implemented");
  }

  /**
   * Query whether an external input's device buffer is authoritative.
   * @return 1 if device authoritative, 0 if host authoritative, -1 on error
   */
  default int dspIsExtInputDeviceAuthoritative(Pointer planHandle, int extIdx) {
    throw new UnsupportedOperationException("dspIsExtInputDeviceAuthoritative not implemented");
  }

  /**
   * Get the DSP execution stream pointer for a plan.
   * @return Opaque pointer to the DSP stream, or null
   */
  default Pointer dspGetExecutionStream(Pointer planHandle) {
    throw new UnsupportedOperationException("dspGetExecutionStream not implemented");
  }

  /**
   * Get the LaunchContext default stream pointer.
   * @return Opaque pointer to the default stream, or null
   */
  default Pointer dspGetDefaultStream() {
    throw new UnsupportedOperationException("dspGetDefaultStream not implemented");
  }

  /**
   * Returns 1 if the segment's arg table needs refreshing (argTableGeneration != capturedArgGeneration),
   * 0 if up-to-date, -1 on error.
   */
  default int getPlanSegmentNeedsArgRefresh(Pointer planHandle, int segIdx) {
    throw new UnsupportedOperationException("getPlanSegmentNeedsArgRefresh not implemented");
  }

  /**
   * Returns the segment's argTableGeneration counter (incremented by bumpArgGeneration).
   * Returns -1 on error.
   */
  default long getPlanSegmentArgGeneration(Pointer planHandle, int segIdx) {
    throw new UnsupportedOperationException("getPlanSegmentArgGeneration not implemented");
  }

  /**
   * Returns the segment's capturedArgGeneration counter (set by markArgsCurrent at capture).
   * Returns -1 on error.
   */
  default long getPlanSegmentCapturedArgGeneration(Pointer planHandle, int segIdx) {
    throw new UnsupportedOperationException("getPlanSegmentCapturedArgGeneration not implemented");
  }

  /**
   * Returns the capturedInputAddrKey for the segment (hash of non-variable external input
   * device addresses recorded at graph capture time). Returns 0 if not yet captured, -1 on error.
   */
  default long getPlanSegmentCapturedInputAddrKey(Pointer planHandle, int segIdx) {
    throw new UnsupportedOperationException("getPlanSegmentCapturedInputAddrKey not implemented");
  }

  // ── Buffer coloring introspection ──────────────────────────────────────────

  /** Whether buffer coloring is currently applied on this plan. */
  default boolean getPlanBufferColoringApplied(Pointer planHandle) { return false; }

  /** Number of colors assigned by coloring (0 if not computed). */
  default int getPlanBufferColoringNumColors(Pointer planHandle) { return 0; }

  /** Estimated bytes saved by coloring. */
  default long getPlanBufferColoringBytesSaved(Pointer planHandle) { return 0; }

  /** Color assigned to a specific slot (-1 if uncolored). */
  default int getPlanSlotColor(Pointer planHandle, int slotIdx) { return -1; }

  // ── Buffer pool introspection ──────────────────────────────────────────────

  /** Total bytes currently pooled on the given device. */
  default long getBufferPoolPooledBytes(int deviceId) { return 0; }

  /** Number of buffers currently in the pool on the given device. */
  default int getBufferPoolPooledCount(int deviceId) { return 0; }

  /** Lifetime acquire count on the given device. */
  default long getBufferPoolTotalAcquired(int deviceId) { return 0; }

  /** Lifetime reuse count (acquires satisfied from pool) on the given device. */
  default long getBufferPoolTotalReused(int deviceId) { return 0; }

  // ── Buffer fingerprint ring (BUF_FP_RING=1) ───────────────────────────────

  /**
   * Drain the device fingerprint ring to host after the decode loop ends.
   * No-op if BUF_FP_RING was not set or plan is null.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   */
  default void drainPlanFingerprintRing(Pointer planHandle) {}

  /**
   * Return per-step XOR fingerprints as a JSON string.
   * Call after drainPlanFingerprintRing(). Returns "null" if not enabled.
   *
   * @param planHandle Handle from compileDynamicShapePlan()
   * @return JSON string or "null"
   */
  default String getPlanFingerprintJson(Pointer planHandle) { return "null"; }

}
