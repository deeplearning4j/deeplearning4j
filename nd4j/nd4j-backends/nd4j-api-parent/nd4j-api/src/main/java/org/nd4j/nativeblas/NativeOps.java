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
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.StdVector;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;


/**
 * A common interface for misc operations
 * needed from c++
 *
 */
public interface NativeOps {

 void dbPrintAllocationTrace(org.nd4j.nativeblas.OpaqueDataBuffer buff);

 org.nd4j.nativeblas.OpaqueDataBuffer intermediateResultDataAt(int index, OpaqueContext contextPointer);

 LongPointer intermediateResultShapeInfoAt(int index, OpaqueContext contextPointer);


 void setIntermediateResult(OpaqueContext contextPointer, int index, org.nd4j.nativeblas.OpaqueDataBuffer buffer, org.nd4j.nativeblas.OpaqueDataBuffer shapeInfo);

 void pushIntermediateResult(OpaqueContext contextPointer, org.nd4j.nativeblas.OpaqueDataBuffer buffer,org.nd4j.nativeblas.OpaqueDataBuffer shapeInfo);


 int numIntermediateResults(OpaqueContext contextPointer);
 PointerPointer intermediateResults(OpaqueContext contextPointer);

 int contextNumInputs(Pointer execTrace);
 int contextNumOutputs(Pointer execTrace);

 int numInputs(Pointer execTrace);
 int numOutputs(Pointer execTrace);
 BooleanPointer bArgs(Pointer execTrace);
 PointerPointer sArgs(Pointer execTrace);
 DoublePointer tArgs(Pointer execTrace);
 LongPointer iArgs(Pointer execTrace);
 PointerPointer inputShapeBuffers(Pointer execTrace);
 PointerPointer outputShapeBuffers(Pointer execTrace);
 BytePointer opName(Pointer execTrace);
 PointerPointer listOpTraces();

 void printOpTrace();

 void purgeOpTrace();

 void toggleOpTrace(boolean opTrace);
 /**
  * Prints device buffers.
  * @param buffer
  */
 void printDeviceBuffer(org.nd4j.nativeblas.OpaqueDataBuffer buffer);

 void copyBuffer(org.nd4j.nativeblas.OpaqueDataBuffer target, long n,  org.nd4j.nativeblas.OpaqueDataBuffer from, long fromOffset, long targetOffset);


 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntPointer shape,  int ndims,
               BytePointer mode/*="w"*/);
 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntPointer shape,  int ndims);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntBuffer shape,  int ndims,
               String mode/*="w"*/);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntBuffer shape,  int ndims);
 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  int[] shape,  int ndims,
               BytePointer mode/*="w"*/);
 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  int[] shape,  int ndims);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntPointer shape,  int ndims,
               String mode/*="w"*/);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntPointer shape,  int ndims);
 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntBuffer shape,  int ndims,
               BytePointer mode/*="w"*/);
 void saveNpy( BytePointer fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  IntBuffer shape,  int ndims);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  int[] shape,  int ndims,
               String mode/*="w"*/);
 void saveNpy( String fname, org.nd4j.nativeblas.OpaqueDataBuffer data,  int[] shape,  int ndims);


 /**
  * This method allows you to specify minimal number of elements per thread/block during op call
  * PLEASE NOTE: Changing this value might and will affect performance.
  *
  * @param value
  */
 void setElementThreshold(int value);

 /**
  * This method allows you to specify minimal number of TADs per thread/block during op call
  * PLEASE NOTE: Changing this value might and will affect performance.
  *
  * @param value
  */
 void setTADThreshold(int value);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParams
  */
 void execIndexReduceScalar(PointerPointer extraPointers,
                            int opNum,
                            OpaqueDataBuffer x,
                            LongPointer xShapeInfo,
                            LongPointer dXShapeInfo,
                            Pointer extraParams,
                            OpaqueDataBuffer z,
                            LongPointer zShapeInfo,
                            LongPointer dZShapeInfo);

 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dXShapeInfo
  * @param extraParams
  * @param result
  * @param resultShapeInfoBuffer
  * @param dResultShapeInfoBuffer
  * @param hDimension
  * @param hDimensionShape
  * @param dDimensionShape
  */
 void execIndexReduce(PointerPointer extraPointers,
                      int opNum,
                      OpaqueDataBuffer x,
                      LongPointer xShapeInfo,
                      LongPointer dXShapeInfo,
                      Pointer extraParams,
                      OpaqueDataBuffer result,
                      LongPointer resultShapeInfoBuffer,
                      LongPointer dResultShapeInfoBuffer,
                      OpaqueDataBuffer hDimension,
                      LongPointer hDimensionShape,
                      LongPointer dDimensionShape);

 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param y
  * @param yShapeInfo
  * @param dyShapeInfo
  * @param result
  * @param resultShapeInfo
  * @param dresultShapeInfo
  * @param hDimension
  * @param hDimensionShape
  * @param dDimensionShape
  */
 void execBroadcast(PointerPointer extraPointers,
                    int opNum,
                    OpaqueDataBuffer x,
                    LongPointer xShapeInfo,
                    LongPointer dxShapeInfo,
                    OpaqueDataBuffer y,
                    LongPointer yShapeInfo,
                    LongPointer dyShapeInfo,
                    OpaqueDataBuffer result,
                    LongPointer resultShapeInfo,
                    LongPointer dresultShapeInfo,
                    OpaqueDataBuffer hDimension,
                    LongPointer hDimensionShape,
                    LongPointer dDimensionShape);

 void execBroadcastBool(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        LongPointer xShapeInfo,
                        LongPointer dxShapeInfo,
                        OpaqueDataBuffer y,
                        LongPointer yShapeInfo,
                        LongPointer dyShapeInfo,
                        OpaqueDataBuffer result,
                        LongPointer resultShapeInfo,
                        LongPointer dresultShapeInfo,
                        Pointer extraParams,
                        OpaqueDataBuffer hDimension,
                        LongPointer hDimensionShape,
                        LongPointer dDimensionShape);


 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param y
  * @param yShapeInfo
  * @param dyShapeInfo
  * @param result
  * @param resultShapeInfo
  * @param dresultShapeInfo
  * @param extraParams
  */
 void execPairwiseTransform(PointerPointer extraPointers,
                            int opNum,
                            OpaqueDataBuffer x,
                            LongPointer xShapeInfo,
                            LongPointer dxShapeInfo,
                            OpaqueDataBuffer y,
                            LongPointer yShapeInfo,
                            LongPointer dyShapeInfo,
                            OpaqueDataBuffer result,
                            LongPointer resultShapeInfo,
                            LongPointer dresultShapeInfo,
                            Pointer extraParams);

 void execPairwiseTransformBool(PointerPointer extraPointers,
                                int opNum,
                                OpaqueDataBuffer x,
                                LongPointer xShapeInfo,
                                LongPointer dxShapeInfo,
                                OpaqueDataBuffer y,
                                LongPointer yShapeInfo,
                                LongPointer dyShapeInfo,
                                OpaqueDataBuffer result,
                                LongPointer resultShapeInfo,
                                LongPointer dresultShapeInfo,
                                Pointer extraParams);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParams
  * @param result
  * @param resultShapeInfo
  */
 void execReduceFloat(PointerPointer extraPointers,
                      int opNum,
                      OpaqueDataBuffer x,
                      LongPointer xShapeInfo,
                      LongPointer dxShapeInfo,
                      Pointer extraParams,
                      OpaqueDataBuffer result,
                      LongPointer resultShapeInfo,
                      LongPointer dresultShapeInfo);


 void execReduceSame(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     Pointer extraParams,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfo,
                     LongPointer dresultShapeInfo);


 void execReduceBool(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     Pointer extraParams,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfo,
                     LongPointer dresultShapeInfo);


 void execReduceLong(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     Pointer extraParams,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfo,
                     LongPointer dresultShapeInfo);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParams
  * @param result
  * @param resultShapeInfo
  */
 void execReduceFloat2(PointerPointer extraPointers,
                       int opNum,
                       OpaqueDataBuffer x,
                       LongPointer xShapeInfo,
                       LongPointer dxShapeInfo,
                       Pointer extraParams,
                       OpaqueDataBuffer result,
                       LongPointer resultShapeInfo,
                       LongPointer dresultShapeInfo,
                       OpaqueDataBuffer hDimension,
                       LongPointer hDimensionShape,
                       LongPointer dDimensionShape);


 void execReduceSame2(PointerPointer extraPointers,
                      int opNum,
                      OpaqueDataBuffer x,
                      LongPointer xShapeInfo,
                      LongPointer dxShapeInfo,
                      Pointer extraParams,
                      OpaqueDataBuffer result,
                      LongPointer resultShapeInfo,
                      LongPointer dresultShapeInfo,
                      OpaqueDataBuffer hDimension,
                      LongPointer hDimensionShape,
                      LongPointer dDimensionShape);

 void execReduceBool2(PointerPointer extraPointers,
                      int opNum,
                      OpaqueDataBuffer x,
                      LongPointer xShapeInfo,
                      LongPointer dxShapeInfo,
                      Pointer extraParams,
                      OpaqueDataBuffer result,
                      LongPointer resultShapeInfo,
                      LongPointer dresultShapeInfo,
                      OpaqueDataBuffer hDimension,
                      LongPointer hDimensionShape,
                      LongPointer dDimensionShape);

 void execReduceLong2(PointerPointer extraPointers,
                      int opNum,
                      OpaqueDataBuffer x,
                      LongPointer xShapeInfo,
                      LongPointer dxShapeInfo,
                      Pointer extraParams,
                      OpaqueDataBuffer result,
                      LongPointer resultShapeInfo,
                      LongPointer dresultShapeInfo,
                      OpaqueDataBuffer hDimension,
                      LongPointer hDimensionShape,
                      LongPointer dDimensionShape);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParamsVals
  * @param y
  * @param yShapeInfo
  * @param result
  * @param resultShapeInfo
  */
 void execReduce3(PointerPointer extraPointers,
                  int opNum,
                  OpaqueDataBuffer x,
                  LongPointer xShapeInfo,
                  LongPointer dxShapeInfo,
                  Pointer extraParamsVals,
                  OpaqueDataBuffer y,
                  LongPointer yShapeInfo,
                  LongPointer dyShapeInfo,
                  OpaqueDataBuffer result,
                  LongPointer resultShapeInfo,
                  LongPointer dresultShapeInfo);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParamsVals
  * @param y
  * @param yShapeInfo
  */
 void execReduce3Scalar(PointerPointer extraPointers, int opNum,
                        OpaqueDataBuffer x,
                        LongPointer xShapeInfo,
                        LongPointer dxShapeInfo,
                        Pointer extraParamsVals,
                        OpaqueDataBuffer y,
                        LongPointer yShapeInfo,
                        LongPointer dyShapeInfo,
                        OpaqueDataBuffer z,
                        LongPointer zShapeInfo,
                        LongPointer dzShapeInfo);

 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param extraParamsVals
  * @param y
  * @param yShapeInfo
  * @param dyShapeInfo
  * @param result
  * @param resultShapeInfoBuffer
  * @param dresultShapeInfoBuffer
  * @param hDimension
  * @param hDimensionShape
  * @param dDimensionShape
  * @param tadOnlyShapeInfo
  * @param tadOffsets
  * @param yTadOnlyShapeInfo
  * @param yTadOffsets
  */
 void execReduce3Tad(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     Pointer extraParamsVals,
                     OpaqueDataBuffer y,
                     LongPointer yShapeInfo,
                     LongPointer dyShapeInfo,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfoBuffer,
                     LongPointer dresultShapeInfoBuffer,
                     OpaqueDataBuffer hDimension,
                     LongPointer hDimensionShape,
                     LongPointer dDimensionShape,
                     LongPointer tadOnlyShapeInfo,  LongPointer tadOffsets,
                     LongPointer yTadOnlyShapeInfo,  LongPointer yTadOffsets);

 void execReduce3All(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     Pointer extraParamsVals,
                     OpaqueDataBuffer y,
                     LongPointer yShapeInfo,
                     LongPointer dyShapeInfo,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfoBuffer,
                     LongPointer dresultShapeInfoBuffer,
                     OpaqueDataBuffer hDimension,
                     LongPointer hDimensionShape,
                     LongPointer dDimensionShape,
                     LongPointer xTadShape,
                     LongPointer xOffsets,
                     LongPointer yTadShape,
                     LongPointer yOffsets);


 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param result
  * @param resultShapeInfo
  * @param scalar
  * @param extraParams
  */
 void execScalar(PointerPointer extraPointers,
                 int opNum,
                 OpaqueDataBuffer x,
                 LongPointer xShapeInfo,
                 LongPointer dxShapeInfo,
                 OpaqueDataBuffer result,
                 LongPointer resultShapeInfo,
                 LongPointer dresultShapeInfo,
                 OpaqueDataBuffer scalar,
                 LongPointer scalarShapeInfo,
                 LongPointer dscalarShapeInfo,
                 Pointer extraParams);

 void execScalarBool(PointerPointer extraPointers,
                     int opNum,
                     OpaqueDataBuffer x,
                     LongPointer xShapeInfo,
                     LongPointer dxShapeInfo,
                     OpaqueDataBuffer result,
                     LongPointer resultShapeInfo,
                     LongPointer dresultShapeInfo,
                     OpaqueDataBuffer scalar,
                     LongPointer scalarShapeInfo,
                     LongPointer dscalarShapeInfo,
                     Pointer extraParams);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParams
  * @param biasCorrected
  */
 void execSummaryStatsScalar(PointerPointer extraPointers,
                             int opNum,
                             OpaqueDataBuffer x,
                             LongPointer xShapeInfo,
                             LongPointer dxShapeInfo,
                             Pointer extraParams,
                             OpaqueDataBuffer z,
                             LongPointer zShapeInfo,
                             LongPointer dzShapeInfo,
                             boolean biasCorrected);

 /**
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param extraParams
  * @param result
  * @param resultShapeInfo
  * @param biasCorrected
  */
 void execSummaryStats(PointerPointer extraPointers,
                       int opNum,
                       OpaqueDataBuffer x,
                       LongPointer xShapeInfo,
                       LongPointer dxShapeInfo,
                       Pointer extraParams,
                       OpaqueDataBuffer result,
                       LongPointer resultShapeInfo,
                       LongPointer dresultShapeInfo,
                       boolean biasCorrected);

 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param extraParams
  * @param result
  * @param resultShapeInfoBuffer
  * @param dresultShapeInfoBuffer
  * @param hDimension
  * @param hDimensionShape
  * @param dDimensionShape
  * @param biasCorrected
  * @param tadShapeInfo
  * @param tadOffsets
  */
 void execSummaryStatsTad(PointerPointer extraPointers,
                          int opNum,
                          OpaqueDataBuffer x,
                          LongPointer xShapeInfo,
                          LongPointer dxShapeInfo,
                          Pointer extraParams,
                          OpaqueDataBuffer result,
                          LongPointer resultShapeInfoBuffer,
                          LongPointer dresultShapeInfoBuffer,
                          OpaqueDataBuffer hDimension,
                          LongPointer hDimensionShape,
                          LongPointer dDimensionShape,
                          boolean biasCorrected,
                          LongPointer tadShapeInfo,
                          LongPointer tadOffsets);


 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param result
  * @param resultShapeInfo
  * @param dresultShapeInfo
  * @param extraParams
  */
 void execTransformFloat(PointerPointer extraPointers,
                         int opNum,
                         OpaqueDataBuffer x,
                         LongPointer xShapeInfo,
                         LongPointer dxShapeInfo,
                         OpaqueDataBuffer result,
                         LongPointer resultShapeInfo,
                         LongPointer dresultShapeInfo,
                         Pointer extraParams);

 void execTransformSame(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        LongPointer xShapeInfo,
                        LongPointer dxShapeInfo,
                        OpaqueDataBuffer result,
                        LongPointer resultShapeInfo,
                        LongPointer dresultShapeInfo,
                        Pointer extraParams);

 void execTransformStrict(PointerPointer extraPointers,
                          int opNum,
                          OpaqueDataBuffer x,
                          LongPointer xShapeInfo,
                          LongPointer dxShapeInfo,
                          OpaqueDataBuffer result,
                          LongPointer resultShapeInfo,
                          LongPointer dresultShapeInfo,
                          Pointer extraParams);

 void execTransformBool(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        LongPointer xShapeInfo,
                        LongPointer dxShapeInfo,
                        OpaqueDataBuffer result,
                        LongPointer resultShapeInfo,
                        LongPointer dresultShapeInfo,
                        Pointer extraParams);

 void execTransformAny(PointerPointer extraPointers,
                       int opNum,
                       OpaqueDataBuffer x,
                       LongPointer xShapeInfo,
                       LongPointer dxShapeInfo,
                       OpaqueDataBuffer result,
                       LongPointer resultShapeInfo,
                       LongPointer dresultShapeInfo,
                       Pointer extraParams);

 /**
  *
  * @param extraPointers
  * @param opNum
  * @param x
  * @param xShapeInfo
  * @param dxShapeInfo
  * @param z
  * @param zShapeInfo
  * @param dzShapeInfo
  * @param scalars
  * @param scalarShapeInfo
  * @param dscalarShapeInfo
  * @param extraParams
  * @param hDimension
  * @param hDimensionShape
  * @param dDimensionShape
  * @param tadShapeInfo
  * @param tadOffsets
  * @param tadShapeInfoZ
  * @param tadOffsetsZ
  */
 void execScalarTad(PointerPointer extraPointers,
                    int opNum,
                    OpaqueDataBuffer x,
                    LongPointer xShapeInfo,
                    LongPointer dxShapeInfo,
                    OpaqueDataBuffer z,
                    LongPointer zShapeInfo,
                    LongPointer dzShapeInfo,
                    OpaqueDataBuffer scalars,
                    LongPointer scalarShapeInfo,
                    LongPointer dscalarShapeInfo,
                    Pointer extraParams,
                    OpaqueDataBuffer hDimension,
                    LongPointer hDimensionShape,
                    LongPointer dDimensionShape,
                    LongPointer tadShapeInfo,
                    LongPointer tadOffsets,
                    LongPointer tadShapeInfoZ,
                    LongPointer tadOffsetsZ);

 void execScalarBoolTad(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        LongPointer xShapeInfo,
                        LongPointer dxShapeInfo,
                        OpaqueDataBuffer z,
                        LongPointer zShapeInfo,
                        LongPointer dzShapeInfo,
                        OpaqueDataBuffer scalars,
                        LongPointer scalarShapeInfo,
                        LongPointer dscalarShapeInfo,
                        Pointer extraParams,
                        OpaqueDataBuffer hDimension,
                        LongPointer hDimensionShape,
                        LongPointer dDimensionShape,
                        LongPointer tadShapeInfo,
                        LongPointer tadOffsets,
                        LongPointer tadShapeInfoZ,
                        LongPointer tadOffsetsZ);


 void specialConcat(PointerPointer extraPointers,
                    int dimension,
                    int numArrays,
                    PointerPointer data, PointerPointer inputShapeInfo,
                    Pointer results,  LongPointer resultShapeInfo,
                    PointerPointer tadPointers,
                    PointerPointer tadOffsets);


 /**
  * Gets the maximum number of open mp threads
  *
  * @return
  */
 int ompGetMaxThreads();

 /**
  * Gets the number of open mp threads
  *
  * @return
  */
 int ompGetNumThreads();

 /**
  * Sets the number of openmp threads
  *
  * @param threads
  */
 void setOmpNumThreads(int threads);

 /**
  * Sets the minimal number of openmp threads for variative methods
  *
  * @param threads
  */
 void setOmpMinThreads(int threads);

 /**
  * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
  */
 void initializeDevicesAndFunctions();

 void initializeFunctions(PointerPointer functions);

 Pointer mallocHost(long memorySize, int flags);

 Pointer mallocDevice(long memorySize, int ptrToDeviceId, int flags);

 int freeHost(Pointer pointer);

 int freeDevice(Pointer pointer, int deviceId);

 Pointer createContext();

 Pointer createStream();

 Pointer createEvent();

 int registerEvent(Pointer event, Pointer stream);

 int destroyEvent(Pointer event);

 int setDevice(int ptrToDeviceId);

 int getDevice();

 int streamSynchronize(Pointer stream);

 int eventSynchronize(Pointer event);

 long getDeviceFreeMemory(int ptrToDeviceId);

 long getDeviceFreeMemoryDefault();

 long getDeviceTotalMemory(int ptrToDeviceId);

 int getDeviceMajor(int ptrToDeviceId);

 int getDeviceMinor(int ptrToDeviceId);

 String getDeviceName(int ptrToDeviceId);


 int memcpySync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

 int memcpyAsync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

 int memcpyConstantAsync(long dst, Pointer src, long size, int flags, Pointer reserved);

 int memsetSync(Pointer dst, int value, long size, int flags, Pointer reserved);

 int memsetAsync(Pointer dst, int value, long size, int flags, Pointer reserved);

 Pointer getConstantSpace();

 int getAvailableDevices();

 void enableDebugMode(boolean reallyEnable);

 void enableVerboseMode(boolean reallyEnable);

 void setGridLimit(int gridSize);

 OpaqueTadPack tadOnlyShapeInfo(LongPointer shapeInfo, LongPointer dimension, long dimensionLength);

 LongPointer getPrimaryShapeInfo(OpaqueTadPack pack);
 LongPointer getPrimaryOffsets(OpaqueTadPack pack);
 LongPointer getSpecialShapeInfo(OpaqueTadPack pack);
 LongPointer getSpecialOffsets(OpaqueTadPack pack);
 long getNumberOfTads(OpaqueTadPack pack);
 int getShapeInfoLength(OpaqueTadPack pack);

 void deleteTadPack(OpaqueTadPack pointer);

 ///////////////

 void pullRows(PointerPointer extraPointers,
               OpaqueDataBuffer x,
               LongPointer xShapeInfo,
               LongPointer dxShapeInfo,
               OpaqueDataBuffer z,
               LongPointer zShapeInfo,
               LongPointer dzShapeInfo,
               long n,
               LongPointer indexes,
               LongPointer tadShapeInfo,
               LongPointer tadOffsets,
               LongPointer zTadShapeInfo,
               LongPointer zTadOffsets);


 ///////////////////////

 void average(PointerPointer extraPointers,
              PointerPointer x,  LongPointer xShapeInfo,
              PointerPointer dx,  LongPointer dxShapeInfo,
              Pointer z,  LongPointer zShapeInfo,
              Pointer dz,  LongPointer dzShapeInfo,
              int n,
              long length,
              boolean propagate);

 ///////////////////////

 void accumulate(PointerPointer extraPointers,
                 PointerPointer x,  LongPointer xShapeInfo,
                 PointerPointer dx,  LongPointer dxShapeInfo,
                 Pointer z,  LongPointer zShapeInfo,
                 Pointer dz,  LongPointer dzShapeInfo,
                 int n,
                 long length);

 ///////////////////////

 void enableP2P(boolean reallyEnable);

 void checkP2P();

 boolean isP2PAvailable();

 //

 void shuffle(PointerPointer extraPointers,
              PointerPointer x,  PointerPointer xShapeInfo,
              PointerPointer dx,  PointerPointer dxShapeInfo,
              PointerPointer z,  PointerPointer zShapeInfo,
              PointerPointer dz,  PointerPointer dzShapeInfo,
              int N,
              IntPointer shuffleMap,
              PointerPointer tadShapeInfo,
              PointerPointer tadOffsets);


 // opType conversion

 void convertTypes(PointerPointer extras, int srcType, Pointer x, long N, int dstType, Pointer z);

 boolean isExperimentalEnabled();

 // GridOps

/*
    // MetaOps
    void execMetaPredicateShape(PointerPointer extras,
                                                int opTypeA, int opNumA,
                                                int opTypeB, int opNumB,
                                                long N,
                                                Pointer x,  LongPointer xShape,
                                                Pointer dx,  LongPointer dxShape,
                                                Pointer y,  LongPointer yShape,
                                                Pointer dy,  LongPointer dyShape,
                                                Pointer z,  LongPointer zShape,
                                                Pointer dz,  LongPointer dzShape,
                                                Pointer extraA, Pointer extraB, double scalarA,
                                                double scalarB);

*/
 /////////////////////////

 void execAggregate(PointerPointer extras, int opNum,
                    PointerPointer arguments,
                    int numArguments,
                    @Cast("sd::LongType **") PointerPointer shapes,
                    int numShapes,
                    IntPointer indexArguments,
                    int numIndexArguments,
                    @Cast("int **") PointerPointer intArrays,
                    int numIntArrays,
                    Pointer realArguments,
                    int numRealArguments,
                    @Cast("nd4j::DataType") int dataType);

 void execAggregateBatch(PointerPointer extras, int numAggregates, int opNum, int maxArgs,
                         int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                         Pointer ptrToArguments, @Cast("nd4j::DataType") int dataType);


 //////////////
 void execRandom(PointerPointer extraPointers,
                 int opNum,
                 Pointer state,
                 OpaqueDataBuffer z,
                 LongPointer zShapeBuffer,
                 LongPointer dzShapeBuffer,
                 Pointer extraArguments);

 void execRandom3(PointerPointer extraPointers,
                  int opNum,
                  Pointer state,
                  OpaqueDataBuffer x,
                  LongPointer xShapeBuffer,
                  LongPointer dxShapeBuffer,
                  OpaqueDataBuffer y,
                  LongPointer yShapeBuffer,
                  LongPointer dyShapeBuffer,
                  OpaqueDataBuffer z,
                  LongPointer zShapeBuffer,
                  LongPointer dzShapeBuffer,
                  Pointer extraArguments);

 void execRandom2(PointerPointer extraPointers,
                  int opNum,
                  Pointer state,
                  OpaqueDataBuffer x,
                  LongPointer xShapeBuffer,
                  LongPointer dxShapeBuffer,
                  OpaqueDataBuffer z,
                  LongPointer zShapeBuffer,
                  LongPointer dzShapeBuffer,
                  Pointer extraArguments);

 ////////////////////


 Pointer initRandom(PointerPointer extraPointers, long seed, long numberOfElements, Pointer pointerToBuffer);

 void refreshBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

 void reSeedBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

 void destroyRandom(Pointer pointer);


 /**
  * Length of a numpy header given a word size and shape buffer
  * @param shapeBuffer the shape buffer to get the header length for
  * @param wordSize the word size
  * @return
  */
 long numpyHeaderLengthWordSize( Pointer shapeBuffer,long wordSize);

 /**
  *
  * Length in bytes of a numpy header + buffer
  */

 long numpyHeaderLength(org.nd4j.nativeblas.OpaqueDataBuffer opaqueDataBuffer, Pointer shapeBuffer);
 /**
  *
  * Length in bytes of the opaque buffer
  */

 long lengthInBytes(org.nd4j.nativeblas.OpaqueDataBuffer buffer);

 /**
  * Create a numpy array from an nd4j
  * array
  *
  * @param data        a pointer to the data
  * @param shapeBuffer the shapebuffer for the nd4j array
  * @param wordSize    the word size (4 for float, 8 for doubles)
  * @return a pointer to a numpy array
  */
 Pointer numpyFromNd4j(Pointer data, Pointer shapeBuffer, long wordSize);


 /**
  * Get the element size for a numpy array
  *
  * @param npyArray the numpy array's address
  *                 to get the length for
  * @return
  */
 int elementSizeForNpyArrayHeader(Pointer npyArray);


 /**
  * @param npyArrayStruct
  * @return
  */
 Pointer dataPointForNumpyStruct(Pointer npyArrayStruct);


 /**
  * Creates a numpy header for nd4j
  *
  * @param data        the data to use
  * @param shapeBuffer the shape buffer for the array
  * @param wordSize    the word size
  * @return
  */
 Pointer numpyHeaderForNd4j(Pointer data, Pointer shapeBuffer, long wordSize, LongPointer length);

 /**
  * Load numpy from a header
  * based on the cnpy parse from header method.
  *
  * @param data the header data to parse
  * @return a pointer to a numpy cnpy:NpyArray struct
  */
 Pointer loadNpyFromHeader(Pointer data);

 /**
  * @param npyArray
  * @return
  */
 Pointer dataPointForNumpyHeader(Pointer npyArray);

 /**
  * Get the shape buffer from a
  * numpy array.
  * **Warning** this allocates memory
  *
  * @param npyArray
  * @return
  */
 Pointer shapeBufferForNumpyHeader(Pointer npyArray);

 /**
  * Used in {@link org.nd4j.linalg.factory.NDArrayFactory#createFromNpyPointer(Pointer)}
  * to allow reuse of an in memory numpy buffer.
  * This is heavily used for python interop
  *
  * @param npyArray the pointer to the numpy array to use
  * @return the pointer for the numpy array
  */
 Pointer dataPointForNumpy(Pointer npyArray);

 /**
  * Get a shape buffer for a numpy array.
  * Used in conjunction with {@link org.nd4j.linalg.factory.NDArrayFactory#createFromNpyPointer(Pointer)}
  *
  * @param npyArray the numpy array to get the shape buffer for
  * @return a pointer representing the shape buffer for numpy
  */
 Pointer shapeBufferForNumpy(Pointer npyArray);

 /**
  * Thie method releases numpy pointer
  * <p>
  * PLEASE NOTE: This method should be ONLY used if pointer/numpy array was originated from file
  *
  * @param npyArray
  */
 void releaseNumpy(Pointer npyArray);


 /**
  * Create a numpy array pointer
  * from a file
  *
  * @param path the path to the file
  * @return
  */
 Pointer numpyFromFile(BytePointer path);


 /**
  * Return the length of a shape buffer
  * based on the pointer
  *
  * @param buffer the buffer pointer to check
  * @return
  */
 int lengthForShapeBufferPointer(Pointer buffer);

 /**
  * Calculate the element size
  * for a numpy array
  *
  * @param npyArray the numpy array to get the
  *                 element size for
  * @return the element size for a given array
  */
 int elementSizeForNpyArray(Pointer npyArray);


 /**
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */
 Pointer pointerForAddress(long address);


 ////// NPZ ///////
 Pointer mapFromNpzFile(BytePointer path);

 int getNumNpyArraysInMap(Pointer map);



 String getNpyArrayNameFromMap(Pointer map, int index,BytePointer buffer);

 Pointer getNpyArrayFromMap(Pointer map, int index);

 Pointer getNpyArrayData(Pointer npArray);

 LongPointer getNpyArrayShape(Pointer npArray);

 int getNpyArrayRank(Pointer npArray);

 char getNpyArrayOrder(Pointer npArray);

 int getNpyArrayElemSize(Pointer npArray);
 ///////


 void tear(PointerPointer extras,
           OpaqueDataBuffer tensor,
           LongPointer xShapeInfo,
           LongPointer dxShapeInfo,
           PointerPointer targets,
           LongPointer zShapeInfo,
           LongPointer tadShapeInfo,
           LongPointer tadOffsets);

 void sort(PointerPointer extraPointers,
           Pointer x, LongPointer xShapeInfo,
           Pointer dx, LongPointer dxShapeInfo,
           boolean descending);



 void sortTad( PointerPointer extraPointers, Pointer hX, LongPointer hXShapeInfo, Pointer dX,
               LongPointer dXShapeInfo, LongPointer dimension,  long dimensionLength,
               LongPointer tadShapeInfo, LongPointer tadOffsets,  boolean descending);
 void sortTad( PointerPointer extraPointers, Pointer hX, LongBuffer hXShapeInfo, Pointer dX,
               LongBuffer dXShapeInfo, LongBuffer dimension,  long dimensionLength,
               LongBuffer tadShapeInfo, LongBuffer tadOffsets,  boolean descending);
 void sortTad( PointerPointer extraPointers, Pointer hX, long[] hXShapeInfo, Pointer dX,
               long[] dXShapeInfo, long[] dimension,long dimensionLength,
               long[] tadShapeInfo, long[] tadOffsets, boolean descending);

 void sortTadByKey( PointerPointer extraPointers, Pointer x, LongPointer xShapeInfo, Pointer dX,
                    LongPointer dXShapeInfo, Pointer y, LongPointer yShapeInfo, Pointer dy,
                    LongPointer dyShapeInfo,  LongPointer dimension, long dimensionLength,boolean descending);
 void sortTadByKey(PointerPointer extraPointers, Pointer x, LongBuffer xShapeInfo, Pointer dX,
                   LongBuffer dXShapeInfo, Pointer y, LongBuffer yShapeInfo, Pointer dy,
                   LongBuffer dyShapeInfo,LongBuffer dimension, long dimensionLength,  boolean descending);
 void sortTadByKey( PointerPointer extraPointers, Pointer x, long[] xShapeInfo, Pointer dX,
                    long[] dXShapeInfo, Pointer y, long[] yShapeInfo, Pointer dy,
                    long[] dyShapeInfo,  long[] dimension, long dimensionLength,  boolean descending);

 void sortTadByValue( PointerPointer extraPointers, Pointer x, LongPointer xShapeInfo, Pointer dx,
                      LongPointer dxShapeInfo, Pointer y, LongPointer yShapeInfo, Pointer dy,
                      LongPointer dyShapeInfo,  LongPointer dimension,
                      long dimensionLength,
                      boolean descending);
 void sortTadByValue( PointerPointer extraPointers, Pointer x, LongBuffer xShapeInfo, Pointer dx,
                      LongBuffer dxShapeInfo, Pointer y, LongBuffer yShapeInfo, Pointer dy,
                      LongBuffer dyShapeInfo,  LongBuffer dimension,
                      long dimensionLength,
                      boolean descending);
 void sortTadByValue( PointerPointer extraPointers, Pointer x, long[] xShapeInfo, Pointer dx,
                      long[] dxShapeInfo, Pointer y, long[] yShapeInfo, Pointer dy,
                      long[] dyShapeInfo,  long[] dimension,
                      long dimensionLength,
                      boolean descending);


 void sortCooIndices(PointerPointer extraPointers,  LongPointer indices, Pointer x, long length,  LongPointer shapeInfo);


 /**
  *
  * @param extraPointers     not used
  * @param indices           DataBuffer containing COO indices for a sparse matrix that is to be raveled/flattened
  * @param flatIndices       DataBuffer where the raveled/flattened indices are to be written to
  * @param length            number of non-zero entries (length of flatIndices)
  * @param shapeInfo   DataBuffer with ShapeInfo for the full matrix to be flattened
  * @param mode              clipMode determines the strategy to use if some of the the passed COO indices does
  *                          not fit into the shape determined by fullShapeBuffer
  *                              0   throw an exception (default)
  *                              1   wrap around shape
  *                              2   clip to shape
  */
 void ravelMultiIndex(PointerPointer extraPointers,  LongPointer indices,  LongPointer flatIndices, long length,  LongPointer shapeInfo, int mode);

 /**
  *
  * @param extraPointers     not used
  * @param indices           DataBuffer where the unraveled COO indices are to be written
  * @param flatIndices       DataBuffer containing the raveled/flattened indices to be unravel
  * @param length            number of non-zero entries (length of flatIndices)
  * @param shapeInfo   DataBuffer with ShapeInfo for the full matrix to be unraveled
  */
 void unravelIndex(PointerPointer extraPointers,  LongPointer indices,  LongPointer flatIndices, long length,  LongPointer shapeInfo);


 LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);

 void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);

 OpaqueResultWrapper executeFlatGraph(PointerPointer extraPointers, Pointer flatBufferPointer);

 long getResultWrapperSize(OpaqueResultWrapper ptr);
 Pointer getResultWrapperPointer(OpaqueResultWrapper ptr);

 String getAllCustomOps();

 String getAllOperations();

 int execCustomOp2(PointerPointer extraPointers, long opHashCode, Pointer context);

 int execCustomOp(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, DoublePointer tArgs, int numTArgs,  LongPointer iArgs, int numIArgs,  BooleanPointer bArgs, int numBArgs, boolean isInplace);

 OpaqueShapeList calculateOutputShapes(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs,  LongPointer iArgs, int numIArgs);

 OpaqueShapeList calculateOutputShapes2(PointerPointer extraPointers, long hash, PointerPointer inputBunffers, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs,  LongPointer iArgs, int numIArgs,  BooleanPointer bArgs, int numBArgs,  IntPointer dArgs, int numDArgs);




 long getShapeListSize(OpaqueShapeList list);
 LongPointer getShape(OpaqueShapeList list, long i);

 int registerGraph(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);

 OpaqueVariablesSet executeStoredGraph(PointerPointer extraPointers, long graphId, PointerPointer inputBuffers, PointerPointer inputShapes, IntPointer inputIndices, int numInputs);

 long getVariablesSetSize(OpaqueVariablesSet set);
 int getVariablesSetStatus(OpaqueVariablesSet set);
 OpaqueVariable getVariable(OpaqueVariablesSet set, long i);
 int getVariableId(OpaqueVariable variable);
 int getVariableIndex(OpaqueVariable variable);
 String getVariableName(OpaqueVariable variable);
 LongPointer getVariableShape(OpaqueVariable variable);
 Pointer getVariableBuffer(OpaqueVariable variable);

 void deleteResultWrapper(Pointer ptr);

 void deleteShapeList(Pointer ptr);

 int unregisterGraph(PointerPointer extraPointers, long graphId);

 void deleteIntArray(Pointer pointer);

 void deleteLongArray(Pointer pointer);

 void deletePointerArray(Pointer pointer);

 void deleteNPArrayStruct(Pointer pointer);

 void deleteNPArrayMap(Pointer pointer);

 void deleteVariablesSet(OpaqueVariablesSet pointer);

 // GraphState creation
 Pointer getGraphState(long id);

 void deleteGraphState(Pointer state);

 int estimateThreshold(PointerPointer extraPointers, Pointer x, LongPointer xShapeInfo, int N, float threshold);

 // this method executes op that requires scope to be present: if/while/cond/whatever
 int execCustomOpWithScope(PointerPointer extraPointers, Pointer state, long opHash, long[] scopes, int numScopes, PointerPointer inputBuffers, PointerPointer inputShapes, int numInputs, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs);

 void scatterUpdate(PointerPointer extraPointers, int opCode, int numOfUpdates,
                    Pointer hX,  LongPointer hXShapeInfo,  LongPointer hxOffsets,
                    Pointer dX,  LongPointer dXShapeInfo,  LongPointer dxOffsets,
                    Pointer hY,  LongPointer hYShapeInfo,  LongPointer hyOffsets,
                    Pointer dY,  LongPointer dYShapeInfo,  LongPointer dyOffsets,
                    Pointer hIndices,  LongPointer hIndicesShapeInfo, Pointer dIndices,  LongPointer dIndicesShapeInfo);

 //void fillUtf8String(PointerPointer extraPointers, String[] string, int numStrings, Pointer buffer);
 Pointer createUtf8String(PointerPointer extraPointers, String string, int length);
 long getUtf8StringLength(PointerPointer extraPointers, Pointer ptr);
 BytePointer getUtf8StringBuffer(PointerPointer extraPointers, Pointer ptr);
 void deleteUtf8String(PointerPointer extraPointers, Pointer ptr);


 void inspectArray(PointerPointer extraPointers, Pointer buffer,  LongPointer shapeInfo, Pointer specialBuffer,  LongPointer specialShapeInfo, @Cast("nd4j::DebugInfo *") Pointer debugInfo);

 /**
  * this method tries to read numBytes bytes from buffer to provoke crash in certain scenarios
  */
 void tryPointer(Pointer extras, Pointer buffer, int numBytesToRead);


 /**
  * This method returns data type from npy header
  *
  * PLEASE NOTE: dont use output directly, use DataType.fromInt(output) instead
  * @param numpyHeader
  * @return
  */
 int dataTypeFromNpyHeader(Pointer numpyHeader);

 OpaqueConstantShapeBuffer shapeBuffer(int rank, LongPointer shape, LongPointer strides, int dtype, char order, long ews, boolean empty);

 OpaqueConstantShapeBuffer shapeBufferEx(int rank, LongPointer shape, LongPointer strides, int dtype, char order, long ews, long extras);

 OpaqueConstantDataBuffer constantBufferDouble(int dtype, DoublePointer data, int length);

 OpaqueConstantDataBuffer constantBufferLong(int dtype, LongPointer data, int length);

 Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf);
 Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf);
 long getConstantDataBufferLength(OpaqueConstantDataBuffer dbf);

 Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf);
 Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf);

 void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer state);
 void deleteConstantDataBuffer(OpaqueConstantDataBuffer state);

 OpaqueContext createGraphContext(int nodeId);
 OpaqueRandomGenerator getGraphContextRandomGenerator(OpaqueContext ptr);
 void markGraphContextInplace(OpaqueContext ptr, boolean reallyInplace);
 void setGraphContextCudaContext(OpaqueContext ptr, Pointer stream, Pointer reductionPointer, Pointer allocationPointer);
 void setGraphContextInputBuffer(OpaqueContext ptr, int index, OpaqueDataBuffer databuffer, OpaqueDataBuffer shapeInfo, OpaqueDataBuffer specialShapeInfo);
 void setGraphContextOutputBuffer(OpaqueContext ptr, int index, OpaqueDataBuffer databuffer, OpaqueDataBuffer shapeInfo, OpaqueDataBuffer specialShapeInfo);



 void setGraphContextInputArrays(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                 PointerPointer specialBuffer, PointerPointer specialShapeInfo);
 void setGraphContextOutputArrays(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                  PointerPointer specialBuffer, PointerPointer specialShapeInfo);
 void setGraphContextInputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                  PointerPointer specialShapeInfo);

 void setGraphContextOutputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                   PointerPointer specialShapeInfo);


 void setShapeBuffer(@Cast("sd::LongType*") LongPointer inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") LongPointer bufferToSet,char order/*='c'*/,int elementWiseStride/*=1*/,@Cast("bool") boolean isEmpty/*=false*/,@Cast("bool") boolean isView/*=false*/);
 void setShapeBuffer(@Cast("sd::LongType*") LongPointer inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") LongPointer bufferToSet);
 void setShapeBuffer(@Cast("sd::LongType*") LongBuffer inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") LongBuffer bufferToSet,char order/*='c'*/,int elementWiseStride/*=1*/,@Cast("bool") boolean isEmpty/*=false*/,@Cast("bool") boolean isView/*=false*/);
 void setShapeBuffer(@Cast("sd::LongType*") LongBuffer inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") LongBuffer bufferToSet);
 void setShapeBuffer(@Cast("sd::LongType*") long[] inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") long[] bufferToSet,char order/*='c'*/,int elementWiseStride/*=1*/,@Cast("bool") boolean isEmpty/*=false*/,@Cast("bool") boolean isView/*=false*/);
 void setShapeBuffer(@Cast("sd::LongType*") long[] inputShapeData,@Cast("sd::DataType") int dt,@Cast("sd::LongType*") long[] bufferToSet);

 void setGraphContextTArguments(OpaqueContext ptr, DoublePointer arguments, int numberOfArguments);
 void setGraphContextIArguments(OpaqueContext ptr, LongPointer arguments, int numberOfArguments);
 void setGraphContextDArguments(OpaqueContext ptr, IntPointer arguments, int numberOfArguments);
 void setGraphContextBArguments(OpaqueContext ptr, BooleanPointer arguments, int numberOfArguments);
 void ctxAllowHelpers(OpaqueContext ptr, boolean reallyAllow);
 void ctxSetExecutionMode(OpaqueContext ptr, int execMode);
 void ctxShapeFunctionOverride(OpaqueContext ptr, boolean reallyOverride);
 void ctxPurge(OpaqueContext ptr);
 void deleteGraphContext(OpaqueContext ptr);

 OpaqueRandomGenerator createRandomGenerator(long rootSeed, long nodeSeed);
 long getRandomGeneratorRootState(OpaqueRandomGenerator ptr);
 long getRandomGeneratorNodeState(OpaqueRandomGenerator ptr);
 void setRandomGeneratorStates(OpaqueRandomGenerator ptr,  long rootSeed/*=0*/,  long nodeSeed/*=0*/);
 float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr,  long index);
 double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr,  long index);
 int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr,  long index);
 long getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr,  long index);
 float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr);
 double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr);
 int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr);
 long getRandomGeneratorNextLong(OpaqueRandomGenerator ptr);
 void deleteRandomGenerator(OpaqueRandomGenerator ptr);



 long getCachedMemory(int deviceId);

 OpaqueLaunchContext defaultLaunchContext();

 Pointer lcScalarPointer(OpaqueLaunchContext lc);
 Pointer lcReductionPointer(OpaqueLaunchContext lc);
 Pointer lcAllocationPointer(OpaqueLaunchContext lc);
 Pointer lcExecutionStream(OpaqueLaunchContext lc);
 Pointer lcCopyStream(OpaqueLaunchContext lc);
 Pointer lcBlasHandle(OpaqueLaunchContext lc);
 Pointer lcSolverHandle(OpaqueLaunchContext lc);

 int lastErrorCode();
 String lastErrorMessage();

 boolean isBlasVersionMatches(int major, int minor, int build);

 int  binaryLevel();
 int optimalLevel();

 boolean isMinimalRequirementsMet();
 boolean isOptimalRequirementsMet();


 OpaqueDataBuffer allocateDataBuffer(long elements, int dataType, boolean allocateBoth);
 OpaqueDataBuffer dbAllocateDataBuffer(long elements, int dataType, boolean allocateBoth);
 OpaqueDataBuffer dbCreateExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special);
 OpaqueDataBuffer dbCreateView(OpaqueDataBuffer dataBuffer, long length, long offset);
 Pointer dbPrimaryBuffer(OpaqueDataBuffer dataBuffer);
 Pointer dbSpecialBuffer(OpaqueDataBuffer dataBuffer);
 long dbBufferLength(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer);
 void dbExpandBuffer(OpaqueDataBuffer dataBuffer, long elements);
 void dbAllocatePrimaryBuffer(OpaqueDataBuffer dataBuffer);
 void dbAllocateSpecialBuffer(OpaqueDataBuffer dataBuffer);
 void dbSetPrimaryBuffer(OpaqueDataBuffer dataBuffer, Pointer primaryBuffer, long numBytes);
 void dbSetSpecialBuffer(OpaqueDataBuffer dataBuffer, Pointer specialBuffer, long numBytes);
 void dbSyncToSpecial(OpaqueDataBuffer dataBuffer);
 void dbSyncToPrimary(OpaqueDataBuffer dataBuffer);
 void dbTickHostRead(OpaqueDataBuffer dataBuffer);
 void dbTickHostWrite(OpaqueDataBuffer dataBuffer);
 void dbTickDeviceRead(OpaqueDataBuffer dataBuffer);
 void dbTickDeviceWrite(OpaqueDataBuffer dataBuffer);
 void deleteDataBuffer(OpaqueDataBuffer dataBuffer);
 void dbClose(OpaqueDataBuffer dataBuffer);
 int  dbLocality(OpaqueDataBuffer dataBuffer);
 int  dbDeviceId(OpaqueDataBuffer dataBuffer);
 int  dbUseCount(OpaqueDataBuffer dataBuffer);
 void  dbSetDeviceId(OpaqueDataBuffer dataBuffer, int deviceId);
 void dbExpand(OpaqueDataBuffer dataBuffer, long newLength);

 boolean isFuncTrace();
 /**
  * Gets the build information of the backend
  *
  * @return
  */
 String buildInfo();
}
