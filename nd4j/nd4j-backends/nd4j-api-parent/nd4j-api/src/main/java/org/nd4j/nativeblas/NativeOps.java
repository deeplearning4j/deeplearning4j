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

import java.nio.LongBuffer;


/**
 * A common interface for misc operations
 * needed from c++
 *
 */
public interface NativeOps {


    /**
     * Prints device buffers.
     * @param buffer
     */
    void printDeviceBuffer(org.nd4j.nativeblas.OpaqueDataBuffer buffer);

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
                               @Cast("sd::LongType *") LongPointer xShapeInfo,
                               @Cast("sd::LongType *") LongPointer dXShapeInfo,
                               Pointer extraParams,
                               OpaqueDataBuffer z,
                               @Cast("sd::LongType *") LongPointer zShapeInfo,
                               @Cast("sd::LongType *") LongPointer dZShapeInfo);

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
                         @Cast("sd::LongType *") LongPointer xShapeInfo,
                         @Cast("sd::LongType *") LongPointer dXShapeInfo,
                         Pointer extraParams,
                         OpaqueDataBuffer result,
                         @Cast("sd::LongType *") LongPointer resultShapeInfoBuffer,
                         @Cast("sd::LongType *") LongPointer dResultShapeInfoBuffer,
                         OpaqueDataBuffer hDimension,
                         @Cast("sd::LongType *") LongPointer hDimensionShape,
                         @Cast("sd::LongType *") LongPointer dDimensionShape);

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
                       @Cast("sd::LongType *") LongPointer xShapeInfo,
                       @Cast("sd::LongType *") LongPointer dxShapeInfo,
                       OpaqueDataBuffer y,
                       @Cast("sd::LongType *") LongPointer yShapeInfo,
                       @Cast("sd::LongType *") LongPointer dyShapeInfo,
                       OpaqueDataBuffer result,
                       @Cast("sd::LongType *") LongPointer resultShapeInfo,
                       @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                       OpaqueDataBuffer hDimension,
                       @Cast("sd::LongType *") LongPointer hDimensionShape,
                       @Cast("sd::LongType *") LongPointer dDimensionShape);

    void execBroadcastBool(PointerPointer extraPointers,
                           int opNum,
                           OpaqueDataBuffer x,
                           @Cast("sd::LongType *") LongPointer xShapeInfo,
                           @Cast("sd::LongType *") LongPointer dxShapeInfo,
                           OpaqueDataBuffer y,
                           @Cast("sd::LongType *") LongPointer yShapeInfo,
                           @Cast("sd::LongType *") LongPointer dyShapeInfo,
                           OpaqueDataBuffer result,
                           @Cast("sd::LongType *") LongPointer resultShapeInfo,
                           @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                           Pointer extraParams,
                           OpaqueDataBuffer hDimension,
                           @Cast("sd::LongType *") LongPointer hDimensionShape,
                           @Cast("sd::LongType *") LongPointer dDimensionShape);


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
                               @Cast("sd::LongType *") LongPointer xShapeInfo,
                               @Cast("sd::LongType *") LongPointer dxShapeInfo,
                               OpaqueDataBuffer y,
                               @Cast("sd::LongType *") LongPointer yShapeInfo,
                               @Cast("sd::LongType *") LongPointer dyShapeInfo,
                               OpaqueDataBuffer result,
                               @Cast("sd::LongType *") LongPointer resultShapeInfo,
                               @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                               Pointer extraParams);

    void execPairwiseTransformBool(PointerPointer extraPointers,
                                   int opNum,
                                   OpaqueDataBuffer x,
                                   @Cast("sd::LongType *") LongPointer xShapeInfo,
                                   @Cast("sd::LongType *") LongPointer dxShapeInfo,
                                   OpaqueDataBuffer y,
                                   @Cast("sd::LongType *") LongPointer yShapeInfo,
                                   @Cast("sd::LongType *") LongPointer dyShapeInfo,
                                   OpaqueDataBuffer result,
                                   @Cast("sd::LongType *") LongPointer resultShapeInfo,
                                   @Cast("sd::LongType *") LongPointer dresultShapeInfo,
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
                         @Cast("sd::LongType *") LongPointer xShapeInfo,
                         @Cast("sd::LongType *") LongPointer dxShapeInfo,
                         Pointer extraParams,
                         OpaqueDataBuffer result,
                         @Cast("sd::LongType *") LongPointer resultShapeInfo,
                         @Cast("sd::LongType *") LongPointer dresultShapeInfo);


    void execReduceSame(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        Pointer extraParams,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfo,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfo);


    void execReduceBool(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        Pointer extraParams,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfo,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfo);


    void execReduceLong(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        Pointer extraParams,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfo,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfo);

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
                          @Cast("sd::LongType *") LongPointer xShapeInfo,
                          @Cast("sd::LongType *") LongPointer dxShapeInfo,
                          Pointer extraParams,
                          OpaqueDataBuffer result,
                          @Cast("sd::LongType *") LongPointer resultShapeInfo,
                          @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                          OpaqueDataBuffer hDimension,
                          @Cast("sd::LongType *") LongPointer hDimensionShape,
                          @Cast("sd::LongType *") LongPointer dDimensionShape);


    void execReduceSame2(PointerPointer extraPointers,
                         int opNum,
                         OpaqueDataBuffer x,
                         @Cast("sd::LongType *") LongPointer xShapeInfo,
                         @Cast("sd::LongType *") LongPointer dxShapeInfo,
                         Pointer extraParams,
                         OpaqueDataBuffer result,
                         @Cast("sd::LongType *") LongPointer resultShapeInfo,
                         @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                         OpaqueDataBuffer hDimension,
                         @Cast("sd::LongType *") LongPointer hDimensionShape,
                         @Cast("sd::LongType *") LongPointer dDimensionShape);

    void execReduceBool2(PointerPointer extraPointers,
                         int opNum,
                         OpaqueDataBuffer x,
                         @Cast("sd::LongType *") LongPointer xShapeInfo,
                         @Cast("sd::LongType *") LongPointer dxShapeInfo,
                         Pointer extraParams,
                         OpaqueDataBuffer result,
                         @Cast("sd::LongType *") LongPointer resultShapeInfo,
                         @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                         OpaqueDataBuffer hDimension,
                         @Cast("sd::LongType *") LongPointer hDimensionShape,
                         @Cast("sd::LongType *") LongPointer dDimensionShape);

    void execReduceLong2(PointerPointer extraPointers,
                         int opNum,
                         OpaqueDataBuffer x,
                         @Cast("sd::LongType *") LongPointer xShapeInfo,
                         @Cast("sd::LongType *") LongPointer dxShapeInfo,
                         Pointer extraParams,
                         OpaqueDataBuffer result,
                         @Cast("sd::LongType *") LongPointer resultShapeInfo,
                         @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                         OpaqueDataBuffer hDimension,
                         @Cast("sd::LongType *") LongPointer hDimensionShape,
                         @Cast("sd::LongType *") LongPointer dDimensionShape);

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
                     @Cast("sd::LongType *") LongPointer xShapeInfo,
                     @Cast("sd::LongType *") LongPointer dxShapeInfo,
                     Pointer extraParamsVals,
                     OpaqueDataBuffer y,
                     @Cast("sd::LongType *") LongPointer yShapeInfo,
                     @Cast("sd::LongType *") LongPointer dyShapeInfo,
                     OpaqueDataBuffer result,
                     @Cast("sd::LongType *") LongPointer resultShapeInfo,
                     @Cast("sd::LongType *") LongPointer dresultShapeInfo);

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
                           @Cast("sd::LongType *") LongPointer xShapeInfo,
                           @Cast("sd::LongType *") LongPointer dxShapeInfo,
                           Pointer extraParamsVals,
                           OpaqueDataBuffer y,
                           @Cast("sd::LongType *") LongPointer yShapeInfo,
                           @Cast("sd::LongType *") LongPointer dyShapeInfo,
                           OpaqueDataBuffer z,
                           @Cast("sd::LongType *") LongPointer zShapeInfo,
                           @Cast("sd::LongType *") LongPointer dzShapeInfo);

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
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        Pointer extraParamsVals,
                        OpaqueDataBuffer y,
                        @Cast("sd::LongType *") LongPointer yShapeInfo,
                        @Cast("sd::LongType *") LongPointer dyShapeInfo,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfoBuffer,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfoBuffer,
                        OpaqueDataBuffer hDimension,
                        @Cast("sd::LongType *") LongPointer hDimensionShape,
                        @Cast("sd::LongType *") LongPointer dDimensionShape,
                        @Cast("sd::LongType *") LongPointer tadOnlyShapeInfo, @Cast("sd::LongType *") LongPointer tadOffsets,
                        @Cast("sd::LongType *") LongPointer yTadOnlyShapeInfo, @Cast("sd::LongType *") LongPointer yTadOffsets);

    void execReduce3All(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        Pointer extraParamsVals,
                        OpaqueDataBuffer y,
                        @Cast("sd::LongType *") LongPointer yShapeInfo,
                        @Cast("sd::LongType *") LongPointer dyShapeInfo,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfoBuffer,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfoBuffer,
                        OpaqueDataBuffer hDimension,
                        @Cast("sd::LongType *") LongPointer hDimensionShape,
                        @Cast("sd::LongType *") LongPointer dDimensionShape,
                        @Cast("sd::LongType *") LongPointer xTadShape,
                        @Cast("sd::LongType *") LongPointer xOffsets,
                        @Cast("sd::LongType *") LongPointer yTadShape,
                        @Cast("sd::LongType *") LongPointer yOffsets);


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
                    @Cast("sd::LongType *") LongPointer xShapeInfo,
                    @Cast("sd::LongType *") LongPointer dxShapeInfo,
                    OpaqueDataBuffer result,
                    @Cast("sd::LongType *") LongPointer resultShapeInfo,
                    @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                    OpaqueDataBuffer scalar,
                    @Cast("sd::LongType *") LongPointer scalarShapeInfo,
                    @Cast("sd::LongType *") LongPointer dscalarShapeInfo,
                    Pointer extraParams);

    void execScalarBool(PointerPointer extraPointers,
                        int opNum,
                        OpaqueDataBuffer x,
                        @Cast("sd::LongType *") LongPointer xShapeInfo,
                        @Cast("sd::LongType *") LongPointer dxShapeInfo,
                        OpaqueDataBuffer result,
                        @Cast("sd::LongType *") LongPointer resultShapeInfo,
                        @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                        OpaqueDataBuffer scalar,
                        @Cast("sd::LongType *") LongPointer scalarShapeInfo,
                        @Cast("sd::LongType *") LongPointer dscalarShapeInfo,
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
                                @Cast("sd::LongType *") LongPointer xShapeInfo,
                                @Cast("sd::LongType *") LongPointer dxShapeInfo,
                                Pointer extraParams,
                                OpaqueDataBuffer z,
                                @Cast("sd::LongType *") LongPointer zShapeInfo,
                                @Cast("sd::LongType *") LongPointer dzShapeInfo,
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
                          @Cast("sd::LongType *") LongPointer xShapeInfo,
                          @Cast("sd::LongType *") LongPointer dxShapeInfo,
                          Pointer extraParams,
                          OpaqueDataBuffer result,
                          @Cast("sd::LongType *") LongPointer resultShapeInfo,
                          @Cast("sd::LongType *") LongPointer dresultShapeInfo,
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
                             @Cast("sd::LongType *") LongPointer xShapeInfo,
                             @Cast("sd::LongType *") LongPointer dxShapeInfo,
                             Pointer extraParams,
                             OpaqueDataBuffer result,
                             @Cast("sd::LongType *") LongPointer resultShapeInfoBuffer,
                             @Cast("sd::LongType *") LongPointer dresultShapeInfoBuffer,
                             OpaqueDataBuffer hDimension,
                             @Cast("sd::LongType *") LongPointer hDimensionShape,
                             @Cast("sd::LongType *") LongPointer dDimensionShape,
                             boolean biasCorrected,
                             @Cast("sd::LongType *") LongPointer tadShapeInfo,
                             @Cast("sd::LongType *") LongPointer tadOffsets);


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
                            @Cast("sd::LongType *") LongPointer xShapeInfo,
                            @Cast("sd::LongType *") LongPointer dxShapeInfo,
                            OpaqueDataBuffer result,
                            @Cast("sd::LongType *") LongPointer resultShapeInfo,
                            @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                            Pointer extraParams);

    void execTransformSame(PointerPointer extraPointers,
                           int opNum,
                           OpaqueDataBuffer x,
                           @Cast("sd::LongType *") LongPointer xShapeInfo,
                           @Cast("sd::LongType *") LongPointer dxShapeInfo,
                           OpaqueDataBuffer result,
                           @Cast("sd::LongType *") LongPointer resultShapeInfo,
                           @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                           Pointer extraParams);

    void execTransformStrict(PointerPointer extraPointers,
                             int opNum,
                             OpaqueDataBuffer x,
                             @Cast("sd::LongType *") LongPointer xShapeInfo,
                             @Cast("sd::LongType *") LongPointer dxShapeInfo,
                             OpaqueDataBuffer result,
                             @Cast("sd::LongType *") LongPointer resultShapeInfo,
                             @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                             Pointer extraParams);

    void execTransformBool(PointerPointer extraPointers,
                           int opNum,
                           OpaqueDataBuffer x,
                           @Cast("sd::LongType *") LongPointer xShapeInfo,
                           @Cast("sd::LongType *") LongPointer dxShapeInfo,
                           OpaqueDataBuffer result,
                           @Cast("sd::LongType *") LongPointer resultShapeInfo,
                           @Cast("sd::LongType *") LongPointer dresultShapeInfo,
                           Pointer extraParams);

    void execTransformAny(PointerPointer extraPointers,
                          int opNum,
                          OpaqueDataBuffer x,
                          @Cast("sd::LongType *") LongPointer xShapeInfo,
                          @Cast("sd::LongType *") LongPointer dxShapeInfo,
                          OpaqueDataBuffer result,
                          @Cast("sd::LongType *") LongPointer resultShapeInfo,
                          @Cast("sd::LongType *") LongPointer dresultShapeInfo,
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
                       @Cast("sd::LongType *") LongPointer xShapeInfo,
                       @Cast("sd::LongType *") LongPointer dxShapeInfo,
                       OpaqueDataBuffer z,
                       @Cast("sd::LongType *") LongPointer zShapeInfo,
                       @Cast("sd::LongType *") LongPointer dzShapeInfo,
                       OpaqueDataBuffer scalars,
                       @Cast("sd::LongType *") LongPointer scalarShapeInfo,
                       @Cast("sd::LongType *") LongPointer dscalarShapeInfo,
                       Pointer extraParams,
                       OpaqueDataBuffer hDimension,
                       @Cast("sd::LongType *") LongPointer hDimensionShape,
                       @Cast("sd::LongType *") LongPointer dDimensionShape,
                       @Cast("sd::LongType *") LongPointer tadShapeInfo,
                       @Cast("sd::LongType *") LongPointer tadOffsets,
                       @Cast("sd::LongType *") LongPointer tadShapeInfoZ,
                       @Cast("sd::LongType *") LongPointer tadOffsetsZ);

    void execScalarBoolTad(PointerPointer extraPointers,
                           int opNum,
                           OpaqueDataBuffer x,
                           @Cast("sd::LongType *") LongPointer xShapeInfo,
                           @Cast("sd::LongType *") LongPointer dxShapeInfo,
                           OpaqueDataBuffer z,
                           @Cast("sd::LongType *") LongPointer zShapeInfo,
                           @Cast("sd::LongType *") LongPointer dzShapeInfo,
                           OpaqueDataBuffer scalars,
                           @Cast("sd::LongType *") LongPointer scalarShapeInfo,
                           @Cast("sd::LongType *") LongPointer dscalarShapeInfo,
                           Pointer extraParams,
                           OpaqueDataBuffer hDimension,
                           @Cast("sd::LongType *") LongPointer hDimensionShape,
                           @Cast("sd::LongType *") LongPointer dDimensionShape,
                           @Cast("sd::LongType *") LongPointer tadShapeInfo,
                           @Cast("sd::LongType *") LongPointer tadOffsets,
                           @Cast("sd::LongType *") LongPointer tadShapeInfoZ,
                           @Cast("sd::LongType *") LongPointer tadOffsetsZ);


    void specialConcat(PointerPointer extraPointers,
                       int dimension,
                       int numArrays,
                       PointerPointer data, PointerPointer inputShapeInfo,
                       Pointer results, @Cast("sd::LongType *") LongPointer resultShapeInfo,
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

    void setVedaDeviceLibFolder(String path);

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

    OpaqueTadPack tadOnlyShapeInfo(LongPointer shapeInfo, IntPointer dimension, int dimensionLength);

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
                  @Cast("sd::LongType *") LongPointer xShapeInfo,
                  @Cast("sd::LongType *") LongPointer dxShapeInfo,
                  OpaqueDataBuffer z,
                  @Cast("sd::LongType *") LongPointer zShapeInfo,
                  @Cast("sd::LongType *") LongPointer dzShapeInfo,
                  long n,
                  @Cast("sd::LongType *") LongPointer indexes,
                  @Cast("sd::LongType *") LongPointer tadShapeInfo,
                  @Cast("sd::LongType *") LongPointer tadOffsets,
                  @Cast("sd::LongType *") LongPointer zTadShapeInfo,
                  @Cast("sd::LongType *") LongPointer zTadOffsets);


    ///////////////////////

    void average(PointerPointer extraPointers,
                 PointerPointer x, @Cast("sd::LongType *") LongPointer xShapeInfo,
                 PointerPointer dx, @Cast("sd::LongType *") LongPointer dxShapeInfo,
                 Pointer z, @Cast("sd::LongType *") LongPointer zShapeInfo,
                 Pointer dz, @Cast("sd::LongType *") LongPointer dzShapeInfo,
                 int n,
                 long length,
                 boolean propagate);

    ///////////////////////

    void accumulate(PointerPointer extraPointers,
                    PointerPointer x, @Cast("sd::LongType *") LongPointer xShapeInfo,
                    PointerPointer dx, @Cast("sd::LongType *") LongPointer dxShapeInfo,
                    Pointer z, @Cast("sd::LongType *") LongPointer zShapeInfo,
                    Pointer dz, @Cast("sd::LongType *") LongPointer dzShapeInfo,
                    int n,
                    long length);

    ///////////////////////

    void enableP2P(boolean reallyEnable);

    void checkP2P();

    boolean isP2PAvailable();

    //

    void shuffle(PointerPointer extraPointers,
                 PointerPointer x, @Cast("sd::LongType *") PointerPointer xShapeInfo,
                 PointerPointer dx, @Cast("sd::LongType *") PointerPointer dxShapeInfo,
                 PointerPointer z, @Cast("sd::LongType *") PointerPointer zShapeInfo,
                 PointerPointer dz, @Cast("sd::LongType *") PointerPointer dzShapeInfo,
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
                                                Pointer x, @Cast("sd::LongType *") LongPointer xShape,
                                                Pointer dx, @Cast("sd::LongType *") LongPointer dxShape,
                                                Pointer y, @Cast("sd::LongType *") LongPointer yShape,
                                                Pointer dy, @Cast("sd::LongType *") LongPointer dyShape,
                                                Pointer z, @Cast("sd::LongType *") LongPointer zShape,
                                                Pointer dz, @Cast("sd::LongType *") LongPointer dzShape,
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
                    @Cast("sd::LongType *") LongPointer zShapeBuffer,
                    @Cast("sd::LongType *") LongPointer dzShapeBuffer,
                    Pointer extraArguments);

    void execRandom3(PointerPointer extraPointers,
                     int opNum,
                     Pointer state,
                     OpaqueDataBuffer x,
                     @Cast("sd::LongType *") LongPointer xShapeBuffer,
                     @Cast("sd::LongType *") LongPointer dxShapeBuffer,
                     OpaqueDataBuffer y,
                     @Cast("sd::LongType *") LongPointer yShapeBuffer,
                     @Cast("sd::LongType *") LongPointer dyShapeBuffer,
                     OpaqueDataBuffer z,
                     @Cast("sd::LongType *") LongPointer zShapeBuffer,
                     @Cast("sd::LongType *") LongPointer dzShapeBuffer,
                     Pointer extraArguments);

    void execRandom2(PointerPointer extraPointers,
                     int opNum,
                     Pointer state,
                     OpaqueDataBuffer x,
                     @Cast("sd::LongType *") LongPointer xShapeBuffer,
                     @Cast("sd::LongType *") LongPointer dxShapeBuffer,
                     OpaqueDataBuffer z,
                     @Cast("sd::LongType *") LongPointer zShapeBuffer,
                     @Cast("sd::LongType *") LongPointer dzShapeBuffer,
                     Pointer extraArguments);

    ////////////////////


    Pointer initRandom(PointerPointer extraPointers, long seed, long numberOfElements, Pointer pointerToBuffer);

    void refreshBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    void reSeedBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    void destroyRandom(Pointer pointer);


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
              @Cast("sd::LongType *") LongPointer xShapeInfo,
              @Cast("sd::LongType *") LongPointer dxShapeInfo,
              PointerPointer targets,
              @Cast("sd::LongType *") LongPointer zShapeInfo,
              @Cast("sd::LongType *") LongPointer tadShapeInfo,
              @Cast("sd::LongType *") LongPointer tadOffsets);

    void sort(PointerPointer extraPointers,
              Pointer x, @Cast("sd::LongType *") LongPointer xShapeInfo,
              Pointer dx, @Cast("sd::LongType *") LongPointer dxShapeInfo,
              boolean descending);


    void sortTad(PointerPointer extraPointers,
                 Pointer x, @Cast("sd::LongType *") LongPointer xShapeInfo,
                 Pointer dx, @Cast("sd::LongType *") LongPointer dxShapeInfo,
                 IntPointer dimension,
                 int dimensionLength,
                 @Cast("sd::LongType *") LongPointer tadShapeInfo,
                 @Cast("sd::LongType *") LongPointer tadOffsets,
                 boolean descending);


    void sortCooIndices(PointerPointer extraPointers, @Cast("sd::LongType *") LongPointer indices, Pointer x, long length, @Cast("sd::LongType *") LongPointer shapeInfo);


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
    void ravelMultiIndex(PointerPointer extraPointers, @Cast("sd::LongType *") LongPointer indices, @Cast("sd::LongType *") LongPointer flatIndices, long length, @Cast("sd::LongType *") LongPointer shapeInfo, int mode);

    /**
     *
     * @param extraPointers     not used
     * @param indices           DataBuffer where the unraveled COO indices are to be written
     * @param flatIndices       DataBuffer containing the raveled/flattened indices to be unravel
     * @param length            number of non-zero entries (length of flatIndices)
     * @param shapeInfo   DataBuffer with ShapeInfo for the full matrix to be unraveled
     */
    void unravelIndex(PointerPointer extraPointers, @Cast("sd::LongType *") LongPointer indices, @Cast("sd::LongType *") LongPointer flatIndices, long length, @Cast("sd::LongType *") LongPointer shapeInfo);


    LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);

    void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);

    OpaqueResultWrapper executeFlatGraph(PointerPointer extraPointers, Pointer flatBufferPointer);

    long getResultWrapperSize(OpaqueResultWrapper ptr);
    Pointer getResultWrapperPointer(OpaqueResultWrapper ptr);

    String getAllCustomOps();

    String getAllOperations();

    int execCustomOp2(PointerPointer extraPointers, long opHashCode, Pointer context);

    int execCustomOp(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, DoublePointer tArgs, int numTArgs, @Cast("sd::LongType *") LongPointer iArgs, int numIArgs, @Cast("bool *") BooleanPointer bArgs, int numBArgs, boolean isInplace);

    OpaqueShapeList calculateOutputShapes(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("sd::LongType *") LongPointer iArgs, int numIArgs);

    OpaqueShapeList calculateOutputShapes2(PointerPointer extraPointers, long hash, PointerPointer inputBunffers, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("sd::LongType *") LongPointer iArgs, int numIArgs, @Cast("bool *") BooleanPointer bArgs, int numBArgs, @Cast("int *") IntPointer dArgs, int numDArgs);

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
                       Pointer hX, @Cast("sd::LongType *") LongPointer hXShapeInfo, @Cast("sd::LongType *") LongPointer hxOffsets,
                       Pointer dX, @Cast("sd::LongType *") LongPointer dXShapeInfo, @Cast("sd::LongType *") LongPointer dxOffsets,
                       Pointer hY, @Cast("sd::LongType *") LongPointer hYShapeInfo, @Cast("sd::LongType *") LongPointer hyOffsets,
                       Pointer dY, @Cast("sd::LongType *") LongPointer dYShapeInfo, @Cast("sd::LongType *") LongPointer dyOffsets,
                       Pointer hIndices, @Cast("sd::LongType *") LongPointer hIndicesShapeInfo, Pointer dIndices, @Cast("sd::LongType *") LongPointer dIndicesShapeInfo);

    //void fillUtf8String(PointerPointer extraPointers, String[] string, int numStrings, Pointer buffer);
    Pointer createUtf8String(PointerPointer extraPointers, String string, int length);
    long getUtf8StringLength(PointerPointer extraPointers, Pointer ptr);
    BytePointer getUtf8StringBuffer(PointerPointer extraPointers, Pointer ptr);
    void deleteUtf8String(PointerPointer extraPointers, Pointer ptr);


    void inspectArray(PointerPointer extraPointers, Pointer buffer, @Cast("sd::LongType *") LongPointer shapeInfo, Pointer specialBuffer, @Cast("sd::LongType *") LongPointer specialShapeInfo, @Cast("nd4j::DebugInfo *") Pointer debugInfo);

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
    void setGraphContextInputArray(OpaqueContext ptr, int index, Pointer buffer, Pointer shapeInfo, Pointer specialBuffer, Pointer specialShapeInfo);
    void setGraphContextOutputArray(OpaqueContext ptr, int index, Pointer buffer, Pointer shapeInfo, Pointer specialBuffer, Pointer specialShapeInfo);
    void setGraphContextInputBuffer(OpaqueContext ptr, int index, OpaqueDataBuffer databuffer, Pointer shapeInfo, Pointer specialShapeInfo);
    void setGraphContextOutputBuffer(OpaqueContext ptr, int index, OpaqueDataBuffer databuffer, Pointer shapeInfo, Pointer specialShapeInfo);



     void setGraphContextInputArrays(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                                  PointerPointer specialBuffer, PointerPointer specialShapeInfo);
     void setGraphContextOutputArrays(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                                   PointerPointer specialBuffer, PointerPointer specialShapeInfo);
     void setGraphContextInputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                                   PointerPointer specialShapeInfo);
     void setGraphContextInputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays,  org.nd4j.nativeblas.OpaqueDataBuffer buffer, PointerPointer shapeInfo,
                                                   PointerPointer specialShapeInfo);
     void setGraphContextOutputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays, PointerPointer buffer, PointerPointer shapeInfo,
                                                    PointerPointer specialShapeInfo);
     void setGraphContextOutputBuffers(org.nd4j.nativeblas.OpaqueContext ptr, int numArrays,  org.nd4j.nativeblas.OpaqueDataBuffer buffer, PointerPointer shapeInfo,
                                                    PointerPointer specialShapeInfo);


    void setShapeBuffer( LongPointer inputShapeData, int dt, LongPointer bufferToSet,char order,int elementWiseStride);
    void setShapeBuffer( LongBuffer inputShapeData,  int dt,  LongBuffer bufferToSet,char order,int elementWiseStride);
    void setShapeBuffer( long[] inputShapeData, int dt, long[] bufferToSet,char order,int elementWiseStride);

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
    void setRandomGeneratorStates(OpaqueRandomGenerator ptr, @Cast("sd::LongType") long rootSeed/*=0*/, @Cast("sd::LongType") long nodeSeed/*=0*/);
    float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, @Cast("sd::LongType") long index);
    double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, @Cast("sd::LongType") long index);
    int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, @Cast("sd::LongType") long index);
    long getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, @Cast("sd::LongType") long index);
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

    /**
     * Gets the build information of the backend
     *
     * @return
     */
    String buildInfo();
}
