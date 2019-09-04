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

package org.nd4j.nativeblas;

import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.linalg.api.buffer.Utf8Buffer;


/**
 * Native interface for
 * op execution on cpu
 * @author Adam Gibson
 */
public interface NativeOps {
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
                                                 Pointer x,
                                                 @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                                 Pointer dX,
                                                 @Cast("Nd4jLong *") LongPointer dXShapeInfo,
                                                 Pointer extraParams,
                                                 Pointer z,
                                                 @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                                 Pointer dZ,
                                                 @Cast("Nd4jLong *") LongPointer dZShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void execIndexReduce(PointerPointer extraPointers,
                                         int opNum,
                                         Pointer x,
                                         @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                         Pointer dX,
                                         @Cast("Nd4jLong *") LongPointer dXShapeInfo,
                                         Pointer extraParams,
                                         Pointer result,
                                         @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer,
                                         Pointer dResult,
                                         @Cast("Nd4jLong *") LongPointer dResultShapeInfoBuffer,
                                         Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                         Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);

    /**
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
    void execBroadcast(PointerPointer extraPointers,
                                       int opNum,
                                       Pointer x,
                                       @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                       Pointer dx,
                                       @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                       Pointer y,
                                       @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                       Pointer dy,
                                       @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                       Pointer result,
                                       @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                       Pointer dresult,
                                       @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                       Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                       Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);

    void execBroadcastBool(PointerPointer extraPointers,
                                           int opNum,
                                           Pointer x,
                                           @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                           Pointer dx,
                                           @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                           Pointer y,
                                           @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                           Pointer dy,
                                           @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                           Pointer result,
                                           @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                           Pointer dresult,
                                           @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                           Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                           Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);


    /**
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void execPairwiseTransform(PointerPointer extraPointers,
                                               int opNum,
                                               Pointer x,
                                               @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                               Pointer dx,
                                               @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                               Pointer y,
                                               @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                               Pointer dy,
                                               @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                               Pointer result,
                                               @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                               Pointer dresult,
                                               @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                               Pointer extraParams);

    void execPairwiseTransformBool(PointerPointer extraPointers,
                                                   int opNum,
                                                   Pointer x,
                                                   @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                                   Pointer dx,
                                                   @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                                   Pointer y,
                                                   @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                                   Pointer dy,
                                                   @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                                   Pointer result,
                                                   @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                                   Pointer dresult,
                                                   @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
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
                                         Pointer x,
                                         @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                         Pointer dx,
                                         @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                         Pointer extraParams,
                                         Pointer result,
                                         @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                         Pointer dresult,
                                         @Cast("Nd4jLong *") LongPointer dresultShapeInfo);


    void execReduceSame(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo);


    void execReduceBool(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo);


    void execReduceLong(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo);

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
                                         Pointer x,
                                         @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                         Pointer dx,
                                         @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                         Pointer extraParams,
                                         Pointer result,
                                         @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                         Pointer dresult,
                                         @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                         Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                         Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);


    void execReduceSame2(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                        Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                        Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);

    void execReduceBool2(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                        Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                        Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);

    void execReduceLong2(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx,
                                        @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParams,
                                        Pointer result,
                                        @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult,
                                        @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                        Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                        Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape);

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
                                     Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                     Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                     Pointer extraParamsVals,
                                     Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                     Pointer dy, @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                     Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                     Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    void execReduce3Scalar(PointerPointer extraPointers, int opNum,
                                           Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                           Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                           Pointer extraParamsVals,
                                           Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                           Pointer dy, @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                           Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                           Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo);

    /**
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
    void execReduce3Tad(PointerPointer extraPointers,
                                     int opNum,
                                     Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                     Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                     Pointer extraParamsVals,
                                     Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                     Pointer dy, @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                     Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer,
                                     Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfoBuffer,
                                     Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                     Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape,
                                     @Cast("Nd4jLong *") LongPointer tadOnlyShapeInfo, @Cast("Nd4jLong *") LongPointer tadOffsets,
                                     @Cast("Nd4jLong *") LongPointer yTadOnlyShapeInfo, @Cast("Nd4jLong *") LongPointer yTadOffsets);

    void execReduce3All(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer extraParamsVals,
                                        Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                        Pointer dy, @Cast("Nd4jLong *") LongPointer dyShapeInfo,
                                        Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer,
                                        Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfoBuffer,
                                        Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                        Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape,
                                        @Cast("Nd4jLong *") LongPointer xTadShape,
                                        @Cast("Nd4jLong *") LongPointer xOffsets,
                                        @Cast("Nd4jLong *") LongPointer yTadShape,
                                        @Cast("Nd4jLong *") LongPointer yOffsets);


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
                                    Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                    Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                    Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                    Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                    Pointer scalar, @Cast("Nd4jLong *") LongPointer scalarShapeInfo,
                                    Pointer dscalar, @Cast("Nd4jLong *") LongPointer dscalarShapeInfo,
                                    Pointer extraParams);

    void execScalarBool(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                        Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                        Pointer scalar, @Cast("Nd4jLong *") LongPointer scalarShapeInfo,
                                        Pointer dscalar, @Cast("Nd4jLong *") LongPointer dscalarShapeInfo,
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
                                                Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                                Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                                Pointer extraParams,
                                                Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                                Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
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
                                          Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                          Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                          Pointer extraParams,
                                          Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                          Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                          boolean biasCorrected);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void execSummaryStatsTad(PointerPointer extraPointers,
                                          int opNum,
                                          Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                          Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                          Pointer extraParams,
                                          Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer,
                                          Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfoBuffer,
                                          Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                          Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape,
                                          boolean biasCorrected,
                                          @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                          @Cast("Nd4jLong *") LongPointer tadOffsets);


    /**
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void execTransformFloat(PointerPointer extraPointers,
                                            int opNum,
                                            Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                            Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                            Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                            Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                            Pointer extraParams);

    void execTransformSame(PointerPointer extraPointers,
                                           int opNum,
                                           Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                           Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                           Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                           Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                           Pointer extraParams);

    void execTransformStrict(PointerPointer extraPointers,
                                             int opNum,
                                             Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                             Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                             Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                             Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                             Pointer extraParams);

    void execTransformBool(PointerPointer extraPointers,
                                           int opNum,
                                           Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                           Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                           Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                           Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                           Pointer extraParams);

    void execTransformAny(PointerPointer extraPointers,
                                           int opNum,
                                           Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                           Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                           Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                           Pointer dresult, @Cast("Nd4jLong *") LongPointer dresultShapeInfo,
                                           Pointer extraParams);

    /**
     * ScalarOp along dimension
     *
     * @param extraPointers   pointers to tadShapes and tadoffsets
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
    void execScalarTad(PointerPointer extraPointers,
                                    int opNum,
                                    Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                    Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                    Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                    Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
                                    Pointer scalars, @Cast("Nd4jLong *") LongPointer scalarShapeInfo,
                                    Pointer dscalars, @Cast("Nd4jLong *") LongPointer dscalarShapeInfo,
                                    Pointer extraParams,
                                    Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                    Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape,
                                    @Cast("Nd4jLong *") LongPointer tadShapeInfo, @Cast("Nd4jLong *") LongPointer tadOffsets,
                                    @Cast("Nd4jLong *") LongPointer tadShapeInfoZ, @Cast("Nd4jLong *") LongPointer tadOffsetsZ);

    void execScalarBoolTad(PointerPointer extraPointers,
                                        int opNum,
                                        Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                        Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                        Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
                                        Pointer scalars, @Cast("Nd4jLong *") LongPointer scalarShapeInfo,
                                        Pointer dscalars, @Cast("Nd4jLong *") LongPointer dscalarShapeInfo,
                                        Pointer extraParams,
                                        Pointer hDimension, @Cast("Nd4jLong *") LongPointer hDimensionShape,
                                        Pointer dDimension, @Cast("Nd4jLong *") LongPointer dDimensionShape,
                                        @Cast("Nd4jLong *") LongPointer tadShapeInfo, @Cast("Nd4jLong *") LongPointer tadOffsets,
                                        @Cast("Nd4jLong *") LongPointer tadShapeInfoZ, @Cast("Nd4jLong *") LongPointer tadOffsetsZ);


    void specialConcat(PointerPointer extraPointers,
                                       int dimension,
                                       int numArrays,
                                       PointerPointer data, PointerPointer inputShapeInfo,
                                       Pointer results, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
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
                                  Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                  Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                  Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                  Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
                                  long n,
                                  @Cast("Nd4jLong *") LongPointer indexes,
                                  @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                  @Cast("Nd4jLong *") LongPointer tadOffsets,
                                  @Cast("Nd4jLong *") LongPointer zTadShapeInfo,
                                  @Cast("Nd4jLong *") LongPointer zTadOffsets);


    ///////////////////////

    void average(PointerPointer extraPointers,
                                 PointerPointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                 PointerPointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                 Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                 Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
                                 int n,
                                 long length,
                                 boolean propagate);

    ///////////////////////

    void accumulate(PointerPointer extraPointers,
                                    PointerPointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                    PointerPointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                    Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                                    Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeInfo,
                                    int n,
                                    long length);

    ///////////////////////

    void enableP2P(boolean reallyEnable);

    void checkP2P();

    boolean isP2PAvailable();

    //

    void shuffle(PointerPointer extraPointers,
                                 PointerPointer x, @Cast("Nd4jLong *") PointerPointer xShapeInfo,
                                 PointerPointer dx, @Cast("Nd4jLong *") PointerPointer dxShapeInfo,
                                 PointerPointer z, @Cast("Nd4jLong *") PointerPointer zShapeInfo,
                                 PointerPointer dz, @Cast("Nd4jLong *") PointerPointer dzShapeInfo,
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
                                                Pointer x, @Cast("Nd4jLong *") LongPointer xShape,
                                                Pointer dx, @Cast("Nd4jLong *") LongPointer dxShape,
                                                Pointer y, @Cast("Nd4jLong *") LongPointer yShape,
                                                Pointer dy, @Cast("Nd4jLong *") LongPointer dyShape,
                                                Pointer z, @Cast("Nd4jLong *") LongPointer zShape,
                                                Pointer dz, @Cast("Nd4jLong *") LongPointer dzShape,
                                                Pointer extraA, Pointer extraB, double scalarA,
                                                double scalarB);

*/
    /////////////////////////

    void execAggregate(PointerPointer extras, int opNum,
                                       PointerPointer arguments,
                                       int numArguments,
                                       @Cast("Nd4jLong **") PointerPointer shapes,
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
                                    Pointer z, @Cast("Nd4jLong *") LongPointer zShapeBuffer,
                                    Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeBuffer,
                                    Pointer extraArguments);

    void execRandom3(PointerPointer extraPointers,
                                    int opNum,
                                    Pointer state,
                                    Pointer x, @Cast("Nd4jLong *") LongPointer xShapeBuffer,
                                    Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeBuffer,
                                    Pointer y, @Cast("Nd4jLong *") LongPointer yShapeBuffer,
                                    Pointer dy, @Cast("Nd4jLong *") LongPointer dyShapeBuffer,
                                    Pointer z, @Cast("Nd4jLong *") LongPointer zShapeBuffer,
                                    Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeBuffer,
                                    Pointer extraArguments);

    void execRandom2(PointerPointer extraPointers,
                                    int opNum,
                                    Pointer state,
                                    Pointer x, @Cast("Nd4jLong *") LongPointer xShapeBuffer,
                                    Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeBuffer,
                                    Pointer z, @Cast("Nd4jLong *") LongPointer zShapeBuffer,
                                    Pointer dz, @Cast("Nd4jLong *") LongPointer dzShapeBuffer,
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

    String getNpyArrayNameFromMap(Pointer map, int index);

    Pointer getNpyArrayFromMap(Pointer map, int index);

    Pointer getNpyArrayData(Pointer npArray);

     LongPointer getNpyArrayShape(Pointer npArray);

    int getNpyArrayRank(Pointer npArray);

    char getNpyArrayOrder(Pointer npArray);

    int getNpyArrayElemSize(Pointer npArray);
    ///////


    void tear(PointerPointer extras,
                              Pointer tensor, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                              Pointer dtensor, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                              PointerPointer targets, @Cast("Nd4jLong *") LongPointer zShapeInfo,
                              @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                              @Cast("Nd4jLong *") LongPointer tadOffsets);


    long encodeBitmap(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, long N, IntPointer dz, float threshold);

    void decodeBitmap(PointerPointer extraPointers, Pointer dx, long N, Pointer dz, LongPointer zShapeInfo);


    void encodeThresholdP1(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, long N, IntPointer dz, float threshold);

    void encodeThresholdP2Int(PointerPointer extraPointers, IntPointer dx, long N, IntPointer dz);

    void encodeThresholdP3(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, IntPointer offsets, long N, IntPointer dz);

    void decodeThreshold(PointerPointer extraPointers, Pointer dx, long N, Pointer dz, LongPointer zShapeInfo);

    void sort(PointerPointer extraPointers,
                              Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                              Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                              boolean descending);


    void sortTad(PointerPointer extraPointers,
                                 Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                 Pointer dx, @Cast("Nd4jLong *") LongPointer dxShapeInfo,
                                 IntPointer dimension,
                                 int dimensionLength,
                                 @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                 @Cast("Nd4jLong *") LongPointer tadOffsets,
                                 boolean descending);


    void sortCooIndices(PointerPointer extraPointers, @Cast("Nd4jLong *") LongPointer indices, Pointer values, long length, int rank);


    LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);

    void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);

    OpaqueResultWrapper executeFlatGraph(PointerPointer extraPointers, Pointer flatBufferPointer);

    long getResultWrapperSize(OpaqueResultWrapper ptr);
    Pointer getResultWrapperPointer(OpaqueResultWrapper ptr);

    String getAllCustomOps();

    String getAllOperations();

    int execCustomOp2(PointerPointer extraPointers, long opHashCode, Pointer context);

    int execCustomOp(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs, @Cast("bool *") BooleanPointer bArgs, int numBArgs, boolean isInplace);

    OpaqueShapeList calculateOutputShapes(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs);

    OpaqueShapeList calculateOutputShapes2(PointerPointer extraPointers, long hash, PointerPointer inputBunffers, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs, @Cast("bool *") BooleanPointer bArgs, int numBArgs);

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
                                       Pointer hX, @Cast("Nd4jLong *") LongPointer hXShapeInfo, @Cast("Nd4jLong *") LongPointer hxOffsets,
                                       Pointer dX, @Cast("Nd4jLong *") LongPointer dXShapeInfo, @Cast("Nd4jLong *") LongPointer dxOffsets,
                                       Pointer hY, @Cast("Nd4jLong *") LongPointer hYShapeInfo, @Cast("Nd4jLong *") LongPointer hyOffsets,
                                       Pointer dY, @Cast("Nd4jLong *") LongPointer dYShapeInfo, @Cast("Nd4jLong *") LongPointer dyOffsets,
                                       IntPointer hIndices, IntPointer dIndices);

    //void fillUtf8String(PointerPointer extraPointers, String[] string, int numStrings, Pointer buffer);
    Pointer createUtf8String(PointerPointer extraPointers, String string, int length);
    long getUtf8StringLength(PointerPointer extraPointers, Pointer ptr);
    BytePointer getUtf8StringBuffer(PointerPointer extraPointers, Pointer ptr);
    void deleteUtf8String(PointerPointer extraPointers, Pointer ptr);


    void inspectArray(PointerPointer extraPointers, Pointer buffer, @Cast("Nd4jLong *") LongPointer shapeInfo, Pointer specialBuffer, @Cast("Nd4jLong *") LongPointer specialShapeInfo, @Cast("nd4j::DebugInfo *") Pointer debugInfo);

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

    OpaqueConstantDataBuffer shapeBuffer(int rank, LongPointer shape, LongPointer strides, int dtype, char order, long ews, boolean empty);

    OpaqueConstantDataBuffer constantBufferDouble(int dtype, DoublePointer data, int length);

    OpaqueConstantDataBuffer constantBufferLong(int dtype, LongPointer data, int length);

    Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf);
    Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf);
    long getConstantDataBufferLength(OpaqueConstantDataBuffer dbf);
    long getConstantDataBufferSizeOf(OpaqueConstantDataBuffer dbf);

    void deleteShapeBuffer(OpaqueConstantDataBuffer state);

    OpaqueContext createGraphContext(int nodeId);
    OpaqueRandomGenerator getGraphContextRandomGenerator(OpaqueContext ptr);
    void markGraphContextInplace(OpaqueContext ptr, boolean reallyInplace);
    void setGraphContextCudaContext(OpaqueContext ptr, Pointer stream, Pointer reductionPointer, Pointer allocationPointer);
    void setGraphContextInputArray(OpaqueContext ptr, int index, Pointer buffer, Pointer shapeInfo, Pointer specialBuffer, Pointer specialShapeInfo);
    void setGraphContextOutputArray(OpaqueContext ptr, int index, Pointer buffer, Pointer shapeInfo, Pointer specialBuffer, Pointer specialShapeInfo);
    void setGraphContextTArguments(OpaqueContext ptr, DoublePointer arguments, int numberOfArguments);
    void setGraphContextIArguments(OpaqueContext ptr, LongPointer arguments, int numberOfArguments);
    void setGraphContextBArguments(OpaqueContext ptr, BooleanPointer arguments, int numberOfArguments);
    void deleteGraphContext(OpaqueContext ptr);

    OpaqueRandomGenerator createRandomGenerator(long rootSeed, long nodeSeed);
    long getRandomGeneratorRootState(OpaqueRandomGenerator ptr);
    long getRandomGeneratorNodeState(OpaqueRandomGenerator ptr);
    void setRandomGeneratorStates(OpaqueRandomGenerator ptr, @Cast("Nd4jLong") long rootSeed/*=0*/, @Cast("Nd4jLong") long nodeSeed/*=0*/);
    int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, @Cast("Nd4jLong") long index);
    long getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, @Cast("Nd4jLong") long index);
    void deleteRandomGenerator(OpaqueRandomGenerator ptr);

    String runLightBenchmarkSuit(boolean printOut);

    String runFullBenchmarkSuit(boolean printOut);

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
}
