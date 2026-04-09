/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for reduction operations with Triton GPU backend.
 * Covers: sum, mean, max, min, prod, variance, stdev across various axis configurations
 * and edge cases.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonReductionTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Sum ─────────────────────────────────────────────────────────────────

    @Test @DisplayName("Sum: Full reduction (all dims) [4, 8]")
    public void testSumFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x);
        TritonTestUtils.runOpTest("testSumFull", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sum: Along axis 0, keepDims=true [4, 8]")
    public void testSumAxis0KeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x, true, 0);
        TritonTestUtils.runOpTest("testSumAxis0KeepDims", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sum: Along axis 1, keepDims=true [4, 8]")
    public void testSumAxis1KeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x, true, 1);
        TritonTestUtils.runOpTest("testSumAxis1KeepDims", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sum: Along axis 1, keepDims=false [4, 8]")
    public void testSumAxis1NoKeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x, 1);
        TritonTestUtils.runOpTest("testSumAxis1NoKeepDims", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Mean ────────────────────────────────────────────────────────────────

    @Test @DisplayName("Mean: Full reduction [4, 8]")
    public void testMeanFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.mean("result", x);
        TritonTestUtils.runOpTest("testMeanFull", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Mean: Along axis 0 [4, 8]")
    public void testMeanAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.mean("result", x, true, 0);
        TritonTestUtils.runOpTest("testMeanAxis0", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Mean: Along axis 1 [4, 8]")
    public void testMeanAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.mean("result", x, true, 1);
        TritonTestUtils.runOpTest("testMeanAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Max / Min via reduceMax/reduceMin ────────────────────────────────────

    @Test @DisplayName("ReduceMax: Full [4, 8]")
    public void testReduceMaxFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        x.max("result", true);
        TritonTestUtils.runOpTest("testReduceMaxFull", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("ReduceMax: Along axis 1 [4, 8]")
    public void testReduceMaxAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        x.max("result", true, 1);
        TritonTestUtils.runOpTest("testReduceMaxAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("ReduceMin: Full [4, 8]")
    public void testReduceMinFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        x.min("result", true);
        TritonTestUtils.runOpTest("testReduceMinFull", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("ReduceMin: Along axis 1 [4, 8]")
    public void testReduceMinAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        x.min("result", true, 1);
        TritonTestUtils.runOpTest("testReduceMinAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Prod ────────────────────────────────────────────────────────────────

    @Test @DisplayName("Prod: Full reduction [4, 4] (small to avoid overflow)")
    public void testProdFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        sd.prod("result", x, true);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 4).addi(0.5f);
        TritonTestUtils.runOpTest("testProdFull", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    // ─── Variance / Stdev (manual) ───────────────────────────────────────────

    @Test @DisplayName("Variance: Manual (mean(sq) - sq(mean)) along axis 1 [4, 8]")
    public void testVariance() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable mean = sd.mean("mean", x, true, 1);
        SDVariable meanSq = sd.mean("meanSq", x.mul("sq", x), true, 1);
        SDVariable variance = meanSq.sub("var", mean.mul("meanSq2", mean));
        sd.nn().relu("result", variance, 0);  // relu to handle floating point negatives
        TritonTestUtils.runOpTest("testVariance", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Stdev: sqrt(variance) along axis 1 [4, 8]")
    public void testStdev() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable mean = sd.mean("mean", x, true, 1);
        SDVariable meanSq = sd.mean("meanSq", x.mul("sq", x), true, 1);
        SDVariable variance = meanSq.sub("var", mean.mul("meanSq2", mean));
        SDVariable varianceClamped = sd.nn().relu("varClamped", variance, 0);
        sd.math().sqrt("result", varianceClamped);
        TritonTestUtils.runOpTest("testStdev", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Edge cases ──────────────────────────────────────────────────────────

    @Test @DisplayName("Sum: Single element [1, 1]")
    public void testSumSingleElement() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1);
        sd.sum("result", x);
        TritonTestUtils.runOpTest("testSumSingleElement", sd, Map.of("x", Nd4j.scalar(DataType.FLOAT, 42.0f)), "result");
        sd.close();
    }

    @Test @DisplayName("Sum: All zeros [4, 8]")
    public void testSumAllZeros() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x);
        TritonTestUtils.runOpTest("testSumAllZeros", sd, Map.of("x", Nd4j.zeros(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sum: All ones [4, 8]")
    public void testSumAllOnes() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.sum("result", x, true, 1);
        TritonTestUtils.runOpTest("testSumAllOnes", sd, Map.of("x", Nd4j.ones(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Mean: Negative values [4, 8]")
    public void testMeanNegativeValues() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.mean("result", x, true, 1);
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8).subi(5.0f);
        TritonTestUtils.runOpTest("testMeanNegativeValues", sd, Map.of("x", xArr), "result");
        sd.close();
    }
}
