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
 * Comprehensive tests for ternary/select operations (where) with Triton GPU backend.
 * Covers: where with explicit bool mask, computed mask, broadcast scalar, clamping,
 * abs via where, and chained where operations.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonTernarySelectTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test @DisplayName("Where: Explicit bool mask [4, 8]")
    public void testWhereExplicitMask() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable mask = createBoolPlaceholder(sd, "mask", 4, 8);
        SDVariable a = sd.constant("a", Nd4j.valueArrayOf(4, 8, 10.0f));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(4, 8, 0.0f));
        sd.where("result", a, b, mask);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray maskArr = createBoolArray(4, 8, 0.5f);
        TritonTestUtils.runOpTest("testWhereExplicitMask", sd, Map.of("x", xArr, "mask", maskArr), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Computed mask from gt [4, 8]")
    public void testWhereComputedMask() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable cond = sd.gt("cond", x, zero);
        SDVariable a = sd.constant("a", Nd4j.valueArrayOf(4, 8, 1.0f));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(4, 8, -1.0f));
        sd.where("result", a, b, cond);
        TritonTestUtils.runOpTest("testWhereComputedMask", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Broadcast scalar values [4, 8]")
    public void testWhereBroadcastScalar() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable cond = sd.gt("cond", x, sd.constant("thresh", Nd4j.scalar(DataType.FLOAT, 0.0f)));
        SDVariable a = sd.constant("a", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable b = sd.constant("b", Nd4j.scalar(DataType.FLOAT, 0.0f));
        sd.where("result", a, b, cond);
        TritonTestUtils.runOpTest("testWhereBroadcastScalar", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Clamp via nested where (clip to [-1, 1]) [4, 8]")
    public void testWhereClamp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable low = sd.constant("low", Nd4j.scalar(DataType.FLOAT, -1.0f));
        SDVariable high = sd.constant("high", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable aboveHigh = sd.gt("above", x, high);
        SDVariable belowLow = sd.lt("below", x, low);
        SDVariable clampedHigh = sd.where("clampHigh", high, x, aboveHigh);
        sd.where("result", low, clampedHigh, belowLow);
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8).muli(3.0f);
        TritonTestUtils.runOpTest("testWhereClamp", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Absolute value via (x < 0 ? -x : x) [4, 8]")
    public void testWhereAbs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable isNeg = sd.lt("isNeg", x, zero);
        SDVariable negX = x.neg("negX");
        sd.where("result", negX, x, isNeg);
        TritonTestUtils.runOpTest("testWhereAbs", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Chained selections (3-way) [4, 8]")
    public void testWhereChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable t1 = sd.constant("t1", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable t2 = sd.constant("t2", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable t3 = sd.constant("t3", Nd4j.scalar(DataType.FLOAT, -1.0f));
        // x > 0.5 ? 1 : (x < -0.5 ? -1 : 0)
        SDVariable high = sd.gt("high", x, sd.constant("hThresh", Nd4j.scalar(DataType.FLOAT, 0.5f)));
        SDVariable low = sd.lt("low", x, sd.constant("lThresh", Nd4j.scalar(DataType.FLOAT, -0.5f)));
        SDVariable step1 = sd.where("step1", t1, t2, high);
        sd.where("result", t3, step1, low);
        TritonTestUtils.runOpTest("testWhereChain", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Where: With broadcast row-vector mask [4, 8]")
    public void testWhereBroadcastMask() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable rowMask = createBoolPlaceholder(sd, "rowMask", 1, 8);
        SDVariable a = sd.constant("a", Nd4j.valueArrayOf(4, 8, 100.0f));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(4, 8, 0.0f));
        sd.where("result", a, b, rowMask);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray maskArr = createBoolArray(1, 8, 0.5f);
        TritonTestUtils.runOpTest("testWhereBroadcastMask", sd, Map.of("x", xArr, "rowMask", maskArr), "result");
        sd.close();
    }

    @Test @DisplayName("Where: Select between two variable tensors [4, 8]")
    public void testWhereTwoVars() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 8);
        SDVariable cond = sd.gt("cond", x, y);
        sd.where("result", x, y, cond);
        // This effectively computes element-wise maximum
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray yArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        TritonTestUtils.runOpTest("testWhereTwoVars", sd, Map.of("x", xArr, "y", yArr), "result");
        sd.close();
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private static SDVariable createBoolPlaceholder(SameDiff sd, String name, int rows, int cols) {
        return sd.placeHolder(name, DataType.BOOL, rows, cols);
    }

    private static INDArray createBoolArray(int rows, int cols, float trueProb) {
        INDArray arr = Nd4j.create(DataType.BOOL, rows, cols);
        INDArray rand = Nd4j.rand(DataType.FLOAT, rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr.putScalar(i, j, rand.getFloat(i, j) < trueProb ? 1.0 : 0.0);
            }
        }
        return arr;
    }
}
