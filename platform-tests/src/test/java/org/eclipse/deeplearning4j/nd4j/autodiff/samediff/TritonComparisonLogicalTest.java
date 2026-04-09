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
 * Comprehensive tests for comparison and logical operations with Triton GPU backend.
 * Covers: gt, gte, lt, lte, eq, neq (all castTo FLOAT), and, or, xor, not,
 * and comparison+where integration.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonComparisonLogicalTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Comparison ops (castTo FLOAT for numeric comparison) ────────────────

    @Test @DisplayName("Greater Than: x > y, cast to float [4, 8]")
    public void testGt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable gt = sd.gt("gt", x, y);
        sd.castTo("result", gt, DataType.FLOAT);
        TritonTestUtils.runOpTest("testGt", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Greater Than or Equal: x >= y, cast to float [4, 8]")
    public void testGte() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable gte = sd.gte("gte", x, y);
        sd.castTo("result", gte, DataType.FLOAT);
        TritonTestUtils.runOpTest("testGte", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Less Than: x < y, cast to float [4, 8]")
    public void testLt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable lt = sd.lt("lt", x, y);
        sd.castTo("result", lt, DataType.FLOAT);
        TritonTestUtils.runOpTest("testLt", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Less Than or Equal: x <= y, cast to float [4, 8]")
    public void testLte() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable lte = sd.lte("lte", x, y);
        sd.castTo("result", lte, DataType.FLOAT);
        TritonTestUtils.runOpTest("testLte", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Equal: x == y, cast to float [4, 8]")
    public void testEq() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable eq = sd.eq("eq", x, y);
        sd.castTo("result", eq, DataType.FLOAT);
        TritonTestUtils.runOpTest("testEq", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Not Equal: x != y, cast to float [4, 8]")
    public void testNeq() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable neq = sd.neq("neq", x, y);
        sd.castTo("result", neq, DataType.FLOAT);
        TritonTestUtils.runOpTest("testNeq", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Logical ops (bool) ──────────────────────────────────────────────────

    @Test @DisplayName("Logical AND: bool AND bool, cast to float [4, 8]")
    public void testAnd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.BOOL, -1, 8);
        SDVariable y = sd.constant("y", createBoolArray(4, 8, 0.7f));
        SDVariable boolResult = sd.booleanAnd("bool_result", x, y);
        boolResult.castTo("result", DataType.FLOAT);
        INDArray xArr = createBoolArray(4, 8, 0.5f);
        TritonTestUtils.runOpTest("testAnd", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Logical OR: bool OR bool, cast to float [4, 8]")
    public void testOr() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.BOOL, -1, 8);
        SDVariable y = sd.constant("y", createBoolArray(4, 8, 0.3f));
        SDVariable boolResult = sd.booleanOr("bool_result", x, y);
        boolResult.castTo("result", DataType.FLOAT);
        INDArray xArr = createBoolArray(4, 8, 0.5f);
        TritonTestUtils.runOpTest("testOr", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Logical XOR: bool XOR bool, cast to float [4, 8]")
    public void testXor() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.BOOL, -1, 8);
        SDVariable y = sd.constant("y", createBoolArray(4, 8, 0.5f));
        SDVariable boolResult = sd.booleanXor("bool_result", x, y);
        boolResult.castTo("result", DataType.FLOAT);
        INDArray xArr = createBoolArray(4, 8, 0.3f);
        TritonTestUtils.runOpTest("testXor", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Logical NOT: 1 - bool (cast to float first) [4, 8]")
    public void testNot() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.BOOL, -1, 8);
        SDVariable xFloat = sd.castTo("x_float", x, DataType.FLOAT);
        // NOT = 1 - x (for boolean as 0/1)
        SDVariable one = sd.constant("one", Nd4j.scalar(1.0f));
        one.sub("result", xFloat);
        INDArray xArr = createBoolArray(4, 8, 0.5f);
        TritonTestUtils.runOpTest("testNot", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    // ─── Comparison + Where integration ──────────────────────────────────────

    @Test @DisplayName("Comparison + Where: gt mask selects between a and b [4, 8]")
    public void testComparisonWhere() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable threshold = sd.constant("threshold", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable cond = sd.gt("cond", x, threshold);
        SDVariable a = sd.constant("a", Nd4j.valueArrayOf(4, 8, 1.0f));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(4, 8, -1.0f));
        sd.where("result", a, b, cond);
        TritonTestUtils.runOpTest("testComparisonWhere", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Where with computed mask: (x > 0) ? x : 0 (ReLU via where) [4, 8]")
    public void testWhereRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable cond = sd.gt("cond", x, zero);
        sd.where("result", x, zero, cond);
        TritonTestUtils.runOpTest("testWhereRelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Helper ──────────────────────────────────────────────────────────────

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
