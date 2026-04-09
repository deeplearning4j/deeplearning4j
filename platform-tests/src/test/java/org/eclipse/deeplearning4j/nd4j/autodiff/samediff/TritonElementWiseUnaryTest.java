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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
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
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for unary element-wise operations with Triton GPU backend.
 * Covers: relu, sigmoid, tanh, gelu, exp, log, abs, sqrt, square, neg, reciprocal,
 * rsqrt, sign, erf, erfc, log1p, ceil, floor, sin, cos, leakyRelu, silu, softplus,
 * relu6, clipByValue.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonElementWiseUnaryTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Parameterized shape sources ─────────────────────────────────────────

    static Stream<ShapeSpec> shapeSource1D() {
        return Stream.of(
            new ShapeSpec("shape_16", 16),
            new ShapeSpec("shape_64", 64),
            new ShapeSpec("shape_256", 256)
        );
    }

    static Stream<ShapeSpec> shapeSource2D() {
        return Stream.of(
            new ShapeSpec("4x8", 4, 8),
            new ShapeSpec("8x16", 8, 16),
            new ShapeSpec("32x64", 32, 64)
        );
    }

    static class ShapeSpec {
        final String name;
        final long[] shape;
        ShapeSpec(String name, int... dims) {
            this.name = name;
            this.shape = new long[dims.length];
            for (int i = 0; i < dims.length; i++) this.shape[i] = dims[i];
        }
        @Override public String toString() { return name; }
    }

    // ─── Activation functions (sd.nn()) ──────────────────────────────────────

    @Test @DisplayName("ReLU: Basic [4, 8]")
    public void testRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().relu("result", x, 0);
        TritonTestUtils.runOpTest("testRelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sigmoid: Basic [4, 8]")
    public void testSigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().sigmoid("result", x);
        TritonTestUtils.runOpTest("testSigmoid", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Tanh: Basic [4, 8]")
    public void testTanh() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().tanh("result", x);
        TritonTestUtils.runOpTest("testTanh", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("GELU: Basic [4, 8]")
    public void testGelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().gelu("result", x);
        TritonTestUtils.runOpTest("testGelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("SiLU (Swish): Basic [4, 8]")
    public void testSilu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().silu("result", x);
        TritonTestUtils.runOpTest("testSilu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Leaky ReLU: alpha=0.01 [4, 8]")
    public void testLeakyRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().leakyRelu("result", x, 0.01);
        TritonTestUtils.runOpTest("testLeakyRelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("ReLU6: Basic [4, 8]")
    public void testRelu6() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().relu6("result", x, 6.0);
        TritonTestUtils.runOpTest("testRelu6", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Math functions (sd.math()) ──────────────────────────────────────────

    @Test @DisplayName("Exp: Basic [4, 8]")
    public void testExp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().exp("result", x);
        TritonTestUtils.runOpTest("testExp", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Log: Positive values [4, 8]")
    public void testLog() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().log("result", x);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 8).addi(0.01f);
        TritonTestUtils.runOpTest("testLog", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Abs: Mixed values [4, 8]")
    public void testAbs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().abs("result", x);
        TritonTestUtils.runOpTest("testAbs", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sqrt: Positive values [4, 8]")
    public void testSqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().sqrt("result", x);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 8).addi(0.01f);
        TritonTestUtils.runOpTest("testSqrt", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Square: Basic [4, 8]")
    public void testSquare() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable result = x.mul("result", x);
        TritonTestUtils.runOpTest("testSquare", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Neg: Basic [4, 8]")
    public void testNeg() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        x.neg("result");
        TritonTestUtils.runOpTest("testNeg", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Reciprocal: Non-zero values [4, 8]")
    public void testReciprocal() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 8).addi(0.1f);
        SDVariable recip = sd.math().reciprocal("result", x);
        TritonTestUtils.runOpTest("testReciprocal", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("RSqrt: Positive values [4, 8]")
    public void testRsqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().rsqrt("result", x);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 8).addi(0.01f);
        TritonTestUtils.runOpTest("testRsqrt", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Sign: Mixed values [4, 8]")
    public void testSign() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().sign("result", x);
        TritonTestUtils.runOpTest("testSign", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Erf: Basic [4, 8]")
    public void testErf() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().erf("result", x);
        TritonTestUtils.runOpTest("testErf", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Erfc: Basic [4, 8]")
    public void testErfc() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().erfc("result", x);
        TritonTestUtils.runOpTest("testErfc", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Log1p: Values > -1 [4, 8]")
    public void testLog1p() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().log1p("result", x);
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 8);
        TritonTestUtils.runOpTest("testLog1p", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Ceil: Basic [4, 8]")
    public void testCeil() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().ceil("result", x);
        TritonTestUtils.runOpTest("testCeil", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Floor: Basic [4, 8]")
    public void testFloor() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().floor("result", x);
        TritonTestUtils.runOpTest("testFloor", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Sin: Basic [4, 8]")
    public void testSin() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().sin("result", x);
        TritonTestUtils.runOpTest("testSin", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Cos: Basic [4, 8]")
    public void testCos() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.math().cos("result", x);
        TritonTestUtils.runOpTest("testCos", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Composite / pattern tests ───────────────────────────────────────────

    @Test @DisplayName("ClipByValue: Clamp to [-1, 1] [4, 8]")
    public void testClipByValue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.clipByValue("result", x, -1.0, 1.0);
        TritonTestUtils.runOpTest("testClipByValue", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8).muli(5.0f)), "result");
        sd.close();
    }

    @Test @DisplayName("Softplus: log(exp(x) + 1) [4, 8]")
    public void testSoftplus() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.nn().softplus("result", x);
        TritonTestUtils.runOpTest("testSoftplus", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Unary chain: x -> exp -> log -> abs -> relu [4, 16]")
    public void testUnaryChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h1 = sd.math().exp("exp1", x);
        SDVariable h2 = sd.math().log("log1", h1);
        SDVariable h3 = sd.math().abs("abs1", h2);
        sd.nn().relu("result", h3, 0);
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        TritonTestUtils.runOpTest("testUnaryChain", sd, Map.of("x", xArr), "result");
        sd.close();
    }

    @Test @DisplayName("Tanh chain: x -> tanh -> mul(scale) -> add(bias) [8, 16]")
    public void testTanhChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(1, 16, 2.0f));
        SDVariable bias = sd.constant("bias", Nd4j.valueArrayOf(1, 16, 0.5f));
        SDVariable h1 = sd.nn().tanh("tanh1", x);
        SDVariable h2 = h1.mul("mul1", scale);
        h2.add("result", bias);
        TritonTestUtils.runOpTest("testTanhChain", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @ParameterizedTest @MethodSource("shapeSource2D")
    @DisplayName("Softmax: Various 2D shapes")
    public void testSoftmax(ShapeSpec spec) {
        SameDiff sd = SameDiff.create();
        int cols = (int) spec.shape[1];
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, cols);
        sd.nn().softmax("result", x, 1);
        INDArray xArr = Nd4j.randn(DataType.FLOAT, (int) spec.shape[0], cols);
        TritonTestUtils.runOpTest("testSoftmax_" + spec.name, sd, Map.of("x", xArr), "result");
        sd.close();
    }
}
