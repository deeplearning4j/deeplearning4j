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
 * Comprehensive tests for matmul + epilogue patterns with Triton GPU backend.
 * Covers: plain mmul, mmul+bias, mmul+relu, mmul+gelu, mmul+silu, mmul+tanh,
 * mmul+sigmoid, mmul+bias+relu, mmul long chain, mmul+residual.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonMatmulEpilogueTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test @DisplayName("Plain Matmul: [8, 16] x [16, 32] -> [8, 32]")
    public void testPlainMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("result", x, w);
        TritonTestUtils.runOpTest("testPlainMatmul", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + Bias: [8, 16] x [16, 32] + [1, 32]")
    public void testMatmulBias() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h = sd.mmul("mm", x, w);
        h.add("result", b);
        TritonTestUtils.runOpTest("testMatmulBias", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + ReLU: [8, 16] x [16, 32] -> relu")
    public void testMatmulRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h = sd.mmul("mm", x, w);
        sd.nn().relu("result", h, 0);
        TritonTestUtils.runOpTest("testMatmulRelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + GELU: [8, 16] x [16, 32] -> gelu")
    public void testMatmulGelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h = sd.mmul("mm", x, w);
        sd.nn().gelu("result", h);
        TritonTestUtils.runOpTest("testMatmulGelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + SiLU: [8, 16] x [16, 32] -> silu")
    public void testMatmulSilu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h = sd.mmul("mm", x, w);
        sd.nn().silu("result", h);
        TritonTestUtils.runOpTest("testMatmulSilu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + Tanh: [8, 16] x [16, 32] -> tanh")
    public void testMatmulTanh() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h = sd.mmul("mm", x, w);
        sd.nn().tanh("result", h);
        TritonTestUtils.runOpTest("testMatmulTanh", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + Sigmoid: [8, 16] x [16, 32] -> sigmoid")
    public void testMatmulSigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h = sd.mmul("mm", x, w);
        sd.nn().sigmoid("result", h);
        TritonTestUtils.runOpTest("testMatmulSigmoid", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + Bias + ReLU: [8, 16] x [16, 32] + bias -> relu")
    public void testMatmulBiasRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h = sd.mmul("mm", x, w);
        SDVariable h2 = h.add("add", b);
        sd.nn().relu("result", h2, 0);
        TritonTestUtils.runOpTest("testMatmulBiasRelu", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul Long Chain: mmul->mul->add->tanh->sigmoid [8, 16]")
    public void testMatmulLongChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(1, 32, 0.5f));
        SDVariable bias = sd.constant("bias", Nd4j.valueArrayOf(1, 32, 0.1f));
        SDVariable h1 = sd.mmul("mm", x, w);
        SDVariable h2 = h1.mul("mul", scale);
        SDVariable h3 = h2.add("add", bias);
        SDVariable h4 = sd.nn().tanh("tanh", h3);
        sd.nn().sigmoid("result", h4);
        TritonTestUtils.runOpTest("testMatmulLongChain", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Matmul + Residual: mmul(x) + x [8, 16] -> [8, 16]")
    public void testMatmulResidual() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable h = sd.mmul("mm", x, w);
        h.add("result", x);
        TritonTestUtils.runOpTest("testMatmulResidual", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }
}
