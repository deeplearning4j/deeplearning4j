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
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
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
 * Comprehensive tests for gated MLP patterns with Triton GPU backend.
 * Covers: silu(x@W_gate) * (x@W_up), various sizes, with residual, with RMSNorm.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonGatedMLPTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 5e-3;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test @DisplayName("Gated MLP: silu(x@W_gate) * (x@W_up) [1, 64] -> [1, 128]")
    public void testGatedMLPBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable W_gate = sd.constant("W_gate", Nd4j.randn(DataType.FLOAT, 64, 128));
        SDVariable W_up = sd.constant("W_up", Nd4j.randn(DataType.FLOAT, 64, 128));

        SDVariable gate = sd.mmul("gate_mm", x, W_gate);
        SDVariable gateAct = sd.nn().silu("gate_act", gate);
        SDVariable up = sd.mmul("up_mm", x, W_up);
        up.mul("result", gateAct);

        runGatedMLPOpTest("testGatedMLPBasic", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Gated MLP: Batched [4, 64] -> [4, 128]")
    public void testGatedMLPBatched() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable W_gate = sd.constant("W_gate", Nd4j.randn(DataType.FLOAT, 64, 128));
        SDVariable W_up = sd.constant("W_up", Nd4j.randn(DataType.FLOAT, 64, 128));

        SDVariable gate = sd.mmul("gate_mm", x, W_gate);
        SDVariable gateAct = sd.nn().silu("gate_act", gate);
        SDVariable up = sd.mmul("up_mm", x, W_up);
        up.mul("result", gateAct);

        runGatedMLPOpTest("testGatedMLPBatched", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Gated MLP + Residual: silu(x@W_gate) * (x@W_up) @ Wo + x [1, 64]")
    public void testGatedMLPResidual() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable W_gate = sd.constant("W_gate", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable W_up = sd.constant("W_up", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable Wo = sd.constant("Wo", Nd4j.randn(DataType.FLOAT, 64, 64));

        SDVariable gate = sd.mmul("gate_mm", x, W_gate);
        SDVariable gateAct = sd.nn().silu("gate_act", gate);
        SDVariable up = sd.mmul("up_mm", x, W_up);
        SDVariable hidden = gateAct.mul("hidden", up);
        SDVariable projected = sd.mmul("proj", hidden, Wo);
        projected.add("result", x);

        runGatedMLPOpTest("testGatedMLPResidual", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Gated MLP + RMSNorm: rms_norm(x) -> silu(x@W_gate) * (x@W_up) [1, 64]")
    public void testGatedMLPRmsNorm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 64));
        SDVariable W_gate = sd.constant("W_gate", Nd4j.randn(DataType.FLOAT, 64, 128));
        SDVariable W_up = sd.constant("W_up", Nd4j.randn(DataType.FLOAT, 64, 128));
        double eps = 1e-5;

        // RMSNorm
        SDVariable squared = x.mul("sq", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtVal = sd.math().rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtVal);
        SDVariable xNormed = normed.mul("x_normed", gamma);

        // Gated MLP
        SDVariable gate = sd.mmul("gate_mm", xNormed, W_gate);
        SDVariable gateAct = sd.nn().silu("gate_act", gate);
        SDVariable up = sd.mmul("up_mm", xNormed, W_up);
        up.mul("result", gateAct);

        runGatedMLPOpTest("testGatedMLPRmsNorm", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("Gated MLP Large: [2, 256] -> [2, 512]")
    public void testGatedMLPLarge() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 256);
        SDVariable W_gate = sd.constant("W_gate", Nd4j.randn(DataType.FLOAT, 256, 512));
        SDVariable W_up = sd.constant("W_up", Nd4j.randn(DataType.FLOAT, 256, 512));

        SDVariable gate = sd.mmul("gate_mm", x, W_gate);
        SDVariable gateAct = sd.nn().silu("gate_act", gate);
        SDVariable up = sd.mmul("up_mm", x, W_up);
        up.mul("result", gateAct);

        runGatedMLPOpTest("testGatedMLPLarge", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 256)), "result");
        sd.close();
    }

    // ─── Helper with relaxed tolerance ───────────────────────────────────────

    private void runGatedMLPOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        Map<String, INDArray> ref = sd.output(ph, outputName);
        INDArray refOutput = ref.get(outputName);
        assertNotNull(refOutput, testName + ": reference output is null");

        org.nd4j.autodiff.samediff.execution.DynamicShapePlan plan =
            NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");

        Pointer planHandle = TritonTestUtils.compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping {} (native executor not supported)", testName);
            return;
        }
        try {
            INDArray[] extInputs = TritonTestUtils.resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResults = TritonTestUtils.executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResults.get(outputName);
            assertNotNull(nativeOutput, testName + ": native output is null");

            assertArrayEquals(refOutput.shape(), nativeOutput.shape(),
                testName + ": shape mismatch");

            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOLERANCE,
                String.format("%s: max diff %.6f exceeds tolerance %.6f",
                    testName, maxDiff, TOLERANCE));

            log.info("{}: PASSED (maxDiff={:.6f})", testName, maxDiff);
        } catch (Exception e) {
            fail(testName + ": execution failed - " + e.getMessage(), e);
        } finally {
            if (planHandle != null) {
                planHandle.close();
            }
        }
    }
}
