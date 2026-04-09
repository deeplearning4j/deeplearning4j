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
 * Comprehensive tests for RMSNorm + Linear patterns with Triton GPU backend.
 * Covers: x^2 -> mean -> +eps -> rsqrt -> x*rsqrt -> *gamma -> @W with various sizes.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonRmsNormLinearTest extends BaseNd4jTestWithBackends {

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

    @Test @DisplayName("RMSNorm + Linear: [1, 64] -> rms_norm(64) -> @W[64, 32] -> [1, 32]")
    public void testRmsNormLinearSmall() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 64));
        SDVariable W = sd.constant("W", Nd4j.randn(DataType.FLOAT, 64, 32));
        double eps = 1e-5;

        SDVariable squared = x.mul("sq", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtVal = sd.math().rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtVal);
        SDVariable scaled = normed.mul("scaled", gamma);
        sd.mmul("result", scaled, W);

        runRmsNormOpTest("testRmsNormLinearSmall", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 64)), "result");
        sd.close();
    }

    @Test @DisplayName("RMSNorm + Linear: [4, 128] -> rms_norm(128) -> @W[128, 64] -> [4, 64]")
    public void testRmsNormLinearMedium() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 128));
        SDVariable W = sd.constant("W", Nd4j.randn(DataType.FLOAT, 128, 64));
        double eps = 1e-5;

        SDVariable squared = x.mul("sq", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtVal = sd.math().rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtVal);
        SDVariable scaled = normed.mul("scaled", gamma);
        sd.mmul("result", scaled, W);

        runRmsNormOpTest("testRmsNormLinearMedium", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 128)), "result");
        sd.close();
    }

    @Test @DisplayName("RMSNorm + Linear: [2, 256] -> rms_norm(256) -> @W[256, 128] -> [2, 128]")
    public void testRmsNormLinearLarge() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 256);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 256));
        SDVariable W = sd.constant("W", Nd4j.randn(DataType.FLOAT, 256, 128));
        double eps = 1e-5;

        SDVariable squared = x.mul("sq", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtVal = sd.math().rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtVal);
        SDVariable scaled = normed.mul("scaled", gamma);
        sd.mmul("result", scaled, W);

        runRmsNormOpTest("testRmsNormLinearLarge", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 256)), "result");
        sd.close();
    }

    @Test @DisplayName("RMSNorm + Linear: Non-trivial gamma [1, 64]")
    public void testRmsNormLinearNonTrivialGamma() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        // Non-trivial gamma: alternating 0.5 and 1.5
        float[] gammaData = new float[64];
        for (int i = 0; i < 64; i++) {
            gammaData[i] = (i % 2 == 0) ? 0.5f : 1.5f;
        }
        SDVariable gamma = sd.constant("gamma", Nd4j.createFromArray(gammaData).reshape(64));
        SDVariable W = sd.constant("W", Nd4j.randn(DataType.FLOAT, 64, 32));
        double eps = 1e-5;

        SDVariable squared = x.mul("sq", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtVal = sd.math().rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtVal);
        SDVariable scaled = normed.mul("scaled", gamma);
        sd.mmul("result", scaled, W);

        runRmsNormOpTest("testRmsNormLinearNonTrivialGamma", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 64)), "result");
        sd.close();
    }

    // ─── Helper with relaxed tolerance ───────────────────────────────────────

    private void runRmsNormOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
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
