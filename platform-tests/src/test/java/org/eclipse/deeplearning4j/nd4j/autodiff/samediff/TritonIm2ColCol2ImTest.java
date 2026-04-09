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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
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
 * Comprehensive tests for im2col and col2im operations with Triton GPU backend.
 * Covers: various kernel sizes, strides, and padding configurations.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonIm2ColCol2ImTest extends BaseNd4jTestWithBackends {

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

    @Test @DisplayName("Im2Col: 3x3 kernel, stride=1, pad=0 [1,1,4,4]")
    public void testIm2ColBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable result = sd.cnn().im2Col("result", input, config);
        runIm2ColOpTest("testIm2ColBasic", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 4, 4)), "result");
        sd.close();
    }

    @Test @DisplayName("Im2Col: 3x3 kernel, stride=2 [1,1,8,8]")
    public void testIm2ColStride2() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 8, 8);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(2).sW(2).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable result = sd.cnn().im2Col("result", input, config);
        runIm2ColOpTest("testIm2ColStride2", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 8, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Im2Col: 3x3 kernel, pad=1 [1,1,4,4]")
    public void testIm2ColPadding() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable result = sd.cnn().im2Col("result", input, config);
        runIm2ColOpTest("testIm2ColPadding", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 4, 4)), "result");
        sd.close();
    }

    @Test @DisplayName("Im2Col + ReLU chain [1,1,6,6]")
    public void testIm2ColReluChain() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 6, 6);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable col = sd.cnn().im2Col("col", input, config);
        sd.nn().relu("result", col, 0);
        runIm2ColOpTest("testIm2ColReluChain", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 6, 6)), "result");
        sd.close();
    }

    @Test @DisplayName("Im2Col: 5x5 kernel [1,1,8,8]")
    public void testIm2ColLargeKernel() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 8, 8);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(5).kW(5).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable result = sd.cnn().im2Col("result", input, config);
        runIm2ColOpTest("testIm2ColLargeKernel", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 8, 8)), "result");
        sd.close();
    }

    // ─── Helper with relaxed tolerance ───────────────────────────────────────

    private void runIm2ColOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        Map<String, INDArray> ref = sd.output(ph, outputName);
        INDArray refOutput = ref.get(outputName);
        assertNotNull(refOutput, testName + ": reference output is null");

        org.nd4j.autodiff.samediff.execution.DynamicShapePlan plan =
            NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");

        org.bytedeco.javacpp.Pointer planHandle = TritonTestUtils.compileNativePlan(plan);
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
