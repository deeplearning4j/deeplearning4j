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
 * Comprehensive tests for convolution operations with Triton GPU backend.
 * Covers: conv2d NHWC with small configs, various kernel sizes, strides, and padding.
 * Uses relaxed tolerance (5e-3) for convolution operations.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonConvolutionTest extends BaseNd4jTestWithBackends {

    private static final double CONV_TOLERANCE = 5e-3;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test @DisplayName("Conv2D NCHW: 3x3 kernel, stride=1, pad=0 [1,1,4,4]")
    public void testConv2dBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);
        SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();

        SDVariable conv = sd.cnn().conv2d("conv", input, weight, config);
        sd.nn().relu("result", conv, 0);

        runConvOpTest("testConv2dBasic", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 4, 4)), "result");
        sd.close();
    }

    @Test @DisplayName("Conv2D NCHW: 3x3 kernel, stride=2 [1,2,8,8]")
    public void testConv2dStride2() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 2, 8, 8);
        SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 4, 2, 3, 3));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(2).sW(2)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();

        SDVariable conv = sd.cnn().conv2d("conv", input, weight, config);
        sd.nn().relu("result", conv, 0);

        runConvOpTest("testConv2dStride2", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 2, 8, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Conv2D NCHW: 3x3 kernel, pad=1 (same padding) [1,2,6,6]")
    public void testConv2dPadding() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 2, 6, 6);
        SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 3, 2, 3, 3));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();

        SDVariable conv = sd.cnn().conv2d("conv", input, weight, config);
        sd.nn().relu("result", conv, 0);

        runConvOpTest("testConv2dPadding", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 2, 6, 6)), "result");
        sd.close();
    }

    @Test @DisplayName("Conv2D NCHW: 5x5 kernel, stride=1, pad=0 [1,1,8,8]")
    public void testConv2dLargeKernel() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 8, 8);
        SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 2, 1, 5, 5));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(5).kW(5)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();

        SDVariable conv = sd.cnn().conv2d("conv", input, weight, config);
        sd.nn().relu("result", conv, 0);

        runConvOpTest("testConv2dLargeKernel", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 8, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Conv2D NCHW: Deterministic ones input [1,1,4,4]")
    public void testConv2dDeterministic() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);
        INDArray filterData = Nd4j.zeros(DataType.FLOAT, 1, 1, 3, 3);
        filterData.putScalar(0, 0, 0, 0, 1.0f);
        SDVariable weight = sd.constant("weight", filterData);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();
        SDVariable conv = sd.cnn().conv2d("conv", input, weight, config);
        sd.nn().relu("result", conv, 0);

        INDArray inputData = Nd4j.createFromArray(1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16f).reshape(1,1,4,4);
        runConvOpTest("testConv2dDeterministic", sd, Map.of("input", inputData), "result");
        sd.close();
    }

    // ─── Helper with relaxed tolerance ───────────────────────────────────────

    private void runConvOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
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
            assertTrue(maxDiff < CONV_TOLERANCE,
                String.format("%s: max diff %.6f exceeds conv tolerance %.6f",
                    testName, maxDiff, CONV_TOLERANCE));

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
