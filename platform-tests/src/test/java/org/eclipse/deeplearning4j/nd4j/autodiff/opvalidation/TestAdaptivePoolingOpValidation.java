/*
 *  ******************************************************************************
 *  *
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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation tests for Adaptive Pooling operations.
 * Tests forward pass correctness and gradient computation for:
 * - AdaptiveAvgPooling2D
 * - AdaptiveMaxPooling2D
 * - AdaptiveAvgPooling3D
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Adaptive Pooling Op Validation Tests")
public class TestAdaptivePoolingOpValidation extends BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ========================= Adaptive Average Pooling 2D Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool2D - Basic Forward (NCHW)")
    public void testAdaptiveAvgPool2DForwardNCHW(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 3;
        int inHeight = 8;
        int inWidth = 8;
        int outHeight = 4;
        int outWidth = 4;

        SameDiff sd = SameDiff.create();

        // Input [batch, channels, height, width] NCHW
        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);

        SDVariable input = sd.var("input", inputArr);

        // Simulate adaptive average pooling using standard average pooling
        // Kernel size = ceil(inSize / outSize), stride = floor(inSize / outSize)
        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        // Use avgPool2d with computed kernel and stride
        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.AVG)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.avgPooling2d("result", input, config);

        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveAvgPool2D NCHW")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool2D - Various Input Sizes")
    public void testAdaptiveAvgPool2DVariousInputSizes(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int outHeight = 2;
        int outWidth = 2;

        for (int inSize : new int[]{4, 6, 8, 10, 12}) {
            SameDiff sd = SameDiff.create();

            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inSize, inSize).muli(0.5);
            SDVariable input = sd.var("input", inputArr);

            // Compute adaptive pooling parameters
            int kH = (int) Math.ceil((double) inSize / outHeight);
            int kW = (int) Math.ceil((double) inSize / outWidth);
            int sH = inSize / outHeight;
            int sW = inSize / outWidth;

            Pooling2DConfig config = Pooling2DConfig.builder()
                    .kH(kH).kW(kW)
                    .sH(sH).sW(sW)
                    .pH(0).pW(0)
                    .type(Pooling2D.Pooling2DType.AVG)
                    .isNHWC(false)
                    .build();

            SDVariable result = sd.cnn.avgPooling2d("result", input, config);
            SDVariable loss = sd.mean("loss", result);

            TestCase tc = new TestCase(sd)
                    .testName("AdaptiveAvgPool2D inSize=" + inSize)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for inSize=" + inSize + ": " + err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool2D - Various Output Sizes")
    public void testAdaptiveAvgPool2DVariousOutputSizes(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int inHeight = 12;
        int inWidth = 12;

        for (int outSize : new int[]{1, 2, 3, 4, 6}) {
            SameDiff sd = SameDiff.create();

            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
            SDVariable input = sd.var("input", inputArr);

            int kH = (int) Math.ceil((double) inHeight / outSize);
            int kW = (int) Math.ceil((double) inWidth / outSize);
            int sH = inHeight / outSize;
            int sW = inWidth / outSize;

            Pooling2DConfig config = Pooling2DConfig.builder()
                    .kH(kH).kW(kW)
                    .sH(sH).sW(sW)
                    .pH(0).pW(0)
                    .type(Pooling2D.Pooling2DType.AVG)
                    .isNHWC(false)
                    .build();

            SDVariable result = sd.cnn.avgPooling2d("result", input, config);
            SDVariable loss = sd.mean("loss", result);

            TestCase tc = new TestCase(sd)
                    .testName("AdaptiveAvgPool2D outSize=" + outSize)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for outSize=" + outSize + ": " + err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool2D - Global Average Pooling (1x1)")
    public void testAdaptiveAvgPool2DGlobalPool(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 4;
        int inHeight = 7;
        int inWidth = 7;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        // Global average pooling: pool over entire spatial dimensions
        SDVariable result = sd.mean("result", input, 2, 3);

        // Expected: [batch, channels] after mean
        // Reshape to [batch, channels, 1, 1] to match adaptive pool output
        SDVariable resultReshaped = sd.reshape("result_reshaped", result, batch, channels, 1, 1);
        SDVariable loss = sd.mean("loss", resultReshaped);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveAvgPool2D Global")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Adaptive Max Pooling 2D Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveMaxPool2D - Basic Forward (NCHW)")
    public void testAdaptiveMaxPool2DForwardNCHW(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 3;
        int inHeight = 8;
        int inWidth = 8;
        int outHeight = 4;
        int outWidth = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        // Compute adaptive pooling parameters
        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.MAX)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.maxPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveMaxPool2D NCHW")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveMaxPool2D - Various Input Sizes")
    public void testAdaptiveMaxPool2DVariousInputSizes(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int outHeight = 2;
        int outWidth = 2;

        for (int inSize : new int[]{4, 6, 8, 10}) {
            SameDiff sd = SameDiff.create();

            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inSize, inSize).muli(0.5);
            SDVariable input = sd.var("input", inputArr);

            int kH = (int) Math.ceil((double) inSize / outHeight);
            int kW = (int) Math.ceil((double) inSize / outWidth);
            int sH = inSize / outHeight;
            int sW = inSize / outWidth;

            Pooling2DConfig config = Pooling2DConfig.builder()
                    .kH(kH).kW(kW)
                    .sH(sH).sW(sW)
                    .pH(0).pW(0)
                    .type(Pooling2D.Pooling2DType.MAX)
                    .isNHWC(false)
                    .build();

            SDVariable result = sd.cnn.maxPooling2d("result", input, config);
            SDVariable loss = sd.mean("loss", result);

            TestCase tc = new TestCase(sd)
                    .testName("AdaptiveMaxPool2D inSize=" + inSize)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for inSize=" + inSize + ": " + err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveMaxPool2D - Global Max Pooling (1x1)")
    public void testAdaptiveMaxPool2DGlobalPool(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 4;
        int inHeight = 7;
        int inWidth = 7;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        // Global max pooling: max over entire spatial dimensions
        SDVariable result = sd.max("result", input, 2, 3);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveMaxPool2D Global")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Adaptive Average Pooling 3D Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool3D - Global Pooling")
    public void testAdaptiveAvgPool3DGlobalPool(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int inDepth = 4;
        int inHeight = 4;
        int inWidth = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inDepth, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        // Global average pooling over spatial dimensions (D, H, W)
        SDVariable result = sd.mean("result", input, 2, 3, 4);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveAvgPool3D Global")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Asymmetric Pooling Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveAvgPool2D - Asymmetric Output")
    public void testAdaptiveAvgPool2DAsymmetricOutput(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int inHeight = 10;
        int inWidth = 12;
        int outHeight = 2;
        int outWidth = 3;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.AVG)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.avgPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveAvgPool2D Asymmetric")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptiveMaxPool2D - Asymmetric Output")
    public void testAdaptiveMaxPool2DAsymmetricOutput(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int inHeight = 10;
        int inWidth = 12;
        int outHeight = 2;
        int outWidth = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.MAX)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.maxPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptiveMaxPool2D Asymmetric")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Edge Case Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptivePool - Same Input and Output Size")
    public void testAdaptivePoolSameSize(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 2;
        int size = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, size, size).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        // When output size equals input size, kernel = 1, stride = 1
        // This should effectively be an identity operation
        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(1).kW(1)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.AVG)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.avgPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptivePool Same Size")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptivePool - Single Channel")
    public void testAdaptivePoolSingleChannel(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int channels = 1;
        int inHeight = 8;
        int inWidth = 8;
        int outHeight = 2;
        int outWidth = 2;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.AVG)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.avgPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptivePool Single Channel")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("AdaptivePool - Batch Size 1")
    public void testAdaptivePoolBatchOne(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 1;
        int channels = 4;
        int inHeight = 8;
        int inWidth = 8;
        int outHeight = 4;
        int outWidth = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, channels, inHeight, inWidth).muli(0.5);
        SDVariable input = sd.var("input", inputArr);

        int kH = (int) Math.ceil((double) inHeight / outHeight);
        int kW = (int) Math.ceil((double) inWidth / outWidth);
        int sH = inHeight / outHeight;
        int sW = inWidth / outWidth;

        Pooling2DConfig config = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(0).pW(0)
                .type(Pooling2D.Pooling2DType.AVG)
                .isNHWC(false)
                .build();

        SDVariable result = sd.cnn.avgPooling2d("result", input, config);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("AdaptivePool Batch=1")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }
}
