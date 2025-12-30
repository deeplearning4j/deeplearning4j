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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.autodiff.samediff.training.GradientAccumulator;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for mixed precision training support in SameDiff.
 * Tests cover:
 * - LossScaleConfig configuration
 * - LossScaler scaling/unscaling and overflow detection
 * - GradientAccumulator accumulation and averaging
 * - TrainingConfig mixed precision builder methods
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
public class MixedPrecisionTrainingTest extends BaseNd4jTestWithBackends {

    // ========== LossScaleConfig Tests ==========

    @Test
    public void testLossScaleConfigDefaults() {
        LossScaleConfig config = LossScaleConfig.builder().build();

        assertEquals(LossScaleConfig.Mode.NONE, config.getMode());
        assertEquals(65536.0, config.getInitialScale());
        assertEquals(1.0, config.getMinScale());
        assertEquals(65536.0, config.getMaxScale());
        assertEquals(2.0, config.getGrowthFactor());
        assertEquals(0.5, config.getBackoffFactor());
        assertEquals(2000, config.getGrowthInterval());
        assertFalse(config.isEnabled());
        assertFalse(config.isDynamic());
    }

    @Test
    public void testLossScaleConfigStaticScaling() {
        LossScaleConfig config = LossScaleConfig.staticScaling(1024.0);

        assertEquals(LossScaleConfig.Mode.STATIC, config.getMode());
        assertEquals(1024.0, config.getInitialScale());
        assertTrue(config.isEnabled());
        assertFalse(config.isDynamic());
    }

    @Test
    public void testLossScaleConfigDynamicScaling() {
        LossScaleConfig config = LossScaleConfig.dynamicScaling();

        assertEquals(LossScaleConfig.Mode.DYNAMIC, config.getMode());
        assertEquals(65536.0, config.getInitialScale());
        assertTrue(config.isEnabled());
        assertTrue(config.isDynamic());
    }

    @Test
    public void testLossScaleConfigDynamicScalingCustomInitial() {
        LossScaleConfig config = LossScaleConfig.dynamicScaling(4096.0);

        assertEquals(LossScaleConfig.Mode.DYNAMIC, config.getMode());
        assertEquals(4096.0, config.getInitialScale());
        assertTrue(config.isEnabled());
        assertTrue(config.isDynamic());
    }

    @Test
    public void testLossScaleConfigBuilder() {
        LossScaleConfig config = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(8192.0)
                .minScale(0.5)
                .maxScale(131072.0)
                .growthFactor(4.0)
                .backoffFactor(0.25)
                .growthInterval(1000)
                .build();

        assertEquals(LossScaleConfig.Mode.DYNAMIC, config.getMode());
        assertEquals(8192.0, config.getInitialScale());
        assertEquals(0.5, config.getMinScale());
        assertEquals(131072.0, config.getMaxScale());
        assertEquals(4.0, config.getGrowthFactor());
        assertEquals(0.25, config.getBackoffFactor());
        assertEquals(1000, config.getGrowthInterval());
    }

    // ========== LossScaler Tests ==========

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerScaling(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.staticScaling(1024.0);
        LossScaler scaler = new LossScaler(config);

        assertEquals(1024.0, scaler.getCurrentScale());

        // Test scalar scaling
        double scaledLoss = scaler.scaleLoss(2.0);
        assertEquals(2048.0, scaledLoss, 1e-6);

        // Test array scaling
        INDArray loss = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray scaledArr = scaler.scaleLoss(loss);
        assertEquals(1024.0, scaledArr.getDouble(0), 1e-6);
        assertEquals(2048.0, scaledArr.getDouble(1), 1e-6);
        assertEquals(3072.0, scaledArr.getDouble(2), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerUnscaling(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.staticScaling(1024.0);
        LossScaler scaler = new LossScaler(config);

        INDArray gradients = Nd4j.create(new double[]{1024.0, 2048.0, 4096.0});
        scaler.unscaleGradients(gradients);

        assertEquals(1.0, gradients.getDouble(0), 1e-6);
        assertEquals(2.0, gradients.getDouble(1), 1e-6);
        assertEquals(4.0, gradients.getDouble(2), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerFiniteGradients(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.dynamicScaling();
        LossScaler scaler = new LossScaler(config);

        // Finite gradients
        INDArray finiteGrads = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        assertTrue(scaler.areGradientsFinite(finiteGrads));

        // Gradients with NaN
        INDArray nanGrads = Nd4j.create(new double[]{1.0, Double.NaN, 3.0});
        assertFalse(scaler.areGradientsFinite(nanGrads));

        // Gradients with Inf
        INDArray infGrads = Nd4j.create(new double[]{1.0, Double.POSITIVE_INFINITY, 3.0});
        assertFalse(scaler.areGradientsFinite(infGrads));

        // Gradients with negative Inf
        INDArray negInfGrads = Nd4j.create(new double[]{1.0, Double.NEGATIVE_INFINITY, 3.0});
        assertFalse(scaler.areGradientsFinite(negInfGrads));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerDynamicGrowth(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(1024.0)
                .growthFactor(2.0)
                .growthInterval(3)  // Grow after 3 successful iterations
                .maxScale(8192.0)
                .build();
        LossScaler scaler = new LossScaler(config);

        assertEquals(1024.0, scaler.getCurrentScale());

        // Simulate 3 successful iterations
        scaler.update(true);  // 1
        assertEquals(1024.0, scaler.getCurrentScale());
        scaler.update(true);  // 2
        assertEquals(1024.0, scaler.getCurrentScale());
        scaler.update(true);  // 3 - should trigger growth
        assertEquals(2048.0, scaler.getCurrentScale());

        // Continue growing
        scaler.update(true);
        scaler.update(true);
        scaler.update(true);
        assertEquals(4096.0, scaler.getCurrentScale());

        // Should cap at maxScale
        scaler.update(true);
        scaler.update(true);
        scaler.update(true);
        assertEquals(8192.0, scaler.getCurrentScale());

        // Should not exceed max
        scaler.update(true);
        scaler.update(true);
        scaler.update(true);
        assertEquals(8192.0, scaler.getCurrentScale());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerDynamicBackoff(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(1024.0)
                .backoffFactor(0.5)
                .minScale(1.0)
                .build();
        LossScaler scaler = new LossScaler(config);

        assertEquals(1024.0, scaler.getCurrentScale());

        // Simulate overflow - should reduce scale
        scaler.update(false);
        assertEquals(512.0, scaler.getCurrentScale());

        scaler.update(false);
        assertEquals(256.0, scaler.getCurrentScale());

        // Continue until min scale
        for (int i = 0; i < 10; i++) {
            scaler.update(false);
        }
        assertEquals(1.0, scaler.getCurrentScale());

        // Should not go below min
        scaler.update(false);
        assertEquals(1.0, scaler.getCurrentScale());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerReset(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(1024.0)
                .growthInterval(2)
                .build();
        LossScaler scaler = new LossScaler(config);

        // Change scale
        scaler.update(false);
        assertNotEquals(1024.0, scaler.getCurrentScale());

        // Reset
        scaler.reset();
        assertEquals(1024.0, scaler.getCurrentScale());
        assertEquals(0, scaler.getConsecutiveFiniteCount());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossScalerUnscaleAndCheck(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.staticScaling(1024.0);
        LossScaler scaler = new LossScaler(config);

        // Finite gradients
        INDArray finiteGrads = Nd4j.create(new double[]{1024.0, 2048.0});
        assertTrue(scaler.unscaleGradientsAndCheck(finiteGrads));
        assertEquals(1.0, finiteGrads.getDouble(0), 1e-6);
        assertEquals(2.0, finiteGrads.getDouble(1), 1e-6);

        // Gradients with NaN (after unscaling)
        INDArray nanGrads = Nd4j.create(new double[]{Double.NaN, 2048.0});
        assertFalse(scaler.unscaleGradientsAndCheck(nanGrads));
    }

    // ========== GradientAccumulator Tests ==========

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorBasic(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(4);

        assertEquals(4, accumulator.getAccumulationSteps());
        assertEquals(0, accumulator.getCurrentStep());
        assertTrue(accumulator.isEnabled());
        assertFalse(accumulator.isReady());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorNoAccumulation(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(1);

        assertFalse(accumulator.isEnabled());
        assertEquals(1, accumulator.getAccumulationSteps());
    }

    @Test
    public void testGradientAccumulatorInvalidSteps() {
        assertThrows(IllegalArgumentException.class, () -> {
            new GradientAccumulator(0);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new GradientAccumulator(-1);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorAccumulate(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(4);

        // Accumulate 4 gradients
        for (int i = 0; i < 4; i++) {
            INDArray grad = Nd4j.create(new double[]{1.0, 2.0, 3.0});
            accumulator.accumulate("weights", grad);
            accumulator.step();
        }

        assertTrue(accumulator.isReady());
        assertTrue(accumulator.hasGradient("weights"));
        assertEquals(1, accumulator.getNumVariables());

        // Get averaged gradients
        Map<String, INDArray> result = accumulator.getAndReset();
        INDArray avgGrad = result.get("weights");

        // Average of 4 identical gradients should be the same as each
        assertEquals(1.0, avgGrad.getDouble(0), 1e-6);
        assertEquals(2.0, avgGrad.getDouble(1), 1e-6);
        assertEquals(3.0, avgGrad.getDouble(2), 1e-6);

        // After reset
        assertEquals(0, accumulator.getCurrentStep());
        assertEquals(0, accumulator.getNumVariables());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorAveraging(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(2);

        // First micro-batch
        INDArray grad1 = Nd4j.create(new double[]{2.0, 4.0});
        accumulator.accumulate("w", grad1);
        accumulator.step();

        // Second micro-batch
        INDArray grad2 = Nd4j.create(new double[]{4.0, 8.0});
        accumulator.accumulate("w", grad2);
        accumulator.step();

        assertTrue(accumulator.isReady());

        Map<String, INDArray> result = accumulator.getAndReset();
        INDArray avgGrad = result.get("w");

        // Average: (2+4)/2 = 3, (4+8)/2 = 6
        assertEquals(3.0, avgGrad.getDouble(0), 1e-6);
        assertEquals(6.0, avgGrad.getDouble(1), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorMultipleVariables(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(2);

        // Accumulate for multiple variables
        Map<String, INDArray> grads1 = new HashMap<>();
        grads1.put("w1", Nd4j.create(new double[]{1.0}));
        grads1.put("w2", Nd4j.create(new double[]{2.0}));
        accumulator.accumulate(grads1);
        accumulator.step();

        Map<String, INDArray> grads2 = new HashMap<>();
        grads2.put("w1", Nd4j.create(new double[]{3.0}));
        grads2.put("w2", Nd4j.create(new double[]{4.0}));
        accumulator.accumulate(grads2);
        accumulator.step();

        assertTrue(accumulator.isReady());
        assertEquals(2, accumulator.getNumVariables());

        Map<String, INDArray> result = accumulator.getAndReset();
        assertEquals(2.0, result.get("w1").getDouble(0), 1e-6);  // (1+3)/2
        assertEquals(3.0, result.get("w2").getDouble(0), 1e-6);  // (2+4)/2
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradientAccumulatorGetAndRemove(Nd4jBackend backend) {
        GradientAccumulator accumulator = new GradientAccumulator(2);

        accumulator.accumulate("w", Nd4j.create(new double[]{2.0}));
        accumulator.step();
        accumulator.accumulate("w", Nd4j.create(new double[]{4.0}));
        accumulator.step();

        INDArray grad = accumulator.getAndRemove("w");
        assertEquals(3.0, grad.getDouble(0), 1e-6);  // (2+4)/2

        assertFalse(accumulator.hasGradient("w"));
    }

    // ========== TrainingConfig Mixed Precision Tests ==========

    @Test
    public void testTrainingConfigMixedPrecision() {
        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(0.001))
                .mixedPrecision()
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        assertTrue(config.isMixedPrecision());
        assertEquals(DataType.FLOAT16, config.getComputeDataType());
        assertEquals(DataType.FLOAT, config.getMasterWeightDataType());
        assertTrue(config.isLossScalingEnabled());
        assertNotNull(config.getLossScaleConfig());
        assertTrue(config.getLossScaleConfig().isDynamic());
    }

    @Test
    public void testTrainingConfigMixedPrecisionBfloat16() {
        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(0.001))
                .mixedPrecisionBfloat16()
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        assertTrue(config.isMixedPrecision());
        assertEquals(DataType.BFLOAT16, config.getComputeDataType());
        assertEquals(DataType.FLOAT, config.getMasterWeightDataType());
        assertFalse(config.isLossScalingEnabled());  // BF16 doesn't need loss scaling
    }

    @Test
    public void testTrainingConfigCustomMixedPrecision() {
        LossScaleConfig lossScaleConfig = LossScaleConfig.staticScaling(2048.0);

        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(0.001))
                .computeDataType(DataType.FLOAT16)
                .masterWeightDataType(DataType.FLOAT)
                .lossScaling(lossScaleConfig)
                .gradientAccumulationSteps(8)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        assertTrue(config.isMixedPrecision());
        assertEquals(DataType.FLOAT16, config.getComputeDataType());
        assertEquals(DataType.FLOAT, config.getMasterWeightDataType());
        assertTrue(config.isLossScalingEnabled());
        assertEquals(2048.0, config.getLossScaleConfig().getInitialScale());
        assertTrue(config.isGradientAccumulationEnabled());
        assertEquals(8, config.getGradientAccumulationSteps());
    }

    @Test
    public void testTrainingConfigNonMixedPrecision() {
        TrainingConfig config = TrainingConfig.builder()
                .updater(new Adam(0.001))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        assertFalse(config.isMixedPrecision());
        assertEquals(DataType.FLOAT, config.getComputeDataType());
        assertFalse(config.isLossScalingEnabled());
        assertFalse(config.isGradientAccumulationEnabled());
    }

    @Test
    public void testTrainingConfigGradientAccumulationValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            TrainingConfig.builder()
                    .updater(new Adam(0.001))
                    .gradientAccumulationSteps(0)
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();
        });
    }

    // ========== Integration Tests ==========

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMixedPrecisionWorkflow(Nd4jBackend backend) {
        // Test the complete workflow of mixed precision training components

        // 1. Configure loss scaling
        LossScaleConfig lossScaleConfig = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(1024.0)
                .growthInterval(2)
                .build();

        // 2. Create loss scaler
        LossScaler scaler = new LossScaler(lossScaleConfig);

        // 3. Create gradient accumulator
        GradientAccumulator accumulator = new GradientAccumulator(2);

        // Simulate training loop
        for (int step = 0; step < 4; step++) {
            // Simulate forward pass - scale loss
            double loss = 0.5;
            double scaledLoss = scaler.scaleLoss(loss);
            assertEquals(loss * scaler.getCurrentScale(), scaledLoss, 1e-6);

            // Simulate backward pass - get gradients
            INDArray gradients = Nd4j.create(new double[]{0.1, 0.2, 0.3}).mul(scaler.getCurrentScale());

            // Unscale and check gradients
            boolean finite = scaler.unscaleGradientsAndCheck(gradients);
            assertTrue(finite);

            // Accumulate gradients
            accumulator.accumulate("w", gradients);
            accumulator.step();

            // Apply updates when ready
            if (accumulator.isReady()) {
                Map<String, INDArray> avgGrads = accumulator.getAndReset();
                assertNotNull(avgGrads.get("w"));

                // Update loss scaler
                scaler.update(true);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMixedPrecisionWithOverflow(Nd4jBackend backend) {
        LossScaleConfig config = LossScaleConfig.builder()
                .mode(LossScaleConfig.Mode.DYNAMIC)
                .initialScale(65536.0)
                .backoffFactor(0.5)
                .build();
        LossScaler scaler = new LossScaler(config);

        // Simulate gradient overflow
        INDArray overflowGradients = Nd4j.create(new double[]{Double.POSITIVE_INFINITY, 1.0});
        scaler.unscaleGradients(overflowGradients);

        assertFalse(scaler.areGradientsFinite(overflowGradients));

        // Update should reduce scale
        double scaleBefore = scaler.getCurrentScale();
        scaler.update(false);
        assertEquals(scaleBefore * 0.5, scaler.getCurrentScale(), 1e-6);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
