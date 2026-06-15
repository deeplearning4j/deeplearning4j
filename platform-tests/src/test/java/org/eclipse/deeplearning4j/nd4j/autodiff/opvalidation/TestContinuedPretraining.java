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
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.config.ContinuedPretrainingConfig;
import org.nd4j.autodiff.samediff.config.FP8TrainingConfig;
import org.nd4j.autodiff.samediff.training.FP8ScaleManager;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for continued pretraining configuration, cosine warmup schedule,
 * FP8 training config, and FP8 scale manager.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Continued Pretraining Tests")
public class TestContinuedPretraining {

    @Test
    @DisplayName("CosineWarmupSchedule - Shape")
    public void testCosineWarmupScheduleShape() {
        CosineWarmupSchedule schedule = new CosineWarmupSchedule(0.001, 0.0, 100, 1000);

        // At iteration 0: linear warmup starts at 0
        double lr0 = schedule.valueAt(0, 0);
        assertEquals(0.0, lr0, 1e-10, "LR at step 0 should be 0 (start of warmup)");

        // At iteration 100: peak LR = maxLR
        double lr100 = schedule.valueAt(100, 0);
        // At step 100, warmup just ended. The cosine phase starts with progress=0, so:
        // lr = minLR + 0.5 * (maxLR - minLR) * (1 + cos(0)) = minLR + (maxLR - minLR) = maxLR
        assertEquals(0.001, lr100, 1e-10, "LR at step 100 should be maxLR (end of warmup)");

        // At iteration 1000: should be at minLR
        double lr1000 = schedule.valueAt(1000, 0);
        assertEquals(0.0, lr1000, 1e-10, "LR at step 1000 should be minLR");

        // At iteration 550 (midpoint of cosine phase): should be between min and max
        double lr550 = schedule.valueAt(550, 0);
        assertTrue(lr550 > 0.0, "LR at step 550 should be > minLR");
        assertTrue(lr550 < 0.001, "LR at step 550 should be < maxLR");
        log.info("LR at step 550: {}", lr550);
    }

    @Test
    @DisplayName("CosineWarmupSchedule - From Ratio")
    public void testCosineWarmupFromRatio() {
        CosineWarmupSchedule schedule = CosineWarmupSchedule.fromRatio(
                0.001, 0.0, 0.1, 1000);

        // warmupRatio=0.1, totalSteps=1000 -> warmupSteps=100
        assertEquals(100, schedule.getWarmupSteps());
        assertEquals(0.001, schedule.getMaxLR(), 1e-10);
        assertEquals(0.0, schedule.getMinLR(), 1e-10);
        assertEquals(1000, schedule.getTotalSteps());

        // Verify value at warmup midpoint
        double lr50 = schedule.valueAt(50, 0);
        assertEquals(0.0005, lr50, 1e-10, "LR at step 50 should be half of maxLR");
    }

    @Test
    @DisplayName("ContinuedPretrainingConfig - Defaults")
    public void testContinuedPretrainingConfigDefaults() {
        ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder().build();

        assertEquals(2048, config.getChunkSize());
        assertEquals(128, config.getChunkOverlap());
        assertEquals(5e-5, config.getLearningRate(), 1e-10);
        assertEquals(0.1, config.getWarmupRatio(), 1e-10);
        assertEquals(1e-6, config.getMinLearningRate(), 1e-10);
        assertEquals(4, config.getGradientAccumulationSteps());
        assertEquals(DataType.BFLOAT16, config.getComputeDataType());
        assertFalse(config.isUseLoRA());
        assertEquals(16, config.getLoraRank());
        assertEquals(32, config.getLoraAlpha());
        assertEquals(0.01, config.getWeightDecay(), 1e-10);
        assertEquals(1.0, config.getMaxGradNorm(), 1e-10);
        assertEquals(1, config.getNumEpochs());
        assertNull(config.getLrSchedule());
        assertNull(config.getGradientCheckpointConfig());
    }

    @Test
    @DisplayName("ContinuedPretrainingConfig - Validation")
    public void testContinuedPretrainingConfigValidation() {
        // chunkSize <= 0 should throw
        assertThrows(IllegalStateException.class, () -> {
            ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
                    .chunkSize(0)
                    .build();
            config.validate();
        });

        // chunkOverlap >= chunkSize should throw
        assertThrows(IllegalStateException.class, () -> {
            ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
                    .chunkSize(100)
                    .chunkOverlap(100)
                    .build();
            config.validate();
        });

        // Negative learningRate should throw
        assertThrows(IllegalStateException.class, () -> {
            ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
                    .learningRate(-1.0)
                    .build();
            config.validate();
        });

        // gradientAccumulationSteps < 1 should throw
        assertThrows(IllegalStateException.class, () -> {
            ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
                    .gradientAccumulationSteps(0)
                    .build();
            config.validate();
        });

        // Valid config should not throw
        assertDoesNotThrow(() -> {
            ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder().build();
            config.validate();
        });
    }

    @Test
    @DisplayName("FP8TrainingConfig - Op Eligibility")
    public void testFP8TrainingConfig() {
        FP8TrainingConfig config = FP8TrainingConfig.builder().build();

        // matmul is eligible
        assertTrue(config.isEligibleForFP8("matmul"), "matmul should be eligible for FP8");
        assertTrue(config.isEligibleForFP8("linear"), "linear should be eligible for FP8");
        assertTrue(config.isEligibleForFP8("dense"), "dense should be eligible for FP8");

        // softmax is excluded
        assertFalse(config.isEligibleForFP8("softmax"), "softmax should NOT be eligible for FP8");
        assertFalse(config.isEligibleForFP8("layernorm"), "layernorm should NOT be eligible for FP8");
        assertFalse(config.isEligibleForFP8("attention"), "attention should NOT be eligible for FP8");
        assertFalse(config.isEligibleForFP8("gelu"), "gelu should NOT be eligible for FP8");
        assertFalse(config.isEligibleForFP8("silu"), "silu should NOT be eligible for FP8");

        // Unknown ops that are not in eligible set
        assertFalse(config.isEligibleForFP8("conv2d"), "conv2d should NOT be eligible for FP8");

        // Default settings
        assertTrue(config.isUseE4M3ForForward());
        assertTrue(config.isPerTensorScaling());
        assertEquals(16, config.getAmaxHistoryLength());
    }

    @Test
    @DisplayName("FP8ScaleManager - Scale Computation")
    public void testFP8ScaleManager() {
        FP8TrainingConfig config = FP8TrainingConfig.builder()
                .amaxHistoryLength(4)
                .build();
        FP8ScaleManager manager = new FP8ScaleManager(config);

        // Initially, scale should be 1.0 (no data yet)
        assertEquals(1.0, manager.getScale("tensor1"), 1e-10);

        // Update amax for forward pass (E4M3, max=448)
        manager.updateAmax("tensor1", 10.0, true);
        double scale1 = manager.getScale("tensor1");
        // scale = 448 / 10.0 = 44.8
        assertEquals(44.8, scale1, 1e-6, "Scale should be FP8_E4M3_MAX / amax = 448/10");

        // Update with larger amax
        manager.updateAmax("tensor1", 100.0, true);
        double scale2 = manager.getScale("tensor1");
        // max(amax_history) = 100.0, scale = 448 / 100 = 4.48
        assertEquals(4.48, scale2, 1e-6, "Scale should use max of history");

        // Inverse scale
        double invScale = manager.getInverseScale("tensor1");
        assertEquals(1.0 / scale2, invScale, 1e-10);

        // Backward pass scale uses E5M2 max = 57344
        manager.updateAmax("grad_tensor", 50.0, false);
        double gradScale = manager.getScale("grad_tensor");
        assertEquals(57344.0 / 50.0, gradScale, 1e-6, "Backward scale should use FP8_E5M2_MAX");

        // Tracked tensor count
        assertEquals(2, manager.getTrackedTensorCount());

        // Clear
        manager.clear();
        assertEquals(0, manager.getTrackedTensorCount());
        assertEquals(1.0, manager.getScale("tensor1"), 1e-10);
    }
}
