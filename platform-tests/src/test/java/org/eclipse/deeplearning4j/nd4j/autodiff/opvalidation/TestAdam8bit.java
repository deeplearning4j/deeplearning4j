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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.BlockQuantizationUtils;
import org.nd4j.linalg.learning.config.Adam8bit;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Adam 8-bit optimizer and block quantization utilities.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Adam 8-bit Optimizer Tests")
public class TestAdam8bit {

    @Test
    @DisplayName("Adam8bit - Default Config")
    public void testAdam8bitConfig() {
        Adam8bit adam = new Adam8bit();

        assertEquals(Adam8bit.DEFAULT_LEARNING_RATE, adam.getLearningRate());
        assertEquals(Adam8bit.DEFAULT_BETA1, adam.getBeta1());
        assertEquals(Adam8bit.DEFAULT_BETA2, adam.getBeta2());
        assertEquals(Adam8bit.DEFAULT_EPSILON, adam.getEpsilon());
        assertEquals(Adam8bit.DEFAULT_BLOCK_SIZE, adam.getBlockSize());
        assertFalse(adam.isPagedOptimizer());
        assertTrue(adam.hasLearningRate());
    }

    @Test
    @DisplayName("Adam8bit - With Learning Rate Schedule")
    public void testAdam8bitWithSchedule() {
        ISchedule schedule = new FixedSchedule(0.01);
        Adam8bit adam = new Adam8bit(schedule);

        assertNotNull(adam.getLearningRateSchedule());
        assertEquals(0.01, adam.getLearningRate(0, 0), 1e-9);
        assertEquals(0.01, adam.getLearningRate(100, 1), 1e-9);
    }

    @Test
    @DisplayName("Block Quantization - Roundtrip Accuracy")
    public void testBlockQuantizationRoundtrip() {
        INDArray original = Nd4j.randn(DataType.FLOAT, 1, 4096);
        int blockSize = 2048;

        Pair<INDArray, INDArray> quantized = BlockQuantizationUtils.quantizeBlocks(original, blockSize);
        INDArray quantizedData = quantized.getFirst();
        INDArray scales = quantized.getSecond();

        assertNotNull(quantizedData);
        assertNotNull(scales);
        assertEquals(original.length(), quantizedData.length());
        assertEquals(BlockQuantizationUtils.numBlocks(original.length(), blockSize), scales.length());

        INDArray dequantized = BlockQuantizationUtils.dequantizeBlocks(quantizedData, scales, blockSize);
        assertNotNull(dequantized);
        assertEquals(original.length(), dequantized.length());

        // Check max error is within INT8 quantization granularity
        INDArray diff = original.reshape(original.length()).sub(dequantized);
        double maxError = diff.amaxNumber().doubleValue();
        double absMax = original.amaxNumber().doubleValue();
        double relativeError = maxError / (absMax + 1e-10);
        log.info("Block quantization max absolute error: {}, relative error: {}", maxError, relativeError);

        // INT8 quantizes to 255 levels, so relative error should be < ~1/127 ~= 0.008 per block
        // With some tolerance for rounding:
        assertTrue(relativeError < 0.02,
                "Relative quantization error too large: " + relativeError);
    }

    @Test
    @DisplayName("Block Quantization - Preserves Zeros")
    public void testBlockQuantizationPreservesZeros() {
        INDArray zeros = Nd4j.zeros(DataType.FLOAT, 1, 4096);

        Pair<INDArray, INDArray> quantized = BlockQuantizationUtils.quantizeBlocks(zeros);
        INDArray dequantized = BlockQuantizationUtils.dequantizeBlocks(
                quantized.getFirst(), quantized.getSecond(), BlockQuantizationUtils.DEFAULT_BLOCK_SIZE);

        // All zeros should remain zeros after roundtrip
        assertEquals(0.0, dequantized.amaxNumber().doubleValue(), 1e-10,
                "Dequantized zeros should remain zeros");
    }

    @Test
    @DisplayName("Adam8bit - State Size")
    public void testAdam8bitStateSize() {
        Adam8bit adam = new Adam8bit();
        // Adam needs m and v states -> 2 * numParams
        assertEquals(2000, adam.stateSize(1000));
        assertEquals(200, adam.stateSize(100));
        assertEquals(0, adam.stateSize(0));
    }

    @Test
    @DisplayName("Adam8bit - Convergence on Quadratic")
    public void testAdam8bitConvergence() {
        // Minimize f(x) = x^2 starting from x=5.0
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", Nd4j.scalar(DataType.FLOAT, 5.0));
        SDVariable loss = x.mul(x);
        loss.rename("loss");

        sd.setLossVariables("loss");

        Adam8bit adam = Adam8bit.builder()
                .learningRate(0.1)
                .build();

        TrainingConfig config = TrainingConfig.builder()
                .updater(adam)
                .dataSetFeatureMapping("x")
                .markLabelsUnused()
                .build();
        sd.setTrainingConfig(config);

        // Run 100 optimization steps
        INDArray dummyInput = Nd4j.scalar(DataType.FLOAT, 0.0);
        for (int i = 0; i < 100; i++) {
            sd.fit(new DataSet(dummyInput, dummyInput));
        }

        double finalX = sd.getVariable("x").getArr().getDouble(0);
        log.info("After 100 steps of Adam8bit, x = {}", finalX);
        assertTrue(Math.abs(finalX) < 0.5,
                "x should converge near 0 after 100 steps, got: " + finalX);
    }
}
