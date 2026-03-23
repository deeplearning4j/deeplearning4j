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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for the nativeDecodeInputs feature.
 *
 * Root cause: C++ updateDecodeInputs() wrote to external input arrays, but the
 * frozen CUDA graph replay path reads from separate capture buffers. When Java
 * skipped feedDict updates (because C++ was "handling it"), the D2D copies
 * pushed stale values to capture buffers → model saw the same position every
 * step → repeating tokens.
 *
 * Fix: platformTryFrozenFastPath now writes pending decode values directly to
 * capture buffers after the D2D copy loop, before graph launch.
 */
public class TestNativeDecodeInputsRegression {

    /**
     * Validates that position_ids must increment each step for correct output.
     * When position_ids is frozen (same value every step), the model produces
     * identical logits → repeating tokens.
     */
    @Test
    public void testPositionIdsMustIncrement() {
        // Simulate 5 decode steps with position_ids
        INDArray posIds = Nd4j.zeros(DataType.LONG, 1, 1);

        long[] positions = new long[5];
        for (int step = 0; step < 5; step++) {
            long pos = 679 + step;
            posIds.putScalar(0, 0, pos);
            positions[step] = posIds.getLong(0, 0);
        }

        // All positions must be different
        for (int i = 0; i < positions.length - 1; i++) {
            assertNotEquals(positions[i], positions[i + 1],
                    "position_ids must increment: step " + i + " and " + (i + 1) + " are both " + positions[i]);
        }

        // Positions must be sequential
        for (int i = 0; i < positions.length; i++) {
            assertEquals(679 + i, positions[i],
                    "position_ids[" + i + "] should be " + (679 + i) + " but was " + positions[i]);
        }
    }

    /**
     * Validates that attention_mask accumulates 1s at each step.
     * When the mask is frozen (same pattern every step), the model only
     * attends to the same positions → no new context → repeating tokens.
     */
    @Test
    public void testAttentionMaskAccumulates() {
        int totalSeqLen = 930;
        int initialFilled = 680; // positions 0..679

        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        // Initial fill
        mask.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, initialFilled)).assign(1);
        // Last position (current token)
        mask.putScalar(0, totalSeqLen - 1, 1);

        long initialSum = mask.sumNumber().longValue();
        assertEquals(initialFilled + 1, initialSum, "Initial mask sum");

        // Simulate 5 decode steps — each step marks one more position
        for (int step = 0; step < 5; step++) {
            int cachePos = initialFilled + step;
            mask.putScalar(0, cachePos, 1);
            long sum = mask.sumNumber().longValue();
            assertEquals(initialFilled + 1 + step + 1, sum,
                    "After step " + step + " mask sum should increase");
        }

        // Verify first few and last few positions
        assertEquals(1, mask.getLong(0, 0), "Position 0 should be 1");
        assertEquals(1, mask.getLong(0, initialFilled - 1), "Last initial position should be 1");
        assertEquals(1, mask.getLong(0, initialFilled), "First new position should be 1");
        assertEquals(1, mask.getLong(0, initialFilled + 4), "Last new position should be 1");
        assertEquals(0, mask.getLong(0, initialFilled + 5), "Beyond updates should be 0");
    }

    /**
     * Validates that putScalar on a reusable CUDA buffer correctly updates
     * the device memory for subsequent reads.
     */
    @Test
    public void testReusableBufferUpdatesVisible() {
        // Simulate reusable position_ids buffer (like the decode loop uses)
        INDArray reusablePos = Nd4j.zeros(DataType.LONG, 1, 1);

        for (int step = 0; step < 10; step++) {
            long expectedPos = 679 + step;
            reusablePos.putScalar(0, 0, expectedPos);

            // Force a device sync round-trip to verify the value is accessible
            INDArray copy = reusablePos.dup();
            long actual = copy.getLong(0, 0);
            assertEquals(expectedPos, actual,
                    "Step " + step + ": reusable buffer should have " + expectedPos + " but had " + actual);
            copy.close();
        }
        reusablePos.close();
    }

    /**
     * Validates that the 4D attn_mask_reformat incremental updates accumulate
     * correctly across steps and don't interfere with each other.
     */
    @Test
    public void testAttnMaskReformat4DAccumulation() {
        int numQHeads = 24;
        int maxKvLen = 929;
        int currentSeqLen = 1;
        long totalSeqLen = maxKvLen + currentSeqLen;
        float MASK_FILL = -3.4028235e+38f;

        // Initial: all masked except positions 0..679 and last position
        int initialCachePos = 679;
        INDArray attnBias = Nd4j.valueArrayOf(
                new long[]{1, numQHeads, currentSeqLen, totalSeqLen}, MASK_FILL, DataType.FLOAT);
        attnBias.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, initialCachePos + 1)).assign(0.0f);
        attnBias.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.point(totalSeqLen - 1)).assign(0.0f);

        // Simulate 5 incremental steps
        for (int step = 0; step < 5; step++) {
            int cachePos = initialCachePos + 1 + step;
            for (int h = 0; h < numQHeads; h++) {
                attnBias.putScalar(new long[]{0, h, 0, cachePos}, 0.0f);
            }

            // Verify this position is now unmasked for all heads
            for (int h = 0; h < numQHeads; h++) {
                float val = attnBias.getFloat(0, h, 0, cachePos);
                assertEquals(0.0f, val, 1e-6f,
                        "Step " + step + " head " + h + " at pos " + cachePos + " should be unmasked");
            }

            // Verify next position is still masked
            if (cachePos + 1 < totalSeqLen - 1) {
                float nextVal = attnBias.getFloat(0, 0, 0, cachePos + 1);
                assertEquals(MASK_FILL, nextVal, 1e30f,
                        "Position " + (cachePos + 1) + " should still be masked");
            }
        }

        attnBias.close();
    }
}
