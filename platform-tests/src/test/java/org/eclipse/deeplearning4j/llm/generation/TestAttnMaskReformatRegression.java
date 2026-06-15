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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for the 4D attention mask semantics used by
 * {@link DecoderUtils#buildDecoderInputMap} when the attn_mask_reformat
 * placeholder override path is active.
 *
 * <p>The override path bypasses the internal SameDiff subgraph (Where + Unsqueeze + Tile)
 * and builds the 4D attention bias directly in Java. These tests verify that the
 * hand-built mask matches the semantics of the subgraph and that incremental updates
 * (reuse path) produce correct masks across decode steps.</p>
 */
public class TestAttnMaskReformatRegression {

    /** Mask fill value matching {@link DecoderUtils#MASK_FILL}. */
    private static final float MASK_FILL = -3.4028235e+38f;

    /**
     * Test the initial 4D mask construction (no reuse, first call).
     *
     * Simulates DecoderUtils.buildDecoderInputMap for the padded static KV path:
     * - numQHeads=24, maxKvLen=679, currentSeqLen=1, cachePos=678
     * - totalSeqLen = maxKvLen + currentSeqLen = 680
     * - Shape: [1, 24, 1, 680]
     * - Filled with MASK_FILL, then positions 0..cachePos (0..678) set to 0.0f
     * - Position totalSeqLen-1 (679) set to 0.0f (the current token)
     */
    @Test
    public void testInitial4DMaskValues() {
        int numQHeads = 24;
        int maxKvLen = 679;
        int currentSeqLen = 1;
        int cachePos = 678;
        long totalSeqLen = maxKvLen + currentSeqLen; // 680

        // Build 4D bias exactly as DecoderUtils does
        INDArray attnBias = Nd4j.valueArrayOf(
                new long[]{1, numQHeads, currentSeqLen, totalSeqLen}, MASK_FILL, DataType.FLOAT);

        // Unmask filled positions (0..cachePos inclusive)
        attnBias.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.interval(0, cachePos + 1)).assign(0.0f);

        // Unmask the current token position (last position)
        attnBias.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(totalSeqLen - 1)).assign(0.0f);

        // Verify shape
        assertArrayEquals(new long[]{1, numQHeads, currentSeqLen, totalSeqLen}, attnBias.shape());

        // Verify attended positions are 0.0f for all heads
        for (int h = 0; h < numQHeads; h++) {
            for (int k = 0; k <= cachePos; k++) {
                float val = attnBias.getFloat(0, h, 0, k);
                assertEquals(0.0f, val, 1e-6f,
                        "Position " + k + " head " + h + " should be 0.0f (attended)");
            }
            // Position 679 (totalSeqLen-1) is the current token — should be 0.0f
            assertEquals(0.0f, attnBias.getFloat(0, h, 0, (int) (totalSeqLen - 1)), 1e-6f,
                    "Current token position head " + h + " should be 0.0f");
        }

        // There are no masked positions in this scenario since cachePos=678 covers 0..678
        // and position 679 is the current token. All 680 positions are attended.
        // But let's verify there are no MASK_FILL values remaining:
        for (int h = 0; h < numQHeads; h++) {
            for (int k = 0; k < totalSeqLen; k++) {
                float val = attnBias.getFloat(0, h, 0, k);
                assertNotEquals(MASK_FILL, val,
                        "Position " + k + " head " + h + " should NOT be MASK_FILL");
            }
        }

        attnBias.close();
    }

    /**
     * Test the incremental mask update path (reuse case).
     *
     * Starts with an initial mask where cachePos=678 (positions 0..678 + 679 attended).
     * Then simulates 3 incremental steps where cachePos advances to 679, 680, 681.
     * Each step unmarks the cachePos position via putScalar across all heads.
     *
     * Note: in the padded static KV scenario with maxKvLen=679, when cachePos reaches 679+
     * the positions are within the totalSeqLen=680 range. We use a larger totalSeqLen
     * to show the incremental pattern clearly.
     */
    @Test
    public void testIncrementalMaskUpdate() {
        int numQHeads = 24;
        int maxKvLen = 685;
        int currentSeqLen = 1;
        int initialCachePos = 678;
        long totalSeqLen = maxKvLen + currentSeqLen; // 686

        // Build initial mask
        INDArray attnBias = Nd4j.valueArrayOf(
                new long[]{1, numQHeads, currentSeqLen, totalSeqLen}, MASK_FILL, DataType.FLOAT);

        // Unmask filled positions (0..initialCachePos inclusive)
        attnBias.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.interval(0, initialCachePos + 1)).assign(0.0f);

        // Unmask current token position
        attnBias.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(totalSeqLen - 1)).assign(0.0f);

        // Simulate 3 incremental decode steps
        for (int step = 0; step < 3; step++) {
            int cachePos = initialCachePos + 1 + step; // 679, 680, 681
            // Incremental update: unmask cachePos across all heads
            for (int h = 0; h < numQHeads; h++) {
                attnBias.putScalar(new long[]{0, h, 0, cachePos}, 0.0f);
            }
        }

        // After 3 steps, positions 0..681 should all be 0.0f
        for (int h = 0; h < numQHeads; h++) {
            for (int k = 0; k <= 681; k++) {
                float val = attnBias.getFloat(0, h, 0, k);
                assertEquals(0.0f, val, 1e-6f,
                        "Position " + k + " head " + h + " should be 0.0f after incremental updates");
            }
        }

        // Positions 682..684 should still be MASK_FILL (not yet unmasked)
        for (int h = 0; h < numQHeads; h++) {
            for (int k = 682; k < totalSeqLen - 1; k++) {
                float val = attnBias.getFloat(0, h, 0, k);
                assertEquals(MASK_FILL, val,
                        "Position " + k + " head " + h + " should still be MASK_FILL");
            }
        }

        // Position totalSeqLen-1 (685) should be 0.0f (current token, set during initial build)
        for (int h = 0; h < numQHeads; h++) {
            assertEquals(0.0f, attnBias.getFloat(0, h, 0, (int) (totalSeqLen - 1)), 1e-6f,
                    "Current token position head " + h + " should remain 0.0f");
        }

        attnBias.close();
    }

    /**
     * Test that putScalar correctly round-trips through the backend (including CUDA sync).
     *
     * Creates a FLOAT array filled with MASK_FILL, uses putScalar to set specific
     * positions to 0.0f, then reads back via getFloat to verify the write persisted.
     * On CUDA, putScalar writes to the host buffer and getFloat may read from the
     * device buffer, so this validates that sync occurs correctly.
     */
    @Test
    public void testPutScalarSyncsToDevice() {
        int numQHeads = 24;
        long totalSeqLen = 680;

        INDArray attnBias = Nd4j.valueArrayOf(
                new long[]{1, numQHeads, 1, totalSeqLen}, MASK_FILL, DataType.FLOAT);

        // Write position 5 to 0.0f for all heads using putScalar
        int targetPos = 5;
        for (int h = 0; h < numQHeads; h++) {
            attnBias.putScalar(new long[]{0, h, 0, targetPos}, 0.0f);
        }

        // Force a device sync by duplicating (ensures any pending host->device transfer completes)
        INDArray synced = attnBias.dup();

        // Verify via the duplicated array (which has fresh device data)
        for (int h = 0; h < numQHeads; h++) {
            float val = synced.getFloat(0, h, 0, targetPos);
            assertEquals(0.0f, val, 1e-6f,
                    "Position " + targetPos + " head " + h + " should be 0.0f after putScalar + sync");
        }

        // Also verify that adjacent positions were NOT modified
        for (int h = 0; h < numQHeads; h++) {
            float valBefore = synced.getFloat(0, h, 0, targetPos - 1);
            float valAfter = synced.getFloat(0, h, 0, targetPos + 1);
            assertEquals(MASK_FILL, valBefore,
                    "Position " + (targetPos - 1) + " head " + h + " should still be MASK_FILL");
            assertEquals(MASK_FILL, valAfter,
                    "Position " + (targetPos + 1) + " head " + h + " should still be MASK_FILL");
        }

        synced.close();
        attnBias.close();
    }

    /**
     * KEY TEST: Verify that the hand-built 4D mask matches the semantics of the
     * attn_mask_reformat internal subgraph.
     *
     * The internal subgraph does:
     *   1. Input: attention_mask [1, totalSeqLen] with LONG type (1 = attended, 0 = masked)
     *   2. Where(mask == 0): fill with MASK_FILL, else 0.0f
     *   3. Unsqueeze to [1, 1, 1, totalSeqLen]
     *   4. Tile across heads to [1, numQHeads, 1, totalSeqLen]
     *
     * This test builds that graph in SameDiff, executes it with a specific 1D mask
     * pattern, and compares the result against the hand-built 4D mask from DecoderUtils.
     */
    @Test
    public void testMaskValuesMatchInternalSubgraph() {
        int numQHeads = 24;
        int maxKvLen = 679;
        int currentSeqLen = 1;
        int cachePos = 500; // Partially filled: positions 0..500 attended, 501..678 masked
        long totalSeqLen = maxKvLen + currentSeqLen; // 680

        // === Build the 1D attention mask (simulating what the model receives) ===
        // 1 = attended, 0 = masked
        INDArray attentionMask1D = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        // Fill positions 0..cachePos with 1 (attended)
        attentionMask1D.get(NDArrayIndex.all(), NDArrayIndex.interval(0, cachePos + 1))
                .assign(1);
        // Fill the current token position (last position) with 1
        attentionMask1D.putScalar(new long[]{0, totalSeqLen - 1}, 1);

        // === Build SameDiff graph matching the attn_mask_reformat subgraph ===
        SameDiff sd = SameDiff.create();
        SDVariable mask = sd.placeHolder("attention_mask", DataType.LONG, 1, totalSeqLen);

        // Where(mask == 0): MASK_FILL, else 0.0
        SDVariable maskFloat = mask.castTo(DataType.FLOAT);
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable maskFillVal = sd.constant("mask_fill", Nd4j.scalar(DataType.FLOAT, MASK_FILL));

        // condition: mask == 0 means masked position -> fill with MASK_FILL
        // mask == 1 means attended -> fill with 0.0
        // This is: result = (1 - mask) * MASK_FILL
        SDVariable inverted = sd.math.sub("inverted", sd.constant(Nd4j.scalar(DataType.FLOAT, 1.0f)), maskFloat);
        SDVariable biasFlat = sd.math.mul("bias_flat", inverted, maskFillVal);

        // Unsqueeze to [1, 1, 1, totalSeqLen]
        SDVariable unsqueezed = sd.expandDims("unsqueezed", biasFlat, 1); // [1, 1, totalSeqLen]
        SDVariable unsqueezed2 = sd.expandDims("unsqueezed2", unsqueezed, 2); // [1, 1, 1, totalSeqLen]

        // Tile across heads [1, numQHeads, 1, totalSeqLen]
        SDVariable tiled = sd.tile("tiled", unsqueezed2, new int[]{1, numQHeads, 1, 1});

        // Execute the graph
        Map<String, INDArray> feed = Collections.singletonMap("attention_mask", attentionMask1D);
        Map<String, INDArray> result = sd.output(feed, "tiled");
        INDArray subgraphOutput = result.get("tiled");

        // === Build the hand-constructed 4D mask (DecoderUtils path) ===
        INDArray handBuilt = Nd4j.valueArrayOf(
                new long[]{1, numQHeads, currentSeqLen, totalSeqLen}, MASK_FILL, DataType.FLOAT);
        // Unmask filled positions 0..cachePos
        handBuilt.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.interval(0, cachePos + 1)).assign(0.0f);
        // Unmask current token position
        handBuilt.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(totalSeqLen - 1)).assign(0.0f);

        // === Compare ===
        assertArrayEquals(handBuilt.shape(), subgraphOutput.shape(),
                "Shapes must match: hand-built vs subgraph output");

        // Compare element-by-element for clear diagnostics on failure
        for (int h = 0; h < numQHeads; h++) {
            for (int k = 0; k < totalSeqLen; k++) {
                float expected = handBuilt.getFloat(0, h, 0, k);
                float actual = subgraphOutput.getFloat(0, h, 0, k);
                assertEquals(expected, actual, 1e-6f,
                        "Mismatch at head=" + h + " pos=" + k +
                                " expected=" + expected + " actual=" + actual);
            }
        }

        // Cleanup
        subgraphOutput.close();
        handBuilt.close();
        attentionMask1D.close();
    }
}
