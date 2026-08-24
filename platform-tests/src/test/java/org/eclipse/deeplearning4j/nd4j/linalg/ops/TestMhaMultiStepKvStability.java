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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for multi-step MHA decode with KV cache, verifying that KV values
 * don't grow unboundedly. This directly investigates the root cause of
 * degenerate VLM output (exploding attention scores).
 *
 * The decode pattern simulates what the decode pipeline does:
 * 1. Prefill: run MHA with Q/K/V from embeddings (seqLen > 1)
 * 2. Pad KV to static size (maxKvLen)
 * 3. Decode steps: run MHA with seqLen=1, scatter new KV entry into static buffer
 *
 * The test verifies that after N decode steps, KV values remain bounded.
 */
public class TestMhaMultiStepKvStability {

    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 16;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;

    /**
     * Test the VIEW-BASED decode path (no freeze / no DSP).
     * Past KV is a growing view [0:cachePos] of the static buffer.
     * MHA concatenates past + new, KvScatter writes last position back.
     */
    @Test
    public void testViewBasedDecodeKvStability() {
        int B = 1;
        int prefillLen = 10;
        int maxKvLen = prefillLen + 50;
        int decodeSteps = 20;

        // Prefill: generate initial KV from random Q/K/V
        INDArray prefillQ = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);

        INDArray[] prefillResult = Nd4j.exec(new OnnxMultiHeadAttention(
                prefillQ, prefillK, prefillV, null, null, null,
                NUM_HEADS, 0.0, false));

        INDArray presentKey = prefillResult[1]; // [B, numHeads, prefillLen, headDim]
        INDArray presentValue = prefillResult[2];

        // Pad to static size
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all()).assign(presentKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all()).assign(presentValue);

        long cachePos = prefillLen;
        double prevKvMax = presentKey.amaxNumber().doubleValue();
        System.out.println("Prefill KV absMax: " + prevKvMax);

        // Decode steps with view-based KV
        for (int step = 0; step < decodeSteps; step++) {
            // New token embedding (simulated)
            INDArray newQ = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);
            INDArray newK = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);
            INDArray newV = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);

            // View-based: pass [0:cachePos] slice as past KV
            INDArray pastKeyView = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
            INDArray pastValueView = staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());

            INDArray[] decodeResult = Nd4j.exec(new OnnxMultiHeadAttention(
                    newQ, newK, newV, null, pastKeyView, pastValueView,
                    NUM_HEADS, 0.0, false));

            INDArray decodePresentKey = decodeResult[1]; // [B, numHeads, cachePos+1, headDim]
            INDArray decodePresentValue = decodeResult[2];

            // Verify present output shapes
            assertEquals(cachePos + 1, decodePresentKey.size(2),
                    "Present key seq len at step " + step);

            // Scatter last position from present into static buffer
            Nd4j.exec(new KvScatter(staticKey, decodePresentKey, cachePos));
            Nd4j.exec(new KvScatter(staticValue, decodePresentValue, cachePos));

            // Check KV value magnitude
            double kvMax = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).amaxNumber().doubleValue();

            System.out.printf("Step %d: cachePos=%d, newKvMax=%.4f, prevKvMax=%.4f%n",
                    step, cachePos, kvMax, prevKvMax);

            // KV values should NOT grow unboundedly
            // Allow 5x the initial max as tolerance (some growth is OK for random inputs)
            assertTrue(kvMax < prevKvMax * 5 + 1.0,
                    "KV values growing unboundedly at step " + step +
                            ": kvMax=" + kvMax + " vs initial=" + prevKvMax);

            cachePos++;
        }

        // Final check: KV magnitude should be in same order of magnitude as initial
        double finalKvMax = staticKey.amaxNumber().doubleValue();
        System.out.println("Final KV absMax: " + finalKvMax + " vs initial: " + prevKvMax);
        assertTrue(finalKvMax < prevKvMax * 10 + 1.0,
                "KV values exploded: final=" + finalKvMax + " vs initial=" + prevKvMax);
    }

    /**
     * Test that the MHA concat properly assembles present_key = concat(past, new).
     * Verify the new entry matches the input K (after reshape/permute).
     */
    @Test
    public void testMhaConcatCorrectness() {
        int B = 1, pastSeq = 5;

        // Create known past KV
        INDArray pastKey = Nd4j.linspace(1, B * NUM_HEADS * pastSeq * HEAD_DIM, B * NUM_HEADS * pastSeq * HEAD_DIM)
                .reshape(B, NUM_HEADS, pastSeq, HEAD_DIM).castTo(DataType.FLOAT);
        INDArray pastValue = pastKey.mul(0.5);

        // New token with known values
        INDArray newQ = Nd4j.ones(DataType.FLOAT, B, 1, HIDDEN);
        INDArray newK = Nd4j.ones(DataType.FLOAT, B, 1, HIDDEN).muli(99.0);
        INDArray newV = Nd4j.ones(DataType.FLOAT, B, 1, HIDDEN).muli(-99.0);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(
                newQ, newK, newV, null, pastKey, pastValue,
                NUM_HEADS, 0.0, false));

        INDArray presentKeyOut = result[1]; // [B, numHeads, pastSeq+1, headDim]
        INDArray presentValueOut = result[2];

        assertEquals(pastSeq + 1, presentKeyOut.size(2), "Present key should have pastSeq+1 positions");
        assertEquals(pastSeq + 1, presentValueOut.size(2), "Present value should have pastSeq+1 positions");

        // Verify past portion preserved
        INDArray pastPortion = presentKeyOut.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all());
        double pastDiff = pastPortion.sub(pastKey).amaxNumber().doubleValue();
        assertTrue(pastDiff < 1e-4,
                "Past KV portion should be preserved, diff=" + pastDiff);

        // Verify new entry is at position pastSeq
        INDArray newEntry = presentKeyOut.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(pastSeq), NDArrayIndex.all());
        // New K was [B, 1, HIDDEN] = 99.0 everywhere
        // After reshape [B, 1, numHeads, headDim] and permute to BHSD, each entry = 99.0
        double newMax = newEntry.amaxNumber().doubleValue();
        double newMin = newEntry.aminNumber().doubleValue();
        assertEquals(99.0, newMax, 0.01, "New key entry max should be 99.0");
        assertEquals(99.0, newMin, 0.01, "New key entry min should be 99.0 (uniform)");
    }

    /**
     * Test KvScatter correctness: verify it overwrites (not accumulates).
     */
    @Test
    public void testKvScatterOverwrites() {
        int B = 1, maxSeq = 10;
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, maxSeq, HEAD_DIM);
        INDArray present = Nd4j.ones(DataType.FLOAT, B, NUM_HEADS, 3, HEAD_DIM).muli(42.0);

        // Scatter at position 5 (should write present's last position = position 2)
        Nd4j.exec(new KvScatter(staticBuf, present, 5));

        // Position 5 should have value 42.0
        INDArray pos5 = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(5), NDArrayIndex.all());
        assertEquals(42.0, pos5.getDouble(0), 0.01);

        // Now scatter different value at same position
        INDArray present2 = Nd4j.ones(DataType.FLOAT, B, NUM_HEADS, 1, HEAD_DIM).muli(7.0);
        Nd4j.exec(new KvScatter(staticBuf, present2, 5));

        // Position 5 should now be 7.0 (overwritten, not 42+7=49)
        INDArray pos5After = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(5), NDArrayIndex.all());
        assertEquals(7.0, pos5After.getDouble(0), 0.01,
                "KvScatter should overwrite, not accumulate");
    }

    /**
     * Test padded-mode decode: full static buffer passed as past KV.
     * This is the path used with DSP/CUDA graphs.
     * The key question: does attending to zero-padded positions cause issues?
     */
    @Test
    public void testPaddedModeDecodeStability() {
        int B = 1;
        int prefillLen = 5;
        int maxKvLen = 20;
        int decodeSteps = 10;

        // Prefill
        INDArray prefillQ = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, B, prefillLen, HIDDEN).muli(0.1);

        INDArray[] prefillResult = Nd4j.exec(new OnnxMultiHeadAttention(
                prefillQ, prefillK, prefillV, null, null, null,
                NUM_HEADS, 0.0, false));

        // Pad to maxKvLen
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all()).assign(prefillResult[1]);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all()).assign(prefillResult[2]);

        long cachePos = prefillLen;
        double initialKvMax = prefillResult[1].amaxNumber().doubleValue();

        // Decode with FULL padded buffer as past KV (simulates DSP/frozen path)
        for (int step = 0; step < decodeSteps; step++) {
            INDArray newQ = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);
            INDArray newK = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);
            INDArray newV = Nd4j.randn(DataType.FLOAT, B, 1, HIDDEN).muli(0.1);

            // Build attention bias: mask out future + unused positions
            // [B, numHeads, 1, maxKvLen+1] — MHA doesn't broadcast head dim
            long totalKv = maxKvLen + 1; // concat will make this
            INDArray attnBias = Nd4j.zeros(DataType.FLOAT, B, NUM_HEADS, 1, totalKv);
            // Mask positions after cachePos (zero padding) and the concat position
            // For positions cachePos+1 to maxKvLen-1, fill with -3.4e38
            float maskFill = -3.4028235e+38f;
            for (long p = cachePos + 1; p < maxKvLen; p++) {
                for (int h = 0; h < NUM_HEADS; h++) {
                    attnBias.putScalar(new long[]{0, h, 0, p}, maskFill);
                }
            }
            // Last position (the concat'd new token at index maxKvLen) should be visible
            // so leave it as 0

            INDArray[] decodeResult = Nd4j.exec(new OnnxMultiHeadAttention(
                    newQ, newK, newV, attnBias, staticKey, staticValue,
                    NUM_HEADS, 0.0, false));

            INDArray decodePresentKey = decodeResult[1]; // [B, numHeads, maxKvLen+1, headDim]
            assertEquals(maxKvLen + 1, decodePresentKey.size(2),
                    "Padded present key should be maxKvLen + 1");

            // Scatter: write last position into static buffer at cachePos
            Nd4j.exec(new KvScatter(staticKey, decodePresentKey, cachePos));
            Nd4j.exec(new KvScatter(staticValue, decodeResult[2], cachePos));

            double kvMax = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).amaxNumber().doubleValue();

            System.out.printf("Padded step %d: cachePos=%d, newKvMax=%.4f%n", step, cachePos, kvMax);

            // KV values should remain bounded
            assertTrue(kvMax < initialKvMax * 10 + 1.0,
                    "Padded mode: KV values growing at step " + step +
                            ": kvMax=" + kvMax + " vs initial=" + initialKvMax);

            cachePos++;
        }
    }

    /**
     * Test that MHA without any past KV produces valid output.
     */
    @Test
    public void testMhaBasicNoPast() {
        int B = 1, seqLen = 5;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(
                q, k, v, null, null, null,
                NUM_HEADS, 0.0, false));

        assertEquals(3, result.length);
        assertArrayEquals(new long[]{B, seqLen, HIDDEN}, result[0].shape(), "Output shape");
        assertArrayEquals(new long[]{B, NUM_HEADS, seqLen, HEAD_DIM}, result[1].shape(), "Present key shape");
        assertArrayEquals(new long[]{B, NUM_HEADS, seqLen, HEAD_DIM}, result[2].shape(), "Present value shape");

        assertFalse(result[0].isNaN().any(), "Output NaN");
        assertFalse(result[0].isInfinite().any(), "Output Inf");
    }

    /**
     * Test with causal masking enabled.
     */
    @Test
    public void testMhaCausalMask() {
        int B = 1, seqLen = 4;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, seqLen, HIDDEN).muli(0.1);

        // Without causal
        INDArray[] noCausal = Nd4j.exec(new OnnxMultiHeadAttention(
                q, k, v, null, null, null,
                NUM_HEADS, 0.0, false));

        // With causal
        INDArray[] withCausal = Nd4j.exec(new OnnxMultiHeadAttention(
                q, k, v, null, null, null,
                NUM_HEADS, 0.0, true));

        // Results should differ (causal mask restricts attention)
        double diff = noCausal[0].sub(withCausal[0]).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6,
                "Causal mask should change output, but diff was " + diff);

        assertFalse(withCausal[0].isNaN().any());
    }

    /**
     * Test GQA (Grouped Query Attention) where K/V have fewer heads than Q.
     */
    @Test
    public void testMhaGqa() {
        int B = 1, seqLen = 4;
        int numQHeads = 8;
        int numKvHeads = 2; // GQA: 8 Q heads, 2 KV heads (4:1 ratio)
        int qHidden = numQHeads * HEAD_DIM;
        int kvHidden = numKvHeads * HEAD_DIM;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, seqLen, qHidden).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, seqLen, kvHidden).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, seqLen, kvHidden).muli(0.1);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(
                q, k, v, null, null, null,
                numQHeads, 0.0, false));

        assertArrayEquals(new long[]{B, seqLen, qHidden}, result[0].shape(), "GQA output shape");
        assertArrayEquals(new long[]{B, numKvHeads, seqLen, HEAD_DIM}, result[1].shape(), "GQA present key shape");

        assertFalse(result[0].isNaN().any());
    }

    @Test
    public void testInPlaceGqaUsesLogicalCachePrefix() {
        int batch = 1;
        int qHeads = 9;
        int kvHeads = 3;
        int headDim = 64;
        int qHidden = qHeads * headDim;
        int kvHidden = kvHeads * headDim;
        int maxKvLen = 64;
        int cachePosition = 11;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        INDArray currentKey = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden).muli(0.1);
        INDArray currentValue = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden).muli(0.1);
        INDArray fullKey = Nd4j.zeros(DataType.FLOAT, batch, kvHeads, maxKvLen, headDim);
        INDArray fullValue = Nd4j.zeros(DataType.FLOAT, batch, kvHeads, maxKvLen, headDim);
        fullKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all())
                .assign(Nd4j.randn(DataType.FLOAT, batch, kvHeads, cachePosition, headDim).muli(0.1));
        fullValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all())
                .assign(Nd4j.randn(DataType.FLOAT, batch, kvHeads, cachePosition, headDim).muli(0.1));

        INDArray pastKey = fullKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all()).dup();
        INDArray pastValue = fullValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all()).dup();
        INDArray reference = Nd4j.exec(new OnnxMultiHeadAttention(
                query, currentKey, currentValue, null, pastKey, pastValue,
                qHeads, 0.0, false))[0].dup();

        INDArray bias = Nd4j.valueArrayOf(new long[]{batch, 1, 1, maxKvLen}, -65504.0f);
        bias.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePosition + 1)).assign(0.0f);
        INDArray position = Nd4j.scalar(DataType.INT64, cachePosition);
        INDArray actual = Nd4j.exec(new OnnxMultiHeadAttention(
                query, currentKey, currentValue, bias, fullKey, fullValue, position,
                qHeads, 0.0, false))[0];

        assertEquals(0.0, reference.sub(actual).amaxNumber().doubleValue(), 1e-4,
                "Padded in-place GQA must match attention over the active prefix");
        INDArray writtenKey = fullKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePosition), NDArrayIndex.all());
        INDArray expectedKey = currentKey.reshape(batch, 1, kvHeads, headDim)
                .permute(0, 2, 1, 3)
                .get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());
        assertEquals(0.0, writtenKey.sub(expectedKey).amaxNumber().doubleValue(), 1e-6,
                "Current key must be written at cache_position");
    }
}
