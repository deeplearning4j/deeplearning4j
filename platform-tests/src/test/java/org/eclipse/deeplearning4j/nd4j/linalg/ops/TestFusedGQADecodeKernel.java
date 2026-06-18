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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for the fusedGQADecodeKernel in FlashAttentionHelper.cu.
 *
 * The GQA decode kernel is dispatched when:
 * - supportedType (FLOAT32/FLOAT64/HALF)
 * - !noGQA (numQHeads != numKvHeads)
 * - isDecode (seqQ == 1)
 * - !needScores && !needLogits
 *
 * These tests verify correctness against a reference computation by comparing
 * the kernel output with manually-computed attention for known inputs.
 *
 * Key configurations tested:
 * - SmolDocling shapes: 9 Q heads, 3 KV heads, headDim=64
 * - Qwen3.5 shapes: 14 Q heads, 2 KV heads, headDim=128
 * - Non-contiguous KV (BHSD→BSHD permuted views) — the production path
 * - Attention bias with masked positions (large negative values)
 * - Full-buffer iteration with sparse valid positions (decode after prefill)
 */
public class TestFusedGQADecodeKernel {

    /**
     * Test GQA decode with SmolDocling-like shapes and contiguous KV.
     * batch=1, numQHeads=9, numKvHeads=3, headDim=64, seqKV=10
     *
     * Uses small seqKV so we can manually verify the attention computation.
     */
    @Test
    @DisplayName("GQA decode - SmolDocling shapes, contiguous KV, no bias")
    public void testGqaDecodeSmolDoclingContiguousNoBias() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64, seqKV = 10;
        int qHidden = numQHeads * headDim;  // 576
        int kvHidden = numKvHeads * headDim; // 192

        // Create inputs in the shape onnx_multi_head_attention expects: [batch, seq, hidden]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);

        // Run via onnx_multi_head_attention (dispatches to fusedGQADecodeKernel on CUDA)
        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, null, null, null,
                numQHeads, 0.0, false));
        INDArray attnOutput = result[0]; // [batch, 1, qHidden]

        // Reference computation: manual GQA attention
        INDArray reference = computeReferenceGQA(query, key, value, null,
                batch, numQHeads, numKvHeads, headDim, seqKV);

        // Compare
        assertArrayEquals(new long[]{batch, 1, qHidden}, attnOutput.shape());
        double maxDiff = attnOutput.sub(reference).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "GQA decode output differs from reference by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Test GQA decode with non-contiguous KV (BHSD→BSHD permuted view).
     * This is the PRODUCTION path for in-place KV write mode in SmolDocling.
     *
     * Compares: running onnx_mha with past_key (triggers BHSD→BSHD permute internally)
     * vs: running onnx_mha with the same data as contiguous KV (no past_key).
     *
     * Both should produce the same attention output since the underlying data is identical.
     */
    @Test
    @DisplayName("GQA decode - non-contiguous BHSD→BSHD KV view vs contiguous")
    public void testGqaDecodeNonContiguousKV() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int pastSeq = 10;  // Past KV positions
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        // Create query: [batch, 1, qHidden]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);

        // Create KV data: [batch, pastSeq, kvHidden] — the actual key/value data
        INDArray pastKvDataBSHD = Nd4j.randn(DataType.FLOAT, batch, pastSeq, kvHidden).muli(0.1);
        INDArray pastVvDataBSHD = Nd4j.randn(DataType.FLOAT, batch, pastSeq, kvHidden).muli(0.1);
        INDArray currentKey = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden).muli(0.1);
        INDArray currentValue = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden).muli(0.1);

        // Path 1: contiguous KV (concatenate past + current as flat BSHD array)
        INDArray fullKey = Nd4j.concat(1, pastKvDataBSHD, currentKey);    // [batch, pastSeq+1, kvHidden]
        INDArray fullValue = Nd4j.concat(1, pastVvDataBSHD, currentValue); // [batch, pastSeq+1, kvHidden]

        INDArray[] contiguousResult = Nd4j.exec(new OnnxMultiHeadAttention(query, fullKey, fullValue,
                null, null, null, numQHeads, 0.0, false));
        INDArray contiguousOutput = contiguousResult[0];

        // Path 2: past KV in BHSD format (triggers internal permute to non-contiguous BSHD view)
        // pastKey expected as [batch, numKvHeads, pastSeq, headDim] (BHSD)
        INDArray pastKeyBHSD = pastKvDataBSHD.reshape(batch, pastSeq, numKvHeads, headDim)
                .permute(0, 2, 1, 3).dup();  // dup to ensure contiguous BHSD
        INDArray pastValueBHSD = pastVvDataBSHD.reshape(batch, pastSeq, numKvHeads, headDim)
                .permute(0, 2, 1, 3).dup();

        INDArray[] pastResult = Nd4j.exec(new OnnxMultiHeadAttention(query, currentKey, currentValue,
                null, pastKeyBHSD, pastValueBHSD, numQHeads, 0.0, false));
        INDArray pastOutput = pastResult[0];

        // Both paths should produce the same attention output
        assertArrayEquals(new long[]{batch, 1, qHidden}, pastOutput.shape());
        double maxDiff = pastOutput.sub(contiguousOutput).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Non-contiguous KV path output differs from contiguous path by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Test GQA decode with attention bias (masked positions).
     * Simulates the decode-after-prefill scenario where most of the KV buffer is masked.
     */
    @Test
    @DisplayName("GQA decode - SmolDocling shapes with attention bias masking")
    public void testGqaDecodeWithBiasMasking() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int totalSeqKV = 100;  // Total KV positions in buffer
        int validPositions = 20;  // Only 20 tokens are valid (rest masked)
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        // Full KV: [batch, totalSeqKV, kvHidden] — positions beyond valid are random noise
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, totalSeqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, totalSeqKV, kvHidden).muli(0.1);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);

        // Bias: 0 for valid positions [0..validPositions-1], -65504 for masked [validPositions..totalSeqKV-1]
        float[] biasData = new float[totalSeqKV];
        for (int i = validPositions; i < totalSeqKV; i++) {
            biasData[i] = -65504.0f;
        }
        INDArray attnBias = Nd4j.create(biasData, new long[]{1, 1, 1, totalSeqKV});

        // Run with full buffer + bias masking
        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, attnBias,
                null, null, numQHeads, 0.0, false));
        INDArray attnOutput = result[0];

        // Reference: compute with ONLY the valid positions (no masking needed)
        INDArray keyValid = key.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all());
        INDArray valueValid = value.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all());
        INDArray reference = computeReferenceGQA(query, keyValid, valueValid, null,
                batch, numQHeads, numKvHeads, headDim, validPositions);

        assertArrayEquals(new long[]{batch, 1, qHidden}, attnOutput.shape());
        double maxDiff = attnOutput.sub(reference).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Masked GQA decode output differs from unmasked reference by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Test GQA decode with FP16 inputs — SmolDocling uses HALF precision.
     * Verifies that the kernel handles FP16 accumulation correctly.
     */
    @Test
    @DisplayName("GQA decode - FP16 with attention bias")
    public void testGqaDecodeFP16WithBias() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int totalSeqKV = 50;
        int validPositions = 15;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        // Create FP16 inputs
        INDArray query = Nd4j.randn(DataType.HALF, batch, 1, qHidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.HALF, batch, totalSeqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.HALF, batch, totalSeqKV, kvHidden).muli(0.1);

        // Bias in HALF (this is what SmolDocling does)
        float[] biasData = new float[totalSeqKV];
        for (int i = validPositions; i < totalSeqKV; i++) {
            biasData[i] = -65504.0f;
        }
        INDArray attnBias = Nd4j.create(biasData, new long[]{1, 1, 1, totalSeqKV}).castTo(DataType.HALF);

        // Run with FP16
        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, attnBias,
                null, null, numQHeads, 0.0, false));
        INDArray attnOutput = result[0];

        // Reference: same computation in FP32 for numerical accuracy
        INDArray queryF32 = query.castTo(DataType.FLOAT);
        INDArray keyValidF32 = key.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all())
                .castTo(DataType.FLOAT);
        INDArray valueValidF32 = value.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all())
                .castTo(DataType.FLOAT);
        INDArray reference = computeReferenceGQA(queryF32, keyValidF32, valueValidF32, null,
                batch, numQHeads, numKvHeads, headDim, validPositions);
        INDArray referenceHalf = reference.castTo(DataType.HALF);

        assertArrayEquals(new long[]{batch, 1, qHidden}, attnOutput.shape());
        // FP16 has lower precision — allow larger tolerance
        double maxDiff = attnOutput.castTo(DataType.FLOAT).sub(referenceHalf.castTo(DataType.FLOAT))
                .amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.05,
                "FP16 GQA decode output differs from reference by " + maxDiff + " (threshold 0.05)");
    }

    /**
     * Test GQA decode matches MHA (non-GQA) when numQHeads == numKvHeads.
     * When headsPerKvHead == 1, the GQA kernel is NOT dispatched (noGQA=true),
     * so this verifies the fallback path. Useful as a sanity check.
     */
    @Test
    @DisplayName("Non-GQA decode - same path as reference commit (cuBLAS fallback)")
    public void testNonGqaDecodeFallback() {
        int batch = 1, numHeads = 4, headDim = 64, seqKV = 20;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden).muli(0.1);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, null, null, null,
                numHeads, 0.0, false));
        INDArray attnOutput = result[0];

        // Reference computation (same as GQA with headsPerKvHead=1)
        INDArray reference = computeReferenceGQA(query, key, value, null,
                batch, numHeads, numHeads, headDim, seqKV);

        assertArrayEquals(new long[]{batch, 1, hidden}, attnOutput.shape());
        double maxDiff = attnOutput.sub(reference).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
                "Non-GQA decode output differs from reference by " + maxDiff + " (threshold 1e-4)");
    }

    /**
     * Test GQA decode with Qwen3.5 shapes: 14 Q heads, 2 KV heads, headDim=128.
     */
    @Test
    @DisplayName("GQA decode - Qwen3.5 shapes (14 Q heads, 2 KV heads, dim=128)")
    public void testGqaDecodeQwenShapes() {
        int batch = 1, numQHeads = 14, numKvHeads = 2, headDim = 128, seqKV = 30;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, null, null, null,
                numQHeads, 0.0, false));
        INDArray attnOutput = result[0];

        INDArray reference = computeReferenceGQA(query, key, value, null,
                batch, numQHeads, numKvHeads, headDim, seqKV);

        assertArrayEquals(new long[]{batch, 1, qHidden}, attnOutput.shape());
        double maxDiff = attnOutput.sub(reference).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Qwen GQA decode output differs from reference by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Stress test: large seqKV with many masked positions (mimics SmolDocling production).
     * maxSeqLen ~1400, only 27 positions valid after prefill.
     */
    @Test
    @DisplayName("GQA decode - production-scale buffer (1400 seqKV, 27 valid)")
    public void testGqaDecodeProductionScale() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int maxSeqLen = 1400;
        int validPositions = 27;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        // Full buffer — noise everywhere (simulates stale/uninitialized data beyond valid range)
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, maxSeqLen, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, maxSeqLen, kvHidden).muli(0.1);

        // Mask: 0 for valid, -65504 for invalid
        float[] biasData = new float[maxSeqLen];
        for (int i = validPositions; i < maxSeqLen; i++) {
            biasData[i] = -65504.0f;
        }
        INDArray attnBias = Nd4j.create(biasData, new long[]{1, 1, 1, maxSeqLen});

        // Full buffer execution with masking
        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, attnBias,
                null, null, numQHeads, 0.0, false));
        INDArray attnOutput = result[0];

        // Reference: only valid positions (no masking needed)
        INDArray keyValid = key.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all());
        INDArray valueValid = value.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validPositions), NDArrayIndex.all());
        INDArray reference = computeReferenceGQA(query, keyValid, valueValid, null,
                batch, numQHeads, numKvHeads, headDim, validPositions);

        assertArrayEquals(new long[]{batch, 1, qHidden}, attnOutput.shape());
        double maxDiff = attnOutput.sub(reference).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Production-scale GQA decode output differs from reference by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Test that GQA decode with full-buffer + bias masking produces the same output
     * as GQA decode with only the valid positions (no masking needed).
     *
     * This tests the core invariant: masking positions with -65504 bias should make
     * them contribute zero to the attention output, equivalent to not having them.
     *
     * Uses only the no-past-KV path (direct K/V input) to avoid concat path differences.
     */
    @Test
    @DisplayName("GQA decode - full buffer + masking equals valid-only (no past KV)")
    public void testGqaDecodeFullBufferMaskingEqualsValidOnly() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int maxSeqLen = 100;
        int validLen = 21;  // Tokens 0..20 are valid
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);

        // Full buffer: [batch, maxSeqLen, kvHidden] — random data everywhere
        INDArray keyFull = Nd4j.randn(DataType.FLOAT, batch, maxSeqLen, kvHidden).muli(0.1);
        INDArray valueFull = Nd4j.randn(DataType.FLOAT, batch, maxSeqLen, kvHidden).muli(0.1);

        // Valid-only slice: [batch, validLen, kvHidden]
        INDArray keyValid = keyFull.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validLen), NDArrayIndex.all()).dup();
        INDArray valueValid = valueFull.get(NDArrayIndex.all(), NDArrayIndex.interval(0, validLen), NDArrayIndex.all()).dup();

        // Path 1: valid only (no masking needed)
        INDArray[] validResult = Nd4j.exec(new OnnxMultiHeadAttention(query, keyValid, valueValid,
                null, null, null, numQHeads, 0.0, false));
        INDArray validOutput = validResult[0];

        // Path 2: full buffer with bias masking
        float[] biasData = new float[maxSeqLen];
        for (int i = validLen; i < maxSeqLen; i++) {
            biasData[i] = -65504.0f;
        }
        INDArray attnBias = Nd4j.create(biasData, new long[]{1, 1, 1, maxSeqLen});

        INDArray[] maskedResult = Nd4j.exec(new OnnxMultiHeadAttention(query, keyFull, valueFull,
                attnBias, null, null, numQHeads, 0.0, false));
        INDArray maskedOutput = maskedResult[0];

        // Both should produce the same output
        assertArrayEquals(new long[]{batch, 1, qHidden}, maskedOutput.shape());
        double maxDiff = maskedOutput.sub(validOutput).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Full-buffer+masked output differs from valid-only by " + maxDiff + " (threshold 1e-3)");
    }

    /**
     * Test basic output sanity: GQA decode should NOT produce all-zeros,
     * should NOT produce NaN/Inf, and token probabilities should be reasonable.
     */
    @Test
    @DisplayName("GQA decode - sanity checks (no NaN, no zeros, bounded output)")
    public void testGqaDecodeSanity() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64, seqKV = 30;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);

        INDArray[] result = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, null, null, null,
                numQHeads, 0.0, false));
        INDArray attnOutput = result[0];

        // Sync to host for checks
        float[] outputData = attnOutput.data().asFloat();

        // No NaN or Inf
        for (int i = 0; i < outputData.length; i++) {
            assertFalse(Float.isNaN(outputData[i]), "Output contains NaN at index " + i);
            assertFalse(Float.isInfinite(outputData[i]), "Output contains Inf at index " + i);
        }

        // Not all zeros
        double maxAbs = attnOutput.amaxNumber().doubleValue();
        assertTrue(maxAbs > 1e-6, "Output is all zeros (max abs = " + maxAbs + ")");

        // Output should be bounded (attention output is a weighted average of V values)
        double valueMax = value.amaxNumber().doubleValue();
        assertTrue(maxAbs <= valueMax * 2.0,
                "Output max (" + maxAbs + ") exceeds 2x max value (" + valueMax + ") — suspicious");
    }

    /**
     * Test multi-step decode: verify that output changes between steps
     * (i.e., the kernel is actually reading updated KV cache positions).
     */
    @Test
    @DisplayName("GQA decode - multi-step produces different outputs per step")
    public void testGqaDecodeMultiStep() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int seqKV = 10;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden).muli(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, kvHidden).muli(0.1);

        // Step 1: attend over seqKV positions
        INDArray[] result1 = Nd4j.exec(new OnnxMultiHeadAttention(query, key, value, null, null, null,
                numQHeads, 0.0, false));

        // Step 2: add one more KV position (different data)
        INDArray newKV = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden).muli(0.5); // larger magnitude
        INDArray keyExtended = Nd4j.concat(1, key, newKV);
        INDArray valueExtended = Nd4j.concat(1, value, newKV);

        INDArray[] result2 = Nd4j.exec(new OnnxMultiHeadAttention(query, keyExtended, valueExtended,
                null, null, null, numQHeads, 0.0, false));

        // Outputs should differ
        double diff = result1[0].sub(result2[0]).amaxNumber().doubleValue();
        assertTrue(diff > 1e-4, "Multi-step outputs are identical (diff=" + diff + ") — kernel may not read new KV");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Reference implementation: manual GQA attention computation
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Computes reference GQA attention output using explicit Java loops.
     * This is the ground truth — no CUDA kernel involved.
     *
     * Q: [batch, 1, qHidden] → reshape to [batch, 1, numQHeads, headDim]
     * K: [batch, seqKV, kvHidden] → reshape to [batch, seqKV, numKvHeads, headDim]
     * V: same as K
     *
     * For each query head h, the corresponding KV head is h / headsPerKvHead.
     * score[h, k] = Q[h, :] · K[kvHead, k, :] * scale
     * attn[h, :] = softmax(score + bias) @ V[kvHead, :, :]
     */
    private INDArray computeReferenceGQA(INDArray query, INDArray key, INDArray value,
                                          INDArray bias,
                                          int batch, int numQHeads, int numKvHeads,
                                          int headDim, int seqKV) {
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;
        int headsPerKvHead = numQHeads / numKvHeads;
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // Reshape to 4D: [batch, seq, heads, dim]
        INDArray qReshaped = query.reshape(batch, 1, numQHeads, headDim);
        INDArray kReshaped = key.reshape(batch, seqKV, numKvHeads, headDim);
        INDArray vReshaped = value.reshape(batch, seqKV, numKvHeads, headDim);

        // Output: [batch, 1, numQHeads, headDim]
        INDArray output = Nd4j.zeros(DataType.FLOAT, batch, 1, numQHeads, headDim);

        for (int b = 0; b < batch; b++) {
            for (int qHead = 0; qHead < numQHeads; qHead++) {
                int kvHead = qHead / headsPerKvHead;

                // Compute scores: Q[b, 0, qHead, :] · K[b, k, kvHead, :]^T for all k
                float[] scores = new float[seqKV];
                for (int k = 0; k < seqKV; k++) {
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; d++) {
                        float q = qReshaped.getFloat(b, 0, qHead, d);
                        float kv = kReshaped.getFloat(b, k, kvHead, d);
                        dot += q * kv;
                    }
                    scores[k] = dot * scale;

                    // Add bias if present
                    if (bias != null) {
                        // Bias may be broadcast — handle common shapes
                        long[] biasShape = bias.shape();
                        float biasVal;
                        if (biasShape.length == 4) {
                            // [batch, heads, seqQ, seqKV] or broadcastable variant
                            int bIdx = (int) Math.min(b, biasShape[0] - 1);
                            int hIdx = (int) Math.min(qHead, biasShape[1] - 1);
                            int sIdx = (int) Math.min(0, biasShape[2] - 1);
                            int kIdx = (int) Math.min(k, biasShape[3] - 1);
                            biasVal = bias.getFloat(bIdx, hIdx, sIdx, kIdx);
                        } else {
                            biasVal = 0.0f;
                        }
                        scores[k] += biasVal;
                    }
                }

                // Softmax
                float maxScore = Float.NEGATIVE_INFINITY;
                for (float s : scores) maxScore = Math.max(maxScore, s);
                float sumExp = 0.0f;
                float[] expScores = new float[seqKV];
                for (int k = 0; k < seqKV; k++) {
                    expScores[k] = (float) Math.exp(scores[k] - maxScore);
                    sumExp += expScores[k];
                }
                for (int k = 0; k < seqKV; k++) {
                    expScores[k] /= sumExp;
                }

                // Weighted sum of V: attn_weights @ V
                for (int d = 0; d < headDim; d++) {
                    float acc = 0.0f;
                    for (int k = 0; k < seqKV; k++) {
                        acc += expScores[k] * vReshaped.getFloat(b, k, kvHead, d);
                    }
                    output.putScalar(new int[]{b, 0, qHead, d}, acc);
                }
            }
        }

        // Reshape back to [batch, 1, qHidden]
        return output.reshape(batch, 1, qHidden);
    }
}
