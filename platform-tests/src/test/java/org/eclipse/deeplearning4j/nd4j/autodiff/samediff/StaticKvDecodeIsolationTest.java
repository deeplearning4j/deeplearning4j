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
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for the static KV cache decode path in DecoderUtils.
 *
 * These tests verify that buildDecoderInputMap produces correct inputs for
 * view-based mode (dspActive=false), and that the static KV buffer + scatter
 * pattern produces equivalent results to passing full KV outputs directly.
 *
 * Motivation: TestVLMDecodeQuality#testManualMultiStepDecode (which passes full
 * KV outputs directly) produces correct output, but StaticKvCacheDecodeLoop
 * (which uses static KV buffers + views + scatter) produces "UserT" garbage.
 * These tests isolate the exact point of divergence.
 */
@Slf4j
@DisplayName("StaticKvDecodeIsolationTest")
public class StaticKvDecodeIsolationTest {

    /**
     * Test 1: buildDecoderInputMap with dspActive=false produces correct
     * view-based inputs — attention_mask is all-ones, KV is view [0:cachePos].
     */
    @Test
    @DisplayName("testBuildDecoderInputMap_viewBased")
    public void testBuildDecoderInputMap_viewBased() {
        int numHeads = 3;
        int headDim = 64;
        int maxKvLen = 100;
        int cachePos = 10;
        long pastSeqLen = 679; // prefill length
        long currentSeqLen = 1;

        // Simulate static KV buffers after prefill
        Map<String, INDArray> staticKvBuffers = new HashMap<>();
        staticKvBuffers.put("past_key_values.0.key",
                Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen, headDim));
        staticKvBuffers.put("past_key_values.0.value",
                Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen, headDim));

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", "past_key_values.0.key", "past_key_values.0.value");

        SameDiff dummyDecoder = SameDiff.create();

        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, 576);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                true, 576, null, false); // dspActive=false

        // attention_mask: should be all-ones [1, cachePos + 1]
        INDArray mask = result.get("attention_mask");
        assertNotNull(mask, "attention_mask should be present");
        assertArrayEquals(new long[]{1, cachePos + 1}, mask.shape(),
                "attention_mask shape should be [1, cachePos+currentSeqLen]");
        assertEquals(cachePos + 1, mask.sumNumber().longValue(),
                "attention_mask should be all ones");

        // past_key_values: should be view [0:cachePos], shape [1,3,cachePos,64]
        INDArray kv = result.get("past_key_values.0.key");
        assertNotNull(kv, "past_key should be present");
        assertArrayEquals(new long[]{1, numHeads, cachePos, headDim}, kv.shape(),
                "past_key should be view [0:cachePos]");

        // position_ids: should be [pastSeqLen]
        INDArray posIds = result.get("position_ids");
        assertEquals(pastSeqLen, posIds.getLong(0, 0),
                "position_ids should be pastSeqLen");

        log.info("View-based buildDecoderInputMap: mask shape={}, kv shape={}, pos={}",
                Arrays.toString(mask.shape()), Arrays.toString(kv.shape()), posIds.getLong(0, 0));

        dummyDecoder.close();
    }

    /**
     * Test 2: buildDecoderInputMap with dspActive=true produces correct
     * padded inputs — attention_mask is sparse, KV is full static buffer.
     */
    @Test
    @DisplayName("testBuildDecoderInputMap_padded")
    public void testBuildDecoderInputMap_padded() {
        int numHeads = 3;
        int headDim = 64;
        int maxKvLen = 100;
        int cachePos = 10;
        long pastSeqLen = 679;
        long currentSeqLen = 1;

        Map<String, INDArray> staticKvBuffers = new HashMap<>();
        staticKvBuffers.put("past_key_values.0.key",
                Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen, headDim));
        staticKvBuffers.put("past_key_values.0.value",
                Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen, headDim));

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", "past_key_values.0.key", "past_key_values.0.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, 576);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                true, 576, null, true); // dspActive=true

        // attention_mask: should be sparse [1, maxKvLen + 1]
        INDArray mask = result.get("attention_mask");
        assertNotNull(mask, "attention_mask should be present");
        assertArrayEquals(new long[]{1, maxKvLen + 1}, mask.shape(),
                "attention_mask shape should be [1, maxKvLen+currentSeqLen]");
        // Should have cachePos ones (past positions) + 1 (current token) = cachePos+1
        assertEquals(cachePos + 1, mask.sumNumber().longValue(),
                "attention_mask should have cachePos+1 ones (past + current)");
        // Last position should be 1 (current token)
        assertEquals(1, mask.getLong(0, maxKvLen),
                "Last position (current token) should be 1");
        // Padding positions should be 0
        if (cachePos < maxKvLen) {
            assertEquals(0, mask.getLong(0, cachePos),
                    "First padding position should be 0");
        }

        // past_key_values: should be full buffer [1,3,maxKvLen,64]
        INDArray kv = result.get("past_key_values.0.key");
        assertNotNull(kv, "past_key should be present");
        assertArrayEquals(new long[]{1, numHeads, maxKvLen, headDim}, kv.shape(),
                "past_key should be full static buffer");

        log.info("Padded buildDecoderInputMap: mask shape={}, kv shape={}, mask sum={}",
                Arrays.toString(mask.shape()), Arrays.toString(kv.shape()), mask.sumNumber());

        dummyDecoder.close();
    }

    /**
     * Test 3: KV scatter correctness — verify that scatterNewKvEntries
     * correctly copies the new token's KV from present output into the static buffer.
     */
    @Test
    @DisplayName("testKvScatterCorrectness")
    public void testKvScatterCorrectness() {
        int numHeads = 3;
        int headDim = 64;
        int maxKvLen = 20;

        // Static buffers with known data at positions 0..4
        INDArray staticKeyBuf = Nd4j.zeros(DataType.FLOAT, 1, numHeads, maxKvLen, headDim);
        INDArray staticValBuf = Nd4j.zeros(DataType.FLOAT, 1, numHeads, maxKvLen, headDim);

        // Fill positions 0..4 with known values
        for (int pos = 0; pos < 5; pos++) {
            staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all()).assign(pos + 1.0f);
            staticValBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all()).assign((pos + 1.0f) * 10);
        }

        Map<String, INDArray> staticKvBuffers = new HashMap<>();
        staticKvBuffers.put("past_key_values.0.key", staticKeyBuf);
        staticKvBuffers.put("past_key_values.0.value", staticValBuf);

        // Simulate decoder output: present_key = past + new entry
        // present shape = [1, numHeads, cachePos+1, headDim] (concat of past view + new)
        int cachePos = 5;
        INDArray presentKey = Nd4j.zeros(DataType.FLOAT, 1, numHeads, cachePos + 1, headDim);
        INDArray presentValue = Nd4j.zeros(DataType.FLOAT, 1, numHeads, cachePos + 1, headDim);

        // Copy past into present (positions 0..cachePos-1)
        for (int pos = 0; pos < cachePos; pos++) {
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all()).assign(pos + 1.0f);
            presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all()).assign((pos + 1.0f) * 10);
        }
        // New entry at last position
        float newKeyVal = 99.0f;
        float newValVal = 990.0f;
        presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newKeyVal);
        presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newValVal);

        Map<String, INDArray> decoderOutputs = new HashMap<>();
        decoderOutputs.put("present.0.key", presentKey);
        decoderOutputs.put("present.0.value", presentValue);

        List<String> keyNames = List.of("present.0.key");
        List<String> valueNames = List.of("present.0.value");

        // Scatter
        DecoderUtils.scatterNewKvEntries(staticKvBuffers, decoderOutputs,
                keyNames, valueNames, maxKvLen, cachePos);

        // Verify: position 5 in static buffer should now have newKeyVal
        INDArray scatteredKey = staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all());
        float actualKeyVal = scatteredKey.getFloat(0, 0, 0);
        assertEquals(newKeyVal, actualKeyVal, 1e-6f,
                "Scattered key value should match new entry");

        INDArray scatteredVal = staticValBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all());
        float actualValVal = scatteredVal.getFloat(0, 0, 0);
        assertEquals(newValVal, actualValVal, 1e-6f,
                "Scattered value should match new entry");

        // Verify previous positions unchanged
        for (int pos = 0; pos < cachePos; pos++) {
            float keyVal = staticKeyBuf.getFloat(0, 0, pos, 0);
            assertEquals(pos + 1.0f, keyVal, 1e-6f,
                    "Position " + pos + " key should be unchanged after scatter");
        }

        // Verify positions after cachePos+1 are still zero
        float afterKeyVal = staticKeyBuf.getFloat(0, 0, cachePos + 1, 0);
        assertEquals(0.0f, afterKeyVal, 1e-6f,
                "Position after scatter should still be zero");

        log.info("KV scatter test passed: new entry at pos {} correctly written", cachePos);
    }

    /**
     * Test 4: View-based KV gives same attention output as full KV.
     * This isolates whether the view [0:cachePos] of a padded static buffer
     * produces the same result as a freshly-created [0:cachePos] tensor.
     */
    @Test
    @DisplayName("testViewVsFullKvEquivalence")
    public void testViewVsFullKvEquivalence() {
        int batch = 1, numHeads = 3, headDim = 64, hidden = numHeads * headDim;
        int cachePos = 5;
        int maxKvLen = 20;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);

        // Create "real" KV data for positions 0..cachePos-1
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, cachePos, headDim).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, cachePos, headDim).mul(0.1);

        // Option A: pass realPastKey directly (what testManualMultiStepDecode does)
        int totalSeq = cachePos + 1;
        INDArray biasA = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
        INDArray attnA = Nd4j.create(DataType.FLOAT, batch, 1, hidden);
        INDArray pkA = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
        INDArray pvA = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasA, realPastKey, realPastValue)
                .addOutputs(attnA, pkA, pvA)
                .addIntegerArguments(numHeads, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Option B: embed in a padded buffer, take view (what StaticKvCacheDecodeLoop does)
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()).assign(realPastKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()).assign(realPastValue);

        INDArray viewKey = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
        INDArray viewValue = staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());

        INDArray biasB = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
        INDArray attnB = Nd4j.create(DataType.FLOAT, batch, 1, hidden);
        INDArray pkB = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
        INDArray pvB = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasB, viewKey, viewValue)
                .addOutputs(attnB, pkB, pvB)
                .addIntegerArguments(numHeads, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Compare
        double maxDiff = attnA.sub(attnB).amaxNumber().doubleValue();
        log.info("Full KV vs view KV attention output max diff: {}", maxDiff);
        assertTrue(maxDiff < 1e-5,
                "View-based KV should produce identical results to full KV. MaxDiff=" + maxDiff);
    }

    /**
     * Test 5: Multi-step decode with static KV buffer + scatter produces same
     * tokens as multi-step decode with full KV output passing.
     *
     * This is the key isolation test: it runs the SAME decode loop two ways
     * and compares token-by-token. If they diverge, the scatter or view is wrong.
     */
    @Test
    @DisplayName("testStaticKvVsFullKvMultiStep")
    public void testStaticKvVsFullKvMultiStep() {
        int batch = 1, numHeads = 2, headDim = 8, hidden = numHeads * headDim;
        int maxKvLen = 30;
        int numSteps = 10;

        // Pre-generate ALL random inputs so both paths use identical data
        INDArray[] queries = new INDArray[numSteps];
        INDArray[] keys = new INDArray[numSteps];
        INDArray[] values = new INDArray[numSteps];
        for (int i = 0; i < numSteps; i++) {
            queries[i] = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
            keys[i] = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
            values[i] = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        }

        // ===== PATH A: Full KV passing (like testManualMultiStepDecode) =====
        List<Integer> tokensA = new java.util.ArrayList<>();
        INDArray pastKeyA = null;
        INDArray pastValueA = null;

        for (int step = 0; step < numSteps; step++) {
            INDArray q = queries[step].dup();
            INDArray k = keys[step].dup();
            INDArray v = values[step].dup();

            int totalSeq = (pastKeyA != null ? (int) pastKeyA.size(2) : 0) + 1;
            INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
            INDArray attn = Nd4j.create(DataType.FLOAT, batch, 1, hidden);
            INDArray pk = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
            INDArray pv = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);

            var builder = org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                    .addOutputs(attn, pk, pv)
                    .addIntegerArguments(numHeads, 0)
                    .addFloatingPointArguments(0.0);
            if (pastKeyA != null) {
                builder.addInputs(q, k, v, bias, pastKeyA, pastValueA);
            } else {
                builder.addInputs(q, k, v);
            }
            Nd4j.exec(builder.build());

            // "Token" = argmax of attention output (simplified)
            int token = attn.reshape(hidden).argMax().getInt(0);
            tokensA.add(token);

            pastKeyA = pk.dup();
            pastValueA = pv.dup();
        }

        // ===== PATH B: Static KV buffer + scatter (like StaticKvCacheDecodeLoop) =====
        List<Integer> tokensB = new java.util.ArrayList<>();
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        int cachePos = 0;

        for (int step = 0; step < numSteps; step++) {
            INDArray q = queries[step].dup();
            INDArray k = keys[step].dup();
            INDArray v = values[step].dup();

            int totalSeq = cachePos + 1;
            INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
            INDArray attn = Nd4j.create(DataType.FLOAT, batch, 1, hidden);
            INDArray pk = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
            INDArray pv = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);

            var builder = org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                    .addOutputs(attn, pk, pv)
                    .addIntegerArguments(numHeads, 0)
                    .addFloatingPointArguments(0.0);
            if (cachePos > 0) {
                INDArray viewKey = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                INDArray viewValue = staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                builder.addInputs(q, k, v, bias, viewKey, viewValue);
            } else {
                builder.addInputs(q, k, v);
            }
            Nd4j.exec(builder.build());

            int token = attn.reshape(hidden).argMax().getInt(0);
            tokensB.add(token);

            // Scatter new entry into static buffer
            INDArray newKey = pk.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newKey);
            INDArray newVal = pv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newVal);
            cachePos++;
        }

        // Compare token sequences
        log.info("Path A (full KV): {}", tokensA);
        log.info("Path B (static+scatter): {}", tokensB);
        assertEquals(tokensA, tokensB,
                "Static KV + scatter should produce identical tokens to full KV passing");
    }

    /**
     * Test 6: Verify that the _causal_mask input produced by buildDecoderInputMap
     * is correct for decode steps (single token attending to all past).
     */
    @Test
    @DisplayName("testCausalMaskForDecodeStep")
    public void testCausalMaskForDecodeStep() {
        // Decode step: currentSeqLen=1, totalSeqLen varies
        for (int totalSeq : new int[]{5, 10, 50, 100}) {
            INDArray mask = DecoderUtils.buildCausalMask(1, totalSeq);
            assertArrayEquals(new long[]{1, 1, 1, totalSeq}, mask.shape(),
                    "Causal mask shape for decode step");
            // For seqLen=1, mask should be all zeros (single token attends to all past)
            assertEquals(0.0, mask.sumNumber().doubleValue(), 1e-6,
                    "Decode causal mask should be all zeros (attend to everything). totalSeq=" + totalSeq);
        }
    }
}
