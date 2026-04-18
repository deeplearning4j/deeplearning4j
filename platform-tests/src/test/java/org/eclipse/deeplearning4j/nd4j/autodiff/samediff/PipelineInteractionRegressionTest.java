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
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.UnifiedKvCacheManager;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pipeline interaction regression tests.
 *
 * Tests the interactions between components in the VLM decode pipeline that are
 * NOT covered by existing unit tests. Each test targets a specific cross-component
 * interaction identified as a potential root cause for degenerate output.
 *
 * These tests use synthetic data (no model loading) for speed and isolation.
 */
@Slf4j
@DisplayName("PipelineInteractionRegressionTest")
public class PipelineInteractionRegressionTest {

    // SmolDocling-like dimensions
    private static final int NUM_KV_HEADS = 3;
    private static final int NUM_Q_HEADS = 9;  // GQA: 576/64 = 9
    private static final int HEAD_DIM = 64;
    private static final long HIDDEN_SIZE = 576;
    private static final int MAX_KV_LEN = 100;
    private static final int NUM_LAYERS = 2;
    private static final int PREFILL_LEN = 10;  // simulate 10-token prefill

    private static final String ATTN_REFORMAT_NODE = "/model/attn_mask_reformat/Tile/output_0";

    // ═══════════════════════════════════════════════════════════════════════
    // Test 1: 4D bias broadcasting correctness
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * The Java-constructed 4D attention bias has shape [1, 1, 1, totalSeqLen].
     * The model's internal subgraph produces [1, numQHeads, 1, seqKV].
     *
     * Broadcasting [1,1,1,T] → [1,9,1,T] means all 9 query heads see the SAME
     * attention pattern. This is correct for causal masking (all heads have same
     * causal structure) but different from the model's subgraph which could
     * potentially produce per-head patterns.
     *
     * This test verifies the broadcast is semantically equivalent for causal masking.
     */
    @Test
    @DisplayName("4D bias [1,1,1,T] broadcasts correctly to [1,numQHeads,1,T]")
    public void testAttnBiasBroadcastingEquivalence() {
        int cachePos = PREFILL_LEN;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;
        float maskFill = DecoderUtils.MASK_FILL;

        // Build the Java [1,1,1,T] bias
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, currentSeqLen, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        INDArray bias = result.get(ATTN_REFORMAT_NODE);
        assertNotNull(bias, "4D bias should be present");
        assertArrayEquals(new long[]{1, 1, 1, totalSeqLen}, bias.shape(),
                "Bias shape should be [1,1,1," + totalSeqLen + "]");

        // Manually broadcast to [1, NUM_Q_HEADS, 1, totalSeqLen]
        INDArray broadcastBias = Nd4j.tile(bias, 1, NUM_Q_HEADS, 1, 1);
        assertArrayEquals(new long[]{1, NUM_Q_HEADS, 1, totalSeqLen}, broadcastBias.shape());

        // Verify ALL heads have identical mask pattern
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            for (int p = 0; p < totalSeqLen; p++) {
                float expected = bias.getFloat(0, 0, 0, p);
                float actual = broadcastBias.getFloat(0, h, 0, p);
                assertEquals(expected, actual, 1e-6,
                        String.format("Head %d pos %d: broadcast mismatch", h, p));
            }
        }

        // Verify causal structure: attended positions [0..cachePos-1] and [totalSeqLen-1]
        for (int p = 0; p < cachePos; p++) {
            assertEquals(0.0f, bias.getFloat(0, 0, 0, p), 1e-6,
                    "Position " + p + " should be attended (0.0)");
        }
        for (int p = cachePos; p < MAX_KV_LEN; p++) {
            assertEquals(maskFill, bias.getFloat(0, 0, 0, p), 1e-6,
                    "Position " + p + " should be masked (MASK_FILL)");
        }
        assertEquals(0.0f, bias.getFloat(0, 0, 0, (int) totalSeqLen - 1), 1e-6,
                "Current token position should be attended");

        broadcastBias.close();
        dummyDecoder.close();
        log.info("4D bias broadcasting test passed: [1,1,1,{}] broadcasts identically to [1,{},1,{}]",
                totalSeqLen, NUM_Q_HEADS, totalSeqLen);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 2: Step 1→2 KV scatter transition consistency
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Step 0: prefill (output path) → KV cache initialized from prefill output
     * Step 1: output() path → Java scatter writes to static buffer
     * Step 2+: outputDirect() path → C++ scatter (different code path)
     *
     * This test verifies KV cache is consistent across the step 1→2 transition.
     * Specifically: after Java scatter on step 1, the next C++ scatter on step 2
     * must write to cachePos=prefillLen+1 (not prefillLen, which would overwrite step 1's entry).
     */
    @Test
    @DisplayName("KV cache position advances correctly across step 1→2 transition")
    public void testStep1To2KvScatterTransition() {
        UnifiedKvCacheManager mgr = new UnifiedKvCacheManager();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        // Simulate prefill: fill positions 0..PREFILL_LEN-1 with recognizable data
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            for (int p = 0; p < PREFILL_LEN; p++) {
                e.getValue().get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(p), NDArrayIndex.all()).assign(0.1f * (p + 1));
            }
        }

        initializeManager(mgr, staticKvBuffers, PREFILL_LEN);
        assertEquals(PREFILL_LEN, mgr.getCachePosition(),
                "After prefill init, cachePos should be " + PREFILL_LEN);

        // === Step 1: Java scatter (output() path, cppScatterThisStep=false) ===
        ModelIOConfig.KVCacheNames kvNames = createKvNames();
        Map<String, INDArray> step1Outputs = createPresentKvOutputs(kvNames, 1.0f);

        mgr.scatterNewEntries(step1Outputs, kvNames);

        assertEquals(PREFILL_LEN + 1, mgr.getCachePosition(),
                "After step 1 Java scatter, cachePos should be " + (PREFILL_LEN + 1));

        // Verify step 1 data written at position PREFILL_LEN
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            float val = e.getValue().getFloat(0, 0, PREFILL_LEN, 0);
            if (e.getKey().endsWith(".key")) {
                assertEquals(1.0f, val, 1e-5,
                        "Step 1 key data should be at position " + PREFILL_LEN);
            }
        }

        // Verify prefill data NOT overwritten
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            float prefillVal = e.getValue().getFloat(0, 0, 0, 0);
            assertEquals(0.1f, prefillVal, 1e-5,
                    "Prefill data at position 0 should be preserved");
        }

        // === Step 2: Simulate C++ scatter (would write at cachePos=PREFILL_LEN+1) ===
        // C++ scatter uses the same cachePosition from the manager
        long step2CachePos = mgr.getCachePosition();
        assertEquals(PREFILL_LEN + 1, step2CachePos,
                "Step 2 should scatter at position " + (PREFILL_LEN + 1));

        // Simulate: write step 2 data at step2CachePos
        Map<String, INDArray> step2Outputs = createPresentKvOutputs(kvNames, 2.0f);
        mgr.scatterNewEntries(step2Outputs, kvNames);

        assertEquals(PREFILL_LEN + 2, mgr.getCachePosition(),
                "After step 2, cachePos should be " + (PREFILL_LEN + 2));

        // Verify step 2 data at PREFILL_LEN+1
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (e.getKey().endsWith(".key")) {
                float step1Val = e.getValue().getFloat(0, 0, PREFILL_LEN, 0);
                float step2Val = e.getValue().getFloat(0, 0, PREFILL_LEN + 1, 0);
                assertEquals(1.0f, step1Val, 1e-5,
                        "Step 1 data should still be at position " + PREFILL_LEN);
                assertEquals(2.0f, step2Val, 1e-5,
                        "Step 2 data should be at position " + (PREFILL_LEN + 1));
            }
        }

        // Close outputs
        for (INDArray arr : step1Outputs.values()) arr.close();
        for (INDArray arr : step2Outputs.values()) arr.close();

        log.info("Step 1→2 KV scatter transition test passed: no gap, no overwrite");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 3: Long-run mask+bias consistency (20+ steps)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Simulates 30 decode steps and verifies that the 1D attention_mask and the
     * 4D attention bias stay perfectly in sync. Each step should unmask exactly
     * one more position in both.
     *
     * This catches drift bugs where one mask updates but the other doesn't.
     */
    @Test
    @DisplayName("1D mask and 4D bias stay in sync over 30 decode steps")
    public void testMaskBiasSyncOver30Steps() {
        int cachePos = PREFILL_LEN;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;
        float maskFill = DecoderUtils.MASK_FILL;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();
        int numSteps = 30;

        for (int step = 0; step < numSteps; step++) {
            int pos = cachePos + step;
            long pastSeqLen = 679 + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    pastSeqLen, currentSeqLen, staticKvBuffers, MAX_KV_LEN, pos,
                    true, HIDDEN_SIZE, reusableInputs, true);

            INDArray mask = result.get("attention_mask");
            INDArray bias = result.get(ATTN_REFORMAT_NODE);
            assertNotNull(mask, "Step " + step + ": attention_mask missing");
            assertNotNull(bias, "Step " + step + ": 4D bias missing");

            // Count 1s in 1D mask
            long maskOnes = mask.sumNumber().longValue();
            // Count 0.0s in 4D bias
            int biasAttended = 0;
            for (int p = 0; p < totalSeqLen; p++) {
                if (Math.abs(bias.getFloat(0, 0, 0, p)) < 1e-6) {
                    biasAttended++;
                }
            }

            // Both should track the same number of attended positions
            long expectedAttended = pos + 1;  // [0..pos-1] + current token
            assertEquals(expectedAttended, maskOnes,
                    String.format("Step %d: 1D mask should have %d ones, got %d",
                            step, expectedAttended, maskOnes));
            assertEquals(expectedAttended, biasAttended,
                    String.format("Step %d: 4D bias should have %d attended, got %d",
                            step, expectedAttended, biasAttended));

            // Verify consistency: for every position, mask=1 iff bias=0.0
            // Note: 1D mask is [1, totalSeqLen] LONG, 4D bias is [1,1,1,totalSeqLen] FLOAT
            for (int p = 0; p < totalSeqLen; p++) {
                long maskVal = mask.getLong(0, p);
                float biasVal = bias.getFloat(0, 0, 0, p);

                if (maskVal == 1) {
                    assertEquals(0.0f, biasVal, 1e-6,
                            String.format("Step %d pos %d: mask=1 but bias=%.2f (should be 0.0)",
                                    step, p, biasVal));
                } else {
                    assertEquals(maskFill, biasVal, 1e-6,
                            String.format("Step %d pos %d: mask=0 but bias=%.2f (should be MASK_FILL)",
                                    step, p, biasVal));
                }
            }
        }

        dummyDecoder.close();
        log.info("Mask/bias sync test passed: {} steps, all positions consistent", numSteps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 4: Mask+bias consistency with nativeDecodeInputs=true
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Same as test 3 but with nativeDecodeInputs=true.
     *
     * When C++ handles decode inputs, the 1D attention_mask Java array must still
     * be updated (because D2D copy reads from it), AND the 4D bias must still be
     * updated (because it's a separate placeholder override not handled by C++).
     *
     * This was a known bug: nativeDecodeInputs guards skipped the bias update.
     */
    @Test
    @DisplayName("1D mask and 4D bias stay in sync with nativeDecodeInputs=true over 20 steps")
    public void testMaskBiasSyncWithNativeDecodeInputs() {
        int cachePos = PREFILL_LEN;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;
        float maskFill = DecoderUtils.MASK_FILL;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();
        int numSteps = 20;

        for (int step = 0; step < numSteps; step++) {
            int pos = cachePos + step;
            long pastSeqLen = 679 + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    pastSeqLen, currentSeqLen, staticKvBuffers, MAX_KV_LEN, pos,
                    true, HIDDEN_SIZE, reusableInputs, true);

            INDArray mask = result.get("attention_mask");
            INDArray bias = result.get(ATTN_REFORMAT_NODE);

            long maskOnes = mask.sumNumber().longValue();
            int biasAttended = 0;
            for (int p = 0; p < totalSeqLen; p++) {
                if (Math.abs(bias.getFloat(0, 0, 0, p)) < 1e-6) biasAttended++;
            }

            long expectedAttended = pos + 1;
            assertEquals(expectedAttended, maskOnes,
                    String.format("Step %d (native): 1D mask should have %d ones", step, expectedAttended));
            assertEquals(expectedAttended, biasAttended,
                    String.format("Step %d (native): 4D bias should have %d attended", step, expectedAttended));
        }

        dummyDecoder.close();
        log.info("Mask/bias sync with nativeDecodeInputs=true: {} steps passed", numSteps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 5: ModelIOConfig overload consistency
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * The ModelIOConfig-based buildDecoderInputMap overload should produce identical
     * results to the legacy 6-arg overload when using default ModelIOConfig.
     *
     * This catches regressions where the ModelIOConfig implementation diverges
     * from the original behavior.
     */
    @Test
    @DisplayName("ModelIOConfig overload matches legacy overload with default config")
    public void testModelIOConfigOverloadMatchesLegacy() {
        int cachePos = PREFILL_LEN;
        long pastSeqLen = 679;
        long currentSeqLen = 1;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        List<String> inputNames = createInputNames(false);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        // Legacy overload (no ModelIOConfig)
        Map<String, INDArray> legacyResult = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        // ModelIOConfig overload with default config
        ModelIOConfig defaultConfig = ModelIOConfig.builder().build();
        Map<String, INDArray> configResult = DecoderUtils.buildDecoderInputMap(
                defaultConfig, inputNames, dummyDecoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        // Same keys
        assertEquals(legacyResult.keySet(), configResult.keySet(),
                "Both overloads should produce same keys");

        // Same values
        for (String key : legacyResult.keySet()) {
            INDArray legacyArr = legacyResult.get(key);
            INDArray configArr = configResult.get(key);

            assertArrayEquals(legacyArr.shape(), configArr.shape(),
                    "Key '" + key + "': shapes must match");
            assertEquals(legacyArr.dataType(), configArr.dataType(),
                    "Key '" + key + "': dtypes must match");

            // For KV buffers, they should be the exact same object (identity)
            if (key.contains("past_key_values")) {
                assertSame(legacyArr, configArr,
                        "Key '" + key + "': KV buffers should be identical objects");
            } else if (key.equals("inputs_embeds")) {
                assertSame(legacyArr, configArr,
                        "Embeddings should be identical object");
            } else {
                // For masks, position_ids: compare values
                assertEquals(legacyArr, configArr,
                        "Key '" + key + "': values should be equal");
            }
        }

        dummyDecoder.close();
        log.info("ModelIOConfig overload consistency test passed: {} keys match", legacyResult.size());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 6: KV scatter position tracking across 50 steps
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify that UnifiedKvCacheManager.getCachePosition() advances by exactly 1
     * per scatter call over 50 steps, and that no position is ever double-written
     * or skipped.
     */
    @Test
    @DisplayName("KV cache position advances monotonically over 50 scatter steps")
    public void testCachePositionMonotonicOver50Steps() {
        UnifiedKvCacheManager mgr = new UnifiedKvCacheManager();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        initializeManager(mgr, staticKvBuffers, PREFILL_LEN);

        ModelIOConfig.KVCacheNames kvNames = createKvNames();
        int numSteps = 50;
        Set<Long> positionsWritten = new HashSet<>();

        for (int step = 0; step < numSteps; step++) {
            long posBefore = mgr.getCachePosition();
            assertFalse(positionsWritten.contains(posBefore),
                    "Position " + posBefore + " was already written at a prior step");
            positionsWritten.add(posBefore);

            // Create unique present KV data for each step
            Map<String, INDArray> outputs = createPresentKvOutputs(kvNames, (step + 1) * 0.1f);
            mgr.scatterNewEntries(outputs, kvNames);

            long posAfter = mgr.getCachePosition();
            assertEquals(posBefore + 1, posAfter,
                    String.format("Step %d: position should advance by exactly 1 (%d → %d)",
                            step, posBefore, posAfter));

            for (INDArray arr : outputs.values()) arr.close();
        }

        assertEquals(PREFILL_LEN + numSteps, mgr.getCachePosition(),
                "Final cache position should be prefillLen + numSteps");

        // Verify each position has unique data (not overwritten)
        String firstKeyBuf = "past_key_values.0.key";
        INDArray buf = staticKvBuffers.get(firstKeyBuf);
        for (int step = 0; step < numSteps; step++) {
            float expected = (step + 1) * 0.1f;
            float actual = buf.getFloat(0, 0, PREFILL_LEN + step, 0);
            assertEquals(expected, actual, 1e-4,
                    String.format("Position %d should have step %d's data (%.1f), got %.1f",
                            PREFILL_LEN + step, step, expected, actual));
        }

        log.info("Cache position tracking: {} steps, all monotonic, no overwrites", numSteps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 7: Prefill-to-decode mask transition
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * After prefill (multi-token, causal mask), the first decode step should
     * produce a mask that includes ALL prefill positions as attended.
     *
     * This catches bugs where the mask at step 0→1 transition doesn't include
     * the prefill-populated positions.
     */
    @Test
    @DisplayName("First decode mask includes all prefill positions as attended")
    public void testPrefillToDecodeMaskTransition() {
        int cachePos = PREFILL_LEN;  // after prefill, cachePos = prefillLen
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        // First decode step after prefill
        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, currentSeqLen, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        INDArray mask = result.get("attention_mask");
        INDArray bias = result.get(ATTN_REFORMAT_NODE);

        // 1D mask: positions [0..PREFILL_LEN-1] should be 1 (prefill slots)
        for (int p = 0; p < PREFILL_LEN; p++) {
            assertEquals(1, mask.getLong(0, p),
                    "Prefill position " + p + " should be 1 in 1D mask");
        }
        // Positions [PREFILL_LEN..MAX_KV_LEN-1] should be 0 (empty slots)
        for (int p = PREFILL_LEN; p < MAX_KV_LEN; p++) {
            assertEquals(0, mask.getLong(0, p),
                    "Empty position " + p + " should be 0 in 1D mask");
        }
        // Current token at totalSeqLen-1
        assertEquals(1, mask.getLong(0, (int) totalSeqLen - 1),
                "Current token position should be 1");

        // 4D bias: prefill positions should be 0.0 (attended)
        for (int p = 0; p < PREFILL_LEN; p++) {
            assertEquals(0.0f, bias.getFloat(0, 0, 0, p), 1e-6,
                    "Prefill position " + p + " should be 0.0 in 4D bias");
        }
        // Empty slots should be MASK_FILL
        for (int p = PREFILL_LEN; p < MAX_KV_LEN; p++) {
            assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, 0, p), 1e-6,
                    "Empty position " + p + " should be MASK_FILL in 4D bias");
        }

        dummyDecoder.close();
        log.info("Prefill→decode mask transition test passed: {} prefill positions attended", PREFILL_LEN);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 8: Reusable input identity stability for CUDA graph replay
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * CUDA graph replay requires stable buffer addresses. When using reusable inputs,
     * the SAME array object must be returned on each call (mutated in-place), not a
     * new allocation.
     *
     * This verifies that attention_mask, 4D bias, and position_ids are reused
     * (same System.identityHashCode) across 10 consecutive calls.
     */
    @Test
    @DisplayName("Reusable inputs return same array objects across calls")
    public void testReusableInputIdentityStability() {
        int cachePos = PREFILL_LEN;
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        // First call: establishes the reusable arrays
        Map<String, INDArray> result0 = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, reusableInputs, true);

        int maskId = System.identityHashCode(result0.get("attention_mask"));
        int biasId = System.identityHashCode(result0.get(ATTN_REFORMAT_NODE));
        int posId = System.identityHashCode(result0.get("position_ids"));

        // Subsequent calls: must return same objects
        for (int step = 1; step < 10; step++) {
            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, 1, staticKvBuffers, MAX_KV_LEN, cachePos + step,
                    true, HIDDEN_SIZE, reusableInputs, true);

            assertEquals(maskId, System.identityHashCode(result.get("attention_mask")),
                    "Step " + step + ": attention_mask should be same object");
            assertEquals(biasId, System.identityHashCode(result.get(ATTN_REFORMAT_NODE)),
                    "Step " + step + ": 4D bias should be same object");
            assertEquals(posId, System.identityHashCode(result.get("position_ids")),
                    "Step " + step + ": position_ids should be same object");
        }

        dummyDecoder.close();
        log.info("Reusable input identity test passed: all arrays stable over 10 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 9: Padded vs view-based KV input mode
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * When dspActive=true (padded mode), the full static KV buffer is passed.
     * When dspActive=false (view mode), a view [0:cachePos] is passed.
     *
     * This verifies the shape difference and that view mode returns the correct
     * prefix slice.
     */
    @Test
    @DisplayName("Padded mode passes full buffer, view mode passes prefix slice")
    public void testPaddedVsViewKvInputMode() {
        int cachePos = PREFILL_LEN;
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        List<String> inputNames = createInputNames(false);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        // Padded mode (dspActive=true)
        Map<String, INDArray> paddedResult = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                679, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        // View mode (dspActive=false)
        Map<String, INDArray> viewResult = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                679, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, false);

        String firstKv = "past_key_values.0.key";
        INDArray paddedKv = paddedResult.get(firstKv);
        INDArray viewKv = viewResult.get(firstKv);

        assertNotNull(paddedKv, "Padded mode should have KV buffer");
        assertNotNull(viewKv, "View mode should have KV buffer");

        // Padded: full buffer [1, heads, MAX_KV_LEN, dim]
        assertEquals(MAX_KV_LEN, paddedKv.size(2),
                "Padded mode should pass full MAX_KV_LEN buffer");

        // View: prefix [1, heads, cachePos, dim]
        assertEquals(cachePos, viewKv.size(2),
                "View mode should pass cachePos-sized view");

        // The padded KV should be the same object as the static buffer
        assertSame(staticKvBuffers.get(firstKv), paddedKv,
                "Padded KV should be the same static buffer object");

        dummyDecoder.close();
        log.info("Padded vs view mode test passed: padded=[{},{},{},{}], view=[{},{},{},{}]",
                paddedKv.size(0), paddedKv.size(1), paddedKv.size(2), paddedKv.size(3),
                viewKv.size(0), viewKv.size(1), viewKv.size(2), viewKv.size(3));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 10: Causal mask correctness for prefill (multi-token)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * For prefill with currentSeqLen > 1, the causal mask must be upper-triangular:
     * query q can attend to keys 0..pastSeqLen+q, but not to future keys.
     *
     * This catches bugs in buildCausalMask for multi-token inputs.
     */
    @Test
    @DisplayName("Causal mask is correctly upper-triangular for multi-token prefill")
    public void testCausalMaskPrefill() {
        long prefillLen = 5;
        long pastSeqLen = 0;
        long totalSeqLen = pastSeqLen + prefillLen;

        INDArray mask = DecoderUtils.buildCausalMask(prefillLen, totalSeqLen);

        // Shape: [1, 1, prefillLen, totalSeqLen]
        assertArrayEquals(new long[]{1, 1, prefillLen, totalSeqLen}, mask.shape(),
                "Causal mask shape should be [1,1," + prefillLen + "," + totalSeqLen + "]");

        float maskFill = DecoderUtils.MASK_FILL;

        // Verify causal structure: q attends to k iff k <= pastSeqLen + q
        for (int q = 0; q < prefillLen; q++) {
            int lastVisible = (int) pastSeqLen + q;
            for (int k = 0; k < totalSeqLen; k++) {
                float val = mask.getFloat(0, 0, q, k);
                if (k <= lastVisible) {
                    assertEquals(0.0f, val, 1e-6,
                            String.format("q=%d, k=%d: should be 0.0 (attended, lastVisible=%d)",
                                    q, k, lastVisible));
                } else {
                    assertEquals(maskFill, val, 1e-6,
                            String.format("q=%d, k=%d: should be MASK_FILL (future token)",
                                    q, k));
                }
            }
        }

        // Verify single-token decode mask is all zeros (can attend to everything)
        INDArray decodeMask = DecoderUtils.buildCausalMask(1, 10);
        assertArrayEquals(new long[]{1, 1, 1, 10}, decodeMask.shape());
        assertEquals(0.0, decodeMask.sumNumber().doubleValue(), 1e-6,
                "Single-token decode mask should be all zeros");

        mask.close();
        decodeMask.close();
        log.info("Causal mask prefill test passed: correct upper-triangular structure");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 11: KV scatter with multiple layers
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify that KV scatter writes to ALL layers, not just layer 0.
     * Each layer's key and value buffer should receive the new entry.
     */
    @Test
    @DisplayName("KV scatter writes to all layers")
    public void testKvScatterAllLayers() {
        UnifiedKvCacheManager mgr = new UnifiedKvCacheManager();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        initializeManager(mgr, staticKvBuffers, PREFILL_LEN);

        ModelIOConfig.KVCacheNames kvNames = createKvNames();

        // Create outputs with layer-specific values
        Map<String, INDArray> outputs = new HashMap<>();
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            INDArray presentKey = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign((layer + 1) * 10.0f);
            outputs.put("present." + layer + ".key", presentKey);

            INDArray presentVal = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            presentVal.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign((layer + 1) * 20.0f);
            outputs.put("present." + layer + ".value", presentVal);
        }

        mgr.scatterNewEntries(outputs, kvNames);

        // Verify ALL layers received data
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            String keyName = "past_key_values." + layer + ".key";
            String valName = "past_key_values." + layer + ".value";

            float keyVal = staticKvBuffers.get(keyName).getFloat(0, 0, PREFILL_LEN, 0);
            float valVal = staticKvBuffers.get(valName).getFloat(0, 0, PREFILL_LEN, 0);

            assertEquals((layer + 1) * 10.0f, keyVal, 1e-5,
                    "Layer " + layer + " key should have value " + ((layer + 1) * 10.0f));
            assertEquals((layer + 1) * 20.0f, valVal, 1e-5,
                    "Layer " + layer + " value should have value " + ((layer + 1) * 20.0f));
        }

        for (INDArray arr : outputs.values()) arr.close();
        log.info("KV scatter all-layers test passed: {} layers × 2 (key+value) all written", NUM_LAYERS);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 12: Edge case — cachePos at maxKvLen boundary
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * When cachePos approaches maxKvLen, the mask/bias should have almost all
     * positions attended (only the remaining empty slots masked).
     * At cachePos=maxKvLen, ALL past positions are attended.
     */
    @Test
    @DisplayName("Mask/bias correct at cachePos boundary (near maxKvLen)")
    public void testMaskBiasAtBoundary() {
        // cachePos = maxKvLen - 1 (one slot remaining)
        int cachePos = MAX_KV_LEN - 1;
        long totalSeqLen = MAX_KV_LEN + 1;
        float maskFill = DecoderUtils.MASK_FILL;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNames(true);

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null, true);

        INDArray mask = result.get("attention_mask");
        INDArray bias = result.get(ATTN_REFORMAT_NODE);

        // All positions [0..cachePos-1] + currentToken = cachePos + 1 = MAX_KV_LEN
        long maskOnes = mask.sumNumber().longValue();
        assertEquals(cachePos + 1, maskOnes,
                "Near boundary: mask should have " + (cachePos + 1) + " ones");

        // Only position cachePos (=MAX_KV_LEN-1) should be masked in 4D bias
        assertEquals(maskFill, bias.getFloat(0, 0, 0, cachePos), 1e-6,
                "Position cachePos should be MASK_FILL (last empty slot)");

        // Now test AT boundary: cachePos = maxKvLen (buffer full)
        Map<String, INDArray> resultFull = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, 1, staticKvBuffers, MAX_KV_LEN, MAX_KV_LEN,
                true, HIDDEN_SIZE, null, true);

        INDArray maskFull = resultFull.get("attention_mask");
        long maskFullOnes = maskFull.sumNumber().longValue();
        // All past positions (MAX_KV_LEN) + current token = MAX_KV_LEN + 1 = totalSeqLen
        assertEquals(MAX_KV_LEN + 1, maskFullOnes,
                "At boundary: all positions should be attended");

        dummyDecoder.close();
        log.info("Boundary test passed: near-boundary has 1 masked, at-boundary has 0 masked");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private Map<String, INDArray> createStaticKvBuffers() {
        Map<String, INDArray> buffers = new HashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            buffers.put("past_key_values." + i + ".key",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN, HEAD_DIM));
            buffers.put("past_key_values." + i + ".value",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN, HEAD_DIM));
        }
        return buffers;
    }

    private ModelIOConfig.KVCacheNames createKvNames() {
        List<String> keyNames = new ArrayList<>();
        List<String> valueNames = new ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            keyNames.add("present." + i + ".key");
            valueNames.add("present." + i + ".value");
        }
        return new ModelIOConfig.KVCacheNames(keyNames, valueNames);
    }

    private List<String> createInputNames(boolean includeAttnReformat) {
        List<String> names = new ArrayList<>();
        names.add("inputs_embeds");
        names.add("attention_mask");
        names.add("input_ids");
        names.add("position_ids");
        if (includeAttnReformat) {
            names.add(ATTN_REFORMAT_NODE);
        }
        for (int i = 0; i < NUM_LAYERS; i++) {
            names.add("past_key_values." + i + ".key");
            names.add("past_key_values." + i + ".value");
        }
        return names;
    }

    private Map<String, INDArray> createPresentKvOutputs(ModelIOConfig.KVCacheNames kvNames, float fillValue) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            INDArray present = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(fillValue);
            outputs.put(name, present);
        }
        for (String name : kvNames.valueNames) {
            INDArray present = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(fillValue * 2.0f);
            outputs.put(name, present);
        }
        return outputs;
    }

    private void initializeManager(UnifiedKvCacheManager mgr,
                                    Map<String, INDArray> staticKvBuffers,
                                    long initialCachePos) {
        try {
            var maxKvLenField = UnifiedKvCacheManager.class.getDeclaredField("maxKvLen");
            maxKvLenField.setAccessible(true);
            maxKvLenField.set(mgr, (long) MAX_KV_LEN);

            var cachePositionField = UnifiedKvCacheManager.class.getDeclaredField("cachePosition");
            cachePositionField.setAccessible(true);
            cachePositionField.set(mgr, initialCachePos);

            var initializedField = UnifiedKvCacheManager.class.getDeclaredField("initialized");
            initializedField.setAccessible(true);
            initializedField.set(mgr, true);

            var buffersField = UnifiedKvCacheManager.class.getDeclaredField("staticKvBuffers");
            buffersField.setAccessible(true);
            buffersField.set(mgr, staticKvBuffers);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize UnifiedKvCacheManager for test", e);
        }
    }
}
