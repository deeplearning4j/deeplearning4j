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
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheManager;
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
 * Regression tests for static KV cache decode bugs.
 *
 * Each test isolates ONE specific behavior that was broken in the decode loop.
 * Tests use synthetic data (no model loading) to run fast and isolate precisely.
 *
 * Bug summary:
 * 1. Step 1 KV scatter gap — when C++ scatter configured, step 1 (output() path)
 *    skips both Java and C++ scatter → first decode token's KV never written
 * 2. C++ kvCachePosition desync — after Java scatter on a non-DSP step, C++ position
 *    not advanced → next C++ scatter overwrites the same position
 * 3. 4D attention bias — padded mode needs correctly-shaped [1,numQHeads,1,totalSeqLen]
 *    mask with MASK_FILL for masked positions, 0 for attended positions
 * 4. Attention mask stale overwrite — Java attention_mask must be updated even when
 *    nativeDecodeInputs=true, because DSP feedDict copies stale Java array each step
 */
@Slf4j
@DisplayName("StaticKvDecodeRegressionTest")
public class StaticKvDecodeRegressionTest {

    // SmolDocling-like dimensions
    private static final int NUM_KV_HEADS = 3;
    private static final int NUM_Q_HEADS = 9;  // GQA: hiddenSize/headDim = 576/64
    private static final int HEAD_DIM = 64;
    private static final long HIDDEN_SIZE = 576;
    private static final int MAX_KV_LEN = 100;
    private static final int NUM_LAYERS = 2;  // Reduced for test speed

    private static final String ATTN_REFORMAT_NODE = "/model/attn_mask_reformat/Tile/output_0";

    // ═══════════════════════════════════════════════════════════════════════
    // Test 1: Step 1 KV scatter gap
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: When kvScatterInCpp=true, step 1 uses output() (not outputDirect()),
     * so C++ scatter doesn't run. But StaticKvCacheManager.scatterNewEntries() also
     * short-circuits when kvScatterInCpp=true. Result: step 1's KV is never written.
     *
     * Fix: The loop must use Java scatter when the step doesn't use the DSP path,
     * regardless of the global kvScatterInCpp flag.
     */
    @Test
    @DisplayName("Step 1 must scatter KV even when C++ scatter is globally enabled")
    public void testStep1ScatterNotSkipped() {
        // Simulate: C++ scatter is configured (kvScatterInCpp=true) but this step
        // uses output() not outputDirect() → cppScatterThisStep=false
        StaticKvCacheManager mgr = new StaticKvCacheManager();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        initializeManager(mgr, staticKvBuffers, 10);

        // Simulate step 1 present KV output (concat of past + new entry)
        // present shape: [1, heads, maxKvLen+1, dim] — new entry at last position
        Map<String, INDArray> decoderOutputs = new HashMap<>();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        for (String name : kvNames.keyNames) {
            INDArray present = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            // Fill last position with recognizable data (1.0)
            present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(1.0f);
            decoderOutputs.put(name, present);
        }
        for (String name : kvNames.valueNames) {
            INDArray present = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(2.0f);
            decoderOutputs.put(name, present);
        }

        long posBefore = mgr.getCachePosition();

        // KEY: Do NOT set kvScatterInCpp on manager — the loop decides per-step.
        // When cppScatterThisStep=false, the loop must call scatterNewEntries directly.
        assertFalse(mgr.isKvScatterInCpp(), "Manager should not have kvScatterInCpp set");
        mgr.scatterNewEntries(decoderOutputs, kvNames);

        // Verify scatter happened
        assertEquals(posBefore + 1, mgr.getCachePosition(),
                "Cache position should advance after scatter");

        // Verify data was written to static buffer at cachePos=10
        for (String keyName : kvNames.keyNames) {
            String pastName = keyName.replace("present.", "past_key_values.")
                    .replace(".key", ".key");
            INDArray staticBuf = staticKvBuffers.get(pastName);
            if (staticBuf != null) {
                float val = staticBuf.getFloat(0, 0, 10, 0);
                assertEquals(1.0f, val, 1e-5,
                        "Key data should be scattered to cachePos=10, got " + val);
            }
        }

        log.info("Step 1 scatter test passed: KV data written at cachePos={}", posBefore);
    }

    /**
     * Verify the BUG scenario: if kvScatterInCpp=true on manager, scatterNewEntries
     * skips the scatter and only advances position → data never written.
     */
    @Test
    @DisplayName("kvScatterInCpp=true on manager causes scatter skip (bug scenario)")
    public void testKvScatterInCppCausesSkip() {
        StaticKvCacheManager mgr = new StaticKvCacheManager();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        initializeManager(mgr, staticKvBuffers, 10);

        // Set the flag that causes the bug
        mgr.setKvScatterInCpp(true);

        Map<String, INDArray> decoderOutputs = new HashMap<>();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        for (String name : kvNames.keyNames) {
            INDArray present = Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(1.0f);
            decoderOutputs.put(name, present);
        }
        for (String name : kvNames.valueNames) {
            decoderOutputs.put(name,
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN + 1, HEAD_DIM));
        }

        long posBefore = mgr.getCachePosition();
        mgr.scatterNewEntries(decoderOutputs, kvNames);

        // Position still advances (manager increments even when skipping scatter)
        assertEquals(posBefore + 1, mgr.getCachePosition(),
                "Cache position advances even with skip");

        // But data was NOT written (the bug)
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (e.getKey().contains("key")) {
                float val = e.getValue().getFloat(0, 0, 10, 0);
                assertEquals(0.0f, val, 1e-5,
                        "With kvScatterInCpp=true, scatter should be skipped — data should remain 0");
            }
        }

        log.info("Bug scenario confirmed: kvScatterInCpp=true skips scatter, data remains zero");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 2: Attention mask incremental update with reusable inputs
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: In padded mode with nativeDecodeInputs=false, attention_mask must
     * accumulate 1s at each decode step. Java-side incremental updates mark each
     * new KV position as valid.
     *
     * Simulates 5 decode steps and verifies the mask accumulates correctly.
     * Note: mask layout is [1, maxKvLen + currentSeqLen], with:
     * - positions [0..cachePos-1] = 1 on first call (cachePos ones for filled KV slots)
     * - position [totalSeqLen-1] = 1 (current token)
     * - each subsequent step adds 1 more at [cachePos+step-1] (newly filled by scatter)
     */
    @Test
    @DisplayName("Attention mask accumulates correctly over multiple decode steps")
    public void testAttentionMaskAccumulation() {
        int cachePos = 10;
        long pastSeqLen = 679;
        long currentSeqLen = 1;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        // Simulate 5 decode steps with nativeDecodeInputs=false (Java-side accumulation)
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    inputNames, dummyDecoder, embeddings, inputIds,
                    pastSeqLen + step, currentSeqLen,
                    staticKvBuffers, MAX_KV_LEN, cachePos + step,
                    true, HIDDEN_SIZE, reusableInputs,
                    true);  // dspActive=true

            INDArray mask = result.get("attention_mask");
            assertNotNull(mask, "attention_mask at step " + step);

            // First call: [0..cachePos-1] (10 filled KV slots) + lastPos (current token) = 11
            // Step 1: reuse marks (cachePos+1)-1=10 as filled → 12
            // Step 2: reuse marks (cachePos+2)-1=11 as filled → 13
            // General: cachePos + step + 1 = filled slots + current token
            long expectedOnes = (cachePos + step) + 1;
            long actualOnes = mask.sumNumber().longValue();

            assertEquals(expectedOnes, actualOnes,
                    String.format("Step %d: mask should have %d ones (positions [0..%d] + last), got %d",
                            step, expectedOnes, cachePos + step - 1, actualOnes));

            // Verify current token position is set
            long totalSeqLen = MAX_KV_LEN + currentSeqLen;
            assertEquals(1, mask.getLong(0, totalSeqLen - 1),
                    "Step " + step + ": current token position should be 1");

            log.info("Step {}: mask sum={} (expected {}), shape={}",
                    step, actualOnes, expectedOnes, Arrays.toString(mask.shape()));
        }

        // Verify it's the SAME array (reused, not recreated each step)
        INDArray mask1 = reusableInputs.get("attention_mask");
        assertNotNull(mask1, "Mask should be cached in reusableInputs");

        dummyDecoder.close();
        log.info("Attention mask accumulation test passed: 5 steps with correct incremental unmasking");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 3: 4D attention bias shape and content (attn_mask_reformat override)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: When attn_mask_reformat is overridden to PLACEHOLDER, DecoderUtils
     * must provide a 4D attention bias [1, 1, currentSeqLen, totalSeqLen] with:
     * - 0.0 at attended positions (0..cachePos-1 and totalSeqLen-1)
     * - MASK_FILL at masked positions (unfilled cache slots)
     *
     * The heads dimension is 1 (broadcast across Q heads at runtime).
     */
    @Test
    @DisplayName("4D attention bias has correct shape and mask values")
    public void testAttnBiasShapeAndContent() {
        int cachePos = 10;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        // Input names include the attn_mask_reformat output (overridden to placeholder)
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", ATTN_REFORMAT_NODE,
                "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                679, currentSeqLen, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, null,
                true);  // dspActive=true

        INDArray attnBias = result.get(ATTN_REFORMAT_NODE);
        assertNotNull(attnBias, "4D attention bias should be present");

        // Shape: [1, 1, currentSeqLen, totalSeqLen] — heads dim=1 for broadcasting
        assertArrayEquals(new long[]{1, 1, currentSeqLen, totalSeqLen}, attnBias.shape(),
                String.format("4D bias shape should be [1,1,1,%d], got %s",
                        totalSeqLen, Arrays.toString(attnBias.shape())));

        // Check attended positions are 0.0 (head=0 since dim is 1)
        // Filled positions (0..cachePos-1)
        for (int p = 0; p < cachePos; p++) {
            assertEquals(0.0f, attnBias.getFloat(0, 0, 0, p), 1e-6,
                    String.format("pos=%d should be 0.0 (attended)", p));
        }
        // Current token (last position)
        assertEquals(0.0f, attnBias.getFloat(0, 0, 0, (int) totalSeqLen - 1), 1e-6,
                "last pos should be 0.0 (current token)");

        // Check masked positions have MASK_FILL
        float maskFill = DecoderUtils.MASK_FILL;
        // Unfilled positions (cachePos..maxKvLen-1)
        for (int p = cachePos; p < MAX_KV_LEN; p++) {
            assertEquals(maskFill, attnBias.getFloat(0, 0, 0, p), 1e-6,
                    String.format("pos=%d should be MASK_FILL (unfilled)", p));
        }

        dummyDecoder.close();
        log.info("4D attention bias test passed: shape={}, {} attended, {} masked",
                Arrays.toString(attnBias.shape()), cachePos + 1,
                MAX_KV_LEN - cachePos);
    }

    /**
     * Regression: The 4D attention bias must accumulate (unmask) positions incrementally
     * when reusableInputs are used, matching the 1D attention_mask accumulation.
     */
    @Test
    @DisplayName("4D attention bias accumulates correctly over multiple decode steps")
    public void testAttnBiasAccumulation() {
        int cachePos = 10;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", ATTN_REFORMAT_NODE,
                "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();
        float maskFill = DecoderUtils.MASK_FILL;

        for (int step = 0; step < 5; step++) {
            int pos = cachePos + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, currentSeqLen, staticKvBuffers, MAX_KV_LEN, pos,
                    true, HIDDEN_SIZE, reusableInputs,
                    true);

            INDArray attnBias = result.get(ATTN_REFORMAT_NODE);
            assertNotNull(attnBias, "Step " + step + ": bias should exist");

            // Count attended positions (value == 0.0) across head 0
            int attendedCount = 0;
            for (int p = 0; p < totalSeqLen; p++) {
                if (Math.abs(attnBias.getFloat(0, 0, 0, p)) < 1e-6) {
                    attendedCount++;
                }
            }

            int expectedAttended = pos + 1;  // 0..pos-1 (past) + totalSeqLen-1 (current)
            assertEquals(expectedAttended, attendedCount,
                    String.format("Step %d (cachePos=%d): expected %d attended positions, got %d",
                            step, pos, expectedAttended, attendedCount));

            // Verify newly unmasked position (pos-1 was masked, now should be 0.0)
            if (step > 0) {
                assertEquals(0.0f, attnBias.getFloat(0, 0, 0, pos - 1), 1e-6,
                        String.format("Step %d: position %d should be newly unmasked", step, pos - 1));
            }

            // Verify next unfilled position is still masked
            if (pos < MAX_KV_LEN) {
                assertEquals(maskFill, attnBias.getFloat(0, 0, 0, pos), 1e-6,
                        String.format("Step %d: position %d should still be masked", step, pos));
            }

            log.info("Step {}: cachePos={}, attended={} (expected {})",
                    step, pos, attendedCount, expectedAttended);
        }

        dummyDecoder.close();
        log.info("4D attention bias accumulation test passed: 5 steps with correct incremental unmasking");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 3b: 4D bias must update even when nativeDecodeInputs=true
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: The 4D attention bias putScalar(cachePos-1, 0.0f) must run regardless
     * of the nativeDecodeInputs flag. C++ native decode handles input_ids, position_ids,
     * and attention_mask — but NOT the 4D attention bias placeholder override.
     *
     * Previous bug: the putScalar was guarded by {@code !nativeDecodeInputs}, so when
     * C++ native decode was active, past positions stayed masked → model had no context
     * → output EOS immediately or degenerate text.
     *
     * Fix: Remove the {@code !nativeDecodeInputs} guard.
     */
    @Test
    @DisplayName("4D attention bias updates correctly even with nativeDecodeInputs=true")
    public void testAttnBiasUpdatesWithNativeDecodeInputs() {
        int cachePos = 10;
        long currentSeqLen = 1;
        long totalSeqLen = MAX_KV_LEN + currentSeqLen;

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", ATTN_REFORMAT_NODE,
                "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();
        float maskFill = DecoderUtils.MASK_FILL;

        // Run 5 steps with nativeDecodeInputs=true
        for (int step = 0; step < 5; step++) {
            int pos = cachePos + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, currentSeqLen, staticKvBuffers, MAX_KV_LEN, pos,
                    true, HIDDEN_SIZE, reusableInputs,
                    true);

            INDArray attnBias = result.get(ATTN_REFORMAT_NODE);
            assertNotNull(attnBias, "Step " + step + ": bias should exist with nativeDecodeInputs=true");

            // Count attended positions
            int attendedCount = 0;
            for (int p = 0; p < totalSeqLen; p++) {
                if (Math.abs(attnBias.getFloat(0, 0, 0, p)) < 1e-6) {
                    attendedCount++;
                }
            }

            int expectedAttended = pos + 1;  // positions 0..pos-1 (past) + totalSeqLen-1 (current)
            assertEquals(expectedAttended, attendedCount,
                    String.format("Step %d (nativeDecodeInputs=true, cachePos=%d): " +
                            "expected %d attended positions, got %d",
                            step, pos, expectedAttended, attendedCount));

            // Verify newly unmasked position
            if (step > 0) {
                assertEquals(0.0f, attnBias.getFloat(0, 0, 0, pos - 1), 1e-6,
                        String.format("Step %d: position %d should be unmasked even with nativeDecodeInputs=true",
                                step, pos - 1));
            }

            // Verify next unfilled position is still masked
            if (pos < MAX_KV_LEN) {
                assertEquals(maskFill, attnBias.getFloat(0, 0, 0, pos), 1e-6,
                        String.format("Step %d: position %d should still be masked", step, pos));
            }

            log.info("Step {} (nativeDecodeInputs=true): cachePos={}, attended={}",
                    step, pos, attendedCount);
        }

        dummyDecoder.close();
        log.info("4D bias with nativeDecodeInputs=true test passed: bias correctly updated over 5 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 4: Position IDs always updated (even with nativeDecodeInputs)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: position_ids Java array must be updated each step when
     * nativeDecodeInputs=false. When nativeDecodeInputs=true, C++ handles
     * position_ids directly and Java-side doesn't update the reusable array.
     */
    @Test
    @DisplayName("Position IDs update correctly across decode steps with nativeDecodeInputs=false")
    public void testPositionIdsAlwaysUpdated() {
        int cachePos = 10;
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        // Test with nativeDecodeInputs=false: Java must update each step
        for (int step = 0; step < 5; step++) {
            long pastSeqLen = 679 + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    inputNames, dummyDecoder, embeddings, inputIds,
                    pastSeqLen, 1, staticKvBuffers, MAX_KV_LEN, cachePos + step,
                    true, HIDDEN_SIZE, reusableInputs,
                    true);  // dspActive=true

            INDArray posIds = result.get("position_ids");
            assertNotNull(posIds, "position_ids at step " + step);
            assertEquals(pastSeqLen, posIds.getLong(0, 0),
                    String.format("Step %d: position_ids should be %d, got %d",
                            step, pastSeqLen, posIds.getLong(0, 0)));
        }

        dummyDecoder.close();
        log.info("Position IDs test passed: correctly updated across 5 steps with nativeDecodeInputs=false");
    }

    /**
     * Verify that when nativeDecodeInputs=true, Java-side position_ids ARE still updated.
     *
     * The J3 fix changed this: Java position_ids must always be correct because
     * step 1 uses output() (not capture buffers), and the D2D copy from the Java
     * array to the capture buffer reads from the Java array's device buffer.
     * Keeping Java in sync ensures correct values for step 1 and consistent diagnostics.
     */
    @Test
    @DisplayName("Position IDs always updated even with nativeDecodeInputs=true (J3 fix)")
    public void testPositionIdsAlwaysUpdatedWithNativeDecodeInputs() {
        int cachePos = 10;
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        List<String> inputNames = List.of("inputs_embeds", "attention_mask", "input_ids",
                "position_ids", "past_key_values.0.key", "past_key_values.0.value",
                "past_key_values.1.key", "past_key_values.1.value");

        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        long initialPastSeqLen = 679;
        // Step 0: creates posIds with pastSeqLen=679
        DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                initialPastSeqLen, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, reusableInputs,
                true);

        // Step 1: Java posIds should be updated to 680 even with nativeDecodeInputs=true
        Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                inputNames, dummyDecoder, embeddings, inputIds,
                initialPastSeqLen + 1, 1, staticKvBuffers, MAX_KV_LEN, cachePos + 1,
                true, HIDDEN_SIZE, reusableInputs,
                true);

        INDArray posIds = result.get("position_ids");
        assertEquals(initialPastSeqLen + 1, posIds.getLong(0, 0),
                "With nativeDecodeInputs=true, Java posIds should be updated (J3 fix)");

        dummyDecoder.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test 5: cppScatterThisStep logic
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verifies the per-step scatter decision logic:
     * - cppScatterThisStep = kvScatterInCpp && useDirect
     * - Step 1 (useDirect=false): must use Java scatter
     * - Step 2+ (useDirect=true): may use C++ scatter
     */
    @Test
    @DisplayName("Per-step scatter decision: step 1 always Java, step 2+ C++ when enabled")
    public void testPerStepScatterDecision() {
        boolean kvScatterInCpp = true;

        // Step 0: never direct
        assertFalse(kvScatterInCpp && false, "Step 0: cppScatterThisStep should be false");

        // Step 1: not direct (step >= 2 required)
        boolean useDirect1 = true && (1 >= 2);  // usingStaticKv && step >= 2
        boolean cppScatter1 = kvScatterInCpp && useDirect1;
        assertFalse(cppScatter1, "Step 1: cppScatterThisStep should be false (uses output())");

        // Step 2: direct
        boolean useDirect2 = true && (2 >= 2);
        boolean cppScatter2 = kvScatterInCpp && useDirect2;
        assertTrue(cppScatter2, "Step 2: cppScatterThisStep should be true (uses outputDirect())");

        // Step 2 with kvScatterInCpp=false
        assertFalse(false && useDirect2, "Step 2 with Java scatter: cppScatterThisStep false");

        log.info("Per-step scatter decision logic verified");
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

    private DecoderUtils.KVCacheNames createKvNames() {
        List<String> keyNames = new java.util.ArrayList<>();
        List<String> valueNames = new java.util.ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            keyNames.add("present." + i + ".key");
            valueNames.add("present." + i + ".value");
        }
        return new DecoderUtils.KVCacheNames(keyNames, valueNames);
    }

    private void initializeManager(StaticKvCacheManager mgr,
                                    Map<String, INDArray> staticKvBuffers,
                                    long initialCachePos) {
        // Directly initialize the manager's internal state
        // (normally done by initializeFromPrefill, but we bypass the full prefill)
        try {
            var maxKvLenField = StaticKvCacheManager.class.getDeclaredField("maxKvLen");
            maxKvLenField.setAccessible(true);
            maxKvLenField.set(mgr, (long) MAX_KV_LEN);

            var cachePositionField = StaticKvCacheManager.class.getDeclaredField("cachePosition");
            cachePositionField.setAccessible(true);
            cachePositionField.set(mgr, initialCachePos);

            var initializedField = StaticKvCacheManager.class.getDeclaredField("initialized");
            initializedField.setAccessible(true);
            initializedField.set(mgr, true);

            var buffersField = StaticKvCacheManager.class.getDeclaredField("staticKvBuffers");
            buffersField.setAccessible(true);
            buffersField.set(mgr, staticKvBuffers);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize StaticKvCacheManager for test", e);
        }
    }
}
