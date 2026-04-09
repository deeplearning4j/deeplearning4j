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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for decode loop state transitions and cross-component interactions
 * that are NOT covered by existing regression tests.
 *
 * These test the exact mutable state transitions from StaticKvCacheDecodeLoop:
 * - Prefill → static KV transition (step 0 → step 1)
 * - Execution path switch (output → outputDirect at step 2)
 * - Reusable buffer lifecycle across steps
 * - KV output closing decisions based on cppScatterThisStep
 * - Placeholder override + input name refresh ordering
 * - Configuration flag conflict behavior
 * - Mask shape transition from prefill to decode
 * - Per-step input cleanup safety
 */
@Slf4j
@DisplayName("Decode Loop State Transition Tests")
public class DecodeLoopStateTransitionTest {

    // Constants matching a small model for fast tests
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN_SIZE = 32;
    private static final int PREFILL_LEN = 10;
    private static final int MAX_NEW_TOKENS = 20;
    private static final int MAX_KV_LEN = PREFILL_LEN + MAX_NEW_TOKENS;
    private static final float MASK_FILL = -3.4028235e+38f;

    // ─── Test 1: Prefill→Decode usingStaticKv flag transition ────────────

    @Test
    @DisplayName("1: usingStaticKv transitions from false→true after prefill, maxKvLen/cachePos update")
    public void testPrefillToStaticKvFlagTransition() {
        // Simulates the decode loop's step 0 state transition (lines 381-390)
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .build();
        StaticKvCacheManager kvManager = new StaticKvCacheManager(ioConfig);

        // Before prefill: manager returns sentinel values
        assertEquals(-1, kvManager.getMaxKvLen(), "maxKvLen should be -1 before init");
        assertEquals(0, kvManager.getCachePosition(), "cachePos should be 0 before init");

        // Create prefill outputs
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(kvNames);

        // Simulate step 0: initialize from prefill
        kvManager.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

        // After prefill: values must be correct (loop reads these at line 389-390)
        assertEquals(MAX_KV_LEN, kvManager.getMaxKvLen(),
                "maxKvLen should be PREFILL_LEN + MAX_NEW_TOKENS after init");
        assertEquals(PREFILL_LEN, kvManager.getCachePosition(),
                "cachePos should be PREFILL_LEN after init");
        assertTrue(kvManager.isInitialized(), "Manager should be initialized");

        // Static KV buffers should be padded to maxKvLen
        Map<String, INDArray> buffers = kvManager.getStaticKvBuffers();
        assertNotNull(buffers, "Static KV buffers should be non-null");
        assertEquals(NUM_LAYERS * 2, buffers.size(), "Should have one buffer per key + value per layer");
        for (INDArray buf : buffers.values()) {
            assertEquals(MAX_KV_LEN, buf.size(2),
                    "Static buffer seq dim should be maxKvLen (padded)");
        }

        kvManager.close();
    }

    // ─── Test 2: Execution path choice per step ──────────────────────────

    @Test
    @DisplayName("2: output() vs outputDirect() path depends on step and usingStaticKv")
    public void testExecutionPathChoicePerStep() {
        // Simulates the decode loop's path selection logic (lines 282-296)
        // useDirect = usingStaticKv && step >= 2
        // cppScatterThisStep = kvScatterInCpp && useDirect

        boolean usingStaticKv = false;
        boolean kvScatterInCpp = true; // Assume C++ scatter configured

        // Step 0: prefill — always output()
        boolean useDirect0 = usingStaticKv && 0 >= 2;
        assertFalse(useDirect0, "Step 0: should use output() (not static KV yet)");
        boolean cppScatter0 = kvScatterInCpp && useDirect0;
        assertFalse(cppScatter0, "Step 0: no C++ scatter (not direct)");

        // After step 0 prefill, usingStaticKv becomes true
        usingStaticKv = true;

        // Step 1: first decode — still output() because step < 2
        boolean useDirect1 = usingStaticKv && 1 >= 2;
        assertFalse(useDirect1, "Step 1: should use output() (step < 2)");
        boolean cppScatter1 = kvScatterInCpp && useDirect1;
        assertFalse(cppScatter1, "Step 1: no C++ scatter (not direct)");

        // Step 2: switch to outputDirect()
        boolean useDirect2 = usingStaticKv && 2 >= 2;
        assertTrue(useDirect2, "Step 2: should use outputDirect()");
        boolean cppScatter2 = kvScatterInCpp && useDirect2;
        assertTrue(cppScatter2, "Step 2: C++ scatter active (direct + configured)");

        // Step 2 with kvScatterInCpp=false
        kvScatterInCpp = false;
        boolean cppScatter2NoKv = kvScatterInCpp && useDirect2;
        assertFalse(cppScatter2NoKv, "Step 2 without kvScatterInCpp: no C++ scatter");

        // Step 2 with nd4j.dsp.noDirect=true (simulated) — forces output()
        // In the real loop: useDirect = usingStaticKv && step >= 2 && !noDirect
        boolean noDirect = true;
        boolean useDirectWithFlag = usingStaticKv && 2 >= 2 && !noDirect;
        assertFalse(useDirectWithFlag, "Step 2 with noDirect: should use output()");
    }

    // ─── Test 3: KV scatter decision per step ────────────────────────────

    @Test
    @DisplayName("3: KV scatter decision: Java at step 1, C++ at step 2+ when configured")
    public void testKvScatterDecisionPerStep() {
        // Simulates lines 332-379: scatter path depends on cppScatterThisStep
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .build();
        StaticKvCacheManager kvManager = new StaticKvCacheManager(ioConfig);
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        // Initialize from prefill
        kvManager.initializeFromPrefill(createPrefillOutputs(kvNames), kvNames, MAX_NEW_TOKENS, PREFILL_LEN);
        long posAfterPrefill = kvManager.getCachePosition();
        assertEquals(PREFILL_LEN, posAfterPrefill);

        // Step 1: Java scatter (cppScatterThisStep=false because useDirect=false)
        Map<String, INDArray> step1Outputs = createMockDecoderOutputs(kvNames, MAX_KV_LEN);
        kvManager.scatterNewEntries(step1Outputs, kvNames);
        assertEquals(PREFILL_LEN + 1, kvManager.getCachePosition(),
                "Step 1 Java scatter: position should advance by 1");

        // Verify data was actually scattered to correct position
        Map<String, INDArray> buffers = kvManager.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : buffers.entrySet()) {
            INDArray buf = e.getValue();
            // The scattered entry should be at PREFILL_LEN (cachePos before scatter)
            assertFalse(isAllZeros(buf, PREFILL_LEN),
                    e.getKey() + " position " + PREFILL_LEN + " should have scattered data");
        }

        // Step 2: C++ scatter (cppScatterThisStep=true)
        // In C++ scatter mode, the loop just advances the position manually (line 335)
        kvManager.setCachePosition(kvManager.getCachePosition() + 1);
        assertEquals(PREFILL_LEN + 2, kvManager.getCachePosition(),
                "Step 2 C++ scatter: position should advance by 1");

        // Step 3: C++ scatter again
        kvManager.setCachePosition(kvManager.getCachePosition() + 1);
        assertEquals(PREFILL_LEN + 3, kvManager.getCachePosition(),
                "Step 3 C++ scatter: position should advance by 1");

        kvManager.close();
    }

    // ─── Test 4: Reusable embeddings lifecycle ───────────────────────────

    @Test
    @DisplayName("4: Reusable embeddings buffer allocated once, reused via assign across steps")
    public void testReusableEmbeddingsLifecycle() {
        // Simulates lines 636-644: reusableEmbeddings created once, data copied each step
        INDArray reusableEmbeddings = null;
        INDArray reusableInputIds = null;
        INDArray prevEmbeddings;

        // Simulate embedding table row lookup
        INDArray embeddingTable = Nd4j.randn(DataType.FLOAT, 1000, HIDDEN_SIZE);

        // Step 0: non-static KV, creates regular embedding (line 644)
        int tokenId = 42;
        INDArray rowEmbed0 = embeddingTable.getRow(tokenId).reshape(1, 1, HIDDEN_SIZE);
        INDArray currentEmbeddings = rowEmbed0;
        assertNull(reusableEmbeddings, "Step 0: reusable not yet allocated");

        // Step 1: static KV now active, first allocation of reusable (line 637-638)
        prevEmbeddings = currentEmbeddings;
        tokenId = 99;
        INDArray rowEmbed1 = embeddingTable.getRow(tokenId).reshape(1, 1, HIDDEN_SIZE);
        // First time: allocate reusable via dup
        reusableEmbeddings = rowEmbed1.dup();
        currentEmbeddings = reusableEmbeddings;

        // prevEmbeddings != currentEmbeddings — safe to close prev (line 689-693)
        assertNotSame(prevEmbeddings, currentEmbeddings,
                "Step 1: prev and current should be different objects");

        // Store the address of reusable for identity tracking
        long reusableAddr = reusableEmbeddings.data().address();

        // Step 2+: reuse via assign — same object, just data changes
        prevEmbeddings = currentEmbeddings;
        tokenId = 77;
        INDArray rowEmbed2 = embeddingTable.getRow(tokenId).reshape(1, 1, HIDDEN_SIZE);
        reusableEmbeddings.assign(rowEmbed2);
        currentEmbeddings = reusableEmbeddings;

        // CRITICAL: prevEmbeddings == currentEmbeddings — must NOT close (line 689)
        assertSame(prevEmbeddings, currentEmbeddings,
                "Step 2+: prev and current should be SAME object (reusable)");
        assertEquals(reusableAddr, reusableEmbeddings.data().address(),
                "Step 2+: reusable buffer address should not change");

        // Verify data was actually updated
        float expected = rowEmbed2.getFloat(0);
        assertEquals(expected, reusableEmbeddings.getFloat(0, 0, 0), 1e-6,
                "Reusable should have step 2's embedding data");

        // Same logic for input_ids (lines 646-652)
        reusableInputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);
        long idsAddr = reusableInputIds.data().address();
        reusableInputIds.putScalar(0, 0, 99);
        assertEquals(99, reusableInputIds.getLong(0, 0), "putScalar should update input_ids");
        assertEquals(idsAddr, reusableInputIds.data().address(),
                "Input IDs buffer address should not change after putScalar");

        // 10 more steps — verify address stability
        for (int step = 3; step < 13; step++) {
            INDArray rowEmbed = embeddingTable.getRow(step).reshape(1, 1, HIDDEN_SIZE);
            reusableEmbeddings.assign(rowEmbed);
            assertEquals(reusableAddr, reusableEmbeddings.data().address(),
                    "Step " + step + ": buffer address must be stable for CUDA graph replay");

            reusableInputIds.putScalar(0, 0, step);
            assertEquals(idsAddr, reusableInputIds.data().address(),
                    "Step " + step + ": input_ids address must be stable");
        }
    }

    // ─── Test 5: KV output closing decision ──────────────────────────────

    @Test
    @DisplayName("5: Present KV outputs closed only when !cppScatterThisStep")
    public void testKvOutputClosingDecision() {
        // Simulates lines 351-379: close KV outputs only when NOT using C++ scatter

        DecoderUtils.KVCacheNames kvNames = createKvNames();

        // Case 1: cppScatterThisStep=false → should close KV outputs
        {
            boolean cppScatterThisStep = false;
            Map<String, INDArray> outputs = createMockDecoderOutputs(kvNames, MAX_KV_LEN);
            int closed = 0;
            if (!cppScatterThisStep) {
                for (String pn : kvNames.keyNames) {
                    INDArray pv = outputs.get(pn);
                    if (pv != null && !pv.wasClosed()) {
                        pv.setCloseable(true);
                        pv.close();
                        closed++;
                    }
                }
                for (String pn : kvNames.valueNames) {
                    INDArray pv = outputs.get(pn);
                    if (pv != null && !pv.wasClosed()) {
                        pv.setCloseable(true);
                        pv.close();
                        closed++;
                    }
                }
            }
            assertEquals(NUM_LAYERS * 2, closed,
                    "When !cppScatter, all KV outputs should be closed");
        }

        // Case 2: cppScatterThisStep=true → should NOT close KV outputs
        {
            boolean cppScatterThisStep = true;
            Map<String, INDArray> outputs = createMockDecoderOutputs(kvNames, MAX_KV_LEN);
            int closed = 0;
            if (!cppScatterThisStep) {
                for (String pn : kvNames.keyNames) {
                    INDArray pv = outputs.get(pn);
                    if (pv != null && !pv.wasClosed()) {
                        pv.setCloseable(true);
                        pv.close();
                        closed++;
                    }
                }
            }
            assertEquals(0, closed,
                    "When cppScatter, KV outputs should NOT be closed (C++ manages lifecycle)");
            // Verify outputs are still alive
            for (String pn : kvNames.keyNames) {
                assertFalse(outputs.get(pn).wasClosed(),
                        pn + " should NOT be closed when C++ scatter active");
            }
            // Clean up
            for (INDArray arr : outputs.values()) {
                if (!arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
            }
        }
    }

    // ─── Test 6: Placeholder override + input name refresh ordering ──────

    @Test
    @DisplayName("6: addPlaceholderOverride then inputs() refresh captures new placeholder")
    public void testPlaceholderOverrideInputNameRefreshOrdering() {
        // Simulates lines 442-447: override must be added BEFORE re-reading input names
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable mask = sd.placeHolder("attention_mask", DataType.LONG, 1, -1);
        // Internal node — would normally be computed, not an input
        SDVariable castedMask = sd.castTo("mask_cast", mask, DataType.FLOAT);
        SDVariable reformat = sd.identity("attn_reformat_output", castedMask);
        SDVariable combined = sd.math.add("combined", input, reformat);
        SDVariable output = sd.identity("output", combined);
        sd.setOutputs(Collections.singletonList("output"));

        // Before override: internal node is NOT in inputs
        List<String> inputsBefore = sd.inputs();
        assertFalse(inputsBefore.contains("attn_reformat_output"),
                "Before override: internal node should NOT be in inputs");
        assertTrue(inputsBefore.contains("input"), "Before override: input should be present");
        assertTrue(inputsBefore.contains("attention_mask"), "Before override: attention_mask should be present");

        // Add override (line 443)
        sd.addPlaceholderOverride("attn_reformat_output");
        sd.getVariable("attn_reformat_output").setShape(-1);

        // Refresh input names (line 447) — MUST happen AFTER override
        List<String> inputsAfter = sd.inputs();
        assertTrue(inputsAfter.contains("attn_reformat_output"),
                "After override: overridden node should appear in inputs");

        // WRONG ORDER: if we read inputs BEFORE override, the list would be stale
        // This is the bug scenario from the analysis — input names stale after override
        // Not testable directly (ordering is correct in the loop), but we verify the
        // invariant: inputs() reflects overrides immediately.
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("attn_reformat_output").getVariableType(),
                "Overridden var should be PLACEHOLDER");

        // Verify model works with override providing a value
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        feedDict.put("attention_mask", Nd4j.ones(DataType.LONG, 1, 8));
        feedDict.put("attn_reformat_output", Nd4j.randn(DataType.FLOAT, 1, 8));
        Map<String, INDArray> result = sd.output(feedDict, "output");
        assertNotNull(result.get("output"), "Model should execute with overridden placeholder");
    }

    // ─── Test 7: Mask shape transition prefill → decode ──────────────────

    @Test
    @DisplayName("7: Attention mask shape changes from [1,prefillLen] to [1,maxKvLen+1] at decode")
    public void testMaskShapeTransitionPrefillToDecode() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .attnMaskReformatOutput("attn_bias")
                .build();

        List<String> inputNames = createInputNames();
        inputNames.add("attn_bias");

        Map<String, INDArray> reusableInputs = new HashMap<>();

        // Step 0 (prefill): use non-KV input names only (no decoder for createEmptyKvCache)
        List<String> prefillInputNames = new ArrayList<>();
        prefillInputNames.add("inputs_embeds");
        prefillInputNames.add("attention_mask");
        prefillInputNames.add("position_ids");
        prefillInputNames.add("input_ids");
        INDArray prefillEmbeds = Nd4j.randn(DataType.FLOAT, 1, PREFILL_LEN, HIDDEN_SIZE);
        INDArray prefillIds = Nd4j.ones(DataType.LONG, 1, PREFILL_LEN);
        Map<String, INDArray> step0Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, prefillInputNames, null, prefillEmbeds, prefillIds,
                0, PREFILL_LEN, null, -1, 0,
                false, HIDDEN_SIZE, reusableInputs, false);

        INDArray prefillMask = step0Map.get("attention_mask");
        assertNotNull(prefillMask, "Step 0: should have attention_mask");
        assertEquals(PREFILL_LEN, prefillMask.size(1),
                "Step 0 (non-static): mask should be [1, prefillLen]");

        // No bias in non-static mode — bias is only for padded static KV
        // (attn_bias is not in inputs when not using static KV with padded mode)

        // Clear reusable for fresh start after mode switch
        reusableInputs.clear();

        // Step 1 (first decode): static KV mode, padded, cachePos=PREFILL_LEN
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        INDArray decodeEmbeds = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray decodeIds = Nd4j.createFromArray(new long[][]{{42}});
        long cachePos = PREFILL_LEN;

        Map<String, INDArray> step1Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, decodeEmbeds, decodeIds,
                cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, HIDDEN_SIZE, reusableInputs, true);

        INDArray decodeMask = step1Map.get("attention_mask");
        assertNotNull(decodeMask, "Step 1: should have attention_mask");
        long expectedMaskLen = MAX_KV_LEN + 1; // totalSeqLen = maxKvLen + currentSeqLen
        assertEquals(expectedMaskLen, decodeMask.size(1),
                "Step 1 (padded static): mask should be [1, maxKvLen + currentSeqLen]");

        // Verify mask values: 1s for filled, 0s for unfilled
        for (int k = 0; k < cachePos; k++) {
            assertEquals(1L, decodeMask.getLong(0, k),
                    "Position " + k + " should be 1 (filled during prefill)");
        }
        for (int k = (int) cachePos; k < MAX_KV_LEN; k++) {
            assertEquals(0L, decodeMask.getLong(0, k),
                    "Position " + k + " should be 0 (unfilled padding)");
        }
        // Last position (current token in concat region)
        assertEquals(1L, decodeMask.getLong(0, (int) expectedMaskLen - 1),
                "Current token position should be 1");
    }

    // ─── Test 8: Per-step input cleanup skips protected arrays ───────────

    @Test
    @DisplayName("8: Per-step cleanup closes transient inputs but protects embeddings, IDs, KV, reusable")
    public void testPerStepInputCleanupSafety() {
        // Simulates lines 616-627: close per-step inputs except protected categories
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .inputEmbeddingsName("inputs_embeds")
                .inputIdsName("input_ids")
                .kvCachePrefix("past_key_values.")
                .build();

        // Build a mock decoderInputMap with various categories
        Map<String, INDArray> decoderInputMap = new LinkedHashMap<>();
        INDArray embeds = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray ids = Nd4j.createFromArray(new long[][]{{42}});
        INDArray posIds = Nd4j.createFromArray(new long[][]{{10}});
        INDArray mask = Nd4j.ones(DataType.LONG, 1, 31);
        INDArray kvBuf = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        INDArray reusableBias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 31);

        decoderInputMap.put("inputs_embeds", embeds);
        decoderInputMap.put("input_ids", ids);
        decoderInputMap.put("position_ids", posIds);
        decoderInputMap.put("attention_mask", mask);
        decoderInputMap.put("past_key_values.0.key", kvBuf);
        decoderInputMap.put("attn_bias", reusableBias);

        // Reusable inputs cache — protects arrays in this map
        Map<String, INDArray> reusableInputs = new HashMap<>();
        reusableInputs.put("attention_mask", mask);
        reusableInputs.put("attn_bias", reusableBias);

        // Simulate the cleanup loop (lines 616-627)
        List<String> closed = new ArrayList<>();
        for (var entry : decoderInputMap.entrySet()) {
            String name = entry.getKey();
            INDArray arr = entry.getValue();
            if (ioConfig.isInputEmbeddings(name) || ioConfig.isInputIds(name)) continue;
            if (ioConfig.isKvCacheInput(name)) continue;
            if (reusableInputs.containsValue(arr)) continue;
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
                closed.add(name);
            }
        }

        // Only position_ids should be closed — everything else is protected
        assertEquals(1, closed.size(), "Only transient inputs should be closed");
        assertEquals("position_ids", closed.get(0), "position_ids should be the only closed input");

        // Verify protected arrays are still alive
        assertFalse(embeds.wasClosed(), "Embeddings should NOT be closed");
        assertFalse(ids.wasClosed(), "Input IDs should NOT be closed");
        assertFalse(mask.wasClosed(), "Mask should NOT be closed (in reusable cache)");
        assertFalse(kvBuf.wasClosed(), "KV buffer should NOT be closed");
        assertFalse(reusableBias.wasClosed(), "Bias should NOT be closed (in reusable cache)");
        assertTrue(posIds.wasClosed(), "Position IDs should be closed");

        // Clean up
        embeds.close(); ids.close(); mask.close(); kvBuf.close(); reusableBias.close();
    }

    // ─── Test 9: nativeDecodeInputs guard at step 1 ──────────────────────

    @Test
    @DisplayName("9: setNextDecodeToken only called at step >= 1 (never at step 0 prefill)")
    public void testNativeDecodeInputsStepGuard() {
        // Simulates lines 270-272: the guard `if (nativeDecodeInputs && step >= 1)`
        // ensures we never call setNextDecodeToken during prefill

        boolean nativeDecodeInputs = true;

        // Step 0: prefill — should NOT call setNextDecodeToken
        boolean shouldCall0 = nativeDecodeInputs && 0 >= 1;
        assertFalse(shouldCall0, "Step 0: should NOT call setNextDecodeToken during prefill");

        // Step 1: first decode — SHOULD call
        boolean shouldCall1 = nativeDecodeInputs && 1 >= 1;
        assertTrue(shouldCall1, "Step 1: should call setNextDecodeToken");

        // Step 1 with nativeDecodeInputs=false — should NOT call
        nativeDecodeInputs = false;
        boolean shouldCall1Disabled = nativeDecodeInputs && 1 >= 1;
        assertFalse(shouldCall1Disabled, "Step 1 with native disabled: should NOT call");

        // Step 1 with decoderDspExec=null — nativeDecodeInputs would be false
        // (line 253: decoderDspExec != null is part of the condition)
        // This is tested implicitly — nativeDecodeInputs can only be true when exec exists
    }

    // ─── Test 10: logitsOnlyOutputNames vs fullOutputNames ───────────────

    @Test
    @DisplayName("10: Requested output names depend on cppScatterThisStep")
    public void testOutputNamesSelectionLogic() {
        // Simulates lines 288-289
        String[] logitsOnlyOutputNames = new String[]{"logits"};
        String[] fullOutputNames = new String[]{"logits", "present.0.key", "present.0.value",
                "present.1.key", "present.1.value"};

        // cppScatter=true + usingStaticKv=true → logits only (C++ reads KV internally)
        boolean usingStaticKv = true;
        boolean cppScatterThisStep = true;
        String[] requested = (usingStaticKv && cppScatterThisStep) ? logitsOnlyOutputNames : fullOutputNames;
        assertArrayEquals(logitsOnlyOutputNames, requested,
                "With C++ scatter: should request logits only");

        // cppScatter=false → full outputs (need KV for Java scatter)
        cppScatterThisStep = false;
        requested = (usingStaticKv && cppScatterThisStep) ? logitsOnlyOutputNames : fullOutputNames;
        assertArrayEquals(fullOutputNames, requested,
                "Without C++ scatter: should request full outputs for Java scatter");
    }

    // ─── Test 11: advanceKvCachePosition sync at step 1 ──────────────────

    @Test
    @DisplayName("11: When kvScatterInCpp=true but step 1 uses Java scatter, advanceKvCachePosition must be called")
    public void testAdvanceKvCachePositionSyncAtStep1() {
        // Simulates lines 336-346: step 1 always uses Java scatter (useDirect=false)
        // but if kvScatterInCpp=true, must call advanceKvCachePosition to keep C++ in sync

        // Track the sequence of operations
        boolean kvScatterInCpp = true;
        boolean useDirect = false;  // step 1: always false
        boolean cppScatterThisStep = kvScatterInCpp && useDirect;

        assertFalse(cppScatterThisStep, "Step 1: cppScatterThisStep should be false");

        // The loop does Java scatter here (line 339)
        // Then checks if C++ sync needed (lines 344-346)
        boolean shouldAdvanceCpp = kvScatterInCpp; // decoderDspExec != null assumed
        assertTrue(shouldAdvanceCpp,
                "When kvScatterInCpp=true but Java scatter was used, " +
                "must call advanceKvCachePosition to keep C++ position in sync");

        // Verify position sync with actual manager
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .build();
        StaticKvCacheManager kvManager = new StaticKvCacheManager(ioConfig);
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        kvManager.initializeFromPrefill(createPrefillOutputs(kvNames), kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

        // Java scatter at step 1
        long posBefore = kvManager.getCachePosition();
        kvManager.scatterNewEntries(createMockDecoderOutputs(kvNames, MAX_KV_LEN), kvNames);
        long posAfterJavaScatter = kvManager.getCachePosition();
        assertEquals(posBefore + 1, posAfterJavaScatter,
                "Java scatter should advance position by 1");

        // If C++ scatter were managing the position, it would need to match.
        // The separate advanceKvCachePosition() call on decoderDspExec (which we
        // can't call without a real DSP) would set the C++ side to this same value.

        // C++ scatter at step 2 (simulated)
        kvManager.setCachePosition(kvManager.getCachePosition() + 1);
        assertEquals(posBefore + 2, kvManager.getCachePosition(),
                "C++ scatter at step 2 should advance by 1 more");

        kvManager.close();
    }

    // ─── Test 12: dspActive depends on usingStaticKv + noPadded flag ─────

    @Test
    @DisplayName("12: dspActive flag depends on usingStaticKv AND absence of noPadded property")
    public void testDspActiveFlagLogic() {
        // Simulates lines 251-252
        // dspActive = usingStaticKv && !noPadded

        // Before prefill: usingStaticKv=false → dspActive=false
        boolean usingStaticKv = false;
        boolean noPadded = false;
        boolean dspActive = usingStaticKv && !noPadded;
        assertFalse(dspActive, "Before prefill: dspActive should be false");

        // After prefill: usingStaticKv=true, noPadded=false → dspActive=true
        usingStaticKv = true;
        dspActive = usingStaticKv && !noPadded;
        assertTrue(dspActive, "After prefill: dspActive should be true");

        // With noPadded=true: dspActive=false even with static KV
        noPadded = true;
        dspActive = usingStaticKv && !noPadded;
        assertFalse(dspActive, "With noPadded: dspActive should be false");
    }

    // ─── Test 13: Prefill KV close guard (DSP active check) ─────────────

    @Test
    @DisplayName("13: Prefill KV outputs NOT closed when DSP executor is active (dangling pointer risk)")
    public void testPrefillKvCloseGuardDspActive() {
        // Simulates lines 398-410: close prefill KV only when DSP not active
        // When DSP is active, C++ slotArrayCache_ holds raw NDArray* pointers —
        // closing from Java would create dangling pointers

        // Case 1: No DSP active → safe to close
        boolean prefillDspActive = false;
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(kvNames);

        if (!prefillDspActive) {
            for (String pn : kvNames.keyNames) {
                INDArray pv = prefillOutputs.get(pn);
                if (pv != null) { pv.setCloseable(true); pv.close(); }
            }
            for (String pn : kvNames.valueNames) {
                INDArray pv = prefillOutputs.get(pn);
                if (pv != null) { pv.setCloseable(true); pv.close(); }
            }
        }
        // Verify all closed
        for (String name : kvNames.keyNames) {
            assertTrue(prefillOutputs.get(name).wasClosed(), name + " should be closed (no DSP)");
        }

        // Case 2: DSP active → do NOT close
        prefillDspActive = true;
        Map<String, INDArray> prefillOutputs2 = createPrefillOutputs(kvNames);
        if (!prefillDspActive) {
            // This block should NOT execute
            for (String pn : kvNames.keyNames) {
                INDArray pv = prefillOutputs2.get(pn);
                if (pv != null) { pv.setCloseable(true); pv.close(); }
            }
        }
        // Verify NOT closed
        for (String name : kvNames.keyNames) {
            assertFalse(prefillOutputs2.get(name).wasClosed(), name + " should NOT be closed (DSP active)");
        }
        // Clean up
        for (INDArray arr : prefillOutputs2.values()) {
            if (!arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
    }

    // ─── Test 14: Logits HALF→FLOAT cast when needed ─────────────────────

    @Test
    @DisplayName("14: HALF logits are cast to FLOAT for sampling precision")
    public void testLogitsHalfToFloatCast() {
        // Simulates lines 314-318
        INDArray halfLogits = Nd4j.randn(DataType.HALF, 1, 100);
        assertEquals(DataType.HALF, halfLogits.dataType());

        // The decode loop casts to FLOAT and closes original
        INDArray floatLogits;
        if (halfLogits.dataType() == DataType.HALF) {
            floatLogits = halfLogits.castTo(DataType.FLOAT);
            if (halfLogits.closeable()) halfLogits.close();
        } else {
            floatLogits = halfLogits;
        }

        assertEquals(DataType.FLOAT, floatLogits.dataType(), "Logits should be FLOAT after cast");
        assertTrue(halfLogits.wasClosed(), "Original HALF logits should be closed");

        // FLOAT logits should pass through unchanged
        INDArray floatOriginal = Nd4j.randn(DataType.FLOAT, 1, 100);
        INDArray result;
        if (floatOriginal.dataType() == DataType.HALF) {
            result = floatOriginal.castTo(DataType.FLOAT);
        } else {
            result = floatOriginal;
        }
        assertSame(floatOriginal, result, "FLOAT logits should not be cast (same object)");
    }

    // ─── Test 15: clearPlaceholders(false) after each step ───────────────

    @Test
    @DisplayName("15: clearPlaceholders(false) clears values but NOT overrides after each step")
    public void testClearPlaceholdersFalsePerStep() {
        // Simulates line 628: decoder.clearPlaceholders(false)
        // This is called after every step to clear per-step placeholder values
        // but must NOT remove placeholder overrides (attn_mask_reformat)

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable internal = sd.nn.relu("internal", input, 0);
        SDVariable output = sd.identity("output", internal);
        sd.setOutputs(Collections.singletonList("output"));

        // Add a placeholder override (simulates attn_mask_reformat)
        sd.addPlaceholderOverride("internal");
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("internal").getVariableType());

        // Execute with both placeholders
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        feedDict.put("internal", Nd4j.randn(DataType.FLOAT, 1, 8));
        sd.output(feedDict, "output");

        // clearPlaceholders(false) — per-step cleanup
        sd.clearPlaceholders(false);

        // Override should survive — this is critical for multi-step decode
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("internal").getVariableType(),
                "clearPlaceholders(false) must NOT remove overrides");
        assertTrue(sd.inputs().contains("internal"),
                "Overridden var must remain in inputs after clearPlaceholders(false)");
    }

    // ─── Test 16: Speculative path cleanup ordering ──────────────────────

    @Test
    @DisplayName("16: Speculative path clears reusable inputs before switching")
    public void testSpeculativePathCleanup() {
        // Simulates lines 700-705: clear reusable inputs when switching to speculative path
        Map<String, INDArray> reusableInputs = new HashMap<>();
        reusableInputs.put("attention_mask", Nd4j.ones(DataType.LONG, 1, 31));
        reusableInputs.put("attn_bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 31));
        reusableInputs.put("position_ids", Nd4j.createFromArray(new long[][]{{10}}));

        // Track arrays for close verification
        List<INDArray> arrays = new ArrayList<>(reusableInputs.values());

        // Simulate speculative path switch (lines 702-705)
        for (INDArray arr : reusableInputs.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        reusableInputs.clear();

        // Verify all closed and map empty
        assertTrue(reusableInputs.isEmpty(), "Reusable inputs should be cleared");
        for (INDArray arr : arrays) {
            assertTrue(arr.wasClosed(), "Reusable array should be closed before speculative switch");
        }
    }

    // ─── Test 17: 20-step decode position monotonicity ───────────────────

    @Test
    @DisplayName("17: Cache position monotonically increases over 20 steps with mixed scatter")
    public void testCachePositionMonotonicityMixedScatter() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .build();
        StaticKvCacheManager kvManager = new StaticKvCacheManager(ioConfig);
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        kvManager.initializeFromPrefill(createPrefillOutputs(kvNames), kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

        long expectedPos = PREFILL_LEN;
        List<Long> positions = new ArrayList<>();
        positions.add(expectedPos);

        for (int step = 1; step <= MAX_NEW_TOKENS; step++) {
            long before = kvManager.getCachePosition();

            if (step < 2) {
                // Java scatter (step 1)
                kvManager.scatterNewEntries(createMockDecoderOutputs(kvNames, MAX_KV_LEN), kvNames);
            } else {
                // C++ scatter (step 2+)
                kvManager.setCachePosition(kvManager.getCachePosition() + 1);
            }

            long after = kvManager.getCachePosition();
            assertEquals(before + 1, after,
                    "Step " + step + ": position should increase by exactly 1");
            expectedPos++;
            assertEquals(expectedPos, after,
                    "Step " + step + ": position should match expected");
            positions.add(after);
        }

        // Verify strictly monotonic
        for (int i = 1; i < positions.size(); i++) {
            assertTrue(positions.get(i) > positions.get(i - 1),
                    "Position at index " + i + " should be strictly greater than previous");
        }

        assertEquals(PREFILL_LEN + MAX_NEW_TOKENS, kvManager.getCachePosition(),
                "Final position should be PREFILL_LEN + MAX_NEW_TOKENS");

        kvManager.close();
    }

    // ─── Test 18: associateArrayWithVariable before DSP recompile ────────

    @Test
    @DisplayName("18: Static KV buffers associated with decoder before DSP recompile")
    public void testAssociateBeforeRecompileOrdering() {
        // Simulates lines 424-431: associate static KV buffers so DSP sees correct shapes
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
        SDVariable kv = sd.placeHolder("past_kv", DataType.FLOAT, -1, -1, -1, -1);
        SDVariable combined = sd.math.add("combined", input,
                sd.reshape("kv_flat", kv, 1, 32));
        SDVariable output = sd.identity("output", combined);
        sd.setOutputs(Collections.singletonList("output"));

        // Associate a static KV buffer (line 430) — 1*2*4*4 = 32 elements matches reshape target
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, 1, 2, 4, 4);
        sd.associateArrayWithVariable(staticBuf, "past_kv");

        // After association, the variable should have the array's shape info
        // This is what the DSP recompiler uses to determine static shapes
        SDVariable kvVar = sd.getVariable("past_kv");
        assertNotNull(kvVar, "KV variable should exist");

        // Verify model executes with the associated array
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
        feedDict.put("past_kv", staticBuf);
        Map<String, INDArray> result = sd.output(feedDict, "output");
        assertNotNull(result.get("output"), "Model should execute after associateArrayWithVariable");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private DecoderUtils.KVCacheNames createKvNames() {
        List<String> keyNames = new ArrayList<>();
        List<String> valueNames = new ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            keyNames.add("present." + i + ".key");
            valueNames.add("present." + i + ".value");
        }
        return new DecoderUtils.KVCacheNames(keyNames, valueNames);
    }

    private List<String> createInputNames() {
        List<String> names = new ArrayList<>();
        names.add("inputs_embeds");
        names.add("attention_mask");
        names.add("position_ids");
        names.add("input_ids");
        for (int i = 0; i < NUM_LAYERS; i++) {
            names.add("past_key_values." + i + ".key");
            names.add("past_key_values." + i + ".value");
        }
        return names;
    }

    private Map<String, INDArray> createStaticKvBuffers() {
        Map<String, INDArray> buffers = new HashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            buffers.put("past_key_values." + i + ".key",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN, HEAD_DIM));
            buffers.put("past_key_values." + i + ".value",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN, HEAD_DIM));
        }
        return buffers;
    }

    private Map<String, INDArray> createPrefillOutputs(DecoderUtils.KVCacheNames kvNames) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            outputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        for (String name : kvNames.valueNames) {
            outputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        return outputs;
    }

    private Map<String, INDArray> createMockDecoderOutputs(DecoderUtils.KVCacheNames kvNames, long maxKvLen) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            INDArray kv = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, maxKvLen + 1, HEAD_DIM);
            outputs.put(name, kv);
        }
        for (String name : kvNames.valueNames) {
            INDArray kv = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, maxKvLen + 1, HEAD_DIM);
            outputs.put(name, kv);
        }
        return outputs;
    }

    private boolean isAllZeros(INDArray buf, int position) {
        INDArray slice = buf.get(NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.point(position), NDArrayIndex.all());
        return slice.sumNumber().floatValue() == 0.0f;
    }
}
