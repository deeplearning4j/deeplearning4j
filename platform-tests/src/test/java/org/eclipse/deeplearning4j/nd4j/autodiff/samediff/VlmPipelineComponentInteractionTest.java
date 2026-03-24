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
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the 10 untested component interactions in the VLM decode pipeline.
 *
 * Each test is fully independent — creates its own mock decoder, KV buffers,
 * and config state. No shared mutable state between tests.
 *
 * Run: cd platform-tests && mvn test -Dtest=VlmPipelineComponentInteractionTest
 */
@Slf4j
public class VlmPipelineComponentInteractionTest {

    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 8;
    private static final int PREFILL_LEN = 10;
    private static final int MAX_NEW_TOKENS = 20;
    private static final long MAX_KV_LEN = PREFILL_LEN + MAX_NEW_TOKENS;
    private static final float MASK_FILL = -3.4028235e+38f;

    // ─── Test 1: ModelIOConfig.discover() accuracy ──────────────────────

    @Test
    @DisplayName("1: ModelIOConfig.discover() finds correct names from SameDiff graph")
    public void testModelIOConfigDiscoverAccuracy() {
        SameDiff sd = SameDiff.create();

        // Create inputs matching SmolDocling naming
        sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, 32);
        sd.placeHolder("attention_mask", DataType.LONG, 1, -1);
        sd.placeHolder("position_ids", DataType.LONG, 1, 1);
        sd.placeHolder("past_key_values.0.key", DataType.FLOAT, 1, NUM_HEADS, -1, HEAD_DIM);
        sd.placeHolder("past_key_values.0.value", DataType.FLOAT, 1, NUM_HEADS, -1, HEAD_DIM);
        sd.placeHolder("past_key_values.1.key", DataType.FLOAT, 1, NUM_HEADS, -1, HEAD_DIM);
        sd.placeHolder("past_key_values.1.value", DataType.FLOAT, 1, NUM_HEADS, -1, HEAD_DIM);

        // Create the attn_mask_reformat node (internal variable, not a placeholder)
        SDVariable attnReformat = sd.constant("/model/attn_mask_reformat/Tile/output_0",
                Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1));

        // Create logits output
        SDVariable logits = sd.constant("logits", Nd4j.zeros(DataType.FLOAT, 1, 1, 100));

        // Create present KV outputs
        sd.constant("present.0.key", Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM));
        sd.constant("present.0.value", Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM));
        sd.constant("present.1.key", Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM));
        sd.constant("present.1.value", Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM));

        // Set outputs so findLogitsOutputName can detect them
        sd.setOutputs(Arrays.asList("logits", "present.0.key", "present.0.value",
                "present.1.key", "present.1.value"));

        ModelIOConfig config = ModelIOConfig.discover(sd);

        assertEquals("inputs_embeds", config.getInputEmbeddingsName(),
                "discover() should find inputs_embeds");
        assertEquals("attention_mask", config.getAttentionMaskName(),
                "discover() should find attention_mask");
        assertEquals("position_ids", config.getPositionIdsName(),
                "discover() should find position_ids");
        assertEquals("past_key_values.", config.getKvCachePrefix(),
                "discover() should detect past_key_values. prefix");
        assertEquals("logits", config.getLogitsOutputName(),
                "discover() should find logits output");
        assertEquals("/model/attn_mask_reformat/Tile/output_0", config.getAttnMaskReformatOutput(),
                "discover() should find attn_mask_reformat node");

        // Verify present→past name mapping
        assertEquals("past_key_values.0.key", config.presentToInputName("present.0.key"),
                "presentToInputName should map present→past_key_values");
        assertEquals("present.0.key", config.inputToPresentName("past_key_values.0.key"),
                "inputToPresentName should map past_key_values→present");

        // Verify query methods
        assertTrue(config.isInputEmbeddings("inputs_embeds"));
        assertTrue(config.isAttentionMask("attention_mask"));
        assertTrue(config.isPositionIds("position_ids"));
        assertTrue(config.isKvCacheInput("past_key_values.0.key"));
        assertFalse(config.isKvCacheInput("attention_mask"));
        assertFalse(config.isInputEmbeddings("attention_mask"));
    }

    // ─── Test 2: BenchmarkConfigApplier.resetModelState() completeness ──

    @Test
    @DisplayName("2: resetModelState() clears all decoder mutations")
    public void testResetModelStateClearsAllMutations() {
        SameDiff sd = SameDiff.create();

        // Create a simple graph with an internal node that can be overridden
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
        SDVariable w = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable internalNode = sd.mmul("internal_result", input, w);
        SDVariable output = sd.nn.relu("output", internalNode, 0);
        sd.setOutputs(Collections.singletonList("output"));

        // Simulate the mutations that the decode loop performs:

        // 1. Associate arrays with variables (use matching shape for the placeholder)
        INDArray inputBuf = Nd4j.zeros(DataType.FLOAT, 1, 32);
        sd.associateArrayWithVariable(inputBuf, "input");

        // 2. Add placeholder override (converts ARRAY/CONSTANT → PLACEHOLDER)
        sd.addPlaceholderOverride("internal_result");

        // 3. Compile a DSP plan
        sd.setDspAutoCompileEnabled(true);

        // Now reset — resetModelState now calls clearPlaceholderOverrides()
        BenchmarkConfigApplier.resetModelState(sd);

        SDVariable internalVar = sd.getVariable("internal_result");
        assertNotNull(internalVar, "Variable should still exist after reset");

        // Verify placeholder override is cleared by resetModelState
        assertNotEquals(VariableType.PLACEHOLDER, internalVar.getVariableType(),
                "Placeholder override should be cleared after resetModelState");

        // Verify the model is executable after reset
        Map<String, INDArray> checkFeed = new HashMap<>();
        checkFeed.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
        Map<String, INDArray> checkResult = sd.output(checkFeed, "output");
        assertNotNull(checkResult.get("output"),
                "Model should be executable after resetModelState");
    }

    // ─── Test 3: BenchmarkConfigApplier.apply() flag ordering ───────────

    @Test
    @DisplayName("3: apply() resets all flags before applying, no leaks between configs")
    public void testApplyFlagOrderingNoLeakBetweenConfigs() {
        Environment env = Nd4j.getEnvironment();

        // Apply a config with Triton + cuBLAS TF32 + capture workspace
        BenchmarkConfig config1 = BenchmarkConfig.create("CONFIG_1")
                .tritonIncludeTypes("NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .cublasTf32(true)
                .tritonTf32(true)
                .tritonNumWarps(4)
                .tritonNumStages(1)
                .dspBatchedGemm(true)
                .maxTokens(100);

        BenchmarkConfigApplier.apply(config1);

        // Verify config1 is applied
        assertTrue(env.cublasTf32Enabled(), "cuBLAS TF32 should be ON after config1");
        assertTrue(env.tritonTf32Enabled(), "Triton TF32 should be ON after config1");
        assertTrue(env.tritonGraphCapture(), "Graph capture should be ON after config1");
        assertTrue(env.tritonConsolidatedArgTable(), "Consolidated arg table should be ON");
        assertTrue(env.tritonArgDirtyTracking(), "Dirty tracking should be ON");
        assertTrue(env.dspBatchedGemm(), "Batched GEMM should be ON");
        assertEquals(4, env.tritonNumWarps(), "Warps should be 4");
        assertEquals(1, env.tritonNumStages(), "Stages should be 1");

        // Now apply a MINIMAL config (SLOT_BY_SLOT, no Triton)
        BenchmarkConfig config2 = BenchmarkConfig.create("CONFIG_2")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .cublasTf32(false)
                .tritonTf32(false)
                .maxTokens(10);

        BenchmarkConfigApplier.apply(config2);

        // ALL Triton flags from config1 must be reset to defaults
        assertFalse(env.cublasTf32Enabled(), "cuBLAS TF32 should be OFF after config2");
        assertFalse(env.tritonTf32Enabled(), "Triton TF32 should be OFF after config2");
        assertFalse(env.tritonGraphCapture(), "Graph capture should be OFF after config2");
        assertFalse(env.tritonConsolidatedArgTable(), "Consolidated arg table should be OFF");
        assertFalse(env.tritonArgDirtyTracking(), "Dirty tracking should be OFF");
        assertFalse(env.dspBatchedGemm(), "Batched GEMM should be OFF");
        assertEquals("", env.tritonIncludeTypes(), "Include types should be empty");
        assertFalse(env.tritonSectionFusion(), "Section fusion should be OFF");
        assertFalse(env.tritonCompileAll(), "Compile all should be OFF");
        assertFalse(env.tritonAllowFallbackCapture(), "Fallback capture should be OFF");
    }

    // ─── Test 4: Post-prefill recompile with attn_mask_reformat override ─

    @Test
    @DisplayName("4: addPlaceholderOverride converts variable and updates input list")
    public void testPlaceholderOverrideConvertsAndUpdatesInputs() {
        SameDiff sd = SameDiff.create();

        // Simulate a decoder graph with attn_mask_reformat as an internal node
        SDVariable attentionMask = sd.placeHolder("attention_mask", DataType.LONG, 1, -1);
        // The attn_mask_reformat subgraph converts 1D mask → 4D bias
        SDVariable castedMask = sd.castTo("attn_mask_cast", attentionMask, DataType.FLOAT);
        SDVariable reshapedMask = sd.reshape("attn_mask_reshape", castedMask, 1, 1, 1, -1);
        SDVariable tiledMask = sd.identity("/model/attn_mask_reformat/Tile/output_0", reshapedMask);

        // Use a 4D input that is compatible with the 4D tiled mask for the add op
        SDVariable inputEmbeds = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, 1, -1);
        SDVariable combined = sd.math.add("combined", inputEmbeds, tiledMask);
        SDVariable output = sd.identity("logits", combined);
        sd.setOutputs(Collections.singletonList("logits"));

        // Before override: attn_mask_reformat/Tile/output_0 should NOT be an input
        List<String> inputsBefore = sd.inputs();
        assertFalse(inputsBefore.contains("/model/attn_mask_reformat/Tile/output_0"),
                "Before override, attn_mask_reformat should not be an input");
        assertTrue(inputsBefore.contains("attention_mask"),
                "Before override, attention_mask should be an input");

        // Add placeholder override — this is what the decode loop does post-prefill
        sd.addPlaceholderOverride("/model/attn_mask_reformat/Tile/output_0");
        sd.getVariable("/model/attn_mask_reformat/Tile/output_0").setShape(-1, -1, -1, -1);

        // After override: attn_mask_reformat/Tile/output_0 should be an input
        List<String> inputsAfter = sd.inputs();
        assertTrue(inputsAfter.contains("/model/attn_mask_reformat/Tile/output_0"),
                "After override, attn_mask_reformat should be an input (PLACEHOLDER)");

        // Verify variable type changed
        SDVariable overriddenVar = sd.getVariable("/model/attn_mask_reformat/Tile/output_0");
        assertEquals(VariableType.PLACEHOLDER, overriddenVar.getVariableType(),
                "Overridden variable should be PLACEHOLDER type");

        // Verify we can provide a value for the overridden placeholder
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("attention_mask", Nd4j.ones(DataType.LONG, 1, 5));
        feedDict.put("inputs_embeds", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 5));
        feedDict.put("/model/attn_mask_reformat/Tile/output_0",
                Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 5));

        // This should not throw — the override should let us pass the 4D mask directly
        Map<String, INDArray> outputs = sd.output(feedDict, "logits");
        assertNotNull(outputs.get("logits"), "Should produce logits with overridden placeholder");
    }

    // ─── Test 5: KvCacheManager ↔ cppScatterThisStep position sync ─────

    @Test
    @DisplayName("5: KvCacheManager position sync across Java/C++ scatter transition")
    public void testKvCacheManagerPositionSyncTransition() {
        ModelIOConfig ioConfig = ModelIOConfig.builder().build();
        StaticKvCacheManager kvManager = new StaticKvCacheManager(ioConfig);

        // Create mock prefill outputs
        Map<String, INDArray> prefillOutputs = new HashMap<>();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        for (String name : kvNames.keyNames) {
            prefillOutputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        for (String name : kvNames.valueNames) {
            prefillOutputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }

        // Initialize from prefill
        kvManager.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);
        assertEquals(PREFILL_LEN, kvManager.getCachePosition(),
                "After prefill, cachePos should be PREFILL_LEN");
        assertEquals(MAX_KV_LEN, kvManager.getMaxKvLen(),
                "maxKvLen should be PREFILL_LEN + MAX_NEW_TOKENS");

        // Step 1: Java scatter (simulates output() path, step < 2)
        Map<String, INDArray> step1Outputs = createMockDecoderOutputs(kvNames);
        kvManager.scatterNewEntries(step1Outputs, kvNames);
        assertEquals(PREFILL_LEN + 1, kvManager.getCachePosition(),
                "After step 1 Java scatter, position should advance by 1");

        // The decode loop would call advanceKvCachePosition() on the C++ side here
        // to keep C++ in sync. Test that KvCacheManager's position is correct.

        // Step 2: C++ scatter (simulates outputDirect() path, step >= 2)
        // When C++ handles scatter, the loop calls setCachePosition manually
        kvManager.setCachePosition(kvManager.getCachePosition() + 1);
        assertEquals(PREFILL_LEN + 2, kvManager.getCachePosition(),
                "After step 2 C++ scatter, position should advance by 1");

        // Step 3: C++ scatter again
        kvManager.setCachePosition(kvManager.getCachePosition() + 1);
        assertEquals(PREFILL_LEN + 3, kvManager.getCachePosition(),
                "After step 3 C++ scatter, position should advance by 1");

        // Verify position monotonically increases over 10 more steps
        for (int i = 0; i < 10; i++) {
            long before = kvManager.getCachePosition();
            kvManager.setCachePosition(kvManager.getCachePosition() + 1);
            assertEquals(before + 1, kvManager.getCachePosition(),
                    "Position must increase by exactly 1 at step " + (i + 4));
        }

        kvManager.close();
    }

    // ─── Test 6: setNextDecodeToken at step 1 safety ────────────────────

    @Test
    @DisplayName("6: setNextDecodeToken at step 1 does not corrupt input arrays")
    public void testSetNextDecodeTokenStep1Safety() {
        // This test verifies that writing device-side inputs (what setNextDecodeToken does)
        // doesn't corrupt the Java-side input arrays that output() uses via feedDict.
        // Since we can't call the actual C++ method in a unit test, we test the Java-side
        // input map construction to verify it always uses the correct values.

        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .build();
        List<String> inputNames = createInputNames();

        // Simulate the reusable inputs cache
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();

        // Step 0 (prefill): build initial input map using static KV mode
        INDArray prefillEmbeds = Nd4j.randn(DataType.FLOAT, 1, PREFILL_LEN, 32);
        INDArray prefillIds = Nd4j.ones(DataType.LONG, 1, PREFILL_LEN);
        Map<String, INDArray> step0Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, prefillEmbeds, prefillIds,
                0, PREFILL_LEN, staticKvBuffers, MAX_KV_LEN, 0,
                true, 32, reusableInputs, true, false);

        assertNotNull(step0Map.get("attention_mask"), "Step 0 should have attention_mask");

        // Step 1 (first decode, uses output() not outputDirect()):
        // nativeDecodeInputs=true simulates what happens when C++ decode inputs are configured
        // but we're still on step 1 which uses output()
        INDArray decodeEmbeds = Nd4j.randn(DataType.FLOAT, 1, 1, 32);
        INDArray decodeIds = Nd4j.createFromArray(new long[][]{{42}});
        long pastSeqLen = PREFILL_LEN;
        long cachePos = PREFILL_LEN;

        Map<String, INDArray> step1Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, decodeEmbeds, decodeIds,
                pastSeqLen, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, 32, reusableInputs, true, true);

        // The key property: Java-side arrays must have correct values regardless of
        // whether setNextDecodeToken will also write device memory.

        // Position IDs should reflect pastSeqLen
        INDArray posIds = step1Map.get("position_ids");
        assertNotNull(posIds, "position_ids should be in the input map");
        assertEquals(pastSeqLen, posIds.getLong(0, 0),
                "position_ids should be pastSeqLen at step 1");

        // Attention mask should have 1s for filled positions
        INDArray mask = step1Map.get("attention_mask");
        assertNotNull(mask, "attention_mask should be in the input map");
        long totalSeqLen = MAX_KV_LEN + 1;
        assertEquals(totalSeqLen, mask.size(1),
                "attention_mask should be [1, maxKvLen + currentSeqLen]");

        // Input IDs should be the decode token
        INDArray ids = step1Map.get("input_ids");
        assertNotNull(ids, "input_ids should be in the input map");
        assertEquals(42, ids.getLong(0, 0),
                "input_ids should match decode token");
    }

    // ─── Test 7: 4D attention bias initial construction + accumulation ──

    @Test
    @DisplayName("7: 4D attention bias construction and step-by-step accumulation")
    public void testAttentionBiasConstructionAndAccumulation() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput("attn_mask_reformat")
                .build();

        List<String> inputNames = new ArrayList<>(createInputNames());
        inputNames.add("attn_mask_reformat");

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        Map<String, INDArray> reusableInputs = new HashMap<>();
        INDArray embeds = Nd4j.randn(DataType.FLOAT, 1, 1, 32);
        INDArray ids = Nd4j.createFromArray(new long[][]{{1}});

        // Step 1: first decode after prefill (cachePos = PREFILL_LEN)
        long cachePos = PREFILL_LEN;
        Map<String, INDArray> step1Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, embeds, ids,
                cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, 32, reusableInputs, true, false);

        INDArray bias = step1Map.get("attn_mask_reformat");
        assertNotNull(bias, "Bias should be present when attnMaskReformatOutput is in input names");
        assertEquals(4, bias.rank(), "Bias should be 4D [1, 1, currentSeqLen, totalSeqLen]");
        long totalSeqLen = MAX_KV_LEN + 1;
        assertEquals(totalSeqLen, bias.size(3), "Bias dim 3 should be maxKvLen + currentSeqLen");

        // Verify: positions [0..cachePos-1] should be 0.0 (attend)
        for (int k = 0; k < cachePos; k++) {
            assertEquals(0.0f, bias.getFloat(0, 0, 0, k), 1e-6,
                    "Position " + k + " should be unmasked (0.0) — within filled KV");
        }
        // Verify: positions [cachePos..maxKvLen-1] should be MASK_FILL (blocked)
        for (int k = (int) cachePos; k < MAX_KV_LEN; k++) {
            assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, k), 1e-6,
                    "Position " + k + " should be masked — unfilled padding");
        }
        // Verify: last position (current token in concat'd region) should be 0.0
        assertEquals(0.0f, bias.getFloat(0, 0, 0, (int) totalSeqLen - 1), 1e-6,
                "Last position should be unmasked — current token");

        // Step 2: advance cachePos, rebuild with reuse
        cachePos = PREFILL_LEN + 1;
        Map<String, INDArray> step2Map = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, embeds, ids,
                cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, 32, reusableInputs, true, false);

        INDArray bias2 = step2Map.get("attn_mask_reformat");
        // Position PREFILL_LEN should now be unmasked (was filled by step 1's scatter)
        assertEquals(0.0f, bias2.getFloat(0, 0, 0, PREFILL_LEN), 1e-6,
                "After step 2, position PREFILL_LEN should be unmasked via putScalar accumulation");
        // Position PREFILL_LEN+1 should still be masked (not yet filled)
        assertEquals(MASK_FILL, bias2.getFloat(0, 0, 0, PREFILL_LEN + 1), 1e-6,
                "Position PREFILL_LEN+1 should still be masked at step 2");

        // Verify 10-step accumulation: each step unmasks exactly one more position
        for (int step = 2; step < 12; step++) {
            cachePos = PREFILL_LEN + step;
            DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, null, embeds, ids,
                    cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                    true, 32, reusableInputs, true, false);
        }
        INDArray finalBias = reusableInputs.get("attn_mask_reformat");
        // All positions [0..PREFILL_LEN+10] should be unmasked
        for (int k = 0; k <= PREFILL_LEN + 10; k++) {
            assertEquals(0.0f, finalBias.getFloat(0, 0, 0, k), 1e-6,
                    "After 12 steps, position " + k + " should be unmasked");
        }
        // Position PREFILL_LEN+11 should still be masked (12 = current step, not yet scattered)
        if (PREFILL_LEN + 11 < MAX_KV_LEN) {
            assertEquals(MASK_FILL, finalBias.getFloat(0, 0, 0, PREFILL_LEN + 11), 1e-6,
                    "Position beyond last scatter should still be masked");
        }
    }

    // ─── Test 8: ensureLocation(DEVICE) after putScalar ─────────────────

    @Test
    @DisplayName("8: ensureLocation(DEVICE) after putScalar triggers H2D sync")
    public void testEnsureLocationAfterPutScalar() {
        // Create a FLOAT array (simulates the 4D attention bias)
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 20);
        bias.assign(MASK_FILL); // Fill with mask value

        // putScalar writes to host buffer
        bias.putScalar(new long[]{0, 0, 0, 5}, 0.0f);

        // Verify the value is visible on host
        assertEquals(0.0f, bias.getFloat(0, 0, 0, 5), 1e-6,
                "putScalar should update the array value");

        // ensureLocation(DEVICE) should trigger H2D sync
        Nd4j.getAffinityManager().ensureLocation(bias,
                org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

        // After sync, reading the value should still be correct
        // (this round-trips through D2H if needed on read)
        assertEquals(0.0f, bias.getFloat(0, 0, 0, 5), 1e-6,
                "Value should survive ensureLocation round-trip");
        assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, 6), 1e-6,
                "Other positions should still have MASK_FILL after sync");

        // Do the same for LONG array (simulates 1D attention mask)
        INDArray mask = Nd4j.zeros(DataType.LONG, 1, 20);
        mask.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                NDArrayIndex.point(0), NDArrayIndex.point(7)
        }, Nd4j.scalar(DataType.LONG, 1));

        Nd4j.getAffinityManager().ensureLocation(mask,
                org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

        assertEquals(1L, mask.getLong(0, 7),
                "LONG mask value should survive ensureLocation round-trip");
        assertEquals(0L, mask.getLong(0, 8),
                "Unmodified LONG mask position should remain 0");
    }

    // ─── Test 9: Logits from outputDirect vs output ─────────────────────

    @Test
    @DisplayName("9: output() and outputDirect() produce identical logits")
    public void testOutputVsOutputDirectLogitsEquivalence() {
        // Build a simple model: logits = matmul(input, weights)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        INDArray weightsArr = Nd4j.randn(DataType.FLOAT, 8, 16);
        SDVariable weights = sd.constant("weights", weightsArr);
        SDVariable logits = sd.mmul("logits", input, weights);
        sd.setOutputs(Collections.singletonList("logits"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 8);
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", inputArr);

        // output() path (step 0-1 in decode loop)
        Map<String, INDArray> result1 = sd.output(feedDict, "logits");
        INDArray logits1 = result1.get("logits").dup(); // dup to detach from session

        // outputDirect() path (step 2+ in decode loop)
        Map<String, INDArray> result2 = sd.outputDirect(feedDict, "logits");
        INDArray logits2 = result2.get("logits").dup();

        // Both should produce identical results
        assertEquals(logits1.shape()[0], logits2.shape()[0], "Batch dim should match");
        assertEquals(logits1.shape()[1], logits2.shape()[1], "Feature dim should match");

        for (int i = 0; i < 16; i++) {
            assertEquals(logits1.getFloat(0, i), logits2.getFloat(0, i), 1e-4,
                    "Logit at position " + i + " should match between output() and outputDirect()");
        }

        // Verify logits from outputDirect are usable for argmax (not closed/stale)
        INDArray argmax2 = Nd4j.argMax(logits2, 1);
        assertTrue(argmax2.getLong(0) >= 0 && argmax2.getLong(0) < 16,
                "argMax on outputDirect logits should produce valid index");

        // Verify both argmax results agree
        INDArray argmax1 = Nd4j.argMax(logits1, 1);
        assertEquals(argmax1.getLong(0), argmax2.getLong(0),
                "argMax should agree between output() and outputDirect() logits");
    }

    // ─── Test 10: clearDynamicShapePlanCache + clearAllCaches ───────────

    @Test
    @DisplayName("10: clearDynamicShapePlanCache during decode recompile invalidates old plan")
    public void testClearDspCacheDuringRecompile() {
        // Build a simple model
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable output = sd.mmul("output", input, weights);
        sd.setOutputs(Collections.singletonList("output"));

        // First execution — establishes session and any cached state
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        sd.output(feedDict, "output");

        // Simulate the post-prefill recompile sequence:
        // 1. Clear DSP plan cache
        sd.clearDynamicShapePlanCache();
        // 2. Clear session caches
        sd.getOrCreateSession().clearAllCaches();

        // After clearing, the model should still be executable
        Map<String, INDArray> result = sd.output(feedDict, "output");
        assertNotNull(result.get("output"), "Model should still produce output after cache clear");
        assertFalse(result.get("output").isEmpty(), "Output should not be empty");

        // Verify the output is valid (not NaN, not all zeros)
        INDArray out = result.get("output");
        boolean hasNonZero = false;
        for (int i = 0; i < out.length(); i++) {
            assertFalse(Float.isNaN(out.getFloat(i)),
                    "Output should not contain NaN after recompile");
            if (out.getFloat(i) != 0.0f) hasNonZero = true;
        }
        assertTrue(hasNonZero, "Output should have non-zero values (valid matmul result)");
    }

    // ─── Test: 1D attention mask accumulation matches 4D bias ───────────

    @Test
    @DisplayName("11: 1D attention mask and 4D bias unmask same positions each step")
    public void testMaskAndBiasPositionConsistency() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput("attn_bias")
                .build();

        List<String> inputNames = new ArrayList<>(createInputNames());
        inputNames.add("attn_bias");

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        Map<String, INDArray> reusableInputs = new HashMap<>();
        INDArray embeds = Nd4j.randn(DataType.FLOAT, 1, 1, 32);
        INDArray ids = Nd4j.createFromArray(new long[][]{{1}});

        // Run 15 decode steps and verify mask/bias consistency at each step
        for (int step = 0; step < 15; step++) {
            long cachePos = PREFILL_LEN + step;
            Map<String, INDArray> inputMap = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, null, embeds, ids,
                    cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                    true, 32, reusableInputs, true, false);

            INDArray mask = inputMap.get("attention_mask");
            INDArray bias = inputMap.get("attn_bias");

            assertNotNull(mask, "Step " + step + ": mask should not be null");
            assertNotNull(bias, "Step " + step + ": bias should not be null");

            // Check consistency: mask=1 should correspond to bias=0.0,
            // mask=0 should correspond to bias=MASK_FILL
            long totalSeqLen = MAX_KV_LEN + 1;
            for (int k = 0; k < (int) Math.min(totalSeqLen, MAX_KV_LEN); k++) {
                long maskVal = mask.getLong(0, k);
                float biasVal = bias.getFloat(0, 0, 0, k);

                if (maskVal == 1) {
                    assertEquals(0.0f, biasVal, 1e-6,
                            "Step " + step + " pos " + k + ": mask=1 must have bias=0.0 (attend)");
                } else {
                    assertEquals(MASK_FILL, biasVal, 1e-6,
                            "Step " + step + " pos " + k + ": mask=0 must have bias=MASK_FILL (block)");
                }
            }
        }
    }

    // ─── Test: StaticKvCacheManager scatter writes correct positions ─────

    @Test
    @DisplayName("12: StaticKvCacheManager Java scatter writes to correct buffer positions")
    public void testStaticKvCacheManagerScatterCorrectPositions() {
        StaticKvCacheManager kvManager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        // Create prefill outputs with distinct values per position
        Map<String, INDArray> prefillOutputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM);
            // Fill each position with its index so we can verify later
            for (int pos = 0; pos < PREFILL_LEN; pos++) {
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).assign(pos + 0.5f);
            }
            prefillOutputs.put(name, kv);
        }
        for (String name : kvNames.valueNames) {
            INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM);
            for (int pos = 0; pos < PREFILL_LEN; pos++) {
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).assign(pos + 100.5f);
            }
            prefillOutputs.put(name, kv);
        }

        kvManager.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

        // Verify prefill data was written correctly
        Map<String, INDArray> buffers = kvManager.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : buffers.entrySet()) {
            INDArray buf = e.getValue();
            boolean isKey = e.getKey().endsWith(".key");
            float baseVal = isKey ? 0.5f : 100.5f;
            for (int pos = 0; pos < PREFILL_LEN; pos++) {
                float expected = pos + baseVal;
                assertEquals(expected, buf.getFloat(0, 0, pos, 0), 1e-4,
                        e.getKey() + " position " + pos + " should have prefill data");
            }
            // Padding positions should be zeros
            assertEquals(0.0f, buf.getFloat(0, 0, PREFILL_LEN, 0), 1e-6,
                    e.getKey() + " padding position should be zero");
        }

        // Scatter 5 decode steps with distinct values
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> decoderOutputs = new HashMap<>();
            float stepVal = (step + 1) * 1000.0f;
            for (String name : kvNames.keyNames) {
                INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
                // New entry is at the last position in the concat'd output
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(stepVal);
                decoderOutputs.put(name, kv);
            }
            for (String name : kvNames.valueNames) {
                INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(MAX_KV_LEN), NDArrayIndex.all()).assign(stepVal + 0.1f);
                decoderOutputs.put(name, kv);
            }

            kvManager.scatterNewEntries(decoderOutputs, kvNames);

            // Verify the entry landed at the correct position
            long expectedPos = PREFILL_LEN + step;
            for (Map.Entry<String, INDArray> e : buffers.entrySet()) {
                boolean isKey = e.getKey().endsWith(".key");
                float expectedVal = isKey ? stepVal : stepVal + 0.1f;
                assertEquals(expectedVal, e.getValue().getFloat(0, 0, (int) expectedPos, 0), 1e-4,
                        e.getKey() + " scatter step " + step + " should be at position " + expectedPos);
            }
        }

        assertEquals(PREFILL_LEN + 5, kvManager.getCachePosition(),
                "After 5 scatters, position should be PREFILL_LEN + 5");

        kvManager.close();
    }

    // ─── Test: ModelIOConfig query methods don't false-match ─────────────

    @Test
    @DisplayName("13: ModelIOConfig query methods are mutually exclusive")
    public void testModelIOConfigQueryMethodsMutuallyExclusive() {
        ModelIOConfig config = ModelIOConfig.builder()
                .inputEmbeddingsName("inputs_embeds")
                .inputIdsName("input_ids")
                .attentionMaskName("attention_mask")
                .causalMaskName("_causal_mask")
                .positionIdsName("position_ids")
                .kvCachePrefix("past_key_values.")
                .build();

        // Each name should match exactly ONE query method
        String[] allNames = {
                "inputs_embeds", "input_ids", "attention_mask",
                "_causal_mask", "position_ids", "past_key_values.0.key"
        };

        for (String name : allNames) {
            int matchCount = 0;
            if (config.isInputEmbeddings(name)) matchCount++;
            if (config.isInputIds(name)) matchCount++;
            if (config.isAttentionMask(name)) matchCount++;
            if (config.isCausalMask(name)) matchCount++;
            if (config.isPositionIds(name)) matchCount++;
            if (config.isKvCacheInput(name)) matchCount++;

            assertEquals(1, matchCount,
                    "Name '" + name + "' should match exactly 1 query method, matched " + matchCount);
        }

        // past_key_values.0.value also matches isKvCacheInput (not isInputEmbeddings etc.)
        assertTrue(config.isKvCacheInput("past_key_values.0.value"));
        assertFalse(config.isInputEmbeddings("past_key_values.0.value"));

        // Unknown name should match nothing
        int unknownMatches = 0;
        if (config.isInputEmbeddings("unknown_var")) unknownMatches++;
        if (config.isInputIds("unknown_var")) unknownMatches++;
        if (config.isAttentionMask("unknown_var")) unknownMatches++;
        if (config.isCausalMask("unknown_var")) unknownMatches++;
        if (config.isPositionIds("unknown_var")) unknownMatches++;
        if (config.isKvCacheInput("unknown_var")) unknownMatches++;
        assertEquals(0, unknownMatches, "Unknown var should match no query method");
    }

    // ─── Test: Multiple resetModelState calls are idempotent ────────────

    @Test
    @DisplayName("14: Multiple resetModelState calls are safe and idempotent")
    public void testResetModelStateIdempotent() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable output = sd.mmul("output", input, weights);
        sd.setOutputs(Collections.singletonList("output"));

        // Execute to create session state
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        sd.output(feedDict, "output");

        // Reset multiple times — should not crash
        for (int i = 0; i < 5; i++) {
            BenchmarkConfigApplier.resetModelState(sd);
        }

        // Model should still be functional after multiple resets
        Map<String, INDArray> result = sd.output(feedDict, "output");
        assertNotNull(result.get("output"), "Model should work after multiple resets");
        assertFalse(Float.isNaN(result.get("output").getFloat(0)),
                "Output should not be NaN after multiple resets");
    }

    // ─── Test: Placeholder override survives clearDynamicShapePlanCache ──

    @Test
    @DisplayName("15: Placeholder override persists through clearDynamicShapePlanCache")
    public void testPlaceholderOverrideSurvivesCacheClear() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable intermediate = sd.nn.relu("intermediate", input, 0);
        SDVariable output = sd.identity("output", intermediate);
        sd.setOutputs(Collections.singletonList("output"));

        // Add placeholder override (simulates attn_mask_reformat override)
        sd.addPlaceholderOverride("intermediate");

        // Verify it's a placeholder now
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "Should be PLACEHOLDER after override");

        // Clear DSP plan cache (done during post-prefill recompile)
        sd.clearDynamicShapePlanCache();

        // The placeholder override should survive cache clearing
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "Placeholder override should survive clearDynamicShapePlanCache");

        // Should still be in the inputs list
        assertTrue(sd.inputs().contains("intermediate"),
                "Overridden var should still be in inputs after cache clear");
    }

    // ─── Test: clearPlaceholders(true) removes overrides ────────────────

    @Test
    @DisplayName("16: clearPlaceholders(true) vs clearPlaceholderOverrides() behavior")
    public void testClearPlaceholdersRemovesOverrides() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable intermediate = sd.nn.relu("intermediate", input, 0);
        SDVariable output = sd.identity("output", intermediate);
        sd.setOutputs(Collections.singletonList("output"));

        // Override
        sd.addPlaceholderOverride("intermediate");
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType());
        assertTrue(sd.inputs().contains("intermediate"));

        // KEY FINDING: clearPlaceholders(true) only clears placeholder VALUES
        // (the INDArrays stored per-thread), NOT the placeholder type OVERRIDES.
        // This is by design — clearPlaceholders manages the feed dict cache,
        // while clearPlaceholderOverrides manages the variable type changes.
        sd.clearPlaceholders(true);

        // The override should STILL be present after clearPlaceholders(true)
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "clearPlaceholders(true) does NOT remove overrides — only clears values");

        // To remove overrides, must call clearPlaceholderOverrides() explicitly
        sd.clearPlaceholderOverrides();
        assertNotEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "After clearPlaceholderOverrides(), override should be removed");

        // The model should be executable without providing the overridden placeholder
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        Map<String, INDArray> result = sd.output(feedDict, "output");
        assertNotNull(result.get("output"),
                "Model should work after clearing placeholder overrides");
    }

    // ─── Test: KvCacheManager close releases buffers ────────────────────

    @Test
    @DisplayName("17: KvCacheManager.close() releases all buffer memory")
    public void testKvCacheManagerCloseReleasesBuffers() {
        StaticKvCacheManager kvManager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        Map<String, INDArray> prefillOutputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            prefillOutputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        for (String name : kvNames.valueNames) {
            prefillOutputs.put(name, Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }

        kvManager.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);
        assertTrue(kvManager.isInitialized(), "Should be initialized after prefill");

        // Get buffer references before close
        Map<String, INDArray> buffers = kvManager.getStaticKvBuffers();
        List<INDArray> bufferRefs = new ArrayList<>(buffers.values());
        assertEquals(NUM_LAYERS * 2, bufferRefs.size(),
                "Should have NUM_LAYERS * 2 buffers (key + value)");

        // Close
        kvManager.close();

        // Verify closed state
        assertFalse(kvManager.isInitialized(), "Should not be initialized after close");
        assertNull(kvManager.getStaticKvBuffers(), "Buffers map should be null after close");

        // Verify the buffer arrays were actually closed
        for (INDArray buf : bufferRefs) {
            assertTrue(buf.wasClosed(), "Buffer should be closed after KvCacheManager.close()");
        }
    }

    // ─── Test: Bias off-by-one at cachePos boundary ─────────────────────

    @Test
    @DisplayName("18: Bias putScalar at cachePos-1 correctly unmasks the PREVIOUS step's position")
    public void testBiasOffByOneAtCachePosBoundary() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput("attn_bias")
                .build();

        List<String> inputNames = new ArrayList<>(createInputNames());
        inputNames.add("attn_bias");

        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        Map<String, INDArray> reusableInputs = new HashMap<>();
        INDArray embeds = Nd4j.randn(DataType.FLOAT, 1, 1, 32);
        INDArray ids = Nd4j.createFromArray(new long[][]{{1}});

        // Step where cachePos = PREFILL_LEN (first decode)
        // The bias should NOT unmask PREFILL_LEN yet (that position hasn't been
        // scattered yet — it happens AFTER this input map is built)
        long cachePos = PREFILL_LEN;
        DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, embeds, ids,
                cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, 32, reusableInputs, true, false);

        INDArray bias = reusableInputs.get("attn_bias");
        // At cachePos = PREFILL_LEN: positions [0..PREFILL_LEN-1] should be unmasked
        assertEquals(0.0f, bias.getFloat(0, 0, 0, PREFILL_LEN - 1), 1e-6,
                "Position PREFILL_LEN-1 should be unmasked (filled during prefill)");
        // Position PREFILL_LEN should still be masked (not yet filled by this step's scatter)
        assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, PREFILL_LEN), 1e-6,
                "Position PREFILL_LEN should be masked (current step hasn't scattered yet)");

        // Step where cachePos = PREFILL_LEN + 1 (second decode)
        // Now cachePos-1 = PREFILL_LEN, so PREFILL_LEN gets unmasked
        cachePos = PREFILL_LEN + 1;
        DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, null, embeds, ids,
                cachePos, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                true, 32, reusableInputs, true, false);

        assertEquals(0.0f, bias.getFloat(0, 0, 0, PREFILL_LEN), 1e-6,
                "After cachePos advances to PREFILL_LEN+1, PREFILL_LEN should be unmasked (cachePos-1)");
        assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, PREFILL_LEN + 1), 1e-6,
                "Position PREFILL_LEN+1 should still be masked (not yet scattered)");
    }

    // ─── Test: KvCacheManager multiple scatter with distinct data ────────

    @Test
    @DisplayName("19: scatterMultipleEntries writes correct contiguous range")
    public void testScatterMultipleEntriesContiguousRange() {
        StaticKvCacheManager kvManager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        Map<String, INDArray> prefillOutputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            prefillOutputs.put(name, Nd4j.ones(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        for (String name : kvNames.valueNames) {
            prefillOutputs.put(name, Nd4j.ones(DataType.FLOAT, 1, NUM_HEADS, PREFILL_LEN, HEAD_DIM).mul(2));
        }

        kvManager.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

        // Create multi-token decoder output (simulates speculative decode with 3 accepted tokens)
        int numAccepted = 3;
        Map<String, INDArray> specOutputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + numAccepted, HEAD_DIM);
            // New entries start at MAX_KV_LEN
            for (int i = 0; i < numAccepted; i++) {
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(MAX_KV_LEN + i), NDArrayIndex.all()).assign((i + 1) * 10.0f);
            }
            specOutputs.put(name, kv);
        }
        for (String name : kvNames.valueNames) {
            INDArray kv = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + numAccepted, HEAD_DIM);
            for (int i = 0; i < numAccepted; i++) {
                kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(MAX_KV_LEN + i), NDArrayIndex.all()).assign((i + 1) * 10.0f + 0.5f);
            }
            specOutputs.put(name, kv);
        }

        kvManager.scatterMultipleEntries(specOutputs, kvNames, numAccepted);

        // Verify all 3 entries landed in the right positions
        Map<String, INDArray> buffers = kvManager.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : buffers.entrySet()) {
            boolean isKey = e.getKey().endsWith(".key");
            for (int i = 0; i < numAccepted; i++) {
                float expected = isKey ? (i + 1) * 10.0f : (i + 1) * 10.0f + 0.5f;
                float actual = e.getValue().getFloat(0, 0, PREFILL_LEN + i, 0);
                assertEquals(expected, actual, 1e-4,
                        e.getKey() + " entry " + i + " at position " + (PREFILL_LEN + i));
            }
        }

        assertEquals(PREFILL_LEN + numAccepted, kvManager.getCachePosition(),
                "cachePosition should advance by numAccepted");

        kvManager.close();
    }

    // ─── Test: apply() followed by resetModelState is a clean slate ──────

    @Test
    @DisplayName("20: apply() then resetModelState() leaves model in clean state for next config")
    public void testApplyThenResetIsCleanSlate() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable intermediate = sd.mmul("intermediate", input, weights);
        SDVariable output = sd.identity("output", intermediate);
        sd.setOutputs(Collections.singletonList("output"));

        // Simulate config 1: apply, mutate model, execute
        BenchmarkConfig config1 = BenchmarkConfig.create("C1")
                .tritonIncludeTypes("NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .cublasTf32(true).tritonTf32(true)
                .maxTokens(100);
        BenchmarkConfigApplier.apply(config1);

        // Add a placeholder override (simulates attn_mask_reformat)
        sd.addPlaceholderOverride("intermediate");
        assertTrue(sd.inputs().contains("intermediate"),
                "intermediate should be in inputs after override");

        // Execute with the override
        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        feedDict.put("intermediate", Nd4j.randn(DataType.FLOAT, 1, 4));
        sd.output(feedDict, "output");

        // Now simulate transition to config 2: reset + apply
        BenchmarkConfigApplier.resetModelState(sd);

        BenchmarkConfig config2 = BenchmarkConfig.create("C2")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .cublasTf32(false).tritonTf32(false)
                .maxTokens(10);
        BenchmarkConfigApplier.apply(config2);

        // After reset, the placeholder override should be gone
        assertNotEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "Placeholder override should be cleared after resetModelState");

        // Model should work WITHOUT providing the overridden placeholder
        Map<String, INDArray> feedDict2 = new HashMap<>();
        feedDict2.put("input", Nd4j.randn(DataType.FLOAT, 1, 8));
        Map<String, INDArray> result = sd.output(feedDict2, "output");
        assertNotNull(result.get("output"), "Model should produce output after reset+apply");

        // Environment flags should reflect config2
        Environment env = Nd4j.getEnvironment();
        assertFalse(env.cublasTf32Enabled(), "cuBLAS TF32 should be OFF for config2");
        assertFalse(env.tritonTf32Enabled(), "Triton TF32 should be OFF for config2");
        assertFalse(env.tritonGraphCapture(), "Graph capture should be OFF for config2");
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

    private Map<String, INDArray> createMockDecoderOutputs(DecoderUtils.KVCacheNames kvNames) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (String name : kvNames.keyNames) {
            INDArray kv = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            outputs.put(name, kv);
        }
        for (String name : kvNames.valueNames) {
            INDArray kv = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, MAX_KV_LEN + 1, HEAD_DIM);
            outputs.put(name, kv);
        }
        return outputs;
    }
}
