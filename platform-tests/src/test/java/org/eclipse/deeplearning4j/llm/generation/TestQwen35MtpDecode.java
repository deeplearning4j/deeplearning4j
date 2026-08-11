/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
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

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end losslessness and engagement oracle for Qwen3.5's bundled NextN
 * predictor. The MTP-enabled GGUF is cached under a distinct local filename so
 * the legacy GGUF without MTP tensors remains available to the n-gram tests.
 */
@Slf4j
public class TestQwen35MtpDecode {

    private static final String MODEL_URL =
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF/resolve/main/"
                    + "Qwen3.5-0.8B-Q4_K_M.gguf";
    private static final String MODEL_FILE = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf";
    private static final String TOKENIZER_URL =
            "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json";
    private static final String TOKENIZER_FILE = "qwen35-0.8B-tokenizer.json";
    private static final String PROMPT =
            "The quick brown fox jumps over the lazy dog. Explain why this sentence is useful.";
    private static final int SPEC_K = 4;
    private static final int TOKENS =
            Integer.getInteger(ND4JSystemProperties.BENCH_MAX_TOKENS, 60);

    private static SameDiff model;
    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }

        File modelFile = LLMModelDownloader.downloadCustom(MODEL_URL, MODEL_FILE);
        SameDiff rawModel = GGMLModelImport.importModel(modelFile.getAbsolutePath());
        assertNotNull(rawModel.getVariable("target_hidden_states"),
                "Target graph must export the pre-final-norm hidden state needed by MTP");
        assertNotNull(rawModel.getVariable("mtp_logits"),
                "MTP-enabled GGUF must import the bundled predictor logits");
        assertNotNull(rawModel.getVariable("mtp_hidden_states"),
                "MTP-enabled GGUF must export predictor carry state");

        ModelIOConfig.KVCacheNames targetKvNames = ModelIOConfig.findKVCacheInputNames(rawModel);
        assertNotNull(targetKvNames, "Target decoder KV cache inputs must be discoverable");
        assertEquals(6, targetKvNames.keyNames.size(),
                "MTP-local key cache must not be classified as a target decoder cache");
        assertEquals(6, targetKvNames.valueNames.size(),
                "MTP-local value cache must not be classified as a target decoder cache");
        assertTrue(targetKvNames.keyNames.stream().noneMatch(name -> name.startsWith("mtp_"))
                        && targetKvNames.valueNames.stream().noneMatch(name -> name.startsWith("mtp_")),
                "Target KV discovery must exclude the isolated MTP cache namespace");

        ModelIOConfig targetIo = ModelIOConfig.discover(rawModel);
        assertEquals("input_ids", targetIo.getInputIdsName(),
                "MTP input IDs must not replace the target decoder input");
        assertEquals("_causal_mask", targetIo.getCausalMaskName(),
                "MTP causal mask must not replace the target decoder mask");
        assertEquals("position_offset", targetIo.getPositionOffsetName(),
                "MTP position offset must not replace the target decoder offset");
        assertEquals("cache_position", targetIo.getCachePositionName(),
                "MTP cache position must not replace the target decoder position");

        List<String> outputs = rawModel.outputs() != null
                ? new ArrayList<>(rawModel.outputs()) : new ArrayList<>();
        long optimizeStart = System.currentTimeMillis();
        model = GraphOptimizer.optimize(rawModel, outputs, GraphOptimizer.defaultOptimizations());
        log.info("[MTP-SETUP] GraphOptimizer: {} -> {} ops in {}ms",
                rawModel.getOps().size(), model.getOps().size(),
                System.currentTimeMillis() - optimizeStart);

        if (model != rawModel) {
            SameDiffMemoryUtils.freeModelArrays(rawModel);
            rawModel.close();
        }

        File tokenizerFile = LLMModelDownloader.downloadCustom(TOKENIZER_URL, TOKENIZER_FILE);
        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
    }

    @AfterAll
    public static void teardown() {
        if (model != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(model);
                model.close();
            } catch (Exception e) {
                log.warn("[MTP-TEARDOWN] Error closing model: {}", e.getMessage());
            }
            model = null;
        }
        tokenizer = null;
    }

    @Test
    public void testBundledMtpIsLosslessAndEngaged() throws Exception {
        SamplingConfig mtpSampling = SamplingConfig.speculative().toBuilder()
                .minNewTokens(TOKENS)
                .build();
        SamplingConfig greedySampling = SamplingConfig.greedy().toBuilder()
                .minNewTokens(TOKENS)
                .build();

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(mtpSampling)
                .maxNewTokens(TOKENS)
                .maxSpeculativeTokens(SPEC_K)
                .maxPrefillLength(64)
                .maxKvCacheLength(Math.max(192, TOKENS + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        GenerationResult mtpResult;
        GenerationResult greedyResult;
        try (GenerationPipeline pipeline = GenerationPipeline.create(config)) {
            for (int warmup = 0; warmup < 2; warmup++) {
                GenerationResult result = pipeline.generate(PROMPT, TOKENS);
                log.info("[MTP-WARMUP {}] tokens={} proposed={} accepted={} steps={}",
                        warmup + 1, result.getTokenIds().length,
                        result.getTotalSpeculativeTokens(), result.getTotalAcceptedTokens(),
                        result.getSpeculativeSteps());
            }

            mtpResult = pipeline.generate(PROMPT, TOKENS);
            log.info("[MTP-METRICS] tokens={} proposed={} accepted={} steps={} acceptance={} "
                            + "tok/s={} decodeTok/s={} lateTok/s={} effectiveTok/s={}",
                    mtpResult.getTokenIds().length, mtpResult.getTotalSpeculativeTokens(),
                    mtpResult.getTotalAcceptedTokens(), mtpResult.getSpeculativeSteps(),
                    mtpResult.getAverageAcceptanceRate(), mtpResult.getTokensPerSecond(),
                    mtpResult.getDecodeTokensPerSecond(),
                    mtpResult.getLateSteadyStateTokensPerSecond(),
                    mtpResult.getEffectiveTokensPerSecond());

            pipeline.setSamplingConfig(greedySampling);
            greedyResult = pipeline.generate(PROMPT, TOKENS);
            log.info("[MTP-GREEDY-METRICS] tokens={} tok/s={} decodeTok/s={} lateTok/s={}",
                    greedyResult.getTokenIds().length, greedyResult.getTokensPerSecond(),
                    greedyResult.getDecodeTokensPerSecond(),
                    greedyResult.getLateSteadyStateTokensPerSecond());
        }

        int[] mtpTokens = mtpResult.getTokenIds();
        int[] greedyTokens = greedyResult.getTokenIds();
        log.info("[MTP-ORACLE] mtp={} greedy={} mtpFirst={} greedyFirst={}",
                mtpTokens.length, greedyTokens.length,
                Arrays.toString(Arrays.copyOf(mtpTokens, Math.min(16, mtpTokens.length))),
                Arrays.toString(Arrays.copyOf(greedyTokens, Math.min(16, greedyTokens.length))));

        assertTrue(mtpResult.getTotalSpeculativeTokens() > 0,
                "Bundled MTP proposed zero tokens; scalar/n-gram fallback is not the MTP path");
        assertTrue(mtpResult.getSpeculativeSteps() > 0,
                "Bundled MTP reported zero speculative steps");
        assertTrue(mtpResult.getTotalAcceptedTokens() > 0,
                "Bundled MTP accepted zero tokens");
        assertTrue(mtpTokens.length >= Math.min(TOKENS, 20),
                "MTP generated too few tokens: " + mtpTokens.length);
        assertArrayEquals(greedyTokens, mtpTokens,
                "Bundled Qwen3.5 MTP must remain token-identical to greedy decode");
    }

    /**
     * Pins the target-model invariant required by lossless speculative decoding: evaluating two
     * causally chained tokens inside the production W=5 envelope must produce the same per-layer
     * rows as evaluating the same tokens as two activeWindow=1 calls on that frozen W=5 plan.
     *
     * <p>The test deliberately requests layer boundaries plus the operation boundaries inside the
     * first hybrid GDN block and first full-attention block. If parity regresses, the numeric
     * discriminator identifies the first graph variable where row 1 departs from the chained
     * scalar reference.</p>
     */
    @Test
    public void testTargetWindowRowsMatchChainedScalarCheckpoints() throws Exception {
        runTargetWindowRowsMatchChainedScalarCheckpoints(true);
    }

    @Test
    public void testTargetWindowRowsMatchChainedScalarOutputOnly() throws Exception {
        runTargetWindowRowsMatchChainedScalarCheckpoints(false);
    }

    private void runTargetWindowRowsMatchChainedScalarCheckpoints(boolean requestAttentionAux) throws Exception {
        final int window = 5;
        final int maxKvLength = 192;
        List<INDArray> owned = new ArrayList<>();

        try {
            ModelIOConfig io = ModelIOConfig.discover(model);
            ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(model);
            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(model, io);
            assertNotNull(kvNames, "Target decoder KV inputs must be discoverable");

            int[] promptTokenIds = tokenizer.encodePrompt(PROMPT, null).getIds();
            int prefillLength = promptTokenIds.length;
            DataType maskType = model.getVariable(io.getCausalMaskName()).dataType();

            // Production parity mode: -Dmtp.parity.paddedPrefill=64 replicates the
            // pipeline's maxPrefillLength padded prefill (pad ids with 0, asl=actual,
            // padded causal mask) so window/scalar writes are comparable against the
            // production decode-loop KV dumps. Default 0 keeps the original unpadded
            // prefill behavior.
            int paddedPrefill = Integer.getInteger("mtp.parity.paddedPrefill", 0);
            int prefillSeqLen = paddedPrefill > prefillLength ? paddedPrefill : prefillLength;
            long[] paddedIds = new long[prefillSeqLen];
            for (int i = 0; i < prefillLength; i++) paddedIds[i] = promptTokenIds[i];

            Map<String, INDArray> prefillInputs = new HashMap<>();
            putOwned(prefillInputs, io.getInputIdsName(),
                    Nd4j.createFromArray(paddedIds).reshape(1, prefillSeqLen), owned);
            putOwned(prefillInputs, io.getPositionOffsetName(), Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, io.getCachePositionName(), Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, "actual_sequence_length",
                    Nd4j.scalar(DataType.INT64, (long) prefillLength), owned);
            putOwned(prefillInputs, io.getCausalMaskName(),
                    prefillSeqLen > prefillLength
                            ? buildPaddedPrefillMaskForTest(prefillLength, prefillSeqLen, maxKvLength, maskType)
                            : DecoderInputBuilder.buildInGraphCausalMask(prefillLength, maxKvLength, maskType), owned);

            for (String name : kvNames.keyNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (String name : kvNames.valueNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(model, pair.inputName);
                assertNotNull(stateShape, "Could not derive recurrent state shape for " + pair.inputName);
                putOwned(prefillInputs, pair.inputName,
                        Nd4j.zeros(model.getVariable(pair.inputName).dataType(), stateShape), owned);
            }

            List<String> prefillOutputNames = new ArrayList<>();
            prefillOutputNames.add(io.getLogitsOutputName());
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                prefillOutputNames.add("k_rope_" + layer);
                prefillOutputNames.add("v_heads_" + layer);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                prefillOutputNames.add(pair.outputName);
            }

            Map<String, INDArray> prefillOutputs = model.output(
                    prefillInputs, prefillOutputNames.toArray(new String[0]));
            ownAll(prefillOutputs, owned);
            INDArray prefillLogits = prefillOutputs.get(io.getLogitsOutputName());
            int firstToken = argMaxToken(prefillLogits, prefillLength - 1);

            Map<String, INDArray> baseKv = new LinkedHashMap<>();
            for (int i = 0; i < kvNames.keyNames.size(); i++) {
                int layer = extractLayerIndex(kvNames.keyNames.get(i));
                INDArray keyRows = prefillOutputs.get("k_rope_" + layer);
                INDArray valueRows = prefillOutputs.get("v_heads_" + layer);
                assertEquals(4, keyRows.rank(), "Expected BSHD K rows for layer " + layer);
                assertEquals(4, valueRows.rank(), "Expected BSHD V rows for layer " + layer);

                INDArray keyCache = own(owned, Nd4j.zeros(keyRows.dataType(),
                        keyRows.size(0), maxKvLength, keyRows.size(2), keyRows.size(3)));
                INDArray valueCache = own(owned, Nd4j.zeros(valueRows.dataType(),
                        valueRows.size(0), maxKvLength, valueRows.size(2), valueRows.size(3)));
                keyCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(keyRows);
                valueCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(valueRows);
                baseKv.put(kvNames.keyNames.get(i), keyCache);
                baseKv.put(kvNames.valueNames.get(i), valueCache);
            }

            Map<String, INDArray> baseStates = new LinkedHashMap<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                baseStates.put(pair.inputName, own(owned, prefillOutputs.get(pair.outputName).dup()));
            }

            // Prefill-boundary comparison against the production pipeline's [GGUF-KV]
            // logs (kRoped min/max per layer) and committed token ids.
            INDArray prefillKRope3 = prefillOutputs.get("k_rope_3");
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH prefill: len={} padded={} firstToken={} "
                            + "k_rope_3 shape={} min={} max={}",
                    prefillLength, prefillSeqLen, firstToken,
                    Arrays.toString(prefillKRope3.shape()),
                    prefillKRope3.minNumber(), prefillKRope3.maxNumber());

            // Match GenerationPipeline's normal warmup: physical W=2, only row 0 active.
            Map<String, INDArray> warmupInputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, baseKv, baseStates, owned);
            setDecodeStep(warmupInputs, io, firstToken, 0, prefillLength, 1,
                    window, maxKvLength, maskType, owned);
            List<String> warmupOutputNames = new ArrayList<>();
            warmupOutputNames.add(io.getLogitsOutputName());
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                warmupOutputNames.add("k_rope_" + layer);
                warmupOutputNames.add("v_heads_" + layer);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                warmupOutputNames.add(pair.outputName);
            }
            Map<String, INDArray> warmupOutputs = model.output(
                    warmupInputs, warmupOutputNames.toArray(new String[0]));
            ownAll(warmupOutputs, owned);
            int secondToken = argMaxToken(warmupOutputs.get(io.getLogitsOutputName()), 0);
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH tokens: firstToken={} secondToken={} "
                    + "(production committed: 271@17, 248068@18)", firstToken, secondToken);
            commitKvRows(baseKv, kvNames, warmupOutputs, prefillLength, 1);

            Map<String, INDArray> postWarmupKv = duplicateArrays(baseKv, owned);
            Map<String, INDArray> postWarmupStates = new LinkedHashMap<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                postWarmupStates.put(pair.inputName,
                        own(owned, warmupOutputs.get(pair.outputName).dup()));
            }

            LinkedHashSet<String> detailedCandidates = new LinkedHashSet<>(Arrays.asList(
                    "embedded",
                    "model.layers.0.input_layernorm",
                    "gdn_qkv_0",
                    "gdn_conv_0",
                    "gdn_gate_proj_0",
                    "gdn_z_reshaped_0",
                    "gdn_gate_act_0",
                    "gdn_out_0",
                    "model.layers.0.gdn.ssm_norm_0",
                    "gdn_gated_0",
                    "gdn_proj_0",
                    "post_attn_0",
                    "model.layers.0.post_attention_layernorm",
                    "gate_0",
                    "up_0",
                    "swish_1",
                    "swiglu_0",
                    "down_0",
                    "model.layers.3.input_layernorm",
                    "q_full_3",
                    "k_3",
                    "v_3",
                    "qg_reshaped_3",
                    "q_3",
                    "attn_gate_3",
                    "k_heads_3",
                    "v_heads_3",
                    "model.layers.3.self_attn.q_norm_3",
                    "model.layers.3.self_attn.k_norm_3",
                    "q_rope_3",
                    "k_rope_3",
                    "attn_out_3",
                    "attn_flat_3",
                    "gate_sigmoid_3",
                    "gated_attn_3",
                    "attn_proj_3",
                    "post_attn_3",
                    "model.layers.3.post_attention_layernorm",
                    "gate_3",
                    "up_3",
                    "swiglu_3",
                    "down_3"));
            LinkedHashSet<String> comparableNames = new LinkedHashSet<>();
            for (String name : detailedCandidates) {
                if (model.hasVariable(name)) {
                    comparableNames.add(name);
                } else {
                    log.info("[MTP-TARGET-PARITY] Optimizer removed detailed checkpoint {}", name);
                }
            }
            for (int layer = 0; layer < 24; layer++) {
                String layerOutput = "layer_out_" + layer;
                assertTrue(model.hasVariable(layerOutput), "Missing target layer boundary " + layerOutput);
                comparableNames.add(layerOutput);
            }
            assertTrue(model.hasVariable("target_hidden_states"), "Missing target hidden-state boundary");
            assertTrue(model.hasVariable(io.getLogitsOutputName()), "Missing target logits boundary");
            comparableNames.add("target_hidden_states");
            comparableNames.add(io.getLogitsOutputName());

            String attentionScoresName = "dot_product_attention_v2:1";
            String attentionLogitsName = "dot_product_attention_v2:2";
            LinkedHashSet<String> requestedNames = new LinkedHashSet<>(comparableNames);
            if (requestAttentionAux) {
                assertTrue(model.hasVariable(attentionScoresName),
                        "Missing first full-attention softmax-score output");
                assertTrue(model.hasVariable(attentionLogitsName),
                        "Missing first full-attention logits output");
                requestedNames.add(attentionScoresName);
                requestedNames.add(attentionLogitsName);
            }
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                requestedNames.add("k_rope_" + layer);
                requestedNames.add("v_heads_" + layer);
            }
            LinkedHashSet<String> stateOutputNames = new LinkedHashSet<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                requestedNames.add(pair.outputName);
                stateOutputNames.add(pair.outputName);
            }
            String[] requested = requestedNames.toArray(new String[0]);

            // Use one stable set of external buffers for warmup, W=5, and both scalar calls. This
            // matches the frozen native plan's pointer-stability contract and avoids cross-plan noise.
            Map<String, INDArray> stableKv = duplicateArrays(postWarmupKv, owned);
            Map<String, INDArray> stableStates = duplicateArrays(postWarmupStates, owned);
            Map<String, INDArray> stableInputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, stableKv, stableStates, owned);

            // Prime this exact output set through the compile transition, then freeze it exactly as the
            // production fixed-buffer path does. Restore model state before every priming call.
            for (int pass = 0; pass < 4; pass++) {
                restoreArrays(stableKv, postWarmupKv);
                restoreArrays(stableStates, postWarmupStates);
                setDecodeStep(stableInputs, io, secondToken, 0, prefillLength + 1,
                        1, window, maxKvLength, maskType, owned);
                Map<String, INDArray> priming = model.output(stableInputs, requested);
                argMaxToken(priming.get(io.getLogitsOutputName()), 0);
                if (pass == 0) {
                    model.getOrCreateSession().getDynamicShapePlanExecutor().setShapesFrozen(true);
                }
                ownAll(priming, owned);
            }

            int vocabSize = (int) warmupOutputs.get(io.getLogitsOutputName()).size(2);
            int draftToken = (secondToken + 1) % vocabSize;

            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    2, window, maxKvLength, maskType, owned);
            // Production-envelope mode: -Dmtp.parity.window4=true evaluates the SAME
            // rows 0-1 inside the production speculative envelope (activeWindow=4,
            // wild draft tokens at rows 2-3, asl=4) instead of activeWindow=2 with
            // zero tail rows. Rows 0-1 are causally invariant to rows>=2 under
            // correct op semantics, so all downstream row-0/row-1 comparisons remain
            // valid — a divergence here reproduces the native decode-loop corruption.
            if (Boolean.getBoolean("mtp.parity.window4")) {
                setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                        4, window, maxKvLength, maskType, owned);
                INDArray specIds = stableInputs.get(io.getInputIdsName());
                specIds.putScalar(0, 2, 332);
                specIds.putScalar(0, 3, 55786);
                log.info("[MTP-TARGET-PARITY] window4 production envelope: ids row2=332 row3=55786 asl=4");
            }
            INDArray windowMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> windowOutputs = model.output(stableInputs, requested);
            ownAll(windowOutputs, owned);
            // Generic SameDiff staging does not expose input side effects to caller arrays.
            // Export and commit K/V rows explicitly so this direct oracle matches native decode state.
            argMaxToken(windowOutputs.get(io.getLogitsOutputName()), 1);
            commitKvRows(stableKv, kvNames, windowOutputs, prefillLength + 1, 2);
            Map<String, INDArray> windowSnapshot = snapshotOutputs(
                    windowOutputs, comparableNames, owned);
            Map<String, INDArray> windowStateSnapshot = snapshotOutputs(
                    windowOutputs, stateOutputNames, owned);
            INDArray windowAttentionScores = requestAttentionAux
                    ? own(owned, windowOutputs.get(attentionScoresName).dup()) : null;
            INDArray windowAttentionLogits = requestAttentionAux
                    ? own(owned, windowOutputs.get(attentionLogitsName).dup()) : null;
            Map<String, INDArray> windowKvSnapshot = duplicateArrays(stableKv, owned);

            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, 0, prefillLength + 1,
                    1, window, maxKvLength, maskType, owned);
            INDArray scalarFirstMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> scalarFirstOutputs = model.output(stableInputs, requested);
            ownAll(scalarFirstOutputs, owned);
            // Consume the token, then commit the exported row before the chained scalar call.
            argMaxToken(scalarFirstOutputs.get(io.getLogitsOutputName()), 0);
            commitKvRows(stableKv, kvNames, scalarFirstOutputs, prefillLength + 1, 1);
            Map<String, INDArray> scalarFirstSnapshot = snapshotOutputs(
                    scalarFirstOutputs, comparableNames, owned);
            Map<String, INDArray> scalarFirstStateSnapshot = snapshotOutputs(
                    scalarFirstOutputs, stateOutputNames, owned);
            INDArray scalarFirstAttentionScores = requestAttentionAux
                    ? own(owned, scalarFirstOutputs.get(attentionScoresName).dup()) : null;
            INDArray scalarFirstAttentionLogits = requestAttentionAux
                    ? own(owned, scalarFirstOutputs.get(attentionLogitsName).dup()) : null;

            // Commit the scalar call's accepted recurrent state for the chained row.
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                stableStates.get(pair.inputName).assign(scalarFirstOutputs.get(pair.outputName));
            }
            setDecodeStep(stableInputs, io, draftToken, 0, prefillLength + 2,
                    1, window, maxKvLength, maskType, owned);
            INDArray scalarSecondMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> scalarSecondOutputs = model.output(stableInputs, requested);
            ownAll(scalarSecondOutputs, owned);
            argMaxToken(scalarSecondOutputs.get(io.getLogitsOutputName()), 0);
            commitKvRows(stableKv, kvNames, scalarSecondOutputs, prefillLength + 2, 1);
            Map<String, INDArray> scalarSecondSnapshot = snapshotOutputs(
                    scalarSecondOutputs, comparableNames, owned);
            Map<String, INDArray> scalarSecondStateSnapshot = snapshotOutputs(
                    scalarSecondOutputs, stateOutputNames, owned);
            INDArray scalarSecondAttentionScores = requestAttentionAux
                    ? own(owned, scalarSecondOutputs.get(attentionScoresName).dup()) : null;
            INDArray scalarSecondAttentionLogits = requestAttentionAux
                    ? own(owned, scalarSecondOutputs.get(attentionLogitsName).dup()) : null;
            Map<String, INDArray> scalarKvSnapshot = duplicateArrays(stableKv, owned);

            String firstFinalStateDivergence = null;
            double firstFinalStateMax = 0.0;
            double firstFinalStateL1 = 0.0;
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        windowStateSnapshot.get(name), scalarSecondStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=finalState name={} max={} l1={}",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstFinalStateDivergence == null) {
                        firstFinalStateDivergence = name;
                        firstFinalStateMax = stateDiff[0];
                        firstFinalStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=finalState first={} max={} l1={}",
                    firstFinalStateDivergence, firstFinalStateMax, firstFinalStateL1);

            // Reproduce native speculative rejection: execute the W-wide target pass,
            // keep its K/V writes, but rerun from the original recurrent input state
            // with only the accepted-prefix length reduced to one row.
            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    2, window, maxKvLength, maskType, owned);
            Map<String, INDArray> acceptedZeroWindowOutputs = model.output(stableInputs, requested);
            ownAll(acceptedZeroWindowOutputs, owned);
            argMaxToken(acceptedZeroWindowOutputs.get(io.getLogitsOutputName()), 1);
            commitKvRows(stableKv, kvNames, acceptedZeroWindowOutputs, prefillLength + 1, 2);
            stableInputs.get("actual_sequence_length").putScalar(new long[]{}, 1L);
            Map<String, INDArray> acceptedZeroRerunOutputs = model.output(stableInputs, requested);
            ownAll(acceptedZeroRerunOutputs, owned);
            argMaxToken(acceptedZeroRerunOutputs.get(io.getLogitsOutputName()), 0);
            Map<String, INDArray> acceptedZeroRerunStateSnapshot = snapshotOutputs(
                    acceptedZeroRerunOutputs, stateOutputNames, owned);

            String firstAcceptedZeroStateDivergence = null;
            double firstAcceptedZeroStateMax = 0.0;
            double firstAcceptedZeroStateL1 = 0.0;
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        acceptedZeroRerunStateSnapshot.get(name), scalarFirstStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=acceptedZeroRerunState "
                                    + "name={} max={} l1={}",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstAcceptedZeroStateDivergence == null) {
                        firstAcceptedZeroStateDivergence = name;
                        firstAcceptedZeroStateMax = stateDiff[0];
                        firstAcceptedZeroStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=acceptedZeroRerunState "
                            + "first={} max={} l1={}",
                    firstAcceptedZeroStateDivergence,
                    firstAcceptedZeroStateMax, firstAcceptedZeroStateL1);

            // Reproduce the production step that proposed three drafts but accepted
            // only the first: execute four rows, retain their K/V writes, then rerun
            // the same fixed W=5 plan with an accepted-prefix length of two.
            int draftToken2 = (draftToken + 1) % vocabSize;
            int draftToken3 = (draftToken + 2) % vocabSize;
            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    4, window, maxKvLength, maskType, owned);
            INDArray partialIds = stableInputs.get(io.getInputIdsName());
            partialIds.putScalar(0, 2, draftToken2);
            partialIds.putScalar(0, 3, draftToken3);
            Map<String, INDArray> partialWindowOutputs = model.output(stableInputs, requested);
            ownAll(partialWindowOutputs, owned);
            argMaxToken(partialWindowOutputs.get(io.getLogitsOutputName()), 3);
            commitKvRows(stableKv, kvNames, partialWindowOutputs, prefillLength + 1, 4);
            stableInputs.get("actual_sequence_length").putScalar(new long[]{}, 2L);
            Map<String, INDArray> partialRerunOutputs = model.output(stableInputs, requested);
            ownAll(partialRerunOutputs, owned);
            argMaxToken(partialRerunOutputs.get(io.getLogitsOutputName()), 1);
            Map<String, INDArray> partialRerunStateSnapshot = snapshotOutputs(
                    partialRerunOutputs, stateOutputNames, owned);

            String firstPartialStateDivergence = null;
            double firstPartialStateMax = 0.0;
            double firstPartialStateL1 = 0.0;
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        partialRerunStateSnapshot.get(name), scalarSecondStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=partialRerunState "
                                    + "name={} max={} l1={}",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstPartialStateDivergence == null) {
                        firstPartialStateDivergence = name;
                        firstPartialStateMax = stateDiff[0];
                        firstPartialStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=partialRerunState "
                            + "first={} max={} l1={}",
                    firstPartialStateDivergence, firstPartialStateMax, firstPartialStateL1);

            String layer3KeyCacheName = kvNames.keyNames.stream()
                    .filter(name -> extractLayerIndex(name) == 3)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Missing layer-3 key cache"));
            logKeyCacheMutation("window", postWarmupKv.get(layer3KeyCacheName),
                    windowKvSnapshot.get(layer3KeyCacheName),
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0));
            logKeyCacheMutation("scalar", postWarmupKv.get(layer3KeyCacheName),
                    scalarKvSnapshot.get(layer3KeyCacheName),
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0));
            double[] windowKeyVsCache = difference(
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0),
                    sequenceRow(windowKvSnapshot.get(layer3KeyCacheName), prefillLength + 1));
            double[] scalarKeyVsCache = difference(
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0),
                    sequenceRow(scalarKvSnapshot.get(layer3KeyCacheName), prefillLength + 1));
            double[] windowVsScalarKey = difference(
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0),
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0));
            log.info("[MTP-TARGET-PARITY] discriminator=keyPersistence "
                            + "windowKeyVsCacheMax={} windowKeyVsCacheL1={} "
                            + "scalarKeyVsCacheMax={} scalarKeyVsCacheL1={} "
                            + "windowVsScalarKeyMax={} windowVsScalarKeyL1={}",
                    windowKeyVsCache[0], windowKeyVsCache[1],
                    scalarKeyVsCache[0], scalarKeyVsCache[1],
                    windowVsScalarKey[0], windowVsScalarKey[1]);
            // Ground-truth arbitration against the native decode-loop KV_ROW_SLICE dumps:
            // raw layer-3 key/value cache components at rows 18..21 (h=0, d=0..1), same
            // slice the production probes print. Row 18's value is the alignment
            // fingerprint between the two position conventions.
            String layer3ValueCacheName = kvNames.valueNames.stream()
                    .filter(name -> extractLayerIndex(name) == 3)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Missing layer-3 value cache"));
            for (Map.Entry<String, Map<String, INDArray>> snap : Map.of(
                    "window", windowKvSnapshot, "scalar", scalarKvSnapshot).entrySet()) {
                INDArray kc = snap.getValue().get(layer3KeyCacheName);
                INDArray vc = snap.getValue().get(layer3ValueCacheName);
                StringBuilder sb = new StringBuilder();
                for (int r = 18; r <= 21 && r < kc.size(1); r++) {
                    sb.append(String.format(" k[%d]=[%.6g,%.6g] v[%d]=[%.6g,%.6g]",
                            r, kc.getDouble(0, r, 0, 0), kc.getDouble(0, r, 0, 1),
                            r, vc.getDouble(0, r, 0, 0), vc.getDouble(0, r, 0, 1)));
                }
                log.info("[MTP-TARGET-PARITY] JAVA-TRUTH {} layer3 cache rows18..21:{}",
                        snap.getKey(), sb);
            }
            INDArray kRopeWin = windowSnapshot.get("k_rope_3");
            INDArray vHeadsWin = windowSnapshot.get("v_heads_3");
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH window k_rope_3 row0=[{},{}] row1=[{},{}] "
                            + "v_heads_3 row0=[{},{}] row1=[{},{}]",
                    kRopeWin.getDouble(0, 0, 0, 0), kRopeWin.getDouble(0, 0, 0, 1),
                    kRopeWin.getDouble(0, 1, 0, 0), kRopeWin.getDouble(0, 1, 0, 1),
                    vHeadsWin.getDouble(0, 0, 0, 0), vHeadsWin.getDouble(0, 0, 0, 1),
                    vHeadsWin.getDouble(0, 1, 0, 0), vHeadsWin.getDouble(0, 1, 0, 1));
            if (requestAttentionAux) {
                logAttentionKeyReference(
                        windowSnapshot.get("q_rope_3"), 1,
                        scalarSecondSnapshot.get("q_rope_3"), 0,
                        windowKvSnapshot.get(layer3KeyCacheName),
                        postWarmupKv.get(layer3KeyCacheName),
                        windowSnapshot.get("k_rope_3"),
                        windowMaskSnapshot, 1,
                        scalarSecondMaskSnapshot, 0,
                        windowAttentionLogits, 1,
                        scalarSecondAttentionLogits, 0,
                        prefillLength + 1);
            }

            double[] maskRow0 = difference(
                    queryRow(windowMaskSnapshot, 0), queryRow(scalarFirstMaskSnapshot, 0));
            double[] maskRow1 = difference(
                    queryRow(windowMaskSnapshot, 1), queryRow(scalarSecondMaskSnapshot, 0));
            log.info("[MTP-TARGET-PARITY] discriminator=mask row0Max={} row0L1={} "
                            + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                    maskRow0[0], maskRow0[1], maskRow1[0], maskRow1[1],
                    Arrays.toString(windowMaskSnapshot.shape()),
                    Arrays.toString(scalarSecondMaskSnapshot.shape()));

            if (requestAttentionAux) {
                double[] scoreRow0 = difference(
                        queryRow(windowAttentionScores, 0), queryRow(scalarFirstAttentionScores, 0));
                double[] scoreRow1 = difference(
                        queryRow(windowAttentionScores, 1), queryRow(scalarSecondAttentionScores, 0));
                log.info("[MTP-TARGET-PARITY] discriminator=attentionScores row0Max={} row0L1={} "
                                + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                        scoreRow0[0], scoreRow0[1], scoreRow1[0], scoreRow1[1],
                        Arrays.toString(windowAttentionScores.shape()),
                        Arrays.toString(scalarSecondAttentionScores.shape()));

                double[] logitsRow0 = difference(
                        queryRow(windowAttentionLogits, 0), queryRow(scalarFirstAttentionLogits, 0));
                double[] logitsRow1 = difference(
                        queryRow(windowAttentionLogits, 1), queryRow(scalarSecondAttentionLogits, 0));
                log.info("[MTP-TARGET-PARITY] discriminator=attentionLogits row0Max={} row0L1={} "
                                + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                        logitsRow0[0], logitsRow0[1], logitsRow1[0], logitsRow1[1],
                        Arrays.toString(windowAttentionLogits.shape()),
                        Arrays.toString(scalarSecondAttentionLogits.shape()));
                logAttentionRowDifferences("attentionLogits", windowAttentionLogits, 1,
                        scalarSecondAttentionLogits, 0, prefillLength + 1, prefillLength + 2);
                logAttentionRowDifferences("attentionScores", windowAttentionScores, 1,
                        scalarSecondAttentionScores, 0, prefillLength + 1, prefillLength + 2);
            }

            String firstKvDivergence = null;
            double firstKvMax = 0.0;
            double firstKvL1 = 0.0;
            for (String name : windowKvSnapshot.keySet()) {
                double[] kvDiff = difference(windowKvSnapshot.get(name), scalarKvSnapshot.get(name));
                if (kvDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=kv name={} max={} l1={}",
                            name, kvDiff[0], kvDiff[1]);
                    if (firstKvDivergence == null) {
                        firstKvDivergence = name;
                        firstKvMax = kvDiff[0];
                        firstKvL1 = kvDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=kv first={} max={} l1={}",
                    firstKvDivergence, firstKvMax, firstKvL1);

            String firstDivergence = null;
            double firstMax = 0.0;
            double firstL1 = 0.0;
            for (String name : comparableNames) {
                double[] row0 = difference(
                        sequenceRow(windowSnapshot.get(name), 0),
                        sequenceRow(scalarFirstSnapshot.get(name), 0));
                double[] row1 = difference(
                        sequenceRow(windowSnapshot.get(name), 1),
                        sequenceRow(scalarSecondSnapshot.get(name), 0));
                if (row0[0] != 0.0 || row1[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] name={} row0Max={} row0L1={} row1Max={} row1L1={}",
                            name, row0[0], row0[1], row1[0], row1[1]);
                    if (firstDivergence == null) {
                        firstDivergence = name;
                        firstMax = Math.max(row0[0], row1[0]);
                        firstL1 = row0[1] + row1[1];
                    }
                }
            }

            assertTrue(firstDivergence == null,
                    (requestAttentionAux ? "Aux-output" : "Output-only")
                            + " W=5 target rows diverged first at " + firstDivergence
                            + " (max=" + firstMax + ", l1=" + firstL1 + ")");
            assertTrue(firstFinalStateDivergence == null,
                    "W=5 final recurrent state diverged first at " + firstFinalStateDivergence
                            + " (max=" + firstFinalStateMax + ", l1=" + firstFinalStateL1 + ")");
            assertTrue(firstAcceptedZeroStateDivergence == null,
                    "Accepted-zero rerun state diverged first at "
                            + firstAcceptedZeroStateDivergence
                            + " (max=" + firstAcceptedZeroStateMax
                            + ", l1=" + firstAcceptedZeroStateL1 + ")");
            assertTrue(firstPartialStateDivergence == null,
                    "Partial accepted-prefix rerun state diverged first at "
                            + firstPartialStateDivergence
                            + " (max=" + firstPartialStateMax
                            + ", l1=" + firstPartialStateL1 + ")");
        } finally {
            model.resetSession();
            closeOwned(owned);
        }
    }

    /** Replicates GenerationPipeline.buildPaddedPrefillCausalMask for production-parity prefill. */
    private static INDArray buildPaddedPrefillMaskForTest(int actualLen, int paddedLen,
                                                          long maxKvLen, DataType dtype) {
        int q0 = paddedLen;
        int k0 = (int) maxKvLen;
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[q0 * k0];
        for (int q = 0; q < q0; q++) {
            int rowOffset = q * k0;
            if (q < actualLen) {
                for (int k = q + 1; k < k0; k++) data[rowOffset + k] = maskVal;
            } else {
                // Keep one safe attention target so FP16 softmax never receives an all-masked row.
                for (int k = 1; k < k0; k++) data[rowOffset + k] = maskVal;
            }
        }
        INDArray mask = Nd4j.create(data, new long[]{1, 1, paddedLen, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }

    private static Map<String, INDArray> newStableDecodeInputs(
            ModelIOConfig io, int window, int maxKvLength, DataType maskType,
            Map<String, INDArray> kv, Map<String, INDArray> states, List<INDArray> owned) {
        Map<String, INDArray> inputs = new HashMap<>();
        putOwned(inputs, io.getInputIdsName(), Nd4j.zeros(DataType.INT64, 1, window), owned);
        putOwned(inputs, io.getCausalMaskName(),
                DecoderInputBuilder.buildInGraphWindowMask(
                        DecoderInputBuilder.chainParents(1, window), 0, 1, window,
                        maxKvLength, maskType), owned);
        putOwned(inputs, io.getPositionOffsetName(), Nd4j.scalar(DataType.INT64, 0L), owned);
        putOwned(inputs, io.getCachePositionName(), Nd4j.scalar(DataType.INT64, 0L), owned);
        putOwned(inputs, "actual_sequence_length", Nd4j.scalar(DataType.INT64, 1L), owned);
        inputs.putAll(kv);
        inputs.putAll(states);
        return inputs;
    }

    private static void setDecodeStep(
            Map<String, INDArray> inputs, ModelIOConfig io,
            int token0, int token1, int position, int activeWindow, int window,
            int maxKvLength, DataType maskType, List<INDArray> owned) {
        INDArray ids = inputs.get(io.getInputIdsName());
        ids.assign(0);
        ids.putScalar(0, 0, token0);
        if (activeWindow > 1) ids.putScalar(0, 1, token1);

        INDArray maskDonor = own(owned, DecoderInputBuilder.buildInGraphWindowMask(
                DecoderInputBuilder.chainParents(activeWindow, window),
                position, activeWindow, window, maxKvLength, maskType));
        inputs.get(io.getCausalMaskName()).assign(maskDonor);
        inputs.get(io.getPositionOffsetName()).putScalar(new long[]{}, (long) position);
        inputs.get(io.getCachePositionName()).putScalar(new long[]{}, (long) position);
        inputs.get("actual_sequence_length").putScalar(new long[]{}, (long) activeWindow);
    }

    private static int argMaxToken(INDArray logits, int row) {
        INDArray argMax = logits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(row), NDArrayIndex.all()).argMax();
        try {
            return argMax.getInt(0);
        } finally {
            argMax.close();
        }
    }

    private static int extractLayerIndex(String cacheName) {
        String[] parts = cacheName.split("\\.");
        return Integer.parseInt(parts[1]);
    }

    private static void commitKvRows(
            Map<String, INDArray> kv, ModelIOConfig.KVCacheNames kvNames,
            Map<String, INDArray> outputs, int cachePosition, int rowCount) {
        for (int i = 0; i < kvNames.keyNames.size(); i++) {
            int layer = extractLayerIndex(kvNames.keyNames.get(i));
            commitCacheRows(kv.get(kvNames.keyNames.get(i)), outputs.get("k_rope_" + layer),
                    cachePosition, rowCount, "key", layer);
            commitCacheRows(kv.get(kvNames.valueNames.get(i)), outputs.get("v_heads_" + layer),
                    cachePosition, rowCount, "value", layer);
        }
    }

    private static void commitCacheRows(
            INDArray cache, INDArray rows, int cachePosition, int rowCount,
            String kind, int layer) {
        assertNotNull(cache, "Missing " + kind + " cache for layer " + layer);
        assertNotNull(rows, "Missing exported " + kind + " rows for layer " + layer);
        assertEquals(4, cache.rank(), "Expected BSHD " + kind + " cache for layer " + layer);
        assertEquals(4, rows.rank(), "Expected BSHD exported " + kind + " rows for layer " + layer);
        assertTrue(rows.size(1) >= rowCount,
                "Insufficient exported " + kind + " rows for layer " + layer);
        cache.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(cachePosition, cachePosition + rowCount),
                        NDArrayIndex.all(), NDArrayIndex.all())
                .assign(rows.get(NDArrayIndex.all(), NDArrayIndex.interval(0, rowCount),
                        NDArrayIndex.all(), NDArrayIndex.all()));
    }

    private static Map<String, INDArray> duplicateArrays(
            Map<String, INDArray> source, List<INDArray> owned) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> entry : source.entrySet()) {
            result.put(entry.getKey(), own(owned, entry.getValue().dup()));
        }
        return result;
    }

    private static void restoreArrays(Map<String, INDArray> destination, Map<String, INDArray> source) {
        for (Map.Entry<String, INDArray> entry : source.entrySet()) {
            destination.get(entry.getKey()).assign(entry.getValue());
        }
    }

    private static Map<String, INDArray> snapshotOutputs(
            Map<String, INDArray> outputs, Set<String> names, List<INDArray> owned) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : names) {
            INDArray output = outputs.get(name);
            assertNotNull(output, "Missing checkpoint output " + name);
            result.put(name, own(owned, output.dup()));
        }
        return result;
    }

    private static INDArray sequenceRow(INDArray array, int row) {
        assertTrue(array.rank() >= 2 && array.size(1) > row,
                "Checkpoint is not sequence-addressable: shape=" + Arrays.toString(array.shape()));
        INDArrayIndex[] indices = new INDArrayIndex[array.rank()];
        Arrays.fill(indices, NDArrayIndex.all());
        indices[1] = NDArrayIndex.point(row);
        return array.get(indices);
    }

    private static INDArray queryRow(INDArray array, int row) {
        int queryAxis = array.rank() - 2;
        assertTrue(queryAxis >= 0 && array.size(queryAxis) > row,
                "Tensor is not query-row-addressable: shape=" + Arrays.toString(array.shape()));
        INDArrayIndex[] indices = new INDArrayIndex[array.rank()];
        Arrays.fill(indices, NDArrayIndex.all());
        indices[queryAxis] = NDArrayIndex.point(row);
        return array.get(indices);
    }

    private static void logAttentionRowDifferences(
            String kind, INDArray window, int windowRow, INDArray scalar, int scalarRow,
            int windowBasePosition, int scalarPosition) {
        assertEquals(4, window.rank(), kind + " window tensor must be [B,H,Q,KV]");
        assertArrayEquals(window.shape(), scalar.shape(), kind + " tensor shape mismatch");
        int logged = 0;
        int different = 0;
        for (int head = 0; head < window.size(1); head++) {
            for (int kv = 0; kv < window.size(3); kv++) {
                double windowValue = window.getDouble(0, head, windowRow, kv);
                double scalarValue = scalar.getDouble(0, head, scalarRow, kv);
                double delta = Math.abs(windowValue - scalarValue);
                if (Double.doubleToLongBits(windowValue) == Double.doubleToLongBits(scalarValue)
                        || delta <= 1.0e-7) {
                    continue;
                }
                different++;
                if (logged < 32) {
                    log.info("[MTP-TARGET-PARITY] discriminator={} head={} kv={} "
                                    + "windowValue={} scalarValue={} absDiff={} "
                                    + "windowBasePosition={} scalarPosition={}",
                            kind, head, kv, windowValue, scalarValue, delta,
                            windowBasePosition, scalarPosition);
                    logged++;
                }
            }
        }
        log.info("[MTP-TARGET-PARITY] discriminator={} differingElements={} logged={}",
                kind, different, logged);
    }

    private static void logAttentionKeyReference(
            INDArray windowQuery, int windowQueryRow,
            INDArray scalarQuery, int scalarQueryRow,
            INDArray keyCache,
            INDArray keyCacheBeforeWrite,
            INDArray currentKeyWindow,
            INDArray windowMask, int windowMaskRow,
            INDArray scalarMask, int scalarMaskRow,
            INDArray windowLogits, int windowLogitsRow,
            INDArray scalarLogits, int scalarLogitsRow,
            int comparedKvPosition) {
        int qHeads = (int) windowQuery.size(2);
        int kvHeads = (int) keyCache.size(2);
        int headDim = (int) windowQuery.size(3);
        int headsPerKvHead = qHeads / kvHeads;
        double scale = 1.0 / Math.sqrt(headDim);
        int searchEnd = Math.min((int) keyCache.size(1), comparedKvPosition + 2);

        for (int qHead = 0; qHead < qHeads; qHead++) {
            int kvHead = qHead / headsPerKvHead;
            double expectedWindow = dotAttentionLogit(
                    windowQuery, windowQueryRow, qHead, keyCache, comparedKvPosition, kvHead,
                    windowMask, windowMaskRow, scale);
            double expectedBeforeWrite = dotAttentionLogit(
                    windowQuery, windowQueryRow, qHead, keyCacheBeforeWrite, comparedKvPosition, kvHead,
                    windowMask, windowMaskRow, scale);
            double expectedCurrentRow0 = dotCurrentWindowAttentionLogit(
                    windowQuery, windowQueryRow, qHead, currentKeyWindow, 0, kvHead,
                    windowMask, windowMaskRow, comparedKvPosition, scale);
            double expectedCurrentRow1 = dotCurrentWindowAttentionLogit(
                    windowQuery, windowQueryRow, qHead, currentKeyWindow, 1, kvHead,
                    windowMask, windowMaskRow, comparedKvPosition, scale);
            double expectedScalar = dotAttentionLogit(
                    scalarQuery, scalarQueryRow, qHead, keyCache, comparedKvPosition, kvHead,
                    scalarMask, scalarMaskRow, scale);
            double observedWindow = windowLogits.getDouble(
                    0, qHead, windowLogitsRow, comparedKvPosition);
            double observedScalar = scalarLogits.getDouble(
                    0, qHead, scalarLogitsRow, comparedKvPosition);

            int nearestWindowKv = -1;
            int nearestScalarKv = -1;
            double nearestWindowDelta = Double.POSITIVE_INFINITY;
            double nearestScalarDelta = Double.POSITIVE_INFINITY;
            for (int kv = 0; kv < searchEnd; kv++) {
                double candidateWindow = dotAttentionLogit(
                        windowQuery, windowQueryRow, qHead, keyCache, kv, kvHead,
                        windowMask, windowMaskRow, scale);
                double candidateScalar = dotAttentionLogit(
                        scalarQuery, scalarQueryRow, qHead, keyCache, kv, kvHead,
                        scalarMask, scalarMaskRow, scale);
                double windowDelta = Math.abs(observedWindow - candidateWindow);
                double scalarDelta = Math.abs(observedScalar - candidateScalar);
                if (windowDelta < nearestWindowDelta) {
                    nearestWindowDelta = windowDelta;
                    nearestWindowKv = kv;
                }
                if (scalarDelta < nearestScalarDelta) {
                    nearestScalarDelta = scalarDelta;
                    nearestScalarKv = kv;
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=keyReference head={} kv={} "
                            + "expectedWindow={} observedWindow={} windowResidual={} windowNearestKv={} "
                            + "windowNearestResidual={} beforeWrite={} beforeWriteResidual={} "
                            + "currentRow0={} currentRow0Residual={} currentRow1={} currentRow1Residual={} "
                            + "expectedScalar={} observedScalar={} scalarResidual={} "
                            + "scalarNearestKv={} scalarNearestResidual={}",
                    qHead, comparedKvPosition,
                    expectedWindow, observedWindow, Math.abs(expectedWindow - observedWindow),
                    nearestWindowKv, nearestWindowDelta,
                    expectedBeforeWrite, Math.abs(expectedBeforeWrite - observedWindow),
                    expectedCurrentRow0, Math.abs(expectedCurrentRow0 - observedWindow),
                    expectedCurrentRow1, Math.abs(expectedCurrentRow1 - observedWindow),
                    expectedScalar, observedScalar, Math.abs(expectedScalar - observedScalar),
                    nearestScalarKv, nearestScalarDelta);
        }
    }

    private static double dotAttentionLogit(
            INDArray query, int queryRow, int qHead,
            INDArray keyCache, int kvPosition, int kvHead,
            INDArray mask, int maskRow, double scale) {
        return dotCurrentWindowAttentionLogit(
                query, queryRow, qHead, keyCache, kvPosition, kvHead,
                mask, maskRow, kvPosition, scale);
    }

    private static double dotCurrentWindowAttentionLogit(
            INDArray query, int queryRow, int qHead,
            INDArray currentKeyWindow, int currentRow, int kvHead,
            INDArray mask, int maskRow, int maskKvPosition, double scale) {
        double dot = 0.0;
        for (int d = 0; d < query.size(3); d++) {
            dot += query.getDouble(0, queryRow, qHead, d)
                    * currentKeyWindow.getDouble(0, currentRow, kvHead, d);
        }
        return dot * scale + mask.getDouble(0, 0, maskRow, maskKvPosition);
    }

    private static void logKeyCacheMutation(
            String kind, INDArray before, INDArray after, INDArray currentKeyRow) {
        double[] total = difference(after, before);
        int changedRows = 0;
        int nearestCurrentRow = -1;
        double nearestCurrentMax = Double.POSITIVE_INFINITY;
        double nearestCurrentL1 = Double.POSITIVE_INFINITY;
        for (int row = 0; row < after.size(1); row++) {
            INDArray afterRow = sequenceRow(after, row);
            double[] mutation = difference(afterRow, sequenceRow(before, row));
            if (mutation[0] != 0.0) {
                changedRows++;
                if (changedRows <= 8) {
                    log.info("[MTP-TARGET-PARITY] discriminator=cacheMutation kind={} row={} max={} l1={}",
                            kind, row, mutation[0], mutation[1]);
                }
            }
            double[] current = difference(afterRow, currentKeyRow);
            if (current[1] < nearestCurrentL1) {
                nearestCurrentRow = row;
                nearestCurrentMax = current[0];
                nearestCurrentL1 = current[1];
            }
        }
        log.info("[MTP-TARGET-PARITY] discriminator=cacheMutation kind={} totalMax={} totalL1={} "
                        + "changedRows={} nearestCurrentRow={} nearestCurrentMax={} nearestCurrentL1={}",
                kind, total[0], total[1], changedRows,
                nearestCurrentRow, nearestCurrentMax, nearestCurrentL1);
    }

    private static double[] difference(INDArray left, INDArray right) {
        INDArray diff = left.sub(right);
        try {
            return new double[]{
                    diff.amaxNumber().doubleValue(),
                    diff.norm1Number().doubleValue()
            };
        } finally {
            diff.close();
        }
    }

    private static void putOwned(
            Map<String, INDArray> map, String name, INDArray array, List<INDArray> owned) {
        map.put(name, own(owned, array));
    }

    private static INDArray own(List<INDArray> owned, INDArray array) {
        owned.add(array);
        return array;
    }

    private static void ownAll(Map<String, INDArray> arrays, List<INDArray> owned) {
        owned.addAll(arrays.values());
    }

    private static void closeOwned(List<INDArray> owned) {
        Set<INDArray> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int i = owned.size() - 1; i >= 0; i--) {
            INDArray array = owned.get(i);
            if (array != null && seen.add(array) && !array.wasClosed()) {
                try {
                    array.close();
                } catch (Exception e) {
                    log.warn("[MTP-TARGET-PARITY] Failed to close array: {}", e.getMessage());
                }
            }
        }
    }
}
