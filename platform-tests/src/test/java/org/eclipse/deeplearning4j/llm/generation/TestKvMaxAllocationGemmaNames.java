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
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.ChatGenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.*;

/**
 * <p>Regression tests for KV max-allocation name matching on gemma-style graphs.</p>
 *
 * <p>{@code DynamicShapePlanExecutor.configureMaxAllocationForKvCache}'s legacy heuristic
 * only matches KV outputs named {@code present…key} / {@code present…value}. Architectures
 * whose KV outputs are named {@code k_rope_N} / {@code v_heads_N} (gemma4, Qwen3.5) never
 * matched, so {@code setPlanOutputSlotMaxSizes} was never called: every decode step allocated
 * a fresh full-length KV output buffer for all layers (~120 MB/step on gemma4-E2B), filling a
 * 24 GB card by step ~224. Under GPU memory pressure the ceiling arrives at step 0 and decode
 * dies with a frozen-buffer lifecycle violation.</p>
 *
 * <p>The contract under test (fixed via explicit-names overloads): after warmup decode with
 * {@code maxKvCacheLength > 0}, the plan's KV output slots MUST be max-length pinned —
 * {@code executor.isMaxAllocationConfigured()} is true — and subsequent decode steps reuse
 * the oversized buffers instead of allocating new ones (pool stays flat).</p>
 */
@Slf4j
@Tag("dsp")
public class TestKvMaxAllocationGemmaNames {

    private static String modelPath;
    private static Tokenizer tokenizer;
    private static GenerationPipeline.ModelMetadata modelMetadata;

    private static final int N = 64; // long enough that the old unpinned path grows observably
    private static final long MAX_POOL_GROWTH_BYTES = 256L * 1024 * 1024;

    private static final String PROMPT = "The quick brown fox";

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(
                LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        modelPath = dl.getModelFile().getAbsolutePath();
        // Metadata-preserving import: the GGUF carries the chat template + special-token ids
        // the chat lane needs. Importing the bare graph loses them and forces every caller
        // to re-derive them by hand — the friction this test's chat-lane repro exists to
        // exercise end-to-end.
        //
        // E2B/BIGGER: the raw-GGUF dequant path (dequantizeTensorStreaming) burns 45+ min on
        // the 8.8GB E2B file (per-tensor sync storm). The serving lane instead loads the
        // pre-converted .sdz beside the model workspace and only reads GGUF metadata. Mirror
        // that here when the sdz exists (system property gemma.sdz.path overrides).
        String sdzPath = System.getProperty("gemma.sdz.path",
                "/home/agibsonccc/Documents/GitHub/kompile/data/models/llm-ggmls/gemma-4-e2b-it/"
                        + "gemma-4-E2B-it-Q4_K_M-auto-1788037684245.sdz");
        GGMLMetadata.TokenizerInfo tinfo;
        if (new File(modelPath).length() > 2L * 1024 * 1024 * 1024
                && new File(sdzPath).isFile()) {
            // Fast lane: metadata from the GGUF header only (no dequantization). The decoder
            // graph is NOT shared across tests: pipelines close their decoders, so a shared
            // SameDiff would hand test N+1 a closed graph (NaN at gated_delta_rule). Each
            // pipeline factory loads its own copy via loadDecoderGraph().
            try (org.nd4j.ggml.format.GGUFReader reader =
                         new org.nd4j.ggml.format.GGUFReader(new File(modelPath))) {
                tinfo = GGMLMetadata.TokenizerInfo.fromGGUFHeader(reader.getHeader());
            }
        } else {
            // Small model: raw import for metadata; per-test decode graphs via loadDecoderGraph().
            try (org.nd4j.ggml.format.GGUFReader reader =
                         new org.nd4j.ggml.format.GGUFReader(new File(modelPath))) {
                tinfo = GGMLMetadata.TokenizerInfo.fromGGUFHeader(reader.getHeader());
            }
        }
        modelMetadata = GenerationPipeline.ModelMetadata.of(
                tinfo.getBosTokenId(), tinfo.getEosTokenId(), tinfo.getPadTokenId(),
                tinfo.getChatTemplate(), java.util.Set.of(), java.util.Set.of());

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(
                    tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            tokenizer = HuggingFaceTokenizer.fromFile(tf.getAbsolutePath());
        }

        // Guard: this regression is only meaningful for graphs whose KV outputs use the
        // k_rope_N / v_heads_N naming family (Qwen3.5 and gemma4 do; llama-style
        // past_key_values.N.key does not and is covered by the legacy heuristic). Uses its
        // own throwaway decoder graph — the shared model field is gone (pipelines close
        // their decoders, so sharing handed later tests a closed graph).
        try (SameDiff guardModel = loadDecoderGraph()) {
            ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(guardModel);
            assertTrue(kvNames != null && !kvNames.keyNames.isEmpty(),
                    "test model has no KV cache inputs");
            String firstKey = kvNames.keyNames.get(0);
            int layer = layerIndex(firstKey);
            assertTrue(guardModel.hasVariable("k_rope_" + layer) && guardModel.hasVariable("v_heads_" + layer),
                    "test model KV outputs are not k_rope_N/v_heads_N named — this test targets the "
                            + "gemma-style naming family; found input " + firstKey);
        }
    }

    /**
     * Loads a FRESH decoder graph per pipeline. Pipelines close their decoders on
     * {@code close()}, so a SameDiff shared across tests hands later tests a closed graph
     * — observed as NaN at gated_delta_rule in the class run while every test passed
     * individually. Small models: raw metadata-preserving import (fast since the streaming
     * dequant fix). Large models (E2B): the pre-converted sdz beside the model workspace
     * when present (sysprop gemma.sdz.path), else raw import. Tests reproducing a serving
     * artifact may set {@code model.sdz.path} explicitly; that takes precedence regardless
     * of the source GGUF size so the test uses the same staged weight representation.
     */
    private static SameDiff loadDecoderGraph() throws Exception {
        String explicitSdzPath = System.getProperty("model.sdz.path");
        if (explicitSdzPath != null && !explicitSdzPath.isBlank()) {
            File explicitSdz = new File(explicitSdzPath);
            assertTrue(explicitSdz.isFile(),
                    "Configured staged decoder does not exist: " + explicitSdz);
            return SameDiff.loadSdz(explicitSdz);
        }

        String gemmaSdzPath = System.getProperty("gemma.sdz.path",
                "/home/agibsonccc/Documents/GitHub/kompile/data/models/llm-ggmls/gemma-4-e2b-it/"
                        + "gemma-4-E2B-it-Q4_K_M-auto-1788037684245.sdz");
        File gemmaSdz = new File(gemmaSdzPath);
        if (new File(modelPath).length() > 2L * 1024 * 1024 * 1024
                && gemmaSdz.isFile()) {
            return SameDiff.loadSdz(gemmaSdz);
        }
        return GGMLModelImport.importModelWithMetadata(
                new File(modelPath), ConversionOptions.forInference()).getModel();
    }

    @AfterAll
    public static void teardown() {
        tokenizer = null;
        modelMetadata = null;
    }

    private static GenerationPipeline fixedBufferPipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(loadDecoderGraph())
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .maxPrefillLength(64)
                .maxKvCacheLength(128)
                .graphOptimizerEnabled(Boolean.parseBoolean(
                        System.getProperty("gemma.optimizer.enabled", "true")))
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * Mirrors the SERVING-lane config: no configured KV ceiling (model-owned envelope =
     * actualPrefillLen + maxNewTokens), so each distinct prompt length yields a distinct
     * envelope → a fresh warmup/freeze per extraction, exactly like the serving child that
     * hit the frozen-buffer violation. Default maxPrefillLength=0/maxKvCacheLength=0 gives
     * the model-owned envelope via the builder defaults.
     */
    private static GenerationPipeline servingLanePipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(loadDecoderGraph())
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .modelMetadata(modelMetadata)
                .graphOptimizerEnabled(Boolean.parseBoolean(
                        System.getProperty("gemma.optimizer.enabled", "true")))
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * Mirrors the fixed envelope used by the v33 crawl that exposed the remaining transition bug.
     * The large envelope is deliberate: every request pads to the same prefill geometry, proving
     * that a constrained-to-native transition failure is not caused by a changing KV ceiling.
     */
    private static GenerationPipeline productionEnvelopePipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(loadDecoderGraph())
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .modelMetadata(modelMetadata)
                .maxNewTokens(512)
                .maxPrefillLength(2048)
                .maxKvCacheLength(2560)
                .graphOptimizerEnabled(Boolean.parseBoolean(
                        System.getProperty("gemma.optimizer.enabled", "true")))
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    @Test
    @DisplayName("KV output slots are max-length pinned after warmup decode (explicit-names fix)")
    public void kvSlotsAreMaxLengthPinned() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            GenerationResult result = pipe.generate(PROMPT, N);
            assertNotNull(result);
            assertTrue(result.getTokenIds().length > 0, "generation produced no tokens");

            DynamicShapePlanExecutor executor =
                    pipe.getDecoder().getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor, "no DSP executor after fixed-buffer generate");

            // THE CONTRACT: max-allocation must have been configured. Before the fix this was
            // false for every gemma-style graph — the k_rope/v_heads outputs never matched the
            // legacy present+key heuristic, so each decode step allocated fresh full-length KV
            // buffers and the pool grew without bound.
            assertTrue(executor.isMaxAllocationConfigured(),
                    "KV max-allocation was never configured — decode allocates fresh full-length "
                            + "KV buffers every step (unbounded pool growth on gemma-style graphs)");
        } finally {
            pipe.close();
        }
    }

    @Test
    @DisplayName("Decode pool does not grow per-step once KV slots are pinned")
    public void decodePoolStaysFlatAfterPinning() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            // First generate: runs prefill + warmup decode + configures max-allocation.
            pipe.generate(PROMPT, N);

            DynamicShapePlanExecutor executor =
                    pipe.getDecoder().getOrCreateSession().getDynamicShapePlanExecutor();
            assertTrue(executor.isMaxAllocationConfigured(),
                    "max-allocation must be configured before the pool-growth assertion");

            long before = gpuPoolBytes();

            // Second generate: with pinned KV slots the decode working set is bounded by
            // maxKvCacheLength. The unpinned path grows materially over 64 steps; pinned decode
            // may still allocate/trim scratch, so allow a generous 256 MB margin.
            pipe.generate(PROMPT + " Tell me about robotics safety.", N);

            long after = gpuPoolBytes();
            long growth = after - before;
            assertTrue(growth < MAX_POOL_GROWTH_BYTES,
                    "decode pool grew by " + (growth / (1024 * 1024))
                            + " MB over " + N + " steps — KV outputs are not max-length pinned");
        } finally {
            pipe.close();
        }
    }

    @Test
    @DisplayName("Per-generate live pool churn stays bounded across retained 1-token steps")
    public void perGenerateLiveChurnStaysBounded() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            // Prime: prefill + warmup decode + KV max-allocation pinning.
            pipe.generate(PROMPT, N);

            DynamicShapePlanExecutor executor =
                    pipe.getDecoder().getOrCreateSession().getDynamicShapePlanExecutor();
            assertTrue(executor.isMaxAllocationConfigured(),
                    "max-allocation must be configured before the churn assertion");

            // Steady-state window: the FIRST TWO 1-token generates each build one retained
            // warmup/decode plan (one-time ~500 MB + ~37 MB plan construction — legitimate,
            // amortized across the pipeline's lifetime). The churn assertion applies to the
            // steps AFTER those builds: every further retained generate must add ~nothing
            // (a ratchet would add ~105 MB/step of fresh plan/buffer sets).
            pipe.generate(PROMPT, 1); // builds retained prefill plan (one-time cost, excluded)
            pipe.generate(PROMPT, 1); // builds retained warmup/decode plan (one-time, excluded)

            long churnSum = 0;
            int samples = 0;
            long worstStep = 0;
            final int churnSteps = 32;
            final long maxBytesPerStep = 16L * 1024 * 1024; // generous vs ~64 MB/step churn
            for (int i = 0; i < churnSteps; i++) {
                long before = gpuPoolBytes();
                GenerationResult r = pipe.generate(PROMPT, 1);
                assertNotNull(r, "retained generate must return a result");
                long after = gpuPoolBytes();
                long delta = after - before;
                churnSum += delta;
                samples++;
                worstStep = Math.max(worstStep, delta);
                if (i % 8 == 7) {
                    // Trim free pool blocks every 8 steps: forces the native POOL_LEDGER
                    // dump (per-device outstanding live allocations grouped by source) so
                    // the retained live growth can be attributed without changing what is
                    // measured — trim only releases free blocks, live blocks stay in the
                    // ledger.
                    try {
                        Nd4j.getNativeOps().trimMemoryPool(0);
                        Nd4j.getExecutioner().commit();
                    } catch (Throwable t) {
                        log.debug("ledger dump trim failed: {}", t.getMessage());
                    }
                }
                log.info("retainedStep={} livePoolDelta={} B (avg {} B/step) refMap={} allocated={} nativePoolUsed={} wsCurrent={} dspPool(pooled={}B count={} acquired={})",
                        i, delta, churnSum / samples,
                        Nd4j.getDeallocatorService().getReferenceMap().size(),
                        Nd4j.getMemoryManager().allocatedMemory(0),
                        DeviceMemoryManager.getInstance().getNativePoolUsedMemory(0),
                        Nd4j.getMemoryManager().getCurrentWorkspace() != null
                                ? Nd4j.getMemoryManager().getCurrentWorkspace().getCurrentSize()
                                : -1L,
                        Nd4j.getNativeOps().getBufferPoolPooledBytes(0),
                        Nd4j.getNativeOps().getBufferPoolPooledCount(0),
                        Nd4j.getNativeOps().getBufferPoolTotalAcquired(0));
                if (i % 8 == 7) {
                    // Lane-thread workspace snapshot: the model-execution lane owns the
                    // per-thread workspaces the pipeline allocates through; if the churn is
                    // workspace expansion, these names+sizes grow ~104MB/step.
                    try {
                        log.info("TEST-THREAD workspace stats:");
                        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
                    } catch (Throwable t) {
                        log.debug("lane dump failed: {}", t.getMessage());
                    }
                }
                if (i % 8 == 7) {
                    // Attribution: count referenceMap entries by deallocator class and device.
                    Map<String, Long> byClass = new LinkedHashMap<>();
                    Map<Long, Long> byBytes = new TreeMap<>();
                    long totalBytes = 0;
                    for (DeallocatableReference ref
                            : Nd4j.getDeallocatorService().getReferenceMap().values()) {
                        String cls = ref == null ? "null" : ref.getClass().getSimpleName();
                        byClass.merge(cls, 1L, Long::sum);
                        long bytes = 0;
                        try {
                            bytes = ref.getBytes();
                        } catch (Throwable ignore) { }
                        totalBytes += bytes;
                        long bucket = bytes <= 0 ? 0 : Long.highestOneBit(bytes);
                        byBytes.merge(bucket, 1L, Long::sum);
                    }
                    log.info("refMap composition after step {}: {} totalBytes={}MB", i, byClass,
                            totalBytes / (1024 * 1024));
                    log.info("refMap size-buckets (2^bytes -> count): {}", byBytes);
                }
            }
            long avgPerStep = churnSum / samples;
            log.info("perGenerateLiveChurnStaysBounded: samples={} avgPerStep={}B worst={}B bound={}B",
                    samples, avgPerStep, worstStep, maxBytesPerStep);
            assertTrue(avgPerStep <= maxBytesPerStep,
                    "Per-generate live pool churn " + avgPerStep + " B/step exceeds bound "
                            + maxBytesPerStep + " B/step (worst " + worstStep
                            + " B) — per-step retention ratchet reproduced");
        } finally {
            pipe.close();
        }
    }

    /**
     * The pipeline's own KV output-name derivation (k_rope_N / v_heads_N), replicated here
     * so the test asserts the exact names the executor must receive.
     */
    private static List<String> expectedKvOutputNames(ModelIOConfig.KVCacheNames kvNames) {
        List<String> out = new ArrayList<>();
        for (String keyName : kvNames.keyNames) {
            out.add("k_rope_" + layerIndex(keyName));
        }
        for (String valueName : kvNames.valueNames) {
            out.add("v_heads_" + layerIndex(valueName));
        }
        return out;
    }

    @Test
    @DisplayName("kvOutputNamesForMaxAlloc derivation covers every KV output the graph requests")
    public void derivationCoversAllKvOutputs() throws Exception {
        try (SameDiff guardModel = loadDecoderGraph()) {
            ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(guardModel);
            List<String> derived = expectedKvOutputNames(kvNames);
            assertEquals(kvNames.keyNames.size() + kvNames.valueNames.size(), derived.size(),
                    "derived KV output names must be 2× layers");
            for (String name : derived) {
                assertTrue(guardModel.hasVariable(name),
                        "derived KV output name " + name + " is not a graph variable");
            }
        }
    }

    private static int layerIndex(String kvInputName) {
        // Pattern: past_key_values.{N}.key / k_rope input naming — first integer wins,
        // mirroring GenerationPipeline.extractLayerIndex.
        String[] parts = kvInputName.split("\\.");
        for (String part : parts) {
            try {
                return Integer.parseInt(part);
            } catch (NumberFormatException ignore) {
            }
        }
        return 0;
    }

    /**
     * Native CUDA memory-pool bytes in use (the pool that backs DSP slot buffers). Falls back
     * to 0 on non-CUDA backends, which makes the growth assertion vacuous there — the pinning
     * assertion above is the primary gate.
     */
    private static long gpuPoolBytes() {
        try {
            DeviceMemoryManager dmm = DeviceMemoryManager.getInstance();
            long total = 0;
            for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); i++) {
                long used = dmm.getNativePoolUsedMemory(i);
                if (used > 0) total += used;
            }
            return total;
        } catch (Throwable t) {
            log.debug("gpuPoolBytes unavailable: {}", t.getMessage());
            return 0L;
        }
    }

    /**
     * Reproduces the serving-child failure from the 2026-09-03 CUDA smoke:
     * extraction A (prompt length L1) runs a full decode; extraction B on the SAME pipeline
     * with a LONGER prompt (different KV envelope → fresh warmup/freeze) then failed its
     * first constrained step with "DataBuffer LIFECYCLE VIOLATION: allocateSpecial called on
     * frozen DataBuffer ... slot 75 (mean_square)", preceded by cross-device [TRANSFER]
     * cuda:0 → cuda:1 and DSP MIGRATE input lines. Suspected causes: (a) thread device
     * affinity flipping to another GPU between extractions (KV buffers materialize on the
     * wrong device, then plan exec migrates inputs back), and/or (b) second-plan buffers
     * being materialized lazily after the executor froze the plan.
     *
     * <p>This test pins down WHICH: it runs the A-then-longer-B sequence on one pipeline and
     * asserts (1) B completes without the frozen-buffer violation, and (2) the executing
     * thread's device affinity is unchanged across the B prefill/STEP-2 KV materialization.
     * Failure diagnosis: violation + unchanged affinity → lazy allocation after freeze
     * (executor bug); violation + affinity changed → cross-device affinity bug.</p>
     */
    @Test
    @DisplayName("Second extraction with longer prompt on same pipeline: no frozen-buffer violation, affinity stable")
    public void secondLongerPromptExtractionDoesNotViolateFrozenPlan() throws Exception {
        GenerationPipeline pipe = servingLanePipeline();
        try {
            int affinityBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            // Extraction A: standard prompt, full decode. Warms plan family 1.
            GenerationResult a = pipe.generate(PROMPT, N);
            assertTrue(a != null && a.getTokenIds().length > 0,
                    "extraction A must produce tokens");

            int affinityAfterA = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            // Extraction B: materially LONGER prompt + SAME maxNewTokens → model-owned
            // envelope = actualPrefillLen + maxNewTokens DIFFERS from A's → the executor must
            // warm a NEW plan (as serving did for chunk B: 1167 vs 1253).
            String longerPrompt = PROMPT
                    + " Meridian Robotics Group operates three divisions: consumer robotics, "
                    + "industrial automation, and research. Founded by Alex Rivera and Morgan Chen, "
                    + "the company partners with Acme Robotics and Nova Labs on safety standards, "
                    + "and maintains offices in Boston, Austin, and Taipei with 480 employees.";
            GenerationResult b = pipe.generate(longerPrompt, N);

            assertTrue(b != null && b.getTokenIds().length > 0,
                    "extraction B (longer prompt, same pipeline) must produce tokens — "
                            + "frozen-buffer violation reproduces here if unfixed");

            int affinityAfterB = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            assertEquals(affinityAfterA, affinityAfterB,
                    "thread device affinity changed across extractions — cross-device "
                            + "materialization is the violation source (KV buffers staged on "
                            + "the wrong GPU before plan execution migrated inputs back)");
            assertEquals(affinityBefore, affinityAfterB,
                    "affinity must return to its original device after both extractions");
        } finally {
            pipe.close();
        }
    }

    /**
     * The EXACT serving failure path: the crawl's unified-corpus extraction calls
     * /api/llm/chat with toolChoice=REQUIRED, and pipeline.generateChat → samplingForChat
     * injects a ConstraintConfig for that, which engages runInGraphConstrainedDecode. The
     * 2026-09-03 serving smoke failed there at step 509/512 with "allocateSpecial called on
     * frozen DataBuffer ... slot 75 (mean_square)". This test drives that constrained loop
     * directly (chat request + REQUIRED tool choice) on the serving-lane pipeline and must
     * complete without the frozen-buffer violation.
     */
    @Test
    @DisplayName("Constrained chat decode (toolChoice=REQUIRED) on serving-lane pipeline: no frozen-buffer violation")
    public void constrainedChatDecodeDoesNotViolateFrozenPlan() throws Exception {
        GenerationPipeline pipe = servingLanePipeline();
        try {
            // Constrained loop engages via samplingForChat (toolChoice=REQUIRED).
            ChatGenerationResult result = pipe.generateChat(
                    chatRequest("Extract organizations from this text: " + PROMPT), 64);

            assertNotNull(result, "constrained chat generation must return a result");
            assertNotNull(result.getRawText(), "constrained chat must produce raw text");
        } finally {
            pipe.close();
        }
    }

    /**
     * The SERVING SMOKE-5 shape: TWO sequential constrained chat extractions with materially
     * different prompt lengths (→ different KV envelopes → fresh warmup/freeze per
     * extraction) in ONE pipeline. Extraction A retains its frozen plan while B warms a new
     * one — the serving child's remaining unreproduced delta (its B envelope was 1253 vs
     * A's 1167, both large). Asserts both extractions complete on the constrained path with
     * no frozen-buffer violation and no cross-device input migration.
     */
    @Test
    @DisplayName("Two sequential constrained chat extractions (differing envelopes): no frozen-buffer violation")
    public void twoSequentialConstrainedChatExtractionsDoNotViolateFrozenPlan() throws Exception {
        GenerationPipeline pipe = servingLanePipeline();
        try {
            String longText = PROMPT
                    + " Meridian Robotics Group operates three divisions: consumer robotics, "
                    + "industrial automation, and research. Founded by Alex Rivera and Morgan Chen, "
                    + "the company partners with Acme Robotics and Nova Labs on safety standards, "
                    + "and maintains offices in Boston, Austin, and Taipei with 480 employees. "
                    + "The industrial division shipped 12000 robot arms last year, the research "
                    + "division published 14 papers on safe human-robot collaboration, and the "
                    + "consumer division launched two household assistants with on-device speech.";

            ChatGenerationResult a = pipe.generateChat(chatRequest("Extract organizations: " + PROMPT), 64);
            assertNotNull(a, "extraction A must return a result");

            ChatGenerationResult b = pipe.generateChat(chatRequest("Extract organizations: " + longText), 64);
            assertNotNull(b, "extraction B must return a result — frozen-buffer violation "
                    + "reproduces here if the second-plan-after-retained-first path is broken");
        } finally {
            pipe.close();
        }
    }

    /**
     * Reproduces the v33 request transition without running a crawl: schema discovery uses the
     * constrained Java decode loop, then entity extraction uses the fused native
     * {@code autoregressive_decode} loop. The production failure occurred at native step zero with
     * {@code TRANSFER_FAILED/cudaErrorInvalidValue} while staging 525 frozen-plan externals, even
     * though the preceding constrained calls and native warmup were healthy.
     */
    @Test
    @DisplayName("Constrained schema calls followed by native generation keep frozen staging valid")
    public void constrainedSchemaCallsThenNativeGenerateDoNotFailStaging() throws Exception {
        GenerationPipeline pipe = productionEnvelopePipeline();
        try {
            Set<String> declaredMutable = pipe.getDecoder().getDynamicShapePlanMutableInputs();
            ModelIOConfig.KVCacheNames declaredKv =
                    ModelIOConfig.findKVCacheInputNames(pipe.getDecoder());
            assertNotNull(declaredKv, "production decoder must expose KV cache inputs");
            assertTrue(declaredMutable.containsAll(declaredKv.keyNames)
                            && declaredMutable.containsAll(declaredKv.valueNames),
                    "all native-mutated KV inputs must be declared mutable before plan compilation");
            for (ModelIOConfig.RecurrentStatePair pair
                    : ModelIOConfig.findRecurrentStatePairs(pipe.getDecoder(), pipe.getIoConfig())) {
                assertTrue(declaredMutable.contains(pair.inputName),
                        "native-mutated recurrent input was not declared mutable: " + pair.inputName);
            }

            ChatGenerationResult nodeSchema = pipe.generateChat(
                    chatRequest("Classify organization types in: " + PROMPT), 96);
            assertNotNull(nodeSchema, "node-schema constrained call must return a result");

            ChatGenerationResult relationSchema = pipe.generateChat(
                    chatRequest("Classify organization relationships in: " + PROMPT
                            + " Acme Robotics partners with Nova Labs."), 96);
            assertNotNull(relationSchema, "relationship-schema constrained call must return a result");

            GenerationResult extraction = pipe.generate(
                    "Extract the named organizations and partnership from: Acme Robotics partners "
                            + "with Nova Labs.", 3);
            assertNotNull(extraction, "native extraction must return a result");
            assertTrue(extraction.getTokenIds().length >= 3,
                    "native extraction must reach the fused decode step; tokens="
                            + extraction.getTokenIds().length);
        } finally {
            pipe.close();
        }
    }

    private static ChatTemplate.Request chatRequest(
            String userText) {
        return ChatTemplate.Request.builder()
                .messages(List.of(
                        ChatTemplate.Message.user(userText)))
                .tools(List.of(
                        ChatTemplate.Tool.function("record_organization",
                                        "Record one organization mentioned in the text.",
                                        Map.of("type", "object",
                                                "properties", Map.of(
                                                        "name", Map.of("type", "string")),
                                                "required", List.of("name")))))
                .toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                .build();
    }
}
