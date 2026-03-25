package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestMethodOrder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for the best benchmark config accuracy issue.
 *
 * TRITON_compileAll_best_ATTN_gc_argOpt_batchOps produces repetitive
 * {@literal '<text\n<text\n<text'} output instead of proper SmolDocling tags.
 * Even SLOT_BY_SLOT baseline repeats — suggesting pipeline-level cause.
 *
 * Tests are ordered to run cheapest first, reuse shared embeddings,
 * and properly reset state between tests to avoid OOM.
 *
 * Key interactions tested:
 * 1. Manual decode (growing KV) vs StaticKvCacheDecodeLoop (padded KV)
 * 2. Attention mask format: contiguous all-ones vs sparse padded
 * 3. outputDirect vs output within StaticKvCacheDecodeLoop
 * 4. Step-by-step token divergence between manual and loop decode
 * 5. Best Triton config end-to-end
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestBenchmarkAccuracyIsolation {

    private SameDiff visionEncoder;
    private SameDiff decoder;
    private SameDiff embedTokens;
    private HuggingFaceTokenizer tokenizer;

    // Shared across tests — computed once in @BeforeAll
    private INDArray benchmarkEmbeddings;
    private int[] benchmarkPromptTokenIds;

    // Reference results from manual decode — set by test 1, used by test 3
    private List<Integer> manualDecodeTokens;

    @BeforeAll
    public void setup() throws Exception {
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedResult.getModelFile().getAbsolutePath());
        visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        log.info("Models loaded: vision={} ops, decoder={} ops, embed={} ops",
                visionEncoder.ops().length, decoder.ops().length, embedTokens.ops().length);

        // Build benchmark embeddings ONCE (512x512 image)
        benchmarkEmbeddings = buildBenchmarkEmbeddings();
        benchmarkPromptTokenIds = getBenchmarkPromptTokenIds();

        // Free vision encoder after embedding computation — we don't need it anymore
        visionEncoder.resetSession();
        visionEncoder.clearPlaceholders(true);
        visionEncoder.clearOpInputs();
        Nd4j.getExecutioner().commit();
        System.gc();

        log.info("Benchmark embeddings: shape={}", Arrays.toString(benchmarkEmbeddings.shape()));
    }

    // =========================================================================
    // Test 1: Manual decode — the BASELINE reference
    // Uses growing KV cache + contiguous all-ones attention mask + sd.output()
    // This is what TestVLMDecodeQuality does and it works correctly.
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("1. Manual decode with benchmark image (512x512) — baseline reference")
    public void testManualDecodeBaseline() {
        resetDecoder();

        manualDecodeTokens = manualDecode(benchmarkEmbeddings.dup(), 15);
        String text = tokenizer.decode(manualDecodeTokens.stream().mapToInt(i -> i).toArray(), false);

        log.info("MANUAL DECODE -> {} tokens: '{}'", manualDecodeTokens.size(), text);

        Set<Integer> unique = new HashSet<>(manualDecodeTokens);
        log.info("Unique: {}/{}", unique.size(), manualDecodeTokens.size());

        for (int i = 0; i < manualDecodeTokens.size(); i++) {
            String decoded = tokenizer.decode(new int[]{manualDecodeTokens.get(i)}, false);
            log.info("  Step {}: id={} text='{}'", i, manualDecodeTokens.get(i), decoded);
        }

        assertTrue(unique.size() >= 5,
                "Manual decode should produce >= 5 unique tokens, got " + unique.size()
                        + ". Text: '" + text + "'");

        // Clean up decoder state to free GPU memory
        resetDecoder();
    }

    // =========================================================================
    // Test 2: StaticKvCacheDecodeLoop with sd.output() only (force no outputDirect)
    // Tests the padded KV + sparse attention mask interaction WITHOUT outputDirect.
    // If this repeats but test 1 passes, the sparse mask is the problem.
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("2. StaticKvCacheDecodeLoop forced sd.output() (no outputDirect)")
    public void testStaticKvLoopForcedOutput() {
        resetDecoder();

        System.setProperty("nd4j.dsp.noDirect", "true");
        try {
            StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(15)
                    .hiddenSize((int) benchmarkEmbeddings.shape()[2])
                    .build();

            GenerationResult result = loop.decode(benchmarkEmbeddings.dup(), benchmarkPromptTokenIds);

            log.info("STATIC_KV + output() -> {} tokens: '{}'", result.getGeneratedTokenCount(), result.getText());

            Set<Integer> unique = new HashSet<>();
            for (int id : result.getTokenIds()) unique.add(id);
            log.info("Unique: {}/{}", unique.size(), result.getGeneratedTokenCount());

            for (int i = 0; i < result.getTokenIds().length; i++) {
                String decoded = tokenizer.decode(new int[]{result.getTokenIds()[i]}, false);
                log.info("  Step {}: id={} text='{}'", i, result.getTokenIds()[i], decoded);
            }

            if (unique.size() < 5 && manualDecodeTokens != null) {
                Set<Integer> manualUnique = new HashSet<>(manualDecodeTokens);
                if (manualUnique.size() >= 5) {
                    fail("StaticKvCacheDecodeLoop repeats with sd.output() while manual decode is diverse. "
                            + "Root cause: padded KV + sparse attention mask interaction. "
                            + "Unique=" + unique.size() + " text='" + result.getText() + "'");
                }
            }
        } finally {
            System.clearProperty("nd4j.dsp.noDirect");
            resetDecoder();
        }
    }

    // =========================================================================
    // Test 3: Step-by-step divergence — at which step does StaticKvCacheDecodeLoop
    // diverge from manual decode? Compares token-by-token.
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("3. Step-by-step divergence: manual vs StaticKvCacheDecodeLoop")
    public void testStepByStepDivergence() {
        if (manualDecodeTokens == null || manualDecodeTokens.isEmpty()) {
            log.warn("Skipping — manual decode tokens not available (test 1 may have failed)");
            return;
        }

        resetDecoder();

        System.setProperty("nd4j.dsp.noDirect", "true");
        try {
            StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(10)
                    .hiddenSize((int) benchmarkEmbeddings.shape()[2])
                    .build();

            GenerationResult loopResult = loop.decode(benchmarkEmbeddings.dup(), benchmarkPromptTokenIds);

            log.info("Manual tokens: {}", manualDecodeTokens.subList(0, Math.min(10, manualDecodeTokens.size())));
            log.info("Loop tokens:   {}", Arrays.toString(loopResult.getTokenIds()));

            int divergeStep = -1;
            int maxCompare = Math.min(Math.min(manualDecodeTokens.size(), loopResult.getTokenIds().length), 10);
            for (int i = 0; i < maxCompare; i++) {
                int manual = manualDecodeTokens.get(i);
                int loopTok = loopResult.getTokenIds()[i];
                String manualText = tokenizer.decode(new int[]{manual}, false);
                String loopText = tokenizer.decode(new int[]{loopTok}, false);
                boolean match = manual == loopTok;
                log.info("Step {}: manual={}('{}') loop={}('{}') {}",
                        i, manual, manualText, loopTok, loopText, match ? "MATCH" : "*** DIVERGE ***");
                if (!match && divergeStep < 0) {
                    divergeStep = i;
                }
            }

            if (divergeStep >= 0) {
                log.warn("FINDING: Outputs diverge at step {}.", divergeStep);
                if (divergeStep == 0) {
                    log.warn("  Diverges at PREFILL — same code path, different result. Check session state.");
                } else if (divergeStep == 1) {
                    log.warn("  Diverges at first DECODE step — padded KV transition is suspect.");
                } else {
                    log.warn("  Diverges at step {} — KV scatter or attention mask accumulation error.", divergeStep);
                }
            } else {
                log.info("FINDING: Manual and StaticKvCacheDecodeLoop produce IDENTICAL tokens for {} steps", maxCompare);
            }
        } finally {
            System.clearProperty("nd4j.dsp.noDirect");
            resetDecoder();
        }
    }

    // =========================================================================
    // Test 4: StaticKvCacheDecodeLoop with view-based KV (no padding)
    // Disables padded mode to test growing KV within StaticKvCacheDecodeLoop.
    // If this works but test 2 fails, the padded mode is the root cause.
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("4. StaticKvCacheDecodeLoop with view-based KV (no padding)")
    public void testStaticKvViewBased() {
        resetDecoder();

        System.setProperty("nd4j.dsp.noPadded", "true");
        System.setProperty("nd4j.dsp.noDirect", "true");
        try {
            StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(15)
                    .hiddenSize((int) benchmarkEmbeddings.shape()[2])
                    .build();

            GenerationResult result = loop.decode(benchmarkEmbeddings.dup(), benchmarkPromptTokenIds);

            log.info("STATIC_KV VIEW-BASED -> {} tokens: '{}'", result.getGeneratedTokenCount(), result.getText());

            Set<Integer> unique = new HashSet<>();
            for (int id : result.getTokenIds()) unique.add(id);
            log.info("Unique: {}/{}", unique.size(), result.getGeneratedTokenCount());

            for (int i = 0; i < result.getTokenIds().length; i++) {
                String decoded = tokenizer.decode(new int[]{result.getTokenIds()[i]}, false);
                log.info("  Step {}: id={} text='{}'", i, result.getTokenIds()[i], decoded);
            }
        } finally {
            System.clearProperty("nd4j.dsp.noPadded");
            System.clearProperty("nd4j.dsp.noDirect");
            resetDecoder();
        }
    }

    // =========================================================================
    // Test 5: StaticKvCacheDecodeLoop with SLOT_BY_SLOT + padded (benchmark baseline)
    // This is what the benchmark's DIAG_SLOT_BY_SLOT_baseline config does.
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("5. StaticKvCacheDecodeLoop + SLOT_BY_SLOT + padded (benchmark baseline)")
    public void testSlotBySlotPadded() {
        resetDecoder();

        BenchmarkConfig config = BenchmarkConfig.create("test_slot_by_slot")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(15).minDiversityPct(0);
        BenchmarkConfigApplier.apply(config);

        decoder.compileNativeDynamicShapePlan(decoder.outputs(), GraphExecutionMode.SLOT_BY_SLOT, true);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(15)
                .hiddenSize((int) benchmarkEmbeddings.shape()[2])
                .build();

        GenerationResult result = loop.decode(benchmarkEmbeddings.dup(), benchmarkPromptTokenIds);

        log.info("SLOT_BY_SLOT PADDED -> {} tokens: '{}'", result.getGeneratedTokenCount(), result.getText());

        Set<Integer> unique = new HashSet<>();
        for (int id : result.getTokenIds()) unique.add(id);
        log.info("Unique: {}/{}", unique.size(), result.getGeneratedTokenCount());

        for (int i = 0; i < result.getTokenIds().length; i++) {
            String decoded = tokenizer.decode(new int[]{result.getTokenIds()[i]}, false);
            log.info("  Step {}: id={} text='{}'", i, result.getTokenIds()[i], decoded);
        }
    }

    // =========================================================================
    // Test 6: Attention mask verification — compare what manual decode
    // and padded mode produce at step 2 for the actual model inputs.
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("6. Attention mask: manual (all-ones) vs padded (sparse) produce same logits at step 2")
    public void testAttentionMaskEquivalence() {
        resetDecoder();
        INDArray embeddingTable = extractEmbeddingTable();
        assertNotNull(embeddingTable, "Embedding table not found");

        long seqLen = benchmarkEmbeddings.shape()[1];
        long hiddenSize = benchmarkEmbeddings.shape()[2];

        String logitsName = findLogitsName();
        List<String> kvOutputNames = findKvOutputNames();
        List<String> allOutputs = new ArrayList<>();
        allOutputs.add(logitsName);
        allOutputs.addAll(kvOutputNames);

        // === PREFILL (same for both paths) ===
        Map<String, INDArray> inputs = buildPrefillInputs(seqLen);
        Map<String, INDArray> prefillResult = decoder.output(inputs, allOutputs.toArray(new String[0]));
        INDArray prefillLogits = prefillResult.get(logitsName);
        INDArray lastLogits = prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int firstToken = lastLogits.argMax().getInt(0);

        // Save KV for manual path
        Map<String, INDArray> manualKv = new HashMap<>();
        for (String kvName : kvOutputNames) {
            String pastName = kvName.replace("present", "past_key_values");
            manualKv.put(pastName, prefillResult.get(kvName).dup());
        }

        // Save KV for padded path (pad to static size)
        long maxKvLen = seqLen + 30;
        Map<String, INDArray> staticKv = new HashMap<>();
        for (String kvName : kvOutputNames) {
            String pastName = kvName.replace("present", "past_key_values");
            INDArray kv = prefillResult.get(kvName);
            long[] shape = kv.shape(); // [1, heads, seqLen, dim]
            INDArray padded = Nd4j.zeros(DataType.FLOAT, shape[0], shape[1], maxKvLen, shape[3]);
            padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, shape[2]), NDArrayIndex.all()).assign(kv);
            staticKv.put(pastName, padded);
        }

        // === STEP 1: Manual path (growing KV, all-ones mask) ===
        resetDecoder();
        INDArray tokenEmbed = embeddingTable.getRow(firstToken).reshape(1, 1, hiddenSize);
        Map<String, INDArray> manualInputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) manualInputs.put(name, tokenEmbed);
            else if (name.equals("attention_mask")) manualInputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen + 1));
            else if (name.equals("position_ids")) manualInputs.put(name, Nd4j.createFromArray(new long[]{seqLen}).reshape(1, 1));
            else if (name.startsWith("past_key_values.") && manualKv.containsKey(name)) manualInputs.put(name, manualKv.get(name));
        }
        Map<String, INDArray> manualStep1 = decoder.output(manualInputs, logitsName);
        INDArray manualLogits1 = manualStep1.get(logitsName).dup();
        int manualToken1 = manualLogits1.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all())
                .argMax().getInt(0);

        // === STEP 1: Padded path (full static KV, sparse mask) ===
        resetDecoder();
        long totalSeqLen = maxKvLen + 1;
        INDArray sparseMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        sparseMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, seqLen)).assign(1);
        sparseMask.putScalar(0, totalSeqLen - 1, 1); // current token

        Map<String, INDArray> paddedInputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) paddedInputs.put(name, tokenEmbed);
            else if (name.equals("attention_mask")) paddedInputs.put(name, sparseMask);
            else if (name.equals("position_ids")) paddedInputs.put(name, Nd4j.createFromArray(new long[]{seqLen}).reshape(1, 1));
            else if (name.startsWith("past_key_values.") && staticKv.containsKey(name)) paddedInputs.put(name, staticKv.get(name));
        }
        Map<String, INDArray> paddedStep1 = decoder.output(paddedInputs, logitsName);
        INDArray paddedLogits1 = paddedStep1.get(logitsName).dup();
        int paddedToken1 = paddedLogits1.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all())
                .argMax().getInt(0);

        // Compare
        String manualText = tokenizer.decode(new int[]{manualToken1}, false);
        String paddedText = tokenizer.decode(new int[]{paddedToken1}, false);
        log.info("Step 1 comparison:");
        log.info("  Manual (all-ones mask, growing KV):  token={} text='{}'", manualToken1, manualText);
        log.info("  Padded (sparse mask, static KV):     token={} text='{}'", paddedToken1, paddedText);

        // Compare top-5
        INDArray manualL = manualLogits1.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all());
        INDArray paddedL = paddedLogits1.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all());

        double maxLogitDiff = manualL.sub(paddedL).amaxNumber().doubleValue();
        log.info("  Max logit difference: {}", maxLogitDiff);

        if (manualToken1 != paddedToken1) {
            log.warn("FINDING: Manual and padded paths produce DIFFERENT tokens at step 1!");
            log.warn("  This means padded KV + sparse attention mask changes model behavior.");
            log.warn("  Root cause: attn_mask_reformat with sparse mask produces different attention bias.");

            // Log top-5 for each
            logTop5("Manual", manualL);
            logTop5("Padded", paddedL);
        } else {
            log.info("  Tokens MATCH at step 1.");
        }

        // Clean up static KV
        for (INDArray buf : staticKv.values()) {
            if (buf != null && !buf.wasClosed()) { buf.setCloseable(true); buf.close(); }
        }
        for (INDArray buf : manualKv.values()) {
            if (buf != null && !buf.wasClosed()) { buf.setCloseable(true); buf.close(); }
        }

        assertEquals(manualToken1, paddedToken1,
                "Padded path should produce same token as manual path at step 1. " +
                        "Manual='" + manualText + "' Padded='" + paddedText + "' maxLogitDiff=" + maxLogitDiff);
    }

    // =========================================================================
    // Test 7: Best Triton config end-to-end (the actual benchmark config)
    // This runs LAST since it's the most expensive and depends on all others passing.
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("7. Best Triton config: TRITON_compileAll_best_ATTN_gc_argOpt_batchOps")
    public void testBestTritonConfig() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping");
            return;
        }

        resetDecoder();

        BenchmarkConfig config = BenchmarkConfig.create("test_triton_best")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .dspBatchedGemm(true)
                .dspCastSinkMatmul(true)
                .maxTokens(100).minDiversityPct(0);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(100)
                .hiddenSize((int) benchmarkEmbeddings.shape()[2])
                .build();

        GenerationResult result = loop.decode(benchmarkEmbeddings.dup(), benchmarkPromptTokenIds);

        log.info("BEST TRITON -> {} tokens: '{}'", result.getGeneratedTokenCount(), result.getText());

        Set<Integer> unique = new HashSet<>();
        for (int id : result.getTokenIds()) unique.add(id);
        double diversityPct = (double) unique.size() / result.getTokenIds().length * 100;
        log.info("Unique: {}/{} ({}%)", unique.size(), result.getGeneratedTokenCount(),
                String.format("%.1f", diversityPct));

        // Performance
        long decodeTokens = result.getGeneratedTokenCount();
        double totalMs = result.getGenerationTimeMs();
        double tokPerSec = decodeTokens > 0 ? decodeTokens * 1000.0 / totalMs : 0;
        log.info("Performance: {} tok/s ({} tokens in {}ms)",
                String.format("%.1f", tokPerSec), decodeTokens, String.format("%.0f", totalMs));

        // Must produce at least 10 tokens
        assertTrue(result.getGeneratedTokenCount() >= 10,
                "Best Triton config should produce >= 10 tokens, got " + result.getGeneratedTokenCount());

        // Check for structural tags (doctag, text, section_header, etc.)
        String text = result.getText();
        boolean hasStructuralTag = text.contains("<doctag>") || text.contains("<text") ||
                text.contains("<section_header") || text.contains("<page") || text.contains("<otsl");
        assertTrue(hasStructuralTag,
                "Output should contain SmolDocling structural tags. Got: '" + text + "'");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void resetDecoder() {
        decoder.resetSession();
        decoder.clearPlaceholders(true);
        decoder.clearOpInputs();
        decoder.clearDynamicShapePlanCache();
        Nd4j.getExecutioner().commit();
    }

    private BufferedImage createBenchmarkImage() {
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 512, 512);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("This is a test page for the", 50, 220);
        g.drawString("SmolDocling VLM pipeline.", 50, 260);
        g.drawString("Section 2: Content", 50, 340);
        g.drawString("Lorem ipsum dolor sit amet,", 50, 400);
        g.drawString("consectetur adipiscing elit.", 50, 440);
        g.dispose();
        return img;
    }

    private INDArray buildBenchmarkEmbeddings() throws Exception {
        BufferedImage image = createBenchmarkImage();
        int targetSize = 512;
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);

        BufferedImage resized = ImageTiler.resizeLongestEdge(image, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resized, targetSize, 9);

        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();

        for (int i = 0; i < splitResult.getTotalFrames(); i++) {
            BufferedImage frame = splitResult.frames.get(i);
            INDArray frameTensor = preprocessor.preprocess(frame);
            INDArray singleFrame = frameTensor.reshape(1, 1, 3, targetSize, targetSize);

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String name : visionInputNames) {
                if (name.equals("pixel_values")) {
                    visionInputMap.put(name, singleFrame);
                } else if (name.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(i);
                    visionInputMap.put(name,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());

            singleFrame.close();
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
        }

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLen = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLen);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        String embedInputName = embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();

        return EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
    }

    private int[] getBenchmarkPromptTokenIds() throws Exception {
        int targetSize = 512;
        BufferedImage image = createBenchmarkImage();
        BufferedImage resized = ImageTiler.resizeLongestEdge(image, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resized, targetSize, 9);

        int imageSeqLen = 64;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLen);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        return tokenizer.encode(chatPrompt, false).getIds();
    }

    private INDArray extractEmbeddingTable() {
        INDArray embeddingTable = null;
        for (SDVariable var : embedTokens.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        return embeddingTable;
    }

    private String findLogitsName() {
        for (String name : decoder.outputs()) {
            if (name.contains("logit")) return name;
        }
        return decoder.outputs().get(0);
    }

    private List<String> findKvOutputNames() {
        List<String> kvNames = new ArrayList<>();
        for (String name : decoder.outputs()) {
            if (name.startsWith("present")) kvNames.add(name);
        }
        return kvNames;
    }

    private Map<String, INDArray> buildPrefillInputs(long seqLen) {
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) inputs.put(name, benchmarkEmbeddings.dup());
            else if (name.equals("attention_mask")) inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            else if (name.equals("position_ids")) inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            else if (name.startsWith("past_key_values.")) inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
        }
        return inputs;
    }

    private void logTop5(String label, INDArray logits) {
        int len = (int) logits.length();
        int[] topIds = new int[5];
        float[] topVals = new float[5];
        Arrays.fill(topVals, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < len; i++) {
            float v = logits.getFloat(i);
            for (int j = 0; j < 5; j++) {
                if (v > topVals[j]) {
                    System.arraycopy(topVals, j, topVals, j + 1, 4 - j);
                    System.arraycopy(topIds, j, topIds, j + 1, 4 - j);
                    topVals[j] = v;
                    topIds[j] = i;
                    break;
                }
            }
        }

        for (int i = 0; i < 5; i++) {
            String text = tokenizer.decode(new int[]{topIds[i]}, false);
            log.info("  {} top-{}: id={} logit={} text='{}'", label, i + 1, topIds[i], topVals[i], text);
        }
    }

    /**
     * Manual decode using sd.output() with growing KV cache — the REFERENCE implementation.
     */
    private List<Integer> manualDecode(INDArray inputsEmbeds, int maxSteps) {
        INDArray embeddingTable = extractEmbeddingTable();
        assertNotNull(embeddingTable, "Embedding table not found");

        long seqLen = inputsEmbeds.shape()[1];
        long hSize = inputsEmbeds.shape()[2];

        String logitsName = findLogitsName();
        List<String> kvOutputNames = findKvOutputNames();

        List<String> allOutputs = new ArrayList<>();
        allOutputs.add(logitsName);
        allOutputs.addAll(kvOutputNames);

        // Prefill
        Map<String, INDArray> inputs = buildPrefillInputs(seqLen);
        inputs.put("inputs_embeds", inputsEmbeds); // use the actual embeddings, not dup

        Map<String, INDArray> result = decoder.output(inputs, allOutputs.toArray(new String[0]));
        INDArray logits = result.get(logitsName);
        INDArray lastLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int firstToken = lastLogits.argMax().getInt(0);

        List<Integer> generated = new ArrayList<>();
        generated.add(firstToken);

        Map<String, INDArray> kvCache = new HashMap<>();
        for (String kvName : kvOutputNames) {
            String pastName = kvName.replace("present", "past_key_values");
            kvCache.put(pastName, result.get(kvName).dup());
        }
        long pastLen = seqLen;

        // Decode steps
        for (int step = 1; step < maxSteps; step++) {
            int prevToken = generated.get(generated.size() - 1);
            INDArray tokenEmbed = embeddingTable.getRow(prevToken).reshape(1, 1, hSize);

            Map<String, INDArray> decInputs = new HashMap<>();
            for (String name : decoder.inputs()) {
                if (name.equals("inputs_embeds")) decInputs.put(name, tokenEmbed);
                else if (name.equals("attention_mask")) decInputs.put(name, Nd4j.ones(DataType.LONG, 1, pastLen + 1));
                else if (name.equals("position_ids")) decInputs.put(name, Nd4j.createFromArray(new long[]{pastLen}).reshape(1, 1));
                else if (name.startsWith("past_key_values.") && kvCache.containsKey(name)) decInputs.put(name, kvCache.get(name));
            }

            result = decoder.output(decInputs, allOutputs.toArray(new String[0]));
            logits = result.get(logitsName);
            INDArray stepLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all());
            int nextToken = stepLogits.argMax().getInt(0);
            generated.add(nextToken);

            // Close old KV before replacing
            for (String kvName : kvOutputNames) {
                String pastName = kvName.replace("present", "past_key_values");
                INDArray oldKv = kvCache.get(pastName);
                if (oldKv != null && !oldKv.wasClosed()) { oldKv.setCloseable(true); oldKv.close(); }
                kvCache.put(pastName, result.get(kvName).dup());
            }
            pastLen++;

            if (nextToken == 49279) { // <end_of_utterance>
                log.info("Manual decode: EOS at step {}", step);
                break;
            }
        }

        // Clean up KV
        for (INDArray kv : kvCache.values()) {
            if (kv != null && !kv.wasClosed()) { kv.setCloseable(true); kv.close(); }
        }

        return generated;
    }
}
