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

package org.eclipse.deeplearning4j.llm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator.QualityReport;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator.ValidationConfig;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator.PerplexityResult;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Qwen3.5 LLM pipeline test with benchmarking and accuracy validation.
 *
 * Loads GGUF models via GGMLModelImport, runs text generation with GenerationPipeline,
 * benchmarks tok/s, and validates output quality (perplexity, diversity, coherence).
 *
 * Modeled after TestSmolDoclingOptimizedPipeline: loads models once, then sweeps
 * a configuration matrix of execution modes and settings.
 *
 * System properties for test control:
 *   -Dqwen.model.size=0.8B          select model size (0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B)
 *   -Dqwen.quant=Q4_K_M             select quantization type
 *   -Dqwen.config=CUDA_GRAPHS       select specific config(s), comma-separated
 *   -Dqwen.max.tokens=50            max generation tokens
 *   -Dqwen.prompt="Hello"           custom test prompt
 *   -Dqwen.tokenizer.path=/path     path to tokenizer.json (optional, extracted from GGUF if absent)
 *
 * Run examples:
 *   cd platform-tests && mvn test -Dtest=TestQwen35Pipeline#testQwen35Pipeline \
 *     -Dqwen.model.size=0.8B -Dqwen.quant=Q4_K_M -Dqwen.config=SLOT_BY_SLOT \
 *     -Dqwen.max.tokens=20
 *
 *   cd platform-tests && mvn test -Dtest=TestQwen35Pipeline#testQwen35Pipeline \
 *     -Dqwen.model.size=0.8B -Dqwen.config=CUDA_GRAPHS \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestQwen35Pipeline {

    // ─── QwenTestConfig ─────────────────────────────────────────────────

    private static class QwenTestConfig {
        String name;
        GraphExecutionMode executionMode;
        String tritonProfile = "BALANCED";
        String tritonIncludeTypes = "";
        boolean tritonSectionFusion;
        boolean tritonGraphCapture;
        boolean tritonCompileAll;
        boolean tritonConsolidatedArgTable;
        boolean tritonArgDirtyTracking;
        boolean tritonAllowFallbackCapture;
        int maxTokens = 20;
        double temperature = 0.0; // greedy by default
        double minDiversityPct = 5.0;
        static QwenTestConfig create(String name) {
            QwenTestConfig c = new QwenTestConfig();
            c.name = name;
            return c;
        }

        QwenTestConfig executionMode(GraphExecutionMode m) { this.executionMode = m; return this; }
        QwenTestConfig tritonProfile(String p) { this.tritonProfile = p; return this; }
        QwenTestConfig tritonIncludeTypes(String t) { this.tritonIncludeTypes = t; return this; }
        QwenTestConfig tritonSectionFusion(boolean b) { this.tritonSectionFusion = b; return this; }
        QwenTestConfig tritonGraphCapture(boolean b) { this.tritonGraphCapture = b; return this; }
        QwenTestConfig tritonCompileAll(boolean b) { this.tritonCompileAll = b; return this; }
        QwenTestConfig tritonConsolidatedArgTable(boolean b) { this.tritonConsolidatedArgTable = b; return this; }
        QwenTestConfig tritonArgDirtyTracking(boolean b) { this.tritonArgDirtyTracking = b; return this; }
        QwenTestConfig tritonAllowFallbackCapture(boolean b) { this.tritonAllowFallbackCapture = b; return this; }
        QwenTestConfig maxTokens(int n) { this.maxTokens = n; return this; }
        QwenTestConfig temperature(double t) { this.temperature = t; return this; }
        QwenTestConfig minDiversityPct(double d) { this.minDiversityPct = d; return this; }
        boolean isTriton() { return !tritonIncludeTypes.isEmpty(); }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(name);
            if (executionMode != null) sb.append(" mode=").append(executionMode);
            if (isTriton()) sb.append(" types=").append(tritonIncludeTypes);
            if (tritonSectionFusion) sb.append(" fusion");
            if (tritonGraphCapture) sb.append(" gc");
            if (tritonCompileAll) sb.append(" compileAll");
            sb.append(" tokens=").append(maxTokens);
            if (temperature > 0) sb.append(" temp=").append(temperature);
            return sb.toString();
        }
    }

    // ─── ConfigResult ───────────────────────────────────────────────────

    private static class ConfigResult {
        final String name;
        boolean passed;
        String failureMessage;
        long resetMs;
        long compileMs;
        long generateMs;
        long validateMs;
        int tokenCount;
        double tokPerSec;
        long firstTokenMs;
        GenerationResult.FinishReason finishReason;
        double diversityRatio;
        double coherenceScore;
        int[] tokenIds;
        int parityMismatches;

        ConfigResult(String name) { this.name = name; }

        String summary() {
            if (!passed) return String.format("[FAIL] %s: %s", name, failureMessage);
            return String.format("[PASS] %s: %d tokens, %.2f tok/s, firstToken=%dms, diversity=%.2f, coherence=%.2f, parity=%s | generate=%dms compile=%dms",
                    name, tokenCount, tokPerSec, firstTokenMs,
                    diversityRatio, coherenceScore,
                    parityMismatches == 0 ? "MATCH" : parityMismatches + " MISMATCHES",
                    generateMs, compileMs);
        }
    }

    // ─── PipelineContext ────────────────────────────────────────────────

    private static class PipelineContext {
        SameDiff model;
        Tokenizer tokenizer;
        LLMModel llmModel;
        QuantType quantType;
        long downloadMs;
        long importMs;
        int modelOps;
    }

    // ─── Configuration Matrix ───────────────────────────────────────────

    private static final String FULL_TRITON_TYPES =
            "ELEMENTWISE,REDUCTION,NORMALIZATION,GATHER,STACK,ATTENTION";
    private static final String COMPILE_ALL_TYPES =
            "CONST_GEN,GATHER,CONCAT,SPLIT,STACK";

    private static List<QwenTestConfig> getAllConfigs() {
        boolean triton = Nd4j.getNativeOps().isTritonAvailable();
        List<QwenTestConfig> configs = new ArrayList<>();

        int maxTokens = Integer.parseInt(System.getProperty("qwen.max.tokens", "20"));

        // Baselines
        configs.add(QwenTestConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens));

        configs.add(QwenTestConfig.create("CUDA_GRAPHS")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .maxTokens(maxTokens));

        if (!triton) return configs;

        // Triton configs
        configs.add(QwenTestConfig.create("TRITON_compileAll_safe")
                .tritonIncludeTypes("GATHER,STACK")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens));

        configs.add(QwenTestConfig.create("TRITON_compileAll_best")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens));

        configs.add(QwenTestConfig.create("TRITON_FULL_fused")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .maxTokens(maxTokens));

        // CUDA_GRAPHS execution mode + Triton compilation.
        // Tests that CUDA graph capture of non-Triton segments coexists correctly
        // with Triton-compiled segments executing outside the captured graph.
        // Note: tritonGraphCapture (capturing CUDA graphs OVER Triton segments)
        // is architecturally unsupported — Triton kernels allocate workspace
        // dynamically, and captured graphs record stale addresses on replay.
        configs.add(QwenTestConfig.create("CUDA_GRAPHS_with_TRITON")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens));

        return configs;
    }

    // ─── Reference Prompts ──────────────────────────────────────────────

    private static final String DEFAULT_PROMPT = "What is the capital of France?";

    private static String getTestPrompt() {
        return System.getProperty("qwen.prompt", DEFAULT_PROMPT);
    }

    // ─── Setup ──────────────────────────────────────────────────────────

    @BeforeAll
    public static void setup() {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
    }

    // ─── Model Loading ──────────────────────────────────────────────────

    private PipelineContext loadModel() throws Exception {
        PipelineContext ctx = new PipelineContext();

        // Determine model size and quant from system properties
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        ctx.llmModel = LLMModel.fromSizeLabel(sizeLabel);
        ctx.quantType = QuantType.valueOf(quantStr);

        log.info("Loading Qwen3.5 model: {} with {} quantization", ctx.llmModel.getName(), ctx.quantType);

        // Download
        long t0 = System.currentTimeMillis();
        DownloadResult downloadResult = LLMModelDownloader.download(ctx.llmModel, ctx.quantType);
        ctx.downloadMs = System.currentTimeMillis() - t0;
        log.info("Download: {}ms (cached={}, size={})",
                ctx.downloadMs, !downloadResult.isDownloadedNow(),
                formatBytes(downloadResult.getFileSizeBytes()));

        // Import GGUF → SameDiff
        t0 = System.currentTimeMillis();
        ctx.model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath());
        ctx.importMs = System.currentTimeMillis() - t0;
        ctx.modelOps = ctx.model.ops().length;
        log.info("Import: {}ms, {} ops", ctx.importMs, ctx.modelOps);

        // Load tokenizer
        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            ctx.tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            // Try to download tokenizer from HuggingFace
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            ctx.tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        }
        log.info("Tokenizer loaded, vocab size: {}", ctx.tokenizer.getVocabSize());

        return ctx;
    }

    // ─── Config Application ─────────────────────────────────────────────

    private void applyConfig(PipelineContext ctx, QwenTestConfig config) {
        SameDiff model = ctx.model;
        Environment env = Nd4j.getEnvironment();

        // Reset model state — clear sessions AND native plan cache to avoid cross-config contamination
        // (backend singletons like OneDNN Graph have negative caches that persist across sessions)
        model.getSessions().clear();
        model.clearDynamicShapePlanCache();
        // Invalidate GPU-resident Triton modules to prevent stale CUfunction handles
        // from corrupting CUDA graph capture when tritonGraphCapture=true.
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (nativeOps.isTritonAvailable()) {
            nativeOps.invalidateTritonCache();
        }
        resetTritonEnvironment(env);

        // Apply execution mode — always set explicitly to avoid stale mode from prior config
        if (config.executionMode != null) {
            model.setGraphExecutionMode(config.executionMode);
        } else {
            model.setGraphExecutionMode(GraphExecutionMode.AUTO);
        }

        // DSP settings
        model.setDspAutoCompileEnabled(true);
        model.setDspNativeAutoCompileEnabled(true);

        // Triton settings
        if (config.isTriton()) {
            BenchmarkConfigApplier.applyTritonProfile(config.tritonProfile);
            env.setTritonIncludeTypes(System.getProperty("nd4j.triton.includeTypes", config.tritonIncludeTypes));
            env.setTritonSectionFusion(config.tritonSectionFusion);
            env.setTritonGraphCapture(booleanProperty("nd4j.triton.graphCapture", config.tritonGraphCapture));
            env.setTritonCompileAll(booleanProperty("nd4j.triton.compileAll", config.tritonCompileAll));
            env.setTritonConsolidatedArgTable(config.tritonConsolidatedArgTable);
            env.setTritonArgDirtyTracking(config.tritonArgDirtyTracking);
            env.setTritonAllowFallbackCapture(config.tritonAllowFallbackCapture);
            env.setTritonSkipKernels(booleanProperty("nd4j.triton.skipKernels", false));
            env.setTritonVerifyKernels(booleanProperty("nd4j.triton.verifyKernels", false));
            log.info("Applied Triton config: types={} fusion={} graphCapture={} compileAll={} skipKernels={} verifyKernels={}",
                    env.tritonIncludeTypes(), env.tritonSectionFusion(), env.tritonGraphCapture(),
                    env.tritonCompileAll(), env.tritonSkipKernels(), env.tritonVerifyKernels());
        }
    }

    private void clearTritonProperties() {
        resetTritonEnvironment(Nd4j.getEnvironment());
        String[] props = {
                "nd4j.dsp.triton.includeTypes", "nd4j.dsp.triton.sectionFusion",
                "nd4j.dsp.triton.graphCapture", "nd4j.dsp.triton.compileAll",
                "nd4j.dsp.triton.consolidatedArgTable", "nd4j.dsp.triton.argDirtyTracking",
                "nd4j.dsp.triton.allowFallbackCapture", "nd4j.dsp.triton.profile"
        };
        for (String prop : props) {
            System.clearProperty(prop);
        }
    }

    private void resetTritonEnvironment(Environment env) {
        env.setTritonIncludeTypes("");
        env.setTritonSectionFusion(false);
        env.setTritonGraphCapture(false);
        env.setTritonCompileAll(false);
        env.setTritonConsolidatedArgTable(false);
        env.setTritonArgDirtyTracking(false);
        env.setTritonAllowFallbackCapture(false);
        env.setTritonSkipKernels(false);
        env.setTritonVerifyKernels(false);
    }

    private boolean booleanProperty(String name, boolean defaultValue) {
        return Boolean.parseBoolean(System.getProperty(name, Boolean.toString(defaultValue)));
    }

    // ─── Generation ─────────────────────────────────────────────────────

    private GenerationResult runGeneration(PipelineContext ctx, QwenTestConfig config, String prompt) {
        SamplingConfig samplingConfig;
        if (config.temperature <= 0) {
            samplingConfig = SamplingConfig.greedy();
        } else {
            samplingConfig = SamplingConfig.builder()
                    .temperature(config.temperature)
                    .topK(50)
                    .topP(0.9)
                    .build();
        }

        try {
            GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                    .decoder(ctx.model)
                    .tokenizer(ctx.tokenizer)
                    .samplingConfig(samplingConfig)
                    .maxNewTokens(config.maxTokens)
                    .graphOptimizerEnabled(Boolean.parseBoolean(
                            System.getProperty("qwen.graph.optimizer", "true")))
                    .dspEnabled(Boolean.parseBoolean(
                            System.getProperty("qwen.dsp", "true")))
                    .build();

            try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
                return pipeline.generate(prompt, config.maxTokens);
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException("Failed to create GenerationPipeline", e);
        }
    }

    // ─── Main Test ──────────────────────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Pipeline: Configuration matrix sweep")
    public void testQwen35Pipeline() throws Exception {
        // Enable debug/verbose if requested via -Dnd4j.debug=true
        if (Boolean.getBoolean("nd4j.debug")) {
            Nd4j.getEnvironment().setDebug(true);
            Nd4j.getEnvironment().setVerbose(true);
            log.info("DEBUG+VERBOSE mode ENABLED — will print per-op execution details");
        }

        PipelineContext ctx = loadModel();

        assertNotNull(ctx.model, "Model must be loaded");
        assertNotNull(ctx.tokenizer, "Tokenizer must be loaded");
        assertTrue(ctx.modelOps > 0, "Model should have ops");

        log.info("Pipeline setup complete: download={}ms import={}ms, {} ops",
                ctx.downloadMs, ctx.importMs, ctx.modelOps);

        List<QwenTestConfig> configs = getAllConfigs();

        // Filter configs via -Dqwen.config=<name>
        String configFilter = System.getProperty("qwen.config");
        if (configFilter != null && !configFilter.isEmpty()) {
            String[] filters = configFilter.split(",");
            configs = configs.stream().filter(c -> {
                for (String f : filters) {
                    String ft = f.trim();
                    if (ft.endsWith("*")) {
                        if (c.name.startsWith(ft.substring(0, ft.length() - 1))) return true;
                    } else {
                        if (c.name.equals(ft)) return true;
                    }
                }
                return false;
            }).collect(Collectors.toList());
            log.info("Config filter '{}' matched {} configs", configFilter, configs.size());
            assertTrue(!configs.isEmpty(), "No configs matched filter: " + configFilter);
        }

        log.info("Running {} configurations:", configs.size());
        for (int i = 0; i < configs.size(); i++) {
            log.info("  [{}] {}", i + 1, configs.get(i));
        }

        String prompt = getTestPrompt();
        List<ConfigResult> results = new ArrayList<>();
        int failures = 0;
        int[] goldenTokens = null; // SLOT_BY_SLOT tokens are the golden reference

        for (int i = 0; i < configs.size(); i++) {
            QwenTestConfig config = configs.get(i);
            log.info("============================================================");
            log.info("[{}/{}] CONFIG: {}", i + 1, configs.size(), config);
            log.info("============================================================");

            ConfigResult cr = new ConfigResult(config.name);
            try {
                // Reset
                long t0 = System.currentTimeMillis();
                clearTritonProperties();
                cr.resetMs = System.currentTimeMillis() - t0;

                // Apply config
                t0 = System.currentTimeMillis();
                applyConfig(ctx, config);
                cr.compileMs = System.currentTimeMillis() - t0;

                // Generate
                t0 = System.currentTimeMillis();
                GenerationResult genResult = runGeneration(ctx, config, prompt);
                cr.generateMs = System.currentTimeMillis() - t0;

                cr.tokenCount = genResult.getGeneratedTokenCount();
                cr.tokPerSec = genResult.getTokensPerSecond();
                cr.firstTokenMs = genResult.getFirstTokenLatencyMs();
                cr.finishReason = genResult.getFinishReason();
                cr.tokenIds = genResult.getTokenIds();

                log.info("Generated {} tokens at {} tok/s: {}",
                        cr.tokenCount, String.format("%.2f", cr.tokPerSec), genResult.getText());
                log.info("Token IDs: {}", Arrays.toString(cr.tokenIds));

                // ── Parity check against SLOT_BY_SLOT golden baseline ──
                if ("SLOT_BY_SLOT".equals(config.name)) {
                    goldenTokens = cr.tokenIds;
                    cr.parityMismatches = 0;
                    log.info("GOLDEN BASELINE: {} tokens captured from SLOT_BY_SLOT", goldenTokens.length);
                } else if (goldenTokens != null && cr.tokenIds != null) {
                    int minLen = Math.min(goldenTokens.length, cr.tokenIds.length);
                    int mismatches = 0;
                    int firstMismatchPos = -1;
                    for (int t = 0; t < minLen; t++) {
                        if (goldenTokens[t] != cr.tokenIds[t]) {
                            if (firstMismatchPos < 0) firstMismatchPos = t;
                            mismatches++;
                        }
                    }
                    mismatches += Math.abs(goldenTokens.length - cr.tokenIds.length);
                    cr.parityMismatches = mismatches;

                    if (mismatches > 0) {
                        log.error("PARITY FAILURE: {} vs SLOT_BY_SLOT — {} mismatches out of {} tokens, " +
                                        "first at position {} (golden={} vs actual={})",
                                config.name, mismatches, Math.max(goldenTokens.length, cr.tokenIds.length),
                                firstMismatchPos,
                                firstMismatchPos >= 0 && firstMismatchPos < goldenTokens.length ? goldenTokens[firstMismatchPos] : "N/A",
                                firstMismatchPos >= 0 && firstMismatchPos < cr.tokenIds.length ? cr.tokenIds[firstMismatchPos] : "N/A");
                        log.error("  Golden tokens: {}", Arrays.toString(goldenTokens));
                        log.error("  Actual tokens: {}", Arrays.toString(cr.tokenIds));
                    } else {
                        log.info("PARITY OK: {} matches SLOT_BY_SLOT exactly ({} tokens)", config.name, minLen);
                    }
                }

                // Validate quality
                t0 = System.currentTimeMillis();
                QualityReport quality = GenerationQualityValidator.validate(genResult);
                cr.validateMs = System.currentTimeMillis() - t0;

                cr.diversityRatio = quality.getDiversityRatio();
                cr.coherenceScore = quality.getCoherenceScore();

                // Assertions — parity with SLOT_BY_SLOT is the primary correctness check
                assertTrue(cr.tokenCount > 0, "Should generate at least 1 token");
                if (goldenTokens != null && !"SLOT_BY_SLOT".equals(config.name)) {
                    assertEquals(0, cr.parityMismatches,
                            String.format("%s produced %d token mismatches vs SLOT_BY_SLOT golden baseline. " +
                                            "Golden: %s, Actual: %s",
                                    config.name, cr.parityMismatches,
                                    Arrays.toString(goldenTokens), Arrays.toString(cr.tokenIds)));
                }

                cr.passed = true;
            } catch (Exception | AssertionError e) {
                cr.passed = false;
                cr.failureMessage = e.getClass().getSimpleName() + ": " + e.getMessage();
                failures++;
                log.error("[{}/{}] Config {} failed", i + 1, configs.size(), config.name, e);
            }
            results.add(cr);
            log.info("  {}", cr.summary());

            // Full lifecycle cleanup: close the model entirely and reload it fresh.
            // Each config (SLOT_BY_SLOT, CUDA_GRAPHS, TRITON) compiles a fundamentally
            // different native plan with different GPU memory requirements. Rather than
            // trying to partially clean up (which leaves model weights, pool reservations,
            // and DeallocatorService queues in indeterminate states), close everything
            // and give the next config a clean GPU with full memory available.
            // The model file is already cached on disk, so reimport is fast (~2-3s).
            var nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            var deviceMgr = org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance();
            int numDevices = deviceMgr.getContextProvider().getDeviceCount();
            long cudaFreeBefore = nativeOps.getDeviceFreeMemory(0);
            long cudaTotalBefore = nativeOps.getDeviceTotalMemory(0);
            long refMapSizeBefore = Nd4j.getDeallocatorService().getReferenceMap().size();
            log.info("  GPU memory BEFORE close: {} | CUDA free={} MB total={} MB | refMap={}",
                    deviceMgr.getMemorySummary(), cudaFreeBefore / (1024*1024),
                    cudaTotalBefore / (1024*1024), refMapSizeBefore);

            ctx.model.close();
            // Flush Triton singleton cache — plan destructor cleans up its own
            // segment entries, but this catches any residual entries from compilation
            // attempts or failed segments that didn't map to a live plan segment.
            nativeOps.invalidateTritonCache();
            long cudaFreeAfterClose = nativeOps.getDeviceFreeMemory(0);
            long refMapSizeAfterClose = Nd4j.getDeallocatorService().getReferenceMap().size();
            log.info("  GPU memory AFTER model.close() + triton invalidate: {} | CUDA free={} MB | refMap={}",
                    deviceMgr.getMemorySummary(), cudaFreeAfterClose / (1024*1024), refMapSizeAfterClose);

            ctx.tokenizer.close();

            // Force-flush all remaining DeallocatorService entries. After model.close(),
            // hundreds of DataBuffer/OpaqueNDArray PhantomReferences remain in the refMap
            // holding GPU memory. System.gc() is advisory and won't reliably collect them
            // in time. Without this flush, ~5GB of GPU memory leaks between configs, leaving
            // insufficient memory for CUDA graph capture in subsequent configs.
            Nd4j.getDeallocatorService().forceFlushAll();

            Nd4j.getMemoryManager().purgeCaches();
            Nd4j.getExecutioner().commit();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }

            // One GC round for any remaining Java-only garbage
            System.gc();
            try { Thread.sleep(200); } catch (InterruptedException ignored) {}
            Nd4j.getExecutioner().commit();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
            {
                long cudaFreeRound = nativeOps.getDeviceFreeMemory(0);
                long refMapSizeRound = Nd4j.getDeallocatorService().getReferenceMap().size();
                log.info("  GPU memory AFTER forceFlush+gc+trim: {} | CUDA free={} MB | refMap={}",
                        deviceMgr.getMemorySummary(), cudaFreeRound / (1024*1024), refMapSizeRound);
            }

            // Diagnostic: compare Java-tracked memory vs actual CUDA memory
            long cudaFreeAfterGc = nativeOps.getDeviceFreeMemory(0);
            long cudaTotal = nativeOps.getDeviceTotalMemory(0);
            long actualUsedMB = (cudaTotal - cudaFreeAfterGc) / (1024*1024);
            log.info("  DIAGNOSTIC: CUDA actual used={} MB vs Java counter={}",
                    actualUsedMB, deviceMgr.getMemorySummary());

            if (i + 1 < configs.size()) {
                // Switch back to device 0 for next config's model load
                deviceMgr.switchDevice(0, "TestQwen35Pipeline", "inter-config cleanup");
                ctx = loadModel();
            }
        }

        // Print full report
        log.info("============================================================");
        log.info("QWEN3.5 CONFIGURATION MATRIX RESULTS: {}/{} passed",
                configs.size() - failures, configs.size());
        log.info("============================================================");
        StringBuilder report = new StringBuilder();
        report.append(String.format("Model: %s (%s)\n", ctx.llmModel.getName(), ctx.quantType));
        report.append(String.format("Pipeline setup: download=%dms import=%dms\n\n", ctx.downloadMs, ctx.importMs));
        for (ConfigResult cr : results) {
            report.append(cr.summary()).append("\n");
        }
        log.info("Full report:\n{}", report);

        // Summary statistics
        double maxTokPerSec = 0;
        String fastestConfig = "";
        for (ConfigResult cr : results) {
            if (!cr.passed) continue;
            if (cr.tokPerSec > maxTokPerSec) {
                maxTokPerSec = cr.tokPerSec;
                fastestConfig = cr.name;
            }
        }
        if (maxTokPerSec > 0) {
            log.info("Fastest: {} at {} tok/s", fastestConfig, String.format("%.2f", maxTokPerSec));
        }

        assertEquals(0, failures, failures + " configuration(s) failed. See log for details.");
    }

    // ─── Perplexity Test ────────────────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Perplexity Evaluation")
    public void testQwen35Perplexity() throws Exception {
        PipelineContext ctx = loadModel();

        assertNotNull(ctx.model, "Model must be loaded");

        // Configure for inference
        ctx.model.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        ctx.model.setDspAutoCompileEnabled(true);
        ctx.model.setDspNativeAutoCompileEnabled(true);

        // Use a short reference text for quick evaluation
        String referenceText = "The quick brown fox jumps over the lazy dog. " +
                "Machine learning is a subset of artificial intelligence that focuses on " +
                "building systems that learn from data. Neural networks are inspired by " +
                "the biological neural networks that constitute animal brains.";

        int contextLength = 512;
        int stride = 256;

        PerplexityResult result = PerplexityEvaluator.evaluate(
                ctx.model, ctx.tokenizer, referenceText, contextLength, stride);

        log.info("Perplexity evaluation:");
        log.info("  Perplexity: {}", String.format("%.4f", result.getPerplexity()));
        log.info("  Bits/byte: {}", String.format("%.4f", result.getBitsPerByte()));
        log.info("  Tokens evaluated: {}", result.getNumTokens());
        log.info("  Time: {}ms", result.getEvaluationTimeMs());

        assertTrue(result.getPerplexity() > 0, "Perplexity must be positive");
        assertTrue(result.getPerplexity() < 1000, "Perplexity should be reasonable (< 1000)");
        assertTrue(result.getNumTokens() > 0, "Should evaluate at least 1 token");
    }

    // ─── Quant Comparison Test ──────────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Quantization Comparison")
    public void testQwen35QuantComparison() throws Exception {
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        int maxTokens = Integer.parseInt(System.getProperty("qwen.max.tokens", "20"));
        String prompt = getTestPrompt();

        // Quants to compare
        QuantType[] quants = {QuantType.Q4_K_M, QuantType.Q8_0};
        String quantFilter = System.getProperty("qwen.quant.compare");
        if (quantFilter != null && !quantFilter.isEmpty()) {
            quants = Arrays.stream(quantFilter.split(","))
                    .map(String::trim)
                    .map(QuantType::valueOf)
                    .toArray(QuantType[]::new);
        }

        log.info("Comparing {} quantizations for {}", quants.length, llmModel.getName());

        List<String> summaries = new ArrayList<>();

        for (QuantType quant : quants) {
            log.info("============================================================");
            log.info("Quantization: {} ({})", quant, llmModel.getName());
            log.info("============================================================");

            try {
                DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quant);
                SameDiff model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath());

                // Load tokenizer
                String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
                File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
                Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());

                model.setDspAutoCompileEnabled(true);
                model.setDspNativeAutoCompileEnabled(true);

                GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                        .decoder(model)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(maxTokens)
                        .graphOptimizerEnabled(Boolean.parseBoolean(
                                System.getProperty("qwen.graph.optimizer", "true")))
                        .dspEnabled(Boolean.parseBoolean(
                                System.getProperty("qwen.dsp", "true")))
                        .build();

                GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig);
                GenerationResult genResult = pipeline.generate(prompt, maxTokens);
                QualityReport quality = GenerationQualityValidator.validate(genResult);

                String summary = String.format("%s: %d tokens, %.2f tok/s, diversity=%.2f, coherence=%.2f, size=%s | %s",
                        quant, genResult.getGeneratedTokenCount(), genResult.getTokensPerSecond(),
                        quality.getDiversityRatio(), quality.getCoherenceScore(),
                        formatBytes(downloadResult.getFileSizeBytes()),
                        genResult.getText().substring(0, Math.min(80, genResult.getText().length())));

                log.info("Result: {}", summary);
                summaries.add(summary);

                tokenizer.close();
            } catch (Exception e) {
                log.error("Failed for quant {}: {}", quant, e.getMessage(), e);
                summaries.add(String.format("%s: FAILED - %s", quant, e.getMessage()));
            }
        }

        log.info("============================================================");
        log.info("QUANTIZATION COMPARISON RESULTS");
        log.info("============================================================");
        for (String s : summaries) {
            log.info("  {}", s);
        }
    }

    // ─── Reference Prompt Validation ────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Reference Prompt Validation")
    public void testQwen35ReferencePrompts() throws Exception {
        PipelineContext ctx = loadModel();

        ctx.model.setDspAutoCompileEnabled(true);
        ctx.model.setDspNativeAutoCompileEnabled(true);

        int maxTokens = Integer.parseInt(System.getProperty("qwen.max.tokens", "50"));
        int passed = 0;
        int total = 0;

        for (Map.Entry<String, String[]> entry : GenerationQualityValidator.REFERENCE_PROMPTS.entrySet()) {
            String prompt = entry.getKey();
            String[] expectedSubstrings = entry.getValue();
            total++;

            log.info("Prompt: {}", prompt);

            GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                    .decoder(ctx.model)
                    .tokenizer(ctx.tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(maxTokens)
                    .graphOptimizerEnabled(Boolean.parseBoolean(
                            System.getProperty("qwen.graph.optimizer", "true")))
                    .dspEnabled(Boolean.parseBoolean(
                            System.getProperty("qwen.dsp", "true")))
                    .build();

            GenerationPipeline pipeline;
            try {
                pipeline = GenerationPipeline.create(pipelineConfig);
            } catch (java.io.IOException e) {
                throw new RuntimeException("Failed to create GenerationPipeline", e);
            }
            GenerationResult genResult = pipeline.generate(prompt, maxTokens);
            log.info("  Output ({} tokens, {} tok/s): {}",
                    genResult.getGeneratedTokenCount(), String.format("%.2f", genResult.getTokensPerSecond()),
                    genResult.getText());

            QualityReport quality;
            if (expectedSubstrings.length > 0) {
                quality = GenerationQualityValidator.validateWithExpected(genResult, expectedSubstrings);
            } else {
                quality = GenerationQualityValidator.validateDiversity(genResult, 0.4);
            }

            if (quality.isPassed()) {
                passed++;
                log.info("  PASSED: {}", quality.summary());
            } else {
                log.warn("  FAILED: {}", quality.summary());
            }
        }

        log.info("Reference prompt results: {}/{} passed", passed, total);
        assertEquals(total, passed, "All reference prompts should pass");
    }

    // ─── Download-Only Test ────────────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Download: model + tokenizer only")
    public void testQwen35Download() throws Exception {
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        QuantType quantType = QuantType.valueOf(quantStr);

        log.info("Downloading Qwen3.5 model: {} with {} quantization", llmModel.getName(), quantType);

        // Download model GGUF
        long t0 = System.currentTimeMillis();
        DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quantType);
        long modelMs = System.currentTimeMillis() - t0;

        File modelFile = downloadResult.getModelFile();
        assertTrue(modelFile.exists(), "Model file not found: " + modelFile);
        assertTrue(modelFile.length() > 0, "Model file is empty");
        log.info("Model downloaded: {} ({}) in {}ms (cached={})",
                modelFile.getName(), formatBytes(downloadResult.getFileSizeBytes()),
                modelMs, !downloadResult.isDownloadedNow());

        // Download tokenizer
        t0 = System.currentTimeMillis();
        String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
        File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
        long tokMs = System.currentTimeMillis() - t0;

        assertTrue(tokenizerFile.exists(), "Tokenizer file not found: " + tokenizerFile);
        assertTrue(tokenizerFile.length() > 0, "Tokenizer file is empty");
        log.info("Tokenizer downloaded: {} ({}) in {}ms",
                tokenizerFile.getName(), formatBytes(tokenizerFile.length()), tokMs);

        // List all cached files
        File[] cached = LLMModelDownloader.listCachedModels();
        log.info("Cache directory: {}", LLMModelDownloader.getCacheDir().getAbsolutePath());
        log.info("Cached models ({}):", cached.length);
        for (File f : cached) {
            log.info("  {} ({})", f.getName(), formatBytes(f.length()));
        }
    }

    // ─── Import-Only Test ─────────────────────────────────────────────

    @Test
    @DisplayName("Qwen3.5 Import: download + GGUF import + tokenizer load (no execution)")
    public void testQwen35Import() throws Exception {
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        QuantType quantType = QuantType.valueOf(quantStr);

        // 1. Download model GGUF
        log.info("Downloading Qwen3.5 model: {} with {} quantization", llmModel.getName(), quantType);
        long t0 = System.currentTimeMillis();
        DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quantType);
        long downloadMs = System.currentTimeMillis() - t0;
        log.info("Model downloaded: {} ({}) in {}ms (cached={})",
                downloadResult.getModelFile().getName(), formatBytes(downloadResult.getFileSizeBytes()),
                downloadMs, !downloadResult.isDownloadedNow());

        // 2. Import GGUF → SameDiff (use FP16 to avoid 8x memory blow-up from FP32 dequantization)
        log.info("Importing GGUF model with FP16 dequantization...");
        t0 = System.currentTimeMillis();
        ConversionOptions importOptions = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .build();
        SameDiff model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath(), importOptions);
        long importMs = System.currentTimeMillis() - t0;

        int opCount = model.ops() != null ? model.ops().length : 0;
        int varCount = model.variables().size();
        List<String> outputs = model.outputs() != null ? new ArrayList<>(model.outputs()) : new ArrayList<>();
        log.info("Import: {}ms, {} ops, {} variables, {} outputs", importMs, opCount, varCount, outputs.size());
        assertTrue(varCount > 0, "Model has no variables after import");
        log.info("Outputs: {}", outputs);

        // Log some variable names for inspection
        List<String> varNames = new ArrayList<>(model.variables().stream()
                .map(v -> v.name()).limit(20).collect(java.util.stream.Collectors.toList()));
        log.info("First {} variables: {}", varNames.size(), varNames);

        // 3. Download and load tokenizer
        t0 = System.currentTimeMillis();
        String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
        File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        long tokMs = System.currentTimeMillis() - t0;

        // Verify tokenizer works
        Encoding testEncoding = tokenizer.encode("Hello, world!");
        int[] testTokens = testEncoding.getIds();
        String decoded = tokenizer.decode(testTokens);
        int vocabSize = tokenizer.getVocabSize();
        log.info("Tokenizer loaded in {}ms, vocab size={}, test encode/decode: '{}' -> {} tokens -> '{}'",
                tokMs, vocabSize, "Hello, world!", testTokens.length, decoded);
        assertTrue(testTokens.length > 0, "Tokenizer produced no tokens");
        assertTrue(vocabSize > 0, "Tokenizer has no vocabulary");

        // Summary
        log.info("=== Import Summary ===");
        log.info("  Model: {} ({})", llmModel.getName(), quantType);
        log.info("  Download: {}ms", downloadMs);
        log.info("  Import: {}ms ({} ops, {} vars)", importMs, opCount, varCount);
        log.info("  Tokenizer: {}ms (vocab={})", tokMs, vocabSize);

        // Cleanup
        model.close();
    }

    // ─── Utility ────────────────────────────────────────────────────────

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }
}
