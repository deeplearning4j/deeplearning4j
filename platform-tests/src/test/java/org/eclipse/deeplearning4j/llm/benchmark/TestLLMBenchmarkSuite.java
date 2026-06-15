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

package org.eclipse.deeplearning4j.llm.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator.QualityReport;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator.PerplexityResult;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Multi-model LLM benchmark suite that exercises different execution backends,
 * kernel fusion, graph optimizer, and Triton compilation across model families.
 *
 * <p>Tests download small GGUF models (Qwen3.5-0.8B, Gemma3-1B, Phi-3-mini, etc.),
 * import them via GGMLModelImport, and benchmark generation throughput under
 * different execution modes: AUTO (optimal), CUDA_GRAPHS, and TRITON.</p>
 *
 * <h3>System properties for test control:</h3>
 * <pre>
 *   -Dbench.models=qwen,gemma        Select model families (comma-separated: qwen,gemma,phi,mistral)
 *   -Dbench.configs=OPTIMAL          Select configs (comma-separated, wildcard with *)
 *   -Dbench.max.tokens=20            Max generation tokens (default: 20)
 *   -Dbench.prompt="Hello"           Custom test prompt
 *   -Dbench.quant=Q4_K_M             Quantization type (default: Q4_K_M)
 *   -Dbench.skip.generation=true     Skip generation tests (import-only)
 *   -Dbackend.artifactId=nd4j-native CPU backend (default: nd4j-cuda-12.9)
 * </pre>
 *
 * <h3>Run examples:</h3>
 * <pre>
 *   # Full suite on CUDA
 *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/llm-bench.log
 *
 *   # Single model, single config
 *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite#testOptimalBaseline \
 *     -Dbench.models=qwen -Dbench.max.tokens=10 2>&1 | tee /tmp/llm-bench.log
 *
 *   # Import benchmark only (no generation, no tokenizer needed)
 *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite#testModelImportBenchmark \
 *     2>&1 | tee /tmp/llm-import.log
 *
 *   # CPU backend
 *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite#testOptimalBaseline \
 *     -Dbackend.artifactId=nd4j-native -Dbench.models=qwen \
 *     2>&1 | tee /tmp/llm-cpu.log
 *
 *   # Triton benchmarks (CUDA only)
 *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite#testTritonCompileBenchmark \
 *     -Dbench.models=qwen -Dlibnd4j.triton=ON 2>&1 | tee /tmp/llm-triton.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestLLMBenchmarkSuite {

    // ─── Model Specification ────────────────────────────────────────────

    /**
     * Defines a model to benchmark: GGUF source + tokenizer source.
     */
    private static class ModelSpec {
        final LLMModel llmModel;
        final QuantType quantType;
        final String tokenizerUrl;
        final String tokenizerCacheFile;
        final String familyKey;
        final boolean requiresAuth;

        ModelSpec(LLMModel llmModel, QuantType quantType,
                  String tokenizerUrl, String tokenizerCacheFile, String familyKey) {
            this(llmModel, quantType, tokenizerUrl, tokenizerCacheFile, familyKey, false);
        }

        ModelSpec(LLMModel llmModel, QuantType quantType,
                  String tokenizerUrl, String tokenizerCacheFile, String familyKey, boolean requiresAuth) {
            this.llmModel = llmModel;
            this.quantType = quantType;
            this.tokenizerUrl = tokenizerUrl;
            this.tokenizerCacheFile = tokenizerCacheFile;
            this.familyKey = familyKey;
            this.requiresAuth = requiresAuth;
        }

        String displayName() {
            return llmModel.getName() + " (" + quantType + ")";
        }
    }

    // ─── Benchmark Result ───────────────────────────────────────────────

    private static class BenchmarkResult {
        final String modelName;
        final String configName;
        boolean passed;
        boolean skipped;
        String failureMessage;
        // Timing
        long importMs;
        long compileMs;
        long generateMs;
        // Generation metrics
        int tokenCount;
        double tokPerSec;
        long firstTokenMs;
        double steadyStateTokPerSec;
        GenerationResult.FinishReason finishReason;
        // Quality
        double diversityRatio;
        double coherenceScore;
        // Model info
        int opCount;
        String architecture;

        BenchmarkResult(String modelName, String configName) {
            this.modelName = modelName;
            this.configName = configName;
        }

        String summary() {
            if (skipped) return String.format("  [SKIP] %-20s %-25s %s", modelName, configName, failureMessage);
            if (!passed) return String.format("  [FAIL] %-20s %-25s %s", modelName, configName, failureMessage);
            if (tokenCount > 0) {
                return String.format("  [PASS] %-20s %-25s %4d tokens %8.2f tok/s  firstToken=%4dms  diversity=%.2f  coherence=%.2f",
                        modelName, configName, tokenCount, tokPerSec, firstTokenMs, diversityRatio, coherenceScore);
            }
            return String.format("  [PASS] %-20s %-25s import=%dms ops=%d arch=%s",
                    modelName, configName, importMs, opCount, architecture);
        }
    }

    // ─── Model Registry ─────────────────────────────────────────────────

    private static final Map<String, ModelSpec> MODEL_REGISTRY = new LinkedHashMap<>();

    static {
        // Qwen3.5-0.8B: public tokenizer, smallest dense model
        MODEL_REGISTRY.put("qwen", new ModelSpec(
                LLMModel.QWEN35_0_8B, QuantType.Q4_K_M,
                "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json",
                "qwen35-0.8B-tokenizer.json",
                "qwen"));

        // Gemma3-1B: Google gated model, tokenizer needs HF_TOKEN auth
        MODEL_REGISTRY.put("gemma", new ModelSpec(
                LLMModel.GEMMA3_1B, QuantType.Q4_K_M,
                "https://huggingface.co/google/gemma-3-1b-it/resolve/main/tokenizer.json",
                "gemma3-1b-tokenizer.json",
                "gemma", true));

        // Phi-3-mini: Microsoft, public tokenizer
        MODEL_REGISTRY.put("phi", new ModelSpec(
                LLMModel.PHI3_MINI_4K, QuantType.Q4_K_M,
                "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json",
                "phi3-mini-tokenizer.json",
                "phi"));

        // Mistral-7B: public tokenizer (larger model, opt-in)
        MODEL_REGISTRY.put("mistral", new ModelSpec(
                LLMModel.MISTRAL_7B, QuantType.Q4_K_M,
                "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.json",
                "mistral-7b-tokenizer.json",
                "mistral"));

        // LFM2-350M-Extract: LiquidAI structured extraction model, F16, public tokenizer
        MODEL_REGISTRY.put("lfm2-extract", new ModelSpec(
                LLMModel.LFM2_350M_EXTRACT, QuantType.F16,
                "https://huggingface.co/LiquidAI/LFM2-350M/resolve/main/tokenizer.json",
                "lfm2-350m-tokenizer.json",
                "lfm2-extract"));
    }

    // ─── Loaded State Cache ─────────────────────────────────────────────

    private final Map<String, SameDiff> loadedModels = new LinkedHashMap<>();
    private final Map<String, Tokenizer> loadedTokenizers = new LinkedHashMap<>();
    private final Map<String, Long> importTimings = new LinkedHashMap<>();
    private final Map<String, String> architectures = new LinkedHashMap<>();

    // ─── Prompts ────────────────────────────────────────────────────────

    private static final String DEFAULT_PROMPT = "What is the capital of France?";

    private static final Map<String, String[]> BENCHMARK_PROMPTS = new LinkedHashMap<>();
    static {
        BENCHMARK_PROMPTS.put("What is the capital of France?", new String[]{"Paris"});
        BENCHMARK_PROMPTS.put("Explain photosynthesis in one sentence.", new String[]{"sun", "light", "plant"});
        BENCHMARK_PROMPTS.put("Count from 1 to 10.", new String[]{"1", "2", "3", "4", "5"});
    }

    /**
     * Extraction prompts for LFM2-350M-Extract: structured data extraction from unstructured text.
     * These use the model's ChatML-style template with a JSON schema system prompt.
     */
    private static final Map<String, String[]> EXTRACTION_PROMPTS = new LinkedHashMap<>();
    static {
        EXTRACTION_PROMPTS.put(
                "Caenorhabditis elegans is a free-living transparent nematode about 1 mm in length that lives in temperate soil environments.",
                new String[]{"elegans", "nematode"});
        EXTRACTION_PROMPTS.put(
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
                new String[]{"Eiffel", "Paris", "1889"});
        EXTRACTION_PROMPTS.put(
                "Tesla Inc. reported revenue of $25.5 billion in Q3 2024, a 7.8% increase from the previous year.",
                new String[]{"Tesla", "25.5", "revenue"});
    }

    // ─── Configuration ──────────────────────────────────────────────────

    private static int getMaxTokens() {
        return Integer.parseInt(System.getProperty("bench.max.tokens", "20"));
    }

    private static String getPrompt() {
        return System.getProperty("bench.prompt", DEFAULT_PROMPT);
    }

    private static QuantType getQuantType() {
        String q = System.getProperty("bench.quant", "Q4_K_M");
        return QuantType.valueOf(q);
    }

    private static boolean skipGeneration() {
        return Boolean.parseBoolean(System.getProperty("bench.skip.generation", "false"));
    }

    // ─── Setup / Teardown ───────────────────────────────────────────────

    @BeforeAll
    public void setup() {
        // no-op — optimizer always runs via GenerationPipeline
    }

    @AfterAll
    public void cleanup() {
        for (Tokenizer t : loadedTokenizers.values()) {
            try { t.close(); } catch (Exception ignored) {}
        }
        loadedTokenizers.clear();
        loadedModels.clear();
    }

    // ─── Model Selection ────────────────────────────────────────────────

    private List<ModelSpec> getSelectedModels() {
        String filter = System.getProperty("bench.models");
        if (filter == null || filter.isEmpty()) {
            // Default: smallest models only (qwen, gemma)
            return List.of(MODEL_REGISTRY.get("qwen"), MODEL_REGISTRY.get("gemma"));
        }
        if (filter.equalsIgnoreCase("all")) {
            return new ArrayList<>(MODEL_REGISTRY.values());
        }
        List<ModelSpec> selected = new ArrayList<>();
        for (String key : filter.split(",")) {
            String k = key.trim().toLowerCase();
            ModelSpec spec = MODEL_REGISTRY.get(k);
            if (spec != null) {
                selected.add(spec);
            } else {
                log.warn("Unknown model key '{}', available: {}", k, MODEL_REGISTRY.keySet());
            }
        }
        assertTrue(!selected.isEmpty(), "No models matched filter: " + filter +
                ". Available: " + MODEL_REGISTRY.keySet());
        return selected;
    }

    // ─── Config Builders ────────────────────────────────────────────────

    private static final String COMPILE_ALL_TYPES = "CONST_GEN,GATHER,CONCAT,SPLIT,STACK";
    private static final String FULL_TRITON_TYPES = "ELEMENTWISE,REDUCTION,NORMALIZATION,GATHER,STACK,ATTENTION";

    private List<BenchmarkConfig> getBaselineConfigs() {
        int maxTokens = getMaxTokens();
        List<BenchmarkConfig> configs = new ArrayList<>();

        // AUTO engages the full DSP backend chain:
        //   CUDA: Triton islands + cuBLAS gap ops + CUDA graph replay
        //   CPU:  OpenVINO → oneDNN → native slot execution
        configs.add(BenchmarkConfig.optimal()
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(maxTokens));

        return configs;
    }

    private List<BenchmarkConfig> getTritonConfigs() {
        int maxTokens = getMaxTokens();
        List<BenchmarkConfig> configs = new ArrayList<>();

        configs.add(BenchmarkConfig.create("TRITON_safe")
                .tritonIncludeTypes("GATHER,STACK")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens));

        configs.add(BenchmarkConfig.create("TRITON_best")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens));

        configs.add(BenchmarkConfig.create("TRITON_full_fused")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .maxTokens(maxTokens));

        configs.add(BenchmarkConfig.create("TRITON_graph_capture")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .maxTokens(maxTokens));

        return configs;
    }

    private List<BenchmarkConfig> getAllConfigs() {
        List<BenchmarkConfig> configs = new ArrayList<>(getBaselineConfigs());
        if (isTritonAvailable()) {
            configs.addAll(getTritonConfigs());
        }
        return filterConfigs(configs);
    }

    private List<BenchmarkConfig> filterConfigs(List<BenchmarkConfig> configs) {
        String filter = System.getProperty("bench.configs");
        if (filter == null || filter.isEmpty()) return configs;

        String[] filters = filter.split(",");
        return configs.stream().filter(c -> {
            for (String f : filters) {
                String ft = f.trim();
                if (ft.endsWith("*")) {
                    if (c.getName().startsWith(ft.substring(0, ft.length() - 1))) return true;
                } else {
                    if (c.getName().equals(ft)) return true;
                }
            }
            return false;
        }).collect(Collectors.toList());
    }

    // ─── Backend Detection ──────────────────────────────────────────────

    private boolean isCudaBackend() {
        String backend = System.getProperty("backend.artifactId", "nd4j-cuda-12.9");
        return backend.contains("cuda");
    }

    private boolean isTritonAvailable() {
        try {
            return Nd4j.getNativeOps().isTritonAvailable();
        } catch (Exception e) {
            return false;
        }
    }

    // ─── Model Loading ──────────────────────────────────────────────────

    private SameDiff loadModel(ModelSpec spec) throws Exception {
        String key = spec.displayName();
        if (loadedModels.containsKey(key)) return loadedModels.get(key);

        log.info("Downloading model: {}", key);
        long t0 = System.currentTimeMillis();
        DownloadResult download = LLMModelDownloader.download(spec.llmModel, spec.quantType);
        long downloadMs = System.currentTimeMillis() - t0;
        log.info("  Download: {}ms (cached={}), size={}",
                downloadMs, !download.isDownloadedNow(), formatBytes(download.getFileSizeBytes()));

        // Inspect metadata before import
        GGMLMetadata metadata = GGMLModelImport.inspectModel(download.getModelFile());
        String arch = metadata.getArchitecture();
        architectures.put(key, arch != null ? arch : "unknown");
        log.info("  Architecture: {}", arch);

        log.info("Importing model: {}", key);
        t0 = System.currentTimeMillis();
        SameDiff model = GGMLModelImport.importModel(download.getModelFile().getAbsolutePath(),
                ConversionOptions.forInference());
        long importMs = System.currentTimeMillis() - t0;
        importTimings.put(key, importMs);
        log.info("  Import: {}ms, {} ops", importMs, model.ops().length);

        loadedModels.put(key, model);
        return model;
    }

    private Tokenizer loadTokenizer(ModelSpec spec) throws Exception {
        String key = spec.displayName();
        if (loadedTokenizers.containsKey(key)) return loadedTokenizers.get(key);

        // Skip gated models when no HF token is configured
        if (spec.requiresAuth) {
            String hfToken = System.getenv("HF_TOKEN");
            if (hfToken == null || hfToken.isEmpty()) {
                hfToken = System.getProperty("hf.token");
            }
            if (hfToken == null || hfToken.isEmpty()) {
                throw new IOException("Skipping " + spec.displayName() + ": gated model requires HF_TOKEN env var or -Dhf.token");
            }
        }

        log.info("Downloading tokenizer for: {}", key);
        File tokenizerFile = LLMModelDownloader.downloadCustom(spec.tokenizerUrl, spec.tokenizerCacheFile);
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        log.info("  Tokenizer loaded, vocab size: {}", tokenizer.getVocabSize());

        loadedTokenizers.put(key, tokenizer);
        return tokenizer;
    }

    // ─── Generation Helper ──────────────────────────────────────────────

    private BenchmarkResult runBenchmark(ModelSpec spec, BenchmarkConfig config, String prompt) {
        String modelName = spec.llmModel.getName();
        BenchmarkResult result = new BenchmarkResult(modelName, config.getName());

        try {
            SameDiff model = loadModel(spec);
            Tokenizer tokenizer = loadTokenizer(spec);

            result.opCount = model.ops().length;
            result.architecture = architectures.getOrDefault(spec.displayName(), "unknown");
            result.importMs = importTimings.getOrDefault(spec.displayName(), 0L);

            // Reset model state
            long t0 = System.currentTimeMillis();
            BenchmarkConfigApplier.resetModelState(model);
            BenchmarkConfigApplier.apply(config);
            if (config.getExecutionMode() != null) {
                model.setGraphExecutionMode(config.getExecutionMode());
            }
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            result.compileMs = System.currentTimeMillis() - t0;

            // Generate via GenerationPipeline (single-model mode: decoder only, no embedTokens)
            int maxTokens = config.getMaxTokens() > 0 ? config.getMaxTokens() : getMaxTokens();
            boolean optimizerEnabled = !"true".equals(System.getProperty("bench.no.optimizer"));
            try (GenerationPipeline pipeline = GenerationPipeline.create(
                    GenerationPipelineConfig.builder()
                            .decoder(model)
                            .tokenizer(tokenizer)
                            .samplingConfig(SamplingConfig.greedy())
                            .graphOptimizerEnabled(optimizerEnabled)
                            .maxNewTokens(maxTokens)
                            .build())) {
                t0 = System.currentTimeMillis();
                GenerationResult genResult = pipeline.generate(prompt, maxTokens);
                result.generateMs = System.currentTimeMillis() - t0;

                result.tokenCount = genResult.getGeneratedTokenCount();
                result.tokPerSec = genResult.getTokensPerSecond();
                result.firstTokenMs = genResult.getFirstTokenLatencyMs();
                result.steadyStateTokPerSec = genResult.getSteadyStateTokensPerSecond();
                result.finishReason = genResult.getFinishReason();

                log.info("[{}] {} → {} tokens at {:.2f} tok/s: {}",
                        config.getName(), modelName, result.tokenCount, result.tokPerSec,
                        truncate(genResult.getText(), 100));

                // Quality validation
                QualityReport quality = GenerationQualityValidator.validate(genResult);
                result.diversityRatio = quality.getDiversityRatio();
                result.coherenceScore = quality.getCoherenceScore();
            }

            assertTrue(result.tokenCount > 0, "Should generate at least 1 token");
            result.passed = true;

        } catch (Exception | AssertionError e) {
            // Auth-skipped gated models are not failures
            if (e.getMessage() != null && e.getMessage().contains("gated model requires HF_TOKEN")) {
                result.skipped = true;
                result.passed = false;
                result.failureMessage = "Skipped: no HF_TOKEN for gated model";
                log.info("[{}] {} SKIPPED: {}", config.getName(), modelName, e.getMessage());
            } else {
                result.passed = false;
                result.failureMessage = e.getClass().getSimpleName() + ": " + truncate(e.getMessage(), 120);
                log.error("[{}] {} FAILED: {}", config.getName(), modelName, e.getMessage(), e);
            }
        }
        return result;
    }

    // ════════════════════════════════════════════════════════════════════
    //  BENCHMARK TESTS
    // ════════════════════════════════════════════════════════════════════

    /**
     * Benchmark GGUF model import time across model families.
     * No tokenizer or generation needed — pure import timing.
     */
    @Test
    @DisplayName("Model Import Benchmark: GGUF → SameDiff timing across architectures")
    public void testModelImportBenchmark() throws Exception {
        List<ModelSpec> models = getSelectedModels();
        List<BenchmarkResult> results = new ArrayList<>();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  MODEL IMPORT BENCHMARK ({} models)", models.size());
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            BenchmarkResult result = new BenchmarkResult(spec.llmModel.getName(), "IMPORT");
            try {
                SameDiff model = loadModel(spec);
                result.importMs = importTimings.getOrDefault(spec.displayName(), 0L);
                result.opCount = model.ops().length;
                result.architecture = architectures.getOrDefault(spec.displayName(), "unknown");
                result.passed = true;

                log.info("  {} → {}ms import, {} ops, arch={}", spec.displayName(),
                        result.importMs, result.opCount, result.architecture);

            } catch (Exception e) {
                result.passed = false;
                result.failureMessage = e.getClass().getSimpleName() + ": " + truncate(e.getMessage(), 120);
                log.error("  {} → FAILED: {}", spec.displayName(), e.getMessage());
            }
            results.add(result);
        }

        printReport("MODEL IMPORT BENCHMARK", results);
        long failures = results.stream().filter(r -> !r.passed && !r.skipped).count();
        assertEquals(0, failures, failures + " model import(s) failed");
    }

    /**
     * Optimal baseline benchmark using AUTO execution mode.
     * AUTO engages the full DSP backend chain:
     *   CUDA: Triton islands + cuBLAS gap ops + CUDA graph replay
     *   CPU:  OpenVINO → oneDNN → native slot execution
     * Works on both CPU and CUDA backends.
     */
    @Test
    @DisplayName("Optimal Baseline: generation throughput per model (AUTO mode)")
    public void testOptimalBaseline() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped via bench.skip.generation");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        BenchmarkConfig config = BenchmarkConfig.optimal()
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(maxTokens);

        log.info("═══════════════════════════════════════════════════════");
        log.info("  OPTIMAL BASELINE ({} models, {} tokens)", models.size(), maxTokens);
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            BenchmarkResult result = runBenchmark(spec, config, prompt);
            results.add(result);
        }

        printReport("OPTIMAL BASELINE", results);
        long failures = results.stream().filter(r -> !r.passed && !r.skipped).count();
        assertEquals(0, failures, failures + " baseline benchmark(s) failed");
    }

    /**
     * Graph dump diagnostic — loads Qwen, runs GraphOptimizer, dumps full summary.
     * Shows pre- and post-optimization op histograms, attention ops, and shapes.
     *
     * Usage:
     *   cd platform-tests && mvn test -Dtest=TestLLMBenchmarkSuite#testGraphDump \
     *     -Dbackend.artifactId=nd4j-native -Dnd4j.optimizer.enabled=true \
     *     2>&1 | tee /tmp/graph-dump.log
     */
    @Test
    @DisplayName("Graph Dump: optimizer + op histogram diagnostic")
    public void testGraphDump() throws Exception {
        ModelSpec spec = MODEL_REGISTRY.get("qwen");
        assertNotNull(spec, "Qwen model not in registry");

        SameDiff raw = loadModel(spec);
        int rawOpCount = raw.ops().length;

        // Pre-optimization op histogram
        Map<String, Integer> rawHist = new java.util.TreeMap<>();
        for (org.nd4j.autodiff.functions.DifferentialFunction op : raw.ops()) {
            rawHist.merge(op.opName(), 1, Integer::sum);
        }

        log.info("═══════════════════════════════════════════════════════");
        log.info("  GRAPH DUMP DIAGNOSTIC — Qwen3.5-0.8B");
        log.info("═══════════════════════════════════════════════════════");
        log.info("");
        log.info("--- RAW GRAPH (pre-optimization) ---");
        log.info("  Total ops: {}", rawOpCount);
        log.info("  Total variables: {}", raw.variables().size());
        log.info("  Inputs: {}", raw.inputs());
        log.info("  Outputs: {}", raw.outputs());
        log.info("");
        log.info("  Op histogram (raw):");
        for (Map.Entry<String, Integer> e : rawHist.entrySet()) {
            log.info("    {}: {}", e.getKey(), e.getValue());
        }

        // Count attention ops specifically
        long rawAttentionOps = rawHist.getOrDefault("dot_product_attention_v2", 0);
        log.info("");
        log.info("  Attention ops (dot_product_attention_v2): {}", rawAttentionOps);

        // Run GraphOptimizer (same as GenerationPipeline.create())
        List<String> outputs = raw.outputs() != null
                ? new ArrayList<>(raw.outputs()) : new ArrayList<>();
        List<org.nd4j.autodiff.samediff.optimize.OptimizerSet> optimizations =
                org.nd4j.autodiff.samediff.optimize.GraphOptimizer.defaultOptimizations().stream()
                        .filter(s -> !(s instanceof org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations))
                        .collect(Collectors.toList());

        long optStart = System.currentTimeMillis();
        SameDiff optimized = org.nd4j.autodiff.samediff.optimize.GraphOptimizer.optimize(
                raw, outputs, optimizations);
        long optMs = System.currentTimeMillis() - optStart;

        int optOpCount = optimized.ops().length;
        Map<String, Integer> optHist = new java.util.TreeMap<>();
        for (org.nd4j.autodiff.functions.DifferentialFunction op : optimized.ops()) {
            optHist.merge(op.opName(), 1, Integer::sum);
        }

        log.info("");
        log.info("--- OPTIMIZED GRAPH (post-GraphOptimizer) ---");
        log.info("  GraphOptimizer: {} -> {} ops ({} removed) in {}ms",
                rawOpCount, optOpCount, rawOpCount - optOpCount, optMs);
        log.info("  Inputs: {}", optimized.inputs());
        log.info("  Outputs: {}", optimized.outputs());
        log.info("");
        log.info("  Op histogram (optimized):");
        for (Map.Entry<String, Integer> e : optHist.entrySet()) {
            log.info("    {}: {}", e.getKey(), e.getValue());
        }

        // Delta: what changed
        log.info("");
        log.info("  Op changes (raw -> optimized):");
        java.util.TreeSet<String> allOps = new java.util.TreeSet<>(rawHist.keySet());
        allOps.addAll(optHist.keySet());
        for (String opName : allOps) {
            int before = rawHist.getOrDefault(opName, 0);
            int after = optHist.getOrDefault(opName, 0);
            if (before != after) {
                log.info("    {}: {} -> {} ({}{})", opName, before, after,
                        after > before ? "+" : "", after - before);
            }
        }

        // Attention ops detail
        long optAttentionOps = optHist.getOrDefault("dot_product_attention_v2", 0);
        log.info("");
        log.info("  Attention ops (dot_product_attention_v2): {}", optAttentionOps);

        // Dump variable types summary
        int constants = 0, variables = 0, placeholders = 0, arrayType = 0;
        int fp16Weights = 0, fp32Weights = 0, otherDtype = 0;
        for (org.nd4j.autodiff.samediff.SDVariable var : optimized.variables()) {
            switch (var.getVariableType()) {
                case CONSTANT: constants++; break;
                case VARIABLE: variables++; break;
                case PLACEHOLDER: placeholders++; break;
                case ARRAY: arrayType++; break;
            }
            org.nd4j.linalg.api.ndarray.INDArray arr = var.getArr();
            if (arr != null && (var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT
                    || var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE)) {
                if (arr.dataType() == org.nd4j.linalg.api.buffer.DataType.HALF) fp16Weights++;
                else if (arr.dataType() == org.nd4j.linalg.api.buffer.DataType.FLOAT) fp32Weights++;
                else otherDtype++;
            }
        }
        log.info("");
        log.info("  Variable breakdown:");
        log.info("    CONSTANT: {}  VARIABLE: {}  PLACEHOLDER: {}  ARRAY: {}",
                constants, variables, placeholders, arrayType);
        log.info("    Weight dtypes: FP16={} FP32={} other={}", fp16Weights, fp32Weights, otherDtype);

        // Dump attention op details (inputs, shapes)
        log.info("");
        log.info("--- ATTENTION OP DETAILS ---");
        int attnIdx = 0;
        for (org.nd4j.autodiff.functions.DifferentialFunction op : optimized.ops()) {
            if ("dot_product_attention_v2".equals(op.opName())) {
                String[] inputNames = optimized.getInputsForOp(op);
                log.info("  Attention[{}]: name={}", attnIdx, op.getOwnName());
                if (inputNames != null) {
                    for (int i = 0; i < inputNames.length; i++) {
                        org.nd4j.autodiff.samediff.SDVariable v = optimized.getVariable(inputNames[i]);
                        String shape = "unknown";
                        if (v != null) {
                            org.nd4j.linalg.api.ndarray.INDArray a = v.getArr();
                            if (a != null) shape = Arrays.toString(a.shape()) + " " + a.dataType();
                            else if (v.isPlaceHolder() && v.placeholderShape() != null) shape = Arrays.toString(v.placeholderShape()) + " (placeholder)";
                            else shape = v.getVariableType() + " (no array yet)";
                        }
                        log.info("    input[{}]: {} -> {}", i, inputNames[i], shape);
                    }
                }
                // Check integer args (iArgs) for mask/dropout config
                attnIdx++;
            }
        }

        // Also dump RMSNorm and SwiGLU fusion ops if present
        long rmsNormCount = optHist.getOrDefault("rms_norm", 0);
        long swishMulCount = optHist.getOrDefault("swish_mul", 0);
        long xwPlusBCount = optHist.getOrDefault("xw_plus_b", 0);
        log.info("");
        log.info("--- FUSION OP SUMMARY ---");
        log.info("  rms_norm:   {} (fused from reduce_mean+sqrt+divide+multiply)", rmsNormCount);
        log.info("  swish_mul:  {} (fused from swish+multiply = SwiGLU)", swishMulCount);
        log.info("  xw_plus_b:  {} (fused matmul+bias)", xwPlusBCount);

        // Write full summary to file for easy inspection
        String summaryFile = "/tmp/qwen-graph-dump.txt";
        try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(summaryFile))) {
            pw.println("=== OPTIMIZED GRAPH SUMMARY ===");
            pw.println("Ops: " + optOpCount);
            pw.println("Variables: " + optimized.variables().size());
            pw.println();
            pw.println(optimized.summary());
        }
        log.info("");
        log.info("Full graph summary written to: {}", summaryFile);
        log.info("═══════════════════════════════════════════════════════");
    }

    /**
     * CUDA Graphs benchmark — captures the execution graph for replay.
     * CUDA backend only.
     */
    @Test
    @DisplayName("CUDA Graphs Benchmark: graph capture + replay throughput")
    public void testCudaGraphsBenchmark() {
        Assumptions.assumeTrue(isCudaBackend(), "CUDA Graphs requires CUDA backend");
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        BenchmarkConfig config = BenchmarkConfig.create("CUDA_GRAPHS")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .maxTokens(maxTokens);

        log.info("═══════════════════════════════════════════════════════");
        log.info("  CUDA GRAPHS BENCHMARK ({} models, {} tokens)", models.size(), maxTokens);
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            BenchmarkResult result = runBenchmark(spec, config, prompt);
            results.add(result);
        }

        printReport("CUDA GRAPHS BENCHMARK", results);
        long failures = results.stream().filter(r -> !r.passed && !r.skipped).count();
        assertEquals(0, failures, failures + " CUDA graphs benchmark(s) failed");
    }

    /**
     * Triton compilation + execution benchmark across models.
     * Tests compile time, kernel fusion, and steady-state throughput.
     * CUDA backend with Triton support required.
     */
    @Test
    @DisplayName("Triton Compile Benchmark: compilation + fused execution throughput")
    public void testTritonCompileBenchmark() {
        Assumptions.assumeTrue(isCudaBackend(), "Triton requires CUDA backend");
        Assumptions.assumeTrue(isTritonAvailable(), "Triton not available");
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        List<BenchmarkConfig> tritonConfigs = getTritonConfigs();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  TRITON COMPILE BENCHMARK ({} models × {} configs)", models.size(), tritonConfigs.size());
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            for (BenchmarkConfig config : tritonConfigs) {
                BenchmarkResult result = runBenchmark(spec, config, prompt);
                results.add(result);
            }
        }

        printReport("TRITON COMPILE BENCHMARK", results);
        long failures = results.stream().filter(r -> !r.passed && !r.skipped).count();
        assertEquals(0, failures, failures + " Triton benchmark(s) failed");
    }

    /**
     * Kernel fusion impact: compares Triton with and without section fusion.
     * Measures the throughput delta from fusing adjacent ops into single kernels.
     */
    @Test
    @DisplayName("Kernel Fusion Impact: section fusion on vs off")
    public void testKernelFusionImpact() {
        Assumptions.assumeTrue(isCudaBackend(), "Triton requires CUDA backend");
        Assumptions.assumeTrue(isTritonAvailable(), "Triton not available");
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        BenchmarkConfig noFusion = BenchmarkConfig.create("TRITON_no_fusion")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(false)
                .tritonCompileAll(true)
                .maxTokens(maxTokens);

        BenchmarkConfig withFusion = BenchmarkConfig.create("TRITON_with_fusion")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens);

        log.info("═══════════════════════════════════════════════════════");
        log.info("  KERNEL FUSION IMPACT ({} models)", models.size());
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            results.add(runBenchmark(spec, noFusion, prompt));
            results.add(runBenchmark(spec, withFusion, prompt));

            // Report delta
            BenchmarkResult noF = results.get(results.size() - 2);
            BenchmarkResult withF = results.get(results.size() - 1);
            if (noF.passed && withF.passed && noF.tokPerSec > 0) {
                double delta = ((withF.tokPerSec - noF.tokPerSec) / noF.tokPerSec) * 100;
                log.info("  {} fusion impact: {:.1f}% ({:.2f} → {:.2f} tok/s)",
                        spec.llmModel.getName(), delta, noF.tokPerSec, withF.tokPerSec);
            }
        }

        printReport("KERNEL FUSION IMPACT", results);
    }

    /**
     * Graph optimizer impact: compares execution with and without FP16 weight pre-casting.
     * The optimizer halves weight memory bandwidth by pre-casting FP32 weights to FP16.
     */
    @Test
    @DisplayName("Graph Optimizer Impact: FP16 pre-cast on vs off")
    public void testGraphOptimizerImpact() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  GRAPH OPTIMIZER IMPACT ({} models)", models.size());
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            // Without optimizer
            System.setProperty("nd4j.optimizer.enabled", "false");
            BenchmarkConfig noOpt = BenchmarkConfig.create("NO_OPTIMIZER")
                    .executionMode(GraphExecutionMode.AUTO)
                    .maxTokens(maxTokens);
            results.add(runBenchmark(spec, noOpt, prompt));

            // With optimizer (FP16 pre-cast)
            System.setProperty("nd4j.optimizer.enabled", "true");
            System.setProperty("nd4j.optimizer.fp16", "true");
            BenchmarkConfig withOpt = BenchmarkConfig.create("WITH_OPTIMIZER_FP16")
                    .executionMode(GraphExecutionMode.AUTO)
                    .maxTokens(maxTokens);
            results.add(runBenchmark(spec, withOpt, prompt));

            // Report delta
            BenchmarkResult noO = results.get(results.size() - 2);
            BenchmarkResult withO = results.get(results.size() - 1);
            if (noO.passed && withO.passed && noO.tokPerSec > 0) {
                double delta = ((withO.tokPerSec - noO.tokPerSec) / noO.tokPerSec) * 100;
                log.info("  {} optimizer impact: {:.1f}% ({:.2f} → {:.2f} tok/s)",
                        spec.llmModel.getName(), delta, noO.tokPerSec, withO.tokPerSec);
            }
        }

        // Restore default
        System.setProperty("nd4j.optimizer.enabled", "true");
        printReport("GRAPH OPTIMIZER IMPACT", results);
    }

    /**
     * Full configuration matrix: all selected models × all available configs.
     * This is the most comprehensive benchmark, combining all execution backends.
     */
    @Test
    @DisplayName("Full Config Matrix: all models × all execution modes")
    public void testFullConfigMatrix() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        List<BenchmarkConfig> configs = getAllConfigs();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  FULL CONFIG MATRIX ({} models × {} configs = {} runs)",
                models.size(), configs.size(), models.size() * configs.size());
        log.info("═══════════════════════════════════════════════════════");
        log.info("  Models: {}", models.stream().map(m -> m.llmModel.getName()).collect(Collectors.joining(", ")));
        log.info("  Configs: {}", configs.stream().map(BenchmarkConfig::getName).collect(Collectors.joining(", ")));

        for (ModelSpec spec : models) {
            for (BenchmarkConfig config : configs) {
                results.add(runBenchmark(spec, config, prompt));
            }
        }

        printReport("FULL CONFIG MATRIX", results);

        // Find fastest config per model
        Map<String, BenchmarkResult> fastest = new LinkedHashMap<>();
        for (BenchmarkResult r : results) {
            if (!r.passed) continue;
            BenchmarkResult current = fastest.get(r.modelName);
            if (current == null || r.tokPerSec > current.tokPerSec) {
                fastest.put(r.modelName, r);
            }
        }
        if (!fastest.isEmpty()) {
            log.info("──── Fastest config per model ────");
            for (Map.Entry<String, BenchmarkResult> e : fastest.entrySet()) {
                log.info("  {} → {} at {:.2f} tok/s", e.getKey(), e.getValue().configName, e.getValue().tokPerSec);
            }
        }

        long failures = results.stream().filter(r -> !r.passed && !r.skipped).count();
        assertEquals(0, failures, failures + " config(s) failed. See log for details.");
    }

    /**
     * Perplexity comparison across model families.
     * Evaluates language modeling quality on a fixed reference text.
     */
    @Test
    @DisplayName("Perplexity Comparison: language modeling quality across models")
    public void testPerplexityComparison() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        List<BenchmarkResult> results = new ArrayList<>();

        String referenceText = "The quick brown fox jumps over the lazy dog. " +
                "Machine learning is a subset of artificial intelligence that focuses on " +
                "building systems that learn from data. Neural networks are inspired by " +
                "the biological neural networks that constitute animal brains. " +
                "Deep learning uses multiple layers of neural networks to progressively " +
                "extract higher-level features from raw input data.";

        log.info("═══════════════════════════════════════════════════════");
        log.info("  PERPLEXITY COMPARISON ({} models)", models.size());
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            BenchmarkResult result = new BenchmarkResult(spec.llmModel.getName(), "PERPLEXITY");
            try {
                SameDiff model = loadModel(spec);
                Tokenizer tokenizer = loadTokenizer(spec);

                model.setGraphExecutionMode(GraphExecutionMode.AUTO);
                model.setDspAutoCompileEnabled(true);
                model.setDspNativeAutoCompileEnabled(true);

                long t0 = System.currentTimeMillis();
                PerplexityResult ppl = PerplexityEvaluator.evaluate(model, tokenizer,
                        referenceText, 512, 256);
                result.generateMs = System.currentTimeMillis() - t0;

                result.opCount = model.ops().length;
                result.architecture = architectures.getOrDefault(spec.displayName(), "unknown");
                result.passed = true;

                log.info("  {} → perplexity={:.4f}, bits/byte={:.4f}, tokens={}, time={}ms",
                        spec.displayName(), ppl.getPerplexity(), ppl.getBitsPerByte(),
                        ppl.getNumTokens(), ppl.getEvaluationTimeMs());

                assertTrue(ppl.getPerplexity() > 0, "Perplexity must be positive");
                assertTrue(ppl.getPerplexity() < 10000, "Perplexity should be reasonable");

            } catch (Exception e) {
                if (e.getMessage() != null && e.getMessage().contains("gated model requires HF_TOKEN")) {
                    result.skipped = true;
                    result.failureMessage = "Skipped: no HF_TOKEN for gated model";
                    log.info("  {} → SKIPPED: {}", spec.displayName(), e.getMessage());
                } else {
                    result.passed = false;
                    result.failureMessage = e.getClass().getSimpleName() + ": " + truncate(e.getMessage(), 120);
                    log.error("  {} → FAILED: {}", spec.displayName(), e.getMessage());
                }
            }
            results.add(result);
        }

        printReport("PERPLEXITY COMPARISON", results);
    }

    /**
     * Quantization comparison: Q4_K_M vs Q8_0 for the same model.
     * Compares file size, import time, generation speed, and output quality.
     */
    @Test
    @DisplayName("Quantization Comparison: Q4_K_M vs Q8_0 speed + quality")
    public void testQuantizationComparison() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        // Only run on Qwen (smallest, always available)
        ModelSpec qwenSpec = MODEL_REGISTRY.get("qwen");
        assertNotNull(qwenSpec, "Qwen model spec must exist");

        QuantType[] quants = {QuantType.Q4_K_M, QuantType.Q8_0};
        String quantFilter = System.getProperty("bench.quant.compare");
        if (quantFilter != null && !quantFilter.isEmpty()) {
            quants = Arrays.stream(quantFilter.split(","))
                    .map(String::trim)
                    .map(QuantType::valueOf)
                    .toArray(QuantType[]::new);
        }

        int maxTokens = getMaxTokens();
        String prompt = getPrompt();
        List<BenchmarkResult> results = new ArrayList<>();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  QUANTIZATION COMPARISON ({} quants, {} tokens)", quants.length, maxTokens);
        log.info("═══════════════════════════════════════════════════════");

        for (QuantType quant : quants) {
            BenchmarkResult result = new BenchmarkResult(qwenSpec.llmModel.getName(), quant.name());
            try {
                // Download this specific quant
                long t0 = System.currentTimeMillis();
                DownloadResult download = LLMModelDownloader.download(qwenSpec.llmModel, quant);
                long downloadMs = System.currentTimeMillis() - t0;

                // Import
                t0 = System.currentTimeMillis();
                SameDiff model = GGMLModelImport.importModel(download.getModelFile().getAbsolutePath(),
                        ConversionOptions.forInference());
                result.importMs = System.currentTimeMillis() - t0;
                result.opCount = model.ops().length;

                // Load tokenizer (shared across quants)
                Tokenizer tokenizer = loadTokenizer(qwenSpec);

                // Generate
                model.setGraphExecutionMode(GraphExecutionMode.AUTO);
                model.setDspAutoCompileEnabled(true);
                model.setDspNativeAutoCompileEnabled(true);

                try (GenerationPipeline pipeline = GenerationPipeline.create(
                        GenerationPipelineConfig.builder()
                                .decoder(model)
                                .tokenizer(tokenizer)
                                .samplingConfig(SamplingConfig.greedy())
                                .maxNewTokens(maxTokens)
                                .build())) {
                    t0 = System.currentTimeMillis();
                    GenerationResult genResult = pipeline.generate(prompt, maxTokens);
                    result.generateMs = System.currentTimeMillis() - t0;

                    result.tokenCount = genResult.getGeneratedTokenCount();
                    result.tokPerSec = genResult.getTokensPerSecond();
                    result.firstTokenMs = genResult.getFirstTokenLatencyMs();

                    QualityReport quality = GenerationQualityValidator.validate(genResult);
                    result.diversityRatio = quality.getDiversityRatio();
                    result.coherenceScore = quality.getCoherenceScore();
                    result.passed = true;

                    log.info("  {} → {} tokens, {:.2f} tok/s, import={}ms, size={}, output: {}",
                            quant, result.tokenCount, result.tokPerSec, result.importMs,
                            formatBytes(download.getFileSizeBytes()),
                            truncate(genResult.getText(), 80));
                }

            } catch (Exception e) {
                if (e.getMessage() != null && e.getMessage().contains("gated model requires HF_TOKEN")) {
                    result.skipped = true;
                    result.failureMessage = "Skipped: no HF_TOKEN for gated model";
                    log.info("  {} → SKIPPED: {}", quant, e.getMessage());
                } else {
                    result.passed = false;
                    result.failureMessage = e.getClass().getSimpleName() + ": " + truncate(e.getMessage(), 120);
                    log.error("  {} → FAILED: {}", quant, e.getMessage());
                }
            }
            results.add(result);
        }

        printReport("QUANTIZATION COMPARISON", results);
    }

    /**
     * Reference prompt accuracy: runs standard prompts and validates expected output.
     * Measures both correctness and generation speed across models.
     * For extraction models (lfm2-extract), uses EXTRACTION_PROMPTS with a JSON schema system prompt.
     */
    @Test
    @DisplayName("Reference Prompt Accuracy: correctness across models + prompts")
    public void testReferencePromptAccuracy() {
        Assumptions.assumeFalse(skipGeneration(), "Generation tests skipped");

        List<ModelSpec> models = getSelectedModels();
        int maxTokens = Math.max(getMaxTokens(), 50); // need enough tokens for meaningful output
        List<BenchmarkResult> results = new ArrayList<>();

        BenchmarkConfig config = BenchmarkConfig.optimal()
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(maxTokens);

        // Select prompts per model: extraction models use EXTRACTION_PROMPTS, others use BENCHMARK_PROMPTS
        int totalPrompts = 0;
        for (ModelSpec spec : models) {
            totalPrompts += getPromptsForModel(spec).size();
        }

        log.info("═══════════════════════════════════════════════════════");
        log.info("  REFERENCE PROMPT ACCURACY ({} models, {} total prompts)", models.size(), totalPrompts);
        log.info("═══════════════════════════════════════════════════════");

        for (ModelSpec spec : models) {
            Map<String, String[]> prompts = getPromptsForModel(spec);
            boolean isExtraction = isExtractionModel(spec);

            for (Map.Entry<String, String[]> entry : prompts.entrySet()) {
                String promptText = entry.getKey();
                String[] expected = entry.getValue();

                // For extraction models, wrap in ChatML-style template with JSON schema
                String prompt = isExtraction ? formatExtractionPrompt(promptText) : promptText;

                BenchmarkResult result = new BenchmarkResult(spec.llmModel.getName(),
                        "PROMPT:" + truncate(promptText, 30));
                try {
                    SameDiff model = loadModel(spec);
                    Tokenizer tokenizer = loadTokenizer(spec);

                    BenchmarkConfigApplier.resetModelState(model);
                    BenchmarkConfigApplier.apply(config);
                    model.setGraphExecutionMode(GraphExecutionMode.AUTO);
                    model.setDspAutoCompileEnabled(true);
                    model.setDspNativeAutoCompileEnabled(true);

                    try (GenerationPipeline pipeline = GenerationPipeline.create(
                            GenerationPipelineConfig.builder()
                                    .decoder(model)
                                    .tokenizer(tokenizer)
                                    .samplingConfig(SamplingConfig.greedy())
                                    .maxNewTokens(maxTokens)
                                    .build())) {
                        GenerationResult genResult = pipeline.generate(prompt, maxTokens);
                        result.tokenCount = genResult.getGeneratedTokenCount();
                        result.tokPerSec = genResult.getTokensPerSecond();
                        result.firstTokenMs = genResult.getFirstTokenLatencyMs();

                        QualityReport quality;
                        if (expected.length > 0) {
                            quality = GenerationQualityValidator.validateWithExpected(genResult, expected);
                        } else {
                            quality = GenerationQualityValidator.validateDiversity(genResult, 0.3);
                        }
                        result.diversityRatio = quality.getDiversityRatio();
                        result.coherenceScore = quality.getCoherenceScore();
                        result.passed = quality.isPassed();

                        if (!result.passed) {
                            result.failureMessage = "Quality check failed: " + quality.summary();
                        }

                        log.info("  [{}] {} → {} | {} tokens, {:.2f} tok/s: {}",
                                result.passed ? "PASS" : "FAIL", spec.llmModel.getName(),
                                truncate(promptText, 40), result.tokenCount, result.tokPerSec,
                                truncate(genResult.getText(), 80));
                    }

                } catch (Exception e) {
                    result.passed = false;
                    result.failureMessage = e.getClass().getSimpleName() + ": " + truncate(e.getMessage(), 120);
                    log.error("  {} → FAILED on '{}': {}", spec.llmModel.getName(), truncate(promptText, 30), e.getMessage());
                }
                results.add(result);
            }
        }

        printReport("REFERENCE PROMPT ACCURACY", results);
    }

    private boolean isExtractionModel(ModelSpec spec) {
        return spec.familyKey.equals("lfm2-extract");
    }

    private Map<String, String[]> getPromptsForModel(ModelSpec spec) {
        return isExtractionModel(spec) ? EXTRACTION_PROMPTS : BENCHMARK_PROMPTS;
    }

    /**
     * Formats an extraction prompt using the LFM2-Extract ChatML template.
     * Uses a generic JSON extraction system prompt that works across input types.
     */
    private String formatExtractionPrompt(String inputText) {
        return "<|im_start|>system\n" +
                "Return data as a JSON object. Extract all key entities, attributes, and values from the input text.\n" +
                "<|im_end|>\n" +
                "<|im_start|>user\n" +
                inputText + "\n" +
                "<|im_end|>\n" +
                "<|im_start|>assistant\n";
    }

    /**
     * Device-specific benchmark: runs on current backend and reports device characteristics.
     * For CUDA: reports GPU name, memory, compute capability.
     * For CPU: reports thread count, instruction set.
     */
    @Test
    @DisplayName("Device Info + Backend-Specific Benchmark")
    public void testDeviceSpecificBenchmark() {
        List<BenchmarkResult> results = new ArrayList<>();

        log.info("═══════════════════════════════════════════════════════");
        log.info("  DEVICE-SPECIFIC BENCHMARK");
        log.info("═══════════════════════════════════════════════════════");

        // Report device info
        boolean cuda = isCudaBackend();
        log.info("  Backend: {}", cuda ? "CUDA" : "CPU");
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            log.info("  Devices: {}", numDevices);
            log.info("  Triton available: {}", isTritonAvailable());
        } catch (Exception e) {
            log.info("  Device info unavailable: {}", e.getMessage());
        }

        if (skipGeneration()) {
            log.info("  Generation skipped — device info only");
            return;
        }

        // Run quick benchmark on default model
        ModelSpec spec = MODEL_REGISTRY.get("qwen");
        int maxTokens = getMaxTokens();
        String prompt = getPrompt();

        // AUTO mode — uses best available backend for this platform
        BenchmarkConfig autoConfig = BenchmarkConfig.create("AUTO")
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(maxTokens);
        results.add(runBenchmark(spec, autoConfig, prompt));

        // SLOT_BY_SLOT correctness baseline
        BenchmarkConfig slotConfig = BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens);
        results.add(runBenchmark(spec, slotConfig, prompt));

        if (cuda) {
            // CUDA-specific configs
            BenchmarkConfig cudaGraphs = BenchmarkConfig.create("CUDA_GRAPHS")
                    .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                    .maxTokens(maxTokens);
            results.add(runBenchmark(spec, cudaGraphs, prompt));

            if (isTritonAvailable()) {
                BenchmarkConfig triton = BenchmarkConfig.create("TRITON_best")
                        .tritonIncludeTypes(COMPILE_ALL_TYPES)
                        .tritonSectionFusion(true)
                        .tritonCompileAll(true)
                        .maxTokens(maxTokens);
                results.add(runBenchmark(spec, triton, prompt));
            }
        }

        printReport("DEVICE-SPECIFIC BENCHMARK", results);
    }

    // ─── Reporting ──────────────────────────────────────────────────────

    private void printReport(String title, List<BenchmarkResult> results) {
        long passed = results.stream().filter(r -> r.passed).count();
        long failed = results.size() - passed;

        log.info("");
        log.info("═══════════════════════════════════════════════════════");
        log.info("  {} RESULTS: {}/{} passed", title, passed, results.size());
        log.info("═══════════════════════════════════════════════════════");
        for (BenchmarkResult r : results) {
            log.info(r.summary());
        }

        // Summary statistics for generation results
        List<BenchmarkResult> genResults = results.stream()
                .filter(r -> r.passed && r.tokenCount > 0)
                .collect(Collectors.toList());
        if (!genResults.isEmpty()) {
            double maxTokPerSec = genResults.stream().mapToDouble(r -> r.tokPerSec).max().orElse(0);
            double minTokPerSec = genResults.stream().mapToDouble(r -> r.tokPerSec).min().orElse(0);
            double avgTokPerSec = genResults.stream().mapToDouble(r -> r.tokPerSec).average().orElse(0);
            long minFirstToken = genResults.stream().mapToLong(r -> r.firstTokenMs).min().orElse(0);

            log.info("──── Summary ────");
            log.info("  Throughput: min={:.2f}  avg={:.2f}  max={:.2f} tok/s", minTokPerSec, avgTokPerSec, maxTokPerSec);
            log.info("  First token latency: min={}ms", minFirstToken);
        }
        if (failed > 0) {
            log.info("  FAILURES: {}", failed);
        }
        log.info("═══════════════════════════════════════════════════════");
    }

    // ─── Utilities ──────────────────────────────────────────────────────

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "(null)";
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
