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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.*;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SmolDocling pipeline test with builder-based configuration matrix.
 *
 * Loads models ONCE, then loops through all meaningful combinations of
 * execution mode, Triton include types, fusion, graph capture, and arg table opts.
 *
 * Uses shared {@link BenchmarkRunner} infrastructure for the reset/configure/compile/decode/validate loop.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestSmolDoclingOptimizedPipeline {

    private static String pdfPath;
    private static int specificPage = -1;
    private static int renderDpi = 150;

    // ─── PipelineContext: shared state loaded once ─────────────────────────

    private static class PipelineContext {
        SameDiff decoder;
        SameDiff embedTokens;
        Tokenizer tokenizer;
        INDArray inputsEmbeds;
        int[] promptTokenIds;
        long hiddenSize;
        // Pipeline setup timings
        long downloadMs;
        long importMs;
        long visionMs;
        long embedMs;
        int visionFrames;
        int decoderOps;
        int embedOps;
    }

    // ─── Configuration matrix: performance-focused configs ──────────────────
    //
    // BEST: compileAll + COMPILE_ALL_TYPES + ATTENTION + GC + argOpt -> 23.28 tok/s (100 tok/s steady)
    // Without ATTENTION: compileAll + COMPILE_ALL_TYPES + GC + argOpt -> 20.47 tok/s (77 tok/s steady)
    // CUDA_GRAPHS baseline (no Triton) -> 11.40 tok/s (40 tok/s steady)
    // SLOT_BY_SLOT baseline -> 5.62 tok/s
    //
    // NEVER compile MATMUL (cuBLAS 2.8x faster), NEVER include SPLIT/CONCAT without compileAll
    // Flash attention (+ATTENTION) gives +30% decode speed with CUDA graph capture
    // dspCastElimination and dspFp16Compute are neutral-to-negative with CUDA graphs

    private static final String FULL_TRITON_TYPES =
            "ELEMENTWISE,REDUCTION,NORMALIZATION,GATHER,STACK,ATTENTION";

    // Best-known compileAll types (from triton-compileall.md: 83 tok/s)
    private static final String COMPILE_ALL_TYPES =
            "CONST_GEN,GATHER,CONCAT,SPLIT,STACK";

    private static List<BenchmarkConfig> getAllConfigs() {
        boolean triton = Nd4j.getNativeOps().isTritonAvailable();
        List<BenchmarkConfig> configs = new ArrayList<>();

        // 1. Baselines
        configs.add(BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("CUDA_GRAPHS")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .maxTokens(50));

        if (!triton) return configs;

        // 2. compileAll: individual section types (bisect crashes)
        for (String singleType : new String[]{"GATHER", "STACK", "CONST_GEN", "CONCAT", "SPLIT"}) {
            configs.add(BenchmarkConfig.create("TRITON_compileAll_" + singleType)
                    .tritonIncludeTypes(singleType)
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .maxTokens(20));
        }

        configs.add(BenchmarkConfig.create("TRITON_compileAll_safe")
                .tritonIncludeTypes("GATHER,STACK")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_gc")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(20));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // Diagnostic: GC + verify to find replay divergence
        configs.add(BenchmarkConfig.create("TRITON_gc_argOpt_VERIFY")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .tritonVerifyKernels(true)
                .tritonVerifyFullSnapshot(true)
                .maxTokens(5)
                .minDiversityPct(0));

        // Diagnostic: GC + force-recapture (re-capture every step, tests freshness)
        configs.add(BenchmarkConfig.create("TRITON_gc_argOpt_FORCE_RECAP")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .tritonForceRecapture(true)
                .maxTokens(10)
                .minDiversityPct(0));

        // Isolation: consolidated arg table only (no dirty tracking)
        configs.add(BenchmarkConfig.create("TRITON_gc_consolidatedOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(false)
                .maxTokens(20)
                .minDiversityPct(0));

        // Isolation: dirty tracking only (no consolidated arg table)
        configs.add(BenchmarkConfig.create("TRITON_gc_dirtyTrackingOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(false)
                .tritonArgDirtyTracking(true)
                .maxTokens(20)
                .minDiversityPct(0));

        // 3. FULL types: attention is biggest win (+23%)
        configs.add(BenchmarkConfig.create("TRITON_FULL_fused")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_FULL_fused_gc")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_FULL_fused_gc_argOpt")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 4. compileAll + FULL types combined
        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL_gc")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 5. Combined high-performance GC configs
        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL_gc_argOpt")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_MAX_PERF_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonProfile("MAX_PERF")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 6. DSP optimization flags (standalone)
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_castElim")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_fp16compute")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspFp16Compute(true)
                .maxTokens(50));

        // 7. DSP optimization flags + GC variants
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_castElim_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_fp16_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspFp16Compute(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 8. MAX_PERF profile (standalone, no GC)
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_MAX_PERF")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonProfile("MAX_PERF")
                .maxTokens(50));

        // 9. Ultimate combined: ATTN + castElim + fp16compute + GC + argOpt
        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_fp16_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .dspFp16Compute(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 10. ATTN + castElim only (no fp16compute), to isolate contributions
        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 11. Batch-zero + batched GEMM node reduction configs
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchZero")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchOps")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .dspBatchedGemm(true)
                .dspCastSinkMatmul(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("CUDA_GRAPHS_batchOps")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .dspBatchedGemm(true)
                .maxTokens(50));

        // Isolation config: batched GEMM only (no batch-zero) to isolate correctness
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        return configs;
    }

    // ─── Setup ─────────────────────────────────────────────────────────────

    @BeforeAll
    public static void setup() {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        System.setProperty("nd4j.optimizer.logApplied", "true");

        pdfPath = System.getProperty("vlm.test.pdf.path");
        String pageStr = System.getProperty("vlm.test.pdf.page");
        if (pageStr != null && !pageStr.isEmpty()) {
            specificPage = Integer.parseInt(pageStr);
        }
        String dpiStr = System.getProperty("vlm.test.pdf.dpi");
        if (dpiStr != null && !dpiStr.isEmpty()) {
            renderDpi = Integer.parseInt(dpiStr);
        }
    }

    // ─── Main test: loads models once, sweeps all configs ──────────────────

    @Test
    @DisplayName("Optimized SmolDocling: Configuration matrix sweep")
    public void testOptimizedDoclingPipeline() throws Exception {
        PipelineContext ctx = loadModelsAndPrepareEmbeddings();

        // Assert pipeline setup produced valid state
        assertNotNull(ctx.decoder, "Decoder model must be loaded");
        assertNotNull(ctx.embedTokens, "EmbedTokens model must be loaded");
        assertNotNull(ctx.tokenizer, "Tokenizer must be loaded");
        assertNotNull(ctx.inputsEmbeds, "Input embeddings must be prepared");
        assertFalse(ctx.inputsEmbeds.wasClosed(), "Input embeddings must not be closed");
        assertTrue(ctx.hiddenSize > 0, "Hidden size must be positive, got: " + ctx.hiddenSize);
        assertTrue(ctx.promptTokenIds.length > 0, "Prompt token IDs must not be empty");
        assertTrue(ctx.decoderOps > 0, "Decoder should have ops");

        log.info("Pipeline setup complete: download={}ms import={}ms vision={}ms embed={}ms",
                ctx.downloadMs, ctx.importMs, ctx.visionMs, ctx.embedMs);
        log.info("  decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}, frames={}",
                ctx.decoderOps, ctx.embedOps, ctx.hiddenSize, ctx.promptTokenIds.length, ctx.visionFrames);

        List<BenchmarkConfig> configs = getAllConfigs();

        // Filter configs by name if vlm.test.configs is set (comma-separated)
        String filterProp = System.getProperty("vlm.test.configs");
        if (filterProp != null && !filterProp.isEmpty()) {
            Set<String> keep = Set.of(filterProp.split(","));
            configs.removeIf(c -> !keep.contains(c.getName()));
            log.info("Filtered to {} configs via vlm.test.configs: {}", configs.size(), keep);
        }

        // Override maxTokens for all configs if vlm.test.maxTokens is set
        String maxTokensOverride = System.getProperty("vlm.test.maxTokens");
        if (maxTokensOverride != null && !maxTokensOverride.isEmpty()) {
            int mt = Integer.parseInt(maxTokensOverride);
            configs.forEach(c -> c.maxTokens(mt));
            log.info("Override maxTokens={} for all {} configs", mt, configs.size());
        }

        List<SameDiff> models = List.of(ctx.decoder, ctx.embedTokens);

        // Compile function: delegates to BenchmarkConfigApplier
        BenchmarkRunner.CompileFunction compileFn = config ->
                BenchmarkConfigApplier.compileModels(
                        ctx.decoder, "decoder", ctx.embedTokens, "embed_tokens", config);

        // Decode function: wraps StaticKvCacheDecodeLoop
        BenchmarkRunner.DecodeFunction decodeFn = config -> {
            StaticKvCacheDecodeLoop decodeLoop = StaticKvCacheDecodeLoop.builder()
                    .decoder(ctx.decoder)
                    .embedTokens(ctx.embedTokens)
                    .tokenizer(ctx.tokenizer)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(config.getMaxTokens())
                    .hiddenSize(ctx.hiddenSize)
                    .build();
            return decodeLoop.decode(ctx.inputsEmbeds, ctx.promptTokenIds);
        };

        // Validate function: structural tags + diversity checks
        BenchmarkRunner.ValidateFunction validateFn = (config, result) ->
                validateResult(config, result);

        // Run the matrix
        List<BenchmarkResult> results = BenchmarkRunner.runMatrix(
                configs, models, compileFn, decodeFn, validateFn, "vlm.config");

        ctx.tokenizer.close();
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);

        // Print report (throws if any config failed)
        StringBuilder pipelineInfo = new StringBuilder();
        pipelineInfo.append(String.format("Pipeline setup: download=%dms import=%dms vision=%dms embed=%dms\n\n",
                ctx.downloadMs, ctx.importMs, ctx.visionMs, ctx.embedMs));
        log.info("{}", pipelineInfo);
        BenchmarkRunner.printReport(results);
    }

    // ─── validateResult ────────────────────────────────────────────────────

    private void validateResult(BenchmarkConfig config, GenerationResult result) {
        String name = config.getName();

        // Basic generation assertions
        assertNotNull(result.getText(), name + ": generated text is null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                name + ": should have generated at least one token");
        assertNotNull(result.getTokenIds(), name + ": token IDs array is null");
        assertEquals(result.getGeneratedTokenCount(), result.getTokenIds().length,
                name + ": token count mismatch with tokenIds array length");
        assertTrue(result.getGenerationTimeMs() > 0,
                name + ": generation time must be positive");
        assertTrue(result.getTokensPerSecond() > 0,
                name + ": tokens/sec must be positive");
        assertNotNull(result.getFinishReason(),
                name + ": finish reason is null");

        String trimmed = result.getText().trim();

        // Structural tag check
        if (config.isExpectStructuralTags() && result.getGeneratedTokenCount() >= 10) {
            boolean hasDocTags = trimmed.contains("<") && trimmed.contains(">");
            if (hasDocTags) {
                Set<String> tagTypes = extractTagTypes(trimmed);
                assertFalse(tagTypes.isEmpty(),
                        name + ": found angle brackets but extracted no tag types");
                boolean hasStructuralTags = tagTypes.stream().anyMatch(t ->
                        t.equals("doctag") || t.equals("page") || t.equals("text") ||
                                t.equals("section_header") || t.equals("otsl") || t.equals("table"));
                assertTrue(hasStructuralTags,
                        name + ": expected structural DocTags in " + result.getGeneratedTokenCount() +
                                " tokens. Tags found: " + tagTypes +
                                ". Text: " + trimmed.substring(0, Math.min(200, trimmed.length())));
            }
        }

        // Degeneracy check
        if (result.getGeneratedTokenCount() >= 10) {
            int[] tokenIds = result.getTokenIds();
            Set<Integer> uniqueTokens = new HashSet<>();
            for (int id : tokenIds) uniqueTokens.add(id);
            double uniqueRatio = (double) uniqueTokens.size() / tokenIds.length;
            log.info("  Token diversity: {}/{} unique ({}%)",
                    uniqueTokens.size(), tokenIds.length, String.format("%.1f", uniqueRatio * 100));
            assertTrue(uniqueRatio > config.getMinDiversityPct() / 100.0,
                    name + ": degenerate output: " + uniqueTokens.size() + "/" + tokenIds.length +
                            " unique (min " + config.getMinDiversityPct() + "%)");
        }

        // Throughput check
        if (result.getGeneratedTokenCount() >= 5) {
            assertTrue(result.getTokensPerSecond() > 0.1,
                    name + ": throughput too low: " +
                            String.format("%.2f", result.getTokensPerSecond()) + " tok/s");
        }
    }

    // ─── KvScatter isolation test ──────────────────────────────────────────

    @Test
    @DisplayName("Test KvScatter op in isolation")
    public void testKvScatterIsolated() {
        int batch = 1, heads = 8, maxKvLen = 100, dim = 64;
        long cachePos = 5;

        INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, maxKvLen + 1, dim);
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

        org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter scatter =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(staticBuf, present, cachePos);

        INDArray[] result = Nd4j.getExecutioner().exec(scatter);
        assertNotNull(result, "KvScatter result is null");
        assertTrue(result.length > 0, "KvScatter result is empty");

        INDArray expectedEntry = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).dup();
        INDArray actualEntry = result[0].get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

        double maxDiff = expectedEntry.sub(actualEntry).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5, "KV scatter should copy present's last pos to static's cachePos, maxDiff=" + maxDiff);

        present.close();
        staticBuf.close();
    }

    // ─── loadModelsAndPrepareEmbeddings: one-time pipeline setup ──────────

    private PipelineContext loadModelsAndPrepareEmbeddings() throws Exception {
        PipelineContext ctx = new PipelineContext();
        Nd4j.getEnvironment().setTritonBuildThreads(4);

        // Download
        long t0 = System.currentTimeMillis();
        log.info("Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        ctx.downloadMs = System.currentTimeMillis() - t0;
        log.info("Download done [{}ms]", ctx.downloadMs);

        // Tokenizer
        ctx.tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        assertNotNull(ctx.tokenizer, "Tokenizer failed to load");
        assertTrue(ctx.tokenizer.getVocabSize() > 0, "Tokenizer vocab size must be positive");
        log.info("Tokenizer loaded: vocab_size={}", ctx.tokenizer.getVocabSize());

        // Import ONNX
        long importStart = System.currentTimeMillis();
        log.info("Importing ONNX models...");
        boolean forceReoptimize = Boolean.getBoolean("vlm.model.cache.disable");
        if (forceReoptimize) {
            OnnxModelCache.invalidateCache(decoderResult.getModelFile().getAbsolutePath());
        }
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        ctx.decoder = models[1];
        ctx.embedTokens = models[2];
        ctx.importMs = System.currentTimeMillis() - importStart;

        assertNotNull(visionEncoder, "Vision encoder import failed");
        assertNotNull(ctx.decoder, "Decoder import failed");
        assertNotNull(ctx.embedTokens, "EmbedTokens import failed");
        ctx.decoderOps = ctx.decoder.getOps().size();
        ctx.embedOps = ctx.embedTokens.getOps().size();
        assertTrue(ctx.decoderOps > 0, "Decoder has no ops");
        assertTrue(ctx.embedOps > 0, "EmbedTokens has no ops");
        assertTrue(visionEncoder.getOps().size() > 0, "Vision encoder has no ops");

        log.info("ONNX import done [{}ms]: vision={} ops, decoder={} ops, embed={} ops",
                ctx.importMs, visionEncoder.getOps().size(), ctx.decoderOps, ctx.embedOps);

        // Log op-type distribution for the decoder to verify optimizer ran
        Map<String, Integer> opCounts = new java.util.TreeMap<>();
        for (var entry : ctx.decoder.getOps().entrySet()) {
            var op = entry.getValue().getOp();
            String opName = op != null ? op.opName() : "null";
            opCounts.merge(opName, 1, Integer::sum);
        }
        log.info("Decoder op distribution ({} total):", ctx.decoderOps);
        opCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(25)
                .forEach(e -> log.info("  {} x {}", e.getValue(), e.getKey()));
        int rmsNormCount = opCounts.getOrDefault("rms_norm", 0);
        log.info("  rms_norm ops: {} (expected ~61 if optimizer ran)", rmsNormCount);
        if (rmsNormCount == 0) {
            log.warn("WARNING: No rms_norm ops found in decoder! GraphOptimizer may not have run. " +
                     "Check nd4j.optimizer.enabled=true and delete stale SDZ caches if needed.");
        }

        // Image preprocessing
        int targetSize = 512;
        BufferedImage pdfImage = loadImageFromPdfOrGenerate();
        assertNotNull(pdfImage, "Failed to load/generate test image");
        assertTrue(pdfImage.getWidth() > 0 && pdfImage.getHeight() > 0, "Test image has zero dimensions");

        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, 9);
        ctx.visionFrames = splitResult.getTotalFrames();
        assertTrue(ctx.visionFrames > 0, "No vision frames produced");

        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
        preprocessor.shutdown();
        assertNotNull(imageInput, "Image preprocessing returned null");
        assertFalse(imageInput.wasClosed(), "Image tensor closed after preprocessing");

        // Vision encoder
        long visionStart = System.currentTimeMillis();
        log.info("Running vision encoder on {} frames...", ctx.visionFrames);
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        assertFalse(visionInputNames.isEmpty(), "Vision encoder has no inputs");
        assertTrue(visionOutputNames.length > 0, "Vision encoder has no outputs");

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < ctx.visionFrames; frameIdx++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

            // DIAGNOSTIC: check pixel values are non-zero
            if (frameIdx == 0) {
                double minVal = singleFrame.minNumber().doubleValue();
                double maxVal = singleFrame.maxNumber().doubleValue();
                double meanVal = singleFrame.meanNumber().doubleValue();
                long zeroCount = singleFrame.eq(0.0).castTo(org.nd4j.linalg.api.buffer.DataType.INT64).sumNumber().longValue();
                long totalElements = singleFrame.length();
                log.info("DIAG pixel_values: shape={}, dtype={}, min={}, max={}, mean={}, zeroCount={}/{} ({}%)",
                        java.util.Arrays.toString(singleFrame.shape()), singleFrame.dataType(),
                        minVal, maxVal, meanVal, zeroCount, totalElements,
                        (100.0 * zeroCount / totalElements));
            }

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            // DIAGNOSTIC: Check shape variable before execution
            if (frameIdx == 0) {
                for (String vname : new String[]{"/Concat_output_0", "/Concat_output_0_"}) {
                    if (visionEncoder.hasVariable(vname)) {
                        INDArray arr = visionEncoder.getArrForVarName(vname);
                        if (arr != null) {
                            Nd4j.getAffinityManager().ensureLocation(arr, org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
                            log.info("DIAG {}: shape={}, dtype={}, values={}", vname, java.util.Arrays.toString(arr.shape()),
                                    arr.dataType(), arr.data().asLong());
                        } else {
                            log.info("DIAG {}: has variable but arr is null", vname);
                        }
                    }
                }
            }
            Map<String, INDArray> visionOutputs;
            try {
                // First, try to compute intermediate outputs to diagnose the Where → Create chain
                if (frameIdx == 0) {
                    try {
                        // Request just the Where output to see what it computes
                        String[] diagOutputs = {"/vision_model/embeddings/Where_output_0"};
                        // Also check if concat variable exists
                        for (String diagVar : new String[]{
                                "/ReduceSum_output_0",
                                "/Equal_1_output_0",
                                "/Not_output_0",
                                "/GatherND_output_0",
                                "/vision_model/embeddings/Shape_output_0",
                                "/vision_model/embeddings/Gather_output_0",
                                "/vision_model/embeddings/Concat_2_output_0",
                                "/vision_model/embeddings/Equal_output_0",
                                "/vision_model/embeddings/Where_output_0"}) {
                            if (visionEncoder.hasVariable(diagVar)) {
                                try {
                                    Map<String, INDArray> diagResult = visionEncoder.output(
                                            visionInputMap, diagVar);
                                    INDArray val = diagResult.get(diagVar);
                                    if (val != null) {
                                        Nd4j.getAffinityManager().ensureLocation(val,
                                                org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
                                        log.info("DIAG COMPUTED {}: shape={}, dtype={}, values={}",
                                                diagVar, java.util.Arrays.toString(val.shape()),
                                                val.dataType(),
                                                val.length() <= 20 ? val.toString() : "len=" + val.length());
                                    }
                                } catch (Exception diagEx) {
                                    log.info("DIAG COMPUTED {} FAILED: {}", diagVar, diagEx.getMessage());
                                }
                            }
                        }
                    } catch (Exception diagEx) {
                        log.info("DIAG intermediate computation failed: {}", diagEx.getMessage());
                    }
                }
                visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            } catch (Exception e) {
                // Dump intermediate variables for debugging
                log.error("Vision encoder failed on frame {}, dumping intermediates:", frameIdx);
                String[] diagVars = {
                    "/Equal_output_0", "/Cast_output_0", "/ReduceSum_output_0",
                    "/Equal_1_output_0", "/Not_output_0", "/NonZero_output_0",
                    "/Transpose_output_0", "/Reshape_output_0", "/GatherND_output_0",
                    "/vision_model/embeddings/Shape_output_0",
                    "/vision_model/embeddings/Gather_output_0",
                    "/vision_model/embeddings/Unsqueeze_2_output_0",
                    "/vision_model/embeddings/Constant_14_output_0",
                    "/vision_model/embeddings/Reshape_1_output_0",
                    "/vision_model/embeddings/Mul_1_output_0",
                    "/vision_model/embeddings/Equal_output_0",
                    "/vision_model/embeddings/ConstantOfShape_output_0",
                    "/vision_model/embeddings/Where_output_0"
                };
                for (String vname : diagVars) {
                    if (visionEncoder.hasVariable(vname)) {
                        INDArray arr = visionEncoder.getArrForVarName(vname);
                        if (arr != null) {
                            try {
                                Nd4j.getAffinityManager().ensureLocation(arr, org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
                                String vals = arr.length() <= 20 ? arr.toString() : "len=" + arr.length();
                                log.error("  DIAG {}: shape={}, dtype={}, values={}", vname,
                                    java.util.Arrays.toString(arr.shape()), arr.dataType(), vals);
                            } catch (Exception ex) {
                                log.error("  DIAG {}: shape={}, dtype={}, ERROR reading: {}", vname,
                                    java.util.Arrays.toString(arr.shape()), arr.dataType(), ex.getMessage());
                            }
                        } else {
                            log.error("  DIAG {}: arr is null (not yet computed?)", vname);
                        }
                    }
                }
                throw e;
            }
            assertNotNull(visionOutputs, "Vision encoder output null for frame " + frameIdx);
            assertFalse(visionOutputs.isEmpty(), "Vision encoder output empty for frame " + frameIdx);

            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            assertNotNull(selected, "Vision encoder selected output null for frame " + frameIdx);
            assertNotNull(selected.tensor, "Vision encoder selected tensor null for frame " + frameIdx);
            assertTrue(selected.tensor.rank() >= 2, "Vision output rank < 2 for frame " + frameIdx);

            INDArray out = selected.tensor.dup();
            assertFalse(out.wasClosed(), "Vision output dup closed for frame " + frameIdx);
            frameEmbeddings.add(out);

            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
        }

        assertEquals(ctx.visionFrames, frameEmbeddings.size(),
                "Frame embedding count mismatch: expected " + ctx.visionFrames);

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }
        imageInput.close();
        ctx.visionMs = System.currentTimeMillis() - visionStart;

        assertFalse(visionEmbeddings.wasClosed(), "Concatenated vision embeddings closed");
        assertTrue(visionEmbeddings.rank() == 3, "Vision embeddings should be rank 3, got " + visionEmbeddings.rank());
        log.info("Vision encoder done [{}ms]: shape={}", ctx.visionMs,
                Arrays.toString(visionEmbeddings.shape()));

        freeModelConstants(visionEncoder, "vision encoder");

        // Build prompt + embeddings
        long embedStart = System.currentTimeMillis();
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(ctx.tokenizer);
        assertTrue(imageTokenId >= 0, "Image token ID should be non-negative");

        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / ctx.visionFrames;
        assertTrue(imageSeqLenPerFrame > 0, "Image seq len per frame must be positive");

        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        ctx.promptTokenIds = ctx.tokenizer.encode(chatPrompt, false).getIds();
        assertTrue(ctx.promptTokenIds.length > 0, "Prompt encoding produced no tokens");

        INDArray promptIdsTensor = Nd4j.createFromArray(ctx.promptTokenIds)
                .reshape(1, ctx.promptTokenIds.length).castTo(DataType.LONG);
        String embedInputName = ctx.embedTokens.inputs().isEmpty() ? "input_ids" : ctx.embedTokens.inputs().get(0);
        String[] embedOutputNames = ctx.embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOutputs = ctx.embedTokens.output(
                Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        assertNotNull(embedOutputs, "EmbedTokens output is null");
        assertFalse(embedOutputs.isEmpty(), "EmbedTokens produced no output");

        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }
        assertNotNull(textEmbeddings, "embed_tokens produced no output");
        assertFalse(textEmbeddings.wasClosed(), "Text embeddings closed after dup");

        ctx.hiddenSize = visionEmbeddings.shape()[2];
        assertEquals(ctx.hiddenSize, textEmbeddings.shape()[2],
                "Hidden size mismatch: vision=" + ctx.hiddenSize + " text=" + textEmbeddings.shape()[2]);
        assertTrue(ctx.hiddenSize > 0, "Hidden size must be positive");

        ctx.inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, ctx.promptTokenIds, imageTokenId);
        assertNotNull(ctx.inputsEmbeds, "Merged embeddings are null");
        assertFalse(ctx.inputsEmbeds.wasClosed(), "Merged embeddings are closed");
        assertTrue(ctx.inputsEmbeds.rank() == 3,
                "Merged embeddings should be rank 3, got " + ctx.inputsEmbeds.rank());

        if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();
        ctx.embedMs = System.currentTimeMillis() - embedStart;
        log.info("Embeddings merged [{}ms]: shape={}", ctx.embedMs,
                Arrays.toString(ctx.inputsEmbeds.shape()));

        return ctx;
    }

    // ─── Utility helpers ──────────────────────────────────────────────────

    private Set<String> extractTagTypes(String text) {
        Set<String> tagTypes = new HashSet<>();
        int idx = 0;
        while (idx < text.length()) {
            int open = text.indexOf('<', idx);
            if (open < 0) break;
            int close = text.indexOf('>', open);
            if (close < 0) break;
            String tag = text.substring(open + 1, close);
            if (tag.startsWith("/")) tag = tag.substring(1);
            int space = tag.indexOf(' ');
            if (space > 0) tag = tag.substring(0, space);
            if (!tag.isEmpty()) tagTypes.add(tag);
            idx = close + 1;
        }
        return tagTypes;
    }

    private BufferedImage loadImageFromPdfOrGenerate() throws IOException {
        if (pdfPath != null && new File(pdfPath).exists()) {
            try (PDDocument document = PDDocument.load(new File(pdfPath))) {
                PDFRenderer renderer = new PDFRenderer(document);
                return renderer.renderImageWithDPI(specificPage >= 0 ? specificPage : 0, renderDpi, ImageType.RGB);
            }
        }
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

    private void freeModelConstants(SameDiff model, String label) {
        int closedArrays = 0;
        long closedBytes = 0;
        for (org.nd4j.autodiff.samediff.ArrayHolder holder :
                new org.nd4j.autodiff.samediff.ArrayHolder[]{model.getConstantArrays(), model.getVariablesArrays()}) {
            for (String name : new ArrayList<>(holder.arrayNames())) {
                INDArray arr = holder.removeArray(name);
                if (arr != null && !arr.wasClosed()) {
                    closedBytes += arr.length() * arr.dataType().width();
                    arr.data().setConstant(false);
                    arr.close();
                    closedArrays++;
                }
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("  Freed {} {} arrays ({}MB)", closedArrays, label, closedBytes / (1024 * 1024));
    }
}
