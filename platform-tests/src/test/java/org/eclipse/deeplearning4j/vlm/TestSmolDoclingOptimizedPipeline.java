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
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SmolDocling pipeline test with builder-based configuration matrix.
 *
 * Loads models ONCE via {@link GenerationPipeline} and {@link VisionEncoder},
 * then loops through all meaningful combinations of execution mode, Triton
 * include types, fusion, graph capture, and arg table opts.
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

    // ─── Configuration matrix: performance-focused configs ──────────────────
    //
    // BEST CURRENT DEFAULT: OPTIMAL / SMOLDOC_IDEAL. Both use compileAll +
    // CONST_GEN,GATHER,GATHER_ND,CONCAT,SPLIT,SPLIT_V,STACK,STRIDED_SLICE,
    // NORMALIZATION,ATTENTION,REDUCTION + CUDA graph replay + arg table opts +
    // dirty tracking + batched GEMM + warps4/stages1.
    //
    // NEVER compile MATMUL (cuBLAS 2.8x faster), NEVER include SPLIT/CONCAT without compileAll
    // Flash attention (+ATTENTION) gives +30% decode speed with CUDA graph capture
    // dspCastElimination is neutral with CUDA graphs
    // FP16: use nd4j.optimizer.fp16=true (pre-cast weights at load) NOT dspFp16Compute (runtime double-cast)

    private static final String FULL_TRITON_TYPES =
            "ELEMENTWISE,REDUCTION,NORMALIZATION,GATHER,STACK,ATTENTION";

    // Best-known compileAll types that achieved 100 tok/s decode (batchops-combined-test.log)
    // CRITICAL: Excludes NORMALIZATION/REDUCTION - Triton compilation is SLOWER than native fallback
    // rms_norm falls back to native CUDA kernel which is faster than Triton for these ops
    private static final String COMPILE_ALL_TYPES =
            "CONST_GEN,GATHER,CONCAT,SPLIT,STACK";

    private static final String COMPILE_ALL_TYPES_WITH_NORM =
            COMPILE_ALL_TYPES + ",NORMALIZATION";

    private static final String COMPILE_ALL_TYPES_WITH_NORM_NO_CONCAT =
            "CONST_GEN,GATHER,SPLIT,STACK,NORMALIZATION";

    private static final String COMPILE_ALL_TYPES_WITH_NORM_AND_MATMUL =
            COMPILE_ALL_TYPES_WITH_NORM + ",MATMUL";

    // The source page renders the heading's decorative drop-cap "M" as a
    // separate visual object. Validate the page-specific prose rather than
    // requiring the OCR model to merge that glyph into the following word.
    private static final String[] SMOLDOC_IDEAL_EXPECTED_SUBSTRINGS = {
            "heroes are set apart from their contemporaries",
            "mythic characters",
            "mythic paths"
    };

    private static final String DEFAULT_LLM_TRITON_INCLUDE_TYPES =
            "CONST_GEN,GATHER,GATHER_ND,CONCAT,SPLIT,SPLIT_V,STACK,STRIDED_SLICE,NORMALIZATION,ATTENTION,REDUCTION";
    private static final int PRODUCTION_MAX_PREFILL = 3072;
    private static final int PRODUCTION_MAX_KV = 4608;

    private static BenchmarkConfig smolDoclingIdealConfig() {
        return BenchmarkConfig.create("SMOLDOC_IDEAL")
                .tritonIncludeTypes(DEFAULT_LLM_TRITON_INCLUDE_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonFusionScoring(false)
                .tritonMergedCaptureThroughViews(true)
                .tritonNumWarps(4).tritonNumStages(1)
                .cublasTf32(true).tritonTf32(true)
                .dspBatchedGemm(true)
                .dspFreezeMergeSegments(true)
                .dspExecutionTiming(Boolean.getBoolean("vlm.test.dspExecutionTiming"))
                .maxTokens(250)
                .minDiversityPct(0)
                .expectedSubstrings(SMOLDOC_IDEAL_EXPECTED_SUBSTRINGS);
    }

    private static List<BenchmarkConfig> getAllConfigs() {
        // Use isCudaAvailable() to guard GPU-only configs (OPTIMAL, SMOLDOC_IDEAL, Triton-dependent).
        // isTritonAvailable() returns true even on CPU (Triton compiler is compiled into CPU libnd4j),
        // so it cannot be used to distinguish CUDA-capable from CPU-only environments.
        boolean cuda = Nd4j.backends().isCudaAvailable();
        List<BenchmarkConfig> configs = new ArrayList<>();

        // Core configs: OPTIMAL (Triton) and SLOT_BY_SLOT (cuBLAS baseline).
        if (cuda) {
            configs.add(BenchmarkConfig.optimal());
            configs.add(smolDoclingIdealConfig());
            // DIAG_TIMING mirrors OPTIMAL but enables per-step native timing instrumentation.
            // Use with: -Dvlm.test.configs=DIAG_TIMING
            // Output: COMPOSITE_REPLAY_TIMING lines with prezero/units/actTick breakdown per step.
            configs.add(BenchmarkConfig.create("DIAG_TIMING")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true).dspFreezeMergeSegments(true)
                    .dspExecutionTiming(true)  // enable per-step native timing
                    .maxTokens(30).minDiversityPct(0));
        }

        // SLOT_BY_SLOT baseline — no Triton, no graph capture, proves model works
        configs.add(BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(100)
                .minDiversityPct(0));

        // Build the extended matrix whenever the caller explicitly selects configs.
        String filterProp = System.getProperty("vlm.test.configs");
        boolean includeAll = filterProp != null && !filterProp.trim().isEmpty();
        if (!includeAll) return configs;

        // ── Additional configs below only run with vlm.test.configs=<name> ──
        if (cuda) {
            configs.add(BenchmarkConfig.create("TRITON_NO_GC")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_argTable")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_batchedGemm")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .dspBatchedGemm(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_tf32")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .cublasTf32(true).tritonTf32(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_graphCapture_only")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_graphCapture_allSettings")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_noGC_allSettings")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(10).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_VERIFY")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonVerifyKernels(true)
                    .maxTokens(3).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("DIAG_Triton_gc_noATTN")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES)
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_graphCapture")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_argTable")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_batchedGemm")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .dspBatchedGemm(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_tf32")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .cublasTf32(true).tritonTf32(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_gc_argTable")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
            configs.add(BenchmarkConfig.create("BISECT_noGC")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        return configs;
    }

    // ─── Setup ─────────────────────────────────────────────────────────────

    @BeforeAll
    public static void setup() {
        // Only enable debug+verbose when explicitly requested via -Dnd4j.env.debug=true / -Dnd4j.env.verbose=true
        if (Boolean.getBoolean("nd4j.env.debug")) {
            Nd4j.getEnvironment().setDebug(true);
        }
        if (Boolean.getBoolean("nd4j.env.verbose")) {
            Nd4j.getEnvironment().setVerbose(true);
        }

        String optEnabled = System.getProperty("nd4j.optimizer.enabled");
        if (optEnabled == null || optEnabled.isEmpty()) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String fp16Prop = System.getProperty("nd4j.optimizer.fp16");
        if (fp16Prop == null || fp16Prop.isEmpty()) {
            // Default fp16 to false on CPU: software HALF emulation is ~13s/token on CPU.
            // FP16 weight pre-casting only benefits CUDA Tensor Cores.
            // When run via run-benchmark.sh, this is already handled by --no-fp16 auto-select;
            // this guard covers direct mvn test invocations without -Dnd4j.optimizer.fp16.
            boolean onCuda = Nd4j.backends().isCudaAvailable();
            System.setProperty("nd4j.optimizer.fp16", onCuda ? "true" : "false");
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
        Nd4j.getEnvironment().setTritonBuildThreads(4);
        boolean captureExternalWorkspaceGaps = Boolean.getBoolean("vlm.test.dspCaptureExternalWorkspaceGaps");
        if (captureExternalWorkspaceGaps) {
            try {
                Class<?> cudaEnvironment = Class.forName("org.nd4j.linalg.jcublas.bindings.Nd4jCuda$Environment");
                Object environment = cudaEnvironment.getMethod("getInstance").invoke(null);
                Object dsp = cudaEnvironment.getMethod("dsp").invoke(environment);
                dsp.getClass().getMethod("setGapCaptureBlockExternalWorkspace", boolean.class)
                        .invoke(dsp, false);
                log.info("Enabled isolated external-workspace gap capture for this CUDA benchmark process");
            } catch (ClassNotFoundException e) {
                log.info("Ignoring CUDA external-workspace gap capture option on a non-CUDA backend");
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Failed to configure CUDA external-workspace gap capture", e);
            }
        }

        // ── Phase 1: Download models ──
        long phaseNs = phaseStart("DOWNLOAD_MODELS", benchmarkInputSummary());
        long downloadMs;
        VLMModelDownloader.DownloadResult visionDl, decoderDl, embedTokensDl, tokenizerDl;
        try {
            long t0 = System.currentTimeMillis();
            visionDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
            decoderDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
            embedTokensDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
            tokenizerDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
            VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
            downloadMs = System.currentTimeMillis() - t0;
        } catch (Throwable t) {
            throw phaseFailure("DOWNLOAD_MODELS", benchmarkInputSummary(), t);
        }
        phaseSuccess("DOWNLOAD_MODELS", phaseNs, "downloadMs=" + downloadMs);

        // ── Phase 2: Load tokenizer ──
        phaseNs = phaseStart("TOKENIZER_LOAD", "tokenizer=" + safeFileName(tokenizerDl.getModelFile()));
        Tokenizer tokenizer;
        try {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDl.getModelFile());
            assertNotNull(tokenizer, "Tokenizer failed to load");
            assertTrue(tokenizer.getVocabSize() > 0, "Tokenizer vocab size must be positive");
        } catch (Throwable t) {
            throw phaseFailure("TOKENIZER_LOAD", "tokenizer=" + safeFileName(tokenizerDl.getModelFile()), t);
        }
        phaseSuccess("TOKENIZER_LOAD", phaseNs, "vocabSize=" + tokenizer.getVocabSize());

        // ── Phase 3: Import ONNX models ──
        phaseNs = phaseStart("IMPORT_MODELS", "decoder=" + safeFileName(decoderDl.getModelFile()));
        long importMs;
        SameDiff visionEncoderSd, decoder, embedTokensSd;
        try {
            long importStart = System.currentTimeMillis();
            boolean forceReoptimize = Boolean.getBoolean("vlm.model.cache.disable");
            if (forceReoptimize) {
                OnnxModelCache.invalidateCache(decoderDl.getModelFile().getAbsolutePath());
            }
            SameDiff[] models = OnnxModelCache.importAllWithCache(
                    visionDl.getModelFile().getAbsolutePath(),
                    decoderDl.getModelFile().getAbsolutePath(),
                    embedTokensDl.getModelFile().getAbsolutePath()
            );
            visionEncoderSd = models[0];
            decoder = models[1];
            embedTokensSd = models[2];
            importMs = System.currentTimeMillis() - importStart;

            assertNotNull(visionEncoderSd, "Vision encoder import failed");
            assertNotNull(decoder, "Decoder import failed");
            assertNotNull(embedTokensSd, "EmbedTokens import failed");
            assertTrue(decoder.getOps().size() > 0, "Decoder has no ops");
            assertTrue(embedTokensSd.getOps().size() > 0, "EmbedTokens has no ops");
        } catch (Throwable t) {
            throw phaseFailure("IMPORT_MODELS", "decoder=" + safeFileName(decoderDl.getModelFile()), t);
        }
        phaseSuccess("IMPORT_MODELS", phaseNs,
                "visionOps=" + visionEncoderSd.getOps().size()
                        + " decoderOps=" + decoder.getOps().size()
                        + " embedOps=" + embedTokensSd.getOps().size());

        // Log decoder op-type distribution
        Map<String, Integer> opCounts = new TreeMap<>();
        for (var entry : decoder.getOps().entrySet()) {
            var op = entry.getValue().getOp();
            String opName = op != null ? op.opName() : "null";
            opCounts.merge(opName, 1, Integer::sum);
        }
        log.info("Decoder op distribution ({} total):", decoder.getOps().size());
        opCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(25)
                .forEach(e -> log.info("  {} x {}", e.getValue(), e.getKey()));

        // ── Phase 4: Image preprocessing ──
        int targetSize = 512;
        phaseNs = phaseStart("IMAGE_PREPROCESS", benchmarkInputSummary());
        BufferedImage pdfImage;
        ImageTiler.SplitImageResult splitResult;
        INDArray imageInput;
        int visionFrames;
        try {
            pdfImage = loadImageFromPdfOrGenerate();
            assertNotNull(pdfImage, "Failed to load/generate test image");
            BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
            splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, -1);
            visionFrames = splitResult.getTotalFrames();
            assertTrue(visionFrames > 0, "No vision frames produced");

            PreprocessorConfig ppConfig = new PreprocessorConfig();
            ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
            ppConfig.setDoRescale(true);
            ppConfig.setRescaleFactor(1.0 / 255.0);
            ppConfig.setDoNormalize(true);
            ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
            ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
            VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
            imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
            preprocessor.shutdown();
            assertNotNull(imageInput, "Image preprocessing returned null");
        } catch (Throwable t) {
            throw phaseFailure("IMAGE_PREPROCESS", benchmarkInputSummary(), t);
        }
        phaseSuccess("IMAGE_PREPROCESS", phaseNs,
                "image=" + pdfImage.getWidth() + "x" + pdfImage.getHeight() + " frames=" + visionFrames);

        // ── Phase 5: Vision encoding via VisionEncoder ──
        phaseNs = phaseStart("VISION_ENCODE", "frames=" + visionFrames);
        INDArray visionEmbeddings;
        long visionMs;
        // Keep visionEncoder accessible after this phase so freeModelMemory() can be called
        // AFTER EmbeddingMerger.mergeEmbeddings() in PIPELINE_SETUP. Calling freeModelMemory()
        // before mergeEmbeddings() risks freeing model constants whose native addresses may be
        // recycled into visionEmbeddings by ArrayCacheMemoryMgr, corrupting the buffer.
        VisionEncoder visionEncoder;
        try {
            visionEncoder = VisionEncoder.builder()
                    .model(visionEncoderSd)
                    .targetSize(targetSize)
                    .build();

            VisionEncoder.Result visionResult = visionEncoder.encode(imageInput, visionFrames, splitResult);
            visionEmbeddings = visionResult.getEmbeddings();
            visionMs = visionResult.getEncodingTimeMs();

            assertFalse(visionEmbeddings.wasClosed(), "Vision embeddings closed");
            assertTrue(visionEmbeddings.rank() == 3, "Vision embeddings should be rank 3, got " + visionEmbeddings.rank());
            if (isDspAssertEnabled()) {
                DspPlanAssertions.assertPhaseReached(visionEncoderSd, PlanPhase.SHAPES_FROZEN,
                        "vision encoder benchmark");
                DspPlanAssertions.assertNoSegmentFailures(visionEncoderSd, "vision encoder benchmark");
                DspPlanAssertions.assertNoCaptureFailures(visionEncoderSd, "vision encoder benchmark");
                DspPlanAssertions.assertNoFallbacks(visionEncoderSd, "vision encoder benchmark");
            }
            log.info("Vision encoder done [{}ms]: shape={}", visionMs, Arrays.toString(visionEmbeddings.shape()));

            imageInput.close();
            // NOTE: freeModelMemory() is intentionally deferred to AFTER mergeEmbeddings() below.
            // See PIPELINE_SETUP phase for the actual call.
        } catch (Throwable t) {
            throw phaseFailure("VISION_ENCODE", "frames=" + visionFrames, t);
        }
        phaseSuccess("VISION_ENCODE", phaseNs,
                "frames=" + visionFrames + " " + summarizeTensor("visionEmbeddings", visionEmbeddings));

        // ── Phase 6: Build GenerationPipeline + merge embeddings ──
        phaseNs = phaseStart("PIPELINE_SETUP", "building GenerationPipeline + merging embeddings");
        GenerationPipeline pipeline;
        INDArray inputsEmbeds;
        int[] promptTokenIds;
        long hiddenSize;
        long embedMs;
        try {
            pipeline = GenerationPipeline.create(
                    GenerationPipelineConfig.builder()
                            .decoder(decoder)
                            .embedTokens(embedTokensSd)
                            .tokenizer(tokenizer)
                            .samplingConfig(SamplingConfig.greedy())
                            .maxNewTokens(256)
                            .maxPrefillLength(PRODUCTION_MAX_PREFILL)
                            .maxKvCacheLength(PRODUCTION_MAX_KV)
                            .benchmarkConfig(BenchmarkConfig.optimal())
                            .build());

            long embedStart = System.currentTimeMillis();
            int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
            assertTrue(imageTokenId >= 0, "Image token ID should be non-negative");

            int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / visionFrames;
            assertTrue(imageSeqLenPerFrame > 0, "Image seq len per frame must be positive");

            String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                    splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
            String chatPrompt = "<|im_start|>User:" + imagePrompt
                    + "Convert this page to docling.<end_of_utterance>\nAssistant:";
            promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
            assertTrue(promptTokenIds.length > 0, "Prompt encoding produced no tokens");

            INDArray textEmbeddings = pipeline.embedTokens(promptTokenIds);
            assertNotNull(textEmbeddings, "embed_tokens produced no output");

            hiddenSize = visionEmbeddings.shape()[2];
            assertEquals(hiddenSize, textEmbeddings.shape()[2],
                    "Hidden size mismatch: vision=" + hiddenSize + " text=" + textEmbeddings.shape()[2]);

            inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
            assertNotNull(inputsEmbeds, "Merged embeddings are null");
            assertFalse(inputsEmbeds.wasClosed(), "Merged embeddings are closed");
            assertTrue(inputsEmbeds.rank() == 3, "Merged embeddings should be rank 3");

            if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();
            embedMs = System.currentTimeMillis() - embedStart;
            log.info("Embeddings merged [{}ms]: shape={}", embedMs, Arrays.toString(inputsEmbeds.shape()));

            // Free vision encoder model memory AFTER mergeEmbeddings() has consumed visionEmbeddings.
            // Deferring until here avoids a non-deterministic buffer-aliasing crash: freeModelMemory()
            // releases ~320 model constant DataBuffers, and ArrayCacheMemoryMgr may immediately recycle
            // one of those native addresses into visionEmbeddings, setting its _lenInBytes=0.
            // mergeEmbeddings() then calls reshape() on the corrupted buffer and throws "buffer is empty".
            visionEncoder.freeModelMemory();
        } catch (Throwable t) {
            throw phaseFailure("PIPELINE_SETUP", "building GenerationPipeline + merging embeddings", t);
        }
        phaseSuccess("PIPELINE_SETUP", phaseNs,
                summarizeTokens("promptTokenIds", promptTokenIds) + " "
                        + summarizeTensor("inputsEmbeds", inputsEmbeds));

        log.info("Pipeline setup complete: download={}ms import={}ms vision={}ms embed={}ms",
                downloadMs, importMs, visionMs, embedMs);
        log.info("  decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}, frames={}",
                decoder.getOps().size(), embedTokensSd.getOps().size(), hiddenSize,
                promptTokenIds.length, visionFrames);

        // ── Phase 7: Configuration matrix sweep ──
        List<BenchmarkConfig> configs = getAllConfigs();

        String filterProp = System.getProperty("vlm.test.configs");
        if (filterProp != null && !filterProp.isEmpty() && !"ALL".equalsIgnoreCase(filterProp)) {
            Set<String> keep = Set.of(filterProp.split(","));
            configs.removeIf(c -> !keep.contains(c.getName()));

            if (configs.isEmpty()) {
                for (String name : keep) {
                    try {
                        GraphExecutionMode mode = GraphExecutionMode.valueOf(name);
                        configs.add(BenchmarkConfig.create(name)
                                .executionMode(mode)
                                .maxTokens(100)
                                .minDiversityPct(0));
                        log.info("Dynamically created config '{}' from GraphExecutionMode enum", name);
                    } catch (IllegalArgumentException ignored) { }
                }
            }
            log.info("Filtered to {} configs via vlm.test.configs: {}", configs.size(), keep);
        }

        String maxTokensOverride = System.getProperty("vlm.test.maxTokens");
        if (maxTokensOverride != null && !maxTokensOverride.isEmpty()) {
            int mt = Integer.parseInt(maxTokensOverride);
            configs.forEach(c -> c.maxTokens(mt));
            log.info("Override maxTokens={} for all {} configs", mt, configs.size());
        }

        // GraphOptimizer may replace the imported decoder while constructing the pipeline.
        // Every benchmark operation and assertion must target the decoder that generation actually uses.
        final SameDiff pipelineDecoder = pipeline.getDecoder();
        List<SameDiff> models = List.of(pipelineDecoder, embedTokensSd);

        // Capture for lambdas
        final INDArray finalInputsEmbeds = inputsEmbeds;
        final int[] finalPromptTokenIds = promptTokenIds;
        final GenerationPipeline finalPipeline = pipeline;

        // Compile function
        BenchmarkRunner.CompileFunction compileFn = config -> {
            String configSummary = summarizeConfig(config);
            long compPhaseNs = phaseStart("CONFIG_COMPILE", configSummary);
            try {
                BenchmarkConfigApplier.compileModels(
                        pipelineDecoder, "decoder", embedTokensSd, "embed_tokens", config);
                logDspState("POST_COMPILE " + config.getName(), pipelineDecoder);
                phaseSuccess("CONFIG_COMPILE", compPhaseNs, configSummary);
            } catch (Throwable t) {
                logDspState("COMPILE_FAILURE " + config.getName(), pipelineDecoder);
                throw phaseFailure("CONFIG_COMPILE", configSummary, t);
            }
        };

        // Decode function — benchmark the GenerationPipeline path by default.
        // Optional old-decoder comparison is available via -Dvlm.test.compareOldDecoder=true
        BenchmarkRunner.DecodeFunction decodeFn = config -> {
            String configSummary = summarizeConfig(config);
            long decPhaseNs = phaseStart("CONFIG_DECODE",
                    configSummary + " "
                            + summarizeTokens("promptTokenIds", finalPromptTokenIds) + " "
                            + summarizeTensor("inputsEmbeds", finalInputsEmbeds));

            try {
                logDspState("PRE_DECODE " + config.getName(), pipelineDecoder);
                if ("OPTIMAL".equals(config.getName()) || "SMOLDOC_IDEAL".equals(config.getName())) {
                    int shortBudget = Math.min(8, config.getMaxTokens());
                    GenerationResult shortResult = finalPipeline.generate(
                            finalInputsEmbeds.dup(), finalPromptTokenIds, shortBudget);
                    Object retainedPlan = currentDecoderPlan(pipelineDecoder);
                    long retainedHandle = currentDecoderPlanHandle(pipelineDecoder);
                    int replayCountBeforeLongRun = decoderReplayCount(pipelineDecoder);
                    assertNotNull(retainedPlan, config.getName() + ": short production-envelope run compiled no plan");
                    assertTrue(retainedHandle != 0L, config.getName() + ": short run has no native plan handle");
                    assertFixedKvEnvelope(pipelineDecoder, PRODUCTION_MAX_KV);

                    GenerationResult result = finalPipeline.generate(
                            finalInputsEmbeds.dup(), finalPromptTokenIds, config.getMaxTokens());
                    assertSame(retainedPlan, currentDecoderPlan(pipelineDecoder),
                            config.getName() + ": active token budget replaced the fixed production plan");
                    assertEquals(retainedHandle, currentDecoderPlanHandle(pipelineDecoder),
                            config.getName() + ": native plan handle changed across active token budgets");
                    int prefixLength = Math.min(shortResult.getTokenIds().length, result.getTokenIds().length);
                    assertTrue(result.getTokenIds().length >= shortResult.getTokenIds().length,
                            config.getName() + ": active-budget run terminated before the short-run prefix");
                    assertArrayEquals(Arrays.copyOf(shortResult.getTokenIds(), prefixLength),
                            Arrays.copyOf(result.getTokenIds(), prefixLength),
                            config.getName() + ": fixed-envelope replay changed the deterministic token prefix");
                    assertDecoderReplayed(pipelineDecoder, config.getName(), replayCountBeforeLongRun);
                    log.info("[PRODUCTION_EQUIVALENCE] {} retained one {}-token KV plan across budgets {} -> {}",
                            config.getName(), PRODUCTION_MAX_KV, shortBudget, config.getMaxTokens());
                    maybeCompareAgainstOldDecoder(
                            config, pipelineDecoder, embedTokensSd, tokenizer, hiddenSize,
                            finalInputsEmbeds, finalPromptTokenIds, result);
                    logDspState("POST_DECODE " + config.getName(), pipelineDecoder);
                    dumpActiveDspReport(config.getName());
                    phaseSuccess("CONFIG_DECODE", decPhaseNs, summarizeResult(result));
                    return result;
                }
                GenerationResult result = finalPipeline.generate(finalInputsEmbeds.dup(), finalPromptTokenIds, config.getMaxTokens());
                maybeCompareAgainstOldDecoder(
                        config,
                        pipelineDecoder,
                        embedTokensSd,
                        tokenizer,
                        hiddenSize,
                        finalInputsEmbeds,
                        finalPromptTokenIds,
                        result);
                logDspState("POST_DECODE " + config.getName(), pipelineDecoder);
                dumpActiveDspReport(config.getName());
                phaseSuccess("CONFIG_DECODE", decPhaseNs, summarizeResult(result));
                return result;
            } catch (Throwable t) {
                logDspState("DECODE_FAILURE " + config.getName(), pipelineDecoder);
                try {
                    dumpActiveDspReport(config.getName());
                } catch (Throwable diagnosticsFailure) {
                    t.addSuppressed(diagnosticsFailure);
                }
                throw phaseFailure("CONFIG_DECODE", configSummary, t);
            }
        };

        // Validate function
        BenchmarkRunner.ValidateFunction validateFn = (config, result) -> {
            long valPhaseNs = phaseStart("FINAL_VALIDATE", config.getName() + " " + summarizeResult(result));
            try {
                if (Boolean.parseBoolean(System.getProperty("vlm.test.skipFinalValidate", "false"))) {
                    log.info("[VALIDATE] Skipping final content validation for {}", config.getName());
                } else {
                    validateResult(config, result);
                }
                // DSP structural assertions (enabled via --dsp-assert / -Dvlm.benchmark.dspAssert=true)
                if (isDspAssertEnabled() && config.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT) {
                    DspPlanAssertions.assertPhaseReached(pipelineDecoder, PlanPhase.SHAPES_FROZEN,
                            config.getName() + " benchmark");
                    DspPlanAssertions.assertNoSegmentFailures(pipelineDecoder, config.getName() + " benchmark");
                    DspPlanAssertions.assertNoCaptureFailures(pipelineDecoder, config.getName() + " benchmark");
                    DspPlanAssertions.assertNoFallbacks(pipelineDecoder, config.getName() + " benchmark");
                    log.info("[DSP_ASSERT] {} — all structural assertions passed", config.getName());
                }
                phaseSuccess("FINAL_VALIDATE", valPhaseNs, config.getName());
            } catch (Throwable t) {
                throw phaseFailure("FINAL_VALIDATE",
                        config.getName() + " " + summarizeResult(result), t);
            }
        };

        // Run the matrix
        List<BenchmarkResult> results = BenchmarkRunner.runMatrix(
                configs, List.of("decoder", "embed_tokens"), models,
                compileFn, decodeFn, validateFn, "vlm.config");

        tokenizer.close();
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);

        StringBuilder pipelineInfo = new StringBuilder();
        pipelineInfo.append(String.format("Pipeline setup: download=%dms import=%dms vision=%dms embed=%dms\n\n",
                downloadMs, importMs, visionMs, embedMs));
        log.info("{}", pipelineInfo);
        BenchmarkRunner.printReport(results);
    }

    private static void dumpActiveDspReport(String configName) {
        String configuredPath = System.getProperty("vlm.diag.dspReportFile");
        if (configuredPath == null || configuredPath.isBlank()) {
            return;
        }

        String safeConfigName = configName.replaceAll("[^A-Za-z0-9._-]", "_");
        Path reportPath = Paths.get(configuredPath.replace("{config}", safeConfigName));
        try {
            Path parent = reportPath.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(reportPath, DspDiagnostics.getJsonReport());
            log.info("[DSP_DIAG] Wrote active decoder report to {}", reportPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write active DSP diagnostics to " + reportPath, e);
        }
    }

    private static Object currentDecoderPlan(SameDiff model) {
        var session = model.getOrCreateSession();
        if (session == null || session.getDynamicShapePlanExecutor() == null) return null;
        return session.getDynamicShapePlanExecutor().getCurrentPlan();
    }

    private static long currentDecoderPlanHandle(SameDiff model) {
        var session = model.getOrCreateSession();
        if (session == null || session.getDynamicShapePlanExecutor() == null
                || session.getDynamicShapePlanExecutor().getNativePlanHandle() == null) return 0L;
        return session.getDynamicShapePlanExecutor().getNativePlanHandle().address();
    }

    private static int decoderReplayCount(SameDiff model) {
        DspDebugger.GraphReplayReport report = DspDebugger.attach(model).analyzeGraphReplay();
        return report.errorMessage == null
                ? report.segments.stream().mapToInt(segment -> segment.replayCount).sum() : 0;
    }

    private static void assertDecoderReplayed(SameDiff model, String label, int priorReplayCount) {
        DspDebugger.GraphReplayReport report = DspDebugger.attach(model).analyzeGraphReplay();
        assertNull(report.errorMessage, label + ": replay report failed: " + report.errorMessage);
        assertEquals(PlanPhase.REPLAYING, report.planPhase, label + ": decoder never reached replay");
        assertTrue(report.isFullyReplaying(), label + ": decoder has a non-replaying segment");
        assertEquals(1, report.numSegments,
                label + ": optimal CUDA decoder replay was fragmented into extra launches");
        int replayCount = report.segments.stream().mapToInt(segment -> segment.replayCount).sum();
        assertTrue(replayCount > priorReplayCount,
                label + ": active-budget invocation added no decoder replays");
    }

    private static void assertFixedKvEnvelope(SameDiff model, long expectedCapacity) {
        var session = model.getOrCreateSession();
        assertNotNull(session, "Decoder session is missing");
        var executor = session.getDynamicShapePlanExecutor();
        assertNotNull(executor, "Decoder DSP executor is missing");
        var plan = executor.getCurrentPlan();
        assertNotNull(plan, "Decoder DSP plan is missing");
        String[] keys = plan.getExternalInputKeys();
        INDArray[] arrays = executor.getExternalInputsSnapshot();
        int kvInputs = 0;
        for (int i = 0; i < keys.length; i++) {
            if (!keys[i].startsWith("past_key_values.")) continue;
            kvInputs++;
            assertNotNull(arrays[i], keys[i]);
            assertEquals(expectedCapacity, arrays[i].size(2),
                    keys[i] + " did not keep the production KV envelope");
        }
        assertEquals(60, kvInputs, "Expected 30 key/value cache pairs in the decoder plan");
    }

    private void maybeCompareAgainstOldDecoder(BenchmarkConfig config,
                                               SameDiff decoder,
                                               SameDiff embedTokensSd,
                                               Tokenizer tokenizer,
                                               long hiddenSize,
                                               INDArray inputsEmbeds,
                                               int[] promptTokenIds,
                                               GenerationResult pipelineResult) throws Exception {
        if (!Boolean.parseBoolean(System.getProperty("vlm.test.compareOldDecoder", "false"))) {
            return;
        }

        ModelIOConfig decoderIOConfig = ModelIOConfig.discover(decoder);
        String specTokensProp = System.getProperty("vlm.speculative.tokens", "0");
        int specTokens = (specTokensProp == null || specTokensProp.isEmpty()) ? 0 : Integer.parseInt(specTokensProp);
        boolean useDraft = config.isUseDraftModel()
                || "true".equalsIgnoreCase(System.getProperty("vlm.speculative.draft"));
        if (useDraft && specTokens == 0) {
            specTokens = config.getDraftModelK() > 0 ? config.getDraftModelK() : 5;
        }

        StaticKvCacheDecodeLoop.StaticKvCacheDecodeLoopBuilder loopBuilder = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokensSd)
                .tokenizer(tokenizer)
                .ioConfig(decoderIOConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(config.getMaxTokens())
                .maxSpeculativeTokens(specTokens)
                .hiddenSize(hiddenSize);
        if (Boolean.parseBoolean(System.getProperty("vlm.benchmark.argmaxTrace", "false"))) {
            loopBuilder.argmaxTraceEnabled(true);
            int topK = Integer.getInteger("vlm.benchmark.argmaxTraceTopK", 5);
            loopBuilder.argmaxTraceTopK(topK);
        }
        int[] referenceTokens = loadReferenceTokenStream();
        if (referenceTokens != null) {
            loopBuilder.referenceTokenStream(referenceTokens);
            log.info("[{}] Reference token stream loaded: {} tokens", config.getName(), referenceTokens.length);
        }

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokensSd);
        GenerationResult oldResult = loopBuilder.build().decode(inputsEmbeds.dup(), promptTokenIds);

        int[] oldTokens = oldResult.getTokenIds();
        int[] newTokens = pipelineResult.getTokenIds();
        int minLen = Math.min(oldTokens.length, newTokens.length);
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (oldTokens[i] != newTokens[i]) {
                firstDivergent = i;
                break;
            }
        }

        log.info("[{}] Old/new comparison: oldLen={} newLen={} firstDivergent={} old='{}' new='{}'",
                config.getName(),
                oldTokens.length,
                newTokens.length,
                firstDivergent,
                oldResult.getText(),
                pipelineResult.getText());

        assertArrayEquals(oldTokens, newTokens,
                config.getName() + ": GenerationPipeline diverged from StaticKvCacheDecodeLoop"
                        + " firstDivergent=" + firstDivergent);
    }

    // ─── Dual-decode test: GenerationPipeline vs StaticKvCacheDecodeLoop ───

    @Test
    @DisplayName("GenerationPipeline (native decode) vs StaticKvCacheDecodeLoop (old) — token parity on mythic PDF")
    public void testGenerationPipelineVsOldDecoder() throws Exception {
        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 30);

        // ── Phase 1: Download models ──
        var visionDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerDl = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDl.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionDl.getModelFile().getAbsolutePath(),
                decoderDl.getModelFile().getAbsolutePath(),
                embedTokensDl.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoderSd = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokensSd = models[2];

        // ── Phase 2: Image preprocessing ──
        int targetSize = 512;
        BufferedImage pdfImage = loadImageFromPdfOrGenerate();
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, -1);

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

        // ── Phase 3: Vision encoding ──
        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(targetSize)
                .build();
        VisionEncoder.Result visionResult = visionEncoder.encode(imageInput, splitResult.getTotalFrames(), splitResult);
        INDArray visionEmbeddings = visionResult.getEmbeddings();
        imageInput.close();
        visionEncoder.freeModelMemory();

        // ── Phase 4: Build GenerationPipeline and merge embeddings ──
        GenerationPipeline pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokensSd)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(maxTokens)
                        .maxPrefillLength(PRODUCTION_MAX_PREFILL)
                        .maxKvCacheLength(PRODUCTION_MAX_KV)
                        .benchmarkConfig(BenchmarkConfig.optimal())
                        .build());

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray textEmbeddings = pipeline.embedTokens(promptTokenIds);
        long hiddenSize = visionEmbeddings.shape()[2];
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        textEmbeddings.close();

        log.info("Dual-decode test: maxTokens={}, promptTokens={}, hiddenSize={}",
                maxTokens, promptTokenIds.length, hiddenSize);

        // ── Phase 5: Run OLD decoder (StaticKvCacheDecodeLoop) ──
        ModelIOConfig decoderIOConfig = ModelIOConfig.discover(decoder);
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokensSd);

        StaticKvCacheDecodeLoop oldLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokensSd)
                .tokenizer(tokenizer)
                .ioConfig(decoderIOConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult oldResult = oldLoop.decode(inputsEmbeds.dup(), promptTokenIds);
        int[] oldTokens = oldResult.getTokenIds();
        String oldText = oldResult.getText();
        log.info("OLD decoder: {} tokens, text='{}'", oldTokens.length, oldText);

        // ── Phase 6: Run NEW pipeline (GenerationPipeline) ──
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokensSd);

        GenerationResult newResult = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
        int[] newTokens = newResult.getTokenIds();
        String newText = newResult.getText();
        log.info("NEW pipeline: {} tokens, text='{}'", newTokens.length, newText);

        // ── Phase 7: Token-by-token comparison ──
        int minLen = Math.min(oldTokens.length, newTokens.length);
        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (oldTokens[i] == newTokens[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }
        double matchRate = minLen > 0 ? (double) matches / minLen : 0.0;
        log.info("Token match rate: {}/{} ({}%) firstDivergent={}",
                matches, minLen, String.format("%.1f", matchRate * 100), firstDivergent);

        if (firstDivergent >= 0) {
            log.error("FIRST DIVERGENCE at step {}: old={} ('{}') new={} ('{}')",
                    firstDivergent,
                    oldTokens[firstDivergent],
                    tokenizer.decode(new int[]{oldTokens[firstDivergent]}, false),
                    newTokens[firstDivergent],
                    tokenizer.decode(new int[]{newTokens[firstDivergent]}, false));
        }

        // ── Phase 8: Content validation ──
        // Both should produce structural DocTags
        assertTrue(oldText.contains("<") && oldText.contains(">"),
                "Old decoder should produce structural tags. Text: " + oldText);
        assertTrue(newText.contains("<") && newText.contains(">"),
                "GenerationPipeline should produce structural tags. Text: " + newText);

        // Both should contain mythic-related content (paragraphs, not just titles)
        String lowerOld = oldText.toLowerCase();
        String lowerNew = newText.toLowerCase();
        boolean oldHasMythic = lowerOld.contains("mythic") || lowerOld.contains("hero")
                || lowerOld.contains("creating a mythic") || lowerOld.contains("path");
        boolean newHasMythic = lowerNew.contains("mythic") || lowerNew.contains("hero")
                || lowerNew.contains("creating a mythic") || lowerNew.contains("path");

        log.info("Old has mythic content: {}", oldHasMythic);
        log.info("New has mythic content: {}", newHasMythic);

        // The critical assertion: tokens MUST match
        assertArrayEquals(oldTokens, newTokens,
                "GenerationPipeline diverges from StaticKvCacheDecodeLoop. "
                        + "Old text: " + oldText + " New text: " + newText);

        // Soft check: if old decoder has mythic content, new should too
        if (oldHasMythic) {
            assertTrue(newHasMythic,
                    "Old decoder has mythic content but GenerationPipeline does not. "
                            + "New text: " + newText);
        }

        tokenizer.close();
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
    }

    // ─── validateResult ────────────────────────────────────────────────────

    private double effectiveThroughput(GenerationResult result) {
        if (result.getLateSteadyStateTokensPerSecond() > 0) {
            return result.getLateSteadyStateTokensPerSecond();
        }
        if (result.getSteadyStateTokensPerSecond() > 0) {
            return result.getSteadyStateTokensPerSecond();
        }
        if (result.getDecodeTokensPerSecond() > 0) {
            return result.getDecodeTokensPerSecond();
        }
        return result.getTokensPerSecond();
    }

    private String effectiveThroughputLabel(GenerationResult result) {
        if (result.getLateSteadyStateTokensPerSecond() > 0) {
            return "late steady-state";
        }
        if (result.getSteadyStateTokensPerSecond() > 0) {
            return "steady-state";
        }
        if (result.getDecodeTokensPerSecond() > 0) {
            return "decode-only";
        }
        return "overall";
    }

    private void validateResult(BenchmarkConfig config, GenerationResult result) {
        String name = config.getName();

        assertNotNull(result.getText(), name + ": generated text is null");
        // Log full generated text for mythic content analysis
        log.info("[FULL_TEXT] {} tokens: {}", result.getGeneratedTokenCount(), result.getText());
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

        if (config.isExpectStructuralTags() && result.getGeneratedTokenCount() >= 10) {
            boolean hasDocTags = trimmed.contains("<") && trimmed.contains(">");
            assertTrue(hasDocTags,
                    name + ": expected structural DocTags in the generated text, but found none. Text: "
                            + trimmed.substring(0, Math.min(200, trimmed.length())));

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

        if (result.getGeneratedTokenCount() >= 10) {
            int[] tokenIds = result.getTokenIds();
            Set<Integer> uniqueTokens = new HashSet<>();
            for (int id : tokenIds) uniqueTokens.add(id);
            double uniqueRatio = (double) uniqueTokens.size() / tokenIds.length;
            log.info("  Token diversity: {}/{} unique ({}%)",
                    uniqueTokens.size(), tokenIds.length, String.format("%.1f", uniqueRatio * 100));
            if (config.getMinDiversityPct() > 0) {
                assertTrue(uniqueRatio > config.getMinDiversityPct() / 100.0,
                        name + ": degenerate output: " + uniqueTokens.size() + "/" + tokenIds.length +
                                " unique (min " + config.getMinDiversityPct() + "%)");
            }
        }

        String[] expectedSubstrings = config.getExpectedSubstrings();
        if (expectedSubstrings != null && expectedSubstrings.length > 0) {
            String lowerText = result.getText().toLowerCase(Locale.ROOT);
            List<String> missing = new ArrayList<>();
            for (String expected : expectedSubstrings) {
                if (expected == null || expected.isBlank()) continue;
                if (!lowerText.contains(expected.toLowerCase(Locale.ROOT))) {
                    missing.add(expected);
                }
            }
            assertTrue(missing.isEmpty(),
                    name + ": generated text is not the expected coherent mythic-heroes passage. Missing "
                            + missing + ". Text: " + trimmed.substring(0, Math.min(500, trimmed.length())));
        }

        if (result.getGeneratedTokenCount() >= 5) {
            assertTrue(result.getTokensPerSecond() > 0.1,
                    name + ": throughput too low: " +
                            String.format("%.2f", result.getTokensPerSecond()) + " tok/s");
        }
        if (("OPTIMAL".equals(name) || "SMOLDOC_IDEAL".equals(name)) && result.getGeneratedTokenCount() >= 20) {
            double effectiveThroughput = effectiveThroughput(result);
            String throughputLabel = effectiveThroughputLabel(result);
            // Minimum throughput: 55 tok/s late steady-state. Current baseline ~59-65 tok/s.
            // Target: 70+ tok/s. Will raise this floor as performance improves.
            assertTrue(effectiveThroughput >= 55.0,
                    name + ": native benchmark target missed: "
                            + throughputLabel + "=" + String.format("%.2f", effectiveThroughput)
                            + " tok/s (target 55.00 tok/s)");
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

    // ─── Utility helpers ──────────────────────────────────────────────────

    private boolean isPhaseLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.phaseLogging", "true"));
    }

    private boolean isTensorFingerprintLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.tensorFingerprints", "false"));
    }

    private boolean isDspStateLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.dspStateLogging", "true"));
    }

    private boolean isDspAssertEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.dspAssert", "false"));
    }

    private int tensorFingerprintSamples() {
        return Integer.getInteger("vlm.benchmark.tensorSampleValues", 8);
    }

    private long phaseStart(String phase, String detail) {
        if (isPhaseLoggingEnabled()) {
            log.info("[PHASE] START {} {}", phase, detail);
        }
        return System.nanoTime();
    }

    private void phaseSuccess(String phase, long startNs, String detail) {
        if (isPhaseLoggingEnabled()) {
            long elapsedMs = (System.nanoTime() - startNs) / 1_000_000;
            log.info("[PHASE] OK {} {}ms {}", phase, elapsedMs, detail);
        }
    }

    private IllegalStateException phaseFailure(String phase, String detail, Throwable cause) {
        log.error("[PHASE] FAIL {} {}: {}", phase, detail, cause.getMessage(), cause);
        return new IllegalStateException("Benchmark phase " + phase + " failed: " + detail, cause);
    }

    private String benchmarkInputSummary() {
        return "pdf=" + (pdfPath != null && !pdfPath.isEmpty() ? pdfPath : "<generated>")
                + " page=" + (specificPage >= 0 ? specificPage : 0)
                + " dpi=" + renderDpi;
    }

    private String summarizeConfig(BenchmarkConfig config) {
        return "config=" + config.getName()
                + " maxTokens=" + config.getMaxTokens()
                + " triton=" + config.isTriton()
                + " compileAll=" + config.isTritonCompileAll()
                + " graphCapture=" + config.isTritonGraphCapture()
                + " noFallbackCapture=" + !config.isTritonAllowFallbackCapture()
                + " batchedGemm=" + config.isDspBatchedGemm();
    }

    private String summarizeTensor(String label, INDArray arr) {
        if (arr == null) return label + "{null}";

        StringBuilder sb = new StringBuilder(label).append("{shape=");
        try {
            sb.append(Arrays.toString(arr.shape()))
                    .append(",dtype=").append(arr.dataType())
                    .append(",length=").append(arr.length())
                    .append(",closed=").append(arr.wasClosed());

            if (isTensorFingerprintLoggingEnabled() && !arr.wasClosed() && arr.length() > 0) {
                INDArray flat = arr.reshape(arr.length());
                long len = flat.length();
                long stride = Math.max(1L, len / Math.max(1, tensorFingerprintSamples()));
                int sampled = 0;
                double sampleMin = Double.POSITIVE_INFINITY;
                double sampleMax = Double.NEGATIVE_INFINITY;
                double sampleSum = 0.0;
                double checksum = 0.0;
                boolean sampleHasNaN = false;
                for (long idx = 0; idx < len && sampled < tensorFingerprintSamples(); idx += stride) {
                    double value = flat.getDouble(idx);
                    sampleMin = Math.min(sampleMin, value);
                    sampleMax = Math.max(sampleMax, value);
                    sampleSum += value;
                    checksum += value * (idx + 1);
                    sampleHasNaN |= Double.isNaN(value);
                    sampled++;
                }
                if (sampled == 0) {
                    double value = flat.getDouble(0);
                    sampleMin = value;
                    sampleMax = value;
                    sampleSum = value;
                    checksum = value;
                    sampleHasNaN = Double.isNaN(value);
                    sampled = 1;
                }
                sb.append(",sampled=").append(sampled)
                        .append(",stride=").append(stride)
                        .append(",sampleMin=").append(String.format("%.6f", sampleMin))
                        .append(",sampleMax=").append(String.format("%.6f", sampleMax))
                        .append(",sampleMean=").append(String.format("%.6f", sampleSum / sampled))
                        .append(",sampleChecksum=").append(String.format("%.6f", checksum))
                        .append(",sampleHasNaN=").append(sampleHasNaN);
            }
        } catch (Throwable t) {
            sb.append("?,fingerprintError=").append(t.getClass().getSimpleName())
                    .append(":").append(t.getMessage());
        }
        return sb.append("}").toString();
    }

    private String summarizeTokens(String label, int[] tokens) {
        if (tokens == null) return label + "{null}";
        int preview = Math.min(tokens.length, 8);
        int tailStart = Math.max(0, tokens.length - 8);
        return label + "{count=" + tokens.length
                + ",head=" + Arrays.toString(Arrays.copyOfRange(tokens, 0, preview))
                + ",tail=" + Arrays.toString(Arrays.copyOfRange(tokens, tailStart, tokens.length))
                + "}";
    }

    private String summarizeResult(GenerationResult result) {
        if (result == null) return "result{null}";
        return "result{tokens=" + result.getGeneratedTokenCount()
                + ",finish=" + result.getFinishReason()
                + ",throughputLabel=" + effectiveThroughputLabel(result)
                + ",throughput=" + String.format("%.2f", effectiveThroughput(result))
                + ",text='" + safeSnippet(result.getText(), 160) + "'"
                + "," + summarizeTokens("tokenIds", result.getTokenIds())
                + "}";
    }

    private String safeSnippet(String text, int maxChars) {
        if (text == null) return "<null>";
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        if (normalized.length() <= maxChars) return normalized;
        return normalized.substring(0, maxChars) + "...";
    }

    private String safeFileName(File file) {
        return file == null ? "<null>" : file.getName();
    }

    private void logDspState(String phase, SameDiff model) {
        if (!isDspStateLoggingEnabled() || model == null) return;
        try {
            DspHandle handle = model.dsp();
            DspHandle.StepSnapshot snapshot = handle.captureStepSnapshot();
            DspDebugger debugger = handle.debugger();
            DspDebugger.PlanReport planReport = debugger.analyzePlan();
            DspDebugger.GraphReplayReport replayReport = debugger.analyzeGraphReplay();

            log.info("[DSP] {} snapshot={} replayedSegments={} slotBySlotSegments={} lastExec={{warmup={}, captured={}, replayed={}, slotBySlot={}, failed={}, total={}, syncLevel={}, streamSyncs={}}}",
                    phase, snapshot, snapshot.replayedSegments(), snapshot.slotBySlotSegments(),
                    handle.lastExecSegmentsWarmup(), handle.lastExecSegmentsCaptured(),
                    handle.lastExecSegmentsReplayed(), handle.lastExecSegmentsSlotBySlot(),
                    handle.lastExecSegmentsFailed(), handle.lastExecSegmentsTotal(),
                    handle.lastExecSyncLevel(), handle.lastExecStreamSyncCount());

            if (planReport.errorMessage != null || replayReport.errorMessage != null) {
                log.info("[DSP] {} plan={} replay={}", phase, planReport.errorMessage, replayReport.errorMessage);
                return;
            }

            List<DspDebugger.SlotInfo> unfrozenOps = planReport.getUnfrozenOps();
            log.info("[DSP] {} planPhase={} pointersStable={} fullyReplaying={} frozenExec={} segments={} replaying={} captureFailures={} stuck={} riskyOps={} unfrozenOps={}",
                    phase,
                    replayReport.planPhase,
                    replayReport.pointersStable,
                    replayReport.isFullyReplaying(),
                    replayReport.frozenExecutionCount,
                    replayReport.numSegments,
                    replayReport.getReplayingSegments().size(),
                    replayReport.getCaptureFailures().size(),
                    replayReport.getStuckSegments().size(),
                    planReport.getRiskyOps().size(),
                    unfrozenOps.size());

            // Log unfrozen ops breakdown by type for performance analysis
            if (!unfrozenOps.isEmpty()) {
                java.util.Map<String, Integer> opTypeCounts = new java.util.TreeMap<>();
                for (DspDebugger.SlotInfo s : unfrozenOps) {
                    opTypeCounts.merge(s.opName, 1, Integer::sum);
                }
                StringBuilder sb = new StringBuilder("[DSP] ").append(phase).append(" UNFROZEN_OPS breakdown: ");
                opTypeCounts.forEach((name, count) -> sb.append(name).append("=").append(count).append(" "));
                log.info(sb.toString().trim());

                // Log first few with details (slot index, flags, state)
                int logged = 0;
                for (DspDebugger.SlotInfo s : unfrozenOps) {
                    if (logged++ >= 20) {
                        log.info("[DSP]   ... and {} more unfrozen ops", unfrozenOps.size() - 20);
                        break;
                    }
                    log.info("[DSP]   slot[{}] op={} state={} flags={} viewCapable={} dataDependent={} shapeDependsOnValues={}",
                            s.index, s.opName, s.state, s.flags,
                            s.isViewCapable(), s.isDataDependent(), s.isShapeDependsOnValues());
                }
            }
        } catch (Throwable t) {
            log.warn("[DSP] {} state unavailable: {}", phase, t.getMessage());
        }
    }

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

    private static int[] loadReferenceTokenStream() {
        return ReferenceTokenStream.loadFromSystemProperty("vlm.benchmark.referenceTokens");
    }
}
