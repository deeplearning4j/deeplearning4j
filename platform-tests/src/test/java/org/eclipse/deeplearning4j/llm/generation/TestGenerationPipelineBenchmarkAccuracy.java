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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Benchmark-equivalent page-10 accuracy tests for GenerationPipeline.
 *
 * <p>These tests intentionally mirror the page-10 setup used by
 * {@code run-benchmark.sh} / {@code TestSmolDoclingOptimizedPipeline}, but keep
 * the assertions narrowly focused on correctness rather than throughput.</p>
 */
@Slf4j
public class TestGenerationPipelineBenchmarkAccuracy {

    private static final int TARGET_SIZE = 512;
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    private static synchronized void ensureBenchmarkInputsLoaded() throws Exception {
        if (loaded) {
            return;
        }

        log.info("Loading benchmark-equivalent page-10 inputs for GenerationPipeline accuracy tests...");

        VLMModelDownloader.DownloadResult visionDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        VLMModelDownloader.DownloadResult decoderDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        VLMModelDownloader.DownloadResult embedTokensDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        VLMModelDownloader.DownloadResult tokenizerDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDl.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionDl.getModelFile().getAbsolutePath(),
                decoderDl.getModelFile().getAbsolutePath(),
                embedTokensDl.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoderSd = models[0];
        decoder = models[1];
        embedTokens = models[2];

        BufferedImage pdfImage = loadBenchmarkPageImage();
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, 9);

        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(9)
                .build();
        VisionEncoder.Result visionResult =
                visionEncoder.encode(imageInput, splitResult.getTotalFrames(), splitResult);
        INDArray visionEmbeddings = visionResult.getEmbeddings();
        imageInput.close();
        visionEncoder.freeModelMemory();

        GenerationPipeline embedPipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(1)
                        .build());
        try {
            int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
            int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
            String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                    splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
            String chatPrompt = "<|im_start|>User:" + imagePrompt
                    + "Convert this page to docling.<end_of_utterance>\nAssistant:";
            promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

            INDArray textEmbeddings = embedPipeline.embedTokens(promptTokenIds);
            hiddenSize = visionEmbeddings.shape()[2];
            inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
            textEmbeddings.close();
        } finally {
            embedPipeline.close();
            visionEmbeddings.close();
        }

        loaded = true;
        log.info("Page-10 benchmark inputs ready: promptTokens={} inputsEmbedsShape={} hiddenSize={}",
                promptTokenIds.length, Arrays.toString(inputsEmbeds.shape()), hiddenSize);
    }

    @Test
    @DisplayName("Benchmark-equivalent GenerationPipeline accuracy on mythic PDF page 10 under OPTIMAL")
    public void testPage10OptimalGenerationPipeline() throws Exception {
        ensureBenchmarkInputsLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = optimalConfig(maxTokens);

        GenerationResult oldResult = runOldDecoder(config, maxTokens);
        GenerationResult newResult = runGenerationPipeline(config, maxTokens);

        logResult("StaticKvCacheDecodeLoop/OPTIMAL", oldResult);
        logResult("GenerationPipeline/OPTIMAL", newResult);

        int[] oldTokens = oldResult.getTokenIds();
        int[] newTokens = newResult.getTokenIds();
        assertArrayEquals(oldTokens, newTokens,
                "GenerationPipeline should match the old decoder token-for-token on page 10 under OPTIMAL");
        assertEquals(oldResult.getText(), newResult.getText(),
                "GenerationPipeline decoded text should match the old decoder on page 10 under OPTIMAL");
    }

    @Test
    @DisplayName("Benchmark-equivalent OPTIMAL parity: old decoder vs GenerationPipeline on mythic PDF page 10")
    public void testPage10OptimalOldVsNewParity() throws Exception {
        ensureBenchmarkInputsLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = optimalConfig(maxTokens);

        GenerationResult oldResult = runOldDecoder(config, maxTokens);
        GenerationResult newResult = runGenerationPipeline(config, maxTokens);

        logResult("StaticKvCacheDecodeLoop/OPTIMAL", oldResult);
        logResult("GenerationPipeline/OPTIMAL", newResult);
        logParity(oldResult, newResult);

        int[] oldTokens = oldResult.getTokenIds();
        int[] newTokens = newResult.getTokenIds();
        int minLen = Math.min(oldTokens.length, newTokens.length);
        assertArrayEquals(oldTokens, Arrays.copyOf(newTokens, minLen),
                "GenerationPipeline diverges from StaticKvCacheDecodeLoop on page 10 under OPTIMAL within the shared prefix");

        if (newTokens.length > oldTokens.length) {
            log.info("Known bug: GenerationPipeline continued {} tokens past old-decoder stop point. "
                    + "oldText='{}' newText='{}'",
                    newTokens.length - oldTokens.length,
                    safeSnippet(oldResult.getText(), 180),
                    safeSnippet(newResult.getText(), 180));
            assertTrue(newResult.getText().startsWith(oldResult.getText()),
                    "GenerationPipeline should contain the same collapse prefix before drifting. old='"
                            + safeSnippet(oldResult.getText(), 180) + "' new='"
                            + safeSnippet(newResult.getText(), 180) + "'");
        }
    }

    @Test
    @DisplayName("GenerationPipeline page-10 config bisection: all modes match old decoder")
    public void testPage10GenerationPipelineConfigBisection() throws Exception {
        ensureBenchmarkInputsLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 60);

        Map<String, BenchmarkConfig> configs = new LinkedHashMap<>();
        configs.put("SLOT_BY_SLOT", BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens)
                .minDiversityPct(0));
        configs.put("CUDA_GRAPHS", BenchmarkConfig.create("CUDA_GRAPHS")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .maxTokens(maxTokens)
                .minDiversityPct(0));
        configs.put("OPTIMAL", optimalConfig(maxTokens));

        List<String> failures = new ArrayList<>();
        for (Map.Entry<String, BenchmarkConfig> entry : configs.entrySet()) {
            String name = entry.getKey();
            BenchmarkConfig config = entry.getValue();

            GenerationResult oldResult = runOldDecoder(config, maxTokens);
            GenerationResult newResult = runGenerationPipeline(config, maxTokens);

            logResult("StaticKvCacheDecodeLoop/" + name, oldResult);
            logResult("GenerationPipeline/" + name, newResult);

            int[] oldTokens = oldResult.getTokenIds();
            int[] newTokens = newResult.getTokenIds();

            assertArrayEquals(oldTokens, newTokens,
                    "GenerationPipeline should match the old decoder token-for-token under " + name);
            assertEquals(oldResult.getText(), newResult.getText(),
                    "GenerationPipeline text should match the old decoder under " + name);
        }
    }

    private static BenchmarkConfig optimalConfig(int maxTokens) {
        return BenchmarkConfig.optimal()
                .maxTokens(maxTokens)
                .minDiversityPct(0);
    }

    private static BenchmarkConfig tritonNoGcConfig(int maxTokens) {
        return BenchmarkConfig.create("TRITON_NO_GC")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .tritonFusionScoring(false)
                .tritonNumWarps(4)
                .tritonNumStages(1)
                .cublasTf32(true)
                .tritonTf32(true)
                .dspBatchedGemm(true)
                .maxTokens(maxTokens)
                .minDiversityPct(0);
    }

    private GenerationResult runGenerationPipeline(BenchmarkConfig config, int maxTokens) throws Exception {
        compileFor(config);
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        GenerationPipeline pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .ioConfig(ioConfig)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(maxTokens)
                        .hiddenSize(hiddenSize)
                        .build());
        try {
            logDspState("PRE GenerationPipeline.generate");
            GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
            logDspState("POST GenerationPipeline.generate");
            return result;
        } finally {
            pipeline.close();
        }
    }

    private GenerationResult runOldDecoder(BenchmarkConfig config, int maxTokens) throws Exception {
        compileFor(config);
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();
        logDspState("PRE StaticKvCacheDecodeLoop.decode");
        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
        logDspState("POST StaticKvCacheDecodeLoop.decode");
        return result;
    }

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static BufferedImage loadBenchmarkPageImage() throws IOException {
        String configuredPath = System.getProperty("vlm.test.pdf.path");
        File pdfFile;
        if (configuredPath != null && !configuredPath.isBlank()) {
            pdfFile = new File(configuredPath);
        } else {
            pdfFile = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
            if (!pdfFile.exists()) {
                pdfFile = new File("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
            }
        }

        assertTrue(pdfFile.exists(), "Benchmark PDF must exist: " + pdfFile.getAbsolutePath());
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        int renderDpi = Integer.getInteger("vlm.test.pdf.dpi", 150);

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
            log.info("Loaded benchmark PDF page {} at {} DPI: {}x{} from {}",
                    pdfPage, renderDpi, image.getWidth(), image.getHeight(), pdfFile.getAbsolutePath());
            return image;
        }
    }

    private static void assertMythicParagraphOutput(String label, GenerationResult result, int maxTokens) {
        String text = result.getText();
        assertNotNull(text, label + ": generated text is null");

        String normalized = text.trim();
        String lower = normalized.toLowerCase();
        int minUsefulTokens = Math.min(maxTokens, 50);

        assertTrue(result.getGeneratedTokenCount() >= minUsefulTokens,
                label + ": generated too few tokens for page-10 paragraph content: "
                        + result.getGeneratedTokenCount());
        assertTrue(normalized.contains("<") && normalized.contains(">"),
                label + ": expected structural tags in output. Text: " + safeSnippet(normalized, 220));
        assertTrue(normalized.contains("<text>") || normalized.contains("<page>") || normalized.contains("<section_header"),
                label + ": expected text/page structural tags, not just generic markup. Text: "
                        + safeSnippet(normalized, 220));
        assertFalse(normalized.startsWith("<doctag><picture>"),
                label + ": collapsed to picture-only output. Text: " + safeSnippet(normalized, 220));
        assertFalse(lower.contains("<end_of_utterance>user:"),
                label + ": fell back into chat transcript instead of docling text. Text: "
                        + safeSnippet(normalized, 220));

        boolean hasMythicPassage = lower.contains("mythic heroes")
                || lower.contains("hytic heroes")
                || (lower.contains("creating a mythic character") && lower.contains("<text>"));
        assertTrue(hasMythicPassage,
                label + ": expected page-10 mythic passage, not title-only output. Text: "
                        + safeSnippet(normalized, 300));
    }

    private static void logResult(String label, GenerationResult result) {
        log.info("{}: generated={} finish={} tps={} text='{}'",
                label,
                result.getGeneratedTokenCount(),
                result.getFinishReason(),
                String.format("%.2f", result.getTokensPerSecond()),
                safeSnippet(result.getText(), 220));
        log.info("{} tokens={}", label, Arrays.toString(result.getTokenIds()));
    }

    private static void logParity(GenerationResult oldResult, GenerationResult newResult) {
        int[] oldTokens = oldResult.getTokenIds();
        int[] newTokens = newResult.getTokenIds();
        int firstDivergent = findFirstDivergentToken(oldTokens, newTokens);
        if (firstDivergent < 0 && oldTokens.length == newTokens.length) {
            log.info("Old/new parity: exact token match ({} tokens)", oldTokens.length);
            return;
        }

        log.error("Old/new parity diverged: oldLen={} newLen={} firstDivergent={} oldText='{}' newText='{}'",
                oldTokens.length,
                newTokens.length,
                firstDivergent,
                safeSnippet(oldResult.getText(), 180),
                safeSnippet(newResult.getText(), 180));
        if (firstDivergent >= 0 && firstDivergent < oldTokens.length && firstDivergent < newTokens.length) {
            log.error("Divergent tokens: old={} new={}", oldTokens[firstDivergent], newTokens[firstDivergent]);
        }
        if (newTokens.length > oldTokens.length) {
            log.error("GenerationPipeline continued {} tokens past old-decoder stop point",
                    newTokens.length - oldTokens.length);
            if (oldTokens.length > 0) {
                log.error("Old-decoder stop prefix: {}", Arrays.toString(
                        Arrays.copyOf(oldTokens, Math.min(oldTokens.length, 30))));
            }
        }
    }

    private static int findFirstDivergentToken(int[] left, int[] right) {
        int minLen = Math.min(left.length, right.length);
        for (int i = 0; i < minLen; i++) {
            if (left[i] != right[i]) {
                return i;
            }
        }
        return left.length == right.length ? -1 : minLen;
    }

    private static void logDspState(String phase) {
        try {
            DspDebugger debugger = DspDebugger.attach(decoder);
            DspDebugger.GraphReplayReport replay = debugger.analyzeGraphReplay();
            DspDebugger.PlanReport plan = debugger.analyzePlan();
            if (replay.errorMessage != null || plan.errorMessage != null) {
                log.info("[DSP] {} plan={} replay={}", phase, plan.errorMessage, replay.errorMessage);
                return;
            }

            log.info("[DSP] {} planPhase={} pointersStable={} fullyReplaying={} replayingSegments={} captureFailures={} frozenExec={}",
                    phase,
                    replay.planPhase,
                    replay.pointersStable,
                    replay.isFullyReplaying(),
                    replay.getReplayingSegments().size(),
                    replay.getCaptureFailures().size(),
                    replay.frozenExecutionCount);
        } catch (Throwable t) {
            log.info("[DSP] {} unavailable: {}", phase, t.getMessage());
        }
    }

    private static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "<null>";
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, maxChars) + "...";
    }
}
