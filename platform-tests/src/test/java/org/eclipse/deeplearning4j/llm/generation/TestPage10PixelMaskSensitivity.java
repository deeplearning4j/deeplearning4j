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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
public class TestPage10PixelMaskSensitivity {

    private static final int TARGET_SIZE = 512;

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static ImageTiler.SplitImageResult baseSplitResult;
    private static INDArray imageInput;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 packed prompt: pixel-mask inflation isolates residual OCR loss after tiler/prompt fixes")
    public void testPage10PackedPromptPixelMaskSensitivity() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        Map<String, ImageTiler.SplitImageResult> variants = new LinkedHashMap<>();
        variants.put("BASE", cloneSplitResult(baseSplitResult, 0, false));
        variants.put("EXPAND_8", cloneSplitResult(baseSplitResult, 8, false));
        variants.put("EXPAND_16", cloneSplitResult(baseSplitResult, 16, false));
        variants.put("FULL_MASK", cloneSplitResult(baseSplitResult, 0, true));

        Map<String, INDArray> encodedVision = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, ImageTiler.SplitImageResult> entry : variants.entrySet()) {
                encodedVision.put(entry.getKey(), encodeVision(entry.getValue()));
            }

            compileFor(config);

            Map<String, GenerationResult> results = new LinkedHashMap<>();
            for (Map.Entry<String, ImageTiler.SplitImageResult> entry : variants.entrySet()) {
                String name = entry.getKey();
                INDArray visionEmbeddings = encodedVision.get(name);
                GenerationResult result = runGeneration(config, entry.getValue(), visionEmbeddings, maxTokens);
                results.put(name, result);
                log.info("{} text='{}'", name, safeSnippet(result.getText(), 260));
            }

            GenerationResult base = results.get("BASE");
            assertTrue(base != null && normalize(base.getText()).contains("heroes are set apart"),
                    "Baseline packed-prompt output no longer contains the expected paragraph fragment. "
                            + summarize(results));

            boolean anyVariantChanged = false;
            for (Map.Entry<String, GenerationResult> entry : results.entrySet()) {
                if (!"BASE".equals(entry.getKey())
                        && !Arrays.equals(base.getTokenIds(), entry.getValue().getTokenIds())) {
                    anyVariantChanged = true;
                    break;
                }
            }
            assertTrue(anyVariantChanged,
                    "Mask variants did not change the token stream at all. " + summarize(results));

            boolean anyImprovedKeyword = results.values().stream()
                    .map(GenerationResult::getText)
                    .map(TestPage10PixelMaskSensitivity::normalize)
                    .anyMatch(text -> text.contains("mythic heroes")
                            || text.contains("hytic heroes")
                            || text.contains("heroes are set apart"));
            assertTrue(anyImprovedKeyword,
                    "No mask variant preserved the target paragraph fragment. " + summarize(results));

            GenerationResult fullMask = results.get("FULL_MASK");
            assertFalse(fullMask != null && Arrays.equals(base.getTokenIds(), fullMask.getTokenIds()),
                    "FULL_MASK matched baseline exactly; mask geometry may not be part of the remaining OCR gap.");
        } finally {
            for (INDArray arr : encodedVision.values()) {
                if (arr != null && !arr.wasClosed()) {
                    arr.close();
                }
            }
        }
    }

    private static synchronized void ensureLoaded() throws Exception {
        if (loaded) {
            return;
        }

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
        visionEncoderSd = models[0];
        decoder = models[1];
        embedTokens = models[2];

        BufferedImage pdfImage = loadBenchmarkPageImage();
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        baseSplitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, 9);

        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        imageInput = VisionEncoderUtils.preprocessFrames(baseSplitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        loaded = true;
        log.info("Pixel-mask sensitivity inputs ready: grid={}x{} frames={} imageInputShape={}",
                baseSplitResult.numRows, baseSplitResult.numCols, baseSplitResult.getTotalFrames(),
                Arrays.toString(imageInput.shape()));
    }

    private GenerationResult runGeneration(BenchmarkConfig config,
                                           ImageTiler.SplitImageResult splitResult,
                                           INDArray visionEmbeddings,
                                           int maxTokens) throws Exception {
        PromptInputs promptInputs = buildPromptInputs(splitResult, visionEmbeddings, maxTokens);
        try {
            ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
            long hiddenSize = visionEmbeddings.shape()[2];
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
                return pipeline.generate(promptInputs.inputsEmbeds, promptInputs.promptTokenIds, maxTokens);
            } finally {
                pipeline.close();
            }
        } finally {
            promptInputs.close();
        }
    }

    private PromptInputs buildPromptInputs(ImageTiler.SplitImageResult splitResult,
                                           INDArray visionEmbeddings,
                                           int maxTokens) throws Exception {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        GenerationPipeline embedPipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(Math.max(1, maxTokens))
                        .build());
        INDArray textEmbeddings = null;
        try {
            textEmbeddings = embedPipeline.embedTokens(promptTokenIds);
            INDArray merged = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
            return new PromptInputs(promptTokenIds, merged);
        } finally {
            if (textEmbeddings != null && !textEmbeddings.wasClosed()) {
                textEmbeddings.close();
            }
            embedPipeline.close();
        }
    }

    private INDArray encodeVision(ImageTiler.SplitImageResult splitResult) {
        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(9)
                .build();
        VisionEncoder.Result visionResult = visionEncoder.encode(
                imageInput, splitResult.getTotalFrames(), splitResult);
        return visionResult.getEmbeddings();
    }

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static ImageTiler.SplitImageResult cloneSplitResult(ImageTiler.SplitImageResult base,
                                                                int inflate,
                                                                boolean fullMask) {
        List<BufferedImage> frames = new ArrayList<>(base.frames);
        List<ImageTiler.ContentRegion> contentRegions = new ArrayList<>(base.contentRegions.size());
        for (ImageTiler.ContentRegion region : base.contentRegions) {
            int width = fullMask ? TARGET_SIZE : Math.min(TARGET_SIZE, region.width + inflate);
            int height = fullMask ? TARGET_SIZE : Math.min(TARGET_SIZE, region.height + inflate);
            contentRegions.add(new ImageTiler.ContentRegion(width, height));
        }
        return new ImageTiler.SplitImageResult(frames, contentRegions, base.numRows, base.numCols);
    }

    private static BufferedImage loadBenchmarkPageImage() throws IOException {
        File pdfFile = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        if (!pdfFile.exists()) {
            pdfFile = new File("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
        }

        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        int renderDpi = Integer.getInteger("vlm.test.pdf.dpi", 150);
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
        }
    }

    private static String normalize(String text) {
        return text == null ? "" : text.toLowerCase();
    }

    private static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "<null>";
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        return normalized.length() <= maxChars ? normalized : normalized.substring(0, maxChars) + "...";
    }

    private static String summarize(Map<String, GenerationResult> results) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, GenerationResult> entry : results.entrySet()) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append(entry.getKey()).append("='")
                    .append(safeSnippet(entry.getValue().getText(), 120))
                    .append("'");
        }
        return sb.toString();
    }

    private static class PromptInputs {
        private final int[] promptTokenIds;
        private final INDArray inputsEmbeds;

        private PromptInputs(int[] promptTokenIds, INDArray inputsEmbeds) {
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
        }

        private void close() {
            if (inputsEmbeds != null && !inputsEmbeds.wasClosed()) {
                inputsEmbeds.close();
            }
        }
    }
}
