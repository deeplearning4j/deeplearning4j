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

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Slf4j
public class TestPage10ResizeInterpolationSensitivity {

    private static final int TARGET_SIZE = 512;
    private static final int MAX_TILES = 9;
    private static final int LONGEST_EDGE = 2048;

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static BufferedImage resizedForTiling;
    private static ImageTiler.SplitImageResult baseSplitResult;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 interpolation sensitivity: resize kernel isolates residual OCR character loss")
    public void testPage10InterpolationSensitivity() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        Map<String, Object> interpolationHints = new LinkedHashMap<>();
        interpolationHints.put("BILINEAR", RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        interpolationHints.put("BICUBIC", RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        interpolationHints.put("NEAREST", RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

        Map<String, ImageTiler.SplitImageResult> variants = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : interpolationHints.entrySet()) {
            variants.put(entry.getKey(), buildSplit(entry.getValue()));
        }

        Map<String, INDArray> encodedVision = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, ImageTiler.SplitImageResult> entry : variants.entrySet()) {
                encodedVision.put(entry.getKey(), encodeVision(entry.getValue()));
            }

            compileFor(config);

            Map<String, GenerationResult> results = new LinkedHashMap<>();
            for (Map.Entry<String, ImageTiler.SplitImageResult> entry : variants.entrySet()) {
                String name = entry.getKey();
                GenerationResult result = runGeneration(
                        config, entry.getValue(), encodedVision.get(name), maxTokens);
                results.put(name, result);
                log.info("{} text='{}'", name, safeSnippet(result.getText(), 260));
            }

            GenerationResult baseline = results.get("BILINEAR");
            assertTrue(baseline != null && normalize(baseline.getText()).contains("heroes are set apart"),
                    "BILINEAR baseline no longer contains the expected paragraph fragment. "
                            + summarize(results));

            boolean anyChanged = false;
            for (Map.Entry<String, GenerationResult> entry : results.entrySet()) {
                if (!"BILINEAR".equals(entry.getKey())
                        && !Arrays.equals(baseline.getTokenIds(), entry.getValue().getTokenIds())) {
                    anyChanged = true;
                    break;
                }
            }
            assertTrue(anyChanged,
                    "Interpolation variants did not change the token stream at all. " + summarize(results));
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
        resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, LONGEST_EDGE);
        baseSplitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, MAX_TILES);

        loaded = true;
        log.info("Interpolation sensitivity inputs ready: resized={}x{} grid={}x{} frames={}",
                resizedForTiling.getWidth(), resizedForTiling.getHeight(),
                baseSplitResult.numRows, baseSplitResult.numCols, baseSplitResult.getTotalFrames());
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
                .maxTiles(MAX_TILES)
                .build();
        try {
            VisionEncoder.Result visionResult = visionEncoder.encode(
                    imageInput, splitResult.getTotalFrames(), splitResult);
            return visionResult.getEmbeddings();
        } finally {
            if (imageInput != null && !imageInput.wasClosed()) {
                imageInput.close();
            }
            visionEncoder.close();
        }
    }

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static ImageTiler.SplitImageResult buildSplit(Object interpolationHint) {
        int numRows = baseSplitResult.numRows;
        int numCols = baseSplitResult.numCols;
        int width = resizedForTiling.getWidth();
        int height = resizedForTiling.getHeight();
        int optimalHeight = (int) Math.ceil((double) height / numRows);
        int optimalWidth = (int) Math.ceil((double) width / numCols);

        List<BufferedImage> frames = new ArrayList<>(baseSplitResult.getTotalFrames());
        List<ImageTiler.ContentRegion> contentRegions = new ArrayList<>(baseSplitResult.getTotalFrames());
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                int startX = c * optimalWidth;
                int startY = r * optimalHeight;
                int endX = Math.min(width, startX + optimalWidth);
                int endY = Math.min(height, startY + optimalHeight);
                PreparedFrame prepared = prepareFrame(
                        resizedForTiling.getSubimage(startX, startY, endX - startX, endY - startY),
                        TARGET_SIZE, interpolationHint);
                frames.add(prepared.image);
                contentRegions.add(prepared.contentRegion);
            }
        }

        PreparedFrame globalFrame = prepareFrame(resizedForTiling, TARGET_SIZE, interpolationHint);
        frames.add(globalFrame.image);
        contentRegions.add(globalFrame.contentRegion);
        return new ImageTiler.SplitImageResult(frames, contentRegions, numRows, numCols);
    }

    private static PreparedFrame prepareFrame(BufferedImage source, int maxSize, Object interpolationHint) {
        ResizeResult resized = resizeToFit(source, maxSize, maxSize, interpolationHint);
        BufferedImage padded = ImageTiler.padToSize(resized.image, maxSize, maxSize);
        return new PreparedFrame(padded, new ImageTiler.ContentRegion(resized.width, resized.height));
    }

    private static ResizeResult resizeToFit(BufferedImage image,
                                            int targetWidth,
                                            int targetHeight,
                                            Object interpolationHint) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width <= targetWidth && height <= targetHeight) {
            return new ResizeResult(image, width, height);
        }
        double scale = Math.min((double) targetWidth / (double) width, (double) targetHeight / (double) height);
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));
        return new ResizeResult(resizeImage(image, newW, newH, interpolationHint), newW, newH);
    }

    private static BufferedImage resizeImage(BufferedImage original,
                                             int targetWidth,
                                             int targetHeight,
                                             Object interpolationHint) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolationHint);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return resized;
    }

    private static BufferedImage loadBenchmarkPageImage() throws IOException {
        String pdfPath = System.getProperty("vlm.test.pdf.path");
        File pdfFile = pdfPath != null ? new File(pdfPath) : new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        assumeTrue(pdfFile.exists(), "PDF not found at " + pdfFile.getAbsolutePath()
                + ". Place pathfinder-mythic.pdf in platform-tests/ or set -Dvlm.test.pdf.path");

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

    private static class ResizeResult {
        private final BufferedImage image;
        private final int width;
        private final int height;

        private ResizeResult(BufferedImage image, int width, int height) {
            this.image = image;
            this.width = width;
            this.height = height;
        }
    }

    private static class PreparedFrame {
        private final BufferedImage image;
        private final ImageTiler.ContentRegion contentRegion;

        private PreparedFrame(BufferedImage image, ImageTiler.ContentRegion contentRegion) {
            this.image = image;
            this.contentRegion = contentRegion;
        }
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
