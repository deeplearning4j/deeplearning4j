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
public class TestPage10WhitePaddingInterpolationMatrix {

    private static final int TARGET_SIZE = 512;
    private static final int LONGEST_EDGE = 2048;

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static BufferedImage pageImage;
    private static File pdfFile;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 white-padding interpolation matrix")
    public void testPage10WhitePaddingInterpolationMatrix() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        Map<String, VariantPolicy> variants = new LinkedHashMap<>();
        variants.put("BLACK_BILINEAR", new VariantPolicy(Color.BLACK, RenderingHints.VALUE_INTERPOLATION_BILINEAR));
        variants.put("WHITE_BILINEAR", new VariantPolicy(Color.WHITE, RenderingHints.VALUE_INTERPOLATION_BILINEAR));
        variants.put("WHITE_BICUBIC", new VariantPolicy(Color.WHITE, RenderingHints.VALUE_INTERPOLATION_BICUBIC));
        variants.put("WHITE_NEAREST", new VariantPolicy(Color.WHITE, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR));

        Map<String, VariantInputs> prepared = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, VariantPolicy> entry : variants.entrySet()) {
                prepared.put(entry.getKey(), buildVariant(entry.getValue(), maxTokens));
            }

            compileFor(config);

            Map<String, GenerationResult> results = new LinkedHashMap<>();
            for (Map.Entry<String, VariantInputs> entry : prepared.entrySet()) {
                GenerationResult result = runGeneration(entry.getValue(), maxTokens);
                results.put(entry.getKey(), result);
                log.info("{} text='{}'", entry.getKey(), safeSnippet(result.getText(), 260));
            }

            GenerationResult baseline = results.get("BLACK_BILINEAR");
            assertTrue(baseline != null && normalize(baseline.getText()).contains("heroes are set apart"),
                    "BLACK_BILINEAR baseline no longer contains the expected paragraph fragment. "
                            + summarize(results));

            boolean anyChanged = false;
            for (Map.Entry<String, GenerationResult> entry : results.entrySet()) {
                if (!"BLACK_BILINEAR".equals(entry.getKey())
                        && !Arrays.equals(baseline.getTokenIds(), entry.getValue().getTokenIds())) {
                    anyChanged = true;
                    break;
                }
            }
            assertTrue(anyChanged,
                    "White-padding interpolation variants did not change the token stream at all. "
                            + summarize(results));
        } finally {
            for (VariantInputs variant : prepared.values()) {
                if (variant != null) {
                    variant.close();
                }
            }
        }
    }

    private static synchronized void ensureLoaded() throws Exception {
        if (loaded) {
            return;
        }

        String pdfPath = System.getProperty("vlm.test.pdf.path");
        pdfFile = pdfPath != null ? new File(pdfPath) : new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        assumeTrue(pdfFile.exists(), "PDF not found at " + pdfFile.getAbsolutePath()
                + ". Place pathfinder-mythic.pdf in platform-tests/ or set -Dvlm.test.pdf.path");

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

        pageImage = loadBenchmarkPageImage();
        loaded = true;
        log.info("White-padding matrix base image ready: pdf={} image={}x{}",
                pdfFile.getAbsolutePath(), pageImage.getWidth(), pageImage.getHeight());
    }

    private static VariantInputs buildVariant(VariantPolicy policy, int maxTokens) throws Exception {
        BufferedImage resized = ImageTiler.resizeLongestEdge(pageImage, LONGEST_EDGE);
        ImageTiler.SplitImageResult splitResult = splitImageWithPolicy(resized, TARGET_SIZE, 3, 3, policy);

        PreprocessorConfig ppConfig = legacyBenchmarkConfig();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(9)
                .build();
        INDArray visionEmbeddings;
        try {
            VisionEncoder.Result visionResult = visionEncoder.encode(
                    imageInput, splitResult.getTotalFrames(), splitResult);
            visionEmbeddings = visionResult.getEmbeddings();
        } finally {
            imageInput.close();
            visionEncoder.close();
        }

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
            return new VariantInputs(promptTokenIds, merged, visionEmbeddings);
        } finally {
            if (textEmbeddings != null && !textEmbeddings.wasClosed()) {
                textEmbeddings.close();
            }
            embedPipeline.close();
        }
    }

    private static GenerationResult runGeneration(VariantInputs variant, int maxTokens) throws Exception {
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        long hiddenSize = variant.visionEmbeddings.shape()[2];
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
            return pipeline.generate(variant.inputsEmbeds, variant.promptTokenIds, maxTokens);
        } finally {
            pipeline.close();
        }
    }

    private static ImageTiler.SplitImageResult splitImageWithPolicy(
            BufferedImage image, int targetSize, int numRows, int numCols, VariantPolicy policy) {
        List<BufferedImage> frames = new ArrayList<>();
        List<ImageTiler.ContentRegion> contentRegions = new ArrayList<>();

        int width = image.getWidth();
        int height = image.getHeight();
        int optimalWidth = (int) Math.ceil((double) width / numCols);
        int optimalHeight = (int) Math.ceil((double) height / numRows);

        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                int startX = c * optimalWidth;
                int startY = r * optimalHeight;
                int endX = Math.min(startX + optimalWidth, width);
                int endY = Math.min(startY + optimalHeight, height);

                BufferedImage tile = image.getSubimage(startX, startY, endX - startX, endY - startY);
                ImageTiler.ResizeResult resized = resizeToFitWithHint(tile, targetSize, targetSize, policy.interpolationHint);
                BufferedImage padded = padToColor(resized.image, targetSize, targetSize, policy.padColor);
                frames.add(padded);
                contentRegions.add(new ImageTiler.ContentRegion(resized.width, resized.height));
            }
        }

        ImageTiler.ResizeResult globalResized = resizeToFitWithHint(
                image, targetSize, targetSize, policy.interpolationHint);
        BufferedImage globalPadded = padToColor(globalResized.image, targetSize, targetSize, policy.padColor);
        frames.add(globalPadded);
        contentRegions.add(new ImageTiler.ContentRegion(globalResized.width, globalResized.height));

        log.info("White-padding matrix split: color={} interpolation={} source={}x{} totalFrames={}",
                policy.padColor, policy.interpolationHint, width, height, frames.size());
        return new ImageTiler.SplitImageResult(frames, contentRegions, numRows, numCols);
    }

    private static ImageTiler.ResizeResult resizeToFitWithHint(
            BufferedImage image, int targetWidth, int targetHeight, Object interpolationHint) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width <= targetWidth && height <= targetHeight) {
            return new ImageTiler.ResizeResult(image, width, height);
        }
        double scale = Math.min((double) targetWidth / (double) width, (double) targetHeight / (double) height);
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));

        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolationHint);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.drawImage(image, 0, 0, newW, newH, null);
        g2d.dispose();
        return new ImageTiler.ResizeResult(resized, newW, newH);
    }

    private static BufferedImage padToColor(
            BufferedImage image, int targetWidth, int targetHeight, Color padColor) {
        if (image.getWidth() == targetWidth && image.getHeight() == targetHeight) {
            return image;
        }

        BufferedImage padded = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = padded.createGraphics();
        g2d.setColor(padColor);
        g2d.fillRect(0, 0, targetWidth, targetHeight);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return padded;
    }

    private static PreprocessorConfig legacyBenchmarkConfig() {
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        return ppConfig;
    }

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static BufferedImage loadBenchmarkPageImage() throws IOException {
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(pdfPage, 150, ImageType.RGB);
        }
    }

    private static String normalize(String text) {
        return text == null ? "" : text.toLowerCase();
    }

    private static String summarize(Map<String, GenerationResult> results) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, GenerationResult> entry : results.entrySet()) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append(entry.getKey())
                    .append("=")
                    .append(safeSnippet(entry.getValue().getText(), 120));
        }
        return sb.toString();
    }

    private static String safeSnippet(String text, int maxLen) {
        if (text == null) {
            return "<null>";
        }
        return text.length() <= maxLen ? text : text.substring(0, maxLen);
    }

    private static final class VariantPolicy {
        private final Color padColor;
        private final Object interpolationHint;

        private VariantPolicy(Color padColor, Object interpolationHint) {
            this.padColor = padColor;
            this.interpolationHint = interpolationHint;
        }
    }

    private static final class VariantInputs implements AutoCloseable {
        private final int[] promptTokenIds;
        private final INDArray inputsEmbeds;
        private final INDArray visionEmbeddings;

        private VariantInputs(int[] promptTokenIds, INDArray inputsEmbeds, INDArray visionEmbeddings) {
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
            this.visionEmbeddings = visionEmbeddings;
        }

        @Override
        public void close() {
            if (inputsEmbeds != null && !inputsEmbeds.wasClosed()) {
                inputsEmbeds.close();
            }
            if (visionEmbeddings != null && !visionEmbeddings.wasClosed()) {
                visionEmbeddings.close();
            }
        }
    }
}
