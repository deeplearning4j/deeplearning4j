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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Slf4j
public class TestPage10MaxTilesCurrentSensitivity {

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
    @DisplayName("Page-10 current-path maxTiles sensitivity")
    public void testPage10CurrentPathMaxTilesSensitivity() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);
        int[] maxTilesValues = new int[]{5, 6, 9};

        Map<String, VariantInputs> variants = new LinkedHashMap<>();
        try {
            for (int maxTiles : maxTilesValues) {
                String name = "MAX_TILES_" + maxTiles;
                variants.put(name, buildVariant(maxTiles, maxTokens));
            }

            compileFor(config);

            Map<String, GenerationResult> results = new LinkedHashMap<>();
            for (Map.Entry<String, VariantInputs> entry : variants.entrySet()) {
                GenerationResult result = runGeneration(entry.getValue(), maxTokens);
                results.put(entry.getKey(), result);
                log.info("{} text='{}'", entry.getKey(), safeSnippet(result.getText(), 260));
            }

            GenerationResult tiles5 = results.get("MAX_TILES_5");
            GenerationResult tiles9 = results.get("MAX_TILES_9");
            assertTrue(tiles5 != null && containsDoclingText(tiles5.getText()),
                    "MAX_TILES_5 no longer produced textual Docling output. " + summarize(results));
            assertTrue(tiles9 != null && containsDoclingText(tiles9.getText()),
                    "MAX_TILES_9 no longer produced textual Docling output. " + summarize(results));
            assertTrue(!Arrays.equals(tiles5.getTokenIds(), tiles9.getTokenIds()),
                    "MAX_TILES_5 and MAX_TILES_9 produced the same token stream. " + summarize(results));
        } finally {
            for (VariantInputs variant : variants.values()) {
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
        log.info("Max-tiles sensitivity base image ready: pdf={} image={}x{}",
                pdfFile.getAbsolutePath(), pageImage.getWidth(), pageImage.getHeight());
    }

    private static VariantInputs buildVariant(int maxTiles, int maxTokens) throws Exception {
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pageImage, LONGEST_EDGE);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, maxTiles);

        PreprocessorConfig ppConfig = legacyBenchmarkConfig();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(maxTiles)
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

    private static boolean containsDoclingText(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.toLowerCase();
        return normalized.contains("<text>")
                || normalized.contains("<section_header")
                || normalized.contains("<page>");
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

    private static class VariantInputs {
        private final int[] promptTokenIds;
        private final INDArray inputsEmbeds;
        private final INDArray visionEmbeddings;

        private VariantInputs(int[] promptTokenIds, INDArray inputsEmbeds, INDArray visionEmbeddings) {
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
            this.visionEmbeddings = visionEmbeddings;
        }

        private void close() {
            if (inputsEmbeds != null && !inputsEmbeds.wasClosed()) {
                inputsEmbeds.close();
            }
            if (visionEmbeddings != null && !visionEmbeddings.wasClosed()) {
                visionEmbeddings.close();
            }
        }
    }
}
