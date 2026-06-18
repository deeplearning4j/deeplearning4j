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
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
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
public class TestGenerationPipelinePromptOrderIsolation {

    private static final int TARGET_SIZE = 512;

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static ImageTiler.SplitImageResult splitResult;
    private static INDArray visionEmbeddings;
    private static long hiddenSize;
    private static boolean loaded;

    private enum PromptVariant {
        DEFAULT,
        GLOBAL_FIRST_SIMPLE,
        GLOBAL_FIRST_PACKED
    }

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
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
        splitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, 9);

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
        visionEmbeddings = visionResult.getEmbeddings();
        hiddenSize = visionEmbeddings.shape()[2];
        imageInput.close();
        visionEncoder.freeModelMemory();

        loaded = true;
        log.info("Prompt-order isolation inputs ready: grid={}x{} frames={} visionShape={}",
                splitResult.numRows, splitResult.numCols, splitResult.getTotalFrames(),
                Arrays.toString(visionEmbeddings.shape()));
    }

    @Test
    @DisplayName("Page-10 prompt-order isolation under OPTIMAL using current production tiler")
    public void testPage10PromptOrderIsolation() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        Map<PromptVariant, GenerationResult> results = new LinkedHashMap<>();
        for (PromptVariant variant : PromptVariant.values()) {
            GenerationResult result = runGenerationPipeline(config, maxTokens, variant);
            results.put(variant, result);
            log.info("{} generated={} finish={} text='{}'",
                    variant,
                    result.getGeneratedTokenCount(),
                    result.getFinishReason(),
                    safeSnippet(result.getText(), 260));
        }

        boolean anyHeroes = results.values().stream()
                .map(GenerationResult::getText)
                .map(TestGenerationPipelinePromptOrderIsolation::normalize)
                .anyMatch(text -> text.contains("mythic heroes")
                        || text.contains("hytic heroes")
                        || text.contains("heroes are set apart"));
        assertTrue(anyHeroes,
                "No prompt-order variant produced the mythic-heroes paragraph. "
                        + summarizeResults(results));
    }

    private GenerationResult runGenerationPipeline(BenchmarkConfig config, int maxTokens, PromptVariant variant) throws Exception {
        compileFor(config);
        PromptInputs promptInputs = buildPromptInputs(variant, maxTokens);
        try {
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
                return pipeline.generate(promptInputs.inputsEmbeds, promptInputs.promptTokenIds, maxTokens);
            } finally {
                pipeline.close();
            }
        } finally {
            promptInputs.close();
        }
    }

    private PromptInputs buildPromptInputs(PromptVariant variant, int maxTokens) throws Exception {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt;
        switch (variant) {
            case DEFAULT:
                imagePrompt = ImagePromptBuilder.buildImagePromptString(
                        splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
                break;
            case GLOBAL_FIRST_SIMPLE:
                imagePrompt = buildGlobalFirstSimplePrompt(
                        splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
                break;
            case GLOBAL_FIRST_PACKED:
                imagePrompt = buildPackedGlobalFirstPrompt(
                        splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
                break;
            default:
                throw new IllegalStateException("Unhandled prompt variant: " + variant);
        }

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

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static String buildGlobalFirstSimplePrompt(int imageRows, int imageCols, int imageSeqLen) {
        String fake = "<fake_token_around_image>";
        String image = "<image>";
        String global = "<global-img>";

        if (imageRows <= 0 || imageCols <= 0) {
            return ImagePromptBuilder.buildImagePromptString(imageRows, imageCols, imageSeqLen);
        }

        StringBuilder sb = new StringBuilder();
        sb.append(fake).append(global);
        for (int i = 0; i < imageSeqLen; i++) {
            sb.append(image);
        }
        sb.append("\n");

        for (int r = 1; r <= imageRows; r++) {
            for (int c = 1; c <= imageCols; c++) {
                sb.append(fake).append("<row_").append(r).append("_col_").append(c).append(">");
                for (int i = 0; i < imageSeqLen; i++) {
                    sb.append(image);
                }
            }
            sb.append("\n");
        }

        sb.append(fake);
        return sb.toString();
    }

    private static String buildPackedGlobalFirstPrompt(int imageRows, int imageCols, int imageSeqLen) {
        int tileCount = imageRows * imageCols;
        int totalSegments = tileCount + 1;
        int[] order = new int[totalSegments];
        order[0] = tileCount;
        for (int i = 1; i < totalSegments; i++) {
            order[i] = i - 1;
        }

        String[] descriptors = new String[totalSegments];
        int idx = 0;
        for (int r = 1; r <= imageRows; r++) {
            for (int c = 1; c <= imageCols; c++) {
                descriptors[idx++] = "<row_" + r + "_col_" + c + ">";
            }
        }
        descriptors[idx] = "<global-img>";

        String fake = "<fake_token_around_image>";
        String image = "<image>";
        StringBuilder sb = new StringBuilder();
        for (int targetIdx = 0; targetIdx < totalSegments; targetIdx++) {
            if (targetIdx > 0 && imageCols > 0 && targetIdx % imageCols == 0) {
                sb.append("\n");
            }
            sb.append(fake).append(descriptors[order[targetIdx]]);
            for (int i = 0; i < imageSeqLen; i++) {
                sb.append(image);
            }
        }
        sb.append(fake);
        return sb.toString();
    }

    private static String summarizeResults(Map<PromptVariant, GenerationResult> results) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<PromptVariant, GenerationResult> entry : results.entrySet()) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append(entry.getKey()).append("='")
                    .append(safeSnippet(entry.getValue().getText(), 140))
                    .append("'");
        }
        return sb.toString();
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
