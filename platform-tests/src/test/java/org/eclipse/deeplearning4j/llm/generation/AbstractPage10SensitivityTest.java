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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Abstract base class for Page-10 sensitivity tests.
 *
 * Holds all shared model fields, model-loading logic, generation helpers,
 * and utility methods used across the suite of Page-10 sensitivity tests.
 * Subclasses provide only the variant-specific test methods.
 */
abstract class AbstractPage10SensitivityTest {

    protected static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(AbstractPage10SensitivityTest.class);

    protected static final int TARGET_SIZE = 512;
    protected static final int MAX_TILES = 9;
    protected static final int LONGEST_EDGE = 2048;

    protected static SameDiff decoder;
    protected static SameDiff embedTokens;
    protected static SameDiff visionEncoderSd;
    protected static Tokenizer tokenizer;
    protected static File pdfFile;
    protected static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    /**
     * Downloads and imports all SmolDocling models. Resolves the benchmark PDF.
     * Must be called (possibly via a subclass-specific ensureLoaded() that delegates here)
     * before any test method accesses the model fields.
     *
     * After this returns, {@code decoder}, {@code embedTokens}, {@code visionEncoderSd},
     * {@code tokenizer}, and {@code pdfFile} are all populated.
     */
    protected static synchronized void ensureModelsLoaded() throws Exception {
        if (loaded) {
            return;
        }

        String pdfPath = System.getProperty("vlm.test.pdf.path");
        pdfFile = pdfPath != null
                ? new File(pdfPath)
                : new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
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

        loaded = true;
    }

    // -------------------------------------------------------------------------
    // Image loading
    // -------------------------------------------------------------------------

    /**
     * Renders the benchmark PDF page using the DPI specified by
     * {@code vlm.test.pdf.dpi} (default 150) and page specified by
     * {@code vlm.test.pdf.page} (default 10).
     */
    protected static BufferedImage loadBenchmarkPageImage() throws IOException {
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        int renderDpi = Integer.getInteger("vlm.test.pdf.dpi", 150);
        return loadBenchmarkPageImage(pdfPage, renderDpi);
    }

    /**
     * Renders the benchmark PDF at a specific DPI. The page number is still
     * taken from the {@code vlm.test.pdf.page} system property (default 10).
     */
    protected static BufferedImage loadBenchmarkPageImage(int renderDpi) throws IOException {
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        return loadBenchmarkPageImage(pdfPage, renderDpi);
    }

    private static BufferedImage loadBenchmarkPageImage(int pdfPage, int renderDpi) throws IOException {
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
        }
    }

    // -------------------------------------------------------------------------
    // Vision encoding
    // -------------------------------------------------------------------------

    /**
     * Encodes a {@link ImageTiler.SplitImageResult} through the vision encoder,
     * using the default legacy preprocessor config ({@link #legacyBenchmarkConfig()}).
     * Returns the concatenated frame embeddings.
     */
    protected static INDArray encodeVision(ImageTiler.SplitImageResult splitResult) {
        return encodeVision(splitResult, MAX_TILES);
    }

    /**
     * Encodes a {@link ImageTiler.SplitImageResult} through the vision encoder
     * with an explicit maxTiles cap.
     */
    protected static INDArray encodeVision(ImageTiler.SplitImageResult splitResult, int maxTiles) {
        PreprocessorConfig ppConfig = legacyBenchmarkConfig();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(maxTiles)
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

    // -------------------------------------------------------------------------
    // Prompt building
    // -------------------------------------------------------------------------

    /**
     * Builds merged (text + vision) prompt embeddings for the given split result
     * and pre-encoded vision embeddings.
     */
    protected static PromptInputs buildPromptInputs(ImageTiler.SplitImageResult splitResult,
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

    // -------------------------------------------------------------------------
    // Generation
    // -------------------------------------------------------------------------

    /**
     * Runs the GenerationPipeline for the given prompt inputs and vision embeddings.
     */
    protected static GenerationResult runGeneration(PromptInputs promptInputs,
                                                    INDArray visionEmbeddings,
                                                    int maxTokens) throws Exception {
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
    }

    /**
     * Runs the GenerationPipeline for the given {@link VariantInputs}.
     */
    protected static GenerationResult runGeneration(VariantInputs variant, int maxTokens) throws Exception {
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

    // -------------------------------------------------------------------------
    // Benchmark compilation
    // -------------------------------------------------------------------------

    protected static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    // -------------------------------------------------------------------------
    // Preprocessor config
    // -------------------------------------------------------------------------

    protected static PreprocessorConfig legacyBenchmarkConfig() {
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        return ppConfig;
    }

    // -------------------------------------------------------------------------
    // Utility predicates and formatters
    // -------------------------------------------------------------------------

    protected static String normalize(String text) {
        return text == null ? "" : text.toLowerCase();
    }

    protected static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "<null>";
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        return normalized.length() <= maxChars ? normalized : normalized.substring(0, maxChars) + "...";
    }

    protected static String summarize(Map<String, GenerationResult> results) {
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

    protected static boolean containsDoclingText(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.toLowerCase();
        return normalized.contains("<text>")
                || normalized.contains("<section_header")
                || normalized.contains("<page>");
    }

    // -------------------------------------------------------------------------
    // Inner classes shared across subclasses
    // -------------------------------------------------------------------------

    /**
     * Holds merged prompt+vision embeddings and the raw prompt token IDs.
     * Used by tests that pre-build all variant inputs before compilation.
     */
    protected static class PromptInputs {
        protected final int[] promptTokenIds;
        protected final INDArray inputsEmbeds;

        protected PromptInputs(int[] promptTokenIds, INDArray inputsEmbeds) {
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
        }

        protected void close() {
            if (inputsEmbeds != null && !inputsEmbeds.wasClosed()) {
                inputsEmbeds.close();
            }
        }
    }

    /**
     * Holds merged prompt+vision embeddings together with the (still-live)
     * vision embedding tensor. Used by tests that need the hidden size from
     * the vision embeddings when constructing the GenerationPipeline.
     */
    protected static class VariantInputs implements AutoCloseable {
        protected final int[] promptTokenIds;
        protected final INDArray inputsEmbeds;
        protected final INDArray visionEmbeddings;

        protected VariantInputs(int[] promptTokenIds, INDArray inputsEmbeds, INDArray visionEmbeddings) {
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

    /**
     * Shared helper for building a {@link VariantInputs} from a split result.
     * Encodes vision frames, builds the prompt, and merges text + vision embeddings.
     * The caller is responsible for closing the returned {@link VariantInputs}.
     */
    protected static VariantInputs buildVariantInputs(ImageTiler.SplitImageResult splitResult,
                                                      int maxTiles,
                                                      int maxTokens) throws Exception {
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
}
