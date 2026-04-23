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
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
public class TestPage10VisionFrameOrderSensitivity {

    private static final int TARGET_SIZE = 512;
    private static final int MAX_TILES = 9;
    private static final int LONGEST_EDGE = 2048;

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static ImageTiler.SplitImageResult splitResult;
    private static INDArray visionEmbeddings;
    private static File pdfFile;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 vision frame-order sensitivity: packed prompt vs tiles-first embedding layout")
    public void testPage10VisionFrameOrderSensitivity() throws Exception {
        ensureLoaded();

        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        Map<String, INDArray> variants = new LinkedHashMap<>();
        variants.put("TILES_THEN_GLOBAL", visionEmbeddings.dup());
        variants.put("GLOBAL_THEN_TILES", reorderGlobalFrameFirst(visionEmbeddings, splitResult.getTotalFrames()));

        try {
            compileFor(config);

            Map<String, GenerationResult> results = new LinkedHashMap<>();
            for (Map.Entry<String, INDArray> entry : variants.entrySet()) {
                GenerationResult result = runGeneration(entry.getValue(), maxTokens);
                results.put(entry.getKey(), result);
                log.info("{} text='{}'", entry.getKey(), safeSnippet(result.getText(), 260));
            }

            GenerationResult baseline = results.get("TILES_THEN_GLOBAL");
            GenerationResult reordered = results.get("GLOBAL_THEN_TILES");
            assertTrue(baseline != null && normalize(baseline.getText()).contains("heroes are set apart"),
                    "Baseline packed-prompt output no longer contains the expected paragraph fragment. "
                            + summarize(results));
            assertFalse(Arrays.equals(baseline.getTokenIds(), reordered.getTokenIds()),
                    "Reordering vision frame blocks did not change the token stream. " + summarize(results));
        } finally {
            for (INDArray arr : variants.values()) {
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

        pdfFile = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        if (!pdfFile.exists()) {
            pdfFile = new File("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
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
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, LONGEST_EDGE);
        splitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, MAX_TILES);

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
            visionEmbeddings = visionResult.getEmbeddings();
        } finally {
            imageInput.close();
            visionEncoder.close();
        }

        loaded = true;
        log.info("Vision frame-order inputs ready: pdf={} frames={} visionShape={}",
                pdfFile.getAbsolutePath(), splitResult.getTotalFrames(), Arrays.toString(visionEmbeddings.shape()));
    }

    private static GenerationResult runGeneration(INDArray activeVisionEmbeddings, int maxTokens) throws Exception {
        PromptInputs promptInputs = buildPromptInputs(activeVisionEmbeddings, maxTokens);
        try {
            ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
            long hiddenSize = activeVisionEmbeddings.shape()[2];
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

    private static PromptInputs buildPromptInputs(INDArray activeVisionEmbeddings, int maxTokens) throws Exception {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) activeVisionEmbeddings.size(1) / splitResult.getTotalFrames();
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
                    textEmbeddings, activeVisionEmbeddings, promptTokenIds, imageTokenId);
            return new PromptInputs(promptTokenIds, merged);
        } finally {
            if (textEmbeddings != null && !textEmbeddings.wasClosed()) {
                textEmbeddings.close();
            }
            embedPipeline.close();
        }
    }

    private static INDArray reorderGlobalFrameFirst(INDArray embeddings, int totalFrames) {
        long totalSeqLen = embeddings.size(1);
        long hiddenSize = embeddings.size(2);
        if (totalFrames <= 1) {
            return embeddings.dup();
        }
        if (totalSeqLen % totalFrames != 0) {
            throw new IllegalStateException("Vision sequence length " + totalSeqLen
                    + " is not divisible by totalFrames=" + totalFrames);
        }

        int seqPerFrame = (int) (totalSeqLen / totalFrames);
        int hidden = (int) hiddenSize;
        float[] src = embeddings.data().asFloat();
        float[] dst = new float[src.length];
        int blockSize = seqPerFrame * hidden;
        int globalFrameIdx = totalFrames - 1;
        int outBlock = 0;

        System.arraycopy(src, globalFrameIdx * blockSize, dst, outBlock * blockSize, blockSize);
        outBlock++;
        for (int frame = 0; frame < globalFrameIdx; frame++) {
            System.arraycopy(src, frame * blockSize, dst, outBlock * blockSize, blockSize);
            outBlock++;
        }

        return Nd4j.create(dst, new long[]{1, totalSeqLen, hidden}, 'c');
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
