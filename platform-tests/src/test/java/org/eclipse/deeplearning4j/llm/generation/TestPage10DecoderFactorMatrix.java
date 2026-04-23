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
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
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
 * Page-10 factor matrix for the current broken benchmark input.
 *
 * This isolates whether the remaining collapse is sensitive to:
 * 1. replay mode / benchmark config
 * 2. old-decoder reusable-vs-fresh decode buffers
 *
 * If all collapsed cases still match across these axes, the next bug is upstream of
 * both replay mode and the old reusable-buffer regression.
 */
@Slf4j
public class TestPage10DecoderFactorMatrix {

    private static final int TARGET_SIZE = 512;
    private static SameDiff visionEncoderSd;
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 control case (maxTiles=5): replay mode and reusable buffers preserve textual output")
    public void testPage10FactorMatrixMaxTiles5Control() throws Exception {
        ensureModelsLoaded();
        VariantInputs variant = buildVariant("PAGE10_MAXTILES5_MATRIX", 10, 2048, 5);
        try {
            Map<String, DecodeSnapshot> results = runMatrix(variant, 40);
            for (Map.Entry<String, DecodeSnapshot> entry : results.entrySet()) {
                String label = entry.getKey();
                GenerationResult result = entry.getValue().result;
                logResult(label, result);
                assertTrue(hasTextualStructure(result.getText()),
                        label + ": maxTiles=5 should retain textual structure. text='"
                                + safeSnippet(result.getText(), 220) + "'");
                assertFalse(isPictureOnlyCollapse(result.getText()),
                        label + ": maxTiles=5 should not collapse to picture-only output. text='"
                                + safeSnippet(result.getText(), 220) + "'");
            }

            assertReusableVsFreshParity(results, "DIRECT");
            assertReusableVsFreshParity(results, "SLOT_BY_SLOT");
            assertReusableVsFreshParity(results, "OPTIMAL");
        } finally {
            variant.close();
        }
    }

    @Test
    @DisplayName("Page-10 default broken case (maxTiles=9): replay mode and reusable buffers all collapse the same way")
    public void testPage10FactorMatrixMaxTiles9BrokenCase() throws Exception {
        ensureModelsLoaded();
        VariantInputs variant = buildVariant("PAGE10_MAXTILES9_MATRIX", 10, 2048, 9);
        try {
            Map<String, DecodeSnapshot> results = runMatrix(variant, 40);
            for (Map.Entry<String, DecodeSnapshot> entry : results.entrySet()) {
                String label = entry.getKey();
                GenerationResult result = entry.getValue().result;
                logResult(label, result);
                assertTrue(isPictureOnlyCollapse(result.getText()),
                        label + ": current maxTiles=9 benchmark input is expected to reproduce the picture-only collapse. text='"
                                + safeSnippet(result.getText(), 220) + "'");
                assertFalse(hasMythicContent(result.getText()),
                        label + ": maxTiles=9 unexpectedly produced mythic-paragraph content. text='"
                                + safeSnippet(result.getText(), 220) + "'");
            }

            assertReusableVsFreshParity(results, "DIRECT");
            assertReusableVsFreshParity(results, "SLOT_BY_SLOT");
            assertReusableVsFreshParity(results, "OPTIMAL");
        } finally {
            variant.close();
        }
    }

    private Map<String, DecodeSnapshot> runMatrix(VariantInputs variant, int maxTokens) throws Exception {
        LinkedHashMap<String, BenchmarkConfig> configs = new LinkedHashMap<>();
        configs.put("DIRECT", null);
        configs.put("SLOT_BY_SLOT", BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens)
                .minDiversityPct(0));
        configs.put("OPTIMAL", BenchmarkConfig.optimal()
                .maxTokens(maxTokens)
                .minDiversityPct(0));

        LinkedHashMap<String, DecodeSnapshot> results = new LinkedHashMap<>();
        for (Map.Entry<String, BenchmarkConfig> entry : configs.entrySet()) {
            String configName = entry.getKey();
            BenchmarkConfig config = entry.getValue();
            for (boolean useReusableBuffers : new boolean[]{true, false}) {
                String label = configName + "/" + (useReusableBuffers ? "REUSE" : "FRESH");
                results.put(label, runOldDecoder(variant, maxTokens, config, useReusableBuffers));
            }
        }
        return results;
    }

    private void assertReusableVsFreshParity(Map<String, DecodeSnapshot> results, String configName) {
        DecodeSnapshot reuse = results.get(configName + "/REUSE");
        DecodeSnapshot fresh = results.get(configName + "/FRESH");
        assertNotNull(reuse, configName + ": missing REUSE result");
        assertNotNull(fresh, configName + ": missing FRESH result");
        assertArrayEquals(reuse.result.getTokenIds(), fresh.result.getTokenIds(),
                configName + ": reusable-vs-fresh buffers changed the token stream unexpectedly."
                        + " reuse='" + safeSnippet(reuse.result.getText(), 180) + "'"
                        + " fresh='" + safeSnippet(fresh.result.getText(), 180) + "'");
        assertEquals(reuse.result.getText(), fresh.result.getText(),
                configName + ": reusable-vs-fresh buffers changed the decoded text unexpectedly.");
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
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
        loaded = true;
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, int maxTiles) throws Exception {
        BenchmarkConfigApplier.resetModelState(visionEncoderSd);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        BufferedImage rendered = loadPageImage(pdfPage);
        if (resizeLongestEdge > 0) {
            rendered = ImageTiler.resizeLongestEdge(rendered, resizeLongestEdge);
        }

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(rendered, TARGET_SIZE, maxTiles);
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

        INDArray visionEmbeddings = encodeWithVisionEncoderWrapper(imageInput, splitResult);
        imageInput.close();

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray textEmbeddings = embedPromptDirect(promptTokenIds);
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        textEmbeddings.close();

        log.info("{}: resize={} maxTiles={} frames={} grid={}x{} mergedShape={} promptTokens={}",
                name, resizeLongestEdge, maxTiles,
                splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols,
                Arrays.toString(inputsEmbeds.shape()), promptTokenIds.length);

        return new VariantInputs(name, promptTokenIds, inputsEmbeds);
    }

    private static BufferedImage loadPageImage(int pdfPage) throws IOException {
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
        int renderDpi = Integer.getInteger("vlm.test.pdf.dpi", 150);
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
            log.info("Loaded benchmark PDF page {} at {} DPI: {}x{} from {}",
                    pdfPage, renderDpi, image.getWidth(), image.getHeight(), pdfFile.getAbsolutePath());
            return image;
        }
    }

    private INDArray encodeManually(INDArray imageInput, ImageTiler.SplitImageResult splitResult) {
        List<String> visionInputNames = visionEncoderSd.inputs();
        String[] visionOutputNames = visionEncoderSd.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();
        try {
            for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
                INDArray frameSlice = imageInput.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(frameIdx),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();

                Map<String, INDArray> visionInputMap = new LinkedHashMap<>();
                for (String inputName : visionInputNames) {
                    if ("pixel_values".equals(inputName)) {
                        visionInputMap.put(inputName, singleFrame);
                    } else if ("pixel_attention_mask".equals(inputName)) {
                        ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                        visionInputMap.put(inputName,
                                ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE));
                    }
                }

                Map<String, INDArray> visionOutputs = visionEncoderSd.output(visionInputMap, visionOutputNames);
                try {
                    VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
                    frameEmbeddings.add(selected.tensor.dup());
                } finally {
                    for (INDArray arr : visionOutputs.values()) {
                        closeArray(arr);
                    }
                    for (INDArray arr : visionInputMap.values()) {
                        closeArray(arr);
                    }
                    closeArray(singleFrame);
                }
            }

            return frameEmbeddings.size() == 1
                    ? frameEmbeddings.get(0).dup()
                    : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
        } finally {
            for (INDArray arr : frameEmbeddings) {
                closeArray(arr);
            }
            resetSameDiff(visionEncoderSd);
        }
    }

    private INDArray encodeWithVisionEncoderWrapper(INDArray imageInput, ImageTiler.SplitImageResult splitResult) {
        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(5)
                .build();
        try {
            VisionEncoder.Result result = visionEncoder.encode(imageInput, splitResult.getTotalFrames(), splitResult);
            return result.getEmbeddings().dup();
        } finally {
            resetSameDiff(visionEncoderSd);
        }
    }

    private static INDArray embedPromptDirect(int[] tokenIds) {
        INDArray inputIds = Nd4j.createFromArray(new int[][]{tokenIds}).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new LinkedHashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, inputIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        try {
            for (INDArray arr : embedOutputs.values()) {
                return arr.dup();
            }
            throw new IllegalStateException("embed_tokens model produced no output");
        } finally {
            inputIds.close();
            for (INDArray arr : embedOutputs.values()) {
                closeArray(arr);
            }
            resetSameDiff(embedTokens);
        }
    }

    private DecodeSnapshot runOldDecoder(VariantInputs variant, int maxTokens, BenchmarkConfig config,
                                         boolean useReusableBuffers) throws Exception {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        if (config != null) {
            BenchmarkConfigApplier.apply(config);
            BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
        } else {
            resetDirectModelState(decoder);
            resetDirectModelState(embedTokens);
        }

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(variant.inputsEmbeds.size(2))
                .useReusableBuffers(useReusableBuffers)
                .build();
        long startNs = System.nanoTime();
        GenerationResult result = loop.decode(variant.inputsEmbeds.dup(), variant.promptTokenIds);
        long elapsedNs = System.nanoTime() - startNs;
        return new DecodeSnapshot(result, elapsedNs);
    }

    private static void resetSameDiff(SameDiff sd) {
        BenchmarkConfigApplier.resetModelState(sd);
        Nd4j.getExecutioner().commit();
    }

    private static void resetDirectModelState(SameDiff sd) {
        sd.clearPlaceholderOverrides();
        sd.clearPlaceholders(true);
        sd.clearOpInputs();
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        Nd4j.getExecutioner().commit();
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

    private static boolean isPictureOnlyCollapse(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim().toLowerCase();
        return normalized.startsWith("<doctag><picture>")
                || (normalized.contains("<picture>") && normalized.contains("<end_of_utterance>"));
    }

    private static boolean hasTextualStructure(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.toLowerCase();
        return normalized.contains("<text>")
                || normalized.contains("<section_header")
                || normalized.contains("<page>");
    }

    private static boolean hasMythicContent(String text) {
        if (text == null) {
            return false;
        }
        String lower = text.toLowerCase();
        return lower.contains("mythic heroes")
                || lower.contains("hytic heroes")
                || lower.contains("creating a mythic character");
    }

    private static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "null";
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, maxChars) + "...";
    }

    private static void closeArray(INDArray arr) {
        if (arr != null && arr.closeable() && !arr.wasClosed()) {
            arr.close();
        }
    }

    private static final class VariantInputs implements AutoCloseable {
        private final String name;
        private final int[] promptTokenIds;
        private final INDArray inputsEmbeds;

        private VariantInputs(String name, int[] promptTokenIds, INDArray inputsEmbeds) {
            this.name = name;
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
        }

        @Override
        public void close() {
            closeArray(inputsEmbeds);
        }
    }

    private static final class DecodeSnapshot {
        private final GenerationResult result;
        private final long elapsedNs;

        private DecodeSnapshot(GenerationResult result, long elapsedNs) {
            this.result = result;
            this.elapsedNs = elapsedNs;
        }
    }
}
