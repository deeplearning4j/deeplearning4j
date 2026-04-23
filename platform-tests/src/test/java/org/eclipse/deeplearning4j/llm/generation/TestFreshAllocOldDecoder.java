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
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * T1: Fresh allocation regression test for StaticKvCacheDecodeLoop.
 *
 * Commit f8e83ff4c9 fixed mythic-heroes output by removing reusable embedding/
 * inputId buffers (fixed-address .assign() caused CUDA graph replay to read
 * stale capture-time data). Commit 2f50e02765 re-introduced reusable buffers.
 *
 * This test runs the old decoder on page 10 (long sequence, 9 tiles) twice:
 *   A) useReusableBuffers=true  (current code — expected broken)
 *   B) useReusableBuffers=false (fresh allocations — expected correct)
 *
 * Pass = B produces mythic content while A does not, confirming the regression.
 */
@Slf4j
public class TestFreshAllocOldDecoder {

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
        if (modelsLoaded) return;

        log.info("Loading SmolDocling models for fresh-alloc regression test...");

        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        int targetSize = 512;
        String pdfPath = System.getProperty("vlm.test.pdf.path",
                "/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 10);
        File pdfFile = new File(pdfPath);
        BufferedImage image;
        if (pdfFile.exists()) {
            log.info("Using mythic PDF: {} page={}", pdfFile.getAbsolutePath(), pdfPage);
            try (org.apache.pdfbox.pdmodel.PDDocument doc = org.apache.pdfbox.pdmodel.PDDocument.load(pdfFile)) {
                org.apache.pdfbox.rendering.PDFRenderer renderer = new org.apache.pdfbox.rendering.PDFRenderer(doc);
                image = renderer.renderImageWithDPI(pdfPage, 150, org.apache.pdfbox.rendering.ImageType.RGB);
            }
        } else {
            log.info("PDF not found, generating synthetic document image");
            image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR);
            java.awt.Graphics2D g = image.createGraphics();
            g.setColor(java.awt.Color.WHITE);
            g.fillRect(0, 0, targetSize, targetSize);
            g.setColor(java.awt.Color.BLACK);
            g.setFont(new java.awt.Font("SansSerif", java.awt.Font.PLAIN, 24));
            g.drawString("Test Document", 50, 100);
            g.drawString("Line 2: Mythic Regression", 50, 150);
            g.dispose();
        }

        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(image, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, 9);

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

        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());
            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
        }

        visionEncoder.clearPlaceholders(false);
        visionEncoder.clearOpInputs();
        visionEncoder.resetSession();
        Nd4j.getExecutioner().commit();

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));

        hiddenSize = visionEmbeddings.size(-1);

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] encoded = tokenizer.encode(chatPrompt, false).getIds();
        promptTokenIds = encoded;

        INDArray tokenIds = Nd4j.createFromArray(encoded).reshape(1, encoded.length).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        assertEquals(promptTokenIds.length, inputsEmbeds.size(1),
                "Merged embedding sequence length must match prompt token length");

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize, promptTokenIds.length);
        modelsLoaded = true;
    }

    private static void resetDecoderState() {
        decoder.clearPlaceholders(false);
        decoder.clearOpInputs();
        decoder.resetSession();
        Nd4j.getExecutioner().commit();
    }

    private GenerationResult runOldDecoder(boolean useReusableBuffers, int maxTokens) throws Exception {
        ensureModelsLoaded();
        resetDecoderState();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .useReusableBuffers(useReusableBuffers)
                .build();

        return loop.decode(inputsEmbeds.dup(), promptTokenIds);
    }

    private boolean hasMythicContent(String text) {
        if (text == null) return false;
        String lower = text.toLowerCase();
        // Match benchmark script criteria exactly
        return lower.contains("mythic heroes")
                || lower.contains("creating a mythic character")
                || lower.contains("hytic heroes");
    }

    private boolean hasStructuralDocTags(String text) {
        if (text == null) return false;
        Set<String> tags = new HashSet<>();
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
            if (!tag.isEmpty()) tags.add(tag);
            idx = close + 1;
        }
        return tags.contains("doctag")
                && (tags.contains("page") || tags.contains("section_header") || tags.contains("text"));
    }

    @Test
    @DisplayName("T1: Fresh allocations restore mythic content on page 10")
    public void testFreshAllocationsRestoreMythicContent() throws Exception {
        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 250);

        // --- Run A: reusable buffers (current code) ---
        log.info("=== T1-A: reusableBuffers=true ===");
        GenerationResult resultReusable = runOldDecoder(true, maxTokens);
        String textReusable = resultReusable.getText();
        int[] tokensReusable = resultReusable.getTokenIds();
        boolean reusableHasMythic = hasMythicContent(textReusable);
        boolean reusableHasStruct = hasStructuralDocTags(textReusable);
        log.info("Reusable buffers: tokens={} mythic={} structural={} text='{}'",
                resultReusable.getGeneratedTokenCount(), reusableHasMythic, reusableHasStruct,
                textReusable.trim().substring(0, Math.min(200, textReusable.trim().length())));

        // --- Run B: fresh allocations (f8e83ff4c9 fix) ---
        log.info("=== T1-B: reusableBuffers=false ===");
        GenerationResult resultFresh = runOldDecoder(false, maxTokens);
        String textFresh = resultFresh.getText();
        int[] tokensFresh = resultFresh.getTokenIds();
        boolean freshHasMythic = hasMythicContent(textFresh);
        boolean freshHasStruct = hasStructuralDocTags(textFresh);
        log.info("Fresh allocations: tokens={} mythic={} structural={} text='{}'",
                resultFresh.getGeneratedTokenCount(), freshHasMythic, freshHasStruct,
                textFresh.trim().substring(0, Math.min(200, textFresh.trim().length())));

        // --- Token stream divergence analysis ---
        int minLen = Math.min(tokensReusable.length, tokensFresh.length);
        int firstDiv = -1;
        int matches = 0;
        for (int i = 0; i < minLen; i++) {
            if (tokensReusable[i] == tokensFresh[i]) {
                matches++;
            } else if (firstDiv < 0) {
                firstDiv = i;
                log.error("FIRST TOKEN DIVERGENCE at step {}: reusable={} ('{}') fresh={} ('{}')",
                        i, tokensReusable[i], tokenizer.decode(new int[]{tokensReusable[i]}, false),
                        tokensFresh[i], tokenizer.decode(new int[]{tokensFresh[i]}, false));
            }
        }
        double matchRate = minLen > 0 ? (double) matches / minLen : 0.0;
        log.info("Token stream comparison: {}/{} match ({}%) firstDivergence={}",
                matches, minLen, String.format("%.1f", matchRate * 100), firstDiv);

        // --- Analysis ---
        log.info("=== T1 RESULT ===");
        log.info("Reusable buffers: mythic={} structural={}", reusableHasMythic, reusableHasStruct);
        log.info("Fresh allocations: mythic={} structural={}", freshHasMythic, freshHasStruct);

        if (firstDiv >= 0) {
            log.error("BUG ISOLATED: Token streams diverge at step {} — reusable buffers cause stale data.", firstDiv);
        }

        if (!reusableHasMythic && freshHasMythic) {
            log.info("T1 PASS: Fresh allocations fixed the regression.");
        } else if (reusableHasMythic && freshHasMythic) {
            log.info("T1 INCONCLUSIVE: Both paths work — bug may be elsewhere.");
        } else if (!reusableHasMythic && !freshHasMythic) {
            log.error("T1 FAIL: Neither path produced mythic content — bug is deeper than buffer reuse.");
        } else {
            log.info("T1 UNEXPECTED: Reusable works but fresh does not.");
        }

        // Hard assertion: if the theory is correct, fresh allocations MUST be correct
        assertTrue(freshHasMythic || freshHasStruct,
                "Fresh allocations should produce correct output on real PDF page 10. "
                        + "If this fails, the reusable-buffer theory is wrong. Text: " + textFresh);
    }
}
