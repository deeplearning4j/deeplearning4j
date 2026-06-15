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
 * Factor-isolation test suite for GenerationPipeline divergence.
 *
 * Compares inputs between StaticKvCacheDecodeLoop (old, correct) and
 * GenerationPipeline (new, degenerate) one factor at a time.
 */
@Slf4j
public class TestGenerationPipelineFactorIsolation {

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

        log.info("Loading SmolDocling models for factor isolation test...");

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
                "platform-tests/pathfinder-mythic.pdf");
        File pdfFile = new File(pdfPath);
        BufferedImage image;
        if (pdfFile.exists()) {
            log.info("Using mythic PDF: {}", pdfFile.getAbsolutePath());
            try (org.apache.pdfbox.pdmodel.PDDocument doc = org.apache.pdfbox.pdmodel.PDDocument.load(pdfFile)) {
                org.apache.pdfbox.rendering.PDFRenderer renderer = new org.apache.pdfbox.rendering.PDFRenderer(doc);
                image = renderer.renderImageWithDPI(0, 150, org.apache.pdfbox.rendering.ImageType.RGB);
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

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(image, targetSize, 9);

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

    // ═══════════════════════════════════════════════════════════════════════════
    // H1: Input map divergence at warmup decode step (step 1 after prefill)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("H1: Compare decoderInputMap between old decoder and GenerationPipeline at step 1")
    public void testH1_InputMapDivergence() throws Exception {
        ensureModelsLoaded();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        int prefillSeqLen = promptTokenIds.length;
        int maxNewTokens = 2; // Just prefill + 1 warmup step
        long maxKvLen = prefillSeqLen + maxNewTokens;

        // ── Old decoder: prefill + build step-1 inputs ──
        decoder.clearPlaceholders(false);
        decoder.clearOpInputs();
        decoder.resetSession();
        Nd4j.getExecutioner().commit();

        StaticKvCacheDecodeLoop oldLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .hiddenSize(hiddenSize)
                .build();

        // Manually run prefill using the old loop's logic
        // We need to replicate what oldLoop.decode() does internally.
        // Instead of calling the full decode, we'll extract the input map
        // after the first decode step by using DecoderInputBuilder directly.

        // Prefill
        INDArray prefillEmbeddings = inputsEmbeds.dup();
        INDArray prefillInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        // Old decoder builds input map via DecoderInputBuilder for step 0 (prefill)
        List<String> decoderInputNames = decoder.inputs();
        Map<String, INDArray> reusableInputs = new HashMap<>();
        boolean dspActive = true;

        Map<String, INDArray> oldPrefillMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                prefillEmbeddings, prefillInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive,
                null, null);

        // Run prefill to get KV caches
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames() != null
                ? ioConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        Map<String, INDArray> prefillOutputs = decoder.output(oldPrefillMap,
                allOutputNames.toArray(new String[0]));

        // Sample first token
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        int firstTokenId = Nd4j.argMax(
                prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits.close();

        // Build static KV buffers
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(keyName);
            staticKvBuffers.put(inputName, padded);
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(valName);
            staticKvBuffers.put(inputName, padded);
        }

        // Build step-1 inputs for old decoder
        INDArray oldEmbeddings = embedTokens(firstTokenId);
        INDArray oldInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        Map<String, INDArray> oldStep1Map = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                oldEmbeddings, oldInputIds,
                prefillSeqLen, 1,
                staticKvBuffers, maxKvLen, prefillSeqLen,
                true, hiddenSize,
                reusableInputs, dspActive,
                null, null);

        // ── New pipeline: replicate the same steps ──
        decoder.clearPlaceholders(false);
        decoder.clearOpInputs();
        decoder.resetSession();
        Nd4j.getExecutioner().commit();

        // Rebuild everything for GenerationPipeline
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .hiddenSize(hiddenSize)
                .build());

        // Run prefill via pipeline (same as old)
        INDArray pipePrefillEmbeddings = inputsEmbeds.dup();
        INDArray pipePrefillInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        Map<String, INDArray> pipePrefillMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                pipePrefillEmbeddings, pipePrefillInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                new HashMap<>(), dspActive,
                null, null);

        Map<String, INDArray> pipePrefillOutputs = decoder.output(pipePrefillMap,
                allOutputNames.toArray(new String[0]));

        INDArray pipePrefillLogits = pipePrefillOutputs.get(ioConfig.getLogitsOutputName());
        int pipeFirstTokenId = Nd4j.argMax(
                pipePrefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(pipePrefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        pipePrefillLogits.close();

        // Build static KV buffers
        Map<String, INDArray> pipeStaticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = pipePrefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(keyName);
            pipeStaticKvBuffers.put(inputName, padded);
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = pipePrefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(valName);
            pipeStaticKvBuffers.put(inputName, padded);
        }

        // Build step-1 inputs for new pipeline (manual, matching GenerationPipeline.generateNative)
        INDArray pipeEmbeddings = embedTokens(pipeFirstTokenId);
        INDArray pipeInputIds = Nd4j.createFromArray(new int[]{pipeFirstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        // Causal mask: [1, 1, 1, maskLen] FLOAT
        long maskLen = maxKvLen + 1;
        INDArray pipeCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maskLen);
        pipeCausalMask.assign(ModelIOConfig.MASK_FILL);
        pipeCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);

        // Attention mask: [1, maskLen] LONG
        INDArray pipeAttentionMask = Nd4j.zeros(DataType.LONG, 1, maskLen);
        for (int i = 0; i < prefillSeqLen; i++) {
            pipeAttentionMask.putScalar(new long[]{0, i}, 1);
        }
        pipeAttentionMask.putScalar(new long[]{0, maskLen - 1}, 1);

        // Position IDs: [1, 1] INT64
        INDArray pipePosIds = Nd4j.createFromArray(new long[]{prefillSeqLen})
                .reshape(1, 1);

        // Build decode input map
        Map<String, INDArray> pipeStep1Map = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) pipeStep1Map.put(embedsName, pipeEmbeddings);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) pipeStep1Map.put(idsName, pipeInputIds);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            pipeStep1Map.put(maskName, pipeAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            pipeStep1Map.put(causalName, pipeCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            pipeStep1Map.put(posName, pipePosIds);
        for (Map.Entry<String, INDArray> entry : pipeStaticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                pipeStep1Map.put(entry.getKey(), entry.getValue());
            }
        }
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(pipeStep1Map.keySet()), decoder, pipeStep1Map);

        // ── Compare input maps ──
        log.info("=== H1: Input Map Comparison ===");
        log.info("Old firstTokenId={}  New firstTokenId={}", firstTokenId, pipeFirstTokenId);
        assertEquals(firstTokenId, pipeFirstTokenId, "Prefill must produce the same first token");

        int mismatches = 0;
        Set<String> allKeys = new TreeSet<>();
        allKeys.addAll(oldStep1Map.keySet());
        allKeys.addAll(pipeStep1Map.keySet());

        for (String key : allKeys) {
            INDArray oldArr = oldStep1Map.get(key);
            INDArray pipeArr = pipeStep1Map.get(key);

            if (oldArr == null && pipeArr == null) {
                log.info("  {}: both null — OK", key);
                continue;
            }
            if (oldArr == null || pipeArr == null) {
                log.error("  {}: old={} pipe={} — MISMATCH", key,
                        oldArr == null ? "null" : Arrays.toString(oldArr.shape()),
                        pipeArr == null ? "null" : Arrays.toString(pipeArr.shape()));
                mismatches++;
                continue;
            }

            if (!Arrays.equals(oldArr.shape(), pipeArr.shape())) {
                log.error("  {}: shapes differ old={} pipe={} — MISMATCH", key,
                        Arrays.toString(oldArr.shape()), Arrays.toString(pipeArr.shape()));
                mismatches++;
                continue;
            }

            if (oldArr.dataType() != pipeArr.dataType()) {
                log.error("  {}: dtypes differ old={} pipe={} — MISMATCH", key,
                        oldArr.dataType(), pipeArr.dataType());
                mismatches++;
                continue;
            }

            // Element-wise comparison with tolerance
            double maxDiff = oldArr.sub(pipeArr).amaxNumber().doubleValue();
            double tolerance = 1e-5;
            if (maxDiff > tolerance) {
                log.error("  {}: maxDiff={} > {} — MISMATCH", key, maxDiff, tolerance);
                // Print first few differing elements
                long[] shape = oldArr.shape();
                long total = oldArr.length();
                int shown = 0;
                for (long i = 0; i < total && shown < 10; i++) {
                    double o = oldArr.getDouble(i);
                    double p = pipeArr.getDouble(i);
                    if (Math.abs(o - p) > tolerance) {
                        log.error("    [{}]: old={} pipe={}", i, o, p);
                        shown++;
                    }
                }
                mismatches++;
            } else {
                log.info("  {}: shape={} dtype={} maxDiff={} — OK", key,
                        Arrays.toString(oldArr.shape()), oldArr.dataType(), maxDiff);
            }
        }

        log.info("H1 Result: {} mismatches / {} keys compared", mismatches, allKeys.size());
        assertEquals(0, mismatches, "Input maps must match perfectly between old decoder and GenerationPipeline");
    }

    private INDArray embedTokens(int tokenId) {
        // Use embedTokens model directly
        INDArray tokenIds = Nd4j.createFromArray(new int[]{tokenId})
                .reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray emb = embedOutputs.values().iterator().next().dup();
        tokenIds.close();
        return emb;
    }

    private static INDArray padKvToStaticSize(INDArray presentKv, long maxKvLen) {
        long[] shape = presentKv.shape();
        long batch = shape[0], heads = shape[1], seqLen = shape[2], dim = shape[3];
        if (seqLen >= maxKvLen) {
            return presentKv.dup();
        }
        INDArray padding = Nd4j.zeros(presentKv.dataType(), batch, heads, maxKvLen - seqLen, dim);
        INDArray result = Nd4j.concat(2, presentKv, padding);
        padding.close();
        return result;
    }
}
