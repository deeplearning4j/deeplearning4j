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
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
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
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Diagnostic breadcrumb tests for the native decode loop divergence.
 *
 * <p>Each test isolates one aspect of the old decoder (StaticKvCacheDecodeLoop,
 * padded layout) vs new pipeline (GenerationPipeline, contiguous layout) to determine
 * which layout is correct and where divergence begins.</p>
 *
 * <p>Run individual tests:</p>
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDecodeLoopDiagnostics#testD1_OldDecoderVsJavaReference_MythicPdf \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestDecodeLoopDiagnostics {

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

        log.info("Loading SmolDocling models for decode loop diagnostic tests...");

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

        int resizeLongestEdge = Integer.getInteger("vlm.test.resizeLongestEdge", 0);
        if (resizeLongestEdge > 0) {
            image = ImageTiler.resizeLongestEdge(image, resizeLongestEdge);
            log.info("Resized mythic image to longestEdge={} -> {}x{}",
                    resizeLongestEdge, image.getWidth(), image.getHeight());
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

        INDArray textEmbeddings = embedPromptDirect(encoded);

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        assertEquals(promptTokenIds.length, inputsEmbeds.size(1),
                "Merged embedding sequence length must match prompt token length");

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize, promptTokenIds.length);
        modelsLoaded = true;
    }

    private static INDArray embedPromptDirect(int[] tokenIds) {
        INDArray inputIds = Nd4j.createFromArray(new int[][]{tokenIds}).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
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
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
        }
    }

    private static INDArray extractEmbeddingTable() {
        for (var varName : embedTokens.variables()) {
            if (varName.name().contains("weight") || varName.name().contains("embed")) {
                INDArray arr = embedTokens.getArrForVarName(varName.name());
                if (arr != null && arr.rank() == 2 && arr.size(1) == hiddenSize) {
                    return arr;
                }
            }
        }
        throw new IllegalStateException("Could not find embedding table in embed_tokens model");
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

    private static void scatterKvToStatic(INDArray presentKv, INDArray staticBuf, long cachePos) {
        long seqLen = presentKv.shape()[2];
        INDArray slice = presentKv.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(slice);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D1: Old decoder (padded layout) vs Java reference (contiguous layout)
    //     on the SAME mythic PDF input.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D1: StaticKvCacheDecodeLoop (padded) vs Java reference (contiguous) on mythic PDF")
    public void testD1_OldDecoderVsJavaReference_MythicPdf() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 10;

        // ── Run old decoder (padded layout via DecoderInputBuilder) ──
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        StaticKvCacheDecodeLoop oldLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult oldResult = oldLoop.decode(inputsEmbeds.dup(), promptTokenIds);
        int[] oldTokens = oldResult.getTokenIds();
        String oldText = oldResult.getText();
        log.info("D1 OLD (padded):  {} tokens, text='{}'", oldTokens.length, oldText);
        log.info("D1 OLD tokens: {}", Arrays.toString(oldTokens));

        // ── Run Java reference (contiguous layout, same as GenerationPipeline) ──
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();
        Map<String, INDArray> reusableInputs = new HashMap<>();
        INDArray prefillEmbeddings = inputsEmbeds.dup();

        // Prefill (same for both)
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                prefillEmbeddings, currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV caches
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(keyName), padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(valName), padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        // Sample first token
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        int firstTokenId = Nd4j.argMax(
                prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits.close();

        // Build contiguous-layout decode arrays (matching GenerationPipeline)
        long maskLen = maxKvLen + 1;
        INDArray decodeAttentionMask = Nd4j.zeros(DataType.LONG, 1, maskLen);
        for (int i = 0; i <= prefillSeqLen; i++) {
            decodeAttentionMask.putScalar(new long[]{0, i}, 1);
        }
        INDArray decodeCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maskLen);
        decodeCausalMask.assign(ModelIOConfig.MASK_FILL);
        decodeCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        INDArray decodePosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        INDArray embeddingTable = extractEmbeddingTable();

        // Decode loop (contiguous layout)
        List<Integer> refTokens = new ArrayList<>();
        refTokens.add(firstTokenId);
        int cachePos = prefillSeqLen;
        int currentToken = firstTokenId;

        for (int step = 0; step < maxTokens - 1; step++) {
            INDArray stepEmbedding = embeddingTable.getRow(currentToken).reshape(1, 1, hiddenSize).dup();
            INDArray stepInputIds = Nd4j.createFromArray(new int[]{currentToken})
                    .reshape(1, 1).castTo(DataType.INT64);

            Map<String, INDArray> decodeInputMap = new HashMap<>();
            String embedsName = ioConfig.getInputEmbeddingsName();
            if (embedsName != null) decodeInputMap.put(embedsName, stepEmbedding);
            String idsName = ioConfig.getInputIdsName();
            if (idsName != null) decodeInputMap.put(idsName, stepInputIds);
            String maskName = ioConfig.getAttentionMaskName();
            if (maskName != null && decoder.hasVariable(maskName))
                decodeInputMap.put(maskName, decodeAttentionMask);
            String causalName = ioConfig.getCausalMaskName();
            if (causalName != null && decoder.hasVariable(causalName))
                decodeInputMap.put(causalName, decodeCausalMask);
            String posName = ioConfig.getPositionIdsName();
            if (posName != null && decoder.hasVariable(posName))
                decodeInputMap.put(posName, decodePosIds);
            for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
                if (decoder.hasVariable(entry.getKey())) {
                    decodeInputMap.put(entry.getKey(), entry.getValue());
                }
            }
            DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                    new ArrayList<>(decodeInputMap.keySet()), decoder, decodeInputMap);

            Map<String, INDArray> decodeOutputs = decoder.output(
                    decodeInputMap, allOutputNames.toArray(new String[0]));

            INDArray logits = decodeOutputs.get(ioConfig.getLogitsOutputName());
            int nextToken = Nd4j.argMax(
                    logits.get(NDArrayIndex.point(0),
                            NDArrayIndex.point(0),
                            NDArrayIndex.all()).dup(), -1).getInt(0);
            logits.close();

            refTokens.add(nextToken);

            // Scatter KV
            for (String keyName : kvNames.keyNames) {
                INDArray presentKv = decodeOutputs.get(keyName);
                INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(keyName));
                if (presentKv != null && staticBuf != null) {
                    scatterKvToStatic(presentKv, staticBuf, cachePos);
                }
            }
            for (String valName : kvNames.valueNames) {
                INDArray presentKv = decodeOutputs.get(valName);
                INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(valName));
                if (presentKv != null && staticBuf != null) {
                    scatterKvToStatic(presentKv, staticBuf, cachePos);
                }
            }

            cachePos++;
            currentToken = nextToken;

            if (cachePos < maskLen) {
                decodeAttentionMask.putScalar(new long[]{0, cachePos}, 1);
                decodeCausalMask.putScalar(new long[]{0, 0, 0, cachePos}, 0.0f);
            }
            decodePosIds.putScalar(new long[]{0, 0}, cachePos);

            if (nextToken == tokenizer.getEosTokenId()) break;
        }

        int[] refTokenArray = refTokens.stream().mapToInt(Integer::intValue).toArray();
        String refText = tokenizer.decode(refTokenArray, true);
        log.info("D1 REF (contiguous): {} tokens, text='{}'", refTokenArray.length, refText);
        log.info("D1 REF tokens: {}", Arrays.toString(refTokenArray));

        // Compare
        int minLen = Math.min(oldTokens.length, refTokenArray.length);
        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (oldTokens[i] == refTokenArray[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }
        double matchRate = minLen > 0 ? (double) matches / minLen : 0.0;
        log.info("D1 Match rate: {}/{} ({}%) firstDivergent={}",
                matches, minLen, String.format("%.1f", matchRate * 100), firstDivergent);

        if (firstDivergent >= 0) {
            log.error("D1 FIRST DIVERGENCE at step {}: old={} ('{}') ref={} ('{}')",
                    firstDivergent,
                    oldTokens[firstDivergent],
                    tokenizer.decode(new int[]{oldTokens[firstDivergent]}, false),
                    refTokenArray[firstDivergent],
                    tokenizer.decode(new int[]{refTokenArray[firstDivergent]}, false));
        }

        // If old decoder diverges from contiguous reference, the padded layout is suspect.
        // We don't hard-fail — this is diagnostic.
        log.info("D1 CONCLUSION: oldDecoder vs contiguousReference matchRate={}", matchRate);
        if (firstDivergent >= 0) {
            log.info("D1 --> Padded layout (old decoder) DIVERGES from contiguous layout at step {}",
                    firstDivergent);
        } else {
            log.info("D1 --> Padded and contiguous layouts produce IDENTICAL tokens");
        }

        // Cleanup
        for (INDArray arr : staticKvBuffers.values()) arr.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D2: Padded vs contiguous layout on the SAME decode step
    //     Build both input maps, run decoder.output(), compare logits.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D2: DecoderInputBuilder padded vs contiguous layout logits comparison")
    public void testD2_PaddedVsContiguousLogits() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 5;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // ── Prefill ──
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV caches
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(keyName), padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(valName), padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        // Sample first token
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        int firstTokenId = Nd4j.argMax(
                prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits.close();

        // ── Warmup decode step: build BOTH layouts, compare ──
        INDArray embeddingTable = extractEmbeddingTable();
        INDArray stepEmbedding = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray stepInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        // --- PADDED layout (old decoder style) ---
        Map<String, INDArray> reusableInputsPadded = new HashMap<>();
        Map<String, INDArray> paddedInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                stepEmbedding, stepInputIds,
                prefillSeqLen, 1,
                staticKvBuffers, maxKvLen, prefillSeqLen,
                true, hiddenSize,
                reusableInputsPadded, true);  // dspActive=true → usePadded=true

        // Log the padded mask
        INDArray paddedMask = paddedInputMap.get(ioConfig.getAttentionMaskName());
        INDArray paddedCausal = paddedInputMap.get(ioConfig.getCausalMaskName());
        log.info("D2 PADDED attention mask shape: {} nonZeroCount={}",
                Arrays.toString(paddedMask.shape()), paddedMask.neq(0).sumNumber().intValue());
        log.info("D2 PADDED causal mask shape: {}", Arrays.toString(paddedCausal.shape()));

        // --- CONTIGUOUS layout (GenerationPipeline style) ---
        long maskLen = maxKvLen + 1;
        INDArray contAttentionMask = Nd4j.zeros(DataType.LONG, 1, maskLen);
        for (int i = 0; i <= prefillSeqLen; i++) {
            contAttentionMask.putScalar(new long[]{0, i}, 1);
        }
        INDArray contCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maskLen);
        contCausalMask.assign(ModelIOConfig.MASK_FILL);
        contCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        INDArray contPosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        Map<String, INDArray> contiguousInputMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) contiguousInputMap.put(embedsName, stepEmbedding);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) contiguousInputMap.put(idsName, stepInputIds);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            contiguousInputMap.put(maskName, contAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            contiguousInputMap.put(causalName, contCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            contiguousInputMap.put(posName, contPosIds);
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                contiguousInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(contiguousInputMap.keySet()), decoder, contiguousInputMap);

        log.info("D2 CONTIGUOUS attention mask shape: {} nonZeroCount={}",
                Arrays.toString(contAttentionMask.shape()), contAttentionMask.neq(0).sumNumber().intValue());
        log.info("D2 CONTIGUOUS causal mask shape: {}", Arrays.toString(contCausalMask.shape()));

        // --- Run both and compare logits ---
        Map<String, INDArray> paddedOutputs = decoder.output(
                paddedInputMap, allOutputNames.toArray(new String[0]));
        Map<String, INDArray> contiguousOutputs = decoder.output(
                contiguousInputMap, allOutputNames.toArray(new String[0]));

        INDArray paddedLogits = paddedOutputs.get(ioConfig.getLogitsOutputName());
        INDArray contiguousLogits = contiguousOutputs.get(ioConfig.getLogitsOutputName());

        // Extract [0,0,:] logits for both
        INDArray paddedLogitsSlice = paddedLogits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();
        INDArray contLogitsSlice = contiguousLogits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();

        int paddedArgmax = Nd4j.argMax(paddedLogitsSlice, -1).getInt(0);
        int contArgmax = Nd4j.argMax(contLogitsSlice, -1).getInt(0);

        INDArray diff = org.nd4j.linalg.ops.transforms.Transforms.abs(paddedLogitsSlice.sub(contLogitsSlice), false);
        double maxDiff = diff.maxNumber().doubleValue();
        double meanDiff = diff.meanNumber().doubleValue();

        log.info("D2 PADDED   argmax token: {} ('{}')", paddedArgmax,
                tokenizer.decode(new int[]{paddedArgmax}, false));
        log.info("D2 CONTIG   argmax token: {} ('{}')", contArgmax,
                tokenizer.decode(new int[]{contArgmax}, false));
        log.info("D2 Logits maxAbsDiff: {} meanAbsDiff: {}", maxDiff, meanDiff);

        // Compare top-5 tokens for both (using argMax on sorted logits)
        int[] paddedTop5 = getTopKIndices(paddedLogitsSlice, 5);
        int[] contTop5 = getTopKIndices(contLogitsSlice, 5);
        log.info("D2 PADDED top-5 tokens: {}", Arrays.toString(paddedTop5));
        log.info("D2 CONTIG top-5 tokens: {}", Arrays.toString(contTop5));

        // Cleanup
        paddedLogits.close();
        contiguousLogits.close();
        for (INDArray arr : staticKvBuffers.values()) arr.close();
        for (INDArray arr : reusableInputsPadded.values()) arr.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D3: Print actual attention mask values from both paths at step 2
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D3: Attention mask value inspection at step 2")
    public void testD3_AttentionMaskValues_Step2() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 5;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // Prefill
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(keyName), padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(valName), padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        int firstTokenId = Nd4j.argMax(
                prefillOutputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillOutputs.get(ioConfig.getLogitsOutputName()).shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillOutputs.get(ioConfig.getLogitsOutputName()).close();

        INDArray embeddingTable = extractEmbeddingTable();

        // ── Step 1 (warmup): run with padded layout, capture mask ──
        Map<String, INDArray> reusableInputsPadded = new HashMap<>();
        INDArray step1Embed = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray step1Ids = Nd4j.createFromArray(new int[]{firstTokenId}).reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> step1InputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                step1Embed, step1Ids,
                prefillSeqLen, 1,
                staticKvBuffers, maxKvLen, prefillSeqLen,
                true, hiddenSize,
                reusableInputsPadded, true);

        INDArray step1Mask = step1InputMap.get(ioConfig.getAttentionMaskName()).dup();
        INDArray step1Causal = step1InputMap.get(ioConfig.getCausalMaskName()).dup();
        log.info("D3 STEP 1 PADDED attentionMask shape={} values(first 20): {}",
                Arrays.toString(step1Mask.shape()),
                step1Mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, Math.min(20, (int) step1Mask.size(1)))));
        log.info("D3 STEP 1 PADDED causalMask shape={} values: {}",
                Arrays.toString(step1Causal.shape()),
                step1Causal.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                        NDArrayIndex.interval(0, Math.min(20, (int) step1Causal.size(3)))));

        Map<String, INDArray> step1Outputs = decoder.output(
                step1InputMap, allOutputNames.toArray(new String[0]));
        int step1Token = Nd4j.argMax(
                step1Outputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup(), -1).getInt(0);

        // Scatter KV for step 1
        for (String keyName : kvNames.keyNames) {
            scatterKvToStatic(step1Outputs.get(keyName),
                    staticKvBuffers.get(ioConfig.presentToInputName(keyName)), prefillSeqLen);
        }
        for (String valName : kvNames.valueNames) {
            scatterKvToStatic(step1Outputs.get(valName),
                    staticKvBuffers.get(ioConfig.presentToInputName(valName)), prefillSeqLen);
        }

        // ── Step 2: build padded layout mask, print values ──
        INDArray step2Embed = embeddingTable.getRow(step1Token).reshape(1, 1, hiddenSize).dup();
        INDArray step2Ids = Nd4j.createFromArray(new int[]{step1Token}).reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> step2InputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                step2Embed, step2Ids,
                prefillSeqLen + 1, 1,
                staticKvBuffers, maxKvLen, prefillSeqLen + 1,
                true, hiddenSize,
                reusableInputsPadded, true);

        INDArray step2Mask = step2InputMap.get(ioConfig.getAttentionMaskName()).dup();
        INDArray step2Causal = step2InputMap.get(ioConfig.getCausalMaskName()).dup();
        log.info("D3 STEP 2 PADDED attentionMask shape={} nonZero={}",
                Arrays.toString(step2Mask.shape()), step2Mask.neq(0).sumNumber().intValue());
        log.info("D3 STEP 2 PADDED causalMask shape={}", Arrays.toString(step2Causal.shape()));

        // Print first 30 values of each mask
        long maskLen = step2Mask.size(1);
        log.info("D3 STEP 2 PADDED attentionMask[0:30]: {}",
                step2Mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, Math.min(30, (int) maskLen))));
        log.info("D3 STEP 2 PADDED causalMask[0:30]: {}",
                step2Causal.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                        NDArrayIndex.interval(0, Math.min(30, (int) step2Causal.size(3)))));

        // Also show what contiguous layout would look like at step 2
        INDArray contMask = Nd4j.zeros(DataType.LONG, 1, maxKvLen + 1);
        for (int i = 0; i <= prefillSeqLen + 1; i++) {
            contMask.putScalar(new long[]{0, i}, 1);
        }
        log.info("D3 STEP 2 CONTIGUOUS attentionMask[0:30]: {}",
                contMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, Math.min(30, (int) contMask.size(1)))));

        // Cleanup
        for (INDArray arr : staticKvBuffers.values()) arr.close();
        step1Mask.close();
        step1Causal.close();
        step2Mask.close();
        step2Causal.close();
        contMask.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D4: Prefill output comparison (both use same path, should match)
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D4: Prefill outputs match between old decoder path and new pipeline path")
    public void testD4_PrefillOutputMatch() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 5;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // Run prefill once
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Sample first token
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        int firstTokenId = Nd4j.argMax(
                prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits.close();

        log.info("D4 Prefill first token: {} ('{}')", firstTokenId,
                tokenizer.decode(new int[]{firstTokenId}, false));

        // Now run prefill AGAIN with fresh state (like old decoder does on its own)
        BenchmarkConfigApplier.resetModelState(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        Map<String, INDArray> reusableInputs2 = new HashMap<>();
        Map<String, INDArray> prefillInputMap2 = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs2, dspActive);
        Map<String, INDArray> prefillOutputs2 = decoder.output(
                prefillInputMap2, allOutputNames.toArray(new String[0]));

        INDArray prefillLogits2 = prefillOutputs2.get(ioConfig.getLogitsOutputName());
        int firstTokenId2 = Nd4j.argMax(
                prefillLogits2.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits2.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits2.close();

        log.info("D4 Prefill first token (second run): {} ('{}')", firstTokenId2,
                tokenizer.decode(new int[]{firstTokenId2}, false));

        assertEquals(firstTokenId, firstTokenId2,
                "D4: Prefill is not deterministic across runs. This is unexpected.");

        // Compare KV cache shapes
        for (String keyName : kvNames.keyNames) {
            INDArray kv1 = prefillOutputs.get(keyName);
            INDArray kv2 = prefillOutputs2.get(keyName);
            assertArrayEquals(kv1.shape(), kv2.shape(),
                    "D4: Prefill KV shape mismatch for " + keyName);
        }

        // Cleanup
        for (INDArray arr : prefillOutputs.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
        for (INDArray arr : prefillOutputs2.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D5: Warmup decode step comparison
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D5: Warmup decode step tokens and logits comparison")
    public void testD5_WarmupDecodeStepComparison() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 5;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // Prefill
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(keyName), padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(valName), padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        int firstTokenId = Nd4j.argMax(
                prefillOutputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillOutputs.get(ioConfig.getLogitsOutputName()).shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillOutputs.get(ioConfig.getLogitsOutputName()).close();

        INDArray embeddingTable = extractEmbeddingTable();

        // --- Old decoder style warmup (padded layout) ---
        Map<String, INDArray> reusableInputsPadded = new HashMap<>();
        INDArray stepEmbed = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray stepIds = Nd4j.createFromArray(new int[]{firstTokenId}).reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> paddedInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                stepEmbed, stepIds,
                prefillSeqLen, 1,
                staticKvBuffers, maxKvLen, prefillSeqLen,
                true, hiddenSize,
                reusableInputsPadded, true);
        Map<String, INDArray> paddedOutputs = decoder.output(
                paddedInputMap, allOutputNames.toArray(new String[0]));
        INDArray paddedLogits = paddedOutputs.get(ioConfig.getLogitsOutputName());
        int paddedToken = Nd4j.argMax(
                paddedLogits.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup(),
                -1).getInt(0);

        // --- New pipeline style warmup (contiguous layout) ---
        long maskLen = maxKvLen + 1;
        INDArray contAttentionMask = Nd4j.zeros(DataType.LONG, 1, maskLen);
        for (int i = 0; i <= prefillSeqLen; i++) {
            contAttentionMask.putScalar(new long[]{0, i}, 1);
        }
        INDArray contCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maskLen);
        contCausalMask.assign(ModelIOConfig.MASK_FILL);
        contCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        INDArray contPosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        Map<String, INDArray> contiguousInputMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) contiguousInputMap.put(embedsName, stepEmbed);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) contiguousInputMap.put(idsName, stepIds);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            contiguousInputMap.put(maskName, contAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            contiguousInputMap.put(causalName, contCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            contiguousInputMap.put(posName, contPosIds);
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                contiguousInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(contiguousInputMap.keySet()), decoder, contiguousInputMap);

        Map<String, INDArray> contiguousOutputs = decoder.output(
                contiguousInputMap, allOutputNames.toArray(new String[0]));
        INDArray contiguousLogits = contiguousOutputs.get(ioConfig.getLogitsOutputName());
        int contiguousToken = Nd4j.argMax(
                contiguousLogits.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup(),
                -1).getInt(0);

        log.info("D5 Warmup decode step 1:");
        log.info("  PADDED    token: {} ('{}')", paddedToken,
                tokenizer.decode(new int[]{paddedToken}, false));
        log.info("  CONTIGUOUS token: {} ('{}')", contiguousToken,
                tokenizer.decode(new int[]{contiguousToken}, false));

        INDArray paddedLogitsSlice = paddedLogits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();
        INDArray contLogitsSlice = contiguousLogits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();
        INDArray diff = org.nd4j.linalg.ops.transforms.Transforms.abs(paddedLogitsSlice.sub(contLogitsSlice), false);
        log.info("D5 Warmup logits maxAbsDiff: {} meanAbsDiff: {}",
                diff.maxNumber().doubleValue(), diff.meanNumber().doubleValue());

        // Cleanup
        paddedLogits.close();
        contiguousLogits.close();
        for (INDArray arr : staticKvBuffers.values()) arr.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D6: KV cache state after warmup
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D6: KV cache state comparison after warmup decode step")
    public void testD6_KvCacheAfterWarmup() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 5;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // Prefill
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV caches for BOTH paths (independent copies)
        Map<String, INDArray> staticKvBuffersPadded = new LinkedHashMap<>();
        Map<String, INDArray> staticKvBuffersContig = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            staticKvBuffersPadded.put(ioConfig.presentToInputName(keyName), padKvToStaticSize(presentKv, maxKvLen));
            staticKvBuffersContig.put(ioConfig.presentToInputName(keyName), padKvToStaticSize(presentKv, maxKvLen));
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            staticKvBuffersPadded.put(ioConfig.presentToInputName(valName), padKvToStaticSize(presentKv, maxKvLen));
            staticKvBuffersContig.put(ioConfig.presentToInputName(valName), padKvToStaticSize(presentKv, maxKvLen));
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        int firstTokenId = Nd4j.argMax(
                prefillOutputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillOutputs.get(ioConfig.getLogitsOutputName()).shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillOutputs.get(ioConfig.getLogitsOutputName()).close();

        INDArray embeddingTable = extractEmbeddingTable();

        // --- Run warmup with PADDED layout, scatter into padded buffers ---
        Map<String, INDArray> reusableInputsPadded = new HashMap<>();
        INDArray stepEmbed = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray stepIds = Nd4j.createFromArray(new int[]{firstTokenId}).reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> paddedInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                stepEmbed, stepIds,
                prefillSeqLen, 1,
                staticKvBuffersPadded, maxKvLen, prefillSeqLen,
                true, hiddenSize,
                reusableInputsPadded, true);
        Map<String, INDArray> paddedOutputs = decoder.output(
                paddedInputMap, allOutputNames.toArray(new String[0]));

        // Scatter into padded buffers
        for (String keyName : kvNames.keyNames) {
            scatterKvToStatic(paddedOutputs.get(keyName),
                    staticKvBuffersPadded.get(ioConfig.presentToInputName(keyName)), prefillSeqLen);
        }
        for (String valName : kvNames.valueNames) {
            scatterKvToStatic(paddedOutputs.get(valName),
                    staticKvBuffersPadded.get(ioConfig.presentToInputName(valName)), prefillSeqLen);
        }

        // --- Run warmup with CONTIGUOUS layout, scatter into contig buffers ---
        long maskLen = maxKvLen + 1;
        INDArray contAttentionMask = Nd4j.zeros(DataType.LONG, 1, maskLen);
        for (int i = 0; i <= prefillSeqLen; i++) {
            contAttentionMask.putScalar(new long[]{0, i}, 1);
        }
        INDArray contCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maskLen);
        contCausalMask.assign(ModelIOConfig.MASK_FILL);
        contCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        INDArray contPosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        Map<String, INDArray> contiguousInputMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) contiguousInputMap.put(embedsName, stepEmbed);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) contiguousInputMap.put(idsName, stepIds);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            contiguousInputMap.put(maskName, contAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            contiguousInputMap.put(causalName, contCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            contiguousInputMap.put(posName, contPosIds);
        for (Map.Entry<String, INDArray> entry : staticKvBuffersContig.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                contiguousInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(contiguousInputMap.keySet()), decoder, contiguousInputMap);

        Map<String, INDArray> contiguousOutputs = decoder.output(
                contiguousInputMap, allOutputNames.toArray(new String[0]));

        // Scatter into contig buffers
        for (String keyName : kvNames.keyNames) {
            scatterKvToStatic(contiguousOutputs.get(keyName),
                    staticKvBuffersContig.get(ioConfig.presentToInputName(keyName)), prefillSeqLen);
        }
        for (String valName : kvNames.valueNames) {
            scatterKvToStatic(contiguousOutputs.get(valName),
                    staticKvBuffersContig.get(ioConfig.presentToInputName(valName)), prefillSeqLen);
        }

        // Compare KV cache states
        for (String keyName : kvNames.keyNames) {
            String inputName = ioConfig.presentToInputName(keyName);
            INDArray paddedKv = staticKvBuffersPadded.get(inputName);
            INDArray contigKv = staticKvBuffersContig.get(inputName);
            INDArray kvDiff = org.nd4j.linalg.ops.transforms.Transforms.abs(paddedKv.sub(contigKv), false);
            double maxKvDiff = kvDiff.maxNumber().doubleValue();
            log.info("D6 KV cache '{}' maxAbsDiff after warmup: {}", inputName, maxKvDiff);
            if (maxKvDiff > 1e-5) {
                log.warn("D6 KV cache divergence detected for {}!", inputName);
            }
        }
        for (String valName : kvNames.valueNames) {
            String inputName = ioConfig.presentToInputName(valName);
            INDArray paddedKv = staticKvBuffersPadded.get(inputName);
            INDArray contigKv = staticKvBuffersContig.get(inputName);
            INDArray kvDiff = org.nd4j.linalg.ops.transforms.Transforms.abs(paddedKv.sub(contigKv), false);
            double maxKvDiff = kvDiff.maxNumber().doubleValue();
            log.info("D6 KV cache '{}' maxAbsDiff after warmup: {}", inputName, maxKvDiff);
            if (maxKvDiff > 1e-5) {
                log.warn("D6 KV cache divergence detected for {}!", inputName);
            }
        }

        // Cleanup
        for (INDArray arr : staticKvBuffersPadded.values()) arr.close();
        for (INDArray arr : staticKvBuffersContig.values()) arr.close();
        paddedOutputs.get(ioConfig.getLogitsOutputName()).close();
        contiguousOutputs.get(ioConfig.getLogitsOutputName()).close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D7: Native decode loop with padded layout inputs
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D7: Native decode loop with padded-layout inputs")
    public void testD7_NativeDecodeWithPaddedLayout() throws Exception {
        ensureModelsLoaded();

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        int maxTokens = 10;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();

        // Prefill
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);
        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Pad KV
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(keyName), padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            staticKvBuffers.put(ioConfig.presentToInputName(valName), padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        int firstTokenId = Nd4j.argMax(
                prefillOutputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillOutputs.get(ioConfig.getLogitsOutputName()).shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillOutputs.get(ioConfig.getLogitsOutputName()).close();

        INDArray embeddingTable = extractEmbeddingTable();

        // Build PADDED layout decode arrays (mimic old decoder)
        long totalSeqLen = maxKvLen + 1;
        INDArray paddedAttentionMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        paddedAttentionMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, prefillSeqLen)).assign(1);
        paddedAttentionMask.putScalar(0, totalSeqLen - 1, 1);

        INDArray paddedCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
        paddedCausalMask.assign(ModelIOConfig.MASK_FILL);
        paddedCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);

        INDArray paddedPosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        // Decode step 1 with padded layout
        INDArray step1Embed = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray step1Ids = Nd4j.createFromArray(new int[]{firstTokenId}).reshape(1, 1).castTo(DataType.INT64);

        Map<String, INDArray> paddedDecodeMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) paddedDecodeMap.put(embedsName, step1Embed);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) paddedDecodeMap.put(idsName, step1Ids);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            paddedDecodeMap.put(maskName, paddedAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            paddedDecodeMap.put(causalName, paddedCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            paddedDecodeMap.put(posName, paddedPosIds);
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                paddedDecodeMap.put(entry.getKey(), entry.getValue());
            }
        }
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(paddedDecodeMap.keySet()), decoder, paddedDecodeMap);

        Map<String, INDArray> paddedOutputs = decoder.output(
                paddedDecodeMap, allOutputNames.toArray(new String[0]));
        int paddedToken = Nd4j.argMax(
                paddedOutputs.get(ioConfig.getLogitsOutputName()).get(
                        NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup(), -1).getInt(0);

        log.info("D7 Padded layout warmup token: {} ('{}')", paddedToken,
                tokenizer.decode(new int[]{paddedToken}, false));

        // Now run the OLD decoder for comparison
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        StaticKvCacheDecodeLoop oldLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(3)
                .hiddenSize(hiddenSize)
                .build();
        GenerationResult oldResult = oldLoop.decode(inputsEmbeds.dup(), promptTokenIds);
        int[] oldTokens = oldResult.getTokenIds();
        log.info("D7 Old decoder tokens: {}", Arrays.toString(oldTokens));

        if (oldTokens.length >= 2) {
            log.info("D7 Comparison: paddedLayoutWarmup={} oldDecoderStep1={} match={}",
                    paddedToken, oldTokens[1], paddedToken == oldTokens[1]);
        }

        // Cleanup
        for (INDArray arr : staticKvBuffers.values()) arr.close();
        paddedOutputs.get(ioConfig.getLogitsOutputName()).close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D8: Tokenizer decode on specific tokens
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D8: Tokenizer decode consistency for known divergence tokens")
    public void testD8_TokenizerDecodeConsistency() throws Exception {
        ensureModelsLoaded();

        int[] tokensToCheck = {49206, 198, 49229, 216, 49205, 49207};
        for (int tok : tokensToCheck) {
            String decoded = tokenizer.decode(new int[]{tok}, false);
            String decodedWithSpecial = tokenizer.decode(new int[]{tok}, true);
            log.info("D8 Token {}: decoded='{}' withSpecial='{}'", tok, decoded, decodedWithSpecial);
        }

        // Also decode pairs to see multi-token effects
        log.info("D8 Token pair [216,49229]: '{}'",
                tokenizer.decode(new int[]{216, 49229}, false));
        log.info("D8 Token pair [216,49206]: '{}'",
                tokenizer.decode(new int[]{216, 49206}, false));
        log.info("D8 Token pair [216,198]: '{}'",
                tokenizer.decode(new int[]{216, 198}, false));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private static int[] getTopKIndices(INDArray logits, int k) {
        // Simple top-k: flatten, copy to array, sort by value descending
        float[] values = logits.toFloatVector();
        int[] indices = new int[values.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        // Sort indices by values descending
        Integer[] boxed = new Integer[indices.length];
        for (int i = 0; i < boxed.length; i++) boxed[i] = i;
        Arrays.sort(boxed, (a, b) -> Float.compare(values[b], values[a]));
        int[] result = new int[Math.min(k, boxed.length)];
        for (int i = 0; i < result.length; i++) result[i] = boxed[i];
        return result;
    }
}
