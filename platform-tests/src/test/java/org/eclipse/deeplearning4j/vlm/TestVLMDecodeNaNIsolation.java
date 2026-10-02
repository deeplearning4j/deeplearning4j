/*
 *  ******************************************************************************
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
package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.InGraphKvDecoderInputs;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.loading.SameDiffOptimizationCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * NaN regression for the SmolDocling decoder through the DSP plan, over a real image-and-text
 * prompt: the prefill and the first decode step after it must produce finite logits and leave
 * no NaN in any plan slot, and the optimized decoder must predict the same next token as the
 * unoptimized one. On failure, {@link DspHandle#firstNaNSlot()} and
 * {@link DspHandle#snapshotAllSlots()} name the slots that hold NaN.
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestVLMDecodeNaNIsolation \
 *     -Dbackend.artifactId=nd4j-cuda-13.1 2&gt;&amp;1 | tee /tmp/vlm-nan-isolation.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestVLMDecodeNaNIsolation {

    private SameDiff decoder;
    private SameDiff embedTokens;
    private HuggingFaceTokenizer tokenizer;
    private INDArray inputsEmbeds;
    private long hiddenSize;
    private String decoderPath;
    private String optimizerEnabled;
    private String optimizerFp16;

    @BeforeAll
    public void setup() throws Exception {
        optimizerEnabled = System.getProperty("nd4j.optimizer.enabled");
        optimizerFp16 = System.getProperty("nd4j.optimizer.fp16");
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");

        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        decoderPath = decoderResult.getModelFile().getAbsolutePath();

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(), decoderPath,
                embedResult.getModelFile().getAbsolutePath());
        SameDiff visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        // Build test image
        int targetSize = 512;
        BufferedImage testImage = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, targetSize, targetSize);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("NaN Isolation Test", 50, 100);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, targetSize, 9);

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
            INDArray pixelMask = null;
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    pixelMask = ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize);
                    visionInputMap.put(inputName, pixelMask);
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());
            closeAll(visionOutputs.values());
            singleFrame.close();
            if (pixelMask != null) {
                pixelMask.close();
            }
        }
        imageInput.close();
        // The vision encoder is needed only for these embeddings
        visionEncoder.close();

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
        closeAll(frameEmbeddings);

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] encoded = tokenizer.encode(chatPrompt, false).getIds();

        INDArray tokenIds = Nd4j.createFromArray(new int[][]{encoded}).castTo(DataType.INT64);
        INDArray textEmbeddings = embed(tokenIds);
        tokenIds.close();

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        textEmbeddings.close();
        visionEmbeddings.close();
        hiddenSize = inputsEmbeds.size(2);

        log.info("Setup complete: decoder={} ops, promptTokens={}, embedShape={}",
                decoder.getOps().size(), encoded.length, inputsEmbeds.shapeInfoToString());
    }

    @AfterAll
    public void tearDown() {
        if (inputsEmbeds != null) inputsEmbeds.close();
        if (decoder != null) decoder.close();
        if (embedTokens != null) embedTokens.close();
        restoreProperty("nd4j.optimizer.enabled", optimizerEnabled);
        restoreProperty("nd4j.optimizer.fp16", optimizerFp16);
    }

    @Test
    @DisplayName("Prefill: finite logits and no NaN slot")
    void testPrefillSlotNaNInspection() {
        long seqLen = inputsEmbeds.size(1);
        Map<String, INDArray> caches = InGraphKvDecoderInputs.kvBuffers(decoder, seqLen + 1);
        Map<String, INDArray> inputs = stepInputs(decoder, inputsEmbeds, 0, caches);
        try {
            INDArray logits = logits(decoder, inputs);
            assertFinite("prefill logits", logits);
            logits.close();
            assertNoNaNSlot("prefill", inputs);
        } finally {
            closeInputs(inputs, caches);
        }
    }

    @Test
    @DisplayName("Decode step 1 after the prefill: finite logits and no NaN slot")
    void testDecodeStep1NaNIsolation() {
        long seqLen = inputsEmbeds.size(1);
        Map<String, INDArray> caches = InGraphKvDecoderInputs.kvBuffers(decoder, seqLen + 1);
        Map<String, INDArray> prefillInputs = stepInputs(decoder, inputsEmbeds, 0, caches);
        Map<String, INDArray> decodeInputs = null;
        INDArray decodeEmbeds = null;
        try {
            // The prefill writes the caches the decode step reads
            INDArray prefillLogits = logits(decoder, prefillInputs);
            assertFinite("prefill logits", prefillLogits);
            int firstToken = lastPositionArgMax(prefillLogits);
            prefillLogits.close();
            log.info("Prefill first token: {} ({})", firstToken, tokenizer.decode(new int[]{firstToken}));

            INDArray tokenId = Nd4j.createFromArray(new long[][]{{firstToken}});
            decodeEmbeds = embed(tokenId);
            tokenId.close();
            decodeInputs = stepInputs(decoder, decodeEmbeds, seqLen, caches);
            INDArray decodeLogits = logits(decoder, decodeInputs);
            assertFinite("decode step 1 logits", decodeLogits);
            decodeLogits.close();
            assertNoNaNSlot("decode step 1", decodeInputs);
        } finally {
            closeInputs(prefillInputs, null);
            if (decodeInputs != null) {
                closeInputs(decodeInputs, null);
            }
            if (decodeEmbeds != null) {
                decodeEmbeds.close();
            }
            closeAll(caches.values());
        }
    }

    @Test
    @DisplayName("Prefill without the optimizer: finite logits and the optimized decoder's next token")
    void testPrefillNoOptimizerNaNComparison() throws Exception {
        long seqLen = inputsEmbeds.size(1);
        int optimizedToken = prefillNextToken(decoder, seqLen);

        // The unoptimized decoder: the cached import with the in-place KV rewrite only
        String enabled = System.getProperty(SameDiffOptimizationCache.OPTIMIZER_ENABLED_PROPERTY);
        System.setProperty(SameDiffOptimizationCache.OPTIMIZER_ENABLED_PROPERTY, "false");
        SameDiff unoptimized;
        try {
            unoptimized = OnnxModelCache.importWithCache(decoderPath);
        } finally {
            restoreProperty(SameDiffOptimizationCache.OPTIMIZER_ENABLED_PROPERTY, enabled);
        }
        try {
            assertTrue(unoptimized != decoder && unoptimized.getOps().size() > decoder.getOps().size(),
                    "expected the unoptimized graph (" + unoptimized.getOps().size()
                            + " ops) to differ from the optimized decoder (" + decoder.getOps().size() + " ops)");
            int unoptimizedToken = prefillNextToken(unoptimized, seqLen);
            log.info("Next token: optimized={} ({}) unoptimized={} ({})",
                    optimizedToken, tokenizer.decode(new int[]{optimizedToken}),
                    unoptimizedToken, tokenizer.decode(new int[]{unoptimizedToken}));
            assertEquals(unoptimizedToken, optimizedToken,
                    "the optimized decoder predicts a different next token than the unoptimized one");
        } finally {
            unoptimized.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** The prompt's next token from a prefill on {@code model}, whose logits must be finite. */
    private int prefillNextToken(SameDiff model, long seqLen) {
        Map<String, INDArray> caches = InGraphKvDecoderInputs.kvBuffers(model, seqLen + 1);
        Map<String, INDArray> inputs = stepInputs(model, inputsEmbeds, 0, caches);
        try {
            INDArray logits = logits(model, inputs);
            assertFinite("prefill logits (" + model.getOps().size() + " ops)", logits);
            int token = lastPositionArgMax(logits);
            logits.close();
            return token;
        } finally {
            closeInputs(inputs, caches);
        }
    }

    /**
     * Step inputs with the embeddings in the model's declared dtype (the input builder passes them
     * through as given).
     */
    private Map<String, INDArray> stepInputs(SameDiff model, INDArray embeddings, long cachePos,
                                             Map<String, INDArray> caches) {
        String embedsName = ModelIOConfig.discover(model).getInputEmbeddingsName();
        DataType declared = model.getVariable(embedsName).dataType();
        INDArray typed = embeddings.dataType() == declared ? embeddings : embeddings.castTo(declared);
        Map<String, INDArray> inputs = InGraphKvDecoderInputs.stepInputs(model, hiddenSize, typed, cachePos, caches);
        if (typed != embeddings) {
            // Owned by this map now; closeInputs frees it with the other per-step arrays
            inputs.put(embedsName, typed);
        }
        return inputs;
    }

    private static INDArray logits(SameDiff model, Map<String, INDArray> inputs) {
        String logitsName = ModelIOConfig.findLogitsOutputName(model);
        return model.output(inputs, logitsName).get(logitsName);
    }

    private INDArray embed(INDArray tokenIds) {
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String name : embedTokens.inputs()) {
            embedInputs.put(name, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray embeddings = embedOutputs.values().iterator().next().dup();
        closeAll(embedOutputs.values());
        return embeddings;
    }

    private static int lastPositionArgMax(INDArray logits) {
        INDArray last = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all());
        INDArray argMax = last.argMax();
        int token = argMax.getInt(0);
        argMax.close();
        return token;
    }

    private static void assertFinite(String what, INDArray values) {
        long nonFinite = Nd4j.getExecutioner().exec(new MatchCondition(values, Conditions.notFinite())).getLong(0);
        assertEquals(0, nonFinite, what + ": " + nonFinite + " of " + values.length() + " values are NaN or infinite");
    }

    /** Replays the current plan with {@code inputs}; no slot may hold NaN afterwards. */
    private void assertNoNaNSlot(String phase, Map<String, INDArray> inputs) {
        DspHandle h = decoder.dsp();
        assertTrue(h.isCompiled(), phase + ": the decoder should have compiled a DSP plan");
        h.replay(inputs);
        int nanSlot = h.firstNaNSlot();
        if (nanSlot >= 0) {
            String nanSlots = h.snapshotAllSlots().values().stream()
                    .filter(summary -> summary.contains("NaN"))
                    .collect(Collectors.joining("\n  "));
            assertEquals(-1, nanSlot, phase + ": plan slots hold NaN:\n  " + nanSlots);
        }
        log.info("{}: no NaN in {} plan slots", phase, h.totalSlots());
    }

    /** Closes the per-step arrays of {@code inputs}: not the prompt embeddings and not the caches. */
    private void closeInputs(Map<String, INDArray> inputs, Map<String, INDArray> caches) {
        for (Map.Entry<String, INDArray> entry : inputs.entrySet()) {
            INDArray value = entry.getValue();
            boolean cache = caches == null ? isCacheValue(inputs, entry.getKey()) : caches.containsValue(value);
            if (value != null && value != inputsEmbeds && !cache && !value.wasClosed()) {
                value.close();
            }
        }
        if (caches != null) {
            closeAll(caches.values());
        }
    }

    private boolean isCacheValue(Map<String, INDArray> inputs, String name) {
        ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(decoder);
        return kvNames.keyNames.contains(name) || kvNames.valueNames.contains(name);
    }

    private static void closeAll(Iterable<INDArray> arrays) {
        for (INDArray arr : arrays) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) {
                arr.close();
            }
        }
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
