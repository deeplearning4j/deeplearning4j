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
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * NaN isolation test for VLM decode using DspHandle API.
 *
 * Uses sd.dsp().firstNaNSlot() and sd.dsp().snapshotAllSlots() to pinpoint
 * exactly which op first produces NaN during plan execution of the SmolDocling
 * decoder.
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestVLMDecodeNaNIsolation \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 -Pcuda \
 *     2>&1 | tee /tmp/vlm-nan-isolation.log
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestVLMDecodeNaNIsolation {

    private SameDiff decoder;
    private SameDiff embedTokens;
    private HuggingFaceTokenizer tokenizer;
    private INDArray inputsEmbeds;
    private int[] promptTokenIds;

    @BeforeAll
    public void setup() throws Exception {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");

        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
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

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] encoded = tokenizer.encode(chatPrompt, false).getIds();
        promptTokenIds = encoded;

        INDArray tokenIds = Nd4j.createFromArray(new int[][]{encoded}).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);

        log.info("Setup complete: decoder={} ops, promptTokens={}, embedShape={}",
                decoder.getOps().size(), promptTokenIds.length, Arrays.toString(inputsEmbeds.shape()));
    }

    @AfterAll
    public void tearDown() {
        if (decoder != null) decoder.close();
        if (embedTokens != null) embedTokens.close();
    }

    /**
     * Test 1: Prefill via sd.output() then inspect all slots for NaN.
     * This tests the prefill (first forward pass with full sequence).
     */
    @Test
    @DisplayName("Prefill: inspect all slots for NaN via DspHandle")
    void testPrefillSlotNaNInspection() {
        long seqLen = inputsEmbeds.shape()[1];
        Map<String, INDArray> inputs = buildPrefillInputs(seqLen);

        String logitsName = findLogitsOutput();

        // Warmup — compiles the plan
        Map<String, INDArray> result = decoder.output(inputs, logitsName);
        INDArray logits = result.get(logitsName);
        assertNotNull(logits, "Prefill should produce logits");

        double logitsSum = logits.sumNumber().doubleValue();
        log.info("Prefill logits sum = {}, hasNaN = {}", logitsSum, Double.isNaN(logitsSum));

        // Now use DspHandle to inspect slots
        DspHandle h = decoder.dsp();
        assertTrue(h.isCompiled(), "Plan should be compiled after output()");

        log.info("Plan: {} total slots, {} ext inputs", h.totalSlots(), h.numExternalInputs());

        // Replay and inspect
        h.replay(inputs);
        int nanSlot = h.firstNaNSlot();
        Map<Integer, String> snapshot = h.snapshotAllSlots();

        log.info("=== PREFILL SLOT SNAPSHOT ({} slots) ===", snapshot.size());
        for (Map.Entry<Integer, String> entry : snapshot.entrySet()) {
            if (entry.getValue().contains("NaN") || entry.getValue().contains("Inf")) {
                log.error("  {}", entry.getValue());
            }
        }

        if (nanSlot >= 0) {
            log.error("FIRST NaN at slot {}", nanSlot);
            // Log surrounding slots for context
            for (int i = Math.max(0, nanSlot - 3); i <= Math.min(nanSlot + 3, h.totalSlots() - 1); i++) {
                String slotInfo = snapshot.get(i);
                if (slotInfo != null) {
                    log.info("  slot[{}]: {}", i, slotInfo);
                }
            }

            // Get the actual NaN array for deeper inspection
            INDArray nanArr = h.getSlotOutput(nanSlot);
            if (nanArr != null) {
                log.info("  NaN slot array shape: {}, dtype: {}",
                        Arrays.toString(nanArr.shape()), nanArr.dataType());
            }
        } else {
            log.info("No NaN found in prefill — prefill is clean");
        }

        // The test reports findings rather than asserting no NaN,
        // since we know the NaN appears during decode, not necessarily prefill
        log.info("Prefill NaN inspection complete. firstNaNSlot = {}", nanSlot);
    }

    /**
     * Test 2: Single decode step after prefill — this is where NaN appears.
     * Build a decode-step input (single token, KV from prefill), replay,
     * and find the first NaN slot.
     */
    @Test
    @DisplayName("Decode step 1: find first NaN slot via DspHandle")
    void testDecodeStep1NaNIsolation() {
        long seqLen = inputsEmbeds.shape()[1];
        Map<String, INDArray> prefillInputs = buildPrefillInputs(seqLen);
        String logitsName = findLogitsOutput();

        // Get all outputs including present KV cache
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsName);
        for (String name : decoder.outputs()) {
            if (name.startsWith("present.")) {
                allOutputNames.add(name);
            }
        }
        log.info("Requesting {} outputs ({} KV pairs)", allOutputNames.size(), allOutputNames.size() - 1);

        // Prefill
        Map<String, INDArray> prefillResult = decoder.output(
                prefillInputs, allOutputNames.toArray(new String[0]));
        INDArray prefillLogits = prefillResult.get(logitsName);
        assertNotNull(prefillLogits);

        // Get first token from prefill
        INDArray lastLogits = prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int firstToken = lastLogits.argMax().getInt(0);
        log.info("Prefill first token: {} ({})", firstToken, tokenizer.decode(new int[]{firstToken}));

        // Build decode step 1 inputs
        // Get token embedding for the predicted token
        INDArray tokenId = Nd4j.createFromArray(new int[]{firstToken}).reshape(1, 1).castTo(DataType.INT64);
        Map<String, INDArray> embedIn = new HashMap<>();
        for (String name : embedTokens.inputs()) {
            embedIn.put(name, tokenId);
        }
        Map<String, INDArray> embedOut = embedTokens.output(embedIn,
                embedTokens.outputs().toArray(new String[0]));
        INDArray decodeEmbeds = embedOut.values().iterator().next().dup();

        // Build decode inputs with KV from prefill
        Map<String, INDArray> decodeInputs = new HashMap<>();
        decodeInputs.put("inputs_embeds", decodeEmbeds);
        decodeInputs.put("attention_mask", Nd4j.ones(DataType.LONG, 1, seqLen + 1));
        decodeInputs.put("position_ids", Nd4j.createFromArray(new long[]{seqLen}).reshape(1, 1));

        // Wire KV from prefill as past_key_values
        // Output names: present.N.key/value → Input names: past_key_values.N.key/value
        for (String outputName : prefillResult.keySet()) {
            if (outputName.startsWith("present.")) {
                // present.0.key → past_key_values.0.key
                String pastName = outputName.replace("present.", "past_key_values.");
                if (decoder.inputs().contains(pastName)) {
                    decodeInputs.put(pastName, prefillResult.get(outputName).dup());
                } else {
                    log.warn("KV mapping: {} → {} not found in decoder inputs", outputName, pastName);
                }
            }
        }

        // Reset decoder session to force fresh plan compilation for decode shape
        decoder.resetSession();

        // Run decode step
        Map<String, INDArray> decodeResult = decoder.output(decodeInputs, logitsName);
        INDArray decodeLogits = decodeResult.get(logitsName);
        assertNotNull(decodeLogits, "Decode step should produce logits");

        double decodeSum = decodeLogits.sumNumber().doubleValue();
        log.info("Decode step 1 logits sum = {}, hasNaN = {}", decodeSum, Double.isNaN(decodeSum));

        // Now use DspHandle to inspect ALL slots
        DspHandle h = decoder.dsp();
        assertTrue(h.isCompiled());

        log.info("Decode plan: {} total slots, {} ext inputs", h.totalSlots(), h.numExternalInputs());

        h.replay(decodeInputs);
        int nanSlot = h.firstNaNSlot();
        Map<Integer, String> snapshot = h.snapshotAllSlots();

        // Report ALL NaN slots
        int nanCount = 0;
        int firstNaN = -1;
        for (Map.Entry<Integer, String> entry : snapshot.entrySet()) {
            if (entry.getValue().contains("NaN")) {
                nanCount++;
                if (firstNaN < 0) firstNaN = entry.getKey();
                log.error("NaN: {}", entry.getValue());
            }
        }

        log.info("=== DECODE STEP 1 SUMMARY ===");
        log.info("Total slots: {}", snapshot.size());
        log.info("NaN slots: {}", nanCount);
        log.info("First NaN slot: {}", firstNaN);

        if (firstNaN >= 0) {
            // Log the 5 slots BEFORE the first NaN — these are the clean inputs
            log.info("=== Slots leading up to first NaN ===");
            for (int i = Math.max(0, firstNaN - 5); i <= firstNaN; i++) {
                String info = snapshot.get(i);
                if (info != null) {
                    log.info("  {}", info);
                }
            }

            // Get the NaN array for further analysis
            INDArray nanArr = h.getSlotOutput(firstNaN);
            if (nanArr != null) {
                log.info("First NaN slot array: shape={}, dtype={}",
                        Arrays.toString(nanArr.shape()), nanArr.dataType());
            }
        }
    }

    /**
     * Test 3: Same as test 1 but with optimizer DISABLED.
     * If NaN disappears, the optimizer is corrupting the fused_rope inputs.
     * If NaN persists, the bug is in the base DSP execution path.
     */
    @Test
    @DisplayName("Prefill WITHOUT optimizer: NaN comparison")
    void testPrefillNoOptimizerNaNComparison() {
        // Disable optimizer for this test
        System.setProperty("nd4j.optimizer.enabled", "false");
        System.setProperty("nd4j.optimizer.fp16", "false");

        // Must reset session to force re-compilation without optimizer
        decoder.resetSession();

        long seqLen = inputsEmbeds.shape()[1];
        Map<String, INDArray> inputs = buildPrefillInputs(seqLen);
        String logitsName = findLogitsOutput();

        Map<String, INDArray> result = decoder.output(inputs, logitsName);
        INDArray logits = result.get(logitsName);
        assertNotNull(logits);

        double logitsSum = logits.sumNumber().doubleValue();
        log.info("Prefill WITHOUT optimizer: logits sum = {}, hasNaN = {}", logitsSum, Double.isNaN(logitsSum));

        DspHandle h = decoder.dsp();
        if (h.isCompiled()) {
            h.replay(inputs);
            int nanSlot = h.firstNaNSlot();
            log.info("WITHOUT optimizer: firstNaNSlot = {}", nanSlot);

            if (nanSlot >= 0) {
                Map<Integer, String> snapshot = h.snapshotAllSlots();
                // Show first few NaN slots
                int count = 0;
                for (Map.Entry<Integer, String> entry : snapshot.entrySet()) {
                    if (entry.getValue().contains("NaN")) {
                        log.error("  {}", entry.getValue());
                        if (++count >= 5) break;
                    }
                }
                log.info("NaN PERSISTS without optimizer — bug is in base DSP execution");
            } else {
                log.info("No NaN without optimizer — OPTIMIZER IS THE CAUSE");
            }
        }

        // Restore optimizer settings
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private Map<String, INDArray> buildPrefillInputs(long seqLen) {
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, inputsEmbeds);
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }
        return inputs;
    }

    private String findLogitsOutput() {
        for (String name : decoder.outputs()) {
            if (name.contains("logit")) return name;
        }
        return decoder.outputs().get(0);
    }
}
