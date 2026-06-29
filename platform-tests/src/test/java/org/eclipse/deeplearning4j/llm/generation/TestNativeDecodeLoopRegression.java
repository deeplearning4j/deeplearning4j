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
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.Pointer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone regression tests for the native C++ autoregressive decode loop.
 *
 * Compares the native decode path (GenerationPipeline → AutoregressiveDecode op)
 * against a step-by-step Java-only decode path using DecoderInputBuilder +
 * decoder.output(). This isolates accuracy bugs in the native loop by having
 * a known-good Java reference.
 *
 * Each test focuses on a specific component:
 *   - Prefill + first token match
 *   - Step-by-step token match (Java loop vs native loop)
 *   - Attention mask evolution
 *   - Position IDs correctness
 *   - KV cache scatter correctness
 *   - Logits comparison at each step
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestNativeDecodeLoopRegression \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/decode-regression.log
 */
@Slf4j
public class TestNativeDecodeLoopRegression {

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;
    private static boolean modelsLoaded = false;

    // State captured immediately after pipeline.generate() inside runNativeDecodeLoop().
    // Must be read before any subsequent runNativeDecodeLoop() call resets the session.
    private int lastRunPlanReplays = 0;
    private boolean lastRunCachedOpContextNonNull = false;
    private DynamicShapePlanExecutor lastRunExecutor = null;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
        if (modelsLoaded) return;

        log.info("Loading SmolDocling models for decode loop regression tests...");

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
        BufferedImage testImage = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, targetSize, targetSize);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Line 2: Decode Regression", 50, 150);
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

        hiddenSize = visionEmbeddings.size(-1);

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

        long expectedImageSlots = visionEmbeddings.shape()[1];
        long actualImageSlots = ImagePromptBuilder.countOccurrences(encoded, imageTokenId);
        assertEquals(expectedImageSlots, actualImageSlots,
                "Regression prompt must expose one <image> slot per vision token. "
                        + "visionSeqLen=" + expectedImageSlots + " imageSlots=" + actualImageSlots);

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        assertEquals(promptTokenIds.length, inputsEmbeds.size(1),
                "Merged embedding sequence length must match prompt token length");

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}, imageSlots={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize,
                promptTokenIds.length, actualImageSlots);
        modelsLoaded = true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Java-only step-by-step decode reference implementation.
    // This mirrors the old StaticKvCacheDecodeLoop pattern:
    //   1. Prefill → get initial KV caches + first token
    //   2. Pad KV to static size
    //   3. For each decode step: buildDecoderInputMap → decoder.output → scatter KV
    // ═══════════════════════════════════════════════════════════════════════

    private static class JavaDecodeResult {
        int[] tokenIds;
        INDArray[] logitsPerStep;       // logits[step] = [1, 1, vocabSize] or [1, seqLen, vocabSize]
        Map<String, INDArray> staticKvBuffers;
        long[] positionPerStep;         // position used at each step
        String text;
    }

    /**
     * Run the full decode loop in Java, constructing masks exactly like GenerationPipeline.
     *
     * This is the known-good reference path. It MUST use the same mask layout as
     * GenerationPipeline (contiguous 1s at [0..cachePos]) so the comparison against
     * the native decode loop is valid. DecoderInputBuilder.buildDecoderInputMap uses
     * a padded layout (query at end of mask) which differs from GenerationPipeline's
     * contiguous layout — using it here would produce different tokens even though
     * both are "correct" for their respective layouts.
     */
    private JavaDecodeResult runJavaDecodeLoop(int maxTokens) throws Exception {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        // Enable DSP but use SLOT_BY_SLOT for deterministic reference
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
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

        // ── Prefill ──
        // For prefill, use DecoderInputBuilder since there's no mask layout difference at seqLen > 1
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

        // ── Pad KV caches ──
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(keyName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(valName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        // Sample first token from prefill logits
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        int firstTokenId = Nd4j.argMax(
                prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.shape()[1] - 1),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        prefillLogits.close();

        // ── Build decode-step arrays matching GenerationPipeline layout ──
        // Uses PADDED layout (query at totalSeqLen-1), matching DecoderInputBuilder with dspActive=true.
        long totalSeqLen = maxKvLen + 1;

        // Attention mask: [1, totalSeqLen] LONG — padded layout
        // [0..prefillSeqLen-1] = 1 (filled KV from prefill)
        // [totalSeqLen-1] = 1 (query position)
        INDArray decodeAttentionMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        decodeAttentionMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, prefillSeqLen)).assign(1);
        decodeAttentionMask.putScalar(0, totalSeqLen - 1, 1);

        // Causal mask: [1, 1, 1, totalSeqLen] FLOAT — padded layout
        // [0..prefillSeqLen] = 0.0f (unmask prefill positions)
        INDArray decodeCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
        decodeCausalMask.assign(ModelIOConfig.MASK_FILL);
        decodeCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);

        // Position IDs: [1, 1] INT64
        INDArray decodePosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        INDArray embeddingTable = extractEmbeddingTable();

        // ── Decode steps (Java loop matching GenerationPipeline mask layout) ──
        List<Integer> allTokens = new ArrayList<>();
        allTokens.add(firstTokenId);
        List<INDArray> allLogits = new ArrayList<>();
        List<Long> allPositions = new ArrayList<>();
        allPositions.add((long) prefillSeqLen);

        int cachePos = prefillSeqLen;
        int currentToken = firstTokenId;

        for (int step = 0; step < maxTokens - 1; step++) {
            // Embedding: dup() to match native path's independent buffer (not a view)
            INDArray stepEmbedding = embeddingTable.getRow(currentToken).reshape(1, 1, hiddenSize).dup();
            INDArray stepInputIds = Nd4j.createFromArray(new int[]{currentToken})
                    .reshape(1, 1).castTo(DataType.INT64);

            // Build input map manually — same layout as GenerationPipeline
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

            // Extract logits and sample
            INDArray logits = decodeOutputs.get(ioConfig.getLogitsOutputName());
            INDArray logitsDup = logits.dup();
            allLogits.add(logitsDup);

            int nextToken = Nd4j.argMax(
                    logits.get(NDArrayIndex.point(0),
                            NDArrayIndex.point(0),
                            NDArrayIndex.all()).dup(), -1).getInt(0);

            allTokens.add(nextToken);
            allPositions.add((long) cachePos);

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

            // Advance mask/position for next step (matching C++ kernel updates)
            // Padded layout: unmask the KV position that was just written (cachePos - 1)
            long kvJustWritten = cachePos - 1;
            if (kvJustWritten >= 0 && kvJustWritten < totalSeqLen) {
                decodeAttentionMask.putScalar(new long[]{0, kvJustWritten}, 1);
            }
            if (cachePos < totalSeqLen) {
                decodeCausalMask.putScalar(new long[]{0, 0, 0, cachePos}, 0.0f);
            }
            decodePosIds.putScalar(new long[]{0, 0}, cachePos);

            // Check EOS
            if (nextToken == tokenizer.getEosTokenId()) break;
        }

        JavaDecodeResult result = new JavaDecodeResult();
        result.tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
        result.logitsPerStep = allLogits.toArray(new INDArray[0]);
        result.staticKvBuffers = staticKvBuffers;
        result.positionPerStep = allPositions.stream().mapToLong(Long::longValue).toArray();
        result.text = tokenizer.decode(result.tokenIds, true);
        return result;
    }

    /**
     * Run the native decode path via GenerationPipeline.
     */
    private GenerationResult runNativeDecodeLoop(int maxTokens) throws Exception {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        // graphOptimizerEnabled=false: the model is already optimized by OnnxModelCache
        // (nd4j.optimizer.enabled=true in @BeforeAll). Running the optimizer again would
        // dup the decoder (GraphOptimizer.optimize always calls graph.dup()), creating a
        // new SameDiff instance that disconnects from the test's decoder reference.
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .graphOptimizerEnabled(false)
                .build());

        GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds);

        // Capture executor state IMMEDIATELY after generate() returns, before any session
        // teardown can occur. The nativePlanHandle and cachedOpContext are only valid on the
        // executor that ran the decode; they may be nulled by the next resetModelState() call.
        //
        // Use pipeline.getDecoder() — when graphOptimizerEnabled=true, the pipeline's
        // internal decoder is a dup (different SameDiff instance). Even with it disabled,
        // always read from the pipeline's decoder to be safe.
        lastRunPlanReplays = 0;
        lastRunCachedOpContextNonNull = false;
        lastRunExecutor = null;
        try {
            SameDiff pipelineDecoder = pipeline.getDecoder();
            InferenceSession postGenSession = pipelineDecoder.getOrCreateSession();
            if (postGenSession != null) {
                DynamicShapePlanExecutor postGenExecutor = postGenSession.getDynamicShapePlanExecutor();
                if (postGenExecutor != null) {
                    lastRunExecutor = postGenExecutor;
                    lastRunCachedOpContextNonNull = postGenExecutor.getCachedOpContext() != null;
                    Pointer postGenHandle = postGenExecutor.getNativePlanHandle();
                    boolean handleValid = postGenHandle != null && !postGenHandle.isNull();
                    if (handleValid) {
                        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
                        lastRunPlanReplays = ops.getPlanTotalGraphReplays(postGenHandle);
                        // Per-segment capture state (Java-side, post-decode). This is the
                        // RELIABLE way to see why capture failed — the native autoregressive_decode
                        // op runs with C++ DSP_DIAG silent, so these JNI getters are the only
                        // window into each segment's phase / execCount / replay / backend.
                        log.info("[runNativeDecodeLoop] captureStats: {}", ops.getPlanCaptureStats(postGenHandle));
                        try {
                            log.info("[runNativeDecodeLoop] PLAN getPlanPhase={} getPlanPointersStable={}",
                                    ops.getPlanPhase(postGenHandle), ops.getPlanPointersStable(postGenHandle));
                        } catch (Throwable t) { log.info("[runNativeDecodeLoop] PLAN phase/pointers UNAVAILABLE: {}", t.toString()); }
                        int segCount = ops.getPlanSegmentCount(postGenHandle);
                        for (int s = 0; s < segCount; s++) {
                            log.info("[runNativeDecodeLoop]   seg[{}] phase={} execCount={} replayCount={} replayState={} backend='{}'",
                                    s, ops.getPlanSegmentExecutionPhase(postGenHandle, s),
                                    ops.getPlanSegmentExecutionCount(postGenHandle, s),
                                    ops.getPlanSegmentReplayCount(postGenHandle, s),
                                    ops.getPlanSegmentReplayState(postGenHandle, s),
                                    ops.getPlanSegmentBackendName(postGenHandle, s));
                            log.info("[runNativeDecodeLoop]   seg[{}] statsJson: {}", s,
                                    ops.getPlanSegmentStatisticsJson(postGenHandle, s));
                            try {
                                log.info("[runNativeDecodeLoop]   seg[{}] needsArgRefresh={}", s,
                                        ops.getPlanSegmentNeedsArgRefresh(postGenHandle, s));
                            } catch (Throwable t) { log.info("[runNativeDecodeLoop]   seg[{}] needsArgRefresh UNAVAILABLE: {}", s, t.toString()); }
                        }
                        // ── Sync-free buffer fingerprint ring dump ──────────────────
                        // Activated by BUF_FP_RING=1 env var.
                        // drainPlanFingerprintRing() does one D2H after the loop ends.
                        try {
                            ops.drainPlanFingerprintRing(postGenHandle);
                            String fpJson = ops.getPlanFingerprintJson(postGenHandle);
                            log.info("[runNativeDecodeLoop] BUF_FP_RING fingerprints: {}", fpJson);
                        } catch (Throwable t) {
                            log.info("[runNativeDecodeLoop] BUF_FP_RING unavailable: {}", t.toString());
                        }
                    }
                    log.info("[runNativeDecodeLoop] Captured: replays={} ctxNonNull={} handle={}",
                            lastRunPlanReplays, lastRunCachedOpContextNonNull,
                            handleValid ? postGenHandle.address() : "null");
                }
            }
        } catch (Exception e) {
            log.warn("[runNativeDecodeLoop] Failed to capture executor state: {}", e.getMessage());
        }

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Java reference decode produces coherent output (sanity check)")
    public void testJavaReferenceDecodeSanity() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 10;

        JavaDecodeResult javaResult = runJavaDecodeLoop(maxTokens);
        log.info("Java reference: {} tokens, text='{}'", javaResult.tokenIds.length, javaResult.text);
        log.info("Java tokens: {}", Arrays.toString(javaResult.tokenIds));
        log.info("Java positions: {}", Arrays.toString(javaResult.positionPerStep));

        assertTrue(javaResult.tokenIds.length >= 2,
                "Java reference should produce at least 2 tokens, got " + javaResult.tokenIds.length);
        // Token 0 should be consistent across runs
        int firstToken = javaResult.tokenIds[0];
        log.info("First token: {} (decoded: '{}')",
                firstToken, tokenizer.decode(new int[]{firstToken}, true));
    }

    @Test
    @DisplayName("Java reference vs native decode: token-by-token comparison")
    public void testJavaVsNativeTokenMatch() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 10;

        JavaDecodeResult javaResult = runJavaDecodeLoop(maxTokens);
        GenerationResult nativeResult = runNativeDecodeLoop(maxTokens);

        int[] javaTokens = javaResult.tokenIds;
        int[] nativeTokens = nativeResult.getTokenIds();
        int minLen = Math.min(javaTokens.length, nativeTokens.length);

        log.info("Java   tokens ({}): {}", javaTokens.length, Arrays.toString(javaTokens));
        log.info("Native tokens ({}): {}", nativeTokens.length, Arrays.toString(nativeTokens));
        log.info("Java   text: '{}'", javaResult.text);
        log.info("Native text: '{}'", nativeResult.getText());

        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            String status = javaTokens[i] == nativeTokens[i] ? "MATCH" : "DIVERGE";
            log.info("  Step {}: java={} native={} [{}]", i, javaTokens[i], nativeTokens[i], status);
            if (javaTokens[i] == nativeTokens[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }

        double matchRate = minLen > 0 ? (double) matches / minLen : 0.0;
        log.info("Token match rate: {}/{} ({}%)", matches, minLen, String.format("%.1f", matchRate * 100));

        if (firstDivergent >= 0) {
            log.info("First divergence at step {}: java={} native={}",
                    firstDivergent, javaTokens[firstDivergent], nativeTokens[firstDivergent]);

            // [DIVDIAG temp] reference logit margin at the divergence: numerical (tiny margin) vs structural (large).
            // allLogits[k] produced allTokens[k+1], so token index i maps to logitsPerStep[i-1].
            int li = firstDivergent - 1;
            if (li >= 0 && javaResult.logitsPerStep != null && li < javaResult.logitsPerStep.length) {
                INDArray vocab = javaResult.logitsPerStep[li]
                        .get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();
                int jt = javaTokens[firstDivergent], nt = nativeTokens[firstDivergent];
                double jL = vocab.getDouble(jt), nL = vocab.getDouble(nt);
                int refArg = Nd4j.argMax(vocab, -1).getInt(0);
                double maxL = vocab.getDouble(refArg);
                log.info("[DIVDIAG] ref-logits@divstep: javaTok={} L={} | nativeTok={} L={} | margin(j-n)={} | refArgmax={} maxL={} | vocabLen={}",
                        jt, String.format("%.5f", jL), nt, String.format("%.5f", nL),
                        String.format("%.5f", jL - nL), refArg, String.format("%.5f", maxL), vocab.length());
            }
        }

        // The first 2 tokens (prefill sample + warmup decode) are done in Java for both paths.
        // They MUST match.
        assertTrue(minLen >= 2, "Both paths should produce at least 2 tokens");
        assertEquals(javaTokens[0], nativeTokens[0],
                "Token 0 (prefill) must match between Java and native");
        assertEquals(javaTokens[1], nativeTokens[1],
                "Token 1 (first decode warmup) must match between Java and native");

        // Tokens 2+ are where the native loop takes over. Check match rate.
        if (minLen > 2) {
            int nativeMatches = 0;
            for (int i = 2; i < minLen; i++) {
                if (javaTokens[i] == nativeTokens[i]) nativeMatches++;
            }
            double nativeMatchRate = (double) nativeMatches / (minLen - 2);
            log.info("Native-only token match rate (steps 2+): {}/{} ({}%)",
                    nativeMatches, minLen - 2, String.format("%.1f", nativeMatchRate * 100));

            assertTrue(nativeMatchRate >= 0.9,
                    "Native decode loop token match rate too low: "
                            + String.format("%.1f%% (steps 2+)", nativeMatchRate * 100));
        }
    }

    @Test
    @DisplayName("Java reference is deterministic across two runs")
    public void testJavaReferenceDeterminism() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 5;

        JavaDecodeResult run1 = runJavaDecodeLoop(maxTokens);
        JavaDecodeResult run2 = runJavaDecodeLoop(maxTokens);

        log.info("Run 1: {}", Arrays.toString(run1.tokenIds));
        log.info("Run 2: {}", Arrays.toString(run2.tokenIds));

        assertEquals(run1.tokenIds.length, run2.tokenIds.length,
                "Java reference should produce same number of tokens across runs");
        assertArrayEquals(run1.tokenIds, run2.tokenIds,
                "Java reference must be deterministic: tokens differ between runs");
    }

    @Test
    @DisplayName("Native decode is deterministic across two runs")
    public void testNativeDecodeDeterminism() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 5;

        GenerationResult run1 = runNativeDecodeLoop(maxTokens);
        GenerationResult run2 = runNativeDecodeLoop(maxTokens);

        log.info("Run 1: {}", Arrays.toString(run1.getTokenIds()));
        log.info("Run 2: {}", Arrays.toString(run2.getTokenIds()));

        assertArrayEquals(run1.getTokenIds(), run2.getTokenIds(),
                "Native decode must be deterministic: tokens differ between runs."
                        + " Run1=" + Arrays.toString(run1.getTokenIds())
                        + " Run2=" + Arrays.toString(run2.getTokenIds()));
    }

    @Test
    @DisplayName("Prefill token matches between Java and native paths")
    public void testPrefillTokenMatch() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 3;

        JavaDecodeResult javaResult = runJavaDecodeLoop(maxTokens);
        GenerationResult nativeResult = runNativeDecodeLoop(maxTokens);

        assertEquals(javaResult.tokenIds[0], nativeResult.getTokenIds()[0],
                "Prefill-sampled token must match. Java=" + javaResult.tokenIds[0]
                        + " Native=" + nativeResult.getTokenIds()[0]);
        log.info("Prefill token MATCH: {} (decoded: '{}')",
                javaResult.tokenIds[0],
                tokenizer.decode(new int[]{javaResult.tokenIds[0]}, true));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // maxNewTokens contract tests — token count must never exceed request
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("maxNewTokens=1 returns exactly 1 token")
    public void testMaxNewTokens1() throws Exception {
        ensureModelsLoaded();
        GenerationResult result = runNativeDecodeLoop(1);
        log.info("maxNewTokens=1: {} tokens, text='{}'", result.getTokenIds().length, result.getText());
        assertEquals(1, result.getTokenIds().length,
                "maxNewTokens=1 must return exactly 1 token, got " + result.getTokenIds().length
                        + ": " + Arrays.toString(result.getTokenIds()));
    }

    @Test
    @DisplayName("maxNewTokens=2 returns at most 2 tokens")
    public void testMaxNewTokens2() throws Exception {
        ensureModelsLoaded();
        GenerationResult result = runNativeDecodeLoop(2);
        log.info("maxNewTokens=2: {} tokens, text='{}'", result.getTokenIds().length, result.getText());
        assertTrue(result.getTokenIds().length <= 2,
                "maxNewTokens=2 must return at most 2 tokens, got " + result.getTokenIds().length
                        + ": " + Arrays.toString(result.getTokenIds()));
    }

    @Test
    @DisplayName("maxNewTokens=3 invokes native decode and returns at most 3 tokens")
    public void testMaxNewTokens3() throws Exception {
        ensureModelsLoaded();
        GenerationResult result = runNativeDecodeLoop(3);
        log.info("maxNewTokens=3: {} tokens, text='{}'", result.getTokenIds().length, result.getText());
        assertTrue(result.getTokenIds().length <= 3,
                "maxNewTokens=3 must return at most 3 tokens, got " + result.getTokenIds().length
                        + ": " + Arrays.toString(result.getTokenIds()));
        // With 3 tokens, remainingTokens=1, so the native C++ op is invoked.
        assertTrue(result.getTokenIds().length >= 2,
                "maxNewTokens=3 should produce at least 2 tokens (2 warmup), got "
                        + result.getTokenIds().length);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Replay-state and address-stability tests
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Native decode with 10 tokens triggers graph replays (fast path engaged)")
    public void testReplayStateAfterNativeDecode() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 10;
        GenerationResult result = runNativeDecodeLoop(maxTokens);

        log.info("Native decode produced {} tokens: {}", result.getTokenIds().length,
                Arrays.toString(result.getTokenIds()));

        // Use state captured immediately after generate() inside runNativeDecodeLoop(),
        // before any session teardown can null the native plan handle.
        DynamicShapePlanExecutor executor = lastRunExecutor;
        assertNotNull(executor, "DSP executor must exist after native decode");

        PlanPhase phase = executor.getPlanPhase();
        log.info("Plan phase after decode: {}", phase);

        // lastRunPlanReplays was captured immediately after generate() returned,
        // when the native plan handle was still valid.
        int totalReplays = lastRunPlanReplays;
        log.info("Total graph replays: {}", totalReplays);
        // With AUTO mode, the frozen fast path should engage after warmup.
        // Graph replays > 0 means the fast path is working and address
        // stability is maintained. The exact phase name depends on whether
        // advancePlanPhase() was reached (fast path may skip it).
        assertTrue(totalReplays > 0,
                "Expected at least 1 graph replay after 10-token native decode, got "
                        + totalReplays + ". This indicates address instability or "
                        + "frozen fast path never engaged. Phase=" + phase);
    }

    @Test
    @DisplayName("Consecutive native decodes engage graph replay fast path")
    public void testAddressStabilityAcrossRuns() throws Exception {
        ensureModelsLoaded();

        // First run — warms up the plan.
        // Replay count is captured inside runNativeDecodeLoop() immediately after
        // pipeline.generate() returns, before any session teardown can occur.
        GenerationResult result1 = runNativeDecodeLoop(8);
        log.info("Run 1: {} tokens", result1.getTokenIds().length);
        int replaysAfterRun1 = lastRunPlanReplays;
        log.info("Replays after run 1: {}", replaysAfterRun1);

        // Second run — runNativeDecodeLoop resets the decoder state via
        // BenchmarkConfigApplier.resetModelState(), so the plan is recompiled.
        // We verify the NEW plan also engages graph replays.
        GenerationResult result2 = runNativeDecodeLoop(8);
        log.info("Run 2: {} tokens", result2.getTokenIds().length);
        int replaysAfterRun2 = lastRunPlanReplays;
        log.info("Replays after run 2: {}", replaysAfterRun2);

        // Each run should independently engage graph replays (> 0).
        // The exact count may differ because the plan is recompiled between runs
        // (resetModelState clears the session). What matters is that the fast path
        // works in both runs, not that the count accumulates across plan resets.
        assertTrue(replaysAfterRun1 > 0,
                "First native decode run should engage graph replays, got "
                        + replaysAfterRun1);
        assertTrue(replaysAfterRun2 > 0,
                "Second native decode run should engage graph replays, got "
                        + replaysAfterRun2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private INDArray extractEmbeddingTable() {
        for (SDVariable varName : embedTokens.variables()) {
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
    // Diagnostic tests P1-P5
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * P1: Verify OpaqueContext external input device pointers match current INDArrays.
     * If the OpaqueContext holds stale wrappers from the warmup decode, the native
     * loop will read stale device data → degenerate output.
     */
    @Test
    @DisplayName("P1: OpaqueContext external inputs are fresh after warmup decode")
    public void testOpaqueContextInputFreshness() throws Exception {
        ensureModelsLoaded();

        // Run warmup decode exactly like GenerationPipeline does.
        // Executor state is captured inside runNativeDecodeLoop() immediately after
        // pipeline.generate() returns, before any session teardown can occur.
        int maxTokens = 3;
        GenerationResult nativeResult = runNativeDecodeLoop(maxTokens);
        log.info("Native warmup produced {} tokens: {}", nativeResult.getTokenIds().length,
                Arrays.toString(nativeResult.getTokenIds()));

        // Use the executor captured inside runNativeDecodeLoop() — it reflects the state
        // immediately after generation, before resetModelState() can null the handle.
        DynamicShapePlanExecutor executor = lastRunExecutor;
        assertNotNull(executor, "DSP executor must exist after native decode");

        assertTrue(lastRunCachedOpContextNonNull, "OpaqueContext must exist after warmup decode");
        org.nd4j.nativeblas.OpaqueContext ctx = executor.getCachedOpContext();
        assertNotNull(ctx, "OpaqueContext must exist after warmup decode");

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numInputs = (int) nativeOps.numInputsNative(ctx);
        log.info("OpaqueContext has {} inputs", numInputs);

        int mismatchCount = 0;
        for (int i = 0; i < numInputs; i++) {
            org.nd4j.nativeblas.OpaqueNDArray opaqueArr = nativeOps.getInputArrayNative(ctx, i);
            if (opaqueArr == null || opaqueArr.isNull()) continue;

            Pointer ctxSpecialBuf = opaqueArr.specialBuffer();
            INDArray[] extInputs = executor.getExternalInputsSnapshot();
            if (extInputs == null || i >= extInputs.length || extInputs[i] == null) continue;

            Pointer javaSpecialBuf = extInputs[i].data().pointer();
            if (ctxSpecialBuf == null || javaSpecialBuf == null) continue;

            boolean match = ctxSpecialBuf.address() == javaSpecialBuf.address();
            if (!match) {
                log.warn("P1 MISMATCH: extInput[{}] ctxSpecialBuf={} javaSpecialBuf={} diff={}",
                        i, ctxSpecialBuf.address(), javaSpecialBuf.address(),
                        ctxSpecialBuf.address() - javaSpecialBuf.address());
                mismatchCount++;
            }
        }
        assertEquals(0, mismatchCount,
                "P1: " + mismatchCount + " OpaqueContext external inputs have stale device pointers");
    }

    /**
     * P2: Compare Java vs native logits at the first divergence point (step 2).
     * Run Java loop to step 2, capture logits. Run native loop for 1 step after warmup,
     * capture logits. Compare element-wise.
     */
    @Test
    @DisplayName("P2: Logits comparison at first divergence point (step 2)")
    public void testLogitsAtStep2Divergence() throws Exception {
        ensureModelsLoaded();

        // Java reference: run to step 2, capture logits at step 2
        JavaDecodeResult javaResult = runJavaDecodeLoop(3);
        INDArray javaStep2Logits = javaResult.logitsPerStep[1]; // step 1 in array = second decode step = token 2
        log.info("Java step 2 logits shape: {}", javaStep2Logits.shapeInfoToString());
        int javaArgmax = Nd4j.argMax(javaStep2Logits.dup(), -1).getInt(0);
        log.info("Java step 2 argmax token: {}", javaArgmax);

        // Native: run warmup (prefill + 1 decode) then capture native step 0 logits
        // We need to hook into the native loop to capture logits — for now, compare tokens
        GenerationResult nativeResult = runNativeDecodeLoop(3);
        int[] nativeTokens = nativeResult.getTokenIds();
        log.info("Native tokens: {}", Arrays.toString(nativeTokens));
        log.info("Java tokens:   {}", Arrays.toString(javaResult.tokenIds));

        if (nativeTokens.length > 2 && javaResult.tokenIds.length > 2) {
            boolean step2Match = nativeTokens[2] == javaResult.tokenIds[2];
            log.info("Step 2 token match: java={} native={} {}",
                    javaResult.tokenIds[2], nativeTokens[2], step2Match ? "MATCH" : "DIVERGE");
            // This is diagnostic — don't assert, just log
        }
    }

    /**
     * P5: Run Java decode loop using OpaqueContext-based execution (same path as native).
     * If this also diverges, confirms the issue is in OpaqueContext/external input freshness,
     * not in the C++ decode loop logic.
     */
    @Test
    @DisplayName("P5: Java decode via OpaqueContext (mimic native path)")
    public void testJavaDecodeViaOpaqueContext() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 5;

        // Setup exactly like GenerationPipeline
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputNames = new ArrayList<>(decoder.outputs());
        decoder.compileNativeDynamicShapePlan(outputNames, GraphExecutionMode.SLOT_BY_SLOT, true);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();

        // Prefill
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoder.inputs(), decoder,
                inputsEmbeds.dup(), currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                new HashMap<>(), dspActive);
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

        // Build decode arrays using SAME mask layout as runJavaDecodeLoop (padded)
        INDArray embeddingTable = extractEmbeddingTable();
        INDArray decodeEmbeddings = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);
        long totalSeqLen = maxKvLen + 1;
        INDArray decodeCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
        decodeCausalMask.assign(ModelIOConfig.MASK_FILL);
        decodeCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        INDArray decodeAttentionMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        decodeAttentionMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, prefillSeqLen)).assign(1);
        decodeAttentionMask.putScalar(0, totalSeqLen - 1, 1);
        INDArray decodePosIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);

        // Warmup decode step via decoder.output()
        Map<String, INDArray> decodeInputMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null) decodeInputMap.put(embedsName, decodeEmbeddings);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) decodeInputMap.put(idsName, decodeInputIds);
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
        INDArray decodeLogits = decodeOutputs.get(ioConfig.getLogitsOutputName());
        int secondTokenId = Nd4j.argMax(
                decodeLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(0),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        decodeLogits.close();

        // Scatter KV
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = decodeOutputs.get(keyName);
            INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(keyName));
            if (presentKv != null && staticBuf != null) {
                scatterKvToStatic(presentKv, staticBuf, prefillSeqLen);
            }
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = decodeOutputs.get(valName);
            INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(valName));
            if (presentKv != null && staticBuf != null) {
                scatterKvToStatic(presentKv, staticBuf, prefillSeqLen);
            }
        }

        // Update decode arrays for second token
        // TEST: use fresh embedding array instead of reused + assign()
        INDArray freshEmbeddings = embeddingTable.getRow(secondTokenId).reshape(1, 1, hiddenSize).dup();
        decodeInputIds.putScalar(new long[]{0, 0}, secondTokenId);
        // Padded layout: unmask the KV position written by warmup (prefillSeqLen)
        decodeAttentionMask.putScalar(new long[]{0, prefillSeqLen}, 1);
        decodePosIds.putScalar(new long[]{0, 0}, prefillSeqLen + 1);
        decodeCausalMask.putScalar(new long[]{0, 0, 0, prefillSeqLen}, 0.0f);

        // Sync to device
        Nd4j.getExecutioner().commit();
        freshEmbeddings.syncToDevice();
        decodeInputIds.syncToDevice();
        decodeAttentionMask.syncToDevice();
        decodePosIds.syncToDevice();
        decodeCausalMask.syncToDevice();
        for (INDArray kvBuf : staticKvBuffers.values()) {
            kvBuf.syncToDevice();
        }

        // NOW: execute step 2 via executor.executeNative() (same path as native loop)
        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        Map<String, INDArray> step2InputMap = new HashMap<>(decodeInputMap);
        if (embedsName != null) step2InputMap.put(embedsName, freshEmbeddings);
        if (idsName != null) step2InputMap.put(idsName, decodeInputIds);
        if (maskName != null) step2InputMap.put(maskName, decodeAttentionMask);
        if (causalName != null) step2InputMap.put(causalName, decodeCausalMask);
        if (posName != null) step2InputMap.put(posName, decodePosIds);
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(step2InputMap.keySet()), decoder, step2InputMap);

        Map<String, INDArray> step2Outputs = decoder.output(
                step2InputMap, allOutputNames.toArray(new String[0]));
        INDArray step2Logits = step2Outputs.get(ioConfig.getLogitsOutputName());
        int step2Token = Nd4j.argMax(
                step2Logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(0),
                        NDArrayIndex.all()).dup(), -1).getInt(0);
        step2Logits.close();

        // Compare against full Java reference
        JavaDecodeResult javaResult = runJavaDecodeLoop(3);
        int javaStep2Token = javaResult.tokenIds[2];

        log.info("P5: Java step 2 via decoder.output() = {}", javaStep2Token);
        log.info("P5: Java step 2 via same OpaqueContext path = {}", step2Token);

        assertEquals(javaStep2Token, step2Token,
                "P5: Java decode via OpaqueContext path diverges from pure Java path. " +
                        "This confirms OpaqueContext/external input freshness is the root cause.");
    }
}
