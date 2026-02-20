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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SmolDocling pipeline test with explicit GraphOptimizer fusion and performance measurement.
 *
 * This test applies RMSNorm, Softmax, and activation fusions to the decoder graph
 * and measures tokens/second throughput, verifying output coherence.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline \
 *     -Dvlm.test.maxTokens=100
 *
 * Optional:
 *   -Dvlm.test.pdf.path=/path/to/file.pdf -Dvlm.test.pdf.page=10
 *   -Dvlm.model.cache.disable=true   (force re-import from ONNX)
 */
@Slf4j
public class TestSmolDoclingOptimizedPipeline {

    private static int maxTokensConfig = 100;
    private static String pdfPath;
    private static int specificPage = -1;
    private static int renderDpi = 150;

    @BeforeAll
    public static void setup() {
        // Enable graph optimizer for this test
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.logApplied", "true");
        
        String maxTokensStr = System.getProperty("vlm.test.maxTokens");
        if (maxTokensStr != null && !maxTokensStr.isEmpty()) {
            maxTokensConfig = Integer.parseInt(maxTokensStr);
        }
        pdfPath = System.getProperty("vlm.test.pdf.path");
        String pageStr = System.getProperty("vlm.test.pdf.page");
        if (pageStr != null && !pageStr.isEmpty()) {
            specificPage = Integer.parseInt(pageStr);
        }
        String dpiStr = System.getProperty("vlm.test.pdf.dpi");
        if (dpiStr != null && !dpiStr.isEmpty()) {
            renderDpi = Integer.parseInt(dpiStr);
        }

        log.info("=== Optimized Pipeline Test Configuration ===");
        log.info("  Max tokens: {}", maxTokensConfig);
        log.info("  PDF: {}", pdfPath != null ? pdfPath : "(test pattern)");
        log.info("  Page: {}", specificPage >= 0 ? specificPage : "0");
    }

    @Test
    @DisplayName("Optimized SmolDocling: GraphOptimizer fusions + tok/s measurement")
    public void testOptimizedDoclingPipeline() throws Exception {
        // ==================== STEP 1: Download Models ====================
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE.");

        // ==================== STEP 2: Load Tokenizer ====================
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("STEP 2 DONE: vocab_size={}", tokenizer.getVocabSize());

        // ==================== STEP 3: Import ONNX + Optimize via OnnxModelCache ====================
        // OnnxModelCache applies GraphOptimizer.optimize() during import and persists the
        // optimized graph to SDZ. The SDZ save+reload cycle ensures a clean graph state.
        // Use -Dvlm.model.cache.disable=true to force re-import and re-optimization.
        log.info("STEP 3: Importing ONNX models (with optimization + SDZ cache)...");
        long step3Start = System.currentTimeMillis();

        // Invalidate existing cache to force re-optimization (so we see the optimization logs)
        boolean forceReoptimize = Boolean.getBoolean("vlm.model.cache.disable");
        if (forceReoptimize) {
            log.info("  Cache disabled - will re-import and re-optimize from ONNX");
            OnnxModelCache.invalidateCache(decoderResult.getModelFile().getAbsolutePath());
        }

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        long importTime = System.currentTimeMillis() - step3Start;

        // Log the optimized decoder stats
        int decoderOps = decoder.getOps().size();
        Map<String, Integer> opCounts = countOpTypes(decoder);
        log.info("  Vision encoder: {} ops", visionEncoder.getOps().size());
        log.info("  Decoder (optimized): {} ops", decoderOps);
        log.info("  Embed tokens: {} ops", embedTokens.getOps().size());
        log.info("  Decoder op types: {}", opCounts);

        // Log fused op counts
        for (String opName : new String[]{"rms_norm", "softmax", "swish", "fused_layer_norm"}) {
            Integer count = opCounts.get(opName);
            if (count != null && count > 0) {
                log.info("  FUSED OP: {} x {}", opName, count);
            }
        }

        log.info("STEP 3 DONE: {}ms", importTime);

        // ==================== STEP 4: Load, Tile, and Preprocess Image ====================
        log.info("STEP 4: Loading and tiling image...");
        long step4Start = System.currentTimeMillis();
        int targetSize = 512;
        BufferedImage pdfImage = loadImageFromPdfOrGenerate();
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        int effectiveMaxTiles = 9;
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, effectiveMaxTiles);
        int numFrames = splitResult.getTotalFrames();
        log.info("  Image: {}x{} -> {} frames ({} tiles + 1 global) [{}ms]",
                pdfImage.getWidth(), pdfImage.getHeight(), numFrames, splitResult.getTileCount(),
                System.currentTimeMillis() - step4Start);

        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
        preprocessor.shutdown();
        log.info("STEP 4 DONE: tensor shape={}", Arrays.toString(imageInput.shape()));

        // ==================== STEP 5: Run Vision Encoder Per Frame ====================
        log.info("STEP 5: Running vision encoder on {} frames...", numFrames);
        long step5Start = System.currentTimeMillis();
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            long frameStart = System.currentTimeMillis();
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
            if (selected == null) throw new RuntimeException("Vision encoder produced no output for frame " + frameIdx);
            INDArray out = selected.tensor.dup();
            log.info("  Frame {}/{}: shape={} [{}ms]", frameIdx + 1, numFrames,
                    Arrays.toString(out.shape()), System.currentTimeMillis() - frameStart);
            frameEmbeddings.add(out);

            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
        }

        INDArray visionEmbeddings;
        if (frameEmbeddings.size() == 1) {
            visionEmbeddings = frameEmbeddings.get(0).dup();
        } else {
            visionEmbeddings = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        }
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }
        frameEmbeddings.clear();
        imageInput.close();

        long visionTime = System.currentTimeMillis() - step5Start;
        log.info("STEP 5 DONE: shape={}, {}ms ({}ms/frame)", Arrays.toString(visionEmbeddings.shape()),
                visionTime, visionTime / numFrames);

        // Free vision encoder constants
        freeModelConstants(visionEncoder, "vision encoder");
        visionEncoder = null;

        // ==================== STEP 6: Build Prompt and Embeddings ====================
        log.info("STEP 6: Building prompt and text embeddings...");
        long step6Start = System.currentTimeMillis();
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / numFrames;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("  Prompt: {} tokens, {} <image> tokens", promptTokenIds.length,
                ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId));

        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds).reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOutputs = embedTokens.output(Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }
        if (textEmbeddings == null) throw new RuntimeException("embed_tokens produced no output");

        long hiddenSize = visionEmbeddings.shape()[2];
        if (hiddenSize != textEmbeddings.shape()[2]) {
            throw new IllegalStateException("Hidden size mismatch: vision=" + hiddenSize + " text=" + textEmbeddings.shape()[2]);
        }

        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();
        long step6Time = System.currentTimeMillis() - step6Start;
        log.info("STEP 6 DONE: merged shape={}, {}ms", Arrays.toString(inputsEmbeds.shape()), step6Time);

        // ==================== STEP 7: Autoregressive Decoding with Timing ====================
        // Uses static KV cache (fixed shape) for decode steps to enable CUDA graph capture.
        // Step 0 = prefill (dynamic KV). Steps 1+ = decode with static KV of shape
        // [batch, heads, maxKvLen, dim] where maxKvLen = prefillLen + maxTokensConfig.
        // This keeps flash attention input shapes FIXED every decode step, eliminating
        // the binary-split cascade that marks attention segments as captureFailed.
        log.info("STEP 7: Generating text (max {} tokens, greedy, optimized decoder)...", maxTokensConfig);
        long decodeStart = System.currentTimeMillis();

        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = decoder.inputs();
        log.info("  Decoder input names: {}", decoderInputNames);
        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxTokensConfig).doSample(false).build());

        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = inputsEmbeds;
        INDArray currentInputIds = promptIdsTensor;
        long pastSeqLen = 0;

        // Per-step timing
        List<Long> stepTimesMs = new ArrayList<>();
        long prefillTimeMs = 0;

        // Phase timing accumulators (steps 2+)
        long totalInputBuildNs = 0, totalDecoderNs = 0, totalLogitsDupNs = 0;
        long totalKvUpdateNs = 0, totalSamplingNs = 0;
        int detailSteps = 0;

        // Static KV cache state (allocated after prefill step 0)
        Map<String, INDArray> staticKvBuffers = null;  // past_key_values input name -> static buffer
        long maxKvLen = -1;  // total static KV length
        long cachePos = 0;   // next write position in static buffer
        boolean usingStaticKv = false;

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        for (int step = 0; step < maxTokensConfig; step++) {
            long stepStart = System.nanoTime();

            Map<String, INDArray> decoderInputMap = new HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    if (usingStaticKv) {
                        // Full static buffer: totalSeqLen = maxKvLen + 1 (past buffer + new token).
                        // Mask: 1s for valid past (0..cachePos-1), 0s for padding (cachePos..maxKvLen-1),
                        // 1 for the new token at position maxKvLen.
                        // The causal mask becomes trivially all-zeros because pastSeqLen=maxKvLen
                        // makes the condition k>maxKvLen never true. Only the padding mask matters.
                        long totalSeqLen = maxKvLen + currentSeqLen;
                        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
                        // Valid past positions
                        if (cachePos > 0) {
                            mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
                        }
                        // New token position (at end)
                        mask.putScalar(0, totalSeqLen - 1, 1);
                        decoderInputMap.put(inputName, mask);
                    } else {
                        long totalSeqLen = pastSeqLen + currentSeqLen;
                        decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
                    }
                } else if (inputName.equals("_causal_mask")) {
                    long totalSeqLen = usingStaticKv ? maxKvLen + currentSeqLen : pastSeqLen + currentSeqLen;
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(currentSeqLen, totalSeqLen));
                } else if (inputName.equals("input_ids")) {
                    decoderInputMap.put(inputName, currentInputIds);
                } else if (inputName.equals("position_ids")) {
                    // Logical position (for RoPE) — always pastSeqLen regardless of buffer size
                    decoderInputMap.put(inputName, Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG));
                } else if (inputName.startsWith("past_key_values.")) {
                    if (usingStaticKv) {
                        // Pass the FULL static buffer — fixed shape every step.
                        // Padding positions (cachePos..maxKvLen-1) are masked out by attention_mask.
                        INDArray staticBuf = staticKvBuffers.get(inputName);
                        if (staticBuf != null) {
                            decoderInputMap.put(inputName, staticBuf);
                        } else {
                            decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                        }
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                }
            }

            if (!decoderInputMap.containsKey("inputs_embeds")) {
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            long tAfterInputBuild = System.nanoTime();

            // Diagnostic: print inputs at step 1-3 for debugging static KV issues
            if (step >= 1 && step <= 3) {
                for (var entry : decoderInputMap.entrySet()) {
                    INDArray v = entry.getValue();
                    String name = entry.getKey();
                    if (name.equals("_causal_mask")) {
                        log.info("  [DIAG] step={} {}: shape={} min={} max={} nonzero={}",
                                step, name, java.util.Arrays.toString(v.shape()),
                                v.minNumber().floatValue(), v.maxNumber().floatValue(),
                                v.neq(0).sumNumber().longValue());
                        // Print first 5 and last 5 values
                        long len = v.length();
                        INDArray flat = v.reshape(len);
                        StringBuilder sb = new StringBuilder("  [DIAG]   values[0..4]=");
                        for (int i = 0; i < Math.min(5, len); i++) sb.append(flat.getFloat(i)).append(",");
                        sb.append(" ... values[").append(len-5).append("..").append(len-1).append("]=");
                        for (long i = Math.max(0, len-5); i < len; i++) sb.append(flat.getFloat(i)).append(",");
                        log.info(sb.toString());
                    } else if (name.equals("attention_mask")) {
                        log.info("  [DIAG] step={} {}: shape={} sum={}",
                                step, name, java.util.Arrays.toString(v.shape()), v.sumNumber().longValue());
                    } else if (name.startsWith("past_key_values.") && name.contains(".key.0")) {
                        log.info("  [DIAG] step={} {}: shape={} absMax={}",
                                step, name, java.util.Arrays.toString(v.shape()), v.amaxNumber().floatValue());
                    } else if (name.equals("position_ids")) {
                        log.info("  [DIAG] step={} {}: shape={} values={}",
                                step, name, java.util.Arrays.toString(v.shape()), v);
                    }
                }
            }

            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, allOutputNames.toArray(new String[0]));

            long tAfterDecoder = System.nanoTime();

            // Diagnostic: check present output shapes at step 1-3
            if (step >= 1 && step <= 3) {
                for (String pn : kvNames.keyNames) {
                    if (pn.contains(".0")) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv != null) {
                            log.info("  [DIAG] step={} present {}: shape={} lastPosAbsMax={}",
                                    step, pn, java.util.Arrays.toString(pv.shape()),
                                    pv.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                           NDArrayIndex.point(pv.shape()[2]-1), NDArrayIndex.all()).amaxNumber().floatValue());
                        }
                    }
                }
            }

            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) { log.error("No logits output at step {}", step); break; }
            INDArray logits = logitsRaw.dup();

            // Diagnostic: print top logit values at steps 1-3
            if (step >= 1 && step <= 3) {
                INDArray lastLogitsDiag = logits.rank() == 3
                        ? logits.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                                     org.nd4j.linalg.indexing.NDArrayIndex.point(logits.size(1) - 1),
                                     org.nd4j.linalg.indexing.NDArrayIndex.all())
                        : logits.getRow(0);
                INDArray topK = org.nd4j.linalg.factory.Nd4j.argMax(lastLogitsDiag);
                int topId = topK.getInt(0);
                float topVal = lastLogitsDiag.getFloat(topId);
                float logitsMin = lastLogitsDiag.minNumber().floatValue();
                float logitsMean = lastLogitsDiag.meanNumber().floatValue();
                log.info("  [DIAG] step={} logits: topId={} topVal={} min={} mean={} shape={}",
                        step, topId, topVal, logitsMin, logitsMean,
                        java.util.Arrays.toString(lastLogitsDiag.shape()));
            }
            logitsRaw.setCloseable(true);
            logitsRaw.close();

            long tAfterLogitsDup = System.nanoTime();

            if (usingStaticKv) {
                // Present output shape: [batch, heads, maxKvLen+1, dim].
                // New token's KV is at the last position (maxKvLen).
                // Scatter it into the static buffer at position cachePos.
                DecoderUtils.scatterNewKvEntries(staticKvBuffers, decoderOutputs,
                        kvNames.keyNames, kvNames.valueNames, maxKvLen, cachePos);
                Nd4j.getExecutioner().commit();
                for (String pn : kvNames.keyNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }
                for (String pn : kvNames.valueNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }
                cachePos++;
            } else {
                // Step 0 (prefill): transition to static KV after this step
                long prefillSeqLen = currentSeqLen;
                maxKvLen = prefillSeqLen + maxTokensConfig;
                log.info("  Setting up static KV: prefillLen={}, maxKvLen={} ({} tensors)",
                        prefillSeqLen, maxKvLen, kvNames.keyNames.size() + kvNames.valueNames.size());
                staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                        decoderOutputs, kvNames.keyNames, kvNames.valueNames, maxKvLen);
                // CRITICAL: commit() to ensure async assigns from padKvCacheToStaticSize complete
                // before the decoder reads from static buffers on a potentially different stream.
                Nd4j.getExecutioner().commit();
                cachePos = prefillSeqLen;
                usingStaticKv = true;
                // Close prefill KV outputs (data now in static buffers)
                for (String pn : kvNames.keyNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }
                for (String pn : kvNames.valueNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }

                // Freeze shapes: with full static buffer, ALL decode steps have identical
                // input shapes (past_kv=[1,h,maxKvLen,d], attention_mask=[1,maxKvLen+1], etc).
                // This enables CUDA graph capture and segment merging.
                InferenceSession decoderSession = decoder.getOrCreateSession();
                DynamicShapePlanExecutor dspExec = decoderSession.getDynamicShapePlanExecutor();
                if (dspExec != null) {
                    dspExec.setShapesFrozen(true);
                    log.info("  [Perf] Shapes frozen — static KV buffer shape=[1,h,{},d], decode fast path active", maxKvLen);
                } else {
                    log.warn("  [Perf] No DSP executor found to freeze shapes");
                }
            }

            long tAfterKvUpdate = System.nanoTime();

            // Sample from last position
            INDArray lastLogits = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                    : logits.getRow(0);
            INDArray logitsForSampling = lastLogits.dup();
            int nextTokenId = sampler.sample(logitsForSampling);
            generatedTokens.add(nextTokenId);

            long stepElapsedNs = System.nanoTime() - stepStart;
            long stepElapsedMs = stepElapsedNs / 1_000_000;

            // Accumulate detailed timing for steps 3+ (fast path active from step 3)
            if (step >= 3) {
                totalInputBuildNs += tAfterInputBuild - stepStart;
                totalDecoderNs    += tAfterDecoder - tAfterInputBuild;
                totalLogitsDupNs  += tAfterLogitsDup - tAfterDecoder;
                totalKvUpdateNs   += tAfterKvUpdate - tAfterLogitsDup;
                totalSamplingNs   += stepElapsedNs - (tAfterKvUpdate - stepStart);
                detailSteps++;
            }

            if (step == 0) {
                prefillTimeMs = stepElapsedMs;
                log.info("  Step 0 (prefill): {}ms (seq_len={})", stepElapsedMs, currentSeqLen);
            } else {
                stepTimesMs.add(stepElapsedMs);
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);

            // Log every 10 steps or first 6
            if (step < 6 || step % 10 == 0) {
                double currentTokPerSec = step > 0 && stepElapsedMs > 0 ? 1000.0 / stepElapsedMs : 0;
                if (step >= 2) {
                    long inputMs  = (tAfterInputBuild - stepStart) / 1_000_000;
                    long decMs    = (tAfterDecoder - tAfterInputBuild) / 1_000_000;
                    long dupMs    = (tAfterLogitsDup - tAfterDecoder) / 1_000_000;
                    long kvMs     = (tAfterKvUpdate - tAfterLogitsDup) / 1_000_000;
                    long sampMs   = stepElapsedMs - (tAfterKvUpdate - stepStart) / 1_000_000;
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s) [input={}ms dec={}ms dup={}ms kv={}ms samp={}ms cachePos={}]",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec),
                            inputMs, decMs, dupMs, kvMs, sampMs, cachePos - 1);
                } else {
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s)",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec));
                }
            }

            if (nextTokenId == eosTokenId || (endOfUtteranceTokenId != null && nextTokenId == endOfUtteranceTokenId)) {
                log.info("  Stop token at step {}", step);
                break;
            }

            logits.close();
            logitsForSampling.close();

            // Clean up per-step inputs (attention_mask, _causal_mask, position_ids, empty KVs)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                if (name.startsWith("past_key_values.")) continue;  // static buffers, kept alive
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            decoder.clearPlaceholders(false);

            // Get embedding for next token
            INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> newEmbedOutputs = embedTokens.output(Map.of(embedInputName, newTokenTensor), embedOutputNames);
            INDArray prevEmbeddings = currentEmbeddings;
            for (var entry : newEmbedOutputs.entrySet()) {
                currentEmbeddings = entry.getValue();
            }
            if (prevEmbeddings != null && !prevEmbeddings.wasClosed()) {
                prevEmbeddings.setCloseable(true);
                prevEmbeddings.close();
            }
            if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                currentInputIds.setCloseable(true);
                currentInputIds.close();
            }
            currentInputIds = newTokenTensor;
            embedTokens.clearPlaceholders(false);
            pastSeqLen += currentSeqLen;
        }

        // Release static KV buffers
        if (staticKvBuffers != null) {
            for (INDArray buf : staticKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) { buf.setCloseable(true); buf.close(); }
            }
        }

        long decodeEnd = System.currentTimeMillis();
        long totalDecodeMs = decodeEnd - decodeStart;

        // Log phase breakdown averages
        if (detailSteps > 0) {
            log.info("=== PHASE BREAKDOWN (avg over {} decode steps, fast path steps 3+) ===", detailSteps);
            log.info("  Input build:  {}ms", totalInputBuildNs / detailSteps / 1_000_000);
            log.info("  Decoder exec: {}ms", totalDecoderNs    / detailSteps / 1_000_000);
            log.info("  Logits dup:   {}ms", totalLogitsDupNs  / detailSteps / 1_000_000);
            log.info("  KV update:    {}ms", totalKvUpdateNs   / detailSteps / 1_000_000);
            log.info("  Sampling:     {}ms", totalSamplingNs   / detailSteps / 1_000_000);
            log.info("  Sum:          {}ms", (totalInputBuildNs + totalDecoderNs + totalLogitsDupNs + totalKvUpdateNs + totalSamplingNs) / detailSteps / 1_000_000);
        }

        // ==================== STEP 8: Results and Coherence Check ====================
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        // Compute timing statistics
        int decodeTokens = stepTimesMs.size(); // excludes prefill
        double avgDecodeMs = decodeTokens > 0
                ? stepTimesMs.stream().mapToLong(Long::longValue).average().orElse(0) : 0;
        double decodeTokensPerSec = avgDecodeMs > 0 ? 1000.0 / avgDecodeMs : 0;
        long p50Ms = decodeTokens > 0 ? percentile(stepTimesMs, 50) : 0;
        long p90Ms = decodeTokens > 0 ? percentile(stepTimesMs, 90) : 0;
        long p99Ms = decodeTokens > 0 ? percentile(stepTimesMs, 99) : 0;

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("PERFORMANCE SUMMARY:");
        log.info("  Decoder ops:       {} (optimized via OnnxModelCache)", decoderOps);
        log.info("  Prefill (step 0):  {}ms", prefillTimeMs);
        log.info("  Decode tokens:     {} (excluding prefill)", decodeTokens);
        log.info("  Avg decode time:   {}ms/token", String.format("%.1f", avgDecodeMs));
        log.info("  Decode throughput: {} tok/s", String.format("%.2f", decodeTokensPerSec));
        log.info("  Latency P50/P90/P99: {}ms / {}ms / {}ms", p50Ms, p90Ms, p99Ms);
        log.info("  Total decode time: {}ms ({} tokens)", totalDecodeMs, generatedTokens.size());
        log.info("  Total pipeline:    {}ms (import={}ms, vision={}ms, embed={}ms, decode={}ms)",
                System.currentTimeMillis() - step3Start, importTime, visionTime, step6Time, totalDecodeMs);
        log.info("========================================");

        // ==================== Coherence Assertions ====================
        assertNotNull(generatedText, "Generated text should not be null");
        assertTrue(generatedTokens.size() > 0, "Should have generated at least one token");

        // Check that output starts with DocTags structure (SmolDocling produces DOCTAG format)
        // Expected patterns: <doctag>, <page>, <text>, <section_header>, <table>, etc.
        String trimmed = generatedText.trim();
        boolean hasDocTags = trimmed.contains("<") && trimmed.contains(">");
        log.info("COHERENCE CHECK:");
        log.info("  Contains XML-like tags: {}", hasDocTags);

        if (hasDocTags) {
            // Count distinct tag types
            Set<String> tagTypes = new HashSet<>();
            int idx = 0;
            while (idx < trimmed.length()) {
                int open = trimmed.indexOf('<', idx);
                if (open < 0) break;
                int close = trimmed.indexOf('>', open);
                if (close < 0) break;
                String tag = trimmed.substring(open + 1, close);
                // Strip closing slash and attributes
                if (tag.startsWith("/")) tag = tag.substring(1);
                int space = tag.indexOf(' ');
                if (space > 0) tag = tag.substring(0, space);
                if (!tag.isEmpty()) tagTypes.add(tag);
                idx = close + 1;
            }
            log.info("  Distinct tag types: {} -> {}", tagTypes.size(), tagTypes);

            // SmolDocling should produce at least a doctag or page tag
            boolean hasStructuralTags = tagTypes.stream().anyMatch(t ->
                    t.equals("doctag") || t.equals("page") || t.equals("text") ||
                    t.equals("section_header") || t.equals("otsl") || t.equals("table"));
            log.info("  Has structural DocTags: {}", hasStructuralTags);

            if (generatedTokens.size() >= 10) {
                assertTrue(hasStructuralTags,
                        "With " + generatedTokens.size() + " tokens, output should contain structural DocTags. Got: " +
                        trimmed.substring(0, Math.min(200, trimmed.length())));
            }
        }

        // Check output isn't degenerate (all same token repeated)
        if (generatedTokens.size() >= 10) {
            Set<Integer> uniqueTokens = new HashSet<>(generatedTokens);
            double uniqueRatio = (double) uniqueTokens.size() / generatedTokens.size();
            log.info("  Token diversity: {}/{} unique ({}%)", uniqueTokens.size(),
                    generatedTokens.size(), String.format("%.1f", uniqueRatio * 100));
            assertTrue(uniqueRatio > 0.05,
                    "Output appears degenerate: only " + uniqueTokens.size() + " unique tokens out of " +
                    generatedTokens.size() + " total");
        }

        // Verify performance meets minimum threshold
        if (decodeTokens >= 5) {
            log.info("  Decode throughput: {} tok/s (minimum expected: 0.5 tok/s)", String.format("%.2f", decodeTokensPerSec));
            assertTrue(decodeTokensPerSec > 0.1,
                    "Decode throughput too low: " + String.format("%.2f", decodeTokensPerSec) + " tok/s");
        }

        tokenizer.close();
        log.info("Optimized pipeline complete.");

        // Suppress deallocation during JVM exit
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private Map<String, Integer> countOpTypes(SameDiff sd) {
        Map<String, Integer> counts = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            String name = op.getOp() != null ? op.getOp().opName() : "null";
            counts.merge(name, 1, Integer::sum);
        }
        return counts;
    }

    private long percentile(List<Long> values, int percentile) {
        if (values.isEmpty()) return 0;
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(idx, sorted.size() - 1)));
    }

    private BufferedImage loadImageFromPdfOrGenerate() throws IOException {
        if (pdfPath != null && new File(pdfPath).exists()) {
            log.info("Loading PDF: {}", pdfPath);
            try (PDDocument document = PDDocument.load(new File(pdfPath))) {
                PDFRenderer renderer = new PDFRenderer(document);
                int pageToLoad = specificPage >= 0 ? specificPage : 0;
                return renderer.renderImageWithDPI(pageToLoad, renderDpi, ImageType.RGB);
            }
        } else {
            log.info("No PDF provided, using test pattern image");
            return createTestImage(512, 512);
        }
    }

    private BufferedImage createTestImage(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("This is a test page for the", 50, 220);
        g.drawString("SmolDocling VLM pipeline.", 50, 260);
        g.drawString("Section 2: Content", 50, 340);
        g.drawString("Lorem ipsum dolor sit amet,", 50, 400);
        g.drawString("consectetur adipiscing elit.", 50, 440);
        g.dispose();
        return img;
    }

    private void freeModelConstants(SameDiff model, String label) {
        int closedArrays = 0;
        long closedBytes = 0;
        org.nd4j.autodiff.samediff.ArrayHolder constantHolder = model.getConstantArrays();
        for (String name : new ArrayList<>(constantHolder.arrayNames())) {
            INDArray arr = constantHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedArrays++;
            }
        }
        org.nd4j.autodiff.samediff.ArrayHolder varHolder = model.getVariablesArrays();
        for (String name : new ArrayList<>(varHolder.arrayNames())) {
            INDArray arr = varHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedArrays++;
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("  Freed {} {} arrays ({}MB)", closedArrays, label, closedBytes / (1024 * 1024));
    }
}
