/*
 * Bisect test: runs baseline decode logic (hardcoded strings, no ModelIOConfig,
 * no KvCacheManager) against the current C++/DSP stack to determine if the
 * degenerate output root cause is Java-side or C++-side.
 *
 * Three tests in progressive order:
 *   1. testBaselineDecodeNoDsp - pure baseline, no DSP at all
 *   2. testBaselineDecodeWithDsp - baseline + DSP recompile (no freeze, no graph capture)
 *   3. testBaselineDecodeWithDspAndAttnOverride - baseline + DSP + attn_mask_reformat override
 *
 * If test 1 passes but 2 fails → DSP recompile is the issue
 * If test 2 passes but 3 fails → attn_mask_reformat override is the issue
 * If all fail → C++ / DSP execution engine is the issue
 * If all pass → Java-side KvCacheManager/ModelIOConfig abstraction is the issue
 */
package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
public class TestBaselineVsHeadDecode {

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;

    @BeforeAll
    static void setup() throws Exception {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");

        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        // Generate test image
        int targetSize = 512;
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 512, 512);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 24));
        g.drawString("CREATING A MYTHIC CHARACTER", 50, 100);
        g.drawString("Heroes of myth and legend have", 50, 140);
        g.drawString("inspired storytellers for centuries.", 50, 180);
        g.dispose();

        BufferedImage resized = ImageTiler.resizeLongestEdge(img, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resized, targetSize, 9);
        int visionFrames = splitResult.getTotalFrames();

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

        // Vision encode
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int f = 0; f < visionFrames; f++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(f),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();
            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
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

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }
        imageInput.close();

        visionEncoder.close();

        // Build prompt + embeddings
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / visionFrames;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray embeddingMatrix = null;
        for (var v : embedTokens.variables()) {
            if (v.getArr() != null && v.getArr().rank() == 2) {
                embeddingMatrix = v.getArr();
                break;
            }
        }
        assertNotNull(embeddingMatrix, "Could not find embedding matrix in embed_tokens model");

        INDArray textEmbeddings = Nd4j.create(DataType.FLOAT, 1, promptTokenIds.length, embeddingMatrix.size(1));
        for (int i = 0; i < promptTokenIds.length; i++) {
            textEmbeddings.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all()},
                    embeddingMatrix.getRow(promptTokenIds[i]));
        }

        hiddenSize = visionEmbeddings.shape()[2];
        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);

        if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();

        log.info("Setup complete: embeddings shape={}, hiddenSize={}, promptLen={}",
                Arrays.toString(inputsEmbeds.shape()), hiddenSize, promptTokenIds.length);
    }

    /**
     * Helper: run baseline decode loop with growing KV cache.
     * Returns the generated token list.
     */
    private List<Integer> runBaselineDecode(int maxNewTokens, boolean compileDsp, boolean addAttnOverride) {
        decoder.getSessions().clear();
        decoder.clearPlaceholderOverrides();

        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = new ArrayList<>(decoder.inputs());

        List<String> fullOutputNameList = new ArrayList<>();
        fullOutputNameList.add(logitsOutputName);
        fullOutputNameList.addAll(kvNames.keyNames);
        fullOutputNameList.addAll(kvNames.valueNames);
        String[] fullOutputNames = fullOutputNameList.toArray(new String[0]);

        if (compileDsp) {
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);
            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);
            log.info("Initial DSP compile done");
        }

        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = inputsEmbeds;
        long currentSeqLen = inputsEmbeds.size(1);
        long pastSeqLen = 0;

        INDArray embeddingMatrix = null;
        for (var v : embedTokens.variables()) {
            if (v.getArr() != null && v.getArr().rank() == 2) {
                embeddingMatrix = v.getArr();
                break;
            }
        }

        // Growing KV cache: each step's "present" outputs become next step's "past" inputs.
        Map<String, INDArray> previousKvOutputs = null;

        for (int step = 0; step < maxNewTokens + 1; step++) {
            // Build inputs for this step
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    long totalSeqLen = pastSeqLen + currentSeqLen;
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .castTo(DataType.LONG).reshape(1, currentSeqLen);
                    decoderInputMap.put(inputName, posIds);
                } else if (inputName.startsWith("past_key_values.")) {
                    if (previousKvOutputs != null) {
                        String presentName = inputName.replace("past_key_values", "present");
                        INDArray prevKv = previousKvOutputs.get(presentName);
                        if (prevKv != null) {
                            decoderInputMap.put(inputName, prevKv.dup());
                        } else {
                            decoderInputMap.put(inputName,
                                    DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                        }
                    } else {
                        decoderInputMap.put(inputName,
                                DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                }
            }
            if (!decoderInputMap.containsKey("inputs_embeds")) {
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            // Clear session between steps
            decoder.getOrCreateSession().clearAllCaches();

            // Run forward pass
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, fullOutputNames);

            // Extract logits and compute argmax on host
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            assertNotNull(logitsRaw, "No logits at step " + step);

            INDArray lastLogits;
            if (logitsRaw.rank() == 3) {
                lastLogits = logitsRaw.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logitsRaw.size(1) - 1), NDArrayIndex.all());
            } else {
                lastLogits = logitsRaw.get(NDArrayIndex.point(0), NDArrayIndex.all());
            }

            Nd4j.getExecutioner().commit();
            INDArray dupLogits = lastLogits.dup();
            float[] logitsArr = dupLogits.data().asFloat();
            dupLogits.close();

            int nextTokenId = 0;
            float maxVal = logitsArr[0];
            for (int li = 1; li < logitsArr.length; li++) {
                if (logitsArr[li] > maxVal) {
                    maxVal = logitsArr[li];
                    nextTokenId = li;
                }
            }
            generatedTokens.add(nextTokenId);

            if (step <= 10) {
                log.info("  Step {} logits: topId={}, topVal={}", step, nextTokenId, maxVal);
            }

            // Save KV outputs for next step
            Nd4j.getExecutioner().commit();

            // Close previous KV arrays to free GPU memory
            if (previousKvOutputs != null) {
                for (INDArray prevKv : previousKvOutputs.values()) {
                    if (prevKv != null && prevKv.closeable() && !prevKv.wasClosed()) {
                        prevKv.close();
                    }
                }
            }
            previousKvOutputs = new HashMap<>();
            for (String keyName : kvNames.keyNames) {
                String presentName = keyName.replace("past_key_values", "present");
                INDArray kv = decoderOutputs.get(presentName);
                if (kv != null) {
                    previousKvOutputs.put(presentName, kv.dup());
                }
            }
            for (String valName : kvNames.valueNames) {
                String presentName = valName.replace("past_key_values", "present");
                INDArray kv = decoderOutputs.get(presentName);
                if (kv != null) {
                    previousKvOutputs.put(presentName, kv.dup());
                }
            }

            // Close input arrays we created this step (NOT the shared embeddings/matrix)
            for (INDArray arr : decoderInputMap.values()) {
                if (arr != null && arr != inputsEmbeds && arr != currentEmbeddings
                        && arr.closeable() && !arr.wasClosed()) {
                    arr.close();
                }
            }

            pastSeqLen += currentSeqLen;

            if (nextTokenId == 2) {
                log.info("  EOS at step {}", step);
                break;
            }

            // Next token embedding (view of embedding matrix — do NOT close)
            currentEmbeddings = embeddingMatrix.getRow(nextTokenId).reshape(1, 1, hiddenSize);
            currentSeqLen = 1;
        }

        // Clean up final KV arrays
        if (previousKvOutputs != null) {
            for (INDArray prevKv : previousKvOutputs.values()) {
                if (prevKv != null && prevKv.closeable() && !prevKv.wasClosed()) {
                    prevKv.close();
                }
            }
        }

        String text = tokenizer.decode(generatedTokens.stream().mapToInt(i -> i).toArray(), true);
        log.info("  Generated {} tokens: {}", generatedTokens.size(), text);

        Set<Integer> unique = new HashSet<>(generatedTokens);
        double ratio = generatedTokens.isEmpty() ? 0 : (double) unique.size() / generatedTokens.size();
        log.info("  Diversity: {}/{} unique ({}%)", unique.size(), generatedTokens.size(),
                String.format("%.1f", ratio * 100));

        return generatedTokens;
    }

    @Test
    public void testBaselineDecodeNoDsp() {
        log.info("=== BASELINE: No DSP, growing KV, hardcoded strings ===");
        // 9 decode steps: enough to detect degenerate repeating but avoids OOM
        // (growing KV + ~4GB intermediates per step exhausts 24GB GPU around step 10)
        List<Integer> tokens = runBaselineDecode(9, false, false);
        Set<Integer> unique = new HashSet<>(tokens);
        double ratio = tokens.isEmpty() ? 0 : (double) unique.size() / tokens.size();
        assertTrue(ratio >= 0.3, "BASELINE (no DSP): degenerate output: " + unique.size() + "/" +
                tokens.size() + " unique (" + String.format("%.1f%%", ratio * 100) + ")");
    }

    @Test
    public void testBaselineDecodeWithDsp() {
        log.info("=== BASELINE + DSP: DSP compile, no freeze, no graph capture ===");
        List<Integer> tokens = runBaselineDecode(9, true, false);
        Set<Integer> unique = new HashSet<>(tokens);
        double ratio = tokens.isEmpty() ? 0 : (double) unique.size() / tokens.size();
        assertTrue(ratio >= 0.3, "BASELINE+DSP: degenerate output: " + unique.size() + "/" +
                tokens.size() + " unique (" + String.format("%.1f%%", ratio * 100) + ")");
    }

    @Test
    public void testBaselineDecodeWithDspAndAttnOverride() {
        log.info("=== BASELINE + DSP + ATTN OVERRIDE: padded mode + 4D bias ===");
        List<Integer> tokens = runBaselineDecode(9, true, true);
        Set<Integer> unique = new HashSet<>(tokens);
        double ratio = tokens.isEmpty() ? 0 : (double) unique.size() / tokens.size();
        assertTrue(ratio >= 0.3, "BASELINE+DSP+ATTN: degenerate output: " + unique.size() + "/" +
                tokens.size() + " unique (" + String.format("%.1f%%", ratio * 100) + ")");
    }
}
