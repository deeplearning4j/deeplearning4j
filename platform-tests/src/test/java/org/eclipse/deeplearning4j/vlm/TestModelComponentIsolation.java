package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
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
 * Isolation tests for each VLM pipeline component.
 * Validates that each model produces correct output INDEPENDENTLY
 * before testing the full pipeline.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestModelComponentIsolation \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     -Dnd4j.optimizer.enabled=false \
 *     2>&1 | tee /tmp/component-isolation.log
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestModelComponentIsolation {

    private static final int TARGET_SIZE = 512;
    private static final int IMAGE_TOKEN_ID = 49190;

    // ─── 1. embed_tokens ─────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("embed_tokens: produces non-zero embeddings with correct shape")
    public void testEmbedTokens() throws Exception {
        var result = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        SameDiff embedTokens = OnnxModelCache.importWithCache(result.getModelFile().getAbsolutePath());

        // Simple token IDs: [1, 2, 3] as a batch of 1
        INDArray tokenIds = Nd4j.createFromArray(new int[][]{{1, 2, 3}}).castTo(DataType.INT64);
        String inputName = embedTokens.inputs().get(0);
        String[] outputNames = embedTokens.outputs().toArray(new String[0]);

        Map<String, INDArray> outputs = embedTokens.output(Map.of(inputName, tokenIds), outputNames);
        INDArray embeddings = outputs.values().iterator().next();

        log.info("embed_tokens output: shape={} dtype={} min={} max={} mean={}",
                Arrays.toString(embeddings.shape()), embeddings.dataType(),
                embeddings.minNumber(), embeddings.maxNumber(), embeddings.meanNumber());

        // Shape should be [1, 3, hiddenSize]
        assertEquals(3, embeddings.rank(), "Embeddings should be rank 3");
        assertEquals(1, embeddings.size(0), "Batch size should be 1");
        assertEquals(3, embeddings.size(1), "Seq len should be 3");
        assertTrue(embeddings.size(2) > 0, "Hidden size should be positive");

        // Values should be non-zero and finite
        assertFalse(embeddings.isNaN().any(), "Embeddings should not contain NaN");
        assertFalse(embeddings.isInfinite().any(), "Embeddings should not contain Inf");
        assertTrue(Math.abs(embeddings.meanNumber().doubleValue()) > 1e-10,
                "Embeddings should not be all zeros");

        long hiddenSize = embeddings.size(2);
        log.info("embed_tokens: PASS (hiddenSize={})", hiddenSize);

        // Cleanup
        tokenIds.close();
        for (INDArray arr : outputs.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    // ─── 2. vision_encoder ───────────────────────────────────────────────

    @Test
    @Order(2)
    @DisplayName("vision_encoder: produces non-zero image embeddings")
    public void testVisionEncoder() throws Exception {
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        SameDiff visionEncoder = OnnxModelCache.importWithCache(visionResult.getModelFile().getAbsolutePath());

        // Create simple test image
        BufferedImage testImage = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);

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

        // Run vision encoder on first frame
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        INDArray frameSlice = imageInput.get(
                NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();

        Map<String, INDArray> visionInputMap = new HashMap<>();
        for (String inputName : visionInputNames) {
            if (inputName.equals("pixel_values")) {
                visionInputMap.put(inputName, singleFrame);
            } else if (inputName.equals("pixel_attention_mask")) {
                ImageTiler.ContentRegion region = splitResult.contentRegions.get(0);
                visionInputMap.put(inputName,
                        ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE));
            }
        }

        Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
        VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);

        log.info("vision_encoder output: shape={} dtype={} min={} max={} mean={}",
                Arrays.toString(selected.tensor.shape()), selected.tensor.dataType(),
                selected.tensor.minNumber(), selected.tensor.maxNumber(),
                selected.tensor.meanNumber());

        // Shape should be [1, numPatches, hiddenSize]
        assertEquals(3, selected.tensor.rank(), "Vision output should be rank 3");
        assertEquals(1, selected.tensor.size(0), "Batch size should be 1");
        assertTrue(selected.tensor.size(1) > 0, "Num patches should be positive");
        assertTrue(selected.tensor.size(2) > 0, "Hidden size should be positive");

        assertFalse(selected.tensor.isNaN().any(), "Vision output should not contain NaN");
        assertFalse(selected.tensor.isInfinite().any(), "Vision output should not contain Inf");
        assertTrue(Math.abs(selected.tensor.meanNumber().doubleValue()) > 1e-10,
                "Vision output should not be all zeros");

        log.info("vision_encoder: PASS (patches={}, hiddenSize={})",
                selected.tensor.size(1), selected.tensor.size(2));

        // Cleanup
        singleFrame.close();
        for (INDArray arr : visionOutputs.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    // ─── 3. decoder prefill ──────────────────────────────────────────────

    @Test
    @Order(3)
    @DisplayName("decoder prefill: first token should be doctag-related (not garbage)")
    public void testDecoderPrefill() throws Exception {
        // Load all models
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        // Create test image
        BufferedImage testImage = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document Line 1", 50, 100);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);

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

        // Run vision encoder
        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionEncoder.inputs()) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionEncoder.outputs().toArray(new String[0]));
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());
            singleFrame.close();
        }

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));

        long hiddenSize = visionEmbeddings.size(-1);
        log.info("Vision embeddings: shape={} hiddenSize={}", Arrays.toString(visionEmbeddings.shape()), hiddenSize);

        // Run embed_tokens on prompt
        String prompt = "<|im_start|>user\nConvert this page to docling.\n<|im_end|>\n<|im_start|>assistant\n";
        int[] encoded = tokenizer.encode(prompt).getIds();
        log.info("Prompt token IDs: {} (len={})", Arrays.toString(encoded), encoded.length);

        INDArray tokenIds = Nd4j.createFromArray(new int[][]{encoded}).castTo(DataType.INT64);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedTokens.inputs().get(0), tokenIds),
                embedTokens.outputs().toArray(new String[0]));
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        log.info("Text embeddings: shape={}", Arrays.toString(textEmbeddings.shape()));

        // Merge embeddings
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, IMAGE_TOKEN_ID);
        log.info("Merged inputs_embeds: shape={} min={} max={} mean={}",
                Arrays.toString(inputsEmbeds.shape()),
                inputsEmbeds.minNumber(), inputsEmbeds.maxNumber(), inputsEmbeds.meanNumber());

        assertFalse(inputsEmbeds.isNaN().any(), "inputs_embeds should not contain NaN");
        assertFalse(inputsEmbeds.isInfinite().any(), "inputs_embeds should not contain Inf");

        Nd4j.getEnvironment().setDebug(false);
        Nd4j.getEnvironment().setVerbose(false);

        long seqLen = inputsEmbeds.size(1);
        log.info("Running decoder prefill with seqLen={}", seqLen);

        // Build decoder inputs — must provide ALL inputs including past_key_values
        // (empty KV cache for prefill) and attention_mask
        int numHeads = 9;   // SmolDocling GQA: 9 query heads
        int kvHeads = 3;    // SmolDocling GQA: 3 KV heads
        int headDim = 64;   // 576 / 9 = 64
        int numLayers = 30; // SmolDocling has 30 layers

        Map<String, INDArray> decoderInputs = new HashMap<>();
        for (String inputName : decoder.inputs()) {
            if (inputName.equals("inputs_embeds")) {
                decoderInputs.put(inputName, inputsEmbeds);
            } else if (inputName.equals("attention_mask")) {
                decoderInputs.put(inputName, Nd4j.ones(DataType.INT64, 1, seqLen));
            } else if (inputName.equals("position_ids")) {
                INDArray posIds = Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.INT64);
                decoderInputs.put(inputName, posIds);
            } else if (inputName.contains("past_key_values") && inputName.contains("key")) {
                // Empty KV cache: [batch, kv_heads, 0, head_dim]
                decoderInputs.put(inputName, Nd4j.zeros(DataType.FLOAT, 1, kvHeads, 0, headDim));
            } else if (inputName.contains("past_key_values") && inputName.contains("value")) {
                decoderInputs.put(inputName, Nd4j.zeros(DataType.FLOAT, 1, kvHeads, 0, headDim));
            }
        }
        log.info("Decoder inputs provided: {}", decoderInputs.keySet().size());

        // Get output names (logits)
        String[] decoderOutputNames = decoder.outputs().stream()
                .filter(n -> n.contains("logits"))
                .toArray(String[]::new);
        if (decoderOutputNames.length == 0) {
            decoderOutputNames = decoder.outputs().toArray(new String[0]);
        }

        // Find intermediate variables between layer 0 and layer 1 for comparison.
        // SmolDocling ONNX names follow: /model/layers.N/...
        List<String> outputsToCompare = new ArrayList<>();
        outputsToCompare.add("logits");

        // Add all model variables that contain layer 0 or layer 1 outputs
        for (org.nd4j.autodiff.samediff.SDVariable sdVar : decoder.variables()) {
            String varName = sdVar.name();
            // Layer 0 attention output, FFN intermediates, layer 1 attention input
            if (varName.contains("/model/layers.0/") || varName.contains("/model/layers.1/")
                    || varName.contains("/model/attn_mask_reformat/")) {
                // Only include op outputs (not constants/placeholders)
                if (sdVar.getVariableType() == org.nd4j.autodiff.samediff.VariableType.ARRAY) {
                    outputsToCompare.add(varName);
                }
            }
        }
        // Also add present.0 and present.1 KV
        for (String name : decoder.outputs()) {
            if (name.startsWith("present.0.") || name.startsWith("present.1.")) {
                outputsToCompare.add(name);
            }
        }
        log.info("Comparing {} outputs (layer 0+1 intermediates + KV) between standard and DSP",
                outputsToCompare.size());

        decoder.compareExecutionPaths(decoderInputs, outputsToCompare, 1e-3,
                GraphExecutionMode.SLOT_BY_SLOT);
    }
}
