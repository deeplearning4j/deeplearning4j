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
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
        // Enable graph optimizer for this test (unless overridden by system property)
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        System.setProperty("nd4j.optimizer.logApplied", "true");
        // FP16 quantization: converts all FLOAT constants/variables to HALF
        // MmulHelper.cu persistent cast cache handles HALF→FLOAT for CUDA graph capture
        // FP16 quantization: gated by property, helps larger models (>1024-dim)
        // For SmolDocling (576-dim), cast overhead negates tensor core benefit
        // System.setProperty("nd4j.optimizer.fp16", "true");
        
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
        Nd4j.getEnvironment().setTritonBuildThreads(4);

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

        // Explicit DSP/native compilation for all models (no lazy compile during decode).
        // When a specific GraphExecutionMode is requested (CUDA_GRAPHS, SLOT_BY_SLOT, AUTO),
        // compile the DSP plan with that execution mode directly (no Triton compilation needed).
        // Only skip precompile when dsp.compilation.mode=NONE is explicitly requested.
        String gemProp = System.getProperty("nd4j.dsp.graphExecutionMode");
        String dspModeProp = System.getProperty("vlm.test.dsp.compilation.mode", "MAX_AUTOTUNE");
        boolean skipPrecompile = "NONE".equalsIgnoreCase(dspModeProp);

        // Check if a non-Triton execution mode is requested — compile DSP with that mode directly
        GraphExecutionMode overrideMode = null;
        if (gemProp != null) {
            try {
                overrideMode = GraphExecutionMode.valueOf(normalizePropertyToken(gemProp));
            } catch (Exception e) {
                log.warn("Invalid nd4j.dsp.graphExecutionMode='{}', ignoring", gemProp);
            }
        }

        if (!skipPrecompile) {
            // Also check if vlm.test.dsp.compilation.mode is actually a GraphExecutionMode name
            // (e.g., SLOT_BY_SLOT, CUDA_GRAPHS, AUTO). If so, treat it as a non-Triton mode override.
            if (overrideMode == null && dspModeProp != null) {
                String dspModeCanonical = normalizePropertyToken(dspModeProp);
                try {
                    GraphExecutionMode gemFromDspProp = GraphExecutionMode.valueOf(dspModeCanonical);
                    if (gemFromDspProp != GraphExecutionMode.TRITON) {
                        overrideMode = gemFromDspProp;
                        log.info("  vlm.test.dsp.compilation.mode='{}' is a GraphExecutionMode, using as override", dspModeProp);
                    }
                } catch (IllegalArgumentException ignored) {
                    // Not a GraphExecutionMode — will be parsed as DspCompilationMode below
                }
            }

            // Set verify kernels flag for ALL modes (Triton, CUDA_GRAPHS, SLOT_BY_SLOT, etc.)
            Nd4j.getEnvironment().setTritonVerifyKernels(boolProp("vlm.test.triton.verifyKernels", false));
            if (Nd4j.getEnvironment().tritonVerifyKernels()) {
                log.info("  Replay verify enabled (tritonVerifyKernels=true)");
            }

            if (overrideMode != null && overrideMode != GraphExecutionMode.TRITON) {
                // Non-Triton mode: compile DSP plan with the requested execution mode.
                // Uses REDUCE_OVERHEAD profile (no Triton compilation) with mode override.
                log.info("STEP 3b: DSP precompile with execution mode {} (no Triton)", overrideMode);
                precompileModelWithMode(visionEncoder, "vision encoder", overrideMode);
                precompileModelWithMode(decoder, "decoder", overrideMode);
                precompileModelWithMode(embedTokens, "embed tokens", overrideMode);
            } else {
                // Triton/default path: use DspCompilationMode profile
                String dspModeCanonical = normalizePropertyToken(dspModeProp);
                DspCompilationMode dspMode;
                try {
                    dspMode = DspCompilationMode.valueOf(dspModeCanonical);
                } catch (Exception e) {
                    throw new IllegalArgumentException(
                            "Invalid vlm.test.dsp.compilation.mode='" + dspModeProp + "' (normalized='" + dspModeCanonical +
                                    "'). Expected one of: " + Arrays.toString(DspCompilationMode.values()) +
                                    " or GraphExecutionMode values: " + Arrays.toString(GraphExecutionMode.values()), e);
                }
                String tritonProfile = configureTritonCompileProfile(dspMode);

                // Triton diagnostic flags (apply after profile, override profile defaults)
                Nd4j.getEnvironment().setTritonGraphCapture(boolProp("vlm.test.triton.graphCapture", true));
                Nd4j.getEnvironment().setTritonSkipKernels(boolProp("vlm.test.triton.skipKernels", false));
                Nd4j.getEnvironment().setTritonVerifyKernels(boolProp("vlm.test.triton.verifyKernels", false));
                Nd4j.getEnvironment().setTritonVerifyKeepNative(boolProp("vlm.test.triton.verifyKeepNative", false));
                Nd4j.getEnvironment().setTritonVerifyFullSnapshot(boolProp("vlm.test.triton.verifyFullSnapshot", false));
                Nd4j.getEnvironment().setTritonMaxSubKernelIndex(intProp("vlm.test.triton.maxSubKernelIndex", -1));
                log.info("  Triton diagnostics: graphCapture={}, skipKernels={}, verifyKernels={}, verifyKeepNative={}, verifyFullSnapshot={}, maxSubKernelIndex={}",
                        Nd4j.getEnvironment().tritonGraphCapture(),
                        Nd4j.getEnvironment().tritonSkipKernels(),
                        Nd4j.getEnvironment().tritonVerifyKernels(),
                        Nd4j.getEnvironment().tritonVerifyKeepNative(),
                        Nd4j.getEnvironment().tritonVerifyFullSnapshot(),
                        Nd4j.getEnvironment().tritonMaxSubKernelIndex());

                if (Nd4j.getNativeOps().isTritonAvailable()) {
                    Nd4j.getNativeOps().resetTritonCounters();
                }
                log.info("STEP 3b: DSP/native precompile all models (requestedMode='{}', normalized='{}', selectedMode={}, tritonProfile={})",
                        dspModeProp, dspModeCanonical, dspMode, tritonProfile);
                logTritonKnobs("Environment before precompile");
                logTritonCounters("before-precompile");
                precompileModelForPipeline(visionEncoder, "vision encoder", dspMode);
                precompileModelForPipeline(decoder, "decoder", dspMode);
                precompileModelForPipeline(embedTokens, "embed tokens", dspMode);
                logTritonCounters("after-precompile");
            }
        } else {
            log.info("STEP 3b: SKIPPING precompile (graphExecutionMode={}, dsp.compilation.mode={})", gemProp, dspModeProp);
        }

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
        log.info("STEP 7: Generating text (max {} tokens, greedy, optimized decoder)...", maxTokensConfig);

        StaticKvCacheDecodeLoop decodeLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokensConfig)
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = decodeLoop.decode(inputsEmbeds, promptTokenIds);
        logTritonCounters("after-decode");

        String generatedText = result.getText();
        long totalDecodeMs = result.getGenerationTimeMs();

        log.info("  Decoder ops:       {} (optimized via OnnxModelCache)", decoderOps);
        log.info("  Total pipeline:    {}ms (import={}ms, vision={}ms, embed={}ms, decode={}ms)",
                System.currentTimeMillis() - step3Start, importTime, visionTime, step6Time, totalDecodeMs);

        // ==================== STEP 8: Coherence Assertions ====================
        assertNotNull(generatedText, "Generated text should not be null");
        assertTrue(result.getGeneratedTokenCount() > 0, "Should have generated at least one token");

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

            if (result.getGeneratedTokenCount() >= 10) {
                assertTrue(hasStructuralTags,
                        "With " + result.getGeneratedTokenCount() + " tokens, output should contain structural DocTags. Got: " +
                        trimmed.substring(0, Math.min(200, trimmed.length())));
            }
        }

        // Check output isn't degenerate (all same token repeated)
        if (result.getGeneratedTokenCount() >= 10) {
            int[] tokenIds = result.getTokenIds();
            Set<Integer> uniqueTokens = new HashSet<>();
            for (int id : tokenIds) uniqueTokens.add(id);
            double uniqueRatio = (double) uniqueTokens.size() / tokenIds.length;
            log.info("  Token diversity: {}/{} unique ({}%)", uniqueTokens.size(),
                    tokenIds.length, String.format("%.1f", uniqueRatio * 100));
            assertTrue(uniqueRatio > 0.05,
                    "Output appears degenerate: only " + uniqueTokens.size() + " unique tokens out of " +
                    tokenIds.length + " total");
        }

        // Verify performance meets minimum threshold
        if (result.getGeneratedTokenCount() >= 5) {
            log.info("  Decode throughput: {} tok/s (minimum expected: 0.5 tok/s)",
                    String.format("%.2f", result.getTokensPerSecond()));
            assertTrue(result.getTokensPerSecond() > 0.1,
                    "Decode throughput too low: " + String.format("%.2f", result.getTokensPerSecond()) + " tok/s");
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

    @Test
    @DisplayName("Test KvScatter op in isolation")
    public void testKvScatterIsolated() {
        int batch = 1, heads = 8, maxKvLen = 100, dim = 64;
        long cachePos = 5;

        INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, maxKvLen + 1, dim);
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

        log.info("present shape: {}, staticBuf shape: {}",
                Arrays.toString(present.shape()), Arrays.toString(staticBuf.shape()));

        // Inplace op: input 0 = static (destination), input 1 = present (source)
        org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter scatter =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(staticBuf, present, cachePos);

        // Log a sample value from present at lastPos before exec
        float presentVal = present.getFloat(0, 0, maxKvLen, 0);
        float staticValBefore = staticBuf.getFloat(0, 0, (int)cachePos, 0);
        log.info("BEFORE exec: present[0,0,{},0]={}, staticBuf[0,0,{},0]={}",
                maxKvLen, presentVal, cachePos, staticValBefore);

        INDArray[] result = Nd4j.getExecutioner().exec(scatter);
        log.info("KvScatter executed, result length: {}", result.length);

        // Check if result[0] is same object as staticBuf
        if (result.length > 0) {
            log.info("result[0] == staticBuf: {}", (result[0] == staticBuf));
            log.info("result[0] data ptr == staticBuf data ptr: {}",
                    (result[0].data().address() == staticBuf.data().address()));
            float resultVal = result[0].getFloat(0, 0, (int)cachePos, 0);
            log.info("result[0][0,0,{},0]={}", cachePos, resultVal);
        }

        float staticValAfter = staticBuf.getFloat(0, 0, (int)cachePos, 0);
        log.info("AFTER exec: staticBuf[0,0,{},0]={}", cachePos, staticValAfter);

        // Verify: the entry at cachePos in staticBuf should match the last position in present
        // Use dup() on views to ensure clean host/device sync
        INDArray expectedEntry = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).dup();
        INDArray actualEntry = result[0].get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

        double maxDiff = expectedEntry.sub(actualEntry).amaxNumber().doubleValue();
        log.info("Max diff between expected and actual: {}", maxDiff);
        assertTrue(maxDiff < 1e-5, "KV scatter should copy present's last pos to static's cachePos");

        present.close();
        staticBuf.close();
        log.info("KvScatter isolation test passed!");
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

    private String configureTritonCompileProfile(DspCompilationMode mode) {
        String defaultProfile = "BALANCED";
        String rawProfile = System.getProperty("vlm.test.triton.profile", defaultProfile);
        String profile = normalizePropertyToken(rawProfile);

        // Op exclusion list and compile-all mode (applied after profile-specific settings)
        // These are set early so profile-specific overrides can also adjust them.
        boolean compileAll = boolProp("vlm.test.triton.compileAll", false);
        String excludeOps = System.getProperty("vlm.test.triton.excludeOps", "");
        String includeTypes = System.getProperty("vlm.test.triton.includeTypes", "");
        Nd4j.getEnvironment().setTritonCompileAll(compileAll);
        if (!excludeOps.isEmpty()) {
            Nd4j.getEnvironment().setTritonExcludeOps(excludeOps);
        }
        if (!includeTypes.isEmpty()) {
            Nd4j.getEnvironment().setTritonIncludeTypes(includeTypes);
        }
        if (compileAll) {
            log.info("  Triton compile-all mode ENABLED, excludeOps='{}', includeTypes='{}'", excludeOps, includeTypes);
        }

        if ("DEBUG_FAST".equals(profile)) {
            // Fast startup profile for validating Triton execution in debug runs.
            Nd4j.getEnvironment().setTritonBuildThreads(intProp("vlm.test.triton.buildThreads", 1));
            Nd4j.getEnvironment().setTritonCacheEnabled(boolProp("vlm.test.triton.cacheEnabled", true));
            Nd4j.getEnvironment().setTritonAlwaysCompile(boolProp("vlm.test.triton.alwaysCompile", false));
            Nd4j.getEnvironment().setTritonMaxSubsegmentOps(intProp("vlm.test.triton.maxSubsegmentOps", 8));
            Nd4j.getEnvironment().setTritonMaxSubsegmentSections(intProp("vlm.test.triton.maxSubsegmentSections", 2));
            Nd4j.getEnvironment().setTritonNumWarps(intProp("vlm.test.triton.numWarps", 2));
            Nd4j.getEnvironment().setTritonNumStages(intProp("vlm.test.triton.numStages", 2));
            Nd4j.getEnvironment().setTritonNumCTAs(intProp("vlm.test.triton.numCTAs", 1));
            Nd4j.getEnvironment().setTritonMaxNreg(intProp("vlm.test.triton.maxNreg", 64));
            Nd4j.getEnvironment().setTritonEnableFpFusion(boolProp("vlm.test.triton.enableFpFusion", false));
            Nd4j.getEnvironment().setTritonDisableLineInfo(boolProp("vlm.test.triton.disableLineInfo", true));
            Nd4j.getEnvironment().setTritonVerbose(boolProp("vlm.test.triton.verbose", true));
            Nd4j.getEnvironment().setTritonDumpSections(boolProp("vlm.test.triton.dumpSections", true));
            Nd4j.getEnvironment().setTritonDumpArgs(boolProp("vlm.test.triton.dumpArgs", false));
            Nd4j.getEnvironment().setTritonKernelDump(boolProp("vlm.test.triton.kernelDump", false));
            Nd4j.getEnvironment().setTritonLogAllPatterns(boolProp("vlm.test.triton.logAllPatterns", false));
            Nd4j.getEnvironment().setTritonCoopTargetBlocks(intProp("vlm.test.triton.coopTargetBlocks", 1));
            return profile;
        }

        if ("BALANCED".equals(profile)) {
            // Balanced profile: ~90%-style runtime throughput with significantly lower compile latency.
            // MAX / all-in-one fused kernel reference (set via system properties when needed):
            //   vlm.test.triton.maxSubsegmentOps=0
            //   vlm.test.triton.maxSubsegmentSections=0
            //   vlm.test.triton.coopTargetBlocks=0
            //   vlm.test.triton.numWarps=8
            //   vlm.test.triton.numStages=2
            //   vlm.test.triton.buildThreads=4
            // Keep MAX values out of defaults so runtime behavior stays user-controlled.
            Nd4j.getEnvironment().setTritonBuildThreads(intProp("vlm.test.triton.buildThreads", 4));
            Nd4j.getEnvironment().setTritonCacheEnabled(boolProp("vlm.test.triton.cacheEnabled", true));
            Nd4j.getEnvironment().setTritonAlwaysCompile(boolProp("vlm.test.triton.alwaysCompile", false));
            Nd4j.getEnvironment().setTritonMaxSubsegmentOps(intProp("vlm.test.triton.maxSubsegmentOps", 0));
            Nd4j.getEnvironment().setTritonMaxSubsegmentSections(intProp("vlm.test.triton.maxSubsegmentSections", 0));
            Nd4j.getEnvironment().setTritonNumWarps(intProp("vlm.test.triton.numWarps", 8));
            Nd4j.getEnvironment().setTritonNumStages(intProp("vlm.test.triton.numStages", 2));
            Nd4j.getEnvironment().setTritonNumCTAs(intProp("vlm.test.triton.numCTAs", 1));
            Nd4j.getEnvironment().setTritonMaxNreg(intProp("vlm.test.triton.maxNreg", 0));
            Nd4j.getEnvironment().setTritonEnableFpFusion(boolProp("vlm.test.triton.enableFpFusion", true));
            Nd4j.getEnvironment().setTritonDisableLineInfo(boolProp("vlm.test.triton.disableLineInfo", true));
            Nd4j.getEnvironment().setTritonVerbose(boolProp("vlm.test.triton.verbose", true));
            Nd4j.getEnvironment().setTritonDumpSections(boolProp("vlm.test.triton.dumpSections", false));
            Nd4j.getEnvironment().setTritonDumpArgs(boolProp("vlm.test.triton.dumpArgs", false));
            Nd4j.getEnvironment().setTritonKernelDump(boolProp("vlm.test.triton.kernelDump", false));
            Nd4j.getEnvironment().setTritonLogAllPatterns(boolProp("vlm.test.triton.logAllPatterns", false));
            Nd4j.getEnvironment().setTritonCoopTargetBlocks(intProp("vlm.test.triton.coopTargetBlocks", 0));
            Nd4j.getEnvironment().setTritonCooperativeLaunch(boolProp("vlm.test.triton.cooperativeLaunch", false));
            return profile;
        }

        if ("MAX_PERF".equals(profile) || "MAX_AUTOTUNE".equals(profile)) {
            // Single fused kernel, no cooperative launch, no splitting, max warps
            Nd4j.getEnvironment().setTritonBuildThreads(intProp("vlm.test.triton.buildThreads", 4));
            Nd4j.getEnvironment().setTritonCacheEnabled(boolProp("vlm.test.triton.cacheEnabled", true));
            Nd4j.getEnvironment().setTritonAlwaysCompile(boolProp("vlm.test.triton.alwaysCompile", false));
            Nd4j.getEnvironment().setTritonMaxSubsegmentOps(intProp("vlm.test.triton.maxSubsegmentOps", 0));
            Nd4j.getEnvironment().setTritonMaxSubsegmentSections(intProp("vlm.test.triton.maxSubsegmentSections", 0));
            Nd4j.getEnvironment().setTritonNumWarps(intProp("vlm.test.triton.numWarps", 8));
            Nd4j.getEnvironment().setTritonNumStages(intProp("vlm.test.triton.numStages", 2));
            Nd4j.getEnvironment().setTritonNumCTAs(intProp("vlm.test.triton.numCTAs", 1));
            Nd4j.getEnvironment().setTritonMaxNreg(intProp("vlm.test.triton.maxNreg", 0));
            Nd4j.getEnvironment().setTritonEnableFpFusion(boolProp("vlm.test.triton.enableFpFusion", true));
            Nd4j.getEnvironment().setTritonDisableLineInfo(boolProp("vlm.test.triton.disableLineInfo", true));
            Nd4j.getEnvironment().setTritonCoopTargetBlocks(intProp("vlm.test.triton.coopTargetBlocks", 0));
            Nd4j.getEnvironment().setTritonCooperativeLaunch(boolProp("vlm.test.triton.cooperativeLaunch", false));
            Nd4j.getEnvironment().setTritonVerbose(boolProp("vlm.test.triton.verbose", false));
            Nd4j.getEnvironment().setTritonDumpSections(boolProp("vlm.test.triton.dumpSections", false));
            Nd4j.getEnvironment().setTritonDumpArgs(boolProp("vlm.test.triton.dumpArgs", false));
            Nd4j.getEnvironment().setTritonKernelDump(boolProp("vlm.test.triton.kernelDump", false));
            Nd4j.getEnvironment().setTritonLogAllPatterns(boolProp("vlm.test.triton.logAllPatterns", false));
            return profile;
        }

        throw new IllegalArgumentException(
                "Invalid vlm.test.triton.profile='" + rawProfile + "' (normalized='" + profile +
                        "'). Expected one of: DEBUG_FAST, BALANCED, MAX_PERF");
    }

    private void logTritonKnobs(String prefix) {
        log.info("  {}: tritonBuildThreads={}, tritonCacheEnabled={}, tritonAlwaysCompile={}, tritonMaxSubsegmentOps={}, tritonMaxSubsegmentSections={}, tritonNumWarps={}, tritonNumStages={}, tritonNumCTAs={}, tritonMaxNreg={}, tritonCoopTargetBlocks={}, tritonEnableFpFusion={}, tritonDisableLineInfo={}, tritonVerbose={}, tritonDumpSections={}, tritonDumpArgs={}, tritonKernelDump={}, tritonLogAllPatterns={}",
                prefix,
                Nd4j.getEnvironment().tritonBuildThreads(),
                Nd4j.getEnvironment().tritonCacheEnabled(),
                Nd4j.getEnvironment().tritonAlwaysCompile(),
                Nd4j.getEnvironment().tritonMaxSubsegmentOps(),
                Nd4j.getEnvironment().tritonMaxSubsegmentSections(),
                Nd4j.getEnvironment().tritonNumWarps(),
                Nd4j.getEnvironment().tritonNumStages(),
                Nd4j.getEnvironment().tritonNumCTAs(),
                Nd4j.getEnvironment().tritonMaxNreg(),
                Nd4j.getEnvironment().tritonCoopTargetBlocks(),
                Nd4j.getEnvironment().tritonEnableFpFusion(),
                Nd4j.getEnvironment().tritonDisableLineInfo(),
                Nd4j.getEnvironment().tritonVerbose(),
                Nd4j.getEnvironment().tritonDumpSections(),
                Nd4j.getEnvironment().tritonDumpArgs(),
                Nd4j.getEnvironment().tritonKernelDump(),
                Nd4j.getEnvironment().tritonLogAllPatterns());
    }

    private void logTritonCounters(String stage) {
        boolean available = Nd4j.getNativeOps().isTritonAvailable();
        long launches = Nd4j.getNativeOps().getTritonKernelLaunchCount();
        long cacheHits = Nd4j.getNativeOps().getTritonCacheHitCount();
        log.info("  Triton counters [{}]: available={} kernelLaunches={} cacheHits={}",
                stage, available, launches, cacheHits);
    }

    private String normalizePropertyToken(String raw) {
        return raw == null ? "" : raw.trim().toUpperCase(Locale.ROOT).replace('-', '_').replace(' ', '_');
    }

    private static int intProp(String key, int defaultValue) {
        return Integer.getInteger(key, defaultValue);
    }

    private static boolean boolProp(String key, boolean defaultValue) {
        return Boolean.parseBoolean(System.getProperty(key, Boolean.toString(defaultValue)));
    }

    private void precompileModelWithMode(SameDiff model, String label, GraphExecutionMode mode) {
        List<String> modelOutputs = model.outputs() == null ? Collections.emptyList() : new ArrayList<>(model.outputs());
        log.info("  Precompiling {} with mode {}: outputs={} ops={}", label, mode, modelOutputs, model.getOps().size());
        long start = System.currentTimeMillis();
        GraphExecutionMode effectiveMode = model.compileNativeDynamicShapePlan(modelOutputs, mode, true);
        long elapsed = System.currentTimeMillis() - start;
        log.info("  {} precompile complete: requestedMode={} effectiveMode={} [{}ms]",
                label, mode, effectiveMode, elapsed);
    }

    private void precompileModelForPipeline(SameDiff model, String label, DspCompilationMode mode) {
        List<String> modelOutputs = model.outputs() == null ? Collections.emptyList() : new ArrayList<>(model.outputs());
        log.info("  Precompiling {}: outputs={} ops={}", label, modelOutputs, model.getOps().size());
        log.debug("    {} compile inputs: {}", label, model.inputs());

        model.setDspAutoCompileEnabled(false);
        model.setDspNativeAutoCompileEnabled(false);
        model.setDspFallbackToAutoIfTritonUnavailable(false);
        boolean tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        log.info("    {} compile controls: autoCompile={} nativeAutoCompile={} fallbackToAuto={} tritonAvailable={}",
                label,
                model.isDspAutoCompileEnabled(),
                model.isDspNativeAutoCompileEnabled(),
                model.isDspFallbackToAutoIfTritonUnavailable(),
                tritonAvailable);
        if (mode == DspCompilationMode.MAX_AUTOTUNE && !tritonAvailable) {
            throw new IllegalStateException("MAX_AUTOTUNE requested for " + label + " but Triton is unavailable");
        }
        log.debug("    {} triton config: buildThreads={} cacheEnabled={} alwaysCompile={} maxSubsegmentOps={} maxSubsegmentSections={} numWarps={} numStages={} numCTAs={} maxNreg={} fpFusion={} disableLineInfo={}",
                label,
                Nd4j.getEnvironment().tritonBuildThreads(),
                Nd4j.getEnvironment().tritonCacheEnabled(),
                Nd4j.getEnvironment().tritonAlwaysCompile(),
                Nd4j.getEnvironment().tritonMaxSubsegmentOps(),
                Nd4j.getEnvironment().tritonMaxSubsegmentSections(),
                Nd4j.getEnvironment().tritonNumWarps(),
                Nd4j.getEnvironment().tritonNumStages(),
                Nd4j.getEnvironment().tritonNumCTAs(),
                Nd4j.getEnvironment().tritonMaxNreg(),
                Nd4j.getEnvironment().tritonEnableFpFusion(),
                Nd4j.getEnvironment().tritonDisableLineInfo());

        long start = System.currentTimeMillis();
        var effectiveMode = modelOutputs.isEmpty()
                ? model.compileNativeDynamicShapePlan(mode)
                : model.compileNativeDynamicShapePlan(modelOutputs, mode);
        long elapsed = System.currentTimeMillis() - start;
        log.info("  {} precompile complete: requestedMode={} effectiveMode={} [{}ms]",
                label, mode, effectiveMode, elapsed);
        if (mode == DspCompilationMode.MAX_AUTOTUNE && effectiveMode != GraphExecutionMode.TRITON) {
            throw new IllegalStateException("MAX_AUTOTUNE requested for " + label +
                    " but effective mode resolved to " + effectiveMode + " (fallback disabled)");
        }
    }
}
