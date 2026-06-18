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
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression isolation test for mythic-PDF DocTag format failure.
 *
 * Compares GenerationPipeline (native decode loop) against StaticKvCacheDecodeLoop
 * (old Java decode loop) on the SAME mythic PDF input.  Token divergence localises
 * the bug to either:
 *   - shared input construction (DecoderInputBuilder) → both diverge from expectation
 *   - GenerationPipeline / native loop → only native diverges
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestMythicPdfRegression \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestMythicPdfRegression {

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

        log.info("Loading SmolDocling models for mythic PDF regression test...");

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
        int pdfPage = Integer.getInteger("vlm.test.pdf.page", 0);
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

    // ─────────────────────────────────────────────────────────────────────────
    // Isolation test 1: old decoder vs new pipeline, token-by-token
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("Old decoder vs GenerationPipeline: token stream divergence on mythic PDF")
    public void testOldDecoderVsGenerationPipeline() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 15;

        // Reference: old StaticKvCacheDecodeLoop (Java-only, known-good for format)
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
        log.info("OLD  decoder: {} tokens, text='{}'", oldTokens.length, oldText);

        // Reset decoder state — old loop may have frozen shapes / compiled DSP
        decoder.clearPlaceholders(false);
        decoder.clearOpInputs();
        decoder.resetSession();
        Nd4j.getExecutioner().commit();

        // Test: GenerationPipeline (native decode loop)
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build());

        GenerationResult newResult = pipeline.generate(inputsEmbeds.dup(), promptTokenIds);
        int[] newTokens = newResult.getTokenIds();
        String newText = newResult.getText();
        log.info("NEW pipeline: {} tokens, text='{}'", newTokens.length, newText);

        // Token-by-token comparison
        int minLen = Math.min(oldTokens.length, newTokens.length);
        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (oldTokens[i] == newTokens[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }
        double matchRate = minLen > 0 ? (double) matches / minLen : 0.0;
        log.info("Token match rate: {}/{} ({}%) firstDivergent={}",
                matches, minLen, String.format("%.1f", matchRate * 100), firstDivergent);

        if (firstDivergent >= 0) {
            log.error("FIRST DIVERGENCE at step {}: old={} ('{}') new={} ('{}')",
                    firstDivergent,
                    oldTokens[firstDivergent],
                    tokenizer.decode(new int[]{oldTokens[firstDivergent]}, false),
                    newTokens[firstDivergent],
                    tokenizer.decode(new int[]{newTokens[firstDivergent]}, false));
        }

        // Structural tag sanity check on OLD decoder (should pass)
        boolean oldHasTags = oldText.contains("<") && oldText.contains(">");
        log.info("Old decoder has structural tags: {}", oldHasTags);

        // Structural tag sanity check on NEW pipeline (fails in benchmark)
        boolean newHasTags = newText.contains("<") && newText.contains(">");
        log.info("New pipeline has structural tags: {}", newHasTags);

        // The critical assertion: old decoder MUST produce structural tags
        assertTrue(oldHasTags,
                "Old decoder failed to produce structural tags — test setup is broken or PDF is missing. "
                        + "Text: " + oldText);
        assertTrue(newHasTags,
                "GenerationPipeline stripped or failed to return structural tags even though the "
                        + "native path should preserve the same token text as the old decoder. "
                        + "Text: " + newText);

        // If old decoder is correct but new pipeline diverges, we have isolated the bug.
        if (firstDivergent >= 0) {
            log.error("BUG ISOLATED: GenerationPipeline diverges from old decoder at step {}. "
                    + "This points to the native decode loop or GenerationPipeline handoff.", firstDivergent);
        }

        // Soft assertion — we expect them to match.  If they don't, the test logs the divergence
        // but does NOT hard-fail so we can inspect output.
        assertTrue(matchRate >= 0.8,
                "GenerationPipeline diverges significantly from old decoder: "
                        + String.format("%.1f%%", matchRate * 100)
                        + " match, first divergence at step " + firstDivergent
                        + ". Old text: " + oldText
                        + ". New text: " + newText);
        assertEquals(oldText, newText,
                "GenerationPipeline returned different decoded text despite matching the old "
                        + "decoder token stream. Old text: " + oldText + " New text: " + newText);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Isolation test 2: GenerationPipeline alone, structural tag check
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("GenerationPipeline structural tag check on mythic PDF")
    public void testGenerationPipelineStructuralTags() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 20;

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build());

        GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds);
        String text = result.getText();
        int[] tokens = result.getTokenIds();

        log.info("GenerationPipeline: {} tokens, text='{}'", tokens.length, text);
        log.info("Token IDs: {}", Arrays.toString(tokens));

        // Decode first few tokens individually to see what they are
        for (int i = 0; i < Math.min(10, tokens.length); i++) {
            String tokText = tokenizer.decode(new int[]{tokens[i]}, true);
            log.info("  Token {}: id={} text='{}'", i, tokens[i], tokText);
        }

        boolean hasStructuralTags = text.contains("<doctag>") || text.contains("<page>")
                || text.contains("<section_header>") || text.contains("<otsl>");

        assertTrue(hasStructuralTags,
                "GenerationPipeline should produce structural DocTags on mythic PDF. "
                        + "Text: " + text);

        // Degenerate output detection: the model should NOT produce only picture tags
        // and then immediately stop. A well-functioning model on this PDF produces
        // content tags like <text>, <section_header>, <otsl> — not just <picture></picture>.
        boolean isDegenerateOutput = text.contains("<picture>") && text.contains("</doctag>")
                && !text.contains("<text>") && !text.contains("<section_header>")
                && !text.contains("<otsl>") && tokens.length < maxTokens;

        assertFalse(isDegenerateOutput,
                "Degenerate output detected: model produced only picture tags and stopped "
                        + "prematurely (" + tokens.length + " tokens). Expected content tags "
                        + "(<text>, <section_header>, <otsl>) in output. Text: " + text);
    }

    /**
     * Degenerate output regression test.
     *
     * The model should produce at least 30 tokens on the mythic PDF page and
     * include content-bearing tags. A degenerate model produces only:
     *   {@code <doctag><picture><loc_0><loc_0><loc_500><loc_500><other></picture></doctag><end_of_utterance>}
     * — 25 tokens with no text/table content, hitting EOS prematurely.
     *
     * Root causes that have triggered this:
     *   1. Stale vision encoder SDZ cache with incorrect FP16 precast
     *   2. Softmax kernel launch bounds crash in vision encoder (error 1: invalid argument)
     *   3. Flash attention launch bounds mismatch (1024 threads vs __launch_bounds__(512))
     */
    @Test
    @DisplayName("Degenerate output detection: model must produce content beyond picture tags")
    public void testDegenerateOutputDetection() throws Exception {
        ensureModelsLoaded();
        int maxTokens = 50;

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build());

        GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds);
        String text = result.getText();
        int[] tokens = result.getTokenIds();

        log.info("Degenerate check: {} tokens, text='{}'", tokens.length, text);
        for (int i = 0; i < Math.min(15, tokens.length); i++) {
            String tokText = tokenizer.decode(new int[]{tokens[i]}, true);
            log.info("  Token {}: id={} text='{}'", i, tokens[i], tokText);
        }

        // Must produce a minimum number of tokens — premature EOS is a sign of broken
        // vision encoder or attention computation
        assertTrue(tokens.length >= 30,
                "Model produced only " + tokens.length + " tokens (expected >= 30). "
                        + "Premature EOS suggests vision encoder or attention bug. Text: " + text);

        // Must contain content tags beyond just picture structure
        boolean hasContentTags = text.contains("<text>") || text.contains("<section_header>")
                || text.contains("<otsl>") || text.contains("<table>");

        assertTrue(hasContentTags,
                "Model output contains no content tags (<text>, <section_header>, <otsl>, <table>). "
                        + "Only structural/picture tags found — this is degenerate output. "
                        + "Tokens: " + tokens.length + " Text: " + text);

        // Count unique token IDs — degenerate output often has very low diversity
        Set<Integer> uniqueTokens = new HashSet<>();
        for (int t : tokens) uniqueTokens.add(t);
        double diversity = (double) uniqueTokens.size() / tokens.length;
        log.info("Token diversity: {}/{} unique ({}%)",
                uniqueTokens.size(), tokens.length, String.format("%.1f", diversity * 100));

        assertTrue(diversity > 0.3,
                "Token diversity too low: " + uniqueTokens.size() + "/" + tokens.length
                        + " unique tokens (" + String.format("%.1f%%", diversity * 100)
                        + "). Model may be stuck in a repetition loop.");
    }

    @Test
    @DisplayName("GenerationPipeline.embedTokens matches embed_tokens model output")
    public void testEmbedTokensParity() throws Exception {
        ensureModelsLoaded();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(1)
                .hiddenSize(hiddenSize)
                .build());

        INDArray direct = embedPromptDirect(promptTokenIds);
        INDArray extracted = pipeline.embedTokens(promptTokenIds);

        assertArrayEquals(direct.shape(), extracted.shape(), "Prompt embedding shapes differ");

        INDArray diff = Transforms.abs(direct.sub(extracted), false);
        double maxDiff = diff.maxNumber().doubleValue();
        log.info("embedTokens parity: shape={} maxAbsDiff={}", Arrays.toString(direct.shape()), maxDiff);

        assertTrue(maxDiff < 1e-6,
                "GenerationPipeline.embedTokens diverges from embed_tokens model output. maxAbsDiff=" + maxDiff);
    }

    @Test
    @DisplayName("Benchmark Triton compile path preserves old decoder output")
    public void testOldDecoderAfterBenchmarkCompile() throws Exception {
        ensureModelsLoaded();
        int maxTokens = Integer.getInteger("vlm.test.maxNewTokens", 15);
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        StaticKvCacheDecodeLoop baselineLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();
        GenerationResult baseline = baselineLoop.decode(inputsEmbeds.dup(), promptTokenIds);

        BenchmarkConfig benchmarkConfig = BenchmarkConfig.create("TRITON_NO_GC")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(maxTokens)
                .minDiversityPct(0);

        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(benchmarkConfig);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", benchmarkConfig);

        StaticKvCacheDecodeLoop configuredLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();
        GenerationResult configured = configuredLoop.decode(inputsEmbeds.dup(), promptTokenIds);

        log.info("Benchmark compile parity: baseline='{}' configured='{}'",
                baseline.getText(), configured.getText());

        assertArrayEquals(baseline.getTokenIds(), configured.getTokenIds(),
                "Benchmark compile path changed old decoder token stream");
        assertEquals(baseline.getText(), configured.getText(),
                "Benchmark compile path changed old decoder decoded text");
    }

}
