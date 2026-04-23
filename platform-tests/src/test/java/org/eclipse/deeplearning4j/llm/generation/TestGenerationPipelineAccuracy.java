package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
public class TestGenerationPipelineAccuracy {

    private static final int TARGET_SIZE = 512;
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoderSd;
    private static Tokenizer tokenizer;
    private static File pdfFile;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("GenerationPipeline reads page 10 OCR content at configured DPI")
    public void testPage10OcrAccuracy() throws Exception {
        ensureLoaded();

        int dpi = Integer.getInteger("vlm.test.pdf.dpi", 150);
        int maxTokens = Integer.getInteger("vlm.test.maxTokens", 100);
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens).minDiversityPct(0);

        BufferedImage pdfImage = loadPageImage(pdfFile, 10, dpi);
        log.info("Running OCR test at {} DPI: image={}x{}", dpi, pdfImage.getWidth(), pdfImage.getHeight());

        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, TARGET_SIZE, 9);

        PreprocessorConfig ppConfig = buildPreprocessorConfig();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(9)
                .build();
        INDArray visionEmbeddings;
        try {
            VisionEncoder.Result visionResult = visionEncoder.encode(
                    imageInput, splitResult.getTotalFrames(), splitResult);
            visionEmbeddings = visionResult.getEmbeddings();
        } finally {
            imageInput.close();
            visionEncoder.close();
        }

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = buildChatPrompt(imagePrompt);
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray textEmbeddings = null;
        GenerationPipeline embedPipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(1)
                        .build());
        try {
            textEmbeddings = embedPipeline.embedTokens(promptTokenIds);
        } finally {
            embedPipeline.close();
        }

        INDArray merged = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        textEmbeddings.close();

        compileFor(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .ioConfig(ioConfig)
                        .samplingConfig(SamplingConfig.greedy())
                        .maxNewTokens(maxTokens)
                        .hiddenSize(576L)
                        .build());
        try {
            GenerationResult result = pipeline.generate(merged, promptTokenIds, maxTokens);
            log.info("DPI_{} text='{}'", dpi, safeSnippet(result.getText(), 300));

            assertOcrOutput(result, maxTokens);
        } finally {
            pipeline.close();
            merged.close();
        }
    }

    private static synchronized void ensureLoaded() throws Exception {
        if (loaded) {
            return;
        }

        pdfFile = discoverPdfFile();

        VLMModelDownloader.DownloadResult visionDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        VLMModelDownloader.DownloadResult decoderDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        VLMModelDownloader.DownloadResult embedTokensDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        VLMModelDownloader.DownloadResult tokenizerDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDl.getModelFile());
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionDl.getModelFile().getAbsolutePath(),
                decoderDl.getModelFile().getAbsolutePath(),
                embedTokensDl.getModelFile().getAbsolutePath()
        );
        visionEncoderSd = models[0];
        decoder = models[1];
        embedTokens = models[2];

        loaded = true;
        log.info("Models loaded. pdf={}", pdfFile.getAbsolutePath());
    }

    private static File discoverPdfFile() {
        String configuredPath = System.getProperty("vlm.test.pdf.path");
        if (configuredPath != null && !configuredPath.isBlank()) {
            return new File(configuredPath);
        }
        File f = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        if (f.exists()) {
            return f;
        }
        return new File("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
    }

    private static BufferedImage loadPageImage(File pdfFile, int page, int dpi) throws IOException {
        assertTrue(pdfFile.exists(), "PDF must exist: " + pdfFile.getAbsolutePath());
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(page, dpi, ImageType.RGB);
        }
    }

    private static PreprocessorConfig buildPreprocessorConfig() {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});
        return config;
    }

    private static String buildChatPrompt(String imagePrompt) {
        return "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
    }

    private static void compileFor(BenchmarkConfig config) {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
    }

    private static void assertOcrOutput(GenerationResult result, int maxTokens) {
        String text = result.getText();
        assertNotNull(text, "Generated text is null");

        String normalized = text.trim().toLowerCase();
        int minUsefulTokens = Math.min(maxTokens, 50);

        assertTrue(result.getGeneratedTokenCount() >= minUsefulTokens,
                "Generated too few tokens: " + result.getGeneratedTokenCount());
        assertTrue(normalized.contains("<"),
                "Expected structural tags in output. Text: " + safeSnippet(text, 220));
        assertTrue(normalized.contains("<text>") || normalized.contains("<page>"),
                "Expected text/page structural tags. Text: " + safeSnippet(text, 220));
        assertFalse(normalized.startsWith("<doctag><picture>"),
                "Collapsed to picture-only output. Text: " + safeSnippet(text, 220));
    }

    private static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "<null>";
        }
        String normalized = text.replace("\n", " ").replace("\r", " ").trim();
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, maxChars) + "...";
    }
}
