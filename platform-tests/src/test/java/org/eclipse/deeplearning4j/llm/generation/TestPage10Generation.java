package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
public class TestPage10Generation {

    @Test
    public void testPage10WithGenerationPipeline() throws Exception {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");

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

        int targetSize = 512;
        File pdfFile = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        if (!pdfFile.exists()) {
            pdfFile = new File("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/pathfinder-mythic.pdf");
        }
        assertTrue(pdfFile.exists(), "PDF must exist at " + pdfFile.getAbsolutePath());

        try (org.apache.pdfbox.pdmodel.PDDocument doc = org.apache.pdfbox.pdmodel.PDDocument.load(pdfFile)) {
            org.apache.pdfbox.rendering.PDFRenderer renderer = new org.apache.pdfbox.rendering.PDFRenderer(doc);
            BufferedImage image = renderer.renderImageWithDPI(10, 150, org.apache.pdfbox.rendering.ImageType.RGB);
            log.info("Rendered page 10: {}x{}", image.getWidth(), image.getHeight());

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

            long hiddenSize = visionEmbeddings.size(-1);
            int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
            int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
            String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                    splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
            String chatPrompt = "<|im_start|>User:" + imagePrompt
                    + "Convert this page to docling.<end_of_utterance>\nAssistant:";
            int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

            INDArray tokenIds = Nd4j.createFromArray(promptTokenIds).reshape(1, promptTokenIds.length).castTo(DataType.INT64);
            Map<String, INDArray> embedInputs = new HashMap<>();
            for (String inputName : embedTokens.inputs()) {
                embedInputs.put(inputName, tokenIds);
            }
            Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                    embedTokens.outputs().toArray(new String[0]));
            INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
            tokenIds.close();

            INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);

            // Test with GenerationPipeline
            ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
            GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(ioConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(250)
                    .hiddenSize(hiddenSize)
                    .build());

            GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, 250);
            log.info("Page 10 GenerationPipeline: {} tokens, text='{}'", result.getTokenIds().length, result.getText());

            boolean hasMythic = result.getText().toLowerCase().contains("mythic")
                    || result.getText().toLowerCase().contains("creating a mythic character");
            log.info("Has mythic content: {}", hasMythic);

            // Also test with old decoder for comparison
            decoder.clearPlaceholders(true);
            decoder.clearOpInputs();
            decoder.resetSession();
            Nd4j.getExecutioner().commit();

            StaticKvCacheDecodeLoop oldLoop = StaticKvCacheDecodeLoop.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(ioConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(250)
                    .hiddenSize(hiddenSize)
                    .build();
            GenerationResult oldResult = oldLoop.decode(inputsEmbeds.dup(), promptTokenIds);
            log.info("Page 10 OldDecoder: {} tokens, text='{}'", oldResult.getTokenIds().length, oldResult.getText());

            boolean oldHasMythic = oldResult.getText().toLowerCase().contains("mythic")
                    || oldResult.getText().toLowerCase().contains("creating a mythic character");
            log.info("Old decoder has mythic content: {}", oldHasMythic);

            pipeline.close();
        }
    }
}
