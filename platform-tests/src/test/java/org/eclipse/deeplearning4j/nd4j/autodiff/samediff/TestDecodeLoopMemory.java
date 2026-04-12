package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests memory growth through the actual StaticKvCacheDecodeLoop with the real SmolDocling model.
 * This isolates whether the decode loop (KV cache, input building, session management)
 * leaks memory, separate from DSP itself.
 *
 * Run: cd platform-tests && mvn test -Dtest=TestDecodeLoopMemory \
 *   -Dbackend.artifactId=nd4j-cuda-12.9 -Dlibnd4j.triton=ON 2>&1 | tee /tmp/decode-loop-mem.log
 */
@Slf4j
public class TestDecodeLoopMemory {

    @Test
    public void testDecodeLoopMemoryGrowth() throws Exception {
        // Load real SmolDocling model
        VLMModelDownloader.DownloadResult decoderResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        VLMModelDownloader.DownloadResult embedResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        VLMModelDownloader.DownloadResult tokResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokResult.getModelFile());
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                decoderResult.getModelFile().getAbsolutePath(),
                embedResult.getModelFile().getAbsolutePath());
        SameDiff decoder = models[0];
        SameDiff embedTokens = models[1];

        // Get hidden size
        String embedInput = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        int[] testIds = tokenizer.encode("Test", false).getIds();
        INDArray testTensor = Nd4j.createFromArray(testIds).reshape(1, testIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOut = embedTokens.output(Map.of(embedInput, testTensor), embedOutputNames);
        INDArray testEmbed = embedOut.values().iterator().next();
        long hiddenSize = testEmbed.shape()[2];

        // Compile SLOT_BY_SLOT (no Triton, no graphs — isolate the decode loop)
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens",
                BenchmarkConfig.create("MEM_TEST").executionMode(GraphExecutionMode.SLOT_BY_SLOT).maxTokens(50));

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        // Short prompt — use text-only to avoid vision encoder
        String prompt = "<|im_start|>User:Test<end_of_utterance>\nAssistant:";
        int[] promptIds = tokenizer.encode(prompt, false).getIds();
        INDArray idsTensor = Nd4j.createFromArray(promptIds).reshape(1, promptIds.length).castTo(DataType.LONG);
        Map<String, INDArray> promptEmbedOut = embedTokens.output(Map.of(embedInput, idsTensor), embedOutputNames);
        INDArray inputEmbeds = promptEmbedOut.values().iterator().next().dup();

        log.info("Prompt: {} tokens, embeds shape={}, hiddenSize={}",
                promptIds.length, Arrays.toString(inputEmbeds.shape()), hiddenSize);

        Nd4j.getExecutioner().commit();
        long freeBefore = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        log.info("GPU free before decode: {} MB", freeBefore / (1024 * 1024));

        // Run 50-token SLOT_BY_SLOT decode
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder).embedTokens(embedTokens).tokenizer(tokenizer)
                .ioConfig(ioConfig).samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(50).hiddenSize(hiddenSize).build();
        GenerationResult result = loop.decode(inputEmbeds, promptIds);

        Nd4j.getExecutioner().commit();
        long freeAfter = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long growth = freeBefore - freeAfter;
        double perToken = growth / (double) result.getGeneratedTokenCount() / (1024.0 * 1024.0);

        log.info("GPU free after decode: {} MB", freeAfter / (1024 * 1024));
        log.info("Generated {} tokens: '{}'", result.getGeneratedTokenCount(), result.getText());
        log.info("RESULT: total growth={} MB, per-token={} MB",
                growth / (1024 * 1024), String.format("%.2f", perToken));

        assertTrue(perToken < 50.0,
                "Per-token GPU growth too high: " + String.format("%.2f MB (max 50.0)", perToken));

        tokenizer.close();
    }
}
