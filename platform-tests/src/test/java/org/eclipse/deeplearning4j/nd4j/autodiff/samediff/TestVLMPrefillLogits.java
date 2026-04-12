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
 * Diagnoses VLM prefill logit quality: short vs long sequences.
 * Tests whether the decoder produces reasonable output for different input lengths.
 *
 * Run: cd platform-tests && mvn test -Dtest=TestVLMPrefillLogits \
 *   -Dbackend.artifactId=nd4j-cuda-12.9 -Dlibnd4j.triton=ON 2>&1 | tee /tmp/prefill-logits.log
 */
@Slf4j
public class TestVLMPrefillLogits {

    @Test
    public void testShortVsLongPrefill() throws Exception {
        // Download models
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
        log.info("Hidden size: {}", hiddenSize);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        // === Test 1: SHORT text-only (known working) ===
        log.info("=== TEST 1: Short text-only (17 tokens) ===");
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens",
                BenchmarkConfig.create("SHORT").executionMode(GraphExecutionMode.SLOT_BY_SLOT).maxTokens(5));

        String shortPrompt = "<|im_start|>User:Convert this.<end_of_utterance>\nAssistant:";
        int[] shortIds = tokenizer.encode(shortPrompt, false).getIds();
        INDArray shortIdsTensor = Nd4j.createFromArray(shortIds).reshape(1, shortIds.length).castTo(DataType.LONG);
        Map<String, INDArray> shortEmbedOut = embedTokens.output(Map.of(embedInput, shortIdsTensor), embedOutputNames);
        INDArray shortEmbeds = shortEmbedOut.values().iterator().next().dup();

        StaticKvCacheDecodeLoop shortLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder).embedTokens(embedTokens).tokenizer(tokenizer)
                .ioConfig(ioConfig).samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(5).hiddenSize(hiddenSize).build();
        GenerationResult shortResult = shortLoop.decode(shortEmbeds, shortIds);

        log.info("SHORT result ({} tokens prefill): {} tokens → '{}'",
                shortIds.length, shortResult.getGeneratedTokenCount(), shortResult.getText());
        for (int i = 0; i < shortResult.getTokenIds().length; i++) {
            log.info("  token[{}]: id={} text='{}'", i, shortResult.getTokenIds()[i],
                    tokenizer.decode(new int[]{shortResult.getTokenIds()[i]}));
        }

        // === Test 2: LONG random embeddings (679 tokens) ===
        log.info("=== TEST 2: Long random embeddings (679 tokens) ===");
        decoder.resetSession();
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens",
                BenchmarkConfig.create("LONG").executionMode(GraphExecutionMode.SLOT_BY_SLOT).maxTokens(5));

        INDArray longEmbeds = Nd4j.randn(DataType.FLOAT, 1, 679, hiddenSize).muli(0.02f);
        int[] longIds = new int[679];
        Arrays.fill(longIds, shortIds[0]);

        StaticKvCacheDecodeLoop longLoop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder).embedTokens(embedTokens).tokenizer(tokenizer)
                .ioConfig(ioConfig).samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(5).hiddenSize(hiddenSize).build();
        GenerationResult longResult = longLoop.decode(longEmbeds, longIds);

        log.info("LONG result (679 tokens prefill): {} tokens → '{}'",
                longResult.getGeneratedTokenCount(), longResult.getText());
        for (int i = 0; i < longResult.getTokenIds().length; i++) {
            log.info("  token[{}]: id={} text='{}'", i, longResult.getTokenIds()[i],
                    tokenizer.decode(new int[]{longResult.getTokenIds()[i]}));
        }

        // === Assertions ===
        assertTrue(shortResult.getGeneratedTokenCount() > 0, "Short must produce tokens");
        assertTrue(longResult.getGeneratedTokenCount() > 0, "Long must produce tokens");

        // Check if tokens are degenerate (all same token, or EOS immediately)
        boolean shortDegenerate = isDegenerate(shortResult.getTokenIds());
        boolean longDegenerate = isDegenerate(longResult.getTokenIds());
        log.info("Short degenerate: {}, Long degenerate: {}", shortDegenerate, longDegenerate);

        if (longDegenerate && !shortDegenerate) {
            log.error("DIAGNOSIS: Short works but long is degenerate — long sequence handling is broken");
        } else if (longDegenerate && shortDegenerate) {
            log.error("DIAGNOSIS: Both degenerate — model is fundamentally broken");
        } else {
            log.info("DIAGNOSIS: Both produce varied tokens — model forward pass OK");
        }

        tokenizer.close();
    }

    private boolean isDegenerate(int[] tokens) {
        if (tokens.length <= 1) return false;
        Set<Integer> unique = new HashSet<>();
        for (int t : tokens) unique.add(t);
        // Degenerate if all same token or only 1-2 unique tokens
        return unique.size() <= 2;
    }
}
