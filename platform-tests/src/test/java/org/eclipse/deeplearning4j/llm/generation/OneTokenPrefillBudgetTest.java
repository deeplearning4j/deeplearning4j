package org.eclipse.deeplearning4j.llm.generation;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

/** Exact cached-model regression; no model download, import or export. */
@EnabledIfSystemProperty(named = "sdx.oneToken.model", matches = ".+")
class OneTokenPrefillBudgetTest {
    @Test
    void oneShotSingleTokenHonorsBudgetAcrossRepeatedRequests() throws Exception {
        Path model = Path.of(System.getProperty("sdx.oneToken.model"));
        assertTrue(Files.isRegularFile(model));
        try (SameDiff graph = SDZSerializer.load(model.toFile(), false);
             HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(
                     Path.of(System.getProperty("sdx.oneToken.tokenizer")).toFile());
             GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                     .decoder(graph).tokenizer(tokenizer).samplingConfig(SamplingConfig.greedy())
                     .maxNewTokens(1).maxPrefillLength(36).maxKvCacheLength(37)
                     .kvCacheStrategy(KvCacheStrategy.QUANTIZED).kvQuantFormat(1)
                     .graphOptimizerEnabled(false).dspEnabled(true)
                     .benchmarkConfig(BenchmarkConfig.cpuCascade()).build())) {
            int[] first = pipeline.generate("Hello. Give one concise helpful response.", 1,
                    SamplingConfig.greedy()).getTokenIds();
            assertEquals(1, first.length, "One-shot generation must not return a warmup token beyond its budget");
            int[] second = pipeline.generate("Hello. Give one concise helpful response.", 1,
                    SamplingConfig.greedy()).getTokenIds();
            assertArrayEquals(first, second, "Repeated independent requests must retain the same first token");
            int[] two = pipeline.generate("Hello. Give one concise helpful response.", 2,
                    SamplingConfig.greedy()).getTokenIds();
            assertEquals(2, two.length, "A later request must still be able to initialize decode");
            assertEquals(first[0], two[0]);
            int[] afterDecode = pipeline.generate("Hello. Give one concise helpful response.", 1,
                    SamplingConfig.greedy()).getTokenIds();
            assertArrayEquals(first, afterDecode, "One-token request after retained decode must reset safely");
        }
    }
}
