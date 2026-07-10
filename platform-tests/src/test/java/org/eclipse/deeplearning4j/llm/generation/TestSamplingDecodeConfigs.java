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
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Full-decoder tests for the sampling decode family (greedy + temperature / top-k / top-p) end-to-end
 * through the native CUDA decode path, and for {@link GenerationPipeline#setSamplingConfig} — one model
 * load driven across many configs without a reload.
 *
 * <p>Complements the individual-op {@code TokenSampleParityTest}. Uses Qwen3.5 0.8B (Q4_K_M).</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestSamplingDecodeConfigs -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestSamplingDecodeConfigs {

    private static final String PROMPT =
            "Once upon a time, in a land far away, there lived a curious inventor who";
    private static final int N = 16;

    private static SameDiff model;
    private static Tokenizer tokenizer;
    private static GenerationPipeline pipeline;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        model = GGMLModelImport.importModel(dl.getModelFile().getAbsolutePath());

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            tokenizer = HuggingFaceTokenizer.fromFile(tf.getAbsolutePath());
        }

        // NOTE (2026-07-06): the fixed-buffer fast path (maxPrefillLength=64 + maxKvCacheLength=128)
        // triggers a DSP replay demotion failure on this Qwen 0.8B GDN-hybrid model after ~15 frozen
        // executions ("plan REPLAYING -> SHAPES_FROZEN ... segment no longer satisfies replay steady
        // state ... kind=demotion") and degenerate output — a DSP/pipeline issue flagged for review, NOT
        // caused by the token_sample / config-surface / mask-builder work. Using the variable-buffer path
        // here (correct, just slower — recompiles per generate). See DECODER_DEV_JOURNAL.md.
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        pipeline = GenerationPipeline.create(cfg);
    }

    @AfterAll
    public static void teardown() {
        if (pipeline != null) pipeline.close();
    }

    /** Greedy decode is deterministic (argmax, no RNG). */
    @Test
    @DisplayName("greedy decode is deterministic across runs")
    public void greedyIsDeterministic() {
        pipeline.setSamplingConfig(SamplingConfig.greedy());
        int[] a = pipeline.generate(PROMPT, N).getTokenIds();
        int[] b = pipeline.generate(PROMPT, N).getTokenIds();
        assertTrue(a.length > 0, "greedy produced no tokens");
        assertArrayEquals(a, b, "greedy decode must be deterministic; a=" + Arrays.toString(a) + " b=" + Arrays.toString(b));
    }

    /**
     * setSamplingConfig swaps the strategy on the ALREADY-LOADED model (no reload) and is reversible:
     * greedy → creative → greedy reproduces the original greedy output. Proves one-load-many-configs.
     */
    @Test
    @DisplayName("config swap on one loaded model is reversible (greedy→creative→greedy)")
    public void configSwapOnOneLoadReversible() {
        pipeline.setSamplingConfig(SamplingConfig.greedy());
        int[] g1 = pipeline.generate(PROMPT, N).getTokenIds();

        pipeline.setSamplingConfig(SamplingConfig.creative());
        GenerationResult creative = pipeline.generate(PROMPT, N);
        assertTrue(creative.getTokenIds().length > 0, "sampling (creative) produced no tokens");
        assertNotNull(creative.getText());
        assertFalse(creative.getText().isBlank(), "sampling produced blank text");

        pipeline.setSamplingConfig(SamplingConfig.greedy());
        int[] g2 = pipeline.generate(PROMPT, N).getTokenIds();
        assertArrayEquals(g1, g2,
                "returning to greedy on the same loaded model must reproduce the greedy output");
    }

    /** Every sampling config variant runs end-to-end and yields valid (non-empty) output. */
    @Test
    @DisplayName("temperature / top-k / top-p / llama.cpp configs all decode to valid output")
    public void samplingConfigsRunEndToEnd() {
        List<SamplingConfig> configs = Arrays.asList(
                SamplingConfig.builder().temperature(0.8).doSample(true).build(),                 // temperature only
                SamplingConfig.builder().temperature(1.0).topK(40).doSample(true).build(),        // top-k
                SamplingConfig.builder().temperature(1.0).topP(0.9).doSample(true).build(),        // top-p (nucleus)
                SamplingConfig.llamaCppDefaults());                                                // top-k + top-p + rep-penalty
        for (SamplingConfig cfg : configs) {
            pipeline.setSamplingConfig(cfg);
            GenerationResult r = pipeline.generate(PROMPT, N);
            assertTrue(r.getTokenIds().length > 0, "config " + cfg + " produced no tokens");
            assertNotNull(r.getText(), "config " + cfg + " produced null text");
            log.info("[{}] -> {} tokens: {}", cfg, r.getTokenIds().length,
                    r.getText().substring(0, Math.min(60, r.getText().length())).replace('\n', ' '));
        }
    }

    /**
     * REPRO (2026-07-06 handoff): the fixed-buffer fast path (maxPrefillLength=64 + maxKvCacheLength=128)
     * demotes REPLAYING -> SHAPES_FROZEN (segment reset to WARMUP mid-replay) around exec 15 on this
     * recurrent GDN-hybrid model. Temporary — diagnostic only, remove once root-caused.
     */
    @Test
    @DisplayName("REPRO: fixed-buffer fast path replay demotion")
    public void reproFixedBufferDemotion() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(24)
                .maxPrefillLength(64)
                .maxKvCacheLength(128)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        GenerationPipeline fixedPipe = GenerationPipeline.create(cfg);
        try {
            // Mirror the handoff scenario: many generates on ONE shared pipeline (the demotion
            // accumulates across generates, not within a single decode), with a config swap.
            for (int g = 0; g < 10; g++) {
                if (g == 3) fixedPipe.setSamplingConfig(SamplingConfig.creative());
                if (g == 5) fixedPipe.setSamplingConfig(SamplingConfig.greedy());
                GenerationResult r = fixedPipe.generate(PROMPT, 16);
                log.info("REPRO gen {} ({} tok): {}", g, r.getTokenIds().length,
                        r.getText().substring(0, Math.min(50, r.getText().length())).replace('\n', ' '));
            }
        } finally {
            fixedPipe.close();
        }
    }
}
