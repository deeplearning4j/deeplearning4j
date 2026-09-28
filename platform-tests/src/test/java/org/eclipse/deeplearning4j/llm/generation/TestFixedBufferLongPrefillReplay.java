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
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Serving-shaped regression for long fixed-buffer prefills on the staged Qwen3.5 SDZ.
 *
 * <p>Crawl r12 (2026-09-28) served qwen35-2b with {@code maxPrefillLength=3072} and
 * {@code maxKvCacheLength=4096}. Prefill 1 compiled the plan and prefill 2 ran it slot-by-slot.
 * Prefill 3 captured the composite graph, and its first replay failed with CUDA 700 (illegal
 * address). The prefill Triton sections, including fused attention over the {@code [1,1,3072,4096]}
 * causal mask, run for the first time in that replay. Each request here uses constrained in-graph
 * KV decoding, as the crawl did, and requests continue past the first replay so steady-state reuse
 * is covered as well.</p>
 *
 * <p>Opt-in: {@code -Dqwen.sdz.path=<model-rq.sdz>}. The tokenizer defaults to
 * {@code tokenizer.json} next to the SDZ. {@code -Dqwen.device.limits=<bytes,bytes,...>} applies the
 * serving per-device memory ceilings, so the multi-GPU placement matches the crawl.</p>
 */
@Slf4j
public class TestFixedBufferLongPrefillReplay {

    private static final int MAX_PREFILL = 3072;
    private static final int MAX_KV = 4096;
    private static final int NEW_TOKENS = 24;
    /** Prompt lengths from the crawl (1483, 2778) plus two more: exec 0/1 warm up, exec 2 captures. */
    private static final int[] PROMPT_TOKEN_TARGETS = {1483, 2100, 2778, 1900};

    private static final String PARAGRAPH =
            "Regional update %d: AMER bookings closed the quarter at plan, with enterprise renewals "
            + "offsetting a softer mid-market pipeline. Finance asked the account teams to reconcile "
            + "deferred revenue against the billing schedule before the forecast review, and flagged "
            + "three contracts whose payment terms changed after signature. ";

    @Test
    @DisplayName("long fixed-buffer prefills survive the first composite-capture replay")
    public void longPrefillsSurviveCompositeCaptureReplay() throws Exception {
        String sdzPath = System.getProperty("qwen.sdz.path");
        assumeTrue(sdzPath != null && new File(sdzPath).isFile(),
                "Set -Dqwen.sdz.path to the staged Qwen SDZ used by serving");
        File modelDir = new File(sdzPath).getAbsoluteFile().getParentFile();
        File tokenizerFile = new File(System.getProperty("qwen.tokenizer.path",
                new File(modelDir, "tokenizer.json").getPath()));
        assumeTrue(tokenizerFile.isFile(), "No tokenizer at " + tokenizerFile);

        applyServingRuntimeConfig();
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoderPath(sdzPath)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(NEW_TOKENS)
                .maxPrefillLength(MAX_PREFILL)
                .maxKvCacheLength(MAX_KV)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .prefixCacheEnabled(false)
                .prefillLastPositionLogitsEnabled(true)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        SamplingConfig constrained = SamplingConfig.greedy().toBuilder()
                .constraintConfig(ConstraintConfig.jsonObject())
                .build();

        GenerationPipeline pipe = GenerationPipeline.create(config);
        try {
            for (int request = 0; request < PROMPT_TOKEN_TARGETS.length; request++) {
                String prompt = promptWithTokens(tokenizer, PROMPT_TOKEN_TARGETS[request], request);
                int promptTokens = tokenizer.encode(prompt, false).getLength();
                long start = System.nanoTime();
                GenerationResult result = pipe.generate(prompt, NEW_TOKENS, constrained);
                log.info("[long-prefill-replay] request={} promptTokens={} generated={} elapsedMs={}",
                        request, promptTokens, result.getGeneratedTokenCount(),
                        (System.nanoTime() - start) / 1_000_000L);
                assertTrue(result.getGeneratedTokenCount() > 0,
                        "request " + request + " (" + promptTokens + " prompt tokens) emitted no tokens");
            }
        } finally {
            pipe.close();
        }
    }

    /** Mirrors ServingSubprocessMain: optimizer on, FP16 off, Triton LLM config, serving memory ceilings. */
    private static void applyServingRuntimeConfig() {
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_FP16) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_FP16, "false");
        }
        Nd4j.scalar(0.0f);
        Environment environment = Nd4j.getEnvironment();
        if (NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable()) {
            environment.applyOptimalLLMConfig();
            if (System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE) == null) {
                System.setProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE, "TRITON");
            }
        }
        String limits = System.getProperty("qwen.device.limits");
        if (limits != null && !limits.isBlank()) {
            String[] parts = limits.split(",");
            int devices = Nd4j.getAffinityManager().getNumberOfDevices();
            assumeTrue(parts.length == devices,
                    "qwen.device.limits has " + parts.length + " entries for " + devices + " devices");
            for (int device = 0; device < parts.length; device++) {
                long limit = Long.parseLong(parts[device].trim());
                environment.setDeviceLimit(device, limit);
                log.info("[long-prefill-replay] device {} memory ceiling {} bytes", device, limit);
            }
        }
    }

    /** Builds a prompt of at least {@code targetTokens} tokens; the variant index keeps requests distinct. */
    private static String promptWithTokens(Tokenizer tokenizer, int targetTokens, int variant) {
        StringBuilder prompt = new StringBuilder(
                "Summarize the regional finance notes below as one JSON object.\n\n");
        int paragraph = variant * 1000;
        while (tokenizer.encode(prompt.toString(), false).getLength() < targetTokens) {
            prompt.append(String.format(PARAGRAPH, paragraph++));
        }
        return prompt.toString();
    }
}
