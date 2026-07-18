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

package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.*;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration test for constrained (JSON / tool-call) decoding.
 *
 * <p>Requires real assets on disk. Skipped automatically when assets are absent
 * so CI stays green on machines without the model files.</p>
 *
 * <p>Asset locations (checked at test startup):
 * <ul>
 *   <li>{@code ~/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf}</li>
 *   <li>{@code ~/.kompile/models/tokenizers/qwen2.5-0.5b/tokenizer.json}</li>
 * </ul>
 * </p>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ConstrainedDecodingIntegrationTest {

    private static final String MODEL_PATH =
            System.getProperty("user.home") + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";
    private static final String TOKENIZER_PATH =
            System.getProperty("user.home") + "/.kompile/models/tokenizers/qwen2.5-0.5b/tokenizer.json";

    /** Tools in the tool-call constraint enum — the exact set used by the graph-reasoning MCP layer. */
    private static final String[] TOOL_NAMES = {
            "ask_graph_verify",
            "graph_reasoning_query",
            "ask_graph_query"
    };

    private static GenerationPipeline pipeline;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @BeforeAll
    static void loadPipeline() throws Exception {
        assumeTrue(new File(MODEL_PATH).exists(),
                "Model not found; skipping integration test: " + MODEL_PATH);
        assumeTrue(new File(TOKENIZER_PATH).exists(),
                "Tokenizer not found; skipping integration test: " + TOKENIZER_PATH);

        Tokenizer tokenizer;
        try {
            tokenizer = HuggingFaceTokenizer.fromFile(TOKENIZER_PATH);
        } catch (UnsatisfiedLinkError | Exception e) {
            assumeTrue(false,
                    "Native tokenizer library unavailable; skipping integration test: " + e.getMessage());
            return; // unreachable — assumeTrue throws; keeps compiler happy
        }

        // Use GGMLModelImport as the custom ModelLoader (keeps samediff-llm decoupled from nd4j-ggml at compile time).
        pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoderPath(MODEL_PATH)
                        .modelLoader(path -> {
                            try {
                                return GGMLModelImport.importModel(path);
                            } catch (Exception e) {
                                throw new java.io.IOException("GGUF import failed: " + e.getMessage(), e);
                            }
                        })
                        .tokenizer(tokenizer)
                        .maxNewTokens(128)
                        .build());
    }

    @AfterAll
    static void closePipeline() {
        if (pipeline != null) {
            pipeline.close();
        }
    }

    // -----------------------------------------------------------------------
    // 1. Tool-call constraint: 10/10 runs must parse as JSON with tool in enum
    // -----------------------------------------------------------------------

    @Test
    @Order(1)
    void toolCallConstraint_10Of10RunsParseable_toolInEnum() {
        SamplingConfig constrainedSampling = SamplingConfig.builder()
                .doSample(true)
                .temperature(0.7f)
                .topK(40)
                .maxNewTokens(128)
                .constraintConfig(ConstraintConfig.toolCall(TOOL_NAMES))
                .build();

        Set<String> allowed = new HashSet<>(Arrays.asList(TOOL_NAMES));
        List<String> failures = new ArrayList<>();

        for (int run = 0; run < 10; run++) {
            SamplingConfig seeded = constrainedSampling.toBuilder()
                    .seed((long) (run * 17 + 42))
                    .build();

            GenerationResult result = pipeline.generate(
                    "You have access to graph reasoning tools. I need to look up whether Alice is connected to Bob.",
                    128,
                    seeded);

            String text = result.getText().trim();

            // Must parse as valid JSON
            JsonNode node;
            try {
                node = MAPPER.readTree(text);
            } catch (Exception e) {
                failures.add("run " + run + ": not valid JSON: [" + text + "] err=" + e.getMessage());
                continue;
            }

            // Must be a JSON object
            if (!node.isObject()) {
                failures.add("run " + run + ": not a JSON object: [" + text + "]");
                continue;
            }

            // Must have a "tool" field
            if (!node.has("tool")) {
                failures.add("run " + run + ": missing 'tool' field: [" + text + "]");
                continue;
            }

            // Tool must be in the allowed enum
            String tool = node.get("tool").asText();
            if (!allowed.contains(tool)) {
                failures.add("run " + run + ": tool '" + tool + "' not in allowed set " + allowed);
            }
        }

        assertTrue(failures.isEmpty(),
                "Tool-call constraint failed on " + failures.size() + " runs:\n  " + String.join("\n  ", failures));
    }

    // -----------------------------------------------------------------------
    // 2. JSON-object constraint: output parses as a JSON object
    // -----------------------------------------------------------------------

    @Test
    @Order(2)
    void jsonObjectConstraint_outputParsesAsJsonObject() {
        SamplingConfig constrainedSampling = SamplingConfig.builder()
                .doSample(true)
                .temperature(0.7f)
                .topK(40)
                .seed(99L)
                .maxNewTokens(128)
                .constraintConfig(ConstraintConfig.jsonObject())
                .build();

        GenerationResult result = pipeline.generate(
                "Summarize the key entity attributes for 'Alice' as a JSON object.",
                128,
                constrainedSampling);

        String text = result.getText().trim();

        JsonNode node;
        try {
            node = MAPPER.readTree(text);
        } catch (Exception e) {
            fail("JSON-mode output is not valid JSON: [" + text + "] err=" + e.getMessage());
            return;
        }

        assertTrue(node.isObject(), "JSON-mode output should be a JSON object, got: [" + text + "]");
    }

    // -----------------------------------------------------------------------
    // 3. Perf: constrained tok/s vs. unconstrained tok/s — informational
    //    (no assertion; just printed so it shows up in the test log)
    // -----------------------------------------------------------------------

    @Test
    @Order(3)
    void perf_constrainedVsUnconstrained_reportTokPerSec() {
        final int WARMUP_TOKENS = 64;
        final int MEASURE_TOKENS = 128;
        final String PROMPT =
                "I need to query the knowledge graph to find Alice. Which tool should I use?";

        // Unconstrained, greedy
        SamplingConfig unconstrained = SamplingConfig.builder()
                .doSample(false)
                .seed(7L)
                .maxNewTokens(WARMUP_TOKENS)
                .build();
        // Warmup
        pipeline.generate(PROMPT, WARMUP_TOKENS, unconstrained);

        // Measure unconstrained
        long t0 = System.currentTimeMillis();
        GenerationResult unconstrainedResult = pipeline.generate(PROMPT, MEASURE_TOKENS, unconstrained);
        long unconstrainedMs = System.currentTimeMillis() - t0;
        int unconstrainedTokens = unconstrainedResult.getTokenIds() != null
                ? unconstrainedResult.getTokenIds().length
                : MEASURE_TOKENS;

        // Constrained, same sampling otherwise
        SamplingConfig constrained = SamplingConfig.builder()
                .doSample(false)
                .seed(7L)
                .maxNewTokens(MEASURE_TOKENS)
                .constraintConfig(ConstraintConfig.toolCall(TOOL_NAMES))
                .build();
        // Warmup (constrained path)
        pipeline.generate(PROMPT, WARMUP_TOKENS, constrained);

        // Measure constrained
        long t1 = System.currentTimeMillis();
        GenerationResult constrainedResult = pipeline.generate(PROMPT, MEASURE_TOKENS, constrained);
        long constrainedMs = System.currentTimeMillis() - t1;
        int constrainedTokens = constrainedResult.getTokenIds() != null
                ? constrainedResult.getTokenIds().length
                : MEASURE_TOKENS;

        double unconstrainedTps = unconstrainedTokens * 1000.0 / Math.max(unconstrainedMs, 1);
        double constrainedTps = constrainedTokens * 1000.0 / Math.max(constrainedMs, 1);
        double overhead = (unconstrainedTps > 0)
                ? (unconstrainedTps - constrainedTps) / unconstrainedTps * 100.0
                : 0.0;

        System.out.printf("[PERF] Unconstrained: %.1f tok/s (%d tokens, %dms)%n",
                unconstrainedTps, unconstrainedTokens, unconstrainedMs);
        System.out.printf("[PERF] Constrained:   %.1f tok/s (%d tokens, %dms)%n",
                constrainedTps, constrainedTokens, constrainedMs);
        System.out.printf("[PERF] Overhead: %.1f%%%n", overhead);

        // Sanity: constrained must not be more than 50% slower on a CPU box.
        // (vocab-sweep O(n*k) with k=256 should add <5ms/step; headline limit is generous.)
        assertTrue(overhead < 50.0,
                String.format("Constrained decoding overhead %.1f%% exceeds 50%% guard — check ConstraintVocabCache", overhead));
    }
}
