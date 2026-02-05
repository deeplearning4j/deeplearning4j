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

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Consumer;

/**
 * High-level text generation utility for autoregressive language models.
 *
 * <p>TextGenerator provides a clean API for generating text from SameDiff models.
 * It handles the autoregressive loop, token sampling, and stopping conditions.</p>
 *
 * <p>Features:</p>
 * <ul>
 *   <li>Configurable sampling strategies (greedy, temperature, top-k, top-p)</li>
 *   <li>Streaming output via callbacks</li>
 *   <li>Stop sequences and EOS detection</li>
 *   <li>Repetition penalty</li>
 *   <li>Context management for long sequences</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * TextGenerator generator = TextGenerator.builder()
 *     .model(decoderModel)
 *     .tokenizer(tokenizer)
 *     .config(SamplingConfig.defaultConfig())
 *     .build();
 *
 * // Simple generation
 * String output = generator.generate("Once upon a time");
 *
 * // Streaming generation
 * generator.generateStreaming("Once upon a time", token -> {
 *     System.out.print(token);
 * });
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@Getter
public class TextGenerator {

    private final SameDiff model;
    private final Tokenizer tokenizer;
    private final SamplingConfig config;
    private final Sampler sampler;

    // Model input/output names
    private final String inputIdsName;
    private final String logitsOutputName;

    // Optional embedding layer for models that use embeddings
    private final SameDiff embeddings;
    private final String embeddingsInputName;
    private final String embeddingsOutputName;

    // Stop sequences
    private final List<int[]> stopSequences;

    @Builder
    public TextGenerator(
            SameDiff model,
            Tokenizer tokenizer,
            SamplingConfig config,
            String inputIdsName,
            String logitsOutputName,
            SameDiff embeddings,
            String embeddingsInputName,
            String embeddingsOutputName,
            List<String> stopStrings) {

        this.model = model;
        this.tokenizer = tokenizer;
        this.config = config != null ? config : SamplingConfig.defaultConfig();
        this.sampler = Sampler.fromConfig(this.config);

        // Default model I/O names
        this.inputIdsName = inputIdsName != null ? inputIdsName : "input_ids";
        this.logitsOutputName = logitsOutputName != null ? logitsOutputName : "logits";

        // Optional embeddings
        this.embeddings = embeddings;
        this.embeddingsInputName = embeddingsInputName != null ? embeddingsInputName : "input_ids";
        this.embeddingsOutputName = embeddingsOutputName != null ? embeddingsOutputName : "inputs_embeds";

        // Convert stop strings to token sequences
        this.stopSequences = new ArrayList<>();
        if (stopStrings != null) {
            for (String s : stopStrings) {
                Encoding enc = tokenizer.encode(s, false);
                stopSequences.add(enc.getIds());
            }
        }
    }

    /**
     * Generate text continuation from a prompt.
     *
     * @param prompt the input prompt
     * @return generated text (excluding the prompt)
     */
    public String generate(String prompt) {
        return generate(prompt, config.getMaxNewTokens());
    }

    /**
     * Generate text continuation with specified max tokens.
     *
     * @param prompt the input prompt
     * @param maxNewTokens maximum tokens to generate
     * @return generated text
     */
    public String generate(String prompt, int maxNewTokens) {
        return generateWithMetrics(prompt, maxNewTokens).getText();
    }

    /**
     * Generate text continuation from a prompt, returning detailed metrics.
     *
     * @param prompt the input prompt
     * @return generation result with metrics
     */
    public GenerationResult generateWithMetrics(String prompt) {
        return generateWithMetrics(prompt, config.getMaxNewTokens());
    }

    /**
     * Generate text continuation with specified max tokens, returning detailed metrics.
     *
     * @param prompt the input prompt
     * @param maxNewTokens maximum tokens to generate
     * @return generation result with metrics
     */
    public GenerationResult generateWithMetrics(String prompt, int maxNewTokens) {
        // Encode prompt
        Encoding encoding = tokenizer.encode(prompt, true);
        int[] promptIds = encoding.getIds();
        int promptTokenCount = promptIds.length;

        // Track generated tokens for repetition penalty
        List<Integer> allTokens = new ArrayList<>();
        for (int id : promptIds) {
            allTokens.add(id);
        }

        // Get EOS token
        int eosToken = config.getEosTokenId() >= 0 ? config.getEosTokenId() : tokenizer.getEosTokenId();

        // Autoregressive generation loop with timing
        INDArray inputIds = Nd4j.createFromArray(promptIds).reshape(1, promptIds.length);
        StringBuilder result = new StringBuilder();
        List<Integer> generatedTokenIds = new ArrayList<>();
        long startNanos = System.nanoTime();
        long firstTokenLatencyNanos = 0;
        int generatedTokenCount = 0;
        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        for (int step = 0; step < maxNewTokens; step++) {
            // Get logits for next token
            INDArray logits = getNextTokenLogits(inputIds);

            // Apply repetition penalty if configured
            if (config.hasRepetitionPenalty()) {
                int[] generated = allTokens.stream().mapToInt(Integer::intValue).toArray();
                logits = SamplerUtils.applyRepetitionPenalty(logits, generated, config.getRepetitionPenalty());
            }

            // Sample next token
            int nextTokenId = sampler.sample(logits);

            // Record first token latency
            if (generatedTokenCount == 0) {
                firstTokenLatencyNanos = System.nanoTime() - startNanos;
            }
            generatedTokenCount++;
            generatedTokenIds.add(nextTokenId);

            // Check for EOS
            if (nextTokenId == eosToken) {
                log.debug("EOS token generated at step {}", step);
                finishReason = GenerationResult.FinishReason.EOS;
                break;
            }

            // Decode and output token
            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            result.append(tokenText);

            // Track token
            allTokens.add(nextTokenId);

            // Check stop sequences
            if (matchesStopSequence(allTokens)) {
                log.debug("Stop sequence matched at step {}", step);
                finishReason = GenerationResult.FinishReason.STOP_SEQUENCE;
                break;
            }

            // Update input for next iteration
            inputIds = appendToken(inputIds, nextTokenId);
        }

        long totalNanos = System.nanoTime() - startNanos;
        long totalMs = totalNanos / 1_000_000;
        long firstTokenMs = firstTokenLatencyNanos / 1_000_000;
        int[] tokenIdArray = generatedTokenIds.stream().mapToInt(Integer::intValue).toArray();

        return GenerationResult.builder()
                .text(result.toString())
                .tokenIds(tokenIdArray)
                .generatedTokenCount(generatedTokenCount)
                .promptTokenCount(promptTokenCount)
                .totalTokenCount(promptTokenCount + generatedTokenCount)
                .finishReason(finishReason)
                .firstTokenLatencyMs(firstTokenMs)
                .generationTimeMs(totalMs)
                .tokensPerSecond(totalNanos > 0 ? (generatedTokenCount * 1_000_000_000.0) / totalNanos : 0)
                .build();
    }

    /**
     * Generate text with streaming output.
     *
     * @param prompt the input prompt
     * @param tokenCallback callback invoked for each generated token
     */
    public void generateStreaming(String prompt, Consumer<String> tokenCallback) {
        generateStreaming(prompt, config.getMaxNewTokens(), tokenCallback);
    }

    /**
     * Generate text with streaming output and max tokens.
     *
     * @param prompt the input prompt
     * @param maxNewTokens maximum tokens to generate
     * @param tokenCallback callback for each token
     */
    public void generateStreaming(String prompt, int maxNewTokens, Consumer<String> tokenCallback) {
        // Encode prompt
        Encoding encoding = tokenizer.encode(prompt, true);
        int[] promptIds = encoding.getIds();

        // Track generated tokens for repetition penalty
        List<Integer> allTokens = new ArrayList<>();
        for (int id : promptIds) {
            allTokens.add(id);
        }

        // Get EOS token
        int eosToken = config.getEosTokenId() >= 0 ? config.getEosTokenId() : tokenizer.getEosTokenId();

        // Autoregressive generation loop
        INDArray inputIds = Nd4j.createFromArray(promptIds).reshape(1, promptIds.length);

        for (int step = 0; step < maxNewTokens; step++) {
            // Get logits for next token
            INDArray logits = getNextTokenLogits(inputIds);

            // Apply repetition penalty if configured
            if (config.hasRepetitionPenalty()) {
                int[] generated = allTokens.stream().mapToInt(Integer::intValue).toArray();
                logits = SamplerUtils.applyRepetitionPenalty(logits, generated, config.getRepetitionPenalty());
            }

            // Sample next token
            int nextTokenId = sampler.sample(logits);

            // Check for EOS
            if (nextTokenId == eosToken) {
                log.debug("EOS token generated at step {}", step);
                break;
            }

            // Decode and output token
            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            if (tokenCallback != null) {
                tokenCallback.accept(tokenText);
            }

            // Track token
            allTokens.add(nextTokenId);

            // Check stop sequences
            if (matchesStopSequence(allTokens)) {
                log.debug("Stop sequence matched at step {}", step);
                break;
            }

            // Update input for next iteration
            inputIds = appendToken(inputIds, nextTokenId);
        }
    }

    /**
     * Generate with custom logits processor.
     *
     * <p>The processor can modify logits before sampling, enabling
     * constrained generation, grammar enforcement, etc.</p>
     *
     * @param prompt the input prompt
     * @param maxNewTokens maximum tokens
     * @param logitsProcessor function that processes logits: (logits, generatedTokens) -> processedLogits
     * @return generated text
     */
    public String generateWithProcessor(String prompt, int maxNewTokens,
                                        BiFunction<INDArray, List<Integer>, INDArray> logitsProcessor) {
        StringBuilder result = new StringBuilder();

        Encoding encoding = tokenizer.encode(prompt, true);
        int[] promptIds = encoding.getIds();

        List<Integer> allTokens = new ArrayList<>();
        for (int id : promptIds) {
            allTokens.add(id);
        }

        int eosToken = config.getEosTokenId() >= 0 ? config.getEosTokenId() : tokenizer.getEosTokenId();
        INDArray inputIds = Nd4j.createFromArray(promptIds).reshape(1, promptIds.length);

        for (int step = 0; step < maxNewTokens; step++) {
            INDArray logits = getNextTokenLogits(inputIds);

            // Apply custom processor
            logits = logitsProcessor.apply(logits, allTokens);

            int nextTokenId = sampler.sample(logits);

            if (nextTokenId == eosToken) {
                break;
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            result.append(tokenText);
            allTokens.add(nextTokenId);

            if (matchesStopSequence(allTokens)) {
                break;
            }

            inputIds = appendToken(inputIds, nextTokenId);
        }

        return result.toString();
    }

    /**
     * Get logits for the next token position.
     *
     * @param inputIds current input token IDs, shape [1, seqLen]
     * @return logits for next token, shape [vocabSize]
     */
    private INDArray getNextTokenLogits(INDArray inputIds) {
        Map<String, INDArray> inputs = new HashMap<>();

        if (embeddings != null) {
            // Two-stage: embeddings then decoder
            Map<String, INDArray> embInputs = new HashMap<>();
            embInputs.put(embeddingsInputName, inputIds);
            Map<String, INDArray> embOutputs = embeddings.output(embInputs, embeddingsOutputName);
            inputs.put("inputs_embeds", embOutputs.get(embeddingsOutputName));
        } else {
            // Direct input_ids to model
            inputs.put(inputIdsName, inputIds);
        }

        Map<String, INDArray> outputs = model.output(inputs, logitsOutputName);
        INDArray logits = outputs.get(logitsOutputName);

        // Get logits for the last position: [batch, seqLen, vocab] -> [vocab]
        if (logits.rank() == 3) {
            long lastPos = logits.size(1) - 1;
            logits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPos), NDArrayIndex.all());
        } else if (logits.rank() == 2) {
            logits = logits.getRow(0);
        }

        return logits;
    }

    /**
     * Append a token to the input sequence.
     *
     * @param inputIds current input IDs
     * @param tokenId token to append
     * @return new input IDs with token appended
     */
    private INDArray appendToken(INDArray inputIds, int tokenId) {
        INDArray newToken = Nd4j.scalar(tokenId).reshape(1, 1);
        return Nd4j.concat(1, inputIds, newToken);
    }

    /**
     * Check if the generated sequence ends with any stop sequence.
     *
     * @param tokens generated tokens
     * @return true if matches a stop sequence
     */
    private boolean matchesStopSequence(List<Integer> tokens) {
        for (int[] stopSeq : stopSequences) {
            if (tokens.size() >= stopSeq.length) {
                boolean matches = true;
                for (int i = 0; i < stopSeq.length; i++) {
                    if (tokens.get(tokens.size() - stopSeq.length + i) != stopSeq[i]) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Create a simple generator with default settings.
     *
     * @param model the decoder model
     * @param tokenizer the tokenizer
     * @return configured generator
     */
    public static TextGenerator simple(SameDiff model, Tokenizer tokenizer) {
        return TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .build();
    }

    /**
     * Create a generator optimized for creative writing.
     *
     * @param model the decoder model
     * @param tokenizer the tokenizer
     * @return creative-optimized generator
     */
    public static TextGenerator creative(SameDiff model, Tokenizer tokenizer) {
        return TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .config(SamplingConfig.creative())
                .build();
    }

    /**
     * Create a generator optimized for precise/factual output.
     *
     * @param model the decoder model
     * @param tokenizer the tokenizer
     * @return precise-optimized generator
     */
    public static TextGenerator precise(SameDiff model, Tokenizer tokenizer) {
        return TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .config(SamplingConfig.precise())
                .build();
    }
}
