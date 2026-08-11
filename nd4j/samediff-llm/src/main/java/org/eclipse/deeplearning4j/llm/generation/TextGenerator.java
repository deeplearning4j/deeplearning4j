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
import org.eclipse.deeplearning4j.llm.generation.batch.BatchGenerationState;
import org.eclipse.deeplearning4j.llm.generation.sampling.Sampler;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
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
                INDArray original = logits;
                int[] generated = allTokens.stream().mapToInt(Integer::intValue).toArray();
                logits = SamplerUtils.applyRepetitionPenalty(logits, generated, config.getRepetitionPenalty());
                if (logits != original) {
                    SameDiffMemoryUtils.safeClose(original);
                }
            }

            // Diagnostic: log logits stats for first 3 tokens — gated on debug level
            // to avoid unconditional dup+reduction overhead in the hot path
            if (log.isDebugEnabled() && step < 3) {
                INDArray logitsDup = logits.dup();
                float maxVal = logitsDup.maxNumber().floatValue();
                float minVal = logitsDup.minNumber().floatValue();
                float meanVal = logitsDup.meanNumber().floatValue();
                // Find top-5 token IDs
                int[] top5Ids = new int[5];
                float[] top5Vals = new float[5];
                for (int t = 0; t < 5; t++) {
                    int idx = logitsDup.argMax(0).getInt(0);
                    top5Ids[t] = idx;
                    top5Vals[t] = logitsDup.getFloat(idx);
                    logitsDup.putScalar(idx, Float.NEGATIVE_INFINITY);
                }
                log.debug("Step {} logits: min={}, max={}, mean={}, top5ids={}, top5vals={}",
                        step, minVal, maxVal, meanVal,
                        java.util.Arrays.toString(top5Ids), java.util.Arrays.toString(top5Vals));
                logitsDup.close();
            }

            // Sample next token
            int nextTokenId = sampler.sample(logits);
            SameDiffMemoryUtils.safeClose(logits);

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

            // Update input for next iteration — close old inputIds
            INDArray oldInputIds = inputIds;
            inputIds = appendToken(inputIds, nextTokenId);
            SameDiffMemoryUtils.safeClose(oldInputIds);
        }

        SameDiffMemoryUtils.safeClose(inputIds);

        long totalNanos = System.nanoTime() - startNanos;
        long totalMs = totalNanos / 1_000_000;
        long firstTokenMs = firstTokenLatencyNanos / 1_000_000;
        int[] tokenIdArray = generatedTokenIds.stream().mapToInt(Integer::intValue).toArray();

        // Decode the complete generated sequence verbatim. Completion generation has no
        // authority to guess chat delimiters or reasoning syntax; generateChat performs
        // model-template-aware normalization at the structured boundary.
        String fullText = tokenIdArray.length > 0 ? tokenizer.decode(tokenIdArray, false) : "";

        return GenerationResult.builder()
                .text(fullText)
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
                INDArray original = logits;
                int[] generated = allTokens.stream().mapToInt(Integer::intValue).toArray();
                logits = SamplerUtils.applyRepetitionPenalty(logits, generated, config.getRepetitionPenalty());
                if (logits != original) {
                    SameDiffMemoryUtils.safeClose(original);
                }
            }

            // Sample next token
            int nextTokenId = sampler.sample(logits);
            SameDiffMemoryUtils.safeClose(logits);

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

            // Update input for next iteration — close old inputIds
            INDArray oldInputIds = inputIds;
            inputIds = appendToken(inputIds, nextTokenId);
            SameDiffMemoryUtils.safeClose(oldInputIds);
        }

        SameDiffMemoryUtils.safeClose(inputIds);
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
            INDArray original = logits;
            logits = logitsProcessor.apply(logits, allTokens);
            if (logits != original) {
                SameDiffMemoryUtils.safeClose(original);
            }

            int nextTokenId = sampler.sample(logits);
            SameDiffMemoryUtils.safeClose(logits);

            if (nextTokenId == eosToken) {
                break;
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            result.append(tokenText);
            allTokens.add(nextTokenId);

            if (matchesStopSequence(allTokens)) {
                break;
            }

            INDArray oldInputIds = inputIds;
            inputIds = appendToken(inputIds, nextTokenId);
            SameDiffMemoryUtils.safeClose(oldInputIds);
        }

        SameDiffMemoryUtils.safeClose(inputIds);
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
        INDArray fullLogits = outputs.get(logitsOutputName);

        // Get logits for the last position: [batch, seqLen, vocab] -> [vocab]
        // Return a dup so the caller owns it and the full logits can be freed
        INDArray lastLogits;
        if (fullLogits.rank() == 3) {
            long lastPos = fullLogits.size(1) - 1;
            lastLogits = fullLogits.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPos), NDArrayIndex.all()).dup();
        } else if (fullLogits.rank() == 2) {
            lastLogits = fullLogits.getRow(0).dup();
        } else {
            lastLogits = fullLogits.dup();
        }

        SameDiffMemoryUtils.safeClose(fullLogits);
        return lastLogits;
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
        INDArray result = Nd4j.concat(1, inputIds, newToken);
        newToken.close();
        return result;
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
     * Generate text for multiple prompts in a batch.
     *
     * <p>All prompts are processed together through the model at each decode step,
     * providing better GPU utilization than sequential generation. Prompts are
     * right-padded to equal length for batching.</p>
     *
     * <p>Each sequence independently tracks its finish condition (EOS, max tokens,
     * stop sequence). Finished sequences contribute padding tokens to the batch
     * but their outputs are ignored.</p>
     *
     * @param prompts array of input prompts
     * @param maxNewTokens maximum tokens to generate per sequence
     * @return array of generation results, one per prompt
     */
    public GenerationResult[] generateBatch(String[] prompts, int maxNewTokens) {
        int batchSize = prompts.length;
        if (batchSize == 0) {
            return new GenerationResult[0];
        }
        if (batchSize == 1) {
            return new GenerationResult[]{generateWithMetrics(prompts[0], maxNewTokens)};
        }

        int eosToken = config.getEosTokenId() >= 0 ? config.getEosTokenId() : tokenizer.getEosTokenId();
        int padToken = config.getPadTokenId() >= 0 ? config.getPadTokenId() : eosToken;

        // Encode all prompts
        int[][] promptIds = new int[batchSize][];
        int maxPromptLen = 0;
        for (int i = 0; i < batchSize; i++) {
            promptIds[i] = tokenizer.encode(prompts[i], true).getIds();
            maxPromptLen = Math.max(maxPromptLen, promptIds[i].length);
        }

        // Right-pad to max prompt length
        int[][] paddedIds = new int[batchSize][maxPromptLen];
        for (int i = 0; i < batchSize; i++) {
            int padLen = maxPromptLen - promptIds[i].length;
            for (int j = 0; j < padLen; j++) {
                paddedIds[i][j] = padToken;
            }
            System.arraycopy(promptIds[i], 0, paddedIds[i], padLen, promptIds[i].length);
        }

        // Build batch state with stop tokens
        int[] additionalStops = stopSequences.isEmpty() ? new int[0] :
                new int[]{stopSequences.get(0).length == 1 ? stopSequences.get(0)[0] : -1};
        BatchGenerationState state = new BatchGenerationState(batchSize, eosToken, additionalStops);

        // Track all tokens per sequence for repetition penalty
        List<List<Integer>> allTokens = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            List<Integer> tokens = new ArrayList<>();
            for (int id : promptIds[i]) tokens.add(id);
            allTokens.add(tokens);
        }

        // Create batched input: [batchSize, seqLen]
        INDArray batchInputIds = Nd4j.createFromArray(paddedIds).castTo(DataType.LONG);

        long startNanos = System.nanoTime();

        for (int step = 0; step < maxNewTokens; step++) {
            // Get logits for all sequences: [batchSize, seqLen, vocabSize]
            INDArray batchLogits = getBatchLogits(batchInputIds);

            // Extract last-position logits: [batchSize, vocabSize]
            INDArray lastLogits;
            if (batchLogits.rank() == 3) {
                lastLogits = batchLogits.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.point(batchLogits.size(1) - 1),
                        NDArrayIndex.all()).dup();
            } else {
                lastLogits = batchLogits.dup();
            }
            SameDiffMemoryUtils.safeClose(batchLogits);

            // Sample batch
            int[] nextTokenIds = sampler.sampleBatch(lastLogits);
            SameDiffMemoryUtils.safeClose(lastLogits);

            // Record and check stop conditions
            long stepNanos = System.nanoTime() - startNanos;
            state.recordTokens(nextTokenIds, stepNanos);

            // Track tokens for repetition penalty
            for (int i = 0; i < batchSize; i++) {
                if (!state.isFinished(i)) {
                    allTokens.get(i).add(nextTokenIds[i]);
                }
            }

            if (state.allFinished()) {
                break;
            }

            // Append tokens for next step
            int[] batchNextTokens = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                batchNextTokens[i] = state.isFinished(i) ? padToken : nextTokenIds[i];
            }
            INDArray nextTokenTensor = Nd4j.createFromArray(batchNextTokens)
                    .reshape(batchSize, 1).castTo(DataType.LONG);
            INDArray oldBatchInputIds = batchInputIds;
            batchInputIds = Nd4j.concat(1, batchInputIds, nextTokenTensor);
            nextTokenTensor.close();
            SameDiffMemoryUtils.safeClose(oldBatchInputIds);
        }

        SameDiffMemoryUtils.safeClose(batchInputIds);

        long totalNanos = System.nanoTime() - startNanos;

        // Decode results
        String[] texts = new String[batchSize];
        int[] promptTokenCounts = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int[] genTokens = state.getTokenArrayForSequence(i);
            texts[i] = tokenizer.decode(genTokens, true);
            promptTokenCounts[i] = promptIds[i].length;
        }

        return state.buildResults(texts, promptTokenCounts, totalNanos);
    }

    /**
     * Generate text for multiple prompts using default max tokens.
     *
     * @param prompts array of input prompts
     * @return array of generation results
     */
    public GenerationResult[] generateBatch(String[] prompts) {
        return generateBatch(prompts, config.getMaxNewTokens());
    }

    /**
     * Generate one output per chunk, batched where possible.
     *
     * @param chunks ordered text chunks (for example: document pages/chunks)
     * @param maxNewTokens maximum tokens to generate per chunk
     * @return generation results aligned with input chunk order
     */
    public GenerationResult[] generateChunks(List<String> chunks, int maxNewTokens) {
        if (chunks == null || chunks.isEmpty()) {
            return new GenerationResult[0];
        }
        return generateBatch(chunks.toArray(new String[0]), maxNewTokens);
    }

    /**
     * Generate one output per chunk using the default max token setting.
     *
     * @param chunks ordered text chunks
     * @return generation results aligned with input chunk order
     */
    public GenerationResult[] generateChunks(List<String> chunks) {
        return generateChunks(chunks, config.getMaxNewTokens());
    }

    /**
     * Generate one combined document output from multiple chunks.
     *
     * @param chunks ordered text chunks
     * @param maxNewTokens maximum tokens to generate per chunk
     * @param chunkDelimiter delimiter inserted between per-chunk outputs
     * @return combined output text
     */
    public String generateDocumentFromChunks(List<String> chunks, int maxNewTokens, String chunkDelimiter) {
        GenerationResult[] results = generateChunks(chunks, maxNewTokens);
        String delimiter = chunkDelimiter == null ? "" : chunkDelimiter;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < results.length; i++) {
            if (i > 0) {
                sb.append(delimiter);
            }
            if (results[i] != null && results[i].getText() != null) {
                sb.append(results[i].getText());
            }
        }
        return sb.toString();
    }

    /**
     * Generate one combined document output from multiple chunks using default max token settings.
     *
     * @param chunks ordered text chunks
     * @param chunkDelimiter delimiter inserted between per-chunk outputs
     * @return combined output text
     */
    public String generateDocumentFromChunks(List<String> chunks, String chunkDelimiter) {
        return generateDocumentFromChunks(chunks, config.getMaxNewTokens(), chunkDelimiter);
    }

    /**
     * Split text into overlapping token chunks.
     *
     * @param text source text to chunk
     * @param chunkTokenSize max tokens per chunk (must be > 0)
     * @param overlapTokens overlap between consecutive chunks (must be &gt;= 0 and &lt; chunkTokenSize)
     * @return ordered chunk strings
     */
    public List<String> chunkText(String text, int chunkTokenSize, int overlapTokens) {
        if (chunkTokenSize <= 0) {
            throw new IllegalArgumentException("chunkTokenSize must be > 0");
        }
        if (overlapTokens < 0 || overlapTokens >= chunkTokenSize) {
            throw new IllegalArgumentException("overlapTokens must be >= 0 and < chunkTokenSize");
        }

        String safeText = text == null ? "" : text;
        int[] tokenIds = tokenizer.encode(safeText, false).getIds();
        if (tokenIds.length == 0) {
            return List.of("");
        }

        int stride = chunkTokenSize - overlapTokens;
        List<String> chunks = new ArrayList<>();
        for (int start = 0; start < tokenIds.length; start += stride) {
            int end = Math.min(start + chunkTokenSize, tokenIds.length);
            int[] chunkIds = Arrays.copyOfRange(tokenIds, start, end);
            chunks.add(tokenizer.decode(chunkIds, false));
            if (end >= tokenIds.length) {
                break;
            }
        }
        return chunks;
    }

    /**
     * Chunk text by token count, then generate one output per chunk.
     *
     * @param text source text
     * @param chunkTokenSize max tokens per chunk (must be > 0)
     * @param overlapTokens overlap between chunks (must be >= 0 and < chunkTokenSize)
     * @param maxNewTokens maximum tokens to generate per chunk
     * @return generation results aligned with chunk order
     */
    public GenerationResult[] generateChunked(String text, int chunkTokenSize, int overlapTokens, int maxNewTokens) {
        List<String> chunks = chunkText(text, chunkTokenSize, overlapTokens);
        return generateChunks(chunks, maxNewTokens);
    }

    /**
     * Chunk text by token count and generate one combined output.
     *
     * @param text source text
     * @param chunkTokenSize max tokens per chunk (must be > 0)
     * @param overlapTokens overlap between chunks (must be >= 0 and < chunkTokenSize)
     * @param maxNewTokens maximum tokens to generate per chunk
     * @param chunkDelimiter delimiter inserted between chunk outputs
     * @return combined chunk output
     */
    public String generateChunkedDocument(String text, int chunkTokenSize, int overlapTokens,
                                          int maxNewTokens, String chunkDelimiter) {
        GenerationResult[] results = generateChunked(text, chunkTokenSize, overlapTokens, maxNewTokens);
        String delimiter = chunkDelimiter == null ? "" : chunkDelimiter;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < results.length; i++) {
            if (i > 0) {
                sb.append(delimiter);
            }
            if (results[i] != null && results[i].getText() != null) {
                sb.append(results[i].getText());
            }
        }
        return sb.toString();
    }

    /**
     * Chunk text by token count and generate one combined output using default max token settings.
     *
     * @param text source text
     * @param chunkTokenSize max tokens per chunk (must be > 0)
     * @param overlapTokens overlap between chunks (must be >= 0 and < chunkTokenSize)
     * @param chunkDelimiter delimiter inserted between chunk outputs
     * @return combined chunk output
     */
    public String generateChunkedDocument(String text, int chunkTokenSize, int overlapTokens, String chunkDelimiter) {
        return generateChunkedDocument(text, chunkTokenSize, overlapTokens, config.getMaxNewTokens(), chunkDelimiter);
    }

    /**
     * Get batched logits from the model.
     *
     * @param batchInputIds input token IDs, shape [batchSize, seqLen]
     * @return logits, shape [batchSize, seqLen, vocabSize] or [batchSize, vocabSize]
     */
    private INDArray getBatchLogits(INDArray batchInputIds) {
        Map<String, INDArray> inputs = new HashMap<>();

        if (embeddings != null) {
            Map<String, INDArray> embInputs = new HashMap<>();
            embInputs.put(embeddingsInputName, batchInputIds);
            Map<String, INDArray> embOutputs = embeddings.output(embInputs, embeddingsOutputName);
            inputs.put("inputs_embeds", embOutputs.get(embeddingsOutputName));
        } else {
            inputs.put(inputIdsName, batchInputIds);
        }

        Map<String, INDArray> outputs = model.output(inputs, logitsOutputName);
        INDArray logits = outputs.get(logitsOutputName);
        // Return a dup so the caller owns it and SameDiff session arrays can be freed
        INDArray result = logits.dup();
        SameDiffMemoryUtils.safeClose(logits);
        return result;
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
