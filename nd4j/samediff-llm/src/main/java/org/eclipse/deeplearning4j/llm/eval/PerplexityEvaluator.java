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

package org.eclipse.deeplearning4j.llm.eval;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Perplexity evaluation utility for language models.
 *
 * Implements sliding-window perplexity computation where the model processes
 * overlapping windows of tokens and the cross-entropy loss is accumulated
 * over the non-overlapping portion of each window.
 */
@Slf4j
public class PerplexityEvaluator {

    private static final String WIKITEXT2_URL =
            "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet";
    // wikitext-2 test split as shipped in the pytorch/examples repo (the previous
    // fairseq wikitext-103 path 404s — it never existed at that location).
    private static final String WIKITEXT2_TXT_URL =
            "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt";

    @Data
    @Builder
    public static class PerplexityResult {
        private final double perplexity;
        private final double bitsPerByte;
        private final int numTokens;
        private final long evaluationTimeMs;
        private final double avgNegLogLikelihood;
    }

    /**
     * Evaluate perplexity of a model on the given text using sliding windows.
     *
     * @param model the SameDiff language model (must output "logits")
     * @param tokenizer the tokenizer
     * @param text evaluation text
     * @param contextLength max context window for the model
     * @param stride step size between windows (stride < contextLength for overlap)
     * @return perplexity result
     */
    public static PerplexityResult evaluate(SameDiff model, Tokenizer tokenizer,
                                             String text, int contextLength, int stride) {
        return evaluate(model, tokenizer, text, contextLength, stride, "input_ids", "logits");
    }

    /**
     * Evaluate perplexity with custom input/output names.
     *
     * <p>Uses {@link DecoderInputBuilder#buildScoringInputMap} to build a complete input map
     * so that hybrid architectures (recurrent states, in-graph KV caches, scalar scalars like
     * position_offset / cache_position) work correctly — not just the bare singleton
     * {@code {inputName → ids}} map. The {@code inputName} parameter is accepted for API
     * backward compatibility but the discovered input_ids name from the graph takes precedence.</p>
     *
     * <p>If {@code logitsName} is not found among the model's declared outputs, the method
     * falls back to {@link ModelIOConfig#findLogitsOutputName(SameDiff)} (handles models that
     * expose {@code "lm_logits"} instead of {@code "logits"}, e.g. Qwen3.5).</p>
     */
    public static PerplexityResult evaluate(SameDiff model, Tokenizer tokenizer,
                                             String text, int contextLength, int stride,
                                             String inputName, String logitsName) {
        long startTime = System.currentTimeMillis();

        int[] tokenIds = tokenizer.encode(text, false).getIds();
        int seqLen = tokenIds.length;
        log.info("Evaluating perplexity on {} tokens with context={}, stride={}", seqLen, contextLength, stride);

        // Resolve the effective logits output name once — fall back to graph discovery when the
        // caller-supplied name (default "logits") is absent from the model's declared outputs.
        String effectiveLogitsName = logitsName;
        if (!model.outputs().contains(logitsName)) {
            String discovered = ModelIOConfig.findLogitsOutputName(model);
            if (discovered != null) {
                effectiveLogitsName = discovered;
                log.info("PerplexityEvaluator: logits output '{}' not in model outputs; using discovered name '{}'",
                        logitsName, effectiveLogitsName);
            }
        }

        double totalNll = 0.0;
        int totalTokens = 0;

        for (int begin = 0; begin < seqLen; begin += stride) {
            int end = Math.min(begin + contextLength, seqLen);
            int targetStart = Math.max(begin, begin == 0 ? 0 : begin + (contextLength - stride));

            // Extract window
            int windowLen = end - begin;
            int[] windowIds = new int[windowLen];
            System.arraycopy(tokenIds, begin, windowIds, 0, windowLen);

            // Build a complete input map (handles standard inputs, GGUF scalars, and recurrent states).
            INDArray inputArr = Nd4j.createFromArray(windowIds).reshape(1, windowLen);
            Map<String, INDArray> inputs = DecoderInputBuilder.buildScoringInputMap(model, inputArr);
            Map<String, INDArray> outputs = model.output(inputs, effectiveLogitsName);
            INDArray logits = outputs.get(effectiveLogitsName);

            // logits shape: [1, seqLen, vocabSize]
            if (logits.rank() == 3) {
                logits = logits.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
            }

            // Compute cross-entropy for target positions
            int localTargetStart = targetStart - begin;
            for (int pos = Math.max(localTargetStart, 1); pos < windowLen; pos++) {
                // Get logits at position pos-1 (predicting token at pos)
                INDArray posLogits = logits.getRow(pos - 1);

                // Log-softmax
                double maxLogit = posLogits.maxNumber().doubleValue();
                INDArray shifted = posLogits.sub(maxLogit);
                INDArray expShifted = Nd4j.math.exp(shifted);
                double logSumExp = Math.log(expShifted.sumNumber().doubleValue()) + maxLogit;

                int targetToken = windowIds[pos];
                double logProb = posLogits.getDouble(targetToken) - logSumExp;

                totalNll -= logProb;
                totalTokens++;
            }

            if (end >= seqLen) break;

            if (totalTokens > 0 && totalTokens % 1000 < stride) {
                double currentPpl = Math.exp(totalNll / totalTokens);
                log.debug("Progress: {}/{} tokens, current perplexity: {:.2f}", totalTokens, seqLen, currentPpl);
            }
        }

        long evalTimeMs = System.currentTimeMillis() - startTime;
        double avgNll = totalTokens > 0 ? totalNll / totalTokens : 0;
        double perplexity = Math.exp(avgNll);
        double bitsPerByte = avgNll / Math.log(2);

        log.info("Perplexity: {:.4f}, bits/byte: {:.4f}, tokens: {}, time: {}ms",
                perplexity, bitsPerByte, totalTokens, evalTimeMs);

        return PerplexityResult.builder()
                .perplexity(perplexity)
                .bitsPerByte(bitsPerByte)
                .numTokens(totalTokens)
                .evaluationTimeMs(evalTimeMs)
                .avgNegLogLikelihood(avgNll)
                .build();
    }

    /**
     * Evaluate perplexity on WikiText-2 test set.
     * Downloads and caches the dataset automatically.
     */
    public static PerplexityResult evaluateWikiText2(SameDiff model, Tokenizer tokenizer,
                                                      int contextLength, int stride) throws IOException {
        String text = loadWikiText2();
        return evaluate(model, tokenizer, text, contextLength, stride);
    }

    /**
     * Load WikiText-2 test set, downloading if needed.
     */
    public static String loadWikiText2() throws IOException {
        String cacheDir = System.getProperty(ND4JSystemProperties.LLM_MODEL_CACHE_DIR,
                System.getProperty("user.home") + "/.cache/dl4j-llm-models");
        File cacheFile = new File(cacheDir, "wikitext2-test.txt");

        if (cacheFile.exists()) {
            log.info("Using cached WikiText-2: {}", cacheFile.getAbsolutePath());
            return Files.readString(cacheFile.toPath(), StandardCharsets.UTF_8);
        }

        log.info("Downloading WikiText-2 test set...");
        cacheFile.getParentFile().mkdirs();

        URL url = new URL(WIKITEXT2_TXT_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(60000);

        int responseCode = conn.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP ||
                responseCode == HttpURLConnection.HTTP_MOVED_PERM ||
                responseCode == 307 || responseCode == 308) {
            String redirectUrl = conn.getHeaderField("Location");
            conn.disconnect();
            conn = (HttpURLConnection) new URL(redirectUrl).openConnection();
            conn.setRequestMethod("GET");
            responseCode = conn.getResponseCode();
        }

        if (responseCode != HttpURLConnection.HTTP_OK) {
            conn.disconnect();
            throw new IOException("Failed to download WikiText-2: HTTP " + responseCode);
        }

        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        conn.disconnect();

        String text = sb.toString();
        Files.writeString(cacheFile.toPath(), text, StandardCharsets.UTF_8);
        log.info("Cached WikiText-2 ({} chars) to {}", text.length(), cacheFile.getAbsolutePath());

        return text;
    }
}
