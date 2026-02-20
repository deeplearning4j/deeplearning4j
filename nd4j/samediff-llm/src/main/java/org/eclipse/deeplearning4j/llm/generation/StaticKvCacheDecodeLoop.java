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
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Autoregressive decode loop with static KV cache for CUDA graph capture.
 *
 * <p>Encapsulates the full decode pipeline: prefill with dynamic KV, transition to
 * fixed-shape static KV buffers, frozen shapes for CUDA graph replay, GPU-side
 * token sampling via {@link TokenSample}, and per-step KV scatter.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
 *     .decoder(decoder)
 *     .embedTokens(embedTokens)
 *     .tokenizer(tokenizer)
 *     .maxNewTokens(256)
 *     .hiddenSize(hiddenSize)
 *     .build();
 * GenerationResult result = loop.decode(prefillEmbeddings, promptTokenIds);
 * }</pre>
 */
@Slf4j
@Builder
public class StaticKvCacheDecodeLoop {

    private final SameDiff decoder;
    private final SameDiff embedTokens;
    private final Tokenizer tokenizer;

    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();
    @Builder.Default
    private final int maxNewTokens = 256;
    @Builder.Default
    private final long hiddenSize = 0;

    private final String embedInputName;
    private final String[] embedOutputNames;
    private final Set<Integer> additionalStopTokenIds;

    /**
     * Run the autoregressive decode loop.
     *
     * @param prefillEmbeddings merged embeddings for the full prompt [1, seqLen, hidden]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @return generation result with text, token IDs, and timing
     */
    public GenerationResult decode(INDArray prefillEmbeddings, int[] promptTokenIds) {
        long decodeStart = System.currentTimeMillis();

        // Resolve I/O names
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = decoder.inputs();
        log.info("  Decoder input names: {}", decoderInputNames);

        String resolvedEmbedInputName = embedInputName;
        if (resolvedEmbedInputName == null) {
            resolvedEmbedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        }
        String[] resolvedEmbedOutputNames = embedOutputNames;
        if (resolvedEmbedOutputNames == null) {
            resolvedEmbedOutputNames = embedTokens.outputs().toArray(new String[0]);
        }

        // Resolve stop tokens
        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        if (endOfUtteranceTokenId != null) {
            stopTokenIds.add(endOfUtteranceTokenId);
        }
        if (additionalStopTokenIds != null) {
            stopTokenIds.addAll(additionalStopTokenIds);
        }

        // Resolve hidden size
        long resolvedHiddenSize = hiddenSize;
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = prefillEmbeddings.shape()[2];
        }

        // Extract embedding weight table for direct lookup (avoids full SameDiff.output() per token).
        // The weight may be stored as CONSTANT or VARIABLE depending on the ONNX import path.
        INDArray embeddingTable = null;
        for (SDVariable var : embedTokens.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        if (embeddingTable != null) {
            log.info("  Using direct embedding lookup: shape={} (bypasses SameDiff.output() per token)",
                    Arrays.toString(embeddingTable.shape()));
        } else {
            log.warn("  Could not extract embedding table, falling back to SameDiff.output() per token");
        }

        // Build output name list
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        // State
        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = prefillEmbeddings;
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        long pastSeqLen = 0;

        // Per-step timing
        List<Long> stepTimesMs = new ArrayList<>();
        long prefillTimeMs = 0;

        // Phase timing accumulators (steps 3+)
        long totalInputBuildNs = 0, totalDecoderNs = 0, totalLogitsDupNs = 0;
        long totalKvUpdateNs = 0, totalSamplingNs = 0;
        int detailSteps = 0;

        // Static KV cache state
        Map<String, INDArray> staticKvBuffers = null;
        long maxKvLen = -1;
        long cachePos = 0;
        boolean usingStaticKv = false;
        boolean kvScatterInCpp = false;  // When true, C++ handles KV scatter — skip Java side

        // Reusable input arrays — avoids per-step allocation of masks/position_ids
        Map<String, INDArray> reusableInputs = new HashMap<>();

        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.nanoTime();
            long currentSeqLen = currentEmbeddings.shape()[1];

            // Build input map (with reusable input cache for decode steps)
            Map<String, INDArray> decoderInputMap = DecoderUtils.buildDecoderInputMap(
                    decoderInputNames, decoder,
                    currentEmbeddings, currentInputIds,
                    pastSeqLen, currentSeqLen,
                    staticKvBuffers, maxKvLen, cachePos,
                    usingStaticKv, resolvedHiddenSize,
                    reusableInputs);

            long tAfterInputBuild = System.nanoTime();

            // Diagnostic logging for steps 1-3
            if (step >= 1 && step <= 3) {
                logDiagnostics(step, decoderInputMap);
            }

            // Run decoder — use fast path when shapes are frozen (skips setCloseable overhead)
            Map<String, INDArray> decoderOutputs;
            if (usingStaticKv && step >= 2) {
                decoderOutputs = decoder.outputDirect(
                        decoderInputMap, allOutputNames.toArray(new String[0]));
            } else {
                decoderOutputs = decoder.output(
                        decoderInputMap, allOutputNames.toArray(new String[0]));
            }

            long tAfterDecoder = System.nanoTime();

            // Diagnostic: present KV shapes at steps 1-3
            if (step >= 1 && step <= 3) {
                logPresentKvDiagnostics(step, decoderOutputs, kvNames);
            }

            // Extract logits — keep raw reference, no dup needed
            // TokenSample now accepts rank-3 [batch, seqLen, vocabSize] directly
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits output at step {}", step);
                break;
            }

            // Diagnostic: top logit values at steps 1-3
            if (step >= 1 && step <= 3) {
                logLogitsDiagnostics(step, logitsRaw);
            }

            long tAfterLogitsDup = System.nanoTime();

            // KV cache update
            if (usingStaticKv) {
                if (kvScatterInCpp) {
                    // C++ frozen fast path handles scatter + position advance automatically
                    cachePos++;
                } else {
                    // Java fallback: scatter new KV entries via view+assign
                    DecoderUtils.scatterNewKvEntries(staticKvBuffers, decoderOutputs,
                            kvNames.keyNames, kvNames.valueNames, maxKvLen, cachePos);
                    cachePos++;
                }
            } else {
                // Step 0 (prefill): transition to static KV
                long prefillSeqLen = currentSeqLen;
                maxKvLen = prefillSeqLen + maxNewTokens;
                log.info("  Setting up static KV: prefillLen={}, maxKvLen={} ({} tensors)",
                        prefillSeqLen, maxKvLen, kvNames.keyNames.size() + kvNames.valueNames.size());
                staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                        decoderOutputs, kvNames.keyNames, kvNames.valueNames, maxKvLen);
                Nd4j.getExecutioner().commit();
                cachePos = prefillSeqLen;
                usingStaticKv = true;

                // Close prefill KV outputs
                for (String pn : kvNames.keyNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }
                for (String pn : kvNames.valueNames) {
                    INDArray pv = decoderOutputs.get(pn);
                    if (pv != null) { pv.setCloseable(true); pv.close(); }
                }

                // Freeze shapes for CUDA graph capture
                InferenceSession decoderSession = decoder.getOrCreateSession();
                DynamicShapePlanExecutor dspExec = decoderSession.getDynamicShapePlanExecutor();
                if (dspExec != null) {
                    dspExec.setShapesFrozen(true);
                    dspExec.setTraceEnabled(true);
                    dspExec.setExecutionTimingEnabled(true);

                    // Configure C++ KV scatter: present outputs → static input buffers
                    // This eliminates 60 copyBuffer + 60 Java view+assign per step
                    boolean cppKvEnabled = !"true".equals(System.getProperty("nd4j.dsp.kvscatter.java", "false"));
                    DynamicShapePlan plan = cppKvEnabled ? dspExec.getCurrentPlan() : null;
                    if (plan != null) {
                        List<String> presentNames = new ArrayList<>();
                        presentNames.addAll(kvNames.keyNames);
                        presentNames.addAll(kvNames.valueNames);
                        List<String> pastNames = new ArrayList<>();
                        for (String pn : presentNames) {
                            pastNames.add(pn.replace("present", "past_key_values"));
                        }
                        dspExec.configureKvCacheRetention(plan, presentNames, pastNames,
                                (int) maxKvLen, (int) cachePos);
                        kvScatterInCpp = true;
                        log.info("  [Perf] C++ KV scatter enabled: {} mappings, pos={}",
                                presentNames.size(), cachePos);
                    }

                    log.info("  [Perf] Shapes frozen — static KV buffer shape=[1,h,{},d], decode fast path active", maxKvLen);
                } else {
                    log.warn("  [Perf] No DSP executor found to freeze shapes");
                }
            }

            long tAfterKvUpdate = System.nanoTime();

            // Sample next token via native GPU op — pass logits directly (rank-3 supported)
            long tSampStart = System.nanoTime();

            TokenSample tokenSampleOp;
            if (samplingConfig.isGreedy()) {
                tokenSampleOp = new TokenSample(logitsRaw);
            } else {
                tokenSampleOp = new TokenSample(logitsRaw,
                        samplingConfig.getTemperature(),
                        samplingConfig.getTopK(),
                        samplingConfig.getTopP(),
                        samplingConfig.getSeed() != null ? samplingConfig.getSeed() : 0L);
            }
            INDArray tokenResult = Nd4j.getExecutioner().exec(tokenSampleOp)[0];
            int nextTokenId = tokenResult.getInt(0);
            long tSampArgmax = System.nanoTime();
            generatedTokens.add(nextTokenId);

            long stepElapsedNs = System.nanoTime() - stepStart;
            long stepElapsedMs = stepElapsedNs / 1_000_000;

            // Log sampling sub-timings
            if (step < 6 || step % 10 == 0) {
                log.info("  [SAMP] step={} total={}ms (native token_sample, no view/dup)",
                        step,
                        (tSampArgmax - tSampStart) / 1_000_000);
            }

            // Accumulate detailed timing for steps 3+
            if (step >= 3) {
                totalInputBuildNs += tAfterInputBuild - stepStart;
                totalDecoderNs    += tAfterDecoder - tAfterInputBuild;
                totalLogitsDupNs  += tAfterLogitsDup - tAfterDecoder;
                totalKvUpdateNs   += tAfterKvUpdate - tAfterLogitsDup;
                totalSamplingNs   += stepElapsedNs - (tAfterKvUpdate - stepStart);
                detailSteps++;
            }

            if (step == 0) {
                prefillTimeMs = stepElapsedMs;
                log.info("  Step 0 (prefill): {}ms (seq_len={})", stepElapsedMs, currentSeqLen);
            } else {
                stepTimesMs.add(stepElapsedMs);
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);

            // Log every 10 steps or first 6
            if (step < 6 || step % 10 == 0) {
                double currentTokPerSec = step > 0 && stepElapsedMs > 0 ? 1000.0 / stepElapsedMs : 0;
                if (step >= 2) {
                    long inputMs  = (tAfterInputBuild - stepStart) / 1_000_000;
                    long decMs    = (tAfterDecoder - tAfterInputBuild) / 1_000_000;
                    long dupMs    = (tAfterLogitsDup - tAfterDecoder) / 1_000_000;
                    long kvMs     = (tAfterKvUpdate - tAfterLogitsDup) / 1_000_000;
                    long sampMs   = stepElapsedMs - (tAfterKvUpdate - stepStart) / 1_000_000;
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s) [input={}ms dec={}ms dup={}ms kv={}ms samp={}ms cachePos={}]",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec),
                            inputMs, decMs, dupMs, kvMs, sampMs, cachePos - 1);
                } else {
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s)",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec));
                }
            }

            // Check stop tokens
            if (stopTokenIds.contains(nextTokenId)) {
                log.info("  Stop token at step {}", step);
                finishReason = GenerationResult.FinishReason.EOS;
                // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
                if (logitsRaw.closeable()) {
                    logitsRaw.close();
                }
                break;
            }

            // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
            if (logitsRaw.closeable()) {
                logitsRaw.close();
            }

            // Clean up per-step inputs (masks, position_ids — NOT embeddings, input_ids, static KV, or reusable)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                if (name.startsWith("past_key_values.")) continue;
                // Skip arrays managed by the reusable inputs cache
                if (reusableInputs.containsValue(arr)) continue;
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            decoder.clearPlaceholders(false);

            // Get embedding for next token
            INDArray prevEmbeddings = currentEmbeddings;
            if (embeddingTable != null) {
                // Direct lookup: single row from embedding table, reshaped to [1, 1, hiddenSize]
                currentEmbeddings = embeddingTable.getRow(nextTokenId).reshape(1, 1, resolvedHiddenSize);
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
            } else {
                // Fallback: full SameDiff execution
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                Map<String, INDArray> newEmbedOutputs = embedTokens.output(
                        Map.of(resolvedEmbedInputName, newTokenTensor), resolvedEmbedOutputNames);
                for (var entry : newEmbedOutputs.entrySet()) {
                    currentEmbeddings = entry.getValue();
                }
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
                embedTokens.clearPlaceholders(false);
            }
            if (prevEmbeddings != null && !prevEmbeddings.wasClosed()) {
                prevEmbeddings.setCloseable(true);
                prevEmbeddings.close();
            }
            pastSeqLen += currentSeqLen;
        }

        // Release reusable input arrays
        for (INDArray arr : reusableInputs.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        reusableInputs.clear();

        // Release static KV buffers
        if (staticKvBuffers != null) {
            for (INDArray buf : staticKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) { buf.setCloseable(true); buf.close(); }
            }
        }

        long totalDecodeMs = System.currentTimeMillis() - decodeStart;

        // Log phase breakdown
        if (detailSteps > 0) {
            log.info("=== PHASE BREAKDOWN (avg over {} decode steps, fast path steps 3+) ===", detailSteps);
            log.info("  Input build:  {}ms", totalInputBuildNs / detailSteps / 1_000_000);
            log.info("  Decoder exec: {}ms", totalDecoderNs    / detailSteps / 1_000_000);
            log.info("  Logits dup:   {}ms", totalLogitsDupNs  / detailSteps / 1_000_000);
            log.info("  KV update:    {}ms", totalKvUpdateNs   / detailSteps / 1_000_000);
            log.info("  Sampling:     {}ms", totalSamplingNs   / detailSteps / 1_000_000);
            log.info("  Sum:          {}ms", (totalInputBuildNs + totalDecoderNs + totalLogitsDupNs + totalKvUpdateNs + totalSamplingNs) / detailSteps / 1_000_000);
        }

        // Build result
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        int decodeTokens = stepTimesMs.size();
        double avgDecodeMs = decodeTokens > 0
                ? stepTimesMs.stream().mapToLong(Long::longValue).average().orElse(0) : 0;
        double decodeTokensPerSec = avgDecodeMs > 0 ? 1000.0 / avgDecodeMs : 0;
        long p50Ms = decodeTokens > 0 ? percentile(stepTimesMs, 50) : 0;
        long p90Ms = decodeTokens > 0 ? percentile(stepTimesMs, 90) : 0;
        long p99Ms = decodeTokens > 0 ? percentile(stepTimesMs, 99) : 0;

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("PERFORMANCE SUMMARY:");
        log.info("  Prefill (step 0):  {}ms", prefillTimeMs);
        log.info("  Decode tokens:     {} (excluding prefill)", decodeTokens);
        log.info("  Avg decode time:   {}ms/token", String.format("%.1f", avgDecodeMs));
        log.info("  Decode throughput: {} tok/s", String.format("%.2f", decodeTokensPerSec));
        log.info("  Latency P50/P90/P99: {}ms / {}ms / {}ms", p50Ms, p90Ms, p99Ms);
        log.info("  Total decode time: {}ms ({} tokens)", totalDecodeMs, generatedTokens.size());
        log.info("========================================");

        if (finishReason == GenerationResult.FinishReason.EOS) {
            return GenerationResult.eos(generatedText, tokenIds, promptTokenIds.length,
                    prefillTimeMs, totalDecodeMs);
        } else {
            return GenerationResult.maxTokens(generatedText, tokenIds, promptTokenIds.length,
                    prefillTimeMs, totalDecodeMs);
        }
    }

    private void logDiagnostics(int step, Map<String, INDArray> decoderInputMap) {
        for (var entry : decoderInputMap.entrySet()) {
            INDArray v = entry.getValue();
            String name = entry.getKey();
            if (name.equals("_causal_mask")) {
                log.info("  [DIAG] step={} {}: shape={} min={} max={} nonzero={}",
                        step, name, Arrays.toString(v.shape()),
                        v.minNumber().floatValue(), v.maxNumber().floatValue(),
                        v.neq(0).sumNumber().longValue());
                long len = v.length();
                INDArray flat = v.reshape(len);
                StringBuilder sb = new StringBuilder("  [DIAG]   values[0..4]=");
                for (int i = 0; i < Math.min(5, len); i++) sb.append(flat.getFloat(i)).append(",");
                sb.append(" ... values[").append(len - 5).append("..").append(len - 1).append("]=");
                for (long i = Math.max(0, len - 5); i < len; i++) sb.append(flat.getFloat(i)).append(",");
                log.info(sb.toString());
            } else if (name.equals("attention_mask")) {
                log.info("  [DIAG] step={} {}: shape={} sum={}",
                        step, name, Arrays.toString(v.shape()), v.sumNumber().longValue());
            } else if (name.startsWith("past_key_values.") && name.contains(".key.0")) {
                log.info("  [DIAG] step={} {}: shape={} absMax={}",
                        step, name, Arrays.toString(v.shape()), v.amaxNumber().floatValue());
            } else if (name.equals("position_ids")) {
                log.info("  [DIAG] step={} {}: shape={} values={}",
                        step, name, Arrays.toString(v.shape()), v);
            }
        }
    }

    private void logPresentKvDiagnostics(int step, Map<String, INDArray> decoderOutputs,
                                          DecoderUtils.KVCacheNames kvNames) {
        for (String pn : kvNames.keyNames) {
            if (pn.contains(".0")) {
                INDArray pv = decoderOutputs.get(pn);
                if (pv != null) {
                    log.info("  [DIAG] step={} present {}: shape={} lastPosAbsMax={}",
                            step, pn, Arrays.toString(pv.shape()),
                            pv.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                    NDArrayIndex.point(pv.shape()[2] - 1), NDArrayIndex.all()).amaxNumber().floatValue());
                }
            }
        }
    }

    private void logLogitsDiagnostics(int step, INDArray logits) {
        INDArray lastLogitsDiag = logits.rank() == 3
                ? logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                : logits.getRow(0);
        INDArray topK = Nd4j.argMax(lastLogitsDiag);
        int topId = topK.getInt(0);
        float topVal = lastLogitsDiag.getFloat(topId);
        float logitsMin = lastLogitsDiag.minNumber().floatValue();
        float logitsMean = lastLogitsDiag.meanNumber().floatValue();
        log.info("  [DIAG] step={} logits: topId={} topVal={} min={} mean={} shape={}",
                step, topId, topVal, logitsMin, logitsMean,
                Arrays.toString(lastLogitsDiag.shape()));
    }

    private static long percentile(List<Long> values, int percentile) {
        if (values.isEmpty()) return 0;
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(idx, sorted.size() - 1)));
    }
}
