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

package org.eclipse.deeplearning4j.llm.generation.speculative;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Speculative decoding loop using n-gram matching for token prediction.
 *
 * <p>This class implements the speculative decoding optimization for autoregressive
 * generation. Instead of generating 1 token per forward pass, it:</p>
 * <ol>
 *   <li>Uses {@link NgramSpeculator} to predict K tokens from n-gram patterns</li>
 *   <li>Embeds the original token + K speculative tokens (K+1 total)</li>
 *   <li>Runs the decoder once with K+1 positions (instead of K+1 separate calls)</li>
 *   <li>Verifies speculative tokens against the model's actual predictions</li>
 *   <li>Accepts matching prefix + 1 correction token</li>
 * </ol>
 *
 * <p>For structured outputs (DocTags, HTML, JSON), this can accept 3-7 tokens per
 * forward pass instead of 1, providing 2-5x throughput improvement.</p>
 *
 * <p>When speculation fails (no n-gram match or all tokens rejected), it falls
 * back to normal single-token decoding with zero overhead.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class SpeculativeDecodeLoop {

    private final NgramSpeculator speculator;
    @Getter private long totalSpeculations;
    @Getter private long totalAccepted;
    @Getter private long totalAttempted;
    @Getter private long speculativeSteps;
    @Getter private long normalSteps;
    private long speculativeWallTimeMs;
    private long speculativeDecoderTimeMs;
    private long speculativeEmbedTimeMs;
    @Getter private boolean disabled;

    // Retry/cooldown state: replaces permanent disable on first failure
    private int consecutiveFailures;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private int cooldownRemaining;
    private static final int COOLDOWN_STEPS = 5;

    private boolean probeCompleted;

    public SpeculativeDecodeLoop(NgramSpeculator speculator) {
        this.speculator = speculator;
    }

    public SpeculativeDecodeLoop() {
        this(new NgramSpeculator());
    }

    /**
     * Probe whether the decoder model supports multi-token input (seqLen > 1).
     *
     * <p>Some models (e.g., SmolDocling) have internal ONNX Expand ops that create
     * a causal mask of shape [1,1,seqLen,seqLen]. When seqLen > 1 and pastSeqLen > 0,
     * this can't broadcast with the external attention mask [1,1,seqLen,totalSeqLen],
     * causing a shape error on every speculative attempt.</p>
     *
     * <p>This method runs a single throwaway decode with seqLen=2 and a minimal
     * KV cache (pastSeqLen=1) to detect this failure upfront. If it fails,
     * speculation is disabled immediately with zero wasted decode steps.</p>
     *
     * <p>Call this after the first successful single-token decode step, when
     * the model is warm and KV cache shape info is available.</p>
     *
     * @param decoder the decoder SameDiff model
     * @param decoderInputNames decoder input names
     * @param logitsOutputName logits output name
     * @param kvCacheNames KV cache output names
     * @param batchSize batch size (use 1 for probe)
     * @param hiddenSize model hidden size
     * @return true if the model supports multi-token decode, false if not
     */
    public boolean probeMultiTokenSupport(
            SameDiff decoder,
            List<String> decoderInputNames,
            String logitsOutputName,
            ModelIOConfig.KVCacheNames kvCacheNames,
            int batchSize,
            long hiddenSize) {

        if (probeCompleted) {
            return !disabled;
        }
        probeCompleted = true;

        // Build a minimal 2-token input with pastSeqLen=1
        // This triggers the Expand broadcast failure if the model can't handle seqLen>1
        int probeSeqLen = 2;
        long probePastSeqLen = 1;
        long probeTotalSeqLen = probeSeqLen + probePastSeqLen;

        Map<String, INDArray> probeInputs = new HashMap<>();
        for (String inputName : decoderInputNames) {
            if (inputName.equals("inputs_embeds")) {
                probeInputs.put(inputName, Nd4j.zeros(DataType.FLOAT, 1, probeSeqLen, hiddenSize));
            } else if (inputName.equals("attention_mask")) {
                probeInputs.put(inputName, Nd4j.ones(DataType.LONG, 1, probeTotalSeqLen));
            } else if (inputName.equals("_causal_mask")) {
                probeInputs.put(inputName, ModelIOConfig.buildCausalMask(1, probeSeqLen, probeTotalSeqLen));
            } else if (inputName.equals("position_ids")) {
                probeInputs.put(inputName, Nd4j.arange(probePastSeqLen, probePastSeqLen + probeSeqLen)
                        .reshape(1, probeSeqLen).castTo(DataType.LONG));
            } else if (inputName.startsWith("past_key_values.")) {
                // Create minimal KV cache with seqLen=1
                probeInputs.put(inputName, ModelIOConfig.createEmptyKvCache(
                        decoder, inputName, 1, hiddenSize));
            }
        }

        // Request only logits (minimal output)
        try {
            Map<String, INDArray> outputs = decoder.output(probeInputs, logitsOutputName);
            // Success — model supports multi-token decode
            log.info("Speculative decoding probe: model supports multi-token decode");

            // Clean up probe outputs
            for (INDArray arr : outputs.values()) {
                SameDiffMemoryUtils.safeClose(arr);
            }
            return true;
        } catch (Exception e) {
            // Failed — model can't handle seqLen > 1 with KV cache
            log.info("Speculative decoding probe: model does NOT support multi-token decode, " +
                    "disabling speculation. Reason: {}", e.getMessage());
            disabled = true;
            return false;
        } finally {
            // Clean up probe inputs
            for (var entry : probeInputs.entrySet()) {
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
            decoder.clearPlaceholders(false);
            // Reset the session to clear any stale shape caches and DynamicShapePlanExecutor
            // state from the probe execution. The next output() call recreates a fresh session.
            decoder.resetSession();
        }
    }

    /**
     * Execute one speculative decode step for a single sequence.
     *
     * <p>This is the core method called from the decode loop. It either performs
     * a speculative step (if n-gram match found) or a normal single-token step.</p>
     *
     * @param generatedTokens tokens generated so far
     * @param lastTokenId the last generated token (current position)
     * @param embedTokens the embedding model
     * @param embedInputName embedding model input name
     * @param embedOutputNames embedding model output names
     * @param decoder the decoder model
     * @param decoderInputNames decoder input names
     * @param logitsOutputName logits output name
     * @param kvCacheNames KV cache output names
     * @param kvCache current KV cache state
     * @param pastSeqLen current past sequence length
     * @param batchSize batch size
     * @param hiddenSize model hidden size
     * @param eosTokenIds set of tokens that terminate generation
     * @return result containing accepted tokens and updated state
     */
    public SpeculativeStepResult step(
            List<Integer> generatedTokens,
            int lastTokenId,
            SameDiff embedTokens,
            String embedInputName,
            String[] embedOutputNames,
            SameDiff decoder,
            List<String> decoderInputNames,
            String logitsOutputName,
            ModelIOConfig.KVCacheNames kvCacheNames,
            Map<String, INDArray> kvCache,
            long pastSeqLen,
            int batchSize,
            long hiddenSize,
            Set<Integer> eosTokenIds) {

        if (disabled) {
            normalSteps++;
            return null;
        }

        if (cooldownRemaining > 0) {
            cooldownRemaining--;
            normalSteps++;
            return null;
        }

        // Try to speculate
        int[] specTokens = speculator.speculate(generatedTokens);

        if (specTokens.length == 0) {
            // No speculation possible — normal single-token step
            normalSteps++;
            return null; // Caller should do normal decode
        }

        speculativeSteps++;
        totalAttempted += specTokens.length;
        long specStepStartMs = System.currentTimeMillis();
        long embedTimeMs = 0L;
        long decoderTimeMs = 0L;

        // Build token sequence: [lastTokenId, spec0, spec1, ..., specK-1]
        int seqLen = 1 + specTokens.length;
        int[] tokenSequence = new int[seqLen];
        tokenSequence[0] = lastTokenId;
        System.arraycopy(specTokens, 0, tokenSequence, 1, specTokens.length);

        // Embed all tokens at once
        long embedStartMs = System.currentTimeMillis();
        INDArray tokenTensor = Nd4j.createFromArray(tokenSequence)
                .reshape(1, seqLen).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, tokenTensor), embedOutputNames);
        INDArray embeddings = embedOutputs.values().iterator().next().dup();
        embedTimeMs = System.currentTimeMillis() - embedStartMs;

        // Tile for batch if needed
        if (batchSize > 1) {
            embeddings = Nd4j.tile(embeddings, batchSize, 1, 1);
        }

        // Build decoder inputs
        long currentSeqLen = seqLen;
        long totalSeqLen = currentSeqLen + pastSeqLen;

        Map<String, INDArray> decoderInputMap = new HashMap<>();
        for (String inputName : decoderInputNames) {
            if (inputName.equals("inputs_embeds")) {
                decoderInputMap.put(inputName, embeddings);
            } else if (inputName.equals("attention_mask")) {
                decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSeqLen));
            } else if (inputName.equals("_causal_mask")) {
                decoderInputMap.put(inputName, ModelIOConfig.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
            } else if (inputName.equals("position_ids")) {
                INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                        .reshape(1, currentSeqLen).castTo(DataType.LONG);
                decoderInputMap.put(inputName, batchSize > 1 ? Nd4j.tile(posIds, batchSize, 1) : posIds);
            } else if (inputName.startsWith("past_key_values.")) {
                String presentName = inputName.replace("past_key_values", "present");
                if (kvCache.containsKey(presentName)) {
                    decoderInputMap.put(inputName, kvCache.get(presentName));
                } else {
                    decoderInputMap.put(inputName, ModelIOConfig.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize));
                }
            }
        }

        // Run decoder once for all positions
        java.util.List<String> allOutputs = new java.util.ArrayList<>();
        allOutputs.add(logitsOutputName);
        allOutputs.addAll(kvCacheNames.keyNames);
        allOutputs.addAll(kvCacheNames.valueNames);

        Map<String, INDArray> decoderOutputs;
        try {
            long decoderStartMs = System.currentTimeMillis();
            decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));
            decoderTimeMs = System.currentTimeMillis() - decoderStartMs;
        } catch (Exception e) {
            // Multi-token decode failed. Use retry/cooldown instead of permanent disable.
            consecutiveFailures++;
            if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                log.warn("Speculative decoding permanently disabled after {} consecutive failures: {}",
                        consecutiveFailures, e.getMessage());
                disabled = true;
            } else {
                log.warn("Speculative decoding failed (attempt {}/{}), cooldown {} steps: {}",
                        consecutiveFailures, MAX_CONSECUTIVE_FAILURES, COOLDOWN_STEPS, e.getMessage());
                cooldownRemaining = COOLDOWN_STEPS;
            }
            // Clean up inputs we created
            SameDiffMemoryUtils.safeClose(tokenTensor);
            embedTokens.clearPlaceholders(false);
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
            SameDiffMemoryUtils.safeClose(embeddings);
            decoder.clearPlaceholders(false);
            recordSpecTiming(specStepStartMs, embedTimeMs, decoderTimeMs);
            return null;
        }

        INDArray logitsRaw = decoderOutputs.get(logitsOutputName);

        if (logitsRaw == null) {
            log.warn("Speculative step produced no logits");
            recordSpecTiming(specStepStartMs, embedTimeMs, decoderTimeMs);
            return null;
        }

        // logits shape: [batchSize, seqLen, vocabSize] or [batchSize, vocabSize]
        // We need logits for each position to verify speculation
        // For batch=1: extract per-position logits and verify
        int vocabSize;
        float[][] logitsPerPosition;

        if (logitsRaw.rank() == 3) {
            vocabSize = (int) logitsRaw.size(2);
            logitsPerPosition = new float[seqLen][];
            for (int p = 0; p < seqLen; p++) {
                logitsPerPosition[p] = logitsRaw.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(p),
                        NDArrayIndex.all()
                ).toFloatVector();
            }
        } else {
            // Rank 2: only last position logits available, can't verify
            log.debug("Speculative step got rank-2 logits, falling back");
            recordSpecTiming(specStepStartMs, embedTimeMs, decoderTimeMs);
            return null;
        }

        // Verify speculation: position 0 predicts token after lastTokenId,
        // which should match specTokens[0]. Position i predicts token after
        // seeing specTokens[0..i-1], which should match specTokens[i].
        int accepted = NgramSpeculator.verifySpeculation(specTokens, logitsPerPosition);
        totalAccepted += accepted;

        // The correction token is at logitsPerPosition[accepted]
        int correctionToken = NgramSpeculator.getCorrectionToken(logitsPerPosition[accepted]);

        // Build result: accepted speculative tokens + correction token
        int[] acceptedTokens = new int[accepted + 1];
        System.arraycopy(specTokens, 0, acceptedTokens, 0, accepted);
        acceptedTokens[accepted] = correctionToken;

        // Check for EOS in accepted tokens
        int eosIdx = -1;
        for (int i = 0; i < acceptedTokens.length; i++) {
            if (eosTokenIds.contains(acceptedTokens[i])) {
                eosIdx = i;
                break;
            }
        }
        if (eosIdx >= 0) {
            // Truncate at EOS
            int[] truncated = new int[eosIdx + 1];
            System.arraycopy(acceptedTokens, 0, truncated, 0, eosIdx + 1);
            acceptedTokens = truncated;
        }

        // We need to trim the KV cache to only include positions up to
        // pastSeqLen + accepted + 1 (the accepted tokens + correction).
        // The decoder produced KV cache for all seqLen positions, but we
        // only want to keep the first (accepted + 1) new positions.
        int validNewPositions = acceptedTokens.length;
        Map<String, INDArray> trimmedKvCache = new HashMap<>();

        for (String keyName : kvCacheNames.keyNames) {
            INDArray kv = decoderOutputs.get(keyName);
            if (kv != null) {
                // KV shape: [batch, numHeads, totalSeqLen, headDim]
                long kvSeqLen = kv.size(2);
                long keepLen = pastSeqLen + validNewPositions;
                if (keepLen < kvSeqLen) {
                    trimmedKvCache.put(keyName, kv.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, keepLen), NDArrayIndex.all()
                    ).dup());
                } else {
                    trimmedKvCache.put(keyName, kv.dup());
                }
            }
        }
        for (String valName : kvCacheNames.valueNames) {
            INDArray kv = decoderOutputs.get(valName);
            if (kv != null) {
                long kvSeqLen = kv.size(2);
                long keepLen = pastSeqLen + validNewPositions;
                if (keepLen < kvSeqLen) {
                    trimmedKvCache.put(valName, kv.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, keepLen), NDArrayIndex.all()
                    ).dup());
                } else {
                    trimmedKvCache.put(valName, kv.dup());
                }
            }
        }

        // Cleanup
        SameDiffMemoryUtils.safeClose(tokenTensor);
        embedTokens.clearPlaceholders(false);

        for (var entry : decoderInputMap.entrySet()) {
            String name = entry.getKey();
            if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
            SameDiffMemoryUtils.safeClose(entry.getValue());
        }
        SameDiffMemoryUtils.safeClose(embeddings);
        decoder.clearPlaceholders(false);

        // Reset failure counter on successful speculation
        consecutiveFailures = 0;

        log.debug("Speculative step: proposed={}, accepted={}, correction={}",
                specTokens.length, accepted, correctionToken);
        recordSpecTiming(specStepStartMs, embedTimeMs, decoderTimeMs);

        return new SpeculativeStepResult(
                acceptedTokens,
                trimmedKvCache,
                validNewPositions,
                eosIdx >= 0
        );
    }

    /**
     * Get speculation statistics.
     */
    public String getStats() {
        double acceptRate = totalAttempted > 0 ? (double) totalAccepted / totalAttempted * 100 : 0;
        double avgAccepted = speculativeSteps > 0 ? (double) totalAccepted / speculativeSteps : 0;
        return String.format("speculative=%d, normal=%d, attempted=%d, accepted=%d (%.1f%%), avg=%.1f tokens/spec",
                speculativeSteps, normalSteps, totalAttempted, totalAccepted, acceptRate, avgAccepted);
    }

    public String getTimingStats() {
        if (speculativeSteps == 0) {
            return "speculative_steps=0";
        }
        long hostOverheadMs = Math.max(0L, speculativeWallTimeMs - speculativeDecoderTimeMs);
        long avgWallMs = speculativeWallTimeMs / speculativeSteps;
        long avgDecoderMs = speculativeDecoderTimeMs / speculativeSteps;
        long avgEmbedMs = speculativeEmbedTimeMs / speculativeSteps;
        long avgHostOverheadMs = hostOverheadMs / speculativeSteps;
        return String.format(
                "speculative_steps=%d, wall=%dms (avg=%dms), decoder=%dms (avg=%dms), embed=%dms (avg=%dms), host_overhead=%dms (avg=%dms)",
                speculativeSteps, speculativeWallTimeMs, avgWallMs,
                speculativeDecoderTimeMs, avgDecoderMs,
                speculativeEmbedTimeMs, avgEmbedMs,
                hostOverheadMs, avgHostOverheadMs);
    }

    private void recordSpecTiming(long stepStartMs, long embedMs, long decoderMs) {
        long wallMs = Math.max(0L, System.currentTimeMillis() - stepStartMs);
        speculativeWallTimeMs += wallMs;
        speculativeEmbedTimeMs += Math.max(0L, embedMs);
        speculativeDecoderTimeMs += Math.max(0L, decoderMs);
    }

    /**
     * Result of a speculative decode step.
     */
    public static class SpeculativeStepResult {
        @Getter private final int[] acceptedTokens;
        @Getter private final Map<String, INDArray> updatedKvCache;
        @Getter private final int newPositions;
        private final boolean hitEos;

        public SpeculativeStepResult(int[] acceptedTokens, Map<String, INDArray> updatedKvCache,
                                     int newPositions, boolean hitEos) {
            this.acceptedTokens = acceptedTokens;
            this.updatedKvCache = updatedKvCache;
            this.newPositions = newPositions;
            this.hitEos = hitEos;
        }

        public boolean hitEos() { return hitEos; }
    }
}
