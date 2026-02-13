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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
    private long totalSpeculations;
    private long totalAccepted;
    private long totalAttempted;
    private long speculativeSteps;
    private long normalSteps;
    private boolean disabled;

    public SpeculativeDecodeLoop(NgramSpeculator speculator) {
        this.speculator = speculator;
    }

    public SpeculativeDecodeLoop() {
        this(new NgramSpeculator());
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
            DecoderUtils.KVCacheNames kvCacheNames,
            Map<String, INDArray> kvCache,
            long pastSeqLen,
            int batchSize,
            long hiddenSize,
            Set<Integer> eosTokenIds) {

        if (disabled) {
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

        // Build token sequence: [lastTokenId, spec0, spec1, ..., specK-1]
        int seqLen = 1 + specTokens.length;
        int[] tokenSequence = new int[seqLen];
        tokenSequence[0] = lastTokenId;
        System.arraycopy(specTokens, 0, tokenSequence, 1, specTokens.length);

        // Embed all tokens at once
        INDArray tokenTensor = Nd4j.createFromArray(tokenSequence)
                .reshape(1, seqLen).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, tokenTensor), embedOutputNames);
        INDArray embeddings = embedOutputs.values().iterator().next().dup();

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
                decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
            } else if (inputName.equals("position_ids")) {
                INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                        .reshape(1, currentSeqLen).castTo(DataType.LONG);
                decoderInputMap.put(inputName, batchSize > 1 ? Nd4j.tile(posIds, batchSize, 1) : posIds);
            } else if (inputName.startsWith("past_key_values.")) {
                String presentName = inputName.replace("past_key_values", "present");
                if (kvCache.containsKey(presentName)) {
                    decoderInputMap.put(inputName, kvCache.get(presentName));
                } else {
                    decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize));
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
            decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));
        } catch (Exception e) {
            // Model doesn't support multi-token KV-cache decode (e.g. ONNX Expand op
            // creates [seqLen, seqLen] causal mask that can't broadcast to [seqLen, totalSeqLen]).
            // Permanently disable speculation and fall back to single-token decode.
            log.warn("Model incompatible with speculative decoding (multi-token KV-cache decode failed), disabling: {}",
                    e.getMessage());
            disabled = true;
            // Clean up inputs we created
            tokenTensor.setCloseable(true);
            tokenTensor.close();
            embedTokens.clearPlaceholders(false);
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            embeddings.setCloseable(true);
            embeddings.close();
            decoder.clearPlaceholders(false);
            return null;
        }

        INDArray logitsRaw = decoderOutputs.get(logitsOutputName);

        if (logitsRaw == null) {
            log.warn("Speculative step produced no logits");
            return null;
        }

        INDArray logits = logitsRaw.dup();

        // logits shape: [batchSize, seqLen, vocabSize] or [batchSize, vocabSize]
        // We need logits for each position to verify speculation
        // For batch=1: extract per-position logits and verify
        int vocabSize;
        float[][] logitsPerPosition;

        if (logits.rank() == 3) {
            vocabSize = (int) logits.size(2);
            logitsPerPosition = new float[seqLen][];
            for (int p = 0; p < seqLen; p++) {
                logitsPerPosition[p] = logits.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(p),
                        NDArrayIndex.all()
                ).dup().toFloatVector();
            }
        } else {
            // Rank 2: only last position logits available, can't verify
            log.debug("Speculative step got rank-2 logits, falling back");
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
        logits.close();
        tokenTensor.setCloseable(true);
        tokenTensor.close();
        embedTokens.clearPlaceholders(false);

        for (var entry : decoderInputMap.entrySet()) {
            String name = entry.getKey();
            INDArray arr = entry.getValue();
            if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
            }
        }
        embeddings.setCloseable(true);
        embeddings.close();
        decoder.clearPlaceholders(false);

        log.debug("Speculative step: proposed={}, accepted={}, correction={}",
                specTokens.length, accepted, correctionToken);

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

    public long getTotalAccepted() { return totalAccepted; }
    public long getTotalAttempted() { return totalAttempted; }
    public long getSpeculativeSteps() { return speculativeSteps; }
    public long getNormalSteps() { return normalSteps; }
    public boolean isDisabled() { return disabled; }

    /**
     * Result of a speculative decode step.
     */
    public static class SpeculativeStepResult {
        private final int[] acceptedTokens;
        private final Map<String, INDArray> updatedKvCache;
        private final int newPositions;
        private final boolean hitEos;

        public SpeculativeStepResult(int[] acceptedTokens, Map<String, INDArray> updatedKvCache,
                                     int newPositions, boolean hitEos) {
            this.acceptedTokens = acceptedTokens;
            this.updatedKvCache = updatedKvCache;
            this.newPositions = newPositions;
            this.hitEos = hitEos;
        }

        public int[] getAcceptedTokens() { return acceptedTokens; }
        public Map<String, INDArray> getUpdatedKvCache() { return updatedKvCache; }
        public int getNewPositions() { return newPositions; }
        public boolean hitEos() { return hitEos; }
    }
}
