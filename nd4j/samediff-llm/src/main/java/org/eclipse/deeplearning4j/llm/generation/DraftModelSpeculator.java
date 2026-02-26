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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Draft-model speculator that runs a smaller SameDiff model to produce speculative tokens.
 *
 * <p>This implements the classic draft-target speculative decoding approach from
 * <em>Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)</em>.
 * A smaller "draft" model (e.g., 135M params) generates K candidate tokens autoregressively,
 * then the larger "target" model verifies all K tokens in a single forward pass.</p>
 *
 * <p>The draft model runs K sequential steps (each cheap due to small model size),
 * and the target model runs 1 step with K+1 positions. When acceptance rate is high,
 * the effective throughput approaches {@code (K+1) / (draftTime + targetTime)} tokens per
 * unit time, versus {@code 1 / targetTime} without speculation.</p>
 *
 * <h3>Architecture</h3>
 * <pre>
 * Draft model (small, fast):
 *   for k = 1..K:
 *     token[k] = argmax(draftModel.forward(token[k-1]))
 *
 * Target model (large, accurate):
 *   logits[0..K] = targetModel.forward([token[0], ..., token[K]])
 *   accepted = verify(token[1..K], logits[0..K-1])
 * </pre>
 *
 * <h3>KV Cache Management</h3>
 * <p>The draft model maintains its own KV cache, separate from the target model.
 * On each speculation round, the draft KV cache is rolled back to the last verified
 * position to avoid accumulating unverified state.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see Speculator
 * @see TreeAttentionVerifier
 */
@Slf4j
public class DraftModelSpeculator implements Speculator {

    @Getter
    private final String name;

    @Getter
    private final int maxSpeculativeTokens;

    /** The draft SameDiff model (smaller than target). */
    private final SameDiff draftModel;

    /** Maps token IDs to embeddings: int[1] -> INDArray[1, 1, hiddenSize]. */
    private final Function<int[], INDArray> embedFunction;

    /** Extracts the greedy next-token from logits: INDArray[1, 1, vocab] -> int. */
    private final Function<INDArray, Integer> decodeFunction;

    /** Draft model decoder input variable names. */
    private final List<String> decoderInputNames;

    /** Logits output name in the draft model. */
    private final String logitsOutputName;

    /** KV cache output names for the draft model. */
    private final DecoderUtils.KVCacheNames kvCacheNames;

    /** Hidden size of the draft model. */
    private final long hiddenSize;

    /** Current draft model KV cache state. */
    private Map<String, INDArray> draftKvCache;

    /** Draft model past sequence length (rolled back after each speculation round). */
    private long draftPastSeqLen;

    /** Saved KV cache at the last verified position (for rollback). */
    private Map<String, INDArray> checkpointKvCache;
    private long checkpointPastSeqLen;

    // Statistics
    private long totalDraftSteps;
    private long totalDraftTimeMs;

    /**
     * Create a draft-model speculator.
     *
     * @param name               human-readable name (e.g., "draft-smollm2-135m")
     * @param draftModel         the draft SameDiff model
     * @param embedFunction      maps token ID array to embeddings [1, seqLen, hiddenSize]
     * @param decodeFunction     extracts greedy token from logits [1, 1, vocabSize]
     * @param decoderInputNames  input variable names of the draft decoder
     * @param logitsOutputName   logits output variable name
     * @param kvCacheNames       KV cache output name pairs
     * @param hiddenSize         hidden dimension of the draft model
     * @param maxSpeculativeTokens maximum tokens to speculate per round (K)
     */
    public DraftModelSpeculator(
            String name,
            SameDiff draftModel,
            Function<int[], INDArray> embedFunction,
            Function<INDArray, Integer> decodeFunction,
            List<String> decoderInputNames,
            String logitsOutputName,
            DecoderUtils.KVCacheNames kvCacheNames,
            long hiddenSize,
            int maxSpeculativeTokens) {

        if (maxSpeculativeTokens < 1) {
            throw new IllegalArgumentException("maxSpeculativeTokens must be >= 1, got " + maxSpeculativeTokens);
        }

        this.name = name;
        this.draftModel = draftModel;
        this.embedFunction = embedFunction;
        this.decodeFunction = decodeFunction;
        this.decoderInputNames = decoderInputNames;
        this.logitsOutputName = logitsOutputName;
        this.kvCacheNames = kvCacheNames;
        this.hiddenSize = hiddenSize;
        this.maxSpeculativeTokens = maxSpeculativeTokens;
        this.draftKvCache = new HashMap<>();
        this.draftPastSeqLen = 0;
    }

    @Override
    public int[] speculate(int[] generatedTokens) {
        if (generatedTokens == null || generatedTokens.length == 0) {
            return new int[0];
        }

        // Rollback to checkpoint if we have one (previous round's speculation may have been partial)
        rollbackToCheckpoint();

        int lastToken = generatedTokens[generatedTokens.length - 1];
        int[] speculated = new int[maxSpeculativeTokens];
        int count = 0;

        long roundStartMs = System.currentTimeMillis();

        // Run draft model autoregressively for K steps
        int currentToken = lastToken;
        for (int k = 0; k < maxSpeculativeTokens; k++) {
            try {
                int nextToken = draftModelStep(currentToken);
                speculated[count++] = nextToken;
                currentToken = nextToken;
            } catch (Exception e) {
                log.warn("Draft model step {} failed: {}", k, e.getMessage());
                break;
            }
        }

        long elapsed = System.currentTimeMillis() - roundStartMs;
        totalDraftSteps += count;
        totalDraftTimeMs += elapsed;

        if (count == 0) {
            return new int[0];
        }

        // Trim to actual count
        if (count < maxSpeculativeTokens) {
            int[] trimmed = new int[count];
            System.arraycopy(speculated, 0, trimmed, 0, count);
            return trimmed;
        }

        return speculated;
    }

    @Override
    public SpeculationResult verify(int[] speculativeTokens, INDArray logitsPerPosition) {
        if (speculativeTokens == null || speculativeTokens.length == 0) {
            // No speculation -- just decode from logits
            int token = argmax(logitsPerPosition, 0);
            return new SpeculationResult(0, token);
        }

        // Verify each speculative token against the target model's greedy prediction
        int accepted = 0;
        for (int i = 0; i < speculativeTokens.length; i++) {
            int modelToken = argmax(logitsPerPosition, i);
            if (modelToken == speculativeTokens[i]) {
                accepted++;
            } else {
                break;
            }
        }

        // Correction token is the target model's prediction at the first rejected position
        int correctionToken = argmax(logitsPerPosition, accepted);

        // Rollback draft KV cache to the verified position
        // accepted tokens were correct, so draft KV cache up to checkpoint + accepted is valid
        // We'll trim on next speculate() call via rollbackToCheckpoint
        saveDraftCheckpoint(accepted);

        return new SpeculationResult(accepted, correctionToken);
    }

    /**
     * Synchronize the draft model's KV cache with the target model's verified state.
     *
     * <p>Call this after each verification round to ensure the draft model's context
     * matches the target model's accepted prefix. This is important when the target
     * model rejects speculative tokens -- the draft KV cache must be rolled back.</p>
     *
     * @param verifiedSeqLen the total sequence length accepted by the target model
     */
    public void syncToVerifiedPosition(long verifiedSeqLen) {
        this.checkpointPastSeqLen = verifiedSeqLen;
        // KV cache trimming happens lazily in rollbackToCheckpoint
    }

    /**
     * Reset the draft model's KV cache state entirely.
     * Call when starting a new generation sequence.
     */
    public void reset() {
        closeKvCache(draftKvCache);
        closeKvCache(checkpointKvCache);
        draftKvCache = new HashMap<>();
        checkpointKvCache = null;
        draftPastSeqLen = 0;
        checkpointPastSeqLen = 0;
    }

    /**
     * Get draft model performance statistics.
     *
     * @return formatted statistics string
     */
    public String getStats() {
        double avgStepMs = totalDraftSteps > 0 ? (double) totalDraftTimeMs / totalDraftSteps : 0;
        return String.format("DraftModelSpeculator[%s]: steps=%d, totalTime=%dms, avgStep=%.1fms",
                name, totalDraftSteps, totalDraftTimeMs, avgStepMs);
    }

    // ========== Internal Methods ==========

    /**
     * Run one autoregressive step of the draft model.
     *
     * @param tokenId the input token ID
     * @return the predicted next token ID (greedy)
     */
    private int draftModelStep(int tokenId) {
        // Embed the token
        INDArray embeddings = embedFunction.apply(new int[]{tokenId});

        // Build decoder input map
        long currentSeqLen = 1;
        long totalSeqLen = currentSeqLen + draftPastSeqLen;

        Map<String, INDArray> inputMap = new HashMap<>();
        for (String inputName : decoderInputNames) {
            if (inputName.equals("inputs_embeds")) {
                inputMap.put(inputName, embeddings);
            } else if (inputName.equals("attention_mask")) {
                inputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
            } else if (inputName.equals("_causal_mask")) {
                inputMap.put(inputName, DecoderUtils.buildCausalMask(1, currentSeqLen, totalSeqLen));
            } else if (inputName.equals("position_ids")) {
                inputMap.put(inputName, Nd4j.createFromArray(new long[]{draftPastSeqLen}).reshape(1, 1));
            } else if (inputName.startsWith("past_key_values.")) {
                String presentName = inputName.replace("past_key_values", "present");
                if (draftKvCache.containsKey(presentName)) {
                    inputMap.put(inputName, draftKvCache.get(presentName));
                } else {
                    inputMap.put(inputName, DecoderUtils.createEmptyKvCache(draftModel, inputName, 1, hiddenSize));
                }
            }
        }

        // Build output names: logits + KV cache
        java.util.List<String> outputNames = new java.util.ArrayList<>();
        outputNames.add(logitsOutputName);
        outputNames.addAll(kvCacheNames.keyNames);
        outputNames.addAll(kvCacheNames.valueNames);

        // Run draft model
        Map<String, INDArray> outputs = draftModel.output(inputMap, outputNames.toArray(new String[0]));

        // Extract logits and decode
        INDArray logits = outputs.get(logitsOutputName);
        int nextToken = decodeFunction.apply(logits);

        // Update draft KV cache
        Map<String, INDArray> newKvCache = new HashMap<>();
        for (String keyName : kvCacheNames.keyNames) {
            INDArray kv = outputs.get(keyName);
            if (kv != null) {
                newKvCache.put(keyName, kv.dup());
            }
        }
        for (String valName : kvCacheNames.valueNames) {
            INDArray kv = outputs.get(valName);
            if (kv != null) {
                newKvCache.put(valName, kv.dup());
            }
        }

        closeKvCache(draftKvCache);
        draftKvCache = newKvCache;
        draftPastSeqLen++;

        // Clean up non-KV outputs
        draftModel.clearPlaceholders(false);

        return nextToken;
    }

    /**
     * Save the current draft KV cache as a checkpoint at the verified position.
     */
    private void saveDraftCheckpoint(int acceptedCount) {
        closeKvCache(checkpointKvCache);

        if (acceptedCount > 0 && !draftKvCache.isEmpty()) {
            checkpointKvCache = new HashMap<>();
            long keepLen = checkpointPastSeqLen + acceptedCount;

            for (Map.Entry<String, INDArray> entry : draftKvCache.entrySet()) {
                INDArray kv = entry.getValue();
                if (kv != null && kv.rank() >= 3) {
                    long kvSeqLen = kv.size(2);
                    if (keepLen < kvSeqLen) {
                        checkpointKvCache.put(entry.getKey(), kv.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, keepLen), NDArrayIndex.all()
                        ).dup());
                    } else {
                        checkpointKvCache.put(entry.getKey(), kv.dup());
                    }
                }
            }
            checkpointPastSeqLen = keepLen;
        } else {
            checkpointKvCache = null;
        }
    }

    /**
     * Roll back the draft model's state to the last checkpoint.
     */
    private void rollbackToCheckpoint() {
        if (checkpointKvCache != null) {
            closeKvCache(draftKvCache);
            draftKvCache = checkpointKvCache;
            draftPastSeqLen = checkpointPastSeqLen;
            checkpointKvCache = null;
        }
    }

    /**
     * Extract argmax token from logits at a given position.
     *
     * @param logits logits tensor, shape [positions, vocabSize] or [1, positions, vocabSize]
     * @param position the position index
     * @return the token ID with the highest logit
     */
    private static int argmax(INDArray logits, int position) {
        INDArray posLogits;
        if (logits.rank() == 3) {
            posLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(position), NDArrayIndex.all());
        } else if (logits.rank() == 2) {
            posLogits = logits.get(NDArrayIndex.point(position), NDArrayIndex.all());
        } else {
            posLogits = logits;
        }

        float maxVal = Float.NEGATIVE_INFINITY;
        int maxIdx = 0;
        long vocabSize = posLogits.length();
        for (int v = 0; v < vocabSize; v++) {
            float val = posLogits.getFloat(v);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = v;
            }
        }
        return maxIdx;
    }

    /**
     * Close and release all arrays in a KV cache map.
     */
    private static void closeKvCache(Map<String, INDArray> kvCache) {
        if (kvCache == null) return;
        for (INDArray arr : kvCache.values()) {
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
            }
        }
        kvCache.clear();
    }
}
