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

package org.eclipse.deeplearning4j.model.benchmark;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages evolving decode-style inputs for multi-step replay validation.
 *
 * <h3>What this does</h3>
 * <p>During autoregressive decode, certain inputs change at each step while others
 * remain constant. This class manages the evolution of those changing inputs so
 * that validation tests can exercise the same compiled native handle across
 * multiple decode steps with realistic input changes.</p>
 *
 * <h3>Evolving Inputs</h3>
 * <ul>
 *   <li><b>position_ids</b> — Increments by 1 each decode step. Shape [1, 1] during
 *       decode (single token position). Starts at the prefill sequence length.</li>
 *   <li><b>attention_mask</b> — Grows by 1 each decode step for growing KV mode,
 *       or maintains fixed shape with sparse pattern for padded KV mode.</li>
 *   <li><b>past_key_values.*</b> — Updated each step with the present KV outputs
 *       from the previous decode step.</li>
 *   <li><b>cache position</b> — Tracks the current position in the static KV buffer
 *       for padded mode decode.</li>
 * </ul>
 *
 * <h3>Evolution Modes</h3>
 * <ul>
 *   <li>{@link EvolutionMode#POSITION_IDS_ONLY} — Only position_ids change; mask and
 *       KV stay fixed. Useful for testing position-sensitive ops in isolation.</li>
 *   <li>{@link EvolutionMode#ATTENTION_MASK_ONLY} — Only attention_mask changes; position
 *       and KV stay fixed. Tests mask reformat subgraph correctness.</li>
 *   <li>{@link EvolutionMode#KV_ONLY} — Only past_key_values change; position and mask
 *       stay fixed. Tests KV scatter/replay correctness.</li>
 *   <li>{@link EvolutionMode#ALL_TOGETHER} — All inputs change each step. Full
 *       decode-style validation — most realistic, catches all interaction bugs.</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>
 *   DecodeInputEvolutor evolutor = DecodeInputEvolutor.builder()
 *       .decoderInputs(allInputNames)
 *       .logitsName("logits")
 *       .kvOutputNames(kvNames)
 *       .prefillResult(prefillOutputs)
 *       .prefillSeqLen(seqLen)
 *       .maxDecodeSteps(maxSteps)
 *       .evolutionMode(EvolutionMode.ALL_TOGETHER)
 *       .build();
 *
 *   for (int step = 0; step &lt; maxSteps; step++) {
 *       Map&lt;String, INDArray&gt; stepInputs = evolutor.buildStepInputs(step);
 *       Map&lt;String, INDArray&gt; stepOutputs = model.output(stepInputs, outputNames);
 *       evolutor.updateFromOutputs(stepOutputs);
 *   }
 * </pre>
 */
public class DecodeInputEvolutor implements AutoCloseable {

    private final List<String> decoderInputNames;
    private final String logitsName;
    private final List<String> kvOutputNames;
    private final long prefillSeqLen;
    private final int maxDecodeSteps;
    private final EvolutionMode evolutionMode;
    private final boolean paddedMode;
    private final long maxKvLen;

    // State that evolves across steps
    private long cachePosition;
    private long currentSeqLen;
    private final Map<String, INDArray> kvBuffers = new HashMap<>();
    private INDArray positionIds;
    private INDArray attentionMask;

    private DecodeInputEvolutor(Builder builder) {
        this.decoderInputNames = builder.decoderInputNames;
        this.logitsName = builder.logitsName;
        this.kvOutputNames = builder.kvOutputNames;
        this.prefillSeqLen = builder.prefillSeqLen;
        this.maxDecodeSteps = builder.maxDecodeSteps;
        this.evolutionMode = builder.evolutionMode;
        this.paddedMode = builder.paddedMode;
        this.maxKvLen = builder.maxKvLen;
        this.cachePosition = 0;
        this.currentSeqLen = prefillSeqLen;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Initialize from prefill result. Call after prefill, before decode steps.
     *
     * @param prefillOutputs outputs from the prefill step (includes present KV)
     */
    public void initializeFromPrefill(Map<String, INDArray> prefillOutputs) {
        cachePosition = prefillSeqLen;
        currentSeqLen = prefillSeqLen;

        // Initialize KV buffers from prefill outputs
        if (paddedMode) {
            // Pad KV to max length
            for (String kvName : kvOutputNames) {
                String pastName = kvName.replace("present", "past_key_values");
                INDArray kv = prefillOutputs.get(kvName);
                if (kv == null) continue;
                long[] shape = kv.shape(); // [1, heads, seqLen, dim]
                INDArray padded = Nd4j.zeros(DataType.FLOAT, shape[0], shape[1], maxKvLen, shape[3]);
                padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, shape[2]), NDArrayIndex.all()).assign(kv);
                kvBuffers.put(pastName, padded);
            }
        } else {
            // Growing KV — just dup the prefill outputs
            for (String kvName : kvOutputNames) {
                String pastName = kvName.replace("present", "past_key_values");
                INDArray kv = prefillOutputs.get(kvName);
                if (kv != null) {
                    kvBuffers.put(pastName, kv.dup());
                }
            }
        }
    }

    /**
     * Build input map for one decode step.
     *
     * @param step zero-based decode step index (0 = first decode after prefill)
     * @return input map ready for model.output() or outputDirect()
     */
    public Map<String, INDArray> buildStepInputs(int step) {
        Map<String, INDArray> inputs = new HashMap<>();
        long stepPosition = prefillSeqLen + step;

        for (String inputName : decoderInputNames) {
            if (inputName.equals("position_ids")) {
                if (evolutionMode == EvolutionMode.POSITION_IDS_ONLY ||
                    evolutionMode == EvolutionMode.ALL_TOGETHER) {
                    // Position changes every step
                    positionIds = Nd4j.createFromArray(new long[]{stepPosition}).reshape(1, 1);
                } else if (positionIds == null) {
                    positionIds = Nd4j.createFromArray(new long[]{prefillSeqLen}).reshape(1, 1);
                }
                inputs.put(inputName, positionIds);

            } else if (inputName.equals("attention_mask")) {
                if (paddedMode) {
                    // Sparse mask in padded mode
                    if (evolutionMode == EvolutionMode.ATTENTION_MASK_ONLY ||
                        evolutionMode == EvolutionMode.ALL_TOGETHER) {
                        long totalLen = maxKvLen + 1;
                        attentionMask = Nd4j.zeros(DataType.LONG, 1, totalLen);
                        attentionMask.get(NDArrayIndex.point(0),
                                NDArrayIndex.interval(0, currentSeqLen)).assign(1);
                        attentionMask.putScalar(0, totalLen - 1, 1);
                    } else if (attentionMask == null) {
                        attentionMask = Nd4j.zeros(DataType.LONG, 1, maxKvLen + 1);
                        attentionMask.get(NDArrayIndex.point(0),
                                NDArrayIndex.interval(0, prefillSeqLen)).assign(1);
                    }
                } else {
                    // Contiguous all-ones mask in growing mode
                    long maskLen = currentSeqLen + 1;
                    attentionMask = Nd4j.ones(DataType.LONG, 1, maskLen);
                }
                inputs.put(inputName, attentionMask);

            } else if (inputName.startsWith("past_key_values.")) {
                if (evolutionMode == EvolutionMode.KV_ONLY ||
                    evolutionMode == EvolutionMode.ALL_TOGETHER) {
                    INDArray kv = kvBuffers.get(inputName);
                    if (kv != null) {
                        inputs.put(inputName, kv);
                    }
                } else {
                    // KV stays at initial state
                    INDArray kv = kvBuffers.get(inputName);
                    if (kv != null) {
                        inputs.put(inputName, kv);
                    }
                }
            }
        }

        return inputs;
    }

    /**
     * Update internal state from decode step outputs.
     * Must be called after each step to evolve KV cache.
     *
     * @param stepOutputs outputs from the decode step (includes present KV)
     */
    public void updateFromOutputs(Map<String, INDArray> stepOutputs) {
        currentSeqLen++;

        if (evolutionMode == EvolutionMode.KV_ONLY ||
            evolutionMode == EvolutionMode.ALL_TOGETHER) {
            // Update KV buffers from present KV outputs
            for (String kvName : kvOutputNames) {
                String pastName = kvName.replace("present", "past_key_values");
                INDArray presentKv = stepOutputs.get(kvName);
                if (presentKv == null) continue;

                // Close old buffer
                INDArray oldKv = kvBuffers.get(pastName);
                if (oldKv != null && !oldKv.wasClosed()) {
                    oldKv.setCloseable(true);
                    oldKv.close();
                }

                if (paddedMode) {
                    // Scatter into padded buffer at cache position
                    INDArray padded = kvBuffers.get(pastName);
                    if (padded != null) {
                        long pos = cachePosition;
                        padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(pos), NDArrayIndex.all())
                                .assign(presentKv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                        NDArrayIndex.point(0), NDArrayIndex.all()));
                    }
                } else {
                    // Growing KV — dup the present output
                    kvBuffers.put(pastName, presentKv.dup());
                }
            }
        }

        if (paddedMode) {
            cachePosition++;
        }
    }

    /**
     * Get current cache position (for padded mode).
     */
    public long getCachePosition() {
        return cachePosition;
    }

    /**
     * Get current sequence length.
     */
    public long getCurrentSeqLen() {
        return currentSeqLen;
    }

    /**
     * Get the logits output name.
     */
    public String getLogitsName() {
        return logitsName;
    }

    /**
     * Get maximum decode steps configured.
     */
    public int getMaxDecodeSteps() {
        return maxDecodeSteps;
    }

    @Override
    public void close() {
        // Close all KV buffers
        for (INDArray kv : kvBuffers.values()) {
            if (kv != null && !kv.wasClosed()) {
                kv.setCloseable(true);
                kv.close();
            }
        }
        kvBuffers.clear();
        if (positionIds != null && !positionIds.wasClosed()) {
            positionIds.setCloseable(true);
            positionIds.close();
        }
        if (attentionMask != null && !attentionMask.wasClosed()) {
            attentionMask.setCloseable(true);
            attentionMask.close();
        }
    }

    // ─── Evolution Mode Enum ────────────────────────────────────────────────

    public enum EvolutionMode {
        /** Only position_ids change each step; mask and KV stay fixed. */
        POSITION_IDS_ONLY,
        /** Only attention_mask changes; position and KV stay fixed. */
        ATTENTION_MASK_ONLY,
        /** Only past_key_values change; position and mask stay fixed. */
        KV_ONLY,
        /** All inputs change each step — full decode-style validation. */
        ALL_TOGETHER
    }

    // ─── Builder ────────────────────────────────────────────────────────────

    public static class Builder {
        private List<String> decoderInputNames;
        private String logitsName = "logits";
        private List<String> kvOutputNames;
        private long prefillSeqLen = 1;
        private int maxDecodeSteps = 10;
        private EvolutionMode evolutionMode = EvolutionMode.ALL_TOGETHER;
        private boolean paddedMode = false;
        private long maxKvLen = 0;

        public Builder decoderInputNames(List<String> names) { this.decoderInputNames = names; return this; }
        public Builder logitsName(String name) { this.logitsName = name; return this; }
        public Builder kvOutputNames(List<String> names) { this.kvOutputNames = names; return this; }
        public Builder prefillSeqLen(long len) { this.prefillSeqLen = len; return this; }
        public Builder maxDecodeSteps(int steps) { this.maxDecodeSteps = steps; return this; }
        public Builder evolutionMode(EvolutionMode mode) { this.evolutionMode = mode; return this; }
        public Builder paddedMode(boolean padded) { this.paddedMode = padded; return this; }
        public Builder maxKvLen(long len) { this.maxKvLen = len; return this; }

        public DecodeInputEvolutor build() {
            return new DecodeInputEvolutor(this);
        }
    }
}
