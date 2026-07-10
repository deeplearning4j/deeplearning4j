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

/**
 * Slot manager for StreamingLLM-style rotating KV cache (attention sinks + sliding window).
 *
 * <h2>Physical layout</h2>
 * <pre>
 *   Slot 0 .. sinkCount-1           : sink tokens (pinned, never evicted)
 *   Slot sinkCount .. maxKvLen-1    : ring buffer for non-sink tokens
 * </pre>
 *
 * <h2>Global → physical mapping</h2>
 * <ul>
 *   <li>Global positions 0 .. sinkCount-1 always map to physical slots 0 .. sinkCount-1 (sinks).</li>
 *   <li>Global positions {@code >= sinkCount} map into the ring: physical slot =
 *       {@code sinkCount + (globalPos - sinkCount) % ringSize} where {@code ringSize = maxKvLen - sinkCount}.</li>
 * </ul>
 *
 * <h2>CUDA-graph safety</h2>
 * The KV buffer shape {@code [B, maxKvLen, H_kv, D]} is FIXED. The physical write position
 * fed to the C++ op is the <em>physical slot</em> returned by {@link #physicalSlot(int)}.
 * When the ring wraps, the C++ op writes the new K/V over the oldest evicted non-sink entry.
 * No buffer reallocation occurs; no DSP plan or CUDA graph is invalidated.
 *
 * <h2>Attention mask</h2>
 * The mask must unmask exactly the physical slots that contain valid K/V:
 * <ul>
 *   <li>Sinks: slots 0 .. sinkCount-1 are always unmasked after the first sinkCount tokens.</li>
 *   <li>Ring: the {@code min(ringSize, globalPosition - sinkCount)} most recently written slots
 *       within sinkCount .. maxKvLen-1 are unmasked; the rest are masked.</li>
 * </ul>
 * Use {@link #buildRotatingDecodeMask(long, long, float)} to compute the mask data array for a
 * given global position. The mask is always of length {@code maxKvLen} (matching the fixed buffer).
 *
 * <h2>RoPE quality caveat (MUST READ)</h2>
 * The keys stored in the KV cache are post-RoPE (rotated at their <em>original global positions</em>
 * by {@code FusedRoPE} before being written by {@code dotProductAttentionV2/kvInPlaceWriteBSHD}).
 * When a non-sink key at global position G is evicted by a new key at global position G', the new
 * key is written to the same physical slot but carries the RoPE encoding of G'. The attention kernel
 * reads the accumulated cache with mixed position encodings: sink keys keep their original positions
 * (correct), and the ring slots carry the positions of whoever last wrote to them. For the current
 * window (recent tokens) positions are monotonically increasing and correctly ordered relative to the
 * new query. For the gap between the sinks and the ring window, the positional encoding jumps —
 * the model sees sinks at positions 0..sinkCount-1 and recent tokens at G-W..G (where W is ring size),
 * which is the exact StreamingLLM approximation accepted in the original paper (Xiao et al. 2023).
 * <p>
 * QUALITY BOUNDARY: token-for-token correctness is guaranteed only when the total generated sequence
 * is {@code <= maxKvLen} tokens (no eviction occurs). Beyond that boundary the attention quality
 * degrades gracefully (empirically acceptable for LLM streaming) but is NOT bit-identical to
 * attending over the full unbounded history.
 * </p>
 * <p>
 * GDN / recurrent-state models (e.g. Qwen GDN): recurrent state (conv / delta-net) is position-free
 * and is NOT affected by ring rotation. Only attention KV buffers are managed here. GDN quality is
 * unchanged by enabling rotating KV.
 * </p>
 *
 * @see GenerationPipelineConfig#isRotatingKvEnabled()
 * @see GenerationPipelineConfig#getRotatingKvSinkCount()
 */
public class RotatingKvSlotMap {

    /** System property to override the default sink count. */
    public static final String SINK_COUNT_PROP = "nd4j.generation.rotatingKv.sinkCount";

    /** Default number of sink tokens if not configured. */
    public static final int DEFAULT_SINK_COUNT = 4;

    /** Total physical KV buffer length. Fixed; never changes after construction. */
    @Getter
    private final int maxKvLen;

    /** Number of pinned sink slots at the start of the buffer. */
    @Getter
    private final int sinkCount;

    /** Ring size: maxKvLen - sinkCount. */
    @Getter
    private final int ringSize;

    /**
     * Construct a slot map.
     *
     * @param maxKvLen  total KV buffer length (fixed; must be {@code > sinkCount})
     * @param sinkCount number of pinned sink tokens (typically 4)
     */
    public RotatingKvSlotMap(int maxKvLen, int sinkCount) {
        if (sinkCount < 0) throw new IllegalArgumentException("sinkCount must be >= 0, got " + sinkCount);
        if (maxKvLen <= sinkCount) throw new IllegalArgumentException(
                "maxKvLen(" + maxKvLen + ") must be > sinkCount(" + sinkCount + ")");
        this.maxKvLen = maxKvLen;
        this.sinkCount = sinkCount;
        this.ringSize = maxKvLen - sinkCount;
    }

    /**
     * Map a global token position to the physical KV-buffer slot where its K/V should be written.
     *
     * <p>Global positions 0..sinkCount-1 map to physical slots 0..sinkCount-1 (sinks). Global
     * positions {@code >= sinkCount} map into the ring modulo {@code ringSize}.</p>
     *
     * @param globalPos the absolute position of the token being written (0-based)
     * @return physical slot index in [0, maxKvLen)
     */
    public int physicalSlot(int globalPos) {
        if (globalPos < sinkCount) return globalPos;
        return sinkCount + ((globalPos - sinkCount) % ringSize);
    }

    /**
     * Whether the ring has wrapped at least once (i.e. eviction has occurred).
     *
     * @param globalPos the current global position (the position being written, 0-based)
     * @return true if {@code globalPos >= maxKvLen}
     */
    public boolean hasWrapped(int globalPos) {
        return globalPos >= maxKvLen;
    }

    /**
     * Build the attention-bias mask data array for a single decode step in rotating mode.
     *
     * <p>Returns a float array of length {@code maxKvLen} where:
     * <ul>
     *   <li>Sink slots 0..min(sinkCount-1, globalPos) are 0.0f (unmasked, always valid after
     *       the first sinkCount global positions have been committed).</li>
     *   <li>Ring slots containing live data are 0.0f (unmasked).</li>
     *   <li>All other slots are {@code maskVal} (masked).</li>
     * </ul>
     * The array is suitable for wrapping into a {@code [1,1,1,maxKvLen]} tensor and assigning
     * into the existing {@code decodeCausalMask} buffer in-place.</p>
     *
     * <p>CUDA-graph safety: this method computes data values only; no buffer shape changes occur.</p>
     *
     * @param globalPos the absolute position of the CURRENT token being fed (its K/V is
     *                  written at {@code physicalSlot(globalPos)}; the query attends to
     *                  all previously committed slots: 0..globalPos-1)
     * @param maskVal   the dtype-safe mask value (-65504f for FP16, -1e9f for FP32)
     * @return float array of length maxKvLen; 0.0f = attend, maskVal = masked
     */
    public float[] buildRotatingDecodeMask(int globalPos, float maskVal) {
        float[] mask = new float[maxKvLen];
        // Default: everything masked
        for (int i = 0; i < maxKvLen; i++) mask[i] = maskVal;

        if (globalPos <= 0) {
            // No tokens committed yet; everything masked
            return mask;
        }

        // ── Unmask sink slots ─────────────────────────────────────────────────────────────────────
        // Sinks at physical slots 0..min(sinkCount-1, globalPos-1) are always valid once written.
        int sinksFilled = Math.min(sinkCount, globalPos);
        for (int s = 0; s < sinksFilled; s++) {
            mask[s] = 0.0f;
        }

        // ── Unmask ring slots ─────────────────────────────────────────────────────────────────────
        // The ring holds non-sink tokens at global positions sinkCount .. globalPos-1.
        // Number of non-sink tokens committed so far:
        int nonSinkCount = Math.max(0, globalPos - sinkCount);
        if (nonSinkCount <= 0) {
            // All committed tokens are sinks; ring has no live data yet
            return mask;
        }

        // The ring holds min(nonSinkCount, ringSize) live entries.
        // When not wrapped: positions sinkCount .. globalPos-1 map to physical slots
        //   sinkCount .. sinkCount + nonSinkCount - 1 (contiguous, no wrap yet).
        // When wrapped: the ring has ringSize entries scattered across sinkCount .. maxKvLen-1;
        //   the live physical slots are exactly ALL slots in [sinkCount, maxKvLen-1]
        //   (because eviction is round-robin and every slot has been written at least once).
        if (!hasWrapped(globalPos)) {
            // Pre-wrap: contiguous slots sinkCount .. sinkCount + nonSinkCount - 1
            for (int s = sinkCount; s < sinkCount + nonSinkCount; s++) {
                mask[s] = 0.0f;
            }
        } else {
            // Post-wrap: the entire ring is live
            for (int s = sinkCount; s < maxKvLen; s++) {
                mask[s] = 0.0f;
            }
        }

        return mask;
    }

    /**
     * Resolve the effective sink count from a config value plus the system property fallback.
     *
     * <p>Priority: {@code configValue > 0} → configValue; else system property
     * {@code nd4j.generation.rotatingKv.sinkCount} if set; else {@link #DEFAULT_SINK_COUNT}.</p>
     */
    public static int resolveSinkCount(int configValue) {
        if (configValue > 0) return configValue;
        String prop = System.getProperty(SINK_COUNT_PROP);
        if (prop != null) {
            try {
                int v = Integer.parseInt(prop.trim());
                if (v >= 0) return v;
            } catch (NumberFormatException ignore) {
            }
        }
        return DEFAULT_SINK_COUNT;
    }

    @Override
    public String toString() {
        return "RotatingKvSlotMap{maxKvLen=" + maxKvLen + ", sinkCount=" + sinkCount
                + ", ringSize=" + ringSize + "}";
    }
}
