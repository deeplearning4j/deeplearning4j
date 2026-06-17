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
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Reusable decode-step diagnostics for VLM/LLM token generation.
 *
 * <p>Consolidates per-step diagnostic checks into one place so any decode loop
 * can call {@link #diagnoseStep} after each decoder invocation. All output goes
 * through SLF4J at INFO level, gated by {@link #enabled}.</p>
 *
 * <h3>Configurable via system properties</h3>
 * <ul>
 *   <li>{@code vlm.decode.diagnostics=true} — enable all diagnostics</li>
 *   <li>{@code vlm.decode.diagnostics.steps=N} — only diagnose steps 0..N (default: all when enabled)</li>
 *   <li>{@code vlm.decode.diagnostics.kvFingerprint=true} — KV cache mutation tracking</li>
 *   <li>{@code vlm.decode.diagnostics.logitHealth=true} — logits tensor health checks</li>
 *   <li>{@code vlm.decode.diagnostics.autoOnFailure=true} — auto-enable on accuracy failure (default: true)</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>
 * DecodeStepDiagnostics diag = new DecodeStepDiagnostics();
 * // In decode loop:
 * diag.diagnoseStep(step, logitsRaw, decoderInputMap, decoderOutputs, kvNames, ioConfig, tokenId);
 * </pre>
 */
@Slf4j
public class DecodeStepDiagnostics {

    private final boolean enabled;
    private final boolean kvFingerprintEnabled;
    private final boolean logitHealthEnabled;
    private final boolean autoOnFailure;
    private final int maxDiagnosticSteps;

    // KV fingerprint state: extInputName -> previous fingerprint hash
    private final Map<String, Long> prevKvFingerprints = new HashMap<>();
    private int kvStaleCount = 0;

    // Logit health tracking
    private int consecutiveZeroLogitSteps = 0;
    private int consecutiveArgmaxZeroSteps = 0;
    private boolean autoEnabled = false;

    public DecodeStepDiagnostics() {
        this.enabled = Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.VLM_DECODE_DIAGNOSTICS, "false"));
        this.kvFingerprintEnabled = enabled ||
                Boolean.parseBoolean(System.getProperty("vlm.decode.diagnostics.kvFingerprint", "false"));
        this.logitHealthEnabled = enabled ||
                Boolean.parseBoolean(System.getProperty("vlm.decode.diagnostics.logitHealth", "false"));
        this.autoOnFailure = Boolean.parseBoolean(
                System.getProperty("vlm.decode.diagnostics.autoOnFailure", "true"));
        this.maxDiagnosticSteps = Integer.getInteger("vlm.decode.diagnostics.steps", Integer.MAX_VALUE);
    }

    private boolean isActive(int step) {
        return (enabled || autoEnabled) && step <= maxDiagnosticSteps;
    }

    /**
     * Run all diagnostic checks for one decode step.
     *
     * @param step             decode step index (0-based)
     * @param logitsRaw        raw logits output from decoder (rank 2 or 3)
     * @param decoderInputs    input map passed to decoder (includes KV buffers)
     * @param decoderOutputs   output map from decoder (includes present KV)
     * @param kvNames          KV cache name mappings (may be null if no KV)
     * @param ioConfig         model I/O config for name classification
     * @param sampledTokenId   the token ID chosen by sampling
     */
    public void diagnoseStep(int step, INDArray logitsRaw,
                             Map<String, INDArray> decoderInputs,
                             Map<String, INDArray> decoderOutputs,
                             ModelIOConfig.KVCacheNames kvNames,
                             ModelIOConfig ioConfig,
                             int sampledTokenId) {
        // Logit health runs even when not "active" — it triggers auto-enable
        LogitHealth health = checkLogitHealth(step, logitsRaw, sampledTokenId);

        if (!isActive(step) && !kvFingerprintEnabled && !logitHealthEnabled) return;

        if (logitHealthEnabled || isActive(step)) {
            logLogitHealth(step, health);
        }

        if ((kvFingerprintEnabled || isActive(step)) && decoderInputs != null && ioConfig != null) {
            checkKvCacheMutation(step, decoderInputs, ioConfig);
        }

        if (isActive(step) && decoderOutputs != null && kvNames != null && ioConfig != null) {
            checkPresentKvHealth(step, decoderOutputs, kvNames);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Logit Health
    // ═════════════════════════════════════════════════════════════════════════

    public enum LogitHealthStatus {
        NORMAL,
        ALL_ZERO,
        ALL_SAME,
        HAS_NAN,
        HAS_INF,
        ARGMAX_ZERO_SUSPICIOUS
    }

    public static class LogitHealth {
        public final LogitHealthStatus status;
        public final int argmaxId;
        public final float argmaxValue;
        public final float min;
        public final float max;
        public final float mean;
        public final int vocabSize;
        public final float[] firstN;

        LogitHealth(LogitHealthStatus status, int argmaxId, float argmaxValue,
                    float min, float max, float mean, int vocabSize, float[] firstN) {
            this.status = status;
            this.argmaxId = argmaxId;
            this.argmaxValue = argmaxValue;
            this.min = min;
            this.max = max;
            this.mean = mean;
            this.vocabSize = vocabSize;
            this.firstN = firstN;
        }
    }

    /**
     * Analyze logits tensor health. Safe to call on any array — uses only getFloat().
     * Never calls CUDA reduction ops which can fail on constant-flagged DSP outputs.
     */
    public LogitHealth checkLogitHealth(int step, INDArray logitsRaw, int sampledTokenId) {
        if (logitsRaw == null) {
            return new LogitHealth(LogitHealthStatus.ALL_ZERO, 0, 0, 0, 0, 0, 0, new float[0]);
        }
        try {
            INDArray lastLogits = logitsRaw.rank() == 3
                    ? logitsRaw.get(NDArrayIndex.point(0),
                                    NDArrayIndex.point(logitsRaw.size(1) - 1),
                                    NDArrayIndex.all())
                    : logitsRaw.getRow(0);

            int len = (int) lastLogits.length();
            // Bulk transfer to host — avoids 49K individual JNI getFloat() calls
            float[] vals = lastLogits.dup().data().asFloat();

            int argmaxId = 0;
            float argmaxVal = vals[0];
            float min = vals[0], max = vals[0];
            double sum = vals[0];
            boolean hasNaN = false, hasInf = false;

            for (int i = 1; i < len; i++) {
                float v = vals[i];
                if (Float.isNaN(v)) hasNaN = true;
                if (Float.isInfinite(v)) hasInf = true;
                if (v > argmaxVal) { argmaxVal = v; argmaxId = i; }
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            float mean = (float) (sum / len);
            float[] firstN = new float[Math.min(8, len)];
            System.arraycopy(vals, 0, firstN, 0, firstN.length);

            LogitHealthStatus status;
            if (hasNaN) {
                status = LogitHealthStatus.HAS_NAN;
            } else if (hasInf) {
                status = LogitHealthStatus.HAS_INF;
            } else if (max == 0.0f && min == 0.0f) {
                status = LogitHealthStatus.ALL_ZERO;
                consecutiveZeroLogitSteps++;
            } else if (max == min) {
                status = LogitHealthStatus.ALL_SAME;
            } else {
                status = LogitHealthStatus.NORMAL;
                consecutiveZeroLogitSteps = 0;
            }

            if (sampledTokenId == 0 && status == LogitHealthStatus.NORMAL) {
                consecutiveArgmaxZeroSteps++;
                if (consecutiveArgmaxZeroSteps >= 2) {
                    status = LogitHealthStatus.ARGMAX_ZERO_SUSPICIOUS;
                }
            } else if (sampledTokenId != 0) {
                consecutiveArgmaxZeroSteps = 0;
            }

            // Auto-enable diagnostics on failure detection
            if (autoOnFailure && !autoEnabled) {
                if (consecutiveZeroLogitSteps >= 2 || consecutiveArgmaxZeroSteps >= 3) {
                    autoEnabled = true;
                    log.warn("[DIAG] AUTO-ENABLED: {} consecutive zero-logit steps or {} consecutive argmax=0 steps — enabling full diagnostics",
                            consecutiveZeroLogitSteps, consecutiveArgmaxZeroSteps);
                }
            }

            return new LogitHealth(status, argmaxId, argmaxVal, min, max, mean, len, firstN);
        } catch (Exception e) {
            log.warn("[DIAG] logit health check failed at step {}: {}", step, e.getMessage());
            return new LogitHealth(LogitHealthStatus.ALL_ZERO, 0, 0, 0, 0, 0, 0, new float[0]);
        }
    }

    private void logLogitHealth(int step, LogitHealth h) {
        String statusTag = h.status == LogitHealthStatus.NORMAL ? "" : " *** " + h.status + " ***";
        log.info("[DIAG] step={} logits: status={} argmax={} val={} min={} max={} mean={} vocab={} first8={}{}",
                step, h.status, h.argmaxId,
                String.format("%.4f", h.argmaxValue),
                String.format("%.4f", h.min),
                String.format("%.4f", h.max),
                String.format("%.6f", h.mean),
                h.vocabSize,
                formatFloats(h.firstN),
                statusTag);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // KV Cache Mutation Detection
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Check that KV cache input buffers have actually changed between decode steps.
     * Computes a lightweight fingerprint (sample 8 positions) of each KV cache input
     * and compares against the previous step. If ALL KV caches have identical
     * fingerprints, that's a strong signal that KV scatter isn't working.
     */
    private void checkKvCacheMutation(int step, Map<String, INDArray> decoderInputs,
                                      ModelIOConfig ioConfig) {
        if (step < 1) {
            // Seed fingerprints on first step
            for (var entry : decoderInputs.entrySet()) {
                if (ioConfig.isKvCacheInput(entry.getKey())) {
                    prevKvFingerprints.put(entry.getKey(), fingerprint(entry.getValue()));
                }
            }
            return;
        }

        int kvCount = 0;
        int unchangedCount = 0;

        for (var entry : decoderInputs.entrySet()) {
            if (!ioConfig.isKvCacheInput(entry.getKey())) continue;
            kvCount++;

            long currentFp = fingerprint(entry.getValue());
            Long prevFp = prevKvFingerprints.get(entry.getKey());

            if (prevFp != null && prevFp == currentFp) {
                unchangedCount++;
                if (isActive(step)) {
                    log.warn("[DIAG] step={} KV_UNCHANGED: '{}' fingerprint={} (same as step {})",
                            step, entry.getKey(), Long.toHexString(currentFp), step - 1);
                }
            }
            prevKvFingerprints.put(entry.getKey(), currentFp);
        }

        if (kvCount > 0 && unchangedCount == kvCount) {
            kvStaleCount++;
            log.error("[DIAG] step={} KV_CACHE_STALE: ALL {} KV cache inputs have identical fingerprints " +
                            "to previous step ({} consecutive stale steps). KV scatter may not be working.",
                    step, kvCount, kvStaleCount);
        } else {
            kvStaleCount = 0;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Present KV Output Health
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Check that present KV outputs from the decoder are non-zero at the last
     * sequence position. Zero present KV → KV scatter writes zeros → attention
     * degrades on subsequent steps.
     */
    private void checkPresentKvHealth(int step, Map<String, INDArray> decoderOutputs,
                                      ModelIOConfig.KVCacheNames kvNames) {
        int zeroCount = 0;
        int totalChecked = 0;

        for (String presentName : kvNames.keyNames) {
            INDArray present = decoderOutputs.get(presentName);
            if (present == null || present.isEmpty()) continue;
            totalChecked++;
            if (isPresentKvZero(present)) {
                zeroCount++;
                if (isActive(step)) {
                    log.warn("[DIAG] step={} PRESENT_KV_ZERO: '{}' shape={} — last-position slice is all zero",
                            step, presentName, Arrays.toString(present.shape()));
                }
            }
        }

        for (String presentName : kvNames.valueNames) {
            INDArray present = decoderOutputs.get(presentName);
            if (present == null || present.isEmpty()) continue;
            totalChecked++;
            if (isPresentKvZero(present)) {
                zeroCount++;
            }
        }

        if (totalChecked > 0 && zeroCount == totalChecked) {
            log.error("[DIAG] step={} ALL_PRESENT_KV_ZERO: all {} present KV outputs are zero at last position. " +
                    "Attention layer is producing zero output — upstream data (embeddings, KV cache) may be corrupt.",
                    step, totalChecked);
        }
    }

    /**
     * Check if a present KV tensor [B, H, S, D] has all-zero values at the last
     * sequence position (the newly computed position).
     */
    private boolean isPresentKvZero(INDArray present) {
        try {
            if (present.rank() != 4) return false;
            long lastPos = present.size(2) - 1;
            INDArray lastSlice = present.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(lastPos),
                    NDArrayIndex.all());
            float[] vals = lastSlice.dup().data().asFloat();
            for (float v : vals) {
                if (v != 0.0f) return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Utilities
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Lightweight fingerprint: FNV-1a hash of 8 sampled float values spread
     * across the array. Avoids full D2H transfer for large KV buffers.
     */
    private static long fingerprint(INDArray arr) {
        if (arr == null || arr.isEmpty()) return 0L;
        long len = arr.length();
        long h = 0xcbf29ce484222325L;
        int samples = (int) Math.min(8, len);
        long stride = Math.max(1, len / samples);
        for (int i = 0; i < samples; i++) {
            long idx = Math.min(i * stride, len - 1);
            int bits = Float.floatToRawIntBits(arr.getFloat(idx));
            h ^= bits;
            h *= 0x100000001b3L;
        }
        return h;
    }

    private static String formatFloats(float[] vals) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vals.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", vals[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Summary report — call at end of decode to log accumulated statistics.
     */
    public void logSummary(int totalSteps) {
        if (kvStaleCount > 0) {
            log.error("[DIAG] SUMMARY: {} KV-stale steps detected out of {} total decode steps",
                    kvStaleCount, totalSteps);
        }
        if (consecutiveZeroLogitSteps > 0) {
            log.error("[DIAG] SUMMARY: {} consecutive zero-logit steps at end of decode", consecutiveZeroLogitSteps);
        }
        if (autoEnabled) {
            log.warn("[DIAG] SUMMARY: diagnostics were auto-enabled during decode due to accuracy failure");
        }
    }

    /**
     * Reset state for a new decode run (e.g., new page/document).
     */
    public void reset() {
        prevKvFingerprints.clear();
        kvStaleCount = 0;
        consecutiveZeroLogitSteps = 0;
        consecutiveArgmaxZeroSteps = 0;
        autoEnabled = false;
    }
}
