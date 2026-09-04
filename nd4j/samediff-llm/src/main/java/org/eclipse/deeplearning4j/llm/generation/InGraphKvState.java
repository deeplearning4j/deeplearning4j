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
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintMasker;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Retained decode state for a single in-graph-KV {@link GenerationPipeline.GenerationSession}.
 *
 * <p>This holds everything produced by prefill + warmup + freeze that must survive <em>across</em>
 * continuation calls so decoding can resume from the current {@code cachePosition} without a session
 * reset or a re-prefill: the static KV / recurrent-state buffers, the decode-step input tensors, the
 * frozen plan handles, the resolved external-input indices, and the running generation state
 * ({@code cachePosition}, {@code lastGeneratedToken}, the full generated-token list, the sampler
 * RNG).</p>
 *
 * <p><strong>Ownership:</strong> the retained {@code INDArray} buffers are owned by this object and
 * freed exactly once in {@link #close()}. The {@code executor} / {@code planHandle} /
 * {@code contextHandle} are owned by the decoder's {@code InferenceSession} and are <em>not</em>
 * freed here.</p>
 *
 * <p><strong>Thread-confinement:</strong> this state is bound to {@link #ownerThreadId}; the decoder's
 * {@code InferenceSession} and frozen plan are thread-affine, so continuation must run on the creating
 * thread. Coordination is lock-free (see {@code GenerationSession}); no monitor is held across a native
 * decode. This class is a plain data holder and performs no locking itself.</p>
 */
@Slf4j
class InGraphKvState implements AutoCloseable {

    // ── Retained native buffers (freed only in close()) ──────────────────────────────────────────
    Map<String, INDArray> staticKvBuffers;
    Map<String, INDArray> recurrentStateBuffers;

    /**
     * Sources of asynchronous recurrent-state copies that must remain alive until a later
     * host-visible decode result establishes the natural completion boundary. Closing a large donor
     * immediately after {@code assign()} can let the CUDA allocator recycle its storage while the D2D
     * copy is still in flight.
     */
    private final List<INDArray> recurrentCopyDonors = new ArrayList<>();

    /**
     * QUANTIZED-strategy buffers: INT8-compressed KV data.
     *
     * <p><b>V1 layout</b> (post-prefill archive only): keys follow
     * {@code past_key_values.{L}.key_q} / {@code past_key_values.{L}.value_q}.
     *
     * <p><b>V2 layout</b> (live storage): keys follow the ORIGINAL variable names
     * {@code past_key_values.{L}.key} / {@code past_key_values.{L}.value}
     * (same as the former {@code staticKvBuffers} keys). This allows the frozen plan's
     * ext-input slot to remain at the same index — only the pointed-to buffer changes
     * from float to INT8. The former {@code _q}-suffixed entries are NOT used in V2.
     *
     * <p>Null when {@code kvCacheStrategy != QUANTIZED}. Freed in {@link #close()}.
     */
    Map<String, INDArray> quantizedKvBuffers;

    /**
     * QUANTIZED-strategy scale buffers: per-token-per-head FLOAT32 scales paired with
     * {@link #quantizedKvBuffers}. Keys follow {@code past_key_values.{L}.key_scale} /
     * {@code past_key_values.{L}.value_scale}.
     * Null when {@code kvCacheStrategy != QUANTIZED}. Freed in {@link #close()}.
     */
    Map<String, INDArray> kvScaleBuffers;

    /** Active KV quantization format (1=INT8, 2=FP8_E4M3, 3=FP8_E5M2, 4=INT4). 0 = not quantized. */
    int kvQuantFormat;

    /**
     * V2 flag: true when the live decode KV storage is INT8 (float buffers freed after prefill).
     * When true, {@link #staticKvBuffers} is null and {@link #quantizedKvBuffers} contains the
     * INT8 live buffers under the original KV variable names (not the _q-suffixed names).
     * ADR 0107 §prefill: staticKvBuffers must be null at the start of decode.
     */
    boolean isQuantizedV2;
    INDArray decodeInputIds;       // [1,1] INT64 — current-token slot, overwritten each step
    INDArray decodeCausalMask;     // [1,1,1,maxKvLen], nullable
    INDArray decodePositionOffset; // scalar INT64, nullable
    INDArray decodeCachePosition;  // scalar INT64, nullable
    INDArray decodeActualSequenceLength; // scalar INT64, nullable; decode recurrent ops always see one token

    /**
     * Prefill-plan external inputs, retained so the fixed-buffer forward-fix can REUSE them (with
     * stable addresses) across consecutive one-shot generate() calls. On the frozen multi-plan switch
     * the prefill plan ([1,prefillSeqLen]) is also frozen/captured, so its external inputs must keep
     * stable device addresses for a correct replay — a fresh tensor per generate would leave the
     * captured graph reading a dangling/stale address (silent gen-3+ degeneration). On reuse the VARYING
     * inputs (input_ids, causal mask) are overwritten in place (assign); the CONSTANT ones (empty-KV
     * sentinels, zero recurrent state, zero position/cache-position scalars) are reused as-is. Retained
     * for BOTH the fixed and variable paths (freed in {@link #close()}); only the fixed path reuses it.
     */
    Map<String, INDArray> prefillInputMap;

    /**
     * Reusable host-side scratch for the per-sample re-prefill temporaries (fresh
     * input-ids array, fresh causal-mask array). Each one-shot {@code generate()}
     * call builds these fresh and drops them right after {@code assign} into the
     * stable-address inputs; allocating them from a retained scratch pool instead of
     * fresh memory keeps a long calibration/decode run from churning native allocator
     * arenas (glibc never returns those to the OS, so RSS ratchets up ~50MB per
     * sample on Android). Contents are fully overwritten by the builder before use,
     * and the pool is sized to the fixed-buffer prefill geometry, so reuse is safe.
     */
    final Map<String, INDArray> reusedScratch = new java.util.HashMap<>();

    // ── Bundled Qwen3.5 MTP predictor state ─────────────────────────────────────────────────────
    /** Independent predictor KV cache. The target and predictor never share mutable cache storage. */
    Map<String, INDArray> mtpKvBuffers;
    /** Stable-address MTP prefill inputs retained for fixed-buffer plan replay. */
    Map<String, INDArray> mtpPrefillInputMap;
    /** Scalar decode inputs retained at stable addresses for the native MTP loop. */
    INDArray mtpInputIds;
    INDArray mtpTargetHiddenStates;
    INDArray mtpCausalMask;
    INDArray mtpPositionOffset;
    INDArray mtpCachePosition;

    /**
     * MTP uses an isolated SameDiff inference session so its scalar plan can coexist with the
     * target model's W-wide verification plan while both branches share immutable graph weights.
     * Unlike {@link #executor}, this session is owned by this state and is cleared in {@link #close()}.
     */
    InferenceSession mtpSession;
    DynamicShapePlanExecutor mtpExecutor;
    Pointer mtpPlanHandle;
    Pointer mtpContextHandle;

    // ── Frozen plan handles (owned by the decoder's InferenceSession — NOT closed here) ──────────
    DynamicShapePlanExecutor executor;
    Pointer planHandle;
    Pointer contextHandle;

    // ── Resolved external-input / output indices (from freeze) ───────────────────────────────────
    int inputIdsExtIdx;
    int causalMaskExtIdx;
    int posOffsetExtIdx;
    int cachePosExtIdx;
    int actualSeqLenExtIdx;
    int logitsOutputIdx;
    int embeddingsExtIdx;
    int maskExtIdx;
    int posIdsExtIdx;
    int[] kvInputExtIndices;
    int[] kvOutputIndices;
    int[] gdnStateExtIndices;
    int[] gdnStateOutputIndices;
    int[] convStateExtIndices;
    int[] convStateOutputIndices;
    int numPlanExternalInputs;
    int numPlanOutputs;
    int numKvPairs;

    /** Target-plan output carrying pre-final-norm hidden rows used to refresh the MTP state. */
    int targetHiddenOutputIdx = -1;
    int mtpInputIdsExtIdx = -1;
    int mtpTargetHiddenExtIdx = -1;
    int mtpCausalMaskExtIdx = -1;
    int mtpPosOffsetExtIdx = -1;
    int mtpCachePosExtIdx = -1;
    int[] mtpKvInputExtIndices;
    int mtpLogitsOutputIdx = -1;
    int mtpHiddenOutputIdx = -1;
    int mtpNumPlanExternalInputs;
    int mtpNumPlanOutputs;

    // ── Running decode state ─────────────────────────────────────────────────────────────────────
    /** Absolute position at which the next-fed token ({@link #lastGeneratedToken}) is written: {@code P + G - 1}. */
    volatile int cachePosition;
    /** The last sampled token — its K/V is not yet written; feeding it resumes the sequence. */
    volatile int lastGeneratedToken;
    /** The full generated-token sequence across all calls (prompt excluded). */
    List<Integer> generatedSoFar;
    Random rng;
    SamplingConfig sampling;
    ConstraintMasker constraintMasker;
    Set<Integer> stopTokenIds;
    int eosTokenId;

    // ── Capacity / shape metadata ────────────────────────────────────────────────────────────────
    long maxKvLen;          // total KV buffer length (the hard capacity ceiling)
    int actualPrefillLen;   // P — real (non-padded) prompt length; drives cachePosition math
    int prefillSeqLen;      // padded prefill length (== actualPrefillLen in variable-size mode)
    int promptTokenCount;   // value reported as GenerationResult.promptTokenCount
    DataType maskDtype;

    // ── Rotating KV cache (StreamingLLM-style) ───────────────────────────────────────────────────
    /**
     * Non-null when rotating KV cache is enabled. Manages the global→physical slot mapping
     * (attention sinks pinned at head, ring of non-sink tokens in the remainder).
     * Null when rotating is disabled (default); all existing code paths are unchanged when null.
     *
     * @see RotatingKvSlotMap
     * @see GenerationPipelineConfig#isRotatingKvEnabled()
     */
    RotatingKvSlotMap rotatingSlotMap;

    // ── Variable names (for the append() outputDirect path) ──────────────────────────────────────
    String inputIdsName;
    String logitsName;
    String causalMaskName;
    String posOffsetName;
    String cachePosName;
    String actualSeqLenName;
    ModelIOConfig.KVCacheNames kvInputNames;
    List<ModelIOConfig.RecurrentStatePair> recurrentStates;
    List<String> decodeOutputNames;

    // ── Guards / concurrency ─────────────────────────────────────────────────────────────────────
    volatile boolean eosReached;
    volatile boolean closed;
    volatile boolean cancelRequested;
    long sessionId;
    long ownerThreadId;

    /**
     * Non-null only for a session that terminated during prefill/warmup (first token was EOS, or the
     * native plan handle was unavailable). The session is already {@link #closed} in that case and the
     * carried result is returned by the session's first {@code generate}.
     */
    GenerationResult terminalResult;

    /**
     * Remaining new-token capacity before the KV buffer is full.
     *
     * <p>When rotating KV cache is enabled ({@link #rotatingSlotMap} is non-null) this always returns
     * {@link Integer#MAX_VALUE}: generation is unbounded (tokens are evicted from the ring rather
     * than the session stopping). The hard-stop guard in {@code decodeInSession} is bypassed only when
     * this returns a positive value, so callers must check {@code rotatingSlotMap != null} or call
     * {@link #isRotatingKvEnabled()} to distinguish "unbounded" from "capacity remaining".</p>
     */
    int remainingCapacity() {
        if (rotatingSlotMap != null) return Integer.MAX_VALUE;
        return (int) (maxKvLen - cachePosition - 1);
    }

    /** Whether rotating KV cache is active for this session. */
    boolean isRotatingKvEnabled() {
        return rotatingSlotMap != null;
    }

    /** Retain ownership of queued-copy sources until the next natural decode completion boundary. */
    void retainRecurrentCopyDonors(List<INDArray> donors) {
        if (donors != null && !donors.isEmpty()) recurrentCopyDonors.addAll(donors);
    }

    /** Release queued-copy sources only after their consumer has produced a host-visible result. */
    void releaseRecurrentCopyDonors() {
        for (INDArray donor : recurrentCopyDonors) safeClose(donor);
        recurrentCopyDonors.clear();
    }

    /**
     * Free all retained buffers exactly once. Idempotent. Does NOT touch the frozen-plan handles
     * (owned by the decoder's InferenceSession).
     *
     * <p>ADR 0107 V2 assertion: if this is a V2 quantized session ({@link #isQuantizedV2}),
     * the float {@code staticKvBuffers} must have been freed at prefill time. A non-null
     * {@code staticKvBuffers} here indicates a live-memory leak — the float KV was never freed.
     */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        // Destroy the isolated predictor plan before releasing any external inputs it references.
        if (mtpSession != null) {
            try {
                mtpSession.clearAllCaches();
            } catch (Exception e) {
                log.warn("[GenerationSession] error clearing isolated MTP session: {}", e.getMessage());
            } finally {
                mtpSession = null;
                mtpExecutor = null;
                mtpPlanHandle = null;
                mtpContextHandle = null;
            }
        }
        // ADR 0107 §prefill invariant: V2 sessions must have freed float KV before decode.
        if (isQuantizedV2 && staticKvBuffers != null) {
            log.warn("[InGraphKvState] V2 quantized session still has non-null staticKvBuffers at close() "
                    + "— float KV was not freed after prefill. This is a live-memory leak.");
        }
        safeClose(decodeInputIds);
        safeClose(decodePositionOffset);
        safeClose(decodeCachePosition);
        safeClose(decodeActualSequenceLength);
        safeClose(decodeCausalMask);
        safeClose(mtpInputIds);
        safeClose(mtpTargetHiddenStates);
        safeClose(mtpCausalMask);
        safeClose(mtpPositionOffset);
        safeClose(mtpCachePosition);
        releaseRecurrentCopyDonors();
        closeAll(mtpKvBuffers);
        closeAll(mtpPrefillInputMap);
        closeAll(recurrentStateBuffers);
        closeAll(staticKvBuffers);
        closeAll(quantizedKvBuffers);
        closeAll(kvScaleBuffers);
        closeAll(prefillInputMap);
        closeAll(reusedScratch);
    }

    private static void closeAll(Map<String, INDArray> buffers) {
        if (buffers == null) return;
        for (INDArray a : buffers.values()) safeClose(a);
    }

    private static void safeClose(INDArray a) {
        if (a == null || a.wasClosed()) return;
        try {
            a.close();
        } catch (Exception e) {
            log.warn("[GenerationSession] error closing retained buffer: {}", e.getMessage());
        }
    }
}
