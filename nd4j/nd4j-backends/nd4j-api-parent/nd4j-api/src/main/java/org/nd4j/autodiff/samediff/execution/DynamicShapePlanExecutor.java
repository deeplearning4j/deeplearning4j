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

package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;

/**
 * Executes a compiled {@link DynamicShapePlan} with pre-wired index-based slot access.
 *
 * <p>Key optimizations over standard InferenceSession execution:</p>
 * <ol>
 *   <li><b>Array-indexed variable storage</b> — {@code INDArray[] slots} replaces
 *       {@code Map<String, SDValue>}</li>
 *   <li><b>Pre-resolved input wiring</b> — integer indices replace {@code op.argNames()}
 *       + HashMap.get()</li>
 *   <li><b>Pre-computed liveness</b> — {@code releaseAtStep[i]} replaces
 *       {@code arrayUseTracker}</li>
 *   <li><b>shapeFunctionOverride(true)</b> — tells C++ to skip redundant shape calc +
 *       prepareOutputs</li>
 *   <li><b>Pre-allocated OpContext pool</b> — {@code OpContext[step]} replaces per-op
 *       pool polling</li>
 *   <li><b>Per-slot shape cache</b> — avoids hash key computation when shapes unchanged</li>
 * </ol>
 *
 * @see DynamicShapePlan
 * @see DynamicShapeSlot
 */
@Slf4j
public class DynamicShapePlanExecutor implements Closeable {

    /** Whether native C++ graph executor is enabled. When true, the entire plan is executed
     *  in C++ via a single JNI call instead of per-op Java dispatch. Default: true. */
    private static final boolean NATIVE_EXECUTOR_ENABLED = !"false".equalsIgnoreCase(
            System.getProperty(ND4JSystemProperties.DSP_NATIVE_EXECUTOR_ENABLED, "true"));

    /** Number of execute() calls on this executor. Used to skip cache validity probe
     *  on the first execution (no cached entries yet). */
    private int executionCount;

    /** Java-side tracking of shapes-frozen state. When true, shape caches don't need
     *  clearing between executions because all shapes are guaranteed constant. */
    private boolean shapesFrozen;

    /** Optional interceptor called after each slot execution. Null by default (zero overhead).
     *  @deprecated Only functional with the removed Java slot-by-slot execution path. */
    @Deprecated
    private SlotOutputInterceptor slotOutputInterceptor;


    private final SameDiff sd;
    private final SessionMemMgr mmgr;

    /** The plan this executor is currently configured for. */
    private DynamicShapePlan currentPlan;

    /** Flat output array slots: stores op outputs by slot index. */
    private INDArray[] outputSlots;

    /** External input array cache: resolved constant/variable/placeholder arrays. */
    private INDArray[] externalInputs;

    /** Pending DataBuffers to close. Used by closeSlotArrayCache for cleanup. */
    private ArrayList<DataBuffer> pendingClose = new ArrayList<>();

    /** Persistent dedup sets for buffer cleanup.
     *  Identity dedup prevents processing the same DataBuffer object twice.
     *  ODB dedup prevents double-close of the same native OpaqueDataBuffer. */
    private Set<DataBuffer> seenIdentity;
    private HashSet<Long> closedOdbAddresses;

    /** Buffers deferred from a flush because their GPU address was still
     *  used by a live slot (view of parent). */
    private ArrayList<DataBuffer> deferredClose = new ArrayList<>();

    /** Cached device count from nativeOps.getAvailableDevices(). Computed once during
     *  initialize(). Defaults to 1 for CPU backend. */
    private int cachedNumDevices = 1;

    /** Slot-indexed array cache: persists across execute() calls for O(1) array reuse.
     *  Used by closeSlotArrayCache for cleanup during session resets. */
    private INDArray[] slotArrayCache;

    /** Accumulated freed buffer count and bytes across all flushes in one execution. */
    private int totalFlushedCount;
    private long totalFlushedBytes;

    /** Native C++ plan handle. Compiled once from the serialized plan on first native
     *  execution attempt. Freed on close(). null means not yet compiled or compilation failed. */
    private Pointer nativePlanHandle;

    /** Track which plan the native handle was compiled from. If the plan changes,
     *  the native handle must be recompiled. */
    private DynamicShapePlan nativePlanSource;

    /** Graph execution mode currently configured on the native plan handle. */
    private GraphExecutionMode configuredGraphExecutionMode = GraphExecutionMode.AUTO;

    /** If native compilation fails, disable native execution for this executor instance
     *  to avoid repeated failure overhead. */
    private boolean nativeExecutorFailed;

    /** If CUDA graph capture fails, disable CUDA graphs but keep using slot-by-slot native execution. */
    private boolean cudaGraphsFailed;

    /** KV cache retention state: when configured, C++ scatters new KV entries
     *  into static input buffers, avoiding 60 copyBuffer round-trips per decode step. */
    private boolean kvCacheRetentionConfigured;

    /** Set of present KV output names managed by C++ scatter. Skip copying these in executeNative(). */
    private Set<String> kvRetentionOutputNames;

    /** Saved KV retention configuration for re-application after plan recompilation.
     *  When the plan changes (e.g., fullOutputNames → logitsOnly), the native handle is freed
     *  and recompiled. These saved params allow automatic KV retention re-configuration
     *  on the new native handle, preventing the scatter from being silently lost. */
    private List<String> savedKvPresentOutputNames;
    private List<String> savedKvPastInputNames;
    private int savedKvMaxLen;
    private int savedKvCurrentPos;

    /** Cached OpaqueContext for native execution. Reused across executeNative() calls
     *  to avoid JNI create/delete overhead (~1-2ms). Freed on close(). */
    private OpaqueContext cachedOpContext;
    private int cachedOpContextInputCount;
    private int cachedOpContextOutputCount;

    /** Zero-copy output cache: when shapesFrozen, wraps C++ output pointers via
     *  dbCreateExternalDataBuffer instead of allocating + copyBuffer per step.
     *  These INDArrays point directly to C++ memory and must NOT be closed by callers.
     *  Cleared on close() and when setShapesFrozen(false) is called. */
    private Map<String, INDArray> zeroCopyOutputCache;

    /** Cached OpaqueNDArray wrappers for external inputs when shapesFrozen.
     *  Avoids recreating wrappers + JNI setGraphContextInputArray calls each step.
     *  Only inputs that changed (by INDArray identity) are re-sent to C++. */
    private OpaqueNDArray[] cachedInputOpaques;
    private INDArray[] cachedInputArrays;

    /** Strong references to ALL OpaqueNDArrays currently registered in the C++ context.
     *  Prevents GC from collecting OpaqueNDArray wrappers (and thus deleting the C++ NDArray
     *  objects they wrap) while the C++ context holds raw NDArray* pointers to them.
     *  Without this, the DeallocatorService can delete C++ NDArrays between steps,
     *  leaving dangling pointers that cause SIGSEGV or "db=(nil)" stale buffer errors.
     *  Always populated after setting inputs, regardless of frozen/non-frozen state. */
    private OpaqueNDArray[] contextInputRefs;

    /** Bitmap: true for external inputs that are placeholders (may be modified on host).
     *  Only these need syncToSpecial on the frozen fast path. Constants never change. */
    private boolean[] inputIsPlaceholder;

    /** Cached indices of placeholder inputs. Built on first frozen call to avoid
     *  iterating all 1332 external inputs every step. Only ~3 are placeholders
     *  (input_ids, attention_mask, position_ids). Saves ~0.5-1ms per step. */
    private int[] placeholderIndices;

    /** True once dummy outputs have been set on the context for frozen execution.
     *  After the first frozen call, C++ manages its own output slots — skip dummy setup. */
    private boolean frozenOutputsInitialized;

    /** Cached requested output names list — avoids allocating a new ArrayList per step. */
    private List<String> cachedRequestedOutputNames;

    /** Count of frozen executeNative() calls. Used to force full input re-set on the
     *  first few calls (warmup + Triton compile) to prevent stale OpaqueNDArray pointers. */
    private int frozenCallCount;

    /** Cached execution stream pointer. Avoids 2 JNI calls per step. */
    private Pointer cachedExecStream;
    private boolean execStreamCached;

    /** Device ID where this DSP executor runs native execution. Determined from the
     *  majority device of external inputs on first executeNative() call. For multi-GPU
     *  scenarios (e.g., draft model on device 1 while target model on device 0), the
     *  entire DSP including CUDA graph capture/replay happens on this device.
     *  -1 means not yet determined. */
    private int nativeExecutionDevice = -1;

    /** Cache of constant replicas for the native execution path. Keyed by external input
     *  index. When constants live on a different device than nativeExecutionDevice (e.g.,
     *  draft model weights on device 1, execution on device 0), replicateToDevice() copies
     *  them to the execution device. Cached here to avoid re-copying 100s of MB of weights
     *  on every decode step. Cleared on close() and when nativeExecutionDevice changes. */
    private Map<Integer, INDArray> nativeConstantReplicaCache;

    /** Maximum KV cache length for pre-allocation. When > 0 and CUDA graphs enabled,
     *  output slots for KV cache are pre-allocated at max size to keep addresses stable.
     *  Can be set programmatically via setMaxKvCacheLength() or via system property
     *  {@link ND4JSystemProperties#DSP_MAX_KV_CACHE_LENGTH}. */
    private int maxKvCacheLength = Integer.parseInt(
            System.getProperty(ND4JSystemProperties.DSP_MAX_KV_CACHE_LENGTH, "0"));

    /** True once max-allocation has been configured (done after the first execution step). */
    private boolean maxAllocationConfigured = false;

    public DynamicShapePlanExecutor(SameDiff sd, SessionMemMgr mmgr) {
        this.sd = sd;
        this.mmgr = mmgr;
    }

    /**
     * Compute a stable cache key for a plan's compiled native handle.
     * Uses sorted requested output names so the same output set always maps to the same key.
     */
    private static String planCacheKey(DynamicShapePlan plan) {
        if (plan == null) return null;
        List<String> sorted = new ArrayList<>(plan.getRequestedOutputs());
        Collections.sort(sorted);
        return String.join(",", sorted);
    }

    /**
     * Initialize the executor for a specific plan.
     *
     * <p>IMPORTANT: This always clears the plan's per-slot shape caches, even if the same plan
     * is being reused. After a session reset, the cached DynamicShapePlan may contain stale
     * DataBuffer references in slot.cachedOutputShapes that point to freed GPU memory.
     * Using these stale references causes memory corruption ("double free or corruption (out)").</p>
     */
    public void initialize(DynamicShapePlan plan) {
        // ALWAYS clear shape caches to avoid stale DataBuffer references from previous sessions.
        plan.clearAllShapeCaches();

        if (currentPlan == plan) {
            // Same plan reused after resetSession() — clear cached external input state
            // to prevent stale arrays from the previous session being reused.
            // Without this, the fast-path in executeNative() would copy old
            // cachedInputArrays (e.g., old gather indices) into the new execution,
            // causing out-of-bounds errors on the gather (embedding lookup) op.
            cachedInputArrays = null;
            cachedInputOpaques = null;
            contextInputRefs = null;
            inputIsPlaceholder = null;
            placeholderIndices = null;
            // Free native plan handle to release CUDA graph replay handles, capture
            // buffers, and workspaces. These hold gigabytes of GPU memory (capture
            // workspace ~512MB, cuBLAS workspace ~256MB, captured graph state, output
            // slot arrays). Without this, resetSession() recovers almost no GPU memory
            // and subsequent pages OOM.
            freeNativePlanHandle();
            nativeExecutorFailed = false;
            executionCount = 0;
            shapesFrozen = false;
            nativeExecutionDevice = -1;
            return;
        }
        // Plan changed — flush old slot cache when switching plans
        closeSlotArrayCache();
        closeZeroCopyOutputCache();
        closeNativeConstantReplicaCache();
        currentPlan = plan;
        int totalSlots = plan.getTotalOutputSlots();
        outputSlots = new INDArray[totalSlots];
        externalInputs = new INDArray[plan.getExternalInputKeys().length];
        pendingClose = new ArrayList<>();
        slotArrayCache = new INDArray[totalSlots];
        // Reset native executor state for new plan
        freeNativePlanHandle();
        nativeExecutorFailed = false;

        // Cache device count once.
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            cachedNumDevices = nativeOps.getAvailableDevices();
        } catch (Exception e) {
            cachedNumDevices = 1;
        }
    }

    private int closeOwnedArrays(Iterable<INDArray> arrays, boolean makeCloseable) {
        if (arrays == null) {
            return 0;
        }

        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (INDArray arr : arrays) {
            if (arr == null) {
                continue;
            }

            try {
                arr.clearOpaqueNDArray();
            } catch (Exception ignored) {
                // Best-effort cleanup only.
            }

            if (makeCloseable && !arr.closeable()) {
                try {
                    arr.setCloseable(true);
                } catch (Exception ignored) {
                    // Best-effort cleanup only.
                }
            }

            DataBuffer buf = arr.data();
            if (buf != null && !buf.wasClosed()) {
                uniqueBuffers.put(buf, Boolean.TRUE);
            }
        }

        int closed = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (buf == null || buf.wasClosed()) {
                continue;
            }

            if (buf.isConstant() && !buf.isAttached()) {
                try {
                    buf.setConstant(false);
                } catch (Exception ignored) {
                    // Best-effort cleanup only.
                }
            }

            if (buf.closeable()) {
                try {
                    buf.close();
                    closed++;
                } catch (Exception ignored) {
                    // Best-effort cleanup only.
                }
            }
        }

        return closed;
    }

    private boolean isCurrentExternalInputBuffer(DataBuffer buf) {
        if (buf == null || externalInputs == null) {
            return false;
        }

        for (INDArray input : externalInputs) {
            if (input == null || input.wasClosed()) {
                continue;
            }

            DataBuffer inputBuf = input.data();
            if (inputBuf == buf) {
                return true;
            }
        }

        return false;
    }

    private boolean isProtectedExternalBuffer(DataBuffer buf) {
        return isCurrentExternalInputBuffer(buf);
    }

    private int closeReplicaCache(Map<Integer, INDArray> cache) {
        if (cache == null || cache.isEmpty()) {
            return 0;
        }

        int closed = closeOwnedArrays(cache.values(), true);
        cache.clear();
        return closed;
    }

    private int closeNativeConstantReplicaCache() {
        int closed = closeReplicaCache(nativeConstantReplicaCache);
        nativeConstantReplicaCache = null;
        return closed;
    }

    /**
     * Clear replica caches to release GPU memory without destroying the compiled plan.
     * Call between pages/sessions to free cross-device constant copies while keeping
     * the native plan handle intact (avoiding expensive recompilation).
     * The replicas will be re-created on the next execution via lazy migration.
     */
    public void clearReplicaCaches() {
        int nativeClosed = closeNativeConstantReplicaCache();
        if (nativeClosed > 0) {
            log.info("DSP clearReplicaCaches: freed {} native replicas", nativeClosed);
        }
        // Also clear cached input arrays since they may reference freed replicas
        if (cachedInputArrays != null) {
            Arrays.fill(cachedInputArrays, null);
        }
        if (cachedInputOpaques != null) {
            Arrays.fill(cachedInputOpaques, 0L);
        }
        // Reset frozen state so inputs are re-checked on next call
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        nativeExecutionDevice = -1;
    }

    private int closeZeroCopyOutputCache() {
        if (zeroCopyOutputCache == null || zeroCopyOutputCache.isEmpty()) {
            zeroCopyOutputCache = null;
            return 0;
        }

        int closed = closeOwnedArrays(zeroCopyOutputCache.values(), true);
        zeroCopyOutputCache.clear();
        zeroCopyOutputCache = null;
        return closed;
    }

    private static final int DIRECT_SLOT_MAPPING_OFFSET = 2;

    private static int encodeDirectOutputSlot(int slotIdx) {
        return -(slotIdx + DIRECT_SLOT_MAPPING_OFFSET);
    }

    private static int findExternalInputIndex(String[] extKeys, String inputName) {
        for (int i = 0; i < extKeys.length; i++) {
            if (extKeys[i].equals(inputName)) {
                return i;
            }
        }
        return -1;
    }

    private static int findOutputSlotIndex(DynamicShapePlan plan, String outputName) {
        Integer requestedSlot = plan.getOutputNameToSlotIndex().get(outputName);
        if (requestedSlot != null) {
            return requestedSlot;
        }

        for (DynamicShapeSlot slot : plan.getSlots()) {
            String[] outputVarNames = slot.getOutputVarNames();
            int[] outputSlotIndices = slot.getOutputSlotIndices();
            if (outputVarNames == null || outputSlotIndices == null) {
                continue;
            }
            int limit = Math.min(outputVarNames.length, outputSlotIndices.length);
            for (int i = 0; i < limit; i++) {
                if (outputName.equals(outputVarNames[i])) {
                    return outputSlotIndices[i];
                }
            }
        }

        return -1;
    }

    /**
     * Configure KV cache retention in the native C++ plan.
     * After this call, executeNative() skips copying KV outputs back to Java;
     * C++ scatters new KV entries into static input buffers internally.
     *
     * <p>The mapping is resolved by output variable name, not only by requested output
     * index, so decode can request logits only while still retaining present KV slots.</p>
     *
     * @param plan               the current compiled plan
     * @param presentOutputNames ordered list of present KV output names
     * @param pastInputNames     ordered list of corresponding past_key_values input names
     * @param maxKvLen           static KV buffer size along sequence dimension
     * @param initialPos         initial cache position (prefillLen)
     * @return true if retention was configured successfully
     */
    public boolean configureKvCacheRetention(DynamicShapePlan plan,
                                             List<String> presentOutputNames,
                                             List<String> pastInputNames,
                                             int maxKvLen, int initialPos) {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) {
            log.warn("configureKvCacheRetention: native plan not yet compiled, skipping");
            return false;
        }
        if (presentOutputNames.size() != pastInputNames.size()) {
            throw new IllegalArgumentException("presentOutputNames and pastInputNames must have the same size");
        }

        String[] extKeys = plan.getExternalInputKeys();
        int numMappings = presentOutputNames.size();
        int[] mappings = new int[numMappings * 3];
        for (int i = 0; i < numMappings; i++) {
            String presentName = presentOutputNames.get(i);
            String pastName = pastInputNames.get(i);

            int presentSlotIdx = findOutputSlotIndex(plan, presentName);
            int pastExtIdx = findExternalInputIndex(extKeys, pastName);
            if (presentSlotIdx < 0 || pastExtIdx < 0) {
                log.warn("configureKvCacheRetention: unresolved mapping present='{}' slot={} past='{}' extIdx={}",
                        presentName, presentSlotIdx, pastName, pastExtIdx);
                return false;
            }

            mappings[i * 3] = encodeDirectOutputSlot(presentSlotIdx);
            mappings[i * 3 + 1] = pastExtIdx;
            mappings[i * 3 + 2] = 2;  // seqDim is always 2 for [B,H,S,D]
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        IntPointer mappingsPtr = new IntPointer(mappings);
        try {
            nativeOps.configurePlanKvCacheRetention(
                    nativePlanHandle, mappingsPtr, numMappings, maxKvLen, initialPos);
        } finally {
            mappingsPtr.close();
        }

        this.kvCacheRetentionConfigured = true;
        this.kvRetentionOutputNames = new HashSet<>(presentOutputNames);

        // Save configuration for re-application after plan recompilation.
        // When the plan changes (e.g., fullOutputNames → logitsOnly), the native handle is
        // freed and recompiled. Without saving, KV retention silently disappears.
        this.savedKvPresentOutputNames = new ArrayList<>(presentOutputNames);
        this.savedKvPastInputNames = new ArrayList<>(pastInputNames);
        this.savedKvMaxLen = maxKvLen;
        this.savedKvCurrentPos = initialPos;

        log.info("KV cache retention configured: {} mappings, maxLen={}, initialPos={}, retainedOutputs={}",
                numMappings, maxKvLen, initialPos, kvRetentionOutputNames.size());
        return true;
    }

    /**
     * Re-apply saved KV cache retention configuration on a new native plan handle.
     * Called automatically by compileNativePlan() when the plan changes but KV retention
     * was previously configured. The present output names must exist in the new plan
     * (they may map to different slot indices).
     */
    private void reapplyKvCacheRetention(DynamicShapePlan plan) {
        if (savedKvPresentOutputNames == null || nativePlanHandle == null) return;

        // Temporarily clear the flag so configureKvCacheRetention can set it fresh
        this.kvCacheRetentionConfigured = false;
        boolean ok = configureKvCacheRetention(plan,
                savedKvPresentOutputNames, savedKvPastInputNames,
                savedKvMaxLen, savedKvCurrentPos);
        if (!ok) {
            log.warn("KV cache retention re-apply failed on new plan — present outputs may not exist in logits-only plan. " +
                     "C++ scatter will be disabled for this plan.");
            // Clear saved state so we don't keep trying
            this.kvCacheRetentionConfigured = false;
        }
    }

    /**
     * Advance the native KV cache position by 1.
     * @return new position
     */
    public int advanceKvCachePosition() {
        if (nativePlanHandle == null || !kvCacheRetentionConfigured) return -1;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int newPos = nativeOps.advancePlanKvCachePosition(nativePlanHandle);
        savedKvCurrentPos = newPos;  // Keep saved pos in sync for plan recompilation
        return newPos;
    }

    /**
     * Reset the native KV cache position.
     */
    public void resetKvCachePosition(int newPos) {
        if (nativePlanHandle == null || !kvCacheRetentionConfigured) return;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.resetPlanKvCachePosition(nativePlanHandle, newPos);
        savedKvCurrentPos = newPos;  // Keep saved pos in sync for plan recompilation
    }

    /**
     * Get the native plan handle for direct JNI calls.
     */
    public Pointer getNativePlanHandle() {
        return nativePlanHandle;
    }

    // ── Decode input direct-update (zero putScalar) ──────────────────────

    private boolean decodeInputsConfigured = false;

    /** External input indices for decode inputs — C++ manages these on device,
     *  so Java must NOT syncToSpecial (which would overwrite device with stale host). */
    private int decodeInputIdsExtIdx = -1;
    private int decodePositionIdsExtIdx = -1;
    private int decodeAttentionMaskExtIdx = -1;

    /**
     * Configure decode input indices for direct device-side updates.
     * Call once after plan compilation. After this, {@link #updateDecodeInputs}
     * writes input_ids, position_ids, and attention_mask directly on device
     * memory — no JNI putScalar, no host↔device round-trips.
     *
     * @param plan      The compiled plan (for external input name→index mapping)
     * @param maxKvLen  Maximum KV cache length
     */
    public void configureDecodeInputs(DynamicShapePlan plan, int maxKvLen) {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) {
            log.warn("configureDecodeInputs: native plan not yet compiled, skipping");
            return;
        }
        String[] extKeys = plan.getExternalInputKeys();
        int inputIdsIdx = -1, posIdsIdx = -1, attnMaskIdx = -1;
        for (int i = 0; i < extKeys.length; i++) {
            switch (extKeys[i]) {
                case "input_ids":      inputIdsIdx = i; break;
                case "position_ids":   posIdsIdx = i; break;
                case "attention_mask": attnMaskIdx = i; break;
            }
        }
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.configurePlanDecodeInputs(nativePlanHandle,
                inputIdsIdx, posIdsIdx, attnMaskIdx, maxKvLen);
        decodeInputsConfigured = true;
        decodeInputIdsExtIdx = inputIdsIdx;
        decodePositionIdsExtIdx = posIdsIdx;
        decodeAttentionMaskExtIdx = attnMaskIdx;
        log.info("Decode inputs configured: inputIds={} posIds={} attnMask={} maxKvLen={}",
                inputIdsIdx, posIdsIdx, attnMaskIdx, maxKvLen);
    }

    /**
     * Set the next decode token and cache position. Call before execute().
     * The C++ execute() will write tokenId → input_ids, cachePos → position_ids,
     * and attention_mask[cachePos-1] = 1 directly on device memory.
     * Single JNI call, zero host↔device round-trips.
     *
     * @param tokenId   Next token ID
     * @param cachePos  Current cache position
     */
    public void setNextDecodeToken(long tokenId, int cachePos) {
        if (!decodeInputsConfigured || nativePlanHandle == null) return;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.setPlanNextDecodeToken(nativePlanHandle, tokenId, cachePos);
    }

    /**
     * Check if decode inputs have been configured.
     */
    public boolean isDecodeInputsConfigured() {
        return decodeInputsConfigured;
    }

    /**
     * Enable/disable "shapes frozen" mode on the native plan.
     * When frozen, shape inference and cache clearing are skipped between executions.
     * Use during static KV decode where all external input shapes are guaranteed constant.
     * The first execution after enabling will still do full shape inference to populate
     * the cache; subsequent executions skip shape work entirely.
     */
    public void setShapesFrozen(boolean frozen) {
        boolean wasFrozen = this.shapesFrozen;
        this.shapesFrozen = frozen;
        if (frozen && !wasFrozen) {
            log.info("FROZEN_TRANSITION: unfrozen → FROZEN (frozenCallCount reset, plan={})",
                    nativePlanHandle != null && !nativePlanHandle.isNull() ? "native" : "java");
            DspDiagnostics.record(DspDiagnostics.SHAPE,
                    "Java: shapes FROZEN (executionCount=" + executionCount + ")");
        } else if (!frozen && wasFrozen) {
            log.info("FROZEN_TRANSITION: FROZEN → unfrozen (caches cleared)");
            DspDiagnostics.record(DspDiagnostics.SHAPE,
                    "Java: shapes UNFROZEN (executionCount=" + executionCount + ")");
        }
        // Always clear frozen-state caches on ANY transition (freeze or unfreeze).
        // When entering frozen mode, stale caches from a previous plan/seqLen would cause
        // shape mismatches (e.g., zeroCopyOutputCache has [1,576] from seqLen=1 but new plan
        // needs [6,576] for seqLen=6). When leaving frozen mode, caches must also be cleared
        // so the next execution does full shape inference.
        {
            closeZeroCopyOutputCache();
            cachedInputOpaques = null;
            cachedInputArrays = null;
            contextInputRefs = null;
            inputIsPlaceholder = null;
            placeholderIndices = null;
            frozenOutputsInitialized = false;
            frozenCallCount = 0;
            cachedExecStream = null;
            execStreamCached = false;
        }
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanShapesFrozen(nativePlanHandle, frozen);
        }
    }

    public boolean isShapesFrozen() {
        return shapesFrozen;
    }

    /**
     * Reset executor state for next-page reuse WITHOUT destroying the native plan handle.
     * Clears all cached inputs, slot arrays, and unfreezes shapes so the next execution
     * does full shape inference (needed for prefill with different seq_len).
     * The native plan handle (compiled Triton kernels + CUDA graph) is preserved.
     */
    public void resetForNextPage() {
        log.info("DSP resetForNextPage: clearing caches, preserving native plan handle");
        if (shapesFrozen) {
            setShapesFrozen(false);
        }
        cachedInputArrays = null;
        cachedInputOpaques = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        placeholderIndices = null;
        closeSlotArrayCache();
        closeZeroCopyOutputCache();
        if (currentPlan != null) {
            currentPlan.clearAllShapeCaches();
        }
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        nativeExecutorFailed = false;
        executionCount = 0;
        nativeExecutionDevice = -1;
    }

    /**
     * Sets an optional interceptor that is called after each slot execution.
     * The interceptor receives the output array directly — implementations
     * must {@code dup()} arrays they want to retain.
     * Pass {@code null} to disable (default).
     * @deprecated Only functional with the removed Java slot-by-slot execution path.
     */
    @Deprecated
    public void setSlotOutputInterceptor(SlotOutputInterceptor interceptor) {
        this.slotOutputInterceptor = interceptor;
    }

    /**
     * @deprecated Only functional with the removed Java slot-by-slot execution path.
     */
    @Deprecated
    public SlotOutputInterceptor getSlotOutputInterceptor() {
        return slotOutputInterceptor;
    }

    /**
     * Enable/disable execution timing breakdown logging on the native plan.
     */
    public void setExecutionTimingEnabled(boolean enabled) {
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanExecutionTimingEnabled(nativePlanHandle, enabled);
        }
    }

    /**
     * Enable/disable trace logging for DSP execution decisions.
     */
    public void setTraceEnabled(boolean enabled) {
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanTraceEnabled(nativePlanHandle, enabled);
        }
    }

    /**
     * Set the maximum KV cache length for pre-allocation.
     * When set > 0 and CUDA graphs are enabled, output slots for KV cache
     * are pre-allocated at max size [batch, numHeads, maxLen, headDim] to keep
     * buffer addresses stable across decode steps. This enables CUDA graph capture.
     * 
     * Must be called before the first execute() call.
     * 
     * @param maxLen Maximum sequence length for KV cache
     */
    public void setMaxKvCacheLength(int maxLen) {
        this.maxKvCacheLength = maxLen;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanMaxKvCacheLength(nativePlanHandle, maxLen);
        }
    }

    /**
     * Get the configured maximum KV cache length.
     */
    public int getMaxKvCacheLength() {
        return maxKvCacheLength;
    }

    /**
     * Whether KV cache retention is configured.
     */
    public boolean isKvCacheRetentionConfigured() {
        return kvCacheRetentionConfigured;
    }

    /**
     * Get the currently compiled DynamicShapePlan (if any).
     */
    public DynamicShapePlan getCurrentPlan() {
        return currentPlan;
    }

    /**
     * Whether a native plan handle is already compiled for the given plan.
     */
    public boolean isNativePlanCompiled(DynamicShapePlan plan) {
        return nativePlanHandle != null && !nativePlanHandle.isNull() && nativePlanSource == plan;
    }

    /**
     * The graph execution mode currently configured on the native plan handle.
     */
    public GraphExecutionMode getConfiguredGraphExecutionMode() {
        return configuredGraphExecutionMode;
    }

    /**
     * Compile (or reuse) the native plan handle and configure graph execution mode.
     *
     * @param plan the DynamicShapePlan to compile
     * @param requestedMode desired graph execution mode, or null to use SameDiff/system property
     * @param fallbackToAutoIfTritonUnavailable if true, TRITON mode degrades to AUTO when Triton is unavailable
     * @return the effective mode configured on the native plan
     */
    public GraphExecutionMode compileNativePlan(DynamicShapePlan plan,
                                                GraphExecutionMode requestedMode,
                                                boolean fallbackToAutoIfTritonUnavailable) {
        if (currentPlan != plan) {
            initialize(plan);
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        boolean planChanged = nativePlanSource != null && nativePlanSource != plan;

        if (nativePlanHandle == null || nativePlanSource != plan) {
            if (planChanged && cudaGraphsFailed) {
                log.info("Native executor: resetting cudaGraphsFailed on plan recompilation");
                cudaGraphsFailed = false;
            }

            freeNativePlanHandle();

            // Check SameDiff's cache for a previously compiled handle with the same outputs
            String cacheKey = planCacheKey(plan);
            Pointer cachedHandle = sd != null ? sd.getCachedNativePlanHandle(cacheKey) : null;
            if (cachedHandle != null) {
                nativePlanHandle = cachedHandle;
                nativePlanSource = plan;
                nativeExecutorFailed = false;
                // Tell C++ to clear cached output slots and redo shape inference.
                // The previous session's output buffers are freed — the plan must
                // re-allocate them on the next execution.
                nativeOps.setPlanShapesFrozen(nativePlanHandle, false);
                shapesFrozen = false;
                maxAllocationConfigured = false;
                log.info("Native executor: restored cached plan handle (key={}, slots={}, inputs={})",
                        cacheKey, plan.getSlots().length, plan.getExternalInputKeys().length);
                // Re-apply KV cache retention if it was configured before
                if (savedKvPresentOutputNames != null && !savedKvPresentOutputNames.isEmpty()) {
                    boolean reapplied = configureKvCacheRetention(plan, savedKvPresentOutputNames,
                            savedKvPastInputNames, savedKvMaxLen, savedKvCurrentPos);
                    log.info("Native executor: KV retention re-applied on cached handle: {}", reapplied);
                }
                // Fall through to mode configuration below
            } else {
                // No cached handle — compile from scratch
                byte[] serialized = plan.serialize();
                if (serialized == null || serialized.length == 0) {
                    nativeExecutorFailed = true;
                    throw new RuntimeException("Native executor: plan serialization returned empty. " +
                            "Cannot compile native plan. No fallback permitted.");
                }

                BytePointer planBytes = new BytePointer(serialized);
                try {
                    nativePlanHandle = nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
                } catch (UnsupportedOperationException e) {
                    nativeExecutorFailed = true;
                    throw new RuntimeException("Native executor: backend does not support compileDynamicShapePlan. " +
                            "No fallback permitted.", e);
                } finally {
                    planBytes.close();
                }

                if (nativePlanHandle == null || nativePlanHandle.isNull()) {
                    nativePlanHandle = null;
                    nativeExecutorFailed = true;
                    throw new RuntimeException("Native executor: compileDynamicShapePlan returned null handle. " +
                            "No fallback permitted.");
                }

                boolean cudaGraphsEnabled = !cudaGraphsFailed && !"false".equalsIgnoreCase(
                        System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "true"));
                if (cudaGraphsEnabled) {
                    try {
                        nativeOps.setPlanCudaGraphsEnabled(nativePlanHandle, true);
                        DspDiagnostics.record(DspDiagnostics.COMPILE,
                                "Java: CUDA graphs ENABLED on native plan");
                    } catch (UnsupportedOperationException e) {
                        DspDiagnostics.record(DspDiagnostics.COMPILE,
                                "Java: CUDA graphs not supported by backend (CPU?)");
                    }
                } else {
                    DspDiagnostics.record(DspDiagnostics.COMPILE,
                            "Java: CUDA graphs DISABLED (cudaGraphsFailed=" + cudaGraphsFailed + ")");
                }

                String jitModeStr = System.getProperty(ND4JSystemProperties.DSP_JIT_MODE, "graph");
                if (!"graph".equalsIgnoreCase(jitModeStr)) {
                    int jitModeInt = 0;  // GRAPH_ONLY
                    if ("jit".equalsIgnoreCase(jitModeStr)) {
                        jitModeInt = 1;  // JIT_ONLY
                    } else if ("graph+jit".equalsIgnoreCase(jitModeStr)) {
                        jitModeInt = 2;  // GRAPH_PLUS_JIT
                    }
                    try {
                        nativeOps.setPlanJitMode(nativePlanHandle, jitModeInt);
                        log.info("Native executor: JIT mode set to {} ({})", jitModeStr, jitModeInt);
                    } catch (UnsupportedOperationException e) {
                        // Backend doesn't support JIT
                    }
                }

                boolean execTiming = "true".equalsIgnoreCase(
                        System.getProperty(ND4JSystemProperties.DSP_EXECUTION_TIMING, "false"));
                if (execTiming) {
                    try {
                        nativeOps.setPlanExecutionTimingEnabled(nativePlanHandle, true);
                    } catch (UnsupportedOperationException e) {
                        // Backend doesn't support timing
                    }
                }

                if (System.getProperty(ND4JSystemProperties.DSP_TRACE) != null) {
                    try {
                        nativeOps.setPlanTraceEnabled(nativePlanHandle, true);
                    } catch (UnsupportedOperationException e) {
                        // Backend doesn't support trace
                    }
                }

                nativePlanSource = plan;
                nativeExecutorFailed = false;
                log.info("Native executor: compiled plan with {} slots, {} external inputs, {} outputs (cudaGraphs={}, shapesFrozen={})",
                        plan.getSlots().length, plan.getExternalInputKeys().length,
                        plan.getRequestedOutputs().size(), cudaGraphsEnabled, false);
                DspDiagnostics.record(DspDiagnostics.COMPILE,
                        "Java: compiled native plan " + plan.getSlots().length + " slots, " +
                        plan.getExternalInputKeys().length + " inputs, " +
                        plan.getRequestedOutputs().size() + " outputs");

                // Re-apply KV cache retention if it was previously configured but lost.
                if (savedKvPresentOutputNames != null && !kvCacheRetentionConfigured) {
                    log.info("Native executor: re-applying KV cache retention on new plan (pos={})", savedKvCurrentPos);
                    reapplyKvCacheRetention(plan);
                }
            }
        }

        GraphExecutionMode resolvedMode = resolveRequestedGraphExecutionMode(requestedMode);
        boolean tritonAvailable = isTritonAvailable(nativeOps);
        GraphExecutionMode effectiveMode = resolvedMode;
        if (effectiveMode == GraphExecutionMode.TRITON &&
                fallbackToAutoIfTritonUnavailable &&
                !tritonAvailable) {
            log.warn("Native executor: TRITON mode requested but Triton is unavailable; falling back to AUTO");
            effectiveMode = GraphExecutionMode.AUTO;
        }

        try {
            nativeOps.setPlanGraphExecutionMode(nativePlanHandle, effectiveMode.getNativeCode());
            configuredGraphExecutionMode = effectiveMode;
            log.info("Native executor: mode resolution requested={} resolved={} effective={} tritonAvailable={} fallbackToAuto={}",
                    requestedMode, resolvedMode, effectiveMode, tritonAvailable, fallbackToAutoIfTritonUnavailable);
            DspDiagnostics.record(DspDiagnostics.BACKEND,
                    "Java: mode resolution requested=" + requestedMode + " effective=" + effectiveMode +
                    " triton=" + tritonAvailable);
        } catch (UnsupportedOperationException e) {
            configuredGraphExecutionMode = GraphExecutionMode.AUTO;
        }

        return effectiveMode;
    }

    private GraphExecutionMode resolveRequestedGraphExecutionMode(GraphExecutionMode requestedMode) {
        if (requestedMode != null) {
            return requestedMode;
        }

        GraphExecutionMode gem = sd.getGraphExecutionMode();
        String gemStr = System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE);
        if (gemStr != null) {
            try {
                try {
                    gem = GraphExecutionMode.valueOf(gemStr.toUpperCase());
                } catch (IllegalArgumentException nameEx) {
                    gem = GraphExecutionMode.fromNativeCode(Integer.parseInt(gemStr));
                }
            } catch (Exception e) {
                log.warn("Invalid graph execution mode '{}', using {}", gemStr, gem);
            }
        }
        return gem;
    }

    private boolean isTritonAvailable(NativeOps nativeOps) {
        try {
            return nativeOps.isTritonAvailable();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Configure max-allocation for KV cache output slots.
     * Called after the first execution step when actual output shapes are known.
     * Finds present_key/present_value outputs and configures C++ to pre-allocate
     * them at maximum sequence length so buffer addresses stay stable for CUDA graphs.
     */
    private void configureMaxAllocationForKvCache(Map<String, INDArray> firstStepResults, DynamicShapePlan plan) {
        if (nativePlanHandle == null || nativePlanHandle.isNull() || maxKvCacheLength <= 0) return;
        if (firstStepResults == null || firstStepResults.isEmpty()) return;

        Map<String, Integer> outputNameToSlot = plan.getOutputNameToSlotIndex();

        List<Integer> kvSlotIndices = new ArrayList<>();
        List<Long> kvMaxSizes = new ArrayList<>();

        // Get shapes from actual output arrays returned by the first execution step.
        // Match logic mirrors DecoderUtils.findKVCacheOutputNames: present+key or present+value.
        for (Map.Entry<String, INDArray> entry : firstStepResults.entrySet()) {
            String outputName = entry.getKey();
            boolean isKvKey   = outputName.contains("present") && outputName.contains("key");
            boolean isKvValue = outputName.contains("present") && outputName.contains("value");
            if (isKvKey || isKvValue) {
                Integer slotIdx = outputNameToSlot.get(outputName);
                if (slotIdx != null && slotIdx >= 0) {
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.rank() == 4) {
                        // Shape is [batch, numHeads, seqLen, headDim]
                        long batchSize = arr.size(0);
                        long numHeads  = arr.size(1);
                        long headDim   = arr.size(3);
                        long maxSize   = batchSize * numHeads * maxKvCacheLength * headDim;

                        kvSlotIndices.add(slotIdx);
                        kvMaxSizes.add(maxSize);
                        log.debug("Max-allocating KV cache slot {} ({}): shape={} -> maxSize={}",
                                slotIdx, outputName, Arrays.toString(arr.shape()), maxSize);
                    }
                }
            }
        }

        if (!kvSlotIndices.isEmpty()) {
            int[] indices = kvSlotIndices.stream().mapToInt(Integer::intValue).toArray();
            long[] sizes   = kvMaxSizes.stream().mapToLong(Long::longValue).toArray();

            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            // Use the int[]/long[] overload — the Pointer overload dispatches to a no-op default
            nativeOps.setPlanOutputSlotMaxSizes(nativePlanHandle, indices.length, indices, sizes);
            nativeOps.setPlanMaxKvCacheLength(nativePlanHandle, maxKvCacheLength);
            log.info("Configured max-allocation for {} KV cache slots with maxSeqLen={}",
                    kvSlotIndices.size(), maxKvCacheLength);
        }
    }

    /**
     * Execute the plan with the given placeholder arrays.
     *
     * @param plan              the compiled plan
     * @param placeholderArrays placeholder name -> INDArray
     * @return map of requested output variable name -> INDArray
     */
    public Map<String, INDArray> execute(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        if (currentPlan != plan) {
            initialize(plan);
        }

        // Native C++ graph executor — no fallback to Java allowed.
        if (NATIVE_EXECUTOR_ENABLED) {
            if (nativeExecutorFailed) {
                throw new RuntimeException("Native DSP executor compilation previously failed. " +
                        "No fallback to Java permitted. Fix the native compilation issue.");
            }
            if (!isNativePlanCompiled(plan) && sd.isDspNativeAutoCompileEnabled()) {
                compileNativePlan(plan, null, sd.isDspFallbackToAutoIfTritonUnavailable());
            }
            // No try/catch — native executor failures must crash, not be masked
            return executeNative(plan, placeholderArrays);
        }

        // Java slot-by-slot execution path has been removed. Native execution is always used.
        throw new UnsupportedOperationException(
                "Java slot-by-slot DSP execution has been removed. " +
                "Native execution (NATIVE_EXECUTOR_ENABLED) must be true.");
    }

    /**
     * Core buffer freeing logic shared by flushPendingClose() and closePendingBuffers().
     * Uses persistent dedup sets (seenIdentity, closedOdbAddresses)
     * that span all flushes within one execute() call.
     *
     * Frees GPU memory on the execution stream (not stream 0) so the pool can reuse
     * freed memory for subsequent allocations on the same stream without cross-stream sync.
     *
     * @param liveGpuAddresses Sorted GPU addresses from live slots (may be null for final close).
     *                         Owner buffers whose allocation range overlaps with any live address
     *                         are deferred — a view in a live slot still references the parent memory.
     * @return int[3] = {freedCount, totalInBatch, freedMB}
     */
    private int[] freePendingBuffers(NativeOps nativeOps, Pointer execStream, long[] liveGpuAddresses) {
        int freedCount = 0;
        long freedBytes = 0;
        int batchSize = pendingClose.size();

        // GPU address dedup is per-batch (not persistent across flushes).
        // Between flushes, the CUDA pool may reuse freed addresses for new allocations.
        // Those are legitimate new allocations that must be freed in the next flush.
        // Persistent GPU dedup would incorrectly skip them → memory leak.
        HashSet<Long> batchGpuAddresses = new HashSet<>();

        for (DataBuffer buf : pendingClose) {
            // NOTE: Don't gate on buf.closeable(). Oversized buffers from slot cache
            // growth factor have data().length() > length() → closeable()=false. But these
            // are OWNED arrays (not sub-views) that the DSP executor allocated. They must
            // be freed to prevent permanent GPU memory leaks. Only skip already-closed buffers.
            // Don't skip isConstant() — constant-poisoned intermediates have already been
            // un-poisoned at release time, but as a safety net, un-poison any remaining ones.
            if (buf == null || buf.wasClosed()) continue;
            // Never touch buffers still owned by current external inputs.
            if (isProtectedExternalBuffer(buf)) continue;
            if (buf.isConstant()) {
                buf.setConstant(false);
            }

            OpaqueDataBuffer odb = buf.opaqueBuffer();
            if (odb == null || odb.isNull()) continue;

            boolean isOwner = nativeOps.dbIsOwner(odb);
            long gpuAddr = 0;

            if (isOwner) {
                Pointer special = nativeOps.dbSpecialBuffer(odb);
                if (special != null && special.address() != 0) {
                    gpuAddr = special.address();
                }
            }

            // Check if this owner buffer's GPU memory is still used by a live slot.
            // This happens when C++ creates a view (reshape/slice/identity) — the view
            // ODB points into the parent's allocation range. If the parent slot is marked
            // dead while a view slot is still live, freeing the parent would invalidate
            // the view's memory. Range check catches both zero-offset and offset views.
            if (isOwner && gpuAddr != 0 && liveGpuAddresses != null && liveGpuAddresses.length > 0) {
                long allocSize = buf.length() * buf.getElementSize();
                if (hasLiveViewInRange(liveGpuAddresses, gpuAddr, allocSize)) {
                    deferredClose.add(buf);
                    continue;
                }
            }

            // Layer 1: Java identity dedup (persistent across flushes)
            if (!seenIdentity.add(buf)) continue;

            // Layer 2: OpaqueDataBuffer address dedup (persistent across flushes)
            long odbAddr = odb.address();
            if (odbAddr != 0 && !closedOdbAddresses.add(odbAddr)) continue;

            // Layer 3: GPU address dedup for OWNER ODBs only (per-batch).
            // Only owner ODBs will actually free GPU memory in C++. Non-owner views
            // (isOwner=false) skip this check so the actual owner can be freed later.
            // This prevents: (a) non-owner views blocking owner frees (the old leak bug),
            // and (b) two different owner ODBs double-freeing the same GPU address.
            if (isOwner && gpuAddr != 0 && !batchGpuAddresses.add(gpuAddr)) continue;

            try {
                long bufBytes = buf.length() * buf.getElementSize();
                freedBytes += bufBytes;
                freedCount++;
                MultiGpuTracer.traceBufferFree(-1,
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        bufBytes, execStream != null ? "dbFreeBuffersOnStream" : "dbFreeBuffersOnly");
                // Free on the execution stream so the pool can reuse this memory
                // for the next cudaMallocAsync on the same stream.
                if (execStream != null) {
                    nativeOps.dbFreeBuffersOnStream(odb, execStream);
                } else {
                    nativeOps.dbFreeBuffersOnly(odb);
                }

                // Sync Java-side lifecycle with the native close that just happened.
                odb.tryMarkForDeallocation();
                odb.setNull();
                OpaqueDataBufferDeallocator deallocator = odb.getDeallocator();
                if (deallocator != null) {
                    deallocator.markDeallocated();
                }
            } catch (Exception e) {
                log.warn("  dbFreeBuffersOnStream failed ({}B): {}",
                        buf.length() * buf.getElementSize(), e.getMessage());
            }
        }

        totalFlushedCount += freedCount;
        totalFlushedBytes += freedBytes;
        return new int[]{freedCount, batchSize, (int) (freedBytes / (1024 * 1024))};
    }

    /**
     * Check if any live GPU address falls within the allocation range [base, base+size).
     * This catches both zero-offset views (same base pointer) and offset views
     * (pointer within the parent's allocation). Uses binary search on sorted live addresses.
     */
    private static boolean hasLiveViewInRange(long[] sortedLiveAddresses, long base, long size) {
        if (sortedLiveAddresses.length == 0 || size <= 0) return false;
        long end = base + size;
        // Binary search for the first address >= base
        int idx = Arrays.binarySearch(sortedLiveAddresses, base);
        if (idx < 0) idx = -(idx + 1); // insertion point
        // Check if any address starting from idx is within [base, end)
        while (idx < sortedLiveAddresses.length && sortedLiveAddresses[idx] < end) {
            return true;
        }
        return false;
    }



    /**
     * Free GPU memory held by cached arrays in the slot cache. Routes buffers through
     * freePendingBuffers() to get full dedup protection (identity, ODB address, GPU address
     * owner-only). Without dedup, views sharing GPU memory with their parent cause double-free
     * heap corruption. Called during close() to prevent GPU memory leaks between
     * execute() calls (e.g., between vision encoder chunks).
     */
    private void closeSlotArrayCache() {
        if (slotArrayCache == null) return;
        log.info("    closeSlotArrayCache: START (length={})", slotArrayCache.length);
        System.out.flush(); System.err.flush();

        // Merge any remaining deferred buffers from mid-execution flushes.
        if (!deferredClose.isEmpty()) {
            pendingClose.addAll(deferredClose);
            deferredClose.clear();
        }

        // Collect eligible buffers from the cache into pendingClose.
        // The persistent dedup sets (seenIdentity, closedOdbAddresses) from the previous
        // execute() call will correctly skip buffers already freed during execution.
        int collected = 0;
        for (int i = 0; i < slotArrayCache.length; i++) {
            INDArray arr = slotArrayCache[i];
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed()) {
                    // Undo setCloseable(false) poisoning from directExecHelper().
                    // Session intermediates are marked constant via setCloseable(false)
                    // → setConstant(true). Without undoing this, the slot cache cannot
                    // free ANY buffers during session reset, leaking all GPU memory.
                    if (buf.isConstant() && !buf.isAttached()) {
                        try {
                            buf.setConstant(false);
                        } catch (Exception ignored) {}
                    }
                    if (!buf.isConstant()) {
                        pendingClose.add(buf);
                        collected++;
                    }
                }
            }
            slotArrayCache[i] = null;
        }

        if (collected == 0) {
            log.info("    closeSlotArrayCache: nothing to free");
            return;
        }

        log.info("    closeSlotArrayCache: collected {} buffers, calling freePendingBuffers", collected);
        System.out.flush(); System.err.flush();

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Pointer stream = DeviceMemoryManager.getInstance().getFreshExecutionStream();

        // Sync execution stream so all GPU kernels using these buffers have completed.
        Nd4j.getExecutioner().commit();

        // Free with full dedup. liveGpuAddresses=null because no slots are live during close().
        int[] stats = freePendingBuffers(nativeOps, stream, null);
        pendingClose.clear();

        log.info("    closeSlotArrayCache: freePendingBuffers done ({}/{} freed, {}MB)", stats[0], stats[1], stats[2]);
        System.out.flush(); System.err.flush();

        // Trim memory pool so freed memory is immediately available
        if (stream != null) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            nativeOps.trimMemoryPoolOnStream(currentDevice, stream);
            // Cross-device frees use stream 0 on target device — sync that stream
            for (int d = 0; d < cachedNumDevices; d++) {
                if (d != currentDevice) {
                    nativeOps.trimMemoryPoolOnStream(d, null);
                }
            }
        }
        log.info("    closeSlotArrayCache: DONE");
        System.out.flush(); System.err.flush();
    }

    /**
     * Execute the plan entirely in C++ via a single JNI call.
     *
     * <p>Requires a previously compiled native plan handle (via {@link #compileNativePlan(DynamicShapePlan, GraphExecutionMode, boolean)}
     * or session-level auto-compile). External inputs are resolved and passed as OpaqueNDArray pointers.
     * C++ handles shape inference, memory allocation, op execution, and release scheduling internally.</p>
     *
     * @return output map if native execution succeeded, or null if native execution
     *         is not available (e.g., backend doesn't support it)
     * @throws RuntimeException if native execution fails (caller should fall back to Java)
     */
    private Map<String, INDArray> executeNative(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Native plan compilation is explicit (or controlled by InferenceSession auto-compile).
        if (!isNativePlanCompiled(plan)) {
            throw new RuntimeException("Native executor: plan not precompiled for native execution. " +
                    "Ensure compileNativePlan() is called before executeNative().");
        }

        DspDiagnostics.record(DspDiagnostics.EXECUTE,
                "Java: executeNative ENTER executionCount=" + executionCount +
                " frozen=" + shapesFrozen + " slots=" + plan.getSlots().length);

         // Resolve external inputs (constants, variables, placeholders)
        // When frozen, cache constant/variable arrays and only re-resolve placeholders
        String[] extKeys = plan.getExternalInputKeys();
        // Debug: identify external input 1331 (slot -1332) for H2D sync diagnostics
        if (Nd4j.getEnvironment().isDebug() && extKeys.length > 1331) {
            log.info("EXT_INPUT_1331: name='{}' total={}", extKeys[1331], extKeys.length);
        }
        INDArray[] extInputs;
        if (shapesFrozen && cachedInputArrays != null && cachedInputArrays.length == extKeys.length) {
            DspDiagnostics.record(DspDiagnostics.EXECUTE,
                    "Java: external inputs FAST PATH (frozen, " + extKeys.length + " cached)");
            // Fast path: reuse cached constant/variable arrays, only re-resolve placeholders.
            // Use a separate array so we don't corrupt cachedInputArrays (needed for identity comparison).
            extInputs = new INDArray[extKeys.length];
            System.arraycopy(cachedInputArrays, 0, extInputs, 0, extKeys.length);
            // Re-resolve any inputs whose DataBuffer has been freed between steps.
            // This can happen when setCloseable(true)+close() is called on KV outputs
            // that share a DataBuffer with past_key_values inputs, or when deferred
            // close evicts constant DataBuffers during long Triton compilations.
            int staleCount = 0;
            int resolvedCount = 0;
            for (int i = 0; i < extKeys.length; i++) {
                if (extInputs[i] != null) {
                    DataBuffer db = extInputs[i].data();
                    if (db == null || db.wasClosed()) {
                        staleCount++;
                        SDVariable var = sd.getVariable(extKeys[i]);
                        VariableType vt = var != null ? var.getVariableType() : null;
                        if (var != null && (vt == VariableType.CONSTANT
                                || vt == VariableType.VARIABLE)) {
                            INDArray fresh = var.getArr();
                            if (fresh != null && fresh.data() != null && !fresh.data().wasClosed()) {
                                extInputs[i] = fresh;
                                // DON'T update cachedInputArrays here — let the fast path
                                // detect the change (extInputs[i] != cachedInputArrays[i])
                                // and call setGraphContextInputArray to update C++ side.
                                resolvedCount++;
                            } else {
                                DspDiagnostics.record(DspDiagnostics.FALLBACK,
                                    "Java: ext[" + i + "] '" + extKeys[i] + "' type=" + vt +
                                    " re-resolved but STILL closed (fresh=" + fresh + ")");
                            }
                        } else if (vt == VariableType.PLACEHOLDER && placeholderArrays != null) {
                            // Placeholder: re-resolve from placeholderArrays map
                            INDArray ph = placeholderArrays.get(extKeys[i]);
                            if (ph != null && ph.data() != null && !ph.data().wasClosed()) {
                                extInputs[i] = ph;
                                // DON'T update cachedInputArrays here — let the fast path
                                // detect the change (extInputs[i] != cachedInputArrays[i])
                                // and call setGraphContextInputArray to update C++ side.
                                resolvedCount++;
                            } else {
                                DspDiagnostics.record(DspDiagnostics.FALLBACK,
                                    "Java: ext[" + i + "] '" + extKeys[i] + "' type=PLACEHOLDER" +
                                    " STALE, placeholder not available in map");
                            }
                        } else {
                            DspDiagnostics.record(DspDiagnostics.FALLBACK,
                                "Java: ext[" + i + "] '" + extKeys[i] + "' type=" + vt +
                                " STALE but not CONST/VAR/PLACEHOLDER — cannot re-resolve!");
                        }
                    }
                }
            }
            if (staleCount > 0) {
                DspDiagnostics.record(DspDiagnostics.MEMORY,
                    "Java: external inputs fast path: " + staleCount + " stale, " +
                    resolvedCount + " resolved, " + (staleCount - resolvedCount) + " unresolvable");
            }
            if (placeholderArrays != null && !placeholderArrays.isEmpty()
                    && inputIsPlaceholder != null) {
                for (int i = 0; i < extKeys.length; i++) {
                    if (inputIsPlaceholder[i]) {
                        INDArray ph = placeholderArrays.get(extKeys[i]);
                        if (ph != null) {
                            extInputs[i] = ph;
                        }
                    }
                }
            }
        } else {
            DspDiagnostics.record(DspDiagnostics.EXECUTE,
                    "Java: external inputs SLOW PATH (resolving " + extKeys.length + " inputs fresh)");
            extInputs = new INDArray[extKeys.length];
            for (int i = 0; i < extKeys.length; i++) {
                String varName = extKeys[i];
                INDArray arr = null;
                if (placeholderArrays != null) {
                    arr = placeholderArrays.get(varName);
                }
                if (arr == null) {
                    SDVariable var = sd.getVariable(varName);
                    if (var != null &&
                            (var.getVariableType() == VariableType.CONSTANT ||
                                    var.getVariableType() == VariableType.VARIABLE)) {
                        arr = var.getArr();
                    }
                }
                if (arr == null) {
                    throw new RuntimeException("Native executor: missing external input '" + varName +
                            "' (index " + i + "/" + extKeys.length + "). " +
                            "All external inputs must be resolved. No fallback permitted.");
                }
                extInputs[i] = arr;
            }
        }

        // Debug: dump external input 1331 value info for H2D sync diagnostics
        if (Nd4j.getEnvironment().isDebug() && extInputs.length > 1331 && extInputs[1331] != null) {
            INDArray ext1331 = extInputs[1331];
            log.info("EXT_INPUT_1331_VALUE: shape={} dtype={} sum={} max={} min={} isAttached={}",
                    java.util.Arrays.toString(ext1331.shape()),
                    ext1331.dataType(),
                    ext1331.length() <= 1000 ? ext1331.sumNumber() : "SKIPPED",
                    ext1331.length() <= 1000 ? ext1331.maxNumber() : "SKIPPED",
                    ext1331.length() <= 1000 ? ext1331.minNumber() : "SKIPPED",
                    ext1331.isAttached());
        }

        // ATTN_DIAG + EXT_INPUT_WRITE diagnostics — only when debug is enabled
        if (log.isDebugEnabled()) {
            for (int i = 0; i < extKeys.length; i++) {
                if (extKeys[i].contains("past_key_values")) {
                    INDArray arr = extInputs[i];
                    log.debug("ATTN_DIAG_JAVA: extIdx={} name='{}' rank={} shape={} empty={} length={} dtype={}",
                            i, extKeys[i],
                            arr != null ? arr.rank() : -1,
                            arr != null ? java.util.Arrays.toString(arr.shape()) : "null",
                            arr != null ? arr.isEmpty() : true,
                            arr != null ? arr.length() : 0,
                            arr != null ? arr.dataType() : "null");
                }
            }

            int nullCount = 0, emptyCount = 0, dataCount = 0;
            for (int i = 0; i < extInputs.length; i++) {
                INDArray arr = extInputs[i];
                if (arr == null) {
                    nullCount++;
                    log.debug("EXT_INPUT_WRITE: idx={} name='{}' shape=null empty=true written=false", i, extKeys[i]);
                } else if (arr.isEmpty() || arr.length() == 0) {
                    emptyCount++;
                    log.debug("EXT_INPUT_WRITE: idx={} name='{}' shape={} empty=true written=false",
                            i, extKeys[i], java.util.Arrays.toString(arr.shape()));
                } else {
                    dataCount++;
                }
            }
            log.debug("EXT_INPUT_WRITE_SUMMARY: total={} null={} empty={} withData={}", extInputs.length, nullCount, emptyCount, dataCount);
        }

        // GATHER DIAGNOSTIC: dump external inputs that are [1,1] INT64 (likely position_ids for gather slot 0)
        for (int i = 0; i < Math.min(extInputs.length, 1333); i++) {
            INDArray arr = extInputs[i];
            if (arr != null && arr.rank() == 2 && arr.shape()[0] == 1 && arr.shape()[1] == 1
                    && arr.dataType() == DataType.INT64) {
                long val = arr.getInt(0);
                log.info("GATHER_DIAG: extIdx={} name='{}' shape=[1,1] INT64 value={} executionCount={}",
                        i, extKeys[i], val, executionCount);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // MULTI-GPU DEVICE COHERENCY
        // ═══════════════════════════════════════════════════════════════════
        // Determine which device holds the majority of external input data.
        // For multi-GPU scenarios (e.g., draft model weights on device 1 after
        // failover from OOM device 0), the entire DSP execution — including
        // CUDA graph capture and replay — must happen on the device where the
        // data lives. Non-peer GPUs cannot cross-access each other's memory.
        //
        // On first call: scan all external inputs to find majority device.
        // On subsequent calls: reuse cached device (data doesn't move).
        // ═══════════════════════════════════════════════════════════════════
        int previousDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean deviceSwitched = false;
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices > 1) {
            NativeOps nOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            if (nativeExecutionDevice < 0) {
                // First call: determine best execution device.
                // Strategy: prefer the device with the most data (data locality).
                // "Most free memory" is wrong for asymmetric multi-GPU (e.g., 24GB + 8GB):
                // the small GPU may have more free memory initially but OOMs when
                // model constants (~5GB) get replicated to it.
                // Tiebreaker: most free memory among devices with equal data.
                long[] deviceBytes = new long[numDevices];
                for (INDArray arr : extInputs) {
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        int devId = nOps.dbDeviceId(arr.data().opaqueBuffer());
                        if (devId >= 0 && devId < numDevices) {
                            deviceBytes[devId] += arr.length() * arr.data().getElementSize();
                        }
                    }
                }

                // Default to the device with the most TOTAL memory (the primary GPU).
                // Do NOT use previousDevice or data-locality heuristics because:
                // - previousDevice may be a small secondary GPU set during warmup
                // - dbDeviceId() is unreliable (often reports 0MB for device 0 even
                //   when GB of model weights are there)
                // The primary GPU (most VRAM) is always the safest choice: it can
                // hold model constants + workspace without OOM.
                int bestDevice = 0;
                long bestTotal = nOps.getDeviceTotalMemory(0);
                for (int d = 1; d < numDevices; d++) {
                    long totalMem = nOps.getDeviceTotalMemory(d);
                    if (totalMem > bestTotal) {
                        bestDevice = d;
                        bestTotal = totalMem;
                    }
                }
                long bestFree = nOps.getDeviceFreeMemory(bestDevice);
                nativeExecutionDevice = bestDevice;
                {
                    StringBuilder sb = new StringBuilder("DSP device selection: ");
                    for (int d = 0; d < numDevices; d++) {
                        long freeMem = nOps.getDeviceFreeMemory(d);
                        long totalMem = nOps.getDeviceTotalMemory(d);
                        sb.append("dev").append(d).append("(data=").append(deviceBytes[d] / (1024 * 1024))
                                .append("MB free=").append(freeMem / (1024 * 1024))
                                .append("MB total=").append(totalMem / (1024 * 1024)).append("MB) ");
                    }
                    sb.append("-> execution on device ").append(bestDevice);
                    log.info(sb.toString());
                }
                if (nativeExecutionDevice != previousDevice) {
                    log.info("DSP native executor: best device={} ({}MB free), switching from device {}",
                            nativeExecutionDevice,
                            bestFree / (1024 * 1024),
                            previousDevice);
                }
            }

            if (nativeExecutionDevice != previousDevice) {
                // Switch to the target device — this changes CUDA context + ContextBuffers
                DeviceMemoryManager.getInstance().switchDevice(nativeExecutionDevice,
                        "DSP.executeNative", "multi-gpu-coherency");
                deviceSwitched = true;

                // Invalidate cached exec stream — it belongs to the previous device
                cachedExecStream = null;
                execStreamCached = false;

                // Migrate any off-device inputs to the target device.
                // For non-peer GPUs, cross-device memory access causes error 700.
                // Use replicateToDevice() instead of dup() because dup() does a direct
                // GPU-to-GPU cudaMemcpy which requires peer access. replicateToDevice()
                // stages through host memory for non-peer GPUs.
                // Cache constant replicas to avoid re-copying model weights every step.
                if (nativeConstantReplicaCache == null) {
                    nativeConstantReplicaCache = new HashMap<>();
                }
                int migratedCount = 0;
                long migratedBytes = 0;
                for (int i = 0; i < extInputs.length; i++) {
                    INDArray arr = extInputs[i];
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        int arrDevice = nOps.dbDeviceId(arr.data().opaqueBuffer());
                        if (arrDevice >= 0 && arrDevice != nativeExecutionDevice) {
                            // Cache any non-placeholder input that doesn't change between
                            // decoder steps. Both CONSTANT and VARIABLE (model weights) are
                            // stable — only placeholders (input_ids, attention_mask, etc.)
                            // change shape/value per step.
                            boolean isPlaceholder = placeholderArrays != null
                                    && extKeys[i] != null && placeholderArrays.containsKey(extKeys[i]);
                            boolean isCacheable = !isPlaceholder;

                            if (isCacheable) {
                                INDArray cached = nativeConstantReplicaCache.get(i);
                                if (cached != null && !cached.wasClosed()
                                        && cached.data() != null && !cached.data().wasClosed()) {
                                    extInputs[i] = cached;
                                    if (cachedInputArrays != null && i < cachedInputArrays.length) {
                                        cachedInputArrays[i] = cached;
                                    }
                                    continue;
                                }
                            }

                            // Cross-device migration via replicateToDevice (handles non-peer GPUs)
                            INDArray migrated = Nd4j.getAffinityManager().replicateToDevice(
                                    nativeExecutionDevice, arr);
                            extInputs[i] = migrated;
                            migratedCount++;
                            migratedBytes += arr.length() * arr.data().getElementSize();

                            // Cache non-placeholder replicas for reuse across decode steps
                            if (isCacheable) {
                                nativeConstantReplicaCache.put(i, migrated);
                            }

                            // Update frozen cache
                            if (cachedInputArrays != null && i < cachedInputArrays.length) {
                                cachedInputArrays[i] = migrated;
                            }
                        }
                    }
                }
                if (migratedCount > 0) {
                    log.info("DSP native executor: migrated {} inputs ({}MB) to device {}, replicaCache={}",
                            migratedCount, migratedBytes / (1024 * 1024), nativeExecutionDevice,
                            nativeConstantReplicaCache != null ? nativeConstantReplicaCache.size() : 0);
                }
            }
        }

        // Wrap remaining execution in try/finally to restore device if we switched
        try {

        // Reuse OpaqueContext across calls to avoid JNI create/delete overhead.
        // Only recreate if input/output count changes.
        int numOutputs = plan.getRequestedOutputs().size();
        int numInputs = extInputs.length;
        if (cachedOpContext == null || numInputs != cachedOpContextInputCount || numOutputs != cachedOpContextOutputCount) {
            if (cachedOpContext != null) {
                nativeOps.deleteGraphContext(cachedOpContext);
            }
            cachedOpContext = nativeOps.createGraphContext(1);
            cachedOpContextInputCount = numInputs;
            cachedOpContextOutputCount = numOutputs;
        }
        OpaqueContext opContext = cachedOpContext;
        {
            // Set inputs on context — when frozen, only update inputs that changed.
            if (shapesFrozen && cachedInputOpaques != null
                    && cachedInputOpaques.length == extInputs.length) {
                // Fast path: only re-set inputs where the INDArray identity changed.
                // For same-identity placeholder inputs (modified via putScalar), sync to device.
                // Constants/variables are never modified on host — skip sync entirely.
                
                // Build placeholderIndices on first frozen call (saves ~0.5-1ms per step)
                if (placeholderIndices == null && inputIsPlaceholder != null) {
                    int count = 0;
                    for (boolean b : inputIsPlaceholder) if (b) count++;
                    placeholderIndices = new int[count];
                    int idx = 0;
                    for (int i = 0; i < inputIsPlaceholder.length; i++) {
                        if (inputIsPlaceholder[i]) placeholderIndices[idx++] = i;
                    }
                    log.info("FROZEN_INPUT_OPT: built placeholderIndices[{}] (extInputs={})", 
                            count, extInputs.length);
                }
                
                // Frozen fast path: only iterate placeholder indices (not all 1332 inputs)
                if (placeholderIndices != null) {
                    DspDiagnostics.record(DspDiagnostics.EXECUTE,
                        "Java: FROZEN_FAST_PATH entering placeholder loop, " +
                        placeholderIndices.length + " placeholders");
                    for (int pi : placeholderIndices) {
                        // When C++ manages decode inputs, skip syncToSpecial for them.
                        // syncToSpecial copies HOST→DEVICE, but C++ updateDecodeInputs
                        // accumulates values on DEVICE directly. Syncing would overwrite
                        // device with stale host data (e.g., wipe accumulated attention_mask).
                        if (decodeInputsConfigured && (pi == decodeInputIdsExtIdx
                                || pi == decodePositionIdsExtIdx
                                || pi == decodeAttentionMaskExtIdx)) {
                            continue;
                        }
                        INDArray arr = extInputs[pi];
                        // ALWAYS re-set placeholder inputs on the opContext.
                        // Multi-GPU migration (replicateToDevice) can change both extInputs
                        // and cachedInputArrays to the same migrated object, making identity
                        // comparison useless. The C++ side needs fresh NDArray* pointers
                        // because the old ones may have closed DataBuffers.
                        if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, pi, opaqueIn);
                            cachedInputOpaques[pi] = opaqueIn;
                            cachedInputArrays[pi] = arr;
                            // Sync to device (placeholder may have been modified on host)
                            if (!arr.isEmpty()) {
                                OpaqueDataBuffer odb = arr.data().opaqueBuffer();
                                if (odb != null && !odb.isNull()) {
                                    odb.syncToSpecial();
                                }
                            }
                        }
                    }
                } else {
                    // Fallback: full iteration (should not happen after first frozen call)
                    for (int i = 0; i < extInputs.length; i++) {
                        boolean staleBuffer = false;
                        if (extInputs[i] != null) {
                            DataBuffer db = extInputs[i].data();
                            if (db == null || db.wasClosed()) {
                                staleBuffer = true;
                                SDVariable var = sd.getVariable(extKeys[i]);
                                if (var != null && (var.getVariableType() == VariableType.CONSTANT
                                        || var.getVariableType() == VariableType.VARIABLE)) {
                                    extInputs[i] = var.getArr();
                                }
                            }
                        }
                        if (extInputs[i] != cachedInputArrays[i] || staleBuffer) {
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                            nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                            cachedInputOpaques[i] = opaqueIn;
                            cachedInputArrays[i] = extInputs[i];
                        } else if (inputIsPlaceholder != null && inputIsPlaceholder[i]) {
                            // Skip decode inputs managed by C++ (same reason as fast path above)
                            if (decodeInputsConfigured && (i == decodeInputIdsExtIdx
                                    || i == decodePositionIdsExtIdx
                                    || i == decodeAttentionMaskExtIdx)) {
                                continue;
                            }
                            INDArray arr = extInputs[i];
                            if (!arr.isEmpty() && arr.data() != null && !arr.data().wasClosed()) {
                                OpaqueDataBuffer odb = arr.data().opaqueBuffer();
                                if (odb != null && !odb.isNull()) {
                                    odb.syncToSpecial();
                                }
                            }
                        }
                    }
                }
            } else {
                // First call or non-frozen: set all inputs and keep strong refs.
                // contextInputRefs prevents GC from collecting OpaqueNDArrays (and thus
                // deleting the C++ NDArray objects they wrap) while the C++ context holds
                // raw NDArray* pointers to them. Without this, closeable arrays (variables,
                // placeholders) whose OpaqueNDArrays are not marked constant can be deleted
                // by the DeallocatorService between steps, causing db=(nil) SIGSEGV.
                contextInputRefs = new OpaqueNDArray[extInputs.length];
                for (int i = 0; i < extInputs.length; i++) {
                    OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                    nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                    contextInputRefs[i] = opaqueIn;
                }
                // Cache for subsequent frozen calls
                if (shapesFrozen) {
                    cachedInputOpaques = new OpaqueNDArray[extInputs.length];
                    cachedInputArrays = new INDArray[extInputs.length];
                    System.arraycopy(extInputs, 0, cachedInputArrays, 0, extInputs.length);
                    inputIsPlaceholder = new boolean[extInputs.length];
                    // Build placeholderIndices for fast path on subsequent calls
                    int placeholderCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        cachedInputOpaques[i] = OpaqueNDArray.fromINDArray(extInputs[i]);
                        // Mark as placeholder if it came from the placeholderArrays map
                        inputIsPlaceholder[i] = placeholderArrays != null
                                && placeholderArrays.containsKey(extKeys[i]);
                        if (inputIsPlaceholder[i]) placeholderCount++;
                    }
                    // Build placeholderIndices array for fast path
                    placeholderIndices = new int[placeholderCount];
                    int idx = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        if (inputIsPlaceholder[i]) placeholderIndices[idx++] = i;
                    }
                    log.info("FROZEN_INPUT_OPT: built placeholderIndices[{}] (extInputs={})", 
                            placeholderCount, extInputs.length);
                }
            }

            // Set empty output slots on context (C++ plan will allocate and fill them)
            // When frozen, skip after first call — C++ manages its own output slots
            if (!shapesFrozen || !frozenOutputsInitialized) {
                for (int i = 0; i < numOutputs; i++) {
                    INDArray dummy = Nd4j.empty(DataType.FLOAT);
                    OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                    nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
                }
                if (shapesFrozen) {
                    frozenOutputsInitialized = true;
                }
            }

            // Get execution stream — cache to avoid 2 JNI calls per step
            Pointer execStream;
            if (execStreamCached) {
                execStream = cachedExecStream;
            } else {
                execStream = null;
                try {
                    OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                    if (lc != null) {
                        execStream = nativeOps.lcExecutionStream(lc);
                        if (execStream != null) execStream.retainReference();
                    }
                } catch (Exception e) {
                    // CPU backend
                }
                if (shapesFrozen) {
                    cachedExecStream = execStream;
                    execStreamCached = true;
                }
            }

            // Clear native shape caches before each execution — unless shapes are frozen.
            // During autoregressive decoding with dynamic shapes, KV cache dimensions grow
            // by 1 each step, so shapes are stale. When frozen, clearing is unnecessary
            // (the C++ side also checks frozen state, but skipping the JNI call saves ~1-2ms).
            if (!shapesFrozen) {
                nativeOps.clearDynamicShapePlanCaches(nativePlanHandle);
            }

            // Track frozen call count for input re-set logic
            if (shapesFrozen) frozenCallCount++;

            // Execute the plan in C++
            long execStart = System.nanoTime();
            int status = nativeOps.executeDynamicShapePlan(
                    nativePlanHandle,
                    opContext,
                    execStream);
            long execMs = (System.nanoTime() - execStart) / 1_000_000;

            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                DspDiagnostics.recordTimed(DspDiagnostics.FALLBACK, -1, -1, "executeNative",
                        execMs * 1000, "Java: native execution FAILED status=" + status +
                        " msg=" + errMsg + " executionCount=" + executionCount);
                throw new RuntimeException("Native plan execution failed with status " + status +
                        ": " + (errMsg != null ? errMsg : "unknown error"));
            }

            DspDiagnostics.recordTimed(DspDiagnostics.EXECUTE, -1, -1, "executeNative",
                    execMs * 1000, "Java: native execution OK " + execMs + "ms" +
                    " frozen=" + shapesFrozen + " executionCount=" + executionCount);

            // NOTE: No need to clearLastError() on success path — error was already
            // cleared on line 4538 when status != 0. If status == 0, there's no error.
            // Removing this unconditional JNI call saves ~1-2us per step.

            // Extract output arrays from context.
            // C++ wrote NDArray* pointers back into the context's output slots.
            long copyStart = System.nanoTime();
            List<String> requestedOutputs;
            if (cachedRequestedOutputNames != null) {
                requestedOutputs = cachedRequestedOutputNames;
            } else {
                requestedOutputs = new ArrayList<>(plan.getRequestedOutputs());
                if (shapesFrozen) {
                    cachedRequestedOutputNames = requestedOutputs;
                }
            }

            // Frozen fast path: reuse pre-allocated destination arrays (skip allocation,
            // only copy non-KV outputs). When kvCacheRetentionConfigured, C++ handles
            // KV scatter internally — skip copying those 60 outputs entirely.
            if (shapesFrozen && zeroCopyOutputCache != null) {
                int copiedOutputs = 0;
                for (int i = 0; i < numOutputs; i++) {
                    String outputName = requestedOutputs.get(i);

                    // Skip KV outputs — C++ already scattered them into static buffers
                    if (kvCacheRetentionConfigured && kvRetentionOutputNames != null
                            && kvRetentionOutputNames.contains(outputName)) {
                        continue;
                    }

                    OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                    if (opaqueOut == null || opaqueOut.isNull()) continue;

                    INDArray cached = zeroCopyOutputCache.get(outputName);
                    if (cached == null) continue;

                    long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);
                    DataType dtype = cached.dataType();

                    Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                    Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);
                    OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                            length, dtype.toInt(), nativePrimary, nativeSpecial);
                    if (srcOdb != null) {
                        try {
                            OpaqueDataBuffer dstOdb = cached.data().opaqueBuffer();
                            if (dstOdb != null) {
                                nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                            }
                        } finally {
                            nativeOps.deleteDataBuffer(srcOdb);
                        }
                    }
                    copiedOutputs++;
                }

                long copyMs = (System.nanoTime() - copyStart) / 1_000_000;
                if (execMs > 100) {
                    log.info("Native executor: exec={}ms copy={}ms (frozen, {}/{} outputs copied)",
                            execMs, copyMs, copiedOutputs, numOutputs);
                }
                return zeroCopyOutputCache;
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                if (opaqueOut == null || opaqueOut.isNull()) {
                    throw new RuntimeException("Native executor: null output at index " + i);
                }

                // Read shape info from the C++ output NDArray
                long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                long[] shape = Shape.shape(shapeInfo);
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);
                char ordering = Shape.order(shapeInfo);

                // Create a Java-owned INDArray with the SAME ordering as the C++ output.
                // The raw buffer copy below is a flat memcpy — the destination must have
                // matching strides so elements are interpreted correctly.
                INDArray result = Nd4j.createUninitialized(dtype, shape, ordering);

                // Get raw pointers — primary may be null on CUDA (data only on GPU)
                Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);

                OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                        length, dtype.toInt(), nativePrimary, nativeSpecial);
                if (srcOdb != null) {
                    try {
                        OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                        if (dstOdb != null) {
                            nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                        }
                    } finally {
                        nativeOps.deleteDataBuffer(srcOdb);
                    }
                }

                String outputName = requestedOutputs.get(i);
                results.put(outputName, result);
            }

            // Cache allocated arrays for reuse on subsequent frozen executions
            if (shapesFrozen && zeroCopyOutputCache == null && !results.isEmpty()) {
                zeroCopyOutputCache = new LinkedHashMap<>(results);
                // Mark cached outputs as non-closeable — they are reused across steps
                for (INDArray arr : zeroCopyOutputCache.values()) {
                    arr.setCloseable(false);
                }
                log.info("Native executor: cached {} output arrays for frozen reuse (skip allocation)", results.size());
            }

            long copyMs = (System.nanoTime() - copyStart) / 1_000_000;
            if (copyMs > 5 || execMs > 100) {
                log.info("Native executor: exec={}ms copy={}ms ({} outputs)", execMs, copyMs, numOutputs);
            }

            // NOTE: Max-allocation for KV cache output slots is disabled. Giving ops a
            // wrong-shaped pre-allocated buffer (e.g. [1,H,2048,D] when the op produces
            // [1,H,2,D]) causes the CUDA kernel to use the wrong shape → OOB reads →
            // cudaErrorIllegalAddress → Status::KERNEL_FAILURE (50). The correct approach
            // requires ops to accept a fixed-size buffer AND a "valid length" parameter
            // (static KV cache pattern), which is a larger architectural change.
            // configureMaxAllocationForKvCache is kept here for future use.

            // Diagnostic: dump first few values of each output to compare with Java executor
            if (Boolean.getBoolean(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS)) {
                for (Map.Entry<String, INDArray> entry : results.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.length() > 0) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("NATIVE_OUT ").append(name).append(" shape=").append(java.util.Arrays.toString(arr.shape()));
                        sb.append(" first5=[");
                        long limit = Math.min(5, arr.length());
                        for (long j = 0; j < limit; j++) {
                            if (j > 0) sb.append(", ");
                            sb.append(String.format("%.6f", arr.getFloat(j)));
                        }
                        sb.append("]");
                        log.info(sb.toString());
                    }
                }
            }

            return results;
        }
        } finally {
            // Restore original device if we switched for multi-GPU coherency
            if (deviceSwitched) {
                DeviceMemoryManager.getInstance().switchDevice(previousDevice,
                        "DSP.executeNative", "restore-device");
            }
        }
    }

    /**
     * Release the native plan handle. If a SameDiff cache is available, saves the handle
     * there for reuse by future sessions. Otherwise frees it immediately.
     */
    private void freeNativePlanHandle() {
        // Free cached OpaqueContext first (it references the plan)
        if (cachedOpContext != null) {
            log.info("    freeNativePlanHandle: deleteGraphContext");
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.deleteGraphContext(cachedOpContext);
            } catch (Exception ignored) {}
            cachedOpContext = null;
        }
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            // Save to SameDiff's cache for reuse across session resets
            String cacheKey = planCacheKey(nativePlanSource);
            if (sd != null && cacheKey != null) {
                log.info("    freeNativePlanHandle: caching handle for reuse (key={})", cacheKey);
                sd.cacheNativePlanHandle(cacheKey, nativePlanHandle);
            } else {
                log.info("    freeNativePlanHandle: freeDynamicShapePlan (handle={})", nativePlanHandle);
                try {
                    NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    nativeOps.freeDynamicShapePlan(nativePlanHandle);
                } catch (Exception e) {
                    log.info("Error freeing native plan handle: {}", e.getMessage());
                }
            }
        }
        nativePlanHandle = null;
        nativePlanSource = null;
        configuredGraphExecutionMode = GraphExecutionMode.AUTO;
        cachedInputOpaques = null;
        cachedInputArrays = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        cachedExecStream = null;
        execStreamCached = false;
        closeZeroCopyOutputCache();
        closeNativeConstantReplicaCache();
        cachedRequestedOutputNames = null;
        nativeExecutionDevice = -1;
        // Reset KV retention flag — the C++ config is on the freed handle.
        // savedKvPresentOutputNames etc. are intentionally NOT cleared so
        // reapplyKvCacheRetention() can restore the config on a new handle.
        kvCacheRetentionConfigured = false;
        decodeInputsConfigured = false;
        decodeInputIdsExtIdx = -1;
        decodePositionIdsExtIdx = -1;
        decodeAttentionMaskExtIdx = -1;
    }

    @Override
    public void close() {
        log.info("  DSP close() step 1: closeSlotArrayCache");
        System.out.flush(); System.err.flush();
        closeSlotArrayCache();
        int nativeReplicaCount = nativeConstantReplicaCache != null ? nativeConstantReplicaCache.size() : 0;
        if (nativeReplicaCount > 0) {
            log.info("  DSP close() step 2: native constant replicas ({})", nativeReplicaCount);
        }
        int nativeReplicasClosed = closeNativeConstantReplicaCache();

        log.info("  DSP close() step 3: outputSlots");
        System.out.flush(); System.err.flush();
        if (outputSlots != null) {
            Arrays.fill(outputSlots, null);
        }
        if (externalInputs != null) {
            Arrays.fill(externalInputs, null);
        }

        // Free cached output wrappers
        int zeroCopyEntries = zeroCopyOutputCache != null ? zeroCopyOutputCache.size() : 0;
        if (zeroCopyEntries > 0) {
            log.info("  DSP close() step 4: zeroCopyOutputCache ({} entries)", zeroCopyEntries);
            System.out.flush(); System.err.flush();
        }
        int zeroCopyClosed = closeZeroCopyOutputCache();

        // Clear cached input/output state
        cachedInputOpaques = null;
        cachedInputArrays = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        cachedExecStream = null;
        execStreamCached = false;

        // Free cached OpaqueContext
        if (cachedOpContext != null) {
            log.info("  DSP close() step 5: deleteGraphContext");
            System.out.flush(); System.err.flush();
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.deleteGraphContext(cachedOpContext);
            } catch (Exception e) {
                log.info("Error freeing cached OpaqueContext: {}", e.getMessage());
            }
            cachedOpContext = null;
        }

        // Free native C++ plan handle if compiled
        log.info("  DSP close() step 6: freeNativePlanHandle");
        System.out.flush(); System.err.flush();
        freeNativePlanHandle();

        currentPlan = null;
        // Clear saved KV retention params — executor is fully closed, no re-apply possible
        savedKvPresentOutputNames = null;
        savedKvPastInputNames = null;
        log.info("  DSP close() complete (nativeReplicasClosed={}, zeroCopyClosed={})",
                nativeReplicasClosed, zeroCopyClosed);
        System.out.flush(); System.err.flush();
    }

    /**
     * Safely close a DataBuffer using dbFreeBuffersOnStream to avoid calling glibc free()
     * on potentially corrupted heap metadata. Falls back to buf.close() if stream unavailable.
     */
    private void safeCloseBuffer(DataBuffer buf, NativeOps nativeOps, Pointer stream) {
        if (buf == null || buf.wasClosed() || buf.isConstant()) return;
        OpaqueDataBuffer odb = buf.opaqueBuffer();
        if (odb != null && !odb.isNull() && stream != null) {
            try {
                nativeOps.dbFreeBuffersOnStream(odb, stream);
                odb.tryMarkForDeallocation();
                odb.setNull();
                OpaqueDataBufferDeallocator deallocator = odb.getDeallocator();
                if (deallocator != null) deallocator.markDeallocated();
            } catch (Exception ignored) {}
        } else {
            try { buf.close(); } catch (Exception ignored) {}
        }
    }
}
