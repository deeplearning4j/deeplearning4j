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
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
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
import org.nd4j.linalg.framework.device.TransferDirection;
import org.nd4j.linalg.framework.device.TransferReason;
import org.nd4j.linalg.framework.device.TransferEvent;
import org.nd4j.linalg.framework.device.TransferSubsystem;
import org.nd4j.linalg.framework.device.ReplicaLeakDetector;
import org.nd4j.linalg.framework.device.PointerStabilityGuard;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

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

    /** C++ error code returned when an input DataBuffer is closed/destroyed/invalid.
     *  Java can detect this and re-resolve the stale input from SameDiff variables. */
    private static final int NATIVE_STATUS_STALE_BUFFER = 5;

    /** Number of execute() calls on this executor. Used to skip cache validity probe
     *  on the first execution (no cached entries yet).
     *  NOTE: C++ also tracks execution count; this Java-side copy avoids a JNI call
     *  on the hot path. Consolidation target for when input resolution moves to C++. */
    private int executionCount;

    /** Java-side tracking of shapes-frozen state. When true, shape caches don't need
     *  clearing between executions because all shapes are guaranteed constant.
     *  NOTE: C++ tracks this via PlanPhase; kept here for Java-side fast-path decisions. */
    private boolean shapesFrozen;

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
     *  execution attempt. Freed on close(). null means not yet compiled or compilation failed.
     *  Can be swapped across executeNative() calls by redispatchForCurrentShapes() when
     *  placeholder shapes change — the C++ NativePlanCache returns the shape-matched plan. */
    private Pointer nativePlanHandle;

    /** Track which plan the native handle was compiled from. If the plan changes,
     *  the native handle must be recompiled. */
    private DynamicShapePlan nativePlanSource;

    /** Cached inputs to dispatchNativePlan — kept across executes so
     *  redispatchForCurrentShapes() can rebuild JNI arguments without re-serializing. */
    private byte[] cachedSerializedPlan;
    private String[] cachedSortedOutputs;
    private String[] cachedPhKeys;

    /** Per-handle settings cached at compile time and re-applied whenever
     *  redispatchForCurrentShapes() swaps in a newly-cached native plan handle. */
    private boolean cachedCudaGraphsEnabled;
    private int cachedJitModeInt = -1;       // -1 = leave default
    private boolean cachedExecTiming;
    private boolean cachedTraceEnabled;
    private int cachedEffectiveGraphModeCode = -1;   // -1 = leave default
    private final java.util.Set<Long> configuredHandleAddresses = new java.util.HashSet<>();

    /** Graph execution mode currently configured on the native plan handle. */
    private GraphExecutionMode configuredGraphExecutionMode = GraphExecutionMode.AUTO;

    /** If native compilation fails, disable native execution for this executor instance
     *  to avoid repeated failure overhead. */
    private boolean nativeExecutorFailed;

    /** If CUDA graph capture fails, disable CUDA graphs but keep using slot-by-slot native execution. */
    private boolean cudaGraphsFailed;

    // Bespoke C++ KV cache retention state was removed — KV cache append now runs as
    // an ordinary in-graph op (KvScatter via KvCacheManager.scatterNewEntries) on every
    // decode step and is captured into the CUDA graph like any other op. The C++ DSP
    // plan is a pure graph executor: no decode-specific or KV-specific lifecycle.

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

    /** Strong references to ALL constant DataBuffers used by this plan.
     *  Prevents session cleanup from closing constants that are shared across
     *  multiple plans (e.g., vision encoder + decoder). These references ensure
     *  the DataBuffers remain alive for the lifetime of this executor.
     *  Populated at compile time, nulled on close(). */
    private IdentityHashMap<DataBuffer, Boolean> protectedConstantBuffers;

    /** Cached indices of placeholder inputs. Built on first frozen call to avoid
     *  iterating all 1332 external inputs every step. Only ~3 are placeholders
     *  (input_ids, attention_mask, position_ids). Saves ~0.5-1ms per step. */
    private int[] placeholderIndices;

    /** Cached indices of non-placeholder small integral control inputs. These
     *  arrays drive value-dependent shape/controller chains and must be refreshed
     *  into the native opContext during frozen execution just like placeholders. */
    private int[] frozenControlInputIndices;

    /** Cached indices of external inputs backed by derived SameDiff variables.
     *  These values are produced by upstream ops outside the current native replay
     *  unit and must be re-resolved on every frozen execution. */
    private int[] frozenDerivedExternalInputIndices;

    /** True once dummy outputs have been set on the context for frozen execution.
     *  After the first frozen call, C++ manages its own output slots — skip dummy setup. */
    private boolean frozenOutputsInitialized;

    /** Cached requested output names list — avoids allocating a new ArrayList per step. */
    private List<String> cachedRequestedOutputNames;

    /**
     * When true, the current call was initiated from SameDiff.outputDirect() — meaning
     * the caller manages output array lifetimes and zeroCopyOutputCache may be safely
     * populated and reused (callers do NOT dup the result).
     *
     * When false (default), the call was initiated from SameDiff.output() which DUPS all
     * results into independent copies before returning them. In that case zeroCopyOutputCache
     * must NOT be used as a fast-path return, because KV close in the caller only touches the
     * duped copies — not the cached originals — leaving the cache stale (it holds the
     * previous step's logits) while appearing valid to the staleness guard.
     *
     * Set to true by SameDiff.outputDirect() BEFORE calling is.output(); reset to false
     * immediately afterward. This field is intentionally not thread-safe — outputDirect()
     * and output() are always called on the same thread that owns this executor.
     */
    private boolean directOutputMode = false;

    /** Allow the SameDiff front-end to tell the executor whether this is a direct-mode call.
     *  Called from SameDiff.outputDirect() before invoking the InferenceSession. */
    public void setDirectOutputMode(boolean directOutputMode) {
        this.directOutputMode = directOutputMode;
    }

    public boolean isDirectOutputMode() {
        return directOutputMode;
    }

    /** Count of frozen executeNative() calls. Used to force full input re-set on the
     *  first few calls (warmup + Triton compile) to prevent stale OpaqueNDArray pointers. */
    private int frozenCallCount;

    /** Frozen external input buffer identities: DataBuffer references captured on first
     *  frozen execution. Used to detect buffer replacement/closure between frozen steps.
     *  Violations are IllegalStateException (hard error, not log). */
    private DataBuffer[] frozenExtBufferSnapshot;
    /** Frozen external input shapes: shape arrays captured on first frozen execution.
     *  Used to detect shape changes during frozen execution. */
    private long[][] frozenExtShapeSnapshot;

    /** Cached execution stream pointer. Avoids 2 JNI calls per step. */
    private Pointer cachedExecStream;
    private boolean execStreamCached;

    /** Device ID where this DSP executor runs native execution. Determined from the
     *  majority device of external inputs on first executeNative() call. For multi-GPU
     *  scenarios (e.g., draft model on device 1 while target model on device 0), the
     *  entire DSP including CUDA graph capture/replay happens on this device.
     *  -1 means not yet determined. */
    private int nativeExecutionDevice = -1;

    /**
     * Ensure the current thread is on this plan's execution device.
     * Call this before building decoder inputs (position_ids, attention_mask, etc.)
     * so that Nd4j array allocations land on the correct GPU. On multi-GPU systems,
     * ops that run between decode steps (token sampling, embedding) can leave the
     * thread on a different device, causing wrong-device allocations that require
     * cross-device migration and produce CUDA error 700 on non-peer GPUs.
     *
     * <p>If the execution device hasn't been determined yet (first call),
     * does NOT preemptively select one. The data-locality-based device selection
     * in {@link #executeNative} will determine the correct device on first execution.
     * Using selectBestGpu() here would pick by free memory alone, which on asymmetric
     * multi-GPU systems (e.g., 24GB + 8GB) routes execution to the small GPU that
     * cannot fit the model weights — producing garbage logits.
     */
    public void ensureExecutionDevice() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices <= 1) return;
        // Only pin if the device was already determined by executeNative's
        // data-locality scan. Do NOT initialize nativeExecutionDevice here —
        // selectBestGpu() picks by free memory which is wrong for asymmetric GPUs.
        if (nativeExecutionDevice < 0) {
            return;  // let executeNative determine the correct device
        }
        int currentDevice = DeviceMemoryManager.getInstance().getCurrentDeviceId();
        if (currentDevice != nativeExecutionDevice) {
            DeviceMemoryManager.getInstance().switchDevice(nativeExecutionDevice,
                    "DSP.ensureExecutionDevice", "pin-to-plan-device");
        }
    }

    /**
     * Resolve the home device for an external input array. Prefer
     * {@link AffinityManager#getDeviceForArray}, which reads the stable
     * AllocationPoint metadata populated at buffer creation time. If that
     * returns an out-of-range value (e.g. host-only arrays that haven't been
     * synced to any GPU), fall back to {@code fallbackDevice}, which is the
     * device we will run on anyway — treating "unknown" as "already correct"
     * avoids needless replication and the syncToSpecial() on the producing
     * device that follows the first kernel launch will then settle it there.
     */
    private int resolveArrayDevice(INDArray arr, int numDevices, int fallbackDevice) {
        try {
            Integer devIdObj = Nd4j.getAffinityManager().getDeviceForArray(arr);
            if (devIdObj != null) {
                int devId = devIdObj.intValue();
                if (devId >= 0 && devId < numDevices) {
                    return devId;
                }
            }
        } catch (Exception ignored) {
            // AffinityManager may throw on arrays whose allocation point is
            // gone (closed, poisoned). Treat as unknown.
        }
        return fallbackDevice;
    }

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
            frozenControlInputIndices = null;
            frozenDerivedExternalInputIndices = null;
            // Release C++ GPU intermediates (planOwnedArrays_, CUDA graph workspaces,
            // replay handles, cuBLAS workspace) BEFORE nulling the Java handle.
            // The C++ plan is cache-owned and survives freeNativePlanHandle(), but its
            // intermediates hold gigabytes of GPU memory. Without this, vision encoder
            // intermediates (~3.6GB per frame) accumulate across resetSession() calls
            // and cause OOM during decoder graph capture.
            releaseGpuIntermediates();
            // Free native plan handle reference (skips C++ plan destruction since it's
            // cache-owned, but clears Java-side cached state).
            freeNativePlanHandle("SESSION_RESET");
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
        freeNativePlanHandle("PLAN_CHANGED");
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
        if (isCurrentExternalInputBuffer(buf)) return true;
        if (protectedConstantBuffers != null && protectedConstantBuffers.containsKey(buf)) return true;
        return false;
    }

    /**
     * Single authoritative check: is this array's data accessible for GPU execution?
     * Checks Java DataBuffer, OpaqueDataBuffer pointer, and wasClosed flag in one place.
     * Empty arrays (shape=[], data=null) are always considered live — they have no
     * DataBuffer by design (Nd4j.empty() creates arrays with null data).
     */
    public static boolean isArrayLive(INDArray arr) {
        if (arr == null || arr.wasClosed()) return false;
        if (arr.isEmpty()) return true;
        DataBuffer db = arr.data();
        if (db == null || db.wasClosed()) return false;
        OpaqueDataBuffer odb = db.opaqueBuffer();
        return odb != null && !odb.isNull();
    }

    /**
     * Returns the set of constant DataBuffers protected by this executor.
     * Used by session cleanup code to avoid closing constants that are still
     * referenced by an active plan.
     */
    public IdentityHashMap<DataBuffer, Boolean> getProtectedConstantBuffers() {
        return protectedConstantBuffers;
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
        // Unregister replicas from leak detector before closing
        ReplicaLeakDetector replicaDetector = null;
        try {
            replicaDetector = Nd4j.framework.device().replicaLeaks();
        } catch (Exception e) {
            // Subsystem may not be initialized
        }
        
        if (replicaDetector != null && nativeConstantReplicaCache != null) {
            for (Map.Entry<Integer, INDArray> entry : nativeConstantReplicaCache.entrySet()) {
                INDArray replica = entry.getValue();
                if (replica != null && !replica.wasClosed()) {
                    replicaDetector.unregisterReplica(replica.getId());
                }
            }
        }
        
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

    /**
     * Collect DataBuffers from the model's constants and variables that must NOT be
     * force-closed during ArrayCacheMemoryMgr cleanup. Without this, "constant-poisoned"
     * intermediates that happen to share DataBuffers with model weights get force-closed,
     * leaving the native DSP plan with stale pointers and preventing CUDA graph replay.
     */
    private IdentityHashMap<DataBuffer, Boolean> collectProtectedModelBuffers() {
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();
        if (sd == null) return protectedBuffers;
        for (SDVariable variable : sd.variables()) {
            if (variable == null) continue;
            VariableType type = variable.getVariableType();
            if (type != VariableType.CONSTANT && type != VariableType.VARIABLE) continue;
            INDArray arr = variable.getArr();
            if (arr != null && !arr.wasClosed() && arr.data() != null) {
                protectedBuffers.put(arr.data(), Boolean.TRUE);
            }
        }
        // Also include DataBuffers captured at compile time — these may no longer
        // be reachable via sd.variables() if the variable's array was swapped.
        if (protectedConstantBuffers != null) {
            protectedBuffers.putAll(protectedConstantBuffers);
        }
        return protectedBuffers;
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

    private static boolean isSmallIntegralControlArray(INDArray arr) {
        if (arr == null || arr.isEmpty()) {
            return false;
        }
        DataType dt = arr.dataType();
        if (dt != DataType.INT32 && dt != DataType.INT64 && dt != DataType.BOOL) {
            return false;
        }
        long len = arr.length();
        return len > 0 && len <= 32;
    }

    private boolean isDerivedExternalInput(String varName) {
        if (sd == null || varName == null) {
            return false;
        }
        Variable meta = sd.getVariables().get(varName);
        if (meta == null) {
            return false;
        }
        String outputOfOp = meta.getOutputOfOp();
        return outputOfOp != null && !outputOfOp.isEmpty();
    }

    private INDArray resolveCanonicalExternalInput(String varName,
                                                   Map<String, INDArray> placeholderArrays) {
        if (varName == null) {
            return null;
        }
        if (placeholderArrays != null) {
            INDArray placeholder = placeholderArrays.get(varName);
            if (placeholder != null) {
                return placeholder;
            }
        }
        if (sd == null) {
            return null;
        }
        SDVariable var = sd.getVariable(varName);
        return var != null ? var.getArr() : null;
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

    // The bespoke KV cache retention + decode-input APIs were removed. KV cache
    // append now runs as an ordinary in-graph op (KvScatter via
    // KvCacheManager.scatterNewEntries) on every decode step, and decoder inputs
    // (input_ids, position_ids, attention_mask) are plain ext input NDArrays that
    // Java writes via normal assign/putScalar. The C++ DSP plan is a pure graph
    // executor with no decode-specific lifecycle.

    /**
     * Get the native plan handle for direct JNI calls.
     */
    public Pointer getNativePlanHandle() {
        return nativePlanHandle;
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
            frozenControlInputIndices = null;
            frozenDerivedExternalInputIndices = null;
            frozenOutputsInitialized = false;
            frozenCallCount = 0;
            cachedExecStream = null;
            execStreamCached = false;
            frozenExtBufferSnapshot = null;
            frozenExtShapeSnapshot = null;
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
     * Get the current plan-level phase from the native C++ plan.
     * Returns the phase that represents the overall plan lifecycle:
     * SLOT_BY_SLOT → SHAPES_FROZEN → POINTERS_STABLE → REPLAYING.
     *
     * @return the current PlanPhase, or null if no native plan is compiled
     */
    public PlanPhase getPlanPhase() {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return null;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int code = nativeOps.getPlanPhase(nativePlanHandle);
        return PlanPhase.fromNativeCode(code);
    }

    /**
     * Check if all buffer pointers are stable across executions.
     * Pointer stability is required before graph capture/replay can begin.
     *
     * @return true if pointers are stable, false otherwise
     */
    public boolean arePointersStable() {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return false;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        return nativeOps.getPlanPointersStable(nativePlanHandle) == 1;
    }

    /**
     * Get the slot state for a specific slot in the native plan.
     *
     * @param slotIdx the slot index
     * @return the SlotState, or null if invalid
     */
    public SlotState getSlotState(int slotIdx) {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return null;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int code = nativeOps.getPlanSlotState(nativePlanHandle, slotIdx);
        return SlotState.fromNativeCode(code);
    }

    /**
     * Get the number of executions since shapes were frozen.
     *
     * @return frozen execution count, or -1 if shapes are not frozen
     */
    public int getFrozenExecutionCount() {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return -1;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        return nativeOps.getPlanFrozenExecutionCount(nativePlanHandle);
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
        frozenControlInputIndices = null;
        frozenDerivedExternalInputIndices = null;

        // Release C++ intermediate GPU memory (CUDA graphs, replay workspaces, cuBLAS workspace,
        // non-weight output slot NDArrays). This also calls closeSlotArrayCache() internally.
        releaseGpuIntermediates();

        closeZeroCopyOutputCache();

        // Drain ArrayCacheMemoryMgr state: deferred close buffers accumulate across pages
        // because nothing drains them between page boundaries. The cache itself holds arrays
        // that will never be reused (different shapes per page). Disable the cache flag so
        // the next executeDynamicShapePlanBased() starts from a clean state.
        // IMPORTANT: Pass model constant/variable buffers as protected so that force-closing
        // "constant-poisoned" intermediates doesn't destroy buffers the native DSP plan still
        // references. Without this, 60+ model constants get closed → stale buffer scan fails
        // → CUDA graph replay can't proceed → 13x slowdown.
        IdentityHashMap<DataBuffer, Boolean> protectedModelBuffers = collectProtectedModelBuffers();
        ArrayCacheMemoryMgr.closeDeferredBuffers(protectedModelBuffers);
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        if (currentPlan != null) {
            currentPlan.clearAllShapeCaches();
        }
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        nativeExecutorFailed = false;
        executionCount = 0;
        // Do NOT reset nativeExecutionDevice here. CUDA graphs are device-specific —
        // once captured on device N, all subsequent executions must stay on device N.
        // Resetting to -1 causes selectBestGpu() to pick a different device (e.g., device 1)
        // on the next page, but captured graphs only exist on the original device, causing
        // status 50 (REPLAY ERROR: hasReplayHandle=0) on the new device.

        // KV cache retention and decode-input direct-update state were removed —
        // decode is now expressed as ordinary in-graph ops and Java-written ext inputs,
        // so there is no bespoke state to reset here.
    }

    /**
     * Reset executor state for next-page decode reuse with MAXIMAL state preservation.
     *
     * <p>This variant is optimized for VLM decode where page shapes are INVARIANT.
     * It preserves:</p>
     * <ul>
     *   <li>Output slot arrays (same shapes for decode)</li>
     *   <li>CUDA graph replay handles</li>
     *   <li>cuBLAS workspace</li>
     *   <li>Batch optimization resources</li>
     * </ul>
     *
     * <p>Only resets:</p>
     * <ul>
     *   <li>KV cache position</li>
     *   <li>Pending decode update flag</li>
     * </ul>
     *
     * <p>Call this between pages when decode shapes are invariant (always [1,1] for input_ids/position_ids).</p>
     */
    public void resetForNextPageDecode() {
        log.info("DSP resetForNextPageDecode: no-op (KV cache is an in-graph scatter; no native state to reset)");
    }

    /**
     * Reserved compatibility hook for the removed external-input staging path.
     *
     * <p>DSP replay now reads canonical external input buffers directly and
     * invalidates/re-captures when those addresses drift, so there is no
     * separate native staging lifecycle to configure here.</p>
     *
     * @param extIndices    Unused
     * @param maxSizes      Unused
     */
    public void setExternalInputMaxSizes(int[] extIndices, long[] maxSizes) {
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            log.info("setExternalInputMaxSizes: no-op; replay uses canonical external buffers directly");
        }
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
    }

    /**
     * Get the configured maximum KV cache length.
     */
    public int getMaxKvCacheLength() {
        return maxKvCacheLength;
    }

    /**
     * Get the currently compiled DynamicShapePlan (if any).
     */
    public DynamicShapePlan getCurrentPlan() {
        return currentPlan;
    }

    /**
     * Whether a native plan has been compiled for the given plan. Compilation caches the
     * serialized bytes and per-handle settings; the actual native handle is obtained per
     * execute via the C++ NativePlanCache (shape-keyed), so we check the cached artifacts
     * rather than {@code nativePlanHandle}, which is swapped by redispatchForCurrentShapes.
     */
    public boolean isNativePlanCompiled(DynamicShapePlan plan) {
        return cachedSerializedPlan != null && nativePlanSource == plan;
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

        if (cachedSerializedPlan == null || nativePlanSource != plan) {
            if (planChanged && cudaGraphsFailed) {
                log.info("Native executor: resetting cudaGraphsFailed on plan recompilation");
                cudaGraphsFailed = false;
            }

            freeNativePlanHandle("PLAN_RECOMPILATION");
            configuredHandleAddresses.clear();

            byte[] serialized = plan.serialize();
            if (serialized == null || serialized.length == 0) {
                nativeExecutorFailed = true;
                throw new RuntimeException("Native executor: plan serialization returned empty. " +
                        "Cannot compile native plan. No fallback permitted.");
            }

            // Cache inputs for shape-keyed dispatch. The actual native plan handle is
            // obtained per-execute through the C++ NativePlanCache; placeholder arrays
            // aren't required to be bound at compile time (they arrive via sd.output(Map)).
            List<String> sortedOutputs = new java.util.ArrayList<>(plan.getRequestedOutputs());
            java.util.Collections.sort(sortedOutputs);

            String[] extKeys = plan.getExternalInputKeys();
            byte[] srcTypes = plan.getExternalInputSourceTypes();
            List<String> phKeys = new java.util.ArrayList<>();
            for (int pi = 0; pi < extKeys.length; pi++) {
                if (srcTypes != null && pi < srcTypes.length
                        && srcTypes[pi] == DynamicShapeSlot.SOURCE_PLACEHOLDER) {
                    phKeys.add(extKeys[pi]);
                }
            }

            cachedSerializedPlan = serialized;
            cachedSortedOutputs = sortedOutputs.toArray(new String[0]);
            cachedPhKeys = phKeys.toArray(new String[0]);

            cachedCudaGraphsEnabled = !cudaGraphsFailed && !"false".equalsIgnoreCase(
                    System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "true"));

            String jitModeStr = System.getProperty(ND4JSystemProperties.DSP_JIT_MODE, "graph");
            if ("graph".equalsIgnoreCase(jitModeStr)) {
                cachedJitModeInt = -1;  // leave default
            } else if ("jit".equalsIgnoreCase(jitModeStr)) {
                cachedJitModeInt = 1;
            } else if ("graph+jit".equalsIgnoreCase(jitModeStr)) {
                cachedJitModeInt = 2;
            } else {
                cachedJitModeInt = 0;  // GRAPH_ONLY fallback
            }

            cachedExecTiming = "true".equalsIgnoreCase(
                    System.getProperty(ND4JSystemProperties.DSP_EXECUTION_TIMING, "false"));
            cachedTraceEnabled = System.getProperty(ND4JSystemProperties.DSP_TRACE) != null;

            nativePlanSource = plan;
            nativeExecutorFailed = false;
            executionCount = 0;
            maxAllocationConfigured = false;
            nativeExecutionDevice = -1;
            log.info("Native executor: compiled plan with {} slots, {} external inputs, {} outputs (cudaGraphs={}, shapesFrozen={})",
                    plan.getSlots().length, plan.getExternalInputKeys().length,
                    plan.getRequestedOutputs().size(), cachedCudaGraphsEnabled, false);
            DspDiagnostics.record(DspDiagnostics.COMPILE,
                    "Java: compiled native plan " + plan.getSlots().length + " slots, " +
                    plan.getExternalInputKeys().length + " inputs, " +
                    plan.getRequestedOutputs().size() + " outputs (dispatch deferred to execute time)");

            // Hold strong references to ALL constant DataBuffers used by this plan.
            // This prevents session cleanup (destroySession / closePooledResources) from
            // closing constants that are shared across multiple plans (e.g., vision encoder
            // + decoder). The protection set is checked by all cleanup paths.
            protectedConstantBuffers = new IdentityHashMap<>();
            int protectedCount = 0;
            for (int i = 0; i < extKeys.length; i++) {
                SDVariable var = sd.getVariable(extKeys[i]);
                if (var != null && var.getVariableType() == VariableType.CONSTANT) {
                    INDArray arr = var.getArr();
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        protectedConstantBuffers.put(arr.data(), Boolean.TRUE);
                        protectedCount++;
                    }
                }
            }
            if (protectedCount > 0) {
                log.info("Native executor: protecting {} constant DataBuffers for plan lifetime", protectedCount);
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

        cachedEffectiveGraphModeCode = effectiveMode.getNativeCode();
        configuredGraphExecutionMode = effectiveMode;
        // Invalidate any previously-configured handles so the new mode is re-applied
        // to every handle the NativePlanCache returns on the next redispatch.
        configuredHandleAddresses.clear();
        log.info("Native executor: mode resolution requested={} resolved={} effective={} tritonAvailable={} fallbackToAuto={}",
                requestedMode, resolvedMode, effectiveMode, tritonAvailable, fallbackToAutoIfTritonUnavailable);
        DspDiagnostics.record(DspDiagnostics.BACKEND,
                "Java: mode resolution requested=" + requestedMode + " effective=" + effectiveMode +
                " triton=" + tritonAvailable + " (applied lazily on first redispatch)");

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
     * Re-dispatch through the C++ NativePlanCache using the current placeholder
     * shape-info pointers. The cache is O(1) for matching shapes (returns the same
     * plan handle) and swaps to a different cached plan when shapes differ. This is
     * what makes shape drift safe — each (outputs, shape-sig) pair gets its own
     * NativeDynamicShapePlan with its own immutable slots.
     *
     * <p>Must be called at the start of every executeNative() after compileNativePlan()
     * has cached the serialized plan bytes and output/placeholder key lists.
     */
    private void redispatchForCurrentShapes(Map<String, INDArray> placeholderArrays) {
        if (cachedSerializedPlan == null) {
            throw new IllegalStateException(
                "redispatchForCurrentShapes: plan not compiled yet — " +
                "compileNativePlan() must run before executeNative().");
        }
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Pointer cache = sd.getOrCreateNativePlanCache();

        BytePointer planBytes = new BytePointer(cachedSerializedPlan);
        BytePointer[] namePointers = new BytePointer[cachedSortedOutputs.length];
        for (int ni = 0; ni < cachedSortedOutputs.length; ni++) {
            // BytePointer(String) appends a NUL terminator, which is required for
            // the C++ std::string(const char*) constructor to read the correct string.
            // BytePointer(byte[]) does NOT append NUL, causing std::string to read
            // garbage bytes past the end, producing unstable outputSetHash values
            // and triggering unnecessary plan re-creation every ~20 decode steps.
            namePointers[ni] = new BytePointer(cachedSortedOutputs[ni]);
        }
        PointerPointer outputNamesPtr = cachedSortedOutputs.length == 0
                ? new PointerPointer(0)
                : new PointerPointer(namePointers);

        List<Pointer> phPtrs = new java.util.ArrayList<>();
        for (String phKey : cachedPhKeys) {
            INDArray arr = placeholderArrays != null ? placeholderArrays.get(phKey) : null;
            if (arr == null) {
                SDVariable v = sd.getVariable(phKey);
                arr = v != null ? v.getArr() : null;
            }
            if (arr == null || arr.shapeInfoDataBuffer() == null) {
                throw new IllegalStateException(
                    "redispatchForCurrentShapes: placeholder '" + phKey +
                    "' has no array at execute time — cannot build shape-keyed cache key.");
            }
            phPtrs.add(arr.shapeInfoDataBuffer().addressPointer());
        }
        PointerPointer phPtrsPacked = phPtrs.isEmpty()
                ? new PointerPointer(0)
                : new PointerPointer(phPtrs.toArray(new Pointer[0]));

        try {
            Pointer newHandle = nativeOps.dispatchNativePlan(
                    cache,
                    planBytes, cachedSerializedPlan.length,
                    outputNamesPtr, cachedSortedOutputs.length,
                    phPtrsPacked, phPtrs.size());
            if (newHandle == null || newHandle.isNull()) {
                String cppError = null;
                try {
                    cppError = nativeOps.lastErrorMessage();
                } catch (Throwable t) {
                    // swallow - diagnostic only
                }
                StringBuilder hex = new StringBuilder();
                for (int i = 0; i < cachedSerializedPlan.length; i++) {
                    if (i % 16 == 0) hex.append(String.format("%n  %04d: ", i));
                    hex.append(String.format("%02x ", cachedSerializedPlan[i] & 0xff));
                }
                throw new RuntimeException(
                    "redispatchForCurrentShapes: dispatchNativePlan returned null. " +
                    "C++ lastErrorMessage=" + (cppError != null && !cppError.isEmpty() ? cppError : "(empty)") +
                    " — planBytes=" + cachedSerializedPlan.length +
                    " outputs=" + cachedSortedOutputs.length +
                    " placeholders=" + phPtrs.size() +
                    " hexDump:" + hex);
            }
            // Swap handle if the cache returned a different plan for current shapes.
            // The C++ cache owns plan lifetimes, so we don't free the old one.
            boolean swapped = nativePlanHandle == null
                    || nativePlanHandle.isNull()
                    || newHandle.address() != nativePlanHandle.address();
            if (swapped) {
                if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
                    // HARD ERROR: if the plan swaps after frozen/replay state is established,
                    // the cache is returning different plans for the same shapes. This means
                    // every decode step creates a new plan, destroying all replay/capture
                    // progress and annihilating throughput. This is a catastrophic bug.
                    if (shapesFrozen || frozenCallCount > 2) {
                        throw new RuntimeException(
                            "PLAN_CACHE_BUG: plan handle swapped from " +
                            Long.toHexString(nativePlanHandle.address()) + " to " +
                            Long.toHexString(newHandle.address()) + " AFTER frozen state was established " +
                            "(frozenCallCount=" + frozenCallCount + ", shapesFrozen=" + shapesFrozen + "). " +
                            "The plan cache is returning different plans for the same shapes on every step, " +
                            "destroying all graph replay progress. This indicates the cache key is not " +
                            "stable across steps (pointer-address vs content-based hashing bug). " +
                            "executionCount=" + executionCount +
                            " phKeys=" + (cachedPhKeys != null ? cachedPhKeys.length : "null") +
                            " outputs=" + (cachedSortedOutputs != null ? cachedSortedOutputs.length : "null"));
                    }
                    // Unpin the old plan so it becomes eligible for LRU eviction.
                    // This MUST happen before the new plan is pinned (which
                    // getOrInsert already did) to avoid dangling pointers — the
                    // old plan's GPU resources are freed on eviction.
                    nativeOps.unpinNativePlan(cache, nativePlanHandle);
                    log.info("redispatchForCurrentShapes: plan swapped from {} to {} — resetting frozen state",
                            nativePlanHandle.address(), newHandle.address());
                    frozenOutputsInitialized = false;
                    frozenCallCount = 0;
                    closeZeroCopyOutputCache();
                    // Clear cached input arrays: the new plan may have different
                    // external input mappings or slot assignments.
                    cachedInputArrays = null;
                    cachedInputOpaques = null;
                    contextInputRefs = null;
                    inputIsPlaceholder = null;
                }
                nativePlanHandle = newHandle;
            }
            // Apply per-handle settings the first time we see each cached handle.
            applySettingsIfNewHandle(nativeOps, newHandle);
        } finally {
            planBytes.close();
        }
    }

    /**
     * Apply the per-handle settings captured at compile time (cudaGraphs, JIT mode,
     * exec timing, trace, graph execution mode) to a native plan handle the first
     * time it is seen. Handles returned repeatedly by the C++ NativePlanCache keep
     * their configuration across executes, so we only configure each address once.
     */
    private void applySettingsIfNewHandle(NativeOps nativeOps, Pointer handle) {
        if (handle == null || handle.isNull()) return;
        long addr = handle.address();
        if (!configuredHandleAddresses.add(addr)) return;

        if (cachedCudaGraphsEnabled) {
            try {
                nativeOps.setPlanCudaGraphsEnabled(handle, true);
                DspDiagnostics.record(DspDiagnostics.COMPILE,
                        "Java: CUDA graphs ENABLED on native plan (addr=" + Long.toHexString(addr) + ")");
            } catch (UnsupportedOperationException e) {
                DspDiagnostics.record(DspDiagnostics.COMPILE,
                        "Java: CUDA graphs not supported by backend (CPU?)");
            }
        } else {
            DspDiagnostics.record(DspDiagnostics.COMPILE,
                    "Java: CUDA graphs DISABLED (cudaGraphsFailed=" + cudaGraphsFailed + ")");
        }

        if (cachedJitModeInt >= 0) {
            try {
                nativeOps.setPlanJitMode(handle, cachedJitModeInt);
                log.info("Native executor: JIT mode set to {}", cachedJitModeInt);
            } catch (UnsupportedOperationException e) {
                // Backend doesn't support JIT
            }
        }

        if (cachedExecTiming) {
            try {
                nativeOps.setPlanExecutionTimingEnabled(handle, true);
            } catch (UnsupportedOperationException e) {
                // Backend doesn't support timing
            }
        }

        if (cachedTraceEnabled) {
            try {
                nativeOps.setPlanTraceEnabled(handle, true);
            } catch (UnsupportedOperationException e) {
                // Backend doesn't support trace
            }
        }

        if (cachedEffectiveGraphModeCode >= 0) {
            try {
                nativeOps.setPlanGraphExecutionMode(handle, cachedEffectiveGraphModeCode);
            } catch (UnsupportedOperationException e) {
                // Backend ignores mode
            }
        }

        // NOTE: Do NOT propagate shapesFrozen here. The C++ plan manages its own frozen
        // state transition independently based on execution count and shape stability.
        // Calling setPlanShapesFrozen prematurely (before any execution) causes phaseWarmup
        // to be dispatched at executeCount_=0, which corrupts slot arrays / shape caches
        // for subsequent phaseSlotBySlot steps, producing wrong output tokens.
        // The Java shapesFrozen flag is used only to gate zeroCopyOutputCache and
        // directOutputMode fast paths on the Java side.
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
            nativeOps.setPlanOutputSlotMaxSizes(nativePlanHandle, indices.length, indices, sizes);
            log.info("Configured max-allocation for {} KV cache slots with maxSeqLen={}",
                    kvSlotIndices.size(), maxKvCacheLength);
        }
    }

    /**
     * Configure native plan KV scatter for CUDA-graph-compatible decode loops.
     *
     * <p>After this call the C++ plan executes batched KV scatter after each execute(),
     * writing present KV outputs into static KV buffers at the current position.
     * This eliminates the Java-side scatterNewEntries() round-trip.</p>
     *
     * <p>Must be called after the first execution step (when slot indices are known)
     * and before the decode loop begins.</p>
     *
     * @param firstStepResults  Output map from the first execution step
     * @param plan              The compiled DSP plan (for slot index lookup)
     * @param staticKvBuffers   Map from present-KV output name to static buffer INDArray
     * @param kvPositionPtr     LongPointer to device-accessible int64 position scalar
     */
    public void configureNativePlanKvScatter(Map<String, INDArray> firstStepResults,
                                              DynamicShapePlan plan,
                                              Map<String, INDArray> staticKvBuffers,
                                              LongPointer kvPositionPtr) {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return;
        if (firstStepResults == null || staticKvBuffers == null || staticKvBuffers.isEmpty()) return;
        if (kvPositionPtr == null) return;

        Map<String, Integer> outputNameToSlot = plan.getOutputNameToSlotIndex();

        List<Integer> presentSlotIndices = new ArrayList<>();
        List<OpaqueNDArray> staticBufList = new ArrayList<>();
        long heads = -1, srcSeqLen = -1, dstSeqLen = -1, dim = -1;
        int dtypeInt = -1;

        for (Map.Entry<String, INDArray> entry : firstStepResults.entrySet()) {
            String outputName = entry.getKey();
            boolean isKvPresent = outputName.contains("present")
                    && (outputName.contains("key") || outputName.contains("value"));
            if (!isKvPresent) continue;

            Integer slotIdx = outputNameToSlot.get(outputName);
            if (slotIdx == null || slotIdx < 0) continue;

            INDArray presentArr = entry.getValue();
            if (presentArr == null || presentArr.rank() != 4) continue;

            // Find the corresponding static buffer
            // The static buffer is keyed by the past input name, which is derived from the present name.
            // Try several common naming conventions:
            INDArray staticBuf = staticKvBuffers.get(outputName);
            if (staticBuf == null) {
                // Try replacing 'present_' with 'past_'
                String pastName = outputName.replace("present_", "past_");
                staticBuf = staticKvBuffers.get(pastName);
            }
            if (staticBuf == null) continue;
            if (staticBuf.rank() != 4) continue;

            presentSlotIndices.add(slotIdx);
            staticBufList.add(OpaqueNDArray.fromINDArray(staticBuf));

            // Extract shape info from first pair (all pairs are assumed uniform)
            if (heads < 0) {
                heads = presentArr.size(1);
                srcSeqLen = presentArr.size(2);
                dstSeqLen = staticBuf.size(2);
                dim = presentArr.size(3);
                dtypeInt = presentArr.dataType().toInt();
            }

            log.debug("KV scatter: slot {} ({}) -> static buf shape={}", slotIdx, outputName,
                    java.util.Arrays.toString(staticBuf.shape()));
        }

        if (presentSlotIndices.isEmpty() || heads < 0) {
            log.warn("configureNativePlanKvScatter: no KV output pairs found — skipping");
            return;
        }

        int numPairs = presentSlotIndices.size();
        int[] slotIndicesArr = presentSlotIndices.stream().mapToInt(Integer::intValue).toArray();

        // Build Pointer[] containing OpaqueNDArray* for each static buffer
        Pointer[] staticPtrs = new Pointer[numPairs];
        for (int i = 0; i < numPairs; i++) {
            staticPtrs[i] = staticBufList.get(i);
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.configurePlanKvScatter(nativePlanHandle,
                slotIndicesArr, staticPtrs, numPairs,
                dtypeInt, heads, srcSeqLen, dstSeqLen, dim,
                kvPositionPtr);

        log.info("Configured native plan KV scatter: {} pairs, heads={}, srcSeq={}, dstSeq={}, dim={}",
                numPairs, heads, srcSeqLen, dstSeqLen, dim);
    }

    /**
     * Reset the KV cache position tracked by the native plan.
     * Call this after prefill with the prefill sequence length.
     *
     * @param position  new cache position (e.g., prefill sequence length)
     */
    public void resetNativePlanKvPosition(long position) {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return;
        NativeOpsHolder.getInstance().getDeviceNativeOps()
                .resetPlanKvCachePosition(nativePlanHandle, position);
    }

    /**
     * Get the current KV cache position tracked by the native plan.
     * Returns -1 if KV scatter is not configured.
     */
    public long getNativePlanKvPosition() {
        if (nativePlanHandle == null || nativePlanHandle.isNull()) return -1L;
        return NativeOpsHolder.getInstance().getDeviceNativeOps()
                .getPlanKvCachePosition(nativePlanHandle);
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
                // Reuse the previously configured execution mode (e.g. SLOT_BY_SLOT)
                // so that plan recompilation after frozen transition doesn't reset to AUTO.
                GraphExecutionMode recompileMode = configuredGraphExecutionMode != GraphExecutionMode.AUTO
                        ? configuredGraphExecutionMode : null;
                compileNativePlan(plan, recompileMode, sd.isDspFallbackToAutoIfTritonUnavailable());
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

        // Build a set of DataBuffers that belong to zeroCopyOutputCache entries.
        // These must NOT be freed here — they're still needed by callers until
        // closeZeroCopyOutputCache() runs later in the releaseGpuIntermediates sequence.
        Set<DataBuffer> outputProtectedBuffers = new HashSet<>();
        if (zeroCopyOutputCache != null) {
            for (INDArray outArr : zeroCopyOutputCache.values()) {
                if (outArr != null && !outArr.wasClosed() && outArr.data() != null) {
                    outputProtectedBuffers.add(outArr.data());
                }
            }
        }

        // Collect eligible buffers from the cache into pendingClose.
        // The persistent dedup sets (seenIdentity, closedOdbAddresses) from the previous
        // execute() call will correctly skip buffers already freed during execution.
        int collected = 0;
        int protectedOutputCount = 0;
        for (int i = 0; i < slotArrayCache.length; i++) {
            INDArray arr = slotArrayCache[i];
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed()) {
                    // Skip buffers that belong to requested outputs (zeroCopyOutputCache).
                    // These will be freed by closeZeroCopyOutputCache() later.
                    if (outputProtectedBuffers.contains(buf)) {
                        protectedOutputCount++;
                        slotArrayCache[i] = null;
                        continue;
                    }
                    // Undo setCloseable(false) poisoning from directExecHelper().
                    // Session intermediates are marked constant via setCloseable(false)
                    // → setConstant(true). Without undoing this, the slot cache cannot
                    // free ANY buffers during session reset, leaking all GPU memory.
                    // BUT: never un-poison real constants protected by the plan.
                    if (buf.isConstant() && !buf.isAttached()) {
                        if (protectedConstantBuffers == null || !protectedConstantBuffers.containsKey(buf)) {
                            try {
                                buf.setConstant(false);
                            } catch (Exception ignored) {}
                        }
                    }
                    if (!buf.isConstant()) {
                        pendingClose.add(buf);
                        collected++;
                    }
                }
            }
            slotArrayCache[i] = null;
        }
        if (protectedOutputCount > 0) {
            log.info("    closeSlotArrayCache: protected {} output buffers from premature free",
                     protectedOutputCount);
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
     * Release all GPU memory held by intermediate computation results on the C++ native
     * plan, while keeping the plan handle alive for reuse. This frees:
     * <ul>
     *   <li>Non-weight NDArrays from output slots (SLOT_OWNED buffers)</li>
     *   <li>Per-segment CUDA graph replay handles (workspaces, host pointers)</li>
     *   <li>cuBLAS workspace (~256 MB)</li>
     *   <li>Batch-zero, batch-D2D, and batched-GEMM device arrays</li>
     *   <li>MmulHelper cast cache (thread-local FP16-to-FP32 staging)</li>
     * </ul>
     *
     * <p>After this call the plan is in a "cold" state — the next {@code execute()} will
     * re-warm (re-detect view producers, re-capture CUDA graphs, re-allocate workspace, etc.)
     * just like the very first execution after compilation.</p>
     *
     * <p>Use this between VLM decode runs to reclaim GPU memory (~14 GB for large models)
     * without the cost of plan re-compilation.</p>
     *
     * <p>Also clears the Java-side slot array cache and trims the CUDA memory pool.</p>
     *
     * @return the number of intermediate NDArrays freed on the C++ side, or 0 if no
     *         native plan is compiled
     */
    public int releaseGpuIntermediates() {
        log.info("releaseGpuIntermediates: START");
        System.out.flush(); System.err.flush();

        // Step 1: Clear Java-side slot array cache (frees Java-managed DataBuffers)
        closeSlotArrayCache();

        // Step 2: Call C++ releaseGpuIntermediates (frees CUDA graphs, replay workspaces,
        //         cuBLAS workspace, and non-weight output slot NDArrays)
        int freedCount = 0;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            freedCount = nativeOps.releaseGpuIntermediates(nativePlanHandle);
            log.info("releaseGpuIntermediates: C++ freed {} intermediate arrays", freedCount);

            // Step 3: Trim CUDA memory pool so freed memory is returned to CUDA
            Pointer stream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
            if (stream != null) {
                int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                nativeOps.trimMemoryPoolOnStream(currentDevice, stream);
                int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
                for (int d = 0; d < numDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPoolOnStream(d, null);
                    }
                }
            }
        } else {
            log.info("releaseGpuIntermediates: no native plan handle, skipping C++ release");
        }

        // Step 4: Reset Java-side execution state for re-warming
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        closeZeroCopyOutputCache();

        log.info("releaseGpuIntermediates: DONE (freed {} C++ arrays)", freedCount);
        System.out.flush(); System.err.flush();
        return freedCount;
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

        // Pin thread to the plan's execution device at the VERY START, before any
        // input resolution or array allocation. Without this, ops that ran between
        // decode steps (token sampling, embedding) can leave the thread on a different
        // device, causing Nd4j.create* / arange / scalar calls during input resolution
        // to allocate arrays on the wrong GPU. On non-peer multi-GPU systems, those
        // wrong-device arrays then require cross-device migration which can produce
        // stale pointers and CUDA error 700.
        if (nativeExecutionDevice >= 0) {
            int currentDevice = DeviceMemoryManager.getInstance().getCurrentDeviceId();
            if (currentDevice != nativeExecutionDevice) {
                DeviceMemoryManager.getInstance().switchDevice(nativeExecutionDevice,
                        "DSP.executeNative", "pin-to-plan-device");
            }
        }

        // Native plan compilation is explicit (or controlled by InferenceSession auto-compile).
        if (!isNativePlanCompiled(plan)) {
            throw new RuntimeException("Native executor: plan not precompiled for native execution. " +
                    "Ensure compileNativePlan() is called before executeNative().");
        }

        // Re-dispatch through the C++ NativePlanCache for current placeholder shapes.
        // O(1) cache hit for matching shapes; swaps to a different plan when shapes drift.
        // This is what lets the plan cache's shape-keyed dispatch enforce slot immutability —
        // each shape-sig gets its own plan with its own bound slots.
        redispatchForCurrentShapes(placeholderArrays);

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
        // Track indices of stale non-placeholder inputs refreshed in the fast path.
        // Declared here (outer scope) so the frozen fast path can access them.
        int[] staleNonPlaceholderIndices = null;
        int staleNonPlaceholderCount = 0;
        if (shapesFrozen && cachedInputArrays != null && cachedInputArrays.length == extKeys.length) {
            DspDiagnostics.record(DspDiagnostics.EXECUTE,
                    "Java: external inputs FAST PATH (frozen, " + extKeys.length + " cached)");
            // Fast path: reuse cached constant/variable arrays, only re-resolve placeholders.
            // Use a separate array so we don't corrupt cachedInputArrays (needed for identity comparison).
            extInputs = new INDArray[extKeys.length];
            System.arraycopy(cachedInputArrays, 0, extInputs, 0, extKeys.length);
            // Re-resolve any inputs whose DataBuffer has been freed between steps.
            // This can happen when setCloseable(true)+close() is called on KV outputs
            // that share a DataBuffer with past_key_values inputs.
            // Constants are protected by protectedConstantBuffers and should never be stale.
            int staleCount = 0;
            int resolvedCount = 0;
            int staleConstantCount = 0;
            int stalePlaceholderCount = 0;
            int staleOtherCount = 0;
            for (int i = 0; i < extKeys.length; i++) {
                if (extInputs[i] != null && !isArrayLive(extInputs[i])) {
                    staleCount++;
                    SDVariable var = sd.getVariable(extKeys[i]);
                    VariableType vt = var != null ? var.getVariableType() : null;
                    if (var != null && (vt == VariableType.CONSTANT
                            || vt == VariableType.VARIABLE)) {
                        staleConstantCount++;
                        INDArray fresh = var.getArr();
                        if (isArrayLive(fresh)) {
                            extInputs[i] = fresh;
                            cachedInputArrays[i] = fresh;
                            if (staleNonPlaceholderIndices == null) staleNonPlaceholderIndices = new int[16];
                            if (staleNonPlaceholderCount >= staleNonPlaceholderIndices.length)
                                staleNonPlaceholderIndices = java.util.Arrays.copyOf(staleNonPlaceholderIndices, staleNonPlaceholderIndices.length * 2);
                            staleNonPlaceholderIndices[staleNonPlaceholderCount++] = i;
                            resolvedCount++;
                        } else {
                            throw new RuntimeException(
                                "LIFECYCLE_ERROR: external input '" + extKeys[i] + "' (type=" + vt +
                                ") DataBuffer was closed between DSP executions — " +
                                "constant/variable was freed while plan active. " +
                                "This indicates a bug in session cleanup: protectedConstantBuffers " +
                                "should have prevented this closure. " +
                                "(dtype=" + (fresh != null ? fresh.dataType() : "unknown") +
                                ", shape=" + (fresh != null ? Arrays.toString(fresh.shape()) : "unknown") + ")");
                        }
                    } else if (vt == VariableType.PLACEHOLDER && placeholderArrays != null) {
                        stalePlaceholderCount++;
                        INDArray ph = placeholderArrays.get(extKeys[i]);
                        if (ph != null && ph.data() != null && !ph.data().wasClosed()) {
                            extInputs[i] = ph;
                            resolvedCount++;
                        } else {
                            DspDiagnostics.record(DspDiagnostics.FALLBACK,
                                "Java: ext[" + i + "] '" + extKeys[i] + "' type=PLACEHOLDER" +
                                " STALE, placeholder not available in map");
                        }
                    } else {
                        staleOtherCount++;
                        DspDiagnostics.record(DspDiagnostics.FALLBACK,
                            "Java: ext[" + i + "] '" + extKeys[i] + "' type=" + vt +
                            " STALE but not CONST/VAR/PLACEHOLDER — cannot re-resolve!");
                    }
                }
            }
            if (staleCount > 0) {
                DspDiagnostics.record(DspDiagnostics.MEMORY,
                    "Java: external inputs fast path: " + staleCount + " stale, " +
                    resolvedCount + " resolved, " + (staleCount - resolvedCount) + " unresolvable");
                log.info("STALE_BUFFER_SCAN: total={} constants={} placeholders={} other={} resolved={}",
                        staleCount, staleConstantCount, stalePlaceholderCount, staleOtherCount, resolvedCount);
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
                                    var.getVariableType() == VariableType.VARIABLE ||
                                    var.getVariableType() == VariableType.ARRAY)) {
                        arr = var.getArr();
                        if (arr != null && !isArrayLive(arr)) {
                            throw new RuntimeException(
                                "LIFECYCLE_ERROR: external input '" + varName + "' (type=" +
                                var.getVariableType() + ") DataBuffer was closed between DSP executions — " +
                                "constant/variable was freed while plan active. " +
                                "This indicates a bug in session cleanup: protectedConstantBuffers " +
                                "should have prevented this closure. " +
                                "(dtype=" + arr.dataType() + ", shape=" + Arrays.toString(arr.shape()) + ")");
                        }
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
                //
                // We use AffinityManager.getDeviceForArray() instead of dbDeviceId()
                // because dbDeviceId queries the CUDA device pointer which may be null
                // for Java-allocated arrays that haven't been synced to device yet.
                // getDeviceForArray() reads the AllocationPoint metadata that the
                // allocator sets at buffer creation time — it's always populated and
                // reflects the logical home device for the buffer.
                long[] deviceBytes = new long[numDevices];
                for (INDArray arr : extInputs) {
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        // Pass -1 as fallback so host-only / unplaced arrays don't
                        // bias selection toward the current thread's device.
                        int devId = resolveArrayDevice(arr, numDevices, -1);
                        if (devId >= 0 && devId < numDevices) {
                            deviceBytes[devId] += arr.length() * arr.data().getElementSize();
                        }
                    }
                }

                // Data locality is the primary selection criterion: native kernels
                // (Triton, cuBLAS, scalarSimpleShaped, etc.) launch with arg tables
                // containing raw device pointers, and those pointers are only valid
                // on the device where the DataBuffer was allocated. Executing on a
                // different device produces CUDA error 700 (illegal memory access),
                // which poisons the CUDA context for the entire process.
                //
                // Pick the device holding the largest share of input bytes. Ties and
                // empty-data edge cases (no external inputs placed on any device)
                // fall back to free-memory selection via selectBestGpu().
                int bestDataDevice = -1;
                long bestDataBytes = 0;
                boolean dataTied = false;
                for (int d = 0; d < numDevices; d++) {
                    if (deviceBytes[d] > bestDataBytes) {
                        bestDataDevice = d;
                        bestDataBytes = deviceBytes[d];
                        dataTied = false;
                    } else if (deviceBytes[d] == bestDataBytes && bestDataBytes > 0) {
                        dataTied = true;
                    }
                }

                int bestDevice;
                if (bestDataDevice >= 0 && !dataTied) {
                    bestDevice = bestDataDevice;
                } else {
                    // No data placed on any device, or exact tie: fall back to
                    // pool-aware free-memory selection.
                    bestDevice = DeviceMemoryManager.getInstance().selectBestGpu();
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

            // Switch CUDA context to the target device if we're not already on it.
            // This is independent from input migration below — the thread may already
            // be on nativeExecutionDevice yet still receive inputs from a different
            // device (second call with the same cached plan but fresh placeholders).
            if (nativeExecutionDevice != previousDevice) {
                DeviceMemoryManager.getInstance().switchDevice(nativeExecutionDevice,
                        "DSP.executeNative", "multi-gpu-coherency");
                deviceSwitched = true;

                // Invalidate cached exec stream — it belongs to the previous device
                cachedExecStream = null;
                execStreamCached = false;
            }

            // Migrate any off-device inputs to nativeExecutionDevice, regardless of
            // whether we just switched devices. The cached nativeExecutionDevice may
            // match previousDevice while inputs still come from a different device
            // (e.g. testCrossDeviceMatmulExecution feeds a device-1 placeholder into
            // a plan that was captured on device 0). Without this migration kernels
            // launch with cross-device pointers and trigger CUDA error 700.
            //
            // For non-peer GPUs, cross-device memory access causes error 700.
            // Use replicateToDevice() instead of dup() because dup() does a direct
            // GPU-to-GPU cudaMemcpy which requires peer access. replicateToDevice()
            // stages through host memory for non-peer GPUs.
            // Cache constant replicas to avoid re-copying model weights every step.
            {
                if (nativeConstantReplicaCache == null) {
                    nativeConstantReplicaCache = new HashMap<>();
                }

                // Get device management subsystems for tracking
                TransferSubsystem transferSubsystem = null;
                ReplicaLeakDetector replicaDetector = null;
                PointerStabilityGuard stabilityGuard = null;
                try {
                    transferSubsystem = Nd4j.framework.device().transfers();
                    replicaDetector = Nd4j.framework.device().replicaLeaks();
                    stabilityGuard = Nd4j.framework.device().pointerStability();
                } catch (Exception e) {
                    // Subsystems may not be initialized - continue without tracking
                }

                int migratedCount = 0;
                long migratedBytes = 0;
                for (int i = 0; i < extInputs.length; i++) {
                    INDArray arr = extInputs[i];
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        int arrDevice = resolveArrayDevice(arr, numDevices, nativeExecutionDevice);
                        if (arrDevice >= 0 && arrDevice != nativeExecutionDevice) {
                            // Check if migration is blocked by frozen buffer (pointer stability)
                            if (stabilityGuard != null && stabilityGuard.isFrozen(arr)) {
                                // Skip migration - buffer is frozen for graph replay
                                if (log.isDebugEnabled()) {
                                    log.debug("Skipping migration for input[{}]: buffer is frozen (graph replay)", i);
                                }
                                continue;
                            }
                            
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
                            log.info("DSP MIGRATE: ext[{}] '{}' shape={} dtype={} from device {} to {} (placeholder={}, view={})",
                                    i, extKeys != null && i < extKeys.length ? extKeys[i] : "?",
                                    java.util.Arrays.toString(arr.shape()), arr.dataType(),
                                    arrDevice, nativeExecutionDevice, isPlaceholder, arr.isView());
                            long startTime = System.nanoTime();
                            INDArray migrated = Nd4j.getAffinityManager().replicateToDevice(
                                    nativeExecutionDevice, arr);
                            long durationNs = System.nanoTime() - startTime;

                            extInputs[i] = migrated;
                            migratedCount++;
                            long elemSize = arr.data().getElementSize();
                            long bytes = arr.length() * elemSize;
                            migratedBytes += bytes;

                            // Record transfer event if tracking is enabled
                            if (transferSubsystem != null && transferSubsystem.isEnabled()) {
                                transferSubsystem.record(TransferEvent.builder()
                                    .variableName(extKeys != null && i < extKeys.length ? extKeys[i] : null)
                                    .sourceDeviceId(arrDevice)
                                    .destDeviceId(nativeExecutionDevice)
                                    .direction(TransferDirection.D2D)
                                    .reason(TransferReason.CONSTANT_REPLICATION)
                                    .bytes(bytes)
                                    .durationNanos(durationNs)
                                    .callerContext("DSP.executeNative")
                                    .build());
                            }

                            // Register replica for leak detection if enabled
                            if (replicaDetector != null && replicaDetector.isEnabled() && isCacheable) {
                                replicaDetector.registerReplica(migrated, 
                                    extKeys != null && i < extKeys.length ? extKeys[i] : "input[" + i + "]",
                                    arrDevice, nativeExecutionDevice);
                            }

                            // Cache non-placeholder replicas for reuse across decode steps
                            if (isCacheable) {
                                nativeConstantReplicaCache.put(i, migrated);
                            }

                            // Update frozen cache AND invalidate cached opaque pointer.
                            // The frozen fast path compares extInputs[i] identity against
                            // cachedInputArrays[i] to decide whether to re-set the C++ opContext.
                            // Migration changes both to the same migrated object, so identity
                            // matches and the stale C++ pointer is never updated.
                            // Nulling cachedInputOpaques[i] forces the frozen path to re-set it.
                            if (cachedInputArrays != null && i < cachedInputArrays.length) {
                                cachedInputArrays[i] = migrated;
                            }
                            if (cachedInputOpaques != null && i < cachedInputOpaques.length) {
                                cachedInputOpaques[i] = null;
                            }
                            // Track migrated non-placeholder for opContext re-set in frozen path.
                            // The frozen fast path only iterates placeholderIndices — migrated
                            // constants/variables would be skipped, leaving C++ with stale pointers.
                            if (!isPlaceholder) {
                                if (staleNonPlaceholderIndices == null) staleNonPlaceholderIndices = new int[16];
                                if (staleNonPlaceholderCount >= staleNonPlaceholderIndices.length)
                                    staleNonPlaceholderIndices = java.util.Arrays.copyOf(
                                            staleNonPlaceholderIndices, staleNonPlaceholderIndices.length * 2);
                                staleNonPlaceholderIndices[staleNonPlaceholderCount++] = i;
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
                
                // Build cached frozen fast-path index lists on first use.
                // Placeholders and derived externals are phase-dynamic and must be
                // re-resolved on every frozen execution. Small integral root externals
                // are phase-stable but still need opContext refresh so value-shaped
                // native chains read current device-visible values.
                if ((placeholderIndices == null || frozenControlInputIndices == null
                        || frozenDerivedExternalInputIndices == null) && inputIsPlaceholder != null) {
                    int count = 0;
                    for (boolean b : inputIsPlaceholder) if (b) count++;
                    placeholderIndices = new int[count];
                    int derivedCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        if (!inputIsPlaceholder[i] && isDerivedExternalInput(extKeys[i])) {
                            derivedCount++;
                        }
                    }
                    frozenDerivedExternalInputIndices = new int[derivedCount];
                    int controlCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        if (!inputIsPlaceholder[i]
                                && !isDerivedExternalInput(extKeys[i])
                                && isSmallIntegralControlArray(extInputs[i])) {
                            controlCount++;
                        }
                    }
                    frozenControlInputIndices = new int[controlCount];
                    int idx = 0;
                    int derivedIdx = 0;
                    int controlIdx = 0;
                    for (int i = 0; i < inputIsPlaceholder.length; i++) {
                        if (inputIsPlaceholder[i]) {
                            placeholderIndices[idx++] = i;
                        } else if (isDerivedExternalInput(extKeys[i])) {
                            frozenDerivedExternalInputIndices[derivedIdx++] = i;
                        } else if (isSmallIntegralControlArray(extInputs[i])) {
                            frozenControlInputIndices[controlIdx++] = i;
                        }
                    }
                    log.info("FROZEN_INPUT_OPT: built placeholderIndices[{}], derivedIndices[{}], controlIndices[{}] (extInputs={})",
                            count, derivedCount, controlCount, extInputs.length);
                    // Validate: check that all non-constant external inputs are in placeholderIndices.
                    // If a PLACEHOLDER variable is missing, it won't get synced on the frozen fast path.
                    int missingCount = 0;
                    for (int i = 0; i < extKeys.length; i++) {
                        if (!inputIsPlaceholder[i]) {
                            SDVariable var = sd.getVariable(extKeys[i]);
                            if (var != null && var.getVariableType() == VariableType.PLACEHOLDER) {
                                missingCount++;
                                log.warn("PLACEHOLDER_SYNC_GAP: ext[{}] '{}' is type PLACEHOLDER " +
                                        "but not in placeholderIndices — will be skipped on frozen fast path",
                                        i, extKeys[i]);
                            }
                        }
                    }
                    if (missingCount > 0) {
                        log.warn("PLACEHOLDER_SYNC_GAP: {} placeholder(s) missing from placeholderIndices " +
                                "(total extInputs={}, placeholderIndices={}). These inputs will NOT be " +
                                "synced during frozen execution.", missingCount, extKeys.length, count);
                    }
                }
                
                // Frozen fast path: only iterate placeholder indices (not all 1332 inputs)
                if (placeholderIndices != null) {
                    DspDiagnostics.record(DspDiagnostics.EXECUTE,
                        "Java: FROZEN_FAST_PATH entering placeholder loop, " +
                        placeholderIndices.length + " placeholders");
                    for (int pi : placeholderIndices) {
                        INDArray arr = extInputs[pi];
                        // Sync placeholder inputs to device. Java putScalar/assign writes to
                        // host and marks host as dirty (tickHostWrite), but the C++ NDArray
                        // wrapper created by OpaqueNDArray.fromINDArray doesn't inherit the
                        // Java-side actuality tracking. Without this sync, the C++ side sees
                        // sAct=true (device up-to-date) and skips the H2D copy, causing stale
                        // data to be used during CUDA graph replay.
                        if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                            arr.syncToDevice();
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, pi, opaqueIn);
                            cachedInputOpaques[pi] = opaqueIn;
                            cachedInputArrays[pi] = arr;
                        }
                    }
                    for (int di : frozenDerivedExternalInputIndices) {
                        INDArray arr = resolveCanonicalExternalInput(extKeys[di], placeholderArrays);
                        if (arr == null || arr.data() == null || arr.data().wasClosed()) {
                            throw new IllegalStateException("Frozen replay phase violation: derived external input '"
                                    + extKeys[di] + "' is not live during frozen execution");
                        }
                        extInputs[di] = arr;
                        if (isSmallIntegralControlArray(arr)) {
                            arr.syncToDevice();
                        }
                        OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                        nativeOps.setGraphContextInputArray(opContext, di, opaqueIn);
                        cachedInputOpaques[di] = opaqueIn;
                        cachedInputArrays[di] = arr;
                    }
                    for (int ci : frozenControlInputIndices) {
                        INDArray arr = resolveCanonicalExternalInput(extKeys[ci], placeholderArrays);
                        if (arr == null) {
                            arr = extInputs[ci];
                        }
                        if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                            extInputs[ci] = arr;
                            arr.syncToDevice();
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, ci, opaqueIn);
                            cachedInputOpaques[ci] = opaqueIn;
                            cachedInputArrays[ci] = arr;
                        }
                    }
                    // Re-set any stale non-placeholder inputs (constants/variables) that
                    // were refreshed in the stale buffer scan above. The placeholder loop
                    // only iterates placeholderIndices, so these would otherwise be skipped,
                    // leaving C++ with stale pointers to freed GPU memory.
                    if (staleNonPlaceholderIndices != null && staleNonPlaceholderCount > 0) {
                        for (int sc = 0; sc < staleNonPlaceholderCount; sc++) {
                            int ci = staleNonPlaceholderIndices[sc];
                            INDArray arr = extInputs[ci];
                            if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                                nativeOps.setGraphContextInputArray(opContext, ci, opaqueIn);
                                cachedInputOpaques[ci] = opaqueIn;
                                DspDiagnostics.record(DspDiagnostics.MEMORY,
                                    "Java: FROZEN_FAST_PATH re-set stale constant ext[" + ci + "] '" + extKeys[ci] + "'");
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
                            // Guard: ensure buffer is live before creating OpaqueNDArray.
                            // Control flow dead branches may have closed buffers.
                            INDArray arrToSet = extInputs[i];
                            if (arrToSet != null && (arrToSet.data() == null || arrToSet.data().wasClosed())) {
                                try {
                                    arrToSet = Nd4j.zeros(arrToSet.dataType(), arrToSet.shape());
                                } catch (Exception e) {
                                    arrToSet = Nd4j.scalar(0.0f);
                                }
                                extInputs[i] = arrToSet;
                            }
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arrToSet);
                            nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                            cachedInputOpaques[i] = opaqueIn;
                            cachedInputArrays[i] = extInputs[i];
                        } else if (inputIsPlaceholder != null && inputIsPlaceholder[i]) {
                            // All placeholders — including decode input_ids / position_ids /
                            // attention_mask — are plain Java-managed NDArrays; sync host to device.
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
                // IMPORTANT: Keep old refs alive until ALL new refs are populated.
                // If we replace contextInputRefs before the loop completes, old OpaqueNDArrays
                // become GC-eligible and the DeallocatorService can delete C++ NDArrays while
                // the opContext still holds raw pointers to them → _buffer becomes nullptr.
                OpaqueNDArray[] oldRefs = contextInputRefs;  // prevent GC of old wrappers
                OpaqueNDArray[] newRefs = new OpaqueNDArray[extInputs.length];
                for (int i = 0; i < extInputs.length; i++) {
                    INDArray arr = extInputs[i];
                    // Guard against closed buffers from control flow dead branches.
                    // Switch/Merge ops selectively enable/disable branches; dead-branch
                    // outputs may have their DataBuffers freed by session cleanup. Creating
                    // an OpaqueNDArray from a closed buffer causes PointerWrapper::pointer()
                    // null dereference in native code. Replace with a zero-filled placeholder
                    // of the same shape/dtype — the native plan's control flow routing will
                    // never actually read from the dead branch's slot.
                    if (arr != null && (arr.data() == null || arr.data().wasClosed())) {
                        // Try to resolve a fresh copy from the variable
                        SDVariable var = sd.getVariable(extKeys[i]);
                        if (var != null) {
                            INDArray fresh = var.getArr();
                            if (fresh != null && fresh.data() != null && !fresh.data().wasClosed()) {
                                arr = fresh;
                            } else {
                                // Create zeros — guard against null shape/dtype from dead arrays
                                try {
                                    arr = Nd4j.zeros(arr.dataType(), arr.shape());
                                } catch (Exception e) {
                                    arr = Nd4j.scalar(0.0f);
                                }
                            }
                        } else {
                            try {
                                arr = Nd4j.zeros(arr.dataType(), arr.shape());
                            } catch (Exception e) {
                                arr = Nd4j.scalar(0.0f);
                            }
                        }
                        extInputs[i] = arr;
                    }
                    if (arr == null || arr.data() == null || arr.data().wasClosed()) {
                        log.error("DEAD_INPUT_AT_SET: ext[{}] '{}' still dead after guard (arr={}, data={})",
                                i, extKeys[i], arr != null ? "id=" + arr.getId() : "null",
                                arr != null ? (arr.data() == null ? "null" : "closed=" + arr.data().wasClosed()) : "N/A");
                        arr = Nd4j.scalar(DataType.FLOAT, 0.0f);
                        extInputs[i] = arr;
                    }
                    OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                    nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                    newRefs[i] = opaqueIn;
                }
                contextInputRefs = newRefs;  // atomically swap after all refs are set
                // Cache for subsequent frozen calls
                if (shapesFrozen) {
                    cachedInputOpaques = new OpaqueNDArray[extInputs.length];
                    cachedInputArrays = new INDArray[extInputs.length];
                    System.arraycopy(extInputs, 0, cachedInputArrays, 0, extInputs.length);
                    inputIsPlaceholder = new boolean[extInputs.length];
                    // Build cached frozen fast-path index lists for subsequent calls.
                    int placeholderCount = 0;
                    int derivedCount = 0;
                    int controlCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        cachedInputOpaques[i] = OpaqueNDArray.fromINDArray(extInputs[i]);
                        // Mark as placeholder if it came from the placeholderArrays map
                        inputIsPlaceholder[i] = placeholderArrays != null
                                && placeholderArrays.containsKey(extKeys[i]);
                        if (inputIsPlaceholder[i]) placeholderCount++;
                        else if (isDerivedExternalInput(extKeys[i])) derivedCount++;
                        else if (isSmallIntegralControlArray(extInputs[i])) controlCount++;
                    }
                    // Build frozen fast-path index arrays
                    placeholderIndices = new int[placeholderCount];
                    frozenDerivedExternalInputIndices = new int[derivedCount];
                    frozenControlInputIndices = new int[controlCount];
                    int idx = 0;
                    int derivedIdx = 0;
                    int controlIdx = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        if (inputIsPlaceholder[i]) placeholderIndices[idx++] = i;
                        else if (isDerivedExternalInput(extKeys[i])) frozenDerivedExternalInputIndices[derivedIdx++] = i;
                        else if (isSmallIntegralControlArray(extInputs[i])) frozenControlInputIndices[controlIdx++] = i;
                    }
                    log.info("FROZEN_INPUT_OPT: built placeholderIndices[{}], derivedIndices[{}], controlIndices[{}] (extInputs={})",
                            placeholderCount, derivedCount, controlCount, extInputs.length);
                    // Validate: check that all non-constant external inputs are in placeholderIndices.
                    int missingCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        if (!inputIsPlaceholder[i]) {
                            SDVariable var = sd.getVariable(extKeys[i]);
                            if (var != null && var.getVariableType() == VariableType.PLACEHOLDER) {
                                missingCount++;
                                log.warn("PLACEHOLDER_SYNC_GAP: ext[{}] '{}' is type PLACEHOLDER " +
                                        "but not in placeholderIndices — will be skipped on frozen fast path",
                                        i, extKeys[i]);
                            }
                        }
                    }
                    if (missingCount > 0) {
                        log.warn("PLACEHOLDER_SYNC_GAP: {} placeholder(s) missing from placeholderIndices " +
                                "(total extInputs={}, placeholderIndices={}). These inputs will NOT be " +
                                "synced during frozen execution.", missingCount, extInputs.length, placeholderCount);
                    }
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

            // ── Java-side lifecycle validation (frozen execution) ──────────────
            // On first frozen execution: external buffer snapshot.
            // On subsequent frozen executions: validate buffers haven't been
            // closed, replaced, or had their shapes change.
            // Violations are IllegalStateException (hard error, not log).
            if (shapesFrozen && frozenCallCount == 1) {
                // Capture snapshot of external input buffers and shapes
                frozenExtBufferSnapshot = new DataBuffer[extInputs.length];
                frozenExtShapeSnapshot = new long[extInputs.length][];
                for (int i = 0; i < extInputs.length; i++) {
                    if (extInputs[i] != null) {
                        frozenExtBufferSnapshot[i] = extInputs[i].data();
                        frozenExtShapeSnapshot[i] = extInputs[i].shape();
                    }
                }
                log.info("LIFECYCLE: captured frozen external input snapshot ({} inputs)", extInputs.length);
            } else if (shapesFrozen && frozenCallCount > 1
                       && frozenExtBufferSnapshot != null) {
                // Validate: no buffer closed, no shape change for non-placeholders
                for (int i = 0; i < Math.min(extInputs.length, frozenExtBufferSnapshot.length); i++) {
                    if (frozenExtBufferSnapshot[i] == null) continue;
                    if (extInputs[i] == null) {
                        throw new IllegalStateException(
                                "LIFECYCLE_ERROR: external input " + i + " (" + extKeys[i] +
                                ") was non-null at freeze but is NULL now — buffer freed during frozen execution");
                    }
                    DataBuffer currentDb = extInputs[i].data();
                    if (currentDb != null && currentDb.wasClosed()) {
                        throw new IllegalStateException(
                                "LIFECYCLE_ERROR: external input " + i + " (" + extKeys[i] +
                                ") DataBuffer is CLOSED during frozen execution — " +
                                "use-after-free will occur. Close frozen session before freeing buffers.");
                    }
                    // Shape check for non-placeholders only (placeholders may have same shape
                    // but different data, which is fine)
                    if (inputIsPlaceholder != null && !inputIsPlaceholder[i]
                            && frozenExtShapeSnapshot[i] != null && extInputs[i].shape() != null) {
                        long[] snapShape = frozenExtShapeSnapshot[i];
                        long[] currShape = extInputs[i].shape();
                        if (!java.util.Arrays.equals(snapShape, currShape)) {
                            throw new IllegalStateException(
                                    "LIFECYCLE_ERROR: external input " + i + " (" + extKeys[i] +
                                    ") shape changed during frozen execution: " +
                                    java.util.Arrays.toString(snapShape) + " → " +
                                    java.util.Arrays.toString(currShape) +
                                    ". Unfreeze shapes before changing constant/variable shapes.");
                        }
                    }
                }
            }

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
                if (status == NATIVE_STATUS_STALE_BUFFER) {
                    // C++ detected a closed/destroyed DataBuffer. This means a constant or
                    // variable was GC'd between Java's input resolution and C++ execution.
                    // Throw a specific exception so callers can re-resolve and retry.
                    throw new IllegalStateException("Stale buffer detected by C++ during DSP execution: " +
                            (errMsg != null ? errMsg : "unknown input"));
                }
                throw new RuntimeException("Native plan execution failed with status " + status +
                        ": " + (errMsg != null ? errMsg : "unknown error"));
            }

            executionCount++;

            DspDiagnostics.recordTimed(DspDiagnostics.EXECUTE, -1, -1, "executeNative",
                    execMs * 1000, "Java: native execution OK " + execMs + "ms" +
                    " frozen=" + shapesFrozen + " executionCount=" + executionCount);

            // ── Always-on: validate output arrays immediately after native returns ──
            // C++ execution succeeded (status=0) but output arrays may still be
            // invalid (null OpaqueDataBuffer, closed buffer, wrong device). Catch
            // these NOW rather than when getFloat()/toFloatVector() crashes later.
            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                if (opaqueOut == null || opaqueOut.isNull()) {
                    throw new IllegalStateException(
                        "ARRAY_INVALID: C++ returned null OpaqueNDArray for output " + i +
                        " after successful DSP execution (executionCount=" + executionCount +
                        ", frozen=" + shapesFrozen + "). " +
                        "This indicates the output slot was never populated by any op.");
                }
                // Check the underlying buffers are accessible
                Pointer specialBuf = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);
                if (length > 0 && (specialBuf == null || specialBuf.isNull())) {
                    Pointer primaryBuf = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                    if (primaryBuf == null || primaryBuf.isNull()) {
                        throw new IllegalStateException(
                            "ARRAY_INVALID: output " + i + " has length=" + length +
                            " but both GPU and host buffers are null after DSP execution " +
                            "(executionCount=" + executionCount + ", frozen=" + shapesFrozen + "). " +
                            "The DataBuffer was likely closed/freed during execution.");
                    }
                }
            }

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
            // copy all requested outputs — KV cache append is now an ordinary in-graph op
            // whose writes land directly in Java-owned static buffers, so nothing gets skipped).
            //
            // IMPORTANT: Only use the zeroCopyOutputCache when directOutputMode=true.
            // When called from SameDiff.output() (directOutputMode=false), the caller DUPs all
            // results into independent copies. KV close in StaticKvCacheDecodeLoop only closes
            // those duped copies — NOT the cached originals — leaving zeroCopyOutputCache holding
            // stale data (previous step's logits) while appearing valid to the staleness guard.
            // Using the stale cache on the next outputDirect() call returns wrong tokens.
            // By skipping the cache entirely for non-direct calls we force fresh allocation,
            // and the cache is only built/used for direct calls where the caller uses the
            // returned references directly (no dup), so KV close invalidates the cache correctly.
            //
            // Guard: if any cached output array has been externally closed (e.g., KV outputs
            // closed by StaticKvCacheDecodeLoop after scatter), the cache is stale. Drop it so
            // we fall through to the allocation path below and rebuild a fresh cache.
            if (directOutputMode && shapesFrozen && zeroCopyOutputCache != null) {
                for (INDArray arr : zeroCopyOutputCache.values()) {
                    if (arr == null || arr.wasClosed() || arr.data() == null || arr.data().wasClosed()) {
                        log.info("Native executor: zeroCopyOutputCache stale (closed array detected) — rebuilding");
                        closeZeroCopyOutputCache();
                        break;
                    }
                }
            }
            if (directOutputMode && shapesFrozen && zeroCopyOutputCache != null) {
                int copiedOutputs = 0;
                for (int i = 0; i < numOutputs; i++) {
                    String outputName = requestedOutputs.get(i);

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

                // Sync stream to ensure async D2D copies complete before returning
                Nd4j.getExecutioner().commit();

                long copyMs = (System.nanoTime() - copyStart) / 1_000_000;
                if (execMs > 100) {
                    log.info("Native executor: exec={}ms copy={}ms (frozen, {}/{} outputs copied)",
                            execMs, copyMs, copiedOutputs, numOutputs);
                }
                return zeroCopyOutputCache;
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            for (int i = 0; i < numOutputs; i++) {
                String outputName = requestedOutputs.get(i);

                // When a constant or variable is directly requested as an output (not an op
                // output), it has no slot index in the native plan. The native plan writes -1
                // for its slot, so the C++ side sets requestedOutputs[i] = nullptr and the
                // opContext output is the initial dummy. Detect this by checking if the output
                // variable is a CONSTANT or VARIABLE in the SameDiff graph (not an op output).
                SDVariable sdVar = sd.getVariable(outputName);
                if (sdVar != null && (sdVar.getVariableType() == VariableType.CONSTANT
                        || sdVar.getVariableType() == VariableType.VARIABLE)) {
                    INDArray sdArr = sd.getArrForVarName(outputName);
                    if (sdArr != null) {
                        results.put(outputName, sdArr.dup());
                        continue;
                    }
                }

                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                if (opaqueOut == null || opaqueOut.isNull()) {
                    throw new RuntimeException("Native executor: null output at index " + i + " for '" + outputName + "'");
                }

                // Read shape info from the C++ output NDArray
                long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                long[] shape = Shape.shape(shapeInfo);
                long[] strides = Shape.stride(shapeInfo);
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);
                char ordering = Shape.order(shapeInfo);

                // Create a Java-owned INDArray with the EXACT strides from the C++ output.
                // The raw buffer copy below is a flat memcpy — the destination must have
                // matching strides so elements are interpreted correctly. If the C++ output
                // has non-contiguous strides (e.g., from a view-based permute op whose shape
                // function inherited the input's strides), using contiguous strides here
                // would mis-interpret the buffer layout and produce wrong results.
                INDArray result = Nd4j.createUninitialized(dtype, shape, strides, ordering);

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

                // If the C++ output had non-contiguous strides (from view ops like permute),
                // dup to contiguous layout so downstream Java code doesn't need to handle
                // non-contiguous arrays. This is a safety net — ideally C++ shape functions
                // should return contiguous strides for output arrays.
                if (shape.length > 1) {
                    boolean isContiguous = true;
                    if (ordering == 'c') {
                        long expected = 1;
                        for (int d = shape.length - 1; d >= 0; d--) {
                            if (strides[d] != expected) { isContiguous = false; break; }
                            expected *= shape[d];
                        }
                    } else {
                        long expected = 1;
                        for (int d = 0; d < shape.length; d++) {
                            if (strides[d] != expected) { isContiguous = false; break; }
                            expected *= shape[d];
                        }
                    }
                    if (!isContiguous) {
                        INDArray contiguous = result.dup(ordering);
                        result = contiguous;
                    }
                }

                // Verify copy correctness: print first values immediately after copyBuffer
                if (Nd4j.getEnvironment().isDebugAndVerbose() && result.rank() >= 2 && result.length() > 0) {
                    result.syncToHost();
                    long lastPos = result.rank() == 3 ? result.size(1) - 1 : 0;
                    StringBuilder sb = new StringBuilder("DSP_COPY_VERIFY[" + outputName + "] shape=");
                    sb.append(java.util.Arrays.toString(result.shape()));
                    sb.append(" first5=[");
                    for (int v = 0; v < Math.min(5, (int)result.length()); v++) {
                        if (v > 0) sb.append(",");
                        sb.append(String.format("%.4f", result.getFloat(v)));
                    }
                    sb.append("] lastPos5=[");
                    if (result.rank() == 3) {
                        long vocabSize = result.size(2);
                        for (int v = 0; v < Math.min(5, (int)vocabSize); v++) {
                            if (v > 0) sb.append(",");
                            sb.append(String.format("%.4f", result.getFloat(0, lastPos, v)));
                        }
                    }
                    sb.append("]");
                    log.info(sb.toString());
                }
                results.put(outputName, result);
            }

            // Synchronize the CUDA stream to ensure all async D2D copies (copyBuffer)
            // have completed before returning. Without this, the caller may destroy the
            // native plan (freeing source output buffers) while the async copies are still
            // in flight — causing the destination arrays to contain garbage or zeros.
            // This is critical for the prefill path where the plan is destroyed and
            // recompiled for static KV shapes immediately after this method returns.
            Nd4j.getExecutioner().commit();

            // Cache allocated arrays for reuse on subsequent frozen direct-mode executions.
            // Only cache when directOutputMode=true: when called from SameDiff.output() the
            // results are duped by the caller so caching here would create a stale cache that
            // holds previous-step logits yet appears valid to the staleness guard.
            if (directOutputMode && shapesFrozen && zeroCopyOutputCache == null && !results.isEmpty()) {
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
            // DO NOT restore to previousDevice. Stay on the plan's execution device
            // (nativeExecutionDevice) so that arrays created between decode steps
            // (position_ids, attention_mask, input_ids) are allocated on the correct
            // device. Restoring to previousDevice causes allocations on the wrong
            // device, requiring cross-device migration on the next step, which leads
            // to CUDA error 700 on non-peer GPU configurations.
            //
            // The thread's device affinity is a global resource — CudaExecutioner's
            // per-op device routing can change it at any time. By staying on the plan's
            // device after execution, we ensure the decode loop's allocations land on
            // the device where they'll be consumed.
        }
    }

    /**
     * Release the native plan handle with a descriptive reason for diagnostics.
     * Handles are always destroyed here so replay state cannot survive across
     * plan changes or session resets.
     *
     * @param reason descriptive reason for plan destruction (e.g., "SESSION_RESET", "PLAN_RECOMPILATION")
     */
    private void freeNativePlanHandle(String reason) {
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            log.info("PLAN_DESTRUCTION: reason='{}' handle={} execCount={} frozen={}",
                    reason, nativePlanHandle.address(), executionCount, shapesFrozen);
        }
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
            // Plan lifetime is managed by sd::graph::NativePlanCache (C++) — do NOT free directly.
            // The cache evicts (and deletes) entries under LRU + memory budget policy.
            // SameDiff.close() frees the cache via freeNativePlanCache(), which releases all entries.
            // Unpin so this plan becomes eligible for eviction when the cache needs space.
            try {
                Pointer cache = sd.getOrCreateNativePlanCache();
                if (cache != null && !cache.isNull()) {
                    NativeOps nativeOps2 = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    nativeOps2.unpinNativePlan(cache, nativePlanHandle);
                }
            } catch (Exception e) {
                log.debug("    freeNativePlanHandle: unpin failed (non-fatal): {}", e.getMessage());
            }
            log.info("    freeNativePlanHandle: handle={} is cache-owned, unpinned", nativePlanHandle);
        }
        nativePlanHandle = null;
        nativePlanSource = null;
        configuredGraphExecutionMode = GraphExecutionMode.AUTO;
        // Dispatch-input cache and per-handle settings must also drop so
        // isNativePlanCompiled() returns false until compileNativePlan() runs again.
        cachedSerializedPlan = null;
        cachedSortedOutputs = null;
        cachedPhKeys = null;
        cachedEffectiveGraphModeCode = -1;
        cachedJitModeInt = -1;
        cachedCudaGraphsEnabled = false;
        cachedExecTiming = false;
        cachedTraceEnabled = false;
        configuredHandleAddresses.clear();
        cachedInputOpaques = null;
        cachedInputArrays = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        placeholderIndices = null;
        frozenControlInputIndices = null;
        frozenDerivedExternalInputIndices = null;
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        cachedExecStream = null;
        execStreamCached = false;
        closeZeroCopyOutputCache();
        closeNativeConstantReplicaCache();
        cachedRequestedOutputNames = null;
        nativeExecutionDevice = -1;
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

        // Free native C++ plan handle reference. The plan is cache-owned and will
        // be cleaned up by the NativePlanCache destructor (or LRU eviction).
        // Do NOT call releaseGpuIntermediates() here: close() is a final cleanup
        // and the cache destructor handles freeing GPU resources. Calling it here
        // would free C++ slot arrays that the cache destructor also frees, causing
        // a double free on JVM shutdown.
        log.info("  DSP close() step 6: freeNativePlanHandle");
        System.out.flush(); System.err.flush();
        freeNativePlanHandle("EXECUTOR_CLOSE");

        currentPlan = null;
        // Release strong refs to constant DataBuffers AFTER all cleanup steps.
        // Now that the plan is fully closed, these constants no longer need protection.
        protectedConstantBuffers = null;
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
