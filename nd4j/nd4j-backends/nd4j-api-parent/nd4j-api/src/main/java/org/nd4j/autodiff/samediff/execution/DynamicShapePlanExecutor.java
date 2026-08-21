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

import lombok.Getter;
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
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeBufferOwner;
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

import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

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

    /**
     * Process-global count of DynamicShapePlanExecutor instances that have reached
     * shapes-frozen state and may have CUDA graphs with baked TAD device pointers.
     *
     * <p>TAD offset device buffers are baked as kernel arguments into captured CUDA
     * graphs during the SHAPES_FROZEN phase. The global TAD cache (ConstantTadHelper)
     * holds the only strong reference to those device allocations. Calling
     * {@code Nd4j.clearTADCache()} while any frozen executor exists frees those
     * device buffers, making the baked CUDA-graph kernel args dangling → illegal
     * access (CUDA error 700) on the next {@code cudaGraphLaunch}.
     *
     * <p>Rules:
     * <ul>
     *   <li>Incremented (with CAS) when this executor transitions from unfrozen → frozen.
     *   <li>Decremented when the executor is reset ({@link #initialize(DynamicShapePlan)}
     *       with same-plan reset) or closed ({@link #close()}).
     * </ul>
     *
     * <p>InferenceSession queries {@link #hasFrozenExecutors()} before calling
     * {@code clearTADCache()}: if any frozen executor exists, the call is suppressed.
     */
    private static final AtomicInteger GLOBAL_FROZEN_EXECUTOR_COUNT =
            new AtomicInteger(0);

    /**
     * Returns true if at least one DynamicShapePlanExecutor process-wide has reached
     * shapes-frozen state (and therefore may have baked TAD device pointers into a
     * captured CUDA graph). When this returns true, {@code Nd4j.clearTADCache()} must
     * NOT be called — doing so would free the baked device buffers and cause err700.
     */
    public static boolean hasFrozenExecutors() {
        return GLOBAL_FROZEN_EXECUTOR_COUNT.get() > 0;
    }

    /**
     * True when THIS executor has incremented {@link #GLOBAL_FROZEN_EXECUTOR_COUNT}.
     * Prevents double-increment and ensures the decrement is paired correctly.
     */
    private boolean registeredAsFrozen;

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

    /** True once shapes have been frozen at least once in this executor's lifetime.
     *  Used to block plan cache swaps after the first freeze — swaps after freeze
     *  indicate a cache key instability bug and cause cascading performance loss.
     *  NOTE: when multi-plan frozen switching is active (shapes change while frozen),
     *  plan swaps due to DIFFERENT shapes are legitimate and not suppressed. */
    private boolean wasEverFrozen;

    /** Hash of placeholder shapes from the last dispatchNativePlan call.
     *  Used to detect when placeholder shapes change between executions so that
     *  the executor can redispatch to a different plan even when frozen.
     *  This enables the VLM multi-page pattern where prefill (seqLen=N) and
     *  decode (seqLen=1) alternate on the same frozen executor. The C++ NativePlanCache
     *  already handles shape-keyed dispatch — this hash just tells Java WHEN to call it. */
    private long lastDispatchedShapeHash;

    private final SameDiff sd;
    private final SessionMemMgr mmgr;

    /** The plan this executor is currently configured for. */
    @Getter
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

    /** Lock for synchronizing native plan execute+readback across threads.
     *  The C++ plan shares output slots (outputSlots_) across all executions.
     *  If thread A's readback (getOutputArrayNative + copyBuffer) races with
     *  thread B's execute (which overwrites the same output slots), A gets
     *  B's results (stale output). This lock spans the execute+readback window. */
    private final ReentrantLock nativeExecLock =
            new ReentrantLock();

    /** Native C++ plan handle. Compiled once from the serialized plan on first native
     *  execution attempt. Freed on close(). null means not yet compiled or compilation failed.
     *  Can be swapped across executeNative() calls by redispatchForCurrentShapes() when
     *  placeholder shapes change — the C++ NativePlanCache returns the shape-matched plan. */
    @Getter
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
    /** Explicit runtime diagnostic overrides survive native plan compilation, reset,
     *  and shape-keyed handle swaps. null means use the JVM property default. */
    private Boolean executionTimingOverride;
    private Boolean traceEnabledOverride;
    private int cachedEffectiveGraphModeCode = -1;   // -1 = leave default
    private final Set<Long> configuredHandleAddresses = new HashSet<>();

    /** Cache pins owned by this executor, including both sides of a frozen prefill/decode switch. */
    private final Map<Long, Pointer> pinnedPlanHandles = new HashMap<>();

    /**
     * Java external-input owners retained by each pinned native plan handle.
     *
     * <p>A frozen executor can alternate between shape-keyed native plans (for example VLM
     * prefill and decode). The active execution replaces {@link #externalInputs}, but the
     * inactive pinned plan still owns its previously-bound input buffers through captured native
     * pointers. Keep one Java snapshot per pinned handle so per-execution cache cleanup cannot
     * close an inactive plan's buffers before that handle is unpinned.</p>
     */
    private final Map<Long, INDArray[]> retainedExternalInputsByPlanHandle = new HashMap<>();

    /**
     * External inputs that are mutable at runtime even though their SameDiff
     * variable type is VARIABLE rather than PLACEHOLDER. Training parameters use
     * this path: their shapes are stable, but Java-side updaters change contents
     * between DSP replay calls.
     */
    private final Set<String> mutableExternalInputNames = new LinkedHashSet<>();
    private final Set<Long> mutableExternalInputsConfiguredHandleAddresses = new HashSet<>();

    /** Graph execution mode currently configured on the native plan handle. */
    private GraphExecutionMode configuredGraphExecutionMode = GraphExecutionMode.AUTO;

    /** If native compilation fails, disable native execution for this executor instance
     *  to avoid repeated failure overhead. */
    private boolean nativeExecutorFailed;

    /** True when the plan has zero slots (all outputs are direct placeholders/constants).
     *  execute() returns passthrough results without native compilation. */
    private boolean zeroSlotPassthrough;

    /** If CUDA graph capture fails, disable CUDA graphs but keep using slot-by-slot native execution. */
    private boolean cudaGraphsFailed;

    // Bespoke C++ KV cache retention state was removed — KV cache append now runs as
    // an ordinary in-graph op (KvScatter via KvCacheManager.scatterNewEntries) on every
    // decode step and is captured into the CUDA graph like any other op. The C++ DSP
    // plan is a pure graph executor: no decode-specific or KV-specific lifecycle.

    /** Cached OpaqueContext for native execution. Reused across executeNative() calls
     *  to avoid JNI create/delete overhead (~1-2ms). Freed on close(). */
    @Getter
    private OpaqueContext cachedOpContext;
    private int cachedOpContextInputCount;
    private int cachedOpContextOutputCount;

    /** Zero-copy output cache: when shapesFrozen, wraps C++ output pointers via
     *  dbCreateExternalDataBuffer instead of allocating + copyBuffer per step.
     *  These INDArrays point directly to C++ memory and must NOT be closed by callers.
     *  Cleared on close(), releaseGpuIntermediates(), and resetForNextPage(). */
    private Map<String, INDArray> zeroCopyOutputCache;

    /** Readback tracing for the frozen zero-copy refresh path: logs src/dst device
     *  addresses per output copy so pool-reuse mis-maps can be correlated with
     *  native DB_DELETE traces. See {@link ND4JSystemProperties#DSP_READBACK_TRACE}. */
    private static final boolean READBACK_TRACE =
            Boolean.getBoolean(ND4JSystemProperties.DSP_READBACK_TRACE);

    /** Cached OpaqueNDArray wrappers for external inputs when shapesFrozen.
     *  Avoids recreating wrappers + JNI setGraphContextInputArray calls each step.
     *  Only inputs that changed (by INDArray identity) are re-sent to C++. */
    private OpaqueNDArray[] cachedInputOpaques;
    private INDArray[] cachedInputArrays;

    /** Reusable working copy of cachedInputArrays for the frozen fast path.
     *  Avoids allocating a new INDArray[1332] array on every decode step. */
    private INDArray[] frozenExtInputsWorkingCopy;

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
    @Getter
    private IdentityHashMap<DataBuffer, Boolean> protectedConstantBuffers;

    /** Cached indices of placeholder inputs. Built on first frozen call to avoid
     *  iterating all 1332 external inputs every step. Only ~3 are placeholders
     *  (input_ids, attention_mask, position_ids). Saves ~0.5-1ms per step. */
    private int[] placeholderIndices;

    /** Cached indices of non-placeholder small integral control inputs. These
     *  arrays drive value-dependent shape/controller chains and must be refreshed
     *  into the native opContext during frozen execution just like placeholders. */
    private int[] frozenControlInputIndices;

    /** Cached indices of non-placeholder VARIABLE-type external inputs.
     *  Only these can be rebound via associateArrayWithVariable(). Built on first
     *  frozen call to avoid HashMap lookups on all 1332+ entries every step.
     *  In LLM models, typically 0-5 entries (cos/sin caches, embedding weights). */
    private int[] cachedVariableTypeIndices;

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
     * Resolve the physical home device for an external input array.
     *
     * <p>Prefer {@link NativeOps#dbDeviceId}, which queries the owning device of an
     * existing CUDA special pointer. The AllocationPoint metadata returned by
     * {@link AffinityManager#getDeviceForArray} is only a logical hint and may be
     * stale after allocator routing or replication. Using that hint first can make
     * a newly compiled plan migrate an already-resident model to another GPU.
     * On asymmetric systems that can select a device which cannot hold the plan.
     *
     * <p>For host-only arrays with no placed special pointer, dbDeviceId falls back
     * to DataBuffer metadata. If neither source yields a valid device, use
     * {@code fallbackDevice}.
     */
    private int resolveArrayDevice(INDArray arr, int numDevices, int fallbackDevice) {
        try {
            if (arr.data() != null && arr.data().opaqueBuffer() != null) {
                NativeOps nOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                int dbDevice = nOps.dbDeviceId(arr.data().opaqueBuffer());
                if (dbDevice >= 0 && dbDevice < numDevices) {
                    return dbDevice;
                }
            }
        } catch (Exception ignored) {
            // A host-only, closed, or poisoned buffer may not have a queryable
            // native allocation. Fall through to logical metadata.
        }
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
            cachedVariableTypeIndices = null;
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
            // Decrement global frozen-executor count before clearing the flag.
            // After SESSION_RESET the CUDA graphs are destroyed (releaseGpuIntermediates above)
            // so the baked TAD pointers no longer exist; it is safe to allow clearTADCache again.
            if (registeredAsFrozen) {
                GLOBAL_FROZEN_EXECUTOR_COUNT.decrementAndGet();
                registeredAsFrozen = false;
            }
            shapesFrozen = false;
            wasEverFrozen = false;
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
        // Capture wasEverFrozen BEFORE freeNativePlanHandle clears Java state.
        // This flag drives the frozen multi-plan switch gate in redispatchForCurrentShapes()
        // (line ~1645: isShapeChange && (shapesFrozen || wasEverFrozen)).
        boolean wasPreviouslyFrozen = wasEverFrozen;
        // Reset native executor state for new plan
        freeNativePlanHandle("PLAN_CHANGED");
        // Decrement global frozen-executor count if this executor was registered.
        // A plan change destroys the old CUDA graphs (freeNativePlanHandle above),
        // so the baked TAD pointers from the old plan no longer exist.
        if (registeredAsFrozen) {
            GLOBAL_FROZEN_EXECUTOR_COUNT.decrementAndGet();
            registeredAsFrozen = false;
        }
        executionCount = 0;
        nativeExecutorFailed = false;
        // FROZEN→FROZEN MULTI-PLAN SWITCH: if the outgoing plan was ever frozen, preserve
        // shapesFrozen/wasEverFrozen so that redispatchForCurrentShapes() takes the frozen
        // multi-plan switch path (gate at ~line 1645). That path calls
        // setPlanShapesFrozen(newHandle, true) on the incoming plan's C++ handle, which
        // starts it in SHAPES_FROZEN phase — skipping SLOT_BY_SLOT warmup and therefore
        // skipping platformClearCastCache(). Without this, platformClearCastCache() frees
        // the FP32 cast buffers that the outgoing frozen plan's CUDA graph baked as device
        // addresses; when the outgoing plan's composite-replay resumes, cuBLAS reads the
        // freed address and produces NaN (test39_RmsNormLinearFp16AfterPlanSwap).
        //
        // FRESH PLAN PATH: if wasPreviouslyFrozen is false (first-ever compile, no prior
        // freeze), shapesFrozen/wasEverFrozen reset to false normally, preserving the
        // standard warmup path for genuinely new plans.
        if (wasPreviouslyFrozen) {
            // Preserve frozen flags: the incoming plan will be started frozen by
            // redispatchForCurrentShapes(). Both shapesFrozen and wasEverFrozen stay true
            // so the gate at ~1645 fires on the very next redispatch call.
            shapesFrozen = true;
            wasEverFrozen = true;
            log.info("initialize: PLAN_CHANGED from frozen plan — preserving shapesFrozen=true " +
                    "so redispatch takes the frozen multi-plan switch path (preserves cast cache)");
        } else {
            // Genuinely new plan (never frozen before): reset frozen flags so the incoming
            // plan starts with a clean SLOT_BY_SLOT warmup, exactly as before.
            shapesFrozen = false;
            wasEverFrozen = false;
        }

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

            // Zero-copy outputs of VIEW chains share the CURRENT external
            // input's DataBuffer (the output IS a view over the input).
            // Force-closing it here kills the plan's installed view slots over
            // that same buffer — the native side then reads a closed parent
            // (isValid()=false -> getSlotOutput length-0 -> "null output slots
            // in REPLAYING phase", longViewChain/EMULATED_REPLAY, task #52).
            // Protected externals are never ours to close, shared or not.
            if (isProtectedExternalBuffer(buf)) {
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
                return detachIfWorkspaceBacked(placeholder, varName);
            }
        }
        if (sd == null) {
            return null;
        }
        SDVariable var = sd.getVariable(varName);
        return var != null ? detachIfWorkspaceBacked(var.getArr(), varName) : null;
    }

    /**
     * DSP plans retain external-input INDArrays across executions, but a
     * workspace-attached array's native DataBuffer holds a raw {@code Workspace*}
     * whose lifetime ends with the Java {@code MemoryWorkspace} scope — NOT with
     * the buffer object. {@code wasClosed()}/{@code isArrayLive()} do NOT catch
     * this: the buffer stays "live" while its {@code _workspace} field dangles,
     * and a later {@code allocateSpecial()} self-heal inside the native plan then
     * calls {@code Workspace::allocateBytes} on the freed workspace object
     * (MALLOC_PERTURB_-poisoned {@code this=0xaaaa...}) — mid-decode SIGSEGV at
     * DataBuffer.cu:821 (hs_err pids 983764/1001349/2010769, Jul 4 2026;
     * reproduces under CPU-load-shifted GC/workspace-cycle timing).
     *
     * Fix at the ingestion boundary: detach (copy off-workspace) any attached
     * array before the plan may retain it. Only fires for attached arrays —
     * zero cost on the normal path.
     */
    private static INDArray detachIfWorkspaceBacked(INDArray arr, String key) {
        if (arr == null || !arr.isAttached()) {
            return arr;
        }
        INDArray detached = arr.detach();
        DspDiagnostics.record(DspDiagnostics.MEMORY,
                "Java: detached workspace-backed external input '" + key
                        + "' before plan ingestion (plan outlives workspace scope)");
        return detached;
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
     * Find the index of an external input by name.
     * Returns -1 if not found.
     */
    public int findExternalInputIndex(String name) {
        if (name == null || currentPlan == null) return -1;
        String[] extKeys = currentPlan.getExternalInputKeys();
        return findExternalInputIndex(extKeys, name);
    }

    /**
     * Find the index of a requested output by name.
     * Returns -1 if not found.
     */
    public int findOutputIndex(String name) {
        if (name == null || currentPlan == null) return -1;
        List<String> outputs = new java.util.ArrayList<>(currentPlan.getRequestedOutputs());
        for (int i = 0; i < outputs.size(); i++) {
            if (outputs.get(i).equals(name)) return i;
        }
        return -1;
    }

    /**
     * Get a snapshot of the current external inputs array.
     * This is the full array of INDArrays that the native plan needs for execution.
     * Returns null if no plan is compiled.
     */
    public INDArray[] getExternalInputsSnapshot() {
        if (externalInputs == null) return null;
        return Arrays.copyOf(externalInputs, externalInputs.length);
    }

    /**
     * Return whether an external-input snapshot is safe for pointer-stable frozen-plan reuse.
     *
     * <p>A strong Java reference is not a liveness guarantee: cleanup may close the array's
     * {@link DataBuffer} while the executor still retains the {@link INDArray}. Reusing such a
     * snapshot would fail on the first in-place update and leave the poisoned frozen plan cached
     * across subsequent requests.</p>
     *
     * @param candidates external inputs captured by a frozen plan
     * @return true only when the snapshot is non-empty and every array and DataBuffer is open
     */
    public static boolean areExternalInputsReusable(INDArray[] candidates) {
        if (candidates == null || candidates.length == 0) return false;
        for (INDArray candidate : candidates) {
            try {
                if (candidate == null || candidate.wasClosed()) return false;
                DataBuffer buffer = candidate.data();
                if (buffer == null || buffer.wasClosed()) return false;
            } catch (RuntimeException e) {
                return false;
            }
        }
        return true;
    }

    /**
     * Retain the Java external-input owners for one pinned native plan handle.
     * The array container is copied because the frozen fast path reuses a mutable working array
     * while alternating between shape-keyed plans.
     */
    private void retainExternalInputsForPlan(long planHandleAddress, INDArray[] inputs) {
        if (planHandleAddress == 0 || inputs == null) return;

        INDArray[] retained = retainedExternalInputsByPlanHandle.get(planHandleAddress);
        if (retained == null || retained.length != inputs.length) {
            retainedExternalInputsByPlanHandle.put(
                    planHandleAddress, Arrays.copyOf(inputs, inputs.length));
            return;
        }

        for (int i = 0; i < inputs.length; i++) {
            retained[i] = inputs[i];
        }
    }

    /**
     * Add every live external-input buffer owned by this executor's pinned native plans to a
     * cleanup protection set.
     *
     * <p>Per-execution SameDiff cleanup is thread-local but an executor may have multiple pinned
     * shape plans. Protecting only the active call's placeholders can close inputs retained by an
     * inactive plan (for example the VLM decode attention mask while page prefill runs).</p>
     */
    public void collectRetainedExternalInputBuffers(
            IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (protectedBuffers == null) return;

        for (INDArray[] inputs : retainedExternalInputsByPlanHandle.values()) {
            collectLiveInputBuffers(inputs, protectedBuffers);
        }
        // Include the current inputs before the first native handle snapshot is published.
        collectLiveInputBuffers(externalInputs, protectedBuffers);
    }

    /** Whether the exact INDArray is retained by any currently pinned native plan. */
    public boolean isRetainedExternalInput(INDArray candidate) {
        if (candidate == null) return false;
        for (INDArray[] inputs : retainedExternalInputsByPlanHandle.values()) {
            if (inputs == null) continue;
            for (INDArray input : inputs) {
                if (input == candidate) return true;
            }
        }
        if (externalInputs != null) {
            for (INDArray input : externalInputs) {
                if (input == candidate) return true;
            }
        }
        return false;
    }

    private static void collectLiveInputBuffers(
            INDArray[] inputs, IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (inputs == null) return;
        for (INDArray input : inputs) {
            if (!isArrayLive(input)) continue;
            DataBuffer buffer = input.data();
            if (buffer != null) {
                protectedBuffers.put(buffer, Boolean.TRUE);
            }
        }
    }

    /**
     * Override a specific external input slot with a new array.
     * Used by DspHandle to inject buffers directly into the plan's ext input array.
     * Package-private — only accessible from the execution package.
     *
     * @param extIdx the external input index
     * @param arr    the array to inject (must not be null)
     */
    void overrideExternalInput(int extIdx, INDArray arr) {
        if (externalInputs == null || extIdx < 0 || extIdx >= externalInputs.length) {
            throw new IndexOutOfBoundsException("overrideExternalInput: extIdx=" + extIdx +
                    " len=" + (externalInputs == null ? 0 : externalInputs.length));
        }
        externalInputs[extIdx] = arr;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            retainExternalInputsForPlan(nativePlanHandle.address(), externalInputs);
        }
    }

    /**
     * Freeze shapes on the native plan, enabling CUDA graph capture and buffer reuse.
     * When frozen, shape inference and cache clearing are skipped between executions.
     * Use during static KV decode where all external input shapes are guaranteed constant.
     * The first execution after enabling will still do full shape inference to populate
     * the cache; subsequent executions skip shape work entirely.
     * <p>
     * Plan phases are strictly linear: SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING.
     * Calling this method with {@code frozen=false} is illegal — backward transitions are
     * architectural errors. If shapes change, call {@link #resetForNextPage()} to destroy the
     * current plan and let the cache compile a fresh entry.
     *
     * @param frozen true to freeze shapes; false throws IllegalArgumentException
     * @throws IllegalArgumentException if frozen is false
     */
    public void setShapesFrozen(boolean frozen) {
        if (!frozen) {
            throw new IllegalArgumentException(
                "LIFECYCLE VIOLATION: setShapesFrozen(false) is illegal. " +
                "Plan phases are strictly linear (SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING). " +
                "Backward transitions are banned. To handle a shape change, call " +
                "resetForNextPage() to destroy the current plan and let the cache compile a fresh one.");
        }
        boolean wasFrozen = this.shapesFrozen;
        if (wasFrozen) {
            // Idempotent: already frozen, nothing to do.
            return;
        }
        enterJavaFrozenState("explicit", -1);
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanShapesFrozen(nativePlanHandle, true);
        }
    }

    private void enterJavaFrozenState(String reason, int nativePhaseCode) {
        this.shapesFrozen = true;
        this.wasEverFrozen = true;
        // Register in the global frozen-executor counter (once per executor lifetime).
        // This prevents clearTADCache() from running while this executor holds baked
        // CUDA-graph kernel args that reference TAD device pointers.
        if (!registeredAsFrozen) {
            GLOBAL_FROZEN_EXECUTOR_COUNT.incrementAndGet();
            registeredAsFrozen = true;
        }
        log.info("FROZEN_TRANSITION: unfrozen → FROZEN (reason={}, nativePhase={}, frozenCallCount reset, plan={}, globalFrozenCount={})",
                reason, nativePhaseCode,
                nativePlanHandle != null && !nativePlanHandle.isNull() ? "native" : "java",
                GLOBAL_FROZEN_EXECUTOR_COUNT.get());
        DspDiagnostics.record(DspDiagnostics.SHAPE,
                "Java: shapes FROZEN (reason=" + reason + ", executionCount=" + executionCount + ")");
        // Clear frozen-state caches when entering frozen mode. Stale caches from a previous
        // plan/seqLen would cause shape mismatches (e.g., zeroCopyOutputCache has [1,576]
        // from seqLen=1 but new plan needs [6,576] for seqLen=6).
        closeZeroCopyOutputCache();
        cachedInputOpaques = null;
        cachedInputArrays = null;
        // Do NOT null contextInputRefs here. Unlike the index caches above, this is the
        // strong-reference array that keeps the C++ OpaqueNDArray wrappers alive while the
        // cachedOpContext still holds raw NDArray* pointers into them. setShapesFrozen() runs
        // BEFORE the context is handed to autoregressive_decode, so clearing the refs here lets
        // the DeallocatorService free those C++ objects → dangling context pointers (stale
        // specialBuffer). It is only safe to replace contextInputRefs atomically in executeNative().
        inputIsPlaceholder = null;
        placeholderIndices = null;
        cachedVariableTypeIndices = null;
        frozenControlInputIndices = null;
        frozenDerivedExternalInputIndices = null;
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        cachedExecStream = null;
        execStreamCached = false;
        frozenExtBufferSnapshot = null;
        frozenExtShapeSnapshot = null;
    }

    public boolean isShapesFrozen() {
        return shapesFrozen;
    }

    /**
     * Get the current plan-level phase from the native C++ plan.
     * Returns the phase that represents the overall plan lifecycle:
     * SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING.
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
     * Reset executor state for next-page reuse.
     * Releases GPU intermediates (CUDA graphs, replay workspaces) and destroys the current
     * native plan handle so the next execution starts with a fresh plan from the cache.
     * Plan phases are strictly linear (SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING) and cannot
     * go backwards — unfreezing is illegal. Destroying the handle and letting the cache create
     * a fresh entry is the only correct reset path.
     */
    public void resetForNextPage() {
        resetForNextPage(true);
    }

    /**
     * Reset executor state for next-page reuse, optionally deferring cleanup of the shared
     * thread-local {@link ArrayCacheMemoryMgr} state.
     *
     * <p>The array cache is shared by every SameDiff executor on the current thread. A component
     * in a multi-model pipeline must therefore pass {@code false}, reset all disposable executors,
     * and let the pipeline owner perform one cache sweep with every surviving model buffer
     * protected. Sweeping here with only this executor's protections can close another executor's
     * retained external inputs.</p>
     *
     * @param clearSharedArrayCache whether this reset owns the complete thread-local cache boundary
     */
    public void resetForNextPage(boolean clearSharedArrayCache) {
        log.info("DSP resetForNextPage: releasing GPU intermediates and destroying native plan handle");
        cachedInputArrays = null;
        cachedInputOpaques = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        placeholderIndices = null;
        cachedVariableTypeIndices = null;
        frozenControlInputIndices = null;
        frozenDerivedExternalInputIndices = null;

        // Release C++ intermediate GPU memory (CUDA graphs, replay workspaces, cuBLAS workspace,
        // non-weight output slot NDArrays). This also calls closeSlotArrayCache() internally.
        // releaseGpuIntermediates does a teardown-only shapesFrozen_=false reset in C++ so the
        // plan goes cold before the handle is unpinned back to the cache.
        releaseGpuIntermediates();

        closeZeroCopyOutputCache();

        // Free the native plan handle — unpin it back to the cache so it is eligible for
        // eviction. The next call to executeDynamicShapePlanBased() will compile a fresh plan
        // for whatever shape the next page uses. Plan phases are linear and immutable, so we
        // cannot reuse a frozen plan for a different shape.
        freeNativePlanHandle("PAGE_RESET");
        // The destroyed plan no longer owns external inputs. Drop these references before a
        // pipeline-level cache sweep so buffers from the reset executor are reclaimable, while
        // preserved executors can still publish their live inputs as protected.
        externalInputs = null;
        // Decrement global frozen-executor count before resetting Java frozen state.
        // releaseGpuIntermediates() above destroyed the CUDA graphs; the baked TAD pointers
        // from the old plan no longer exist. It is safe to allow clearTADCache again.
        if (registeredAsFrozen) {
            GLOBAL_FROZEN_EXECUTOR_COUNT.decrementAndGet();
            registeredAsFrozen = false;
        }
        // Reset Java-side tracking that freeNativePlanHandle clears (shapesFrozen is on the
        // Java side only; the C++ plan is gone).
        shapesFrozen = false;
        wasEverFrozen = false;
        nativeExecutorFailed = false;
        executionCount = 0;
        lastDispatchedShapeHash = 0;

        if (clearSharedArrayCache) {
            // Drain ArrayCacheMemoryMgr state only when this reset owns the entire thread-local
            // cache boundary. Multi-model pipelines defer this sweep until every disposable
            // executor is reset, then protect the external inputs of executors that survive.
            // IMPORTANT: Pass model constant/variable buffers as protected so that force-closing
            // "constant-poisoned" intermediates doesn't destroy buffers the native DSP plan still
            // references. Without this, 60+ model constants get closed → stale buffer scan fails
            // → CUDA graph replay can't proceed → 13x slowdown.
            IdentityHashMap<DataBuffer, Boolean> protectedModelBuffers = collectProtectedModelBuffers();
            ArrayCacheMemoryMgr.closeDeferredBuffers(protectedModelBuffers);
            ArrayCacheMemoryMgr.clearCacheState();
            ArrayCacheMemoryMgr.setEnableCache(false);
        }
        if (currentPlan != null) {
            currentPlan.clearAllShapeCaches();
        }
        frozenOutputsInitialized = false;
        frozenCallCount = 0;
        // Do NOT reset nativeExecutionDevice here. The next page should use the same GPU device
        // for consistency (CUDA contexts, memory topology, etc. are device-specific).

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
     * Clear Java-side output caches and trim the CUDA memory pool, WITHOUT destroying
     * the native plan handle or CUDA graphs.
     *
     * <p>Use this between pages for models whose plans are preserved across pages
     * (vision encoder, embedTokens). The slot array cache and zero-copy output cache
     * hold references to intermediate GPU buffers from the last execution. Clearing
     * them allows the JavaCPP deallocator to free those native buffers, preventing
     * GPU memory from accumulating across pages.</p>
     *
     * <p>Unlike {@link #releaseGpuIntermediates()}, this does NOT call the C++
     * {@code releaseGpuIntermediates()} which would destroy CUDA graphs and replay
     * state. The frozen plan handle stays intact for shape-keyed switching.</p>
     */
    public void clearOutputCaches() {
        log.info("DSP clearOutputCaches: clearing Java-side caches (plan preserved)");
        closeSlotArrayCache();
        closeZeroCopyOutputCache();
        frozenOutputsInitialized = false;

        // Trim CUDA memory pool so freed buffers are returned to the driver
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            try {
                Pointer stream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                if (stream != null) {
                    NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                    nativeOps.trimMemoryPoolOnStream(currentDevice, stream);
                }
            } catch (Exception e) {
                log.debug("clearOutputCaches: trim failed (non-fatal): {}", e.getMessage());
            }
        }
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
        executionTimingOverride = enabled;
        cachedExecTiming = enabled;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanExecutionTimingEnabled(nativePlanHandle, enabled);
        }
    }

    /**
     * Enable/disable trace logging for DSP execution decisions.
     */
    public void setTraceEnabled(boolean enabled) {
        traceEnabledOverride = enabled;
        cachedTraceEnabled = enabled;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanTraceEnabled(nativePlanHandle, enabled);
        }
    }

    /**
     * Enable/disable shape-only dry-run mode on the native plan.
     *
     * When enabled, executeSlot() runs the full DSP dispatch machinery — slot
     * iteration, shape caching, frozen-constant checks, identity/fusion detection,
     * segment dispatch, output allocation — but SKIPS the actual op kernel execution.
     * Outputs retain their values from the previous real execution (uninitialized
     * on the very first pass).
     *
     * Purpose: measure pure dispatch/infrastructure overhead independently from
     * compute.  With 1683 ops per CPU decode step and 185 ms of kernel time versus
     * 367 ms of dispatch overhead, this mode lets dispatch optimizations be
     * profiled and iterated ~100x faster.
     *
     * Can also be activated via system property {@code nd4j.dsp.shape.only=true}.
     *
     * @param enabled true to enable shape-only mode, false to disable
     */
    public void setShapeOnlyMode(boolean enabled) {
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.setPlanShapeOnlyMode(nativePlanHandle, enabled);
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
     * Configure max-allocation for KV cache output slots from an already-run
     * decode step. This is used by native decode loops that warm up once, then
     * replay the same native plan directly.
     */
    public boolean configureMaxAllocationForKvCache(Map<String, INDArray> firstStepResults) {
        return configureMaxAllocationForKvCache(firstStepResults, (Collection<String>) null);
    }

    /**
     * Configure max-allocation for explicitly known KV cache output names.
     */
    public boolean configureMaxAllocationForKvCache(Map<String, INDArray> firstStepResults,
                                                    Collection<String> kvOutputNames) {
        if (currentPlan == null) return false;
        Set<String> explicitKvOutputs = kvOutputNames == null
                ? null : new LinkedHashSet<>(kvOutputNames);
        boolean configured = configureMaxAllocationForKvCache(firstStepResults, currentPlan, explicitKvOutputs);
        if (configured) {
            maxAllocationConfigured = true;
        }
        return configured;
    }

    /**
     * Whether a native plan has been compiled for the given plan. Compilation caches the
     * serialized bytes and per-handle settings; the actual native handle is obtained per
     * execute via the C++ NativePlanCache (shape-keyed), so we check the cached artifacts
     * rather than {@code nativePlanHandle}, which is swapped by redispatchForCurrentShapes.
     */
    public boolean isNativePlanCompiled(DynamicShapePlan plan) {
        if (zeroSlotPassthrough && nativePlanSource == plan) return true;
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

        // Zero-slot plans (all outputs are direct placeholders/constants) need no native
        // compilation. This happens when the GraphOptimizer eliminates all ops (e.g.
        // pow(x,1) → x, div(x,1) → x). execute() handles the passthrough.
        DynamicShapeSlot[] planSlots = plan.getSlots();
        if (planSlots == null || planSlots.length == 0) {
            zeroSlotPassthrough = true;
            nativePlanSource = plan;
            log.debug("Native executor: zero-slot plan detected, using passthrough mode");
            return requestedMode != null ? requestedMode : GraphExecutionMode.SLOT_BY_SLOT;
        }
        zeroSlotPassthrough = false;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        boolean planChanged = nativePlanSource != null && nativePlanSource != plan;

        if (cachedSerializedPlan == null || nativePlanSource != plan) {
            if (planChanged && cudaGraphsFailed) {
                log.info("Native executor: resetting cudaGraphsFailed on plan recompilation");
                cudaGraphsFailed = false;
            }

            freeNativePlanHandle("PLAN_RECOMPILATION");
            configuredHandleAddresses.clear();
            mutableExternalInputsConfiguredHandleAddresses.clear();

            // --- Disk cache: try loading serialized plan bytes from disk ---
            byte[] serialized = null;
            long structureHash = 0;
            boolean loadedFromDiskCache = false;

            if (DspPlanDiskCache.isEnabled() && !DspPlanDiskCache.isForceRecompile()) {
                // Try model-identity lookup first (works without serializing the plan)
                byte[] diskBytes = DspPlanDiskCache.tryLoadByModelIdentity(
                        plan.getRequestedOutputs(), plan.getExternalInputKeys(),
                        plan.getSlots().length);
                if (diskBytes != null && DynamicShapePlan.isValidSerializedPlan(diskBytes)) {
                    // Validate the cached plan matches the current graph structure.
                    // The model identity hash only covers variable names and slot count,
                    // so graphs with different iArgs (e.g. different reshape dimensions)
                    // can collide. Serialize the current plan and compare structure hashes
                    // to catch stale cache entries.
                    byte[] freshSerialized = plan.serialize();
                    long freshHash = DynamicShapePlan.computeStructureHash(freshSerialized);
                    long diskHash = DynamicShapePlan.computeStructureHash(diskBytes);
                    if (freshHash == diskHash) {
                        serialized = diskBytes;
                        structureHash = diskHash;
                        loadedFromDiskCache = true;
                        log.info("Native executor: loaded plan from disk cache (model identity hit, validated, {} bytes)", diskBytes.length);
                    } else {
                        log.info("Native executor: disk cache plan STALE (hash mismatch: disk=0x{} fresh=0x{}), recompiling",
                                Long.toHexString(diskHash), Long.toHexString(freshHash));
                        serialized = freshSerialized;
                        structureHash = freshHash;
                    }
                }
            }

            if (serialized == null) {
                serialized = plan.serialize();
                if (serialized == null || serialized.length == 0) {
                    nativeExecutorFailed = true;
                    throw new RuntimeException("Native executor: plan serialization returned empty. " +
                            "Cannot compile native plan. No fallback permitted.");
                }
                structureHash = DynamicShapePlan.computeStructureHash(serialized);
            }

            // Cache inputs for shape-keyed dispatch. The actual native plan handle is
            // obtained per-execute through the C++ NativePlanCache; placeholder arrays
            // aren't required to be bound at compile time (they arrive via sd.output(Map)).
            List<String> sortedOutputs = new java.util.ArrayList<>(plan.getRequestedOutputs());
            java.util.Collections.sort(sortedOutputs);

            String[] extKeys = plan.getExternalInputKeys();
            // Hash ALL external input shapes into the plan cache key, not just
            // placeholders. Any external shape change (KV cache growth, attention
            // mask resize, etc.) must dispatch to a different plan instance.
            // Intermediates are deterministic from externals, so only externals matter.
            List<String> phKeys = new java.util.ArrayList<>();
            for (int pi = 0; pi < extKeys.length; pi++) {
                phKeys.add(extKeys[pi]);
            }

            cachedSerializedPlan = serialized;
            cachedSortedOutputs = sortedOutputs.toArray(new String[0]);
            cachedPhKeys = phKeys.toArray(new String[0]);

            // --- Disk cache: store serialized plan bytes to disk ---
            if (DspPlanDiskCache.isEnabled() && !loadedFromDiskCache && !DspPlanDiskCache.exists(structureHash)) {
                String outputSetStr = String.join(",", cachedSortedOutputs);
                DspPlanDiskCache.store(structureHash, serialized,
                        plan.getSlots().length, extKeys.length,
                        plan.getRequestedOutputs().size(), outputSetStr);
                // Also store the model identity → structure hash mapping for cross-JVM lookup
                DspPlanDiskCache.storeModelIdentityIndex(
                        plan.getRequestedOutputs(), extKeys, plan.getSlots().length, structureHash);
            }

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

            cachedExecTiming = executionTimingOverride != null
                    ? executionTimingOverride
                    : "true".equalsIgnoreCase(
                            System.getProperty(ND4JSystemProperties.DSP_EXECUTION_TIMING, "false"));
            cachedTraceEnabled = traceEnabledOverride != null
                    ? traceEnabledOverride
                    : System.getProperty(ND4JSystemProperties.DSP_TRACE) != null;

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
                if (var != null && (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE)) {
                    INDArray arr = var.getArr();
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        protectedConstantBuffers.put(arr.data(), Boolean.TRUE);
                        protectedCount++;
                    }
                }
            }
            if (protectedCount > 0) {
                log.info("Native executor: protecting {} constant/variable DataBuffers for plan lifetime", protectedCount);
            }
        }

        GraphExecutionMode resolvedMode = resolveRequestedGraphExecutionMode(requestedMode);
        boolean tritonAvailable = isTritonAvailable(nativeOps);
        GraphExecutionMode effectiveMode = resolveEffectiveGraphExecutionMode(
                resolvedMode, tritonAvailable, fallbackToAutoIfTritonUnavailable);
        if (resolvedMode == GraphExecutionMode.TRITON && effectiveMode == GraphExecutionMode.AUTO) {
            log.warn("Native executor: TRITON mode requested but Triton is unavailable; falling back to AUTO");
        }

        cachedEffectiveGraphModeCode = effectiveMode.getNativeCode();
        configuredGraphExecutionMode = effectiveMode;
        // Invalidate any previously-configured handles so the new mode is re-applied
        // to every handle the NativePlanCache returns on the next redispatch.
        configuredHandleAddresses.clear();
        mutableExternalInputsConfiguredHandleAddresses.clear();
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

    /**
     * Resolve only policy owned by the Java layer. NativeDynamicShapePlan is the
     * authority for backend capability selection because it can discover both GPU
     * and CPU graph backends (oneDNN, OpenVINO, Arm Compute, MLIR, NNAPI, MLX).
     */
    static GraphExecutionMode resolveEffectiveGraphExecutionMode(
            GraphExecutionMode resolvedMode,
            boolean tritonAvailable,
            boolean fallbackToAutoIfTritonUnavailable) {
        if (resolvedMode == GraphExecutionMode.TRITON &&
                fallbackToAutoIfTritonUnavailable &&
                !tritonAvailable) {
            return GraphExecutionMode.AUTO;
        }
        return resolvedMode;
    }

    private boolean isTritonAvailable(NativeOps nativeOps) {
        try {
            return nativeOps.isTritonAvailable();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Compute a hash of all placeholder shapes from the current execution's inputs.
     * Used to detect when shapes change between frozen executions so that the executor
     * can switch between shape-keyed plans in the C++ cache.
     *
     * @param placeholderArrays the placeholder name → array map from sd.output()
     * @return a hash combining all placeholder shape info, or 0 if no placeholders
     */
    private long computePlaceholderShapeHash(Map<String, INDArray> placeholderArrays) {
        if (cachedPhKeys == null || cachedPhKeys.length == 0) return 0;
        long hash = 17;
        // D1a per-token trim: once shapes are frozen, only DYNAMIC-shape external inputs
        // (placeholders + derived KV/mask + integral control) can change shape between
        // tokens. Constant weights have fixed shapes, so hashing all ~N ext inputs (incl.
        // every weight) each token is wasted work (thousands of map lookups/token on VLM).
        // When the frozen classifier is built, hash only that dynamic subset (indices into
        // cachedPhKeys, which mirrors getExternalInputKeys()); otherwise fall back to all
        // keys (pre-freeze / classifier absent). Excluding fixed-shape weights cannot change
        // the hash's ability to distinguish dynamic-shape configs; the bounds guard + fallback
        // keep it safe if indices ever misalign.
        final int[][] dynIdxGroups = (placeholderIndices != null)
                ? new int[][]{ placeholderIndices, frozenDerivedExternalInputIndices, frozenControlInputIndices }
                : null;
        if (dynIdxGroups != null) {
            for (int[] group : dynIdxGroups) {
                if (group == null) continue;
                for (int gi : group) {
                    if (gi < 0 || gi >= cachedPhKeys.length) continue;
                    hash = hashPlaceholderKeyShape(placeholderArrays, hash, cachedPhKeys[gi]);
                }
            }
            return hash;
        }
        for (String phKey : cachedPhKeys) {
            hash = hashPlaceholderKeyShape(placeholderArrays, hash, phKey);
        }
        return hash;
    }

    /** Fold one external-input's shape (dims + rank delimiter) into the running plan-cache hash. */
    private long hashPlaceholderKeyShape(Map<String, INDArray> placeholderArrays, long hash, String phKey) {
        INDArray arr = placeholderArrays != null ? placeholderArrays.get(phKey) : null;
        if (arr == null) {
            SDVariable v = sd.getVariable(phKey);
            arr = v != null ? v.getArr() : null;
        }
        if (arr != null) {
            long[] shape = arr.shape();
            for (long dim : shape) {
                hash = hash * 31 + dim;
            }
            // Include rank as a delimiter to distinguish e.g. [2,3] from [6]
            hash = hash * 31 + shape.length;
        }
        return hash;
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
     *
     * @param isShapeChangeExpected true when caller detected that the placeholder shape hash
     *        changed since the last dispatch. When true, the C++ cache may return a different
     *        plan handle for this executor's frozen multi-plan switch. This remains the same
     *        borrower: the executor pins every acquired handle until reset, and the native
     *        staging path refreshes runtime inputs without discarding captured plan state.
     */
    private void redispatchForCurrentShapes(Map<String, INDArray> placeholderArrays,
                                            boolean isShapeChangeExpected) {
        if (cachedSerializedPlan == null) {
            throw new IllegalStateException(
                "redispatchForCurrentShapes: plan not compiled yet — " +
                "compileNativePlan() must run before executeNative().");
        }
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        requireCurrentDispatchAbi(nativeOps);
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
            // NOTE: closed arrays are safe to shape-key here — shape-info buffers are
            // ConstantShapeHelper-owned and outlive the array's DataBuffer, and the key
            // must stay stable across a borrower's close/reopen of same-shaped inputs
            // (rerouting closed arrays to the any-shape sentinel changed plan cache
            // keys and regressed lifecycle/staleness gates).
            if (arr == null || arr.shapeInfoDataBuffer() == null) {
                // Empty constants (e.g., scalar placeholders compiled as EMPTY_CONSTANT)
                // may have no backing array at execute time. Use a Pointer(0) as the
                // shape sentinel — the C++ cache key treats null shape pointers as "any shape"
                // which is correct for empty/scalar constants whose shape never changes.
                phPtrs.add(new Pointer());
                continue;
            }
            phPtrs.add(arr.shapeInfoDataBuffer().addressPointer());
        }
        PointerPointer phPtrsPacked = phPtrs.isEmpty()
                ? new PointerPointer(0)
                : new PointerPointer(phPtrs.toArray(new Pointer[0]));

        try {
            // Pass mode as part of cache key — each mode gets its own plan (one flow).
            int modeForDispatch = cachedEffectiveGraphModeCode >= 0
                    ? cachedEffectiveGraphModeCode : 0;
            // newBorrower is an executor-lifecycle boundary, not a plan-shape boundary.
            // Mark only the first dispatch from this Java executor. A cache hit can then
            // contain external-fed views minted by a previous executor and native code must
            // validate them. Prefill/decode shape switches after that are same-borrower
            // re-dispatches: both handles stay pinned by this executor and invalidating either
            // plan would discard its captures, force a full warmup allocation on every A/B/A
            // switch, and eventually exhaust CUDA memory.
            //
            // This intentionally matches the native binding contract. Runtime input contents
            // and same-shape replacement arrays are refreshed through the external-input
            // staging path; they do not constitute a borrower change.
            int newBorrower = (nativePlanHandle == null || nativePlanHandle.isNull()) ? 1 : 0;
            Pointer newHandle = nativeOps.dispatchNativePlan(
                    cache,
                    planBytes, cachedSerializedPlan.length,
                    outputNamesPtr, cachedSortedOutputs.length,
                    phPtrsPacked, phPtrs.size(),
                    modeForDispatch, newBorrower);
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
            // Track every cache pin acquired by dispatchNativePlan. Frozen multi-plan
            // switches intentionally retain both handles until the executor is reset.
            pinnedPlanHandles.put(newHandle.address(), newHandle);
            // Swap handle if the cache returned a different plan for current shapes.
            // The C++ cache owns plan lifetimes, so we don't free the old one.
            boolean swapped = nativePlanHandle == null
                    || nativePlanHandle.isNull()
                    || newHandle.address() != nativePlanHandle.address();
            if (swapped) {
                if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
                    // Check if this swap is due to a legitimate shape change (multi-plan switching)
                    // or a same-shape cache key instability bug.
                    long currentShapeHash = computePlaceholderShapeHash(placeholderArrays);
                    boolean isShapeChange = (lastDispatchedShapeHash != 0 && currentShapeHash != lastDispatchedShapeHash);

                    if ((shapesFrozen || frozenCallCount > 2) && !isShapeChange) {
                        // HARD ERROR: plan swapped for the SAME shapes after frozen state —
                        // the cache is returning different plans for identical shapes. This means
                        // every decode step creates a new plan, destroying all replay/capture
                        // progress and annihilating throughput. This is a catastrophic bug.
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

                    if (isShapeChange && (shapesFrozen || wasEverFrozen)) {
                        // FROZEN MULTI-PLAN SWITCH: shapes changed while frozen. This is the
                        // VLM multi-page pattern (prefill seq=N ↔ decode seq=1). The C++ cache
                        // returned a different plan for the new shape — accept it. Both handles
                        // stay pinned in the C++ cache so we can switch between them freely;
                        // pinnedPlanHandles records both so reset/close can release them.
                        log.info("redispatchForCurrentShapes: frozen multi-plan switch from {} to {} " +
                                "(shapeHash {} → {}, shapesFrozen={}, executionCount={})",
                                Long.toHexString(nativePlanHandle.address()),
                                Long.toHexString(newHandle.address()),
                                lastDispatchedShapeHash, currentShapeHash,
                                shapesFrozen, executionCount);
                        nativePlanHandle = newHandle;
                        // Clear per-shape caches — the new plan has different input mappings.
                        // Keep the wrappers currently installed in cachedOpContext strongly reachable
                        // until executeNative() has populated every replacement and atomically swaps
                        // contextInputRefs. Clearing them here creates a raw-pointer UAF window during
                        // prefill/decode plan switches.
                        cachedInputArrays = null;
                        cachedInputOpaques = null;
                        inputIsPlaceholder = null;
                        frozenExtInputsWorkingCopy = null;
                        frozenExtBufferSnapshot = null;
                        frozenExtShapeSnapshot = null;
                        frozenOutputsInitialized = false;
                        closeZeroCopyOutputCache();
                        // Reset frozen-call counter and variable-type index cache for the new plan.
                        // After a frozen multi-plan switch, the new plan's C++ executeCount_ is 0
                        // (reset by phaseFreeze). Java's frozenCallCount must also be 0 so the
                        // first execution of the new plan takes the frozenCallCount==1 snapshot
                        // path (line 3418) and the C++ broadPrepare=true gate (executeCount_<=1)
                        // ensures ALL external inputs — including CONSTANT/SOURCE_CONSTANT weight
                        // buffers — receive prepareSpecialUse() → host→device sync before the
                        // first CUDA graph capture on the new plan.
                        //
                        // Without this reset: frozenCallCount is e.g. 250 from the old plan →
                        // the snapshot is never taken for the new plan → no lifecycle validation.
                        // More critically, cachedVariableTypeIndices (built for old plan's input
                        // layout) may map wrong indices for the new plan, causing stale VARIABLE
                        // data to be skipped on the frozen fast path.
                        //
                        // Risk: none. The new plan starts fresh (executeCount_=0) so resetting
                        // frozenCallCount to 0 correctly reflects that. The snapshot will be
                        // captured on the very next execute() call (frozenCallCount 0→1).
                        frozenCallCount = 0;
                        cachedVariableTypeIndices = null;
                        // Freeze the new plan handle too — it needs frozen C++ state for replay
                        NativeOps nOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                        nOps.setPlanShapesFrozen(nativePlanHandle, true);
                    } else if (wasEverFrozen && !isShapeChange) {
                        // PLAN SWAP SUPPRESSION: same shapes but different handle after freeze.
                        // This is a cache key instability bug. Suppress the swap to prevent
                        // cascading performance loss.
                        nativeOps.unpinNativePlan(cache, newHandle);
                        pinnedPlanHandles.remove(newHandle.address());
                        configuredHandleAddresses.remove(newHandle.address());
                        mutableExternalInputsConfiguredHandleAddresses.remove(newHandle.address());
                        log.warn("redispatchForCurrentShapes: SUPPRESSED plan swap from {} to {} " +
                                "(wasEverFrozen=true, shapesFrozen={}, executionCount={}). " +
                                "Keeping existing plan to preserve graph replay state.",
                                nativePlanHandle.address(), newHandle.address(),
                                shapesFrozen, executionCount);
                        if (!shapesFrozen) {
                            shapesFrozen = true;
                        }
                    } else {
                        // Unpin the old plan so it becomes eligible for LRU eviction.
                        // This MUST happen before the new plan is pinned (which
                        // getOrInsert already did) to avoid dangling pointers — the
                        // old plan's GPU resources are freed on eviction.
                        nativeOps.unpinNativePlan(cache, nativePlanHandle);
                        pinnedPlanHandles.remove(nativePlanHandle.address());
                        configuredHandleAddresses.remove(nativePlanHandle.address());
                        mutableExternalInputsConfiguredHandleAddresses.remove(nativePlanHandle.address());
                        log.info("redispatchForCurrentShapes: plan swapped from {} to {} — resetting frozen state",
                                nativePlanHandle.address(), newHandle.address());
                        frozenOutputsInitialized = false;
                        frozenCallCount = 0;
                        closeZeroCopyOutputCache();
                        // Clear cached input arrays: the new plan may have different
                        // external input mappings or slot assignments. cachedOpContext survives the
                        // plan swap, so retain its current wrappers until the next full population
                        // atomically replaces contextInputRefs.
                        cachedInputArrays = null;
                        cachedInputOpaques = null;
                        inputIsPlaceholder = null;
                        nativePlanHandle = newHandle;
                    }
                } else {
                    nativePlanHandle = newHandle;
                }
            }
            // Apply per-handle settings the first time we see each cached handle.
            applySettingsIfNewHandle(nativeOps, newHandle);
        } finally {
            planBytes.close();
        }
    }

    private static void requireCurrentDispatchAbi(NativeOps nativeOps) {
        try {
            java.lang.reflect.Method dispatch = nativeOps.getClass().getMethod(
                    "dispatchNativePlan",
                    Pointer.class, Pointer.class, long.class,
                    Pointer.class, long.class, Pointer.class, long.class,
                    int.class, int.class);
            if (dispatch.getDeclaringClass() == NativeOps.class) {
                Package bindingPackage = nativeOps.getClass().getPackage();
                String implementationVersion = bindingPackage == null
                        ? null : bindingPackage.getImplementationVersion();
                throw new IllegalStateException(
                        "Mixed or stale ND4J native artifacts: active binding "
                                + nativeOps.getClass().getName()
                                + " does not override the current 9-argument dispatchNativePlan ABI"
                                + (implementationVersion == null ? "" : " (version " + implementationVersion + ")")
                                + ". Rebuild nd4j-api, the native preset, backend, and platform classifier "
                                + "from the same commit.");
            }
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException(
                    "ND4J native binding does not expose the current dispatchNativePlan ABI: "
                            + nativeOps.getClass().getName(), e);
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
        if (!configuredHandleAddresses.add(addr)) {
            applyMutableExternalInputsIfNeeded(nativeOps, handle);
            return;
        }

        // Only enable CUDA graph capture if the configured mode supports it.
        // SLOT_BY_SLOT and EMULATED_REPLAY must never capture/replay CUDA graphs.
        boolean modeSupportsGraphCapture = configuredGraphExecutionMode.requiresGraphBackend();
        if (cachedCudaGraphsEnabled && modeSupportsGraphCapture) {
            try {
                nativeOps.setPlanCudaGraphsEnabled(handle, true);
                DspDiagnostics.record(DspDiagnostics.COMPILE,
                        "Java: CUDA graphs ENABLED on native plan (addr=" + Long.toHexString(addr) +
                        " mode=" + configuredGraphExecutionMode + ")");
            } catch (UnsupportedOperationException e) {
                DspDiagnostics.record(DspDiagnostics.COMPILE,
                        "Java: CUDA graphs not supported by backend (CPU?)");
            }
        } else {
            DspDiagnostics.record(DspDiagnostics.COMPILE,
                    "Java: CUDA graphs DISABLED (cudaGraphsFailed=" + cudaGraphsFailed +
                    " modeSupportsGraphCapture=" + modeSupportsGraphCapture +
                    " mode=" + configuredGraphExecutionMode + ")");
        }

        if (cachedJitModeInt >= 0) {
            try {
                nativeOps.setPlanJitMode(handle, cachedJitModeInt);
                log.info("Native executor: JIT mode set to {}", cachedJitModeInt);
            } catch (UnsupportedOperationException e) {
                // Backend doesn't support JIT
            }
        }

        try {
            nativeOps.setPlanExecutionTimingEnabled(handle, cachedExecTiming);
        } catch (UnsupportedOperationException e) {
            // Backend doesn't support timing
        }

        try {
            nativeOps.setPlanTraceEnabled(handle, cachedTraceEnabled);
        } catch (UnsupportedOperationException e) {
            // Backend doesn't support trace
        }

        // NOTE: graphExecutionMode is part of the cache key — each mode gets its
        // own plan at creation time (one flow, no reclassification). No need to call
        // setPlanGraphExecutionMode here.

        // NOTE: Do NOT propagate shapesFrozen here. The C++ plan manages its own frozen
        // state transition independently based on execution count and shape stability.
        // Calling setPlanShapesFrozen prematurely (before any execution) causes phaseWarmup
        // to be dispatched at executeCount_=0, which corrupts slot arrays / shape caches
        // for subsequent phaseSlotBySlot steps, producing wrong output tokens.
        // The Java shapesFrozen flag is used only to gate zeroCopyOutputCache and
        // directOutputMode fast paths on the Java side.
        applyMutableExternalInputsIfNeeded(nativeOps, handle);
    }

    /**
     * Monotonically add external inputs that native replay must treat as mutable.
     * Native plans can safely add variable externals, but cannot unmark an input
     * after freeze/capture analysis has seen it.
     */
    public void addMutableExternalInputs(Collection<String> names) {
        if (names == null || names.isEmpty()) {
            return;
        }

        boolean changed = false;
        for (String name : names) {
            if (name != null && mutableExternalInputNames.add(name)) {
                changed = true;
            }
        }

        if (changed) {
            mutableExternalInputsConfiguredHandleAddresses.clear();
            if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
                applyMutableExternalInputsIfNeeded(
                        NativeOpsHolder.getInstance().getDeviceNativeOps(), nativePlanHandle);
            }
        }
    }

    private void applyMutableExternalInputsIfNeeded(NativeOps nativeOps, Pointer handle) {
        if (handle == null || handle.isNull() || currentPlan == null || mutableExternalInputNames.isEmpty()) {
            return;
        }

        long addr = handle.address();
        if (!mutableExternalInputsConfiguredHandleAddresses.add(addr)) {
            return;
        }

        String[] extKeys = currentPlan.getExternalInputKeys();
        int marked = 0;
        for (int i = 0; i < extKeys.length; i++) {
            if (mutableExternalInputNames.contains(extKeys[i])) {
                nativeOps.markPlanExternalInputVariable(handle, i);
                marked++;
            }
        }

        if (marked > 0) {
            DspDiagnostics.record(DspDiagnostics.EXECUTE,
                    "Java: marked " + marked + " mutable VARIABLE external inputs on native plan addr="
                            + Long.toHexString(addr));
        }
    }

    /**
     * Configure max-allocation for KV cache output slots.
     * Called after the first execution step when actual output shapes are known.
     * Finds present_key/present_value outputs and configures C++ to pre-allocate
     * them at maximum sequence length so buffer addresses stay stable for CUDA graphs.
     */
    private boolean configureMaxAllocationForKvCache(Map<String, INDArray> firstStepResults, DynamicShapePlan plan) {
        return configureMaxAllocationForKvCache(firstStepResults, plan, null);
    }

    private boolean configureMaxAllocationForKvCache(Map<String, INDArray> firstStepResults, DynamicShapePlan plan,
                                                     Set<String> explicitKvOutputNames) {
        if (nativePlanHandle == null || nativePlanHandle.isNull() || maxKvCacheLength <= 0) return false;
        if (firstStepResults == null || firstStepResults.isEmpty()) return false;

        Map<String, Integer> outputNameToSlot = plan.getOutputNameToSlotIndex();

        List<Integer> kvSlotIndices = new ArrayList<>();
        List<Long> kvMaxSizes = new ArrayList<>();

        // Get shapes from actual output arrays returned by the first execution step.
        // Match logic mirrors ModelIOConfig.findKVCacheOutputNames: present+key or present+value.
        for (Map.Entry<String, INDArray> entry : firstStepResults.entrySet()) {
            String outputName = entry.getKey();
            boolean isExplicitKv = explicitKvOutputNames != null && explicitKvOutputNames.contains(outputName);
            boolean isKvKey   = outputName.contains("present") && outputName.contains("key");
            boolean isKvValue = outputName.contains("present") && outputName.contains("value");
            if (isExplicitKv || isKvKey || isKvValue) {
                Integer slotIdx = outputNameToSlot.get(outputName);
                if (slotIdx == null || slotIdx < 0) {
                    int resolvedSlot = findOutputSlotIndex(plan, outputName);
                    slotIdx = resolvedSlot >= 0 ? resolvedSlot : null;
                }
                if (slotIdx != null && slotIdx >= 0) {
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.rank() == 4) {
                        // Shape is [batch, numHeads, seqLen, headDim]
                        long batchSize = arr.size(0);
                        long numHeads  = arr.size(1);
                        long headDim   = arr.size(3);
                        long configuredMaxSize = batchSize * numHeads * maxKvCacheLength * headDim;
                        long currentSize = arr.length();
                        long maxSize = Math.max(configuredMaxSize, currentSize);

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
            return true;
        }

        String message = "KV cache max-allocation requested (maxSeqLen=" + maxKvCacheLength
                + ") but no KV output slots were matched. outputs=" + firstStepResults.keySet()
                + " explicitKvOutputs=" + explicitKvOutputNames;
        if (explicitKvOutputNames != null) {
            log.warn(message);
        } else {
            log.debug(message);
        }
        return false;
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
                    Arrays.toString(staticBuf.shape()));
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

        // Identity passthrough: when the optimizer eliminates all ops (e.g. pow(x,1) → x),
        // the plan has zero slots and all outputs map directly to external inputs (placeholders
        // or constants). Return them directly without native plan compilation.
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) {
            Map<String, INDArray> result = new LinkedHashMap<>();
            Set<String> requested = plan.getRequestedOutputs();
            String[] extKeys = plan.getExternalInputKeys();
            byte[] extTypes = plan.getExternalInputSourceTypes();
            Set<String> extPlaceholders = new HashSet<>();
            for (int i = 0; i < extKeys.length; i++) {
                if (extTypes[i] == DynamicShapeSlot.SOURCE_PLACEHOLDER) {
                    extPlaceholders.add(extKeys[i]);
                }
            }
            for (String output : requested) {
                if (placeholderArrays != null && placeholderArrays.containsKey(output)) {
                    result.put(output, placeholderArrays.get(output));
                } else if (extPlaceholders.contains(output)) {
                    result.put(output, placeholderArrays.get(output));
                } else {
                    // Constant or variable — resolve from SameDiff
                    INDArray arr = sd.getArrForVarName(output);
                    if (arr != null) {
                        result.put(output, arr);
                    } else {
                        throw new IllegalStateException("Zero-op plan output '" + output +
                                "' is not a placeholder and has no array in SameDiff.");
                    }
                }
            }
            return result;
        }

        // Native C++ graph executor — no fallback to Java allowed.
        if (NATIVE_EXECUTOR_ENABLED) {
            if (nativeExecutorFailed) {
                throw new RuntimeException("Native DSP executor compilation previously failed. " +
                        "No fallback to Java permitted. Fix the native compilation issue.");
            }
            // Detect mode changes after initial compilation. Resolve through the same
            // SameDiff/system-property path and Triton fallback used by compileNativePlan()
            // so mode comparison stays stable throughout the plan lifetime. Native code
            // remains responsible for selecting the available GPU or CPU graph backend.
            GraphExecutionMode currentSdMode = resolveRequestedGraphExecutionMode(null);
            boolean tritonAvailableForCurrentMode =
                    currentSdMode != GraphExecutionMode.TRITON ||
                    isTritonAvailable(NativeOpsHolder.getInstance().getDeviceNativeOps());
            GraphExecutionMode effectiveCurrentMode = resolveEffectiveGraphExecutionMode(
                    currentSdMode,
                    tritonAvailableForCurrentMode,
                    sd.isDspFallbackToAutoIfTritonUnavailable());
            if (isNativePlanCompiled(plan) && effectiveCurrentMode != configuredGraphExecutionMode) {
                log.info("Native executor: mode change detected ({} -> {} effective={}), recompiling native plan",
                        configuredGraphExecutionMode, currentSdMode, effectiveCurrentMode);
                compileNativePlan(plan, currentSdMode, sd.isDspFallbackToAutoIfTritonUnavailable());
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

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Pointer stream = null;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            stream = nativeOps.dspGetExecutionStream(nativePlanHandle);
        }
        if (stream == null) {
            stream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
        }

        // Free with full dedup. liveGpuAddresses=null because no slots are live during close().
        int[] stats = freePendingBuffers(nativeOps, stream, null);
        pendingClose.clear();

        log.info("    closeSlotArrayCache: freePendingBuffers done ({}/{} freed, {}MB)", stats[0], stats[1], stats[2]);

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
        return freedCount;
    }

    /**
     * Returns an output wrapper owned by the exact backend that created the graph context.
     */
    private static OpaqueNDArray getOwnedOutput(OpaqueContext context, int outputIndex) {
        NativeBufferOwner owner = context.backendOwner();
        OpaqueNDArray output = owner.nativeOps().getOutputArrayNative(context, outputIndex);
        if (output != null && !output.isNull()) {
            output.attachOwner(owner);
        }
        return output;
    }

    /**
     * Resolve a flat native output-slot index back to its Java plan producer.
     * Native lifecycle validation reports flat slots because it intentionally does
     * not retain Java variable names; appending this context turns an otherwise
     * opaque ownership failure into an actionable op/output diagnosis.
     */
    private static String describePlanOutputSlot(DynamicShapePlan plan, String errorMessage) {
        if (plan == null || errorMessage == null) {
            return "";
        }

        int marker = errorMessage.indexOf("slot ");
        if (marker < 0) {
            return "";
        }
        int start = marker + "slot ".length();
        int end = start;
        while (end < errorMessage.length() && Character.isDigit(errorMessage.charAt(end))) {
            end++;
        }
        if (end == start) {
            return "";
        }

        final int outputSlot;
        try {
            outputSlot = Integer.parseInt(errorMessage.substring(start, end));
        } catch (NumberFormatException ignored) {
            return "";
        }

        DynamicShapeSlot[] planSlots = plan.getSlots();
        if (planSlots == null) {
            return " [flat output slot " + outputSlot + ", Java plan has no slots]";
        }
        for (int step = 0; step < planSlots.length; step++) {
            DynamicShapeSlot slot = planSlots[step];
            if (slot == null || slot.getOutputSlotIndices() == null) {
                continue;
            }
            int[] indices = slot.getOutputSlotIndices();
            String[] names = slot.getOutputVarNames();
            for (int output = 0; output < indices.length; output++) {
                if (indices[output] == outputSlot) {
                    String variable = names != null && output < names.length
                            ? names[output] : "<unnamed>";
                    return " [plan step " + step + ", op='" + slot.getOpName()
                            + "', output=" + output + ", variable='" + variable + "']";
                }
            }
        }
        return " [flat output slot " + outputSlot + " has no Java plan producer]";
    }

    /**
     * Retain the exact OpaqueNDArray wrapper most recently installed at an input index in
     * {@link #cachedOpContext}. The native context stores a raw NDArray pointer; retaining an older
     * wrapper for the same INDArray is insufficient because every OpaqueNDArray wrapper owns a
     * distinct native NDArray object.
     */
    private void retainContextInputRef(int index, OpaqueNDArray opaque, int inputCount) {
        if (opaque == null || index < 0 || index >= inputCount) {
            return;
        }
        if (contextInputRefs == null || contextInputRefs.length != inputCount) {
            OpaqueNDArray[] retained = new OpaqueNDArray[inputCount];
            if (contextInputRefs != null) {
                System.arraycopy(contextInputRefs, 0, retained, 0,
                        Math.min(contextInputRefs.length, retained.length));
            }
            contextInputRefs = retained;
        }
        contextInputRefs[index] = opaque;
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
        NativeBufferOwner backendOwner =
                MultiBackendNativeOpsHolder.getInstance().getOwnerForNativeOps(nativeOps);

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
        //
        // When shapes are frozen, we still need to detect when placeholder shapes CHANGE
        // (e.g., VLM prefill seq=N → decode seq=1 → prefill seq=N on next page).
        // Each distinct shape gets its own plan in the C++ cache. The C++ cache handles
        // shape-keyed dispatch natively — we just need to call it when shapes change.
        // Skip redispatch ONLY when frozen AND shapes are the same (performance optimization).
        long currentShapeHash = computePlaceholderShapeHash(placeholderArrays);
        boolean shapesChanged = (lastDispatchedShapeHash != 0 && currentShapeHash != lastDispatchedShapeHash);
        boolean needsRedispatch = (nativePlanHandle == null || nativePlanHandle.isNull())
                || (!shapesFrozen && !wasEverFrozen)
                || shapesChanged;
        if (needsRedispatch) {
            redispatchForCurrentShapes(placeholderArrays, shapesChanged);
        }
        lastDispatchedShapeHash = currentShapeHash;
        applyMutableExternalInputsIfNeeded(nativeOps, nativePlanHandle);

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
            // Use a reusable working array so we don't corrupt cachedInputArrays (needed for identity comparison)
            // but also don't allocate a new array on every decode step (1332+ entries × 250 tokens = GC pressure).
            if (frozenExtInputsWorkingCopy == null || frozenExtInputsWorkingCopy.length != extKeys.length) {
                frozenExtInputsWorkingCopy = new INDArray[extKeys.length];
            }
            System.arraycopy(cachedInputArrays, 0, frozenExtInputsWorkingCopy, 0, extKeys.length);
            extInputs = frozenExtInputsWorkingCopy;
            // Re-resolve VARIABLE-type inputs that may have been rebound via
            // associateArrayWithVariable(). That method updates SameDiff's storage
            // (variablesArrays) but not cachedInputArrays, so the frozen fast path
            // would silently use the old INDArray forever. Re-resolve from
            // sd.getVariable(name).getArr() and update extInputs[i] (but NOT
            // cachedInputArrays[i]) so the generic catch-all at the bottom of
            // the frozen path detects the identity change and properly rebinds
            // the C++ opContext with setGraphContextInputArray.
            //
            // Use cached variable-type indices to avoid HashMap lookups on all
            // 1332+ entries every step. Only VARIABLE-type entries (not CONSTANT,
            // not PLACEHOLDER) can be rebound — typically 0-5 entries in LLM models.
            if (cachedVariableTypeIndices == null) {
                // Build index cache on first frozen call
                int count = 0;
                for (int i = 0; i < extKeys.length; i++) {
                    if (inputIsPlaceholder != null && inputIsPlaceholder[i]) continue;
                    SDVariable var = sd.getVariable(extKeys[i]);
                    if (var != null && var.getVariableType() == VariableType.VARIABLE) {
                        count++;
                    }
                }
                cachedVariableTypeIndices = new int[count];
                int idx = 0;
                for (int i = 0; i < extKeys.length; i++) {
                    if (inputIsPlaceholder != null && inputIsPlaceholder[i]) continue;
                    SDVariable var = sd.getVariable(extKeys[i]);
                    if (var != null && var.getVariableType() == VariableType.VARIABLE) {
                        cachedVariableTypeIndices[idx++] = i;
                    }
                }
                DspDiagnostics.record(DspDiagnostics.EXECUTE,
                        "Java: cached " + count + " VARIABLE-type ext input indices out of "
                                + extKeys.length + " total");
            }
            int variableRebindCount = 0;
            for (int vi : cachedVariableTypeIndices) {
                INDArray current = sd.getVariable(extKeys[vi]).getArr();
                if (current != null && current != cachedInputArrays[vi]) {
                    extInputs[vi] = detachIfWorkspaceBacked(current, extKeys[vi]);
                    variableRebindCount++;
                }
            }
            if (variableRebindCount > 0) {
                DspDiagnostics.record(DspDiagnostics.EXECUTE,
                        "Java: FROZEN_FAST_PATH re-resolved " + variableRebindCount
                                + " VARIABLE-type inputs (associateArrayWithVariable rebind)");
            }
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
                            // The Java layer can re-resolve to the new array, but the C++ frozen
                            // plan holds raw NDArray* pointers baked into segment snapshots and
                            // CUDA graph arg tables from the ORIGINAL buffer. Those pointers are
                            // now dangling (the old buffer was freed). setGraphContextInputArray
                            // updates the opContext, but frozen segments don't re-read from opContext —
                            // they use their baked snapshot. Continuing would cause a use-after-free
                            // crash in native code (SIGSEGV in Workspace::allocateBytes).
                            // Throw here so callers get a clean Java exception instead of a JVM crash.
                            throw new RuntimeException(
                                "LIFECYCLE_ERROR: external input '" + extKeys[i] + "' (type=" + vt +
                                ") DataBuffer was closed and re-resolved to a new buffer between " +
                                "frozen DSP executions. The C++ frozen plan holds baked pointers to " +
                                "the OLD buffer which is now freed — continuing would cause a " +
                                "use-after-free crash. Close the plan executor or unfreeze shapes " +
                                "before swapping variable buffers. " +
                                "(dtype=" + fresh.dataType() +
                                ", shape=" + Arrays.toString(fresh.shape()) + ")");
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
                        if (ph != null && isArrayLive(ph)) {
                            extInputs[i] = detachIfWorkspaceBacked(ph, extKeys[i]);
                            resolvedCount++;
                        } else {
                            // Stale placeholder with no live replacement in the map.
                            // Passing a closed DataBuffer to native code causes SIGSEGV —
                            // the C++ plan dereferences freed GPU memory in Workspace::allocateBytes.
                            // Throw a clean Java exception instead of crashing the JVM.
                            throw new RuntimeException(
                                "LIFECYCLE_ERROR: external input '" + extKeys[i] + "' (type=PLACEHOLDER)" +
                                " DataBuffer was closed between DSP executions and no live replacement " +
                                "was provided in placeholderArrays. " +
                                (ph != null ? "(placeholder present but closed)" :
                                    "(placeholder missing from map)") +
                                " — cannot proceed, would use-after-free in native code.");
                        }
                    } else {
                        staleOtherCount++;
                        // Stale input of unrecognized type (null variable or unexpected VariableType).
                        // We cannot re-resolve it. Passing the closed DataBuffer to native code
                        // causes SIGSEGV. Throw a clean Java exception.
                        throw new RuntimeException(
                            "LIFECYCLE_ERROR: external input '" + extKeys[i] + "' (type=" + vt + ")" +
                            " DataBuffer was closed between DSP executions and cannot be re-resolved — " +
                            "variable type is not CONSTANT/VARIABLE/PLACEHOLDER. " +
                            "Continuing would use-after-free in native code.");
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
                            if (!isArrayLive(ph)) {
                                throw new RuntimeException(
                                    "LIFECYCLE_ERROR: placeholder input '" + extKeys[i] +
                                    "' DataBuffer was closed — cannot pass freed buffer to native plan.");
                            }
                            extInputs[i] = detachIfWorkspaceBacked(ph, extKeys[i]);
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
                    if (arr != null && !isArrayLive(arr)) {
                        throw new RuntimeException(
                            "LIFECYCLE_ERROR: external input '" + varName + "' (type=PLACEHOLDER)" +
                            " DataBuffer was closed — cannot pass freed buffer to native plan.");
                    }
                    arr = detachIfWorkspaceBacked(arr, varName);
                }
                if (arr == null) {
                    SDVariable var = sd.getVariable(varName);
                    if (var != null &&
                            (var.getVariableType() == VariableType.CONSTANT ||
                                    var.getVariableType() == VariableType.VARIABLE ||
                                    var.getVariableType() == VariableType.ARRAY)) {
                        arr = detachIfWorkspaceBacked(var.getArr(), varName);
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
                            "All external inputs must be resolved. No fallback permitted." +
                            " | extKeys=" + Arrays.toString(extKeys) +
                            " planOutputs=" + (currentPlan != null ? currentPlan.getRequestedOutputs() : null) +
                            " planId=" + System.identityHashCode(currentPlan) +
                            " sdVars=" + (sd != null ? sd.variableMap().keySet() : null));
                }
                extInputs[i] = arr;
            }
        }

        // Publish the resolved external input array so snapshots and lifecycle
        // checks see the same buffers that will be bound into the native context.
        this.externalInputs = extInputs;
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            retainExternalInputsForPlan(nativePlanHandle.address(), extInputs);
        }

        // Debug metadata only; value reductions would force host reads on CUDA.
        if (Nd4j.getEnvironment().isDebug() && extInputs.length > 1331 && extInputs[1331] != null) {
            INDArray ext1331 = extInputs[1331];
            log.info("EXT_INPUT_1331_METADATA: shape={} dtype={} length={} isAttached={}",
                    Arrays.toString(ext1331.shape()),
                    ext1331.dataType(),
                    ext1331.length(),
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
                            arr != null ? Arrays.toString(arr.shape()) : "null",
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
                            i, extKeys[i], Arrays.toString(arr.shape()));
                } else {
                    dataCount++;
                }
            }
            log.debug("EXT_INPUT_WRITE_SUMMARY: total={} null={} empty={} withData={}", extInputs.length, nullCount, emptyCount, dataCount);
        }

        // GATHER DIAGNOSTIC: dump external inputs that are [1,1] INT64 (likely position_ids for gather slot 0)
        // Gated behind isDebugEnabled — getInt(0) forces D2H sync, log.info() allocates strings.
        // Running unconditionally on every decode step costs ~1-5ms/step from the D2H sync alone.
        if (log.isDebugEnabled()) {
            for (int i = 0; i < Math.min(extInputs.length, 1333); i++) {
                INDArray arr = extInputs[i];
                if (arr != null && arr.rank() == 2 && arr.shape()[0] == 1 && arr.shape()[1] == 1
                        && arr.dataType() == DataType.INT64) {
                    long val = arr.getInt(0);
                    log.debug("GATHER_DIAG: extIdx={} name='{}' shape=[1,1] INT64 value={} executionCount={}",
                            i, extKeys[i], val, executionCount);
                }
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
                // resolveArrayDevice() prefers actual native pointer ownership.
                // AllocationPoint metadata is only the fallback for host-only inputs:
                // allocator routing and replication can leave that logical metadata
                // pointing at a different device than the resident CUDA allocation.
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

                // ── Multi-GPU sharding: per-external-input consuming device ─────────────
                // For a sharded plan, an external weight consumed ONLY by a secondary-device
                // segment must live on that device, NOT be pulled to the primary execution
                // device below. Pulling it back device-ping-pongs the weight: the device-1
                // segment migrates it to device 1 (warmup), then this loop yanks it to device 0,
                // and on captured-graph replay the baked device-1 address is stale → CUDA err700
                // (illegal access) in the device-1 op. Compute each external input's consuming
                // device from the plan's slot targetDeviceIds; -1=unknown, -2=mixed(both devices).
                int[] extConsumerDevice = null;
                if (numDevices > 1 && currentPlan != null && currentPlan.getSlots() != null) {
                    extConsumerDevice = new int[extInputs.length];
                    Arrays.fill(extConsumerDevice, -1);
                    for (DynamicShapeSlot slot : currentPlan.getSlots()) {
                        int slotDev = slot.getTargetDeviceId();
                        if (slotDev < 0) continue;  // unassigned (single-GPU path) — no constraint
                        int[] srcs = slot.getInputSourceIndices();
                        if (srcs == null) continue;
                        for (int s : srcs) {
                            if (s >= 0) continue;              // internal slot input
                            int extIdx = -(s + 1);
                            if (extIdx < 0 || extIdx >= extConsumerDevice.length) continue;
                            if (extConsumerDevice[extIdx] == -1) extConsumerDevice[extIdx] = slotDev;
                            else if (extConsumerDevice[extIdx] != slotDev) extConsumerDevice[extIdx] = -2;
                        }
                    }
                }

                int migratedCount = 0;
                long migratedBytes = 0;
                for (int i = 0; i < extInputs.length; i++) {
                    INDArray arr = extInputs[i];
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        int arrDevice = resolveArrayDevice(arr, numDevices, nativeExecutionDevice);
                        // Sharding: an input consumed solely by a secondary (non-primary) device
                        // stays on that device — the native per-segment path placed it there and
                        // the captured graph baked its address. Do NOT pull it to the primary.
                        if (extConsumerDevice != null
                                && extConsumerDevice[i] >= 0
                                && extConsumerDevice[i] != nativeExecutionDevice) {
                            continue;
                        }
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

                            // Cross-device migration via async peer copy on the DSP stream.
                            // Uses cudaMemcpyPeerAsync — the CUDA driver handles P2P or
                            // host-staging internally, all async on the execution stream.
                            // No sync D2H, no host relay, no pipeline drain.
                            log.info("DSP MIGRATE: ext[{}] '{}' shape={} dtype={} from device {} to {} (placeholder={}, view={})",
                                    i, extKeys != null && i < extKeys.length ? extKeys[i] : "?",
                                    Arrays.toString(arr.shape()), arr.dataType(),
                                    arrDevice, nativeExecutionDevice, isPlaceholder, arr.isView());
                            long startTime = System.nanoTime();

                            // For views, dup first to get a contiguous buffer
                            INDArray srcArr = arr.isView() ? arr.dup() : arr;

                            // Allocate destination buffer on the target device
                            INDArray migrated;
                            try (MemoryWorkspace ws =
                                    Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                migrated = Nd4j.createUninitialized(srcArr.dataType(), srcArr.shape(), srcArr.ordering());
                            }

                            // Async cross-device copy: cudaMemcpyPeerAsync on DSP stream.
                            // The copy is ordered on the same stream the plan executes on,
                            // so data is visible when the plan reads it — no sync needed.
                            Pointer execStream = nativePlanHandle != null
                                    ? nativeOps.dspGetExecutionStream(nativePlanHandle)
                                    : null;
                            nativeOps.dbAsyncCrossDeviceCopy(
                                    migrated.data().opaqueBuffer(),
                                    srcArr.data().opaqueBuffer(),
                                    execStream);

                            // Clean up view dup if we created one
                            if (srcArr != arr) {
                                srcArr.close();
                            }

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
                                    staleNonPlaceholderIndices = Arrays.copyOf(
                                            staleNonPlaceholderIndices, staleNonPlaceholderIndices.length * 2);
                                staleNonPlaceholderIndices[staleNonPlaceholderCount++] = i;
                            }
                        }
                    }
                }
                if (migratedCount > 0) {
                    // No commit() needed: dbAsyncCrossDeviceCopy uses cudaMemcpyPeerAsync
                    // on the DSP execution stream — the plan executes on the same stream,
                    // so CUDA stream ordering guarantees the copy completes first.

                    // Invalidate zeroCopyOutputCache: migrated inputs mean the plan must
                    // re-execute with the new data.
                    if (zeroCopyOutputCache != null) {
                        closeZeroCopyOutputCache();
                    }
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
            cachedOpContext = OpaqueContext.create(backendOwner, 1);
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
                        if (arr != null && isArrayLive(arr)) {
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, pi, opaqueIn);
                            retainContextInputRef(pi, opaqueIn, extInputs.length);
                            cachedInputOpaques[pi] = opaqueIn;
                            cachedInputArrays[pi] = arr;
                        }
                    }
                    for (int di : frozenDerivedExternalInputIndices) {
                        INDArray arr = resolveCanonicalExternalInput(extKeys[di], placeholderArrays);
                        if (!isArrayLive(arr)) {
                            throw new IllegalStateException("Frozen replay phase violation: derived external input '"
                                    + extKeys[di] + "' is not live during frozen execution");
                        }
                        extInputs[di] = arr;
                        OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                        nativeOps.setGraphContextInputArray(opContext, di, opaqueIn);
                        retainContextInputRef(di, opaqueIn, extInputs.length);
                        cachedInputOpaques[di] = opaqueIn;
                        cachedInputArrays[di] = arr;
                    }
                    for (int ci : frozenControlInputIndices) {
                        INDArray arr = resolveCanonicalExternalInput(extKeys[ci], placeholderArrays);
                        if (arr == null) {
                            arr = extInputs[ci];
                        }
                        if (arr != null && isArrayLive(arr)) {
                            extInputs[ci] = arr;
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, ci, opaqueIn);
                            retainContextInputRef(ci, opaqueIn, extInputs.length);
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
                            if (arr != null && isArrayLive(arr)) {
                                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                                nativeOps.setGraphContextInputArray(opContext, ci, opaqueIn);
                                retainContextInputRef(ci, opaqueIn, extInputs.length);
                                cachedInputOpaques[ci] = opaqueIn;
                                DspDiagnostics.record(DspDiagnostics.MEMORY,
                                    "Java: FROZEN_FAST_PATH re-set stale constant ext[" + ci + "] '" + extKeys[ci] + "'");
                            }
                        }
                    }

                    // Generic catch-all: any external input whose INDArray identity changed
                    // must be rebound into the native context, even if it was not classified
                    // as a placeholder, derived external, or small integral control input.
                    // This keeps the frozen fast path correct for graphs where mutable
                    // execution inputs are surfaced as non-placeholder externals.
                    int genericRebindCount = 0;
                    for (int i = 0; i < extInputs.length; i++) {
                        INDArray arr = extInputs[i];
                        if (arr == null || arr == cachedInputArrays[i]) {
                            continue;
                        }
                        if (!isArrayLive(arr)) {
                            throw new IllegalStateException(
                                    "FROZEN_FAST_PATH: external input '" + extKeys[i]
                                            + "' changed identity but is not live");
                        }

                        OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                        nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                        retainContextInputRef(i, opaqueIn, extInputs.length);
                        cachedInputOpaques[i] = opaqueIn;
                        cachedInputArrays[i] = arr;
                        genericRebindCount++;
                    }
                    if (genericRebindCount > 0) {
                        DspDiagnostics.record(DspDiagnostics.EXECUTE,
                                "Java: FROZEN_FAST_PATH rebound " + genericRebindCount
                                        + " identity-changed external inputs");
                    }
                } else {
                    // Fallback: full iteration (should not happen after first frozen call)
                    for (int i = 0; i < extInputs.length; i++) {
                        boolean staleBuffer = false;
                        if (extInputs[i] != null && !extInputs[i].isEmpty()) {
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
                            if (arrToSet != null && !arrToSet.isEmpty()
                                    && (arrToSet.data() == null || arrToSet.data().wasClosed())) {
                                try {
                                    arrToSet = Nd4j.zeros(arrToSet.dataType(), arrToSet.shape());
                                } catch (Exception e) {
                                    arrToSet = Nd4j.scalar(0.0f);
                                }
                                extInputs[i] = arrToSet;
                            }
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arrToSet);
                            nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
                            retainContextInputRef(i, opaqueIn, extInputs.length);
                            cachedInputOpaques[i] = opaqueIn;
                            cachedInputArrays[i] = extInputs[i];
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
                    //
                    // IMPORTANT: empty arrays (e.g., Const with shape [0]) legitimately have
                    // data() == null (BaseNDArray.createBufferForDescriptor returns null for
                    // non-scalar empty arrays). They are NOT dead branches. OpaqueNDArray.fromINDArray
                    // handles them correctly via the nativeLength==0 path. Skip the dead-branch
                    // guard entirely for empty arrays — only apply it to non-empty arrays whose
                    // data buffer is null or closed (genuinely dead control-flow branches).
                    if (arr != null && !arr.isEmpty() && (arr.data() == null || arr.data().wasClosed())) {
                        // Try to resolve a fresh copy from the variable
                        SDVariable var = sd.getVariable(extKeys[i]);
                        if (var != null) {
                            INDArray fresh = var.getArr();
                            if (fresh != null && !fresh.isEmpty() && fresh.data() != null && !fresh.data().wasClosed()) {
                                arr = fresh;
                            } else if (fresh != null && fresh.isEmpty()) {
                                // Fresh copy is also empty — use it as-is
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
                    if (arr == null || (!arr.isEmpty() && (arr.data() == null || arr.data().wasClosed()))) {
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

            // Set placeholder output slots on context — C++ plan will allocate and fill them.
            // Must be done every call (not just first frozen call) because C++ may
            // reorder output slot indices across executions, and skipping this causes
            // multi-output graphs to return wrong outputs after shape freezing.
            // Use a zero scalar rather than Nd4j.empty() because the empty singleton's
            // native shapeInfo layout can produce "not empty but null buffer" in
            // createOpaqueNDArray on first execution. Any valid array works here —
            // C++ replaces the output buffer during execution.
            for (int i = 0; i < numOutputs; i++) {
                INDArray dummy = Nd4j.scalar(DataType.FLOAT, 0.0f);
                OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
            }

            // Get execution stream — resolve it FRESH every execution; never reuse a
            // cached raw CUDA stream pointer. The plan-owned stream (dspGetExecutionStream)
            // is destroyed+recreated across recompiles (platformFreePlanResources deletes
            // it), and the LaunchContext fallback stream is thread-local — a cached pointer
            // to either can dangle. When that stale pointer is later handed to
            // dbFreeBuffersOnStream, cudaFreeAsync fails with CUDA 201 (invalid device
            // context). Re-resolving is a trivial JNI getter relative to a decode step.
            // (Native platformBeginExecution also no longer trusts this pointer for the
            // execution path — it uses its own ownedStream_ / live thread-local stream.)
            Pointer execStream = null;
            try {
                // Prefer plan-owned stream (created after first execution)
                if (nativePlanHandle != null) {
                    execStream = nativeOps.dspGetExecutionStream(nativePlanHandle);
                }
                // Fallback to LaunchContext default before plan has executed
                if (execStream == null) {
                    OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                    if (lc != null) {
                        execStream = nativeOps.lcExecutionStream(lc);
                    }
                }
                if (execStream != null) execStream.retainReference();
            } catch (Exception e) {
                // CPU backend
            }

            // Clear native shape caches before each execution — unless shapes are frozen.
            // During autoregressive decoding with dynamic shapes, KV cache dimensions grow
            // by 1 each step, so shapes are stale. When frozen, clearing is unnecessary.
            // Native phases are strictly linear; if a cache-owned native plan is already
            // SHAPES_FROZEN/REPLAYING while the Java wrapper is not, mirror the native
            // lifecycle instead of attempting an illegal rollback to SLOT_BY_SLOT.
            if (!shapesFrozen && nativePlanHandle != null && !nativePlanHandle.isNull()) {
                int currentNativePhase = nativeOps.getPlanPhase(nativePlanHandle);
                if (log.isTraceEnabled()) {
                    log.trace("SHAPE_RESET_CHECK: !shapesFrozen, C++ phase={}, handle=0x{}", currentNativePhase,
                            Long.toHexString(nativePlanHandle.address()));
                }
                if (currentNativePhase >= 1) {
                    enterJavaFrozenState("native-phase-sync-before-execute", currentNativePhase);
                }
            }
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
                        if (!Arrays.equals(snapShape, currShape)) {
                            throw new IllegalStateException(
                                    "LIFECYCLE_ERROR: external input " + i + " (" + extKeys[i] +
                                    ") shape changed during frozen execution: " +
                                    Arrays.toString(snapShape) + " → " +
                                    Arrays.toString(currShape) +
                                    ". Unfreeze shapes before changing constant/variable shapes.");
                        }
                    }
                }
            }

            // Final safety check: ensure no closed DataBuffer will reach native code.
            // This catches any remaining paths where a closed buffer slipped through
            // earlier validation. A SIGSEGV in Workspace::allocateBytes or DataBuffer::migrate
            // is far harder to diagnose than a clean Java exception thrown here.
            for (int i = 0; i < extInputs.length; i++) {
                INDArray arr = extInputs[i];
                if (arr != null && !arr.isEmpty()) {
                    DataBuffer db = arr.data();
                    if (db != null && db.wasClosed()) {
                        throw new IllegalStateException(
                            "LIFECYCLE_ERROR: external input " + i + " ('" + extKeys[i] +
                            "') has a CLOSED DataBuffer immediately before native execution — " +
                            "use-after-free would occur in native code. " +
                            "This indicates a validation gap upstream.");
                    }
                }
            }

            // Execute the plan in C++ + readback outputs atomically.
            // The native plan shares output slots across all executions. If another
            // thread calls execute() before this thread finishes readback (copyBuffer),
            // it overwrites the output slots — causing stale/wrong results. The lock
            // spans execute + readback to prevent this race.
            nativeExecLock.lock();
            try {
            if (log.isTraceEnabled()) {
                log.trace("DSP_EXEC_PRE: handle={} executionCount={} numInputs={} numOutputs={} frozen={}",
                        nativePlanHandle != null ? "0x" + Long.toHexString(nativePlanHandle.address()) : "null",
                        executionCount, numInputs, numOutputs, shapesFrozen);
            }
            long execStart = System.nanoTime();
            int status = nativeOps.executeDynamicShapePlan(
                    nativePlanHandle,
                    opContext,
                    execStream);
            long execMs = (System.nanoTime() - execStart) / 1_000_000;
            if (log.isTraceEnabled()) {
                log.trace("DSP_EXEC_POST: status={} execMs={} executionCount={}", status, execMs, executionCount);
            }

            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                String planSlotContext = describePlanOutputSlot(plan, errMsg);
                DspDiagnostics.recordTimed(DspDiagnostics.FALLBACK, -1, -1, "executeNative",
                        execMs * 1000, "Java: native execution FAILED status=" + status +
                        " msg=" + errMsg + planSlotContext + " executionCount=" + executionCount);
                if (DspDiagnostics.isEnabled(DspDiagnostics.LIFECYCLE)) {
                    try {
                        String lifecycleReport = DspDiagnostics.getPlanReport();
                        if (lifecycleReport != null && !lifecycleReport.isBlank()) {
                            log.error("DSP lifecycle diagnostics captured at native failure:\n{}", lifecycleReport);
                        }
                        String lifecycleJson = DspDiagnostics.getJsonReport();
                        if (lifecycleJson != null && !lifecycleJson.isBlank()) {
                            Deque<String> transitions = new ArrayDeque<>();
                            for (String line : lifecycleJson.split("\\R")) {
                                String upper = line.toUpperCase(Locale.ROOT);
                                if (upper.contains("RESET_FOR_WARMUP")
                                        || upper.contains("SEGMENT_EXEC_RESET")
                                        || upper.contains("INVALIDAT")
                                        || upper.contains("EVICT")
                                        || upper.contains("REACTIVATE")
                                        || upper.contains("REBIND")
                                        || upper.contains("PROTECTED_EXT")
                                        || upper.contains("POINTERS_UNSTABLE")
                                        || upper.contains("NEW_BORROWER")
                                        || upper.contains("UNSEAL")
                                        || upper.contains("OOM")) {
                                    // A large plan can emit one UNSEAL event per slot. Keep enough
                                    // history to retain the single causal rebind event that precedes
                                    // those transitions instead of logging only the resulting churn.
                                    if (transitions.size() == 1000) {
                                        transitions.removeFirst();
                                    }
                                    transitions.addLast(line);
                                }
                            }
                            if (!transitions.isEmpty()) {
                                log.error("DSP lifecycle transitions preceding native failure:\n{}",
                                        String.join(System.lineSeparator(), transitions));
                            }
                        }
                    } catch (Throwable diagnosticsError) {
                        log.warn("Unable to capture DSP lifecycle diagnostics after native failure: {}",
                                diagnosticsError.getMessage());
                    }
                }
                if (status == NATIVE_STATUS_STALE_BUFFER) {
                    // C++ detected a closed/destroyed DataBuffer. This means a constant or
                    // variable was GC'd between Java's input resolution and C++ execution.
                    // Throw a specific exception so callers can re-resolve and retry.
                    throw new IllegalStateException("Stale buffer detected by C++ during DSP execution: " +
                            (errMsg != null ? errMsg : "unknown input") + planSlotContext);
                }
                throw new RuntimeException("Native plan execution failed with status " + status +
                        ": " + (errMsg != null ? errMsg : "unknown error") + planSlotContext);
            }

            executionCount++;

            // Sync Java-side shapesFrozen/wasEverFrozen from the native plan phase after each execution.
            // The C++ plan advances SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING autonomously via
            // auto-seal (fires inside execute() on the first slot-by-slot pass). Java must mirror
            // this state so redispatchForCurrentShapes() takes the frozen multi-plan switch path
            // (line 1645) instead of the standard swap path (line 1683).
            //
            // Without this sync: plan swaps always take the else-branch which resets frozenCallCount=0
            // and clears cachedInputArrays. More critically, on the next plan's FIRST execution
            // (planLifecycle_.isSlotBySlot()=true), platformClearCastCache() is called, destroying
            // the persistent HALF-weight FP32 cast entries that compositeReplay baked at capture time.
            // When the original plan resumes compositeReplay, it finds an empty or stale cast cache,
            // reuses a wrong entry (sourceChanged=false check skips assign), and cuBLAS reads
            // partially-written garbage → NaN at index 2 of lm_logits output.
            //
            // With this sync: plan swaps use the frozen multi-plan switch path which:
            //   (a) calls setPlanShapesFrozen(newHandle, true) so the new plan starts SHAPES_FROZEN,
            //       skipping slot-by-slot warmup and therefore skipping platformClearCastCache()
            //   (b) keeps both plan handles pinned, preserving cast cache across swaps
            //   (c) does NOT reset frozenCallCount, preserving the frozen execution fast path
            //
            // CUDA graphs and TAD device pointers: when frozen or replaying, CUDA graphs may have
            // TAD device pointers baked in as kernel args. Register in the global counter so
            // InferenceSession suppresses clearTADCache() until this executor is closed/reset.
            if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
                int nativePhaseCode = nativeOps.getPlanPhase(nativePlanHandle);
                // PlanPhase native codes: SLOT_BY_SLOT=0, SHAPES_FROZEN=1, REPLAYING=2
                if (nativePhaseCode >= 1) {
                    if (!registeredAsFrozen) {
                        GLOBAL_FROZEN_EXECUTOR_COUNT.incrementAndGet();
                        registeredAsFrozen = true;
                        log.info("DSP_TAD_GUARD: plan reached native phase {} — registered in global frozen count {}",
                                nativePhaseCode, GLOBAL_FROZEN_EXECUTOR_COUNT.get());
                    }
                    // Sync Java frozen flags so subsequent plan swaps use the frozen multi-plan
                    // switch path (preserves cast cache, avoids platformClearCastCache in new plan).
                    // This is the fix for the HALF-weight FP16 NaN after plan swap
                    // (VLM lm_logits NaN, test39_RmsNormLinearFp16AfterPlanSwap).
                    if (!shapesFrozen) {
                        shapesFrozen = true;
                        wasEverFrozen = true;
                        log.info("DSP_PHASE_SYNC: native plan phase={} — syncing Java shapesFrozen=true " +
                                "(preserves cast cache across plan swaps, fixes FP16 NaN after swap)",
                                nativePhaseCode);
                    }
                }
            }

            DspDiagnostics.recordTimed(DspDiagnostics.EXECUTE, -1, -1, "executeNative",
                    execMs * 1000, "Java: native execution OK " + execMs + "ms" +
                    " frozen=" + shapesFrozen + " executionCount=" + executionCount);

            // ── Always-on: validate output arrays immediately after native returns ──
            // C++ execution succeeded (status=0) but output arrays may still be
            // invalid (null OpaqueDataBuffer, closed buffer, wrong device). Catch
            // these NOW rather than when getFloat()/toFloatVector() crashes later.
            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = getOwnedOutput(opContext, i);
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
            // results into independent copies. KV close in the decode loop only closes
            // those duped copies — NOT the cached originals — leaving zeroCopyOutputCache holding
            // stale data (previous step's logits) while appearing valid to the staleness guard.
            // Using the stale cache on the next outputDirect() call returns wrong tokens.
            // By skipping the cache entirely for non-direct calls we force fresh allocation,
            // and the cache is only built/used for direct calls where the caller uses the
            // returned references directly (no dup), so KV close invalidates the cache correctly.
            //
            // Guard: if any cached output array has been externally closed (e.g., KV outputs
            // closed by the GenerationPipeline after scatter), the cache is stale. Drop it so
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
                boolean cacheDtypeMismatch = false;
                for (int i = 0; i < numOutputs; i++) {
                    String outputName = requestedOutputs.get(i);

                    OpaqueNDArray opaqueOut = getOwnedOutput(opContext, i);
                    if (opaqueOut == null || opaqueOut.isNull()) continue;

                    INDArray cached = zeroCopyOutputCache.get(outputName);
                    if (cached == null) continue;

                    // Empty arrays have no data to copy — skip the buffer copy entirely.
                    // The cached array already has ARRAY_EMPTY from first execution.
                    if (cached.isEmpty()) {
                        copiedOutputs++;
                        continue;
                    }

                    long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);

                    // CRITICAL: Use the dtype reported by C++ for the actual output array, NOT
                    // the cached array's dtype.  When Triton compilation fails and the native
                    // CUDA fallback runs, the C++ output dtype may change relative to the dtype
                    // that was cached on the first execution (e.g. DOUBLE from warmup →  FLOAT
                    // after model-variable normalization).  Using the cached DOUBLE dtype to wrap
                    // a FLOAT device buffer causes dbCreateExternalDataBuffer to interpret the
                    // buffer as having 8-byte elements when it only has 4-byte elements, reading
                    // twice as many bytes as exist — the upper half is uninitialized GPU memory,
                    // which reads back as NaN when the DOUBLE-typed destination array is synced.
                    long[] nativeShapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                    DataType nativeDtype = ArrayOptionsHelper.dataType(nativeShapeInfo);

                    if (nativeDtype != cached.dataType()) {
                        // C++ output dtype changed since cache was built (e.g. Triton fallback
                        // path changed dtype from DOUBLE warmup to FLOAT native execution).
                        // The cache is no longer valid — drop it and rebuild below.
                        log.info("Native executor: zeroCopyOutputCache dtype mismatch for '{}' " +
                                 "(cached={}, native={}) — invalidating cache and rebuilding",
                                 outputName, cached.dataType(), nativeDtype);
                        cacheDtypeMismatch = true;
                        break;
                    }

                    DataType dtype = nativeDtype;  // guaranteed == cached.dataType() here

                    Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);
                    // On CUDA, prefer D2D copy from the device buffer. Calling
                    // getOpaqueNDArrayBuffer triggers buffer() → syncToPrimary which
                    // only syncs when primary is null (first call). On subsequent
                    // executions the primary buffer is already allocated so buffer()
                    // returns the STALE host pointer from the first sync without
                    // re-syncing, causing all-zero outputs. By passing null for
                    // primary when special is available, we force memcpyWithT to use
                    // the device-to-device path which always reads fresh GPU data.
                    Pointer nativePrimary = (nativeSpecial == null || nativeSpecial.isNull())
                            ? nativeOps.getOpaqueNDArrayBuffer(opaqueOut) : null;
                    OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                            length, dtype.toInt(), nativePrimary, nativeSpecial);
                    if (srcOdb != null) {
                        try {
                            OpaqueDataBuffer dstOdb = cached.data().opaqueBuffer();
                            if (dstOdb != null) {
                                // Readback trace (-Dnd4j.dsp.readbackTrace=true): source vs
                                // destination device addresses for the frozen zero-copy refresh.
                                // Ground-truth instrument for the close-weight readback mis-map
                                // (batch-only wrong 'afterClose' values with correct native slots):
                                // correlates with native DB_DELETE_BUFFERS / pool-reuse traces to
                                // show whether src or dst was freed-and-repurposed.
                                if (READBACK_TRACE) {
                                    Pointer dstSpecial = nativeOps.dbSpecialBuffer(dstOdb);
                                    Pointer dstPrimary = nativeOps.dbPrimaryBuffer(dstOdb);
                                    log.info("READBACK_TRACE out='{}' i={} len={} srcSpecial=0x{} "
                                            + "dstSpecial=0x{} dstPrimary=0x{} dstArrId={} dstClosed={}",
                                            outputName, i, length,
                                            Long.toHexString(nativeSpecial != null ? nativeSpecial.address() : 0L),
                                            Long.toHexString(dstSpecial != null ? dstSpecial.address() : 0L),
                                            Long.toHexString(dstPrimary != null ? dstPrimary.address() : 0L),
                                            System.identityHashCode(cached),
                                            cached.wasClosed());
                                }
                                nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                            }
                        } finally {
                            nativeOps.deleteDataBuffer(srcOdb);
                        }
                    }
                    copiedOutputs++;
                }

                if (cacheDtypeMismatch) {
                    closeZeroCopyOutputCache();
                    // Fall through to the fresh-allocation path below which rebuilds
                    // zeroCopyOutputCache with the correct dtype.
                } else {

                // Update outputSlots from zeroCopyOutputCache for introspection access
                if (outputSlots != null) {
                    Map<String, Integer> outputNameToSlot = plan.getOutputNameToSlotIndex();
                    for (Map.Entry<String, INDArray> entry : zeroCopyOutputCache.entrySet()) {
                        Integer slotIdx = outputNameToSlot.get(entry.getKey());
                        if (slotIdx == null || slotIdx < 0) {
                            slotIdx = findOutputSlotIndex(plan, entry.getKey());
                        }
                        if (slotIdx != null && slotIdx >= 0 && slotIdx < outputSlots.length) {
                            outputSlots[slotIdx] = entry.getValue();
                        }
                    }
                }

                // Sync execution stream to ensure async D2D copies are complete before
                // returning cached arrays. Same rationale as the non-frozen path below.
                Nd4j.getExecutioner().commit();

                long copyMs = (System.nanoTime() - copyStart) / 1_000_000;
                if (execMs > 100) {
                    log.info("Native executor: exec={}ms copy={}ms (frozen, {}/{} outputs copied)",
                            execMs, copyMs, copiedOutputs, numOutputs);
                }
                return zeroCopyOutputCache;
                }  // end else (no dtype mismatch)
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

                // Placeholders also have no slot in the native plan (slot -1).
                // Return the caller-supplied placeholder array directly.
                if (sdVar != null && sdVar.getVariableType() == VariableType.PLACEHOLDER
                        && placeholderArrays != null) {
                    INDArray phArr = placeholderArrays.get(outputName);
                    if (phArr != null) {
                        results.put(outputName, phArr.dup());
                        continue;
                    }
                }

                OpaqueNDArray opaqueOut = getOwnedOutput(opContext, i);
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
                if (DspDiagnostics.isEnabled(DspDiagnostics.SHAPE)) {
                    DspDiagnostics.record(DspDiagnostics.SHAPE,
                            "Java: native output '" + outputName + "' shape=" + Arrays.toString(shape)
                                    + " strides=" + Arrays.toString(strides) + " order=" + ordering);
                }

                // Empty-tensor short-circuit: when the C++ output is empty (ARRAY_EMPTY bit set
                // in shapeInfo, or zero-element shape), create a Java empty array with the
                // ARRAY_EMPTY flag preserved. Nd4j.createUninitialized() always creates DENSE
                // arrays, which would lose the ARRAY_EMPTY flag and cause equalShapes() to
                // return false when comparing against TF reference arrays that DO have ARRAY_EMPTY.
                boolean isEmptyOutput = Shape.isEmpty(shapeInfo) || length == 0;
                INDArray result;
                if (isEmptyOutput) {
                    result = Nd4j.emptyWithShape(shape, dtype);
                    results.put(outputName, result);
                    continue;
                }

                // Create a Java-owned INDArray with the EXACT strides from the C++ output.
                // The raw buffer copy below is a flat memcpy — the destination must have
                // matching strides so elements are interpreted correctly. If the C++ output
                // has non-contiguous strides (e.g., from a view-based permute op whose shape
                // function inherited the input's strides), using contiguous strides here
                // would mis-interpret the buffer layout and produce wrong results.
                result = Nd4j.createUninitialized(dtype, shape, strides, ordering);

                // Get raw pointers — prefer device buffer for D2D copy.
                // On CUDA, getOpaqueNDArrayBuffer() calls buffer() which triggers
                // syncToPrimary. This only syncs when the primary buffer is null
                // (first execution). On subsequent executions the primary is already
                // allocated, so buffer() returns STALE host data without re-syncing.
                // By passing null for primary when the device buffer is available, we
                // force memcpyWithT to use D2D copy from the always-fresh GPU buffer.
                Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);
                Pointer nativePrimary = (nativeSpecial == null || nativeSpecial.isNull())
                        ? nativeOps.getOpaqueNDArrayBuffer(opaqueOut) : null;
                OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                        length, dtype.toInt(), nativePrimary, nativeSpecial);
                if (srcOdb != null) {
                    try {
                        OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                        if (dstOdb != null) {
                            if (READBACK_TRACE) {
                                Pointer dstSpecial = nativeOps.dbSpecialBuffer(dstOdb);
                                Pointer dstPrimary = nativeOps.dbPrimaryBuffer(dstOdb);
                                log.info("READBACK_TRACE FRESH out='{}' i={} len={} opaque=0x{} srcSpecial=0x{} "
                                                + "srcPrimary=0x{} dstSpecial=0x{} dstPrimary=0x{} dstArrId={}",
                                        outputName, i, length, Long.toHexString(opaqueOut.address()),
                                        Long.toHexString(nativeSpecial != null ? nativeSpecial.address() : 0L),
                                        Long.toHexString(nativePrimary != null ? nativePrimary.address() : 0L),
                                        Long.toHexString(dstSpecial != null ? dstSpecial.address() : 0L),
                                        Long.toHexString(dstPrimary != null ? dstPrimary.address() : 0L),
                                        System.identityHashCode(result));
                            }
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

                if (DspDiagnostics.isEnabled(DspDiagnostics.SHAPE)) {
                    DspDiagnostics.record(DspDiagnostics.SHAPE,
                            "Java: readback output '" + outputName + "' strides="
                                    + Arrays.toString(result.stride()) + " order=" + result.ordering());
                }

                if (Nd4j.getEnvironment().isDebugAndVerbose() && result.rank() >= 2 && result.length() > 0) {
                    log.info("DSP_COPY_VERIFY[{}] shape={} dtype={} len={} asyncCopy=true",
                            outputName, Arrays.toString(result.shape()),
                            result.dataType(), result.length());
                }
                results.put(outputName, result);
            }

            // Sync the execution stream to ensure all async D2D copies (copyBuffer above)
            // are complete before returning results to the caller. Without this, the caller
            // receives INDArrays whose device buffers may still have in-flight D2D memcpy
            // operations, causing zero/stale data on host read (syncToPrimary uses stream 0
            // which has no ordering guarantee with the LC default stream's async copies).
            Nd4j.getExecutioner().commit();

            // Populate the outputSlots array with the results at their corresponding slot indices.
            // The outputSlots field is allocated at plan initialization but was never populated
            // with actual arrays during execution. Tests and introspection APIs access this field
            // via reflection to inspect the live output arrays after execution.
            if (outputSlots != null) {
                Map<String, Integer> outputNameToSlot = plan.getOutputNameToSlotIndex();
                for (Map.Entry<String, INDArray> entry : results.entrySet()) {
                    Integer slotIdx = outputNameToSlot.get(entry.getKey());
                    if (slotIdx == null || slotIdx < 0) {
                        slotIdx = findOutputSlotIndex(plan, entry.getKey());
                    }
                    if (slotIdx != null && slotIdx >= 0 && slotIdx < outputSlots.length) {
                        outputSlots[slotIdx] = entry.getValue();
                    }
                }
            }

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
                if (READBACK_TRACE) {
                    for (Map.Entry<String, INDArray> e : zeroCopyOutputCache.entrySet()) {
                        OpaqueDataBuffer odb = e.getValue().data().opaqueBuffer();
                        Pointer sp = odb != null ? nativeOps.dbSpecialBuffer(odb) : null;
                        Pointer pp = odb != null ? nativeOps.dbPrimaryBuffer(odb) : null;
                        log.info("READBACK_TRACE CACHE_BUILD out='{}' dstSpecial=0x{} dstPrimary=0x{} dstArrId={}",
                                e.getKey(),
                                Long.toHexString(sp != null ? sp.address() : 0L),
                                Long.toHexString(pp != null ? pp.address() : 0L),
                                System.identityHashCode(e.getValue()));
                    }
                }
                log.info("Native executor: cached {} output arrays for frozen reuse (skip allocation)", results.size());
            }

            long copyMs = (System.nanoTime() - copyStart) / 1_000_000;
            if (copyMs > 5 || execMs > 100) {
                log.info("Native executor: exec={}ms copy={}ms ({} outputs)", execMs, copyMs, numOutputs);
            }

            // Configure max-allocation for KV cache output slots after the first execution.
            // This pre-allocates oversized DataBuffers at max capacity so that subsequent
            // steps can reuse the same buffer with a new shape wrapper. The C++ plan
            // creates NDArrays with the actual output shape (not the max shape) — only the
            // underlying buffer is oversized. This keeps buffer addresses stable for CUDA
            // graph replay while giving op kernels the correct shape info.
            if (!maxAllocationConfigured && maxKvCacheLength > 0) {
                maxAllocationConfigured = configureMaxAllocationForKvCache(results, plan);
            }

            // Diagnostic: dump first few values of each output to compare with Java executor
            if (Boolean.getBoolean(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS)) {
                for (Map.Entry<String, INDArray> entry : results.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.length() > 0) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("NATIVE_OUT ").append(name).append(" shape=").append(Arrays.toString(arr.shape()));
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
        } finally {
            nativeExecLock.unlock();
        }
        }  // end bare block from input-setting
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
        // Plan lifetime is managed by sd::graph::NativePlanCache (C++) — do NOT free directly.
        // The cache evicts (and deletes) entries under LRU + memory budget policy.
        // Release every pin owned by this executor. This matters for frozen prefill/decode
        // switching, where the previous plan remains pinned while the new shape is active.
        try {
            Pointer cache = sd.getOrCreateNativePlanCache();
            if (cache != null && !cache.isNull()) {
                NativeOps nativeOps2 = NativeOpsHolder.getInstance().getDeviceNativeOps();
                List<Pointer> handles = new ArrayList<>(pinnedPlanHandles.values());
                if (handles.isEmpty() && nativePlanHandle != null && !nativePlanHandle.isNull()) {
                    // Backward-compatible fallback for a handle acquired before lease tracking.
                    handles.add(nativePlanHandle);
                }
                int released = 0;
                for (Pointer handle : handles) {
                    if (handle == null || handle.isNull()) continue;
                    try {
                        nativeOps2.unpinNativePlan(cache, handle);
                        released++;
                    } catch (Exception e) {
                        log.debug("    freeNativePlanHandle: unpin failed for handle={} (non-fatal): {}",
                                handle.address(), e.getMessage());
                    }
                }
                log.info("    freeNativePlanHandle: released {} cache pin(s)", released);
            }
        } catch (Exception e) {
            log.debug("    freeNativePlanHandle: cache cleanup failed (non-fatal): {}", e.getMessage());
        }
        pinnedPlanHandles.clear();
        retainedExternalInputsByPlanHandle.clear();
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
        mutableExternalInputsConfiguredHandleAddresses.clear();
        cachedInputOpaques = null;
        cachedInputArrays = null;
        contextInputRefs = null;
        inputIsPlaceholder = null;
        placeholderIndices = null;
        cachedVariableTypeIndices = null;
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
        closeSlotArrayCache();
        int nativeReplicaCount = nativeConstantReplicaCache != null ? nativeConstantReplicaCache.size() : 0;
        if (nativeReplicaCount > 0) {
            log.info("  DSP close() step 2: native constant replicas ({})", nativeReplicaCount);
        }
        int nativeReplicasClosed = closeNativeConstantReplicaCache();

        log.info("  DSP close() step 3: outputSlots");
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
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.deleteGraphContext(cachedOpContext);
            } catch (Exception e) {
                log.info("Error freeing cached OpaqueContext: {}", e.getMessage());
            }
            cachedOpContext = null;
        }

        lastDispatchedShapeHash = 0;

        // Decrement global frozen-executor count before freeing the native plan handle.
        // Once close() runs, the CUDA graphs are destroyed so their baked TAD kernel args
        // are gone; clearTADCache() may proceed safely when no other frozen executor remains.
        if (registeredAsFrozen) {
            GLOBAL_FROZEN_EXECUTOR_COUNT.decrementAndGet();
            registeredAsFrozen = false;
        }

        // Free native C++ plan handle reference. The plan is cache-owned and will
        // be cleaned up by the NativePlanCache destructor (or LRU eviction).
        // Do NOT call releaseGpuIntermediates() here: close() is a final cleanup
        // and the cache destructor handles freeing GPU resources. Calling it here
        // would free C++ slot arrays that the cache destructor also frees, causing
        // a double free on JVM shutdown.
        log.info("  DSP close() step 6: freeNativePlanHandle");
        freeNativePlanHandle("EXECUTOR_CLOSE");

        currentPlan = null;
        // Release strong refs to constant DataBuffers AFTER all cleanup steps.
        // Now that the plan is fully closed, these constants no longer need protection.
        protectedConstantBuffers = null;
        log.info("  DSP close() complete (nativeReplicasClosed={}, zeroCopyClosed={})",
                nativeReplicasClosed, zeroCopyClosed);
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
