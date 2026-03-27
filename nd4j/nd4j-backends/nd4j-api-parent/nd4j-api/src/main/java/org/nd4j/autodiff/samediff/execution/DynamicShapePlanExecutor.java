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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueConstantShapeBuffer;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueNDArrayArr;
import org.nd4j.nativeblas.OpaqueShapeList;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;

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

    private static final boolean TIMING_ENABLED = Boolean.parseBoolean(
            System.getProperty(ND4JSystemProperties.INFERENCE_TIMING, "false"));

    // Temporary: serialize parallel worker op execution to test if concurrent C++ calls cause heap corruption.
    // When nd4j.dsp.serialExec=true, only one worker thread executes at a time.
    private static final boolean SERIAL_EXEC = Boolean.getBoolean(ND4JSystemProperties.DSP_SERIAL_EXEC);
    private static final Object EXEC_LOCK = new Object();

    private static final boolean SHAPE_OVERRIDE = Boolean.parseBoolean(
            System.getProperty(ND4JSystemProperties.DSP_SHAPE_OVERRIDE, "true"));

    /** Whether native C++ graph executor is enabled. When true, the entire plan is executed
     *  in C++ via a single JNI call instead of per-op Java dispatch. Falls back to Java
     *  executor on any failure. Default: true. */
    private static final boolean NATIVE_EXECUTOR_ENABLED = !"false".equalsIgnoreCase(
            System.getProperty(ND4JSystemProperties.DSP_NATIVE_EXECUTOR_ENABLED, "true"));

    /** Cache the nd4j.dsp.trace property check at class init time instead of calling
     *  System.getProperty() on every slot (1962 per frame * 20 frames = 39,240 calls).
     *  System.getProperty() acquires a lock on System.properties each invocation.
     *  When enabled, System.err.println + flush() is synchronous and unbuffered. */
    private static final boolean DSP_TRACE_ENABLED = System.getProperty(ND4JSystemProperties.DSP_TRACE) != null;

    /** Post-slot CUDA error check interval. Checking lastErrorCode() after every op is
     *  a JNI call that adds ~2us * 1962 ops = ~4ms per frame. In non-debug mode, check
     *  only every N ops. Set nd4j.dsp.errorCheckInterval=1 to check every op (for debugging).
     *  Default: 50 (check every 50 ops, ~39x fewer JNI calls). */
    private static final int ERROR_CHECK_INTERVAL = Integer.getInteger(ND4JSystemProperties.DSP_ERROR_CHECK_INTERVAL, 50);

    /** Whether view-producer detection has been completed (first execution establishes
     *  the slotIsViewProducer[] array; subsequent executions can skip pre/post GPU address
     *  comparison for known non-view-producer slots). */
    private boolean viewProducerDetectionDone;

    /** Number of execute() calls on this executor. Used to skip cache validity probe
     *  on the first execution (no cached entries yet). */
    private int executionCount;

    /** Whether op type logging has been done for the current plan. Only log once per plan
     *  instead of on every execute() call. With 1962 slots, building the map takes ~0.5ms. */
    private boolean opTypesLogged;

    /** Java-side tracking of shapes-frozen state. When true, shape caches don't need
     *  clearing between executions because all shapes are guaranteed constant. */
    private boolean shapesFrozen;

    /** Optional interceptor called after each slot execution. Null by default (zero overhead). */
    private SlotOutputInterceptor slotOutputInterceptor;


    private final SameDiff sd;
    private final SessionMemMgr mmgr;

    /** The plan this executor is currently configured for. */
    private DynamicShapePlan currentPlan;

    /** Flat output array slots: stores op outputs by slot index. */
    private INDArray[] outputSlots;

    /** External input array cache: resolved constant/variable/placeholder arrays. */
    private INDArray[] externalInputs;

    /** Whether constants/variables in externalInputs have been resolved and cached.
     *  When true, only placeholders are re-resolved on subsequent execute() calls. */
    private boolean externalConstantsResolved;

    /** DataBuffers belonging to model weights (constants/variables). These must NEVER be
     *  un-poisoned or closed — doing so corrupts model weights and produces garbage output.
     *  Built once after resolveExternalInputs and used as defense-in-depth in release/flush. */
    private IdentityHashMap<DataBuffer, Boolean> protectedWeightBuffers;

    /** BitSet tracking which slots are currently live (have valid arrays). */
    private BitSet liveSlots;

    /** Pending DataBuffers to close. Accumulated during execution and periodically flushed
     *  to reclaim GPU memory mid-execution (not just at the end). */
    private ArrayList<DataBuffer> pendingClose = new ArrayList<>();

    /** Persistent dedup sets across all flushes within one execute() call.
     *  Identity dedup prevents processing the same DataBuffer object twice (views sharing parents).
     *  ODB dedup prevents double-close of the same native OpaqueDataBuffer.
     *  GPU address dedup is per-batch (local to each freePendingBuffers call) to prevent
     *  double-free of the same GPU address within a single flush batch, while allowing
     *  pool-reused addresses to be freed in subsequent flushes. */
    private Set<DataBuffer> seenIdentity;
    private HashSet<Long> closedOdbAddresses;

    /** Buffers deferred from a mid-execution flush because their GPU address was still
     *  used by a live slot (view of parent). Re-checked on the next flush when the view
     *  slot may have been released. */
    private ArrayList<DataBuffer> deferredClose = new ArrayList<>();

    /** Flush pendingClose every RELEASE_FLUSH_INTERVAL ops during execution to reduce
     *  peak GPU memory. Vision encoder with 1962 ops accumulates ~10GB of dead intermediates
     *  if we only flush at the end. Periodic flushing reduces peak by ~50%. */
    private static final int RELEASE_FLUSH_INTERVAL = Integer.getInteger(ND4JSystemProperties.DSP_FLUSH_INTERVAL, 100);

    /** Per-slot eviction threshold for selective cache eviction. Arrays smaller than this
     *  survive eviction (scalars, shapes, small intermediates reused in decode). Only arrays
     *  larger than this threshold are evicted when total cache exceeds 512MB.
     *  Default 64KB — covers typical scalar/shape utility arrays. */
    private static final long PER_SLOT_EVICTION_THRESHOLD = Long.getLong(
            ND4JSystemProperties.DSP_PER_SLOT_EVICTION_THRESHOLD, 64L * 1024);

    /** Byte threshold below which freePendingBuffers uses a fast path that skips
     *  GPU address dedup and live view range check. For decode steps with tiny
     *  intermediates (seq_len=1), aliasing is extremely unlikely. */
    private static final long FAST_CLOSE_THRESHOLD = Long.getLong(
            ND4JSystemProperties.DSP_FAST_CLOSE_THRESHOLD, 10L * 1024 * 1024);

    /** Byte threshold for memory-pressure flush. When accumulated pendingClose bytes exceed
     *  this, flush immediately instead of waiting for the op-count interval. Prevents
     *  multi-GB intermediate accumulation between flush intervals (e.g., 95 ops × 48MB = 4.5GB). */
    private static final long FLUSH_BYTE_THRESHOLD = Long.getLong(
            ND4JSystemProperties.DSP_FLUSH_BYTE_THRESHOLD, 256L * 1024 * 1024);

    /** Persistent buffer pool for cross-execution array reuse (avoids mmgr round-trip each step). */
    private LocalBufferPool localPool;

    /** Devices that hit unrecoverable CUDA errors (e.g., error 700 from OOM cascades).
     *  Once a device is marked failed, all remaining ops redirect to device 0. Reset per execute(). */
    private Set<Integer> failedDevices;

    /** Per-device P2P accessibility from device 0. Computed once during initialize().
     *  isPeerAccessible[d] is true if device d is device 0 or has P2P access from device 0.
     *  Used to gate shape buffer correction (ensureShapeOnDevice) — only needed for non-P2P devices. */
    private boolean[] isPeerAccessible;

    /** Cached device count from nativeOps.getAvailableDevices(). Computed once during
     *  initialize() to avoid repeated JNI calls (~2us each × 7+ call sites per execute()).
     *  Device count doesn't change at runtime. Defaults to 1 for CPU backend. */
    private int cachedNumDevices = 1;

    /** Per-slot device ID cache. Tracks the device each output slot was allocated on.
     *  Parallel to outputSlots[]. -1 means unknown (requires JNI dbDeviceId() fallback).
     *  Updated when outputs are allocated (Step 4) and when failover changes the device.
     *  Used in Step 1b to avoid dbDeviceId() JNI calls for op-output inputs (~7000 calls
     *  per vision frame at ~2us each = ~14ms saved). */
    private int[] outputSlotDeviceIds;

    /** Per-external-input device ID cache. Parallel to externalInputs[].
     *  -1 means unresolved (will be queried via JNI and cached on first access).
     *  External inputs (constants, variables) don't change device between slots. */
    private int[] externalInputDeviceIds;

    /** Slot-indexed array cache: persists across execute() calls for O(1) array reuse.
     *  Same slot always produces the same shape in autoregressive decoding, so we cache
     *  by slot index instead of using TreeMap lookup. Non-view arrays are stored here
     *  when released; views go to pendingClose. Output slots are NOT cached (claimed by caller). */
    private INDArray[] slotArrayCache;

    /** Tracks which slots produce view outputs (C++ replaces our pre-allocated buffer).
     *  Set after first execution; skip allocation for these slots on subsequent steps. */
    private boolean[] slotIsViewProducer;

    /** Tracks which slot indices are final outputs (claimed by caller). These slots must
     *  always have real allocations even if they're view producers. */
    private BitSet outputSlotSet;

    /** Persistent OpContext pool (avoids native allocation each step). */
    private final ArrayDeque<OpContext> ctxPool = new ArrayDeque<>();

    // Timing accumulators
    private long timingWireInputsNs, timingSyncNs, timingShapeNs, timingAllocNs, timingExecNs, timingReleaseNs;
    private int timingShapeHits, timingShapeMisses;
    private int timingZeroSkipped, timingZeroApplied;
    private int timingPoolHits, timingPoolMisses;
    private int timingViewSkips, timingFreshAllocs;
    // Cache miss diagnostics
    private Map<String, Integer> timingCacheMissReasons = new HashMap<>();
    private int timingCacheLeakedConstant;
    private long timingCacheLeakedConstantBytes;
    // Per-op-type timing: opName -> [totalNs, count, maxNs]
    private Map<String, long[]> perOpTimingNs;
    // Time bucket counters: <1ms, 1-10ms, 10-100ms, >100ms
    private int timingBucketSub1ms, timingBucket1to10ms, timingBucket10to100ms, timingBucketOver100ms;
    // Release diagnostics
    private int pendingCloseCount, pendingCloseViewCount;
    private long pendingCloseBytes;
    private int totalFlushedCount;
    private long totalFlushedBytes;
    // Cross-device replica diagnostics
    private int replicaCount;
    private long replicaBytes;
    private int replicaToDev0Count, replicaToDev1Count;
    private long replicaToDev0Bytes, replicaToDev1Bytes;
    private int wrongDeviceCacheEjections;
    private int replicaCacheHits;
    private int shapeBufferCorrections;

    /** Cache of constant external inputs replicated to non-primary devices.
     *  Key: (extIdx << 16) | targetDevice. Value: replicated INDArray on target device.
     *  Only populated for arrays with isConstant() data buffers (model weights).
     *  Persists across execute() calls — constants don't change between decode steps. */
    private Map<Integer, INDArray> constantReplicaCache;

    /** Self-managed native workspace for C++ op temporaries. Created lazily when the
     *  SessionMemMgr doesn't provide one (e.g., ArrayCacheMemoryMgr). On CUDA this uses
     *  cudaHostAlloc (pinned host memory) so overruns from C++ ALLOCATE macro stay within
     *  the workspace buffer instead of corrupting the glibc heap.
     *
     *  CRITICAL: Must be large enough for 2x growth factor allocations. With 2.3M elements
     *  and 2x growth, we need 4.6M elements = ~18MB for FLOAT32. 32MB provides headroom. */
    private Pointer ownNativeWorkspace;
    private static final long DSP_NATIVE_WORKSPACE_BYTES = 32L * 1024 * 1024;

    /** Counter for workspace scope reset throttling. Tracks how many ops have executed
     *  since the last workspace ScopeOut/ScopeIn reset. The workspace fills after ~30 ops
     *  (32MB / ~1MB avg temp per op), so resetting every WORKSPACE_RESET_INTERVAL ops
     *  prevents spill while saving ~96% of JNI calls (2 JNI calls × 1962 ops = ~3924 saved
     *  per vision frame). Set nd4j.dsp.workspaceResetInterval=1 to reset every op. */
    private int workspaceOpsSinceReset;
    private static final int WORKSPACE_RESET_INTERVAL = Integer.getInteger(
            ND4JSystemProperties.DSP_WORKSPACE_RESET_INTERVAL, 25);

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
            return;
        }
        // Plan changed — flush old pool and slot cache when switching plans
        if (localPool != null) {
            localPool.flushTo(mmgr);
        }
        closeSlotArrayCache();
        currentPlan = plan;
        int totalSlots = plan.getTotalOutputSlots();
        outputSlots = new INDArray[totalSlots];
        outputSlotDeviceIds = new int[totalSlots];
        Arrays.fill(outputSlotDeviceIds, -1);
        externalInputs = new INDArray[plan.getExternalInputKeys().length];
        externalInputDeviceIds = new int[plan.getExternalInputKeys().length];
        Arrays.fill(externalInputDeviceIds, -1);
        externalConstantsResolved = false;
        liveSlots = new BitSet(totalSlots);
        pendingClose = new ArrayList<>();
        localPool = new LocalBufferPool();
        slotArrayCache = new INDArray[totalSlots];
        slotIsViewProducer = new boolean[totalSlots];
        // Reset per-plan state for new plan
        viewProducerDetectionDone = false;
        opTypesLogged = false;
        // Reset native executor state for new plan
        freeNativePlanHandle();
        nativeExecutorFailed = false;

        // Cache device count and compute per-device P2P accessibility once.
        // Used by ensureShapeOnDevice() and all multi-GPU paths to avoid repeated
        // getAvailableDevices() JNI calls (~2us each × 7+ sites per execute()).
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            cachedNumDevices = nativeOps.getAvailableDevices();
            isPeerAccessible = new boolean[cachedNumDevices];
            isPeerAccessible[0] = true; // device 0 is always accessible from itself
            for (int d = 1; d < cachedNumDevices; d++) {
                isPeerAccessible[d] = nativeOps.isPeerAccessSupported(0, d);
            }
        } catch (Exception e) {
            // CPU backend or unavailable — no multi-GPU, no shape correction needed
            cachedNumDevices = 1;
            isPeerAccessible = null;
        }
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
            zeroCopyOutputCache = null;
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
     * Sets an optional interceptor that is called after each slot execution.
     * The interceptor receives the output array directly — implementations
     * must {@code dup()} arrays they want to retain.
     * Pass {@code null} to disable (default).
     */
    public void setSlotOutputInterceptor(SlotOutputInterceptor interceptor) {
        this.slotOutputInterceptor = interceptor;
    }

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

            byte[] serialized = plan.serialize();
            if (serialized == null || serialized.length == 0) {
                log.warn("Native executor: plan serialization returned empty, cannot compile native plan");
                nativeExecutorFailed = true;
                return GraphExecutionMode.AUTO;
            }

            BytePointer planBytes = new BytePointer(serialized);
            try {
                nativePlanHandle = nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
            } catch (UnsupportedOperationException e) {
                log.debug("Native executor: backend does not support compileDynamicShapePlan");
                nativeExecutorFailed = true;
                return GraphExecutionMode.AUTO;
            } finally {
                planBytes.close();
            }

            if (nativePlanHandle == null || nativePlanHandle.isNull()) {
                log.warn("Native executor: compileDynamicShapePlan returned null handle");
                nativePlanHandle = null;
                nativeExecutorFailed = true;
                return GraphExecutionMode.AUTO;
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
            // When the plan changes (e.g., fullOutputNames → logitsOnly), freeNativePlanHandle()
            // resets kvCacheRetentionConfigured. Re-apply saved params on the new handle
            // so C++ scatter continues to work.
            if (savedKvPresentOutputNames != null && !kvCacheRetentionConfigured) {
                log.info("Native executor: re-applying KV cache retention on new plan (pos={})", savedKvCurrentPos);
                reapplyKvCacheRetention(plan);
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

        // Try native C++ graph executor if enabled and not previously failed.
        // This executes the entire plan in C++ via a single JNI call, avoiding
        // per-op Java→JNI→C++ round-trips (~15-20μs each × 1962 ops = ~30ms overhead).
        if (NATIVE_EXECUTOR_ENABLED && !nativeExecutorFailed) {
            if (!isNativePlanCompiled(plan) && sd.isDspNativeAutoCompileEnabled()) {
                compileNativePlan(plan, null, sd.isDspFallbackToAutoIfTritonUnavailable());
            }
            try {
                Map<String, INDArray> nativeResult = executeNative(plan, placeholderArrays);
                if (nativeResult != null) {
                    return nativeResult;
                }
                // null means native execution not available, fall through to Java
            } catch (Exception e) {
                // No fallback — native executor failures must be fixed, not masked
                log.error("Native executor failed — no fallback to Java allowed: {}", e.getMessage());
                throw new RuntimeException("Native DSP executor failed at plan execution. " +
                        "No fallback permitted. Fix the native executor. Error: " + e.getMessage(), e);
            }
        }

        // Clear per-slot shape caches between executions — unless shapes are frozen.
        // During autoregressive decoding with dynamic shapes, KV cache dimensions grow
        // by 1 each step, so shapes computed in the previous step are stale.
        // When shapesFrozen=true, all shapes are guaranteed constant so clearing is unnecessary.
        if (!shapesFrozen) {
            plan.clearAllShapeCaches();
        }

        pendingCloseBytes = 0;  // Reset unconditionally (used by byte-based flush trigger)
        workspaceOpsSinceReset = 0;  // Start fresh for workspace reset throttling
        if (TIMING_ENABLED) {
            timingWireInputsNs = timingSyncNs = timingShapeNs = timingAllocNs = timingExecNs = timingReleaseNs = 0;
            timingShapeHits = timingShapeMisses = 0;
            timingZeroSkipped = timingZeroApplied = 0;
            timingPoolHits = timingPoolMisses = 0;
            timingViewSkips = timingFreshAllocs = 0;
            timingCacheMissReasons.clear();
            timingCacheLeakedConstant = 0;
            timingCacheLeakedConstantBytes = 0;
            if (perOpTimingNs == null) perOpTimingNs = new HashMap<>();
            perOpTimingNs.clear();
            timingBucketSub1ms = timingBucket1to10ms = timingBucket10to100ms = timingBucketOver100ms = 0;
            pendingCloseCount = pendingCloseViewCount = 0;
        }

        // Clear output slots from the previous execution.
        Arrays.fill(outputSlots, null);
        Arrays.fill(outputSlotDeviceIds, -1);
        // Only clear placeholder external inputs — constants/variables are cached.
        if (externalConstantsResolved) {
            byte[] sourceTypes = plan.getExternalInputSourceTypes();
            for (int i = 0; i < externalInputs.length; i++) {
                byte srcType = (sourceTypes != null && i < sourceTypes.length) ? sourceTypes[i] : -1;
                if (srcType == DynamicShapeSlot.SOURCE_PLACEHOLDER || srcType < 0) {
                    externalInputs[i] = null;
                    externalInputDeviceIds[i] = -1;
                }
            }
        } else {
            Arrays.fill(externalInputs, null);
            Arrays.fill(externalInputDeviceIds, -1);
        }
        liveSlots.clear();
        pendingClose.clear();

        // Initialize persistent dedup sets for this execution.
        seenIdentity = Collections.newSetFromMap(new IdentityHashMap<>());
        closedOdbAddresses = new HashSet<>();
        deferredClose.clear();
        totalFlushedCount = 0;
        totalFlushedBytes = 0;
        replicaCount = 0;
        replicaBytes = 0;
        replicaToDev0Count = replicaToDev1Count = 0;
        replicaToDev0Bytes = replicaToDev1Bytes = 0;
        wrongDeviceCacheEjections = 0;
        replicaCacheHits = 0;
        shapeBufferCorrections = 0;

        // Resolve external inputs (constants, variables, placeholders)
        resolveExternalInputs(plan, placeholderArrays);

        // Build protection set for weight DataBuffers (once).
        // These must NEVER be un-poisoned or closed — doing so corrupts model weights.
        if (protectedWeightBuffers == null) {
            protectedWeightBuffers = new IdentityHashMap<>();
            byte[] srcTypes = plan.getExternalInputSourceTypes();
            for (int i = 0; i < externalInputs.length; i++) {
                if (externalInputs[i] != null && srcTypes != null && i < srcTypes.length
                        && (srcTypes[i] == DynamicShapeSlot.SOURCE_CONSTANT || srcTypes[i] == DynamicShapeSlot.SOURCE_VARIABLE)) {
                    DataBuffer db = externalInputs[i].data();
                    if (db != null) protectedWeightBuffers.put(db, Boolean.TRUE);
                }
            }
            log.info("DSP: built protectedWeightBuffers set with {} entries", protectedWeightBuffers.size());
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        // Clear stale CUDA error state from previous execution's cleanup (closePendingBuffers,
        // evictOversizedSlotCache, trimPool). Without this, a CUDA error from step N's cleanup
        // propagates as a sticky error into step N+1, causing the first memsetAsync/op to fail.
        nativeOps.clearLastError();

        DynamicShapeSlot[] slots = plan.getSlots();
        if (localPool == null) localPool = new LocalBufferPool();

        // Get a fresh execution stream for this execute() call. Between calls,
        // intermediate JNI operations (closePendingBuffers, trimPool, mmgr.scopeOut)
        // can trigger ContextBuffers release, invalidating cached stream handles.
        // Using getFreshExecutionStream() ensures we get a currently-valid stream.
        Pointer execStream = null;
        try {
            execStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
            if (execStream != null) execStream.retainReference();
        } catch (Exception e) {
            // CPU backend or unavailable — try fallback
            try {
                OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                if (lc != null) {
                    execStream = nativeOps.lcExecutionStream(lc);
                    if (execStream != null) execStream.retainReference();
                }
            } catch (Exception e2) {
                // CPU backend
            }
        }

        executionCount++;

        // Build set of output slot indices — these slots must always have real allocations
        // even if they're view producers, because the caller needs the data.
        outputSlotSet = new BitSet();
        failedDevices = new HashSet<>();
        for (int si : plan.getOutputNameToSlotIndex().values()) {
            if (si >= 0) outputSlotSet.set(si);
        }

        // FIXME: Parallel execution (DeviceWorker threads) causes latent heap corruption
        // ("double free or corruption (out)") that manifests on the second chunk. The corruption
        // occurs even with serialized execution (nd4j.dsp.serialExec=true), ruling out
        // concurrency as the root cause. Likely a C++ thread-local state or ContextBuffers
        // lifecycle issue with worker threads. Sequential mode handles multi-device execution
        // correctly via CudaExecutioner.ensureDeviceCoherency() which migrates buffers between
        // devices transparently. Enable parallel mode explicitly with nd4j.dsp.forceParallel=true.
        if (plan.getNumDistinctDevices() > 1 && plan.getSuccessors() != null
                && Boolean.getBoolean(ND4JSystemProperties.DSP_FORCE_PARALLEL)) {
            log.debug("DSP executing in PARALLEL mode ({} devices, {} ops)",
                    plan.getNumDistinctDevices(), slots.length);
            return executeParallel(plan, placeholderArrays, nativeOps, execStream);
        }

        // Log unique op types once per plan (not every execute() call).
        // Building the map iterates all 1962 slots — wasted work on decode steps.
        if (!opTypesLogged) {
            java.util.Map<String, Integer> opCounts = new java.util.LinkedHashMap<>();
            for (DynamicShapeSlot s : slots) {
                String name = s.getOp().opName();
                opCounts.merge(name, 1, Integer::sum);
            }
            log.debug("DSP unique ops ({}): {}", opCounts.size(), opCounts);
            opTypesLogged = true;
        }

        // Suppress autoGc during DSP execution. DSP manages its own memory via
        // pendingClose + flushPendingClose. The DeallocatorService's autoGcWindow (default 100ms)
        // calls System.gc() every 100ms when the PhantomReference queue is empty (which it always
        // is during DSP since we close buffers directly). This causes 10+ Full GCs/sec, each
        // ~144ms, consuming more time than actual computation.
        // Use Integer.MAX_VALUE instead of 0 since CudaConfiguration.setNoGcWindowMs rejects < 1.
        int savedAutoGcWindow = Nd4j.getMemoryManager().getAutoGcWindow();
        Nd4j.getMemoryManager().setAutoGcWindow(Integer.MAX_VALUE);

        // Cache the current device ID once before the main loop. executeSlot() uses this
        // instead of calling getDeviceForCurrentThread() per-slot, saving ~1962 JNI calls
        // per vision frame (~4ms at ~2us/call). The finally block in executeSlot() restores
        // the device after each slot, so this value stays accurate across iterations.
        int cachedDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        Map<String, INDArray> results = new LinkedHashMap<>();
        try {
            for (int stepIdx = 0; stepIdx < slots.length; stepIdx++) {
                DynamicShapeSlot slot = slots[stepIdx];
                OpContext ctx = ctxPool.pollFirst();
                if (ctx == null) {
                    ctx = Nd4j.getExecutioner().buildContext();
                }
                ctx.purge();

                try {
                    if (stepIdx % 500 == 0 || stepIdx == slots.length - 1) {
                        log.debug("DSP step {}/{}: op={}", stepIdx, slots.length, slot.getOpName());
                    }
                    // Trace-level per-slot logging to stderr for crash diagnosis.
                    // When a native crash (SIGSEGV) occurs, the last stderr line identifies
                    // the slot that was executing. Uses stderr because it's unbuffered and
                    // survives native crashes (unlike log4j which may have buffered output).
                    // Property check is cached at class init (DSP_TRACE_ENABLED) to avoid
                    // System.getProperty() lock acquisition on every slot.
                    if (DSP_TRACE_ENABLED) {
                        System.err.println("DSP slot " + stepIdx + "/" + slots.length + ": " + slot.getOpName());
                        System.err.flush();
                    }
                    long tSlot0 = TIMING_ENABLED ? System.nanoTime() : 0;
                    executeSlot(slot, ctx, nativeOps, localPool, execStream, cachedDeviceId);
                    if (slotOutputInterceptor != null) {
                        int[] outSlotIds = slot.getOutputSlotIndices();
                        String[] outVarNames = slot.getOutputVarNames();
                        if (outSlotIds != null) {
                            for (int oi = 0; oi < outSlotIds.length; oi++) {
                                INDArray outArr = outputSlots[outSlotIds[oi]];
                                slotOutputInterceptor.onSlotOutput(stepIdx, slot.getOpName(),
                                        outVarNames != null && oi < outVarNames.length ? outVarNames[oi] : "slot_" + outSlotIds[oi],
                                        outArr, outSlotIds[oi]);
                            }
                        }
                    }
                    if (TIMING_ENABLED) {
                        long slotNs = System.nanoTime() - tSlot0;
                        double slotMs = slotNs / 1_000_000.0;
                        // Accumulate per-op-type timing
                        String opName = slot.getOpName();
                        long[] stats = perOpTimingNs.get(opName);
                        if (stats == null) {
                            stats = new long[3]; // [totalNs, count, maxNs]
                            perOpTimingNs.put(opName, stats);
                        }
                        stats[0] += slotNs;
                        stats[1]++;
                        if (slotNs > stats[2]) stats[2] = slotNs;
                        // Time bucket distribution
                        if (slotMs < 1.0) timingBucketSub1ms++;
                        else if (slotMs < 10.0) timingBucket1to10ms++;
                        else if (slotMs < 100.0) timingBucket10to100ms++;
                        else timingBucketOver100ms++;
                        // Log slow ops (>10ms)
                        if (slotMs > 10.0) {
                            log.info("SLOW OP slot {}: {} took {}ms", stepIdx, opName,
                                    String.format("%.1f", slotMs));
                        }
                    }

                    // Post-slot CUDA error check: detect sticky errors after the slot that
                    // caused them. lastErrorCode() is a JNI call (~2us each). With 1962 ops
                    // per vision frame, checking every op adds ~4ms/frame. Check every N ops
                    // (ERROR_CHECK_INTERVAL, default 50) to reduce JNI overhead by ~50x while
                    // still catching errors within 50 ops of occurrence. Always check the
                    // last slot to catch errors before output claiming.
                    if (ERROR_CHECK_INTERVAL <= 1
                            || stepIdx % ERROR_CHECK_INTERVAL == 0
                            || stepIdx == slots.length - 1) {
                        int postSlotErr = nativeOps.lastErrorCode();
                        if (postSlotErr != 0) {
                            String postSlotMsg = nativeOps.lastErrorMessage();
                            log.error("CUDA error {} after slot {} ({}): {}",
                                    postSlotErr, stepIdx, slot.getOpName(), postSlotMsg);
                            nativeOps.clearLastError();
                            // Don't throw immediately — let the DSP continue so we get more diagnostic info.
                            // The error is sticky on the GPU side, so subsequent slots will also fail.
                        }
                    }
                } catch (Exception e) {
                    log.error("Error executing slot {} ({}): {}", stepIdx, slot.getOpName(), e.getMessage());
                    // Log input details for diagnosis
                    String[] inVarNames = slot.getInputVarNames();
                    int[] inSrcIdx = slot.getInputSourceIndices();
                    byte[] inSrcTypes = slot.getInputSourceTypes();
                    if (inVarNames != null) {
                        for (int ii = 0; ii < inVarNames.length; ii++) {
                            String srcDesc;
                            if (inSrcTypes != null && ii < inSrcTypes.length) {
                                switch (inSrcTypes[ii]) {
                                    case DynamicShapeSlot.SOURCE_CONSTANT: srcDesc = "CONST"; break;
                                    case DynamicShapeSlot.SOURCE_VARIABLE: srcDesc = "VAR"; break;
                                    case DynamicShapeSlot.SOURCE_PLACEHOLDER: srcDesc = "PH"; break;
                                    case DynamicShapeSlot.SOURCE_OP_OUTPUT: srcDesc = "OP[" + inSrcIdx[ii] + "]"; break;
                                    default: srcDesc = "?"; break;
                                }
                            } else {
                                srcDesc = "idx=" + (inSrcIdx != null ? inSrcIdx[ii] : "?");
                            }
                            INDArray inArr = (inSrcIdx != null && inSrcIdx[ii] >= 0)
                                    ? outputSlots[inSrcIdx[ii]]
                                    : (inSrcIdx != null ? externalInputs[-(inSrcIdx[ii] + 1)] : null);
                            String shapeStr = inArr != null
                                    ? java.util.Arrays.toString(inArr.shape()) + " " + inArr.dataType()
                                    : "null";
                            log.error("  Input[{}] '{}' src={} shape={}", ii, inVarNames[ii], srcDesc, shapeStr);
                        }
                    }
                    // Clear the slot cache on failure — a failed execution may leave
                    // cached arrays with corrupted shape info or stale GPU pointers.
                    // Without this, every subsequent execute() hits the same stale cache.
                    closeSlotArrayCache();
                    throw new RuntimeException("DynamicShapePlan execution failed at step " + stepIdx +
                            " (" + slot.getOpName() + ")", e);
                }

                ctx.purgeForReuse();
                ctxPool.offerFirst(ctx);

                // NOTE: Workspace scope-out/scope-in (reset) is now handled inside
                // executeSlot() after op execution (line ~1690). The previous duplicate
                // reset here added 2 unnecessary JNI calls per slot (~4000 JNI calls
                // per 1962-op vision encoder frame). The executeSlot reset is sufficient
                // because no workspace allocations occur between executeSlot return and
                // the next executeSlot call — only Java-side release bookkeeping.

                // Mark dead slots for deferred close. Don't close now because:
                // (1) GPU kernels may still be using the buffer on the execution stream
                // (2) View arrays share GPU pointers — dedup is only safe post-commit
                long tRelease0 = TIMING_ENABLED ? System.nanoTime() : 0;
                int[] toRelease = plan.getReleaseAtStep()[stepIdx];
                for (int slotIdx : toRelease) {
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null && liveSlots.get(slotIdx)) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed()) {
                            // Check view status BEFORE un-poisoning to protect weight buffers
                            boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[slotIdx];
                            boolean isWeightBuffer = protectedWeightBuffers != null && protectedWeightBuffers.containsKey(buf);
                            if (!isViewSlot && !isWeightBuffer && buf.isConstant()) {
                                // Un-poison constant-poisoned intermediates so they can be freed.
                                // Skip view slots (share input's buffer) and weight buffers.
                                buf.setConstant(false);
                            }
                            if (isViewSlot) {
                                // View producer — the buffer belongs to the input (C++ made
                                // this a view of the input's GPU memory). Don't close or cache
                                // it; the input slot manages its own buffer lifecycle.
                            } else if (slotArrayCache != null) {
                                // Non-view producer — cache for O(1) reuse on next execute().
                                // Don't gate on closeable(): oversized buffers (growth factor
                                // > 1.0) have data().length() > length() → closeable()=false,
                                // but these are OWNED arrays safe to cache and reuse.
                                INDArray prev = slotArrayCache[slotIdx];
                                if (prev != null && !prev.wasClosed()) {
                                    DataBuffer pbuf = prev.data();
                                    if (pbuf != null && !pbuf.wasClosed()) {
                                        boolean prevIsWeight = protectedWeightBuffers != null && protectedWeightBuffers.containsKey(pbuf);
                                        if (!prevIsWeight && pbuf.isConstant()) pbuf.setConstant(false);
                                        if (!prevIsWeight) pendingClose.add(pbuf);
                                    }
                                }
                                slotArrayCache[slotIdx] = arr;
                            } else {
                                pendingClose.add(buf);
                            }
                            if (!isViewSlot) {
                                pendingCloseBytes += buf.length() * buf.getElementSize();
                                if (TIMING_ENABLED) pendingCloseCount++;
                            }
                        }
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                // Flush dead buffers to reclaim GPU memory mid-execution.
                // Two triggers: op-count interval (every 100 ops) AND byte threshold (256MB).
                // The byte trigger prevents multi-GB accumulation between op-count boundaries
                // (e.g., 95 ops × 48MB = 4.5GB trapped before next interval flush).
                if (!pendingClose.isEmpty() && (
                        (stepIdx > 0 && stepIdx % RELEASE_FLUSH_INTERVAL == 0) ||
                        pendingCloseBytes >= FLUSH_BYTE_THRESHOLD)) {
                    flushPendingClose(nativeOps, execStream);
                    pendingCloseBytes = 0;
                }

                if (TIMING_ENABLED) timingReleaseNs += System.nanoTime() - tRelease0;
            }

            // Claim output arrays directly from slots — no dup, no D2H sync.
            // The caller owns the returned arrays and must close them when done.
            // This avoids 61 per-output dup + commit + getFloat calls per decode step
            // (~3.5x speedup for autoregressive decoding with KV cache).
            Nd4j.getExecutioner().commit();
            {
                Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
                int viewFlagFixCount = 0;
                int lengthViewCount = 0;
                for (Map.Entry<String, Integer> entry : outputMap.entrySet()) {
                    int slotIdx = entry.getValue();
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null) {
                        // Fix IS_VIEW flag set by C++ ops. C++ ops (concat, reshape, etc.)
                        // sometimes set the IS_VIEW flag in shapeInfo even for outputs that
                        // OWN their GPU memory (not actual views). This makes isView()=true,
                        // which causes arr.close() to skip freeing GPU memory → 30MB/step leak.
                        // Only clear the flag if the array actually owns its buffer (not a
                        // true view-producer slot) and length()==data().length().
                        if (arr.isView() && arr.data() != null && !arr.data().wasClosed()) {
                            boolean isViewProducer = slotIsViewProducer != null && slotIsViewProducer[slotIdx];
                            long arrLen = arr.length();
                            long dataLen = arr.data().length();
                            boolean lengthView = arrLen < dataLen;
                            boolean flagView = ArrayOptionsHelper.isView(arr.shapeInfoJava());
                            if (!isViewProducer && !lengthView && flagView) {
                                // C++ set the IS_VIEW flag but this is not a true view.
                                // Clear the flag so close() works correctly.
                                long[] shapeInfo = arr.shapeInfoJava();
                                long options = shapeInfo[shapeInfo.length - 3];
                                options &= ~ArrayOptionsHelper.IS_VIEW;
                                shapeInfo[shapeInfo.length - 3] = options;
                                viewFlagFixCount++;
                            } else if (lengthView) {
                                lengthViewCount++;
                            }
                        }
                        results.put(entry.getKey(), arr);
                        // Remove from slots so closePendingBuffers won't free it
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                if (viewFlagFixCount > 0 || lengthViewCount > 0) {
                    log.info("  Output view fix: {} flag-only views fixed, {} length-based views (unfixable)",
                            viewFlagFixCount, lengthViewCount);
                }
                // Log first few length-based view details to understand why data().length() > length()
                if (lengthViewCount > 0) {
                    int logged = 0;
                    for (Map.Entry<String, Integer> entry2 : outputMap.entrySet()) {
                        INDArray arr2 = results.get(entry2.getKey());
                        if (arr2 != null && arr2.data() != null && arr2.length() < arr2.data().length()) {
                            if (logged < 3) {
                                log.info("    View detail: {} arrLen={} dataLen={} ratio={:.2f} shape={} dtype={}",
                                        entry2.getKey(), arr2.length(), arr2.data().length(),
                                        (double) arr2.data().length() / arr2.length(),
                                        java.util.Arrays.toString(arr2.shape()), arr2.dataType());
                                logged++;
                            }
                        }
                    }
                }
            }

            // Collect remaining live slots (non-output intermediates) for cleanup.
            // Skip view-producer slots — their buffers belong to the input.
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && liveSlots.get(i)) {
                    boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[i];
                    if (!isViewSlot) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            pendingClose.add(buf);
                        }
                    }
                    outputSlots[i] = null;
                    liveSlots.clear(i);
                }
            }

            // Now close ALL pending buffers in one batch. commit() syncs the execution stream
            // first, ensuring all GPU kernels are done before we free their buffers.
            // GPU address dedup prevents double-free for view arrays sharing parent buffers.
            // No new allocations happen between closes, so address reuse is impossible.
            closePendingBuffers(nativeOps, execStream);

            // Evict oversized slot cache after prefill steps. During autoregressive decoding,
            // step 0 (prefill) allocates intermediates for full sequence length (e.g., seq=679),
            // while subsequent decode steps use seq=1. Cached arrays from prefill hold GBs of
            // GPU memory that won't be reused. Clear the cache if total cached bytes > 512MB
            // so subsequent threads can create CUDA contexts/streams.
            evictOversizedSlotCache(nativeOps, execStream);

            // After first successful execution, view-producer detection is complete.
            // Subsequent executions skip pre/post GPU address comparison (saves ~2 JNI
            // calls per output per slot — significant for 1962-op vision encoder).
            if (!viewProducerDetectionDone) {
                viewProducerDetectionDone = true;
                if (TIMING_ENABLED) {
                    int viewCount = 0;
                    if (slotIsViewProducer != null) {
                        for (boolean b : slotIsViewProducer) if (b) viewCount++;
                    }
                    log.info("  View-producer detection complete: {} view-producer slots", viewCount);
                }
            }

            if (TIMING_ENABLED) {
                printTimingSummary(slots.length, localPool);
            }

            // Diagnostic: dump first few values of each output to compare with native executor
            if (Boolean.getBoolean(ND4JSystemProperties.DSP_JAVA_DUMP_OUTPUTS)) {
                for (Map.Entry<String, INDArray> entry : results.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.length() > 0) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("JAVA_OUT ").append(name).append(" shape=").append(java.util.Arrays.toString(arr.shape()));
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
            // Restore autoGc window so DeallocatorService resumes normal GC for non-DSP code
            Nd4j.getMemoryManager().setAutoGcWindow(savedAutoGcWindow);
            // Safety: null out remaining slots without closing (already handled above)
            if (outputSlots != null) {
                Arrays.fill(outputSlots, null);
            }
            if (liveSlots != null) {
                liveSlots.clear();
            }
        }
    }

    /**
     * Flush pending dead buffers mid-execution to reclaim GPU memory.
     * Syncs the execution stream, frees dead buffers on that same stream via
     * dbFreeBuffersOnStream(), then trims the pool on the same stream so the
     * pool can reuse the freed memory for subsequent allocations.
     */
    private void flushPendingClose(NativeOps nativeOps, Pointer execStream) {
        // Merge previously deferred buffers (parents whose GPU memory was still live).
        if (!deferredClose.isEmpty()) {
            pendingClose.addAll(deferredClose);
            deferredClose.clear();
        }
        if (pendingClose.isEmpty()) return;

        // Re-fetch a fresh stream pointer. The cached execStream may be stale — C++
        // ContextBuffers::release() can free the underlying cudaStream_t during device
        // context switches in executeSlot(). Dereferencing a freed stream → SIGSEGV.
        Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
        if (freshStream == null) freshStream = execStream; // fallback to cached

        // Sync execution stream so all GPU kernels using these buffers have completed.
        Nd4j.getExecutioner().commit();

        // Fast path: for decode steps with small pending close (< 10MB), skip the
        // expensive collectLiveGpuAddresses() call. With seq_len=1 arrays, aliasing
        // is extremely unlikely and the overhead of iterating all live slots + JNI
        // calls for each slot's GPU address outweighs the risk.
        long estimatedBytes = 0;
        for (DataBuffer buf : pendingClose) {
            if (buf != null && !buf.wasClosed()) {
                estimatedBytes += buf.length() * buf.getElementSize();
            }
        }
        boolean fastPath = estimatedBytes < FAST_CLOSE_THRESHOLD;

        long[] liveGpuAddresses;
        if (fastPath) {
            // Skip live address collection — no range check, just identity + ODB dedup
            liveGpuAddresses = null;
        } else {
            // Build sorted array of GPU addresses from live slots. Owner buffers whose
            // allocation range overlaps with any live address are deferred — a view in a
            // live slot still needs the parent's GPU memory. Range check catches offset
            // views (e.g., strided_slice) that exact-match would miss.
            liveGpuAddresses = collectLiveGpuAddresses(nativeOps);
        }
        int[] stats = freePendingBuffers(nativeOps, freshStream, liveGpuAddresses);
        // Trim the pool on the execution stream so freed memory is immediately reusable.
        // Without this, cudaFreeAsync enqueues frees but the pool can't reuse until synced.
        if (freshStream != null) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            nativeOps.trimMemoryPoolOnStream(currentDevice, freshStream);
            // Also trim device 1 if multi-GPU. Cross-device frees go to device 1's
            // default stream (see dbFreeBuffersOnStream cross-device path).
            // Use trimMemoryPoolOnStream with null stream to sync stream 0 on the
            // target device before trimming — that's where the cross-device frees land.
            if (cachedNumDevices > 1) {
                for (int d = 0; d < cachedNumDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPool(d);
                    }
                }
            }
        }
        if (!deferredClose.isEmpty()) {
            log.debug("  Mid-exec flush: freed {}/{} buffers ({}MB), deferred {} (live views), total freed: {}MB",
                    stats[0], stats[1], stats[2], deferredClose.size(), totalFlushedBytes / (1024 * 1024));
        } else {
            log.debug("  Mid-exec flush: freed {}/{} buffers ({}MB), total freed so far: {}MB",
                    stats[0], stats[1], stats[2], totalFlushedBytes / (1024 * 1024));
        }
        pendingClose.clear();
    }

    /**
     * Close all pending DataBuffers after execution completes (final flush).
     * Sync the execution stream first, then close with GPU address dedup.
     * Frees on the execution stream so pool memory is immediately reusable.
     */
    private void closePendingBuffers(NativeOps nativeOps, Pointer execStream) {
        // Merge any remaining deferred buffers from mid-execution flushes.
        // At final close, no slots are live so all deferred buffers can be freed.
        if (!deferredClose.isEmpty()) {
            pendingClose.addAll(deferredClose);
            deferredClose.clear();
        }

        if (pendingClose.isEmpty() && totalFlushedCount == 0) return;

        // Re-fetch a fresh stream pointer (same reason as flushPendingClose).
        Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
        if (freshStream == null) freshStream = execStream;

        if (!pendingClose.isEmpty()) {
            Nd4j.getExecutioner().commit();
            freePendingBuffers(nativeOps, freshStream, null);
        }

        // Trim the pool so cudaFreeAsync-enqueued frees are processed and memory is
        // returned to the driver. Without this, poolUsed grows across execute() calls
        // because the frees are stream-ordered but never synced before the next allocation.
        if (freshStream != null) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            nativeOps.trimMemoryPoolOnStream(currentDevice, freshStream);
            // Trim all other devices — cross-device frees went to their default streams.
            // Use trimMemoryPoolOnStream with null to sync stream 0 on target device.
            if (cachedNumDevices > 1) {
                for (int d = 0; d < cachedNumDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPool(d);
                    }
                }
            }
        }

        log.debug("  Deferred close: {}/{} buffers ({}MB)",
                totalFlushedCount, totalFlushedCount, totalFlushedBytes / (1024 * 1024));
        pendingClose.clear();
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
            // Never touch weight buffers — un-poisoning them corrupts model weights
            if (protectedWeightBuffers != null && protectedWeightBuffers.containsKey(buf)) continue;
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
     * Collect GPU addresses from all live slots. Used during mid-execution flushes
     * to prevent freeing parent buffers whose GPU memory is still referenced by
     * view arrays in live slots.
     *
     * Returns a sorted array of GPU addresses for range-based overlap checks.
     * Views created by C++ ops (reshape, slice, etc.) may have non-zero offsets
     * from their parent allocation, so exact-match address checks are insufficient.
     */
    private long[] collectLiveGpuAddresses(NativeOps nativeOps) {
        int count = 0;
        long[] addresses = new long[liveSlots.cardinality()];
        for (int i = liveSlots.nextSetBit(0); i >= 0; i = liveSlots.nextSetBit(i + 1)) {
            INDArray arr = outputSlots[i];
            if (arr == null) continue;
            DataBuffer buf = arr.data();
            if (buf == null || buf.wasClosed()) continue;
            OpaqueDataBuffer odb = buf.opaqueBuffer();
            if (odb == null || odb.isNull()) continue;
            Pointer special = nativeOps.dbSpecialBuffer(odb);
            if (special != null && special.address() != 0) {
                addresses[count++] = special.address();
            }
        }
        long[] result = count == addresses.length ? addresses : Arrays.copyOf(addresses, count);
        Arrays.sort(result);
        return result;
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

    private void resolveExternalInputs(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        String[] keys = plan.getExternalInputKeys();
        byte[] sourceTypes = plan.getExternalInputSourceTypes();
        for (int i = 0; i < keys.length; i++) {
            byte srcType = (sourceTypes != null && i < sourceTypes.length) ? sourceTypes[i] : -1;

            // Constants and variables: resolve once and cache across execute() calls.
            // They don't change between steps in autoregressive decoding.
            if (externalConstantsResolved &&
                    (srcType == DynamicShapeSlot.SOURCE_CONSTANT || srcType == DynamicShapeSlot.SOURCE_VARIABLE)) {
                // Already resolved and cached — skip re-resolution.
                // Still need ensureLocation for variables (may be modified between calls).
                if (srcType == DynamicShapeSlot.SOURCE_VARIABLE && externalInputs[i] != null) {
                    Nd4j.getAffinityManager().ensureLocation(externalInputs[i], AffinityManager.Location.DEVICE);
                }
                continue;
            }

            String varName = keys[i];
            INDArray arr = null;

            // Try placeholder first (always re-resolve — values change each step)
            if (placeholderArrays != null) {
                arr = placeholderArrays.get(varName);
            }

            // Then try constant/variable from SameDiff
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null &&
                        (var.getVariableType() == VariableType.CONSTANT ||
                                var.getVariableType() == VariableType.VARIABLE)) {
                    arr = var.getArr();
                }
            }

            externalInputs[i] = arr;

            // Unconditional HOST→DEVICE sync for ALL external inputs.
            // Without this, arrays created/modified on host (e.g., attention bias,
            // position_ids, attention_mask) stay host-only and C++ reads zeros.
            if (arr != null) {
                Nd4j.getAffinityManager().ensureLocation(arr, AffinityManager.Location.DEVICE);
            }
        }
        externalConstantsResolved = true;
    }

    private void executeSlot(DynamicShapeSlot slot, OpContext ctx, NativeOps nativeOps,
                             LocalBufferPool localPool, Pointer execStream, int cachedDeviceId) {
        DifferentialFunction fn = slot.getOp();

        // Step 0: Device placement. Use the cached device ID from execute() instead of
        // querying getDeviceForCurrentThread() per-slot (~2us JNI overhead × 1962 slots).
        // The execute() loop restores the device after each executeSlot() via the finally
        // block, so cachedDeviceId is always accurate at entry.
        int previousDeviceId = cachedDeviceId;
        int targetDevice = slot.getTargetDeviceId();
        // Resolve unset device placement (-1) to the cached thread device.
        // Then check if the resolved device has enough free memory for C++ ContextBuffers
        // workspace (16MB reduction + 16MB allocation + streams). Without this check,
        // ops run on device 0 (thread affinity) even when it has only ~50MB free,
        // causing ContextBuffers to failover to device 1 and create cross-device
        // stream/data mismatches that crash with error 700/900.
        if (targetDevice < 0) {
            targetDevice = cachedDeviceId;
        }
        // If the planned device hit an unrecoverable CUDA error (e.g., OOM cascade),
        // redirect to the first non-failed device. When device 0 fails, we redirect
        // to device 1 (or the next available). When device 1+ fails, redirect to 0.
        if (failedDevices != null && failedDevices.contains(targetDevice)) {
            int redirectDevice = 0;
            for (int d = 0; d < cachedNumDevices; d++) {
                if (!failedDevices.contains(d)) {
                    redirectDevice = d;
                    break;
                }
            }
            targetDevice = redirectDevice;
        }
        // Free memory gate: if target device can't afford ContextBuffers workspace (16MB)
        // plus some execution headroom, route to the device with most free memory.
        // This prevents the ContextBuffers flip-flop where it initializes via failover
        // on a different device than the op data, causing cross-device crashes.
        if (cachedNumDevices > 1) {
            long freeMem = nativeOps.getDeviceFreeMemory(targetDevice);
            long MIN_WORKSPACE_HEADROOM = 128L * 1024 * 1024; // 128MB for workspace + intermediates
            if (freeMem < MIN_WORKSPACE_HEADROOM) {
                long bestFree = -1;
                int bestDevice = targetDevice;
                for (int d = 0; d < cachedNumDevices; d++) {
                    if (failedDevices != null && failedDevices.contains(d)) continue;
                    long dfree = nativeOps.getDeviceFreeMemory(d);
                    if (dfree > bestFree) {
                        bestFree = dfree;
                        bestDevice = d;
                    }
                }
                if (bestDevice != targetDevice) {
                    log.debug("DSP executeSlot: device {} has only {}MB free (need {}MB), routing to device {} ({}MB free) for op {}",
                            targetDevice, freeMem / (1024*1024), MIN_WORKSPACE_HEADROOM / (1024*1024),
                            bestDevice, bestFree / (1024*1024), slot.getOpName());
                    targetDevice = bestDevice;
                }
            }
        }
        boolean deviceSwitchOccurred = false;
        if (previousDeviceId != targetDevice) {
            deviceSwitchOccurred = true;
            DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "slot-device-placement");
            // Re-fetch execution stream for the target device. The execStream passed
            // in was cached from the original device's launch context — using it for
            // cudaMemsetAsync on a different device's memory fails for non-P2P GPUs.
            Pointer deviceStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
            if (deviceStream != null) execStream = deviceStream;
        }

        try {
        // Track replicated input copies for explicit close after execution.
        // replicateToDevice() creates new arrays that leak forever without explicit close
        // (GC cleanup is broken for GPU buffers). Both Step 1b and Step 4c create copies.
        List<DataBuffer> replicatedInputBuffers = null;

        // Step 1: Wire inputs
        long tWire0 = TIMING_ENABLED ? System.nanoTime() : 0;
        int[] inputSourceIndices = slot.getInputSourceIndices();
        INDArray[] inputArrays = slot.getInputArraysBuffer();

        for (int i = 0; i < inputSourceIndices.length; i++) {
            int srcIdx = inputSourceIndices[i];
            if (srcIdx >= 0) {
                inputArrays[i] = outputSlots[srcIdx];
            } else {
                int extIdx = -(srcIdx + 1);
                inputArrays[i] = externalInputs[extIdx];
            }

            if (inputArrays[i] == null) {
                throw new IllegalStateException("Null input at index " + i + " for op " + slot.getOpName() +
                        ", input var: " + slot.getInputVarNames()[i]);
            }
        }
        // Step 1b: Migrate inputs to target device if cross-device.
        // Use dbDeviceId (C++ DataBuffer._deviceId) instead of getDeviceForArray (Java-side)
        // because Java doesn't know about C++ allocation failover — an array may have been
        // intended for device 1 but allocated on device 0 by CudaMemoryPool::allocateFailover.
        //
        // For VIEW inputs: pre-dup to contiguous before replication and track the intermediate
        // buffer for freeing on the execution stream. replicateToDevice's internal dup() also
        // handles views, but its close() frees via dbClose() → default stream (nullptr). DSP
        // trims only the execution stream, so default-stream frees accumulate without being
        // processed → ~30MB/step leak. By pre-duping here and tracking in replicatedInputBuffers,
        // the intermediate is freed on the execution stream by freePendingBuffers → trimmed properly.
        if (targetDevice >= 0) {
            boolean migrated = false;
            for (int i = 0; i < inputArrays.length; i++) {
                INDArray input = inputArrays[i];
                if (input != null && !input.isEmpty() && input.data() != null) {
                    // Use cached device IDs instead of JNI dbDeviceId() per input.
                    // Op outputs use outputSlotDeviceIds[], externals use externalInputDeviceIds[].
                    // Falls back to JNI only when cache is -1 (unknown), which happens on
                    // failover or first access of external inputs.
                    int srcIdx2 = inputSourceIndices[i];
                    int inputDevice = -1;
                    if (srcIdx2 >= 0) {
                        inputDevice = outputSlotDeviceIds[srcIdx2];
                    } else {
                        int extIdx2 = -(srcIdx2 + 1);
                        inputDevice = externalInputDeviceIds[extIdx2];
                    }
                    // Fall back to JNI if cached value is unknown (-1)
                    if (inputDevice < 0) {
                        OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                        if (inputOdb != null && !inputOdb.isNull()) {
                            inputDevice = nativeOps.dbDeviceId(inputOdb);
                        }
                        // Cache the resolved value for future lookups
                        if (inputDevice >= 0 && srcIdx2 < 0) {
                            int extIdx2 = -(srcIdx2 + 1);
                            externalInputDeviceIds[extIdx2] = inputDevice;
                        }
                    }
                    if (inputDevice >= 0 && inputDevice != targetDevice) {
                        // Check constant replica cache first. Model weights (constants) don't
                        // change between decode steps — cache their replicas to avoid re-copying
                        // 194MB of weights from device 0 to device 1 on every single token.
                        // CRITICAL: Do NOT cache PLACEHOLDER arrays. setCloseable(false) poisons
                        // the DataBuffer.isConstant() flag (via setConstant(true)), causing
                        // placeholders like attention_mask to appear constant. If cached, stale
                        // replicas from the previous step are reused, causing shape mismatches
                        // (e.g., attention_mask [1,680] cached from step N used at step N+1
                        // where the correct shape is [1,681]).
                        boolean isConstant = input.data() != null && input.data().isConstant();
                        boolean isExternal = srcIdx2 < 0;
                        // Check slot's inputSourceTypes — placeholders are NEVER truly constant
                        byte[] srcTypes = slot.getInputSourceTypes();
                        boolean isPlaceholder = (srcTypes != null && i < srcTypes.length &&
                                srcTypes[i] == DynamicShapeSlot.SOURCE_PLACEHOLDER);
                        boolean isTrulyConstant = isConstant && !isPlaceholder;
                        int cacheKey = isExternal ? ((-(srcIdx2 + 1)) << 16) | targetDevice : -1;

                        INDArray cachedReplica = null;
                        if (isTrulyConstant && isExternal && constantReplicaCache != null) {
                            cachedReplica = constantReplicaCache.get(cacheKey);
                            if (cachedReplica != null && !cachedReplica.wasClosed()) {
                                inputArrays[i] = cachedReplica;
                                if (TIMING_ENABLED) replicaCacheHits++;
                                migrated = true;
                                MultiGpuTracer.traceInputMigration(-1, i, inputDevice, targetDevice,
                                        input.length() * input.data().getElementSize(),
                                        input.isView(), true, true);
                                continue;
                            }
                        }

                        MultiGpuTracer.traceInputMigration(-1, i, inputDevice, targetDevice,
                                input.length() * input.data().getElementSize(),
                                input.isView(), isConstant, false);

                        // Pre-dup views to contiguous on the source device. This avoids
                        // replicateToDevice's internal dup() which frees the intermediate
                        // on the default stream (not trimmed by DSP's execution stream trim).
                        // CRITICAL: dup() must run on the SOURCE device — if the current
                        // thread is on a different device (e.g., DeviceWorker[1] on device 1,
                        // but input data on device 0), dup() would try a direct GPU copy
                        // between non-P2P devices → error 700 (illegal memory access).
                        INDArray inputToReplicate = input;
                        if (input.isView()) {
                            DeviceMemoryManager.getInstance().switchDevice(inputDevice, "DSP.executeSlot", "view-dup-source");
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                inputToReplicate = input.dup(input.ordering());
                            }
                            DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "view-dup-restore");
                            // Track contiguous intermediate for freeing on execution stream
                            DataBuffer dupBuf = inputToReplicate.data();
                            if (dupBuf != null && !dupBuf.isConstant()) {
                                if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                replicatedInputBuffers.add(dupBuf);
                            }
                        }
                        INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                        inputArrays[i] = replica;

                        // Cache constant replicas for reuse across decode steps
                        // Only cache truly constant arrays (not placeholders poisoned by setCloseable)
                        if (isTrulyConstant && isExternal) {
                            if (constantReplicaCache == null) constantReplicaCache = new HashMap<>();
                            constantReplicaCache.put(cacheKey, replica);
                            // Don't track in replicatedInputBuffers — cached replicas
                            // persist across execute() calls, freed when executor closes
                        } else {
                            // Track non-constant replica for explicit close after execution
                            DataBuffer replicaBuf = replica.data();
                            if (replicaBuf != null && !replicaBuf.isConstant()) {
                                if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                replicatedInputBuffers.add(replicaBuf);
                            }
                        }
                        replicaCount++;
                        long rBytes = replica.length() * replica.data().getElementSize();
                        replicaBytes += rBytes;
                        if (targetDevice == 0) { replicaToDev0Count++; replicaToDev0Bytes += rBytes; }
                        else if (targetDevice == 1) { replicaToDev1Count++; replicaToDev1Bytes += rBytes; }
                        migrated = true;
                    }
                }
            }
            if (migrated) {
                log.trace("Migrated inputs to device {} for op {}", targetDevice, slot.getOpName());
            }
        }
        ctx.setInputArrays(inputArrays);
        if (TIMING_ENABLED) timingWireInputsNs += System.nanoTime() - tWire0;

        // Step 2: Sync INT/LONG inputs if needed
        long tSync0 = TIMING_ENABLED ? System.nanoTime() : 0;
        if (slot.isNeedsIntLongSync()) {
            syncIntLongInputs(inputArrays, slot.isDataDependent(), nativeOps);
        }
        if (TIMING_ENABLED) timingSyncNs += System.nanoTime() - tSync0;

        // Step 3: Compute output shapes
        long tShape0 = TIMING_ENABLED ? System.nanoTime() : 0;
        List<DataBuffer> outShapes = getOrComputeShapes(slot, ctx, fn, inputArrays, nativeOps);
        if (outShapes == null || outShapes.isEmpty()) {
            throw new IllegalStateException("No output shapes for op " + slot.getOpName());
        }
        // Debug: log strided_slice shape details to diagnose DSP shape mismatch
        if ("strided_slice".equals(slot.getOpName())) {
            StringBuilder sb = new StringBuilder();
            sb.append("DSP strided_slice debug [step=").append(slot.getStepIndex()).append("]:");
            sb.append(" iArgs=").append(java.util.Arrays.toString(slot.getIArgs()));
            sb.append(" numInputs=").append(inputArrays.length);
            for (int di = 0; di < inputArrays.length; di++) {
                INDArray in = inputArrays[di];
                sb.append(" in[").append(di).append("]=");
                if (in == null) { sb.append("null"); }
                else {
                    sb.append(java.util.Arrays.toString(in.shape())).append("/").append(in.dataType());
                    if (in.length() <= 8) {
                        // Print small tensor values (begin/end/strides)
                        Nd4j.getExecutioner().commit();
                        nativeOps.dbForceSyncToPrimary(in.data().opaqueBuffer());
                        sb.append("=").append(in);
                    }
                }
            }
            for (int di = 0; di < outShapes.size(); di++) {
                long[] si = outShapes.get(di).asLong();
                sb.append(" outShape[").append(di).append("]=").append(java.util.Arrays.toString(Shape.shape(si)));
            }
            log.info(sb.toString());
        }
        if (TIMING_ENABLED) timingShapeNs += System.nanoTime() - tShape0;

        // Step 4: Allocate outputs
        long tAlloc0 = TIMING_ENABLED ? System.nanoTime() : 0;
        int[] outputSlotIndices = slot.getOutputSlotIndices();
        INDArray[] outputArrays = new INDArray[outShapes.size()];

        for (int i = 0; i < outShapes.size(); i++) {
            DataBuffer shapeBuffer = outShapes.get(i);
            long[] shapeInfo = shapeBuffer.asLong();
            DataType dt = Shape.dataType(shapeInfo);
            long[] actualShape = Shape.shape(shapeInfo);

            INDArray out = null;
            int slotIdx = (i < outputSlotIndices.length) ? outputSlotIndices[i] : -1;

            // For view-capable INTERMEDIATE CustomOp slots, use an empty placeholder.
            // C++ will replace it with a view of the input's buffer (zero data copy).
            // Detection sources (OR'd):
            //   - Compile-time: slot.isViewCapableOp() — reshape, expand_dims, squeeze, permute
            //   - Runtime: slotIsViewProducer[slotIdx] — detected after first execution
            // We skip OUTPUT slots — the caller needs the data, so we must allocate
            // even though C++ will replace the buffer.
            // We also skip non-CustomOp (legacy) ops because the Java executor validates
            // X.length == Z.length BEFORE calling C++.
            boolean isViewSlotCompileTime = slot.isViewCapableOp();
            boolean isViewSlotRuntime = slotIdx >= 0 && slotIsViewProducer != null && slotIsViewProducer[slotIdx];
            if (slotIdx >= 0 && (isViewSlotCompileTime || isViewSlotRuntime)
                    && !outputSlotSet.get(slotIdx) && slot.isCustomOp()) {
                out = Nd4j.empty(dt);
                outputArrays[i] = out;
                outputSlots[slotIdx] = out;
                liveSlots.set(slotIdx);
                if (TIMING_ENABLED) timingViewSkips++;
                // Clear any stale cached array for view producer slots to prevent
                // use-after-free when the slot is reused in future executions.
                if (slotArrayCache != null && slotArrayCache[slotIdx] != null) {
                    INDArray stale = slotArrayCache[slotIdx];
                    DataBuffer sbuf = stale.data();
                    if (sbuf != null && !sbuf.wasClosed() && !sbuf.isConstant()) {
                        pendingClose.add(sbuf);
                    }
                    slotArrayCache[slotIdx] = null;
                }
                // Mark as view producer immediately (compile-time knowledge)
                if (isViewSlotCompileTime && slotIsViewProducer != null) {
                    slotIsViewProducer[slotIdx] = true;
                }
                continue;
            }

            if (Shape.isEmpty(shapeInfo) || numElements(actualShape) == 0) {
                out = Nd4j.emptyWithShape(actualShape, dt);
            } else {
                // Try slot-indexed cache first (O(1) lookup, no TreeMap)
                if (slotIdx >= 0 && slotArrayCache != null) {
                    INDArray cached = slotArrayCache[slotIdx];
                    if (cached != null && !cached.wasClosed()) {
                        // Validate cached array's native shape info before reuse.
                        // A C++ op buffer overrun can corrupt constant shape info on the host heap.
                        // The Java-side jvmShapeInfo (long[]) is immune (JVM-managed memory).
                        // Detect corruption here to avoid propagating it through execution.
                        DataBuffer cachedShapeInfo = cached.shapeInfoDataBuffer();
                        if (cachedShapeInfo != null && cachedShapeInfo.length() > 0) {
                            long cachedRank = cachedShapeInfo.getLong(0);
                            if (cachedRank < 0 || cachedRank > 32) {
                                log.error("Slot cache shape info corruption at slot {} ({}): " +
                                        "native rank={} (0x{}), shapeInfo constant={}, " +
                                        "nativeAddr=0x{}. Evicting from cache.",
                                        slotIdx, slot.getOpName(),
                                        cachedRank, Long.toHexString(cachedRank),
                                        cachedShapeInfo.isConstant(),
                                        Long.toHexString(cachedShapeInfo.pointer().address()));
                                // Don't reuse corrupted cache entry — force fresh allocation.
                                DataBuffer cbufCorrupt = cached.data();
                                if (cbufCorrupt != null && !cbufCorrupt.wasClosed() && !cbufCorrupt.isConstant()) {
                                    pendingClose.add(cbufCorrupt);
                                }
                                slotArrayCache[slotIdx] = null;
                                cached = null;
                            }
                        }
                    }
                    if (cached != null && !cached.wasClosed()) {
                        DataBuffer cbuf = cached.data();
                        // NOTE: Don't gate on cbuf.closeable() here. Oversized buffers
                        // (from growth factor > 1.0) have data().length() > length() which
                        // makes closeable()=false. But these are OWNED arrays that we
                        // allocated — safe to reshape and reuse.
                        if (cbuf != null && !cbuf.wasClosed()
                                && cached.dataType() == dt
                                && cbuf.length() >= numElements(actualShape)) {
                            // Check device locality for multi-GPU. If the cached array is on
                            // a different device, DON'T use it — allocate fresh on the target
                            // device instead. Using cross-device memory causes illegal memory
                            // access for non-P2P GPUs, and relying on Step 4c to "detect" it
                            // as a failover creates unnecessary cross-device migrations that
                            // compound memory pressure and corrupt CUDA stream state.
                            boolean wrongDevice = false;
                            if (targetDevice >= 0) {
                                OpaqueDataBuffer cachedOdb = cbuf.opaqueBuffer();
                                if (cachedOdb != null && !cachedOdb.isNull()) {
                                    int cachedDevice = nativeOps.dbDeviceId(cachedOdb);
                                    if (cachedDevice >= 0 && cachedDevice != targetDevice) {
                                        wrongDevice = true;
                                    }
                                }
                            }
                            if (wrongDevice) {
                                // Wrong device — put cached array in pendingClose.
                                // Don't set out — let the code fall through to fresh allocation.
                                pendingClose.add(cbuf);
                                slotArrayCache[slotIdx] = null;
                                wrongDeviceCacheEjections++;
                            } else {
                            reshapeBuffer(cached, actualShape);
                            // Invalidate the cached OpaqueNDArray so a fresh C++ NDArray*
                            // wrapper is created from the current Java-side shape info.
                            // Without this, the stale OpaqueNDArray from the previous execution
                            // may have a C++ shape info pointer that was modified by the C++ op
                            // (e.g., scalar ops can modify output shapes in-place). The Java
                            // side still has the correct shape, but the cached C++ wrapper doesn't.
                            cached.clearOpaqueNDArray();
                            if (fastZero(cached, nativeOps, execStream)) {
                                out = cached;
                                slotArrayCache[slotIdx] = null;
                                if (TIMING_ENABLED) { timingPoolHits++; timingZeroApplied++; }
                            } else {
                                // Both async and sync memset failed → GPU memory is invalid.
                                // This can happen if the buffer was freed through an aliased ODB.
                                log.warn("Slot cache: invalid GPU memory at slot {} ({}), allocating fresh",
                                        slotIdx, slot.getOpName());
                                slotArrayCache[slotIdx] = null;
                                // out remains null → falls through to fresh allocation
                            }
                            } // end else (correct device)
                        } else if (TIMING_ENABLED && cbuf != null) {
                            // Diagnose cache miss reason
                            String reason = !cbuf.closeable() ? "not-closeable(const=" + cbuf.isConstant() + ")"
                                    : cbuf.wasClosed() ? "closed"
                                    : cached.dataType() != dt ? "dtype-mismatch"
                                    : cbuf.length() < numElements(actualShape) ? "too-small(" + cbuf.length() + "<" + numElements(actualShape) + ")"
                                    : "unknown";
                            if (timingCacheMissReasons.size() < 20) {
                                timingCacheMissReasons.merge(reason, 1, Integer::sum);
                            }
                        }
                    }
                    if (out == null) {
                        // Cached array is stale or wrong type — close it.
                        // Don't gate on closeable() — oversized buffers from growth factor
                        // are OWNED arrays safe to free. Only skip true constants.
                        if (cached != null && !cached.wasClosed()) {
                            DataBuffer cbuf = cached.data();
                            if (cbuf != null && !cbuf.wasClosed() && !cbuf.isConstant()) {
                                pendingClose.add(cbuf);
                            } else if (TIMING_ENABLED && cbuf != null && !cbuf.wasClosed() && cbuf.isConstant()) {
                                timingCacheLeakedConstant++;
                                timingCacheLeakedConstantBytes += cbuf.length() * cbuf.getElementSize();
                            }
                        }
                        slotArrayCache[slotIdx] = null;
                    }
                }
                if (out == null) {
                    if (slotIdx >= 0 && outputSlotSet.get(slotIdx)) {
                        // Output arrays are claimed by caller and NOT reused by slot cache.
                        // Allocate DIRECTLY via Nd4j.create() — NOT through mmgr (ArrayCacheMemoryMgr)
                        // which adds 5% headroom (growthFactor=1.05), making data().length() > length()
                        // → isView()=true → closeable()=false → close() silently fails → 30MB/step leak.
                        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            out = Nd4j.create(dt, actualShape);
                        }
                    } else if (slotIdx >= 0 && slotArrayCache != null) {
                        // Intermediate slots get 2x headroom for slot cache reuse.
                        out = allocateForSlotCache(dt, actualShape);
                    } else {
                        out = allocateWithHeadroom(dt, actualShape);
                    }
                    if (TIMING_ENABLED) { timingPoolMisses++; timingFreshAllocs++; }
                }
            }
            outputArrays[i] = out;

            if (slotIdx >= 0) {
                outputSlots[outputSlotIndices[i]] = outputArrays[i];
                liveSlots.set(outputSlotIndices[i]);
                // Cache the ACTUAL device ID for this output slot. C++ allocateFailover
                // may silently place the buffer on a different GPU than the target device.
                // Using the planned targetDevice here caused stale cache entries that
                // prevented Step 1b from migrating cross-device inputs → CUDA error 700.
                int actualDeviceId = targetDevice >= 0 ? targetDevice : 0;
                if (out != null && out.data() != null) {
                    OpaqueDataBuffer outOdb = out.data().opaqueBuffer();
                    if (outOdb != null && !outOdb.isNull()) {
                        int realDevice = nativeOps.dbDeviceId(outOdb);
                        if (realDevice >= 0) actualDeviceId = realDevice;
                    }
                }
                outputSlotDeviceIds[outputSlotIndices[i]] = actualDeviceId;
            }
        }
        ctx.setOutputArrays(outputArrays);
        if (TIMING_ENABLED) timingAllocNs += System.nanoTime() - tAlloc0;

        // Save GPU buffer addresses BEFORE execution for view-producer detection.
        // After execution, if C++ modifies an output in-place to be a view, the GPU
        // address changes (points to input's buffer instead of our pre-allocation).
        // We compare pre/post addresses to detect TRUE view producers, avoiding false
        // positives from 2x headroom (which makes isView()=true for all large arrays).
        // We save raw long addresses (not ODB objects) because C++ modifies the native
        // DataBuffer in-place — the ODB wraps the same C++ object and would return the
        // NEW address after modification.
        //
        // OPTIMIZATION: Once view-producer detection is complete (after first execution),
        // skip the expensive JNI calls (dbSpecialBuffer per output per slot). The
        // slotIsViewProducer[] array is stable across executions since view behavior
        // depends on the op, not the data. This saves ~2 JNI calls per output per slot.
        long[] preExecGpuAddrs = null;
        if (!viewProducerDetectionDone) {
            preExecGpuAddrs = new long[outputArrays.length];
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray arr = outputArrays[i];
                if (arr != null && !arr.isEmpty()) {
                    DataBuffer buf = arr.data();
                    if (buf != null && !buf.wasClosed()) {
                        OpaqueDataBuffer odb = buf.opaqueBuffer();
                        if (odb != null && !odb.isNull()) {
                            Pointer special = nativeOps.dbSpecialBuffer(odb);
                            if (special != null) preExecGpuAddrs[i] = special.address();
                        }
                    }
                }
            }
        }

        // Step 4c: If output allocation failed over to a different device (target OOM),
        // handle based on P2P capability:
        // - P2P available: switch execution to failover device (cross-device memory works)
        // - No P2P: emergency reclaim on original device and retry there (cross-device
        //   memory access causes cudaErrorIllegalAddress on non-P2P GPUs)
        if (targetDevice >= 0) {
            int failoverDevice = -1;
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray out = outputArrays[i];
                if (out != null && !out.isEmpty() && out.data() != null) {
                    OpaqueDataBuffer odb = out.data().opaqueBuffer();
                    if (odb != null && !odb.isNull()) {
                        int actualDevice = nativeOps.dbDeviceId(odb);
                        if (actualDevice >= 0 && actualDevice != targetDevice) {
                            failoverDevice = actualDevice;
                            break;
                        }
                    }
                }
            }
            if (failoverDevice >= 0) {
                boolean p2pAccess = nativeOps.isPeerAccessSupported(targetDevice, failoverDevice);

                if (!p2pAccess) {
                    // Non-P2P failover: can't execute cross-device. Free misplaced outputs,
                    // emergency flush pending buffers on original device, then retry allocation.
                    log.info("Op {} output OOM on device {}, failover to non-P2P device {} — emergency reclaim on device {}",
                            slot.getOpName(), targetDevice, failoverDevice, targetDevice);

                    // Save shapes/dtypes and free misplaced outputs on the failover device
                    int originalTarget = targetDevice;
                    DataType[] retryDtypes = new DataType[outputArrays.length];
                    long[][] retryShapes = new long[outputArrays.length][];
                    for (int i = 0; i < outputArrays.length; i++) {
                        INDArray out = outputArrays[i];
                        if (out == null || out.isEmpty()) continue;
                        DataBuffer db = out.data();
                        if (db == null) continue;
                        OpaqueDataBuffer odb = db.opaqueBuffer();
                        if (odb == null || odb.isNull()) continue;
                        int actualDevice = nativeOps.dbDeviceId(odb);
                        if (actualDevice == failoverDevice) {
                            retryDtypes[i] = out.dataType();
                            retryShapes[i] = out.shape();
                            try {
                                nativeOps.dbFreeBuffersOnly(odb);
                                odb.tryMarkForDeallocation();
                                odb.setNull();
                                OpaqueDataBufferDeallocator dealloc = odb.getDeallocator();
                                if (dealloc != null) dealloc.markDeallocated();
                            } catch (Exception e) {
                                log.warn("Failed to free misplaced output on device {}: {}", failoverDevice, e.getMessage());
                            }
                            outputArrays[i] = null;
                        }
                    }

                    // Switch back to original device
                    DeviceMemoryManager.getInstance().switchDevice(originalTarget, "DSP.executeSlot", "retry-restore");
                    nativeOps.clearLastError();
                    Nd4j.getExecutioner().commit();

                    // Emergency flush ALL pending + deferred buffers to maximize memory recovery
                    if (!pendingClose.isEmpty() || !deferredClose.isEmpty()) {
                        flushPendingClose(nativeOps, execStream);
                    }

                    // Re-fetch fresh stream after device switch + flush
                    Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                    if (freshStream != null) execStream = freshStream;
                    nativeOps.clearLastError();

                    // Retry allocation on original device
                    boolean retryOk = true;
                    for (int i = 0; i < outputArrays.length; i++) {
                        if (retryShapes[i] == null) continue;
                        try {
                            int slotIdx = (i < outputSlotIndices.length) ? outputSlotIndices[i] : -1;
                            boolean isOutputSlot = slotIdx >= 0 && outputSlotSet.get(slotIdx);
                            INDArray newOut;
                            if (isOutputSlot) {
                                newOut = Nd4j.create(retryDtypes[i], retryShapes[i]);
                            } else if (slotIdx >= 0 && slotArrayCache != null) {
                                newOut = allocateForSlotCache(retryDtypes[i], retryShapes[i]);
                            } else {
                                newOut = Nd4j.create(retryDtypes[i], retryShapes[i]);
                            }
                            // Check which device the allocation landed on
                            OpaqueDataBuffer retryOdb = newOut.data().opaqueBuffer();
                            if (retryOdb != null && !retryOdb.isNull()) {
                                int retryDevice = nativeOps.dbDeviceId(retryOdb);
                                if (retryDevice >= 0 && retryDevice != originalTarget) {
                                    // Allocation landed on different device — accept it and
                                    // switch execution there with full input migration.
                                    // Mark the original device as failed — CUDA error 700
                                    // from OOM cascades is sticky and unrecoverable.
                                    // All remaining ops planned for this device will redirect to device 0.
                                    log.warn("Emergency reclaim insufficient on device {} — output for {} allocated on device {}, switching execution. Device {} marked failed for remaining ops.",
                                            originalTarget, slot.getOpName(), retryDevice, originalTarget);
                                    if (failedDevices != null) failedDevices.add(originalTarget);
                                    targetDevice = retryDevice;
                                    deviceSwitchOccurred = true;
                                    DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "oom-failover");
                                    nativeOps.clearLastError();
                                    Nd4j.getExecutioner().commit();
                                    Pointer freshStream2 = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                                    if (freshStream2 != null) execStream = freshStream2;
                                    // Migrate inputs to the new device (non-P2P can't cross-access)
                                    for (int j = 0; j < inputArrays.length; j++) {
                                        INDArray input = inputArrays[j];
                                        if (input != null && !input.isEmpty() && input.data() != null) {
                                            int inputDevice = -1;
                                            OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                                            if (inputOdb != null && !inputOdb.isNull()) {
                                                inputDevice = nativeOps.dbDeviceId(inputOdb);
                                            }
                                            if (inputDevice >= 0 && inputDevice != targetDevice) {
                                                INDArray inputToReplicate = input;
                                                if (input.isView()) {
                                                    // Switch to source device for dup() — non-P2P devices can't cross-access
                                                    DeviceMemoryManager.getInstance().switchDevice(inputDevice, "DSP.executeSlot", "failover-view-dup-source");
                                                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                                        inputToReplicate = input.dup(input.ordering());
                                                    }
                                                    DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "failover-view-dup-restore");
                                                    DataBuffer dupBuf = inputToReplicate.data();
                                                    if (dupBuf != null && !dupBuf.isConstant()) {
                                                        if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                                        replicatedInputBuffers.add(dupBuf);
                                                    }
                                                }
                                                INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                                                inputArrays[j] = replica;
                                                DataBuffer replicaBuf = replica.data();
                                                if (replicaBuf != null && !replicaBuf.isConstant()) {
                                                    if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                                    replicatedInputBuffers.add(replicaBuf);
                                                }
                                            }
                                        }
                                    }
                                    ctx.setInputArrays(inputArrays);
                                    nativeOps.clearLastError();
                                }
                            }
                            outputArrays[i] = newOut;
                            fastZero(newOut, nativeOps, execStream);
                            if (slotIdx >= 0) {
                                outputSlots[slotIdx] = newOut;
                                liveSlots.set(slotIdx);
                                // Update cached device ID after failover reallocation.
                                // Use JNI to get the actual device since retry may land on a different one.
                                OpaqueDataBuffer retryOdb2 = newOut.data().opaqueBuffer();
                                if (retryOdb2 != null && !retryOdb2.isNull()) {
                                    outputSlotDeviceIds[slotIdx] = nativeOps.dbDeviceId(retryOdb2);
                                }
                            }
                            // Update pre-exec GPU address for view-producer detection
                            if (preExecGpuAddrs != null) {
                                DataBuffer buf = newOut.data();
                                if (buf != null && !buf.wasClosed()) {
                                    OpaqueDataBuffer odb2 = buf.opaqueBuffer();
                                    if (odb2 != null && !odb2.isNull()) {
                                        Pointer special = nativeOps.dbSpecialBuffer(odb2);
                                        if (special != null) preExecGpuAddrs[i] = special.address();
                                    }
                                }
                            }
                        } catch (Exception e) {
                            log.error("Emergency reclaim retry failed for output {} of {}: {}",
                                    i, slot.getOpName(), e.getMessage());
                            retryOk = false;
                            break;
                        }
                    }

                    if (!retryOk) {
                        throw new RuntimeException("OOM on device " + originalTarget + " for op " +
                                slot.getOpName() + " — emergency reclaim freed insufficient memory. " +
                                "Consider reducing model size or sequence length.");
                    }
                    ctx.setOutputArrays(outputArrays);
                    nativeOps.clearLastError();
                    log.info("  Emergency reclaim succeeded — continuing on device {}", targetDevice);

                } else {
                    // P2P failover: transparently switch execution to failover device.
                    // Cross-device memory access works via P2P, so inputs/outputs on different
                    // devices can be accessed directly.
                    log.info("Op {} output failed over from device {} to P2P device {} (OOM) — switching execution",
                            slot.getOpName(), targetDevice, failoverDevice);
                    if (!pendingClose.isEmpty()) {
                        flushPendingClose(nativeOps, execStream);
                    }
                    targetDevice = failoverDevice;
                    deviceSwitchOccurred = true;
                    DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "p2p-failover");
                    nativeOps.clearLastError();
                    Nd4j.getExecutioner().commit();
                    nativeOps.clearLastError();
                    Pointer deviceStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                    if (deviceStream != null) execStream = deviceStream;
                    // Re-migrate inputs to the failover device
                    for (int i = 0; i < inputArrays.length; i++) {
                        INDArray input = inputArrays[i];
                        if (input != null && !input.isEmpty() && input.data() != null) {
                            int inputDevice = -1;
                            OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                            if (inputOdb != null && !inputOdb.isNull()) {
                                inputDevice = nativeOps.dbDeviceId(inputOdb);
                            }
                            if (inputDevice >= 0 && inputDevice != targetDevice) {
                                // Pre-dup views (same reason as Step 1b)
                                // Switch to source device for dup() — non-P2P devices can't cross-access
                                INDArray inputToReplicate = input;
                                if (input.isView()) {
                                    DeviceMemoryManager.getInstance().switchDevice(inputDevice, "DSP.executeSlot", "failover2-view-dup-source");
                                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                        inputToReplicate = input.dup(input.ordering());
                                    }
                                    DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.executeSlot", "failover2-view-dup-restore");
                                    DataBuffer dupBuf = inputToReplicate.data();
                                    if (dupBuf != null && !dupBuf.isConstant()) {
                                        if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                        replicatedInputBuffers.add(dupBuf);
                                    }
                                }
                                INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                                inputArrays[i] = replica;
                                DataBuffer replicaBuf = replica.data();
                                if (replicaBuf != null && !replicaBuf.isConstant()) {
                                    if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                    replicatedInputBuffers.add(replicaBuf);
                                }
                            }
                        }
                    }
                    ctx.setInputArrays(inputArrays);
                    nativeOps.clearLastError();
                    for (int i = 0; i < outputArrays.length; i++) {
                        INDArray out = outputArrays[i];
                        if (out != null && !out.isEmpty() && out.data() != null) {
                            fastZero(out, nativeOps, execStream);
                        }
                    }
                    nativeOps.clearLastError();
                    ctx.setOutputArrays(outputArrays);
                    // Update cached device IDs for outputs that failed over
                    for (int i = 0; i < outputSlotIndices.length; i++) {
                        int si = outputSlotIndices[i];
                        if (si >= 0) outputSlotDeviceIds[si] = failoverDevice;
                    }
                }
            }
        }

        // Step 4d: Shape buffer correction for non-P2P devices.
        // ConstantShapeHelper caches shape buffers per-device, but allocateFailover() may
        // place them on a different device. For non-P2P GPUs, accessing a shape buffer on
        // the wrong device causes CUDA error 700 (illegal memory access). Detect and fix
        // by creating a fresh shape buffer from the Java-side jvmShapeInfo on the target device.
        if (targetDevice >= 0 && isPeerAccessible != null
                && targetDevice < isPeerAccessible.length && !isPeerAccessible[targetDevice]) {
            int corrections = 0;
            for (INDArray out : outputArrays) {
                if (ensureShapeOnDevice(out, targetDevice, nativeOps)) corrections++;
            }
            for (INDArray in : inputArrays) {
                if (ensureShapeOnDevice(in, targetDevice, nativeOps)) corrections++;
            }
            if (corrections > 0) {
                shapeBufferCorrections += corrections;
                if (TIMING_ENABLED) {
                    log.debug("Op {}: corrected {} shape buffers to device {}", slot.getOpName(), corrections, targetDevice);
                }
            }
        }

        // Step 5: Execute
        // Clear any stale CUDA errors from fastZero or allocation before op execution.
        // Without this, a failed cudaMemsetAsync (e.g., from cross-device buffer zeroing)
        // leaves a stale error that CudaExecutioner picks up via lastErrorCode() after
        // execCustomOp2, causing a spurious "cudaMemsetAsync failed" exception.
        // OPTIMIZATION: Only clear when a device switch or failover occurred in this slot.
        // Same-device slots can't produce stale errors from fastZero/allocation.
        // Saves ~1800 JNI calls per vision frame (most slots don't switch devices).
        if (deviceSwitchOccurred) {
            nativeOps.clearLastError();
        }
        // Disable shape override for:
        // 1. Data-dependent shapes — Java may read stale GPU values before host sync
        // 2. View-capable ops — C++ must create its own output (a view of the input buffer);
        //    our pre-allocated empty placeholder cannot be used as the output buffer.
        boolean enableShapeOverride = SHAPE_OVERRIDE && !slot.isDataDependent() && !slot.isViewCapableOp();
        ctx.shapeFunctionOverride(enableShapeOverride);

        // Attach native workspace to OpContext — this allows C++ ops to use bump allocation
        // for internal temporaries instead of per-op malloc/cudaMalloc. Without this, C++ op
        // temporary buffer overruns corrupt the regular malloc heap metadata, causing
        // "double free or corruption (out)" SIGABRT crashes.
        // OPTIMIZATION: Skip mmgr.getNativeWorkspacePointer() after the first call returns
        // null, since the mmgr type doesn't change between ops. This saves one virtual
        // method call per slot (~1962 calls per vision encoder frame).
        Pointer wsPtr = ownNativeWorkspace != null ? null : mmgr.getNativeWorkspacePointer();
        if (wsPtr == null && ownNativeWorkspace == null) {
            log.debug("DSP: mmgr ({}) returned null workspace, creating own", mmgr.getClass().getSimpleName());
        }
        if (wsPtr == null) {
            // mmgr doesn't provide a workspace (e.g., ArrayCacheMemoryMgr). Create our own.
            if (ownNativeWorkspace == null) {
                try {
                    NativeOps ws_nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    ownNativeWorkspace = ws_nativeOps.createNativeWorkspace(DSP_NATIVE_WORKSPACE_BYTES);
                    if (ownNativeWorkspace != null) {
                        ws_nativeOps.workspaceScopeIn(ownNativeWorkspace);
                        log.debug("DSP: created native workspace ({}MB) for C++ op temporaries",
                                DSP_NATIVE_WORKSPACE_BYTES / (1024 * 1024));
                    } else {
                        log.warn("DSP: createNativeWorkspace returned null");
                    }
                } catch (Exception e) {
                    log.warn("Failed to create DSP native workspace: {}", e.getMessage());
                }
            }
            wsPtr = ownNativeWorkspace;
        }
        if (wsPtr != null) {
            ctx.attachWorkspace(wsPtr);
        }

        long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
        // Pre-exec shape validation: ensure allocated buffers can hold the data C++ will write.
        // With shapeFunctionOverride=true, C++ skips shape calc and uses our pre-allocated buffers.
        // If Java calculated wrong shapes, C++ will overflow → heap corruption.
        // OPTIMIZATION: Only validate when shape override is active — if C++ is computing
        // shapes itself (data-dependent ops), this check is redundant.
        if (enableShapeOverride) {
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray outArr = outputArrays[i];
                if (outArr != null && !outArr.isEmpty()) {
                    long javaLen = outArr.length();
                    long bufLen = outArr.data() != null ? outArr.data().length() : 0;
                    // Buffer must be at least as large as the declared shape
                    if (bufLen < javaLen) {
                        log.error("CRITICAL: Buffer too small for {} output[{}]: shape declares {} elements but buffer only has {}. Disabling shape override to prevent heap corruption.",
                                slot.getOpName(), i, javaLen, bufLen);
                        // Disable shape override for this op - let C++ calculate correct shapes
                        ctx.shapeFunctionOverride(false);
                        break;
                    }
                }
            }
        }

        if (slot.isCustomOp()) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            ctx.setSArguments(slot.getSArgs() == null ? new String[0] : slot.getSArgs());
            // For view-capable ops, call initializeOutputs to let the op create its
            // view with the correct offset/strides before C++ execution. This is
            // critical for strided_slice which needs a non-zero offset into the input
            // buffer. If initializeOutputs returns false, it has set up the output
            // already and we can update the slot tracking.
            boolean viewOpProducedActualView = false;
            if (slot.isViewCapableOp() && fn instanceof CustomOp) {
                CustomOp customOp = (CustomOp) fn;
                // Sync op-level inputArguments from OpContext. DSP sets inputs on the
                // OpContext, not on the op object. But initializeOutputs() accesses
                // this.inputArguments directly (e.g., ReshapeNoCopy needs inputArguments.get(0)
                // to create a view sharing the input's data buffer). Without this sync,
                // inputArguments is empty → IndexOutOfBoundsException → slot fails.
                // Always sync (not just when empty) because the OpContext may have different
                // arrays than a previous execution.
                if (customOp instanceof DynamicCustomOp) {
                    DynamicCustomOp dco = (DynamicCustomOp) customOp;
                    dco.setInputArguments((INDArray[]) null);
                    dco.outputArguments().clear();
                    List<INDArray> ctxInputs = ctx.getInputArrays();
                    if (ctxInputs != null) {
                        for (INDArray input : ctxInputs) {
                            if (input != null) {
                                dco.addInputArgument(input);
                            }
                        }
                    }
                }
                customOp.initializeOutputs(ctx);
                // Propagate outputs regardless of return value — semantics differ across ops
                // (Permute/StridedSlice return false, ReshapeNoCopy/base return true on success).
                // In DSP the return value is irrelevant; we just need outputs propagated.
                if (customOp.numOutputArguments() > 0) {
                    INDArray viewOut = customOp.getOutputArgument(0);
                    if (viewOut != null && (!viewOut.isEmpty() || viewOut.rank() > 0)) {
                        // Determine if this is a TRUE view (shares DataBuffer with an input)
                        // vs. a newly allocated buffer (from super.initializeOutputs()).
                        // Only true views should be marked as view producers — their buffers
                        // belong to the input and must never be freed/un-poisoned.
                        boolean isActualView = false;
                        DataBuffer outBuf = viewOut.data();
                        if (outBuf != null) {
                            List<INDArray> inputs = ctx.getInputArrays();
                            if (inputs != null) {
                                for (INDArray in : inputs) {
                                    if (in != null && in.data() == outBuf) {
                                        isActualView = true;
                                        break;
                                    }
                                }
                            }
                        }
                        ctx.setOutputArray(0, viewOut);
                        int[] outSlotIndices = slot.getOutputSlotIndices();
                        if (outSlotIndices.length > 0 && outSlotIndices[0] >= 0) {
                            int viewSlotIdx = outSlotIndices[0];
                            outputSlots[viewSlotIdx] = viewOut;
                            if (slotArrayCache != null) slotArrayCache[viewSlotIdx] = viewOut;
                            if (slotIsViewProducer != null) slotIsViewProducer[viewSlotIdx] = isActualView;
                        }
                        viewOpProducedActualView = isActualView;
                    }
                }
            }
            // For view-capable ops where initializeOutputs created a true view
            // (output shares the input's DataBuffer), skip C++ execution entirely.
            // The view is already correct — C++ permute/reshape/squeeze/expand_dims
            // would just check same-buffer and return OK (no-op). Skipping avoids
            // C++ prepareOutputs (with shapeFunctionOverride=false) which runs
            // calculateOutputShape and creates a C++ NDArray* for the output through
            // OpaqueNDArray. This OpaqueNDArray goes through ConstantShapeHelper cache
            // lookup which may return a different shapeInfo (C-contiguous strides)
            // than the view's permuted strides, corrupting downstream ops that read
            // through the cached OpaqueNDArray.
            if (!viewOpProducedActualView) {
                Nd4j.exec((CustomOp) fn, ctx);
            }
        } else {
            Nd4j.exec((Op) fn, ctx);
        }

        // Detach workspace after execution to prevent shape computation from
        // allocating shape buffers in workspace on ctx reuse (ShapeList::destroy
        // calls delete[] on them → SIGSEGV).
        if (wsPtr != null) {
            ctx.detachWorkspace();
            // Reset workspace offset periodically. RELEASE is a no-op for workspace
            // allocations, so temp memory accumulates. ScopeOut+ScopeIn resets the
            // bump pointer so the workspace can be reused by the next op.
            // This MUST happen periodically for ALL workspaces (mmgr-provided or
            // self-managed), otherwise the workspace fills after ~30 ops and all
            // subsequent ops spill to cudaHostAlloc, accumulating thousands of spill
            // entries whose tracking vector grows via glibc realloc.
            // OPTIMIZATION: Reset every WORKSPACE_RESET_INTERVAL ops instead of every
            // op. Saves ~96% of 2 JNI calls × 1962 ops = ~3700 JNI calls per vision frame.
            workspaceOpsSinceReset++;
            if (workspaceOpsSinceReset >= WORKSPACE_RESET_INTERVAL) {
                nativeOps.workspaceScopeOut(wsPtr);
                nativeOps.workspaceScopeIn(wsPtr);
                workspaceOpsSinceReset = 0;
            }
        }

        // After execution, C++ may have replaced output arrays or modified them in-place
        // to be views. Detect both cases and mark slots as view producers for future steps.
        List<INDArray> ctxOutputs = ctx.getOutputArrays();
        int maxTracked = Math.min(ctxOutputs != null ? ctxOutputs.size() : 0, outputSlotIndices.length);

        if (ctxOutputs != null) {
            for (int i = 0; i < maxTracked; i++) {
                INDArray ctxOut = ctxOutputs.get(i);
                int si = outputSlotIndices[i];
                if (ctxOut == null || si < 0) continue;

                if (ctxOut != outputArrays[i]) {
                    // Case 1: C++ replaced the output with a different object (new view).
                    if (slotIsViewProducer != null) slotIsViewProducer[si] = true;
                    if (!outputArrays[i].isEmpty()) {
                        DataBuffer buf = outputArrays[i].data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            pendingClose.add(buf);
                        }
                    }
                    outputSlots[si] = ctxOut;
                } else if (!viewProducerDetectionDone
                        && slotIsViewProducer != null && !slotIsViewProducer[si]
                        && preExecGpuAddrs != null) {
                    // Case 2: Check if C++ modified the output's GPU buffer in-place.
                    // Compare pre-execution GPU address with current address.
                    // If they differ, C++ replaced the buffer with a view of the input.
                    // We must NOT use isView() here — 2x headroom allocation makes
                    // isView()=true for ALL arrays >256 elements (false positive).
                    // OPTIMIZATION: Skip on subsequent executions (viewProducerDetectionDone)
                    // since view behavior is stable per-op, not per-data.
                    long preAddr = (i < preExecGpuAddrs.length) ? preExecGpuAddrs[i] : 0;
                    if (preAddr != 0) {
                        DataBuffer currentBuf = ctxOut.data();
                        OpaqueDataBuffer currentOdb = (currentBuf != null) ? currentBuf.opaqueBuffer() : null;
                        long currentAddr = 0;
                        if (currentOdb != null && !currentOdb.isNull()) {
                            Pointer special = nativeOps.dbSpecialBuffer(currentOdb);
                            if (special != null) currentAddr = special.address();
                        }
                        if (currentAddr != 0 && preAddr != currentAddr) {
                            // GPU address changed — C++ made this a view of the input.
                            // The original allocation at preAddr is orphaned (one-time cost
                            // on the first execute(); subsequent calls use Nd4j.empty()).
                            slotIsViewProducer[si] = true;
                        }
                    }
                }
            }
        }

        // Release untracked output arrays
        for (int i = 0; i < outputArrays.length; i++) {
            boolean tracked = (i < outputSlotIndices.length && outputSlotIndices[i] >= 0);
            if (!tracked) {
                INDArray arr = outputArrays[i];
                if (ctxOutputs != null && i < ctxOutputs.size()) {
                    INDArray ctxOut = ctxOutputs.get(i);
                    if (ctxOut != null && ctxOut != arr) {
                        arr = ctxOut;
                    }
                }
                if (arr != null) {
                    DataBuffer buf = arr.data();
                    if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                        pendingClose.add(buf);
                    }
                }
            }
        }

        // Release replicated input copies to prevent GPU memory leak.
        // replicateToDevice() creates new arrays for cross-device inputs in Step 1b and
        // Step 4c failover. Without explicit close, these leak forever (GC cleanup is broken
        // for GPU buffers — PhantomRef strong reference cycle). The live view range check
        // in flushPendingClose() safely defers any buffer whose GPU memory is still
        // referenced by a live output slot (e.g., if C++ made an output view of the input).
        if (replicatedInputBuffers != null) {
            for (DataBuffer buf : replicatedInputBuffers) {
                if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                    pendingClose.add(buf);
                }
            }
        }

        if (TIMING_ENABLED) timingExecNs += System.nanoTime() - tExec0;

        } finally {
            // Always restore the caller's device context, even after transparent failover.
            // Use targetDevice to detect if a switch happened instead of querying
            // getDeviceForCurrentThread() (saves ~1962 JNI calls per vision frame).
            // If targetDevice >= 0 and differs from previousDeviceId, we switched at line 1012.
            // If failover changed targetDevice mid-execution, we also need to restore.
            if (targetDevice >= 0 && targetDevice != previousDeviceId) {
                DeviceMemoryManager.getInstance().switchDevice(previousDeviceId, "DSP.executeSlot", "restore-caller-device");
            }
        }
    }

    private void syncIntLongInputs(INDArray[] inputs, boolean isDataDependent, NativeOps nativeOps) {
        boolean needsSync = isDataDependent;
        if (!needsSync) {
            for (INDArray in : inputs) {
                if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG ||
                        in.dataType() == DataType.BOOL)
                        && !in.data().isConstant()
                        && in.length() <= 32) {
                    needsSync = true;
                    break;
                }
            }
        }

        if (needsSync) {
            Nd4j.getExecutioner().commit();
            for (INDArray in : inputs) {
                if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG ||
                        in.dataType() == DataType.BOOL)
                        && !in.data().isConstant()) {
                    if (isDataDependent || in.length() <= 32) {
                        nativeOps.dbForceSyncToPrimary(in.data().opaqueBuffer());
                    }
                }
            }
        }
    }

    private List<DataBuffer> getOrComputeShapes(DynamicShapeSlot slot, OpContext ctx,
                                                  DifferentialFunction fn, INDArray[] inputArrays,
                                                  NativeOps nativeOps) {
        long shapeKey = computeShapeKey(slot, inputArrays);

        if (slot.isShapeCacheValid(shapeKey) && !slot.isDataDependent()) {
            if (TIMING_ENABLED) timingShapeHits++;
            return slot.getCachedOutputShapes();
        }

        if (TIMING_ENABLED) timingShapeMisses++;

        List<DataBuffer> outShapes = null;

        if (fn instanceof DynamicCustomOp) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            ctx.setSArguments(slot.getSArgs() == null ? new String[0] : slot.getSArgs());
            outShapes = ((DynamicCustomOp) fn).calculateOutputShapeFromInputs(ctx);
        }

        if (outShapes == null || outShapes.isEmpty()) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                if (fn instanceof CustomOp) {
                    ctx.setIArguments(slot.getIArgs());
                    ctx.setTArguments(slot.getTArgs());
                    ctx.setBArguments(slot.getBArgs());
                    ctx.setDArguments(slot.getDArgs());
                    ctx.setSArguments(slot.getSArgs() == null ? new String[0] : slot.getSArgs());

                    // Use the slot's opName for hash lookup, not fn.opHash().
                    // The compiler may remap ops (e.g., reshape → reshape_no_copy) and
                    // store the remapped name in the slot while keeping the original op
                    // object. fn.opHash() would return the ORIGINAL op's hash (e.g.,
                    // "reshape"), causing the C++ side to call the wrong shape function
                    // with incompatible iArgs format.
                    String slotOpName = slot.getOpName();
                    long opHash;
                    if (!slotOpName.equals(fn.opName())) {
                        // Op was remapped — look up hash by the slot's actual op name
                        Map<String, org.nd4j.linalg.api.ops.CustomOpDescriptor> customOps = Nd4j.getExecutioner().getCustomOperations();
                        org.nd4j.linalg.api.ops.CustomOpDescriptor desc = customOps.get(slotOpName);
                        if (desc == null) {
                            throw new RuntimeException("Op name " + slotOpName + " not found in custom operations registry");
                        }
                        opHash = desc.getHash();
                    } else {
                        opHash = ((CustomOp) fn).opHash();
                    }
                    // Always use calculateOutputShapes2 which syncs INT/LONG input values to host.
                    // syncIntLongInputs() already syncs these before shape computation, so this
                    // is effectively zero additional cost. Eliminates the need for a hardcoded
                    // VALUE_DEPENDENT_SHAPE_OPS list to gate the sync/no-sync decision.
                    OpaqueShapeList shapeList = nativeOps.calculateOutputShapes2(null, opHash,
                            ctx.contextPointer());

                    if (nativeOps.lastErrorCode() != 0 || shapeList == null) {
                        throw new RuntimeException("Shape calculation failed for op " +
                                slot.getOpName() + ": " + nativeOps.lastErrorMessage());
                    }

                    outShapes = new ArrayList<>();
                    int numShapes = (int) nativeOps.getShapeListSize(shapeList);
                    for (int e = 0; e < numShapes; e++) {
                        outShapes.add(readShapeFromNative(nativeOps, shapeList, e));
                    }
                    nativeOps.deleteShapeList(shapeList);
                } else {
                    outShapes = fn.calculateOutputShape(ctx);
                }
            }
        }

        if (!slot.isDataDependent() && outShapes != null && !outShapes.isEmpty()) {
            slot.updateShapeCache(shapeKey, outShapes);
        }

        return outShapes;
    }

    private static DataBuffer readShapeFromNative(NativeOps nativeOps, OpaqueShapeList list, int index) {
        LongPointer ptr = new PagedPointer(nativeOps.getShape(list, index)).asLongPointer();
        int rank = (int) ptr.get(0);
        int len = Shape.shapeInfoLength(rank);
        long[] shapeInfo = new long[len];
        ptr.capacity(len);
        ptr.get(shapeInfo, 0, len);

        OpaqueConstantShapeBuffer csb = nativeOps.cacheAndStoreShapeBuffer(shapeInfo);
        if (csb == null) {
            throw new RuntimeException("Failed to cache shape buffer");
        }

        Pointer primaryPtr = nativeOps.getConstantShapeBufferPrimary(csb);
        Pointer specialPtr = nativeOps.getConstantShapeBufferSpecial(csb);

        DataBuffer buffer;
        if (specialPtr != null && specialPtr.address() != 0) {
            buffer = Nd4j.createBuffer(primaryPtr, specialPtr, len, DataType.INT64);
        } else {
            buffer = Nd4j.createBuffer(primaryPtr, len, DataType.INT64);
        }
        buffer.setConstant(true);
        return buffer;
    }

    private long computeShapeKey(DynamicShapeSlot slot, INDArray[] inputArrays) {
        long hash = slot.getOpNameHash();
        for (INDArray in : inputArrays) {
            if (in != null) {
                for (long dim : in.shape()) {
                    hash ^= dim;
                    hash *= 0x517CC1B727220A95L;
                }
                hash ^= in.dataType().ordinal();
                hash *= 0x9E3779B97F4A7C15L;

                // Always hash INT/LONG input values for small tensors — matches C++ computeShapeKey()
                // which unconditionally hashes these. Eliminates VALUE_DEPENDENT_SHAPE_OPS as a
                // correctness concern (missing ops caused stale cache hits).
                if ((in.dataType() == DataType.INT || in.dataType() == DataType.LONG)
                        && in.length() > 0 && in.length() <= 32
                        && in.data() != null && !in.data().wasClosed()) {
                    for (long j = 0; j < in.length(); j++) {
                        hash ^= in.getLong(j);
                        hash *= 0x517CC1B727220A95L;
                    }
                }
            }
        }
        long[] iArgs = slot.getIArgs();
        if (iArgs != null) {
            for (long arg : iArgs) {
                hash ^= arg;
                hash *= 0x9E3779B97F4A7C15L;
            }
        }
        return hash;
    }

    // Minimum padding elements added to every DSP output allocation.
    // Some C++ ops overrun their output buffers by a few bytes/elements, corrupting
    // adjacent glibc malloc chunk headers. This padding absorbs the overrun.
    // 256 elements * 4 bytes (FLOAT32) = 1KB safety margin per buffer.
    // For 2000 ops, total overhead is ~2MB — negligible vs the ~9GB of intermediates.
    private static final long OUTPUT_PADDING_ELEMENTS = 256;

    private INDArray allocateWithHeadroom(DataType dataType, long[] shape) {
        long requiredElements = numElements(shape);
        if (requiredElements <= 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }

        // Scalars (shape=[]) and small arrays: allocate exactly, no padding needed.
        // The overruns happen in large intermediate ops, not scalar reductions.
        if (shape.length == 0 || requiredElements <= OUTPUT_PADDING_ELEMENTS) {
            return mmgr.allocate(true, dataType, shape);
        }

        // Over-allocate to protect against C++ op output buffer overruns.
        // The padding absorbs any small overruns that would otherwise corrupt
        // adjacent malloc metadata, causing "double free or corruption" SIGABRT.
        double gf = ArrayCacheMemoryMgr.getGrowthFactor().get();
        long growthElements = gf > 1.0 ? (long) (requiredElements * gf) : requiredElements;
        long allocElements = Math.max(growthElements, requiredElements + OUTPUT_PADDING_ELEMENTS);

        INDArray oversized = mmgr.allocate(true, dataType, allocElements);
        reshapeBuffer(oversized, shape);
        return oversized;
    }

    /** Growth factor for slot cache refills. Arrays that grow each step (attention scores,
     *  KV cache intermediates) need headroom so the cached array lasts many steps.
     *
     *  With growth > 1.0, data().length() > length() after reshape, causing
     *  closeable()=false (BaseNDArray line 6471). Previously, closeable() gates in
     *  release/cache/free paths blocked ALL cleanup → permanent GPU memory leak.
     *  FIX: All closeable() gates replaced with isConstant() checks throughout
     *  DynamicShapePlanExecutor, allowing oversized buffers to be properly freed/reused.
     *
     *  For autoregressive decoding, KV cache arrays grow by 1 token per step.
     *  With growth factor 2.0 at 679 tokens → buffer for 1358 → no cache miss
     *  until step 679. Without headroom, EVERY step has ~133 cache misses. */
    private static final double SLOT_CACHE_GROWTH_FACTOR = Double.parseDouble(
            System.getProperty(ND4JSystemProperties.DSP_SLOT_CACHE_GROWTH_FACTOR, "1.0"));

    private INDArray allocateForSlotCache(DataType dataType, long[] shape) {
        long requiredElements = numElements(shape);
        if (requiredElements <= 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }
        if (shape.length == 0 || requiredElements <= OUTPUT_PADDING_ELEMENTS) {
            return mmgr.allocate(true, dataType, shape);
        }
        long allocElements = Math.max(
                (long) (requiredElements * SLOT_CACHE_GROWTH_FACTOR),
                requiredElements + OUTPUT_PADDING_ELEMENTS);
        INDArray oversized = mmgr.allocate(true, dataType, allocElements);
        reshapeBuffer(oversized, shape);
        return oversized;
    }

    /**
     * Fast buffer zeroing using direct memset instead of the full assign(0) op dispatch path.
     *
     * Always fetches a FRESH stream from the current device's LaunchContext. The execStream
     * pointer cached earlier in executeSlot can become stale: C++ ContextBuffers::release()
     * frees the underlying cudaStream_t* when device context changes occur during intermediate
     * JNI calls (shape computation, sync operations). Dereferencing a freed stream pointer
     * causes SIGSEGV in the CUDA driver (si_addr near 0x0, garbage stream handle).
     *
     * Falls back to synchronous memset if async fails or no valid stream is available.
     */
    /**
     * Ensure an array's shape buffer is on the target device. If not (due to
     * CudaMemoryPool::allocateFailover placing it on a different device), create
     * a fresh non-constant shape buffer from the Java-side jvmShapeInfo copy.
     *
     * <p>Each fresh buffer allocates host + device memory (~200 bytes each).
     * Host allocation counts against Pointer.maxPhysicalBytes() (JavaCPP off-heap limit).</p>
     *
     * @return true if shape buffer was corrected, false if already correct or skipped
     */
    private boolean ensureShapeOnDevice(INDArray array, int targetDevice, NativeOps nativeOps) {
        if (array == null || array.isEmpty()) return false;
        DataBuffer shapeDb = array.shapeInfoDataBuffer();
        if (shapeDb == null) return false;
        OpaqueDataBuffer shapeOdb = shapeDb.opaqueBuffer();
        if (shapeOdb == null || shapeOdb.isNull()) return false;
        int shapeDevice = nativeOps.dbDeviceId(shapeOdb);
        if (shapeDevice < 0 || shapeDevice == targetDevice) return false;

        // Check off-heap headroom before allocating.
        // Shape info is tiny (~200 bytes host + device) but respect the limit.
        long offHeapUsed = Pointer.physicalBytes();
        long offHeapMax = Pointer.maxPhysicalBytes();
        if (offHeapMax > 0 && offHeapUsed > offHeapMax * 95 / 100) {
            log.warn("Shape buffer on device {} needs copy to device {} but off-heap at {}% ({}/{}MB) — skipping",
                    shapeDevice, targetDevice,
                    offHeapMax > 0 ? (offHeapUsed * 100 / offHeapMax) : 0,
                    offHeapUsed / (1024*1024), offHeapMax / (1024*1024));
            return false;
        }

        long[] jvmShape = array.shapeInfoJava();
        DataBuffer freshShape = Nd4j.createBufferDetached(jvmShape);
        ((BaseNDArray) array).setShapeInfoDataBuffer(freshShape);
        return true;
    }

    /**
     * Zero a buffer's GPU (or host) memory. Returns true on success, false if the
     * CUDA memset failed (e.g., stale GPU pointer from a freed slot cache entry).
     * Callers should discard the buffer and allocate fresh on failure.
     */
    private static boolean fastZero(INDArray arr, NativeOps nativeOps, Pointer execStream) {
        DataBuffer buf = arr.data();
        if (buf == null || buf.wasClosed()) return true;

        OpaqueDataBuffer opaque = buf.opaqueBuffer();
        long bytes = buf.length() * buf.getElementSize();

        Pointer specialPtr = nativeOps.dbSpecialBuffer(opaque);
        if (specialPtr != null && specialPtr.address() != 0) {
            // CRITICAL: Check the buffer's actual device. C++ allocateFailover may have
            // silently placed the buffer on a different GPU than the current thread's device.
            // Using the wrong device's stream for memsetAsync on non-P2P GPUs causes
            // CUDA error 700 (illegal memory access) that is only detected at the next
            // stream sync, corrupting the CUDA context.
            int bufferDevice = nativeOps.dbDeviceId(opaque);
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            boolean switchedDevice = false;
            if (bufferDevice >= 0 && bufferDevice != currentDevice) {
                DeviceMemoryManager.getInstance().switchDevice(bufferDevice,
                        "DSP.fastZero", "buffer-device-align");
                switchedDevice = true;
            }

            try {
                // Fetch a fresh stream for the buffer's device.
                Pointer streamToUse = null;
                try {
                    streamToUse = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                } catch (Exception e) {
                    // Fall through to sync path
                }
                if (streamToUse == null || streamToUse.address() == 0) {
                    streamToUse = execStream;
                }

                if (streamToUse != null && streamToUse.address() != 0) {
                    nativeOps.memsetAsync(specialPtr, 0, bytes, 0, streamToUse);
                    int err = nativeOps.lastErrorCode();
                    if (err != 0) {
                        nativeOps.clearLastError();
                        nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
                        int err2 = nativeOps.lastErrorCode();
                        if (err2 != 0) {
                            nativeOps.clearLastError();
                            return false;
                        }
                    }
                } else {
                    nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
                }
                nativeOps.dbTickDeviceWrite(opaque);
            } finally {
                if (switchedDevice) {
                    DeviceMemoryManager.getInstance().switchDevice(currentDevice,
                            "DSP.fastZero", "restore-device");
                }
            }
        } else {
            Pointer primaryPtr = nativeOps.dbPrimaryBuffer(opaque);
            if (primaryPtr != null && primaryPtr.address() != 0) {
                nativeOps.memsetSync(primaryPtr, 0, bytes, 0, null);
            }
        }
        return true;
    }

    private static long numElements(long[] shape) {
        if (shape == null || shape.length == 0) {
            return 1;
        }
        long prod = 1;
        for (long d : shape) {
            prod *= d;
        }
        return prod;
    }

    private static boolean reshapeBuffer(INDArray arr, long[] shape) {
        if (arr == null || shape == null || shape.length == 0) {
            return false;
        }
        if (Arrays.equals(arr.shape(), shape)) {
            return false;
        }
        long[] newStrides = Nd4j.getStrides(shape, arr.ordering());
        int[] intShape = ArrayUtil.toInts(shape);
        int[] intStrides = ArrayUtil.toInts(newStrides);
        ((BaseNDArray) arr).setShapeAndStride(intShape, intStrides);
        ((BaseNDArray) arr).assignNewId();
        return true;
    }

    /**
     * Local buffer pool for intra-execution array reuse.
     */
    private static final class LocalBufferPool {
        private final Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> pools = new EnumMap<>(DataType.class);
        private final double largerArrayMaxMultiple;
        private final Set<INDArray> pooledRefs = Collections.newSetFromMap(new IdentityHashMap<>());
        private boolean lastAcquireReshaped;
        private int releaseAccepted;
        private int releaseRejected;
        private long currentPoolBytes;
        private static final long MAX_POOL_BYTES = Long.parseLong(
                System.getProperty(ND4JSystemProperties.DSP_POOL_MAX_BYTES,
                        String.valueOf(2L * 1024 * 1024 * 1024)));

        private LocalBufferPool() {
            this.largerArrayMaxMultiple = ArrayCacheMemoryMgr.getLargerArrayMaxMultiple().get();
        }

        INDArray acquire(DataType dataType, long[] shape) {
            if (shape == null || shape.length == 0) return null;
            long requiredElements = numElements(shape);
            if (requiredElements <= 0) return null;

            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.get(dataType);
            if (tree == null || tree.isEmpty()) return null;

            long maxElements = (long) (requiredElements * largerArrayMaxMultiple);
            Map.Entry<Long, ArrayDeque<INDArray>> entry = tree.ceilingEntry(requiredElements);
            while (entry != null) {
                long bufferElements = entry.getKey();
                if (bufferElements > maxElements) break;
                ArrayDeque<INDArray> deque = entry.getValue();
                while (deque != null && !deque.isEmpty()) {
                    INDArray arr = deque.poll();
                    if (arr == null) continue;
                    if (deque.isEmpty()) tree.remove(bufferElements);
                    DataBuffer buf = arr.data();
                    // NOTE: Don't gate on buf.closeable() — oversized buffers are OWNED, safe to reuse
                    if (arr.wasClosed() || buf == null || buf.wasClosed() || buf.isConstant()) continue;
                    if (arr.dataType() != dataType) continue;

                    pooledRefs.remove(arr);
                    currentPoolBytes -= bufferElements * dataType.width();
                    lastAcquireReshaped = reshapeBuffer(arr, shape);
                    // Clear stale OpaqueNDArray — see slot cache comment above
                    arr.clearOpaqueNDArray();
                    return arr;
                }
                entry = tree.higherEntry(bufferElements);
            }
            return null;
        }

        boolean release(INDArray arr) {
            if (arr == null || arr.wasClosed()) return false;
            DataBuffer buf = arr.data();
            // NOTE: Don't gate on buf.closeable() — oversized buffers are OWNED, safe to pool
            if (buf == null || buf.wasClosed() || buf.isConstant()) {
                releaseRejected++;
                return false;
            }
            long thisBytes = buf.length() * arr.dataType().width();
            if (currentPoolBytes + thisBytes > MAX_POOL_BYTES) {
                releaseRejected++;
                return false;
            }
            if (!pooledRefs.add(arr)) return false;
            DataType dt = arr.dataType();
            long bufferElements = buf.length();
            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.computeIfAbsent(dt, k -> new TreeMap<>());
            tree.computeIfAbsent(bufferElements, k -> new ArrayDeque<>()).add(arr);
            currentPoolBytes += thisBytes;
            releaseAccepted++;
            return true;
        }

        void flushTo(SessionMemMgr mmgr) {
            try {
                Nd4j.getExecutioner().commit();
            } catch (Exception ignored) {}

            int pooledCount = 0;
            long pooledBytes = 0;
            int closedCount = 0;
            for (TreeMap<Long, ArrayDeque<INDArray>> tree : pools.values()) {
                for (ArrayDeque<INDArray> deque : tree.values()) {
                    for (INDArray arr : deque) {
                        if (arr == null || arr.wasClosed()) continue;
                        DataBuffer buf = arr.data();
                        if (buf == null || buf.wasClosed()) continue;
                        pooledBytes += buf.length() * arr.dataType().width();
                        pooledCount++;
                        // NOTE: Don't gate on buf.closeable() — use isConstant() instead
                        if (!buf.isConstant() && !arr.isView()) {
                            try {
                                buf.close();
                                closedCount++;
                            } catch (Exception ignored) {}
                        }
                    }
                }
            }
            pools.clear();
            pooledRefs.clear();
            currentPoolBytes = 0;
            log.debug("  LocalBufferPool flushTo: {} buffers ({}MB), closed={}, pooled={}, rejected={}",
                    pooledCount, pooledBytes / (1024 * 1024), closedCount,
                    releaseAccepted, releaseRejected);
            releaseAccepted = 0;
            releaseRejected = 0;
        }
    }

    private void printTimingSummary(int opCount, LocalBufferPool localPool) {
        double totalMs = (timingWireInputsNs + timingSyncNs + timingShapeNs +
                timingAllocNs + timingExecNs + timingReleaseNs) / 1_000_000.0;
        log.info("=== DynamicShapePlanExecutor Timing ({} ops, {}ms total) ===",
                opCount, String.format("%.1f", totalMs));
        log.info("  Wire inputs:  {}ms", String.format("%.2f", timingWireInputsNs / 1_000_000.0));
        log.info("  INT/LONG sync: {}ms", String.format("%.2f", timingSyncNs / 1_000_000.0));
        log.info("  Shape calc:   {}ms (hits={}, misses={})",
                String.format("%.2f", timingShapeNs / 1_000_000.0), timingShapeHits, timingShapeMisses);
        log.info("  Mem alloc:    {}ms (cache hits={}, fresh={}, view skips={}, zero skipped={}, zero applied={})",
                String.format("%.2f", timingAllocNs / 1_000_000.0),
                timingPoolHits, timingFreshAllocs, timingViewSkips, timingZeroSkipped, timingZeroApplied);
        if (!timingCacheMissReasons.isEmpty()) {
            log.info("  Cache miss reasons: {}", timingCacheMissReasons);
        }
        if (timingCacheLeakedConstant > 0) {
            log.info("  Cache LEAKED (constant/non-closeable, not freed): {} arrays, {}MB",
                    timingCacheLeakedConstant, timingCacheLeakedConstantBytes / (1024 * 1024));
        }
        log.info("  Native exec:  {}ms", String.format("%.2f", timingExecNs / 1_000_000.0));
        log.info("  Release:      {}ms", String.format("%.2f", timingReleaseNs / 1_000_000.0));
        // Per-op time bucket distribution
        log.info("  Op time distribution: <1ms={}, 1-10ms={}, 10-100ms={}, >100ms={}",
                timingBucketSub1ms, timingBucket1to10ms, timingBucket10to100ms, timingBucketOver100ms);
        // Per-op-type timing histogram (top 20 by total time)
        if (perOpTimingNs != null && !perOpTimingNs.isEmpty()) {
            List<Map.Entry<String, long[]>> sorted = new ArrayList<>(perOpTimingNs.entrySet());
            sorted.sort((a, b) -> Long.compare(b.getValue()[0], a.getValue()[0]));
            int limit = Math.min(sorted.size(), 20);
            log.info("  --- Per-Op Timing (top {} of {} op types) ---", limit, sorted.size());
            for (int i = 0; i < limit; i++) {
                Map.Entry<String, long[]> entry = sorted.get(i);
                long[] s = entry.getValue();
                double opTotalMs = s[0] / 1_000_000.0;
                long opCount2 = s[1];
                double opAvgMs = opTotalMs / opCount2;
                double opMaxMs = s[2] / 1_000_000.0;
                double pctOfTotal = totalMs > 0 ? (opTotalMs / totalMs * 100.0) : 0;
                log.info("    {}: {}ms total ({}%), count={}, avg={}ms, max={}ms",
                        entry.getKey(),
                        String.format("%.1f", opTotalMs),
                        String.format("%.1f", pctOfTotal),
                        opCount2,
                        String.format("%.2f", opAvgMs),
                        String.format("%.1f", opMaxMs));
            }
        }
        int viewProducerCount = 0;
        if (slotIsViewProducer != null) {
            for (boolean b : slotIsViewProducer) if (b) viewProducerCount++;
        }
        log.info("  Pending close: {} buffers ({}MB), viewProducerSlots={}",
                pendingCloseCount, pendingCloseBytes / (1024 * 1024), viewProducerCount);
        if (replicaCount > 0 || replicaCacheHits > 0) {
            log.info("  Cross-device replicas: {} new, {}MB (toDev0: {} {}MB, toDev1: {} {}MB) | {} cached hits",
                    replicaCount, replicaBytes / (1024 * 1024),
                    replicaToDev0Count, replicaToDev0Bytes / (1024 * 1024),
                    replicaToDev1Count, replicaToDev1Bytes / (1024 * 1024),
                    replicaCacheHits);
        }
        if (wrongDeviceCacheEjections > 0) {
            log.info("  Wrong-device cache ejections: {}", wrongDeviceCacheEjections);
        }
        if (shapeBufferCorrections > 0) {
            log.info("  Shape buffer corrections (non-P2P): {}", shapeBufferCorrections);
        }
        // GPU memory pool stats (per-device)
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            LongPointer usedPtr = new LongPointer(1);
            LongPointer reservedPtr = new LongPointer(1);
            nativeOps.getMemoryPoolStats(0, usedPtr, reservedPtr);
            long usedMB = usedPtr.get() / (1024 * 1024);
            long reservedMB = reservedPtr.get() / (1024 * 1024);
            if (cachedNumDevices > 1) {
                LongPointer usedPtr1 = new LongPointer(1);
                LongPointer reservedPtr1 = new LongPointer(1);
                nativeOps.getMemoryPoolStats(1, usedPtr1, reservedPtr1);
                long usedMB1 = usedPtr1.get() / (1024 * 1024);
                long reservedMB1 = reservedPtr1.get() / (1024 * 1024);
                log.info("  GPU memory pool: dev0 used={}MB reserved={}MB, dev1 used={}MB reserved={}MB",
                        usedMB, reservedMB, usedMB1, reservedMB1);
            } else {
                log.info("  GPU memory pool: used={}MB, reserved={}MB", usedMB, reservedMB);
            }
        } catch (Exception e) {
            // Not available on CPU backend
        }
    }

    // =====================================================================================
    // PARALLEL EXECUTION
    // =====================================================================================

    /**
     * Execute the plan in parallel across multiple devices using dependency-driven scheduling.
     * Each device has a dedicated worker thread. Steps become ready when all predecessors complete.
     * Output slots are freed when all consumers have executed.
     */
    private Map<String, INDArray> executeParallel(DynamicShapePlan plan,
                                                    Map<String, INDArray> placeholderArrays,
                                                    NativeOps nativeOps, Pointer cachedExecStream) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int numSteps = slots.length;
        int[] predCounts = plan.getPredecessorCounts();
        int[][] successors = plan.getSuccessors();
        int[] consumerCounts = plan.getConsumerCounts();
        int[] rootSlots = plan.getRootSlots();

        // Shared state
        AtomicIntegerArray predecessorRemaining = new AtomicIntegerArray(numSteps);
        for (int i = 0; i < numSteps; i++) {
            predecessorRemaining.set(i, predCounts[i]);
        }
        AtomicIntegerArray consumerRemaining = new AtomicIntegerArray(plan.getTotalOutputSlots());
        for (int i = 0; i < consumerCounts.length; i++) {
            consumerRemaining.set(i, consumerCounts[i]);
        }

        // outputSlots[] and externalInputs[] are initialized by execute() caller
        // liveSlots is not thread-safe (BitSet); use AtomicIntegerArray as flags instead
        AtomicIntegerArray liveFlags = new AtomicIntegerArray(plan.getTotalOutputSlots());

        // Group slots by target device
        Map<Integer, List<Integer>> deviceSlotMap = new LinkedHashMap<>();
        for (int i = 0; i < numSteps; i++) {
            int dev = slots[i].getTargetDeviceId();
            if (dev < 0) dev = 0;
            deviceSlotMap.computeIfAbsent(dev, k -> new ArrayList<>());
        }
        int numDevices = deviceSlotMap.size();

        // Per-device ready queues
        Map<Integer, BlockingQueue<Integer>> readyQueues = new LinkedHashMap<>();
        for (int dev : deviceSlotMap.keySet()) {
            readyQueues.put(dev, new LinkedBlockingQueue<>());
        }

        // Completion tracking
        CountDownLatch completionLatch = new CountDownLatch(numSteps);
        AtomicReference<Throwable> workerError = new AtomicReference<>();

        // Poison pill for shutdown
        int POISON = -1;

        // Seed root slots into their device queues
        for (int root : rootSlots) {
            int dev = slots[root].getTargetDeviceId();
            if (dev < 0) dev = 0;
            BlockingQueue<Integer> q = readyQueues.get(dev);
            if (q != null) q.offer(root);
        }

        // Create and start device worker threads.
        // Each worker creates its own C++ workspace ON its thread after setting device affinity.
        // The workspace provides bump allocation for C++ op temporaries, preventing glibc
        // heap corruption from buffer overruns in native ops.
        List<DeviceWorker> workers = new ArrayList<>();
        List<Thread> workerThreads = new ArrayList<>();
        for (Map.Entry<Integer, BlockingQueue<Integer>> entry : readyQueues.entrySet()) {
            int deviceId = entry.getKey();
            BlockingQueue<Integer> readyQueue = entry.getValue();
            DeviceWorker worker = new DeviceWorker(
                    deviceId, readyQueue, plan, nativeOps,
                    predecessorRemaining, consumerRemaining, liveFlags,
                    successors, readyQueues, slots,
                    completionLatch, workerError, POISON);
            workers.add(worker);
            Thread t = new Thread(worker, "DSP-DeviceWorker-" + deviceId);
            t.setDaemon(true);
            workerThreads.add(t);
        }

        // Start all workers
        MultiGpuTracer.traceParallelExec("parallel-begin",
                "workers=" + workers.size() + " steps=" + numSteps);
        for (Thread t : workerThreads) {
            t.start();
        }

        // Wait for all steps to complete
        try {
            boolean completed = completionLatch.await(10, TimeUnit.MINUTES);
            if (!completed) {
                throw new RuntimeException("DSP parallel execution timed out after 10 minutes. " +
                        completionLatch.getCount() + " steps remaining.");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("DSP parallel execution interrupted", e);
        }

        // Poison all queues to stop workers
        for (BlockingQueue<Integer> q : readyQueues.values()) {
            q.offer(POISON);
        }

        // Wait for workers to finish cleanup
        for (Thread t : workerThreads) {
            try { t.join(5000); } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Check for errors
        MultiGpuTracer.traceParallelExec("parallel-end",
                "workers=" + workers.size() + " error=" + (workerError.get() != null));
        Throwable err = workerError.get();
        if (err != null) {
            if (err instanceof RuntimeException) throw (RuntimeException) err;
            throw new RuntimeException("DSP parallel execution failed", err);
        }

        // Flush per-device pending buffers
        for (DeviceWorker worker : workers) {
            int prevDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            try {
                MultiGpuTracer.traceParallelExec("flush-device",
                        "device=" + worker.deviceId + " pending=" + worker.devicePendingClose.size());
                DeviceMemoryManager.getInstance().switchDevice(worker.deviceId, "DSP.execute", "worker-flush");
                Nd4j.getExecutioner().commit();
                Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                if (!worker.devicePendingClose.isEmpty()) {
                    pendingClose.addAll(worker.devicePendingClose);
                    if (freshStream != null) {
                        freePendingBuffers(nativeOps, freshStream, null);
                        nativeOps.trimMemoryPoolOnStream(worker.deviceId, freshStream);
                    }
                    pendingClose.clear();
                }
            } finally {
                DeviceMemoryManager.getInstance().switchDevice(prevDev, "DSP.execute", "restore-after-flush");
            }
        }

        // Per-worker workspaces are created/destroyed on the worker threads themselves
        // (in DeviceWorker.run() / finally block), so no cleanup needed here.

        // Collect results — same logic as sequential path
        Nd4j.getExecutioner().commit();
        Map<String, INDArray> results = new LinkedHashMap<>();
        Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
        int viewFlagFixCount = 0;
        for (Map.Entry<String, Integer> entry : outputMap.entrySet()) {
            int slotIdx = entry.getValue();
            INDArray arr = outputSlots[slotIdx];
            if (arr != null) {
                // Fix IS_VIEW flag (same as sequential path)
                if (arr.isView() && arr.data() != null && !arr.data().wasClosed()) {
                    boolean isViewProducer = slotIsViewProducer != null && slotIsViewProducer[slotIdx];
                    long arrLen = arr.length();
                    long dataLen = arr.data().length();
                    boolean lengthView = arrLen < dataLen;
                    boolean flagView = ArrayOptionsHelper.isView(arr.shapeInfoJava());
                    if (!isViewProducer && !lengthView && flagView) {
                        long[] shapeInfo = arr.shapeInfoJava();
                        long options = shapeInfo[shapeInfo.length - 3];
                        options &= ~ArrayOptionsHelper.IS_VIEW;
                        shapeInfo[shapeInfo.length - 3] = options;
                        viewFlagFixCount++;
                    }
                }
                results.put(entry.getKey(), arr);
                outputSlots[slotIdx] = null;
            }
        }
        if (viewFlagFixCount > 0) {
            log.info("  Output view fix (parallel): {} flag-only views fixed", viewFlagFixCount);
        }

        // Clean remaining live slots — group by device for proper per-device cleanup.
        // In multi-GPU execution, slots may hold buffers on different devices.
        // Freeing all on device 0's stream could cause issues with cross-device cleanup.
        if (cachedNumDevices <= 1) {
            // Single-GPU: simple path
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && liveFlags.get(i) == 1) {
                    boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[i];
                    if (!isViewSlot) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            pendingClose.add(buf);
                        }
                    }
                    outputSlots[i] = null;
                }
            }
            closePendingBuffers(nativeOps, cachedExecStream);
        } else {
            // Multi-GPU: group remaining buffers by their planned target device
            DynamicShapeSlot[] planSlots = plan.getSlots();
            Map<Integer, List<DataBuffer>> perDeviceBuffers = new HashMap<>();
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && liveFlags.get(i) == 1) {
                    boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[i];
                    if (!isViewSlot) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            int devId = (i < planSlots.length) ? planSlots[i].getTargetDeviceId() : 0;
                            if (devId < 0) devId = 0;
                            perDeviceBuffers.computeIfAbsent(devId, k -> new ArrayList<>()).add(buf);
                        }
                    }
                    outputSlots[i] = null;
                }
            }
            // Free per-device with proper device context and stream
            int mainDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            for (Map.Entry<Integer, List<DataBuffer>> entry : perDeviceBuffers.entrySet()) {
                int devId = entry.getKey();
                DeviceMemoryManager.getInstance().switchDevice(devId, "DSP.close", "per-device-cleanup");
                Nd4j.getExecutioner().commit();
                Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                pendingClose.addAll(entry.getValue());
                freePendingBuffers(nativeOps, freshStream, null);
                nativeOps.trimMemoryPoolOnStream(devId, freshStream);
                pendingClose.clear();
            }
            DeviceMemoryManager.getInstance().switchDevice(mainDevice, "DSP.close", "restore-after-cleanup");
            // Final close for any deferred buffers
            closePendingBuffers(nativeOps, cachedExecStream);
        }

        if (TIMING_ENABLED) {
            printTimingSummary(numSteps, localPool);
        }

        return results;
    }

    /**
     * Per-device worker for async DSP execution. Each worker runs on a dedicated thread
     * with GPU device affinity. Workers poll a ready queue for steps whose predecessors
     * have all completed, execute them, and notify successors.
     */
    private class DeviceWorker implements Runnable {
        final int deviceId;
        final BlockingQueue<Integer> readyQueue;
        final DynamicShapePlan plan;
        final NativeOps nativeOps;
        final AtomicIntegerArray predecessorRemaining;
        final AtomicIntegerArray consumerRemaining;
        final AtomicIntegerArray liveFlags;
        final int[][] successors;
        final Map<Integer, BlockingQueue<Integer>> allReadyQueues;
        final DynamicShapeSlot[] slots;
        final CountDownLatch completionLatch;
        final AtomicReference<Throwable> workerError;
        final int POISON;

        // Per-device state (no cross-thread contention)
        final ArrayList<DataBuffer> devicePendingClose = new ArrayList<>();
        final Set<DataBuffer> deviceSeenIdentity = Collections.newSetFromMap(new IdentityHashMap<>());
        final HashSet<Long> deviceClosedOdbAddresses = new HashSet<>();
        final ArrayDeque<OpContext> deviceCtxPool = new ArrayDeque<>();
        final Map<Integer, INDArray> deviceConstantReplicaCache = new HashMap<>();
        Pointer workerWorkspace;  // Created on worker thread in run()

        DeviceWorker(int deviceId, BlockingQueue<Integer> readyQueue, DynamicShapePlan plan,
                     NativeOps nativeOps, AtomicIntegerArray predecessorRemaining,
                     AtomicIntegerArray consumerRemaining, AtomicIntegerArray liveFlags,
                     int[][] successors, Map<Integer, BlockingQueue<Integer>> allReadyQueues,
                     DynamicShapeSlot[] slots, CountDownLatch completionLatch,
                     AtomicReference<Throwable> workerError, int POISON) {
            this.deviceId = deviceId;
            this.readyQueue = readyQueue;
            this.plan = plan;
            this.nativeOps = nativeOps;
            this.predecessorRemaining = predecessorRemaining;
            this.consumerRemaining = consumerRemaining;
            this.liveFlags = liveFlags;
            this.successors = successors;
            this.allReadyQueues = allReadyQueues;
            this.slots = slots;
            this.completionLatch = completionLatch;
            this.workerError = workerError;
            this.POISON = POISON;
        }

        @Override
        public void run() {
            try {
                // Set device affinity for this thread
                MultiGpuTracer.traceDeviceSwitch("DSP.DeviceWorker.run",
                        -1, deviceId, "worker-init");
                DeviceMemoryManager.getInstance().switchDevice(deviceId, "DSP.DeviceWorker", "worker-init");

                // Do NOT skip device coherency — OOM failover can place outputs on a different
                // device than the worker, and subsequent ops need coherency to migrate inputs.
                // The coherency check is lightweight (just device ID comparison) and no-ops when
                // all arrays are already on the correct device.

                // Create per-worker workspace ON this thread (after device affinity set).
                // Must be created here, not on main thread, so cudaHostAlloc happens
                // in the correct CUDA device context. HOST-ONLY (0 GPU, 8MB host).
                try {
                    this.workerWorkspace = nativeOps.createNativeWorkspace(8 * 1024 * 1024);
                    log.info("DeviceWorker[{}] created workspace: {}", deviceId,
                            workerWorkspace != null ? workerWorkspace.address() : "null");
                } catch (Exception e) {
                    log.warn("DeviceWorker[{}] failed to create workspace: {}", deviceId, e.getMessage());
                    this.workerWorkspace = null;
                }

                // Get execution stream for this device
                Pointer execStream = null;
                try {
                    OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                    if (lc != null) {
                        execStream = nativeOps.lcExecutionStream(lc);
                        if (execStream != null) {
                            execStream.retainReference();
                            MultiGpuTracer.traceStreamOp("retain", Thread.currentThread().getName(),
                                    deviceId, execStream.address());
                        }
                    }
                } catch (Exception e) { /* CPU backend */ }
                while (workerError.get() == null) {
                    Integer stepIdx;
                    try {
                        stepIdx = readyQueue.poll(100, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                    if (stepIdx == null) continue;
                    if (stepIdx == POISON) break;

                    DynamicShapeSlot slot = slots[stepIdx];

                    // Get or create OpContext
                    OpContext ctx = deviceCtxPool.pollFirst();
                    if (ctx == null) {
                        ctx = Nd4j.getExecutioner().buildContext();
                    }
                    ctx.purge();

                    try {
                        MultiGpuTracer.traceOpExec(stepIdx, deviceId, slot.getOpName(), null, null);
                        if (SERIAL_EXEC) {
                            synchronized (EXEC_LOCK) {
                                executeSlotForWorker(slot, ctx, nativeOps, execStream, stepIdx, this.workerWorkspace);
                            }
                        } else {
                            executeSlotForWorker(slot, ctx, nativeOps, execStream, stepIdx, this.workerWorkspace);
                        }
                    } catch (Exception e) {
                        log.error("DeviceWorker[{}] error at step {} ({}): {}",
                                deviceId, stepIdx, slot.getOpName(), e.getMessage(), e);
                        workerError.compareAndSet(null, e);
                        // Count down remaining steps to unblock the latch
                        completionLatch.countDown();
                        // Poison all queues
                        for (BlockingQueue<Integer> q : allReadyQueues.values()) {
                            q.offer(POISON);
                        }
                        break;
                    }

                    ctx.purgeForReuse();
                    deviceCtxPool.offerFirst(ctx);

                    // Notify successors
                    int[] succs = successors[stepIdx];
                    for (int succ : succs) {
                        if (predecessorRemaining.decrementAndGet(succ) == 0) {
                            // Successor is ready — route to its device's queue
                            int succDev = slots[succ].getTargetDeviceId();
                            if (succDev < 0) succDev = 0;
                            BlockingQueue<Integer> succQueue = allReadyQueues.get(succDev);
                            if (succQueue != null) {
                                succQueue.offer(succ);
                            }
                        }
                    }

                    // Decrement consumer counts for consumed output slots, release when done
                    int[] inputSrcIndices = slot.getInputSourceIndices();
                    for (int srcIdx : inputSrcIndices) {
                        if (srcIdx >= 0) {
                            if (consumerRemaining.decrementAndGet(srcIdx) == 0) {
                                // Slot is dead — release it
                                INDArray arr = outputSlots[srcIdx];
                                if (arr != null && liveFlags.compareAndSet(srcIdx, 1, 0)) {
                                    DataBuffer buf = arr.data();
                                    if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                                        boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[srcIdx];
                                        if (!isViewSlot) {
                                            // Cache for reuse or pending close
                                            if (slotArrayCache != null) {
                                                INDArray prev = slotArrayCache[srcIdx];
                                                if (prev != null && !prev.wasClosed()) {
                                                    DataBuffer pbuf = prev.data();
                                                    if (pbuf != null && !pbuf.wasClosed() && !pbuf.isConstant()) {
                                                        devicePendingClose.add(pbuf);
                                                    }
                                                }
                                                slotArrayCache[srcIdx] = arr;
                                            } else {
                                                devicePendingClose.add(buf);
                                            }
                                        }
                                    }
                                    outputSlots[srcIdx] = null;
                                }
                            }
                        }
                    }

                    // Periodic flush
                    if (devicePendingClose.size() >= RELEASE_FLUSH_INTERVAL) {
                        Nd4j.getExecutioner().commit();
                        Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
                        if (freshStream != null) {
                            for (DataBuffer buf : devicePendingClose) {
                                if (buf == null || buf.wasClosed() || buf.isConstant()) continue;
                                OpaqueDataBuffer odb = buf.opaqueBuffer();
                                if (odb == null || odb.isNull()) continue;
                                if (!deviceSeenIdentity.add(buf)) continue;
                                long odbAddr = odb.address();
                                if (odbAddr != 0 && !deviceClosedOdbAddresses.add(odbAddr)) continue;
                                try {
                                    nativeOps.dbFreeBuffersOnStream(odb, freshStream);
                                    odb.tryMarkForDeallocation();
                                    odb.setNull();
                                    OpaqueDataBufferDeallocator deallocator = odb.getDeallocator();
                                    if (deallocator != null) deallocator.markDeallocated();
                                } catch (Exception e) {
                                    log.warn("DeviceWorker[{}] flush failed: {}", deviceId, e.getMessage());
                                }
                            }
                            devicePendingClose.clear();
                            nativeOps.trimMemoryPoolOnStream(deviceId, freshStream);
                        }
                    }

                    completionLatch.countDown();
                }
            } catch (Throwable t) {
                workerError.compareAndSet(null, t);
                // Unblock latch
                while (completionLatch.getCount() > 0) completionLatch.countDown();
            } finally {
                // Destroy workspace created on this thread
                if (this.workerWorkspace != null) {
                    try {
                        nativeOps.destroyNativeWorkspace(this.workerWorkspace);
                    } catch (Exception e) {
                        log.debug("DeviceWorker[{}] workspace cleanup failed: {}", deviceId, e.getMessage());
                    }
                    this.workerWorkspace = null;
                }
            }
        }

        /**
         * Execute a single slot on this worker's device. Reuses the core executeSlot logic
         * but without device save/restore (worker is already on the correct device).
         */
        private void executeSlotForWorker(DynamicShapeSlot slot, OpContext ctx, NativeOps nativeOps,
                                           Pointer execStream, int stepIdx, Pointer workerWorkspace) {
            DifferentialFunction fn = slot.getOp();
            int targetDevice = deviceId;
            List<DataBuffer> replicatedInputBuffers = null;

            // Step 1: Wire inputs
            int[] inputSourceIndices = slot.getInputSourceIndices();
            INDArray[] inputArrays = new INDArray[inputSourceIndices.length];

            for (int i = 0; i < inputSourceIndices.length; i++) {
                int srcIdx = inputSourceIndices[i];
                if (srcIdx >= 0) {
                    inputArrays[i] = outputSlots[srcIdx];
                } else {
                    int extIdx = -(srcIdx + 1);
                    inputArrays[i] = externalInputs[extIdx];
                }
                if (inputArrays[i] == null) {
                    throw new IllegalStateException("Null input at index " + i + " for op " + slot.getOpName() +
                            ", input var: " + slot.getInputVarNames()[i] + " (step " + stepIdx + ")");
                }
            }

            // Step 1b: Migrate inputs to target device if cross-device
            for (int i = 0; i < inputArrays.length; i++) {
                INDArray input = inputArrays[i];
                if (input != null && !input.isEmpty() && input.data() != null) {
                    int inputDevice = -1;
                    OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                    if (inputOdb != null && !inputOdb.isNull()) {
                        inputDevice = nativeOps.dbDeviceId(inputOdb);
                    }
                    if (inputDevice >= 0 && inputDevice != targetDevice) {
                        boolean isConstant = input.data() != null && input.data().isConstant();
                        int srcIdx2 = inputSourceIndices[i];
                        boolean isExternal = srcIdx2 < 0;
                        int cacheKey = isExternal ? ((-(srcIdx2 + 1)) << 16) | targetDevice : -1;

                        // Check constant replica cache
                        INDArray cachedReplica = null;
                        if (isConstant && isExternal) {
                            cachedReplica = deviceConstantReplicaCache.get(cacheKey);
                            if (cachedReplica != null && !cachedReplica.wasClosed()) {
                                inputArrays[i] = cachedReplica;
                                MultiGpuTracer.traceInputMigration(stepIdx, i, inputDevice, targetDevice,
                                        input.length() * input.data().getElementSize(),
                                        input.isView(), true, true);
                                continue;
                            }
                        }

                        long inputBytes = input.length() * input.data().getElementSize();
                        MultiGpuTracer.traceInputMigration(stepIdx, i, inputDevice, targetDevice,
                                inputBytes, input.isView(), isConstant, false);

                        // CRITICAL: dup() must run on the SOURCE device — non-P2P devices
                        // can't cross-access GPU memory. Switch to source device for dup(),
                        // then restore to target device for replication.
                        INDArray inputToReplicate = input;
                        if (input.isView()) {
                            DeviceMemoryManager.getInstance().switchDevice(inputDevice, "DSP.DeviceWorker", "worker-view-dup-source");
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                inputToReplicate = input.dup(input.ordering());
                            }
                            DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DSP.DeviceWorker", "worker-view-dup-restore");
                            DataBuffer dupBuf = inputToReplicate.data();
                            if (dupBuf != null && !dupBuf.isConstant()) {
                                if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                replicatedInputBuffers.add(dupBuf);
                            }
                        }
                        INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                        inputArrays[i] = replica;

                        if (isConstant && isExternal) {
                            deviceConstantReplicaCache.put(cacheKey, replica);
                        } else {
                            DataBuffer replicaBuf = replica.data();
                            if (replicaBuf != null && !replicaBuf.isConstant()) {
                                if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                replicatedInputBuffers.add(replicaBuf);
                            }
                        }
                    }
                }
            }
            ctx.setInputArrays(inputArrays);

            // Step 2: Sync INT/LONG inputs if needed
            if (slot.isNeedsIntLongSync()) {
                syncIntLongInputs(inputArrays, slot.isDataDependent(), nativeOps);
            }

            // Step 3: Compute output shapes
            List<DataBuffer> outShapes = getOrComputeShapes(slot, ctx, fn, inputArrays, nativeOps);
            if (outShapes == null || outShapes.isEmpty()) {
                throw new IllegalStateException("No output shapes for op " + slot.getOpName());
            }

            // Step 4: Allocate outputs
            int[] outputSlotIndices = slot.getOutputSlotIndices();
            INDArray[] outputArrays = new INDArray[outShapes.size()];

            for (int i = 0; i < outShapes.size(); i++) {
                DataBuffer shapeBuffer = outShapes.get(i);
                long[] shapeInfo = shapeBuffer.asLong();
                DataType dt = Shape.dataType(shapeInfo);
                long[] actualShape = Shape.shape(shapeInfo);

                INDArray out = null;
                int slotIdx = (i < outputSlotIndices.length) ? outputSlotIndices[i] : -1;

                // View-producer optimization (same as sequential)
                if (slotIdx >= 0 && slotIsViewProducer != null && slotIsViewProducer[slotIdx]
                        && !outputSlotSet.get(slotIdx) && slot.isCustomOp()) {
                    out = Nd4j.empty(dt);
                    outputArrays[i] = out;
                    outputSlots[slotIdx] = out;
                    liveFlags.set(slotIdx, 1);
                    continue;
                }

                if (Shape.isEmpty(shapeInfo) || numElements(actualShape) == 0) {
                    out = Nd4j.emptyWithShape(actualShape, dt);
                } else {
                    // Try slot cache
                    if (slotIdx >= 0 && slotArrayCache != null) {
                        INDArray cached = slotArrayCache[slotIdx];
                        if (cached != null && !cached.wasClosed()) {
                            DataBuffer cbuf = cached.data();
                            if (cbuf != null && !cbuf.wasClosed()
                                    && cached.dataType() == dt
                                    && cbuf.length() >= numElements(actualShape)) {
                                boolean wrongDevice = false;
                                if (targetDevice >= 0) {
                                    OpaqueDataBuffer cachedOdb = cbuf.opaqueBuffer();
                                    if (cachedOdb != null && !cachedOdb.isNull()) {
                                        int cachedDevice = nativeOps.dbDeviceId(cachedOdb);
                                        if (cachedDevice >= 0 && cachedDevice != targetDevice) {
                                            wrongDevice = true;
                                        }
                                    }
                                }
                                if (wrongDevice) {
                                    devicePendingClose.add(cbuf);
                                    slotArrayCache[slotIdx] = null;
                                } else {
                                    reshapeBuffer(cached, actualShape);
                                    cached.clearOpaqueNDArray();
                                    if (fastZero(cached, nativeOps, execStream)) {
                                        out = cached;
                                        slotArrayCache[slotIdx] = null;
                                    } else {
                                        slotArrayCache[slotIdx] = null;
                                    }
                                }
                            }
                        }
                        if (out == null && cached != null && !cached.wasClosed()) {
                            DataBuffer cbuf = cached.data();
                            if (cbuf != null && !cbuf.wasClosed() && !cbuf.isConstant()) {
                                devicePendingClose.add(cbuf);
                            }
                            slotArrayCache[slotIdx] = null;
                        }
                    }
                    if (out == null) {
                        if (slotIdx >= 0 && outputSlotSet.get(slotIdx)) {
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                out = Nd4j.create(dt, actualShape);
                            }
                        } else if (slotIdx >= 0 && slotArrayCache != null) {
                            out = allocateForSlotCache(dt, actualShape);
                        } else {
                            out = allocateWithHeadroom(dt, actualShape);
                        }
                    }
                }
                outputArrays[i] = out;
                if (slotIdx >= 0) {
                    outputSlots[slotIdx] = outputArrays[i];
                    liveFlags.set(slotIdx, 1);
                }
            }
            ctx.setOutputArrays(outputArrays);

            // Save GPU addresses for view detection (skip once detection is complete)
            long[] preExecGpuAddrs = null;
            if (!viewProducerDetectionDone) {
                preExecGpuAddrs = new long[outputArrays.length];
                for (int i = 0; i < outputArrays.length; i++) {
                    INDArray arr = outputArrays[i];
                    if (arr != null && !arr.isEmpty()) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed()) {
                            OpaqueDataBuffer odb = buf.opaqueBuffer();
                            if (odb != null && !odb.isNull()) {
                                Pointer special = nativeOps.dbSpecialBuffer(odb);
                                if (special != null) preExecGpuAddrs[i] = special.address();
                            }
                        }
                    }
                }
            }

            // Step 5: Execute
            nativeOps.clearLastError();
            ctx.shapeFunctionOverride(SHAPE_OVERRIDE);

            // Attach per-worker workspace for C++ op temporaries.
            // Each DeviceWorker creates its own workspace on its thread (in run()),
            // so there's no cross-thread contention on the bump allocator offset.
            // Without workspace, C++ ops use aligned_alloc on the glibc heap, and
            // buffer overruns corrupt adjacent heap metadata → "double free or corruption" crash.
            //
            // scopeIn() resets the bump offset so each op reuses the same workspace memory.
            // Workspace is detached immediately after execution to prevent shape computation
            // from allocating shape buffers in the workspace (ShapeList::destroy() would
            // call delete[] on workspace pointers → SIGSEGV).
            if (workerWorkspace != null) {
                nativeOps.workspaceScopeIn(workerWorkspace);
                ctx.attachWorkspace(workerWorkspace);
            }

            if (slot.isCustomOp()) {
                ctx.setIArguments(slot.getIArgs());
                ctx.setTArguments(slot.getTArgs());
                ctx.setBArguments(slot.getBArgs());
                ctx.setDArguments(slot.getDArgs());
                ctx.setSArguments(slot.getSArgs() == null ? new String[0] : slot.getSArgs());
                Nd4j.exec((CustomOp) fn, ctx);
            } else {
                Nd4j.exec((Op) fn, ctx);
            }

            // Detach workspace + scopeOut immediately after execution.
            // purge()/purgeForReuse() do NOT clear workspace attachment.
            if (workerWorkspace != null) {
                ctx.detachWorkspace();
                nativeOps.workspaceScopeOut(workerWorkspace);
            }

            // View-producer detection
            List<INDArray> ctxOutputs = ctx.getOutputArrays();
            int maxTracked = Math.min(ctxOutputs != null ? ctxOutputs.size() : 0, outputSlotIndices.length);
            if (ctxOutputs != null) {
                for (int i = 0; i < maxTracked; i++) {
                    INDArray ctxOut = ctxOutputs.get(i);
                    int si = outputSlotIndices[i];
                    if (ctxOut == null || si < 0) continue;

                    if (ctxOut != outputArrays[i]) {
                        if (slotIsViewProducer != null) slotIsViewProducer[si] = true;
                        if (!outputArrays[i].isEmpty()) {
                            DataBuffer buf = outputArrays[i].data();
                            if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                                devicePendingClose.add(buf);
                            }
                        }
                        outputSlots[si] = ctxOut;
                    } else if (!viewProducerDetectionDone
                            && slotIsViewProducer != null && !slotIsViewProducer[si]
                            && preExecGpuAddrs != null) {
                        long preAddr = (i < preExecGpuAddrs.length) ? preExecGpuAddrs[i] : 0;
                        if (preAddr != 0) {
                            DataBuffer currentBuf = ctxOut.data();
                            OpaqueDataBuffer currentOdb = (currentBuf != null) ? currentBuf.opaqueBuffer() : null;
                            long currentAddr = 0;
                            if (currentOdb != null && !currentOdb.isNull()) {
                                Pointer special = nativeOps.dbSpecialBuffer(currentOdb);
                                if (special != null) currentAddr = special.address();
                            }
                            if (currentAddr != 0 && preAddr != currentAddr) {
                                slotIsViewProducer[si] = true;
                            }
                        }
                    }
                }
            }

            // Release untracked outputs
            for (int i = 0; i < outputArrays.length; i++) {
                boolean tracked = (i < outputSlotIndices.length && outputSlotIndices[i] >= 0);
                if (!tracked) {
                    INDArray arr = outputArrays[i];
                    if (ctxOutputs != null && i < ctxOutputs.size()) {
                        INDArray ctxOut = ctxOutputs.get(i);
                        if (ctxOut != null && ctxOut != arr) arr = ctxOut;
                    }
                    if (arr != null) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            devicePendingClose.add(buf);
                        }
                    }
                }
            }

            // Release replicated input copies
            if (replicatedInputBuffers != null) {
                for (DataBuffer buf : replicatedInputBuffers) {
                    if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                        devicePendingClose.add(buf);
                    }
                }
            }
        }
    }

    /**
     * Evict slot cache if total cached memory exceeds threshold. After a prefill step
     * (large seq_len), cached arrays hold GBs of GPU memory that won't be reused by
     * decode steps (seq_len=1). Without eviction, CUDA stream creation on new threads
     * fails with cudaErrorMemoryAllocation because the pool reservation is too large.
     *
     * Threshold: 512MB. Decode step caches are typically < 10MB, so this only fires
     * after prefill steps.
     */
    private void evictOversizedSlotCache(NativeOps nativeOps, Pointer execStream) {
        if (slotArrayCache == null) return;

        // Estimate total cached bytes
        long totalCachedBytes = 0;
        int cachedCount = 0;
        for (INDArray arr : slotArrayCache) {
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed()) {
                    totalCachedBytes += buf.length() * buf.getElementSize();
                    cachedCount++;
                }
            }
        }

        // Only evict when cache exceeds threshold. Cached arrays' GPU memory is valid
        // (pool tracks them as live allocations). The only issue was stale CUDA streams
        // (error 400) during fastZero, which is now handled by sync memset fallback.
        long thresholdBytes = 512L * 1024 * 1024; // 512MB
        if (totalCachedBytes <= thresholdBytes) {
            return;
        }

        int evicted = 0;
        for (int i = 0; i < slotArrayCache.length; i++) {
            INDArray arr = slotArrayCache[i];
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                    pendingClose.add(buf);
                }
                evicted++;
            }
            slotArrayCache[i] = null;
        }

        if (!pendingClose.isEmpty()) {
            // Merge deferred buffers from mid-execution flushes
            if (!deferredClose.isEmpty()) {
                pendingClose.addAll(deferredClose);
                deferredClose.clear();
            }
            Nd4j.getExecutioner().commit();
            // Re-fetch a fresh stream pointer. The original execStream may have become
            // stale: intermediate JNI calls during execution can trigger ContextBuffers::release()
            // which frees the underlying cudaStream_t. Dereferencing a stale stream pointer
            // in cudaFreeAsync causes SIGSEGV in the CUDA driver.
            Pointer freshStream = DeviceMemoryManager.getInstance().getFreshExecutionStream();
            freePendingBuffers(nativeOps, freshStream, null);
            pendingClose.clear();
            // Trim pool so freed memory is available
            if (freshStream != null) {
                int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                nativeOps.trimMemoryPoolOnStream(currentDevice, freshStream);
                for (int d = 0; d < cachedNumDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPool(d);
                    }
                }
            }
        }

        log.debug("Slot cache eviction: evicted={}/{} entries, was {}MB (threshold {}MB)",
                evicted, cachedCount, totalCachedBytes / (1024 * 1024),
                thresholdBytes / (1024 * 1024));
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
                if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                    pendingClose.add(buf);
                    collected++;
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
            log.debug("Native executor: plan not precompiled for native execution");
            return null;
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
            for (int i = 0; i < extKeys.length; i++) {
                if (extInputs[i] != null) {
                    DataBuffer db = extInputs[i].data();
                    if (db == null || db.wasClosed()) {
                        SDVariable var = sd.getVariable(extKeys[i]);
                        if (var != null && (var.getVariableType() == VariableType.CONSTANT
                                || var.getVariableType() == VariableType.VARIABLE)) {
                            INDArray fresh = var.getArr();
                            if (fresh != null) {
                                extInputs[i] = fresh;
                                cachedInputArrays[i] = fresh;
                            }
                        }
                    }
                }
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
                    log.warn("Native executor: missing external input '{}', falling back to Java", varName);
                    return null;
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
                // First call: determine majority device by data volume
                long[] deviceBytes = new long[numDevices];
                for (INDArray arr : extInputs) {
                    if (arr != null && arr.data() != null && !arr.data().wasClosed()) {
                        int devId = nOps.dbDeviceId(arr.data().opaqueBuffer());
                        if (devId >= 0 && devId < numDevices) {
                            deviceBytes[devId] += arr.length() * arr.data().getElementSize();
                        }
                    }
                }
                int bestDevice = 0;
                long bestBytes = deviceBytes[0];
                for (int d = 1; d < numDevices; d++) {
                    if (deviceBytes[d] > bestBytes) {
                        bestDevice = d;
                        bestBytes = deviceBytes[d];
                    }
                }
                nativeExecutionDevice = bestDevice;
                if (nativeExecutionDevice != previousDevice) {
                    log.info("DSP native executor: majority device={} ({}MB), switching from device {}",
                            nativeExecutionDevice,
                            bestBytes / (1024 * 1024),
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
                            // Check constant replica cache first — model weights don't change
                            boolean isConstant = arr.data().isConstant();
                            boolean isPlaceholder = placeholderArrays != null
                                    && extKeys[i] != null && placeholderArrays.containsKey(extKeys[i]);
                            // Placeholders poisoned by setCloseable(false) appear constant —
                            // never cache them (stale shapes from previous step)
                            boolean isTrulyConstant = isConstant && !isPlaceholder;

                            if (isTrulyConstant) {
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

                            // Cache constant replicas for reuse across decode steps
                            if (isTrulyConstant) {
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
                    log.info("DSP native executor: migrated {} inputs ({}MB) to device {}",
                            migratedCount, migratedBytes / (1024 * 1024), nativeExecutionDevice);
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
                        if (arr != cachedInputArrays[pi]) {
                            // Placeholder array identity changed — re-set input (rare)
                            OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(arr);
                            nativeOps.setGraphContextInputArray(opContext, pi, opaqueIn);
                            cachedInputOpaques[pi] = opaqueIn;
                            cachedInputArrays[pi] = arr;
                        } else if (!arr.isEmpty() && arr.data() != null && !arr.data().wasClosed()) {
                            // Same array, sync to device (placeholder may have been modified on host)
                            OpaqueDataBuffer odb = arr.data().opaqueBuffer();
                            if (odb != null && !odb.isNull()) {
                                odb.syncToSpecial();
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
                        OpaqueDataBuffer dstOdb = cached.data().opaqueBuffer();
                        if (dstOdb != null) {
                            nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
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
                    OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                    if (dstOdb != null) {
                        nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
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
     * Free the native plan handle if it exists.
     */
    private void freeNativePlanHandle() {
        // Free cached OpaqueContext first (it references the plan)
        if (cachedOpContext != null) {
            log.info("    freeNativePlanHandle: deleteGraphContext");
            System.out.flush(); System.err.flush();
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.deleteGraphContext(cachedOpContext);
            } catch (Exception ignored) {}
            cachedOpContext = null;
        }
        if (nativePlanHandle != null && !nativePlanHandle.isNull()) {
            log.info("    freeNativePlanHandle: freeDynamicShapePlan (handle={})", nativePlanHandle);
            System.out.flush(); System.err.flush();
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.freeDynamicShapePlan(nativePlanHandle);
                log.info("    freeNativePlanHandle: freeDynamicShapePlan DONE");
                System.out.flush(); System.err.flush();
            } catch (Exception e) {
                log.info("Error freeing native plan handle: {}", e.getMessage());
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
        // Clear native constant replica cache — replicas reference the old plan's device
        if (nativeConstantReplicaCache != null) {
            nativeConstantReplicaCache.clear();
            nativeConstantReplicaCache = null;
        }
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
        log.info("  DSP close() step 1: flushTo");
        System.out.flush(); System.err.flush();
        if (localPool != null) {
            localPool.flushTo(mmgr);
            localPool = null;
        }
        log.info("  DSP close() step 2: closeSlotArrayCache");
        System.out.flush(); System.err.flush();
        closeSlotArrayCache();
        log.info("  DSP close() step 3: constant replicas ({})", constantReplicaCache != null ? constantReplicaCache.size() : 0);
        System.out.flush(); System.err.flush();
        if (constantReplicaCache != null) {
            constantReplicaCache.clear();
            constantReplicaCache = null;
        }
        if (nativeConstantReplicaCache != null) {
            log.info("  DSP close() step 3b: native constant replicas ({})", nativeConstantReplicaCache.size());
            nativeConstantReplicaCache.clear();
            nativeConstantReplicaCache = null;
        }
        log.info("  DSP close() step 4: ctxPool ({})", ctxPool.size());
        System.out.flush(); System.err.flush();
        ctxPool.clear();

        log.info("  DSP close() step 5: outputSlots");
        System.out.flush(); System.err.flush();
        if (outputSlots != null) {
            Arrays.fill(outputSlots, null);
        }
        if (externalInputs != null) {
            Arrays.fill(externalInputs, null);
        }
        // Destroy self-managed native workspace if we created one
        if (ownNativeWorkspace != null) {
            log.info("  DSP close() step 5b: destroyNativeWorkspace");
            System.out.flush(); System.err.flush();
            try {
                NativeOps wsOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                wsOps.workspaceScopeOut(ownNativeWorkspace);
                wsOps.destroyNativeWorkspace(ownNativeWorkspace);
            } catch (Exception ignored) {}
            ownNativeWorkspace = null;
        }

        // Free cached output wrappers
        if (zeroCopyOutputCache != null) {
            log.info("  DSP close() step 6: zeroCopyOutputCache ({} entries)", zeroCopyOutputCache.size());
            System.out.flush(); System.err.flush();
            for (INDArray arr : zeroCopyOutputCache.values()) {
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            zeroCopyOutputCache = null;
        }

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
            log.info("  DSP close() step 7: deleteGraphContext");
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
        log.info("  DSP close() step 8: freeNativePlanHandle");
        System.out.flush(); System.err.flush();
        freeNativePlanHandle();

        currentPlan = null;
        externalConstantsResolved = false;
        // Clear saved KV retention params — executor is fully closed, no re-apply possible
        savedKvPresentOutputNames = null;
        savedKvPastInputNames = null;
        log.info("  DSP close() complete");
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
