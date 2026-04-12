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

package org.nd4j.autodiff.samediff.internal;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.execution.ExecutionNode;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.DAGCache;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.CleanupDiagnostics;
import org.nd4j.autodiff.samediff.internal.memory.HashDependencyTracker;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.VariableUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.*;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.bytedeco.javacpp.LongPointer;

import org.nd4j.shade.wstx.util.StringUtil;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray, Pair<SameDiffOp,OpContext>> {
    private static final String SCOPE_PANIC_MSG = "If required, arrays in workspaces can be detached using INDArray.detach() before being passed to the SameDiff instance.\n" +
            "Alternatively, arrays defined in a workspace must be replaced after the workspace has been closed.";


    // Ops with data-dependent output shapes that can't be cached even with non-INT/LONG inputs.
    // Where: output length depends on number of true elements in BOOL input.
    // unique: output length depends on number of unique values.
    // Ops with data-dependent output shapes (shape depends on input VALUES, not just input shapes)
    // These ops are never cacheable and always need INT/LONG input sync.
    private static final Set<String> DATA_DEPENDENT_OUTPUT_OPS = Set.of("Where", "unique");

    protected static final String KERAS_TRAIN_TEST = "keras_learning_phase";
    //freed array ids to track for allocation, sometimes SDValues contain dup arrays that get freed twice.
    //we track the ids to avoid double frees
    protected final ThreadLocal<Set<Long>> freedArraysTl = ThreadLocal.withInitial(LinkedHashSet::new);

    @Getter
    @Setter
    private SessionMemMgr mmgr;     //Used for allocating and deallocating memory
    /**
     * Array use tracker: What needs to happen before the array can be closed/released?
     * As the name suggests, the INDArrays are tracked using object identity, not equality
     */
    private final ThreadLocal<AbstractDependencyTracker<SDValue, Dep>> arrayUseTrackerTl =
            ThreadLocal.withInitial(HashDependencyTracker::new);


    // Use per-thread OpContexts to prevent sharing across concurrent execution.
    private final ThreadLocal<Map<String,OpContext>> opContextsTl =
            ThreadLocal.withInitial(ConcurrentHashMap::new);

    // Thread-local OpContext pool: reuse OpContexts via purge() instead of close()/rebuild.
    // This avoids the native alloc/dealloc cost of creating a new OpContext per op per step.
    private final ThreadLocal<Deque<OpContext>> opContextPoolTl =
            ThreadLocal.withInitial(ArrayDeque::new);

    // DAG cache for avoiding expensive convergence process
    protected final DAGCache dagCache = new DAGCache();

    // Cached inputs list to avoid iterating all variables on every output() call
    private volatile List<String> cachedInputsList;

    // Cached constant/variable values to avoid repeated PatriciaTrie lookups on every output() call.
    // Key is a DAGCache.CacheKey-like identifier; value is a pre-built map of const/var name -> SDValue.
    private volatile Map<String, SDValue> cachedConstVarValues;

    // Shape cache: keyed by (opName hash + input shapes hash) to avoid redundant calculateOutputShape JNI calls.
    // For autoregressive generation, most ops have identical input shapes across decode steps.
    private final ThreadLocal<Map<Long, List<DataBuffer>>> outputShapeCacheTl =
            ThreadLocal.withInitial(HashMap::new);

    // Output array cache: reuse intermediate arrays whose shapes don't change between steps.
    // Keyed by variable name. Only non-output (intermediate) arrays are cached.
    private final ThreadLocal<Map<String, INDArray>> outputArrayCacheTl =
            ThreadLocal.withInitial(HashMap::new);

    // Requested outer-frame outputs from the most recent execution. These provide the
    // post-inference ARRAY lookup contract used by SameDiff/SDVariable without restoring
    // the old behavior of retaining every intermediate in nodeValueOutputs.
    private final ThreadLocal<Map<String, SDValue>> latestRequestedOutputsTl =
            ThreadLocal.withInitial(LinkedHashMap::new);

    // The current DAG being executed. Set at the start of executeOperations() so that
    // sub-methods (executeNode, executeRegularOperation, etc.) can look up variable consumers
    // for proper last-consumer-based liveness tracking without passing the DAG through every call.
    private ForwardExecutionDAG currentExecutionDAG;

    // Live DataBuffer reference counts: tracks how many arrays in variableValues share each DataBuffer.
    // Used by mid-execution release to determine if it's safe to close an array without
    // corrupting views (reshape/permute outputs) that share the same DataBuffer.
    // Count > 1 means another live array shares the buffer — do NOT close.
    // Reset at start of each executeOperations() call.
    private final ThreadLocal<java.util.IdentityHashMap<org.nd4j.linalg.api.buffer.DataBuffer, Integer>>
            liveDataBufferRefsTl = ThreadLocal.withInitial(java.util.IdentityHashMap::new);

    private java.util.IdentityHashMap<org.nd4j.linalg.api.buffer.DataBuffer, Integer> liveDataBufferRefs() {
        return liveDataBufferRefsTl.get();
    }

    /** Increment live reference count for the DataBuffer of the given array. */
    private void trackLiveBuffer(INDArray array) {
        if (array == null || array.data() == null) return;
        org.nd4j.linalg.api.buffer.DataBuffer buf = array.data();
        liveDataBufferRefs().merge(buf, 1, Integer::sum);
    }

    /** Decrement live reference count and remove entry when it reaches zero. */
    private void untrackLiveBuffer(INDArray array) {
        if (array == null || array.data() == null) return;
        org.nd4j.linalg.api.buffer.DataBuffer buf = array.data();
        java.util.IdentityHashMap<org.nd4j.linalg.api.buffer.DataBuffer, Integer> refs = liveDataBufferRefs();
        Integer count = refs.get(buf);
        if (count == null) return;
        if (count <= 1) {
            refs.remove(buf);
        } else {
            refs.put(buf, count - 1);
        }
    }

    /** Return true if the DataBuffer is exclusively owned (ref count == 1 or not tracked). */
    private boolean isBufferExclusivelyOwned(INDArray array) {
        if (array == null || array.data() == null) return true;
        Integer count = liveDataBufferRefs().get(array.data());
        return count == null || count <= 1;
    }

    // DataBuffers from externally-provided placeholder arrays. These must NOT be closed during
    // resetSession() — they are owned by the caller, not by this session.
    @Getter
    private final java.util.IdentityHashMap<org.nd4j.linalg.api.buffer.DataBuffer, Boolean> externalPlaceholderBuffers = new java.util.IdentityHashMap<>();


    // ---- Java-side execution timing ----
    // Enable with -Dorg.nd4j.inference.timing=true
    // Tracks where wall-clock time goes during graph execution (shape calc, alloc, exec, etc.)
    private static final boolean TIMING_ENABLED = Boolean.parseBoolean(
            System.getProperty(ND4JSystemProperties.INFERENCE_TIMING, "false"));
    // Per-call accumulators (reset each output() call)
    private long timingDagLookupNs, timingInitValuesNs, timingExecLoopNs, timingOutputDupNs, timingCleanupNs;
    // Per-op phase accumulators within executeOperations
    private long timingOpContextAcquireNs, timingInputResolveNs, timingShapeCalcNs;
    private long timingMemAllocNs, timingNativeExecNs, timingResultStoreNs;
    private long timingDepTrackNs, timingArrayReleaseNs;
    private long timingIntLongSyncNs;
    private int timingIntLongSyncCount, timingIntLongSyncSkipCount;
    private int timingOpCount, timingShapeCacheHits, timingShapeCacheMisses;
    private int timingReshapeTotal, timingReshapeViewUsed, timingReshapeViewSkipped;

    private void resetTimingCounters() {
        timingDagLookupNs = timingInitValuesNs = timingExecLoopNs = timingOutputDupNs = timingCleanupNs = 0;
        timingOpContextAcquireNs = timingInputResolveNs = timingShapeCalcNs = 0;
        timingMemAllocNs = timingNativeExecNs = timingResultStoreNs = 0;
        timingDepTrackNs = timingArrayReleaseNs = 0;
        timingIntLongSyncNs = 0;
        timingIntLongSyncCount = timingIntLongSyncSkipCount = 0;
        timingOpCount = timingShapeCacheHits = timingShapeCacheMisses = 0;
        timingReshapeTotal = timingReshapeViewUsed = timingReshapeViewSkipped = 0;
    }

    private void printTimingSummary() {
        double totalMs = (timingDagLookupNs + timingInitValuesNs + timingExecLoopNs +
                timingOutputDupNs + timingCleanupNs) / 1_000_000.0;
        log.info("=== InferenceSession Timing ({} ops, {}ms total) ===", timingOpCount,
                String.format("%.1f", totalMs));
        log.info("  DAG lookup:       {}ms", String.format("%.2f", timingDagLookupNs / 1_000_000.0));
        log.info("  Init values:      {}ms", String.format("%.2f", timingInitValuesNs / 1_000_000.0));
        log.info("  Exec loop:        {}ms", String.format("%.2f", timingExecLoopNs / 1_000_000.0));
        log.info("    OpCtx acquire:  {}ms", String.format("%.2f", timingOpContextAcquireNs / 1_000_000.0));
        log.info("    Input resolve:  {}ms", String.format("%.2f", timingInputResolveNs / 1_000_000.0));
        log.info("    INT/LONG sync:  {}ms (synced={}, skipped={})",
                String.format("%.2f", timingIntLongSyncNs / 1_000_000.0), timingIntLongSyncCount, timingIntLongSyncSkipCount);
        log.info("    Shape calc:     {}ms (hits={}, misses={})",
                String.format("%.2f", timingShapeCalcNs / 1_000_000.0), timingShapeCacheHits, timingShapeCacheMisses);
        long[] cc = ArrayCacheMemoryMgr.getCacheCounters();
        log.info("    Mem alloc:      {}ms (exact={}, capacity={}, miss={}, viewSkip={}, cached={}, capKeys={})",
                String.format("%.2f", timingMemAllocNs / 1_000_000.0), cc[0], cc[1], cc[2], cc[3], cc[4], cc[5]);
        ArrayCacheMemoryMgr.resetCacheCounters();
        if (timingReshapeTotal > 0) {
            log.info("    Reshape views:  {}/{} used, {} skipped (non-contiguous)",
                    timingReshapeViewUsed, timingReshapeTotal, timingReshapeViewSkipped);
        }
        log.info("    Native exec:    {}ms", String.format("%.2f", timingNativeExecNs / 1_000_000.0));
        log.info("    Result store:   {}ms", String.format("%.2f", timingResultStoreNs / 1_000_000.0));
        log.info("    Dep tracking:   {}ms", String.format("%.2f", timingDepTrackNs / 1_000_000.0));
        log.info("    Array release:  {}ms", String.format("%.2f", timingArrayReleaseNs / 1_000_000.0));
        log.info("  Output dup:       {}ms", String.format("%.2f", timingOutputDupNs / 1_000_000.0));
        log.info("  Cleanup:          {}ms", String.format("%.2f", timingCleanupNs / 1_000_000.0));
    }

    // ---- DynamicShapePlan-based execution (for autoregressive inference with dynamic shapes) ----
    // When enabled, pre-compiles graph wiring (input/output index mapping, liveness schedule) once,
    // then executes using flat array-indexed slots instead of string-keyed HashMaps.
    // Unlike static ExecutionPlan, this handles shapes that change every step (e.g., growing KV cache).
    // Execution can auto-compile plans only if SameDiff.dspAutoCompileEnabled is true.
    // Disable this path entirely with -Dorg.nd4j.inference.dynamicShapePlan=false.
    private static volatile boolean DYNAMIC_SHAPE_PLAN_ENABLED = Boolean.parseBoolean(
            System.getProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true"));

    /**
     * Enable or disable DynamicShapePlan execution at runtime (primarily for tests).
     */
    public static void setDynamicShapePlanEnabled(boolean enabled) {
        DYNAMIC_SHAPE_PLAN_ENABLED = enabled;
    }

    /**
     * Accessor for tests.
     */
    public static boolean isDynamicShapePlanEnabled() {
        return DYNAMIC_SHAPE_PLAN_ENABLED;
    }

    protected volatile DynamicShapePlan dynamicShapePlan;
    protected final ThreadLocal<DynamicShapePlanExecutor> dynamicShapePlanExecutorTl = new ThreadLocal<>();

    // ---- Conditional pool trim ----
    // During steady-state decode, the CUDA memory pool reuses freed memory without trimming.
    // Trimming every step wastes time on cudaStreamSynchronize + cudaMemPoolTrimTo when there's
    // nothing meaningful to return. Only trim periodically and always on step 0/1
    // (prefill→decode transition where large buffers are freed).
    protected int dspStepCount = 0;
    protected static final int TRIM_INTERVAL = Integer.getInteger(ND4JSystemProperties.DSP_TRIM_INTERVAL, 10);

    private Set<Long> freedArrays() {
        return freedArraysTl.get();
    }

    private AbstractDependencyTracker<SDValue, Dep> arrayUseTracker() {
        return arrayUseTrackerTl.get();
    }

    public AbstractDependencyTracker<SDValue, Dep> getArrayUseTracker() {
        return arrayUseTracker();
    }

    /**
     * Get the DynamicShapePlanExecutor for the current thread (if any).
     * Can be used to configure native plan settings like shapesFrozen mode.
     */
    public DynamicShapePlanExecutor getDynamicShapePlanExecutor() {
        return dynamicShapePlanExecutorTl.get();
    }

    /**
     * Explicitly compile (or reuse) a DynamicShapePlan for the given outputs.
     * This does not execute the graph.
     */
    public DynamicShapePlan compileDynamicShapePlan(@NonNull List<String> requestedOutputs) {
        List<String> outputs = resolveRequestedOutputs(requestedOutputs);
        Set<String> allRequired = new LinkedHashSet<>(outputs);
        ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
            ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
            return builder.buildForwardDAG(allRequired);
        });

        DynamicShapePlan plan = getOrCompileDynamicShapePlan(dag, outputs, true);
        if (plan == null) {
            throw new IllegalStateException("DynamicShapePlan compilation failed (control flow or unsupported graph)");
        }
        dynamicShapePlan = plan;
        return plan;
    }

    /**
     * Explicitly compile a native DSP plan handle and configure graph execution mode.
     *
     * @return effective graph execution mode configured on the native plan
     */
    public GraphExecutionMode compileNativeDynamicShapePlan(@NonNull List<String> requestedOutputs,
                                                            GraphExecutionMode requestedMode,
                                                            boolean fallbackToAutoIfTritonUnavailable) {
        DynamicShapePlan plan = compileDynamicShapePlan(requestedOutputs);
        DynamicShapePlanExecutor executor = dynamicShapePlanExecutorTl.get();
        if (executor == null) {
            executor = new DynamicShapePlanExecutor(sameDiff, mmgr);
            dynamicShapePlanExecutorTl.set(executor);
        }

        GraphExecutionMode effectiveMode = executor.compileNativePlan(
                plan, requestedMode, fallbackToAutoIfTritonUnavailable);
        if (!executor.isNativePlanCompiled(plan)) {
            throw new IllegalStateException("Native DSP plan compilation failed");
        }
        return effectiveMode;
    }

    /**
     * Clear only the node output buffers and plan, preserving the array cache
     * and model constants. Use this during decode recompile cycles where
     * stale prefill outputs must be cleared but model weights must survive.
     */
    public void clearNodeOutputsOnly() {
        DynamicShapePlan sessionPlan = dynamicShapePlan;
        if (sessionPlan != null) {
            sessionPlan.close();
            dynamicShapePlan = null;
        }
        int nodeOutputsClosed = closeNodeValueOutputBuffers();
        if (nodeOutputsClosed > 0) {
            log.info("clearNodeOutputsOnly: closed {} node output buffers", nodeOutputsClosed);
        }
    }

    /**
     * Clear all caches (DAG cache, plan cache, constant/variable cache).
     * Call when the graph structure changes.
     */
    public void clearAllCaches() {
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = snapshotProtectedBuffersForCleanup();
        DynamicShapePlan sessionPlan = dynamicShapePlan;
        if (sessionPlan != null) {
            sessionPlan.close();
            dynamicShapePlan = null;
        }

        int nodeOutputsClosed = closeNodeValueOutputBuffers();
        closePooledResources();
        flushArrayCache(protectedBuffers);

        if (nodeOutputsClosed > 0) {
            log.info("Session cache invalidation closed {} node output buffers", nodeOutputsClosed);
        }
    }

    public void setArrayUseTracker(AbstractDependencyTracker<SDValue, Dep> tracker) {
        arrayUseTrackerTl.set(tracker);
    }

    public Map<String,OpContext> getOpContexts() {
        return opContextsTl.get();
    }

    private Map<String,OpContext> opContexts() {
        return opContextsTl.get();
    }

    private void collectCloseableBuffers(SDValue value, IdentityHashMap<DataBuffer, Boolean> buffers) {
        if (value == null) {
            return;
        }

        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor != null && !tensor.wasClosed() && tensor.data() != null) {
                    buffers.put(tensor.data(), Boolean.TRUE);
                }
                break;
            case LIST:
                if (value.getListValue() != null) {
                    for (INDArray arr : value.getListValue()) {
                        if (arr != null && !arr.wasClosed() && arr.data() != null) {
                            buffers.put(arr.data(), Boolean.TRUE);
                        }
                    }
                }
                break;
            case DICT:
                if (value.getDictValue() != null) {
                    for (INDArray arr : value.getDictValue().values()) {
                        if (arr != null && !arr.wasClosed() && arr.data() != null) {
                            buffers.put(arr.data(), Boolean.TRUE);
                        }
                    }
                }
                break;
            default:
                break;
        }
    }

    private int closeLatestRequestedOutputBuffers() {
        Map<String, SDValue> latestOutputs = latestRequestedOutputsTl.get();
        if (latestOutputs == null || latestOutputs.isEmpty()) {
            return 0;
        }

        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = snapshotProtectedBuffersForCleanup();
        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (SDValue value : latestOutputs.values()) {
            collectCloseableBuffers(value, uniqueBuffers);
        }

        int closed = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (buf == null || protectedBuffers.containsKey(buf)) {
                continue;
            }
            if (buf.closeable()) {
                try {
                    buf.close();
                    closed++;
                } catch (Exception ignored) {
                    // Buffer may already be deallocated or owned elsewhere.
                }
            }
        }
        latestOutputs.clear();
        return closed;
    }

    private IdentityHashMap<DataBuffer, Boolean> snapshotProtectedBuffersForCleanup() {
        IdentityHashMap<DataBuffer, Boolean> snapshot = new IdentityHashMap<>();
        snapshot.putAll(externalPlaceholderBuffers);
        for (SDVariable variable : sameDiff.variables()) {
            if (variable == null) {
                continue;
            }
            VariableType type = variable.getVariableType();
            if (type != VariableType.CONSTANT && type != VariableType.VARIABLE) {
                continue;
            }

            INDArray arr = variable.getArr();
            if (arr != null && !arr.wasClosed() && arr.data() != null) {
                snapshot.put(arr.data(), Boolean.TRUE);
            }
        }
        return snapshot;
    }

    private int closeNodeValueOutputBuffers() {
        if (nodeValueOutputs.isEmpty()) {
            return 0;
        }

        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = snapshotProtectedBuffersForCleanup();
        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (SDValue value : nodeValueOutputs.values()) {
            collectCloseableBuffers(value, uniqueBuffers);
        }

        int closed = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (buf == null || protectedBuffers.containsKey(buf)) {
                continue;
            }

            // Never force-close constant buffers — they are model weights that must
            // survive session cleanup. The protectedBuffers snapshot may miss them if
            // identity ops alias a constant's DataBuffer to an output slot, causing the
            // buffer to appear in nodeValueOutputs with a different identity.
            if (buf.isConstant()) {
                continue;
            }

            if (buf.closeable()) {
                try {
                    buf.close();
                    closed++;
                } catch (Exception ignored) {
                    // Buffer may already be deallocated or owned elsewhere.
                }
            }
        }

        nodeValueOutputs.clear();
        return closed;
    }

    /**
     * Close all pooled OpContexts and clear accumulated thread-local state.
     * Must be called during resetSession() to prevent stale native pointers
     * from being reused in subsequent sessions.
     */
    public void closePooledResources() {
        Map<String, OpContext> activeContexts = opContextsTl.get();
        int activeClosed = 0;
        for (OpContext ctx : activeContexts.values()) {
            if (ctx != null) {
                try {
                    ctx.close();
                    activeClosed++;
                } catch (Exception e) {
                    // Already closed or invalid; ignore
                }
            }
        }
        activeContexts.clear();

        // Close pooled OpContexts that were returned via purge() during execution.
        // These hold live native pointers that become stale after session reset.
        Deque<OpContext> pool = opContextPoolTl.get();
        int pooledClosed = 0;
        for (OpContext ctx : pool) {
            if (ctx != null) {
                try {
                    ctx.close();
                    pooledClosed++;
                } catch (Exception e) {
                    // Already closed or invalid; ignore
                }
            }
        }
        pool.clear();

        // Clear freed arrays tracker to avoid stale entries
        freedArraysTl.get().clear();

        // Clear dependency tracking and live buffer refs held by this thread's last execution.
        arrayUseTrackerTl.get().clear();
        liveDataBufferRefs().clear();

        // Clear output shape and array caches
        outputShapeCacheTl.get().clear();
        outputArrayCacheTl.get().clear();

        // Reset DSP step counter so the next session starts with immediate trim
        dspStepCount = 0;

        // Session-local caches do not survive resetSession(): a new session instance will be
        // created on the next output() call and can retrieve any reusable DynamicShapePlan
        // from SameDiff's model-level cache.
        cachedConstVarValues = null;
        cachedInputsList = null;
        currentExecutionDAG = null;
        dynamicShapePlan = null;
        dagCache.clear();

        // Close the DynamicShapePlanExecutor (slot cache, native workspace, dedup sets).
        // The executor holds GPU-allocated arrays in its slot cache; closing it frees them
        // and trims the pool, making memory available for subsequent model phases (e.g.,
        // decoder after vision encoder). A new executor is created on the next output() call.
        boolean planExecutorClosed = false;
        DynamicShapePlanExecutor executor = dynamicShapePlanExecutorTl.get();
        if (executor != null) {
            try {
                executor.close();
                planExecutorClosed = true;
            } catch (Exception e) {
                log.warn("Error closing DynamicShapePlanExecutor during session cleanup: {}", e.getMessage());
            }
            dynamicShapePlanExecutorTl.remove();
        }

        int latestOutputsClosed = closeLatestRequestedOutputBuffers();
        externalPlaceholderBuffers.clear();

        opContextsTl.remove();
        opContextPoolTl.remove();
        freedArraysTl.remove();
        arrayUseTrackerTl.remove();
        outputShapeCacheTl.remove();
        outputArrayCacheTl.remove();
        latestRequestedOutputsTl.remove();
        liveDataBufferRefsTl.remove();

        // Sync streams and trim pool to release freed memory back to the driver.
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int nDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            Nd4j.getExecutioner().commit();
            for (int d = 0; d < nDevices; d++) {
                nativeOps.trimMemoryPoolOnStream(d, null);
            }
        } catch (Exception e) {
            log.debug("Pool trim during closePooledResources failed: {}", e.getMessage());
        }

        if (activeClosed > 0 || pooledClosed > 0 || planExecutorClosed || latestOutputsClosed > 0) {
            log.info("Session cleanup closed activeCtx={}, pooledCtx={}, planExecutor={}, latestOutputs={}",
                    activeClosed, pooledClosed, planExecutorClosed, latestOutputsClosed);
        }
    }

    /**
     * Flush the array cache, freeing all cached intermediate arrays.
     * Call this between phases (e.g., after prefill, before decode) to release
     * large cached arrays that are no longer needed. The cache will refill with
     * appropriately-sized arrays for the new phase.
     *
     * @param protectedBuffers DataBuffers to skip (e.g., static KV buffers).
     *                         May be null if no protection is needed.
     */
    public void flushArrayCache(IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (mmgr instanceof ArrayCacheMemoryMgr) {
            ((ArrayCacheMemoryMgr) mmgr).close(protectedBuffers);
            log.info("Flushed ArrayCacheMemoryMgr");
        }
    }

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
        mmgr = new ArrayCacheMemoryMgr();
    }

    /**
     * Create an InferenceSession with a custom SessionMemMgr.
     * Use {@link org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr}
     * for workspace-backed memory management during inference.
     *
     * @param sameDiff the SameDiff instance
     * @param memMgr   the memory manager to use for allocations
     */
    public InferenceSession(@NonNull SameDiff sameDiff, @NonNull SessionMemMgr memMgr) {
        super(sameDiff);
        mmgr = memMgr;
    }


    @Override
    @SneakyThrows
    public ExecutionResult output(@NonNull List<String> variables,
                                  Map<String, INDArray> placeholderValues,
                                  Map<String, SDValue> otherPlaceHolderValues,
                                  MultiDataSet batch,
                                  Collection<String> requiredActivations,
                                  List<Listener> listeners, At at) {

        // Clear freed arrays tracking from previous execution
        freedArrays().clear();
        latestRequestedOutputsTl.get().clear();

        // Clear and repopulate externally-provided placeholder DataBuffers.
        // Without clearing, this map accumulates references to DataBuffers from ALL previous
        // output() calls (e.g., 5 draft model steps per speculative decode round), preventing
        // GC of closed DataBuffer objects and causing Java heap growth.
        externalPlaceholderBuffers.clear();
        if (placeholderValues != null) {
            for (Map.Entry<String, INDArray> phEntry : placeholderValues.entrySet()) {
                INDArray arr = phEntry.getValue();
                if (arr != null && !arr.wasClosed() && arr.data() != null) {
                    externalPlaceholderBuffers.put(arr.data(), Boolean.TRUE);
                }
            }
        }

        if (TIMING_ENABLED) resetTimingCounters();

        log.debug("Executing forward pass for {} variables", variables.size());

        Map<String, SDValue> filteredResults = null;
        boolean dspHandledExecution = false;
        try {
            // Prepare all required outputs
            Set<String> allRequired = new LinkedHashSet<>(variables);
            if (requiredActivations != null) {
                allRequired.addAll(requiredActivations);
            }

            // Build corrected DAG with caching (replaces broken initSubgraph)
            long t0 = TIMING_ENABLED ? System.nanoTime() : 0;
            ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
                ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
                return builder.buildForwardDAG(allRequired);
            });
            if (TIMING_ENABLED) timingDagLookupNs = System.nanoTime() - t0;

            // Debug: Print execution plan when debug mode is enabled
            if (Nd4j.getEnvironment().isDebug()) {
                List<ExecutionNode> execOrder = dag.getFrameAwareExecutionOrder();
                StringBuilder sb = new StringBuilder();
                sb.append("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
                sb.append("║                    SAMEDIFF EXECUTION PLAN (DEBUG MODE)                      ║\n");
                sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
                sb.append("║ Requested Outputs: ").append(variables).append("\n");
                sb.append("║ Total Operations: ").append(execOrder.size()).append("\n");
                sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");

                // List all placeholders with their shapes/types
                sb.append("║ PLACEHOLDERS:\n");
                Set<String> placeholders = dag.getRequiredPlaceholders();
                if (placeholders.isEmpty()) {
                    sb.append("║   (none)\n");
                } else {
                    for (String ph : placeholders) {
                        SDVariable var = sameDiff.getVariable(ph);
                        sb.append("║   • ").append(ph);
                        if (var != null) {
                            sb.append(" : dtype=").append(var.dataType());
                            if (var.getShape() != null) {
                                sb.append(", shape=").append(formatShape(var.getShape()));
                            }
                        }
                        sb.append("\n");
                    }
                }

                // List all constants
                sb.append("╠──────────────────────────────────────────────────────────────────────────────╣\n");
                sb.append("║ CONSTANTS:\n");
                Set<String> constants = dag.getConstants();
                if (constants.isEmpty()) {
                    sb.append("║   (none)\n");
                } else {
                    for (String c : constants) {
                        SDVariable var = sameDiff.getVariable(c);
                        sb.append("║   • ").append(c);
                        if (var != null) {
                            sb.append(" : dtype=").append(var.dataType());
                            if (var.getShape() != null) {
                                sb.append(", shape=").append(formatShape(var.getShape()));
                            }
                        }
                        sb.append("\n");
                    }
                }

                // List all trainable variables
                sb.append("╠──────────────────────────────────────────────────────────────────────────────╣\n");
                sb.append("║ TRAINABLE VARIABLES:\n");
                Set<String> trainableVars = dag.getVariables();
                if (trainableVars.isEmpty()) {
                    sb.append("║   (none)\n");
                } else {
                    for (String v : trainableVars) {
                        SDVariable var = sameDiff.getVariable(v);
                        sb.append("║   • ").append(v);
                        if (var != null) {
                            sb.append(" : dtype=").append(var.dataType());
                            if (var.getShape() != null) {
                                sb.append(", shape=").append(formatShape(var.getShape()));
                            }
                        }
                        sb.append("\n");
                    }
                }

                // List all graph variables
                sb.append("╠──────────────────────────────────────────────────────────────────────────────╣\n");
                sb.append("║ ALL GRAPH VARIABLES (").append(sameDiff.getVariables().size()).append(" total):\n");
                for (String varName : sameDiff.getVariables().keySet()) {
                    SDVariable var = sameDiff.getVariable(varName);
                    if (var != null) {
                        sb.append("║   • ").append(String.format("%-30s", var.name()));
                        sb.append(" [").append(String.format("%-11s", var.getVariableType())).append("] ");
                        sb.append("dtype=").append(var.dataType());
                        if (var.getShape() != null) {
                            sb.append(", shape=").append(formatShape(var.getShape()));
                        }
                        sb.append("\n");
                    }
                }

                // Execution order with full details
                sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
                sb.append("║ EXECUTION ORDER (").append(execOrder.size()).append(" nodes):\n");
                sb.append("╠──────────────────────────────────────────────────────────────────────────────╣\n");

                int opIndex = 0;
                for (ExecutionNode node : execOrder) {
                    SameDiffOp sdOp = sameDiff.getOps().get(node.getOperationName());
                    DifferentialFunction op = sdOp != null ? sdOp.getOp() : null;
                    String opType = op != null ? op.opName() : "?";
                    String opClass = op != null ? op.getClass().getSimpleName() : "?";

                    sb.append("║ ").append(String.format("%4d", opIndex++)).append(". ");
                    sb.append(node.getOperationName()).append("\n");
                    sb.append("║       Type: ").append(opType).append(" (").append(opClass).append(")\n");
                    sb.append("║       NodeType: ").append(node.getNodeType()).append("\n");

                    // Show inputs with variable info
                    List<String> inputs = node.getInputVariables();
                    if (!inputs.isEmpty()) {
                        sb.append("║       Inputs:\n");
                        for (int i = 0; i < inputs.size(); i++) {
                            String inputName = inputs.get(i);
                            SDVariable inputVar = sameDiff.getVariable(inputName);
                            sb.append("║         [").append(i).append("] ").append(inputName);
                            if (inputVar != null) {
                                sb.append(" : ").append(inputVar.getVariableType());
                                sb.append(", dtype=").append(inputVar.dataType());
                                if (inputVar.getShape() != null) {
                                    sb.append(", shape=").append(formatShape(inputVar.getShape()));
                                }
                            }
                            sb.append("\n");
                        }
                    }

                    // Show outputs with variable info
                    List<String> outputs = node.getOutputVariables();
                    if (!outputs.isEmpty()) {
                        sb.append("║       Outputs:\n");
                        for (int i = 0; i < outputs.size(); i++) {
                            String outputName = outputs.get(i);
                            SDVariable outputVar = sameDiff.getVariable(outputName);
                            sb.append("║         [").append(i).append("] ").append(outputName);
                            if (outputVar != null) {
                                sb.append(" : ").append(outputVar.getVariableType());
                                sb.append(", dtype=").append(outputVar.dataType());
                                if (outputVar.getShape() != null) {
                                    sb.append(", shape=").append(formatShape(outputVar.getShape()));
                                }
                            }
                            sb.append("\n");
                        }
                    }

                    // Show dependencies
                    Set<String> deps = node.getDependsOnOperations();
                    if (deps != null && !deps.isEmpty()) {
                        sb.append("║       DependsOn: ").append(deps).append("\n");
                    }

                    sb.append("║\n");
                }
                sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
                log.info(sb.toString());
            }

            // Enter memory manager scope before any allocations
            mmgr.scopeIn();

            // ---- DynamicShapePlan fast path ----
            // Check DSP BEFORE preprocessPlaceholders to avoid iterating all variables
            // (6000+ in large models) when DSP will handle execution. DSP resolves
            // external inputs directly and doesn't use arrayUseTracker.
            if (DYNAMIC_SHAPE_PLAN_ENABLED &&
                (otherPlaceHolderValues == null || otherPlaceHolderValues.isEmpty())) {
                try {
                    // Suppress cross-device routing during DSP execution.
                    // Without this, selectTargetDevice() in CudaExecutioner can migrate
                    // input tensors to a different GPU when memory pressure is detected
                    // (cudaMemGetInfo reports low free memory). During frozen DSP decode,
                    // all data should stay on the model's home device — cross-device
                    // migration causes stale CUDA context state → SIGSEGV.
                    OpaqueDataBuffer.suppressCrossDeviceRouting(true);
                    // Lightweight type casting: cast mismatched placeholder dtypes without
                    // the full preprocessPlaceholders overhead (arrayUseTracker iteration).
                    Map<String, INDArray> dspPlaceholders = castPlaceholderTypes(placeholderValues);
                    long memBeforeDsp = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemoryDefault() / (1024 * 1024);
                    Map<String, SDValue> dynamicPlanResults = executeDynamicShapePlanBased(
                            dag, dspPlaceholders, allRequired, variables);
                    long memAfterDsp = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemoryDefault() / (1024 * 1024);
                    if (memBeforeDsp - memAfterDsp > 10) {
                        log.info("[DSP_MEM_LEAK] executeDynamicShapePlanBased consumed {}MB (before={}MB after={}MB) dspStep={}",
                                memBeforeDsp - memAfterDsp, memBeforeDsp, memAfterDsp, dspStepCount);
                    }
                    if (dynamicPlanResults == null) {
                        // DSP not available — close any cast copies to prevent GPU memory leak.
                        // castPlaceholderTypes returns a NEW map with cast arrays when types mismatch.
                        // Without closing, these cast arrays leak each output() call.
                        if (dspPlaceholders != placeholderValues && dspPlaceholders != null) {
                            for (Map.Entry<String, INDArray> entry : dspPlaceholders.entrySet()) {
                                INDArray castArr = entry.getValue();
                                // Only close arrays that are NOT the same object as the original placeholder
                                if (placeholderValues != null && castArr != placeholderValues.get(entry.getKey())) {
                                    if (castArr != null && !castArr.wasClosed()) {
                                        castArr.close();
                                    }
                                }
                            }
                        }
                    }
                    if (dynamicPlanResults != null) {
                        log.debug("[EXEC-PATH] DSP path succeeded for {} outputs", dynamicPlanResults.size());
                        filteredResults = dynamicPlanResults;
                        dspHandledExecution = true;
                        dspStepCount++;

                        // Check if shapes are frozen (steady-state decode with CUDA graph replay).
                        // When frozen, the C++ fast path already does cudaStreamSynchronize before
                        // graph launch, so outputs are ready when executeDynamicShapePlan() returns.
                        // We can skip commit() and make trim periodic to save ~2-5ms/step.
                        DynamicShapePlanExecutor dspExec = dynamicShapePlanExecutorTl.get();
                        boolean frozen = dspExec != null && dspExec.isShapesFrozen();

                        // Periodically trim the CUDA memory pool to reclaim freed memory.
                        if (dspStepCount % TRIM_INTERVAL == 0 || dspStepCount <= 2) {
                            Nd4j.getExecutioner().commit();
                            NativeOps trimOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                            int nDevicesTrim = Nd4j.getAffinityManager().getNumberOfDevices();
                            for (int d = 0; d < nDevicesTrim; d++) {
                                trimOps.trimMemoryPool(d);
                            }
                        }

                        log.debug("DynamicShapePlan-based execution completed successfully");
                        // Do NOT call mmgr.scopeOut() here. The finally block handles both
                        // detaching workspace-backed arrays and closing the workspace scope
                        // in the correct order. Closing the workspace here causes the finally
                        // block's detach() to fail with "leaked workspace pointer" because
                        // the workspace is no longer active when detach checks isScopeActive().
                        // Close any cast placeholder copies to prevent GPU memory leak.
                        // castPlaceholderTypes() allocates new arrays when dtypes mismatch.
                        // These must be freed after DSP execution completes.
                        if (dspPlaceholders != placeholderValues && dspPlaceholders != null) {
                            for (Map.Entry<String, INDArray> entry : dspPlaceholders.entrySet()) {
                                INDArray castArr = entry.getValue();
                                if (placeholderValues != null && castArr != placeholderValues.get(entry.getKey())) {
                                    if (castArr != null && !castArr.wasClosed()) {
                                        castArr.close();
                                    }
                                }
                            }
                        }
                        // Restore cross-device routing suppression before returning.
                        OpaqueDataBuffer.suppressCrossDeviceRouting(false);
                        return ExecutionResult.builder().valueOutputs(filteredResults).build();
                    }
                } catch (Exception e) {
                    log.error("DynamicShapePlan-based execution failed — no fallback allowed: {}", e.getMessage());
                    throw new RuntimeException("DSP execution failed. No fallback to standard path. " +
                            "Fix the DSP executor. Error: " + e.getMessage(), e);
                } finally {
                    // Always restore cross-device routing when leaving the DSP path,
                    // whether it succeeded, fell through, or threw an exception.
                    OpaqueDataBuffer.suppressCrossDeviceRouting(false);
                }
            }

            // Preprocess placeholders using existing logic (needed for standard path)
            Map<String, INDArray> processedPlaceholders = preprocessPlaceholders(placeholderValues, at);

            log.debug("[EXEC-PATH] Falling through to standard executeOperations (DSP_ENABLED={})",
                    DYNAMIC_SHAPE_PLAN_ENABLED);
            // Reset growth factor to 1.0 but keep cache ENABLED.
            // The DSP side effect enables cache + growth factor 1.1x before checking if plan exists.
            // Growth factor > 1.0 creates oversized buffers (output arrays become views) and
            // Growth factor is now scoped to DSP execution via withGrowthFactor() in
            // executeDynamicShapePlanBased(). No need to reset here — the scope auto-restores.
            // IMPORTANT: Do NOT disable the cache here. The cache prevents premature DataBuffer
            // close in the standard executeOperations path — the no-cache cleanup aggressively
            // closes DataBuffers of intermediate variables, which destroys shared buffers for
            // view-producing ops (reshape_no_copy, permute, etc.).
            Map<String, SDValue> processedOtherPlaceholders = preprocessValuePlaceholders(otherPlaceHolderValues, at);
            // Suppress cross-device routing during execution.
            OpaqueDataBuffer.suppressCrossDeviceRouting(true);
            // Execute with corrected ordering
            long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
            Map<String, SDValue> results = executeOperations(dag, processedPlaceholders,
                    processedOtherPlaceholders, allRequired, listeners, at, batch);
            if (TIMING_ENABLED) timingExecLoopNs = System.nanoTime() - tExec0;
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);

            // Commit all pending operations before accessing results.
            // This ensures async operations (CUDA streams, etc.) complete before values are read.
            // Without this, multi-threaded or subprocess contexts may read stale/uninitialized data.
            Nd4j.getExecutioner().commit();

            // Post-process results using existing logic
            Map<String, SDValue> finalResults = postProcessOutputValues(results);

            // Return only requested variables
            filteredResults = new HashMap<>();
            for (String var : variables) {
                if (finalResults.containsKey(var)) {
                    SDValue val = finalResults.get(var);
                    filteredResults.put(var, val);
                }
            }

            // Debug: Print execution summary
            if (Nd4j.getEnvironment().isDebug()) {
                StringBuilder sb = new StringBuilder();
                sb.append("\n╔══════════════════════════════════════════════════════════════════╗\n");
                sb.append("║              EXECUTION COMPLETE - RESULTS SUMMARY                ║\n");
                sb.append("╠══════════════════════════════════════════════════════════════════╣\n");
                sb.append("║ Total Results: ").append(filteredResults.size()).append("\n");
                for (Map.Entry<String, SDValue> entry : filteredResults.entrySet()) {
                    sb.append("║   ").append(entry.getKey()).append(": ");
                    SDValue val = entry.getValue();
                    if (val == null) {
                        sb.append("null");
                    } else if (val.getSdValueType() == SDValueType.TENSOR) {
                        INDArray arr = val.getTensorValue();
                        if (arr != null) {
                            sb.append("shape=").append(formatShape(arr.shape()))
                              .append(", dtype=").append(arr.dataType());
                        } else {
                            sb.append("null tensor");
                        }
                    } else {
                        sb.append(val.getSdValueType());
                    }
                    sb.append("\n");
                }
                sb.append("╚══════════════════════════════════════════════════════════════════╝\n");
                log.info(sb.toString());
            }

            // Mark output dependencies as satisfied so buffers can be released before clearing tracker
            for (String outputVar : variables) {
                arrayUseTracker().markSatisfied(new ReqOutputDep(outputVar), true);
            }

            // Release any arrays that became releasable (will NOT release outputs we're returning)
            if (arrayUseTracker().hasNewAllSatisfied()) {
                arrayUseTracker().getNewAllSatisfiedList(); // Just pop them off the queue
            }

            // Clear array use tracker to prevent stale dependencies from accumulating
            arrayUseTracker().clear();
        } finally {
            long tCleanup0 = TIMING_ENABLED ? System.nanoTime() : 0;
            // Close any remaining OpContext instances.
            // Skip when DSP handled execution: DSP manages its own OpContext pool
            // and the standard session's opContexts map is empty.
            if (!dspHandledExecution) {
                try {
                    for (Map.Entry<String, OpContext> entry : opContexts().entrySet()) {
                        OpContext ctx = entry.getValue();
                        if (ctx != null) {
                            try {
                                ctx.close();
                            } catch (Exception e) {
                                log.warn("Error closing OpContext: {}", e.getMessage());
                            }
                        }
                    }
                } catch (Exception e) {
                    log.warn("Error iterating OpContexts for cleanup: {}", e.getMessage());
                } finally {
                    opContexts().clear();
                }
            }

            // Detach workspace-backed output arrays BEFORE scopeOut resets the workspace.
            // Views and workspace-backed arrays become invalid when the workspace cycles.
            if (mmgr.isWorkspaceBacked() && filteredResults != null) {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (Map.Entry<String, SDValue> entry : filteredResults.entrySet()) {
                        SDValue val = entry.getValue();
                        if (val != null && val.getSdValueType() == SDValueType.TENSOR) {
                            INDArray arr = val.getTensorValue();
                            if (arr != null && arr.isAttached()) {
                                entry.setValue(SDValue.create(arr.detach()));
                            }
                        } else if (val != null && val.getSdValueType() == SDValueType.LIST) {
                            List<INDArray> list = val.getListValue();
                            boolean needsDetach = false;
                            for (INDArray arr : list) {
                                if (arr != null && arr.isAttached()) {
                                    needsDetach = true;
                                    break;
                                }
                            }
                            if (needsDetach) {
                                List<INDArray> detached = new java.util.ArrayList<>(list.size());
                                for (INDArray arr : list) {
                                    detached.add(arr != null && arr.isAttached() ? arr.detach() : arr);
                                }
                                entry.setValue(SDValue.create(detached));
                            }
                        }
                    }
                }
            }

            // Exit memory manager scope to release/recycle cached arrays
            if (mmgr != null) {
                try {
                    mmgr.scopeOut();
                } catch (Exception e) {
                    log.warn("Error in memory manager scope out: {}", e.getMessage());
                }
            }

            // TAD packs accumulate during operation execution and MUST be cleared after EVERY inference.
            // Skip when DSP handled execution: DSP manages its own memory lifecycle and the
            // synchronized JNI call adds ~10-50ms overhead per token.
            if (!dspHandledExecution) {
                org.nd4j.linalg.factory.Nd4j.clearTADCache();
            }

            // Note: nodeValueOutputs is intentionally NOT cleared or closed here.
            // Closing arrays causes double-free crashes because native data buffers
            // are shared across multiple INDArray objects (views, reshapes). Even clearing
            // without closing causes GC-triggered JavaCPP deallocation to hit freed buffers.
            // Memory reclamation between output() calls relies on mmgr.close() above
            // which releases the array cache, and on the arrayUseTracker releasing
            // intermediate arrays during execution.

            if (TIMING_ENABLED) {
                timingCleanupNs = System.nanoTime() - tCleanup0;
                printTimingSummary();
            }
            log.debug("CLEANUP: all phases complete");
        }

        latestRequestedOutputsTl.get().putAll(filteredResults);
        return ExecutionResult.builder().valueOutputs(filteredResults).build();
    }

    @Override
    public boolean contains(String variable, String frame, int iteration, FrameIter parentFrameIter) {
        if (super.contains(variable, frame, iteration, parentFrameIter)) {
            return true;
        }

        return isLatestRequestedOutputLookup(frame, iteration, parentFrameIter)
                && latestRequestedOutputsTl.get().containsKey(variable);
    }

    @Override
    public SDValue get(String variable, String frame, int iteration, FrameIter parentFrameIter,
                       boolean enforceExistence) {
        SDValue out = super.get(variable, frame, iteration, parentFrameIter, false);
        if (out != null || super.contains(variable, frame, iteration, parentFrameIter)) {
            return out;
        }

        if (isLatestRequestedOutputLookup(frame, iteration, parentFrameIter)) {
            SDValue latest = latestRequestedOutputsTl.get().get(variable);
            if (latest != null || latestRequestedOutputsTl.get().containsKey(variable)) {
                return latest;
            }
        }

        if (enforceExistence) {
            Preconditions.checkNotNull(null,
                    "No output found for variable %s (frame %s, iteration %s)", variable, frame, iteration);
        }
        return null;
    }

    private boolean isLatestRequestedOutputLookup(String frame, int iteration, FrameIter parentFrameIter) {
        return OUTER_FRAME.equals(frame) && iteration == 0 && parentFrameIter == null;
    }


    private List<String> resolveRequestedOutputs(List<String> requestedOutputs) {
        if (requestedOutputs != null && !requestedOutputs.isEmpty()) {
            return requestedOutputs;
        }

        List<String> configuredOutputs = sameDiff.outputs();
        Preconditions.checkState(configuredOutputs != null && !configuredOutputs.isEmpty(),
                "No outputs were provided and SameDiff has no configured outputs");
        return configuredOutputs;
    }

    private DynamicShapePlan getOrCompileDynamicShapePlan(ForwardExecutionDAG dag,
                                                          List<String> requestedOutputs,
                                                          boolean allowCompile) {
        Set<String> outputSet = new LinkedHashSet<>(requestedOutputs);

        DynamicShapePlan plan = dynamicShapePlan;
        if (plan != null) {
            Set<String> existingOutputs = new LinkedHashSet<>(plan.getRequestedOutputs());
            if (!existingOutputs.equals(outputSet)) {
                plan = null;
            }
        }

        if (plan == null) {
            plan = sameDiff.getCachedDynamicShapePlan(outputSet);
            if (plan != null) {
                dynamicShapePlan = plan;
                return plan;
            }

            if (!allowCompile) {
                return null;
            }

            log.debug("Compiling DynamicShapePlan for {} outputs", outputSet.size());
            plan = DynamicShapePlanCompiler.compile(sameDiff, dag, outputSet);
            if (plan == null) {
                log.debug("DynamicShapePlan compilation returned null");
                return null;
            }
            if (plan.isHasControlFlowOps()) {
                log.debug("DynamicShapePlan has control flow ops - native CF execution enabled");
            }

            plan.assignDevices();
            sameDiff.cacheDynamicShapePlan(outputSet, plan);
            log.debug("DynamicShapePlan compiled: {}", plan.getSummary());
        }

        dynamicShapePlan = plan;
        return plan;
    }

    /**
     * Execute using DynamicShapePlan-based path: pre-compile graph wiring once, then execute
     * with flat array-indexed slots and per-slot shape cache. Handles dynamic shapes (e.g.,
     * growing KV cache in autoregressive generation).
     * Compilation is explicit unless SameDiff.dspAutoCompileEnabled is true.
     *
     * @return results as SDValue map, or null if DynamicShapePlan execution is not possible
     */
    protected Map<String, SDValue> executeDynamicShapePlanBased(ForwardExecutionDAG dag,
                                                               Map<String, INDArray> placeholderArrays,
                                                               Set<String> allRequired,
                                                               List<String> requestedOutputs) {
        // DynamicShapePlan is allocation-heavy: force-enable the ArrayCacheMemoryMgr cache
        // to reduce cudaMalloc churn for growing shapes (KV cache).
        // NOTE: Cache must be enabled before the plan check — it also prevents premature
        // DataBuffer close in the standard executeOperations path.
        // Growth factor is scoped to DSP execution only (via withGrowthFactor) to prevent
        // leaking oversized buffers into standard execution if the plan returns null.
        if (mmgr instanceof ArrayCacheMemoryMgr) {
            ArrayCacheMemoryMgr.setEnableCache(true);
        }

        DynamicShapePlan plan = getOrCompileDynamicShapePlan(
                dag, requestedOutputs, sameDiff.isDspAutoCompileEnabled());
        if (plan == null) {
            log.debug("DynamicShapePlan not available (auto-compile disabled or compilation unsupported), using standard path");
            // Undo the cache enable — standard executeOperations() must NOT cache arrays
            // because nothing drains the cache between operations in the standard path.
            if (mmgr instanceof ArrayCacheMemoryMgr) {
                ArrayCacheMemoryMgr.setEnableCache(false);
            }
            return null;
        }

        // Get or create DynamicShapePlanExecutor (ThreadLocal per InferenceSession).
        // The executor persists between output() calls on the same session, so the slot
        // array cache enables O(1) array reuse in autoregressive decode steps.
        DynamicShapePlanExecutor executor = dynamicShapePlanExecutorTl.get();
        if (executor == null) {
            executor = new DynamicShapePlanExecutor(sameDiff, mmgr);
            dynamicShapePlanExecutorTl.set(executor);
        }

        if (!executor.isNativePlanCompiled(plan)) {
            executor.compileNativePlan(plan, null, sameDiff.isDspFallbackToAutoIfTritonUnavailable());
        }

        // Execute with scoped growth factor.
        // Growth factor 1.1x is active ONLY during DSP execution — if the plan returned
        // null above, the growth factor stays at its default (1.0) and standard execution
        // won't create oversized buffers that leak stale data via cuBLAS partial writes.
        long memBeforeExec = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemoryDefault() / (1024 * 1024);
        Map<String, INDArray> rawResults;
        try (AutoCloseable gfScope = ArrayCacheMemoryMgr.withGrowthFactor(1.1)) {
            rawResults = executor.execute(plan, placeholderArrays);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("DSP execution failed", e);
        }
        long memAfterExec = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemoryDefault() / (1024 * 1024);
        if (memBeforeExec - memAfterExec > 10) {
            log.info("[DSP_MEM_INNER] executor.execute consumed {}MB (before={}MB after={}MB) frozen={}",
                    memBeforeExec - memAfterExec, memBeforeExec, memAfterExec,
                    executor.isShapesFrozen());
        }

        // Wrap results as SDValues
        Map<String, SDValue> sdResults = new LinkedHashMap<>();
        for (String outputName : requestedOutputs) {
            INDArray arr = rawResults.get(outputName);
            if (arr != null) {
                sdResults.put(outputName, SDValue.create(arr));
            }
        }

        return sdResults;
    }

    private Map<String, SDValue> executeOperations(ForwardExecutionDAG dag,
                                                   Map<String, INDArray> placeholderValues,
                                                   Map<String, SDValue> otherPlaceholderValues,
                                                   Set<String> allRequired,
                                                   List<Listener> listeners,
                                                   At at,
                                                   MultiDataSet batch) {

        Map<String, SDValue> variableValues = new LinkedHashMap<>();
        Map<String, SDValue> results = new LinkedHashMap<>();
        Set<String> completedOps = new LinkedHashSet<>();

        // Reset live DataBuffer reference tracking for this execution pass.
        liveDataBufferRefs().clear();

        // Store DAG reference so sub-methods can look up variable consumers
        // for last-consumer-based liveness tracking without passing DAG through every call.
        currentExecutionDAG = dag;

        // Initialize constants, variables, and placeholders
        long tInit0 = TIMING_ENABLED ? System.nanoTime() : 0;
        initializeValues(variableValues, dag, placeholderValues, otherPlaceholderValues);
        if (TIMING_ENABLED) timingInitValuesNs = System.nanoTime() - tInit0;


        // Execute operations in topological order.
        // Use getExecutionOrder() (not getFrameAwareExecutionOrder()) to preserve
        // dependency ordering for control flow ops. Frame-aware ordering incorrectly
        // groups Enter ops into child frames, placing them after their consumers.
        List<ExecutionNode> executionOrder = dag.getExecutionOrder();

        // Suppress autoGc during InferenceSession execution. InferenceSession now manages
        // its own intermediate array lifecycle via consumer-based OpDep dependencies:
        // arrays are freed when their last consumer op completes. The DeallocatorService's
        // autoGcWindow (default 100ms) calls System.gc() periodically, causing Full GC pauses
        // that waste time since intermediates are freed explicitly.
        // Use Integer.MAX_VALUE instead of 0 since CudaConfiguration.setNoGcWindowMs rejects < 1.
        int savedAutoGcWindow = Nd4j.getMemoryManager().getAutoGcWindow();
        Nd4j.getMemoryManager().setAutoGcWindow(Integer.MAX_VALUE);

        try {
        // Detect while-loop regions and build skip sets for control flow
        Set<String> skipOps = new HashSet<>();
        Map<String, WhileLoopRegion> whileLoopRegions = detectWhileLoopRegions(dag);
        Set<String> opsInWhileLoops = new HashSet<>();
        for (WhileLoopRegion r : whileLoopRegions.values()) {
            opsInWhileLoops.addAll(r.allOps());
        }
        // Track which while-loop regions have been executed
        Set<String> executedWhileLoops = new HashSet<>();

        if (log.isDebugEnabled() && !whileLoopRegions.isEmpty()) {
            for (Map.Entry<String, WhileLoopRegion> rEntry : whileLoopRegions.entrySet()) {
                WhileLoopRegion r = rEntry.getValue();
                log.debug("While-loop region '{}': {} merges, {} condOps, {} switches, {} bodyOps, {} nextIter, {} exits",
                        r.frameName, r.mergeOps.size(), r.condOps.size(),
                        r.switchOps.size(), r.bodyOps.size(), r.nextIterOps.size(), r.exitOps.size());
            }
        }

        int totalOps = executionOrder.size();
        int opIdx = 0;
        for (ExecutionNode node : executionOrder) {
            opIdx++;
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT ||
                    node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET) {
                // Skip - already handled in initialization
                continue;
            }

            String opName = node.getOperationName();

            // Skip ops marked inactive (e.g., inactive if-branch after Switch)
            if (skipOps.contains(opName)) {
                continue;
            }

            // If this op is part of a while-loop, check if it's the first Merge
            // of that region and execute the entire loop
            if (opsInWhileLoops.contains(opName)) {
                // Find which region this op belongs to
                String regionKey = null;
                for (Map.Entry<String, WhileLoopRegion> entry : whileLoopRegions.entrySet()) {
                    if (entry.getValue().allOps().contains(opName)) {
                        regionKey = entry.getKey();
                        break;
                    }
                }
                if (regionKey != null && !executedWhileLoops.contains(regionKey)) {
                    // Check if we're at the first op of this region in execution order
                    WhileLoopRegion region = whileLoopRegions.get(regionKey);
                    if (region.mergeOps.contains(opName)) {
                        // Execute the entire while loop
                        executeWhileLoop(region, dag, variableValues, completedOps,
                                allRequired, listeners, at, batch);
                        executedWhileLoops.add(regionKey);
                    }
                }
                // Skip individual execution - handled by executeWhileLoop
                if (executedWhileLoops.contains(regionKey)) {
                    continue;
                }
            }

            // For Merge nodes (in if-conditionals), relax readiness: only need
            // at least one input value to be non-null, not all dependencies completed
            DifferentialFunction nodeOp = node.getOperation();
            if (nodeOp instanceof Merge) {
                // Check if at least one input has a value
                boolean hasInput = false;
                for (String input : node.getInputVariables()) {
                    if (variableValues.get(input) != null) {
                        hasInput = true;
                        break;
                    }
                }
                if (!hasInput) {
                    // All inputs null/missing — this Merge is in a fully inactive branch.
                    // Skip it and propagate skip to downstream consumers.
                    log.debug("Merge {} has no available inputs (fully inactive branch), skipping", opName);
                    completedOps.add(opName);
                    // BFS from Merge outputs to mark downstream ops for skipping
                    for (String output : node.getOutputVariables()) {
                        Queue<String> inactiveQueue = new LinkedList<>();
                        Set<String> consumers = dag.getVariableConsumers().get(output);
                        if (consumers != null) inactiveQueue.addAll(consumers);
                        while (!inactiveQueue.isEmpty()) {
                            String consumer = inactiveQueue.poll();
                            if (skipOps.contains(consumer)) continue;
                            ExecutionNode consumerNode = dag.getOperationNodes().get(consumer);
                            if (consumerNode == null) continue;
                            // Stop at next Merge — let it decide based on its own inputs
                            if (consumerNode.getOperation() instanceof Merge) continue;
                            skipOps.add(consumer);
                            for (String consumerOutput : consumerNode.getOutputVariables()) {
                                Set<String> nextConsumers = dag.getVariableConsumers().get(consumerOutput);
                                if (nextConsumers != null) inactiveQueue.addAll(nextConsumers);
                            }
                        }
                    }
                    continue;
                }
            } else {
                // Check dependencies are satisfied
                if (!node.isReadyToExecute(completedOps)) {
                    Set<String> missing = new HashSet<>(node.getDependsOnOperations());
                    missing.removeAll(completedOps);
                    throw new IllegalStateException("Operation " + opName +
                            " not ready. Missing dependencies: " + missing);
                }
            }

            // Execute the operation
            executeNode(node, variableValues, allRequired, listeners, at, batch);

            // Mark as completed
            long tDep0 = TIMING_ENABLED ? System.nanoTime() : 0;
            completedOps.add(opName);

            // After Switch, mark inactive branch ops for skipping
            if (nodeOp instanceof Switch) {
                markInactiveBranchForSkipping(node, dag, variableValues, skipOps);
            }

            // NOTE: We intentionally do NOT sync results to nodeValueOutputs here.
            // nodeValueOutputs is part of AbstractSession's old execution path and is
            // never read during our executeOperations flow. Writing to it causes unbounded
            // memory growth during autoregressive generation (5000+ intermediate arrays
            // per step x hundreds of steps). All lookups during execution use the local
            // variableValues map instead.

            // Mark operation dependency as satisfied in arrayUseTracker
            Dep opDep = new OpDep(opName, OUTER_FRAME, 0, null);
            arrayUseTracker().markSatisfied(opDep, true);

            if (TIMING_ENABLED) timingDepTrackNs += System.nanoTime() - tDep0;

            // Mid-execution release: free arrays whose all consumers have completed.
            // Uses liveDataBufferRefs to safely skip arrays whose DataBuffer is shared
            // with another live array (reshape/permute/transpose views).
            // Only arrays whose buffer is exclusively owned (ref count == 1) are closed.
            long tRelease0 = TIMING_ENABLED ? System.nanoTime() : 0;
            if (arrayUseTracker().hasNewAllSatisfied()) {
                List<SDValue> satisfied = arrayUseTracker().getNewAllSatisfiedList();
                for (SDValue value : satisfied) {
                    if (value == null) continue;
                    switch (value.getSdValueType()) {
                        case TENSOR: {
                            INDArray arr = value.getTensorValue();
                            if (arr == null || arr.wasClosed() || freedArrays().contains(arr.getId())) break;
                            if (arr.data() != null && arr.data().isConstant()) {
                                // Never close constants, but DO untrack so ref counts stay accurate
                                untrackLiveBuffer(arr);
                                break;
                            }
                            // Safe to close only when this array's DataBuffer is not shared with
                            // any other live tracked array (view safety check).
                            if (isBufferExclusivelyOwned(arr)) {
                                untrackLiveBuffer(arr);
                                mmgr.release(arr);
                                freedArrays().add(arr.getId());
                            }
                            // If buffer is shared, leave it — it will be cleaned up at end-of-execution.
                            break;
                        }
                        case LIST: {
                            if (value.getListValue() != null) {
                                for (INDArray arr : value.getListValue()) {
                                    if (arr == null || arr.wasClosed() || freedArrays().contains(arr.getId())) continue;
                                    if (arr.data() != null && arr.data().isConstant()) continue;
                                    if (isBufferExclusivelyOwned(arr)) {
                                        untrackLiveBuffer(arr);
                                        mmgr.release(arr);
                                        freedArrays().add(arr.getId());
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            if (TIMING_ENABLED) timingArrayReleaseNs += System.nanoTime() - tRelease0;
        }

        } catch (Exception execException) {
            // Execution failed — close intermediate DataBuffers to prevent GPU memory
            // accumulation across retries/frames. Without this, 1953+ ops' intermediates
            // leak on each failed vision encoder frame → GPU OOM on subsequent frames.
            try {
                emergencyCleanupIntermediates(dag, placeholderValues, variableValues);
            } catch (Exception cleanupEx) {
                log.warn("Emergency cleanup failed: {}", cleanupEx.getMessage());
            }
            throw execException;
        } finally {
            // Restore autoGc window and cross-device routing to previous values
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            Nd4j.getMemoryManager().setAutoGcWindow(savedAutoGcWindow);
            currentExecutionDAG = null;
        }

        // Ensure all async CUDA operations complete before accessing results.
        // Without this, dup() may copy stale/incomplete data from GPU buffers
        // that haven't finished being written by the last op's kernel.
        Nd4j.getExecutioner().commit();

        // Collect output arrays and detach them (create independent copies).
        // This breaks shared data buffer references with intermediates so
        // intermediates can be safely closed to free GPU memory immediately.
        long tDup0 = TIMING_ENABLED ? System.nanoTime() : 0;
        for(String output : allRequired) {
            if(!variableValues.containsKey(output)) {
                log.debug("Output '{}' not in variableValues after execution, skipping", output);
                continue;
            }
            SDValue val = variableValues.get(output);
            if (val == null) {
                log.debug("Output '{}' has null value (inactive control flow branch), skipping", output);
                continue;
            }
            if (val != null && val.getSdValueType() == SDValueType.TENSOR) {
                INDArray arr = val.getTensorValue();
                if (arr != null) {
                    // dup() creates an independent copy with its own data buffer
                    results.put(output, SDValue.create(arr.dup()));
                } else {
                    results.put(output, val);
                }
            } else if (val != null && val.getSdValueType() == SDValueType.LIST) {
                List<INDArray> list = val.getListValue();
                if (list != null) {
                    List<INDArray> duped = new java.util.ArrayList<>(list.size());
                    for (INDArray arr : list) {
                        duped.add(arr != null ? arr.dup() : null);
                    }
                    results.put(output, SDValue.create(duped));
                } else {
                    results.put(output, val);
                }
            } else {
                results.put(output, val);
            }
        }

        if (TIMING_ENABLED) timingOutputDupNs = System.nanoTime() - tDup0;

        // Handle intermediate arrays: either return to cache or close directly.
        Set<String> preserveNames = new HashSet<>();
        preserveNames.addAll(dag.getConstants());
        preserveNames.addAll(dag.getVariables());
        if (placeholderValues != null) {
            preserveNames.addAll(placeholderValues.keySet());
        }
        boolean cacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();

        if (cacheEnabled) {
            // Cache-enabled path: release to cache, then force-close non-closeable arrays.
            // mmgr.release() silently DROPS non-closeable arrays (closeable() returns false
            // when DataBuffer is oversized from growth factor). These dropped arrays leak GPU
            // memory unless we force-close their DataBuffers directly.
            int releasedToCache = 0;
            int forceClosedCount = 0;
            long forceClosedBytes = 0;
            Set<INDArray> seen = Collections.newSetFromMap(new IdentityHashMap<>());
            IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();

            // Collect protected DataBuffers (constants, variables, placeholders)
            for (String name : preserveNames) {
                SDValue value = variableValues.get(name);
                if (value == null) continue;
                if (value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray t = value.getTensorValue();
                    if (t != null && !t.wasClosed() && t.data() != null) {
                        protectedBuffers.put(t.data(), Boolean.TRUE);
                    }
                } else if (value.getSdValueType() == SDValueType.LIST && value.getListValue() != null) {
                    for (INDArray a : value.getListValue()) {
                        if (a != null && !a.wasClosed() && a.data() != null) {
                            protectedBuffers.put(a.data(), Boolean.TRUE);
                        }
                    }
                }
            }

            // Two-pass approach: HashMap iteration order is non-deterministic, so a
            // non-closeable view might be processed BEFORE its closeable parent. If we
            // force-close the view's DataBuffer before the parent is cached, the shared
            // buffer is destroyed and the cached parent gets a dead buffer.
            //
            // Pass 1: Cache all closeable arrays and collect their DataBuffers as protected.
            // Pass 2: Force-close non-closeable arrays, skipping any with protected DataBuffers.
            List<INDArray> nonCloseableArrays = new ArrayList<>();

            // Pass 1: cache closeable arrays, build complete protection set
            for (Map.Entry<String, SDValue> entry : variableValues.entrySet()) {
                if (preserveNames.contains(entry.getKey())) continue;
                SDValue value = entry.getValue();
                if (value == null) continue;
                switch (value.getSdValueType()) {
                    case TENSOR:
                        INDArray tensor = value.getTensorValue();
                        if (tensor != null && !tensor.wasClosed() && seen.add(tensor)
                                && !freedArrays().contains(tensor.getId())) {
                            if (tensor.closeable()) {
                                mmgr.release(tensor);
                                DataBuffer cachedBuf = tensor.data();
                                if (cachedBuf != null) {
                                    protectedBuffers.put(cachedBuf, Boolean.TRUE);
                                }
                            } else {
                                nonCloseableArrays.add(tensor);
                            }
                            freedArrays().add(tensor.getId());
                            releasedToCache++;
                        }
                        break;
                    case LIST:
                        if (value.getListValue() != null) {
                            for (INDArray arr : value.getListValue()) {
                                if (arr != null && !arr.wasClosed() && seen.add(arr)
                                        && !freedArrays().contains(arr.getId())) {
                                    if (arr.closeable()) {
                                        mmgr.release(arr);
                                        DataBuffer cachedBuf = arr.data();
                                        if (cachedBuf != null) {
                                            protectedBuffers.put(cachedBuf, Boolean.TRUE);
                                        }
                                    } else {
                                        nonCloseableArrays.add(arr);
                                    }
                                    freedArrays().add(arr.getId());
                                    releasedToCache++;
                                }
                            }
                        }
                        break;
                }
            }

            // Pass 2: route non-closeable arrays through mmgr.release() which handles
            // them via the deferred close path. Direct force-close here is unsafe because
            // it can destroy DataBuffers shared with cached arrays (the cache returned an
            // oversized buffer, making the array non-closeable, but the buffer is still
            // valid and referenced by the cached entry). The deferred close path in
            // closeDeferredBuffers() protects all cached arrays' DataBuffers.
            for (INDArray arr : nonCloseableArrays) {
                mmgr.release(arr);
            }
            variableValues.clear();

            // Close deferred buffers from non-closeable arrays dropped by release().
            // Now that all ops are complete, views are no longer in use, so it's safe.
            long[] deferredStats = ArrayCacheMemoryMgr.closeDeferredBuffers(protectedBuffers);
            int deferredClosed = (int) deferredStats[0];
            long deferredBytes = deferredStats[1];

            // Close ALL cached arrays to reclaim GPU memory for the next step.
            // The cache served its purpose during this step's execution: preventing premature
            // DataBuffer close for view-producing ops (reshape_no_copy, permute, etc.).
            // Now that execution is complete and outputs have been dup'd, we can safely
            // close all cached arrays. cudaFreeAsync returns memory to the pool, and
            // cudaMallocAsync reuses freed pool entries — no explicit trim needed.
            // IMPORTANT: Do NOT call trimMemoryPoolOnStream here. Trim physically frees
            // pool entries back to the OS, but can corrupt live arrays whose addresses
            // coincide with freed pool entries (the pool reuses virtual addresses).
            if (mmgr instanceof ArrayCacheMemoryMgr) {
                ((ArrayCacheMemoryMgr) mmgr).close(protectedBuffers);
            } else {
                mmgr.close();
            }
            Nd4j.getExecutioner().commit();

            try (LongPointer pu = new LongPointer(1);
                 LongPointer pr = new LongPointer(1)) {
                NativeOpsHolder.getInstance().getDeviceNativeOps().getMemoryPoolStats(
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(), pu, pr);
                log.debug("Cleanup: released={} to cache, force-closed={} ({}MB), deferred-closed={} ({}MB), poolUsed={}MB",
                        releasedToCache - forceClosedCount, forceClosedCount, forceClosedBytes / (1024 * 1024),
                        deferredClosed, deferredBytes / (1024 * 1024),
                        pu.get() / (1024 * 1024));
            } catch (Exception e) {
                log.debug("Cleanup: released={} to cache, force-closed={} ({}MB), deferred-closed={} ({}MB)",
                        releasedToCache - forceClosedCount, forceClosedCount, forceClosedBytes / (1024 * 1024),
                        deferredClosed, deferredBytes / (1024 * 1024));
            }
        } else {
            // No-cache path: close DataBuffers directly to free GPU memory immediately.
            // First, collect DataBuffers belonging to constants/variables/placeholders.
            IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();
            for (String name : preserveNames) {
                SDValue value = variableValues.get(name);
                if (value == null) continue;
                switch (value.getSdValueType()) {
                    case TENSOR:
                        INDArray tensor = value.getTensorValue();
                        if (tensor != null && !tensor.wasClosed() && tensor.data() != null) {
                            protectedBuffers.put(tensor.data(), Boolean.TRUE);
                        }
                        break;
                    case LIST:
                        if (value.getListValue() != null) {
                            for (INDArray arr : value.getListValue()) {
                                if (arr != null && !arr.wasClosed() && arr.data() != null) {
                                    protectedBuffers.put(arr.data(), Boolean.TRUE);
                                }
                            }
                        }
                        break;
                }
            }

            IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
            for (Map.Entry<String, SDValue> entry : variableValues.entrySet()) {
                if (preserveNames.contains(entry.getKey())) continue;
                SDValue value = entry.getValue();
                if (value == null) continue;
                switch (value.getSdValueType()) {
                    case TENSOR:
                        INDArray tensor = value.getTensorValue();
                        if (tensor != null && !tensor.wasClosed() && tensor.data() != null) {
                            boolean isProtected = protectedBuffers.containsKey(tensor.data());
                            if (!isProtected) {
                                uniqueBuffers.put(tensor.data(), Boolean.TRUE);
                            }
                        }
                        break;
                    case LIST:
                        if (value.getListValue() != null) {
                            for (INDArray arr : value.getListValue()) {
                                if (arr != null && !arr.wasClosed() && arr.data() != null
                                        && !protectedBuffers.containsKey(arr.data())) {
                                    uniqueBuffers.put(arr.data(), Boolean.TRUE);
                                }
                            }
                        }
                        break;
                }
            }
            log.info("CLEANUP: variableValues={} entries, preserveNames={}, uniqueBuffers={}, protectedBuffers={}",
                    variableValues.size(), preserveNames.size(), uniqueBuffers.size(), protectedBuffers.size());
            variableValues.clear();

            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            Nd4j.getExecutioner().commit();

            // Pool stats BEFORE cleanup
            long poolUsedBeforeMB = 0;
            try (LongPointer pBefore = new LongPointer(2);
                 LongPointer pResBefore = new LongPointer(1)) {
                nativeOps.getMemoryPoolStats(Nd4j.getAffinityManager().getDeviceForCurrentThread(), pBefore, pResBefore);
                poolUsedBeforeMB = pBefore.get(0) / (1024 * 1024);
            } catch (Exception e) { /* ignore */ }

            int releasedCount = 0;
            long releasedBytes = 0;
            int skippedConstant = 0, skippedAttached = 0, skippedReleased = 0;
            int forceClosedConstant = 0;
            long forceClosedConstantBytes = 0;
            int closeErrors = 0;
            nativeOps.dbCloseResetDiagnostics();
            OpaqueDataBuffer.resetCloseBufferStats();
            for (DataBuffer buf : uniqueBuffers.keySet()) {
                if (buf.closeable()) {
                    try {
                        releasedBytes += buf.length() * buf.getElementSize();
                        buf.close();
                        releasedCount++;
                    } catch (Exception e) {
                        closeErrors++;
                        log.info("Error closing buffer: {}", e.getMessage());
                    }
                } else {
                    if (buf.wasClosed()) {
                        skippedReleased++;
                    } else if (buf.isAttached()) {
                        skippedAttached++;
                    } else if (buf.isConstant()) {
                        // Buffer is marked constant — leave it alone. Force-closing constant
                        // buffers was destroying model constant values needed by subsequent
                        // DSP executions (scalar epsilon, scale, attention mask values).
                        // The memory "leak" from poisoned intermediates is acceptable — they
                        // are typically small scalars and get cleaned up on session/model close.
                        skippedConstant++;
                    }
                }
            }
            try (LongPointer nativeStats = new LongPointer(9)) {
                nativeOps.dbCloseGetDiagnostics(nativeStats);
                log.info("Native dbClose stats: total={}, null={}, constant={}, alreadyClosed={}, noDataBuffer={}, notOwner={}, deviceError={}, deleted={}, freedBytes={}MB",
                        nativeStats.get(0), nativeStats.get(1), nativeStats.get(2), nativeStats.get(3),
                        nativeStats.get(4), nativeStats.get(5), nativeStats.get(6), nativeStats.get(7),
                        nativeStats.get(8) / (1024 * 1024));
                nativeOps.dbCloseResetDiagnostics();
                log.info("Java {}", OpaqueDataBuffer.getCloseBufferStats());
            } catch (Exception e) {
                log.debug("Could not get native dbClose diagnostics: {}", e.getMessage());
            }
            if (closeErrors > 0) {
                log.warn("Close errors: {}", closeErrors);
            }
            if (releasedCount > 0 || uniqueBuffers.size() > 0) {
                // commit() syncs the current thread's execution stream
                Nd4j.getExecutioner().commit();
                try {
                    // Trim all devices — cross-device frees use stream 0 on target device
                    for (int d = 0; d < numDevices; d++) {
                        nativeOps.trimMemoryPoolOnStream(d, null);
                    }
                } catch (Exception e) {
                    log.debug("Failed to trim memory pool: {}", e.getMessage());
                }

                // Pool stats after cleanup + trim
                long poolUsedMB = 0;
                try (LongPointer pStats = new LongPointer(2);
                     LongPointer pResStats = new LongPointer(1)) {
                    nativeOps.getMemoryPoolStats(Nd4j.getAffinityManager().getDeviceForCurrentThread(), pStats, pResStats);
                    poolUsedMB = pStats.get(0) / (1024 * 1024);
                } catch (Exception e) { /* ignore */ }
                log.info("CLEANUP: Released {} of {} unique data buffers ({} MB), force-closed {} constant-poisoned ({}MB). Pool: before={}MB after={}MB delta={}MB. Skipped: constant={}, attached={}, released={}",
                        releasedCount, uniqueBuffers.size(), releasedBytes / (1024 * 1024),
                        forceClosedConstant, forceClosedConstantBytes / (1024 * 1024),
                        poolUsedBeforeMB, poolUsedMB, poolUsedMB - poolUsedBeforeMB,
                        skippedConstant, skippedAttached, skippedReleased);
            }
        }

        // Clear cross-device constant replica cache so replicas are freed between forward passes.
        // Without this, constant replicas accumulate indefinitely in multi-device execution.
        Nd4j.getExecutioner().clearConstantReplicaCache();

        return results;
    }



    /**
     * Get all dependent values for a variable as a formatted string.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @return Formatted string with dependent values
     */
    public String getDependentValuesString(Map<String, SDValue> variableValues, String variableName) {
        Map<String, String> deps = getDependentValuesMap(variableValues, variableName);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : deps.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Get all dependent values for a variable as a map.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @return Map of variable names to their values
     */
    public Map<String, String> getDependentValuesMap(Map<String, SDValue> variableValues, String variableName) {
        Map<String, String> result = new LinkedHashMap<>();
        Set<String> visited = new HashSet<>();
        collectDependentValues(variableValues, variableName, result, visited);
        return result;
    }

    private void collectDependentValues(Map<String, SDValue> variableValues, String varName,
                                        Map<String, String> result, Set<String> visited) {
        if (visited.contains(varName)) {
            return;
        }
        visited.add(varName);

        // Add this variable's value
        SDValue value = variableValues.get(varName);
        if (value != null) {
            result.put(varName, formatValue(value));
        }

        // Find the op that produces this variable
        for (SameDiffOp op : sameDiff.getOps().values()) {
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                // Collect values from all inputs
                if (op.getInputsToOp() != null) {
                    for (String input : op.getInputsToOp()) {
                        collectDependentValues(variableValues, input, result, visited);
                    }
                }
                break;
            }
        }
    }

    private String formatValue(SDValue value) {
        if (value.getSdValueType() == SDValueType.TENSOR) {
            INDArray arr = value.getTensorValue();
            if (arr == null) return "null";
            if (arr.isScalar()) return String.valueOf(arr.getDouble(0));
            return Arrays.toString(arr.shape()) + " = " + arr.toString().replaceAll("\\s+", " ").trim();
        } else if (value.getSdValueType() == SDValueType.LIST) {
            return "List[" + value.getListValue().size() + "]";
        }
        return value.toString();
    }
    private void initializeValues(Map<String, SDValue> variableValues,
                                  ForwardExecutionDAG dag,
                                  Map<String, INDArray> placeholderValues,
                                  Map<String, SDValue> otherPlaceholderValues) {

        // Use cached constant/variable values to avoid thousands of PatriciaTrie lookups per call.
        // Constants and model weights don't change between autoregressive steps.
        Map<String, SDValue> constVarCache = cachedConstVarValues;
        if (constVarCache == null) {
            constVarCache = new HashMap<>();
            for (String constName : dag.getConstants()) {
                INDArray constValue = getConstantOrVariable(constName);
                if (constValue != null) {
                    constVarCache.put(constName, SDValue.create(constValue));
                }
            }
            for (String varName : dag.getVariables()) {
                INDArray varValue = getConstantOrVariable(varName);
                if (varValue != null) {
                    constVarCache.put(varName, SDValue.create(varValue));
                }
            }
            cachedConstVarValues = constVarCache;
        }
        variableValues.putAll(constVarCache);

        // Initialize placeholders (these change every call)
        if (placeholderValues != null) {
            for (Map.Entry<String, INDArray> entry : placeholderValues.entrySet()) {
                variableValues.put(entry.getKey(), SDValue.create(entry.getValue()));
            }
        }

        if (otherPlaceholderValues != null) {
            variableValues.putAll(otherPlaceholderValues);
        }
    }

// =============================================================================
// OVERRIDE 4: Execute single node (add this to InferenceSession)
// =============================================================================

    /**
     * Emergency cleanup: close intermediate DataBuffers when execution fails.
     * Protects constants, variables, and placeholder DataBuffers from being closed.
     */
    private void emergencyCleanupIntermediates(ForwardExecutionDAG dag, Map<String, INDArray> placeholderValues, Map<String, SDValue> variableValues) {
        Set<String> preserveNames = new HashSet<>();
        if (dag != null) {
            preserveNames.addAll(dag.getConstants());
            preserveNames.addAll(dag.getVariables());
        }
        if (placeholderValues != null) {
            preserveNames.addAll(placeholderValues.keySet());
        }

        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();
        for (String name : preserveNames) {
            SDValue value = variableValues.get(name);
            if (value == null) continue;
            if (value.getSdValueType() == SDValueType.TENSOR) {
                INDArray t = value.getTensorValue();
                if (t != null && !t.wasClosed() && t.data() != null) {
                    protectedBuffers.put(t.data(), Boolean.TRUE);
                }
            }
        }

        int closed = 0;
        for (Map.Entry<String, SDValue> entry : variableValues.entrySet()) {
            if (preserveNames.contains(entry.getKey())) continue;
            SDValue value = entry.getValue();
            if (value == null) continue;
            if (value.getSdValueType() == SDValueType.TENSOR) {
                INDArray t = value.getTensorValue();
                if (t != null && !t.wasClosed() && t.data() != null
                        && !protectedBuffers.containsKey(t.data())) {
                    try {
                        if (t.data().isConstant()) t.data().setConstant(false);
                        t.data().close();
                        closed++;
                    } catch (Exception e) { /* ignore */ }
                }
            }
        }
        variableValues.clear();
        if (closed > 0) {
            log.info("Emergency cleanup: closed {} intermediate DataBuffers after execution failure", closed);
        }
    }

    /**
     * Formats a shape array as a readable string for debug output.
     */
    private String formatShape(long[] shape) {
        if (shape == null) return "null";
        if (shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Formats input information for debug output.
     */
    private String formatInputsDebug(ExecutionNode node, Map<String, SDValue> variableValues) {
        StringBuilder sb = new StringBuilder();
        List<String> inputs = node.getInputVariables();
        if (inputs == null || inputs.isEmpty()) {
            sb.append("    Inputs: (none)\n");
        } else {
            sb.append("    Inputs:\n");
            for (int i = 0; i < inputs.size(); i++) {
                String inputName = inputs.get(i);
                SDValue value = variableValues.get(inputName);
                sb.append("      [").append(i).append("] ").append(inputName).append(": ");
                if (value == null) {
                    sb.append("null");
                } else if (value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = value.getTensorValue();
                    if (arr == null) {
                        sb.append("null tensor");
                    } else {
                        sb.append("shape=").append(formatShape(arr.shape()))
                          .append(", dtype=").append(arr.dataType());
                    }
                } else if (value.getSdValueType() == SDValueType.LIST) {
                    List<INDArray> list = value.getListValue();
                    sb.append("list[").append(list != null ? list.size() : 0).append("]");
                } else {
                    sb.append(value.getSdValueType());
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Formats output information for debug output.
     */
    private String formatOutputsDebug(ExecutionNode node, Map<String, SDValue> variableValues) {
        StringBuilder sb = new StringBuilder();
        List<String> outputs = node.getOutputVariables();
        if (outputs == null || outputs.isEmpty()) {
            sb.append("    Outputs: (none)\n");
        } else {
            sb.append("    Outputs:\n");
            for (int i = 0; i < outputs.size(); i++) {
                String outputName = outputs.get(i);
                SDVariable outVar = sameDiff.getVariable(outputName);
                sb.append("      [").append(i).append("] ").append(outputName);
                if (outVar != null) {
                    sb.append(": dtype=").append(outVar.dataType());
                    if (outVar.getShape() != null) {
                        sb.append(", expectedShape=").append(formatShape(outVar.getShape()));
                    }
                }
                // Check if already computed
                SDValue computed = variableValues.get(outputName);
                if (computed != null && computed.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = computed.getTensorValue();
                    if (arr != null) {
                        sb.append(" -> actualShape=").append(formatShape(arr.shape()));
                    }
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Formats comprehensive node metadata for error messages.
     * This includes the operation name, input variable names with their shapes/values,
     * and expected output variable names.
     *
     * @param node the execution node that failed
     * @param variableValues current variable values map
     * @return formatted string with node metadata for error context
     */
    private String formatNodeErrorContext(ExecutionNode node, Map<String, SDValue> variableValues) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n========================================\n");
        sb.append("SAMEDIFF NODE EXECUTION ERROR CONTEXT\n");
        sb.append("========================================\n");

        String opName = node.getOperationName();
        sb.append("Operation Name: ").append(opName).append("\n");

        // Get the SameDiff op for additional info
        SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
        if (sameDiffOp != null) {
            DifferentialFunction op = sameDiffOp.getOp();
            if (op != null) {
                sb.append("Op Type: ").append(op.opName()).append("\n");
                sb.append("Op Class: ").append(op.getClass().getSimpleName()).append("\n");
            }
        }

        // Input variables with details
        List<String> inputs = node.getInputVariables();
        sb.append("\nINPUT VARIABLES (").append(inputs != null ? inputs.size() : 0).append("):\n");
        if (inputs != null && !inputs.isEmpty()) {
            for (int i = 0; i < inputs.size(); i++) {
                String inputName = inputs.get(i);
                sb.append("  [").append(i).append("] '").append(inputName).append("'");

                SDValue value = variableValues.get(inputName);
                if (value == null) {
                    sb.append(" -> VALUE NOT FOUND IN CONTEXT");
                } else if (value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = value.getTensorValue();
                    if (arr == null) {
                        sb.append(" -> null tensor");
                    } else {
                        sb.append("\n      Shape: ").append(formatShape(arr.shape()));
                        sb.append(", DataType: ").append(arr.dataType());
                        sb.append(", Closed: ").append(arr.wasClosed());
                        if (arr.data() != null && !arr.wasClosed()) {
                            try {
                                sb.append("\n      Buffer Address: 0x").append(Long.toHexString(arr.data().pointer().address()));
                            } catch (Exception e) {
                                sb.append("\n      Buffer Address: <released>");
                            }
                        }
                        // For scalars, show the value
                        if (arr.isScalar() && !arr.wasClosed()) {
                            try {
                                if (arr.dataType() == DataType.INT64) {
                                    long val = arr.getLong(0);
                                    sb.append("\n      Scalar Value: ").append(val);
                                    sb.append(" (hex: 0x").append(Long.toHexString(val)).append(")");
                                } else if (arr.dataType().isFPType()) {
                                    sb.append("\n      Scalar Value: ").append(arr.getDouble(0));
                                }
                            } catch (Exception e) {
                                sb.append("\n      Scalar Value: <error reading value>");
                            }
                        }
                    }
                } else if (value.getSdValueType() == SDValueType.LIST) {
                    List<INDArray> list = value.getListValue();
                    sb.append(" -> list[").append(list != null ? list.size() : 0).append("]");
                } else {
                    sb.append(" -> ").append(value.getSdValueType());
                }
                sb.append("\n");
            }
        } else {
            sb.append("  (no inputs)\n");
        }

        // Output variables
        List<String> outputs = node.getOutputVariables();
        sb.append("\nOUTPUT VARIABLES (").append(outputs != null ? outputs.size() : 0).append("):\n");
        if (outputs != null && !outputs.isEmpty()) {
            for (int i = 0; i < outputs.size(); i++) {
                String outputName = outputs.get(i);
                sb.append("  [").append(i).append("] '").append(outputName).append("'");

                SDVariable outVar = sameDiff.getVariable(outputName);
                if (outVar != null) {
                    sb.append(" -> dtype=").append(outVar.dataType());
                    if (outVar.getShape() != null) {
                        sb.append(", expectedShape=").append(formatShape(outVar.getShape()));
                    }
                }
                sb.append("\n");
            }
        } else {
            sb.append("  (no outputs)\n");
        }

        sb.append("========================================\n");
        return sb.toString();
    }

    /**
     * Add consumer-based dependencies for an output variable's SDValue.
     * Instead of using ExecDoneDep (which defers all frees to end-of-execution),
     * this registers an OpDep for each consuming operation. When the last consumer
     * op completes and calls markSatisfied(OpDep), the array becomes releasable
     * immediately, freeing GPU memory mid-execution.
     *
     * Falls back to ExecDoneDep only when no consumer information is available
     * (e.g., when currentExecutionDAG is null or the variable has no consumers).
     *
     * @param outputValue The SDValue wrapping the output array
     * @param varName     The variable name (output of the producing op)
     * @param allRequired Set of required output variable names
     */
    private void addConsumerDependencies(SDValue outputValue, String varName, Set<String> allRequired) {
        // Track this array's DataBuffer for mid-execution release safety.
        // ALL arrays are tracked (including those marked constant) to ensure
        // correct ref counting for shared buffers (reshape/permute views).
        // Without this, a view's parent buffer gets released while the view
        // is still live — causing use-after-free SIGSEGV.
        if (outputValue != null && outputValue.getSdValueType() == SDValueType.TENSOR) {
            INDArray arr = outputValue.getTensorValue();
            if (arr != null && arr.data() != null) {
                trackLiveBuffer(arr);
            }
        }

        if (allRequired.contains(varName)) {
            // This is a final output: protect from deallocation until explicitly released
            arrayUseTracker().addDependency(outputValue, new ReqOutputDep(varName));
            return;
        }

        // Look up consuming ops from the DAG
        ForwardExecutionDAG dag = currentExecutionDAG;
        if (dag != null) {
            Set<ExecutionNode> consumerNodes = dag.getConsumerNodes(varName);
            if (consumerNodes != null && !consumerNodes.isEmpty()) {
                for (ExecutionNode consumerNode : consumerNodes) {
                    Dep d = new OpDep(consumerNode.getOperationName(), OUTER_FRAME, 0, null);
                    arrayUseTracker().addDependency(outputValue, d);
                }
                return;
            }
        }

        // Fallback: no consumer info available, use ExecDoneDep (freed at end of execution)
        arrayUseTracker().addDependency(outputValue, new ExecDoneDep());
    }

    private void executeNode(ExecutionNode node,
                             Map<String, SDValue> variableValues,
                             Set<String> allRequired,
                             List<Listener> listeners,
                             At at,
                             MultiDataSet batch) {

        String opName = node.getOperationName();
        boolean debugMode = Nd4j.getEnvironment().isDebug();
        long startTime = debugMode ? System.nanoTime() : 0;

        if (debugMode) {
            SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
            DifferentialFunction op = sameDiffOp != null ? sameDiffOp.getOp() : null;

            StringBuilder sb = new StringBuilder();
            sb.append("\n┌──────────────────────────────────────────────────────────────────────────────┐\n");
            sb.append("│ EXECUTING: ").append(opName).append("\n");
            sb.append("├──────────────────────────────────────────────────────────────────────────────┤\n");
            sb.append("│ Op Type: ").append(op != null ? op.opName() : "unknown");
            sb.append(" (").append(op != null ? op.getClass().getSimpleName() : "unknown").append(")\n");
            sb.append("│ Node Type: ").append(node.getNodeType()).append("\n");

            // Frame info for control flow
            if (node.getFrameInfo() != null) {
                sb.append("│ Frame: ").append(node.getFrameInfo()).append("\n");
            }

            // Dependencies
            Set<String> deps = node.getDependsOnOperations();
            if (deps != null && !deps.isEmpty()) {
                sb.append("│ Dependencies: ").append(deps).append("\n");
            }

            // Input variables with full details
            sb.append("├──────────────────────────────────────────────────────────────────────────────┤\n");
            sb.append("│ INPUT VARIABLES:\n");
            List<String> inputs = node.getInputVariables();
            if (inputs == null || inputs.isEmpty()) {
                sb.append("│   (none)\n");
            } else {
                for (int i = 0; i < inputs.size(); i++) {
                    String inputName = inputs.get(i);
                    SDVariable inputVar = sameDiff.getVariable(inputName);
                    SDValue value = variableValues.get(inputName);

                    sb.append("│   [").append(i).append("] ").append(inputName).append("\n");
                    if (inputVar != null) {
                        sb.append("│       varType: ").append(inputVar.getVariableType()).append("\n");
                        sb.append("│       dtype: ").append(inputVar.dataType()).append("\n");
                        if (inputVar.getShape() != null) {
                            sb.append("│       declaredShape: ").append(formatShape(inputVar.getShape())).append("\n");
                        }
                    }
                    if (value != null) {
                        if (value.getSdValueType() == SDValueType.TENSOR) {
                            INDArray arr = value.getTensorValue();
                            if (arr != null) {
                                sb.append("│       runtimeShape: ").append(formatShape(arr.shape())).append("\n");
                                sb.append("│       runtimeDtype: ").append(arr.dataType()).append("\n");
                                // Show first few values for small arrays or scalars
                                if (arr.isScalar()) {
                                    sb.append("│       value: ").append(arr.getDouble(0)).append("\n");
                                } else if (arr.length() <= 6) {
                                    sb.append("│       values: ").append(arr.toStringFull()).append("\n");
                                }
                            } else {
                                sb.append("│       runtime: null tensor\n");
                            }
                        } else if (value.getSdValueType() == SDValueType.LIST) {
                            List<INDArray> list = value.getListValue();
                            sb.append("│       runtime: list[").append(list != null ? list.size() : 0).append("]\n");
                        }
                    } else {
                        sb.append("│       runtime: NOT YET COMPUTED\n");
                    }
                }
            }

            // Output variables with expected info
            sb.append("├──────────────────────────────────────────────────────────────────────────────┤\n");
            sb.append("│ OUTPUT VARIABLES:\n");
            List<String> outputs = node.getOutputVariables();
            if (outputs == null || outputs.isEmpty()) {
                sb.append("│   (none)\n");
            } else {
                for (int i = 0; i < outputs.size(); i++) {
                    String outputName = outputs.get(i);
                    SDVariable outputVar = sameDiff.getVariable(outputName);

                    sb.append("│   [").append(i).append("] ").append(outputName).append("\n");
                    if (outputVar != null) {
                        sb.append("│       varType: ").append(outputVar.getVariableType()).append("\n");
                        sb.append("│       dtype: ").append(outputVar.dataType()).append("\n");
                        if (outputVar.getShape() != null) {
                            sb.append("│       expectedShape: ").append(formatShape(outputVar.getShape())).append("\n");
                        }
                    }
                }
            }
            sb.append("└──────────────────────────────────────────────────────────────────────────────┘\n");

            log.info(sb.toString());
        }

        try {
            // Get the operation
            SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
            if (sameDiffOp == null) {
                throw new IllegalStateException("Operation not found: " + opName);
            }

            DifferentialFunction op = sameDiffOp.getOp();

            // Handle special control flow operations directly
            if (op instanceof Identity) {
                executeIdentityNode(node, variableValues);
                return;
            }

            // Using opName() is more robust than instanceof for control flow ops
            String opNameLowerCase = op.opName().toLowerCase();
            switch (opNameLowerCase) {
                case "switch":
                    executeSwitchNode(node, variableValues, op);
                    return;
                case "enter":
                    executeEnterNode(node, variableValues, op);
                    return;
                case "exit":
                    executeExitNode(node, variableValues, op);
                    return;
                case "next_iteration":
                    executeNextIterationNode(node, variableValues, op);
                    return;
                case "merge":
                    executeMergeNode(node, variableValues, op, allRequired);
                    return;
                case "loop_cond":
                    executeLoopCondNode(node, variableValues, op);
                    return;
            }


            if (op instanceof BaseTensorOp) {
                executeTensorArrayNode(node, variableValues, op, allRequired);
                return;
            }

            // For regular operations, use the existing doExec infrastructure
            executeRegularOperation(node, variableValues, allRequired, listeners, at, batch);

        } catch (Exception e) {
            // Format comprehensive node context for the error message
            String nodeContext = formatNodeErrorContext(node, variableValues);
            log.error("Failed to execute operation: {}\n{}", opName, nodeContext, e);
            throw new RuntimeException("Operation execution failed: " + opName + nodeContext, e);
        } finally {
            // Log completion with timing and output shapes
            if (debugMode) {
                long endTime = System.nanoTime();
                double elapsedMs = (endTime - startTime) / 1_000_000.0;

                StringBuilder sb = new StringBuilder();
                sb.append("┌─ COMPLETED: ").append(opName);
                sb.append(" [").append(String.format("%.3f", elapsedMs)).append(" ms] ─────────────────────────────────┐\n");

                // Show actual computed outputs after execution
                List<String> outputs = node.getOutputVariables();
                if (outputs != null && !outputs.isEmpty()) {
                    sb.append("│ COMPUTED OUTPUTS:\n");
                    for (int i = 0; i < outputs.size(); i++) {
                        String outputName = outputs.get(i);
                        SDValue computed = variableValues.get(outputName);

                        sb.append("│   [").append(i).append("] ").append(outputName).append("\n");
                        if (computed != null) {
                            if (computed.getSdValueType() == SDValueType.TENSOR) {
                                INDArray arr = computed.getTensorValue();
                                if (arr != null) {
                                    sb.append("│       shape: ").append(formatShape(arr.shape())).append("\n");
                                    sb.append("│       dtype: ").append(arr.dataType()).append("\n");
                                    // Show values for small arrays or scalars
                                    // All value access is wrapped in try-catch because arrays may be
                                    // on a different GPU device in multi-GPU DSP execution, and
                                    // accessing values triggers CUDA ops that can fail across devices.
                                    if (arr.isScalar()) {
                                        try {
                                            sb.append("│       value: ").append(arr.getDouble(0)).append("\n");
                                        } catch (Exception e) {
                                            sb.append("│       value: (unavailable)\n");
                                        }
                                    } else if (arr.length() <= 6) {
                                        try {
                                            sb.append("│       values: ").append(arr.toStringFull()).append("\n");
                                        } catch (Exception e) {
                                            sb.append("│       values: (unavailable)\n");
                                        }
                                    } else {
                                        // Show summary for larger arrays (skip min/max for non-numeric types)
                                        DataType dt = arr.dataType();
                                        if (dt != DataType.BOOL && dt != DataType.UTF8 && dt != DataType.UTF16 && dt != DataType.UTF32) {
                                            try {
                                                sb.append("│       min: ").append(arr.minNumber()).append(", max: ").append(arr.maxNumber()).append("\n");
                                            } catch (Exception e) {
                                                // min/max may fail for cross-device arrays (CUDA error 700)
                                                sb.append("│       (").append(arr.length()).append(" elements, stats unavailable)\n");
                                            }
                                        } else {
                                            sb.append("│       (non-numeric type, ").append(arr.length()).append(" elements)\n");
                                        }
                                    }
                                } else {
                                    sb.append("│       result: null tensor\n");
                                }
                            } else if (computed.getSdValueType() == SDValueType.LIST) {
                                List<INDArray> list = computed.getListValue();
                                sb.append("│       result: list[").append(list != null ? list.size() : 0).append("]\n");
                            }
                        } else {
                            sb.append("│       result: NOT COMPUTED (null)\n");
                        }
                    }
                }
                sb.append("└──────────────────────────────────────────────────────────────────────────────┘\n");

                log.info(sb.toString());
            }
        }
    }

    private void executeIdentityNode(ExecutionNode node, Map<String, SDValue> variableValues) {
        String opName = node.getOperationName();
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Identity operation " + opName + " has no inputs or outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for Identity operation " + opName);
        }

        variableValues.put(outputVar, inputValue);
    }

    private void executeSwitchNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        Switch switchOp = (Switch) op;
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.size() < 2) {
            throw new IllegalStateException("Switch operation requires at least 2 inputs");
        }

        String dataInput = inputs.get(0);
        String predicateInput = inputs.get(1);

        SDValue dataValue = variableValues.get(dataInput);
        SDValue predicateValue = variableValues.get(predicateInput);

        if (dataValue == null || predicateValue == null) {
            throw new IllegalStateException("Switch inputs not available: data=" + (dataValue != null) +
                    ", predicate=" + (predicateValue != null));
        }

        INDArray predicate = predicateValue.getTensorValue();
        boolean condition = predicate.getDouble(0) != 0.0;

        // Switch outputs: [false_output, true_output]
        if (outputs.size() >= 2) {
            if (condition) {
                variableValues.put(outputs.get(1), dataValue); // true branch
                variableValues.put(outputs.get(0), null);       // false branch (null)
            } else {
                variableValues.put(outputs.get(0), dataValue);  // false branch
                variableValues.put(outputs.get(1), null);       // true branch (null)
            }
        }
    }

    private void executeEnterNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        Enter enterOp = (Enter) op;
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Enter operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for Enter operation");
        }

        // Enter just forwards the input to the output (entering a new frame)
        variableValues.put(outputVar, inputValue);
    }

    private void executeExitNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Exit operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for Exit operation");
        }

        // Exit forwards the input to the parent frame
        variableValues.put(outputVar, inputValue);
    }

    private void executeNextIterationNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("NextIteration operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for NextIteration operation");
        }

        // NextIteration forwards input to next iteration
        variableValues.put(outputVar, inputValue);
    }

    private void executeMergeNode(ExecutionNode node, Map<String, SDValue> variableValues,
                                   DifferentialFunction op, Set<String> allRequired) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.size() < 2 || outputs.isEmpty()) {
            throw new IllegalStateException("Merge operation requires at least 2 inputs and 1 output");
        }

        String outputVar = outputs.get(0);

        // Find the first available input (standard Merge behavior)
        for (String inputVar : inputs) {
            SDValue inputValue = variableValues.get(inputVar);
            if (inputValue != null) {
                variableValues.put(outputVar, inputValue);

                // Add dependency tracking for the Merge output
                // This is crucial: the Merge shares the SDValue with its input,
                // so we need to add a dependency to prevent the array from being
                // freed when the input's dependency is satisfied
                addConsumerDependencies(inputValue, outputVar, allRequired);
                return;
            }
        }

        throw new IllegalStateException("No inputs available for Merge operation " + node.getOperationName());
    }

    private void executeLoopCondNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("LoopCond operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for LoopCond operation");
        }

        // LoopCond forwards boolean condition
        variableValues.put(outputVar, inputValue);
    }

    // ======================== Control Flow Helpers ========================

    /**
     * Detect while-loop regions in the DAG by finding Merge ops whose inputs
     * include both an Enter output and a NextIteration output.
     */
    private Map<String, WhileLoopRegion> detectWhileLoopRegions(ForwardExecutionDAG dag) {
        Map<String, WhileLoopRegion> regions = new LinkedHashMap<>();
        List<ExecutionNode> executionOrder = dag.getExecutionOrder();
        Map<String, ExecutionNode> opNodes = dag.getOperationNodes();

        // Find all Merge ops that have a NextIteration input (while-loop Merges)
        // Group by frame name from the associated Enter op
        Map<String, List<String>> frameToMerges = new LinkedHashMap<>();
        Map<String, String> mergeToFrame = new HashMap<>();

        for (ExecutionNode node : executionOrder) {
            if (!(node.getOperation() instanceof Merge)) continue;

            // Check if any input comes from a NextIteration op
            boolean hasNextIterInput = false;
            String frameName = null;
            for (String inputVar : node.getInputVariables()) {
                String producerOp = dag.getVariableProducers().get(inputVar);
                if (producerOp == null) continue;
                ExecutionNode producer = opNodes.get(producerOp);
                if (producer == null) continue;
                if (producer.getOperation() instanceof NextIteration) {
                    hasNextIterInput = true;
                }
                if (producer.getOperation() instanceof Enter) {
                    Enter enter = (Enter) producer.getOperation();
                    frameName = enter.getFrameName();
                }
            }

            if (hasNextIterInput && frameName != null) {
                frameToMerges.computeIfAbsent(frameName, k -> new ArrayList<>())
                        .add(node.getOperationName());
                mergeToFrame.put(node.getOperationName(), frameName);
            }
        }

        if (frameToMerges.isEmpty()) return regions;

        // For each frame, build the complete while-loop region
        for (Map.Entry<String, List<String>> entry : frameToMerges.entrySet()) {
            String frameName = entry.getKey();
            List<String> mergeOps = entry.getValue();

            WhileLoopRegion region = new WhileLoopRegion();
            region.frameName = frameName;
            region.mergeOps = new ArrayList<>(mergeOps);
            region.condOps = new ArrayList<>();
            region.loopCondOp = null;
            region.switchOps = new ArrayList<>();
            region.bodyOps = new ArrayList<>();
            region.nextIterOps = new ArrayList<>();
            region.exitOps = new ArrayList<>();

            // Walk the execution order and classify ops by their role in the loop.
            // An op belongs to this loop if it consumes (transitively) from a Merge
            // output and feeds into a NextIteration or Exit.
            // Simpler approach: classify all control flow ops with matching frame,
            // and all ops between Switch:true outputs and NextIteration inputs.

            // First pass: find LoopCond, Switch, NextIteration, Exit ops for this frame
            Set<String> mergeOutputs = new HashSet<>();
            for (String mergeOp : mergeOps) {
                ExecutionNode mergeNode = opNodes.get(mergeOp);
                if (mergeNode != null) {
                    mergeOutputs.addAll(mergeNode.getOutputVariables());
                }
            }

            // Find all ops reachable from merge outputs (condition + body)
            Set<String> reachableOps = new HashSet<>();
            Queue<String> toVisit = new LinkedList<>();
            // Seed with merge output variables
            for (String mergeOutput : mergeOutputs) {
                Set<String> consumers = dag.getVariableConsumers().get(mergeOutput);
                if (consumers != null) {
                    for (String consumer : consumers) {
                        if (!mergeOps.contains(consumer)) {
                            toVisit.add(consumer);
                        }
                    }
                }
            }

            while (!toVisit.isEmpty()) {
                String current = toVisit.poll();
                if (reachableOps.contains(current)) continue;
                reachableOps.add(current);

                ExecutionNode currentNode = opNodes.get(current);
                if (currentNode == null) continue;

                // Don't traverse past Exit ops (they leave the loop)
                if (currentNode.getOperation() instanceof Exit) continue;
                // Don't traverse past NextIteration (they feed back to Merge)
                if (currentNode.getOperation() instanceof NextIteration) continue;

                // Add consumers of this op's outputs
                for (String output : currentNode.getOutputVariables()) {
                    Set<String> consumers = dag.getVariableConsumers().get(output);
                    if (consumers != null) {
                        for (String consumer : consumers) {
                            if (!reachableOps.contains(consumer) && !mergeOps.contains(consumer)) {
                                toVisit.add(consumer);
                            }
                        }
                    }
                }
            }

            // Classify reachable ops
            // Condition ops: between Merge and Switch (including LoopCond)
            // Body ops: between Switch:true and NextIteration
            // We use execution order to maintain proper ordering
            //
            // IMPORTANT: Only classify a Switch as a while-loop Switch if its data input
            // (first input) comes from a Merge output. This distinguishes while-loop
            // switches (switchOp(merged[i], condResult)) from nested if-conditional
            // switches that happen to be inside the loop body.
            Set<String> switchOutputTrueVars = new HashSet<>();
            for (String opName : reachableOps) {
                ExecutionNode n = opNodes.get(opName);
                if (n == null) continue;
                DifferentialFunction op = n.getOperation();
                if (op instanceof LoopCond) {
                    region.loopCondOp = opName;
                } else if (op instanceof Switch) {
                    // Check if this is a while-loop Switch (data input from Merge)
                    // vs a nested if Switch (data input from elsewhere in the body)
                    boolean isWhileSwitch = false;
                    List<String> inputs = n.getInputVariables();
                    if (!inputs.isEmpty()) {
                        String dataInput = inputs.get(0); // First input is data
                        // Check if the data input is produced by one of our Merge ops
                        String producerOp = dag.getVariableProducers().get(dataInput);
                        if (producerOp != null && mergeOps.contains(producerOp)) {
                            isWhileSwitch = true;
                        }
                    }

                    if (isWhileSwitch) {
                        region.switchOps.add(opName);
                        // Track true-branch outputs (index 1)
                        List<String> outputs = n.getOutputVariables();
                        if (outputs.size() >= 2) {
                            switchOutputTrueVars.add(outputs.get(1));
                        }
                    }
                    // Non-while Switches will be classified as bodyOps later
                } else if (op instanceof NextIteration) {
                    region.nextIterOps.add(opName);
                } else if (op instanceof Exit) {
                    region.exitOps.add(opName);
                }
            }

            // Also find Exit ops (reachable from Switch false outputs)
            for (String switchOp : region.switchOps) {
                ExecutionNode switchNode = opNodes.get(switchOp);
                if (switchNode == null) continue;
                List<String> outputs = switchNode.getOutputVariables();
                if (outputs.size() >= 2) {
                    String falseOutput = outputs.get(0);
                    Set<String> consumers = dag.getVariableConsumers().get(falseOutput);
                    if (consumers != null) {
                        for (String consumer : consumers) {
                            ExecutionNode consumerNode = opNodes.get(consumer);
                            if (consumerNode != null && consumerNode.getOperation() instanceof Exit) {
                                region.exitOps.add(consumer);
                                reachableOps.add(consumer);
                            }
                        }
                    }
                }
            }

            // Now classify into condOps vs bodyOps using execution order
            // condOps: reachable from Merge but not from Switch:true
            // bodyOps: reachable from Switch:true outputs
            Set<String> bodyReachable = new HashSet<>();
            Queue<String> bodyQueue = new LinkedList<>();
            for (String trueVar : switchOutputTrueVars) {
                Set<String> consumers = dag.getVariableConsumers().get(trueVar);
                if (consumers != null) bodyQueue.addAll(consumers);
            }
            while (!bodyQueue.isEmpty()) {
                String current = bodyQueue.poll();
                if (bodyReachable.contains(current)) continue;
                ExecutionNode cn = opNodes.get(current);
                if (cn == null) continue;
                if (cn.getOperation() instanceof NextIteration) {
                    bodyReachable.add(current);
                    continue;
                }
                bodyReachable.add(current);
                for (String output : cn.getOutputVariables()) {
                    Set<String> consumers = dag.getVariableConsumers().get(output);
                    if (consumers != null) {
                        for (String c : consumers) {
                            if (!bodyReachable.contains(c)) bodyQueue.add(c);
                        }
                    }
                }
            }

            // Use execution order for proper ordering
            for (ExecutionNode ordered : executionOrder) {
                String name = ordered.getOperationName();
                if (!reachableOps.contains(name)) continue;
                if (region.switchOps.contains(name) || region.nextIterOps.contains(name) ||
                        region.exitOps.contains(name) || name.equals(region.loopCondOp)) {
                    continue;
                }
                if (bodyReachable.contains(name)) {
                    region.bodyOps.add(name);
                } else {
                    region.condOps.add(name);
                }
            }

            regions.put(frameName, region);
            log.debug("Detected while-loop region '{}': {} merges, {} cond ops, {} switches, {} body ops, {} nextIter, {} exits",
                    frameName, region.mergeOps.size(), region.condOps.size(),
                    region.switchOps.size(), region.bodyOps.size(),
                    region.nextIterOps.size(), region.exitOps.size());
        }

        return regions;
    }

    /**
     * Execute a while-loop region iteratively until the condition is false.
     */
    private void executeWhileLoop(WhileLoopRegion region, ForwardExecutionDAG dag,
                                   Map<String, SDValue> variableValues,
                                   Set<String> completedOps,
                                   Set<String> allRequired,
                                   List<Listener> listeners, At at, MultiDataSet batch) {
        Map<String, ExecutionNode> opNodes = dag.getOperationNodes();
        int maxIterations = 10000;

        // Build a map from NextIteration output var → Merge input var
        // so we can find which Merge input comes from NextIteration
        Set<String> nextIterOutputVars = new HashSet<>();
        for (String nextIterOp : region.nextIterOps) {
            ExecutionNode nextIterNode = opNodes.get(nextIterOp);
            if (nextIterNode != null) {
                nextIterOutputVars.addAll(nextIterNode.getOutputVariables());
            }
        }

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // 1. Execute Merge ops
            // On iteration 0: pick the Enter input (first non-null)
            // On iteration 1+: pick the NextIteration input (which was just updated)
            for (String mergeOp : region.mergeOps) {
                ExecutionNode mergeNode = opNodes.get(mergeOp);
                if (mergeNode == null) continue;

                if (iteration == 0) {
                    // First iteration: use standard Merge (picks Enter input)
                    executeNode(mergeNode, variableValues, allRequired, listeners, at, batch);
                } else {
                    // Subsequent iterations: prefer the NextIteration input
                    List<String> inputs = mergeNode.getInputVariables();
                    List<String> outputs = mergeNode.getOutputVariables();
                    String outputVar = outputs.get(0);

                    SDValue nextIterValue = null;
                    for (String input : inputs) {
                        if (nextIterOutputVars.contains(input)) {
                            nextIterValue = variableValues.get(input);
                            break;
                        }
                    }

                    if (nextIterValue != null) {
                        variableValues.put(outputVar, nextIterValue);
                    } else {
                        // Fallback to standard Merge
                        executeNode(mergeNode, variableValues, allRequired, listeners, at, batch);
                    }
                }
                completedOps.add(mergeOp);
            }

            // 2. Execute condition ops (between Merge and LoopCond)
            for (String condOp : region.condOps) {
                ExecutionNode condNode = opNodes.get(condOp);
                if (condNode != null) {
                    executeNode(condNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(condOp);
                }
            }

            // 3. Execute LoopCond
            if (region.loopCondOp != null) {
                ExecutionNode loopCondNode = opNodes.get(region.loopCondOp);
                if (loopCondNode != null) {
                    executeNode(loopCondNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(region.loopCondOp);
                }
            }

            // 4. Execute Switch ops
            for (String switchOp : region.switchOps) {
                ExecutionNode switchNode = opNodes.get(switchOp);
                if (switchNode != null) {
                    executeNode(switchNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(switchOp);
                }
            }

            // 5. Check condition: if any Switch routed to false (exit), stop looping
            boolean conditionTrue = true;
            for (String switchOp : region.switchOps) {
                ExecutionNode switchNode = opNodes.get(switchOp);
                if (switchNode == null) continue;
                List<String> outputs = switchNode.getOutputVariables();
                if (outputs.size() >= 2) {
                    // If true output is null, condition is false
                    SDValue trueOutput = variableValues.get(outputs.get(1));
                    if (trueOutput == null) {
                        conditionTrue = false;
                        break;
                    }
                }
            }

            if (!conditionTrue) {
                // Execute Exit ops to forward false-branch values out of the loop
                for (String exitOp : region.exitOps) {
                    ExecutionNode exitNode = opNodes.get(exitOp);
                    if (exitNode != null) {
                        executeNode(exitNode, variableValues, allRequired, listeners, at, batch);
                        completedOps.add(exitOp);
                    }
                }
                log.debug("While loop '{}' exited after {} iterations", region.frameName, iteration + 1);
                break;
            }

            // 6. Execute body ops (with nested control flow support)
            // Body ops may contain nested if-conditionals (Switch/Merge pairs).
            // After executing a nested Switch, mark inactive branch ops for skipping.
            // Reset skip set each iteration since if-condition may change between iterations.
            Set<String> bodySkipOps = new HashSet<>();
            for (String bodyOp : region.bodyOps) {
                if (bodySkipOps.contains(bodyOp)) {
                    continue; // Skip inactive branch ops from nested if
                }
                ExecutionNode bodyNode = opNodes.get(bodyOp);
                if (bodyNode != null) {
                    // For nested Merge ops (from if-conditionals), relax readiness:
                    // only need at least one non-null input
                    DifferentialFunction bodyNodeOp = bodyNode.getOperation();
                    if (bodyNodeOp instanceof Merge) {
                        boolean hasInput = false;
                        for (String input : bodyNode.getInputVariables()) {
                            if (variableValues.get(input) != null) {
                                hasInput = true;
                                break;
                            }
                        }
                        if (!hasInput) {
                            log.warn("Nested Merge {} in while body has no available inputs, skipping", bodyOp);
                            continue;
                        }
                    }

                    executeNode(bodyNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(bodyOp);

                    // After nested Switch, mark inactive branch for skipping
                    if (bodyNodeOp instanceof Switch) {
                        markInactiveBranchForSkipping(bodyNode, dag, variableValues, bodySkipOps);
                    }
                }
            }

            // 7. Execute NextIteration ops (feed values back to Merge)
            for (String nextIterOp : region.nextIterOps) {
                ExecutionNode nextIterNode = opNodes.get(nextIterOp);
                if (nextIterNode != null) {
                    executeNode(nextIterNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(nextIterOp);
                }
            }
        }
    }

    /**
     * After a Switch op executes, mark all ops on the inactive branch for skipping.
     * BFS from null Switch outputs through consumer ops. Stop at Merge nodes.
     */
    private void markInactiveBranchForSkipping(ExecutionNode switchNode, ForwardExecutionDAG dag,
                                                Map<String, SDValue> variableValues,
                                                Set<String> skipOps) {
        List<String> outputs = switchNode.getOutputVariables();
        if (outputs.size() < 2) return;

        for (String output : outputs) {
            SDValue val = variableValues.get(output);
            if (val != null) continue; // Active branch

            // This output is null (inactive). BFS through consumers.
            Queue<String> queue = new LinkedList<>();
            Set<String> consumers = dag.getVariableConsumers().get(output);
            if (consumers != null) {
                queue.addAll(consumers);
            }

            while (!queue.isEmpty()) {
                String consumerOp = queue.poll();
                if (skipOps.contains(consumerOp)) continue;

                ExecutionNode consumerNode = dag.getOperationNodes().get(consumerOp);
                if (consumerNode == null) continue;

                // Stop at Merge nodes — they handle null inputs by picking the other
                if (consumerNode.getOperation() instanceof Merge) continue;

                skipOps.add(consumerOp);

                // Continue BFS through this op's outputs
                for (String consumerOutput : consumerNode.getOutputVariables()) {
                    Set<String> nextConsumers = dag.getVariableConsumers().get(consumerOutput);
                    if (nextConsumers != null) {
                        queue.addAll(nextConsumers);
                    }
                }
            }
        }
    }

    /**
     * Represents a while-loop region in the graph. Contains all ops organized
     * by their role in the loop execution cycle.
     */
    private static class WhileLoopRegion {
        String frameName;
        List<String> mergeOps;       // Merge ops that start the loop
        List<String> condOps;        // Ops between Merge outputs and LoopCond (ordered)
        String loopCondOp;           // The LoopCond op
        List<String> switchOps;      // Switch ops (one per loop var)
        List<String> bodyOps;        // Body ops in execution order
        List<String> nextIterOps;    // NextIteration ops
        List<String> exitOps;        // Exit ops

        Set<String> allOps() {
            Set<String> all = new HashSet<>();
            all.addAll(mergeOps);
            all.addAll(condOps);
            if (loopCondOp != null) all.add(loopCondOp);
            all.addAll(switchOps);
            all.addAll(bodyOps);
            all.addAll(nextIterOps);
            all.addAll(exitOps);
            return all;
        }
    }

    private void executeTensorArrayNode(ExecutionNode node, Map<String, SDValue> variableValues,
                                         DifferentialFunction op, Set<String> allRequired) {
        // Convert to VarId-based approach for tensor array operations
        // This maintains compatibility with existing tensor array handling

        FrameIter frameIter = new FrameIter(OUTER_FRAME, 0, null);
        Set<VarId> opInputs = new HashSet<>();
        Set<VarId> allIterInputs = new HashSet<>();

        // Convert string-based inputs to VarIds for tensor array compatibility
        for (String inputVar : node.getInputVariables()) {
            VarId vid = new VarId(inputVar, OUTER_FRAME, 0, null);
            // Store the variable value in nodeValueOutputs for tensor array ops
            SDValue value = variableValues.get(inputVar);
            if (value != null) {
                putNodeValue(value, vid);
            }
            opInputs.add(vid);
        }

        try {
            ExecutionResult result = getOutputsHelperTensorArrayOps(op, frameIter, opInputs, allIterInputs, variableValues);

            // Store results back in variableValues and track for deallocation
            if (result.hasValues()) {
                Map<String, SDValue> outputs = result.getValueOutputs();
                for (Map.Entry<String, SDValue> entry : outputs.entrySet()) {
                    String varName = entry.getKey();
                    SDValue outputValue = entry.getValue();
                    variableValues.put(varName, outputValue);
                    addConsumerDependencies(outputValue, varName, allRequired);
                }
            } else if (result.hasSingle()) {
                List<String> outputNames = node.getOutputVariables();
                for (int i = 0; i < outputNames.size() && i < result.numResults(); i++) {
                    INDArray resultArray = result.resultAt(i);
                    if (resultArray != null) {
                        String varName = outputNames.get(i);
                        SDValue outputValue = SDValue.create(resultArray);
                        variableValues.put(varName, outputValue);
                        addConsumerDependencies(outputValue, varName, allRequired);
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to execute tensor array operation " + node.getOperationName(), e);
        }
    }

    private void executeRegularOperation(ExecutionNode node, Map<String, SDValue> variableValues,
                                         Set<String> allRequired, List<Listener> listeners,
                                         At at, MultiDataSet batch) {
        String opName = node.getOperationName();
        SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
        DifferentialFunction op = sameDiffOp.getOp();

        // Reuse OpContext from pool when available, avoiding native alloc/dealloc per op.
        long tOpCtx0 = TIMING_ENABLED ? System.nanoTime() : 0;
        Deque<OpContext> pool = opContextPoolTl.get();
        OpContext opContext = pool.pollFirst();
        if (opContext == null) {
            opContext = Nd4j.getExecutioner().buildContext();
        }
        if (TIMING_ENABLED) timingOpContextAcquireNs += System.nanoTime() - tOpCtx0;

        // Prepare inputs
        long tInput0 = TIMING_ENABLED ? System.nanoTime() : 0;
        String[] argNames = op.argNames();
        if (argNames == null || argNames.length == 0) {
            // Safety check: if no input names are available but the operation requires inputs,
            // throw an exception to prevent native code crashes from null/uninitialized pointers.
            if (op instanceof CustomOp) {
                CustomOpDescriptor desc = ((CustomOp) op).getDescriptor();
                // getNumInputs() returns -1 for variadic, 0 for no inputs required, >0 for fixed number
                if (desc != null && desc.getNumInputs() > 0) {
                    throw new IllegalStateException("Operation " + opName + " (" + op.opName() +
                            ") has no registered input variable names, but descriptor requires " +
                            desc.getNumInputs() + " inputs. This indicates an improperly constructed operation. " +
                            "Ensure all required inputs are connected before execution.");
                }
            }
        }
        if (argNames != null && argNames.length > 0) {
            INDArray[] inputArrays = new INDArray[argNames.length];

            for (int i = 0; i < argNames.length; i++) {
                String argName = argNames[i];
                SDValue argValue = variableValues.get(argName);

                if (argValue == null) {
                    // Start of fix: Manually resolve Merge op outputs if not found
                    SDVariable argVar = sameDiff.getVariable(argName);
                    if (argVar != null && argVar.getCreator() != null) {
                        SameDiffOp producerOp = sameDiff.getOps().get(argVar.getCreator().getOwnName());
                        if (producerOp != null && producerOp.getOp() instanceof org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge) {
                            for (String mergeInputName : producerOp.getInputsToOp()) {
                                if(variableValues.containsKey(mergeInputName)) {
                                    SDValue v = variableValues.get(mergeInputName);
                                    if (v != null) {
                                        argValue = v;
                                        variableValues.put(argName, argValue); // Cache the resolved value
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // End of fix
                }

                if (argValue == null) {
                    // This can happen if a variable is an output of a control flow op (like Switch) and that branch was not taken
                    SDVariable variable = sameDiff.getVariable(argName);
                    if (variable != null && (variable.isConstant() || variable.getVariableType() == VariableType.VARIABLE)) {
                        inputArrays[i] = getConstantOrVariable(argName);
                    } else {
                        inputArrays[i] = null;
                    }
                } else {
                    switch (argValue.getSdValueType()) {
                        case TENSOR:
                            inputArrays[i] = argValue.getTensorValue();
                            break;
                        case LIST:
                            // For list values, try to use the first non-null, non-closed element
                            List<INDArray> list = argValue.getListValue();
                            inputArrays[i] = null;  //Default to null
                            if (list != null && !list.isEmpty()) {
                                for (INDArray arr : list) {
                                    if (arr != null && !arr.wasClosed()) {
                                        inputArrays[i] = arr;
                                        break;
                                    }
                                }
                            }
                            break;
                        default:
                            throw new IllegalStateException("Unsupported SDValue type for op " + opName + " input " + argName + ": " + argValue.getSdValueType());
                    }
                }

                // Centralized check for null or closed arrays, performed immediately after resolving each input.
                // This prevents nulls from ever reaching the OpContext.
                if (inputArrays[i] != null && inputArrays[i].wasClosed()) {
                    throw new ND4JIllegalStateException("Input array was closed! For argument '" + argName + "' at index " + i + " for op '" + op.getOwnName() + "' (" + op.opName() + ") with id " + inputArrays[i].getId());
                }

                // Use-after-free corruption checks for scalar INT64 arrays.
                // Guarded by isDebug() because getLong(0) forces a D→H sync on GPU,
                // causing thousands of GPU-blocking transfers per step.
                if (Nd4j.getEnvironment().isDebug() && inputArrays[i] != null && inputArrays[i].isScalar() && inputArrays[i].dataType() == DataType.INT64) {
                    long value = inputArrays[i].getLong(0);
                    // InteropDataBuffer magic number: 0xAB12CD34EF56 = 188097240559446
                    if (value == 0xAB12CD34EF56L) {
                        throw new ND4JIllegalStateException(
                            "USE-AFTER-FREE DETECTED: Input scalar at index " + i + " for op '" + op.getOwnName() +
                            "' contains InteropDataBuffer magic number (0xAB12CD34EF56). " +
                            "This indicates the array's data buffer was deallocated and the memory was reused. " +
                            "Variable name: '" + argName + "'. Array id: " + inputArrays[i].getId() + ". " +
                            "This is a critical memory management bug - please report this issue.");
                    }
                    if (value > 0x100000000000L && value != Long.MAX_VALUE && value != Long.MIN_VALUE) {
                        throw new ND4JIllegalStateException(
                            "USE-AFTER-FREE DETECTED: Input scalar at index " + i + " for op '" + op.getOwnName() +
                            "' contains pointer-like value: " + value + " (hex: 0x" + Long.toHexString(value) + "). " +
                            "This indicates the array's data buffer was deallocated and memory was reused for another allocation. " +
                            "Variable name: '" + argName + "'. Array id: " + inputArrays[i].getId() + ". " +
                            "Array wasClosed: " + inputArrays[i].wasClosed() + ". " +
                            "Data buffer address: 0x" + Long.toHexString(inputArrays[i].data().pointer().address()) + ". " +
                            "This is a critical memory management bug.");
                    }
                }
            }

            for(int i = 0; i < inputArrays.length; i++) {
                if(inputArrays[i] == null) {
                    // Check if this null is from a Switch operation (inactive branch)
                    // In that case, we should skip this entire operation rather than throw an exception
                    String argName = argNames[i];
                    SDVariable argVar = sameDiff.getVariable(argName);
                    boolean isFromSwitch = false;
                    if (argVar != null && argVar.getCreator() != null) {
                        DifferentialFunction creator = argVar.getCreator();
                        if (creator instanceof org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch) {
                            isFromSwitch = true;
                        }
                    }

                    if (isFromSwitch) {
                        // This operation depends on an inactive Switch branch - skip execution
                        // The Merge node will select the output from the active branch
                        if (log.isTraceEnabled()) {
                            log.trace("Skipping operation {} because input {} is from inactive Switch branch",
                                    op.getOwnName(), argName);
                        }
                        // Return opContext to pool before returning
                        try {
                            opContext.purgeForReuse();
                            pool.offerFirst(opContext);
                        } catch (Exception e) {
                            try { opContext.close(); } catch (Exception ignored) {}
                            if (log.isTraceEnabled()) {
                                log.trace("Error returning OpContext to pool while skipping {}: {}", opName, e.getMessage());
                            }
                        }
                        // Store null for this operation's outputs so Merge can properly select the active branch
                        for (String outputName : node.getOutputVariables()) {
                            variableValues.put(outputName, null);
                        }
                        return; // Skip this operation entirely
                    }

                    // For non-Switch cases, this indicates a real problem
                    throw new ND4JIllegalStateException("Unexpected null input array at index " + i + " for operation '" +
                            op.getOwnName() + "' (op name: " + op.opName() + "). Input variable name: '" + argNames[i] + "'. " +
                            "This indicates a failure in control flow management, where a null from an inactive branch " +
                            "was propagated past a Merge node. \nDependent values for this operation:\n" +
                            getDependentValuesString(variableValues, argNames[i]));
                }
            }

            opContext.setInputArrays(inputArrays);
            // DIAGNOSTIC: check if inputs are closed right after setting them
            if ("gather".equals(op.opName())) {
                for (int di = 0; di < inputArrays.length; di++) {
                    if (inputArrays[di] != null) {
                        log.info("DIAG-SET-INPUT: gather op '{}' input[{}] '{}': closed={} id={} identityHash={}",
                                op.getOwnName(), di, argNames[di],
                                inputArrays[di].wasClosed(), inputArrays[di].getId(),
                                System.identityHashCode(inputArrays[di]));
                    }
                }
            }
        }
        if (TIMING_ENABLED) timingInputResolveNs += System.nanoTime() - tInput0;

        // Handle different operation types
        // NOTE: Do NOT call clearOpaqueNDArraysFromOpContext here!
        // The OpaqueNDArrays are cached in the INDArrays and shared with the OpContext's
        // OpaqueNDArrayArr. Closing them prematurely can cause dangling pointers and
        // heap corruption. The OpaqueNDArrays will be properly cleaned up when:
        // 1. The OpContext is closed (immediately after operation execution below)
        // 2. The INDArrays are garbage collected

        // Attach native workspace to OpContext if available - this allows C++ ops to
        // use bump allocation for internal temporaries instead of per-op malloc/cudaMalloc
        org.bytedeco.javacpp.Pointer wsPtr = mmgr.getNativeWorkspacePointer();
        if (wsPtr != null) {
            opContext.attachWorkspace(wsPtr);
        }

        try {
            if (op instanceof CustomOp) {
                executeCustomOp((CustomOp) op, opContext, node, variableValues, allRequired);
            } else if (op instanceof Op) {
                executeStandardOp((Op) op, opContext, node, variableValues, allRequired);
            } else {
                throw new UnsupportedOperationException("Unsupported operation type: " + op.getClass().getName());
            }

            // View safety: if workspace-backed, replace view outputs with detached copies
            // to prevent dangling pointers when workspace resets on next scopeIn/scopeOut cycle
            if (mmgr.isWorkspaceBacked()) {
                List<String> viewCheckNames = node.getOutputVariables();
                for (String outName : viewCheckNames) {
                    SDValue val = variableValues.get(outName);
                    if (val != null && val.getSdValueType() == SDValueType.TENSOR) {
                        INDArray arr = val.getTensorValue();
                        if (arr != null && arr.isView()) {
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                INDArray detached = arr.dup();
                                variableValues.put(outName, SDValue.create(detached));
                            }
                        }
                    }
                }
            }

            // Call listeners after operation execution
            if (listeners != null && !listeners.isEmpty()) {
                // Collect output arrays for the listener
                List<String> outputNames = node.getOutputVariables();
                INDArray[] outputArrays = new INDArray[outputNames.size()];
                for (int i = 0; i < outputNames.size(); i++) {
                    SDValue val = variableValues.get(outputNames.get(i));
                    outputArrays[i] = val != null ? val.getTensorValue() : null;
                }

                for (Listener l : listeners) {
                    if (l.isActive(at.operation())) {
                        l.opExecution(sameDiff, at, batch, sameDiffOp, opContext, outputArrays);

                        // Also call activationAvailable for each output
                        for (int i = 0; i < outputNames.size(); i++) {
                            if (outputArrays[i] != null) {
                                l.activationAvailable(sameDiff, at, batch, sameDiffOp, outputNames.get(i), outputArrays[i]);
                            }
                        }
                    }
                }
            }
        } finally {
            // Detach workspace before returning OpContext to pool.
            // Do NOT commit() here - the op has already been enqueued on the CUDA stream.
            // Calling commit() per-op forces a GPU sync which serializes all work.
            if (wsPtr != null) {
                opContext.detachWorkspace();
            }

            // Return OpContext to pool after purging, avoiding native alloc/dealloc overhead.
            // purge() clears input/output arrays but keeps the native context alive for reuse.
            try {
                opContext.purgeForReuse();
                opContextPoolTl.get().offerFirst(opContext);
            } catch (Exception e) {
                try { opContext.close(); } catch (Exception ignored) {}
                if (log.isTraceEnabled()) {
                    log.trace("Error returning OpContext to pool for {}: {}", opName, e.getMessage());
                }
            }
        }
    }

    private void executeCustomOp(CustomOp customOp, OpContext opContext, ExecutionNode node,
                                 Map<String, SDValue> variableValues, Set<String> allRequired) {
        DynamicCustomOp dynOp = (DynamicCustomOp) customOp;

        // Set op arguments (NOT input arrays - they are already set by executeRegularOperation)
        // customOp.inputArguments() returns stale arrays from graph import/construction,
        // NOT the arrays from the current execution context. The correct inputs were
        // already set by executeRegularOperation via opContext.setInputArrays(inputArrays).
        // Only set arguments when non-empty to avoid unnecessary array allocations and JNI calls.
        opContext.setIArguments(dynOp.iArgs());
        opContext.setTArguments(dynOp.tArgs());
        opContext.setDArguments(dynOp.dArgs());
        opContext.setBArguments(dynOp.bArgs());
        opContext.setSArguments(dynOp.sArgs());

        // Calculate output shapes with caching: for autoregressive generation, most ops
        // have identical input shapes across decode steps. Cache by (opName + input shapes + args).
        // Ops with data-dependent output shapes (Where, unique) are never cacheable.
        // Ops with INT/LONG inputs (reshape, permute, create, etc.) ARE cacheable but we
        // include the actual input values in the cache key to handle different values correctly.
        boolean isDataDependent = DATA_DEPENDENT_OUTPUT_OPS.contains(customOp.opName());
        boolean cacheable = !isDataDependent;
        boolean hasIntLongInputs = false;
        if (cacheable) {
            List<INDArray> cacheCheckInputs = opContext.getInputArrays();
            for (int ci = 0; ci < cacheCheckInputs.size(); ci++) {
                INDArray cin = cacheCheckInputs.get(ci);
                if (cin != null && (cin.dataType() == DataType.INT || cin.dataType() == DataType.LONG)) {
                    hasIntLongInputs = true;
                    break;
                }
            }
        }

        // For ops with INT/LONG inputs, sync those inputs from device to host BEFORE
        // Check shape cache FIRST before syncing INT/LONG arrays.
        // This avoids expensive device-to-host sync for cache hits (majority of ops).
        long tShape0 = TIMING_ENABLED ? System.nanoTime() : 0;
        List<DataBuffer> outShape = null;
        Map<Long, List<DataBuffer>> shapeCache = null;
        long shapeKey = 0;
        if (cacheable) {
            shapeCache = outputShapeCacheTl.get();
            shapeKey = computeShapeKey(customOp.opName(), opContext);
            outShape = shapeCache.get(shapeKey);
        }

        if (outShape != null) {
            // Cache hit - no need for INT/LONG sync or shape calculation
            if (TIMING_ENABLED) {
                timingShapeCacheHits++;
                timingIntLongSyncSkipCount++;
            }
        } else {
            // Cache miss - need to sync INT/LONG arrays and calculate shape
            // INT/LONG arrays need device-to-host sync before C++ shape functions can read them.
            // Optimizations:
            // 1. Skip sync for constant buffers - their host data is always current.
            // 2. Only sync small arrays (length <= 32) - these carry shape/axis parameters.
            // 3. Data-dependent ops (Where, unique) always need full sync.
            long tSync0 = TIMING_ENABLED ? System.nanoTime() : 0;
            if (hasIntLongInputs || isDataDependent) {
                boolean needsSync = isDataDependent;
                if (!needsSync) {
                    List<INDArray> inputArrays = opContext.getInputArrays();
                    for (int i = 0; i < inputArrays.size(); i++) {
                        INDArray in = inputArrays.get(i);
                        if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                                && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG
                                || in.dataType() == DataType.BOOL)
                                && !in.data().isConstant()
                                && in.length() <= 32) {
                            needsSync = true;
                            break;
                        }
                    }
                }
                if (needsSync) {
                    if (TIMING_ENABLED) timingIntLongSyncCount++;
                    Nd4j.getExecutioner().commit();
                    NativeOps syncNativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    List<INDArray> inputArrays = opContext.getInputArrays();
                    for (int i = 0; i < inputArrays.size(); i++) {
                        INDArray in = inputArrays.get(i);
                        if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                                && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG
                                || in.dataType() == DataType.BOOL)
                                && !in.data().isConstant()) {
                            if (isDataDependent || in.length() <= 32) {
                                syncNativeOps.dbForceSyncToPrimary(in.data().opaqueBuffer());
                            }
                        }
                    }
                } else {
                    if (TIMING_ENABLED) timingIntLongSyncSkipCount++;
                }
            }
            if (TIMING_ENABLED) timingIntLongSyncNs += System.nanoTime() - tSync0;

            if (TIMING_ENABLED) timingShapeCacheMisses++;

            // Calculate shape outside workspace scope so shape buffers are heap-allocated
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                outShape = dynOp.calculateOutputShape(opContext);
            }
            if (outShape == null || outShape.isEmpty()) {
                throw new IllegalStateException("No output shapes calculated for op: " + customOp.opName());
            }
            if (cacheable && shapeCache != null) {
                shapeCache.put(shapeKey, outShape);
            }
        }
        if (TIMING_ENABLED) timingShapeCalcNs += System.nanoTime() - tShape0;

        // Allocate output arrays
        long tAlloc0 = TIMING_ENABLED ? System.nanoTime() : 0;
        List<String> outputNames = node.getOutputVariables();
        INDArray[] outputArrays = new INDArray[outShape.size()];

        // Check if op needs zeroed output buffers (sparse output ops like where, scatter_nd, unique)
        boolean requiresZeroed = customOp.requiresZeroedOutput();

        // Get input arrays for potential view creation
        List<INDArray> inputArrays = opContext.getInputArrays();

        // Special handling for reshape: check if view is possible
        boolean isReshapeOp = "reshape".equals(customOp.opName()) || "reshape_no_copy".equals(customOp.opName());
        boolean reshapeViewPossible = false; // DISABLED: view optimization causes use-after-free on CPU (shared buffer released while view alive)
        char reshapeOrder = 'c';
        if (isReshapeOp && inputArrays != null && !inputArrays.isEmpty() && outShape.size() == 1) {
            if (TIMING_ENABLED) timingReshapeTotal++;
            INDArray reshapeInput = inputArrays.get(0);
            if (reshapeInput != null && !reshapeInput.isEmpty()) {
                long[] iArgs = customOp.iArgs();
                // Determine order: from iArgs if available, else default to 'c'
                if (iArgs != null && iArgs.length > 0) {
                    // First iArg is -order (negative of 'c' or 'f')
                    reshapeOrder = (char) -iArgs[0];
                } else {
                    reshapeOrder = 'c'; // Default to C order for ONNX reshape with shape tensor
                }
                boolean isFOrder = (reshapeOrder == 'f');
                long[] targetShape = Shape.shape(outShape.get(0).asLong());
                reshapeViewPossible = Shape.ableToReshapeWithView(reshapeInput, isFOrder, targetShape);
                if (TIMING_ENABLED) {
                    if (reshapeViewPossible) timingReshapeViewUsed++;
                    else timingReshapeViewSkipped++;
                }
            }
        }

        try {
            for (int i = 0; i < outShape.size(); i++) {
                DataBuffer shapeBuffer = outShape.get(i);
                long[] shapeInfo = shapeBuffer.asLong();

                // Special case: reshape with view possible
                if (isReshapeOp && reshapeViewPossible && i == 0) {
                    INDArray input = inputArrays.get(0);
                    long[] actualShape = Shape.shape(shapeInfo);
                    long[] strides = Nd4j.getStrides(actualShape, reshapeOrder);
                    outputArrays[i] = Nd4j.create(input.data(), actualShape, strides, input.offset(), reshapeOrder, true);
                    continue;
                }

                // Check if C++ shape function indicates output should be a view of an input
                int copyOffsetInputIndex = ArrayOptionsHelper.getCopyOffsetInputIndex(shapeInfo);
                if (copyOffsetInputIndex >= 0 && inputArrays != null && copyOffsetInputIndex < inputArrays.size()) {
                    // View case: create output sharing the specified input's buffer and offset
                    INDArray input = inputArrays.get(copyOffsetInputIndex);
                    long[] actualShape = Shape.shape(shapeInfo);
                    long[] strides = Shape.stride(shapeInfo);
                    char ordering = Shape.order(shapeInfo);

                    // Create output as a view sharing the input's buffer at the input's offset
                    outputArrays[i] = Nd4j.create(input.data(), actualShape, strides, input.offset(), ordering, true);
                } else if (Shape.isEmpty(shapeInfo)) {
                    // Empty array case: use allocateFromDescriptor to preserve ARRAY_EMPTY flag
                    boolean isOutput = i < outputNames.size() && allRequired.contains(outputNames.get(i));
                    outputArrays[i] = mmgr.allocateFromDescriptor(isOutput, shapeBuffer);
                } else {
                    // Standard case: allocate new array (including empty arrays with 0 dimensions)
                    DataType dt = Shape.dataType(shapeInfo);
                    long[] actualShape = Shape.shape(shapeInfo);

                    boolean isOutput = i < outputNames.size() && allRequired.contains(outputNames.get(i));
                    outputArrays[i] = mmgr.allocate(isOutput, dt, actualShape, requiresZeroed);
                }
            }

            opContext.setOutputArrays(outputArrays);
        } finally {
            // Shape buffers from calculateOutputShape() are CACHED by ConstantShapeHelper - do not close.
        }
        if (TIMING_ENABLED) timingMemAllocNs += System.nanoTime() - tAlloc0;

        // Re-sync input arrays to device after calculateOutputShape().
        // Shape functions (e.g. Where) call syncToHost() on inputs to read data values,
        // which marks the host buffer as primary. The GPU kernel then accesses a stale
        // device buffer, producing wrong results (e.g. Where selects wrong branch →
        // broadcast_to gets [1,1] instead of [1,32]). Re-syncing ensures device buffers
        // are current before the kernel launches.
        if (isDataDependent) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            List<INDArray> syncInputs = opContext.getInputArrays();
            for (int si = 0; si < syncInputs.size(); si++) {
                INDArray in = syncInputs.get(si);
                if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()) {
                    nativeOps.dbSyncToSpecial(in.data().opaqueBuffer());
                }
            }
        }

        // Execute the operation
        // DIAGNOSTIC: check if inputs are closed right before exec
        {
            List<INDArray> preExecInputs = opContext.getInputArrays();
            for (int di = 0; di < preExecInputs.size(); di++) {
                INDArray dArr = preExecInputs.get(di);
                if (dArr != null && dArr.wasClosed()) {
                    String opNameDiag = customOp.opName();
                    String ownNameDiag = dynOp.getOwnName();
                    log.warn("DIAG-PRE-EXEC: Op '{}' ({}) input[{}] is CLOSED right before Nd4j.exec. " +
                            "id={} identityHash={} shape={} dtype={}",
                            ownNameDiag, opNameDiag, di, dArr.getId(),
                            System.identityHashCode(dArr),
                            java.util.Arrays.toString(dArr.shape()), dArr.dataType());
                    String[] diagArgNames = dynOp.argNames();
                    if (diagArgNames != null && di < diagArgNames.length) {
                        INDArray modelArr = sameDiff.getArrForVarName(diagArgNames[di]);
                        log.warn("DIAG-PRE-EXEC:   model array for '{}': closed={} id={} identityHash={} sameObject={}",
                                diagArgNames[di],
                                modelArr == null ? "null" : modelArr.wasClosed(),
                                modelArr == null ? -1 : modelArr.getId(),
                                modelArr == null ? -1 : System.identityHashCode(modelArr),
                                modelArr == dArr);
                    }
                }
            }
        }
        long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
        INDArray[] execOutputArrays = Nd4j.exec(dynOp, opContext);
        if (TIMING_ENABLED) {
            timingNativeExecNs += System.nanoTime() - tExec0;
            timingOpCount++;
        }

        // After exec, use the arrays actually written by the executor.
        // On multi-GPU systems, ensureDeviceCoherency may replace output arrays in opContext
        // with fresh device-local copies (e.g. when the op migrates from device 0 to device 1).
        // The local outputArrays[] still holds the original (pre-migration) device-0 arrays,
        // which were never written to.  Use execOutputArrays[] — the return value from exec —
        // which reflects whatever is currently in opContext after the op ran.
        // Fall back to outputArrays[i] only when execOutputArrays is shorter (shouldn't happen).
        final INDArray[] effectiveOutputs = (execOutputArrays != null && execOutputArrays.length >= outputArrays.length)
                ? execOutputArrays : outputArrays;

        // Release any pre-exec output arrays that were replaced by ensureDeviceCoherency.
        // When an op migrates to a different device, ensureDeviceCoherency allocates fresh
        // device-local output buffers and replaces them in opContext.  The original arrays
        // (in outputArrays[]) were allocated by this method and are not referenced anywhere
        // else — they must be freed immediately to prevent GPU OOM.
        // Skip if effectiveOutputs IS outputArrays (no migration happened).
        if (effectiveOutputs != outputArrays) {
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray preExecArr = outputArrays[i];
                if (preExecArr == null || preExecArr.wasClosed()) continue;
                // Only close if it's not a view (views share another array's buffer;
                // closing would corrupt the source) and not already the effective output.
                boolean isEffective = (i < effectiveOutputs.length && effectiveOutputs[i] == preExecArr);
                if (!isEffective && !preExecArr.isView()) {
                    try {
                        preExecArr.close();
                    } catch (Exception e) {
                        // Non-fatal: log and continue
                        log.warn("executeCustomOp: failed to close orphaned pre-exec output[{}]: {}", i, e.getMessage());
                    }
                }
            }
        }

        // Store results and track for deallocation
        long tStore0 = TIMING_ENABLED ? System.nanoTime() : 0;
        for (int i = 0; i < effectiveOutputs.length && i < outputNames.size(); i++) {
            SDValue outputValue = SDValue.create(effectiveOutputs[i]);
            variableValues.put(outputNames.get(i), outputValue);

            // Add to arrayUseTracker so it can be released when no longer needed
            String varName = outputNames.get(i);
            addConsumerDependencies(outputValue, varName, allRequired);
        }
        if (TIMING_ENABLED) timingResultStoreNs += System.nanoTime() - tStore0;
    }

    private void executeStandardOp(Op op, OpContext opContext, ExecutionNode node,
                                   Map<String, SDValue> variableValues, Set<String> allRequired) {

        // For reduce ops with axis as second input, extract dimensions before shape calculation.
        // TF semantics: empty axis = no reduction (return input unchanged).
        if (op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3
                && opContext.getInputArrays().size() >= 2) {
            handleReduceOpAxis(op, opContext);
        }

        // Handle empty reduce: allocate output with input shape, skip shape calculation
        boolean emptyReduce = (op instanceof BaseReduceOp) && ((BaseReduceOp) op).isEmptyReduce();

        List<String> outputNames = node.getOutputVariables();
        INDArray outputArray = null;

        if (emptyReduce) {
            // TF semantics: empty axis = no reduction, output shape = input shape
            long tAlloc0 = TIMING_ENABLED ? System.nanoTime() : 0;
            INDArray input = opContext.getInputArray(0);
            boolean isOutput = !outputNames.isEmpty() && allRequired.contains(outputNames.get(0));
            outputArray = mmgr.allocate(isOutput, input.dataType(), input.shape());
            opContext.setOutputArray(0, outputArray);
            if (TIMING_ENABLED) timingMemAllocNs += System.nanoTime() - tAlloc0;
        } else {
            // Calculate output shape outside workspace scope so shape buffers are heap-allocated
            long tShape0 = TIMING_ENABLED ? System.nanoTime() : 0;
            List<DataBuffer> outputShape;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                outputShape = ((BaseOp) op).calculateOutputShape(opContext);
            }
            if (outputShape == null || outputShape.isEmpty()) {
                throw new IllegalStateException("No output shape calculated for op: " + op.opName());
            }
            if (TIMING_ENABLED) {
                timingShapeCalcNs += System.nanoTime() - tShape0;
                timingShapeCacheMisses++; // standard ops don't use shape cache
            }

            long tAlloc0 = TIMING_ENABLED ? System.nanoTime() : 0;
            try {
                DataBuffer shapeBuffer = outputShape.get(0);
                boolean isOutput = !outputNames.isEmpty() && allRequired.contains(outputNames.get(0));
                outputArray = mmgr.allocateFromDescriptor(isOutput, shapeBuffer);

                opContext.setOutputArray(0, outputArray);
            } finally {
            }
            if (TIMING_ENABLED) timingMemAllocNs += System.nanoTime() - tAlloc0;
        }

        // Execute the operation
        long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
        Nd4j.exec(op, opContext);
        if (TIMING_ENABLED) {
            timingNativeExecNs += System.nanoTime() - tExec0;
            timingOpCount++;
        }

        // Trace removed — use --debug flag for InferenceSession tracing

        // Store result and track for deallocation
        long tStore0 = TIMING_ENABLED ? System.nanoTime() : 0;
        if (!outputNames.isEmpty()) {
            SDValue outputValue = SDValue.create(outputArray);
            String varName = outputNames.get(0);
            variableValues.put(varName, outputValue);
            addConsumerDependencies(outputValue, varName, allRequired);
        }
        if (TIMING_ENABLED) timingResultStoreNs += System.nanoTime() - tStore0;
    }

    private void handleReduceOpAxis(Op op, OpContext opContext) {
        if (opContext.getInputArrays().size() >= 2) {
            INDArray axisArray = opContext.getInputArray(1);
            // FIX: Check both isEmpty() flag AND length() == 0.
            // TF empty axis tensor has shape [0] with length 0, but may not have ARRAY_EMPTY flag set.
            if (axisArray.length() > 0 && !axisArray.isEmpty()) {
                long[] axis = axisArray.toLongVector();
                int rank = opContext.getInputArray(0).rank();
                axis = Shape.normalizeAxis(rank, axis);
                ((DifferentialFunction) op).setDimensions(axis);
                ((BaseReduceOp) op).setEmptyReduce(false);
            } else {
                ((DifferentialFunction) op).setDimensions(null);
                ((BaseReduceOp) op).setEmptyReduce(true);
            }
        }
    }


    @Override
    protected Map<String, INDArray> preprocessPlaceholders(Map<String, INDArray> placeholders, At at) {
        arrayUseTracker().clear();

        //We'll also use this method as a "pre execution" hook-in, to mark variables as something we should never deallocate
        //This occurs by never marking these "ConstantDep" and "VariableDep" instances as satisfied, so there's always
        // an unsatisfied dependency for them in the array use tracker
        //TODO we shouldn't be clearing this on every single iteration, in 99.5% of cases variables will be same as last iteration...
        for (SDVariable v : sameDiff.variables()) {
            if (v.getVariableType() == VariableType.CONSTANT) {
                INDArray arr = v.getArr();
                if (arr != null) {
                    arrayUseTracker().addDependency(SDValue.create(arr), new ConstantDep(v.name()));
                }
            } else if (v.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = v.getArr();
                if (arr != null) {
                    arrayUseTracker().addDependency(SDValue.create(arr), new VariableDep(v.name()));
                }
            }
        }

        //Workaround for some TF/Keras based models that require explicit train/test as a placeholder
        boolean kerasWorkaround = false;
        // Cache inputs list to avoid iterating all 7000+ variables on every call
        List<String> phs = cachedInputsList;
        if (phs == null) {
            phs = sameDiff.inputs();
            cachedInputsList = phs;
        }
        if (phs != null && !phs.isEmpty()) {
            for (String s : phs) {
                if (s.endsWith(KERAS_TRAIN_TEST) && !placeholders.containsKey(s)) {
                    // The behaviour of some Keras layers (like GRU) differs depending on whether the model is training.
                    // We provide this value directly, unless the user has provided this manually
                    INDArray scalar = mmgr.allocate(false, DataType.BOOL).assign(at.operation().isTrainingPhase());
                    placeholders = new HashMap<>(placeholders); //Array might be singleton, or otherwise unmodifiable
                    placeholders.put(s, scalar);
                    kerasWorkaround = true;
                }
            }
        }


        if (placeholders == null || placeholders.isEmpty()) {
            return placeholders;
        }

        //Handle casting of the input array automatically.
        //The idea here is to avoid unexpected errors if the user (for example) tries to perform inference with a double
        // array for a float placeholder
        //TODO eventually we might have ops that support multiple input types, and hence won't need this casting
        Map<String, INDArray> out = new HashMap<>();
        for (Map.Entry<String, INDArray> e : placeholders.entrySet()) {
            Preconditions.checkState(sameDiff.hasVariable(e.getKey()), "Invalid placeholder passed for execution: " +
                    "No variable/placeholder with name %s exists", e.getKey());
            INDArray arr = e.getValue();
            SDValue arrValue = SDValue.create(arr);
            //First: check workspaces
            if (arr.isAttached()) {
                MemoryWorkspace ws = arr.data() == null ? null : arr.data().getParentWorkspace();
                if (ws != null && ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {
                    if (!ws.isScopeActive()) {
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses leaked workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace the array was defined in is no longer open.\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces()
                                + "\n" + SCOPE_PANIC_MSG);
                    }

                    if (ws.getGenerationId() != arr.data().getGenerationId())
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses outdated workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace array was defined in has been closed and reopened at least once since array creation. Array WS iteration: " +
                                arr.data().getGenerationId() + ". Workspace current iteration: " +
                                ws.getGenerationId() + "\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
                }
            }


            //Second: cast the input to the required type
            //TODO For the casting case, we SHOULD actually deallocate this when we're done with it, which is usually sooner than "exec done"
            DataType dt = sameDiff.getVariable(e.getKey()).dataType();
            if (kerasWorkaround && e.getKey().endsWith(KERAS_TRAIN_TEST)) {
                arrayUseTracker().addDependency(arrValue, new ExecDoneDep());
            } else if (arr.dataType() == dt) {
                //Mark as a placeholder array in the array use tracker, so we never deallocate this array...
                arrayUseTracker().addDependency(arrValue,
                        PlaceholderDep.builder()
                        .phName(e.getKey())
                        .frame(currentFrame).parentFrame(currParentFrame)
                        .build());
            } else {
                INDArray cast = mmgr.allocate(false, dt, arr.shape());
                cast.assign(arr);
                arr = cast;
                //This array CAN be deallocated once consumed, because of the cast
                //TODO we can likely close this sooner
                arrayUseTracker().addDependency(arrValue, ExecDoneDep.builder()
                        .frame(currentFrame)
                        .parentFrame(currParentFrame)
                        .build());
            }
            out.put(e.getKey(), arr);
        }

        return out;
    }

    /**
     * Lightweight placeholder type casting for the DynamicShapePlan fast path.
     * Only casts arrays whose dtype doesn't match the placeholder's declared dtype.
     * Skips the expensive arrayUseTracker iteration in preprocessPlaceholders.
     */
    protected Map<String, INDArray> castPlaceholderTypes(Map<String, INDArray> placeholders) {
        if (placeholders == null || placeholders.isEmpty()) {
            return placeholders;
        }
        Map<String, INDArray> out = null;
        for (Map.Entry<String, INDArray> e : placeholders.entrySet()) {
            SDVariable var = sameDiff.getVariable(e.getKey());
            if (var != null && e.getValue() != null && e.getValue().dataType() != var.dataType()) {
                if (out == null) {
                    out = new HashMap<>(placeholders);
                }
                out.put(e.getKey(), e.getValue().castTo(var.dataType()));
            }
        }
        return out != null ? out : placeholders;
    }

    @Override
    protected Map<String, SDValue> postProcessOutputValues(Map<String, SDValue> output) {
        //For any queued (not yet processed) ops - mark them as satisfied, so we can deallocate any arrays
        // that are waiting on them
        if (dt.hasNewAllSatisfied()) {
            List<ExecStep> execSteps = dt.getNewAllSatisfiedList();
            for (ExecStep es : execSteps) {
                if (es.getType() == ExecType.OP) {
                    OpDep od = new OpDep(es.getName(), es.getFrameIter().getFrame(), es.getFrameIter().getIteration(), es.getFrameIter().getParentFrame());
                    arrayUseTracker().markSatisfied(od, true);
                }
            }
        }

        //Also mark "end of execution" for array dependency tracker. Mainly used for TensorArray arrays at present.
        //TODO Optimize for reduced memory for some TensorArray operations - i.e., close/deallocate earlier
        arrayUseTracker().markSatisfied(new ExecDoneDep(), true);
        if (arrayUseTracker().hasNewAllSatisfied()) {
            List<SDValue> l = arrayUseTracker().getNewAllSatisfiedList();
            for (SDValue value : l) {
                switch(value.getSdValueType()) {
                    case LIST:
                        for(INDArray arr : value.getListValue())
                            if(arr != null && !freedArrays().contains(arr.getId())) {
                                mmgr.release(arr);
                                freedArrays().add(arr.getId());
                            }
                        break;
                    case TENSOR:
                        if(!freedArrays().contains(value.getTensorValue().getId())) {
                            mmgr.release(value.getTensorValue());
                            freedArrays().add(value.getTensorValue().getId());
                        }
                        break;
                }
            }
        }

        return output;
    }



    @Override
    public ExecutionResult getOutputs(Pair<SameDiffOp, OpContext> opPair,
                                      FrameIter outputFrameIter,
                                      Set<VarId> opInputs,
                                      Set<VarId> allIterInputs,
                                      Set<String> constAndPhInputs,
                                      List<Listener> listeners,
                                      At at, MultiDataSet batch,
                                      Set<String> allReqVariables,
                                      Map<String, SDValue> otherPlaceHolders) {
        SameDiffOp op = opPair.getFirst();
        at.setFrameIter(outputFrameIter);
        if (listeners != null && listeners.size() > 0) {
            SameDiffOp sdOp = sameDiff.getOps().get(op.getOp().getOwnName());
            for (Listener l : listeners) {
                if (l.isActive(at.operation()))
                    l.preOpExecution(sameDiff, at, sdOp, opPair.getSecond());
            }
        }

        if(Nd4j.getEnvironment().isDebug()) {
            log.info("Executing samediff op: " + op.getName());
        }

        ExecutionResult out = doExec(
                op.getOp(),
                opPair.getRight(),
                outputFrameIter, opInputs,
                allIterInputs,
                constAndPhInputs,
                otherPlaceHolders);
        List<String> opOutNames = op.getOutputsOfOp();

        if (log.isTraceEnabled()) {
            StringBuilder sb = new StringBuilder();
            sb.append(op.getName()).append(" - ").append(outputFrameIter).append(" outputs: ");
            for (int i = 0; i < out.numResults(); i++) {
                if (i > 0)
                    sb.append(", ");
                if(out.hasSingle())
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            out.resultAt(i) == null ? null :  out.resultAt(i) .getId()).append(")");

                else if(out.hasValues()) {
                    SDValue value = out.valueWithKeyAtIndex(i, false);
                    //append either the list of associated array ids or the singular one similar to the singular array case
                    String append = value != null && value.getSdValueType() == SDValueType.LIST ? StringUtil.concatEntries(value.getListValue().stream()
                            .map(input -> input == null ? "" : input.getId()).collect(Collectors.toList()),",",",") : value != null ? String.valueOf(value.getTensorValue().getId()) : null;
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            value == null ? null : append).append(")");

                }
            }
            log.trace(sb.toString());
        }

        //Call listeners, before we (maybe) deallocate input arrays
        if (listeners != null && listeners.size() > 0) {
            Map<String, INDArray> namedOuts = null;

            for (Listener l : listeners) {
                if (l.isActive(at.operation())) {
                    //Lazily create map, only if required
                    if (namedOuts == null) {
                        Map<String, INDArray> namedOutsBuilder = new HashMap<>();

                        for (int i = 0; i < out.numResults(); i++)
                            namedOutsBuilder.put(op.outputsOfOp.get(i), out.resultAt(i));
                        namedOuts = Collections.unmodifiableMap(namedOutsBuilder);
                    }


                    l.opExecution(sameDiff, at, batch, op, opPair.getSecond(), out.outputsToArray(opOutNames));

                    for (String varName : namedOuts.keySet()) {
                        l.activationAvailable(sameDiff, at, batch, op, varName, namedOuts.get(varName));
                    }
                }
            }
        }
        // op.getOp().clearArrays(); // Removed for thread safety when sharing ops across threads
        if(opPair.getSecond() != null)
            opPair.getSecond().purge();


        //Record array uses for memory management/deallocation
        SameDiffOp o = sameDiff.getOps().get(op.getName());
        List<String> outVarNames = o.getOutputsOfOp();
        for (int i = 0; i < out.numResults(); i++) {
            if (out.hasSingle() && out.resultAt(i) == null   || out.hasValues()
                    && out.valueWithKeyAtIndex(i, false) == null
                    && o.getOp() instanceof Switch)
                continue;   //Switch case: we only ever get one of 2 outputs, other is null (branch not executed)
            String name = outVarNames.get(i);
            Variable v = sameDiff.getVariables().get(name);
            List<String> inputsForOps = v.getInputsForOp();
            if (inputsForOps != null) {
                for (String opName : inputsForOps) {
                    //Only add dependencies if we actually need the op this feeds into, otherwise the dependency
                    // will never be marked as satisfied
                    if (!subgraphOps.contains(opName))
                        continue;

                    SameDiffOp forOp = sameDiff.getOps().get(opName);

                    //TODO do switch or merge need special handling also?
                    if (forOp.getOp() instanceof Enter) {
                        Enter e = (Enter) forOp.getOp();
                        if (e.isConstant()) {
                        /*
                        Constant enter case: Need to keep this array around for the entire duration of the frame, including
                        any nested frames, and all iterations.
                        Unfortunately, we don't know exactly when we're done with a frame for good
                        This isn't a great solution, but other possibilities (frame close, trying to detect all exit ops,
                        detecting return to parent frame, etc all fail in certain circumstances, such as due to control dependencies
                        on variables).
                         */
                            Dep d = new ExecDoneDep();
                            addToArrayTracker(out,i,d);
                        } else {
                            Dep d = new OpDep(opName, e.getFrameName(), 0, outputFrameIter);
                            addToArrayTracker(out,i,d);
                        }
                    } else if (forOp.getOp() instanceof NextIteration) {
                        //The array is needed by the NEXT iteration op, not the current one
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration() + 1, outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else if (forOp.getOp() instanceof Exit) {
                        //The array is needed at the EXIT frame (i.e., parent frame), not the inner/just executed one
                        FrameIter fi = outputFrameIter.getParentFrame();
                        Dep d = new OpDep(opName, fi.getFrame(), fi.getIteration(), fi.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else {
                        //All other ops...
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    }
                }
            }

            if (OUTER_FRAME.equals(outputFrameIter.getFrame()) && allReqVariables.contains(name)) {
                //This variable is an output, record that in the array use tracker, so we don't deallocate it
                //the specific value here
                addToArrayTracker(out,i,new ReqOutputDep(name));
            } else {
                SDValue array = null;
                if (out.getValueOutputs() != null) {
                    array = out.valueWithKeyAtIndex(i, false);
                } else if (out.getOutputs() != null) {
                    INDArray arr = out.resultAt(i);
                    if (arr != null) {
                        array = SDValue.create(arr);
                    }
                }
                if (array != null) {
                    arrayUseTracker().addDependency(array, new ExecDoneDep());
                }
            }
        }

        //Mark current op dependency as satisfied...
        Dep d = new OpDep(op.getName(), outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
        arrayUseTracker().markSatisfied(d, true);


        //Close any no longer required arrays
        if (arrayUseTracker().hasNewAllSatisfied()) {
            List<SDValue> canClose = arrayUseTracker().getNewAllSatisfiedList();
            for (SDValue value : canClose) {
                if (log.isTraceEnabled()) {
                    if(value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray arr = value.getTensorValue();
                        log.trace("Closing array... id={}, {}", arr.getId(), arr.shapeInfoToString());

                    }
                }

                //don't free anything that's an output
                boolean containsOutput = false;
                for(String output : outVarNames) {
                    if(op.getOutputsOfOp().contains(output)) {
                        containsOutput = true;
                    }
                }

                if(!(op.getOp() instanceof Switch))
                    switch(value.getSdValueType()) {
                        case TENSOR:
                            if(!freedArrays().contains(value.getTensorValue().getId()) &&
                                    !containsOutput) {
                                mmgr.release(value.getTensorValue());
                                freedArrays().add(value.getTensorValue().getId());
                            }
                            break;
                        case LIST:
                            for(INDArray arr : value.getListValue())
                                if(arr != null && !freedArrays().contains(arr.getId()) && !containsOutput) {
                                    mmgr.release(arr);
                                    freedArrays().add(arr.getId());
                                }
                            break;
                    }

            }
        }

        return out;
    }


    private void addToArrayTracker(ExecutionResult out,int i,Dep d) {
        if(out.hasSingle()) {
            arrayUseTracker().addDependency(SDValue.create(out.resultOrValueAt(i,false)), d);       //Op defined by "d" needs to be executed before specified array can be closed
        } else {
            arrayUseTracker().addDependency(out.valueWithKeyAtIndex(i,false),d);
        }
    }

    public ExecutionResult doExec(DifferentialFunction op,
                                  OpContext opContext,
                                  FrameIter outputFrameIter,
                                  Set<VarId> opInputs, Set<VarId> allIterInputs,
                                  Set<String> constAndPhInputs,
                                  Map<String, SDValue> otherPlaceHolders) {

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size())
                + (allIterInputs == null ? 0 : allIterInputs.size());

        boolean constPhInput = (opInputs == null || opInputs.size() == 0) && (allIterInputs == null || allIterInputs.size() == 0);

        // Initialize visualization if enabled - mimic output() method
        if (visualizationEnabled && visualizer != null) {
            // Prepare visualization data
            List<String> stepInputs = getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs);
            List<String> stepOutputs = getStepOutputsForVisualization(op);
            String executionStatus = "INITIALIZING";
            String detailedStatus = String.format("Operation: %s, Type: %s, Inputs: %d",
                    op.getOwnName(), op.getClass().getSimpleName(), totalInputs);

            visualizer.recordStep(
                    ExecType.OP,
                    op.getOwnName(),
                    outputFrameIter,
                    stepInputs,
                    stepOutputs,
                    executionStatus + " | " + detailedStatus
            );
        }

        // Execution status tracking for visualization
        String executionStatus = "SUCCESS";
        String detailedStatus = "";
        List<String> outputNames = new ArrayList<>();

        try {
            if (op instanceof Identity) {
                Identity i = (Identity) op;
                String[] argNames = i.argNames();
                Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", (Object) argNames);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue orig = getSdValue(vid);

                executionStatus = "SUCCESS";
                detailedStatus = "Identity passthrough";
                outputNames.add(vid.getVariable());

                ExecutionResult result = ExecutionResult.createValue(vid.getVariable(), orig);

                // Record successful execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames[0]),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Switch) {
                Switch s = (Switch) op;
                String[] argNames = s.argNames();       //Order: input, boolean array
                VarId vidPredicate = outputFrameIter.toVarId(argNames[1]);
                SDValue sdValuePred = getSdValue(vidPredicate);
                INDArray predicate = sdValuePred.getSdValueType() == SDValueType.LIST ? sdValuePred.getListValue().get(0) :
                        sdValuePred.getTensorValue();
                if(predicate != null && predicate.isEmpty()) {
                    predicate = Nd4j.scalar(false);
                }
                if(predicate == null && !constAndPhInputs.isEmpty() && constAndPhInputs.contains(argNames[1])) {
                    //Constant predicate...
                    predicate = getTensorFromOutputs(new VarId(argNames[1], OUTER_FRAME, 0, null));
                }
                Preconditions.checkNotNull(predicate, "Error during graph execution: Predicate array was null. VarId=%s", vidPredicate);
                Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue sdValue = getSdValue(vid);
                Map<String,SDValue> values = new LinkedHashMap<>();
                ExecutionResult.ExecutionResultBuilder executionResultBuilder = ExecutionResult.builder()
                        .valueOutputs(values);

                boolean predicateValue = predicate.getDouble(0) != 0.0;
                String branchTaken = predicateValue ? "RIGHT" : "LEFT";
                executionStatus = "SWITCH_" + branchTaken;
                detailedStatus = String.format("SWITCH decision: %s branch taken (frame: %s, iter: %d) | Exec count: 1, Switches: 1",
                        branchTaken, outputFrameIter.getFrame(), outputFrameIter.getIteration());

                if (predicate.getDouble(0) == 0.0) {
                    //tensorflow import case
                    if(vid.getVariable().equals(vidPredicate.getVariable())) {
                        SDValue sdValue1 = SDValue.create(Arrays.asList(sdValue.getTensorValue(), null));
                        values.put(vidPredicate.getVariable(),sdValue1);
                        putNodeValue(sdValue1,vid);
                        VarId varId1 = new VarId(vid.getVariable() + ":1", vid.getFrame(), vid.getIteration(),vid.getParentFrame());
                        putNodeValue(sdValue1,varId1);
                        outputNames.add(vid.getVariable());
                    } else {
                        values.put(vid.getVariable(),sdValue);
                        values.put(vidPredicate.getVariable(),null);
                        outputNames.add(vid.getVariable());
                    }
                } else {
                    //tensorflow import case
                    if(vid.getVariable().equals(vidPredicate.getVariable())) {
                        SDValue sdValue1 = SDValue.create(Arrays.asList(null,sdValue.getTensorValue()));
                        values.put(vidPredicate.getVariable(),sdValue1);
                        values.put(vidPredicate.getVariable() + ":1",sdValue1);
                        outputNames.add(vidPredicate.getVariable());
                    } else {
                        values.put(vid.getVariable(),null);
                        values.put(vidPredicate.getVariable(),sdValue);
                        outputNames.add(vidPredicate.getVariable());
                    }
                }

                ExecutionResult result = executionResultBuilder.build();

                // Record switch execution with enhanced details
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Enter) {
                Enter e = (Enter) op;
                String[] input = e.argNames();
                Preconditions.checkState(input.length == 1, "Expected only 1 arg name for enter op: got %s", (Object) input);
                Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for Enter op \"%s\", got %s+%s", e.getOwnName(), opInputs, constAndPhInputs);

                VarId inputVarId;
                if (constPhInput) {
                    inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
                } else if (allIterInputs != null && allIterInputs.size() > 0) {
                    inputVarId = allIterInputs.iterator().next();
                } else {
                    inputVarId = opInputs.iterator().next();
                }

                inputVarId.setVariable(VariableUtils.stripVarSuffix(inputVarId.getVariable()));

                // Create explicit dependency aliases for cross-frame access
                SameDiffOp enterOp = sameDiff.getOps().get(e.getOwnName());
                List<String> enterOutputs = enterOp.getOutputsOfOp();
                String outFrame = e.getFrameName();
                FrameIter enterOutFrameIter = new FrameIter(outFrame, 0, outputFrameIter);

                if (enterOutputs != null) {
                    for (String outputVar : enterOutputs) {
                        String inputVar = enterOp.getInputsToOp().get(0);

                        ExecStep expectedStep = new ExecStep(ExecType.OP, outputVar, enterOutFrameIter);
                        ExecStep actualStep = new ExecStep(ExecType.OP, e.getOwnName(), enterOutFrameIter);
                        dt.createDependeeAlias(expectedStep, actualStep);

                        log.debug("Created Enter dependency alias: {} -> {} (frame: {} -> {})",
                                outputVar, inputVar, outputFrameIter.getFrame(), outFrame);
                    }

                    // Handle frame transition to ensure cross-frame dependencies are properly established
                    handleFrameTransition(e.getOwnName(), outputFrameIter, enterOutFrameIter, enterOutputs);

                    // Validate that dependent Merge operations will be able to find the Enter outputs
                    validateMergeDependencies(enterOutFrameIter, enterOutputs);
                }

                ExecutionResult result;
                if(nodeValueOutputs.containsKey(inputVarId)) {
                    SDValue value = getSdValue(inputVarId);
                    if(value != null && value.getSdValueType() == SDValueType.LIST) {
                        result = ExecutionResult.createValue(inputVarId.getVariable(), value);
                    } else if(value != null &&  value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray inArr = getTensorFromOutputs(inputVarId);
                        if (inArr == null) {
                            Preconditions.throwStateEx("Could not find array for Enter operation %s with output %s (frame=%s, iteration=%s)",
                                    op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), enterOutFrameIter.getFrame(), enterOutFrameIter.getIteration());
                        }
                        result = ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
                    } else {
                        throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + inputVarId);
                    }
                } else {
                    INDArray inArr = getTensorFromOutputs(inputVarId);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for Enter operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), enterOutFrameIter.getFrame(), enterOutFrameIter.getIteration());
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
                }

                return result;
            }else if (op instanceof Exit) {
                //Exit node forwards input to parent frame
                VarId inputVarId;
                if (constPhInput) {
                    //Constant or placeholder
                    inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
                } else if (allIterInputs != null && allIterInputs.size() > 0) {
                    inputVarId = allIterInputs.iterator().next();
                } else {
                    inputVarId = opInputs.iterator().next();
                }

                executionStatus = "EXIT_FRAME";
                detailedStatus = String.format("EXIT from frame '%s' (iter: %d) | Variables exiting: %d",
                        outputFrameIter.getFrame(), outputFrameIter.getIteration(), 1);
                outputNames.add(inputVarId.getVariable());

                SDValue sdValue = getSdValue(inputVarId);
                ExecutionResult result = ExecutionResult.createValue(inputVarId.getVariable(), sdValue);

                // Record exit execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(inputVarId.getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof NextIteration) {
                //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
                Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for NextIteration: got %s+%s", opInputs, constAndPhInputs);
                VarId in = (allIterInputs != null && !allIterInputs.isEmpty() ? allIterInputs.iterator().next() : opInputs.iterator().next());
                Preconditions.checkState(outputFrameIter.getFrame().equals(in.getFrame()), "Expected same frame for NextIteration input vs. output:" +
                        " got input %s, output %s", in, outputFrameIter);
                Preconditions.checkState(outputFrameIter.getIteration() == in.getIteration() + 1, "Expected output iteration for NextIteration output to" +
                        " be 1 larger than the input iteration. Input: %s, output %s", in, outputFrameIter);

                executionStatus = "NEXT_ITERATION";
                detailedStatus = String.format("NEXT_ITERATION in frame '%s' (iter: %d -> %d)",
                        outputFrameIter.getFrame(), in.getIteration(), outputFrameIter.getIteration());
                outputNames.add(in.getVariable());

                ExecutionResult result;
                if(nodeValueOutputs.containsKey(in) && getSdValue(in) != null) {
                    SDValue value = getSdValue(in);
                    if(value != null && value.getSdValueType() == SDValueType.LIST) {
                        result = ExecutionResult.createValue(in.getVariable(),value);
                    } else if(value != null && value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray inArr = getTensorFromOutputs(in);
                        if (inArr == null) {
                            Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                    op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                        }
                        result = ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
                    } else {
                        throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + in);
                    }
                } else {
                    INDArray inArr = getTensorFromOutputs(in);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
                }

                // Record next iteration execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(in.getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Merge) {
                Merge m = (Merge) op;
                String[] in = sameDiff.getInputsForOp(op);

                // Multi-frame input resolution for Merge operations
                List<VarId> candidateInputs = new ArrayList<>();
                List<SDValue> availableValues = new ArrayList<>();

                for (String inputName : in) {
                    SDValue foundValue = null;
                    VarId foundVarId = null;

                    // Strategy 1: Current frame lookup
                    VarId currentFrameVid = outputFrameIter.toVarId(inputName);
                    foundValue = getSdValue(currentFrameVid);
                    if (foundValue != null) {
                        candidateInputs.add(currentFrameVid);
                        availableValues.add(foundValue);
                        continue;
                    }

                    // Strategy 2: Cross-frame lookup for Enter operations
                    for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
                        VarId storedVid = entry.getKey();

                        if (storedVid.getFrame().equals(outputFrameIter.getFrame())) {
                            String producerOp = findVariableProducer(storedVid.getVariable());
                            if (producerOp != null) {
                                SameDiffOp producer = sameDiff.getOps().get(producerOp);
                                if (producer != null && producer.getOp() instanceof Enter) {
                                    List<String> enterOutputs = producer.getOutputsOfOp();
                                    if (enterOutputs != null && enterOutputs.contains(inputName)) {
                                        foundValue = entry.getValue();
                                        foundVarId = storedVid;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Strategy 3: Look for alias mappings in dependency tracker
                    if (foundValue == null) {
                        VarId aliasVid = resolveVarIdAlias(currentFrameVid);
                        if (aliasVid != null && !aliasVid.equals(currentFrameVid)) {
                            foundValue = getSdValue(aliasVid);
                            foundVarId = aliasVid;
                        }
                    }

                    if (foundValue != null) {
                        candidateInputs.add(foundVarId != null ? foundVarId : currentFrameVid);
                        availableValues.add(foundValue);
                    }
                }

                if (log.isDebugEnabled()) {
                    log.debug("Merge operation {} found {} available inputs out of {} expected",
                            m.getOwnName(), availableValues.size(), in.length);
                    for (int i = 0; i < candidateInputs.size(); i++) {
                        log.debug("  Input {}: {} -> {}", i, candidateInputs.get(i),
                                availableValues.get(i) != null ? "AVAILABLE" : "NULL");
                    }
                }

                if (availableValues.isEmpty()) {
                    throw new IllegalStateException(String.format(
                            "Merge node %s has no available inputs (expected: %s, frame: %s) - cross-frame dependency resolution failure",
                            m.getOwnName(), Arrays.toString(in), outputFrameIter));
                }

                // Use first available input (standard Merge behavior)
                SDValue selectedValue = availableValues.get(0);
                VarId selectedVarId = candidateInputs.get(0);

                log.trace("Merge {} selected input: {} from frame {}",
                        m.getOwnName(), selectedVarId.getVariable(), selectedVarId.getFrame());

                ExecutionResult result;
                if(selectedValue.getSdValueType() == SDValueType.LIST) {
                    result = ExecutionResult.createValue(selectedVarId.getVariable(), selectedValue);
                } else if(selectedValue.getSdValueType() == SDValueType.TENSOR) {
                    INDArray inArr = getTensorFromOutputs(selectedVarId);
                    if (inArr == null) {
                        inArr = selectedValue.getTensorValue();
                    }
                    if (inArr == null) {
                        throw new IllegalStateException(String.format(
                                "Could not resolve tensor for Merge operation %s input %s (frame=%s, iteration=%s)",
                                op.getOwnName(), selectedVarId.getVariable(), outputFrameIter.getFrame(), outputFrameIter.getIteration()));
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(selectedVarId.getVariable()), new INDArray[]{inArr});
                } else {
                    throw new IllegalStateException("Illegal value type " + selectedValue.getSdValueType() + " for Merge input " + selectedVarId);
                }

                return result;
            }else if (op instanceof LoopCond) {
                //LoopCond just forwards scalar boolean to output
                LoopCond lc = (LoopCond) op;
                String[] argNames = lc.argNames();
                Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", (Object) argNames);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue getValue = getSdValue(vid);
                if(getValue.getTensorValue() == null) {
                    throw new IllegalStateException("Node value output at " + vid.getVariable() + " was not a boolean tensor!");
                }
                Preconditions.checkNotNull(getValue, "Input to LoopCond op must not be null");
                Preconditions.checkState(getValue.getTensorValue().isScalar() && getValue.getTensorValue().dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");

                boolean conditionValue = getValue.getTensorValue().getDouble(0) != 0.0;
                executionStatus = "SUCCESS";
                detailedStatus = String.format("LoopCond forwarded: %s", conditionValue);
                outputNames.add(vid.getVariable());

                ExecutionResult result = ExecutionResult.createValue(vid.getVariable(), getValue);

                // Record loop condition execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames[0]),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof BaseTensorOp) {
                //TensorOps - special cases...
                executionStatus = "SUCCESS";
                detailedStatus = "TensorArray operation";

                ExecutionResult result = getOutputsHelperTensorArrayOps(op, outputFrameIter, opInputs, allIterInputs, otherPlaceHolders);

                // Record tensor op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Identity) {
                List<VarId> orderedInputs = new ArrayList<>(opInputs);
                SDValue sdValue = getSdValue(orderedInputs.get(0));
                executionStatus = "SUCCESS";
                detailedStatus = "Identity operation";
                outputNames.add(op.outputVariablesNames()[0]);

                ExecutionResult result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);

                // Record identity execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(orderedInputs.get(0).getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Assign) {
                List<VarId> orderedInputs = new ArrayList<>(opInputs);
                executionStatus = "SUCCESS";
                detailedStatus = "Assign operation";
                outputNames.add(op.outputVariablesNames()[0]);

                ExecutionResult result;
                if(orderedInputs.size() > 1) {
                    SDValue sdValue = getSdValue(orderedInputs.get(0));
                    SDValue sdValue1 = getSdValue(orderedInputs.get(1));
                    switch(sdValue.getSdValueType()) {
                        case TENSOR:
                            Assign c = (Assign) op;
                            try {
                                Nd4j.exec(c, opContext);
                                result = ExecutionResult.createFrom(c,opContext);
                            } finally {
                                // LEAK FIX: Clean up Assign operation's internal arrays
                                // ((DifferentialFunction) c).clearArrays(); // Removed for thread safety
                            }
                            break;
                        case LIST:
                            result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue1);
                            break;
                        default:
                            throw new IllegalStateException("Unknown SDValue type: " + sdValue.getSdValueType());
                    }
                } else {
                    SDValue sdValue = getSdValue(orderedInputs.get(0));
                    result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);
                }

                // Record assign execution
                if (visualizationEnabled && visualizer != null) {
                    List<String> inputs = new ArrayList<>();
                    for (VarId vid : orderedInputs) {
                        inputs.add(vid.getVariable());
                    }
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            inputs,
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof GradientBackwardsMarker) {
                INDArray out = mmgr.allocate(false, DataType.FLOAT).assign(1.0f);
                executionStatus = "SUCCESS";
                detailedStatus = "Gradient backwards marker";
                outputNames.add("gradientbackwardsmarker");

                ExecutionResult result = ExecutionResult.createFrom(Arrays.asList("gradientbackwardsmarker"), new INDArray[]{out});

                // Track for deallocation
                if (!freedArrays().contains(out.getId())) {
                    mmgr.release(out);
                    freedArrays().add(out.getId());
                }

                // Record gradient marker execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Collections.emptyList(),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof CreateView) {
                Map<String,VarId> inputVars = new LinkedHashMap<>();
                String[] argNames = op.argNames();
                for(Iterator<VarId> iter = opInputs.iterator(); iter.hasNext();) {
                    VarId varId  = iter.next();
                    inputVars.put(varId.getVariable(),varId);
                }

                executionStatus = "SUCCESS";
                detailedStatus = String.format("CreateView with %d indices", argNames.length - 1);
                outputNames.add(op.outputVariablesNames()[0]);

                SDValue sdValue = getSdValue(inputVars.get(argNames[0]));
                if(sdValue == null) {
                    sdValue = SDValue.create(opContext.getInputArray(0));
                }
                INDArray[] indices = new INDArray[argNames.length - 1];
                for(int i = 1; i < argNames.length; i++) {
                    indices[i - 1] = getSdValue(inputVars.get(argNames[i])).getTensorValue();
                }

                INDArray from = CreateView.createFrom(sdValue.getTensorValue(), indices);
                from.setCloseable(false);
                sdValue.getTensorValue().setCloseable(false);
                for(INDArray arr : indices)
                    arr.setCloseable(false);
                ExecutionResult result = ExecutionResult.createFrom(op.outputVariablesNames()[0], from);

                // Record create view execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof ExternalErrorsFunction) {
                ExternalErrorsFunction fn = (ExternalErrorsFunction) op;
                String n = fn.getGradPlaceholderName();
                INDArray arr = getTensorFromOutputs(new VarId(n, OUTER_FRAME, 0, null));
                Preconditions.checkState(arr != null, "Could not find external errors placeholder array: %s", arr);

                executionStatus = "SUCCESS";
                detailedStatus = "External errors function";
                outputNames.add(n);

                INDArray out = mmgr.allocate(false, arr.dataType(), arr.shape());
                out.assign(arr);
                ExecutionResult result = ExecutionResult.createFrom(Arrays.asList(n), new INDArray[]{out});

                // Track for deallocation - this is typically an output array
                arrayUseTracker().addDependency(SDValue.create(out), new ReqOutputDep(n));

                // Record external errors execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(n),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Invoke) {
                Invoke invoke = (Invoke) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Invoke with %d inputs", invoke.getInputVarNames().length);
                outputNames.addAll(Arrays.asList(invoke.getOutputVarNames()));

                boolean hasValues = false;
                for(VarId varId : opInputs) {
                    //need to invoke with values
                    if(nodeValueOutputs.containsKey(varId)) {
                        hasValues = true;
                        break;
                    }
                }

                //no need to check placeholders if other values are present
                if(!hasValues)
                    for(Map.Entry<String,SDValue> entry : otherPlaceHolders.entrySet()) {
                        if(constAndPhInputs.contains(entry.getKey())) {
                            hasValues = true;
                            break;
                        }
                    }

                Map<String,INDArray> inputs = new LinkedHashMap<>();
                Map<String,SDValue> valueInputs = new LinkedHashMap<>();
                //need to pull from tensor arrays
                if(!hasValues) {
                    //simple linear scan of inputs over inputs
                    int currInput = 0;
                    for(VarId opInput : opInputs) {
                        inputs.put(opInput.getVariable(),opContext.getInputArray(currInput));
                        currInput++;
                    }
                } else {
                    //simple linear scan of inputs over inputs
                    Map<String,VarId> varIdsByVariable = new HashMap<>();
                    for(VarId opInput : opInputs) {
                        varIdsByVariable.put(opInput.getVariable(),opInput);
                    }

                    for(int i = 0; i < invoke.getInputVarNames().length; i++) {
                        VarId opInput = varIdsByVariable.get(invoke.getInputVarNames()[i]);
                        if(constAndPhInputs.contains(invoke.getInputVarNames()[i])) {
                            if(otherPlaceHolders.containsKey(invoke.getInputVarNames()[i]))
                                valueInputs.put(invoke.getInputVarNames()[i],otherPlaceHolders.get(invoke.getInputVarNames()[i]));
                            else if(inputs.containsKey(invoke.getInputVarNames()[i]))
                                valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(inputs.get(invoke.getInputVarNames()[i])));
                        }else if(sameDiff.getArrForVarName(invoke.getInputVarNames()[i]) != null) {
                            valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(sameDiff.getArrForVarName(invoke.getInputVarNames()[i])));
                        }  else if(nodeValueOutputs.containsKey(opInput)) {
                            valueInputs.put(opInput.getVariable(), getSdValue(opInput));
                        } else {
                            valueInputs.put(opInput.getVariable(),SDValue.create(opContext.getInputArray(i)));
                        }
                    }
                }

                if(valueInputs.size() + inputs.size() != op.args().length) {
                    throw new IllegalArgumentException("Value inputs and inputs combined did not fulfill all arguments. Inputs were: " + Arrays.toString(op.argNames()) + " for op name " + op.getOwnName());
                }

                ExecutionResult result = Invoke.doInvoke(invoke,inputs,valueInputs);

                // Record invoke execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(invoke.getInputVarNames()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Assert) {
                Assert a = (Assert) op;
                boolean condition = !opContext.getInputArray(0).isEmpty() && opContext.getInputArray(0).getDouble(0) != 0.0;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Assert condition: %s", condition);

                if(!condition) {
                    //Assertion failed
                    String s = "Assertion failed for operation \"" + op.getOwnName() + "\" during execution";
                    if(a.numInputArguments() >= 3) {
                        INDArray msg = opContext.getInputArray(2);
                        if (msg != null && msg.dataType() == DataType.UTF8) {
                            s += ": " + msg.getString(0);
                        }
                    }
                    if(a.numInputArguments() >= 5) {
                        INDArray arr = opContext.getInputArray(4);
                        s += "\n" + arr;
                    }

                    // Record failed assertion
                    if (visualizationEnabled && visualizer != null) {
                        visualizer.recordStep(
                                ExecType.OP,
                                op.getOwnName(),
                                outputFrameIter,
                                getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                                Collections.emptyList(),
                                "EXECUTION_FAILED | Assertion failed: " + s
                        );
                    }

                    throw new IllegalStateException(s);
                }

                ExecutionResult result = ExecutionResult.createFrom(a,opContext);

                // Record successful assertion
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof CustomOp) {
                CustomOp c = (CustomOp) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("CustomOp: %s", c.opName());

                ExecutionResult result;
                try {
                    Nd4j.exec(c, opContext);
                    result = ExecutionResult.createFrom((DifferentialFunction) c,opContext);
                } finally {
                    // LEAK FIX: Clean up operation's internal arrays to prevent memory leaks
                    // This must happen AFTER createFrom() captures the results
                    // ((DifferentialFunction) c).clearArrays(); // Removed for thread safety
                }

                // Record custom op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Op) {
                Op o = (Op) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Op: %s", o.opName());

                ExecutionResult result;
                try {
                    Nd4j.exec(o, opContext);
                    result = ExecutionResult.createFrom((DifferentialFunction)o,opContext);
                } finally {
                    // LEAK FIX: Clean up operation's internal arrays (dimensionz, scalarValue, etc.)
                    // This must happen AFTER createFrom() captures the results
                    // ((DifferentialFunction) o).clearArrays(); // Removed for thread safety
                }

                // Record op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else {
                throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
            }

        } catch (Exception e) {
            // Enhanced error visualization - mimic output() method error handling
            executionStatus = "EXECUTION_FAILED";
            detailedStatus = String.format("Exception: %s - %s", e.getClass().getSimpleName(), e.getMessage());

            // Record failed execution with comprehensive context
            if (visualizationEnabled && visualizer != null) {
                String failureContext = generateOperationFailureContext(op, opInputs, constAndPhInputs, allIterInputs, e);

                visualizer.recordStep(
                        ExecType.OP,
                        op.getOwnName(),
                        outputFrameIter,
                        getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                        getStepOutputsForVisualization(op),
                        executionStatus + " | " + detailedStatus + " | " + failureContext
                );

                // Enhanced failure analysis for control flow operations
                if (op instanceof Switch || op instanceof Merge || op instanceof Enter ||
                        op instanceof Exit || op instanceof NextIteration || op instanceof LoopCond) {

                    visualizer.analyzeControlFlowFailure(op, opInputs, allIterInputs, constAndPhInputs,
                            outputFrameIter, nodeValueOutputs, e);
                }
            }

            throw e; // Re-throw the exception
        }
    }



    /**
     * Initialize variable values from constants, variables, and placeholders
     */
    private void initializeVariableValues(Map<String, SDValue> variableValues,
                                          ForwardExecutionDAG dag,
                                          Map<String, INDArray> placeholderValues,
                                          Map<String, SDValue> otherPlaceholderValues) {

        // Initialize constants
        for (String constName : dag.getConstants()) {
            INDArray constValue = getConstantOrVariable(constName);
            if (constValue != null) {
                variableValues.put(constName, SDValue.create(constValue));
            }
        }

        // Initialize variables
        for (String varName : dag.getVariables()) {
            INDArray varValue = getConstantOrVariable(varName);
            if (varValue != null) {
                variableValues.put(varName, SDValue.create(varValue));
            }
        }

        // Initialize placeholders
        if (placeholderValues != null) {
            for (Map.Entry<String, INDArray> entry : placeholderValues.entrySet()) {
                variableValues.put(entry.getKey(), SDValue.create(entry.getValue()));
            }
        }

        if (otherPlaceholderValues != null) {
            variableValues.putAll(otherPlaceholderValues);
        }
    }

    // Helper methods for visualization - mimic the output() method approach

    private List<String> getStepInputsForVisualization(Set<VarId> opInputs, Set<String> constAndPhInputs, Set<VarId> allIterInputs) {
        List<String> stepInputs = new ArrayList<>();

        if (opInputs != null) {
            for (VarId vid : opInputs) {
                stepInputs.add(vid.getVariable());
            }
        }

        if (constAndPhInputs != null) {
            stepInputs.addAll(constAndPhInputs);
        }

        if (allIterInputs != null) {
            for (VarId vid : allIterInputs) {
                stepInputs.add(vid.getVariable());
            }
        }

        return stepInputs;
    }

    private List<String> getStepOutputsForVisualization(DifferentialFunction op) {
        if (op.outputVariablesNames() != null) {
            return Arrays.asList(op.outputVariablesNames());
        }

        // For ops that might not have standard output variable names
        SameDiffOp sdOp = sameDiff.getOps().get(op.getOwnName());
        if (sdOp != null && sdOp.getOutputsOfOp() != null) {
            return sdOp.getOutputsOfOp();
        }

        return Collections.emptyList();
    }

    private String generateOperationFailureContext(DifferentialFunction op, Set<VarId> opInputs,
                                                   Set<String> constAndPhInputs, Set<VarId> allIterInputs,
                                                   Exception e) {
        StringBuilder context = new StringBuilder();

        context.append("Op: ").append(op.getClass().getSimpleName());
        context.append(", Inputs: ").append(opInputs != null ? opInputs.size() : 0);
        context.append(", ConstPh: ").append(constAndPhInputs != null ? constAndPhInputs.size() : 0);
        context.append(", AllIter: ").append(allIterInputs != null ? allIterInputs.size() : 0);

        // Add input availability analysis
        if (opInputs != null) {
            int availableInputs = 0;
            for (VarId vid : opInputs) {
                if (getSdValue(vid) != null) {
                    availableInputs++;
                }
            }
            context.append(", Available: ").append(availableInputs).append("/").append(opInputs.size());
        }

        return context.toString();
    }

    private SDValue getPreviousValue(VarId varId) {
        return getPreviousValue(varId,1);
    }

    private SDValue getPreviousValue(VarId varId,int offset) {
        VarId ret = new VarId(varId.getVariable(), varId.getFrame(), varId.getIteration() - offset,varId.getParentFrame());
        return nodeValueOutputs.get(ret);
    }

    private SDValue getValueAtIteration(String var,String frame, int iteration,FrameIter parentFrame) {
        VarId varId = new VarId(var,frame,iteration,parentFrame);
        return nodeValueOutputs.get(varId);
    }

    /**
     * Forward pass for TensorArray ops
     */
    public ExecutionResult getOutputsHelperTensorArrayOps(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs, Map<String, SDValue> otherPlaceHolders) {
        /*
        TODO: TensorArray memory management note: For now, we'll close any INDArrays stored in the TensorArray at the end of
        graph execution. This uses more memory than necessary for an earlier close strategy, but simplifies memory management.
        This should be revisited and optimized later
         */

        if (op instanceof TensorArray) {
            //Create a TensorArray
            VarId vid = outputFrameIter.toVarId(op.outputVariable().name());
            if(nodeValueOutputs.containsKey(vid)) {
                // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
                return ExecutionResult.createValue(vid.getVariable(),nodeValueOutputs.get(vid));
            }
            Preconditions.checkState(!nodeValueOutputs.containsKey(vid), "TensorArray already exists for %s when executing TensorArrayV3", vid);
            List<INDArray> createList = new ArrayList<>();

            if(op.args().length > 0) {
                SDVariable size = op.arg(0);
                INDArray arr = size.getArr();
                TensorArray tensorArray = (TensorArray) op;
                long[] requiredShape = tensorArray.args().length > 1 ? tensorArray.requiredShape() : null;
                for(int i = 0; i  < arr.getInt(0); i++) {
                    createList.add(null);
                }

            }


            SDValue listValue = SDValue.create(createList);
            putNodeValue(listValue, vid);

            // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
            return ExecutionResult.createValue(vid.getVariable(),listValue);
        } else if (op instanceof TensorArrayRead) {
            //Do lookup and return
            //Input 0 is the TensorArray (or dummy variable that represents it). Sometimes (for import) this can be like (TensorArray -> Enter -> TensorArrayRead)
            //Input 1 is the index
            SDVariable idxSDV = op.arg(1);
            INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
            Preconditions.checkState(idxArr.isScalar(), "TensorArrayRead input argument 1 should be scalar - has shape %ndShape", idxArr);
            int i = idxArr.getInt(0);

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array

            //Work out the frame/iteration:
            VarId v = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (v == null && allIterInputs != null) {
                v = lookup(inTensorArray.name(), allIterInputs, false);
            }


            Preconditions.checkState(v != null, "Could not find input %s", inTensorArray.name());

            TensorArray tensorArray1 = TensorArray.getTensorArray(sameDiff, inTensorArray);

            List<INDArray> list = null;
            if(!nodeValueOutputs.containsKey(v)) {
                TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
                SDVariable output = tensorArray.getVar();
                list = getTensorArraysInSession(output.name());

            } else {
                list = getSdValue(v).getListValue();
            }

            //we specify a shape every element should be and validate it
            if(tensorArray1.args().length > 1) {
                long[] inputShapeArr = tensorArray1.requiredShape();
                for(int j = 0; j < list.size(); j++) {
                    if(list.get(j) != null)
                        if(!Arrays.equals(inputShapeArr,list.get(j).shape()) && inputShapeArr.length > 0) {
                            throw new IllegalArgumentException("Element " + j  + " of list " + v.getVariable() + " did not have correct shape of " + Arrays.toString(inputShapeArr) + " was shape " + Arrays.toString(list.get(j).shape()));
                        }

                }
            }
            Preconditions.checkState(list != null, "Could not find TensorList for %s", v);
            Preconditions.checkState(list.size() > i, "Cannot get index %s from TensorList of size %s (array not present?) - VarId=%s", i, list.size(), v);

            INDArray out = list.get(i);

            log.trace("Reading item at index " + i + " for list " + v + " with value " + out + " with list of " + list);
            return ExecutionResult.createFrom(v.getVariable(),out);
        } else if (op instanceof TensorArrayWrite) {
            //TensorArrayWrite - also has a scalar 0.0 that it returns...
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            //Work out the varid (frame/iteration) of the tensor array:
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }



            //create new tensor array for placeholder referencing a passed in variable
            if(tArr == null && inTensorArray.getVariableType() == VariableType.PLACEHOLDER) {
                VarId varId = new VarId(inTensorArray.name(),outputFrameIter.getFrame(),outputFrameIter.getIteration(),outputFrameIter.getParentFrame());
                tArr = varId;
                SDValue sdValue = otherPlaceHolders.get(inTensorArray.name());
                //putNodeValue(sdValue, tArr);
            }

            Preconditions.checkState(tArr != null, "Could not find input %s", inTensorArray.name());



            //Input 0 is the TensorArray (or dummy variable that represents it) - but sometimes Enter, in TensorArray -> Enter -> TensorARrayRead
            //Input 1 is the index
            //Input 2 is the value to write

            String idxName = op.arg(1).name();
            SDVariable idxSDV = sameDiff.getVariable(idxName);
            INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
            Preconditions.checkState(idxArr.isScalar(), "Index variable ID for TensorArrayWrite should be a scalar, got %ndShape", idxArr);
            int idx = idxArr.getInt(0);

            String inName = op.arg(2).name();
            SDVariable inSDV = sameDiff.getVariable(inName);
            INDArray arr = getArray(inSDV, opInputs, allIterInputs);
            Preconditions.checkState(arr != null, "Could not find array for %s", inName);
            TensorArray tArrOp = TensorArray.getTensorArray(sameDiff,inTensorArray);
            tArr = new VarId(tArrOp.outputVariable().name(),OUTER_FRAME,0,null);
            if(tArrOp.args().length > 1) {
                long[] shape = tArrOp.arg(1).getArr().toLongVector();
                if(!Arrays.equals(arr.shape(),shape) && shape.length > 0) {
                    throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(arr.shape()) + " must be " + Arrays.toString(shape) + " for op " + op.getOwnName() + " and tensor array " + tArrOp.getOwnName());
                }
            }


            Preconditions.checkState(nodeValueOutputs.containsKey(tArr), "Tensor array does not exist for %s", tArr);
            //TODO is this always safe to insert by index for all execution orders?
            SDValue sdValue1 = getSdValue(tArr);
            List<INDArray> l = sdValue1.getListValue(); //.set(idx, arr);
            if(idx < 0 && l != null && !l.isEmpty()) {
                idx += l.size() + 1;
            } else if(idx < 0) {
                idx = 0;
            }
            while (l.size() <= idx) {
                //Can't use set(int, E) if index >= size
                l.add(null);
            }

            setArrayAtIndex(l, idx, arr);
            log.trace("Setting item at index " + idx + " for list " + tArr + " with value " + arr + " with whole list of after write " + l + " and value array " + arr);
            log.trace("Writing value " + inSDV + " to list " + tArr.getVariable() + " at iteration " + tArr.getIteration());

            //Add a dependency
            Dep d = new ExecDoneDep();
            arrayUseTracker().addDependency(sdValue1, d);
            return ExecutionResult.createValue(op.outputVariable().name(),sdValue1);
        } else if (op instanceof TensorArraySize) {
            //Index 0 is the TensorArray (or dummy variable that represents it)
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            List<INDArray> l = getSdValue(tArr).getListValue();
            int size = l == null ? 0 : l.size();
            INDArray scalar = mmgr.allocate(false, DataType.INT).assign(size);
            return ExecutionResult.createFrom(tensorArray.getVar().name(),scalar);
        } else if (op instanceof TensorArrayConcat) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }
            List<INDArray> l = getSdValue(tArr).getListValue();

            Concat c = new Concat(0, l.stream().map(INDArray::detach).filter(input -> input != null && !input.wasClosed()).collect(Collectors.toList())
                    .toArray(new INDArray[0]));
            try (OpContext opContext = Nd4j.getExecutioner().buildContext()) {
                opContext.setInputArrays(c.inputArguments());
                opContext.setOutputArrays(c.outputArguments());
                opContext.setIArguments(c.iArgs());
                opContext.setTArguments(c.tArgs());
                opContext.setBArguments(c.bArgs());
                opContext.setDArguments(c.dArgs());
                opContext.setSArguments(c.sArgs());
                List<DataBuffer> shape;
                try (MemoryWorkspace ws2 = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    shape = c.calculateOutputShape(opContext);
                }
                try {
                    // shape.get(0) is OWNED by the output array after allocation - don't close it
                    INDArray out = mmgr.allocateFromDescriptor(false, shape.get(0));
                    c.setOutputArgument(0, out);
                    Nd4j.getExecutioner().exec(c, opContext);
                    c.setInputArguments(new INDArray[0]);
                    return ExecutionResult.createFrom(tArr.getVariable(),out);
                } finally {
                    // NOTE: c.clearArrays() was removed.
                    // The original "LEAK FIX" was causing a use-after-free bug by deallocating the input arrays
                    // which are part of the persistent TensorArray state. It also deallocated the output array
                    // 'out' before it was returned.
                    // The temporary OpContext is closed by the try-with-resources block, which cleans up native resources.
                    // The input arrays are persistent state. The output array is returned to the caller.

                    // NOTE: Shape buffers returned by calculateOutputShape() are CACHED by ConstantShapeHelper
                    // and must NOT be closed here - they are shared across operations.
                    // Closing them would corrupt the shape cache, leading to use-after-free bugs.
                }            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (op instanceof TensorArrayGather) {
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            List<INDArray> l = getSdValue(tArr).getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayGather should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayGather should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);

            int[] idxArrInt = idxArr.toIntVector();
            log.trace("Gathering op " + op.getOwnName() + " from indices " + Arrays.toString(idxArrInt) + " named " + indicesName + " from list " + tArr.getVariable());
            if(idxArrInt.length > 0) {
                //Edge case: -1 means "all"
                List<INDArray> newList = new ArrayList<>();
                if (idxArrInt.length == 1 || idxArrInt.length > 0 &&  idxArrInt[0]  < 0) {
                    newList.addAll(l);
                } else {
                    for (int id : idxArrInt) {
                        Preconditions.checkState(id >= 0, "Index for TensorArrayGather must be >= 0, got %s", id);
                        if(l.get(id) != null) {
                            log.trace("Gathering op " + op.getOwnName() + " at index " + id + " adding value " + l.get(id).toStringFull() + " from full list " + l);
                            newList.add(l.get(id));

                        }
                    }
                }

                Stack s = new Stack(newList.stream().map(INDArray::detach).filter(input -> input != null && !input.wasClosed()).collect(Collectors.toList())
                        .toArray(new INDArray[0]), null, 0);
                try (OpContext opContext = Nd4j.getExecutioner().buildContext()) {
                    opContext.setInputArrays(s.inputArguments());
                    opContext.setOutputArrays(s.outputArguments());
                    opContext.setIArguments(s.iArgs());
                    opContext.setTArguments(s.tArgs());
                    opContext.setBArguments(s.bArgs());
                    opContext.setDArguments(s.dArgs());
                    opContext.setSArguments(s.sArgs());
                    List<DataBuffer> shape;
                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        shape = s.calculateOutputShape(opContext);
                    }
                    try {
                        // shape.get(0) is OWNED by the output array after allocation - don't close it
                        INDArray out = mmgr.allocateFromDescriptor(false, shape.get(0));
                        s.setOutputArgument(0, out);
                        Nd4j.getExecutioner().exec(s, opContext);
                        return ExecutionResult.createFrom(tArr.getVariable(),out);
                    } finally {
                        // NOTE: s.clearArrays() was removed.
                        // The original "LEAK FIX" was causing a use-after-free bug by deallocating the input arrays
                        // which are part of the persistent TensorArray state. It also deallocated the output array
                        // 'out' before it was returned.
                        // The temporary OpContext is closed by the try-with-resources block, which cleans up native resources.
                        // The input arrays are persistent state. The output array is returned to the caller.

                        // NOTE: Shape buffers returned by calculateOutputShape() are CACHED by ConstantShapeHelper
                        // and must NOT be closed here - they are shared across operations.
                        // Closing them would corrupt the shape cache, leading to use-after-free bugs.
                    }                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else {
                return ExecutionResult.createFrom(tArr.getVariable(),Nd4j.zeros(op.arg().dataType(),0));
            }

        } else if (op instanceof TensorArrayScatter) {
            //Scatter values from a rank (N+1)d tensor into specific indices of the TensorArray
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)
            //Input 2: The values to scatter

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray ta = TensorArray.getTensorArray(sameDiff,inTensorArray);
            VarId tArr = (opInputs == null ? null : lookup(ta.outputVariablesNames()[0], opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(ta.outputVariablesNames()[0], allIterInputs, false);
            }

            SDValue retValue = getSdValue(tArr);
            List<INDArray> l = retValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayScatter should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayScatter should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);
            int[] idxs = idxArr.toIntVector();

            String valuesName = op.arg(2).name();
            SDVariable valuesSDV = sameDiff.getVariable(valuesName);
            INDArray valuesArr = getArray(valuesSDV, opInputs, allIterInputs);

            while (l.size() < idxs.length) { //Can't use set(int, E) if index >= size
                l.add(null);
            }


            //Edge case: idxs being [-1] means "all sub arrays" (i.e., "unstack" case)
            if (idxs.length == 1 && idxs[0] == -1) {
                idxs = ArrayUtil.range(0, (int) valuesArr.size(0));
            }

            for(int i = 0; i < idxs.length; i++) {
                if(valuesArr.size(0) < idxs[i]) {
                    throw new IllegalArgumentException("Unable to obtain slice from values array named " + valuesName +  " with shape " + Arrays.toString(valuesArr.shape()) + " at index " + idxs[i] + " at node named " + op.getOwnName()  + " with inputs " + Arrays.toString(op.argNames()));
                }
            }

            for (int i = 0; i < idxs.length; i++) {
                if(idxs[i] >= valuesArr.size(0)) {
                    throw new IllegalStateException("Unable to pull slice from value array " + valuesSDV.name() + " of shape " + Arrays.toString(valuesArr.shape()) + " index was" + idxs[i]  + " all indices were " + Arrays.toString(idxs));
                }
                INDArray getView = valuesArr.slice(idxs[i]);
                INDArray get = mmgr.dup(getView);
                if(ta.args().length > 1) {
                    long[] shape = ta.arg(1).getArr().toLongVector();
                    if(!Arrays.equals(get.shape(),shape) && shape.length > 0) {
                        throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(get.shape()) + " must be " + shape + " for op " + op.getOwnName() + " and tensor array " + ta.getOwnName());
                    }
                }
                SDValue newValue = SDValue.create(get);
                int outIdx = idxs[i];
                if (valuesArr.rank() == 1 && get.rank() > 0) {
                    get = get.reshape();
                }

                //reflect the expanded storage
                if(outIdx >= l.size()) {
                    while(l.size() <= outIdx) {
                        l.add(null);
                    }
                }

                log.trace("Scattering item at index " + i + " for list " + tArr + " with value " + get + " from whole list of " + l + " from values array " + valuesArr.toStringFull() + " named " + valuesSDV.name());
                setArrayAtIndex(l, outIdx, get);

                //Add dependency for values array until end of execution
                arrayUseTracker().addDependency(newValue, new ExecDoneDep());
            }


            return ExecutionResult.createValue(valuesName,retValue);
        } else if (op instanceof TensorArraySplit) {
            //Split values from a rank (N+1)d tensor into sequential indices of the TensorArray
            //For example, orig=[8,2] sizearray with split (4,4) means TensorArray[0] = orig[0:4,:] and TensorArray[1] = orig[4:8,:]
            //Input 0: the TensorArray
            //Input 1: The values to split
            //Input 2: the size of each split (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            SDValue sdValue = getSdValue(tArr);
            List<INDArray> l = sdValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String splitName = op.arg(1).name();
            INDArray splitArr = getArray(sameDiff.getVariable(splitName), opInputs, allIterInputs);


            String sizeName = op.arg(2).name();
            SDVariable sizeSDV = sameDiff.getVariable(sizeName);
            INDArray sizeArr = getArray(sizeSDV, opInputs, allIterInputs);
            Preconditions.checkState(sizeArr.isVector(), "Indices variable for TensorArraySplit should be a vector, got %ndShape for %s", sizeArr, sizeName);
            Preconditions.checkState(sizeArr.dataType().isIntType(), "Indices variable for TensorArraySplit should be an integer type, got %s for array %s", sizeArr.dataType(), sizeName);
            int[] sizes = sizeArr.toIntVector();

            while (l.size() <= sizes.length) { //Can't use set(int, E) if index >= size
                l.add(null);
            }

            INDArrayIndex[] idx = ArrayUtil.nTimes(splitArr.rank(), NDArrayIndex.all(), INDArrayIndex.class);
            int soFar = 0;
            for (int i = 0; i < sizes.length; i++) {
                idx[0] = NDArrayIndex.interval(soFar, soFar + sizes[i]);
                INDArray sub = mmgr.dup(splitArr.get(idx));
                SDValue subValue = SDValue.create(sub);
                setArrayAtIndex(l, i, sub);
                soFar += sizes[i];

                //Add dependency for values array until end of execution
                arrayUseTracker().addDependency(subValue, new ExecDoneDep());
            }

            return ExecutionResult.createValue(sizeName,sdValue);
        } else if (op instanceof TensorArrayRemove) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            SDVariable index = op.arg(1);
            List<INDArray> l = getTensorArraysInSession(inTensorArray.name());
            if(l == null)
                l = new ArrayList<>();
            else if(l != null)
                l.remove(index.getArr(true).getInt(0));
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            //setup an extra reference to the removed list
            putNodeValue(SDValue.create(l), tArr);
            return ExecutionResult.createValue(tArr.getVariable(),l);
        }

        else {
            throw new IllegalStateException("Execution support not yet implemented for: " + op.getClass().getName());
        }
    }


    private Map<Pair<String,Integer>,SDValue> valuesFor(String varName) {
        Map<Pair<String,Integer>,SDValue> ret = new HashMap<>();
        for(Map.Entry<VarId,SDValue> values : nodeValueOutputs.entrySet()) {
            if(values.getKey().getVariable().equals(varName)) {
                ret.put(Pair.of(values.getKey().getVariable(),values.getKey().getIteration()),values.getValue());
            }
        }

        return ret;
    }


    @Override
    public INDArray getConstantOrVariable(String variableName) {
        SDVariable v = sameDiff.getVariable(variableName);
        Preconditions.checkState(sameDiff.getVariable(variableName).isConstant() || v.getVariableType() == VariableType.VARIABLE,
                "Variable %s is not a constant", variableName);
        INDArray arr = sameDiff.getArrForVarName(variableName);
        // Mark constant arrays' DataBuffers as constant so the multi-GPU replica cache
        // routes them to constantReplicaCache (cached across ops within a forward pass)
        // instead of pendingReplicaClose (closed after each op). Without this, constant
        // replicas like sin_cache/cos_cache get closed prematurely between ops.
        if (arr != null && arr.data() != null && v.isConstant() && !arr.data().isConstant()) {
            arr.data().setConstant(true);
        }
        return arr;
    }

    @Override
    public Pair<SameDiffOp,OpContext> getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                                           Set<String> constAndPhInputs, Map<String, INDArray> placeholderValues, Set<String> allReqVariables, Map<String, SDValue> otherPlaceholders) {
        SameDiffOp sdo = sameDiff.getOps().get(opName);
        DifferentialFunction df = sdo.getOp();

        //TODO Switch to OpContext - and make sure executing like that is thread safe (i.e., array fields in ops are not used etc)

        Preconditions.checkNotNull(df, "No differential function found with name \"%s\"", opName);

        if (df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof BaseTensorOp || df instanceof Invoke) {
            //Control dependencies and tensor ops (like TensorArray, TensorArrayRead etc) don't need inputs set, execution is a special case
            return new Pair<>(sdo, null);
        }

        //Infer the args based on the inputs (variable + frame + iteration)
        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numNonConstInsAllIters = (allIterInputs == null ? 0 : allIterInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        if (numArgs != (numNonConstIns + numConstPhIns + numNonConstInsAllIters)) {
            if (numArgs > 1) {
                //Might be due to repeated inputs
                Set<String> uniqueArgNames = new LinkedHashSet<>();
                Collections.addAll(uniqueArgNames, argNames);

            } else {
                Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, argNames, opInputs, constAndPhInputs);
            }
        }

        INDArray[] args = null;
        if (argNames != null && argNames.length > 0) {
            args = new INDArray[argNames.length];
            int i = 0;
            for (String s : argNames) {
                SDVariable v = sameDiff.getVariable(s);
                if (v.isConstant()) {
                    args[i] = v.getArr();
                } else if (v.getVariableType() == VariableType.VARIABLE) {
                    args[i] = v.getArr();
                } else if (v.isPlaceHolder()) {
                    if(placeholderValues != null && placeholderValues.containsKey(s))
                        args[i] = placeholderValues.get(s);
                    else if(otherPlaceholders != null && otherPlaceholders.containsKey(s)) {
                        args[i] = otherPlaceholders.get(s).getTensorValue();
                    }
                    else
                        throw new IllegalArgumentException("No array was provided for required placeholder variable \"%s\"".format(s));
                } else {
                    VarId vid = lookup(s, opInputs, allIterInputs, true);
                    SDValue getValue = getSdValue(vid);
                    if(getValue != null)
                        switch(getValue.getSdValueType()) {
                            case TENSOR:
                                args[i] = getValue.getTensorValue();
                                break;
                            case LIST:
                                DifferentialFunction variableOutputOp = sameDiff.getVariableOutputOp(s);
                                //tensorflow import case: when switch is imported and 2 are input names are equal
                                //we output a list with 1 value that's null and 1 that's not
                                if(variableOutputOp instanceof Switch && variableOutputOp.argNames().length == 2 && variableOutputOp.argNames()[0].equals(variableOutputOp.argNames()[1])) {
                                    //find the non null value
                                    for(int j = 0; j < getValue.getListValue().size(); j++) {
                                        if(getValue.getListValue().get(j) !=  null) {
                                            args[i] = getValue.getListValue().get(j);
                                            break;
                                        }
                                    }
                                }
                                else
                                    args[i] = Nd4j.empty(DataType.FLOAT);
                                break;

                        }
                }


                Preconditions.checkNotNull(args[i], "Could not parameterize op %s: array %s (variable %s) is null", opName, i, v.name());
                i++;
            }
        }

        if(df.needsConfigure()) {
            SDVariable[] vars = df.args();
            for(int i = 0; i < vars.length; i++) {
                vars[i].setShape(args[i].shape());
            }

            df.configureWithSameDiff(sameDiff);
        }


        //Set the op inputs and output arguments
        //Note that when we are in a loop (and non-first iteration), we want to allocate new arrays even if shapes are
        // ok: this is because we need the values in past iterations for backprop (potentially)
        //TODO let's find a way to use in-place modification for loops where possible to reduce memory requirements
        boolean isLoop = !frameIter.getFrame().equals(OUTER_FRAME) && frameIter.getIteration() > 0;

        OpContext oc = opContexts().get(opName);
        if(oc == null) {
            oc = Nd4j.getExecutioner().buildContext();
            opContexts().put(opName, oc);
        }

        if (df instanceof CustomOp) {
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            if (df instanceof Identity || df instanceof CreateView) {
                if (args != null) {
                    oc.setInputArrays(args);
                }

                //set a dummy result to be replaced
                oc.setOutputArrays(args[0]);
                //We don't need to allocate an output array for Identity, we pass through the input array without copying
                return new Pair<>(sdo, oc);
            }

            oc.setArgs(args, customOp.iArgs(), customOp.dArgs(), customOp.tArgs(), customOp.bArgs(), customOp.sArgs());

            //input and output should be same for assign
            if((df instanceof Assign)) {
                oc.setOutputArray(0, oc.getInputArray(0));

            } else {
                List<DataBuffer> outShape;
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    outShape = customOp.calculateOutputShape(oc);
                }
                Preconditions.checkState(outShape != null && outShape.size() > 0, "Failed to calculate output shapes for op %s (%s) - no shapes were returned by calculateOutputShape()", customOp.opName(), customOp.getOwnName());
                String[] outNames = df.outputVariablesNames();
                Preconditions.checkState(outNames.length == outShape.size(), "Error in operation shape calculation for op \"%s\": Got %s op output shapes for an operation" +
                        " with %s outputs (number of shapes and outputs must be equal)", df.opName(), outShape.size(), outNames.length);
                try {
                    for (int i = 0; i < outShape.size(); i++) {
                        DataBuffer reqShape = outShape.get(i);
                        long[] asJava = reqShape.asLong();;
                        //Issue: many ops have multiple valid output datatypes, and output shape calc can't at present know which: https://github.com/eclipse/deeplearning4j/issues/6872
                        //As a workaround, we'll use the output variable datatype instead.
                        DataType dt = sameDiff.getVariable(outNames[i]).dataType();
                        DataType currDT = reqShape.dataType();
                        if (dt != currDT) {
                            Shape.setExtras(asJava,Shape.extras(asJava));
                        }

                        //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                        boolean isOutput = allReqVariables.contains(outNames[i]);
                        // Note: the shape buffer created here is OWNED by the output array allocated below
                        // Do NOT close it - the INDArray owns and manages the shape buffer's lifecycle
                        DataBuffer newShapeBuffer = Nd4j.createBuffer(asJava);
                        INDArray out = mmgr.allocateFromDescriptor(false, newShapeBuffer);
                        if(Shape.isEmpty(asJava) && !out.isEmpty()) {
                            throw new IllegalStateException("Output shape was empty, but created array was not.");
                        }

                        oc.setOutputArray(i, out);
                    }
                } finally {
                    // NOTE: Shape buffers returned by calculateOutputShape() are CACHED by ConstantShapeHelper
                    // and must NOT be closed here - they are shared across operations.
                    // Closing them would corrupt the shape cache, leading to use-after-free bugs that
                    // manifest as JVM heap corruption (crashes in SymbolTable::do_lookup, G1 GC, etc.)
                }
            }


        } else if (df instanceof Op) {
            Op op = (Op) df;

            boolean axisArg = false;
            boolean emptyReduce = false;
            if (op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3 && df.argNames().length == 2) {
                //2nd input should be treated as integer axis arg...
                SDVariable axisArgVar = df.arg(1);
                Preconditions.checkState(axisArgVar.dataType().isIntType(), "Legacy op %s input 1 (axis) was expected to be an integer type, is %s", df.getClass(), axisArgVar.dataType());

                INDArray arr = getArray(axisArgVar, opInputs, allIterInputs);
                Preconditions.checkState(arr != null, "Could not get axis argument for op %s: %s", df.getOwnName(), df.getClass());
                if (!arr.isEmpty()) {
                    long[] axis = arr.toLongVector();
                    int rank = args[0].rank();
                    axis = Shape.normalizeAxis(rank, axis);
                    df.setDimensions(axis);
                    ((BaseReduceOp) op).setEmptyReduce(false);
                } else {
                    df.setDimensions(null);
                    emptyReduce = true;
                    //Note: edge case: [x,y].sum(empty) = [x,y] for TF import compatibility.
                    //Note also that empty is not the same as int[0] as in INDArray.sum(new int[0])
                    ((BaseReduceOp) op).setEmptyReduce(true);
                }
                axisArg = true;
            } else if (op instanceof ScalarOp) {
                if (df.argNames().length == 2) {
                    //Scalar ops: 2nd input should be treated as scalar...
                    SDVariable scalarVar = df.arg(1);
                    INDArray scalar = getArray(scalarVar, opInputs, allIterInputs);
                    Preconditions.checkState(scalar != null, "Could not get scalar argument for op %s: %s", df.getOwnName(), df.getClass());
                    Preconditions.checkState(scalar.isScalar(), "Scalar argument for op %s (%s) is not a scalar: has shape %ndShape", df.getOwnName(), df.getClass(), scalar);
                    ((ScalarOp) op).setScalar(scalar);
                } else if (df.argNames().length == 1) {
                    // Handle scalar ops where the scalar was set directly (e.g., in.add(2.0))
                    // In this case, the scalar value is stored in the op's scalarValue field
                    INDArray existingScalar = ((ScalarOp) op).scalar();
                    if (existingScalar == null) {
                        log.warn("ScalarOp {} has no scalar value set - operations may produce incorrect results", df.getOwnName());
                    }
                    // The scalar is already set in the op from construction, no action needed
                }
            }

            if (args != null && args.length > 0) {
                oc.setInputArray(0, args[0]);
                if (args.length == 2 && !axisArg)
                    oc.setInputArray(1, args[1]);
            }


            //Check output shape; allocate a new Z if required
            //For example, if minibatch size has changed since last op execution
            boolean isOutput = allReqVariables.contains(((BaseOp) op).outputVariablesNames()[0]);
            if (emptyReduce) {
                //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                INDArray z = mmgr.allocate(false, oc.getInputArray(0).dataType(), oc.getInputArray(0).shape());
                oc.setOutputArray(0, z);
            } else {
                List<DataBuffer> outputShape;
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    outputShape = ((BaseOp) op).calculateOutputShape(oc);
                }
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                // Note: outputShape.get(0) (stored as lsd) is OWNED by the output array after allocation
                // We must NOT close it as it becomes the array's shape descriptor
                DataBuffer lsd = outputShape.get(0);
                INDArray z = mmgr.allocateFromDescriptor(isOutput, lsd);
                oc.setOutputArray(0, z);
                // Since size == 1, no other buffers to clean up
            }
        }

        return new Pair<>(sdo, oc);
    }


    protected INDArray getArray(SDVariable sdv, Collection<VarId> opInputs, Collection<VarId> allIterInputs) {
        String n = sdv.name();
        if (sdv.getVariableType() == VariableType.CONSTANT || sdv.getVariableType() == VariableType.VARIABLE) {
            return getConstantOrVariable(n);

        }   else {
            VarId inVarId = lookup(n, opInputs, allIterInputs, false);
            Preconditions.checkState(inVarId != null, "Could not find array for variable %s", sdv.name());
            return getTensorFromOutputs(inVarId);
        }
    }

    /**
     * DEPRECATED: Do NOT use this method!
     *
     * This method was causing NaN values and potential heap corruption by closing OpaqueNDArrays
     * that are still referenced by the OpContext's OpaqueNDArrayArr. The OpaqueNDArrays are
     * CACHED in the INDArrays and SHARED with the OpContext. Closing them here causes:
     * 1. Dangling pointers in the OpContext's OpaqueNDArrayArr (PointerPointer still has old ptrs)
     * 2. Potential use-after-free if native code accesses the freed sd::NDArray* objects
     * 3. NaN values appearing in reduce operations due to reading freed/corrupted memory
     *
     * The OpaqueNDArrays are properly cleaned up when:
     * 1. The OpContext is closed (at end of inference via opContexts().clear())
     * 2. The INDArrays are garbage collected (their cached OpaqueNDArrays are cleaned up)
     *
     * @param opContext The OpContext (not used - method is deprecated)
     * @deprecated This method causes memory corruption. Let the normal cleanup path handle it.
     */
    @Deprecated
    @SuppressWarnings("unused")
    private void clearOpaqueNDArraysFromOpContext(OpContext opContext) {
        // DO NOT USE THIS METHOD
        // Left here for documentation purposes only
        log.warn("clearOpaqueNDArraysFromOpContext is deprecated and should not be called - ignoring");
    }

    /**
     * Compute a hash key for output shape caching based on op name, input array shapes,
     * and all op arguments (iArgs, tArgs, dArgs, bArgs, sArgs) since they all affect output shapes.
     */
    private long computeShapeKey(String opName, OpContext opContext) {
        long hash = opName.hashCode() * 0x9E3779B97F4A7C15L;
        List<INDArray> inputs = opContext.getInputArrays();
        for (int i = 0; i < inputs.size(); i++) {
            INDArray in = inputs.get(i);
            if (in != null) {
                long[] s = in.shape();
                for (long dim : s) {
                    hash ^= dim;
                    hash *= 0x517CC1B727220A95L;
                }
                hash ^= in.dataType().ordinal();
                hash *= 0x9E3779B97F4A7C15L;
                // For INT/LONG inputs, include actual values in key.
                // These carry shape/axis/permutation parameters whose VALUES
                // determine output shape (e.g. permute order, reshape target, gather axis).
                // Data is already synced to host by the caller.
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
        // Include iArgs in key (e.g. conv2d kernel sizes, strides, padding)
        List<Long> iArgs = opContext.getIArguments();
        if (iArgs != null) {
            for (Long arg : iArgs) {
                hash ^= arg;
                hash *= 0x517CC1B727220A95L;
            }
        }
        // Include tArgs (floating point arguments)
        List<Double> tArgs = opContext.getTArguments();
        if (tArgs != null) {
            for (Double arg : tArgs) {
                hash ^= Double.doubleToLongBits(arg);
                hash *= 0x9E3779B97F4A7C15L;
            }
        }
        // Include dArgs (data type arguments)
        List<DataType> dArgs = opContext.getDArguments();
        if (dArgs != null) {
            for (DataType dt : dArgs) {
                hash ^= dt.ordinal();
                hash *= 0x517CC1B727220A95L;
            }
        }
        // Include bArgs (boolean arguments)
        List<Boolean> bArgs = opContext.getBArguments();
        if (bArgs != null) {
            for (Boolean b : bArgs) {
                hash ^= b ? 1L : 0L;
                hash *= 0x9E3779B97F4A7C15L;
            }
        }
        // Include sArgs (string arguments)
        List<String> sArgs = opContext.getSArguments();
        if (sArgs != null) {
            for (String s : sArgs) {
                hash ^= (s == null ? 0L : s.hashCode());
                hash *= 0x517CC1B727220A95L;
            }
        }
        return hash;
    }

}
