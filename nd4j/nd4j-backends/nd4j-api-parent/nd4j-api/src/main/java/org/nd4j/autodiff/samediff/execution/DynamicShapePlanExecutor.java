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
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
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
import org.nd4j.nativeblas.OpaqueShapeList;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

import java.io.Closeable;
import java.util.*;

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
            System.getProperty("org.nd4j.inference.timing", "false"));

    private static final boolean SHAPE_OVERRIDE = Boolean.parseBoolean(
            System.getProperty("org.nd4j.inference.dynamicShapePlan.shapeOverride", "true"));


    private final SameDiff sd;
    private final SessionMemMgr mmgr;

    /** The plan this executor is currently configured for. */
    private DynamicShapePlan currentPlan;

    /** Flat output array slots: stores op outputs by slot index. */
    private INDArray[] outputSlots;

    /** External input array cache: resolved constant/variable/placeholder arrays. */
    private INDArray[] externalInputs;

    /** BitSet tracking which slots are currently live (have valid arrays). */
    private BitSet liveSlots;

    /** Pending DataBuffers to close after execution completes.
     *  Closing is deferred because: (1) buf.close() calls cudaFreeAsync(stream 0) which
     *  races with kernels on the execution stream, and (2) view arrays share GPU pointers
     *  with parents, requiring GPU-address dedup that's only safe without intervening allocs. */
    private ArrayList<DataBuffer> pendingClose;

    /** Persistent buffer pool for cross-execution array reuse (avoids mmgr round-trip each step). */
    private LocalBufferPool localPool;

    /** Persistent OpContext pool (avoids native allocation each step). */
    private final ArrayDeque<OpContext> ctxPool = new ArrayDeque<>();

    // Timing accumulators
    private long timingWireInputsNs, timingSyncNs, timingShapeNs, timingAllocNs, timingExecNs, timingReleaseNs;
    private int timingShapeHits, timingShapeMisses;
    private int timingZeroSkipped, timingZeroApplied;
    private int timingPoolHits, timingPoolMisses;
    // Release diagnostics
    private int pendingCloseCount, pendingCloseViewCount;
    private long pendingCloseBytes;

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
        // Plan changed — flush old pool when switching plans
        if (localPool != null) {
            localPool.flushTo(mmgr);
        }
        currentPlan = plan;
        outputSlots = new INDArray[plan.getTotalOutputSlots()];
        externalInputs = new INDArray[plan.getExternalInputKeys().length];
        liveSlots = new BitSet(plan.getTotalOutputSlots());
        pendingClose = new ArrayList<>();
        localPool = new LocalBufferPool();
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

        if (TIMING_ENABLED) {
            timingWireInputsNs = timingSyncNs = timingShapeNs = timingAllocNs = timingExecNs = timingReleaseNs = 0;
            timingShapeHits = timingShapeMisses = 0;
            timingZeroSkipped = timingZeroApplied = 0;
            timingPoolHits = timingPoolMisses = 0;
            pendingCloseCount = pendingCloseViewCount = 0;
            pendingCloseBytes = 0;
        }

        // Clear output slots from the previous execution.
        Arrays.fill(outputSlots, null);
        Arrays.fill(externalInputs, null);
        liveSlots.clear();
        if (pendingClose == null) pendingClose = new ArrayList<>();
        pendingClose.clear();

        // Resolve external inputs (constants, variables, placeholders)
        resolveExternalInputs(plan, placeholderArrays);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapeSlot[] slots = plan.getSlots();
        if (localPool == null) localPool = new LocalBufferPool();

        // Cache the execution stream once per execute() call.
        Pointer execStream = null;
        try {
            OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
            if (lc != null) {
                execStream = nativeOps.lcExecutionStream(lc);
                if (execStream != null) execStream.retainReference();
            }
        } catch (Exception e) {
            // CPU backend or unavailable
        }

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
                        log.info("DSP step {}/{}: op={}", stepIdx, slots.length, slot.getOpName());
                    }
                    executeSlot(slot, ctx, nativeOps, localPool, execStream);
                } catch (Exception e) {
                    log.error("Error executing slot {} ({}): {}", stepIdx, slot.getOpName(), e.getMessage());
                    throw new RuntimeException("DynamicShapePlan execution failed at step " + stepIdx +
                            " (" + slot.getOpName() + ")", e);
                }

                ctx.purgeForReuse();
                ctxPool.offerFirst(ctx);

                // Mark dead slots for deferred close. Don't close now because:
                // (1) GPU kernels may still be using the buffer on the execution stream
                // (2) View arrays share GPU pointers — dedup is only safe post-commit
                long tRelease0 = TIMING_ENABLED ? System.nanoTime() : 0;
                int[] toRelease = plan.getReleaseAtStep()[stepIdx];
                for (int slotIdx : toRelease) {
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null && liveSlots.get(slotIdx)) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                            pendingClose.add(buf);
                            if (TIMING_ENABLED) {
                                pendingCloseBytes += buf.length() * buf.getElementSize();
                                pendingCloseCount++;
                                if (arr.isView()) pendingCloseViewCount++;
                            }
                        }
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                if (TIMING_ENABLED) timingReleaseNs += System.nanoTime() - tRelease0;
            }

            // Collect output arrays (dup before we close intermediates).
            // CRITICAL: commit() first so the async dup copy reads completed GPU data,
            // then syncToHost() so the host buffer has valid data before closePendingBuffers
            // frees the source GPU buffers.
            Nd4j.getExecutioner().commit();
            try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
                for (Map.Entry<String, Integer> entry : outputMap.entrySet()) {
                    int slotIdx = entry.getValue();
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null) {
                        INDArray duped = arr.dup();
                        Nd4j.getExecutioner().commit();
                        // Force host sync while all GPU buffers still valid.
                        // On CUDA, getFloat triggers lazyAllocateHostPointer + cudaMemcpy.
                        // Must happen BEFORE closePendingBuffers frees intermediate GPU buffers.
                        if (duped.length() > 0 && !duped.isEmpty()) {
                            duped.getFloat(0);
                        }
                        results.put(entry.getKey(), duped);
                    }
                }
            }

            // Collect remaining live slots (output slots, etc.) into the SAME pendingClose
            // batch. This is critical: using a single closePendingBuffers call ensures GPU
            // address dedup catches shared buffers between intermediates and output slots.
            // Two separate calls would have separate closedAddresses sets → double-free.
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && liveSlots.get(i)) {
                    DataBuffer buf = arr.data();
                    if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                        pendingClose.add(buf);
                    }
                    outputSlots[i] = null;
                    liveSlots.clear(i);
                }
            }

            // Now close ALL pending buffers in one batch. commit() syncs the execution stream
            // first, ensuring all GPU kernels are done before we free their buffers.
            // GPU address dedup prevents double-free for view arrays sharing parent buffers.
            // No new allocations happen between closes, so address reuse is impossible.
            closePendingBuffers(nativeOps);

            if (TIMING_ENABLED) {
                printTimingSummary(slots.length, localPool);
            }

            return results;
        } finally {
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
     * Close all pending DataBuffers after execution completes.
     * Sync the execution stream first, then close with GPU address dedup.
     *
     * Uses dbFreeBuffersOnly() which frees GPU memory via cudaFreeAsync and abandons
     * the host-side pinned buffer (via madvise(MADV_DONTNEED)) rather than calling free().
     * This avoids the overhead of full DataBuffer destruction for intermediates that are
     * immediately discarded. The DataBuffer C++ object (~200B) leaks but total is
     * &lt;500KB per execution.
     */
    private void closePendingBuffers(NativeOps nativeOps) {
        if (pendingClose.isEmpty()) return;

        // Sync execution stream so all GPU kernels are complete
        Nd4j.getExecutioner().commit();

        // Three layers of dedup to prevent double-free:
        Set<DataBuffer> seenIdentity = Collections.newSetFromMap(new IdentityHashMap<>(pendingClose.size()));
        HashSet<Long> closedOdbAddresses = new HashSet<>(pendingClose.size());
        HashSet<Long> closedGpuAddresses = new HashSet<>(pendingClose.size());
        int freedCount = 0;
        long freedBytes = 0;
        int identitySkip = 0, odbSkip = 0, gpuSkip = 0;

        for (DataBuffer buf : pendingClose) {
            if (buf == null || buf.wasClosed() || !buf.closeable() || buf.isConstant()) continue;

            // Layer 1: Java identity dedup
            if (!seenIdentity.add(buf)) {
                identitySkip++;
                continue;
            }

            OpaqueDataBuffer odb = buf.opaqueBuffer();
            if (odb == null || odb.isNull()) continue;

            // Layer 2: OpaqueDataBuffer address dedup
            long odbAddr = odb.address();
            if (odbAddr != 0 && !closedOdbAddresses.add(odbAddr)) {
                odbSkip++;
                continue;
            }

            // Layer 3: GPU memory address dedup (views share parent GPU pointers)
            long gpuAddr = 0;
            Pointer special = nativeOps.dbSpecialBuffer(odb);
            if (special != null && special.address() != 0) {
                gpuAddr = special.address();
            }
            if (gpuAddr != 0 && !closedGpuAddresses.add(gpuAddr)) {
                gpuSkip++;
                continue;
            }

            try {
                freedBytes += buf.length() * buf.getElementSize();
                freedCount++;
                nativeOps.dbFreeBuffersOnly(odb);
            } catch (Exception e) {
                log.warn("  dbFreeBuffersOnly failed ({}B): {}",
                        buf.length() * buf.getElementSize(), e.getMessage());
            }
        }

        log.info("  Deferred close: {}/{} buffers ({}MB), skips: identity={}, odb={}, gpu={}",
                freedCount, pendingClose.size(), freedBytes / (1024 * 1024),
                identitySkip, odbSkip, gpuSkip);
        pendingClose.clear();
    }

    private void resolveExternalInputs(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        String[] keys = plan.getExternalInputKeys();
        for (int i = 0; i < keys.length; i++) {
            String varName = keys[i];
            INDArray arr = null;

            // Try placeholder first
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
        }
    }

    private void executeSlot(DynamicShapeSlot slot, OpContext ctx, NativeOps nativeOps,
                             LocalBufferPool localPool, Pointer execStream) {
        DifferentialFunction fn = slot.getOp();

        // Step 0: Device placement
        int previousDeviceId = -1;
        int targetDevice = slot.getTargetDeviceId();
        if (targetDevice >= 0) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (currentDevice != targetDevice) {
                previousDeviceId = currentDevice;
                Nd4j.getAffinityManager().unsafeSetDevice(targetDevice);
            }
        }

        try {
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

            INDArray out;
            if (Shape.isEmpty(shapeInfo) || numElements(actualShape) == 0) {
                out = Nd4j.emptyWithShape(actualShape, dt);
            } else {
                out = localPool.acquire(dt, actualShape);
                if (out == null) {
                    out = allocateWithHeadroom(dt, actualShape);
                    if (TIMING_ENABLED) timingPoolMisses++;
                } else {
                    if (TIMING_ENABLED) timingPoolHits++;
                    if (TIMING_ENABLED) timingZeroApplied++;
                    fastZero(out, nativeOps, execStream);
                }
            }
            outputArrays[i] = out;

            if (i < outputSlotIndices.length && outputSlotIndices[i] >= 0) {
                outputSlots[outputSlotIndices[i]] = outputArrays[i];
                liveSlots.set(outputSlotIndices[i]);
            }
        }
        ctx.setOutputArrays(outputArrays);
        if (TIMING_ENABLED) timingAllocNs += System.nanoTime() - tAlloc0;

        // Step 5: Execute
        ctx.shapeFunctionOverride(SHAPE_OVERRIDE);

        // Attach native workspace to OpContext if available — this allows C++ ops to
        // use bump allocation for internal temporaries instead of per-op malloc/cudaMalloc.
        // Without this, C++ op temporary buffer overruns corrupt the regular malloc heap
        // metadata, causing "double free or corruption (out)" crashes.
        // The standard InferenceSession path always does this (see executeRegularOperation).
        Pointer wsPtr = mmgr.getNativeWorkspacePointer();
        if (wsPtr != null) {
            ctx.attachWorkspace(wsPtr);
        }

        long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
        if (slot.isCustomOp()) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            Nd4j.exec((CustomOp) fn, ctx);
        } else {
            Nd4j.exec((Op) fn, ctx);
        }

        // After execution, C++ may have replaced output arrays. Update tracking.
        List<INDArray> ctxOutputs = ctx.getOutputArrays();
        int maxTracked = Math.min(ctxOutputs != null ? ctxOutputs.size() : 0, outputSlotIndices.length);

        if (ctxOutputs != null) {
            for (int i = 0; i < maxTracked; i++) {
                INDArray ctxOut = ctxOutputs.get(i);
                if (ctxOut != null && outputSlotIndices[i] >= 0 && ctxOut != outputArrays[i]) {
                    // C++ replaced this output — defer close of orphaned original
                    DataBuffer buf = outputArrays[i].data();
                    if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                        pendingClose.add(buf);
                    }
                    outputSlots[outputSlotIndices[i]] = ctxOut;
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
                    if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                        pendingClose.add(buf);
                    }
                }
            }
        }

        if (TIMING_ENABLED) timingExecNs += System.nanoTime() - tExec0;

        } finally {
            if (previousDeviceId >= 0) {
                Nd4j.getAffinityManager().unsafeSetDevice(previousDeviceId);
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
            outShapes = ((DynamicCustomOp) fn).calculateOutputShapeFromInputs(ctx);
        }

        if (outShapes == null || outShapes.isEmpty()) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                if (fn instanceof CustomOp) {
                    ctx.setIArguments(slot.getIArgs());
                    ctx.setTArguments(slot.getTArgs());
                    ctx.setBArguments(slot.getBArgs());
                    ctx.setDArguments(slot.getDArgs());

                    long opHash = ((CustomOp) fn).opHash();
                    OpaqueShapeList shapeList;
                    if (!slot.isOutputShapeDependsOnInputValues()) {
                        shapeList = nativeOps.calculateOutputShapesNoSync(null, opHash,
                                ctx.contextPointer());
                    } else {
                        shapeList = nativeOps.calculateOutputShapes2(null, opHash,
                                ctx.contextPointer());
                    }

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
        boolean includeValues = slot.isOutputShapeDependsOnInputValues();
        for (INDArray in : inputArrays) {
            if (in != null) {
                for (long dim : in.shape()) {
                    hash ^= dim;
                    hash *= 0x517CC1B727220A95L;
                }
                hash ^= in.dataType().ordinal();
                hash *= 0x9E3779B97F4A7C15L;

                if (includeValues
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG)
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

    private INDArray allocateWithHeadroom(DataType dataType, long[] shape) {
        long requiredElements = numElements(shape);
        if (requiredElements <= 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }

        if (mmgr instanceof ArrayCacheMemoryMgr) {
            return mmgr.allocate(true, dataType, shape);
        }

        double gf = ArrayCacheMemoryMgr.getGrowthFactor().get();
        long overAllocThreshold = Math.max(ArrayCacheMemoryMgr.getSmallArrayThreshold().get(), 10_000);
        if (gf <= 1.0 || requiredElements <= overAllocThreshold) {
            return mmgr.allocate(true, dataType, shape);
        }

        long allocElements = (long) (requiredElements * gf);
        INDArray oversized = mmgr.allocate(true, dataType, allocElements);
        reshapeBuffer(oversized, shape);
        return oversized;
    }

    /**
     * Fast buffer zeroing using direct memset instead of the full assign(0) op dispatch path.
     */
    private static void fastZero(INDArray arr, NativeOps nativeOps, Pointer execStream) {
        DataBuffer buf = arr.data();
        if (buf == null || buf.wasClosed()) return;

        OpaqueDataBuffer opaque = buf.opaqueBuffer();
        long bytes = buf.length() * buf.getElementSize();

        Pointer specialPtr = nativeOps.dbSpecialBuffer(opaque);
        if (specialPtr != null && specialPtr.address() != 0) {
            if (execStream != null && execStream.address() != 0) {
                nativeOps.memsetAsync(specialPtr, 0, bytes, 0, execStream);
            } else {
                nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
            }
            nativeOps.dbTickDeviceWrite(opaque);
        } else {
            Pointer primaryPtr = nativeOps.dbPrimaryBuffer(opaque);
            if (primaryPtr != null && primaryPtr.address() != 0) {
                nativeOps.memsetSync(primaryPtr, 0, bytes, 0, null);
            }
        }
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
                System.getProperty("org.nd4j.dsp.pool.maxBytes",
                        String.valueOf(512L * 1024 * 1024)));

        private LocalBufferPool() {
            this.largerArrayMaxMultiple = ArrayCacheMemoryMgr.getLargerArrayMaxMultiple().get();
        }

        INDArray acquire(DataType dataType, long[] shape) {
            // TEMP: disable pool reuse pending investigation
            if (true) return null;
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
                    if (arr.wasClosed() || buf == null || buf.wasClosed() || !buf.closeable()) continue;
                    if (arr.dataType() != dataType) continue;

                    pooledRefs.remove(arr);
                    currentPoolBytes -= bufferElements * dataType.width();
                    lastAcquireReshaped = reshapeBuffer(arr, shape);
                    return arr;
                }
                entry = tree.higherEntry(bufferElements);
            }
            return null;
        }

        void release(INDArray arr) {
            if (arr == null || arr.wasClosed()) return;
            DataBuffer buf = arr.data();
            if (buf == null || buf.wasClosed() || !buf.closeable()) {
                releaseRejected++;
                return;
            }
            if (!pooledRefs.add(arr)) return;
            DataType dt = arr.dataType();
            long bufferElements = buf.length();
            long thisBytes = bufferElements * dt.width();
            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.computeIfAbsent(dt, k -> new TreeMap<>());
            tree.computeIfAbsent(bufferElements, k -> new ArrayDeque<>()).add(arr);
            currentPoolBytes += thisBytes;
            releaseAccepted++;
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
                        if (buf.closeable() && !arr.isView()) {
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
            log.info("  LocalBufferPool flushTo: {} buffers ({}MB), closed={}, pooled={}, rejected={}",
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
        log.info("  Mem alloc:    {}ms (pool hits={}, pool misses={}, zero skipped={}, zero applied={})",
                String.format("%.2f", timingAllocNs / 1_000_000.0),
                timingPoolHits, timingPoolMisses, timingZeroSkipped, timingZeroApplied);
        log.info("  Native exec:  {}ms", String.format("%.2f", timingExecNs / 1_000_000.0));
        log.info("  Pending close: {} buffers ({}MB), views={}",
                pendingCloseCount, pendingCloseBytes / (1024 * 1024), pendingCloseViewCount);
        // GPU memory pool stats
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            LongPointer usedPtr = new LongPointer(1);
            LongPointer reservedPtr = new LongPointer(1);
            nativeOps.getMemoryPoolStats(0, usedPtr, reservedPtr);
            long usedMB = usedPtr.get() / (1024 * 1024);
            long reservedMB = reservedPtr.get() / (1024 * 1024);
            log.info("  GPU memory pool: used={}MB, reserved={}MB", usedMB, reservedMB);
        } catch (Exception e) {
            // Not available on CPU backend
        }
    }

    @Override
    public void close() {
        if (localPool != null) {
            localPool.flushTo(mmgr);
            localPool = null;
        }
        for (OpContext ctx : ctxPool) {
            try { ctx.close(); } catch (Exception ignored) {}
        }
        ctxPool.clear();

        if (outputSlots != null) {
            Arrays.fill(outputSlots, null);
        }
        if (externalInputs != null) {
            Arrays.fill(externalInputs, null);
        }
        currentPlan = null;
    }
}
