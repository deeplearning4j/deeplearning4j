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

    // Enable shapeFunctionOverride by default to skip redundant C++ shape calculation.
    // When Java-side calculates shapes and pre-allocates outputs, C++ doesn't need to redo it.
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

    /** Persistent buffer pool for cross-execution array reuse (avoids mmgr round-trip each step). */
    private LocalBufferPool localPool;

    /** Persistent OpContext pool (avoids native allocation each step). */
    private final ArrayDeque<OpContext> ctxPool = new ArrayDeque<>();

    // Timing accumulators
    private long timingWireInputsNs, timingSyncNs, timingShapeNs, timingAllocNs, timingExecNs, timingReleaseNs;
    private int timingShapeHits, timingShapeMisses;
    private int timingZeroSkipped, timingZeroApplied;
    private int timingPoolHits, timingPoolMisses;

    public DynamicShapePlanExecutor(SameDiff sd, SessionMemMgr mmgr) {
        this.sd = sd;
        this.mmgr = mmgr;
    }

    /**
     * Initialize the executor for a specific plan.
     */
    public void initialize(DynamicShapePlan plan) {
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
        }

        // Clear output slots from the previous execution.
        // We do NOT call mmgr.release() here because these arrays may have already been
        // released during the previous execution's step-by-step release, or they may be
        // final-output arrays that were dup'd for the caller.
        Arrays.fill(outputSlots, null);
        Arrays.fill(externalInputs, null);
        liveSlots.clear();

        // Resolve external inputs (constants, variables, placeholders)
        resolveExternalInputs(plan, placeholderArrays);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapeSlot[] slots = plan.getSlots();
        // Persistent pool — buffers survive across execute() calls, avoiding mmgr
        // round-trip overhead. Lazy init here in case initialize() wasn't called.
        if (localPool == null) localPool = new LocalBufferPool();

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
                    executeSlot(slot, ctx, nativeOps, localPool);
                } catch (Exception e) {
                    log.error("Error executing slot {} ({}): {}", stepIdx, slot.getOpName(), e.getMessage());
                    throw new RuntimeException("DynamicShapePlan execution failed at step " + stepIdx +
                            " (" + slot.getOpName() + ")", e);
                }

                ctx.purgeForReuse();
                ctxPool.offerFirst(ctx);

                // Release dead arrays from previous steps into local pool (ORT-style reuse).
                long tRelease0 = TIMING_ENABLED ? System.nanoTime() : 0;
                int[] toRelease = plan.getReleaseAtStep()[stepIdx];
                for (int slotIdx : toRelease) {
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null && liveSlots.get(slotIdx)) {
                        localPool.release(arr);
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                if (TIMING_ENABLED) timingReleaseNs += System.nanoTime() - tRelease0;
            }

            // Collect output arrays using pre-built index map (O(1) per output).
            // dup() ensures caller gets correctly-sized independent copies —
            // DSP-allocated arrays may have over-allocated buffers (growth factor)
            // which could confuse downstream ops when passed back as inputs.
            try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
                for (Map.Entry<String, Integer> entry : outputMap.entrySet()) {
                    int slotIdx = entry.getValue();
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null) {
                        results.put(entry.getKey(), arr.dup());
                    }
                }
            }

            if (TIMING_ENABLED) {
                printTimingSummary(slots.length, localPool);
            }

            return results;
        } finally {
            // Release any remaining live arrays into persistent pool for next execution's reuse.
            // These are final-output arrays and any not yet released by the liveness schedule.
            if (outputSlots != null && liveSlots != null) {
                for (int i = 0; i < outputSlots.length; i++) {
                    INDArray arr = outputSlots[i];
                    if (arr != null && liveSlots.get(i)) {
                        localPool.release(arr);
                        outputSlots[i] = null;
                        liveSlots.clear(i);
                    }
                }
            }
            // Pool and ctxPool are persistent — don't flush/close here.
            // They persist across execute() calls and are cleaned up in close().
        }
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
                             LocalBufferPool localPool) {
        DifferentialFunction fn = slot.getOp();

        // Step 1: Wire inputs using pre-allocated buffer (avoids per-step allocation)
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

        // Step 3: Compute output shapes (using pre-cached opName hash + direct array access)
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
                    // Skip zeroing ONLY for matmul - BLAS GEMM is contractually guaranteed to
                    // write C[i,j] = sum(A[i,k]*B[k,j]) for every (i,j).
                    // Broader FULLY_WRITING_OPS skip was tested and produced wrong output —
                    // many ops in that set have edge cases where they don't fully write.
                    String opName = slot.getOpName();
                    boolean canSkipZero = "matmul".equals(opName) || "mmul".equals(opName);
                    if (TIMING_ENABLED) {
                        if (canSkipZero) {
                            timingZeroSkipped++;
                        } else {
                            timingZeroApplied++;
                        }
                    }
                    if (!canSkipZero) {
                        // Use direct memset instead of assign(0) to avoid full op dispatch
                        // overhead (ScalarSet creation → executor → JNI → C++ op dispatch).
                        fastZero(out, nativeOps);
                    }
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

        // Step 5: Set shapeFunctionOverride and execute
        // Default false: let C++ handle shape validation and prepareOutputs.
        // Enable via system property for performance (skips redundant shape calc).
        ctx.shapeFunctionOverride(SHAPE_OVERRIDE);

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

        // After execution, C++ may have replaced output arrays with different references.
        // Update slot tracking and release the original (now orphaned) arrays.
        List<INDArray> ctxOutputs = ctx.getOutputArrays();
        int maxTracked = Math.min(ctxOutputs != null ? ctxOutputs.size() : 0, outputSlotIndices.length);

        if (ctxOutputs != null) {
            for (int i = 0; i < maxTracked; i++) {
                INDArray ctxOut = ctxOutputs.get(i);
                if (ctxOut != null && outputSlotIndices[i] >= 0 && ctxOut != outputArrays[i]) {
                    // C++ replaced this output — release original, track replacement
                    localPool.release(outputArrays[i]);
                    outputSlots[outputSlotIndices[i]] = ctxOut;
                }
            }
        }

        // Release untracked output arrays: those with outputSlotIndices[i] == -1
        // (not consumed by any downstream op) or extra shapes beyond outputSlotIndices.length.
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
                    localPool.release(arr);
                }
            }
        }

        if (TIMING_ENABLED) timingExecNs += System.nanoTime() - tExec0;
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
        // Compute shape key from inputs using pre-cached opName hash and direct array access
        long shapeKey = computeShapeKey(slot, inputArrays);

        // Check per-slot cache
        if (slot.isShapeCacheValid(shapeKey) && !slot.isDataDependent()) {
            if (TIMING_ENABLED) timingShapeHits++;
            return slot.getCachedOutputShapes();
        }

        // Cache miss — try Java-side shape inference first (avoids JNI overhead)
        if (TIMING_ENABLED) timingShapeMisses++;

        List<DataBuffer> outShapes = null;

        // Try Java-side shape calculation from the op itself
        if (fn instanceof DynamicCustomOp) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            outShapes = ((DynamicCustomOp) fn).calculateOutputShapeFromInputs(ctx);
        }

        // Fall back to native shape function if Java-side didn't provide a result
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
                        // Shape-only op: skip D2H sync (shape function only reads shape info)
                        shapeList = nativeOps.calculateOutputShapesNoSync(null, opHash,
                                ctx.contextPointer());
                    } else {
                        // Value-dependent op: need sync for scalar/tensor value reads
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

        // Update per-slot cache
        if (!slot.isDataDependent() && outShapes != null && !outShapes.isEmpty()) {
            slot.updateShapeCache(shapeKey, outShapes);
        }

        return outShapes;
    }

    /**
     * Read a shape info buffer from a native OpaqueShapeList result and route through
     * ConstantShapeHelper cache for stable, device-aware pointers.
     */
    private static DataBuffer readShapeFromNative(NativeOps nativeOps, OpaqueShapeList list, int index) {
        LongPointer ptr = new PagedPointer(nativeOps.getShape(list, index)).asLongPointer();
        int rank = (int) ptr.get(0);
        int len = Shape.shapeInfoLength(rank);
        long[] shapeInfo = new long[len];
        ptr.capacity(len);
        ptr.get(shapeInfo, 0, len);

        // Route through ConstantShapeHelper C++ cache for stable, device-aware pointers.
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

    /**
     * Compute a shape key for cache lookup using pre-cached opName hash and direct
     * array access (avoids ctx.getInputArrays() list copy and String.hashCode()).
     *
     * <p>For ops whose output shape depends only on input shapes (not values), INT/LONG
     * input values are excluded from the key. This avoids: (1) expensive CUDA D2H sync
     * from {@code in.getLong(j)} on device arrays, and (2) false cache misses when INT/LONG
     * values change but input shapes are unchanged (e.g., position counters, seq_len scalars
     * consumed by ops like add/multiply whose output shape is independent of values).</p>
     */
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

                // Only include INT/LONG values for ops whose output shape depends on them
                // (reshape, strided_slice, tile, pad, etc.). For all other ops, input shapes
                // + dtypes are sufficient and we avoid expensive CUDA D2H sync.
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
        // Include iArgs from pre-frozen slot (avoids ctx.getIArguments() list copy)
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

        // Always use detached=true so that DSP-allocated arrays are independent of
        // any active workspace. Non-detached arrays become workspace-attached and
        // non-closeable, causing native GPU memory leaks when the DSP executor
        // tries to release them (localPool rejects non-closeable arrays).
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
     * Bypasses: ScalarSet op creation → OpExecutioner.exec() → JNI → C++ op dispatcher → CUDA kernel.
     * Replaces with: JNI → cudaMemset (single native call, ~10-50us vs ~250us for assign(0)).
     */
    private static void fastZero(INDArray arr, NativeOps nativeOps) {
        DataBuffer buf = arr.data();
        if (buf == null || buf.wasClosed()) return;

        OpaqueDataBuffer opaque = buf.opaqueBuffer();
        long bytes = buf.length() * buf.getElementSize();

        // Try device (special) buffer first (CUDA backend)
        Pointer specialPtr = nativeOps.dbSpecialBuffer(opaque);
        if (specialPtr != null && specialPtr.address() != 0) {
            nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
            // Mark device as authoritative so host reads trigger sync
            nativeOps.dbTickDeviceWrite(opaque);
        } else {
            // CPU backend: zero primary (host) buffer directly
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

    /**
     * Reshape a buffer to a new shape. Returns true if the shape was actually changed
     * (and device state was invalidated via dbSetDeviceId(buf, -1)), or false if the
     * shape already matched (no-op, device state remains valid).
     */
    private static boolean reshapeBuffer(INDArray arr, long[] shape) {
        if (arr == null || shape == null || shape.length == 0) {
            return false;
        }
        if (Arrays.equals(arr.shape(), shape)) {
            return false; // Shape matches — no change, device state still valid
        }
        long[] newStrides = Nd4j.getStrides(shape, arr.ordering());
        int[] intShape = ArrayUtil.toInts(shape);
        int[] intStrides = ArrayUtil.toInts(newStrides);
        ((BaseNDArray) arr).setShapeAndStride(intShape, intStrides);
        ((BaseNDArray) arr).assignNewId();
        if (arr.data() != null) {
            Nd4j.getNativeOps().dbSetDeviceId(arr.data().opaqueBuffer(), -1);
        }
        return true; // Shape changed, device state invalidated
    }

    /**
     * Local buffer pool for intra-execution array reuse. Recycles INDArrays by
     * DataType and buffer size using TreeMap-based ceiling lookups.
     *
     * <p>Uses {@code DataBuffer.closeable()} instead of {@code INDArray.closeable()} for
     * pool eligibility. Over-allocated arrays (from growth factor) have
     * {@code data().length() > length()} which makes {@code isView()} return true and
     * {@code INDArray.closeable()} return false, even though these arrays exclusively
     * own their buffer and are safe to pool. Checking the DataBuffer directly avoids
     * this false positive, allowing ~583 arrays (~5GB) per frame to be reused instead
     * of force-closed via cudaFree.</p>
     */
    private static final class LocalBufferPool {
        private final Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> pools = new EnumMap<>(DataType.class);
        private final double largerArrayMaxMultiple;
        /** Tracks pooled array references to prevent double-add (which causes
         *  "Array was released multiple times" in ArrayCacheMemoryMgr.release). */
        private final Set<INDArray> pooledRefs = Collections.newSetFromMap(new IdentityHashMap<>());

        /** Whether the last acquire() call required reshapeBuffer to change the shape.
         *  When false, the buffer's device state is still valid from its previous use
         *  and assign(0) can be skipped for fully-writing ops. */
        private boolean lastAcquireReshaped;

        private int releaseAccepted;
        private int releaseRejected;

        private LocalBufferPool() {
            this.largerArrayMaxMultiple = ArrayCacheMemoryMgr.getLargerArrayMaxMultiple().get();
        }

        INDArray acquire(DataType dataType, long[] shape) {
            if (shape == null || shape.length == 0) {
                return null; // Don't pool scalars — reshapeBuffer can't handle empty shape
            }
            long requiredElements = numElements(shape);
            if (requiredElements <= 0) {
                return null;
            }

            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.get(dataType);
            if (tree == null || tree.isEmpty()) {
                return null;
            }

            long maxElements = (long) (requiredElements * largerArrayMaxMultiple);
            Map.Entry<Long, ArrayDeque<INDArray>> entry = tree.ceilingEntry(requiredElements);
            while (entry != null) {
                long bufferElements = entry.getKey();
                if (bufferElements > maxElements) {
                    break;
                }
                ArrayDeque<INDArray> deque = entry.getValue();
                while (deque != null && !deque.isEmpty()) {
                    INDArray arr = deque.poll();
                    if (arr == null) {
                        continue;
                    }
                    if (deque.isEmpty()) {
                        tree.remove(bufferElements);
                    }
                    // Check DataBuffer directly: over-allocated arrays have isView()=true
                    // (data().length() > length()) but exclusively own their buffer.
                    DataBuffer buf = arr.data();
                    if (arr.wasClosed() || buf == null || buf.wasClosed() || !buf.closeable()) {
                        continue;
                    }
                    if (arr.dataType() != dataType) {
                        continue;
                    }

                    pooledRefs.remove(arr);
                    lastAcquireReshaped = reshapeBuffer(arr, shape);
                    return arr;
                }
                entry = tree.higherEntry(bufferElements);
            }
            return null;
        }

        boolean wasLastAcquireReshaped() {
            return lastAcquireReshaped;
        }

        void release(INDArray arr) {
            if (arr == null || arr.wasClosed()) return;
            DataBuffer buf = arr.data();
            if (buf == null || buf.wasClosed()) return;
            // Check DataBuffer.closeable() instead of INDArray.closeable():
            // Over-allocated arrays (growth factor) have data().length() > length()
            // which makes isView() return true and INDArray.closeable() return false.
            // But these arrays exclusively own their buffer — pooling them is safe and
            // avoids 583+ cudaFree calls per frame (~5GB).
            if (!buf.closeable()) {
                releaseRejected++;
                return;
            }
            // Prevent double-add of the same array reference (identity check).
            // Can happen when C++ replaces an output with one of the input arrays.
            if (!pooledRefs.add(arr)) {
                return; // already in pool
            }
            DataType dt = arr.dataType();
            long bufferElements = buf.length();
            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.computeIfAbsent(dt, k -> new TreeMap<>());
            tree.computeIfAbsent(bufferElements, k -> new ArrayDeque<>()).add(arr);
            releaseAccepted++;
        }

        void flushTo(SessionMemMgr mmgr) {
            int flushed = 0;
            int skippedDup = 0;
            long flushedBytes = 0;
            // Track IDs we've already flushed to avoid double-release to mmgr.
            // C++ op execution can replace outputs with input arrays, causing the
            // same underlying buffer (same id) to appear under multiple slot indices.
            Set<Long> flushedIds = new HashSet<>();
            for (TreeMap<Long, ArrayDeque<INDArray>> tree : pools.values()) {
                for (ArrayDeque<INDArray> deque : tree.values()) {
                    for (INDArray arr : deque) {
                        if (arr == null || arr.wasClosed()) continue;
                        DataBuffer buf = arr.data();
                        if (buf == null || buf.wasClosed() || !buf.closeable()) continue;
                        long id = arr.getId();
                        if (!flushedIds.add(id)) {
                            skippedDup++;
                            continue;
                        }
                        flushedBytes += buf.length() * arr.dataType().width();
                        try {
                            mmgr.release(arr);
                            flushed++;
                        } catch (Exception e) {
                            // ArrayCacheMemoryMgr may reject arrays already in its LRU
                            // (e.g., from C++ output replacement creating alias chains).
                            // Safe to skip — the mmgr still holds the buffer reference.
                            skippedDup++;
                        }
                    }
                }
            }
            pools.clear();
            pooledRefs.clear();
            if (TIMING_ENABLED) {
                log.info("  LocalBufferPool: flushed {} arrays ({}MB), pooled={}, rejected={}, dupSkipped={}",
                        flushed, flushedBytes / (1024 * 1024),
                        releaseAccepted, releaseRejected, skippedDup);
            }
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
        log.info("  Release:      {}ms", String.format("%.2f", timingReleaseNs / 1_000_000.0));
        // GPU memory pool stats (if available)
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            org.bytedeco.javacpp.LongPointer usedPtr = new org.bytedeco.javacpp.LongPointer(1);
            org.bytedeco.javacpp.LongPointer reservedPtr = new org.bytedeco.javacpp.LongPointer(1);
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
        // Flush persistent pool back to SessionMemMgr
        if (localPool != null) {
            localPool.flushTo(mmgr);
            localPool = null;
        }
        // Close persistent OpContexts
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
