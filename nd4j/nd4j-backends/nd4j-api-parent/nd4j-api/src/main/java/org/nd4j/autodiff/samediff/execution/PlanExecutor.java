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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Closeable;
import java.util.*;

/**
 * Executes a compiled {@link ExecutionPlan} with pre-allocated, persistent buffers
 * and zero per-op shape inference overhead.
 *
 * <p>Key optimizations over standard SameDiff execution:</p>
 * <ol>
 *   <li><b>shapeFunctionOverride</b> — tells the C++ native side to skip
 *       {@code prepareOutputs}/calculateOutputShape entirely when output arrays
 *       are already provided via OpContext. This is the single most impactful
 *       optimization.</li>
 *   <li><b>Persistent buffer pool</b> — intermediate INDArrays are allocated once
 *       on first execution and reused across subsequent calls with the same plan.
 *       No per-call Nd4j.create() for intermediates.</li>
 *   <li><b>True memory reuse</b> — REUSE buffers share the same underlying memory
 *       as the buffer they reuse (liveness analysis guarantees no overlap in time).</li>
 *   <li><b>Pre-allocated OpContexts</b> — one OpContext per slot, reused across calls.</li>
 *   <li><b>Output arrays allocated outside workspace</b> — graph outputs are allocated
 *       as regular arrays (not workspace-backed), eliminating the detach copy.</li>
 * </ol>
 *
 * @see ExecutionPlan
 * @see ExecutionPlanCompiler
 */
@Slf4j
public class PlanExecutor implements Closeable {

    private final SameDiff sd;

    /** Pre-allocated INDArray pool indexed by buffer allocation index. Persists across calls. */
    private INDArray[] bufferPool;

    /** Backing DataBuffer pools per datatype for contiguous allocations. */
    private Map<DataType, DataBuffer> dataTypePools;

    /** Pre-allocated OpContext pool indexed by slot index. Persists across calls. */
    private OpContext[] opContextPool;

    /** Whether the executor has been initialized for a specific plan. */
    private boolean initialized;

    /** The plan this executor is currently configured for. */
    private ExecutionPlan currentPlan;

    public PlanExecutor(SameDiff sd) {
        this.sd = sd;
    }

    /**
     * Accessor for tests and diagnostics.
     */
    public ExecutionPlan getCurrentPlan() {
        return currentPlan;
    }

    /**
     * Accessor for tests and diagnostics.
     */
    public INDArray[] getBufferPool() {
        return bufferPool;
    }

    /**
     * Accessor for tests and diagnostics.
     */
    public Map<DataType, DataBuffer> getDataTypePools() {
        return dataTypePools;
    }

    /**
     * Initialize (or reinitialize) the executor for a specific plan.
     * Pre-allocates the buffer pool and OpContexts. This runs once per plan shape.
     */
    public void initialize(ExecutionPlan plan) {
        if (initialized && currentPlan == plan) {
            return;
        }

        cleanup();
        currentPlan = plan;

        // Create persistent buffer pool
        bufferPool = new INDArray[plan.getBufferPoolSize()];

        // Allocate all intermediate and output buffers up front
        allocateBufferPoolOnce(plan);

        // Pre-allocate OpContexts
        opContextPool = new OpContext[plan.getSlots().size()];
        for (int i = 0; i < plan.getSlots().size(); i++) {
            opContextPool[i] = Nd4j.getExecutioner().buildContext();
        }

        initialized = true;

        log.debug("PlanExecutor initialized: {} buffers, {} ops, {} bytes total",
                plan.getBufferPoolSize(), plan.getSlots().size(), plan.getTotalHostBytes());
    }

    /**
     * Allocate the buffer pool once. Intermediate arrays are created as regular
     * arrays (not workspace-backed) so they persist across calls.
     *
     * <p>For REUSE buffers, we create an array that shares the same underlying
     * DataBuffer as the buffer being reused (just with a different shape view),
     * since liveness analysis guarantees non-overlapping lifetimes.</p>
     */
    private void allocateBufferPoolOnce(ExecutionPlan plan) {
        Map<DataType, Long> poolBytesByType = computePoolSizes(plan);
        dataTypePools = new EnumMap<>(DataType.class);

        try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<DataType, Long> entry : poolBytesByType.entrySet()) {
                DataType dt = entry.getKey();
                long bytes = entry.getValue();
                if (bytes <= 0) {
                    continue;
                }
                long elements = (bytes + dt.width() - 1) / dt.width();
                dataTypePools.put(dt, Nd4j.createBuffer(dt, elements, true));
            }

            char ordering = Nd4j.order();
            for (BufferAllocation alloc : plan.getAllocations()) {
                switch (alloc.getKind()) {
                    case ALLOCATE:
                    case OUTPUT:
                    case REUSE:
                        long[] shape = alloc.getShape();
                        if (shape == null || alloc.getSizeInBytes() <= 0) {
                            break;
                        }
                        if (hasZeroDimension(shape)) {
                            bufferPool[alloc.getIndex()] = Nd4j.emptyWithShape(shape, alloc.getDataType());
                            break;
                        }

                        DataBuffer pool = dataTypePools.get(alloc.getDataType());
                        if (pool == null) {
                            bufferPool[alloc.getIndex()] = Nd4j.create(alloc.getDataType(), shape);
                            break;
                        }

                        long elemSize = alloc.getDataType().width();
                        long offsetBytes = alloc.getOffsetInPool();
                        if (offsetBytes % elemSize != 0) {
                            log.warn("Misaligned pool offset {} for dtype {} ({} bytes) - falling back to standalone allocation",
                                    offsetBytes, alloc.getDataType(), elemSize);
                            bufferPool[alloc.getIndex()] = Nd4j.create(alloc.getDataType(), shape);
                            break;
                        }

                        long offsetElems = offsetBytes / elemSize;
                        long[] strides = Nd4j.getStrides(shape, ordering);
                        bufferPool[alloc.getIndex()] = Nd4j.create(pool, shape, strides, offsetElems, ordering);
                        break;

                    case PRE_EXISTING:
                    case CONSTANT:
                    case VARIABLE:
                        // Resolved from SameDiff at execution time
                        break;
                }
            }
        }
    }

    /**
     * Execute the plan with the given placeholder arrays.
     *
     * @param plan              the compiled execution plan
     * @param placeholderArrays placeholder name -> INDArray
     * @return map of requested output variable name -> INDArray
     */
    public Map<String, INDArray> execute(ExecutionPlan plan, Map<String, INDArray> placeholderArrays) {
        if (!initialized || currentPlan != plan) {
            initialize(plan);
        }

        // Execute all op slots with pre-allocated arrays and shapeFunctionOverride
        for (int slotIdx = 0; slotIdx < plan.getSlots().size(); slotIdx++) {
            OpSlot slot = plan.getSlots().get(slotIdx);
            executeSlot(plan, slot, slotIdx, placeholderArrays);
        }

        // Collect output arrays. Because output buffers are regular arrays (not workspace),
        // we return fresh copies so the caller owns them and the buffer pool stays stable.
        Map<String, INDArray> results = new LinkedHashMap<>();
        try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            for (String outputName : plan.getRequestedOutputs()) {
                BufferAllocation alloc = plan.getBufferMap().get(outputName);
                if (alloc != null && bufferPool[alloc.getIndex()] != null) {
                    results.put(outputName, bufferPool[alloc.getIndex()].dup());
                }
            }
        }

        return results;
    }

    private void executeSlot(ExecutionPlan plan, OpSlot slot, int slotIdx,
                             Map<String, INDArray> placeholderArrays) {
        OpContext ctx = opContextPool[slotIdx];
        ctx.purge(); // Reset inputs/outputs without deallocating the native context

        DifferentialFunction fn = slot.getOp();

        // Wire inputs
        List<INDArray> inputArrays = new ArrayList<>(slot.getInputBufferIndices().length);
        for (int i = 0; i < slot.getInputBufferIndices().length; i++) {
            int bufIdx = slot.getInputBufferIndices()[i];
            INDArray input;
            if (bufIdx >= 0 && bufferPool[bufIdx] != null) {
                input = bufferPool[bufIdx];
            } else {
                input = resolveExternalInput(slot.getInputVarNames()[i], placeholderArrays);
            }
            if (input != null) {
                inputArrays.add(input);
            }
        }
        ctx.setInputArrays(inputArrays);

        // Handle dynamic shape ops — fall back to runtime shape calculation
        if (slot.isRequiresDynamicShapeInference()) {
            executeDynamicSlot(plan, slot, ctx, placeholderArrays);
            return;
        }

        // Wire pre-allocated outputs
        for (int i = 0; i < slot.getOutputBufferIndices().length; i++) {
            int bufIdx = slot.getOutputBufferIndices()[i];
            if (bufIdx >= 0 && bufferPool[bufIdx] != null) {
                ctx.setOutputArray(i, bufferPool[bufIdx]);
            }
        }

        // THE CRITICAL OPTIMIZATION: Tell the C++ native side that output shapes
        // are already set and it should NOT call calculateOutputShape/prepareOutputs.
        // This bypasses the most expensive per-op overhead.
        // This works on all backends (CPU, CUDA) via ctxShapeFunctionOverride().
        ctx.shapeFunctionOverride(true);

        // Set arguments and execute
        if (slot.isCustomOp()) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            Nd4j.exec((CustomOp) fn, ctx);
        } else {
            Nd4j.exec((Op) fn, ctx);
        }
    }

    private void executeDynamicSlot(ExecutionPlan plan, OpSlot slot, OpContext ctx,
                                    Map<String, INDArray> placeholderArrays) {
        DifferentialFunction fn = slot.getOp();

        // Dynamic ops still need shape inference at runtime, so do NOT set shapeFunctionOverride
        if (slot.isCustomOp()) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());

            List<DataBuffer> outShapes = fn.calculateOutputShape(ctx);
            if (outShapes != null && !outShapes.isEmpty()) {
                try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (int i = 0; i < outShapes.size() && i < slot.getOutputBufferIndices().length; i++) {
                        long[] shapeInfo = outShapes.get(i).asLong();
                        long[] shape = Shape.shape(shapeInfo);
                        DataType dt = Shape.dataType(shapeInfo);
                        int bufIdx = slot.getOutputBufferIndices()[i];

                        INDArray output = reuseOrAllocateDynamic(bufIdx, dt, shape);
                        if (bufIdx >= 0) {
                            bufferPool[bufIdx] = output;
                        }
                        ctx.setOutputArray(i, output);
                    }
                }
            }

            // Set shapeFunctionOverride AFTER we've set the outputs from shape calc
            ctx.shapeFunctionOverride(true);
            Nd4j.exec((CustomOp) fn, ctx);
        } else {
            List<DataBuffer> outShapes = fn.calculateOutputShape(ctx);
            if (outShapes != null && !outShapes.isEmpty()) {
                long[] shapeInfo = outShapes.get(0).asLong();
                long[] shape = Shape.shape(shapeInfo);
                DataType dt = Shape.dataType(shapeInfo);
                int bufIdx = slot.getOutputBufferIndices().length > 0 ? slot.getOutputBufferIndices()[0] : -1;

                try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    INDArray output = reuseOrAllocateDynamic(bufIdx, dt, shape);
                    if (bufIdx >= 0) {
                        bufferPool[bufIdx] = output;
                    }
                    ctx.setOutputArray(0, output);
                }
            }

            ctx.shapeFunctionOverride(true);
            Nd4j.exec((Op) fn, ctx);
        }
    }

    private INDArray resolveExternalInput(String varName, Map<String, INDArray> placeholderArrays) {
        if (placeholderArrays != null) {
            INDArray arr = placeholderArrays.get(varName);
            if (arr != null) return arr;
        }

        SDVariable var = sd.getVariable(varName);
        if (var != null) {
            if (var.getVariableType() == VariableType.CONSTANT ||
                    var.getVariableType() == VariableType.VARIABLE) {
                return var.getArr();
            }
        }

        log.warn("Could not resolve external input: {}", varName);
        return null;
    }

    private Map<DataType, Long> computePoolSizes(ExecutionPlan plan) {
        EnumMap<DataType, Long> poolBytesByType = new EnumMap<>(DataType.class);
        for (BufferAllocation alloc : plan.getAllocations()) {
            if (alloc.getKind() == BufferAllocKind.PRE_EXISTING ||
                alloc.getKind() == BufferAllocKind.CONSTANT ||
                alloc.getKind() == BufferAllocKind.VARIABLE) {
                continue;
            }

            long[] shape = alloc.getShape();
            if (shape == null || alloc.getSizeInBytes() <= 0 || hasZeroDimension(shape)) {
                continue;
            }

            long end = alloc.getOffsetInPool() + alloc.getSizeInBytes();
            DataType dt = alloc.getDataType();
            Long prev = poolBytesByType.get(dt);
            if (prev == null || end > prev) {
                poolBytesByType.put(dt, end);
            }
        }
        return poolBytesByType;
    }

    private INDArray reuseOrAllocateDynamic(int bufferIndex, DataType dataType, long[] shape) {
        if (shape == null) {
            return null;
        }
        if (hasZeroDimension(shape)) {
            return Nd4j.emptyWithShape(shape, dataType);
        }
        if (bufferIndex >= 0) {
            INDArray existing = bufferPool[bufferIndex];
            if (existing != null && !existing.wasClosed() &&
                existing.dataType() == dataType && Arrays.equals(existing.shape(), shape)) {
                return existing;
            }
        }
        return Nd4j.create(dataType, shape);
    }

    private boolean hasZeroDimension(long[] shape) {
        for (long dim : shape) {
            if (dim == 0) {
                return true;
            }
        }
        return false;
    }

    private void cleanup() {
        if (opContextPool != null) {
            for (OpContext ctx : opContextPool) {
                if (ctx != null) {
                    try {
                        ctx.close();
                    } catch (Exception e) {
                        // ignore
                    }
                }
            }
            opContextPool = null;
        }

        if (dataTypePools != null) {
            for (DataBuffer buffer : dataTypePools.values()) {
                if (buffer != null) {
                    try {
                        buffer.close();
                    } catch (Exception e) {
                        // ignore cleanup errors
                    }
                }
            }
            dataTypePools = null;
        }

        bufferPool = null;
        initialized = false;
        currentPlan = null;
    }

    @Override
    public void close() {
        cleanup();
    }
}
