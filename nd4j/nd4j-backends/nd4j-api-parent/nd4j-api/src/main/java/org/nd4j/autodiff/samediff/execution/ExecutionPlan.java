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

import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.*;

/**
 * An immutable, compiled execution plan for a SameDiff graph.
 *
 * <p>Analogous to ONNX Runtime's {@code SequentialExecutionPlan} combined with its
 * {@code MemoryPatternGroup}, an ExecutionPlan captures:</p>
 * <ol>
 *   <li>A linearized, topologically-sorted list of {@link OpSlot}s to execute</li>
 *   <li>A complete buffer layout with pre-computed offsets and reuse assignments</li>
 *   <li>Total memory required so a single workspace allocation can back all intermediates</li>
 * </ol>
 *
 * <p>The plan is compiled by {@link ExecutionPlanCompiler} from a {@link ForwardExecutionDAG}
 * and a set of concrete placeholder shapes.  It is cached by {@link ExecutionPlanCache}
 * keyed on (DAG topology, placeholder shape signature) so that subsequent calls with
 * the same shapes reuse the plan without recompilation.</p>
 *
 * <p>At execution time, {@link PlanExecutor} allocates one contiguous workspace of
 * {@link #totalHostBytes} bytes, places pre-created INDArray shells at the computed
 * offsets, wires them into pre-allocated OpContexts, and executes each slot in order
 * with no shape inference and no per-op allocation.</p>
 *
 * @see ExecutionPlanCompiler
 * @see PlanExecutor
 * @see ExecutionPlanCache
 */
@Data
public class ExecutionPlan {

    /** Shape signature used to match this plan to a set of placeholder shapes. */
    private final long shapeSignature;

    /** Ordered list of operations to execute. */
    private final List<OpSlot> slots;

    /** Total host-side bytes needed for all ALLOCATE/OUTPUT/REUSE buffers. */
    private final long totalHostBytes;

    /** Buffer allocation descriptors, keyed by variable name. */
    private final Map<String, BufferAllocation> bufferMap;

    /** All buffer allocations in index order (for direct array-based access). */
    private final List<BufferAllocation> allocations;

    /** The set of output variable names this plan was compiled for. */
    private final Set<String> requestedOutputs;

    /** Number of buffer slots (max index + 1). */
    private final int bufferPoolSize;

    /** Whether any slot requires dynamic shape inference (fallback path). */
    private final boolean hasDynamicShapeOps;

    /**
     * Alignment in bytes for buffer offsets within the workspace pool.
     * 64 bytes aligns to cache lines and AVX-512 vectors.
     */
    public static final int ALIGNMENT = 64;

    public ExecutionPlan(long shapeSignature,
                         List<OpSlot> slots,
                         long totalHostBytes,
                         Map<String, BufferAllocation> bufferMap,
                         List<BufferAllocation> allocations,
                         Set<String> requestedOutputs) {
        this.shapeSignature = shapeSignature;
        this.slots = Collections.unmodifiableList(slots);
        this.totalHostBytes = totalHostBytes;
        this.bufferMap = Collections.unmodifiableMap(bufferMap);
        this.allocations = Collections.unmodifiableList(allocations);
        this.requestedOutputs = Collections.unmodifiableSet(requestedOutputs);
        this.bufferPoolSize = allocations.size();

        boolean dynamic = false;
        for (OpSlot slot : slots) {
            if (slot.isRequiresDynamicShapeInference()) {
                dynamic = true;
                break;
            }
        }
        this.hasDynamicShapeOps = dynamic;
    }

    /**
     * Align a byte size up to the plan's alignment boundary.
     */
    public static long align(long bytes) {
        return (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    /**
     * Get a human-readable summary of this plan.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("ExecutionPlan {\n");
        sb.append("  ops: ").append(slots.size()).append("\n");
        sb.append("  buffers: ").append(bufferPoolSize).append("\n");
        sb.append("  totalHostBytes: ").append(totalHostBytes)
          .append(" (").append(totalHostBytes / (1024 * 1024)).append(" MB)\n");
        sb.append("  outputs: ").append(requestedOutputs).append("\n");
        sb.append("  hasDynamicShapeOps: ").append(hasDynamicShapeOps).append("\n");

        // Count by alloc kind
        Map<BufferAllocKind, Integer> kindCounts = new EnumMap<>(BufferAllocKind.class);
        for (BufferAllocation alloc : allocations) {
            kindCounts.merge(alloc.getKind(), 1, Integer::sum);
        }
        sb.append("  allocKinds: ").append(kindCounts).append("\n");

        // Buffer reuse stats
        long totalAllocated = 0;
        long totalLogical = 0;
        for (BufferAllocation alloc : allocations) {
            totalLogical += alloc.getSizeInBytes();
            if (alloc.getKind() == BufferAllocKind.ALLOCATE || alloc.getKind() == BufferAllocKind.OUTPUT) {
                totalAllocated += alloc.getSizeInBytes();
            }
        }
        if (totalLogical > 0) {
            double savings = 1.0 - ((double) totalAllocated / totalLogical);
            sb.append("  memorySavingsFromReuse: ").append(String.format("%.1f%%", savings * 100)).append("\n");
        }

        sb.append("}");
        return sb.toString();
    }
}
