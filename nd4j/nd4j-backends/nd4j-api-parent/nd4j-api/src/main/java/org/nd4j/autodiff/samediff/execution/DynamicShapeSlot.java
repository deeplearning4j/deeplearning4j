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

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Per-op descriptor with pre-compiled wiring for the DynamicShapePlan.
 *
 * <p>Each slot replaces string-keyed HashMap lookups with flat array-indexed access.
 * Input wiring is encoded as integer indices into the flat output array or external
 * input arrays (constants, variables, placeholders).</p>
 *
 * <p>Index conventions for {@code inputSourceIndices}:</p>
 * <ul>
 *   <li>{@code >= 0}: index into the flat {@code INDArray[] slots} output array</li>
 *   <li>{@code < 0}: index into external arrays: {@code -(index + 1)} into the external input array</li>
 * </ul>
 */
@Data
@Builder
public class DynamicShapeSlot {

    /** Source type constants for inputSourceTypes. */
    public static final byte SOURCE_CONSTANT = 0;
    public static final byte SOURCE_VARIABLE = 1;
    public static final byte SOURCE_PLACEHOLDER = 2;
    public static final byte SOURCE_OP_OUTPUT = 3;

    /** The operation name (for diagnostics and fallback). */
    private final String opName;

    /** The operation instance. */
    private final DifferentialFunction op;

    /** Whether this is a CustomOp (true) or legacy Op (false). */
    private final boolean customOp;

    /**
     * Input source indices. For each input:
     * - {@code >= 0}: flat output slot index (from a prior op's output)
     * - {@code < 0}: external input, decoded as {@code -(index + 1)} into the external inputs array
     */
    private final int[] inputSourceIndices;

    /** Source type per input: SOURCE_CONSTANT, SOURCE_VARIABLE, SOURCE_PLACEHOLDER, or SOURCE_OP_OUTPUT. */
    private final byte[] inputSourceTypes;

    /** Variable names for each input (used for external resolution). */
    private final String[] inputVarNames;

    /** Indices into the flat output array for each output of this op. */
    private final int[] outputSlotIndices;

    /** Variable names for each output (for result collection). */
    private final String[] outputVarNames;

    /** Frozen integer arguments. */
    private final long[] iArgs;

    /** Frozen float arguments. */
    private final double[] tArgs;

    /** Frozen boolean arguments. */
    private final boolean[] bArgs;

    /** Frozen datatype arguments. */
    private final DataType[] dArgs;

    /** Per-slot cached shape key for fast comparison. 0 = not yet computed. */
    private long cachedShapeKey;

    /** Per-slot cached output shapes. null = not yet computed. */
    private List<DataBuffer> cachedOutputShapes;

    /** Whether this op has INT/LONG inputs that need device-to-host sync before shape calc. */
    private final boolean needsIntLongSync;

    /** Whether this op has data-dependent output shapes (Where, unique). */
    private final boolean isDataDependent;

    /** Whether this op requires dynamic shape inference (reshape with computed shape input, etc). */
    private final boolean requiresDynamicShapeInference;

    /**
     * Whether outputs of this op need to be zeroed before execution.
     * Ops that fully write every output element (matmul, add, softmax, etc.) can safely
     * skip zeroing of reused buffers. Ops that partially write or have data-dependent
     * output sizes (scatter, pad, Where) need zeroed buffers to avoid stale data corruption.
     */
    private final boolean needsZeroedOutput;

    /**
     * Whether this op's output shape depends on input tensor VALUES (not just shapes).
     * Examples: reshape (shape from input tensor), strided_slice (begin/end/strides values),
     * tile (repeat counts), pad (padding values), fill (target shape values).
     * When false, the shape cache key only uses input shapes + dtypes, skipping expensive
     * INT/LONG value reads (which trigger CUDA D2H sync) and eliminating false cache misses
     * from value changes that don't affect output shapes.
     */
    private final boolean outputShapeDependsOnInputValues;

    /** Index of this slot in the plan (for diagnostics). */
    private final int stepIndex;

    /** Pre-computed hash of opName, avoiding String.hashCode() per step in shape key computation. */
    private final long opNameHash;

    /** Pre-allocated input array buffer to avoid per-step allocation. */
    private final transient INDArray[] inputArraysBuffer;

    /**
     * Update the per-slot shape cache.
     */
    public void updateShapeCache(long shapeKey, List<DataBuffer> outputShapes) {
        this.cachedShapeKey = shapeKey;
        this.cachedOutputShapes = outputShapes;
    }

    /**
     * Check if the cached shapes match the given key.
     */
    public boolean isShapeCacheValid(long shapeKey) {
        return cachedShapeKey != 0 && cachedShapeKey == shapeKey;
    }
}
