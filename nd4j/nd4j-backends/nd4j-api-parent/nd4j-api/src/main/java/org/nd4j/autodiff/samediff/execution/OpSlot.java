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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * A compiled operation slot inside an {@link ExecutionPlan}.
 *
 * <p>Each OpSlot captures everything needed to execute one operation without
 * any runtime shape inference or dynamic allocation:</p>
 * <ul>
 *   <li>The operation instance and its name</li>
 *   <li>Indices into the plan's buffer pool for inputs and outputs</li>
 *   <li>Pre-computed output shapes and data types</li>
 *   <li>All integer, float, boolean, and datatype arguments</li>
 * </ul>
 *
 * <p>An index of {@code -1} in {@code inputBufferIndices} means the input
 * comes from a placeholder or constant (resolved at execution time from the
 * feed map or the SameDiff instance).</p>
 */
@Data
@AllArgsConstructor
@Builder
public class OpSlot {
    /** SameDiffOp name (unique within the graph). */
    private final String opName;

    /** The operation instance. */
    private final DifferentialFunction op;

    /** Whether this is a CustomOp (true) or legacy Op (false). */
    private final boolean customOp;

    /**
     * Buffer pool indices for each input.
     * {@code -1} means the input is a placeholder/constant/variable resolved externally.
     */
    private final int[] inputBufferIndices;

    /** Variable names for each input (parallel to inputBufferIndices). */
    private final String[] inputVarNames;

    /** Buffer pool indices for each output. */
    private final int[] outputBufferIndices;

    /** Variable names for each output (parallel to outputBufferIndices). */
    private final String[] outputVarNames;

    /** Pre-computed output shapes (one per output). */
    private final long[][] outputShapes;

    /** Pre-computed output data types (one per output). */
    private final DataType[] outputDataTypes;

    /** Integer arguments for the operation. */
    private final long[] iArgs;

    /** Float arguments for the operation. */
    private final double[] tArgs;

    /** Boolean arguments for the operation. */
    private final boolean[] bArgs;

    /** DataType arguments for the operation. */
    private final DataType[] dArgs;

    /**
     * If true, this op requires dynamic shape inference at runtime
     * (e.g., Reshape with a data-dependent shape input).
     * The executor will fall back to calculateOutputShape for this slot.
     */
    private final boolean requiresDynamicShapeInference;
}
