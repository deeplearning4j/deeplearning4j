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

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Callback interface invoked after each slot execution in {@link DynamicShapePlanExecutor}.
 * Implementations that need to retain array data must call {@code output.dup()} since
 * the output array may be reused or closed on subsequent steps.
 */
@FunctionalInterface
public interface SlotOutputInterceptor {

    /**
     * Called after a slot produces an output array.
     *
     * @param stepIdx       the slot index within the plan (0-based)
     * @param opName        the operation name (e.g., "matmul", "add")
     * @param outputVarName the SameDiff variable name for this output
     * @param output        the output array (may be null if the op produced no output for this slot)
     * @param outputSlotIdx the index in the flat outputSlots[] array
     */
    void onSlotOutput(int stepIdx, String opName, String outputVarName, INDArray output, int outputSlotIdx);
}
