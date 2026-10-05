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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;

public interface BroadcastOp extends Op {

    /**
     * Dimension to do the vector op along. Along dimension 1 for row vector ops,  along 0 for column vector ops
     */
    long[] getDimension();

    /** Resolve any inferred axes against the input shapes used by this execution. */
    default long[] getDimension(OpContext context) {
        return getDimension();
    }

    /** Set the dimension for the vector op. */
    void setDimension(long... dimension);

    boolean validateDataTypes(boolean experimentalOp);

    /** Validate current context operands before TAD construction or native dispatch. */
    default boolean validateDataTypes(OpContext context, boolean experimentalMode) {
        INDArray x = context == null ? x() : context.getInputArray(0);
        INDArray y = context == null ? y() : context.getInputArray(1);
        INDArray z = context == null ? z() : context.getOutputArray(0);
        Broadcast.validateBroadcastDims(x, y, z, getDimension(context));
        if (getOpType() == Type.BROADCAST_BOOL) {
            if (x.dataType() != y.dataType() || !z.isB()) {
                throw new IllegalArgumentException("Boolean broadcast requires matching X/Y types and BOOL Z for " + opName());
            }
        } else {
            if (z.dataType() != x.dataType() && z.dataType() != y.dataType()) {
                throw new IllegalArgumentException("Broadcast Z type must be either X or Y type for " + opName());
            }
            if (!experimentalMode && x.dataType() != y.dataType() && !y.isB()) {
                throw new IllegalArgumentException("Broadcast X and Y must have the same data type for " + opName());
            }
            if (opNum() != 1 && (x.isR() || y.isR()) && !z.isR()) {
                throw new IllegalArgumentException("Broadcast output must be floating point when an input is floating point for " + opName());
            }
        }
        return true;
    }

    Type getOpType();

    INDArray dimensions();
}
