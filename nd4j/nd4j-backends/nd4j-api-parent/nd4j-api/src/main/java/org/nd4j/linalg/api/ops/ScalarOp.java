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

public interface ScalarOp extends Op {

    /**The normal scalar
     *@return the scalar
     */
    INDArray scalar();

    /**
     * This method allows to set scalar
     * @param scalar
     */
    void setScalar(Number scalar);

    void setScalar(INDArray scalar);

    /**
     * This method returns target dimensions for this op
     * @return
     */
    INDArray dimensions();

    long[] getDimension();

    void setDimension(long... dimension);

    boolean validateDataTypes(boolean experimentalMode);

    /** Validate the operands actually supplied to execution, without rebinding the op's arrays. */
    default boolean validateDataTypes(OpContext context, boolean experimentalMode) {
        INDArray x = context == null ? x() : context.getInputArray(0);
        INDArray y = context == null ? y() : context.getInputArray(1);
        INDArray z = context == null ? z() : context.getOutputArray(0);
        if (x == null || z == null) {
            throw new IllegalArgumentException("Scalar execution requires X and Z arrays for " + opName());
        }
        if (x.length() != z.length()) {
            throw new IllegalArgumentException("Scalar X and Z lengths must match for " + opName());
        }
        if (getOpType() == Type.SCALAR_BOOL) {
            if (!z.isB()) {
                throw new IllegalArgumentException("Scalar boolean output must have BOOL type for " + opName());
            }
        } else {
            if ((x.isR() || (y != null && y.isR())) && !z.isR()) {
                throw new IllegalArgumentException("Scalar output must be floating point when an input is floating point for " + opName());
            }
            if (!experimentalMode && y != null && x.dataType() != y.dataType() && !y.isB()) {
                throw new IllegalArgumentException("Scalar X and Y must have the same data type for " + opName());
            }
        }
        return true;
    }

    Type getOpType();
}
