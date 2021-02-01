/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2019 Konduit, KK.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.experimental;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Arrays;

public class NDValidation {

    private NDValidation() {
    }

    /**
     * Validate that the operation is being applied on a numerical INDArray (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to perform operation on
     */
    protected static void validateNumerical(String opName, INDArray v, String inputName) {
        if (v == null)
            return;
        if (v.dataType() == DataType.BOOL || v.dataType() == DataType.UTF8)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to input \"" + inputName + "\" with non-numerical data type " + v.dataType());
    }


    /**
     * Validate that the operation is being applied on an integer type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateInteger(String opName, INDArray v, String inputName) {
        if (v == null)
            return;
        if (!v.dataType().isIntType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to input \"" + inputName + "\" with non-integer data type " + v.dataType());
    }


    /**
     * Validate that the operation is being applied on an floating point type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateFloatingPoint(String opName, INDArray v, String inputName) {
        if (v == null)
            return;
        if (!v.dataType().isFPType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to input \"" + inputName + "\" with non-floating point data type " + v.dataType());
    }


    /**
     * Validate that the operation is being applied on a boolean type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateBool(String opName, INDArray v, String inputName) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to inputName \"" + inputName + "\" with non-boolean point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on array with the exact same datatypes
     *
     * @param opName        Operation name to print in the exception
     * @param vars          Variable to perform operation on
     */
    protected static void validateSameType(String opName, INDArray... vars) {
        if (isSameType(vars)){
            return;
        }
        else{
            DataType[] dtypes = new DataType[vars.length];
            for (int j = 0; j < vars.length; j++) {
                dtypes[j] = vars[j].dataType();
            }
            throw new IllegalStateException("Cannot perform operation \"" + opName + "\" to inputs with different datatypes:" +
                    " datatypes " + Arrays.toString(dtypes));
        }
    }

    /**
     * Is the operation being applied on array with the exact same datatypes?
     *
     * @param vars          Variable to perform operation on
     */
    protected static boolean isSameType(INDArray... vars) {
        if (vars.length > 1) {
            DataType first = vars[0].dataType();
            for (int i = 1; i < vars.length; i++) {
                if (first != vars[i].dataType()) {
                    return false;
                }
            }
        }
        return true;
    }
}
