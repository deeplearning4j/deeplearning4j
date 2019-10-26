/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Arrays;

public class SDValidation {

    private SDValidation() {
    }

    /**
     * Validate that the operation is being applied on a numerical SDVariable (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to perform operation on
     */
    protected static void validateNumerical(String opName, SDVariable v) {
        if (v == null)
            return;
        if (v.dataType() == DataType.BOOL || v.dataType() == DataType.UTF8)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to variable \"" + v.name() + "\" with non-numerical data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a numerical SDVariable (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateNumerical(String opName, String inputName, SDVariable v) {
        if (v == null)
            return;
        if (v.dataType() == DataType.BOOL || v.dataType() == DataType.UTF8)
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an numerical type type; got variable \"" +
                    v.name() + "\" with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on numerical SDVariables (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v1     Variable to validate datatype for (input to operation)
     * @param v2     Variable to validate datatype for (input to operation)
     */
    protected static void validateNumerical(String opName, SDVariable v1, SDVariable v2) {
        if (v1.dataType() == DataType.BOOL || v1.dataType() == DataType.UTF8 || v2.dataType() == DataType.BOOL || v2.dataType() == DataType.UTF8)
            throw new IllegalStateException("Cannot perform operation \"" + opName + "\" on variables  \"" + v1.name() + "\" and \"" +
                    v2.name() + "\" if one or both variables are non-numerical: " + v1.dataType() + " and " + v2.dataType());
    }

    /**
     * Validate that the operation is being applied on an integer type SDVariable
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateInteger(String opName, SDVariable v) {
        if (v == null)
            return;
        if (!v.dataType().isIntType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to variable \"" + v.name() + "\" with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on an integer type SDVariable
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    protected static void validateInteger(String opName, String inputName, SDVariable v) {
        if (v == null)
            return;
        if (!v.dataType().isIntType())
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an integer type; got variable \"" +
                    v.name() + "\" with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on an floating point type SDVariable
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateFloatingPoint(String opName, SDVariable v) {
        if (v == null)
            return;
        if (!v.dataType().isFPType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to variable \"" + v.name() + "\" with non-floating point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a floating point type SDVariable
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    protected static void validateFloatingPoint(String opName, String inputName, SDVariable v) {
        if (v == null)
            return;
        if (!v.dataType().isFPType())
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an floating point type; got variable \"" +
                    v.name() + "\" with non-floating point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a boolean type SDVariable
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    protected static void validateBool(String opName, SDVariable v) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to variable \"" + v.name() + "\" with non-boolean point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a boolean type SDVariable
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    protected static void validateBool(String opName, String inputName, SDVariable v) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an boolean variable; got variable \"" +
                    v.name() + "\" with non-boolean data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on boolean SDVariables
     *
     * @param opName Operation name to print in the exception
     * @param v1     Variable to validate datatype for (input to operation)
     * @param v2     Variable to validate datatype for (input to operation)
     */
    protected static void validateBool(String opName, SDVariable v1, SDVariable v2) {
        if (v1.dataType() != DataType.BOOL || v2.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot perform operation \"" + opName + "\" on variables  \"" + v1.name() + "\" and \"" +
                    v2.name() + "\" if one or both variables are non-boolean: " + v1.dataType() + " and " + v2.dataType());
    }

    /**
     * Validate that the operation is being applied on array with the exact same datatypes (which may optionally be
     * restricted to numerical SDVariables only (not boolean or utf8))
     *
     * @param opName        Operation name to print in the exception
     * @param numericalOnly If true, the variables must all be the same type, and must be numerical (not boolean/utf8)
     * @param vars          Variable to perform operation on
     */
    protected static void validateSameType(String opName, boolean numericalOnly, SDVariable... vars) {
        if (vars.length == 0)
            return;
        if (vars.length == 1) {
            if (numericalOnly) {
                validateNumerical(opName, vars[0]);
            }
        } else {
            DataType first = vars[0].dataType();
            if (numericalOnly)
                validateNumerical(opName, vars[0]);
            for (int i = 1; i < vars.length; i++) {
                if (first != vars[i].dataType()) {
                    String[] names = new String[vars.length];
                    DataType[] dtypes = new DataType[vars.length];
                    for (int j = 0; j < vars.length; j++) {
                        names[j] = vars[j].name();
                        dtypes[j] = vars[j].dataType();
                    }
                    throw new IllegalStateException("Cannot perform operation \"" + opName + "\" to variables with different datatypes:" +
                            " Variable names " + Arrays.toString(names) + ", datatypes " + Arrays.toString(dtypes));
                }
            }
        }
    }

}
