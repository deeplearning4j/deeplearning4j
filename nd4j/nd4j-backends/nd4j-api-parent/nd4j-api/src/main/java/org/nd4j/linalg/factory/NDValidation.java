/* *****************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public class NDValidation {

    private NDValidation() {
    }

    /**
     * Validate that the operation is being applied on a numerical INDArray (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc) don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to perform operation on
     */
    public static void validateNumerical(String opName, INDArray v) {
        if (v == null)
            return;
        if (v.dataType() == DataType.BOOL || v.dataType() == DataType.UTF8)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to array with non-numerical data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on numerical INDArrays (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc) don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to perform operation on
     */
    public static void validateNumerical(String opName, INDArray[] v) {
        if (v == null)
            return;
        for (int i = 0; i < v.length; i++) {
            if (v[i].dataType() == DataType.BOOL || v[i].dataType() == DataType.UTF8)
                throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to input array " + i + " with non-numerical data type " + v[i].dataType());
        }
    }

    /**
     * Validate that the operation is being applied on a numerical INDArray (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc) don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    public static void validateNumerical(String opName, String inputName, INDArray v) {
        if (v == null)
            return;
        if (v.dataType() == DataType.BOOL || v.dataType() == DataType.UTF8)
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an numerical type type;" +
                    " got array with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on numerical INDArrays (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc) don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to perform operation on
     */
    public static void validateNumerical(String opName, String inputName, INDArray[] v) {
        if (v == null)
            return;
        for (int i = 0; i < v.length; i++) {
            if (v[i].dataType() == DataType.BOOL || v[i].dataType() == DataType.UTF8)
                throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to input \"" + inputName + "\" array " + i + " with non-numerical data type " + v[i].dataType());
        }
    }

    /**
     * Validate that the operation is being applied on numerical INDArrays (not boolean or utf8).
     * Some operations (such as sum, norm2, add(Number) etc don't make sense when applied to boolean/utf8 arrays
     *
     * @param opName Operation name to print in the exception
     * @param v1     Variable to validate datatype for (input to operation)
     * @param v2     Variable to validate datatype for (input to operation)
     */
    public static void validateNumerical(String opName, INDArray v1, INDArray v2) {
        if (v1.dataType() == DataType.BOOL || v1.dataType() == DataType.UTF8 || v2.dataType() == DataType.BOOL || v2.dataType() == DataType.UTF8)
            throw new IllegalStateException("Cannot perform operation \"" + opName + "\" on arrays if one or both variables" +
                    " are non-numerical: got " + v1.dataType() + " and " + v2.dataType());
    }

    /**
     * Validate that the operation is being applied on an integer type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    public static void validateInteger(String opName, INDArray v) {
        if (v == null)
            return;
        if (!v.dataType().isIntType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to array with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on an integer type INDArray
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    public static void validateInteger(String opName, String inputName, INDArray v) {
        if (v == null)
            return;
        if (!v.dataType().isIntType())
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName + "\" must be an integer" +
                    " type; got array with non-integer data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on an floating point type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    public static void validateFloatingPoint(String opName, INDArray v) {
        if (v == null)
            return;
        if (!v.dataType().isFPType())
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to array with non-floating point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a floating point type INDArray
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    public static void validateFloatingPoint(String opName, String inputName, INDArray v) {
        if (v == null)
            return;
        if (!v.dataType().isFPType())
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName +
                    "\" must be an floating point type; got array with non-floating point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a boolean type INDArray
     *
     * @param opName Operation name to print in the exception
     * @param v      Variable to validate datatype for (input to operation)
     */
    public static void validateBool(String opName, INDArray v) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to array with non-boolean point data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on a boolean type INDArray
     *
     * @param opName    Operation name to print in the exception
     * @param inputName Name of the input to the op to validate
     * @param v         Variable to validate datatype for (input to operation)
     */
    public static void validateBool(String opName, String inputName, INDArray v) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Input \"" + inputName + "\" for operation \"" + opName +
                    "\" must be an boolean variable; got array with non-boolean data type " + v.dataType());
    }

    /**
     * Validate that the operation is being applied on boolean INDArrays
     *
     * @param opName Operation name to print in the exception
     * @param v1     Variable to validate datatype for (input to operation)
     * @param v2     Variable to validate datatype for (input to operation)
     */
    public static void validateBool(String opName, INDArray v1, INDArray v2) {
        if (v1.dataType() != DataType.BOOL || v2.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot perform operation \"" + opName + "\" on array if one or both variables are non-boolean: "
                    + v1.dataType() + " and " + v2.dataType());
    }

    /**
     * Validate that the operation is being applied on array with the exact same datatypes (which may optionally be
     * restricted to numerical INDArrays only (not boolean or utf8))
     *
     * @param opName        Operation name to print in the exception
     * @param numericalOnly If true, the variables must all be the same type, and must be numerical (not boolean/utf8)
     * @param vars          Variable to perform operation on
     */
    public static void validateSameType(String opName, boolean numericalOnly, INDArray... vars) {
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
                    DataType[] dtypes = new DataType[vars.length];
                    for (int j = 0; j < vars.length; j++) {
                        dtypes[j] = vars[j].dataType();
                    }
                    throw new IllegalStateException("Cannot perform operation \"" + opName + "\" to arrays with different datatypes:" +
                            " Got arrays with datatypes " + Arrays.toString(dtypes));
                }
            }
        }
    }

    public static boolean isSameType(INDArray x, INDArray y) {
        return x.dataType() == y.dataType();
    }
}
