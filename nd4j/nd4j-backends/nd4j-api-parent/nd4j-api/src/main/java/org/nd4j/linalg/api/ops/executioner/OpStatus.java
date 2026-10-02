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

package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.exception.ND4JIllegalStateException;

public enum OpStatus {
    ND4J_STATUS_OK,
    ND4J_STATUS_BAD_INPUT,
    ND4J_STATUS_BAD_SHAPE,
    ND4J_STATUS_BAD_RANK,
    ND4J_STATUS_BAD_PARAMS,
    ND4J_STATUS_BAD_OUTPUT,
    ND4J_STATUS_BAD_RNG,
    ND4J_STATUS_BAD_EPSILON,
    ND4J_STATUS_BAD_GRADIENTS,
    ND4J_STATUS_BAD_BIAS,

    ND4J_STATUS_BAD_GRAPH,
    ND4J_STATUS_BAD_LENGTH,
    ND4J_STATUS_BAD_DIMENSIONS,
    ND4J_STATUS_BAD_ORDER,
    ND4J_STATUS_BAD_ARGUMENTS,
    ND4J_STATUS_VALIDATION,
    ND4J_STATUS_DOUBLE_WRITE,
    ND4J_STATUS_DOUBLE_READ,
    ND4J_STATUS_KERNEL_FAILURE;

    /**
     * The name of a native status code (sd::Status); NATIVE_EXCEPTION for -1, which native entry
     * points such as the DSP plan entries return when they caught an exception; UNKNOWN(code) for
     * any other code this enum lacks.
     */
    public static String nameOf(int val) {
        switch (val) {
            case -1:
                return "NATIVE_EXCEPTION";
            case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9:
            case 20: case 30: case 31: case 32: case 33: case 34: case 40: case 45: case 50:
                return byNumber(val).name();
            default:
                return "UNKNOWN(" + val + ")";
        }
    }

    public static OpStatus byNumber(int val) {
        switch (val) {
            case 0:
                return ND4J_STATUS_OK;
            case 1:
                return ND4J_STATUS_BAD_INPUT;
            case 2:
                return ND4J_STATUS_BAD_SHAPE;
            case 3:
                return ND4J_STATUS_BAD_RANK;
            case 4:
                return ND4J_STATUS_BAD_PARAMS;
            case 5:
                return ND4J_STATUS_BAD_OUTPUT;
            case 6:
                return ND4J_STATUS_BAD_RNG;
            case 7:
                return ND4J_STATUS_BAD_EPSILON;
            case 8:
                return ND4J_STATUS_BAD_GRADIENTS;
            case 9:
                return ND4J_STATUS_BAD_BIAS;
            case 20:
                return ND4J_STATUS_VALIDATION;
            case 30:
                return ND4J_STATUS_BAD_GRAPH;
            case 31:
                return ND4J_STATUS_BAD_LENGTH;
            case 32:
                return ND4J_STATUS_BAD_DIMENSIONS;
            case 33:
                return ND4J_STATUS_BAD_ORDER;
            case 34:
                return ND4J_STATUS_BAD_ARGUMENTS;
            case 40:
                return ND4J_STATUS_DOUBLE_WRITE;
            case 45:
                return ND4J_STATUS_DOUBLE_READ;
            case 50:
                return ND4J_STATUS_KERNEL_FAILURE;
            default:
                throw new ND4JIllegalStateException("Unknown status given: " + val);
        }
    }
}
