/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.api

enum class DataType {
    NDARRAY,        // Any NDArray type (input only) - INDArray or SDVariable
    FLOATING_POINT, // Any floating point data type
    INT, // integer data type
    LONG, //long, signed int64 datatype
    NUMERIC, // any floating point or integer data type
    BOOL, // boolean data type
    STRING, //String value
    // Arg only
    DATA_TYPE, // tensor data type
    CONDITION, // A condition
    LOSS_REDUCE, // Loss reduction mode
    ENUM; // defines an enum along with possibleValues property in Arg

    fun isTensorDataType() = setOf(NDARRAY, FLOATING_POINT, INT, LONG, NUMERIC, BOOL).contains(this)
}