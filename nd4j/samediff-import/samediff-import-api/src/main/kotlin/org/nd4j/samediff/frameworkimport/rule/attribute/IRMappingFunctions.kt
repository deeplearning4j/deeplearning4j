/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.samediff.frameworkimport.rule.attribute

import org.nd4j.common.base.Preconditions
import org.nd4j.ir.TensorNamespace


//source:  https://github.com/eclipse/deeplearning4j/blob/63fa3c2ef3c4e5e33cdb99bb4804997b40ad4590/libnd4j/include/array/DataType.h#L39

/**
 * Referenced from https://github.com/eclipse/deeplearning4j/blob/63fa3c2ef3c4e5e33cdb99bb4804997b40ad4590/libnd4j/include/array/DataType.h
 * Used to convert ints to data types. These ints are used in certain ops where an int is taken in for expressing a data type.
 */
fun dataTypeFromInt(inputInt: Int): TensorNamespace.DataType {
    when(inputInt) {
        1 -> return TensorNamespace.DataType.BOOL
        2 -> return TensorNamespace.DataType.BFLOAT16
        3 -> return TensorNamespace.DataType.FLOAT16
        4 -> return TensorNamespace.DataType.FLOAT
        5 -> return TensorNamespace.DataType.FLOAT
        6 -> return TensorNamespace.DataType.DOUBLE
        7 -> return TensorNamespace.DataType.INT8
        8 -> return TensorNamespace.DataType.INT16
        9 -> return TensorNamespace.DataType.INT32
        10 -> return TensorNamespace.DataType.INT64
        11 -> return TensorNamespace.DataType.UINT8
        12 -> return TensorNamespace.DataType.UINT16
        13 -> return TensorNamespace.DataType.UINT32
        14 -> return TensorNamespace.DataType.UINT64
        17 -> return TensorNamespace.DataType.BFLOAT16
        50,51,52 -> return TensorNamespace.DataType.STRING
        else -> return TensorNamespace.DataType.UNDEFINED

    }
}

/**
 * Reverse of [dataTypeFromInt]
 * converts an int argument to a [TensorNamespace.DataType]
 */
fun intArgFromDataType(inputDataType: TensorNamespace.DataType): Int {
   Preconditions.checkNotNull(inputDataType,"Data type must not be null!")
    when(inputDataType) {
        TensorNamespace.DataType.BOOL -> return 1
        TensorNamespace.DataType.BFLOAT16 -> return 17
        TensorNamespace.DataType.FLOAT16 -> return 3
        TensorNamespace.DataType.FLOAT16 -> return 4
        TensorNamespace.DataType.FLOAT -> return 5
        TensorNamespace.DataType.DOUBLE -> return 6
        TensorNamespace.DataType.INT8 -> return 7
        TensorNamespace.DataType.INT16 -> return 8
        TensorNamespace.DataType.INT32 -> return 9
        TensorNamespace.DataType.INT64 -> return 10
        TensorNamespace.DataType.UINT8 -> return 11
        TensorNamespace.DataType.UINT16 -> return 12
        TensorNamespace.DataType.UINT32 -> return 13
        TensorNamespace.DataType.UINT64 -> return 14
        TensorNamespace.DataType.BFLOAT16 -> return 17
        TensorNamespace.DataType.STRING -> return 50
        TensorNamespace.DataType.UNDEFINED,TensorNamespace.DataType.UNRECOGNIZED -> return 100
        else ->  throw IllegalArgumentException("No data type found for $inputDataType")

    }

    return -1
}


