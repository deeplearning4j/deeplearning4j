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
package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.nd4j.ir.TensorNamespace
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRDataTypeValue
import org.tensorflow.framework.DataType

class TensorflowIRDataType(inputDataType: DataType): IRDataType<DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: DataType): IRDataTypeValue {
        when(input) {
            DataType.DT_BOOL, DataType.DT_BOOL_REF -> return IRDataTypeValue.DT_BOOL
            DataType.DT_BFLOAT16, DataType.DT_BFLOAT16_REF -> return IRDataTypeValue.DT_BFLOAT16
            DataType.DT_COMPLEX128, DataType.DT_COMPLEX128_REF -> return IRDataTypeValue.DT_COMPLEX128
            DataType.DT_COMPLEX64, DataType.DT_COMPLEX64_REF -> return IRDataTypeValue.DT_COMPLEX64
            DataType.DT_DOUBLE, DataType.DT_DOUBLE_REF -> return IRDataTypeValue.DT_DOUBLE
            DataType.DT_FLOAT, DataType.DT_FLOAT_REF -> return IRDataTypeValue.DT_FLOAT
            DataType.DT_HALF, DataType.DT_HALF_REF -> return IRDataTypeValue.DT_HALF
            DataType.DT_INT8, DataType.DT_INT8_REF -> return IRDataTypeValue.DT_INT8
            DataType.DT_UINT8, DataType.DT_UINT8_REF -> return IRDataTypeValue.DT_UINT8
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return IRDataTypeValue.DT_UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> return IRDataTypeValue.DT_UINT32
            DataType.DT_INT16, DataType.DT_INT16_REF -> return IRDataTypeValue.DT_INT16
            DataType.DT_INT32, DataType.DT_INT32_REF -> return IRDataTypeValue.DT_INT32
            DataType.DT_INT64, DataType.DT_INT64_REF -> return IRDataTypeValue.DT_INT64
            DataType.DT_QINT8, DataType.DT_QINT8_REF -> return IRDataTypeValue.DT_QINT8
            DataType.DT_QINT16, DataType.DT_QINT16_REF -> return IRDataTypeValue.DT_QINT16
            DataType.DT_QINT32, DataType.DT_QINT32_REF -> return IRDataTypeValue.DT_QINT32
            DataType.DT_STRING, DataType.DT_STRING_REF -> return IRDataTypeValue.DT_STRING
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return IRDataTypeValue.DT_UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> return IRDataTypeValue.DT_UINT32
            DataType.DT_UINT64, DataType.DT_UINT64_REF -> return IRDataTypeValue.DT_UINT64

        }

        return IRDataTypeValue.DT_INVALID
    }



    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): DataType {
        return this.dataType
    }

    override fun nd4jDataType(): org.nd4j.linalg.api.buffer.DataType {
        when(this.dataType) {
            DataType.DT_BOOL, DataType.DT_BOOL_REF -> return org.nd4j.linalg.api.buffer.DataType.BOOL
            DataType.DT_FLOAT, DataType.DT_FLOAT_REF -> return org.nd4j.linalg.api.buffer.DataType.FLOAT
            DataType.DT_STRING, DataType.DT_STRING_REF -> return org.nd4j.linalg.api.buffer.DataType.UTF8
            DataType.DT_BFLOAT16, DataType.DT_BFLOAT16_REF -> return org.nd4j.linalg.api.buffer.DataType.BFLOAT16
            DataType.DT_INT64, DataType.DT_INT64_REF -> return org.nd4j.linalg.api.buffer.DataType.INT64
            DataType.DT_HALF, DataType.DT_HALF_REF -> return org.nd4j.linalg.api.buffer.DataType.FLOAT16
            DataType.DT_INT8, DataType.DT_INT8_REF -> return org.nd4j.linalg.api.buffer.DataType.INT8
            DataType.DT_INT16, DataType.DT_INT16_REF -> return org.nd4j.linalg.api.buffer.DataType.INT16
            DataType.DT_INT32, DataType.DT_INT32_REF -> return org.nd4j.linalg.api.buffer.DataType.INT32
            DataType.DT_DOUBLE, DataType.DT_DOUBLE_REF -> return org.nd4j.linalg.api.buffer.DataType.DOUBLE
            DataType.DT_UINT8, DataType.DT_UINT8_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT8
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT32
            DataType.DT_UINT64, DataType.DT_UINT64_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT64
        }

        return org.nd4j.linalg.api.buffer.DataType.UNKNOWN
    }

    override fun nameSpaceDataType(): TensorNamespace.DataType {
        when(this.dataType) {
            DataType.DT_BOOL, DataType.DT_BOOL_REF -> return TensorNamespace.DataType.BOOL
            DataType.DT_FLOAT, DataType.DT_FLOAT_REF -> return TensorNamespace.DataType.FLOAT
            DataType.DT_STRING, DataType.DT_STRING_REF -> return TensorNamespace.DataType.STRING
            DataType.DT_BFLOAT16, DataType.DT_BFLOAT16_REF -> return TensorNamespace.DataType.BFLOAT16
            DataType.DT_INT64, DataType.DT_INT64_REF -> return TensorNamespace.DataType.INT64
            DataType.DT_HALF, DataType.DT_HALF_REF -> return TensorNamespace.DataType.FLOAT16
            DataType.DT_INT16, DataType.DT_INT16_REF -> return TensorNamespace.DataType.INT16
            DataType.DT_INT32, DataType.DT_INT32_REF -> return TensorNamespace.DataType.INT32
            DataType.DT_INT8,DataType.DT_INT8_REF -> return TensorNamespace.DataType.INT8
            DataType.DT_UINT8,DataType.DT_UINT8_REF -> return TensorNamespace.DataType.UINT8
            DataType.DT_DOUBLE, DataType.DT_DOUBLE_REF -> return TensorNamespace.DataType.DOUBLE
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return TensorNamespace.DataType.UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> return TensorNamespace.DataType.UINT32
            DataType.DT_UINT64, DataType.DT_UINT64_REF -> return TensorNamespace.DataType.UINT64
        }

        return TensorNamespace.DataType.UNDEFINED
    }

}