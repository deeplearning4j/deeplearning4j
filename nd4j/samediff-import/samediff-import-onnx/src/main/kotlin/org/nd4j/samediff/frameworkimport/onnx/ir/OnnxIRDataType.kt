/* Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRDataTypeValue

class OnnxIRDataType(inputDataType: Onnx.TensorProto.DataType): IRDataType<Onnx.TensorProto.DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: Onnx.TensorProto.DataType): IRDataTypeValue {
        when(input) {
            Onnx.TensorProto.DataType.UINT64 ->  return IRDataTypeValue.DT_UINT64
            Onnx.TensorProto.DataType.UINT32 ->  return IRDataTypeValue.DT_UINT32
            Onnx.TensorProto.DataType.UINT16 ->  return IRDataTypeValue.DT_UINT16
            Onnx.TensorProto.DataType.FLOAT16 -> return IRDataTypeValue.DT_HALF
            Onnx.TensorProto.DataType.STRING -> return IRDataTypeValue.DT_STRING
            Onnx.TensorProto.DataType.FLOAT ->  return IRDataTypeValue.DT_FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return IRDataTypeValue.DT_DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return IRDataTypeValue.DT_BOOL
            Onnx.TensorProto.DataType.INT64 -> return IRDataTypeValue.DT_INT64
            Onnx.TensorProto.DataType.INT32 ->  return IRDataTypeValue.DT_INT32
            Onnx.TensorProto.DataType.INT16 -> return IRDataTypeValue.DT_INT16
            Onnx.TensorProto.DataType.COMPLEX64 ->  return IRDataTypeValue.DT_COMPLEX64
            Onnx.TensorProto.DataType.COMPLEX128 ->  return IRDataTypeValue.DT_COMPLEX128
            Onnx.TensorProto.DataType.UNDEFINED, Onnx.TensorProto.DataType.UNRECOGNIZED ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }

        return IRDataTypeValue.DT_INVALID
    }

    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): Onnx.TensorProto.DataType {
        return this.dataType
    }

    override fun nd4jDataType(): DataType {
        when(this.dataType) {
            Onnx.TensorProto.DataType.UINT64 ->  return  return DataType.INT64
            Onnx.TensorProto.DataType.UINT32 ->  return return DataType.INT32
            Onnx.TensorProto.DataType.UINT16 ->  return return DataType.INT16
            Onnx.TensorProto.DataType.FLOAT16 -> return   return DataType.FLOAT16
            Onnx.TensorProto.DataType.STRING -> return  return DataType.UTF8
            Onnx.TensorProto.DataType.FLOAT ->  return  return DataType.FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return  return DataType.DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return  return DataType.BOOL
            Onnx.TensorProto.DataType.INT64 -> return  return DataType.INT64
            Onnx.TensorProto.DataType.INT32 ->  return  return DataType.INT32
            Onnx.TensorProto.DataType.INT16 -> return  return DataType.INT16

        }

        return  return DataType.UNKNOWN

    }

    override fun nameSpaceDataType(): TensorNamespace.DataType {
        when(this.dataType) {
            Onnx.TensorProto.DataType.UINT64 ->  return  return TensorNamespace.DataType.INT64
            Onnx.TensorProto.DataType.UINT32 ->  return return TensorNamespace.DataType.INT32
            Onnx.TensorProto.DataType.UINT16 ->  return return TensorNamespace.DataType.INT16
            Onnx.TensorProto.DataType.FLOAT16 -> return   return TensorNamespace.DataType.FLOAT16
            Onnx.TensorProto.DataType.STRING -> return  return TensorNamespace.DataType.STRING
            Onnx.TensorProto.DataType.FLOAT ->  return TensorNamespace.DataType.FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return TensorNamespace.DataType.DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return  return TensorNamespace.DataType.BOOL
            Onnx.TensorProto.DataType.INT64 -> return  return TensorNamespace.DataType.INT64
            Onnx.TensorProto.DataType.INT32 ->  return  return TensorNamespace.DataType.INT32
            Onnx.TensorProto.DataType.INT16 -> return  return TensorNamespace.DataType.INT16

        }

        return TensorNamespace.DataType.UNDEFINED
    }

}