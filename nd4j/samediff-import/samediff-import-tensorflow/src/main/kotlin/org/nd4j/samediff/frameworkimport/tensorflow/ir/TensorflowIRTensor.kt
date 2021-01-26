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
package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto

class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto, DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return  tensor.tensorShape.dimList.map { it.size }

    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<DataType> {
        return TensorflowIRDataType(tensor.dtype)
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensor.tensorShape.dimCount) {
            builder.addDims(tensor.tensorShape.getDim(i).size)
        }

        when(tensor.dtype) {
            DataType.DT_UINT8 -> builder.dataType = TensorNamespace.DataType.UINT8.ordinal
            DataType.DT_UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            DataType.DT_INT8 -> builder.dataType = TensorNamespace.DataType.INT8.ordinal
            DataType.DT_UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            DataType.DT_UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            DataType.DT_UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            DataType.DT_HALF -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            DataType.DT_STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            DataType.DT_FLOAT -> builder.dataType = TensorNamespace.DataType.FLOAT.ordinal
            DataType.DT_DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            DataType.DT_BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            DataType.DT_INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            DataType.DT_INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            DataType.DT_INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            DataType.DT_BFLOAT16 -> builder.dataType = TensorNamespace.DataType.BFLOAT16.ordinal
            DataType.DT_COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            DataType.DT_COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            DataType.UNRECOGNIZED -> builder.dataType = TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleValList != null && tensor.doubleValCount > 0) {
            builder.addAllDoubleData(tensor.doubleValList)
        }

        if(tensor.stringValList != null && tensor.stringValCount > 0) {
            builder.addAllStringData(tensor.stringValList)
        }

        if(tensor.floatValList != null && tensor.floatValCount > 0) {
            builder.addAllFloatData(tensor.floatValList)
        }

        if(tensor.intValList != null && tensor.intValCount > 0) {
            builder.addAllInt32Data(tensor.intValList)
        }

        if(tensor.uint64ValList != null && tensor.uint64ValCount > 0) {
            builder.addAllInt64Data(tensor.uint64ValList)
        }

        if(tensor.int64ValList != null && tensor.int64ValCount > 0) {
            builder.addAllInt64Data(tensor.int64ValList)
        }

        if(tensor.halfValList != null && tensor.halfValCount > 0) {
            builder.addAllHalfVal(tensor.halfValList)
        }

        if(tensor.boolValList != null && tensor.boolValCount > 0) {
            builder.addAllBoolVal(tensor.boolValList)
        }

        if(tensor.tensorContent != null) {
            builder.rawData = tensor.tensorContent
        }


        return builder.build()
    }

    override fun rawValue(): TensorProto {
        return tensor
    }

    override fun toNd4jNDArray(): INDArray {
        if(tensor.dtype == DataType.UNRECOGNIZED || tensor.dtype == DataType.DT_INVALID)
            return Nd4j.empty()
         return ndarrayFromNameSpaceTensor(toArgTensor())
    }
}