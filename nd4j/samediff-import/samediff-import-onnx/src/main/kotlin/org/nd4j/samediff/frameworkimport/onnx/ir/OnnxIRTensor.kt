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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor

class OnnxIRTensor(input: Onnx.TensorProto): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return tensor.dimsList
    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[tensor.dataType.ordinal])
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensor.dimsCount) {
            builder.addDims(tensor.getDims(i))
        }

        when(tensor.dataType) {
            Onnx.TensorProto.DataType.UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            Onnx.TensorProto.DataType.UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            Onnx.TensorProto.DataType.UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            Onnx.TensorProto.DataType.FLOAT16 -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            Onnx.TensorProto.DataType.STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            Onnx.TensorProto.DataType.FLOAT -> builder.dataType  = TensorNamespace.DataType.FLOAT.ordinal
            Onnx.TensorProto.DataType.DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            Onnx.TensorProto.DataType.BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            Onnx.TensorProto.DataType.INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            Onnx.TensorProto.DataType.INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            Onnx.TensorProto.DataType.INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            Onnx.TensorProto.DataType.COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            Onnx.TensorProto.DataType.COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            Onnx.TensorProto.DataType.UNDEFINED, Onnx.TensorProto.DataType.UNRECOGNIZED ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleDataList != null && tensor.doubleDataCount > 0) {
            builder.addAllDoubleData(tensor.doubleDataList)
        }

        if(tensor.stringDataList != null && tensor.stringDataCount > 0) {
            builder.addAllStringData(tensor.stringDataList)
        }

        if(tensor.floatDataList != null && tensor.floatDataCount > 0) {
            builder.addAllFloatData(tensor.floatDataList)
        }

        if(tensor.int32DataList != null && tensor.int32DataCount > 0) {
            builder.addAllInt32Data(tensor.int32DataList)
        }

        if(tensor.int64DataCount != null && tensor.int64DataCount > 0) {
            builder.addAllInt64Data(tensor.int64DataList)
        }

        if(tensor.uint64DataList != null && tensor.uint64DataCount > 0) {
            builder.addAllInt64Data(tensor.uint64DataList)
        }

        if(tensor.rawData != null) {
            builder.rawData = tensor.rawData
        }

        builder.dataType = tensor.dataType.ordinal

        return builder.build()
    }

    override fun rawValue(): Onnx.TensorProto {
        return tensor
    }

    override fun toNd4jNDArray(): INDArray {
        return ndarrayFromNameSpaceTensor(toArgTensor())
    }


}