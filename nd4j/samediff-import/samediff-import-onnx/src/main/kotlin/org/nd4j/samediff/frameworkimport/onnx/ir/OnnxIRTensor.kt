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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor

class OnnxIRTensor: IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {

    constructor(input: Onnx.TensorProto) {
        tensors = arrayOf(input)
    }

    constructor(inputs: Array<Onnx.TensorProto>) {
        tensors = inputs

    }

    var tensors: Array<Onnx.TensorProto> = arrayOf()


    override fun shape(): List<Long> {
        return tensors[0].dimsList
    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[tensors[0].dataType])
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        return createArgTensorFrom(tensors[0])
    }


    override fun rawValue(): Onnx.TensorProto {
        return tensors[0]
    }

    override fun toNd4jNDArray(): INDArray {
        return ndarrayFromNameSpaceTensor(toArgTensor())
    }

    override fun rawValues(): Array<Onnx.TensorProto> {
        return tensors
    }

    override fun toArgTensors(): Array<TensorNamespace.TensorProto> {
       return tensors.map { input -> createArgTensorFrom(input) }.toTypedArray()
    }

    override fun toNd4jNdarrays(): Array<INDArray> {
       return tensors.map { input -> ndarrayFromNameSpaceTensor(createArgTensorFrom(input))}.toTypedArray()

    }

    private fun createArgTensorFrom(tensorProto: Onnx.TensorProto): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensorProto.dimsCount) {
            builder.addDims(tensorProto.getDims(i))
        }

        when(tensorProto.dataType) {
            Onnx.TensorProto.DataType.UINT64.ordinal -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            Onnx.TensorProto.DataType.UINT32.ordinal -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            Onnx.TensorProto.DataType.UINT16.ordinal -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            Onnx.TensorProto.DataType.FLOAT16.ordinal -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            Onnx.TensorProto.DataType.STRING.ordinal -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            Onnx.TensorProto.DataType.FLOAT.ordinal -> builder.dataType  = TensorNamespace.DataType.FLOAT.ordinal
            Onnx.TensorProto.DataType.DOUBLE.ordinal -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            Onnx.TensorProto.DataType.BOOL.ordinal -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            Onnx.TensorProto.DataType.INT64.ordinal -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            Onnx.TensorProto.DataType.INT32.ordinal -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            Onnx.TensorProto.DataType.INT16.ordinal -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            Onnx.TensorProto.DataType.COMPLEX64.ordinal -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            Onnx.TensorProto.DataType.COMPLEX128.ordinal -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            Onnx.TensorProto.DataType.UNDEFINED.ordinal, Onnx.TensorProto.DataType.UNRECOGNIZED.ordinal ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensorProto.doubleDataList != null && tensorProto.doubleDataCount > 0) {
            builder.addAllDoubleData(tensorProto.doubleDataList)
        }

        if(tensorProto.stringDataList != null && tensorProto.stringDataCount > 0) {
            builder.addAllStringData(tensorProto.stringDataList)
        }

        if(tensorProto.floatDataList != null && tensorProto.floatDataCount > 0) {
            builder.addAllFloatData(tensorProto.floatDataList)
        }

        if(tensorProto.int32DataList != null && tensorProto.int32DataCount > 0) {
            builder.addAllInt32Data(tensorProto.int32DataList)
        }

        if(tensorProto.int64DataCount != null && tensorProto.int64DataCount > 0) {
            builder.addAllInt64Data(tensorProto.int64DataList)
        }

        if(tensorProto.uint64DataList != null && tensorProto.uint64DataCount > 0) {
            builder.addAllInt64Data(tensorProto.uint64DataList)
        }

        if(tensorProto.rawData != null) {
            builder.rawData = tensorProto.rawData
        }

        builder.dataType = tensorProto.dataType

        return builder.build()
    }


}