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
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor
import java.nio.Buffer
import java.nio.ByteBuffer

class OnnxIRTensor(input: Onnx.TensorProto): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return tensor.dimsList
    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[tensor.dataType])
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensor.dimsCount) {
            builder.addDims(tensor.getDims(i))
        }

        when(tensor.dataType) {
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

        // rawData in protobuf is never null - check if it's non-empty
        if(tensor.rawData != null && !tensor.rawData.isEmpty) {
            builder.rawData = tensor.rawData
        }

        builder.dataType = tensor.dataType

        return builder.build()
    }

    override fun rawValue(): Onnx.TensorProto {
        return tensor
    }

    override fun toNd4jNDArray(): INDArray {
        return ndarrayFromNameSpaceTensor(toArgTensor())
    }

    /**
     * Optimized direct loading of ONNX tensor raw data to INDArray.
     * Avoids building intermediate TensorNamespace.TensorProto protobuf.
     *
     * Uses CPU-only buffer allocation by default for lazy GPU migration.
     * Model weights are loaded to CPU memory first, and GPU allocation
     * is deferred until the data is actually needed during execution.
     * This significantly reduces model load time and memory pressure.
     */
    private fun loadOnnxRawDataDirect(tensor: Onnx.TensorProto): INDArray {
        val shape = tensor.dimsList.toLongArray()
        val onnxDtype = Onnx.TensorProto.DataType.values()[tensor.dataType]
        val dtype = convertOnnxDataType(onnxDtype)

        var totalLen = if (shape.isNotEmpty()) {
            var prod = 1L
            for (dim in shape) prod *= dim
            prod
        } else {
            1L
        }
        if (totalLen < 1) totalLen = 1

        // Decode raw bytes into typed Java arrays, then use Nd4j.createBuffer(typedArray).
        // ONNX raw_data is always LITTLE_ENDIAN.
        // Nd4j.createBuffer(ByteBuffer, DataType, int) has issues with INT64 on CUDA
        // (corrupted pointer values in device buffer), so we use typed arrays instead.
        val protoBuffer = tensor.rawData.asReadOnlyByteBuffer()
        protoBuffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)

        val dataBuffer: DataBuffer = when (dtype) {
            DataType.FLOAT -> {
                val arr = FloatArray(totalLen.toInt())
                protoBuffer.asFloatBuffer().get(arr)
                Nd4j.createBuffer(arr)
            }
            DataType.DOUBLE -> {
                val arr = DoubleArray(totalLen.toInt())
                protoBuffer.asDoubleBuffer().get(arr)
                Nd4j.createBuffer(arr)
            }
            DataType.INT64 -> {
                val arr = LongArray(totalLen.toInt())
                protoBuffer.asLongBuffer().get(arr)
                Nd4j.createBuffer(arr)
            }
            DataType.INT32 -> {
                val arr = IntArray(totalLen.toInt())
                protoBuffer.asIntBuffer().get(arr)
                Nd4j.createBuffer(arr)
            }
            DataType.INT16, DataType.BFLOAT16, DataType.FLOAT16 -> {
                val arr = ShortArray(totalLen.toInt())
                protoBuffer.asShortBuffer().get(arr)
                Nd4j.createTypedBuffer(arr, dtype)
            }
            else -> {
                // For other types (INT8, UINT8, BOOL, etc.), copy raw bytes
                val byteBuffer = ByteBuffer.allocateDirect((totalLen * dtype.width()).toInt())
                byteBuffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                byteBuffer.put(protoBuffer)
                (byteBuffer as Buffer).rewind()
                Nd4j.createBuffer(byteBuffer, dtype, totalLen.toInt())
            }
        }

        return if (shape.isNotEmpty() && dataBuffer.length() > 0) {
            Nd4j.create(dataBuffer).reshape('c', *shape)
        } else {
            Nd4j.create(dataBuffer)
        }
    }

    private fun convertOnnxDataType(onnxType: Onnx.TensorProto.DataType): DataType {
        return when (onnxType) {
            Onnx.TensorProto.DataType.FLOAT -> DataType.FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> DataType.DOUBLE
            Onnx.TensorProto.DataType.INT32 -> DataType.INT32
            Onnx.TensorProto.DataType.INT64 -> DataType.INT64
            Onnx.TensorProto.DataType.INT16 -> DataType.INT16
            Onnx.TensorProto.DataType.INT8 -> DataType.INT8
            Onnx.TensorProto.DataType.UINT8 -> DataType.UINT8
            Onnx.TensorProto.DataType.UINT16 -> DataType.UINT16
            Onnx.TensorProto.DataType.UINT32 -> DataType.UINT32
            Onnx.TensorProto.DataType.UINT64 -> DataType.UINT64
            Onnx.TensorProto.DataType.FLOAT16 -> DataType.FLOAT16
            Onnx.TensorProto.DataType.BFLOAT16 -> DataType.BFLOAT16
            Onnx.TensorProto.DataType.BOOL -> DataType.BOOL
            else -> DataType.FLOAT
        }
    }


}