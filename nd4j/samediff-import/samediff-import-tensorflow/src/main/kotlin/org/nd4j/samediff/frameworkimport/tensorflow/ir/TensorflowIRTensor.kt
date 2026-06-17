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
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.DataType as Nd4jDataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto
import java.lang.IllegalArgumentException
import java.nio.Buffer
import java.nio.ByteBuffer

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


        var hasNormalData = false


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
            DataType.DT_INVALID -> builder.dataType = TensorNamespace.DataType.UNRECOGNIZED.ordinal
            DataType.DT_FLOAT_REF -> builder.dataType = TensorNamespace.DataType.FLOAT.ordinal
            DataType.DT_DOUBLE_REF -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            DataType.DT_INT32_REF -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            DataType.DT_UINT8_REF ->builder.dataType = TensorNamespace.DataType.UINT8.ordinal
            DataType.DT_INT16_REF -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            DataType.DT_INT8_REF -> builder.dataType = TensorNamespace.DataType.INT8.ordinal
            DataType.DT_STRING_REF -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            DataType.DT_INT64_REF -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            DataType.DT_BOOL_REF -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            DataType.DT_BFLOAT16_REF -> builder.dataType = TensorNamespace.DataType.BFLOAT16.ordinal
            DataType.DT_UINT16_REF -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            DataType.DT_HALF_REF -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            DataType.DT_UINT32_REF -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            DataType.DT_UINT64_REF -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            else -> throw IllegalArgumentException("Unsupported type ${tensor.dtype}")
        }


        if(tensor.doubleValList != null && tensor.doubleValCount > 0) {
            hasNormalData = true
            builder.addAllDoubleData(tensor.doubleValList)
        }

        if(tensor.stringValList != null && tensor.stringValCount > 0) {
            hasNormalData = true
            builder.addAllStringData(tensor.stringValList)
        }

        if(tensor.floatValList != null && tensor.floatValCount > 0) {
            hasNormalData = true
            builder.addAllFloatData(tensor.floatValList)
        }

        if(tensor.intValList != null && tensor.intValCount > 0) {
            hasNormalData = true
            builder.addAllInt32Data(tensor.intValList)
        }

        if(tensor.uint64ValList != null && tensor.uint64ValCount > 0) {
            hasNormalData = true
            builder.addAllInt64Data(tensor.uint64ValList)
        }

        if(tensor.int64ValList != null && tensor.int64ValCount > 0) {
            hasNormalData = true
            builder.addAllInt64Data(tensor.int64ValList)
        }

        if(tensor.halfValList != null && tensor.halfValCount > 0) {
            hasNormalData = true
            builder.addAllHalfVal(tensor.halfValList)
        }

        if(tensor.boolValList != null && tensor.boolValCount > 0) {
            hasNormalData = true
            builder.addAllBoolVal(tensor.boolValList)
        }

        if(tensor.tensorContent != null && !hasNormalData) {
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

        // OPTIMIZATION: For tensorContent (raw data), convert directly without building intermediate protobuf
        // Check that there's no "normal" data fields populated
        if (tensor.tensorContent != null && !tensor.tensorContent.isEmpty &&
            tensor.floatValCount == 0 && tensor.doubleValCount == 0 &&
            tensor.intValCount == 0 && tensor.int64ValCount == 0 &&
            tensor.uint64ValCount == 0 && tensor.halfValCount == 0 &&
            tensor.boolValCount == 0 && tensor.stringValCount == 0) {
            return loadTensorContentDirect(tensor)
        }

        // Fall back to standard path for non-raw data (float_val, int_val, etc.)
        return ndarrayFromNameSpaceTensor(toArgTensor())
    }

    /**
     * Optimized direct loading of TensorFlow tensor content to INDArray.
     * Avoids building intermediate TensorNamespace.TensorProto protobuf.
     */
    private fun loadTensorContentDirect(tensor: TensorProto): INDArray {
        val shape = tensor.tensorShape.dimList.map { it.size }.toLongArray()
        val dtype = convertTfDataType(tensor.dtype)

        var totalLen = if (shape.isNotEmpty()) {
            var prod = 1L
            for (dim in shape) prod *= dim
            prod
        } else {
            1L
        }
        if (totalLen < 1) totalLen = 1

        // Get ByteBuffer view directly from protobuf (no copy)
        val protoBuffer = tensor.tensorContent.asReadOnlyByteBuffer()

        // Allocate direct buffer and copy once
        val byteBuffer = ByteBuffer.allocateDirect((totalLen * dtype.width()).toInt())
        byteBuffer.put(protoBuffer)
        (byteBuffer as Buffer).rewind()

        val rawDataBuffer = Nd4j.createBuffer(byteBuffer, dtype, totalLen.toInt())
        return if (shape.isNotEmpty() && rawDataBuffer.length() > 0) {
            Nd4j.create(rawDataBuffer as DataBuffer).reshape('c', *shape)
        } else {
            Nd4j.create(rawDataBuffer as DataBuffer)
        }
    }

    private fun convertTfDataType(tfType: DataType): Nd4jDataType {
        return when (tfType) {
            DataType.DT_FLOAT, DataType.DT_FLOAT_REF -> Nd4jDataType.FLOAT
            DataType.DT_DOUBLE, DataType.DT_DOUBLE_REF -> Nd4jDataType.DOUBLE
            DataType.DT_INT32, DataType.DT_INT32_REF -> Nd4jDataType.INT32
            DataType.DT_INT64, DataType.DT_INT64_REF -> Nd4jDataType.INT64
            DataType.DT_INT16, DataType.DT_INT16_REF -> Nd4jDataType.INT16
            DataType.DT_INT8, DataType.DT_INT8_REF -> Nd4jDataType.INT8
            DataType.DT_UINT8, DataType.DT_UINT8_REF -> Nd4jDataType.UINT8
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> Nd4jDataType.UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> Nd4jDataType.UINT32
            DataType.DT_UINT64, DataType.DT_UINT64_REF -> Nd4jDataType.UINT64
            DataType.DT_HALF, DataType.DT_HALF_REF -> Nd4jDataType.FLOAT16
            DataType.DT_BFLOAT16, DataType.DT_BFLOAT16_REF -> Nd4jDataType.BFLOAT16
            DataType.DT_BOOL, DataType.DT_BOOL_REF -> Nd4jDataType.BOOL
            else -> Nd4jDataType.FLOAT
        }
    }
}