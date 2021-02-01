/* ******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.samediff.frameworkimport

import org.bytedeco.javacpp.indexer.Bfloat16ArrayIndexer
import org.bytedeco.javacpp.indexer.HalfIndexer
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.common.util.ArrayUtil
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.nativeblas.Nd4jCpu.FLOAT32
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.shade.protobuf.ByteString
import java.lang.IllegalArgumentException
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.util.*
import kotlin.collections.ArrayList
import java.lang.reflect.Field


fun isOutputFrameworkAttributeName(name: String, opDescriptor: OpNamespace.OpDescriptor): Boolean {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType != OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
            && argDescriptor.argType != OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR
    }
        .map { inputArg -> inputArg.name }.contains(name)
}

fun isNd4jTensorName(name: String, opDescriptor: OpNamespace.OpDescriptor): Boolean {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
        .map { inputArg -> inputArg.name }
        .contains(name)
}

fun argDescriptorType(name: String, opDescriptor: OpNamespace.OpDescriptor): OpNamespace.ArgDescriptor.ArgType {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.name == name }[0].argType
}

fun OpNamespace.OpDescriptorList.findOp(opName: String): OpNamespace.OpDescriptor {
    val opRet = this.opListList.firstOrNull {opDescriptor -> opDescriptor.name == opName }
    if(opRet == null) {
        throw IllegalArgumentException("Op name $opName not found!")
    }
    return opRet
}

fun ArgDescriptor(block: OpNamespace.ArgDescriptor.Builder.() -> Unit): OpNamespace.ArgDescriptor {
    return OpNamespace.ArgDescriptor.newBuilder()
        .apply(block).build()
}

fun NameSpaceTensor(block: TensorNamespace.TensorProto.Builder.() -> Unit): TensorNamespace.TensorProto {
    return TensorNamespace.TensorProto.newBuilder()
        .apply(block).build()
}

fun TensorNamespace.TensorProto.Builder.RawData(rawData: ByteArray) {
    this.rawData = ByteString.copyFrom(rawData)
}

fun TensorNamespace.TensorProto.Builder.IntData(intData: List<Int>) {
    this.addAllInt32Data(intData)
}

fun TensorNamespace.TensorProto.Builder.FloatData(floatData: List<Float>) {
    this.addAllFloatData(floatData)
}

fun TensorNamespace.TensorProto.Builder.DoubleData(doubleData: List<Double>) {
    this.addAllDoubleData(doubleData)
}

fun TensorNamespace.TensorProto.Builder.StringData(stringData: List<String>) {
    this.addAllStringData(stringData.map { input -> ByteString.copyFrom(input.toByteArray(Charset.defaultCharset())) })
}

fun TensorNamespace.TensorProto.Builder.Int64Data(intData: List<Long>) {
    this.addAllInt64Data(intData)
}

fun TensorNamespace.TensorProto.Builder.Dims(shape: List<Long>) {
    shape.forEach { this.addDims(it) }
}

fun convertNd4jDataTypeFromNameSpaceTensorDataType(dataType: TensorNamespace.DataType): DataType {
    return when(dataType) {
        TensorNamespace.DataType.UINT32 -> return DataType.UINT32
        TensorNamespace.DataType.UINT8 -> return DataType.UINT8
        TensorNamespace.DataType.INT64 -> return DataType.INT64
        TensorNamespace.DataType.INT16 -> return DataType.INT16
        TensorNamespace.DataType.UINT64 ->  return DataType.UINT64
        TensorNamespace.DataType.DOUBLE ->  return DataType.DOUBLE
        TensorNamespace.DataType.FLOAT ->  return DataType.FLOAT
        TensorNamespace.DataType.FLOAT16 ->  return DataType.FLOAT16
        TensorNamespace.DataType.FLOAT16 -> return DataType.FLOAT16
        TensorNamespace.DataType.INT32 ->  return DataType.INT32
        TensorNamespace.DataType.STRING ->  return DataType.UTF8
        TensorNamespace.DataType.BOOL -> return DataType.BOOL
        TensorNamespace.DataType.BFLOAT16 -> return DataType.BFLOAT16
        TensorNamespace.DataType.INT8 -> return DataType.INT8
        TensorNamespace.DataType.UINT16 -> return DataType.UINT16
        TensorNamespace.DataType.UNDEFINED, TensorNamespace.DataType.UNRECOGNIZED -> return DataType.UNKNOWN
        else -> {
            throw IllegalArgumentException("Illegal data type $dataType")
        }
    }
}

fun convertNameSpaceTensorDataTypeFromNd4jDataType(dataType: DataType): TensorNamespace.DataType {
    return when(dataType) {
        DataType.UINT32 ->  return TensorNamespace.DataType.UINT32
        DataType.INT64, DataType.LONG ->  return TensorNamespace.DataType.INT64
        DataType.UINT64 ->  return TensorNamespace.DataType.UINT64
        DataType.DOUBLE ->  return TensorNamespace.DataType.DOUBLE
        DataType.FLOAT ->  return TensorNamespace.DataType.FLOAT
        DataType.FLOAT16, DataType.HALF ->  return TensorNamespace.DataType.FLOAT16
        DataType.HALF -> return TensorNamespace.DataType.FLOAT16
        DataType.INT32, DataType.INT ->  return TensorNamespace.DataType.INT32
        DataType.UTF8 ->  return TensorNamespace.DataType.STRING
        DataType.BOOL -> return TensorNamespace.DataType.BOOL
        DataType.BFLOAT16 -> return TensorNamespace.DataType.BFLOAT16
        DataType.SHORT, DataType.INT8 -> return TensorNamespace.DataType.INT8
        DataType.UINT16 -> return TensorNamespace.DataType.UINT16
        DataType.BYTE, DataType.UINT8, DataType.UBYTE -> return TensorNamespace.DataType.UINT8
        else -> {
            throw IllegalArgumentException("Illegal data type $dataType")
        }
    }
}

fun ndarrayFromNameSpaceTensor(inputTensor: TensorNamespace.TensorProto): INDArray {
    val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(TensorNamespace.DataType.values()[inputTensor.dataType])
    val shape = inputTensor.dimsList.toLongArray()
    val totalLen = ArrayUtil.prod(*shape)
    //note for all cases here scalars can be either zero shape with 1 element or rank >= 1 with 1 element
    when(dtype) {
        DataType.FLOAT -> {
            val floatArray = inputTensor.floatDataList.toFloatArray()
            if(floatArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else  if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(floatArray[0])
            } else if(totalLen != floatArray.size) {
                //broadcast case
                if(floatArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,floatArray[0])
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${floatArray.size}")
            }

            val dataBuffer = Nd4j.createBuffer(floatArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }

        DataType.DOUBLE -> {
            val doubleArray = inputTensor.doubleDataList.toDoubleArray()
            if(doubleArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else  if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(doubleArray[0])
            }
            else if(totalLen != doubleArray.size) {
                //broadcast case
                if(doubleArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,doubleArray[0])
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${doubleArray.size}")

            }

            val dataBuffer = Nd4j.createBuffer(doubleArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }

        DataType.FLOAT16,DataType.HALF -> {
            val halfArray = inputTensor.halfValList.toIntArray()
            if(halfArray.isEmpty()) {
                return loadDataBufferFromRawData(inputTensor)
            } else if(totalLen <= 1 && shape.isEmpty()) {
                val convertedFloat = HalfIndexer.toFloat(halfArray[0])
                return Nd4j.scalar(convertedFloat).castTo(DataType.FLOAT16)
            } else if(totalLen != halfArray.size) {
                //broadcast case
                if(halfArray.size == 1) {
                    val convertedFloat = HalfIndexer.toFloat(halfArray[0])
                    return Nd4j.valueArrayOf(shape,convertedFloat).castTo(DataType.FLOAT16)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${halfArray.size}")
            }

            val dataBuffer = Nd4j.createBuffer(DataType.FLOAT,halfArray.size.toLong(),false)

            for(i in 0 until halfArray.size) {
                dataBuffer.put(i.toLong(),HalfIndexer.toFloat(halfArray[i]))
            }

            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.FLOAT16)
        }

        DataType.BFLOAT16 -> {
            val halfArray = inputTensor.halfValList.toIntArray()
            if(halfArray.isEmpty()) {
                return loadDataBufferFromRawData(inputTensor)
            } else if(totalLen <= 1 && shape.isEmpty()) {
                val convertedFloat = Bfloat16ArrayIndexer.toFloat(halfArray[0])
                return Nd4j.scalar(convertedFloat).castTo(DataType.BFLOAT16)
            } else if(totalLen != halfArray.size) {
                //broadcast case
                if(halfArray.size == 1) {
                    val convertedFloat = Bfloat16ArrayIndexer.toFloat(halfArray[0])
                    return Nd4j.valueArrayOf(shape,convertedFloat).castTo(DataType.BFLOAT16)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${halfArray.size}")
            }

            val dataBuffer = Nd4j.createBuffer(DataType.FLOAT,halfArray.size.toLong(),false)

            for(i in 0 until halfArray.size) {
                dataBuffer.put(i.toLong(),Bfloat16ArrayIndexer.toFloat(halfArray[i]))
            }

            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.BFLOAT16)
        }


        DataType.INT64 -> {
            val longArray = inputTensor.int64DataList.toLongArray()
            if(longArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)

            else  if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(longArray[0])
            } else   if(totalLen != longArray.size) {
                //broadcast case
                if(longArray.size == 1) {
                    return Nd4j.zeros(*shape).addi(longArray[0]).castTo(DataType.INT64)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${longArray.size}")
            }

            val dataBuffer = Nd4j.createBuffer(longArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }

        DataType.INT32 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0])
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,intArray[0])
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }
            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }

        DataType.INT16 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0]).castTo(DataType.INT16)
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,intArray[0]).castTo(DataType.INT16)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }
            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.INT16)
        }

        DataType.INT8 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0]).castTo(DataType.INT8)
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,intArray[0]).castTo(DataType.INT8)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }
            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.INT8)
        }


        DataType.UINT8 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0]).castTo(DataType.UINT8)
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,intArray[0]).castTo(DataType.UINT8)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }
            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.UINT8)
        }

        DataType.UINT16 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0]).castTo(DataType.UINT16)
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    return Nd4j.valueArrayOf(shape,intArray[0]).castTo(DataType.UINT16)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }
            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape).castTo(DataType.UINT16)
        }

        DataType.BOOL -> {
            val intArray = inputTensor.boolValList.toBooleanArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(intArray[0])
            }
            else if(totalLen != intArray.size) {
                //broadcast case
                if(intArray.size == 1) {
                    val booleanList = ArrayList<Boolean>()
                    for(i in 0 until totalLen) {
                        booleanList.add(intArray[0])
                    }
                    return Nd4j.create(booleanList.toBooleanArray()).reshape(*shape)
                }
                else
                    throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${intArray.size}")
            }

            return Nd4j.create(intArray).reshape(*shape)
        }

        DataType.UTF8 -> {
            val stringList = inputTensor.stringDataList.map { input -> input.toStringUtf8() }
            if(stringList.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            else  if(totalLen <= 1 && shape.isEmpty()) {
                return Nd4j.scalar(stringList[0])
            } else if(totalLen != stringList.size) {
                //broadcast case
                if(stringList.size == 1) {
                    val newStringList = ArrayList<String>()
                    for(i in 0 until totalLen) {
                        newStringList.add(stringList[0])
                    }

                    return Nd4j.create(newStringList).reshape(*shape)
                }
                throw IllegalArgumentException("Shape of ${Arrays.toString(shape)} did not match length ${stringList.size}")
            }
            return Nd4j.create(stringList).reshape(*shape)
        }

        DataType.UNKNOWN -> {
            val ret = Nd4j.empty()
            return ret
        }

        else -> {
            return loadDataBufferFromRawData(inputTensor)
        }

    }

    throw IllegalArgumentException("Illegal type found for conversion ${dtype}")
}

fun loadDataBufferFromRawData(inputTensor: TensorNamespace.TensorProto): INDArray {
    val shape = inputTensor.dimsList.toLongArray()
    val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(TensorNamespace.DataType.values()[inputTensor.dataType])
    val byteArray = inputTensor.rawData.toByteArray()
    //note: scalar can be zero
    val totalLen = ArrayUtil.prod(*shape)
    if(totalLen < 1) {
        if(shape.isNotEmpty()) {
            return Nd4j.zeros(*shape).castTo(dtype)
        }
        else
            return Nd4j.empty(dtype)
    }

    val byteBuffer = ByteBuffer.allocateDirect(totalLen * dtype.width())
    byteBuffer.put(byteArray)
    byteBuffer.rewind()
    val rawDataBuffer = Nd4j.createBuffer(byteBuffer, dtype, totalLen, 0)
    if(shape.isNotEmpty() && totalLen > 0) {
        if(rawDataBuffer.length() > 1)
            return Nd4j.create(rawDataBuffer).reshape(*shape)
        return Nd4j.empty(dtype)
    }
    return Nd4j.create(rawDataBuffer)
}

fun nameSpaceTensorFromNDarray(ndarray: INDArray): TensorNamespace.TensorProto {
    val nameSpaceDataType = convertNameSpaceTensorDataTypeFromNd4jDataType(ndarray.dataType()).ordinal
    when(ndarray.dataType()) {
        DataType.INT64 -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                Int64Data(ndarray.data().asLong().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.INT32 -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                IntData(ndarray.data().asInt().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.DOUBLE -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                DoubleData(ndarray.data().asDouble().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.FLOAT -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                FloatData(ndarray.data().asFloat().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.UTF8 -> {
            val stringList = ArrayList<String>()
            for(i in 0 until ndarray.length()) {
                stringList.add(ndarray.getString(i))
            }

            return NameSpaceTensor {
                dataType = nameSpaceDataType
                StringData(stringList)
                Dims(ndarray.shape().asList())
            }
        }

        else -> {
            throw IllegalArgumentException("Illegal data type ${ndarray.dataType()}")
        }
    }

}

fun lookupIndexForArgDescriptor(
    argDescriptorName: String,
    opDescriptorName: String,
    argDescriptorType: OpNamespace.ArgDescriptor.ArgType
): Int {
    val op =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(opDescriptorName)
    val names = op.argDescriptorList.map { argDescriptor -> argDescriptor.name }
    if(!names.contains(argDescriptorName)) {
        throw IllegalArgumentException("Invalid name $argDescriptorName for op $opDescriptorName passed in. $argDescriptorName not found in $opDescriptorName. Available names were ${names}")
    }
    val ret =  op
        .argDescriptorList.firstOrNull { argDescriptor -> argDescriptor.name == argDescriptorName &&
                argDescriptor.argType == argDescriptorType }
    if(ret == null)
        return -1
    else return ret.argIndex
}

fun createVariable(varName: String, varType: VariableType, sameDiff: SameDiff, shape: List<Long>, dataType: DataType): SDVariable {
    return SDVariable(varName, varType, sameDiff, shape.toLongArray(), dataType)
}

fun descriptorsForName(
    name: String,
    argDescriptors: Collection<OpNamespace.ArgDescriptor>): List<OpNamespace.ArgDescriptor> {
    return argDescriptors.filter { argDescriptor -> argDescriptor.name == name }!!
}

fun setNameForFunctionFromDescriptors(argDescriptors: Collection<OpNamespace.ArgDescriptor>, func: DifferentialFunction) {
    val fields = ArrayList<Field>()
    fields.addAll(func.javaClass.declaredFields.toList())
    fields.addAll(func.javaClass.superclass.declaredFields.toList())
    fields.forEach { field ->
        if(hasArgDescriptorWithNameAndType(argDescriptors, field.name)) {
            val descriptors = descriptorsForName(field.name, argDescriptors)
            descriptors.forEach { descriptor ->
                when(descriptor.argType) {
                    OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                        if(Boolean.javaClass.isAssignableFrom(field.type) || Boolean::class.javaPrimitiveType!!.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.boolValue)
                        }
                    }

                    OpNamespace.ArgDescriptor.ArgType.STRING -> {
                        if(field.type.isAssignableFrom(String::class.java)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.stringValue)
                        }
                    }

                    OpNamespace.ArgDescriptor.ArgType.INT64, OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                        if(Int.javaClass.isAssignableFrom(field.type) || Int::class.javaPrimitiveType!!.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.int64Value.toInt())
                        }

                        if(Long.javaClass.isAssignableFrom(field.type) || Long::class.javaPrimitiveType!!.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.int64Value)
                        }

                        if(DataType::javaClass.javaClass.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, DataType.fromInt(descriptor.int64Value.toInt()))
                        }

                    }

                    OpNamespace.ArgDescriptor.ArgType.FLOAT, OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                        if(Float.javaClass.isAssignableFrom(field.type) || Float::class.javaPrimitiveType!!.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.doubleValue.toFloat())
                        }

                        if(Double.javaClass.isAssignableFrom(field.type) || Double::class.javaPrimitiveType!!.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(field, func, descriptor.doubleValue)
                        }
                    }

                    OpNamespace.ArgDescriptor.ArgType.DATA_TYPE -> {
                        if(DataType::class.java.isAssignableFrom(field.type)) {
                            field.isAccessible = true
                            ReflectionUtils.setField(
                                field,
                                func,
                                convertNd4jDataTypeFromNameSpaceTensorDataType(descriptor.dataTypeValue)
                            )
                        }
                    }

                }

            }

        }
    }

}

fun hasArgDescriptorWithNameAndType(argDescriptors: Collection<OpNamespace.ArgDescriptor>, name: String): Boolean {
    return argDescriptors.map { input -> input.name}.contains(name)
}


/**
 * @return The specified name without the leading "^" character (if any) that appears for control dependencies
 */
fun stripControl(name: String): String {
    return if (name.startsWith("^")) {
        name.substring(1)
    } else name
}

/**
 * Remove the ":1" etc suffix for a variable name to get the op name
 *
 * @param varName Variable name
 * @return Variable name without any number suffix
 */
fun stripVarSuffix(varName: String): String {
    if (varName.matches(regex = Regex(".*:\\d+"))) {
        val idx = varName.lastIndexOf(':')
        return varName.substring(0, idx)
    }
    return varName
}