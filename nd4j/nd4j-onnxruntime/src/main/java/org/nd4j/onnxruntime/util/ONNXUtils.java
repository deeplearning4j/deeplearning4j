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
package org.nd4j.onnxruntime.util;

import onnx.Onnx;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.onnxruntime.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.runner.enums.ONNXType;
import org.nd4j.shade.guava.primitives.Longs;
import org.slf4j.Logger;

import java.util.List;
import java.util.stream.Collectors;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;
import static org.nd4j.linalg.api.buffer.DataType.*;

public class ONNXUtils {


    /**
     * Return the {@link INDArray} from a sequence
     * @param outValue the input sequence to get the ndarrays from
     * @param ortAllocator the allocator to use to retrieve relevant memory
     * @return the equivalent arrays
     */
    public static INDArray[] ndarraysFromSequence(Value outValue,OrtAllocator ortAllocator) {
        Preconditions.checkState(outValue.HasValue(),"No value found in specified value!");
        INDArray[] ret = new INDArray[(int) outValue.GetCount()];
        for(int i = 0; i < ret.length; i++) {
            INDArray retValue = getArray(outValue.GetValue(i,ortAllocator));
            ret[i] = retValue;
        }

        return ret;
    }


    /**
     * Create a sequence from a list of tensors
     * returning a {@link ValueVector} equivalent
     * using {@link #getTensor(INDArray, MemoryInfo)}
     *
     * @param sequence the sequence to get
     * @param memoryInfo the memory info to use for allocation
     * @return
     */
    public static ValueVector getSequence(List<INDArray> sequence,MemoryInfo memoryInfo) {
        ValueVector valueVector = new ValueVector(sequence.size());
        for(int i = 0; i < sequence.size(); i++) {
            valueVector.put(getTensor(sequence.get(i),memoryInfo));
        }

        return valueVector;
    }


    /**
     * Get the onnx type of the output
     * @param session the session to get the input for
     * @param i the index of the output
     * @return
     */
    public static ONNXType getTypeForOutput(Session session,int i) {
        TypeInfo typeInfo = session.GetOutputTypeInfo(i);
        return ONNXType.values()[typeInfo.GetONNXType()];
    }


    /**
     * Get the onnx type of the input
     * @param session the session to get the output type info from
     * @param i the index of the input
     * @return the relevant type information
     */
    public static ONNXType getTypeForInput(Session session,long i) {
        TypeInfo typeInfo = session.GetInputTypeInfo(i);
        return ONNXType.values()[typeInfo.GetONNXType()];
    }


    /**
     * Returns a zeroed array of the input data.
     * This array's shape and data type are determined
     * from {@link Onnx.ValueInfoProto#getType()}
     * tensor field.
     * given the value type. Mainly used for quick debugging/
     * testing.
     * @param valueInfoProto the value info proto
     *                       to get the shape information from
     * @return the sample tensor
     */
    public static INDArray getSampleForValueInfo(Onnx.ValueInfoProto valueInfoProto) {
        Preconditions.checkState(valueInfoProto.hasType(),"Value info must have a type!");
        Onnx.TypeProto.Tensor tensorType = valueInfoProto.getType().getTensorType();
        long[] shape = Longs.toArray(tensorType.getShape().getDimList().stream().map(input -> input.getDimValue()).collect(Collectors.toList()));
        DataType type = dataTypeForOnnxType(tensorType.getElemType());
        return Nd4j.create(type,shape);
    }

    /**
     *
     * @param expected
     * @param array
     */
    public static void validateType(DataType expected, INDArray array) {
        if (!array.dataType().equals(expected))
            throw new RuntimeException("INDArray data type (" + array.dataType() + ") does not match required ONNX data type (" + expected + ")");
    }

    /**
     * Return a {@link DataType}
     * for the onnx data type
     * @param dataType the equivalent nd4j data type
     * @return
     */
    public static DataType dataTypeForOnnxType(int dataType) {
        if(dataType == dataType) {
            return FLOAT;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            return INT8;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            return DOUBLE;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            return BOOL;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return UINT8;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
            return UINT16;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
            return INT16;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return INT32;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return INT64;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return FLOAT16;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
            return UINT32;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
            return UINT64;
        } else if(dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
            return BFLOAT16;
        }
        else
            throw new IllegalArgumentException("Illegal data type " + dataType);
    }

    /**
     * Convert the onnx type for the given data type
     * @param dataType
     * @return
     */
    public static int onnxTypeForDataType(DataType dataType) {
        if(dataType == FLOAT) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        } else if(dataType == INT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        } else if(dataType == DOUBLE) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        } else if(dataType == BOOL) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        } else if(dataType == UINT8) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        } else if(dataType == UINT16) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        } else if(dataType == INT16) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        } else if(dataType == INT32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        } else if(dataType == INT64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        } else if(dataType == FLOAT16) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        } else if(dataType == UINT32) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        } else if(dataType == UINT64) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        } else if(dataType == BFLOAT16) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        }
        else
            throw new IllegalArgumentException("Illegal data type " + dataType);
    }


    /**
     * Convert an onnx {@link Value}
     *  in to an {@link INDArray}
     * @param value the value to convert
     * @return
     */
    public static INDArray getArray(Value value) {
        DataType dataType = dataTypeForOnnxType(value.GetTypeInfo().GetONNXType());
        LongVector shape = value.GetTensorTypeAndShapeInfo().GetShape();
        long[] shapeConvert;
        if(shape != null) {
            shapeConvert = new long[(int) value.GetTensorTypeAndShapeInfo().GetDimensionsCount()];
            for(int  j = 0; j < shapeConvert.length; j++) {
                shapeConvert[j] = shape.get(j);
            }
        } else {
            shapeConvert = new long[]{1};
        }

        DataBuffer getBuffer = getDataBuffer(value);
        Preconditions.checkState(dataType.equals(getBuffer.dataType()),"Data type must be equivalent as specified by the onnx metadata.");
        return Nd4j.create(getBuffer,shapeConvert,Nd4j.getStrides(shapeConvert),0);
    }


    /**
     * Get the onnx log level relative to the given slf4j logger.
     * Trace or debug will return ORT_LOGGING_LEVEL_VERBOSE
     * Info will return: ORT_LOGGING_LEVEL_INFO
     * Warn returns ORT_LOGGING_LEVEL_WARNING
     * Error returns error ORT_LOGGING_LEVEL_ERROR
     *
     * The default is info
     * @param logger the slf4j logger to get the onnx log level for
     * @return
     */
    public static int getOnnxLogLevelFromLogger(Logger logger) {
        if(logger.isTraceEnabled() || logger.isDebugEnabled()) {
            return ORT_LOGGING_LEVEL_VERBOSE;
        }
        else if(logger.isInfoEnabled()) {
            return ORT_LOGGING_LEVEL_INFO;
        }
        else if(logger.isWarnEnabled()) {
            return ORT_LOGGING_LEVEL_WARNING;
        }
        else if(logger.isErrorEnabled()) {
            return ORT_LOGGING_LEVEL_ERROR;
        }

        return ORT_LOGGING_LEVEL_INFO;

    }

    /**
     * Get an onnx tensor from an ndarray.
     * @param ndArray the ndarray to get the value from
     * @param memoryInfo the {@link MemoryInfo} to use.
     *                   Can be created with:
     *                   MemoryInfo memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
     * @return
     */
    public static Value getTensor(INDArray ndArray, MemoryInfo memoryInfo) {
        if(ndArray == null) {
            throw new IllegalArgumentException("Cannot create tensor from null array");
        }
        
        if(ndArray.isEmpty()) {
            // Handle empty arrays properly - they still have shape information
            long[] shape = ndArray.shape();
            LongPointer dims = new LongPointer(shape);
            
            // For empty arrays, we need at least a minimal buffer to avoid "not enough space" error
            // Create a dummy buffer with 1 element
            Pointer dummyData;
            int elementSize;
            int onnxType = onnxTypeForDataType(ndArray.dataType());
            
            switch (ndArray.dataType()) {
                case FLOAT:
                    dummyData = new FloatPointer(1);
                    elementSize = 4;
                    break;
                case DOUBLE:
                    dummyData = new DoublePointer(1);
                    elementSize = 8;
                    break;
                case INT:
                    dummyData = new IntPointer(1);
                    elementSize = 4;
                    break;
                case LONG:
                    dummyData = new LongPointer(1);
                    elementSize = 8;
                    break;
                case BYTE:
                    dummyData = new BytePointer(1);
                    elementSize = 1;
                    break;
                case SHORT:
                    dummyData = new ShortPointer(1);
                    elementSize = 2;
                    break;
                case BOOL:
                    dummyData = new BoolPointer(1);
                    elementSize = 1;
                    break;
                default:
                    // Default to float for other types
                    dummyData = new FloatPointer(1);
                    elementSize = 4;
                    onnxType = onnxTypeForDataType(FLOAT);
            }
            
            Value ret = Value.CreateTensor(
                    memoryInfo.asOrtMemoryInfo(),
                    dummyData,
                    elementSize,
                    dims,
                    ndArray.rank(),
                    onnxType);
            return ret;
        }

        Pointer inputTensorValuesPtr = ndArray.data().pointer();
        Pointer inputTensorValues = inputTensorValuesPtr;
        long sizeInBytes = ndArray.length() * ndArray.data().getElementSize();

        LongPointer dims = new LongPointer(ndArray.shape());
        Value ret =  Value.CreateTensor(
                memoryInfo.asOrtMemoryInfo(),
                inputTensorValues,
                sizeInBytes,
                dims,
                ndArray.rank(),
                onnxTypeForDataType(ndArray.dataType()));
        return  ret;
    }

    /**
     * Get the data buffer from the given value
     * @param tens the values to get
     * @return the equivalent data buffer
     */
    public static DataBuffer getDataBuffer(Value tens) {
        if(tens.isNull())
            throw new IllegalArgumentException("Native underlying tensor value was null!");
        try (PointerScope scope = new PointerScope()) {
            DataBuffer buffer = null;
            int type = tens.GetTensorTypeAndShapeInfo().GetElementType();
            long size = tens.GetTensorTypeAndShapeInfo().GetElementCount();
            switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    FloatPointer pFloat = tens.GetTensorMutableDataFloat().capacity(size);
                    FloatIndexer floatIndexer = FloatIndexer.create(pFloat);
                    buffer = Nd4j.createBuffer(pFloat, DataType.FLOAT, size, floatIndexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    BytePointer pUint8 = tens.GetTensorMutableDataUByte().capacity(size);
                    Indexer uint8Indexer = ByteIndexer.create(pUint8);
                    buffer = Nd4j.createBuffer(pUint8, DataType.UINT8, size, uint8Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    BytePointer pInt8 = tens.GetTensorMutableDataByte().capacity(size);
                    Indexer int8Indexer = ByteIndexer.create(pInt8);
                    buffer = Nd4j.createBuffer(pInt8, DataType.UINT8, size, int8Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                    ShortPointer pUint16 = tens.GetTensorMutableDataUShort().capacity(size);
                    Indexer uint16Indexer = ShortIndexer.create(pUint16);
                    buffer = Nd4j.createBuffer(pUint16, DataType.UINT16, size, uint16Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                    ShortPointer pInt16 = tens.GetTensorMutableDataShort().capacity(size);
                    Indexer int16Indexer = ShortIndexer.create(pInt16);
                    buffer = Nd4j.createBuffer(pInt16, INT16, size, int16Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    IntPointer pInt32 = tens.GetTensorMutableDataInt().capacity(size);
                    Indexer int32Indexer = IntIndexer.create(pInt32);
                    buffer = Nd4j.createBuffer(pInt32, DataType.INT32, size, int32Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    LongPointer pInt64 = tens.GetTensorMutableDataLong().capacity(size);
                    Indexer int64Indexer = LongIndexer.create(pInt64);
                    buffer = Nd4j.createBuffer(pInt64, DataType.INT64, size, int64Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
                    BytePointer pString = tens.GetTensorMutableDataByte().capacity(size);
                    Indexer stringIndexer = ByteIndexer.create(pString);
                    buffer = Nd4j.createBuffer(pString, DataType.INT8, size, stringIndexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    BoolPointer pBool = tens.GetTensorMutableDataBool().capacity(size);
                    Indexer boolIndexer = BooleanIndexer.create(new BooleanPointer(pBool)); //Converting from JavaCPP Bool to Boolean here - C++ bool type size is not defined, could cause problems on some platforms
                    buffer = Nd4j.createBuffer(pBool, DataType.BOOL, size, boolIndexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    ShortPointer pFloat16 = tens.GetTensorMutableDataShort().capacity(size);
                    Indexer float16Indexer = ShortIndexer.create(pFloat16);
                    buffer = Nd4j.createBuffer(pFloat16, DataType.FLOAT16, size, float16Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                    DoublePointer pDouble = tens.GetTensorMutableDataDouble().capacity(size);
                    Indexer doubleIndexer = DoubleIndexer.create(pDouble);
                    buffer = Nd4j.createBuffer(pDouble, DataType.DOUBLE, size, doubleIndexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                    IntPointer pUint32 = tens.GetTensorMutableDataUInt().capacity(size);
                    Indexer uint32Indexer = IntIndexer.create(pUint32);
                    buffer = Nd4j.createBuffer(pUint32, DataType.UINT32, size, uint32Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                    LongPointer pUint64 = tens.GetTensorMutableDataULong().capacity(size);
                    Indexer uint64Indexer = LongIndexer.create(pUint64);
                    buffer = Nd4j.createBuffer(pUint64, DataType.UINT64, size, uint64Indexer);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
                    ShortPointer pBfloat16 = tens.GetTensorMutableDataShort().capacity(size);
                    Indexer bfloat16Indexer = ShortIndexer.create(pBfloat16);
                    buffer = Nd4j.createBuffer(pBfloat16, DataType.BFLOAT16, size, bfloat16Indexer);
                    break;
                default:
                    throw new RuntimeException("Unsupported data type encountered");
            }
            return buffer;
        }
    }

}
