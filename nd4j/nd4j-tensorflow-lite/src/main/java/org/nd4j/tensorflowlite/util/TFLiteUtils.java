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
package org.nd4j.tensorflowlite.util;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.tensorflowlite.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;
import static org.nd4j.linalg.api.buffer.DataType.*;

public class TFLiteUtils {

    /**
     * Return a {@link DataType}
     * for the tflite data type
     * @param dataType the equivalent nd4j data type
     * @return
     */
    public static DataType dataTypeForTfliteType(int dataType) {
        switch (dataType) {
            case kTfLiteFloat32: return FLOAT;
            case kTfLiteInt32:   return INT32;
            case kTfLiteUInt8:   return UINT8;
            case kTfLiteInt64:   return INT64;
            case kTfLiteString:  return UTF8;
            case kTfLiteBool:    return BOOL;
            case kTfLiteInt16:   return INT16;
            case kTfLiteInt8:    return INT8;
            case kTfLiteFloat16: return FLOAT16;
            case kTfLiteFloat64: return DOUBLE;
            case kTfLiteUInt64:  return UINT64;
            case kTfLiteUInt32:  return UINT32;
            case kTfLiteComplex64:
            case kTfLiteComplex128:
            case kTfLiteResource:
            case kTfLiteVariant:
            default: throw new IllegalArgumentException("Illegal data type " + dataType);
        }
    }

    /**
     * Convert the tflite type for the given data type
     * @param dataType
     * @return
     */
    public static int tfliteTypeForDataType(DataType dataType) {
        switch (dataType) {
            case DOUBLE: return kTfLiteFloat64;
            case FLOAT:  return kTfLiteFloat32;
            case HALF:   return kTfLiteFloat16;
            case LONG:   return kTfLiteInt64;
            case INT:    return kTfLiteInt32;
            case SHORT:  return kTfLiteInt16;
            case UBYTE:  return kTfLiteUInt8;
            case BYTE:   return kTfLiteInt8;
            case BOOL:   return kTfLiteBool;
            case UTF8:   return kTfLiteString;
            case UINT32: return kTfLiteUInt32;
            case UINT64: return kTfLiteUInt64;
            case COMPRESSED:
            case BFLOAT16:
            case UINT16:
            default: throw new IllegalArgumentException("Illegal data type " + dataType);
        }
    }

    /**
     * Convert a {@link TfLiteTensor}
     *  in to an {@link INDArray}
     * @param value the value to convert
     * @return
     */
    public static INDArray getArray(TfLiteTensor value) {
        DataType dataType = dataTypeForTfliteType(value.type());
        TfLiteIntArray shape = value.dims();
        long[] shapeConvert;
        if(shape != null) {
            shapeConvert = new long[shape.size()];
            for (int i = 0; i < shapeConvert.length; i++) {
                shapeConvert[i] = shape.data(i);
            }
        } else {
            shapeConvert = new long[]{1};
        }

        DataBuffer getBuffer = getDataBuffer(value);
        Preconditions.checkState(dataType.equals(getBuffer.dataType()),"Data type must be equivalent as specified by the tflite metadata.");
        return Nd4j.create(getBuffer,shapeConvert,Nd4j.getStrides(shapeConvert),0);
    }

    /**
     * Get the data buffer from the given tensor
     * @param tens the values to get
     * @return the equivalent data buffer
     */
    public static DataBuffer getDataBuffer(TfLiteTensor tens) {
        DataBuffer buffer = null;
        int type = tens.type();
        long size = tens.bytes();
        Pointer data = tens.data().data();
        switch (type) {
            case kTfLiteFloat32:
                FloatPointer pFloat = new FloatPointer(data).capacity(size / 4);
                FloatIndexer floatIndexer = FloatIndexer.create(pFloat);
                buffer = Nd4j.createBuffer(pFloat, DataType.FLOAT, size / 4, floatIndexer);
                break;
            case kTfLiteInt32:
                IntPointer pInt32 = new IntPointer(data).capacity(size / 4);
                Indexer int32Indexer = IntIndexer.create(pInt32);
                buffer = Nd4j.createBuffer(pInt32, DataType.INT32, size / 4, int32Indexer);
                break;
            case kTfLiteUInt8:
                BytePointer pUint8 = new BytePointer(data).capacity(size);
                Indexer uint8Indexer = ByteIndexer.create(pUint8);
                buffer = Nd4j.createBuffer(pUint8, DataType.UINT8, size, uint8Indexer);
                break;
            case kTfLiteInt64:
                LongPointer pInt64 = new LongPointer(data).capacity(size / 8);
                Indexer int64Indexer = LongIndexer.create(pInt64);
                buffer = Nd4j.createBuffer(pInt64, DataType.INT64, size / 8, int64Indexer);
                break;
            case kTfLiteString:
                BytePointer pString = new BytePointer(data).capacity(size);
                Indexer stringIndexer = ByteIndexer.create(pString);
                buffer = Nd4j.createBuffer(pString, DataType.INT8, size, stringIndexer);
                break;
            case kTfLiteBool:
                BoolPointer pBool = new BoolPointer(data).capacity(size);
                Indexer boolIndexer = BooleanIndexer.create(new BooleanPointer(pBool)); //Converting from JavaCPP Bool to Boolean here - C++ bool type size is not defined, could cause problems on some platforms
                buffer = Nd4j.createBuffer(pBool, DataType.BOOL, size, boolIndexer);
                break;
            case kTfLiteInt16:
                ShortPointer pInt16 = new ShortPointer(data).capacity(size / 2);
                Indexer int16Indexer = ShortIndexer.create(pInt16);
                buffer = Nd4j.createBuffer(pInt16, INT16, size / 2, int16Indexer);
                break;
            case kTfLiteInt8:
                BytePointer pInt8 = new BytePointer(data).capacity(size);
                Indexer int8Indexer = ByteIndexer.create(pInt8);
                buffer = Nd4j.createBuffer(pInt8, DataType.UINT8, size, int8Indexer);
                break;
            case kTfLiteFloat16:
                ShortPointer pFloat16 = new ShortPointer(data).capacity(size / 2);
                Indexer float16Indexer = ShortIndexer.create(pFloat16);
                buffer = Nd4j.createBuffer(pFloat16, DataType.FLOAT16, size / 2, float16Indexer);
                break;
            case kTfLiteFloat64:
                DoublePointer pDouble = new DoublePointer(data).capacity(size / 8);
                Indexer doubleIndexer = DoubleIndexer.create(pDouble);
                buffer = Nd4j.createBuffer(pDouble, DataType.DOUBLE, size / 8, doubleIndexer);
                break;
            case kTfLiteUInt64:
                LongPointer pUint64 = new LongPointer(data).capacity(size / 8);
                Indexer uint64Indexer = LongIndexer.create(pUint64);
                buffer = Nd4j.createBuffer(pUint64, DataType.UINT64, size / 8, uint64Indexer);
                break;
            case kTfLiteUInt32:
                IntPointer pUint32 = new IntPointer(data).capacity(size / 4);
                Indexer uint32Indexer = IntIndexer.create(pUint32);
                buffer = Nd4j.createBuffer(pUint32, DataType.UINT32, size / 4, uint32Indexer);
                break;
            case kTfLiteComplex64:
            case kTfLiteComplex128:
            case kTfLiteResource:
            case kTfLiteVariant:
            default:
                throw new RuntimeException("Unsupported data type encountered");
        }
        return buffer;
    }

}
