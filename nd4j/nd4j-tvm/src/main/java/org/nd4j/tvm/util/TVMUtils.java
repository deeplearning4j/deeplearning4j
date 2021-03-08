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
package org.nd4j.tvm.util;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.tvm.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.tvm.global.tvm_runtime.*;
import static org.nd4j.linalg.api.buffer.DataType.*;

public class TVMUtils {

    /**
     * Return a {@link DataType}
     * for the tvm data type
     * @param dataType the equivalent nd4j data type
     * @return
     */
    public static DataType dataTypeForTvmType(DLDataType dataType) {
        if(dataType.code() == kDLInt && dataType.bits() == 8) {
            return INT8;
        } else if(dataType.code() == kDLInt && dataType.bits() == 16) {
            return INT16;
        } else if(dataType.code() == kDLInt && dataType.bits() == 32) {
            return INT32;
        } else if(dataType.code() == kDLInt && dataType.bits() == 64) {
            return INT64;
        } else if(dataType.code() == kDLUInt && dataType.bits() == 8) {
            return UINT8;
        } else if(dataType.code() == kDLUInt && dataType.bits() == 16) {
            return UINT16;
        } else if(dataType.code() == kDLUInt && dataType.bits() == 32) {
            return UINT32;
        } else if(dataType.code() == kDLUInt && dataType.bits() == 64) {
            return UINT64;
        } else if(dataType.code() == kDLFloat && dataType.bits() == 16) {
            return FLOAT16;
        } else if(dataType.code() == kDLFloat && dataType.bits() == 32) {
            return FLOAT;
        } else if(dataType.code() == kDLFloat && dataType.bits() == 64) {
            return DOUBLE;
        } else if(dataType.code() == kDLBfloat && dataType.bits() == 16) {
            return BFLOAT16;
        } else
            throw new IllegalArgumentException("Illegal data type code " + dataType.code() + " with bits " + dataType.bits());
    }

    /**
     * Convert the tvm type for the given data type
     * @param dataType
     * @return
     */
    public static DLDataType tvmTypeForDataType(DataType dataType) {
        if(dataType == INT8) {
            return new DLDataType().code((byte)kDLInt).bits((byte)8).lanes((short)1);
        } else if(dataType == INT16) {
            return new DLDataType().code((byte)kDLInt).bits((byte)16).lanes((short)1);
        } else if(dataType == INT32) {
            return new DLDataType().code((byte)kDLInt).bits((byte)32).lanes((short)1);
        } else if(dataType == INT64) {
            return new DLDataType().code((byte)kDLInt).bits((byte)64).lanes((short)1);
        } else if(dataType == UINT8) {
            return new DLDataType().code((byte)kDLUInt).bits((byte)8).lanes((short)1);
        } else if(dataType == UINT16) {
            return new DLDataType().code((byte)kDLUInt).bits((byte)16).lanes((short)1);
        } else if(dataType == UINT32) {
            return new DLDataType().code((byte)kDLUInt).bits((byte)32).lanes((short)1);
        } else if(dataType == UINT64) {
            return new DLDataType().code((byte)kDLUInt).bits((byte)64).lanes((short)1);
        } else if(dataType == FLOAT16) {
            return new DLDataType().code((byte)kDLFloat).bits((byte)16).lanes((short)1);
        } else if(dataType == FLOAT) {
            return new DLDataType().code((byte)kDLFloat).bits((byte)32).lanes((short)1);
        } else if(dataType == DOUBLE) {
            return new DLDataType().code((byte)kDLFloat).bits((byte)64).lanes((short)1);
        } else if(dataType == BFLOAT16) {
            return new DLDataType().code((byte)kDLBfloat).bits((byte)16).lanes((short)1);
        } else
            throw new IllegalArgumentException("Illegal data type " + dataType);
    }

    /**
     * Convert an tvm {@link DLTensor}
     *  in to an {@link INDArray}
     * @param value the tensor to convert
     * @return
     */
    public static INDArray getArray(DLTensor value) {
        DataType dataType = dataTypeForTvmType(value.dtype());
        LongPointer shape = value.shape();
        LongPointer stride = value.strides();
        long[] shapeConvert;
        if(shape != null) {
            shapeConvert = new long[value.ndim()];
            shape.get(shapeConvert);
        } else {
            shapeConvert = new long[]{1};
        }
        long[] strideConvert;
        if(stride != null) {
            strideConvert = new long[value.ndim()];
            stride.get(strideConvert);
        } else {
            strideConvert = Nd4j.getStrides(shapeConvert);
        }
        long size = 1;
        for (int i = 0; i < shapeConvert.length; i++) {
            size *= shapeConvert[i];
        }
        size *= value.dtype().bits() / 8;

        DataBuffer getBuffer = getDataBuffer(value,size);
        Preconditions.checkState(dataType.equals(getBuffer.dataType()),"Data type must be equivalent as specified by the tvm metadata.");
        return Nd4j.create(getBuffer,shapeConvert,strideConvert,0);
    }

    /**
     * Get an tvm tensor from an ndarray.
     * @param ndArray the ndarray to get the value from
     * @param ctx the {@link DLContext} to use.
     * @return
     */
    public static DLTensor getTensor(INDArray ndArray, DLContext ctx) {
        DLTensor ret = new DLTensor();
        ret.data(ndArray.data().pointer());
        ret.ctx(ctx);
        ret.ndim(ndArray.rank());
        ret.dtype(tvmTypeForDataType(ndArray.dataType()));
        ret.shape(new LongPointer(ndArray.shape()));
        ret.strides(new LongPointer(ndArray.stride()));
        ret.byte_offset(ndArray.offset());
        return ret;
    }

    /**
     * Get the data buffer from the given value
     * @param tens the values to get
     * @return the equivalent data buffer
     */
    public static DataBuffer getDataBuffer(DLTensor tens, long size) {
        DataBuffer buffer = null;
        DataType type = dataTypeForTvmType(tens.dtype());
        switch (type) {
            case BYTE:
                BytePointer pInt8 = new BytePointer(tens.data()).capacity(size);
                Indexer int8Indexer = ByteIndexer.create(pInt8);
                buffer = Nd4j.createBuffer(pInt8, type, size, int8Indexer);
                break;
            case SHORT:
                ShortPointer pInt16 = new ShortPointer(tens.data()).capacity(size);
                Indexer int16Indexer = ShortIndexer.create(pInt16);
                buffer = Nd4j.createBuffer(pInt16, type, size, int16Indexer);
                break;
            case INT:
                IntPointer pInt32 = new IntPointer(tens.data()).capacity(size);
                Indexer int32Indexer = IntIndexer.create(pInt32);
                buffer = Nd4j.createBuffer(pInt32, type, size, int32Indexer);
                break;
            case LONG:
                LongPointer pInt64 = new LongPointer(tens.data()).capacity(size);
                Indexer int64Indexer = LongIndexer.create(pInt64);
                buffer = Nd4j.createBuffer(pInt64, type, size, int64Indexer);
                break;
            case UBYTE:
                BytePointer pUint8 = new BytePointer(tens.data()).capacity(size);
                Indexer uint8Indexer = UByteIndexer.create(pUint8);
                buffer = Nd4j.createBuffer(pUint8, type, size, uint8Indexer);
                break;
            case UINT16:
                ShortPointer pUint16 = new ShortPointer(tens.data()).capacity(size);
                Indexer uint16Indexer = UShortIndexer.create(pUint16);
                buffer = Nd4j.createBuffer(pUint16, type, size, uint16Indexer);
                break;
            case UINT32:
                IntPointer pUint32 = new IntPointer(tens.data()).capacity(size);
                Indexer uint32Indexer = UIntIndexer.create(pUint32);
                buffer = Nd4j.createBuffer(pUint32, type, size, uint32Indexer);
                break;
            case UINT64:
                LongPointer pUint64 = new LongPointer(tens.data()).capacity(size);
                Indexer uint64Indexer = LongIndexer.create(pUint64);
                buffer = Nd4j.createBuffer(pUint64, type, size, uint64Indexer);
                break;
            case HALF:
                ShortPointer pFloat16 = new ShortPointer(tens.data()).capacity(size);
                Indexer float16Indexer = HalfIndexer.create(pFloat16);
                buffer = Nd4j.createBuffer(pFloat16, type, size, float16Indexer);
                break;
            case FLOAT:
                FloatPointer pFloat =  new FloatPointer(tens.data()).capacity(size);
                FloatIndexer floatIndexer = FloatIndexer.create(pFloat);
                buffer = Nd4j.createBuffer(pFloat, type, size, floatIndexer);
                break;
            case DOUBLE:
                DoublePointer pDouble =  new DoublePointer(tens.data()).capacity(size);
                Indexer doubleIndexer = DoubleIndexer.create(pDouble);
                buffer = Nd4j.createBuffer(pDouble, type, size, doubleIndexer);
                break;
            case BFLOAT16:
                ShortPointer pBfloat16 = new ShortPointer(tens.data()).capacity(size);
                Indexer bfloat16Indexer = Bfloat16Indexer.create(pBfloat16);
                buffer = Nd4j.createBuffer(pBfloat16, type, size, bfloat16Indexer);
                break;
            default:
                throw new RuntimeException("Unsupported data type encountered");
        }
        return buffer;
    }

}
