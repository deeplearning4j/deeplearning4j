/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class DataBufferStruct extends Struct {

    @Getter
    private DataBuffer dataBuffer;

    public DataBufferStruct(DataBuffer dataBuffer) {
        this.dataBuffer = dataBuffer;
    }

    public DataBufferStruct(ByteBuffer byteBuffer,int offset) {
        __init(offset,byteBuffer);
    }

    public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
    public DataBufferStruct __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

    /**
     * Create a {@link DataBuffer} from a
     * byte buffer. This is meant to be used with flatbuffers
     * @param bb the flat buffers buffer
     * @param bb_pos the position to start from
     * @param type the type of buffer to create
     * @param length the length of the buffer to create
     * @return the created databuffer
     */
    public static DataBuffer createFromByteBuffer(ByteBuffer bb, int bb_pos, DataType type, int length) {
        bb.order(ByteOrder.LITTLE_ENDIAN);
        int elementSize = DataTypeUtil.lengthForDtype(type);
        DataBuffer ret = Nd4j.createBuffer(ByteBuffer.allocateDirect(length *   elementSize),type,length,0);

        switch(type) {
            case DOUBLE:
                for(int i = 0; i < ret.length(); i++) {
                    double doubleGet = bb.getDouble(bb.capacity() - bb_pos + (i * elementSize));
                    ret.put(i,doubleGet);
                }
                break;
            case FLOAT:
                for(int i = 0; i < ret.length(); i++) {
                    float floatGet = bb.getFloat(bb.capacity() - bb_pos + (i * elementSize));
                    ret.put(i,floatGet);
                }
                break;
            case INT:
                for(int i = 0; i < ret.length(); i++) {
                    int intGet = bb.getInt(bb.capacity() - bb_pos  + (i * elementSize));
                    ret.put(i,intGet);
                }
                break;
            case LONG:
                for(int i = 0; i < ret.length(); i++) {
                    long longGet = bb.getLong(bb.capacity() - bb_pos  + (i * elementSize));
                    ret.put(i,longGet);
                }
                break;
        }

        return ret;
    }


    /**
     * Create a data buffer struct within
     * the passed in {@link FlatBufferBuilder}
     * @param bufferBuilder the existing flatbuffer
     *                      to use to serialize the {@link DataBuffer}
     * @param create the databuffer to serialize
     * @return an int representing the offset of the buffer
     */
    public static int createDataBufferStruct(FlatBufferBuilder bufferBuilder,DataBuffer create) {
        bufferBuilder.prep(create.getElementSize(), (int) create.length() * create.getElementSize());
        for(int i = (int) (create.length() - 1); i >= 0; i--) {
            switch(create.dataType()) {
                case DOUBLE:
                    double putDouble = create.getDouble(i);
                    bufferBuilder.putDouble(putDouble);
                    break;
                case FLOAT:
                    float putFloat = create.getFloat(i);
                    bufferBuilder.putFloat(putFloat);
                    break;
                case INT:
                    int putInt = create.getInt(i);
                    bufferBuilder.putInt(putInt);
                    break;
                case LONG:
                    long putLong = create.getLong(i);
                    bufferBuilder.putLong(putLong);
            }
        }

        return bufferBuilder.offset();

    }
}
