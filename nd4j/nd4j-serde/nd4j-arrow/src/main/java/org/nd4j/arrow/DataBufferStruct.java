package org.nd4j.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class DataBufferStruct extends Struct {

    private DataBuffer dataBuffer;

    public DataBufferStruct(DataBuffer dataBuffer) {
        this.dataBuffer = dataBuffer;
    }

    public DataBufferStruct(ByteBuffer byteBuffer,int offset) {
        __init(offset,byteBuffer);
    }

    public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
    public DataBufferStruct __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

    public static DataBuffer createFromByteBuffer(ByteBuffer bb,int bb_pos,DataBuffer.Type type,int length,int elementSize) {
        bb.order(ByteOrder.LITTLE_ENDIAN);
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
