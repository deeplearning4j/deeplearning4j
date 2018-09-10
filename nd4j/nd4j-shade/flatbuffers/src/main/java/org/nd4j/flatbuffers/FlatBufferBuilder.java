package org.nd4j.flatbuffers;

import java.nio.ByteBuffer;

public class FlatBufferBuilder extends com.google.flatbuffers.FlatBufferBuilder {

    public FlatBufferBuilder(int initial_size, com.google.flatbuffers.FlatBufferBuilder.ByteBufferFactory bb_factory) {
        super(initial_size, bb_factory);
    }

    public FlatBufferBuilder(int initial_size) {
        super(initial_size);
    }

    public FlatBufferBuilder(){
        super();
    }

    public FlatBufferBuilder(ByteBuffer existing_bb, com.google.flatbuffers.FlatBufferBuilder.ByteBufferFactory bb_factory) {
        super(existing_bb, bb_factory);
    }

    public FlatBufferBuilder(ByteBuffer existing_bb) {
        super(existing_bb);
    }
    public int createString(String s){
        return super.createString(s);
    }

}
