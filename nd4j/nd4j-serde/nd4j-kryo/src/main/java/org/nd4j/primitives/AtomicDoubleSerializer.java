package org.nd4j.primitives;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.primitives.AtomicDouble;

/**
 * Serializer for AtomicDouble (needs a serializer due to long field being transient...)
 */
public class AtomicDoubleSerializer extends Serializer<AtomicDouble> {
    @Override
    public void write(Kryo kryo, Output output, AtomicDouble a) {
        output.writeDouble(a.get());
    }

    @Override
    public AtomicDouble read(Kryo kryo, Input input, Class<AtomicDouble> a) {
        return new AtomicDouble(input.readDouble());
    }
}
