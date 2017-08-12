package org.nd4j.linalg.primitives.serde;

import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

public class JsonSerializerAtomicBoolean extends JsonSerializer<AtomicBoolean> {
    @Override
    public void serialize(AtomicBoolean atomicDouble, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
        jsonGenerator.writeBoolean(atomicDouble.get());
    }
}
