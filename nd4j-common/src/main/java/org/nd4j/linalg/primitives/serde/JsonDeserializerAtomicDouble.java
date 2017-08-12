package org.nd4j.linalg.primitives.serde;

import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

public class JsonDeserializerAtomicDouble extends JsonDeserializer<AtomicDouble> {
    @Override
    public AtomicDouble deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        double value = node.asDouble();
        return new AtomicDouble(value);
    }
}
