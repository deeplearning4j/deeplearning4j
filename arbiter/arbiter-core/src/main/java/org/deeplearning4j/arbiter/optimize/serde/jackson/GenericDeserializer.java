package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;

/**
 * Created by Alex on 15/02/2017.
 */
public class GenericDeserializer extends JsonDeserializer<Object> {
    @Override
    public Object deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
        JsonNode node = p.getCodec().readTree(p);
        String className = node.get("@class").asText();
        Class<?> c;
        try {
            c = Class.forName(className);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        JsonNode valueNode = node.get("value");
        Object o = new ObjectMapper().treeToValue(valueNode, c);
        return o;
    }
}
