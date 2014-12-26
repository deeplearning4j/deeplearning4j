package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.deeplearning4j.optimize.api.StepFunction;

import java.io.IOException;

/**
 * Created by agibsonccc on 12/25/14.
 */
public class StepFunctionDeSerializer extends JsonDeserializer<StepFunction> {
    @Override
    public StepFunction deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String rngClazz = node.textValue();
        try {
            Class<? extends StepFunction> clazz = (Class<? extends StepFunction>) Class.forName(rngClazz);
            StepFunction gen = clazz.newInstance();
            return gen;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
