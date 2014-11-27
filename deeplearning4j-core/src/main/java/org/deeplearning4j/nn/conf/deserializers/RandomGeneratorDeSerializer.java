package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.IOException;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class RandomGeneratorDeSerializer extends JsonDeserializer<RandomGenerator> {
    @Override
    public RandomGenerator deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String rngClazz = node.textValue();
        int seed = node.asInt();
        JsonNode rngClazz1 = node.get("rng");
        try {
            Class<? extends RandomGenerator> clazz = (Class<? extends RandomGenerator>) Class.forName(rngClazz);
            RandomGenerator gen = clazz.newInstance();
            gen.setSeed(seed);
            return gen;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
