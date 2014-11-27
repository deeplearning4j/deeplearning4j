package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.IOException;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class RandomGeneratorSerializer extends JsonSerializer<RandomGenerator> {
    @Override
    public void serialize(RandomGenerator value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        jgen.writeStringField("rng",value.getClass().getName());
    }
}
