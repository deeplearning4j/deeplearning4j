package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.deeplearning4j.optimize.api.StepFunction;

import java.io.IOException;

/**
 * Created by agibsonccc on 12/25/14.
 */
public class StepFunctionSerializer extends JsonSerializer<StepFunction> {
    @Override
    public void serialize(StepFunction value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        jgen.writeStringField("stepFunction",value.getClass().getName());

    }
}
