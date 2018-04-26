package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * Created by Alex on 15/02/2017.
 */
public class GenericSerializer extends JsonSerializer<Object> {
    @Override
    public void serialize(Object o, JsonGenerator j, SerializerProvider serializerProvider)
                    throws IOException, JsonProcessingException {
        j.writeStartObject();
        j.writeStringField("@class", o.getClass().getName());
        j.writeObjectField("value", o);
        j.writeEndObject();
    }
}
