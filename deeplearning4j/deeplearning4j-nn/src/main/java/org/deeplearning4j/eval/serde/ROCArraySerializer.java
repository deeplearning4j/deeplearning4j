package org.deeplearning4j.eval.serde;

import org.deeplearning4j.eval.ROC;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * Custom Jackson serializer for ROC[]. Simply delegates to {@link ROCSerializer} internally.
 *
 * @author Alex Black
 */
public class ROCArraySerializer extends JsonSerializer<ROC[]> {
    private static final ROCSerializer serializer = new ROCSerializer();

    @Override
    public void serialize(ROC[] rocs, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException, JsonProcessingException {
        jsonGenerator.writeStartArray();
        for (ROC r : rocs) {
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("@class", ROC.class.getName());
            serializer.serialize(r, jsonGenerator, serializerProvider);
            jsonGenerator.writeEndObject();
        }
        jsonGenerator.writeEndArray();
    }
}
