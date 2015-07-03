package org.deeplearning4j.ui.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public class GradientSerializer extends JsonSerializer<Gradient> {
    @Override
    public void serialize(Gradient gradient, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeStartObject();
        for(Map.Entry<String,INDArray> entry : gradient.gradientForVariable().entrySet()) {
            jsonGenerator.writeArrayFieldStart(entry.getKey());
            INDArray view = entry.getValue().linearView();
            for(int i = 0; i < view.length(); i++) {
                jsonGenerator.writeNumber(view.getDouble(i));
            }

            jsonGenerator.writeEndArray();
        }

        jsonGenerator.writeEndObject();

    }

}

