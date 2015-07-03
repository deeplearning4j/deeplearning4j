package org.deeplearning4j.ui.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class ModelSerializer extends JsonSerializer<Model> {
    @Override
    public void serialize(Model model, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeStartObject();
        for(String entry : model.conf().variables()) {
            jsonGenerator.writeArrayFieldStart(entry);
            INDArray view = model.getParam(entry).linearView();
            for(int i = 0; i < view.length(); i++) {
                jsonGenerator.writeNumber(view.getDouble(i));
            }

            jsonGenerator.writeEndArray();
        }

        jsonGenerator.writeEndObject();
    }
}
