package org.deeplearning4j.ui.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class VectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        INDArray view = indArray.linearView();
        jsonGenerator.writeStartArray();
        for(int i = 0; i < view.length(); i++) {
            jsonGenerator.writeNumber(view.getDouble(i));
        }

        jsonGenerator.writeEndArray();
    }
}
