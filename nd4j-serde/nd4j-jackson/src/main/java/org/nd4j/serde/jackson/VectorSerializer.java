package org.nd4j.serde.jackson;


import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class VectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeStartObject();
        DataBuffer view = indArray.data();
        jsonGenerator.writeArrayFieldStart("data");
        for(int i = 0; i < view.length(); i++) {
            jsonGenerator.writeNumber(view.getDouble(i));
        }

        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("shape");
        for(int i = 0; i < indArray.rank(); i++) {
            jsonGenerator.writeNumber(indArray.size(i));
        }
        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("stride");
        for(int i = 0; i < indArray.rank(); i++)
            jsonGenerator.writeNumber(indArray.stride(i));
        jsonGenerator.writeEndArray();

        jsonGenerator.writeNumberField("offset", indArray.offset());
        jsonGenerator.writeStringField("type",indArray instanceof IComplexNDArray ? "complex" : "real");
        jsonGenerator.writeNumberField("rank",indArray.rank());
        jsonGenerator.writeNumberField("numElements",view.length());
        jsonGenerator.writeStringField("ordering",String.valueOf(indArray.ordering()));
        jsonGenerator.writeEndObject();
    }
}
