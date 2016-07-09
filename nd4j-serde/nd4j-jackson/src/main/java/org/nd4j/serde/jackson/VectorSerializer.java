package org.nd4j.serde.jackson;


import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class VectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        if (indArray.isView())
            indArray = indArray.dup(indArray.ordering());
        jsonGenerator.writeStartObject();
        DataBuffer view = indArray.data();
        jsonGenerator.writeArrayFieldStart("dataBuffer");
        for(int i = 0; i < view.length(); i++) {
            jsonGenerator.writeNumber(view.getDouble(i));
        }

        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("shapeField");
        for(int i = 0; i < indArray.rank(); i++) {
            jsonGenerator.writeNumber(indArray.size(i));
        }
        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("strideField");
        for(int i = 0; i < indArray.rank(); i++)
            jsonGenerator.writeNumber(indArray.stride(i));
        jsonGenerator.writeEndArray();

        jsonGenerator.writeNumberField("offsetField", indArray.offset());
        jsonGenerator.writeStringField("typeField", indArray instanceof IComplexNDArray ? "complex" : "real");
        jsonGenerator.writeNumberField("rankField", indArray.rank());
        jsonGenerator.writeNumberField("numElements", view.length());
        jsonGenerator.writeStringField("orderingField", String.valueOf(indArray.ordering()));
        jsonGenerator.writeEndObject();
    }
}
