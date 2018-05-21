package org.nd4j.shade.serde.jackson.shaded;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class NDArraySerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException {
        String toBase64 = Nd4jBase64.base64String(indArray);
        jsonGenerator.writeStartObject();
        jsonGenerator.writeStringField("array", toBase64);
        jsonGenerator.writeEndObject();
    }
}
