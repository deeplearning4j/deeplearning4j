package org.nd4j.shade.serde.jackson.ndarray;


import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

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
