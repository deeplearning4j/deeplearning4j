package org.nd4j.shade.serde.jackson.ndarray;


import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class NDArraySerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.writeTxtString(indArray,bos);
        bos.flush();
        String toWrite = new String(bos.toByteArray());
        jsonGenerator.writeRaw(toWrite);
    }
}
