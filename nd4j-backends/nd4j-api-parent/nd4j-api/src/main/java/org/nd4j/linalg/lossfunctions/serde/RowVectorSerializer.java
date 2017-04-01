package org.nd4j.linalg.lossfunctions.serde;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * Simple JSON serializer for use in {@link org.nd4j.linalg.lossfunctions.ILossFunction} weight serialization.
 * Serializes an INDArray as a double[]
 *
 * @author Alex Black
 */
public class RowVectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray array, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException {
        if (array.isView()) {
            array = array.dup();
        }
        double[] dArr = array.data().asDouble();
        jsonGenerator.writeObject(dArr);
    }
}
