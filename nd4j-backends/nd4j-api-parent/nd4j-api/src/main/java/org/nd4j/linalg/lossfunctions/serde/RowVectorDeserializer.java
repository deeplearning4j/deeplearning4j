package org.nd4j.linalg.lossfunctions.serde;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Simple JSON deserializer for use in {@link org.nd4j.linalg.lossfunctions.ILossFunction} loss function weight serialization.
 * Used in conjunction with {@link RowVectorSerializer}
 *
 * @author Alex Black
 */
public class RowVectorDeserializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
                    throws IOException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        if (node == null)
            return null;
        int size = node.size();
        double[] d = new double[size];
        for (int i = 0; i < size; i++) {
            d[i] = node.get(i).asDouble();
        }

        return Nd4j.create(d);
    }
}
