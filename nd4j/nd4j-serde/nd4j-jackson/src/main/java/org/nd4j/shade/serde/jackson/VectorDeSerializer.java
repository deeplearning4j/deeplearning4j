package org.nd4j.shade.serde.jackson;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * @author Adam Gibson
 */

public class VectorDeSerializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        JsonNode arr = node.get("dataBuffer");
        int rank = node.get("rankField").asInt();
        int numElements = node.get("numElements").asInt();
        int offset = node.get("offsetField").asInt();
        JsonNode shape = node.get("shapeField");
        JsonNode stride = node.get("strideField");
        String type = node.get("typeField").asText();
        int[] realShape = new int[rank];
        int[] realStride = new int[rank];
        DataBuffer buff = Nd4j.createBuffer(numElements);
        for (int i = 0; i < numElements; i++) {
            buff.put(i, arr.get(i).asDouble());
        }

        String ordering = node.get("orderingField").asText();
        for (int i = 0; i < rank; i++) {
            realShape[i] = shape.get(i).asInt();
            realStride[i] = stride.get(i).asInt();
        }

        INDArray ret = type.equals("real") ? Nd4j.create(buff, realShape, realStride, offset, ordering.charAt(0))
                        : Nd4j.createComplex(buff, realShape, realStride, offset, ordering.charAt(0));
        return ret;
    }
}
