package org.nd4j.shade.serde.dwjackson.ndarray;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * @author Adam Gibson
 */

public class NDArrayDeSerializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String raw = node.asText();
        ByteArrayInputStream bis = new ByteArrayInputStream(raw.getBytes());
        INDArray ret = Nd4j.readTxtString(bis);
        return ret;
    }
}
