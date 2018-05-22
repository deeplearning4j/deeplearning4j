package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.IOException;

/**
 * Deserialize either an int[] to an int[], or a single int x to int[]{x,x}
 *
 * Used when supporting a configuration format change from single int value to int[], as for Upsampling2D
 * between 1.0.0-alpha and 1.0.0-beta
 *
 * @author Alex Black
 */
public class LegacyIntArrayDeserializer extends JsonDeserializer<int[]> {
    @Override
    public int[] deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode n = jp.getCodec().readTree(jp);
        if(n.isArray()){
            ArrayNode an = (ArrayNode)n;
            int size = an.size();
            int[] out = new int[size];
            for( int i=0; i<size; i++ ){
                out[i] = an.get(i).asInt();
            }
            return out;
        } else if(n.isNumber()){
            int v = n.asInt();
            return new int[]{v,v};
        } else {
            throw new IllegalStateException("Could not deserialize value: " + n);
        }
    }
}
