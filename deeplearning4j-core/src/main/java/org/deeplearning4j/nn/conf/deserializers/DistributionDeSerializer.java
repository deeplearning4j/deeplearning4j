package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import org.apache.commons.math3.distribution.RealDistribution;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.json.JSONObject;

import java.io.IOException;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class DistributionDeSerializer extends JsonDeserializer<RealDistribution> {
    @Override
    public RealDistribution deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        JsonNode node = jp.getCodec().readTree(jp);
        String val = node.textValue();
        JSONObject obj = new JSONObject(val);
        String clazz = obj.getString("distclass");
        obj.remove("distclass");
        try {
            Class<? extends RealDistribution> clazz2 = (Class<? extends RealDistribution>) Class.forName(clazz);
            RealDistribution ret = mapper.readValue(obj.toString(),clazz2);
            return ret;
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return null;
    }
}
