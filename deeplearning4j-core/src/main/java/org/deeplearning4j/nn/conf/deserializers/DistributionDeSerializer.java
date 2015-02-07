package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import org.apache.commons.math3.distribution.RealDistribution;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.util.Dl4jReflection;
import org.json.JSONObject;

import java.io.IOException;
import java.io.StringReader;
import java.util.Properties;

/**
 * De serializer.
 * Reads value as :
 * class name \t properties
 *
 * @author Adam Gibson
 */
public class DistributionDeSerializer extends JsonDeserializer<RealDistribution> {
    @Override
    public RealDistribution deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String val = node.textValue();
        String[] split = val.split("\t");
        try {
            Class<? extends RealDistribution> clazz2 = (Class<? extends RealDistribution>) Class.forName(split[0]);
            RealDistribution ret =  clazz2.newInstance();
            Properties props = new Properties();
            props.load(new StringReader(split[1]));
            Dl4jReflection.setProperties(ret,props);
            return ret;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }
}
