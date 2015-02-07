package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.distribution.RealDistribution;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.util.Dl4jReflection;

import java.io.IOException;

/**
 *
 * Write the field as follows:
 * dist : value \t properties
 * @author Adam Gibson
 */
public class DistributionSerializer extends JsonSerializer<RealDistribution> {
    @Override
    public void serialize(RealDistribution value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        try {
            jgen.writeStringField("dist",value.getClass().getName() + "\t" + Dl4jReflection.getFieldsAsProperties(value,null));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}
