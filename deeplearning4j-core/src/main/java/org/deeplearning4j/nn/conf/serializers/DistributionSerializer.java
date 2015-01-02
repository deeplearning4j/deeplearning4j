package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.distribution.RealDistribution;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.io.IOException;

/**
 *
 *
 * @author Adam Gibson
 */
public class DistributionSerializer extends JsonSerializer<RealDistribution> {
    @Override
    public void serialize(RealDistribution value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
      jgen.writeStringField("distclass",value.getClass().getName());


    }
}
