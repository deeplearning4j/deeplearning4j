package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.distribution.RealDistribution;

import java.io.IOException;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class DistributionSerializer extends JsonSerializer<RealDistribution> {
    @Override
    public void serialize(RealDistribution value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        String write = mapper.writeValueAsString(value).substring(1);
        String all = "{\"distclass\":\"" + value.getClass().getName() + "\"," + write;
        jgen.writeString(all);


    }
}
