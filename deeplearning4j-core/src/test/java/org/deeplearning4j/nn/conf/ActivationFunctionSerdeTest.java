package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.deeplearning4j.nn.conf.deserializers.ActivationFunctionDeSerializer;
import org.deeplearning4j.nn.conf.serializers.ActivationFunctionSerializer;
import org.junit.Test;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class ActivationFunctionSerdeTest {

    @Test
    public void testSerde() throws Exception {
        ActivationFunction softmax = Activations.softmax();
        ObjectMapper mapper = new ObjectMapper();
        SimpleModule module = new SimpleModule();
        module.addDeserializer(ActivationFunction.class,new ActivationFunctionDeSerializer());
        module.addSerializer(ActivationFunction.class,new ActivationFunctionSerializer());
        mapper.registerModule(module);
        String val = mapper.writeValueAsString(softmax);
        assertTrue(val.contains(softmax.getClass().getName() + ":" + false));
    }



}
