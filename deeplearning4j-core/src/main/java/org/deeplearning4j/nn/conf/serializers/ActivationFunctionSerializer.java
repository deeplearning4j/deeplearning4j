package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.SoftMax;

import java.io.IOException;
import java.lang.reflect.Field;

/**
 * Handles activation function serde
 * @author Adam Gibson
 */
public class ActivationFunctionSerializer extends JsonSerializer<ActivationFunction> {
    @Override
    public void serialize(ActivationFunction value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        if(value instanceof SoftMax) {
            SoftMax max = (SoftMax) value;
            try {
                Field f = SoftMax.class.getDeclaredField("rows");
                f.setAccessible(true);
                boolean val = f.getBoolean(max);
                jgen.writeStringField("activationFunction", value.getClass().getName() + ":" + val);

            } catch (Exception e) {
                e.printStackTrace();
            }



        }
        else
            jgen.writeString(value.getClass().getName());


    }
}
