package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;

import java.io.IOException;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class ActivationFunctionDeSerializer extends JsonDeserializer<ActivationFunction> {
    @Override
    public ActivationFunction deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String val1 = node.textValue();
        if(val1.contains("SoftMax")) {
            try {
                String[] valSplit = val1.split(":");
                boolean val2 = Boolean.parseBoolean(valSplit[1]);
                if(val2)
                    return Activations.softMaxRows();
                return Activations.softmax();

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else {
            try {
                Class<? extends ActivationFunction> clazz = (Class<? extends ActivationFunction>) Class.forName(val1);
                return clazz.newInstance();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return null;
    }
}
