/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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
public class ActivationFunctionDeSerializer extends JsonDeserializer<ActivationFunction>  {
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
