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
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import org.deeplearning4j.optimize.api.StepFunction;

import java.io.IOException;

/**
 * Created by agibsonccc on 12/25/14.
 */
public class StepFunctionDeSerializer extends JsonDeserializer<StepFunction> {
    @Override
    public StepFunction deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String rngClazz = node.textValue();
        try {
            Class<? extends StepFunction> clazz = (Class<? extends StepFunction>) Class.forName(rngClazz);
            StepFunction gen = clazz.newInstance();
            return gen;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
