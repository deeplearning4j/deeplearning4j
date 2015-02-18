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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;

import java.io.IOException;
import java.lang.reflect.Constructor;

/**
 * De serializes a layer factory.
 * The layer factory's value consists of a comma separated class list containing
 * the layer factory class name followed by the layer class name it represents
 * @author Adam Gibson
 */
public class LayerFactoryDeSerializer extends JsonDeserializer<LayerFactory> {
    @Override
    public LayerFactory deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String val1 = node.textValue();
        String[] split = val1.split(",");
        String clazz1 = split[0];
        String layerClazz = split[1];

        try {
            Class<? extends LayerFactory> clazz = (Class<? extends LayerFactory>) Class.forName(clazz1);
            Class<? extends Layer> layerClazz1 = (Class<? extends Layer>) Class.forName(layerClazz);
            Constructor<?> c = clazz.getConstructor(Class.class);
            return (LayerFactory) c.newInstance(layerClazz1);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
