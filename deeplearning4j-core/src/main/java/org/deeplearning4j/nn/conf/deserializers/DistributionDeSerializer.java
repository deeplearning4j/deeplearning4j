/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import org.deeplearning4j.util.Dl4jReflection;
import org.nd4j.linalg.api.rng.distribution.Distribution;

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
public class DistributionDeSerializer extends JsonDeserializer<Distribution> {
    @Override
    public Distribution deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String val = node.textValue();
        String[] split = val.split("\t");
        try {
            Class<? extends Distribution> clazz2 = (Class<? extends Distribution>) Class.forName(split[0]);
            Distribution ret =  clazz2.newInstance();
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
