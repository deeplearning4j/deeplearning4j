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

package org.arbiter.nn.conf.deserializers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.io.StringReader;
import java.util.Properties;
import org.arbiter.optimize.api.IterationListener;
import org.arbiter.util.Dl4jReflection;

/**
 * Created by agibsonccc on 2/10/15.
 */
public class IterationListenerDeSerializer extends JsonDeserializer<IterationListener> {
    @Override
    public IterationListener deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        JsonNode node = jp.getCodec().readTree(jp);
        String val = node.asText();
        String[] split = val.split("\t");
       if(split.length >= 2) {
           try {
               Class<? extends IterationListener> clazz2 = (Class<? extends IterationListener>) Class.forName(split[0]);
               IterationListener ret =  clazz2.newInstance();
               Properties props = new Properties();
               props.load(new StringReader(split[1]));
               Dl4jReflection.setProperties(ret, props);
               return ret;
           } catch (Exception e) {
               e.printStackTrace();
           }
       }


        return null;
    }
}
