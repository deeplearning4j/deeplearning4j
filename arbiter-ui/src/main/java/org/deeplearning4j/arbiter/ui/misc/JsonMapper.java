/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.ui.misc;


import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.IOException;

/**
 * Simple class for serializing to / deserializing from JSON, using default ObjectMapper configuration
 *
 * @author Alex Black
 */
public class JsonMapper {

    private static final ObjectMapper MAPPER = getMapper();

    private JsonMapper(){

    }

    private static ObjectMapper getMapper(){
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        mapper.setVisibilityChecker(mapper.getSerializationConfig().getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.NONE));

        return mapper;
    }

    public static String asJson(Object o){
        try{
            return getMapper().writeValueAsString(o);
        } catch (JsonProcessingException e){
            throw new RuntimeException(e);
        }
    }

    public static <T> T fromJson(String json, Class<? extends T> type){
        try{
            return getMapper().readValue(json, type);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static ObjectMapper getInstance(){
        return MAPPER;
    }

}
