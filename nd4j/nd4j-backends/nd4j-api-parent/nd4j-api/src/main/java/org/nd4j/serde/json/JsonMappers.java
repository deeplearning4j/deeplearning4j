/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.serde.json;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.common.primitives.serde.JsonDeserializerAtomicBoolean;
import org.nd4j.common.primitives.serde.JsonDeserializerAtomicDouble;
import org.nd4j.common.primitives.serde.JsonSerializerAtomicBoolean;
import org.nd4j.common.primitives.serde.JsonSerializerAtomicDouble;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

/**
 * JSON mappers for serializing/deserializing objects
 *
 * @author Alex Black
 */
@Slf4j
public class JsonMappers {

    private static ObjectMapper jsonMapper = configureMapper(new ObjectMapper());
    private static ObjectMapper yamlMapper = configureMapper(new ObjectMapper(new YAMLFactory()));

    /**
     * @return The default/primary ObjectMapper for deserializing JSON objects
     */
    public static ObjectMapper getMapper(){
        return jsonMapper;
    }

    /**
     * @return The default/primary ObjectMapper for deserializing JSON objects
     */
    public static ObjectMapper getYamlMapper(){
        return jsonMapper;
    }

    private static ObjectMapper configureMapper(ObjectMapper ret) {
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, false);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        SimpleModule atomicModule = new SimpleModule();
        atomicModule.addSerializer(AtomicDouble.class, new JsonSerializerAtomicDouble());
        atomicModule.addSerializer(AtomicBoolean.class, new JsonSerializerAtomicBoolean());
        atomicModule.addDeserializer(AtomicDouble.class, new JsonDeserializerAtomicDouble());
        atomicModule.addDeserializer(AtomicBoolean.class, new JsonDeserializerAtomicBoolean());
        ret.registerModule(atomicModule);
        //Serialize fields only, not using getters
        ret.setVisibilityChecker(ret.getSerializationConfig().getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.ANY)
        );
        return ret;
    }
}
