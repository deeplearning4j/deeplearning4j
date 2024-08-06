/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.serde;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.serde.legacy.LegacyJsonFormat;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

@Slf4j
public class JsonMappers {

    private static ThreadLocal<ObjectMapper> jsonMapper = ThreadLocal.withInitial(() -> {
        ObjectMapper om = new ObjectMapper();
        JsonMapperUtil.configureMapper(om);
        return om;
    });
    private static ThreadLocal<ObjectMapper> yamlMapper = ThreadLocal.withInitial(() -> {
        ObjectMapper om = new ObjectMapper();
        JsonMapperUtil.configureMapper(om);
        return om;
    });


    private static ThreadLocal<ObjectMapper> legacyMapper  = ThreadLocal.withInitial(() -> {
        ObjectMapper mapper = LegacyJsonFormat.getMapper100alpha();
        JsonMapperUtil.configureMapper(mapper);
        return mapper;
    });;


    /**
     * @return The default/primary ObjectMapper for deserializing JSON network configurations in DL4J
     */
    public static ObjectMapper getMapper() {
        if(jsonMapper.get() == null) {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonMapperUtil.configureMapper(objectMapper);
            jsonMapper.set(objectMapper);
        }
        return jsonMapper.get();
    }

    public static  ObjectMapper getLegacyMapper() {
        if(legacyMapper.get() == null) {
            ObjectMapper mapper = LegacyJsonFormat.getMapper100alpha();
            JsonMapperUtil.configureMapper(mapper);
            legacyMapper.set(mapper);
        }
        return legacyMapper.get();
    }

    /**
     * @return The default/primary ObjectMapper for deserializing network configurations in DL4J (YAML format)
     */
    public static ObjectMapper getMapperYaml() {
        if(jsonMapper.get() == null) {
            jsonMapper.set(new ObjectMapper(new YAMLFactory()));
            JsonMapperUtil.configureMapper(jsonMapper.get());
        }
        return yamlMapper.get();
    }

}
