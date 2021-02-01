/*
 *  ******************************************************************************
 *  *
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

package org.datavec.api.transform.serde;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.transform.serde.legacy.LegacyJsonFormat;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

/**
 * JSON mappers for deserializing neural net configurations, etc.
 *
 * @author Alex Black
 */
@Slf4j
public class JsonMappers {

    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;
    private static ObjectMapper legacyMapper;       //For 1.0.0-alpha and earlier TransformProcess etc

    static {
        jsonMapper = new ObjectMapper();
        yamlMapper = new ObjectMapper(new YAMLFactory());
        configureMapper(jsonMapper);
        configureMapper(yamlMapper);
    }

    public static synchronized ObjectMapper getLegacyMapper(){
        if(legacyMapper == null){
            legacyMapper = LegacyJsonFormat.legacyMapper();
            configureMapper(legacyMapper);
        }
        return legacyMapper;
    }

    /**
     * @return The default/primary ObjectMapper for deserializing JSON network configurations in DL4J
     */
    public static ObjectMapper getMapper(){
        return jsonMapper;
    }

    /**
     * @return The default/primary ObjectMapper for deserializing network configurations in DL4J (YAML format)
     */
    public static ObjectMapper getMapperYaml() {
        return yamlMapper;
    }

    private static void configureMapper(ObjectMapper ret) {
        ret.registerModule(new JodaModule());
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        ret.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        ret.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        ret.setVisibility(PropertyAccessor.CREATOR, JsonAutoDetect.Visibility.ANY);     //Need this otherwise JsonProperty annotations on constructors won't be seen
    }

}
