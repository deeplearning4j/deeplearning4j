/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.serde;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.config.DL4JSystemProperties;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.conf.serde.legacy.LegacyJsonFormat;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.serde.json.LegacyIActivationDeserializer;
import org.nd4j.serde.json.LegacyILossFunctionDeserializer;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.cfg.MapperConfig;
import org.nd4j.shade.jackson.databind.deser.BeanDeserializerModifier;
import org.nd4j.shade.jackson.databind.introspect.Annotated;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.introspect.AnnotationMap;
import org.nd4j.shade.jackson.databind.introspect.JacksonAnnotationIntrospector;
import org.nd4j.shade.jackson.databind.jsontype.TypeResolverBuilder;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.util.OneTimeLogger;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * JSON mappers for deserializing neural net configurations, etc.
 *
 * @author Alex Black
 */
@Slf4j
public class JsonMappers {

    private static ObjectMapper jsonMapper = new ObjectMapper();
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    private static ObjectMapper legacyMapper;

    static {
        configureMapper(jsonMapper);
        configureMapper(yamlMapper);
    }

    /**
     * @return The default/primary ObjectMapper for deserializing JSON network configurations in DL4J
     */
    public static ObjectMapper getMapper(){
        return jsonMapper;
    }

    public static synchronized ObjectMapper getLegacyMapper(){
        if(legacyMapper == null){
            legacyMapper = LegacyJsonFormat.getMapper100alpha();
            configureMapper(legacyMapper);
        }
        return legacyMapper;
    }

    /**
     * @return The default/primary ObjectMapper for deserializing network configurations in DL4J (YAML format)
     */
    public static ObjectMapper getMapperYaml() {
        return yamlMapper;
    }

    private static void configureMapper(ObjectMapper ret) {
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);

        SimpleModule customDeserializerModule = new SimpleModule();
        customDeserializerModule.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                                                          JsonDeserializer<?> deserializer) {
                //Use our custom deserializers to handle backward compatibility for updaters -> IUpdater
                if (beanDesc.getBeanClass() == MultiLayerConfiguration.class) {
                    return new MultiLayerConfigurationDeserializer(deserializer);
                } else if (beanDesc.getBeanClass() == ComputationGraphConfiguration.class) {
                    return new ComputationGraphConfigurationDeserializer(deserializer);
                }
                return deserializer;
            }
        });

        ret.registerModule(customDeserializerModule);
    }
}
