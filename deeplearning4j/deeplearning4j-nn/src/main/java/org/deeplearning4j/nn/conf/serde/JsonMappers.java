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

import java.lang.annotation.Annotation;
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

    /**
     * @deprecated Use {@link DL4JSystemProperties#CUSTOM_REGISTRATION_PROPERTY}
     */
    @Deprecated
    public static String CUSTOM_REGISTRATION_PROPERTY = DL4JSystemProperties.CUSTOM_REGISTRATION_PROPERTY;

    static {
        String p = System.getProperty(DL4JSystemProperties.CUSTOM_REGISTRATION_PROPERTY);
        if(p != null && !p.isEmpty()){
            String[] split = p.split(",");
            List<Class<?>> list = new ArrayList<>();
            for(String s : split){
                try{
                    Class<?> c = Class.forName(s);
                    list.add(c);
                } catch (Throwable t){
                    log.warn("Error parsing {} system property: class \"{}\" could not be loaded",DL4JSystemProperties.CUSTOM_REGISTRATION_PROPERTY, s, t);
                }
            }

            if(list.size() > 0){
                try {
                    NeuralNetConfiguration.registerLegacyCustomClassesForJSONList(list);
                } catch (Throwable t){
                    log.warn("Error registering custom classes for legacy JSON deserialization ({} system property)",DL4JSystemProperties.CUSTOM_REGISTRATION_PROPERTY, t);
                }
            }
        }
    }

    private static ObjectMapper jsonMapper = new ObjectMapper();
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    /*
    Note to developers: The following JSON mappers are for handling legacy format JSON.
    Note that after 1.0.0-alpha, the JSON subtype format for layers, preprocessors, graph vertices,
    etc were changed from a wrapper object, to an "@class" field.
    However, in an attempt to not break saved networks, these mappers are part of the solution.

    How legacy loading works (same pattern for all types - Layer, GraphVertex, InputPreprocesor etc)
    1. Layers etc that have an "@class" field are deserialized as normal
    2. Layers that don't have such a field are mapped (via Layer @JsonTypeInfo) to the LegacyLayerDeserializerHelper class.
    3. LegacyLayerDeserializerHelper has a @JsonDeserialize annotation - we use LegacyLayerDeserialize to handle it
    4. LegacyLayerDeserializer has a list of old names (present in the legacy format JSON) and the corresponding class names
    5. BaseLegacyDeserializer (that LegacyLayerDeserializer extends) does a lookup and handles the deserialization

    Now, as to why we have one ObjectMapper for each type: We can't use the default JSON mapper for the legacy format,
    as it'll fail due to not having the expected "@class" annotation.
    Consequently, we need to tell Jackson to ignore that specific annotation and deserialize to the specified
    class anyway. The ignoring is done via an annotation introspector, defined below in this class.
    However, we can't just use a single annotation introspector (and hence ObjectMapper) for loading legacy values of
    all types - if we did, then any nested types would fail (i.e., an IActivation in a Layer - the IActivation couldn't
    be deserialized correctly, as the annotation would be ignored).

     */
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatLayer = new ObjectMapper();
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatVertex = new ObjectMapper();
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatPreproc = new ObjectMapper();
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatIActivation = new ObjectMapper();
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatILoss = new ObjectMapper();
    @Getter
    private static ObjectMapper jsonMapperLegacyFormatReconstruction = new ObjectMapper();

    static {
        configureMapper(jsonMapper);
        configureMapper(yamlMapper);
        configureMapper(jsonMapperLegacyFormatLayer);
        configureMapper(jsonMapperLegacyFormatVertex);
        configureMapper(jsonMapperLegacyFormatPreproc);
        configureMapper(jsonMapperLegacyFormatIActivation);
        configureMapper(jsonMapperLegacyFormatILoss);
        configureMapper(jsonMapperLegacyFormatReconstruction);

        jsonMapperLegacyFormatLayer.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(Layer.class)));
        jsonMapperLegacyFormatVertex.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(GraphVertex.class)));
        jsonMapperLegacyFormatPreproc.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(InputPreProcessor.class)));
        jsonMapperLegacyFormatIActivation.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(IActivation.class)));
        jsonMapperLegacyFormatILoss.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(ILossFunction.class)));
        jsonMapperLegacyFormatReconstruction.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(ReconstructionDistribution.class)));

        LegacyIActivationDeserializer.setLegacyJsonMapper(jsonMapperLegacyFormatIActivation);
        LegacyILossFunctionDeserializer.setLegacyJsonMapper(jsonMapperLegacyFormatILoss);
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


    /**
     * Custom Jackson Introspector to ignore the {@code @JsonTypeYnfo} annotations on layers etc.
     * This is so we can deserialize legacy format JSON without recursing infinitely, by selectively ignoring
     * a set of JsonTypeInfo annotations
     */
    @AllArgsConstructor
    private static class IgnoreJsonTypeInfoIntrospector extends JacksonAnnotationIntrospector {

        private List<Class> classList;

        @Override
        protected TypeResolverBuilder<?> _findTypeResolver(MapperConfig<?> config, Annotated ann, JavaType baseType) {
            if(ann instanceof AnnotatedClass){
                AnnotatedClass c = (AnnotatedClass)ann;
                Class<?> annClass = c.getAnnotated();

                boolean isAssignable = false;
                for(Class c2 : classList){
                    if(c2.isAssignableFrom(annClass)){
                        isAssignable = true;
                        break;
                    }
                }

                if( isAssignable ){
                    AnnotationMap annotations = (AnnotationMap) ((AnnotatedClass) ann).getAnnotations();
                    if(annotations == null || annotations.annotations() == null){
                        //Probably not necessary - but here for safety
                        return super._findTypeResolver(config, ann, baseType);
                    }

                    AnnotationMap newMap = null;
                    for(Annotation a : annotations.annotations()){
                        Class<?> annType = a.annotationType();
                        if(annType == JsonTypeInfo.class){
                            //Ignore the JsonTypeInfo annotation on the Layer class
                            continue;
                        }
                        if(newMap == null){
                            newMap = new AnnotationMap();
                        }
                        newMap.add(a);
                    }
                    if(newMap == null)
                        return null;

                    //Pass the remaining annotations (if any) to the original introspector
                    AnnotatedClass ann2 = c.withAnnotations(newMap);
                    return super._findTypeResolver(config, ann2, baseType);
                }
            }
            return super._findTypeResolver(config, ann, baseType);
        }
    }
}
