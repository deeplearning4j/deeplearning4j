/*
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.datavec.api.transform.serde;

import lombok.AllArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.WritableComparator;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.condition.column.ColumnCondition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.transform.sequence.window.WindowFunction;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.serde.json.LegacyIActivationDeserializer;
import org.nd4j.serde.json.LegacyILossFunctionDeserializer;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.cfg.MapperConfig;
import org.nd4j.shade.jackson.databind.introspect.Annotated;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.introspect.AnnotationMap;
import org.nd4j.shade.jackson.databind.introspect.JacksonAnnotationIntrospector;
import org.nd4j.shade.jackson.databind.jsontype.TypeResolverBuilder;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

import java.lang.annotation.Annotation;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * JSON mappers for deserializing neural net configurations, etc.
 *
 * @author Alex Black
 */
@Slf4j
public class JsonMappers {

    /**
     * This system property is provided as an alternative to {@link #registerLegacyCustomClassesForJSON(Class[])}
     * Classes can be specified in comma-separated format
     */
    public static String CUSTOM_REGISTRATION_PROPERTY = "org.datavec.config.custom.legacyclasses";

    static {
        String p = System.getProperty(CUSTOM_REGISTRATION_PROPERTY);
        if(p != null && !p.isEmpty()){
            String[] split = p.split(",");
            List<Class<?>> list = new ArrayList<>();
            for(String s : split){
                try{
                    Class<?> c = Class.forName(s);
                    list.add(c);
                } catch (Throwable t){
                    log.warn("Error parsing {} system property: class \"{}\" could not be loaded",CUSTOM_REGISTRATION_PROPERTY, s, t);
                }
            }

            if(list.size() > 0){
                try {
                    registerLegacyCustomClassesForJSONList(list);
                } catch (Throwable t){
                    log.warn("Error registering custom classes for legacy JSON deserialization ({} system property)",CUSTOM_REGISTRATION_PROPERTY, t);
                }
            }
        }
    }

    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;

    static {
        jsonMapper = new ObjectMapper();
        yamlMapper = new ObjectMapper(new YAMLFactory());
        configureMapper(jsonMapper);
        configureMapper(yamlMapper);
    }

    private static Map<Class, ObjectMapper> legacyMappers = new ConcurrentHashMap<>();


    /**
     * Register a set of classes (Transform, Filter, etc) for JSON deserialization.<br>
     * <br>
     * This is required ONLY when BOTH of the following conditions are met:<br>
     * 1. You want to load a serialized TransformProcess, saved in 1.0.0-alpha or before, AND<br>
     * 2. The serialized TransformProcess has a custom Transform, Filter, etc (i.e., one not defined in DL4J)<br>
     * <br>
     * By passing the classes of these custom classes here, DataVec should be able to deserialize them, in spite of the JSON
     * format change between versions.
     *
     * @param classes Classes to register
     */
    public static void registerLegacyCustomClassesForJSON(Class<?>... classes) {
        registerLegacyCustomClassesForJSONList(Arrays.<Class<?>>asList(classes));
    }

    /**
     * @see #registerLegacyCustomClassesForJSON(Class[])
     */
    public static void registerLegacyCustomClassesForJSONList(List<Class<?>> classes){
        //Default names (i.e., old format for custom JSON format)
        List<Pair<String,Class>> list = new ArrayList<>();
        for(Class<?> c : classes){
            list.add(new Pair<String,Class>(c.getSimpleName(), c));
        }
        registerLegacyCustomClassesForJSON(list);
    }

    /**
     * Set of classes that can be registered for legacy deserialization.
     */
    private static List<Class<?>> REGISTERABLE_CUSTOM_CLASSES = (List<Class<?>>) Arrays.<Class<?>>asList(
            Transform.class,
            ColumnAnalysis.class,
            ColumnCondition.class,
            Filter.class,
            ColumnMetaData.class,
            CalculateSortedRank.class,
            Schema.class,
            SequenceComparator.class,
            SequenceSplit.class,
            WindowFunction.class,
            Writable.class,
            WritableComparator.class
    );

    /**
     * Register a set of classes (Layer, GraphVertex, InputPreProcessor, IActivation, ILossFunction, ReconstructionDistribution
     * ONLY) for JSON deserialization, with custom names.<br>
     * Using this method directly should never be required (instead: use {@link #registerLegacyCustomClassesForJSON(Class[])}
     * but is added in case it is required in non-standard circumstances.
     */
    public static void registerLegacyCustomClassesForJSON(List<Pair<String,Class>> classes){
        for(Pair<String,Class> p : classes){
            String s = p.getFirst();
            Class c = p.getRight();
            //Check if it's a valid class to register...
            boolean found = false;
            for( Class<?> c2 : REGISTERABLE_CUSTOM_CLASSES){
                if(c2.isAssignableFrom(c)){
                    Map<String,String> map = LegacyMappingHelper.legacyMappingForClass(c2);
                    map.put(p.getFirst(), p.getSecond().getName());
                    found = true;
                }
            }

            if(!found){
                throw new IllegalArgumentException("Cannot register class for legacy JSON deserialization: class " +
                        c.getName() + " is not a subtype of classes " + REGISTERABLE_CUSTOM_CLASSES);
            }
        }
    }


    /**
     * Get the legacy JSON mapper for the specified class.<br>
     *
     * <b>NOTE</b>: This is intended for internal backward-compatibility use.
     *
     * Note to developers: The following JSON mappers are for handling legacy format JSON.
     * Note that after 1.0.0-alpha, the JSON subtype format for Transforms, Filters, Conditions etc were changed from
     * a wrapper object, to an "@class" field. However, to not break all saved transforms networks, these mappers are
     * part of the solution.<br>
     * <br>
     * How legacy loading works (same pattern for all types - Transform, Filter, Condition etc)<br>
     * 1. Transforms etc JSON that has a "@class" field are deserialized as normal<br>
     * 2. Transforms JSON that don't have such a field are mapped (via Layer @JsonTypeInfo) to LegacyMappingHelper.TransformHelper<br>
     * 3. LegacyMappingHelper.TransformHelper has a @JsonDeserialize annotation - we use LegacyMappingHelper.LegacyTransformDeserializer to handle it<br>
     * 4. LegacyTransformDeserializer has a list of old names (present in the legacy format JSON) and the corresponding class names
     * 5. BaseLegacyDeserializer (that LegacyTransformDeserializer extends) does a lookup and handles the deserialization
     *
     * Now, as to why we have one ObjectMapper for each type: We can't use the default JSON mapper for the legacy format,
     * as it'll fail due to not having the expected "@class" annotation.
     * Consequently, we need to tell Jackson to ignore that specific annotation and deserialize to the specified
     * class anyway. The ignoring is done via an annotation introspector, defined below in this class.
     * However, we can't just use a single annotation introspector (and hence ObjectMapper) for loading legacy values of
     * all types - if we did, then any nested types would fail (i.e., an Condition in a Transform - the Transform couldn't
     * be deserialized correctly, as the annotation would be ignored).
     *
     */
    public static synchronized ObjectMapper getLegacyMapperFor(@NonNull Class<?> clazz){
        if(!legacyMappers.containsKey(clazz)){
            ObjectMapper m = new ObjectMapper();
            configureMapper(m);
            m.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector(Collections.<Class>singletonList(clazz)));
            legacyMappers.put(clazz, m);
        }
        return legacyMappers.get(clazz);
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
