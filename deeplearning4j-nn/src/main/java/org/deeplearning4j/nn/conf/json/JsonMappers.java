package org.deeplearning4j.nn.conf.json;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.serde.ComputationGraphConfigurationDeserializer;
import org.deeplearning4j.nn.conf.serde.MultiLayerConfigurationDeserializer;
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

public class JsonMappers {

    private static ObjectMapper jsonMapper = new ObjectMapper();
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
    private static ObjectMapper jsonMapperLegacyFormat = new ObjectMapper();

    static {
        configureMapper(jsonMapper);
        configureMapper(yamlMapper);
        configureMapper(jsonMapperLegacyFormat);

        jsonMapperLegacyFormat.setAnnotationIntrospector(new IgnoreJsonTypeInfoIntrospector());
    }

    public static ObjectMapper getMapper(){
        return jsonMapper;
    }

    public static ObjectMapper getMapperYaml() {
        return yamlMapper;
    }

    public static ObjectMapper getMapperLegacyJson(){
        return jsonMapperLegacyFormat;
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
     * This is so we can deserialize legacy format JSON without recursing infinitely
     */
    private static class IgnoreJsonTypeInfoIntrospector extends JacksonAnnotationIntrospector {
        @Override
        protected TypeResolverBuilder<?> _findTypeResolver(MapperConfig<?> config, Annotated ann, JavaType baseType) {
            if(ann instanceof AnnotatedClass){
                AnnotatedClass c = (AnnotatedClass)ann;
                Class<?> annClass = c.getAnnotated();
                if (Layer.class.isAssignableFrom(annClass)) {
                    System.out.println("Ignore annotation for: " + annClass);
                    AnnotationMap annotations = (AnnotationMap) ((AnnotatedClass) ann).getAnnotations();

                    AnnotationMap newMap = new AnnotationMap();
                    for(Annotation a : annotations.annotations()){
                        Class<?> annType = a.annotationType();
                        System.out.println(a + "\t" + annType);
                        if(annType == JsonTypeInfo.class){
                            //Ignore the JsonTypeInfo annotation on the
                            continue;
                        }
                        newMap.add(a);

                    }
                    return null;
                }
            }

            return super._findTypeResolver(config, ann, baseType);
        }

    }

}
