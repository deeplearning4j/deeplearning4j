package org.deeplearning4j.nn.conf.serde;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
import java.util.Collections;
import java.util.List;

public class JsonMappers {

    private static ObjectMapper jsonMapper = new ObjectMapper();
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
    private static ObjectMapper jsonMapperLegacyFormat = null;      //new ObjectMapper();

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
