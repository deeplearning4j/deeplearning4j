package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.module.SimpleAbstractTypeResolver;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.reflections.Reflections;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by agibsonccc on 3/10/17.
 */
public class JacksonReflectionLoader {

    /**
     *
     * @param types
     * @return
     */
    public static ObjectMapper findTypesFor(List<Class<?>> types) {
        return findTypesFor(types, true);
    }


    /**
     * Get the implementations for a given list of classes.
     * These classes MUST be abstract classes or interfaces
     * @param types the types to get hte sub classes for
     * @return a map containing a list of interface names to
     * implementation types
     */
    public static Map<String, String> getImpls(List<Class<?>> types) {
        Map<String, String> classes = new HashMap<>();
        for (Class<?> type : types) {
            Reflections reflections = new Reflections();
            Set<Class<?>> subClasses = (Set<Class<?>>) reflections.getSubTypesOf(type);
            if (subClasses.size() > 1) {
                throw new IllegalArgumentException(String.format(
                                "Class " + type + " opType can't be inferred. There is more than %d of sub class for the given class",
                                subClasses.size()));
            } else if (subClasses.isEmpty())
                throw new IllegalArgumentException("No class implementation found for " + type.getCanonicalName());


            classes.put(type.getCanonicalName(), subClasses.iterator().next().getCanonicalName());
        }

        return classes;
    }

    /**
     *
     * @param types
     * @param json
     * @return
     */
    public static ObjectMapper findTypesFor(List<Class<?>> types, boolean json) {
        return withTypes(json ? new ObjectMapper() : new ObjectMapper(new YAMLFactory()), getImpls(types));

    }

    /**
     *
     * @param objectMapper
     * @param typeImpls
     * @return
     */
    public static ObjectMapper withTypes(ObjectMapper objectMapper, Map<String, String> typeImpls) {
        SimpleAbstractTypeResolver abstractTypeResolver = new SimpleAbstractTypeResolver();
        SimpleModule simpleModule = new SimpleModule();
        simpleModule.setAbstractTypes(abstractTypeResolver);

        for (Map.Entry<String, String> types : typeImpls.entrySet()) {
            try {
                Class interfaceClazz = Class.forName(types.getKey());
                if (!interfaceClazz.isInterface())
                    throw new IllegalArgumentException(
                                    "Class key must be an interface. Found " + interfaceClazz.getSimpleName());
                Class implClazz = Class.forName(types.getValue());
                if (Modifier.isAbstract(implClazz.getModifiers()) || implClazz.isInterface())
                    throw new IllegalArgumentException("Class value must be a concrete  implementation. Found "
                                    + implClazz.getSimpleName());
                abstractTypeResolver.addMapping(interfaceClazz, implClazz);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        objectMapper.registerModule(simpleModule);
        return objectMapper;

    }

    /**
     * Instantiate the given class opType
     * @param clazz the class to instantiate
     * @param json the json to instantiate from
     * @param objectMapper the object mapper to
     * @param <T>
     * @return
     * @throws IOException
     */
    public static <T> T instantiateType(Class<T> clazz, String json, ObjectMapper objectMapper) throws IOException {
        return objectMapper.readValue(json, clazz);
    }

}
