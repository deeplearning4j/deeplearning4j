package org.deeplearning4j.translate;

import org.deeplearning4j.util.Dl4jReflection;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeTranslatorUtils {

    public static <T> Map<String, Map<String, Object>> getFieldValueList(T caffeInst, Map<String, String> paramMappings)
            throws IllegalAccessException{
        // Map caffeFieldName(String) to builderFieldName(String) to caffeFieldValue(Object)
        Map<String, Map<String, Object>> paramMap = new HashMap<>();
        // Get all the fields from the caffeInst
        Field[] allCaffeInstFields = Dl4jReflection.getAllFields(caffeInst.getClass());
        for (Field caffeInstField : allCaffeInstFields) {
            String caffeFieldName = caffeInstField.getName();
            String builderFieldName = paramMappings.get(caffeFieldName);
            caffeInstField.setAccessible(true);
            Object caffeInstFieldValue = caffeInstField.get(caffeInst);
            Map<String, Object> innerMap = new HashMap<>();
            innerMap.put(builderFieldName, caffeInstFieldValue);
            paramMap.put(caffeFieldName, innerMap);
        }
        return paramMap;
    }

    public static Map regularTranslation(Object caffeFieldValue, String builderFieldName,
                                          Map<String, Object> builderParamMap) {
        if (caffeFieldValue instanceof Float) {
            double caffeDoubleValue = (double)caffeFieldValue;
            builderParamMap.put(builderFieldName, caffeDoubleValue);
        } else {
            builderParamMap.put(builderFieldName, builderFieldName);
        }
        return builderParamMap;
    }


//    private static <T> T writeToBuilderContainer(T builder, T solverNet) {
//        // overwrite builder with fields with layer fields
//        Class<?> layerClazz = layer.getClass();
//        Field[] neuralNetConfFields = Dl4jReflection.getAllFields(configInst.getClass());
//        Field[] layerFields = Dl4jReflection.getAllFields(layerClazz);
//        for(Field neuralNetField : neuralNetConfFields) {
//            neuralNetField.setAccessible(true);
//            for(Field layerField : layerFields) {
//                layerField.setAccessible(true);
//                if(neuralNetField.getName().equals(layerField.getName())) {
//                    try {
//                        Object layerFieldValue = layerField.get(layer);
//                        if(layerFieldValue != null ) {
//                            if(neuralNetField.getType().isAssignableFrom(layerField.getType())){
//                                //Same class, or neuralNetField is superclass/superinterface of layer field
//                                if(!ClassUtils.isPrimitiveOrWrapper(layerField.getType()) ){
//                                    neuralNetField.set(configInst, layerFieldValue);
//                                } else {
//                                    //Primitive -> autoboxed by Field.get(...). Hence layerFieldValue is never null for primitive fields,
//                                    // even if not explicitly set (due to default value for primitives)
//                                    //Convention here is to use Double.NaN, Float.NaN, Integer.MIN_VALUE, etc. as defaults in layer configs
//                                    // to signify 'not set'
//                                    Class<?> primitiveClass = layerField.getType();
//                                    if( primitiveClass == double.class || primitiveClass == Double.class ){
//                                        if( !Double.isNaN((double)layerFieldValue) ){
//                                            neuralNetField.set(configInst, layerFieldValue);
//                                        }
//                                    } else if( primitiveClass == float.class || primitiveClass == Float.class ){
//                                        if( !Float.isNaN((float)layerFieldValue) ){
//                                            neuralNetField.set(configInst, layerFieldValue);
//                                        }
//                                    } else if( primitiveClass == int.class || primitiveClass == Integer.class ){
//                                        if( ((int)layerFieldValue) != Integer.MIN_VALUE ){
//                                            neuralNetField.set(configInst, layerFieldValue);
//                                        }
//                                    } else if( primitiveClass == long.class || primitiveClass == Long.class ){
//                                        if( ((long)layerFieldValue) != Long.MIN_VALUE ){
//                                            neuralNetField.set(configInst, layerFieldValue);
//                                        }
//                                    } else if( primitiveClass == char.class || primitiveClass == Character.class ){
//                                        if( ((char)layerFieldValue) != Character.MIN_VALUE ){
//                                            neuralNetField.set(configInst, layerFieldValue);
//                                        }
//                                    } else {
//                                        //Boolean: can only be true/false. No usable 'not set' value -> need some other workaround
//                                        //Short, Byte: probably never used
//                                        throw new RuntimeException("Primitive type not settable via reflection");
//                                    }
//                                }
//                            }
//                        }
//                    } catch(Exception e) {
//                        throw new RuntimeException(e);
//                    }
//                }
//            }
//        }
//        return configInst;
//    }

}
