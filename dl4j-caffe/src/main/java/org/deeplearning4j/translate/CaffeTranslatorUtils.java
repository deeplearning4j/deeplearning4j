package org.deeplearning4j.translate;

import org.deeplearning4j.util.Dl4jReflection;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

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
            String builderFieldName;
            try {
                // Get the BuilderFieldName from the caffeInstFieldName
                builderFieldName = paramMappings.get(caffeFieldName);
            } catch (NoSuchElementException e) {
                throw new NoSuchElementException(String.format("Cannot find the '%s' field in current mappings."
                        , caffeFieldName));
            }
            caffeInstField.setAccessible(true);
            Object caffeInstFieldValue = caffeInstField.get(caffeInst);
            Map<String, Object> innerMap = new HashMap<>();
            innerMap.put(builderFieldName, caffeInstFieldValue);
            paramMap.put(caffeFieldName, innerMap);
        }
        return paramMap;
    }

    public static <T> void setFieldFromMap(T builderLikeObject, Map<String,
            Object> builderParamMap) throws NoSuchFieldException, IllegalAccessException{
        // Loop through the map of builderFieldName mapped to correct builderFieldValue
        for (Map.Entry<String, Object> entry : builderParamMap.entrySet()) {
            String builderFieldName = entry.getKey();
            Object builderFieldValue = entry.getValue();
            // Get the Field based on the name of the field
            Field builderField = builderLikeObject.getClass().getDeclaredField(builderFieldName);
            // Allow access to the field
            builderField.setAccessible(true);
            // Set the value to the field in the object
            builderField.set(builderLikeObject, builderFieldValue);
        }
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
}
