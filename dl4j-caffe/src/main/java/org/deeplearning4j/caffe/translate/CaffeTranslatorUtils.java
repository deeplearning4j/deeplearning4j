package org.deeplearning4j.caffe.translate;

import org.apache.commons.lang3.reflect.FieldUtils;

import java.lang.reflect.Field;
import java.util.*;

/**
 * @author jeffreytang
 */
public class CaffeTranslatorUtils {


    public static <T> Map<String, Map<String, Object>>
            caffeField2builderField2caffeVal(T caffeInst, Map<String, String> paramMappings)
            throws IllegalAccessException{

        // Map caffeFieldName(String) to builderFieldName(String) to caffeFieldValue(Object)
        Map<String, Map<String, Object>> paramMap = new HashMap<>();

        // Loop through pre-defined mappings between caffe and builder classes
        for (Map.Entry<String, String> entry : paramMappings.entrySet()) {
            String caffeFieldName = entry.getKey();
            String builderFieldName = entry.getValue();
            // If there is not a mapping yet, ignore
            if (builderFieldName != null && !builderFieldName.isEmpty()) {
                try {
                    // Put a nested hashmap
                    Field caffeField = FieldUtils.getField(caffeInst.getClass(), caffeFieldName, true);
                    Object caffeFieldValue = caffeField.get(caffeInst);
                    Map<String, Object> innerMap = new HashMap<>();
                    innerMap.put(builderFieldName, caffeFieldValue);
                    paramMap.put(caffeFieldName, innerMap);
                } catch (NoSuchElementException e) {
                    throw new NoSuchElementException(
                            String.format("NoSuchElementException: %s is not a caffe field", caffeFieldName)
                    );
                }
            }
        }
        return paramMap;
    }

    public static List<List<Object>> caffeFieldBuilderFieldCaffeValIter(Map<String, Map<String, Object>> paramMap) {

        List<List<Object>> lst = new ArrayList<>();

        // Loop through the map of map and unpack into list
        for (Map.Entry<String, Map<String, Object>> entry : paramMap.entrySet()) {
            // CaffeFieldName
            String caffeField = entry.getKey();

            // HashMap with BuilderFieldName and CaffeFieldValue
            Map.Entry builderField2CaffeVal = entry.getValue().entrySet().iterator().next();
            // BuilderFieldName
            String builderField = (String) builderField2CaffeVal.getKey();
            // If the corresponding builderField is not empty or null
            if (builderField != null && !builderField.isEmpty()) {
                // CaffeFieldVal
                Object caffeVal = builderField2CaffeVal.getValue();
                lst.add(Arrays.asList(caffeField, builderField, caffeVal));
            }
        }
        return lst;
    }

    public static void translation2BuilderFieldBuilderValMap(List<List<Object>> caffeFieldBuilderFieldCaffeValIter,
                                                            Map<String, Object> builderParamMap,
                                                            CaffeSpecialTranslator functionInterface) {

        for (List<Object> iter : caffeFieldBuilderFieldCaffeValIter) {
            String caffeField = (String) iter.get(0);
            String builderField = (String) iter.get(1);
            Object caffeVal = iter.get(2);
            // Just the caffeValue as the value of the builderField
            builderParamMap.put(builderField, caffeVal);
            // Takes care of special translations
            functionInterface.specialTranslation(caffeField, caffeVal, builderField, builderParamMap);
        }
    }

    public static <T> void applyMapToBuilder(T builderLikeObject, Map<String, Object> builderParamMap)
            throws NoSuchFieldException, IllegalAccessException{

        // Loop through the map of builderFieldName mapped to correct builderFieldValue
        for (Map.Entry<String, Object> entry : builderParamMap.entrySet()) {
            String builderFieldName = entry.getKey();
            Object builderFieldValue = entry.getValue();
            //DEBUG
            System.out.println(String.format("Field: %s   Value: %s", builderFieldName, builderFieldValue));
            //
            if (builderFieldName != null && !builderFieldName.isEmpty()) {
                // Get the Field based on the name of the field
                Field builderField = FieldUtils.getField(builderLikeObject.getClass(), builderFieldName, true);
                builderField.setAccessible(true);
                builderField.set(builderLikeObject, builderFieldValue);
            }
        }
    }

}
