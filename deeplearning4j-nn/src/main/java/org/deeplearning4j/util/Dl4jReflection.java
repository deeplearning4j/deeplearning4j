/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.util;


import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * @author Adam Gibson
 */
public class Dl4jReflection {
    private Dl4jReflection() {}

    /**
     * Gets the empty constructor from a class
     * @param clazz the class to get the constructor from
     * @return the empty constructor for the class
     */
    public static Constructor<?> getEmptyConstructor(Class<?> clazz) {
        Constructor<?> c = clazz.getDeclaredConstructors()[0];
        for (int i = 0; i < clazz.getDeclaredConstructors().length; i++) {
            if (clazz.getDeclaredConstructors()[i].getParameterTypes().length < 1) {
                c = clazz.getDeclaredConstructors()[i];
                break;
            }
        }

        return c;
    }


    public static Field[] getAllFields(Class<?> clazz) {
        // Keep backing up the inheritance hierarchy.
        Class<?> targetClass = clazz;
        List<Field> fields = new ArrayList<>();

        do {
            fields.addAll(Arrays.asList(targetClass.getDeclaredFields()));
            targetClass = targetClass.getSuperclass();
        } while (targetClass != null && targetClass != Object.class);

        return fields.toArray(new Field[fields.size()]);
    }

    /**
     * Sets the properties of the given object
     * @param obj the object o set
     * @param props the properties to set
     */
    public static void setProperties(Object obj, Properties props) throws Exception {
        for (Field field : obj.getClass().getDeclaredFields()) {
            field.setAccessible(true);
            if (props.containsKey(field.getName())) {
                set(field, obj, props.getProperty(field.getName()));
            }

        }
    }

    /* sets a field with a fairly basic strategy */
    private static void set(Field field, Object obj, String value) throws Exception {
        Class<?> clazz = field.getType();
        field.setAccessible(true);
        if (clazz.equals(Double.class) || clazz.equals(double.class)) {
            double val = Double.valueOf(value);
            field.set(obj, val);
        } else if (clazz.equals(String.class)) {
            field.set(obj, value);
        } else if (clazz.equals(Integer.class) || clazz.equals(int.class)) {
            int val = Integer.parseInt(value);
            field.set(obj, val);
        } else if (clazz.equals(Float.class) || clazz.equals(float.class)) {
            float f = Float.parseFloat(value);
            field.set(obj, f);
        }
    }


    /**
     * Get fields as properties
     * @param obj the object to get fields for
     * @param clazzes the classes to use for reflection and properties.
     *                T
     * @return the fields as properties
     */
    public static Properties getFieldsAsProperties(Object obj, Class<?>[] clazzes) throws Exception {
        Properties props = new Properties();
        for (Field field : obj.getClass().getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers()))
                continue;
            field.setAccessible(true);
            Class<?> type = field.getType();
            if (clazzes == null || contains(type, clazzes)) {
                Object val = field.get(obj);
                if (val != null)
                    props.put(field.getName(), val.toString());

            }
        }

        return props;
    }


    private static boolean contains(Class<?> test, Class<?>[] arr) {
        for (Class<?> c : arr)
            if (c.equals(test))
                return true;
        return false;
    }

}
