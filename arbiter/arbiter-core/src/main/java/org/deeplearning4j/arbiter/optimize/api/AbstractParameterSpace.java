package org.deeplearning4j.arbiter.optimize.api;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 23/07/2017.
 */
public abstract class AbstractParameterSpace<T> implements ParameterSpace<T> {

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        Map<String, ParameterSpace> m = new LinkedHashMap<>();

        //Need to manually build and walk the class heirarchy...
        Class<?> currClass = this.getClass();
        List<Class<?>> classHeirarchy = new ArrayList<>();
        while (currClass != Object.class) {
            classHeirarchy.add(currClass);
            currClass = currClass.getSuperclass();
        }

        for (int i = classHeirarchy.size() - 1; i >= 0; i--) {
            //Use reflection here to avoid a mass of boilerplate code...
            Field[] allFields = classHeirarchy.get(i).getDeclaredFields();

            for (Field f : allFields) {

                String name = f.getName();
                Class<?> fieldClass = f.getType();
                boolean isParamSpacefield = ParameterSpace.class.isAssignableFrom(fieldClass);

                if (!isParamSpacefield) {
                    continue;
                }

                f.setAccessible(true);

                ParameterSpace<?> p;
                try {
                    p = (ParameterSpace<?>) f.get(this);
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e);
                }

                if (p != null) {
                    m.put(name, p);
                }
            }
        }

        return m;
    }

}
