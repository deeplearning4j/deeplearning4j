package org.deeplearning4j.util;

import java.lang.reflect.Constructor;

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
        for(int i = 0; i < clazz.getDeclaredConstructors().length; i++) {
            if(clazz.getDeclaredConstructors()[i].getParameterTypes().length < 1) {
                c = clazz.getDeclaredConstructors()[i];
                break;
            }
        }

       return c;
    }

}
