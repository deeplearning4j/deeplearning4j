package org.deeplearning4j.linalg.util;

/**
 * Reflect utilities
 */
public final class ReflectionUtil {

    /**
     * Create a class array from the given array of objects
     * @param objects the objects to get classes for
     * @return the classes for each object in the array
     */
    public static Class<?>[] classesFor(Object[] objects) {
        Class<?>[] ret = new Class<?>[objects.length];
        for(int i = 0; i < objects.length; i++) {
            ret[i] = objects[i].getClass();
        }
        return ret;
    }


}
