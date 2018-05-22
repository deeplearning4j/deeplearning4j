package org.deeplearning4j.arbiter.util;

import java.util.Arrays;

/**
 * @author Alex Black
 */
public class ObjectUtils {

    private ObjectUtils() {}

    /**
     * Get the string representation of the object. Arrays, including primitive arrays, are printed using
     * Arrays.toString(...) methods.
     *
     * @param v Value to convert to a string
     * @return String representation
     */
    public static String valueToString(Object v) {
        if (v.getClass().isArray()) {
            if (v.getClass().getComponentType().isPrimitive()) {
                Class<?> c = v.getClass().getComponentType();
                if (c == int.class) {
                    return Arrays.toString((int[]) v);
                } else if (c == double.class) {
                    return Arrays.toString((double[]) v);
                } else if (c == float.class) {
                    return Arrays.toString((float[]) v);
                } else if (c == long.class) {
                    return Arrays.toString((long[]) v);
                } else if (c == byte.class) {
                    return Arrays.toString((byte[]) v);
                } else if (c == short.class) {
                    return Arrays.toString((short[]) v);
                } else {
                    return v.toString();
                }
            } else {
                return Arrays.toString((Object[]) v);
            }
        } else {
            return v.toString();
        }
    }
}
