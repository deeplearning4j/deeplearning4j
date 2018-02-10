package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.val;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.lang.reflect.Field;

public abstract class BaseConvolutionConfig {

    /**
     * Get the value for a given property
     * for this function
     *
     * @param property the property to get
     * @return the value for the function if it exists
     */
    public Object getValue(Field property) {
        try {
            return property.get(this);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }

        return null;
    }


    /**
     * Set the value for this function.
     * Note that if value is null an {@link ND4JIllegalStateException}
     * will be thrown.
     *
     * @param target the target field
     * @param value  the value to set
     */
    public void setValueFor(Field target, Object value) {
        if (value == null) {
            throw new ND4JIllegalStateException("Unable to set field " + target + " using null value!");
        }

        value = ensureProperType(target, value);

        try {
            target.set(this, value);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    private Object ensureProperType(Field targetType, Object value) {
        val firstClass = targetType.getType();
        val valueType = value.getClass();
        if (!firstClass.equals(valueType)) {
            if (firstClass.equals(int[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.intValue();
                }

                int otherValue = (int) value;
                int[] setValue = new int[]{otherValue};
                return setValue;
            } else if (firstClass.equals(Integer[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.intValue();
                }

                Integer otherValue = (Integer) value;
                Integer[] setValue = new Integer[]{otherValue};
                return setValue;
            } else if (firstClass.equals(long[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.longValue();
                }

                long otherValue = (long) value;
                long[] setValue = new long[]{otherValue};
                return setValue;

            } else if (firstClass.equals(Long[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.longValue();
                }

                Long otherValue = (Long) value;
                Long[] setValue = new Long[]{otherValue};
                return setValue;

            } else if (firstClass.equals(double[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.doubleValue();
                }


                double otherValue = (double) value;
                double[] setValue = new double[]{otherValue};
                return setValue;

            } else if (firstClass.equals(Double[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.doubleValue();
                }


                Double otherValue = (Double) value;
                Double[] setValue = new Double[]{otherValue};
                return setValue;

            } else if (firstClass.equals(float[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.floatValue();
                }


                float otherValue = (float) value;
                float[] setValue = new float[]{otherValue};
                return setValue;

            } else if (firstClass.equals(Float[].class)) {
                if (value instanceof Number) {
                    Number number = (Number) value;
                    value = number.floatValue();
                }


                Float otherValue = (Float) value;
                Float[] setValue = new Float[]{otherValue};
                return setValue;

            }
        }

        return value;
    }


}
