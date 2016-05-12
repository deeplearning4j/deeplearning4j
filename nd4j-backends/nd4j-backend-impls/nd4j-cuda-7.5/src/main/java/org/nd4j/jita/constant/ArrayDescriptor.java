package org.nd4j.jita.constant;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class ArrayDescriptor {
    public enum DataType {
        INT,
        FLOAT,
        DOUBLE,
        LONG,
    }

    private int hashCode = -1;
    private final DataType dtype;

    public ArrayDescriptor(int[] array) {
        hashCode = Arrays.hashCode(array);
        dtype = DataType.INT;
    }

    public ArrayDescriptor(float[] array) {
        hashCode = Arrays.hashCode(array);
        dtype = DataType.FLOAT;
    }

    public ArrayDescriptor(double[] array) {
        hashCode = Arrays.hashCode(array);
        dtype = DataType.DOUBLE;
    }

    public ArrayDescriptor(long[] array) {
        hashCode = Arrays.hashCode(array);
        dtype = DataType.LONG;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ArrayDescriptor that = (ArrayDescriptor) o;

        if (dtype != that.dtype) return false;
        return hashCode == that.hashCode;

    }

    @Override
    public int hashCode() {
        int result = hashCode;
        result = 31 * result + (dtype != null ? dtype.hashCode() : 0);
        return result;
    }
}
