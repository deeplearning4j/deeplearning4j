package org.nd4j.bytebuddy.arrays.assign;

/**
 * @author Adam Gibson
 */
public interface AssignValue {
    /**
     * Assign the given value
     * @param arr the array
     * @param index the index
     * @param value the value
     */
    void assign(int[] arr,int index,int value);

}
