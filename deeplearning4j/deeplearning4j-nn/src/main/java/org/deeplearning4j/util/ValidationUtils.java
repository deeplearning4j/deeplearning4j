package org.deeplearning4j.util;

import com.google.common.base.Preconditions;
import java.util.Arrays;

public class ValidationUtils {

    /**
     * Reformats the input array to a length 1 array.
     *
     * If the array is length 1, returns the array
     *
     * @param data An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 1 that represents the input
     */
    public static int[] validate1(int[] data, String paramName){
        if(data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1,
                "Need either 1 " +
                        paramName + " values, got " +
                        data.length + " values: " +
                        Arrays.toString(data));

        return data;
    }

    /**
     * Reformats the input array to a length 2 array.
     * 
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     * 
     * @param data An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static int[] validate2(int[] data, String paramName){
        if(data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 2,
                "Need either 1 or 2 " + 
                        paramName + " values, got " + 
                        data.length + " values: " + 
                        Arrays.toString(data));
        
        if(data.length == 1){
            return new int[]{data[0], data[0]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 3 array.
     *
     * If the array is length 1, returns [a, a, a]
     * If the array is length 3, returns the array.
     *
     * @param data An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 3 that represents the input
     */
    public static int[] validate3(int[] data, String paramName){
        if(data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 3,
                "Need either 1 or 3 " +
                        paramName + " values, got " +
                        data.length + " values: " +
                        Arrays.toString(data));

        if(data.length == 1){
            return new int[]{data[0], data[0], data[0]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 4 array.
     *
     * If the array is length 1, returns [a, a, a, a]
     * If the array is length 2, return [a, a, b, b]
     * If the array is length 4, returns the array.
     *
     * @param data An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 4 that represents the input
     */
    public static int[] validate4(int[] data, String paramName){
        if(data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 2 || data.length == 4,
                "Need either 1, 2, or 4 " +
                        paramName + " values, got " +
                        data.length + " values: " +
                        Arrays.toString(data));

        if(data.length == 1){
            return new int[]{data[0], data[0], data[0], data[0]};
        } else if(data.length == 2){
            return new int[]{data[0], data[0], data[1], data[1]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 6 array.
     *
     * If the array is length 1, returns [a, a, a, a, a, a]
     * If the array is length 3, return [a, a, b, b, c, c]
     * If the array is length 6, returns the array.
     *
     * @param data An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 6 that represents the input
     */
    public static int[] validate6(int[] data, String paramName){
        if(data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 3 || data.length == 6,
                "Need either 1, 3, or 6 " +
                        paramName + " values, got " +
                        data.length + " values: " +
                        Arrays.toString(data));

        if(data.length == 1){
            return new int[]{data[0], data[0], data[0], data[0], data[0], data[0]};
        } else if(data.length == 3){
            return new int[]{data[0], data[0], data[1], data[1], data[2], data[2]};
        } else {
            return data;
        }
    }
}
