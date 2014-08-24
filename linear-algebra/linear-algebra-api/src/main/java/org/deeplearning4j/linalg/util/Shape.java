package org.deeplearning4j.linalg.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Encapsulates all shape related logic (vector of 0 dimension is a scalar is equivalent to
 *                                       a vector of length 1...)
 */
public class Shape {


    /**
     * Gets rid of any singleton dimensions of the given array
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static int[] squeeze(int[] shape) {
        List<Integer> ret = new ArrayList<>();

        for(int i = 0; i < shape.length; i++)
            if(shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArray(ret);
    }


    public static int nonZeroDimension(int[] shape) {
        if(shape[0] == 1 && shape.length > 1)
            return shape[1];
        return shape[0];
    }


    /**
     * Returns whether 2 shapes are equals by checking for dimension semantics
     * as well as array equality
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the shapes are equivalent
     */
    public static boolean shapeEquals(int[] shape1,int[] shape2) {
        return scalarEquals(shape1,shape2) || Arrays.equals(shape1,shape2) || squeezeEquals(shape1,shape2);
    }

    /**
     * Returns true if the given shapes are both scalars (0 dimension or shape[0] == 1)
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the 2 shapes are equal based on scalar rules
     */
    public static boolean scalarEquals(int[] shape1,int[] shape2) {
        if(shape1.length == 0) {
            if(shape2.length == 1 && shape2[0] == 1)
                return true;
        }

        else if(shape2.length == 0) {
            if(shape1.length == 1 && shape1[0] == 1)
                return true;
        }

        return false;
    }

    public static boolean isRowVectorShape(int[] shape) {
        return
                (shape.length == 2
                &&  shape[0] == 1) ||
                shape.length == 1;

    }

    public static boolean isColumnVectorShape(int[] shape) {
        return
                (shape.length == 2
                        &&  shape[1] == 1);

    }



    /**
     * Returns true for the case where
     * singleton dimensions are being compared
     * @param test1 the first to test
     * @param test2 the second to test
     * @return true if the arrays
     * are equal with the singleton dimension omitted
     */
   public static boolean squeezeEquals(int[] test1,int[] test2) {
       int[] s1 = squeeze(test1);
       int[] s2 = squeeze(test2);
       return scalarEquals(s1,s2) || Arrays.equals(s1,s2);
   }


}
