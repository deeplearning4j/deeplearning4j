/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Encapsulates all shape related logic (vector of 0 dimension is a scalar is equivalent to
 * a vector of length 1...)
 *
 * @author Adam Gibson
 */
public class Shape {
    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static int[] squeeze(int[] shape, int[] stride) {
        List<Integer> ret = new ArrayList<>();

        for (int i = 0; i < shape.length; i++)
            if (shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArray(ret);
    }


    public static int[] sizeForAxes(int[] axes, int[] shape) {
        int[] ret = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            ret[i] = shape[axes[i]];
        }
        return ret;
    }


    /**
     * Returns whether the given shape is a vector
     *
     * @param shape the shape to test
     * @return whether the given shape is a vector
     */
    public static boolean isVector(int[] shape) {
        if (shape.length > 2 || shape.length < 1)
            return false;
        else {
            int len = ArrayUtil.prod(shape);
            return shape[0] == len || shape[1] == len;
        }
    }

    /**
     * Returns whether the passed in shape is a matrix
     *
     * @param shape whether the passed in shape is a matrix
     * @return true if the shape is a matrix false otherwise
     */
    public static boolean isMatrix(int[] shape) {
        if (shape.length != 2)
            return false;
        return !isVector(shape);
    }


    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static int[] squeeze(int[] shape) {
        List<Integer> ret = new ArrayList<>();

        for (int i = 0; i < shape.length; i++)
            if (shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArray(ret);
    }


    public static int nonZeroDimension(int[] shape) {
        if (shape[0] == 1 && shape.length > 1)
            return shape[1];
        return shape[0];
    }


    /**
     * Returns whether 2 shapes are equals by checking for dimension semantics
     * as well as array equality
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the shapes are equivalent
     */
    public static boolean shapeEquals(int[] shape1, int[] shape2) {
        if (isColumnVectorShape(shape1)) {
            if (isColumnVectorShape(shape2)) {
                return Arrays.equals(shape1, shape2);
            }

        }

        if (isRowVectorShape(shape1)) {
            if (isRowVectorShape(shape2)) {
                int[] shape1Comp = squeeze(shape1);
                int[] shape2Comp = squeeze(shape2);
                return Arrays.equals(shape1Comp, shape2Comp);
            }
        }

        return scalarEquals(shape1, shape2) || Arrays.equals(shape1, shape2);
    }


    /**
     * Returns true if the given shapes are both scalars (0 dimension or shape[0] == 1)
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the 2 shapes are equal based on scalar rules
     */
    public static boolean scalarEquals(int[] shape1, int[] shape2) {
        if (shape1.length == 0) {
            if (shape2.length == 1 && shape2[0] == 1)
                return true;
        } else if (shape2.length == 0) {
            if (shape1.length == 1 && shape1[0] == 1)
                return true;
        }

        return false;
    }

    public static boolean isRowVectorShape(int[] shape) {
        return
                (shape.length == 2
                        && shape[0] == 1) ||
                        shape.length == 1;

    }

    public static boolean isColumnVectorShape(int[] shape) {
        return
                (shape.length == 2
                        && shape[1] == 1);

    }


    /**
     * Returns true for the case where
     * singleton dimensions are being compared
     *
     * @param test1 the first to test
     * @param test2 the second to test
     * @return true if the arrays
     * are equal with the singleton dimension omitted
     */
    public static boolean squeezeEquals(int[] test1, int[] test2) {
        int[] s1 = squeeze(test1);
        int[] s2 = squeeze(test2);
        return scalarEquals(s1, s2) || Arrays.equals(s1, s2);
    }


}
