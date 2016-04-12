/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.util;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * IComplexNDArray operations
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayUtil {


    public static IComplexNDArray exp(IComplexNDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns the exponential of a complex ndarray
     *
     * @param toExp the ndarray to convert
     * @return the exponential of the specified
     * ndarray
     */
    public static IComplexNDArray expi(IComplexNDArray toExp) {
        IComplexNDArray flattened = toExp.ravel();
        for (int i = 0; i < flattened.length(); i++) {
            IComplexNumber n = flattened.getComplex(i);
            flattened.put(i, Nd4j.scalar(ComplexUtil.exp(n)));
        }
        return flattened.reshape(toExp.shape());
    }

    /**
     * Center an array
     *
     * @param arr   the arr to center
     * @param shape the shape of the array
     * @return the center portion of the array based on the
     * specified shape
     */
    public static IComplexNDArray center(IComplexNDArray arr, int[] shape) {
        if (arr.length() < ArrayUtil.prod(shape))
            return arr;
        for (int i = 0; i < shape.length; i++)
            if (shape[i] < 1)
                shape[i] = 1;

        INDArray shapeMatrix = NDArrayUtil.toNDArray(shape);
        INDArray currShape = NDArrayUtil.toNDArray(arr.shape());

        INDArray startIndex = Transforms.floor(currShape.sub(shapeMatrix).divi(Nd4j.scalar(2)));
        INDArray endIndex = startIndex.add(shapeMatrix);
        INDArrayIndex[] indexes = Indices.createFromStartAndEnd(startIndex, endIndex);

        if (shapeMatrix.length() > 1)
            return arr.get(indexes);


        else {
            IComplexNDArray ret = Nd4j.createComplex(new int[]{(int) shapeMatrix.getDouble(0)});
            int start = (int) startIndex.getDouble(0);
            int end = (int) endIndex.getDouble(0);
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.putScalar(count++, arr.getComplex(i));
            }

            return ret;
        }


    }

    /**
     * Truncates an ndarray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     *
     * @param nd the ndarray to truncate
     * @param n  the number of elements to truncate to
     * @return the truncated ndarray
     */
    public static IComplexNDArray truncate(IComplexNDArray nd, int n, int dimension) {


        if (nd.isVector()) {
            IComplexNDArray truncated = Nd4j.createComplex(new int[]{1,n});
            for (int i = 0; i < n; i++)
                truncated.putScalar(i, nd.getComplex(i));

            return truncated;
        }


        if (nd.size(dimension) > n) {
            int[] shape = ArrayUtil.copy(nd.shape());
            shape[dimension] = n;
            IComplexNDArray ret = Nd4j.createComplex(shape);
            IComplexNDArray ndLinear = nd.linearView();
            IComplexNDArray retLinear = ret.linearView();
            for(int i = 0; i < ret.length(); i++)
                retLinear.putScalar(i,ndLinear.getComplex(i));
            return ret;

        }

        return nd;

    }

    /**
     * Pads an ndarray with zeros
     *
     * @param nd          the ndarray to pad
     * @param targetShape the the new shape
     * @return the padded ndarray
     */
    public static IComplexNDArray padWithZeros(IComplexNDArray nd, int[] targetShape) {
        if (Arrays.equals(nd.shape(), targetShape))
            return nd;
        //no padding required
        if (ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        IComplexNDArray ret = Nd4j.createComplex(targetShape);
        INDArrayIndex[] targetShapeIndex = NDArrayIndex.createCoveringShape(nd.shape());
        ret.put(targetShapeIndex,nd);
        return ret;

    }


}
