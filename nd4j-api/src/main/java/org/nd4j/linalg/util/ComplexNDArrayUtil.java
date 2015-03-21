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


import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

        INDArray shapeMatrix = ArrayUtil.toNDArray(shape);
        for (int i = 0; i < shape.length; i++)
            if (shape.length < 1)
                throw new IllegalArgumentException("Illegal shape passed in with value < 0");
        INDArray currShape = ArrayUtil.toNDArray(arr.shape());

        INDArray startIndex = currShape.sub(shapeMatrix).divi(Nd4j.scalar(2));
        INDArray endIndex = startIndex.add(shapeMatrix);
        if (shapeMatrix.length() > 1) {
            arr = Nd4j.createComplex(arr.get(NDArrayIndex.interval((int) startIndex.getDouble(0), (int) endIndex.getDouble(0) + 1), NDArrayIndex.interval((int) startIndex.getDouble(1), (int) endIndex.getDouble(1) + 1)));
        } else {
            IComplexNDArray ret = Nd4j.createComplex(new int[]{(int) shapeMatrix.getDouble(0)});
            int start = (int) startIndex.getDouble(0);
            int end = (int) endIndex.getDouble(0);
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.putScalar(count++, arr.getComplex(i));
            }

            return ret;
        }


        return arr;
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
            IComplexNDArray truncated = Nd4j.createComplex(new int[]{n});
            for (int i = 0; i < n; i++)
                truncated.put(i, nd.getScalar(i));
            return truncated;
        }


        if (nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if (nd.isVector()) {
                IComplexNDArray ret = Nd4j.createComplex(targetShape);
                int count = 0;
                for (int i = 0; i < nd.length(); i += nd.stride()[dimension]) {
                    ret.put(count++, nd.getScalar(i));

                }
                return ret;
            } else if (nd.isMatrix()) {
                List<IComplexDouble> list = new ArrayList<>();
                //row
                if (dimension == 0) {
                    for (int i = 0; i < nd.rows(); i++) {
                        IComplexNDArray row = nd.getRow(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

                            list.add(row.getComplex(j).asDouble());
                        }
                    }
                } else if (dimension == 1) {
                    for (int i = 0; i < nd.columns(); i++) {
                        IComplexNDArray row = nd.getColumn(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

                            list.add(row.getComplex(j).asDouble());
                        }
                    }
                } else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

            }


            if (dimension == 0) {
                List<IComplexNDArray> slices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    IComplexNDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return Nd4j.createComplex(slices, targetShape);

            } else {
                List<IComplexDouble> list = new ArrayList<>();
                int numElementsPerSlice = ArrayUtil.prod(ArrayUtil.removeIndex(targetShape, 0));
                for (int i = 0; i < nd.slices(); i++) {
                    IComplexNDArray slice = nd.slice(i).ravel();
                    for (int j = 0; j < numElementsPerSlice; j++)
                        list.add((IComplexDouble) slice.getScalar(j).element());
                }

                assert list.size() == ArrayUtil.prod(targetShape) : "Illegal shape for length " + list.size();

                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

            }


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
        System.arraycopy(nd.data().asDouble(), 0, ret.data().asDouble(), 0, nd.data().length());
        return ret;

    }


}
