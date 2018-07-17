/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.checkutil;

import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 *
 * This class contains utility methods for generating NDArrays for use in unit tests
 * The idea is to generate arrays with a specific shape, after various operations have been undertaken on them
 * So output is after get, reshape, transpose, permute, tensorAlongDimension etc operations have been done<br>
 * Most useful methods:<br>
 *     - getAllTestMatricesWithShape
 *     - getAll4dTestArraysWithShape
 *     - getAll4dTestArraysWithShape
 * @author Alex Black
 */
public class NDArrayCreationUtil {
    private NDArrayCreationUtil() {}

    /** Get an array of INDArrays (2d) all with the specified shape. Pair<INDArray,String> returned to aid
     * debugging: String contains information on how to reproduce the matrix (i.e., which function, and arguments)
     * Each NDArray in the returned array has been obtained by applying an operation such as transpose, tensorAlongDimension,
     * etc to an original array.
     */
    public static List<Pair<INDArray, String>> getAllTestMatricesWithShape(char ordering, int rows, int cols,
                    int seed) {
        List<Pair<INDArray, String>> all = new ArrayList<>();
        Nd4j.getRandom().setSeed(seed);
        all.add(new Pair<>(Nd4j.linspace(1, rows * cols, rows * cols).reshape(ordering, rows, cols),
                        "Nd4j..linspace(1,rows * cols,rows * cols).reshape(rows,cols)"));

        all.add(getTransposedMatrixWithShape(ordering, rows, cols, seed));

        all.addAll(getSubMatricesWithShape(ordering, rows, cols, seed));

        all.addAll(getTensorAlongDimensionMatricesWithShape(ordering, rows, cols, seed));

        all.add(getPermutedWithShape(ordering, rows, cols, seed));
        all.add(getReshapedWithShape(ordering, rows, cols, seed));

        return all;
    }


    /** Get an array of INDArrays (2d) all with the specified shape. Pair<INDArray,String> returned to aid
     * debugging: String contains information on how to reproduce the matrix (i.e., which function, and arguments)
     * Each NDArray in the returned array has been obtained by applying an operation such as transpose, tensorAlongDimension,
     * etc to an original array.
     */
    public static List<Pair<INDArray, String>> getAllTestMatricesWithShape(long rows, long cols, long seed) {
        List<Pair<INDArray, String>> all = new ArrayList<>();
        Nd4j.getRandom().setSeed(seed);
        all.add(new Pair<>(Nd4j.linspace(1L, rows * cols, rows * cols).reshape(rows, cols),
                        "Nd4j..linspace(1,rows * cols,rows * cols).reshape(rows,cols)"));

        all.add(getTransposedMatrixWithShape(rows, cols, seed));

        all.addAll(getSubMatricesWithShape(rows, cols, seed));

        all.addAll(getTensorAlongDimensionMatricesWithShape(rows, cols, seed));

        all.add(getPermutedWithShape(rows, cols, seed));
        all.add(getReshapedWithShape(rows, cols, seed));

        return all;
    }

    /**
     * Test utility to sweep shapes given a rank
     * Given a rank will generate random test matrices that will cover all cases of a shape with a '1' anywhere in the shape
     * as well a shape with random ints that are not 0 or 1
     * eg. rank 2: 1,1; 1,2; 2,1; 2,2; 3,4
     * Motivated by TADs that often hit bugs when a "1" occurs as the size of a dimension
     *
     * @param rank any rank including true scalars i.e rank >= 0
     * @param order what order array to return i.e 'c' or 'f' order arrays
     * @return List of arrays and the shapes as strings
     */
    public static List<Pair<INDArray, String>> getTestMatricesWithVaryingShapes(int rank, char order) {
        List<Pair<INDArray, String>> all = new ArrayList<>();
        if (rank == 0) {
            //scalar
            all.add(new Pair<>(Nd4j.trueScalar(Nd4j.rand(1, 1).getDouble(0)), "{}"));
            return all;
        }
        //generate all possible combinations with a 1 and a 2
        int maxCount = (int) Math.pow(2.0, rank);
        int[] defaultOnes = new int[rank];
        Arrays.fill(defaultOnes, 1);
        //use binary and just add 1
        for (int i = 0; i < maxCount; i++) {
            int num = i;
            int[] iShape = ArrayUtils.clone(defaultOnes);
            int b = 0;
            while (num > 0) {
                iShape[b] = (num % 2) + 1;
                b++;
                num /= 2;
            }
            all.add(new Pair<>(Nd4j.rand(order, iShape), ArrayUtils.toString(iShape)));
        }
        // add a random shape of correct rank with elements > 2 that is not too big
        int[] aRandomShape = new int[rank];
        Random ran = new Random();
        for (int i = 0; i < rank; i++) {
            aRandomShape[i] = 2 + ran.nextInt(6);
        }
        all.add(new Pair<>(Nd4j.rand(order, aRandomShape), ArrayUtils.toString(aRandomShape)));
        return all;
    }


    public static Pair<INDArray, String> getTransposedMatrixWithShape(char ordering, int rows, int cols, int seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray out = Nd4j.linspace(1, rows * cols, rows * cols).reshape(ordering, cols, rows);
        return new Pair<>(out.transpose(), "getTransposedMatrixWithShape(" + rows + "," + cols + "," + seed + ")");
    }

    public static Pair<INDArray, String> getTransposedMatrixWithShape(long rows, long cols, long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray out = Nd4j.linspace(1, rows * cols, rows * cols).reshape(cols, rows);
        return new Pair<>(out.transpose(), "getTransposedMatrixWithShape(" + rows + "," + cols + "," + seed + ")");
    }

    public static List<Pair<INDArray, String>> getSubMatricesWithShape(long rows, long cols, long seed) {
        return getSubMatricesWithShape(Nd4j.order(), rows, cols, seed);
    }

    public static List<Pair<INDArray, String>> getSubMatricesWithShape(char ordering, long rows, long cols, long seed) {
        //Create 3 identical matrices. Could do get() on single original array, but in-place modifications on one
        //might mess up tests for another
        Nd4j.getRandom().setSeed(seed);
        long[] shape = new long[] {2 * rows + 4, 2 * cols + 4};
        int len = ArrayUtil.prod(shape);
        INDArray orig = Nd4j.linspace(1, len, len).reshape(ordering, shape);
        INDArray first = orig.get(NDArrayIndex.interval(0, rows), NDArrayIndex.interval(0, cols));
        Nd4j.getRandom().setSeed(seed);
        orig = Nd4j.linspace(1, len, len).reshape(shape);
        INDArray second = orig.get(NDArrayIndex.interval(3, rows + 3), NDArrayIndex.interval(3, cols + 3));
        Nd4j.getRandom().setSeed(seed);
        orig = Nd4j.linspace(1, len, len).reshape(ordering, shape);
        INDArray third = orig.get(NDArrayIndex.interval(rows, 2 * rows), NDArrayIndex.interval(cols, 2 * cols));

        String baseMsg = "getSubMatricesWithShape(" + rows + "," + cols + "," + seed + ")";
        List<Pair<INDArray, String>> list = new ArrayList<>(3);
        list.add(new Pair<>(first, baseMsg + ".get(0)"));
        list.add(new Pair<>(second, baseMsg + ".get(1)"));
        list.add(new Pair<>(third, baseMsg + ".get(2)"));
        return list;
    }



    public static List<Pair<INDArray, String>> getTensorAlongDimensionMatricesWithShape(char ordering, long rows,
                    long cols, long seed) {
        Nd4j.getRandom().setSeed(seed);
        //From 3d NDArray: do various tensors. One offset 0, one offset > 0
        //[0,1], [0,2], [1,0], [1,2], [2,0], [2,1]
        INDArray[] out = new INDArray[12];

        INDArray temp01 = Nd4j.linspace(1, cols * rows * 4, cols * rows * 4).reshape(cols, rows, 4);
        out[0] = temp01.javaTensorAlongDimension(0, 0, 1).reshape(rows, cols);
        long[] temp01Shape = new long[] {cols, rows, 4};
        int len = ArrayUtil.prod(temp01Shape);
        temp01 = Nd4j.linspace(1, len, len).reshape(temp01Shape);
        out[1] = temp01.javaTensorAlongDimension(2, 0, 1).reshape(rows, cols);

        Nd4j.getRandom().setSeed(seed);
        INDArray temp02 = Nd4j.linspace(1, len, len).reshape(new long[] {cols, 4, rows});
        out[2] = temp02.javaTensorAlongDimension(0, 0, 2).reshape(rows, cols);
        temp02 = Nd4j.linspace(1, len, len).reshape(cols, 4, rows);
        out[3] = temp02.javaTensorAlongDimension(2, 0, 2).reshape(rows, cols);

        INDArray temp10 = Nd4j.linspace(1, len, len).reshape(rows, cols, 4);
        out[4] = temp10.javaTensorAlongDimension(0, 1, 0).reshape(rows, cols);
        temp10 = Nd4j.linspace(1, len, len).reshape(rows, cols, 4);
        out[5] = temp10.javaTensorAlongDimension(2, 1, 0).reshape(rows, cols);

        INDArray temp12 = Nd4j.linspace(1, len, len).reshape(4, cols, rows);
        out[6] = temp12.javaTensorAlongDimension(0, 1, 2).reshape(rows, cols);
        temp12 = Nd4j.linspace(1, len, len).reshape(4, cols, rows);
        out[7] = temp12.javaTensorAlongDimension(2, 1, 2).reshape(rows, cols);

        INDArray temp20 = Nd4j.linspace(1, len, len).reshape(rows, 4, cols);
        out[8] = temp20.javaTensorAlongDimension(0, 2, 0).reshape(rows, cols);
        temp20 = Nd4j.linspace(1, len, len).reshape(rows, 4, cols);
        out[9] = temp20.javaTensorAlongDimension(2, 2, 0).reshape(rows, cols);

        INDArray temp21 = Nd4j.linspace(1, len, len).reshape(4, rows, cols);
        out[10] = temp21.javaTensorAlongDimension(0, 2, 1).reshape(rows, cols);
        temp21 = Nd4j.linspace(1, len, len).reshape(4, rows, cols);
        out[11] = temp21.javaTensorAlongDimension(2, 2, 1).reshape(rows, cols);

        String baseMsg = "getTensorAlongDimensionMatricesWithShape(" + rows + "," + cols + "," + seed + ")";
        List<Pair<INDArray, String>> list = new ArrayList<>(12);

        for (int i = 0; i < out.length; i++)
            list.add(new Pair<>(out[i], baseMsg + ".get(" + i + ")"));

        return list;
    }

    public static List<Pair<INDArray, String>> getTensorAlongDimensionMatricesWithShape(long rows, long cols, long seed) {
        return getTensorAlongDimensionMatricesWithShape(Nd4j.order(), rows, cols, seed);
    }


    public static Pair<INDArray, String> getPermutedWithShape(char ordering, long rows, long cols, long seed) {
        Nd4j.getRandom().setSeed(seed);
        long len = rows * cols;
        INDArray arr = Nd4j.linspace(1, len, len).reshape(cols, rows);
        return new Pair<>(arr.permute(1, 0), "getPermutedWithShape(" + rows + "," + cols + "," + seed + ")");
    }

    public static Pair<INDArray, String> getPermutedWithShape(long rows, long cols, long seed) {
        return getPermutedWithShape(Nd4j.order(), rows, cols, seed);
    }


    public static Pair<INDArray, String> getReshapedWithShape(char ordering, long rows, long cols, long seed) {
        Nd4j.getRandom().setSeed(seed);
        long[] origShape = new long[3];
        if (rows % 2 == 0) {
            origShape[0] = rows / 2;
            origShape[1] = cols;
            origShape[2] = 2;
        } else if (cols % 2 == 0) {
            origShape[0] = rows;
            origShape[1] = cols / 2;
            origShape[2] = 2;
        } else {
            origShape[0] = 1;
            origShape[1] = rows;
            origShape[2] = cols;
        }

        int len = ArrayUtil.prod(origShape);
        INDArray orig = Nd4j.linspace(1, len, len).reshape(ordering, origShape);
        return new Pair<>(orig.reshape(ordering, rows, cols),
                        "getReshapedWithShape(" + rows + "," + cols + "," + seed + ")");
    }

    public static Pair<INDArray, String> getReshapedWithShape(long rows, long cols, long seed) {
        return getReshapedWithShape(Nd4j.order(), rows, cols, seed);
    }

    public static List<Pair<INDArray, String>> getAll3dTestArraysWithShape(int seed, int[] shape) {
        return getAll3dTestArraysWithShape(seed, ArrayUtil.toLongArray(shape));
    }

    public static List<Pair<INDArray, String>> getAll3dTestArraysWithShape(long seed, long... shape) {
        if (shape.length != 3)
            throw new IllegalArgumentException("Shape is not length 3");

        List<Pair<INDArray, String>> list = new ArrayList<>();

        String baseMsg = "getAll3dTestArraysWithShape(" + seed + "," + Arrays.toString(shape) + ").get(";


        val len = ArrayUtil.prodLong(shape);
        //Basic 3d in C and F orders:
        Nd4j.getRandom().setSeed(seed);
        INDArray stdC = Nd4j.linspace(1, len, len).reshape('c', shape);
        INDArray stdF = Nd4j.linspace(1, len, len).reshape('f', shape);
        list.add(new Pair<>(stdC, baseMsg + "0)/Nd4j.linspace(1,len,len)(" + Arrays.toString(shape) + ",'c')"));
        list.add(new Pair<>(stdF, baseMsg + "1)/Nd4j.linspace(1,len,len(" + Arrays.toString(shape) + ",'f')"));

        //Various sub arrays:
        list.addAll(get3dSubArraysWithShape(seed, shape));

        //TAD
        list.addAll(get3dTensorAlongDimensionWithShape(seed, shape));

        //Permuted
        list.addAll(get3dPermutedWithShape(seed, shape));

        //Reshaped
        list.addAll(get3dReshapedWithShape(seed, shape));

        return list;
    }

    public static List<Pair<INDArray, String>> get3dSubArraysWithShape(int seed, int[] shape) {
        return get3dSubArraysWithShape(seed, ArrayUtil.toLongArray(shape));
    }

    public static List<Pair<INDArray, String>> get3dSubArraysWithShape(long seed, long... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get3dSubArraysWithShape(" + seed + "," + Arrays.toString(shape) + ")";
        //Create and return various sub arrays:
        Nd4j.getRandom().setSeed(seed);
        val newShape1 = Arrays.copyOf(shape, shape.length);
        newShape1[0] += 5;
        int len = ArrayUtil.prod(newShape1);
        INDArray temp1 = Nd4j.linspace(1, len, len).reshape(newShape1);
        INDArray subset1 = temp1.get(NDArrayIndex.interval(2, shape[0] + 2), NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset1, baseMsg + ".get(0)"));

        val newShape2 = Arrays.copyOf(shape, shape.length);
        newShape2[1] += 5;
        int len2 = ArrayUtil.prod(newShape2);
        INDArray temp2 = Nd4j.linspace(1, len2, len2).reshape(newShape2);
        INDArray subset2 = temp2.get(NDArrayIndex.all(), NDArrayIndex.interval(3, shape[1] + 3), NDArrayIndex.all());
        list.add(new Pair<>(subset2, baseMsg + ".get(1)"));

        val newShape3 = Arrays.copyOf(shape, shape.length);
        newShape3[2] += 5;
        int len3 = ArrayUtil.prod(newShape3);
        INDArray temp3 = Nd4j.linspace(1, len3, len3).reshape(newShape3);
        INDArray subset3 = temp3.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(4, shape[2] + 4));
        list.add(new Pair<>(subset3, baseMsg + ".get(2)"));

        val newShape4 = Arrays.copyOf(shape, shape.length);
        newShape4[0] += 5;
        newShape4[1] += 5;
        newShape4[2] += 5;
        int len4 = ArrayUtil.prod(newShape4);
        INDArray temp4 = Nd4j.linspace(1, len4, len4).reshape(newShape4);
        INDArray subset4 = temp4.get(NDArrayIndex.interval(4, shape[0] + 4), NDArrayIndex.interval(3, shape[1] + 3),
                        NDArrayIndex.interval(2, shape[2] + 2));
        list.add(new Pair<>(subset4, baseMsg + ".get(3)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get3dTensorAlongDimensionWithShape(int seed, int[] shape) {
        return get3dTensorAlongDimensionWithShape(seed, ArrayUtil.toLongArray(shape));
    }

    public static List<Pair<INDArray, String>> get3dTensorAlongDimensionWithShape(long seed, long... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get3dTensorAlongDimensionWithShape(" + seed + "," + Arrays.toString(shape) + ")";

        //Create some 4d arrays and get subsets using 3d TAD on them
        //This is not an exhaustive list of possible 3d arrays from 4d via TAD

        Nd4j.getRandom().setSeed(seed);
        //            int[] shape4d1 = {shape[2],shape[1],shape[0],3};
        val shape4d1 = new long[]{shape[0], shape[1], shape[2], 3};
        int lenshape4d1 = ArrayUtil.prod(shape4d1);
        INDArray orig1a = Nd4j.linspace(1, lenshape4d1, lenshape4d1).reshape(shape4d1);
        INDArray tad1a = orig1a.javaTensorAlongDimension(0, 0, 1, 2);
        INDArray orig1b = Nd4j.linspace(1, lenshape4d1, lenshape4d1).reshape(shape4d1);
        INDArray tad1b = orig1b.javaTensorAlongDimension(1, 0, 1, 2);

        list.add(new Pair<>(tad1a, baseMsg + ".get(0)"));
        list.add(new Pair<>(tad1b, baseMsg + ".get(1)"));

        long[] shape4d2 = {3, shape[0], shape[1], shape[2]};
        int lenshape4d2 = ArrayUtil.prod(shape4d2);
        INDArray orig2 = Nd4j.linspace(1, lenshape4d2, lenshape4d2).reshape(shape4d2);
        INDArray tad2 = orig2.javaTensorAlongDimension(1, 1, 2, 3);
        list.add(new Pair<>(tad2, baseMsg + ".get(2)"));

        long[] shape4d3 = {shape[0], shape[1], 3, shape[2]};
        int lenshape4d3 = ArrayUtil.prod(shape4d3);
        INDArray orig3 = Nd4j.linspace(1, lenshape4d3, lenshape4d3).reshape(shape4d3);
        INDArray tad3 = orig3.javaTensorAlongDimension(1, 1, 3, 0);
        list.add(new Pair<>(tad3, baseMsg + ".get(3)"));

        long[] shape4d4 = {shape[0], 3, shape[1], shape[2]};
        int lenshape4d4 = ArrayUtil.prod(shape4d4);
        INDArray orig4 = Nd4j.linspace(1, lenshape4d4, lenshape4d4).reshape(shape4d4);
        INDArray tad4 = orig4.javaTensorAlongDimension(1, 2, 0, 3);
        list.add(new Pair<>(tad4, baseMsg + ".get(4)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get3dPermutedWithShape(int seed, int[] shape) {
        return get3dPermutedWithShape(seed, ArrayUtil.toLongArray(shape));
    }

    public static List<Pair<INDArray, String>> get3dPermutedWithShape(long seed, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        long[] createdShape = {shape[1], shape[2], shape[0]};
        int lencreatedShape = ArrayUtil.prod(createdShape);
        INDArray arr = Nd4j.linspace(1, lencreatedShape, lencreatedShape).reshape(createdShape);
        INDArray permuted = arr.permute(2, 0, 1);
        return Collections.singletonList(new Pair<>(permuted,
                        "get3dPermutedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }

    public static List<Pair<INDArray, String>> get3dReshapedWithShape(int seed, int[] shape) {
        return get3dReshapedWithShape(seed, ArrayUtil.toLongArray(shape));
    }

    public static List<Pair<INDArray, String>> get3dReshapedWithShape(long seed, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        long[] shape2d = {shape[0] * shape[2], shape[1]};
        int lenshape2d = ArrayUtil.prod(shape2d);
        INDArray array2d = Nd4j.linspace(1, lenshape2d, lenshape2d).reshape(shape2d);
        INDArray array3d = array2d.reshape(shape);
        return Collections.singletonList(new Pair<>(array3d,
                        "get3dReshapedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }

    public static List<Pair<INDArray, String>> getAll4dTestArraysWithShape(int seed, int... shape) {
        if (shape.length != 4)
            throw new IllegalArgumentException("Shape is not length 4");

        List<Pair<INDArray, String>> list = new ArrayList<>();

        String baseMsg = "getAll4dTestArraysWithShape(" + seed + "," + Arrays.toString(shape) + ").get(";

        //Basic 4d in C and F orders:
        Nd4j.getRandom().setSeed(seed);
        int len = ArrayUtil.prod(shape);
        INDArray stdC = Nd4j.linspace(1, len, len).reshape('c', ArrayUtil.toLongArray(shape));
        INDArray stdF = Nd4j.linspace(1, len, len).reshape('f', ArrayUtil.toLongArray(shape));
        list.add(new Pair<>(stdC, baseMsg + "0)/Nd4j.rand(" + Arrays.toString(shape) + ",'c')"));
        list.add(new Pair<>(stdF, baseMsg + "1)/Nd4j.rand(" + Arrays.toString(shape) + ",'f')"));

        //Various sub arrays:
        list.addAll(get4dSubArraysWithShape(seed, shape));

        //TAD
        list.addAll(get4dTensorAlongDimensionWithShape(seed, shape));

        //Permuted
        list.addAll(get4dPermutedWithShape(seed, shape));

        //Reshaped
        list.addAll(get4dReshapedWithShape(seed, shape));

        return list;
    }

    public static List<Pair<INDArray, String>> get4dSubArraysWithShape(int seed, int... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get4dSubArraysWithShape(" + seed + "," + Arrays.toString(shape) + ")";
        //Create and return various sub arrays:
        Nd4j.getRandom().setSeed(seed);
        int[] newShape1 = Arrays.copyOf(shape, shape.length);
        newShape1[0] += 5;
        int len = ArrayUtil.prod(newShape1);
        INDArray temp1 = Nd4j.linspace(1, len, len).reshape(ArrayUtil.toLongArray(newShape1));
        INDArray subset1 = temp1.get(NDArrayIndex.interval(2, shape[0] + 2), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all());
        list.add(new Pair<>(subset1, baseMsg + ".get(0)"));

        int[] newShape2 = Arrays.copyOf(shape, shape.length);
        newShape2[1] += 5;
        int len2 = ArrayUtil.prod(newShape2);
        INDArray temp2 = Nd4j.linspace(1, len2, len2).reshape(ArrayUtil.toLongArray(newShape2));
        INDArray subset2 = temp2.get(NDArrayIndex.all(), NDArrayIndex.interval(3, shape[1] + 3), NDArrayIndex.all(),
                        NDArrayIndex.all());
        list.add(new Pair<>(subset2, baseMsg + ".get(1)"));

        int[] newShape3 = Arrays.copyOf(shape, shape.length);
        newShape3[2] += 5;
        int len3 = ArrayUtil.prod(newShape3);
        INDArray temp3 = Nd4j.linspace(1, len3, len3).reshape(ArrayUtil.toLongArray(newShape3));
        INDArray subset3 = temp3.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(4, shape[2] + 4),
                        NDArrayIndex.all());
        list.add(new Pair<>(subset3, baseMsg + ".get(2)"));

        int[] newShape4 = Arrays.copyOf(shape, shape.length);
        newShape4[3] += 5;
        int len4 = ArrayUtil.prod(newShape4);
        INDArray temp4 = Nd4j.linspace(1, len4, len4).reshape(ArrayUtil.toLongArray(newShape4));
        INDArray subset4 = temp4.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(3, shape[3] + 3));
        list.add(new Pair<>(subset4, baseMsg + ".get(3)"));

        int[] newShape5 = Arrays.copyOf(shape, shape.length);
        newShape5[0] += 5;
        newShape5[1] += 5;
        newShape5[2] += 5;
        newShape5[3] += 5;
        int len5 = ArrayUtil.prod(newShape5);
        INDArray temp5 = Nd4j.linspace(1, len5, len5).reshape(ArrayUtil.toLongArray(newShape5));
        INDArray subset5 = temp5.get(NDArrayIndex.interval(4, shape[0] + 4), NDArrayIndex.interval(3, shape[1] + 3),
                        NDArrayIndex.interval(2, shape[2] + 2), NDArrayIndex.interval(1, shape[3] + 1));
        list.add(new Pair<>(subset5, baseMsg + ".get(4)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get4dTensorAlongDimensionWithShape(int seed, int... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get4dTensorAlongDimensionWithShape(" + seed + "," + Arrays.toString(shape) + ")";

        //Create some 5d arrays and get subsets using 4d TAD on them
        //This is not an exhausive list of possible 4d arrays from 5d via TAD
        Nd4j.getRandom().setSeed(seed);
        int[] shape4d1 = {3, shape[0], shape[1], shape[2], shape[3]};
        int len = ArrayUtil.prod(shape4d1);
        INDArray orig1a = Nd4j.linspace(1, len, len).reshape(ArrayUtil.toLongArray(shape4d1));
        INDArray tad1a = orig1a.javaTensorAlongDimension(0, 1, 2, 3, 4);
        INDArray orig1b = Nd4j.linspace(1, len, len).reshape(ArrayUtil.toLongArray(shape4d1));
        INDArray tad1b = orig1b.javaTensorAlongDimension(2, 1, 2, 3, 4);

        list.add(new Pair<>(tad1a, baseMsg + ".get(0)"));
        list.add(new Pair<>(tad1b, baseMsg + ".get(1)"));

        int[] shape4d2 = {3, shape[0], shape[1], shape[2], shape[3]};
        int len2 = ArrayUtil.prod(shape4d2);
        INDArray orig2 = Nd4j.linspace(1, len2, len2).reshape(ArrayUtil.toLongArray(shape4d2));
        INDArray tad2 = orig2.javaTensorAlongDimension(1, 3, 4, 2, 1);
        list.add(new Pair<>(tad2, baseMsg + ".get(2)"));

        int[] shape4d3 = {shape[0], shape[1], 3, shape[2], shape[3]};
        int len3 = ArrayUtil.prod(shape4d3);
        INDArray orig3 = Nd4j.linspace(1, len3, len3).reshape(ArrayUtil.toLongArray(shape4d3));
        INDArray tad3 = orig3.javaTensorAlongDimension(1, 4, 1, 3, 0);
        list.add(new Pair<>(tad3, baseMsg + ".get(3)"));

        int[] shape4d4 = {shape[0], shape[1], shape[2], shape[3], 3};
        int len4 = ArrayUtil.prod(shape4d4);
        INDArray orig4 = Nd4j.linspace(1, len4, len4).reshape(ArrayUtil.toLongArray(shape4d4));
        INDArray tad4 = orig4.javaTensorAlongDimension(1, 2, 0, 3, 1);
        list.add(new Pair<>(tad4, baseMsg + ".get(4)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get4dPermutedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] createdShape = {shape[1], shape[3], shape[2], shape[0]};
        INDArray arr = Nd4j.rand(createdShape);
        INDArray permuted = arr.permute(3, 0, 2, 1);
        return Collections.singletonList(new Pair<>(permuted,
                        "get4dPermutedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }

    public static List<Pair<INDArray, String>> get4dReshapedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] shape2d = {shape[0] * shape[2], shape[1] * shape[3]};
        INDArray array2d = Nd4j.rand(shape2d);
        INDArray array3d = array2d.reshape(ArrayUtil.toLongArray(shape));
        return Collections.singletonList(new Pair<>(array3d,
                        "get4dReshapedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }



    public static List<Pair<INDArray, String>> getAll5dTestArraysWithShape(int seed, int... shape) {
        if (shape.length != 5)
            throw new IllegalArgumentException("Shape is not length 5");

        List<Pair<INDArray, String>> list = new ArrayList<>();

        String baseMsg = "getAll5dTestArraysWithShape(" + seed + "," + Arrays.toString(shape) + ").get(";

        //Basic 5d in C and F orders:
        Nd4j.getRandom().setSeed(seed);
        INDArray stdC = Nd4j.rand(shape, 'c');
        INDArray stdF = Nd4j.rand(shape, 'f');
        list.add(new Pair<>(stdC, baseMsg + "0)/Nd4j.rand(" + Arrays.toString(shape) + ",'c')"));
        list.add(new Pair<>(stdF, baseMsg + "1)/Nd4j.rand(" + Arrays.toString(shape) + ",'f')"));

        //Various sub arrays:
        list.addAll(get5dSubArraysWithShape(seed, shape));

        //TAD
        list.addAll(get5dTensorAlongDimensionWithShape(seed, shape));

        //Permuted
        list.addAll(get5dPermutedWithShape(seed, shape));

        //Reshaped
        list.addAll(get5dReshapedWithShape(seed, shape));

        return list;
    }

    public static List<Pair<INDArray, String>> get5dSubArraysWithShape(int seed, int... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get5dSubArraysWithShape(" + seed + "," + Arrays.toString(shape) + ")";
        //Create and return various sub arrays:
        Nd4j.getRandom().setSeed(seed);
        int[] newShape1 = Arrays.copyOf(shape, shape.length);
        newShape1[0] += 5;
        INDArray temp1 = Nd4j.rand(newShape1);
        INDArray subset1 = temp1.get(NDArrayIndex.interval(2, shape[0] + 2), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset1, baseMsg + ".get(0)"));

        int[] newShape2 = Arrays.copyOf(shape, shape.length);
        newShape2[1] += 5;
        INDArray temp2 = Nd4j.rand(newShape2);
        INDArray subset2 = temp2.get(NDArrayIndex.all(), NDArrayIndex.interval(3, shape[1] + 3), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset2, baseMsg + ".get(1)"));

        int[] newShape3 = Arrays.copyOf(shape, shape.length);
        newShape3[2] += 5;
        INDArray temp3 = Nd4j.rand(newShape3);
        INDArray subset3 = temp3.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(4, shape[2] + 4),
                        NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset3, baseMsg + ".get(2)"));

        int[] newShape4 = Arrays.copyOf(shape, shape.length);
        newShape4[3] += 5;
        INDArray temp4 = Nd4j.rand(newShape4);
        INDArray subset4 = temp4.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(3, shape[3] + 3), NDArrayIndex.all());
        list.add(new Pair<>(subset4, baseMsg + ".get(3)"));

        int[] newShape5 = Arrays.copyOf(shape, shape.length);
        newShape5[4] += 5;
        INDArray temp5 = Nd4j.rand(newShape5);
        INDArray subset5 = temp5.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(3, shape[4] + 3));
        list.add(new Pair<>(subset5, baseMsg + ".get(4)"));

        int[] newShape6 = Arrays.copyOf(shape, shape.length);
        newShape6[0] += 5;
        newShape6[1] += 5;
        newShape6[2] += 5;
        newShape6[3] += 5;
        newShape6[4] += 5;
        INDArray temp6 = Nd4j.rand(newShape6);
        INDArray subset6 = temp6.get(NDArrayIndex.interval(4, shape[0] + 4), NDArrayIndex.interval(3, shape[1] + 3),
                        NDArrayIndex.interval(2, shape[2] + 2), NDArrayIndex.interval(1, shape[3] + 1),
                        NDArrayIndex.interval(2, shape[4] + 2));
        list.add(new Pair<>(subset6, baseMsg + ".get(5)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get5dTensorAlongDimensionWithShape(int seed, int... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get5dTensorAlongDimensionWithShape(" + seed + "," + Arrays.toString(shape) + ")";

        //Create some 6d arrays and get subsets using 5d TAD on them
        //This is not an exhausive list of possible 5d arrays from 6d via TAD
        Nd4j.getRandom().setSeed(seed);
        int[] shape4d1 = {3, shape[0], shape[1], shape[2], shape[3], shape[4]};
        INDArray orig1a = Nd4j.rand(shape4d1);
        INDArray tad1a = orig1a.javaTensorAlongDimension(0, 1, 2, 3, 4, 5);
        INDArray orig1b = Nd4j.rand(shape4d1);
        INDArray tad1b = orig1b.javaTensorAlongDimension(2, 1, 2, 3, 4, 5);

        list.add(new Pair<>(tad1a, baseMsg + ".get(0)"));
        list.add(new Pair<>(tad1b, baseMsg + ".get(1)"));

        int[] shape4d2 = {3, shape[0], shape[1], shape[2], shape[3], shape[4]};
        INDArray orig2 = Nd4j.rand(shape4d2);
        INDArray tad2 = orig2.javaTensorAlongDimension(1, 3, 5, 4, 2, 1);
        list.add(new Pair<>(tad2, baseMsg + ".get(2)"));

        int[] shape4d3 = {shape[0], shape[1], shape[2], shape[3], shape[4], 2};
        INDArray orig3 = Nd4j.rand(shape4d3);
        INDArray tad3 = orig3.javaTensorAlongDimension(1, 4, 1, 3, 2, 0);
        list.add(new Pair<>(tad3, baseMsg + ".get(3)"));

        int[] shape4d4 = {shape[0], shape[1], shape[2], shape[3], 3, shape[4]};
        INDArray orig4 = Nd4j.rand(shape4d4);
        INDArray tad4 = orig4.javaTensorAlongDimension(1, 5, 2, 0, 3, 1);
        list.add(new Pair<>(tad4, baseMsg + ".get(4)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get5dPermutedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] createdShape = {shape[1], shape[4], shape[3], shape[2], shape[0]};
        INDArray arr = Nd4j.rand(createdShape);
        INDArray permuted = arr.permute(4, 0, 3, 2, 1);
        return Collections.singletonList(new Pair<>(permuted,
                        "get5dPermutedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }

    public static List<Pair<INDArray, String>> get5dReshapedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] shape2d = {shape[0] * shape[2], shape[4], shape[1] * shape[3]};
        INDArray array3d = Nd4j.rand(shape2d);
        INDArray array5d = array3d.reshape(ArrayUtil.toLongArray(shape));
        return Collections.singletonList(new Pair<>(array5d,
                        "get5dReshapedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }



    public static List<Pair<INDArray, String>> getAll6dTestArraysWithShape(int seed, int... shape) {
        if (shape.length != 6)
            throw new IllegalArgumentException("Shape is not length 6");

        List<Pair<INDArray, String>> list = new ArrayList<>();

        String baseMsg = "getAll6dTestArraysWithShape(" + seed + "," + Arrays.toString(shape) + ").get(";

        //Basic 5d in C and F orders:
        Nd4j.getRandom().setSeed(seed);
        INDArray stdC = Nd4j.rand(shape, 'c');
        INDArray stdF = Nd4j.rand(shape, 'f');
        list.add(new Pair<>(stdC, baseMsg + "0)/Nd4j.rand(" + Arrays.toString(shape) + ",'c')"));
        list.add(new Pair<>(stdF, baseMsg + "1)/Nd4j.rand(" + Arrays.toString(shape) + ",'f')"));

        //Various sub arrays:
        list.addAll(get6dSubArraysWithShape(seed, shape));

        //Permuted
        list.addAll(get6dPermutedWithShape(seed, shape));

        //Reshaped
        list.addAll(get6dReshapedWithShape(seed, shape));

        return list;
    }

    public static List<Pair<INDArray, String>> get6dSubArraysWithShape(int seed, int... shape) {
        List<Pair<INDArray, String>> list = new ArrayList<>();
        String baseMsg = "get6dSubArraysWithShape(" + seed + "," + Arrays.toString(shape) + ")";
        //Create and return various sub arrays:
        Nd4j.getRandom().setSeed(seed);
        int[] newShape1 = Arrays.copyOf(shape, shape.length);
        newShape1[0] += 5;
        INDArray temp1 = Nd4j.rand(newShape1);
        INDArray subset1 = temp1.get(NDArrayIndex.interval(2, shape[0] + 2), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset1, baseMsg + ".get(0)"));

        int[] newShape2 = Arrays.copyOf(shape, shape.length);
        newShape2[1] += 5;
        INDArray temp2 = Nd4j.rand(newShape2);
        INDArray subset2 = temp2.get(NDArrayIndex.all(), NDArrayIndex.interval(3, shape[1] + 3), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset2, baseMsg + ".get(1)"));

        int[] newShape3 = Arrays.copyOf(shape, shape.length);
        newShape3[2] += 5;
        INDArray temp3 = Nd4j.rand(newShape3);
        INDArray subset3 = temp3.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(4, shape[2] + 4),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset3, baseMsg + ".get(2)"));

        int[] newShape4 = Arrays.copyOf(shape, shape.length);
        newShape4[3] += 5;
        INDArray temp4 = Nd4j.rand(newShape4);
        INDArray subset4 = temp4.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(3, shape[3] + 3), NDArrayIndex.all(), NDArrayIndex.all());
        list.add(new Pair<>(subset4, baseMsg + ".get(3)"));

        int[] newShape5 = Arrays.copyOf(shape, shape.length);
        newShape5[4] += 5;
        INDArray temp5 = Nd4j.rand(newShape5);
        INDArray subset5 = temp5.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(3, shape[4] + 3), NDArrayIndex.all());
        list.add(new Pair<>(subset5, baseMsg + ".get(4)"));

        int[] newShape6 = Arrays.copyOf(shape, shape.length);
        newShape6[5] += 5;
        INDArray temp6 = Nd4j.rand(newShape6);
        INDArray subset6 = temp6.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.interval(1, shape[5] + 1));
        list.add(new Pair<>(subset6, baseMsg + ".get(5)"));

        int[] newShape7 = Arrays.copyOf(shape, shape.length);
        newShape7[0] += 5;
        newShape7[1] += 5;
        newShape7[2] += 5;
        newShape7[3] += 5;
        newShape7[4] += 5;
        newShape7[5] += 5;
        INDArray temp7 = Nd4j.rand(newShape7);
        INDArray subset7 = temp7.get(NDArrayIndex.interval(4, shape[0] + 4), NDArrayIndex.interval(3, shape[1] + 3),
                        NDArrayIndex.interval(2, shape[2] + 2), NDArrayIndex.interval(1, shape[3] + 1),
                        NDArrayIndex.interval(2, shape[4] + 2), NDArrayIndex.interval(3, shape[5] + 3));
        list.add(new Pair<>(subset7, baseMsg + ".get(6)"));

        return list;
    }

    public static List<Pair<INDArray, String>> get6dPermutedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] createdShape = {shape[1], shape[4], shape[5], shape[3], shape[2], shape[0]};
        INDArray arr = Nd4j.rand(createdShape);
        INDArray permuted = arr.permute(5, 0, 4, 3, 1, 2);
        return Collections.singletonList(new Pair<>(permuted,
                        "get6dPermutedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }

    public static List<Pair<INDArray, String>> get6dReshapedWithShape(int seed, int... shape) {
        Nd4j.getRandom().setSeed(seed);
        int[] shape3d = {shape[0] * shape[2], shape[4] * shape[5], shape[1] * shape[3]};
        INDArray array3d = Nd4j.rand(shape3d);
        INDArray array6d = array3d.reshape(ArrayUtil.toLongArray(shape));
        return Collections.singletonList(new Pair<>(array6d,
                        "get6dReshapedWithShape(" + seed + "," + Arrays.toString(shape) + ").get(0)"));
    }


    /**
     * Create an ndarray
     * of
     * @param seed
     * @param rank
     * @param numShapes
     * @return
     */
    public static int[][] getRandomBroadCastShape(long seed, int rank, int numShapes) {
        Nd4j.getRandom().setSeed(seed);
        INDArray coinFlip = Nd4j.getDistributions().createBinomial(1, 0.5).sample(new int[] {numShapes, rank});
        int[][] ret = new int[(int) coinFlip.rows()][(int) coinFlip.columns()];
        for (int i = 0; i < coinFlip.rows(); i++) {
            for (int j = 0; j < coinFlip.columns(); j++) {
                int set = coinFlip.getInt(i, j);
                if (set > 0)
                    ret[i][j] = set;
                else {
                    //anything from 0 to 9
                    ret[i][j] = Nd4j.getRandom().nextInt(9) + 1;
                }
            }
        }

        return ret;
    }


    /**
     * Generate a random shape to
     * broadcast to
     * given a randomly generated
     * shape with 1s in it as inputs
     * @param inputShapeWithOnes
     * @param seed
     * @return
     */
    public static int[] broadcastToShape(int[] inputShapeWithOnes, long seed) {
        Nd4j.getRandom().setSeed(seed);
        int[] shape = new int[inputShapeWithOnes.length];
        for (int i = 0; i < shape.length; i++) {
            if (inputShapeWithOnes[i] == 1) {
                shape[i] = Nd4j.getRandom().nextInt(9) + 1;
            } else
                shape[i] = inputShapeWithOnes[i];
        }

        return shape;
    }

    public static long[] broadcastToShape(long[] inputShapeWithOnes, long seed) {
        Nd4j.getRandom().setSeed(seed);
        val shape = new long[inputShapeWithOnes.length];
        for (int i = 0; i < shape.length; i++) {
            if (inputShapeWithOnes[i] == 1) {
                shape[i] = Nd4j.getRandom().nextInt(9) + 1;
            } else
                shape[i] = inputShapeWithOnes[i];
        }

        return shape;
    }

}
