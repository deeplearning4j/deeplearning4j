/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.primitives.Pair;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpaceND;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for NDArray reshape, permute, transpose, tile, and related structural operations.
 * Extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayReshapePermuteTest extends BaseNd4jTestWithBackends {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000;
    }

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorAlongDimension(Nd4jBackend backend) {
        val shape = new long[] {4, 5, 7};
        int length = ArrayUtil.prod(shape);
        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape(shape);

        int[] dim0s = {0, 1, 2, 0, 1, 2};
        int[] dim1s = {1, 0, 0, 2, 2, 1};

        double[] sums = {1350., 1350., 1582, 1582, 630, 630};

        for (int i = 0; i < dim0s.length; i++) {
            int firstDim = dim0s[i];
            int secondDim = dim1s[i];
            INDArray tad = arr.tensorAlongDimension(0, firstDim, secondDim);
            tad.sumNumber();
            //            assertEquals("I " + i + " failed ",sums[i],tad.sumNumber().doubleValue(),1e-1);
        }

        INDArray testMem = Nd4j.create(10, 10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetDouble(Nd4jBackend backend) {
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30, DataType.DOUBLE).data(), new long[] {3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray slice0 = swapped.slice(0).slice(1);
        INDArray assertion = Nd4j.create(new double[] {2, 4, 6, 8, 10});
        assertEquals(assertion, slice0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDimShuffle(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[] {0, 'x', 1}, new int[] {0, 1}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[] {1, 'x', 0}, new int[] {1, 0}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, reverse.shape()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(100).data(), new long[] {5, 5, 4}).castTo(DataType.DOUBLE);
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new long[] {4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).reshape(1, -1);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());

        INDArray linspaced = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray transposed = Nd4j.create(new double[] {1, 3, 2, 4}, new long[] {2, 2});
        INDArray linSpacedT = linspaced.transpose();
        assertEquals(transposed, linSpacedT);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTile(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray repeated = x.repeat(0, 2);
        assertEquals(8, repeated.length());
        INDArray repeatAlongDimension = x.repeat(1, new long[] {2});
        INDArray assertionRepeat = Nd4j.create(new double[][] {{1, 1, 2, 2}, {3, 3, 4, 4}});
        assertArrayEquals(new long[] {2, 4}, assertionRepeat.shape());
        assertEquals(assertionRepeat, repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[] {0, 1, 2}).reshape(1, 3);
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{0, 1, 2, 0, 1, 2}, {0, 1, 2, 0, 1, 2}});
        assertEquals(assertion, tile);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeOneReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[] {0, 1, 2});
        INDArray newShape = arr.reshape(-1);
        assertEquals(newShape, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20, DataType.DOUBLE).data(), new long[] {5, 4}).castTo(DataType.DOUBLE);
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);

        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray permuted = toPermute.permute(2, 1, 0);
        INDArray assertion = Nd4j.create(new double[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(permuted, assertion);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermutei(Nd4jBackend backend) {
        //Check in-place permute vs. copy array permute

        //2d:
        INDArray orig = Nd4j.linspace(1, 3 * 4, 3 * 4, DataType.DOUBLE).reshape('c', 3, 4).castTo(DataType.DOUBLE);
        INDArray exp01 = orig.permute(0, 1);
        INDArray exp10 = orig.permute(1, 0);
        List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE);
        List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE);
        for (int i = 0; i < list1.size(); i++) {
            INDArray p1 = list1.get(i).getFirst().assign(orig).permutei(0, 1);
            INDArray p2 = list2.get(i).getFirst().assign(orig).permutei(1, 0);

            assertEquals(exp01, p1);
            assertEquals(exp10, p2);

            assertEquals(3, p1.rows());
            assertEquals(4, p1.columns());

            assertEquals(4, p2.rows());
            assertEquals(3, p2.columns());
        }

        //2d, v2
        orig = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 1, 4);
        exp01 = orig.permute(0, 1);
        exp10 = orig.permute(1, 0);
        list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345, DataType.DOUBLE);
        list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345, DataType.DOUBLE);
        for (int i = 0; i < list1.size(); i++) {
            INDArray p1 = list1.get(i).getFirst().assign(orig).permutei(0, 1);
            INDArray p2 = list2.get(i).getFirst().assign(orig).permutei(1, 0);

            assertEquals(exp01, p1);
            assertEquals(exp10, p2);

            assertEquals(1, p1.rows());
            assertEquals(4, p1.columns());
            assertEquals(4, p2.rows());
            assertEquals(1, p2.columns());
            assertTrue(p1.isRowVector());
            assertFalse(p1.isColumnVector());
            assertFalse(p2.isRowVector());
            assertTrue(p2.isColumnVector());
        }

        //3d:
        INDArray orig3d = Nd4j.linspace(1, 3 * 4 * 5, 3 * 4 * 5, DataType.DOUBLE).reshape('c', 3, 4, 5);
        INDArray exp012 = orig3d.permute(0, 1, 2);
        INDArray exp021 = orig3d.permute(0, 2, 1);
        INDArray exp120 = orig3d.permute(1, 2, 0);
        INDArray exp102 = orig3d.permute(1, 0, 2);
        INDArray exp201 = orig3d.permute(2, 0, 1);
        INDArray exp210 = orig3d.permute(2, 1, 0);

        List<Pair<INDArray, String>> list012 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list021 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list120 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list102 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list201 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list210 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);

        for (int i = 0; i < list012.size(); i++) {
            INDArray p1 = list012.get(i).getFirst().assign(orig3d).permutei(0, 1, 2);
            INDArray p2 = list021.get(i).getFirst().assign(orig3d).permutei(0, 2, 1);
            INDArray p3 = list120.get(i).getFirst().assign(orig3d).permutei(1, 2, 0);
            INDArray p4 = list102.get(i).getFirst().assign(orig3d).permutei(1, 0, 2);
            INDArray p5 = list201.get(i).getFirst().assign(orig3d).permutei(2, 0, 1);
            INDArray p6 = list210.get(i).getFirst().assign(orig3d).permutei(2, 1, 0);

            assertEquals(exp012, p1);
            assertEquals(exp021, p2);
            assertEquals(exp120, p3);
            assertEquals(exp102, p4);
            assertEquals(exp201, p5);
            assertEquals(exp210, p6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteiShape(Nd4jBackend backend) {
        INDArray row = Nd4j.create(1, 10).castTo(DataType.DOUBLE);

        INDArray permutedCopy = row.permute(1, 0);
        INDArray permutedInplace = row.permutei(1, 0);

        assertArrayEquals(new long[] {10, 1}, permutedCopy.shape());
        assertArrayEquals(new long[] {10, 1}, permutedInplace.shape());

        assertEquals(10, permutedCopy.rows());
        assertEquals(10, permutedInplace.rows());

        assertEquals(1, permutedCopy.columns());
        assertEquals(1, permutedInplace.columns());

        INDArray col = Nd4j.create(10, 1);
        INDArray cPermutedCopy = col.permute(1, 0);
        INDArray cPermutedInplace = col.permutei(1, 0);

        assertArrayEquals(new long[] {1, 10}, cPermutedCopy.shape());
        assertArrayEquals(new long[] {1, 10}, cPermutedInplace.shape());

        assertEquals(1, cPermutedCopy.rows());
        assertEquals(1, cPermutedInplace.rows());

        assertEquals(10, cPermutedCopy.columns());
        assertEquals(10, cPermutedInplace.columns());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSwapAxes(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(0, 7, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray assertion = n.permute(2, 1, 0);
        INDArray permuteTranspose = assertion.slice(1).slice(1);
        INDArray validate = Nd4j.create(new double[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(validate, assertion);

        INDArray thirty = Nd4j.linspace(1, 30, 30, DataType.DOUBLE).reshape(3, 5, 2);
        INDArray swapped = thirty.swapAxes(2, 1);
        INDArray slice = swapped.slice(0).slice(0);
        INDArray assertion2 = Nd4j.create(new double[] {1, 3, 5, 7, 9});
        assertEquals(assertion2, slice);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data(), new long[] {4, 3, 2});
        INDArray reshaped = arr.reshape(2, 3, 4);
        assertEquals(arr.length(), reshaped.length());
        assertEquals(true, Arrays.equals(new long[] {4, 3, 2}, arr.shape()));
        assertEquals(true, Arrays.equals(new long[] {2, 3, 4}, reshaped.shape()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTemp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new long[] {2, 2, 2}).castTo(DataType.DOUBLE);
        INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
        INDArray out = permuted.reshape(4, 2);

        int countZero = 0;
        for (int i = 0; i < 8; i++)
            if (out.getDouble(i) == 0.0)
                countZero++;
        assertEquals(countZero, 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadPermuteEquals(Nd4jBackend backend) {
        INDArray d3c = Nd4j.linspace(1, 5, 5, DataType.DOUBLE).reshape('c', 1, 5, 1);
        INDArray d3f = d3c.dup('f');

        INDArray tadCi = d3c.tensorAlongDimension(0, 1, 2).permutei(1, 0);
        INDArray tadFi = d3f.tensorAlongDimension(0, 1, 2).permutei(1, 0);

        INDArray tadC = d3c.tensorAlongDimension(0, 1, 2).permute(1, 0);
        INDArray tadF = d3f.tensorAlongDimension(0, 1, 2).permute(1, 0);

        assertArrayEquals(tadCi.shape(), tadC.shape());
        assertArrayEquals(tadCi.stride(), tadC.stride());
        assertArrayEquals(tadCi.data().asDouble(), tadC.data().asDouble(), 1e-8);
        assertEquals(tadC, tadCi.dup());
        assertEquals(tadC, tadCi);

        assertArrayEquals(tadFi.shape(), tadF.shape());
        assertArrayEquals(tadFi.stride(), tadF.stride());
        assertArrayEquals(tadFi.data().asDouble(), tadF.data().asDouble(), 1e-8);

        assertEquals(tadF, tadFi.dup());
        assertEquals(tadF, tadFi);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeFailure(Nd4jBackend backend) {
        assertThrows(RuntimeException.class, () -> {
            val a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
            val b = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
            val score = a.mmul(b);
            val reshaped1 = score.reshape(2, 100);
            val reshaped2 = score.reshape(2, 1);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeScalar(Nd4jBackend backend) {
        val scalar = Nd4j.scalar(2.0f);
        val newShape = scalar.reshape(1, 1, 1, 1);

        assertEquals(4, newShape.rank());
        assertArrayEquals(new long[]{1, 1, 1, 1}, newShape.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeVector(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6});
        val newShape = vector.reshape(3, 2);

        assertEquals(2, newShape.rank());
        assertArrayEquals(new long[]{3, 2}, newShape.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarSqueeze(Nd4jBackend backend) {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1, 1});
        val output = Nd4j.scalar(0.0f);
        val exp = Nd4j.scalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarVectorSqueeze(Nd4jBackend backend) {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1});

        assertArrayEquals(new long[]{1}, scalar.shape());

        val output = Nd4j.scalar(0.0f);
        val exp = Nd4j.scalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorSqueeze(Nd4jBackend backend) {
        val vector = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new long[]{1, 6});
        val output = Nd4j.createFromArray(new float[] {0, 0, 0, 0, 0, 0});
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6});

        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(vector)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{6}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrixReshape(Nd4jBackend backend) {
        val matrix = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {3, 3});
        val exp = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {9});

        val reshaped = matrix.reshape(-1);

        assertArrayEquals(exp.shape(), reshaped.shape());
        assertEquals(exp, reshaped);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTile_1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        val exp = Nd4j.create(new double[] {1.000000, 2.000000, 3.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 4.000000, 5.000000, 6.000000, 1.000000, 2.000000, 3.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 4.000000, 5.000000, 6.000000}, new int[] {4, 6});
        val output = Nd4j.create(4, 6);

        val op = DynamicCustomOp.builder("tile")
                .addInputs(array)
                .addIntegerArguments(2, 2)
                .addOutputs(output)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose_Custom() {
        INDArray arr = Nd4j.linspace(1, 15, 15, DataType.DOUBLE).reshape(5, 3);
        INDArray out = Nd4j.create(3, 5);

        val op = DynamicCustomOp.builder("transpose")
                .addInputs(arr)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        val exp = arr.transpose();
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransposei() {
        INDArray arr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        INDArray ti = arr.transposei();
        assertArrayEquals(new long[]{4, 3}, ti.shape());
        assertArrayEquals(new long[]{4, 3}, arr.shape());

        assertTrue(arr == ti);  //Should be same object
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeEnforce() {
        INDArray arr = Nd4j.create(new long[]{2, 2}, 'c');
        INDArray arr2 = arr.reshape('c', true, 4, 1);

        INDArray arr1a = Nd4j.create(new long[]{2, 3}, 'c').get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray arr3 = arr1a.reshape('c', false, 4, 1);
        boolean isView = arr3.isView();
        assertFalse(isView);     //Should be copy

        try {
            INDArray arr4 = arr1a.reshape('c', true, 4, 1);
            fail("Expected exception");
        } catch (ND4JIllegalStateException e) {
            assertTrue(e.getMessage().contains("Unable to reshape array as view"), e.getMessage());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatSimple() {
        INDArray arr = Nd4j.createFromArray(new double[][]{
                {1, 2, 3}, {4, 5, 6}});

        INDArray r0 = arr.repeat(0, 2);

        INDArray exp0 = Nd4j.createFromArray(new double[][]{
                {1, 2, 3},
                {1, 2, 3},
                {4, 5, 6},
                {4, 5, 6}});

        assertEquals(exp0, r0);

        INDArray r1 = arr.repeat(1, 2);
        INDArray exp1 = Nd4j.createFromArray(new double[][]{
                {1, 1, 2, 2, 3, 3}, {4, 4, 5, 5, 6, 6}});
        assertEquals(exp1, r1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatStrided(Nd4jBackend backend) {
        // Create a 2D array (shape 5x5)
        INDArray array = Nd4j.arange(25).reshape(5, 5);

        // Get first column (shape 5x1)
        INDArray slice = array.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(5, 1);

        // Repeat column on sliced array (shape 5x3)
        INDArray repeatedSlice = slice.repeat(1, (long) 3);

        // Same thing but copy array first
        INDArray repeatedDup = slice.dup().repeat(1, (long) 3);

        // Check result
        assertEquals(repeatedSlice, repeatedDup);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceKeepDimsShape() {
        INDArray arr = Nd4j.create(3, 4);
        INDArray out = arr.sum(true, 1);
        assertArrayEquals(new long[]{3, 1}, out.shape());

        INDArray out2 = arr.sum(true, 0);
        assertArrayEquals(new long[]{1, 4}, out2.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceRow() {
        double[] data = new double[]{15.0, 16.0};
        INDArray vector = Nd4j.createFromArray(data).reshape(1, 2);
        INDArray slice = vector.slice(0);
        assertEquals(vector.reshape(2), slice);
        slice.assign(-1);
        assertEquals(Nd4j.createFromArray(-1.0, -1.0).reshape(1, 2), vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceMatrix() {
        INDArray arr = Nd4j.arange(4).reshape(2, 2);
        arr.slice(0);
        arr.slice(1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBatchToSpace(Nd4jBackend backend) {
        INDArray out = Nd4j.create(DataType.FLOAT, 2, 4, 5);
        DynamicCustomOp c = new BatchToSpaceND();

        c.addInputArgument(
                Nd4j.rand(DataType.FLOAT, new int[]{4, 4, 3}),
                Nd4j.createFromArray(1, 2),
                Nd4j.createFromArray(new int[][]{ new int[]{0, 0}, new int[]{0, 1} })
        );
        c.addOutputArgument(out);
        Nd4j.getExecutioner().exec(c);

        List<DataBuffer> l = c.calculateOutputShape();

        //from [4,4,3] to [2,4,6] then crop to [2,4,5]
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyReshapingMinus1() {
        INDArray arr0 = Nd4j.create(DataType.FLOAT, 2, 0);
        INDArray arr1 = Nd4j.create(DataType.FLOAT, 0, 1, 2);

        INDArray out0 = Nd4j.exec(new Reshape(arr0, Nd4j.createFromArray(2, 0, -1)))[0];
        INDArray out1 = Nd4j.exec(new Reshape(arr1, Nd4j.createFromArray(-1, 1)))[0];
        INDArray out2 = Nd4j.exec(new Reshape(arr1, Nd4j.createFromArray(10, -1)))[0];

        assertArrayEquals(new long[]{2, 0, 1}, out0.shape());
        assertArrayEquals(new long[]{0, 1}, out1.shape());
        assertArrayEquals(new long[]{10, 0}, out2.shape());
    }
}
