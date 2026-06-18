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

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for vstack, hstack, concat, and stack operations extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayStackConcatTest extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackDifferentOrders(Nd4jBackend backend) {
        INDArray expected = Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3);

        for (char order : new char[] {'c', 'f'}) {

            INDArray arr1 = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape( 2, 3).dup('c');
            INDArray arr2 = Nd4j.linspace(7, 9, 3, DataType.DOUBLE).reshape(1, 3).dup('c');

            Nd4j.factory().setOrder(order);

            INDArray merged = Nd4j.vstack(arr1, arr2);

            assertEquals( expected, merged,"Failed for [" + order + "] order");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackEdgeCase(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray vstacked = Nd4j.vstack(arr);
        assertEquals(arr.reshape(1,4), vstacked);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat(Nd4jBackend backend) {
        INDArray A = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(2, 2, 2);
        INDArray B = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        INDArray concat = Nd4j.concat(0, A, B);
        assertTrue(Arrays.equals(new long[] {5, 2, 2}, concat.shape()));

        INDArray columnConcat = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray concatWith = Nd4j.zeros(2, 3);
        INDArray columnWiseConcat = Nd4j.concat(0, columnConcat, concatWith);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatHorizontally(Nd4jBackend backend) {
        INDArray rowVector = Nd4j.ones(1, 5);
        INDArray other = Nd4j.ones(1, 5);
        INDArray concat = Nd4j.hstack(other, rowVector);
        assertEquals(rowVector.rows(), concat.rows());
        assertEquals(rowVector.columns() * 2, concat.columns());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecialConcat1(Nd4jBackend backend) {
        for (int i = 0; i < 10; i++) {
            List<INDArray> arrays = new ArrayList<>();
            for (int x = 0; x < 10; x++) {
                arrays.add(Nd4j.create(1, 100).assign(x).castTo(DataType.DOUBLE));
            }

            INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
            assertEquals(10, matrix.rows());
            assertEquals(100, matrix.columns());

            for (int x = 0; x < 10; x++) {
                assertEquals(x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
                assertEquals(arrays.get(x), matrix.getRow(x).reshape(1,matrix.size(1)));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecialConcat2(Nd4jBackend backend) {
        List<INDArray> arrays = new ArrayList<>();
        for (int x = 0; x < 10; x++) {
            arrays.add(Nd4j.create(new double[] {x, x, x, x, x, x}).reshape(1, 6));
        }

        INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
        assertEquals(10, matrix.rows());
        assertEquals(6, matrix.columns());

        for (int x = 0; x < 10; x++) {
            assertEquals(x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
            assertEquals(arrays.get(x), matrix.getRow(x).reshape(1, matrix.size(1)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorScalarConcat(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[] {1, 2});
        val scalar = Nd4j.scalar(3.0f);

        val output = Nd4j.createFromArray(new float[]{0, 0, 0});
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3});

        val op = DynamicCustomOp.builder("concat")
                .addInputs(vector, scalar)
                .addOutputs(output)
                .addIntegerArguments(0) // axis
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(exp.shape(), Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(exp.shape(), output.shape());
        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat_1(Nd4jBackend backend) {
        for(char order : new char[]{'c', 'f'}) {

            INDArray arr1 = Nd4j.create(new double[]{1, 2}, new long[]{1, 2}, order);
            INDArray arr2 = Nd4j.create(new double[]{3, 4}, new long[]{1, 2}, order);

            INDArray out = Nd4j.concat(0, arr1, arr2);
            Nd4j.getExecutioner().commit();
            INDArray exp = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
            assertEquals(exp, out,String.valueOf(order));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStack(){
        INDArray in = Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4);
        INDArray in2 = in.add(100);

        for( int i=-3; i<3; i++ ){
            INDArray out = Nd4j.stack(i, in, in2);
            long[] expShape;
            switch (i){
                case -3:
                case 0:
                    expShape = new long[]{2,3,4};
                    break;
                case -2:
                case 1:
                    expShape = new long[]{3,2,4};
                    break;
                case -1:
                case 2:
                    expShape = new long[]{3,4,2};
                    break;
                default:
                    throw new RuntimeException(String.valueOf(i));
            }
            assertArrayEquals(expShape, out.shape(),String.valueOf(i));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackRank1(){
        List<INDArray> list = new ArrayList<>();
        list.add(Nd4j.linspace(1,3,3, DataType.DOUBLE));
        list.add(Nd4j.linspace(4,6,3, DataType.DOUBLE));
        list.add(Nd4j.linspace(7,9,3, DataType.DOUBLE));

        INDArray out = Nd4j.vstack(list);
        INDArray exp = Nd4j.createFromArray(new double[][]{
                {1,2,3},
                {4,5,6},
                {7,8,9}});
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackHStack1d(Nd4jBackend backend) {
        INDArray rowVector1 = Nd4j.create(new double[]{1,2,3});
        INDArray rowVector2 = Nd4j.create(new double[]{4,5,6});

        INDArray vStack = Nd4j.vstack(rowVector1, rowVector2);      //Vertical stack:   [3]+[3] to [2,3]
        INDArray hStack = Nd4j.hstack(rowVector1, rowVector2);      //Horizontal stack: [3]+[3] to [6]

        assertEquals(Nd4j.createFromArray(1.0,2,3,4,5,6).reshape('c', 2, 3), vStack);
        assertEquals(Nd4j.createFromArray(1.0,2,3,4,5,6), hStack);
    }
}
