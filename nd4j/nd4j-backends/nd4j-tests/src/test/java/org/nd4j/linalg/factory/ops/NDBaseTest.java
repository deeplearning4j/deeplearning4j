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

package org.nd4j.linalg.factory.ops;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.jupiter.api.Assertions.*;

public class NDBaseTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering(){
        return 'c';
    }

    // TODO: Comment from the review. We'll remove the new NDBase() at some point.

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAll(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.BOOL, 3, 3);
        INDArray y = base.all(x, 1);
        INDArray y_exp = Nd4j.createFromArray(false, false, false);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAny(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.BOOL);
        INDArray y = base.any(x, 1);
        INDArray y_exp = Nd4j.createFromArray(true, true, true);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testArgmax(Nd4jBackend backend) {
        NDBase base = new NDBase();

        INDArray x = Nd4j.createFromArray(new double[][]{{0.75, 0.5, 0.25}, {0.5, 0.75, 0.25}, {0.5, 0.25, 0.75}});
        INDArray y = base.argmax(x, 0); //with default keepdims
        INDArray y_exp = Nd4j.createFromArray(0L, 1L, 2L);
        assertEquals(y_exp, y);

        y = base.argmax(x, false, 0); //with explicit keepdims false
        assertEquals(y_exp, y);

        y = base.argmax(x, true, 0); //with keepdims true
        y_exp = Nd4j.createFromArray(new long[][]{{0L, 1L, 2L}}); //expect different shape.
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testArgmin(Nd4jBackend backend) {
        //Copy Paste from argmax, replaced with argmin.
        NDBase base = new NDBase();

        INDArray x = Nd4j.createFromArray(new double[][]{{0.75, 0.5, 0.25}, {0.5, 0.75, 0.25}, {0.5, 0.25, 0.75}});
        INDArray y = base.argmin(x, 0); //with default keepdims
        INDArray y_exp = Nd4j.createFromArray(1L, 2L, 0L);
        assertEquals(y_exp, y);

        y = base.argmin(x, false, 0); //with explicit keepdims false
        assertEquals(y_exp, y);

        y = base.argmin(x, true, 0); //with keepdims true
        y_exp = Nd4j.createFromArray(new long[][]{{1L, 2L, 0L}}); //expect different shape.
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testConcat(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = Nd4j.ones(DataType.DOUBLE, 3, 3);

        INDArray z = base.concat(0, x, y);
        assertArrayEquals(new long[]{6, 3}, z.shape());

        z = base.concat(1, x, y);
        assertArrayEquals(new long[]{3, 6}, z.shape());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCumprod(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.cumprod(x, false, false, 0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 10.0, 18.0}, {28.0, 80.0, 162.0}});
        assertEquals(y_exp, y);

        y = base.cumprod(x, false, false, 1);
        y_exp = Nd4j.createFromArray(new double[][]{{1.0, 2.0, 6.0}, {4.0, 20.0, 120.0}, {7.0, 56.0, 504.0}});
        assertEquals(y_exp, y);

    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCumsum(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.cumsum(x, false, false, 0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.0, 2.0, 3.0}, {5.0, 7.0, 9.0}, {12.0, 15.0, 18.0}});
        assertEquals(y_exp, y);

        y = base.cumsum(x, false, false, 1);
        y_exp = Nd4j.createFromArray(new double[][]{{1.0, 3.0, 6.0}, {4.0, 9.0, 15.0}, {7.0, 15.0, 24.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDot(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 3);
        INDArray y = base.dot(x, x, 0);
        INDArray y_exp = Nd4j.scalar(14.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDynamicpartition(Nd4jBackend backend) {
        //Try to execute the sample in the code dcumentation:
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 5);
        int numPartitions = 2;
        int[] partitions = new int[]{1, 0, 0, 1, 0};
        //INDArray y = base.dynamicPartition(x, partitions, numPartitions); TODO: Fix
        //TODO: crashes here. Op needs fixing.

    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDynamicStitch(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        //INDArray y = base.dynamicStitch(new INDArray[]{x, x}, 0); TODO: Fix
        //TODO: crashes here. Op needs fixing.  Bad constructor, as previously flagged. Both input and indices need to be INDArrays
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarEq(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.eq(x, 0.0);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testEq(Nd4jBackend backend) {
        //element wise  eq.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.eq(x, x);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testExpandDims(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,2).reshape(1,2);
        INDArray y = base.expandDims(x, 0);
        INDArray y_exp = x.reshape(1, 1, 2);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testFill(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(2, 2);
        INDArray y = base.fill(x, DataType.DOUBLE, 1.1);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.1, 1.1}, {1.1, 1.1}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testGather(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        int[] ind = new int[]{0};
        INDArray y = base.gather(x, ind, 0);
        INDArray y_exp = Nd4j.createFromArray(0.0, 0.0, 0.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarGt(Nd4jBackend backend) {
        //Scalar gt.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.gt(x, -0.1);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testGt(Nd4jBackend backend) {
        //element wise  gt.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray x1 = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray y = base.gt(x1, x);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarGte(Nd4jBackend backend) {
        //Scalar gte.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.gte(x, -0.1);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testGte(Nd4jBackend backend) {
        //element wise  gte.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray x1 = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray y = base.gte(x1, x);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testIdentity(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.identity(x);
        assertEquals(x, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testInvertPermutation(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(2,0,1);
        INDArray y = base.invertPermutation(x);
        INDArray exp = Nd4j.createFromArray(1,2,0);
        assertEquals(exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testisNumericTensor(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.isNumericTensor(x);
        assertEquals(Nd4j.scalar(true), y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLinspace(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray y = base.linspace(DataType.DOUBLE, 0.0, 9.0, 19);
        //TODO: test crashes.
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarLt(Nd4jBackend backend) {
        //Scalar lt.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.lt(x, 0.1);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLt(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x1 = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray x = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray y = base.lt(x1, x);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarLte(Nd4jBackend backend) {
        //Scalar gt.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.lte(x, 0.1);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLte(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x1 = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray x = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray y = base.lte(x1, x);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMatchCondition(Nd4jBackend backend) {
        // same test as TestMatchTransformOp,
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1.0, 1.0, 1.0, 0.0, 1.0, 1.0);
        INDArray y = base.matchCondition(x, Conditions.epsEquals(0.0));
        INDArray y_exp = Nd4j.createFromArray(false, false, false, true, false, false);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMatchConditionCount(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1.0, 1.0, 1.0, 0.0, 1.0, 1.0);
        INDArray y = base.matchConditionCount(x, Conditions.epsEquals(0.0));
        assertEquals(Nd4j.scalar(1L), y);

        x = Nd4j.eye(3);
        y = base.matchConditionCount(x, Conditions.epsEquals(1.0), true, 1);
        INDArray y_exp = Nd4j.createFromArray(new Long[][]{{1L}, {1L}, {1L}});
        assertEquals(y_exp, y);

        y = base.matchConditionCount(x, Conditions.epsEquals(1.0), true, 0);
        y_exp = Nd4j.createFromArray(new Long[][]{{1L, 1L, 1L}});
        assertEquals(y_exp, y);

        y = base.matchConditionCount(x, Conditions.epsEquals(1.0), false, 1);
        y_exp = Nd4j.createFromArray(1L, 1L, 1L);
        assertEquals(y_exp, y);

        y = base.matchConditionCount(x, Conditions.epsEquals(1.0), false, 0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMax(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.max(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(1.0f, 1.0f, 1.0f);
        assertEquals(y_exp, y);

        y = base.max(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{1.0f, 1.0f, 1.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMean(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.mean(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(0.333333f, 0.333333f, 0.333333f);
        assertEquals(y_exp, y);

        y = base.mean(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{0.333333f, 0.333333f, 0.333333f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMin(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.min(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(0.0f, 0.0f, 0.0f);
        assertEquals(y_exp, y);

        y = base.min(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{0.0f, 0.0f, 0.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMmulTranspose(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 3);
        INDArray y = Nd4j.rand(DataType.FLOAT, 5, 4);
        INDArray exp = x.transpose().mmul(y.transpose());
        INDArray z = new NDBase().mmul(x, y, true, true, false);
        assertEquals(exp, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMmul(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray x1 = Nd4j.eye(3).castTo(DataType.DOUBLE);
        INDArray y = base.mmul(x, x1);
        assertEquals(y, x);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarNeq(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray y = base.neq(x, 1.0);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNeq(Nd4jBackend backend) {
        //element wise  eq.
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(DataType.DOUBLE, 3, 3);
        INDArray x1 = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray y = base.neq(x, x1);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, true, true}, {true, true, true}, {true, true, true}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNorm1(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.norm1(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(1.0f, 1.0f, 1.0f);
        assertEquals(y_exp, y);

        y = base.norm1(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{1.0f, 1.0f, 1.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNorm2(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.norm2(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(1.0f, 1.0f, 1.0f);
        assertEquals(y_exp, y);

        y = base.norm2(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{1.0f, 1.0f, 1.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNormMax(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.normmax(x, 0);
        INDArray  y_exp = Nd4j.createFromArray(1.0f, 1.0f, 1.0f);
        assertEquals(y_exp, y);

        y = base.normmax(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{1.0f, 1.0f, 1.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testOneHot(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(0.0, 1.0, 2.0);
        INDArray y = base.oneHot(x, 1, 0, 1.0, 0.0);
        INDArray y_exp = Nd4j.createFromArray(new float[][]{{1.0f, 0.0f, 0.0f}});
        assertEquals(y_exp, y);

        y = base.oneHot(x, 1);
        y_exp = Nd4j.createFromArray(new float[][]{{1.0f},{ 0.0f}, {0.0f}});
        assertEquals(y_exp, y);

        y = base.oneHot(x, 1, 0, 1.0, 0.0, DataType.DOUBLE);
        y_exp = Nd4j.createFromArray(new double[][]{{1.0, 0.0, 0.0}});
        assertEquals(y_exp, y); //TODO: Looks like we're getting back the wrong datatype.       https://github.com/eclipse/deeplearning4j/issues/8607
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testOnesLike(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3, 3);
        INDArray y = base.onesLike(x);
        INDArray  y_exp = Nd4j.createFromArray(1, 1);
        assertEquals(y_exp, y);

        y = base.onesLike(x, DataType.INT64);
        y_exp = Nd4j.createFromArray(1L, 1L);
        assertEquals(y_exp, y); //TODO: Getting back a double array, not a long.    https://github.com/eclipse/deeplearning4j/issues/8605
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testPermute(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray y = base.permute(x, 1,0);
        assertArrayEquals(new long[]{3, 2}, y.shape());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testProd(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3).castTo(DataType.FLOAT);
        INDArray y = base.prod(x, 0);
        INDArray y_exp = Nd4j.createFromArray(0.0f, 0.0f, 0.0f);
        assertEquals(y_exp, y);

        y = base.prod(x, true, 0);
        y_exp = Nd4j.createFromArray(new float[][]{{0.0f, 0.0f, 0.0f}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testRange(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray y = base.range(0.0, 3.0, 1.0, DataType.DOUBLE);
        INDArray y_exp = Nd4j.createFromArray(0.0, 1.0, 2.0);
        assertEquals(y_exp, y); //TODO: Asked for DOUBLE, got back a FLOAT Array.   https://github.com/eclipse/deeplearning4j/issues/8606
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testRank(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3);
        INDArray y = base.rank(x);
        INDArray y_exp = Nd4j.scalar(2);
        System.out.println(y);
        assertEquals(y_exp, y);
    }

    /*
      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testRepeat(Nd4jBackend backend) {
        fail("AB 2020/01/09 - Not sure what this op is supposed to do...");
        NDBase base = new NDBase();
        INDArray x = Nd4j.eye(3);
        INDArray y = base.repeat(x, 0);
        //TODO: fix, crashes the JVM.
    }
     */


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testReplaceWhere(Nd4jBackend backend) {
        // test from BooleanIndexingTest.
        NDBase base = new NDBase();
        INDArray array1 = Nd4j.createFromArray( 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        INDArray array2 = Nd4j.createFromArray( 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

        INDArray y = base.replaceWhere(array1, array2 , Conditions.greaterThan(4));
        INDArray y_exp = Nd4j.createFromArray( 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testReshape(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray shape = Nd4j.createFromArray(new long[] {3, 3});
        INDArray y = base.reshape(x, shape);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.0, 2.0, 3.0}, { 4.0, 5.0, 6.0}, { 7.0, 8.0, 9.0} } );
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testReverse(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 6).reshape(2, 3);
        INDArray y = base.reverse(x, 0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{ 4.0, 5.0, 6.0},{1.0, 2.0, 3.0} } );
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testReverseSequence(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        INDArray seq_kengths = Nd4j.createFromArray(2,3,1);
        INDArray y = base.reverseSequence(x, seq_kengths);

        INDArray y_exp = Nd4j.createFromArray(new double[][]{{ 2.0, 1.0, 3.0},{6.0, 5.0, 4.0},{7.0, 8.0, 9.0} } );
        assertEquals(y_exp, y);

        y = base.reverseSequence(x, seq_kengths, 0, 1);
        y_exp = Nd4j.createFromArray(new double[][]{{ 4.0, 8.0, 3.0},{1.0, 5.0, 6.0},{7.0, 2.0, 9.0} } );
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarFloorMod(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.scalarFloorMod(x, 2.0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{ 1.0, 0.0, 1.0},{0.0, 1.0, 0.0}, { 1.0, 0.0, 1.0} } );
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarMax(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.scalarMax(x, 5.0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{ 5.0, 5.0, 5.0},{5.0, 5.0, 6.0}, { 7.0, 8.0, 9.0} } );
        assertEquals(y_exp, y);
        //System.out.println(y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarMin(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.scalarMin(x, 5.0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{ 1.0, 2.0, 3.0},{4.0, 5.0, 5.0}, { 5.0, 5.0, 5.0} } );
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScalarSet(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1.0, 2.0, 0.0, 4.0, 5.0);
        INDArray y = base.scalarSet(x, 1.0);
        INDArray y_exp = Nd4j.ones(DataType.DOUBLE, 5);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterAdd(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10);
        INDArray y = base.scatterAdd(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterDiv(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10).add(1.0);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10).add(1.0);
        INDArray y = base.scatterDiv(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterMax(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10).add(1.0);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10).add(1.0);
        INDArray y = base.scatterMax(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterMin(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10).add(1.0);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10).add(1.0);
        INDArray y = base.scatterMin(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterMul(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10).add(1.0);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10).add(1.0);
        INDArray y = base.scatterMul(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testScatterSub(Nd4jBackend backend) {
        NDBase base = new NDBase();

        //from testScatterOpGradients.
        INDArray x = Nd4j.ones(DataType.DOUBLE, 20, 10).add(1.0);
        INDArray indices = Nd4j.createFromArray(3, 4, 5, 10, 18);
        INDArray updates = Nd4j.ones(DataType.DOUBLE, 5, 10).add(1.0);
        INDArray y = base.scatterSub(x,indices, updates);

        y = y.getColumn(0);
        INDArray  y_exp = Nd4j.createFromArray(2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0);
        assertEquals(y_exp, y);
    }



    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSegmentMax(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3, 6, 1, 4, 9,2, 2);
        INDArray segmentIDs = Nd4j.createFromArray(0,0,1,1,1,2,2);
        INDArray y = base.segmentMax(x, segmentIDs);
        INDArray y_exp = Nd4j.createFromArray(6, 9, 2);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSegmentMean(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3.0, 6.0, 1.0, 4.0, 9.0,2.0, 2.0);
        INDArray segmentIDs = Nd4j.createFromArray(0,0,1,1,1,2,2);
        INDArray y = base.segmentMean(x, segmentIDs);
        INDArray y_exp = Nd4j.createFromArray(4.5, 4.6667, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSegmentMin(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3.0, 6.0, 1.0, 4.0, 9.0,2.0, 2.0);
        INDArray segmentIDs = Nd4j.createFromArray(0,0,1,1,1,2,2);
        INDArray y = base.segmentMin(x, segmentIDs);
        INDArray y_exp = Nd4j.createFromArray(3.0, 1.0, 2.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSegmentProd(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3.0, 6.0, 1.0, 4.0, 9.0,2.0, 2.0);
        INDArray segmentIDs = Nd4j.createFromArray(0,0,1,1,1,2,2);
        INDArray y = base.segmentProd(x, segmentIDs);
        INDArray y_exp = Nd4j.createFromArray(18.0, 36.0, 4.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSegmentSum(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(3.0, 6.0, 1.0, 4.0, 9.0,2.0, 2.0);
        INDArray segmentIDs = Nd4j.createFromArray(0,0,1,1,1,2,2);
        INDArray y = base.segmentSum(x, segmentIDs);
        INDArray y_exp = Nd4j.createFromArray(9.0, 14.0, 4.0);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSequenceMask(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray length = Nd4j.createFromArray(1, 3, 2);
        int maxlength = 5;
        DataType dt = DataType.BOOL;
        INDArray y = base.sequenceMask(length, maxlength, dt);
        INDArray y_exp = Nd4j.createFromArray(new boolean[][]{{true, false, false, false, false}, {true, true, true, false, false}, {true, true, false, false, false}});
        assertEquals(y_exp, y);

        y = base.sequenceMask(length, maxlength, DataType.FLOAT);
        y_exp = y_exp.castTo(DataType.FLOAT);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testShape(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(3,3);
        INDArray y = base.shape(x);
        INDArray y_exp = Nd4j.createFromArray(3L, 3L);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSize(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(3,3);
        INDArray y = base.size(x);
        assertEquals(Nd4j.scalar(9L), y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSizeAt(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(10,20, 30);
        INDArray y = base.sizeAt(x, 1);
        assertEquals(Nd4j.scalar(20L), y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSlice(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 6).reshape(2, 3);
        INDArray y = base.slice(x, new int[]{0,1}, 2,1);
        INDArray y_exp = Nd4j.create(new double[][]{{2.0}, {5.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSquaredNorm(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3, 3);
        INDArray y = base.squaredNorm(x, 0);
        INDArray y_exp = Nd4j.createFromArray(66.0, 93.0, 126.0);
        assertEquals(y_exp, y);

        y = base.squaredNorm(x, true, 0);
        y_exp = Nd4j.createFromArray(new double[][]{{66.0, 93.0, 126.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSqueeze(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 10).reshape(2,1,5);
        INDArray y = base.squeeze(x,1);
        INDArray exp = x.reshape(2, 5);
        assertEquals(exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testStack(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 3);
        INDArray y = base.stack(1 , x);
        // TODO: Op definition looks wrong. Compare stack in Nd4j.
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testStandardDeviation(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 4);
        INDArray y = base.standardDeviation(x, false, 0);
        assertEquals(Nd4j.scalar(1.118034), y);

        x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        y = base.standardDeviation(x, false, true, 0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{2.4494898, 2.4494898, 2.4494898}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testStridedSlice(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        INDArray y = base.stridedSlice(x, new long[]{0,1}, new long[] {3,3}, 2,1);

        INDArray y_exp = Nd4j.createFromArray(new double[][]{{2.0, 3.0}, {8.0, 9.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSum(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        INDArray y = base.sum(x, 0);
        INDArray y_exp = Nd4j.createFromArray(12.0, 15.0, 18.0);
        assertEquals(y_exp, y);

        y = base.sum(x, true, 0);
        assertEquals(y_exp.reshape(1,3), y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testTensorMul(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        INDArray y = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        int[] dimX = new int[] {1};
        int[] dimY = new int[] {0};
        boolean transposeX = false;
        boolean transposeY = false;
        boolean transposeResult = false;

        INDArray res = base.tensorMmul(x, y, dimX, dimY, transposeX, transposeY, transposeResult);
        // org.nd4j.linalg.exception.ND4JIllegalStateException: Op name tensordot - no output arrays were provided and calculateOutputShape failed to execute

        INDArray exp = x.mmul(y);
        assertEquals(exp, res);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testTile(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 4).reshape(2,2);
        INDArray repeat = Nd4j.createFromArray(2, 3);
        INDArray y = base.tile(x, repeat); // the sample from the code docs.

        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.0, 2.0, 1.0, 2.0, 1.0, 2.0}, {3.0, 4.0, 3.0, 4.0, 3.0, 4.0}, {1.0, 2.0, 1.0, 2.0, 1.0, 2.0}, {3.0, 4.0, 3.0, 4.0, 3.0, 4.0}});
        assertEquals(y_exp, y);

        y = base.tile(x, 2, 3); // the sample from the code docs.
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testTranspose(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        INDArray y = base.transpose(x);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{1.0, 4.0, 7.0}, {2.0, 5.0, 8.0}, {3.0, 6.0, 9.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsegmentMax(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,3,2,6,4,9,8);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentMax(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray(6,9,8);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsegmentMean(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,3,2,6,4,9,8).castTo(DataType.FLOAT);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentMean(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray(4.5f,4.6667f, 5.0f);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsegmentedMin(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,3,2,6,4,9,8);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentMin(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray(3,1,2);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsegmentProd(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,3,2,6,4,9,8);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentProd(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray(18,36,16);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsortedSegmentSqrtN(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1.0,3.0,2.0,6.0,4.0,9.0,8.0);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentSqrtN(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray( 6.3640,    8.0829,    7.0711);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUnsortedSegmentSum(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.createFromArray(1,3,2,6,4,9,8);
        INDArray segmentIDs = Nd4j.createFromArray(1,0,2,0,1,1,2);
        INDArray y = base.unsortedSegmentSum(x, segmentIDs, 3);
        INDArray y_exp = Nd4j.createFromArray(9,14,10);
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testVariance(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 4);
        INDArray y = base.variance(x, false, 0);
        assertEquals(Nd4j.scalar(1.250), y);

        x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
        y = base.variance(x, false, true, 0);
        INDArray y_exp = Nd4j.createFromArray(new double[][]{{6.0, 6.0, 6.0}});
        assertEquals(y_exp, y);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testZerosLike(Nd4jBackend backend) {
        NDBase base = new NDBase();
        INDArray x = Nd4j.zeros(3,3);
        INDArray y = base.zerosLike(x);
        assertEquals(x, y);
        assertNotSame(x, y);
    }
}
