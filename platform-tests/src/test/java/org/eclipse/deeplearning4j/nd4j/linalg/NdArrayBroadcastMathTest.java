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
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMin;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMin;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastEqualTo;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThan;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastLessThan;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for NDArray broadcasting and element-wise math operations extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayBroadcastMathTest extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoBroadcastShape(Nd4jBackend backend) {
        val assertion = new long[]{2,2,2,5};
        val shapeTest = Shape.broadcastOutputShape(new long[]{2,1,2,1},new long[]{2,1,5});
        assertArrayEquals(assertion,shapeTest);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")

    public void testAutoBroadcastAdd(Nd4jBackend backend) {
        INDArray left = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,1,2,1);
        INDArray right = Nd4j.linspace(1,10,10, DataType.DOUBLE).reshape(2,1,5);
        INDArray assertion = Nd4j.create(new double[]{2,3,4,5,6,3,4,5,6,7,7,8,9,10,11,8,9,10,11,12,4,5,6,7,8,5,6,7,8,9,9,10,11,12,13,10,11,12,13,14}).reshape(2,2,2,5);
        INDArray test = left.add(right);
        assertEquals(assertion,test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAudoBroadcastAddMatrix(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,2);
        INDArray row = Nd4j.ones(1, 2);
        INDArray assertion = arr.add(1.0);
        INDArray test = arr.add(row);
        assertEquals(assertion,test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarOps(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.addi(Nd4j.scalar(1d));
        n.subi(Nd4j.scalar(1.0d));
        n.muli(Nd4j.scalar(1.0d));
        n.divi(Nd4j.scalar(1.0d));

        n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27, n.sumNumber().doubleValue(), 1e-1,getFailureMessage(backend));
        INDArray a = n.slice(2);
        assertEquals( true, Arrays.equals(new long[] {3, 3}, a.shape()),getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSubiRowVector(Nd4jBackend backend) {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 2, 2);
        INDArray row1 = oneThroughFour.getRow(1).dup();
        oneThroughFour.subiRowVector(row1);
        INDArray result = Nd4j.create(new double[] {-2, -2, 0, 0}, new long[] {2, 2});
        assertEquals(result, oneThroughFour,getFailureMessage(backend));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddiRowVectorWithScalar(Nd4jBackend backend) {
        INDArray colVector = Nd4j.create(5, 1).assign(0.0);
        INDArray scalar = Nd4j.create(1, 1).assign(0.0);
        scalar.putScalar(0, 1);

        assertEquals(scalar.getDouble(0), 1.0, 0.0);

        colVector.addiRowVector(scalar); //colVector is all zeros after this
        for (int i = 0; i < 5; i++)
            assertEquals(colVector.getDouble(i), 1.0, 0.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCasting(Nd4jBackend backend) {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1).castTo(DataType.DOUBLE);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][] {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4).castTo(DataType.DOUBLE);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][] {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}});
        assertEquals(testR2, r2);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastRepeated(Nd4jBackend backend) {
        INDArray z = Nd4j.create(1, 4, 4, 3);
        INDArray bias = Nd4j.create(1, 3);
        BroadcastOp op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op);
//        System.out.println("First: OK");
        //OK at this point: executes successfully


        z = Nd4j.create(1, 4, 4, 3);
        bias = Nd4j.create(1, 3);
        op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op); //Crashing here, when we are doing exactly the same thing as before...
//        System.out.println("Second: OK");
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddVectorWithOffset(Nd4jBackend backend) {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new double[] {1, 2, 4, 5}, new long[] {2, 2});
        assertEquals(result, oneThroughFour,getFailureMessage(backend));


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast1d(Nd4jBackend backend) {
        int[] shape = {4, 3, 2};
        int[] toBroadcastDims = new int[] {0, 1, 2};
        int[][] toBroadcastShapes = new int[][] {{1, 4}, {1, 3}, {1, 2}};

        //Expected result values in buffer: c order, need to reshape to {4,3,2}. Values taken from 0.4-rc3.8
        double[][] expFlat = new double[][] {
                {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0,
                        4.0, 4.0, 4.0, 4.0, 4.0},
                {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
                        1.0, 2.0, 2.0, 3.0, 3.0},
                {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                        2.0, 1.0, 2.0, 1.0, 2.0}};

        double[][] expLinspaced = new double[][] {
                {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0},
                {2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 14.0, 15.0, 17.0, 18.0, 20.0,
                        21.0, 20.0, 21.0, 23.0, 24.0, 26.0, 27.0},
                {2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 14.0, 16.0, 16.0, 18.0, 18.0,
                        20.0, 20.0, 22.0, 22.0, 24.0, 24.0, 26.0}};

        for (int i = 0; i < toBroadcastDims.length; i++) {
            int dim = toBroadcastDims[i];
            int[] vectorShape = toBroadcastShapes[i];
            int length = ArrayUtil.prod(vectorShape);

            INDArray zC = Nd4j.create(shape, 'c');
            zC.setData(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data());
            for (int tad = 0; tad < zC.tensorsAlongDimension(dim); tad++) {
                INDArray javaTad = zC.tensorAlongDimension(tad, dim);

            }

            INDArray zF = Nd4j.create(shape, 'f');
            zF.assign(zC);
            INDArray toBroadcast = Nd4j.linspace(1, length, length, DataType.DOUBLE);

            Op opc = new BroadcastAddOp(zC, toBroadcast, zC, dim);
            Op opf = new BroadcastAddOp(zF, toBroadcast, zF, dim);
            INDArray exp = Nd4j.create(expLinspaced[i], shape, 'c');
            INDArray expF = Nd4j.create(shape, 'f');
            expF.assign(exp);

            Nd4j.getExecutioner().exec(opc);
            Nd4j.getExecutioner().exec(opf);

            assertEquals(exp, zC);
            assertEquals(exp, zF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSubRowVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray row = Nd4j.linspace(1, 3, 3, DataType.DOUBLE);
        INDArray test = matrix.subRowVector(row);
        INDArray assertion = Nd4j.create(new double[][] {{0, 0, 0}, {3, 3, 3}});
        assertEquals(assertion, test);

        INDArray threeByThree = Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3);
        INDArray offsetTest = threeByThree.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        assertEquals(2, offsetTest.rows());
        INDArray offsetAssertion = Nd4j.create(new double[][] {{3, 3, 3}, {6, 6, 6}});
        INDArray offsetSub = offsetTest.subRowVector(row);
        assertEquals(offsetAssertion, offsetSub);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDivide(Nd4jBackend backend) {
        INDArray two = Nd4j.create(new double[] {2, 2, 2, 2}).castTo(DataType.DOUBLE);
        INDArray div = two.div(two);
        assertEquals(Nd4j.ones(4), div);

        INDArray half = Nd4j.create(new double[] {0.5f, 0.5f, 0.5f, 0.5f}, new long[] {2, 2});
        INDArray divi = Nd4j.create(new double[] {0.3f, 0.6f, 0.9f, 0.1f}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new double[] {1.6666666f, 0.8333333f, 0.5555556f, 5}, new long[] {2, 2});
        INDArray result = half.div(divi);
        assertEquals(assertion, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMulRowVector(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        arr.muliRowVector(Nd4j.linspace(1, 2, 2, DataType.DOUBLE));
        INDArray assertion = Nd4j.create(new double[][] {{1, 4}, {3, 8}});

        assertEquals(assertion, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFTimesCAddiRow(Nd4jBackend backend) {

        INDArray arrF = Nd4j.create(2, 3, 'f').assign(1.0).castTo(DataType.DOUBLE);
        INDArray arrC = Nd4j.create(2, 3, 'c').assign(1.0).castTo(DataType.DOUBLE);
        INDArray arr2 = Nd4j.create(new long[] {3, 4}, 'c').assign(1.0).castTo(DataType.DOUBLE);

        INDArray mmulC = arrC.mmul(arr2); //[2,4] with elements 3.0
        INDArray mmulF = arrF.mmul(arr2); //[2,4] with elements 3.0
        assertArrayEquals(mmulC.shape(), new long[] {2, 4});
        assertArrayEquals(mmulF.shape(), new long[] {2, 4});
        assertTrue(arrC.equals(arrF));

        INDArray row = Nd4j.zeros(1, 4).assign(0.0).addi(0.5).castTo(DataType.DOUBLE);
        mmulC.addiRowVector(row); //OK
        mmulF.addiRowVector(row); //Exception

        assertTrue(mmulC.equals(mmulF));

        for (int i = 0; i < mmulC.length(); i++)
            assertEquals(mmulC.getDouble(i), 3.5, 1e-1); //OK
        for (int i = 0; i < mmulF.length(); i++)
            assertEquals(mmulF.getDouble(i), 3.5, 1e-1); //Exception
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMuliRowVector(Nd4jBackend backend) {
        INDArray arrC = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape('c', 3, 2);
        INDArray arrF = Nd4j.create(new long[] {3, 2}, 'f').assign(arrC);

        INDArray temp = Nd4j.create(new long[] {2, 11}, 'c');
        INDArray vec = temp.get(NDArrayIndex.all(), NDArrayIndex.interval(9, 10)).transpose();
        vec.assign(Nd4j.linspace(1, 2, 2, DataType.DOUBLE));

        //Passes if we do one of these...
        //        vec = vec.dup('c');
        //        vec = vec.dup('f');

//        System.out.println("Vec: " + vec);

        INDArray outC = arrC.muliRowVector(vec);
        INDArray outF = arrF.muliRowVector(vec);

        double[][] expD = new double[][] {{1, 4}, {3, 8}, {5, 12}};
        INDArray exp = Nd4j.create(expD);

        assertEquals(exp, outC);
        assertEquals(exp, outF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDiv(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00,
                2.00, 2.00, 2.00, 2.00, -1.00, -1.00, -1.00, -1.00, -2.00, -2.00, -2.00, -2.00, -1.00, -1.00,
                -1.00, -1.00, -2.00, -2.00, -2.00, -2.00}).reshape(2, 16);

        INDArray denom = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00,
                1.00, 2.00, 2.00, 2.00, 2.00});

        INDArray expected = Nd4j.create(
                new double[] {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1.,
                        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,},
                new long[] {2, 16});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastDivOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDiv2(){
        INDArray arr = Nd4j.ones(DataType.DOUBLE, 1, 64, 125, 125).muli(2);
        INDArray vec = Nd4j.ones(DataType.DOUBLE, 64).muli(2);

        INDArray exp = Nd4j.ones(DataType.DOUBLE, 1, 64, 125, 125);
        INDArray out = arr.like();

        for( int i=0; i<10; i++ ) {
            out.assign(0.0);
            Nd4j.getExecutioner().exec(new BroadcastDivOp(arr, vec, out, 1));
            assertEquals(exp, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMult(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {1, 4, 9, 16, 25, 36, 49, 64, -1, -4, -9, -16, -25, -36, -49, -64},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastMulOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastSub(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {0, 0, 0, 0, 0, 0, 0, 0, -2, -4, -6, -8, -10, -12, -14, -16},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastSubOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAdd(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0,},
                new long[] {2, 8});
        INDArray dup = num.dup();
        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastAddOp(num, denom, dup, -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRSubi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.ones(2);
        INDArray n2Assertion = Nd4j.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion, nRsubi);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4);
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 5);
        assertEquals(answer, rdiv);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRdivScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4).castTo(DataType.DOUBLE);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 0.25).castTo(DataType.DOUBLE);
        assertEquals(rdiv, answer);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDivi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.valueArrayOf(new long[] {1, 2}, 4).castTo(DataType.DOUBLE);
        INDArray n2Assertion = Nd4j.valueArrayOf(new long[] {1, 2}, 0.5).castTo(DataType.DOUBLE);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseAdd(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray linspace2 = linspace.dup();
        INDArray assertion = Nd4j.create(new double[][] {{2, 4}, {6, 8}});
        linspace.addi(linspace2);
        assertEquals(assertion, linspace);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCast(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray broadCasted = n.broadcast(5, 4);
        for (int i = 0; i < broadCasted.rows(); i++) {
            INDArray row = broadCasted.getRow(i);
            assertEquals(n, broadCasted.getRow(i));
        }

        INDArray broadCast2 = broadCasted.getRow(0).broadcast(5, 4);
        assertEquals(broadCasted, broadCast2);


        INDArray columnBroadcast = n.reshape(4,1).broadcast(4, 5);
        for (int i = 0; i < columnBroadcast.columns(); i++) {
            INDArray column = columnBroadcast.getColumn(i);
            assertEquals(column, n);
        }

        INDArray fourD = Nd4j.create(1, 2, 1, 1);
        INDArray broadCasted3 = fourD.broadcast(1, 2, 36, 36);
        assertTrue(Arrays.equals(new long[] {1, 2, 36, 36}, broadCasted3.shape()));



        INDArray ones = Nd4j.ones(1, 1, 1).broadcast(2, 1, 1);
        assertArrayEquals(new long[] {2, 1, 1}, ones.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarBroadcast(Nd4jBackend backend) {
        INDArray fiveThree = Nd4j.ones(5, 3);
        INDArray fiveThreeTest = Nd4j.scalar(1.0).broadcast(5, 3);
        assertEquals(fiveThree, fiveThreeTest);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseOps(Nd4jBackend backend) {
        INDArray n1 = Nd4j.scalar(1.0);
        INDArray n2 = Nd4j.scalar(2.0);
        INDArray nClone = n1.add(n2);
        assertEquals(Nd4j.scalar(3.0), nClone);
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = Nd4j.scalar(3.0);
        INDArray n4 = Nd4j.scalar(4.0);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.scalar(1.0), subbed);
        assertEquals(Nd4j.scalar(12.0), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeakyRelu(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(-1, 1, 10, DataType.DOUBLE);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().exec(new LeakyReLU(arr, 0.01));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeakyRelu2(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(-1, 1, 10, DataType.DOUBLE);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().exec(new LeakyReLU(arr, 0.01));

//        System.out.println("Expected: " + Arrays.toString(expected));
//        System.out.println("Actual:   " + Arrays.toString(out.data().asDouble()));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseMixedC(Nd4jBackend backend) {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
//        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

//        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseMixedF(Nd4jBackend backend) {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).dup('f');
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
//        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).dup('f').mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

//        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast3d2d(Nd4jBackend backend) {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
//                System.out.println(orderArr + "\t" + orderbc);
                INDArray arrOrig = Nd4j.ones(3, 4, 5).dup(orderArr);

                //Broadcast on dimensions 0,1
                INDArray bc01 = Nd4j.create(new double[][] {{1, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 0}}).dup(orderbc);

                INDArray result01 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc01, result01, 0, 1));

                for (int i = 0; i < 5; i++) {
                    INDArray subset = result01.tensorAlongDimension(i, 0, 1);//result01.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));
                    assertEquals(bc01, subset);
                }

                //Broadcast on dimensions 0,2
                INDArray bc02 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result02 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc02, result02, 0, 2));

                for (int i = 0; i < 4; i++) {
                    INDArray subset = result02.tensorAlongDimension(i, 0, 2); //result02.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all());
                    assertEquals(bc02, subset);
                }

                //Broadcast on dimensions 1,2
                INDArray bc12 = Nd4j.create(
                                new double[][] {{1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result12 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc12, result12, 1, 2));

                for (int i = 0; i < 3; i++) {
                    INDArray subset = result12.tensorAlongDimension(i, 1, 2);//result12.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
                    assertEquals( bc12, subset,"Failed for subset [" + i + "] orders [" + orderArr + "/" + orderbc + "]");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast4d2d(Nd4jBackend backend) {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
//                System.out.println(orderArr + "\t" + orderbc);
                INDArray arrOrig = Nd4j.ones(3, 4, 5, 6).dup(orderArr);

                //Broadcast on dimensions 0,1
                INDArray bc01 = Nd4j.create(new double[][] {{1, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 0}}).dup(orderbc);

                INDArray result01 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result01, bc01, result01, 0, 1));

                for (int d2 = 0; d2 < 5; d2++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result01.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(d2),
                                NDArrayIndex.point(d3));
                        assertEquals(bc01, subset);
                    }
                }

                //Broadcast on dimensions 0,2
                INDArray bc02 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result02 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result02, bc02, result02, 0, 2));

                for (int d1 = 0; d1 < 4; d1++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result02.get(NDArrayIndex.all(), NDArrayIndex.point(d1), NDArrayIndex.all(),
                                NDArrayIndex.point(d3));
                        assertEquals(bc02, subset);
                    }
                }

                //Broadcast on dimensions 0,3
                INDArray bc03 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}})
                        .dup(orderbc);

                INDArray result03 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result03, bc03, result03, 0, 3));

                for (int d1 = 0; d1 < 4; d1++) {
                    for (int d2 = 0; d2 < 5; d2++) {
                        INDArray subset = result03.get(NDArrayIndex.all(), NDArrayIndex.point(d1),
                                NDArrayIndex.point(d2), NDArrayIndex.all());
                        assertEquals(bc03, subset);
                    }
                }

                //Broadcast on dimensions 1,2
                INDArray bc12 = Nd4j.create(
                                new double[][] {{1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result12 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result12, bc12, result12, 1, 2));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result12.get(NDArrayIndex.point(d0), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(d3));
                        assertEquals(bc12, subset);
                    }
                }

                //Broadcast on dimensions 1,3
                INDArray bc13 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1},
                        {1, 1, 1, 0, 0, 1}}).dup(orderbc);

                INDArray result13 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result13, bc13, result13, 1, 3));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d2 = 0; d2 < 5; d2++) {
                        INDArray subset = result13.get(NDArrayIndex.point(d0), NDArrayIndex.all(),
                                NDArrayIndex.point(d2), NDArrayIndex.all());
                        assertEquals(bc13, subset);
                    }
                }

                //Broadcast on dimensions 2,3
                INDArray bc23 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0},
                        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0}}).dup(orderbc);

                INDArray result23 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result23, bc23, result23, 2, 3));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d1 = 0; d1 < 4; d1++) {
                        INDArray subset = result23.get(NDArrayIndex.point(d0), NDArrayIndex.point(d1),
                                NDArrayIndex.all(), NDArrayIndex.all());
                        assertEquals(bc23, subset);
                    }
                }

            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison1(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {true, true, true, false, false});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();
//        log.info("original: \n{}", initial);

        Nd4j.getExecutioner().exec(new BroadcastLessThan(initial, mask, result, 1));

        Nd4j.getExecutioner().commit();
//        log.info("Comparison ----------------------------------------------");
        for (int i = 0; i < initial.rows(); i++) {
            val row = result.getRow(i);
            assertEquals(exp, row,"Failed at row " + i);
//            log.info("-------------------");
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison2(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, false, true, true});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThan(initial, mask, result, 1));



        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, result.getRow(i),"Failed at row " + i);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison3(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, true, true, true});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i + 1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThanOrEqual(initial, mask, result, 1));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, result.getRow(i),"Failed at row " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison4(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, true, false, false});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i + 1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastEqualTo(initial, mask, result, 1 ));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals( exp, result.getRow(i),"Failed at row " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRsub1(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(5).assign(2.0);
        INDArray exp_0 = Nd4j.ones(5).assign(2.0);
        INDArray exp_1 = Nd4j.create(5).assign(-1);

        Nd4j.getExecutioner().commit();

        INDArray res = arr.rsub(1.0);

        assertEquals(exp_0, arr);
        assertEquals(exp_1, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMin(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{2, 3, 3, 4, 5}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new BroadcastMin(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 5}), matrix.getRow(r));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMax(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3, 2, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new BroadcastMax(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 5}), matrix.getRow(r));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAMax(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3, 2, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, -4, -5});

        Nd4j.getExecutioner().exec(new BroadcastAMax(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, -4, -5}), matrix.getRow(r));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAMin(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{2, 3, 3, 4, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, -5});

        Nd4j.getExecutioner().exec(new BroadcastAMin(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 1}), matrix.getRow(r));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPow1(Nd4jBackend backend) {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {2.0, 4.0, 8.0});
        val res = Transforms.pow(argX, argY);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDiv1(Nd4jBackend backend) {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {0.5, 1.0, 1.5});
        val res = argX.rdiv(argY);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDiv(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{2,2,2});
        val y = Nd4j.create(new double[]{4,6,8});
        val result = Nd4j.createUninitialized(DataType.DOUBLE, 3);

        assertEquals(DataType.DOUBLE, x.dataType());
        assertEquals(DataType.DOUBLE, y.dataType());
        assertEquals(DataType.DOUBLE, result.dataType());

        val op = DynamicCustomOp.builder("RDiv")
                .addInputs(x,y)
                .addOutputs(result)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(Nd4j.create(new double[]{2, 3, 4}), result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRdiv()    {
        final INDArray a = Nd4j.create(new double[]{2.0, 2.0, 2.0, 2.0});
        final INDArray b = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0});
        final INDArray c = Nd4j.create(new double[]{2.0, 2.0}).reshape(2, 1);
        final INDArray d = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0}).reshape(2, 2);

        final INDArray expected = Nd4j.create(new double[]{2.0, 1.0, 0.5, 0.25});
        final INDArray expected2 = Nd4j.create(new double[]{2.0, 1.0, 0.5, 0.25}).reshape(2, 2);

        assertEquals(expected, a.div(b));
        assertEquals(expected, b.rdiv(a));
        assertEquals(expected, b.rdiv(2));
        assertEquals(expected2, d.rdivColumnVector(c));

        assertEquals(expected, b.rdiv(Nd4j.scalar(2.0)));
        assertEquals(expected, b.rdivColumnVector(Nd4j.scalar(2)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRsub()    {
        final INDArray a = Nd4j.create(new double[]{2.0, 2.0, 2.0, 2.0});
        final INDArray b = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0});
        final INDArray c = Nd4j.create(new double[]{2.0, 2.0}).reshape(2, 1);
        final INDArray d = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0}).reshape('c',2, 2);

        final INDArray expected = Nd4j.create(new double[]{1.0, 0.0, -2.0, -6.0});
        final INDArray expected2 = Nd4j.create(new double[]{1, 0, -2.0, -6.0}).reshape('c',2, 2);

        assertEquals(expected, a.sub(b));
        assertEquals(expected, b.rsub(a));
        assertEquals(expected, b.rsub(2));
        assertEquals(expected2, d.rsubColumnVector(c));

        assertEquals(expected, b.rsub(Nd4j.scalar(2)));
        assertEquals(expected, b.rsubColumnVector(Nd4j.scalar(2)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast_1(Nd4jBackend backend) {
        val array1 = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).reshape(5, 1, 2).broadcast(5, 4, 2);
        val array2 = Nd4j.linspace(1, 20, 20, DataType.DOUBLE).reshape(5, 4, 1).broadcast(5, 4, 2);
        val exp = Nd4j.create(new double[] {2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 17.0f, 17.0f, 18.0f, 20.0f, 21.0f, 21.0f, 22.0f, 22.0f, 23.0f, 23.0f, 24.0f, 26.0f, 27.0f, 27.0f, 28.0f, 28.0f, 29.0f, 29.0f, 30.0f}).reshape(5,4,2);

        array1.addi(array2);

        assertEquals(exp, array1);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddiColumnEdge(){
        INDArray arr1 = Nd4j.create(1, 5);
        arr1.addiColumnVector(Nd4j.ones(1));
        assertEquals(Nd4j.ones(1,5), arr1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseScalar_1(Nd4jBackend backend) {
        val exp_1 = Nd4j.create(new double[]{2.0, 3.0, 4.0}, new long[]{3});
        val exp_2 = Nd4j.create(new double[]{0.0, 1.0, 2.0}, new long[]{3});
        val exp_3 = Nd4j.create(new double[]{1.0, 2.0, 3.0}, new long[]{3});
        val arrayX = Nd4j.create(new double[]{1.0, 2.0, 3.0}, new long[]{3});
        val arrayY = Nd4j.scalar(1.0);

        val arrayZ_1 = arrayX.add(arrayY);
        assertEquals(exp_1, arrayZ_1);

        val arrayZ_2 = arrayX.sub(arrayY);
        assertEquals(exp_2, arrayZ_2);

        val arrayZ_3 = arrayX.div(arrayY);
        assertEquals(exp_3, arrayZ_3);

        val arrayZ_4 = arrayX.mul(arrayY);
        assertEquals(exp_3, arrayZ_4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLTOE_1(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val y = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ex = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val ey = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ez = Nd4j.create(new boolean[]{true, true, true, false});
        val z = Transforms.lessThanOrEqual(x, y, true);

        assertEquals(ex, x);
        assertEquals(ey, y);

        assertEquals(ez, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGTOE_1(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val y = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ex = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val ey = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ez = Nd4j.create(new boolean[]{false, true, true, true}, new long[]{4}, DataType.BOOL);
        val z = Transforms.greaterThanOrEqual(x, y, true);

        val str = ez.toString();
//        log.info("exp: {}", str);

        assertEquals(ex, x);
        assertEquals(ey, y);

        assertEquals(ez, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastInvalid() {
        assertThrows(IllegalStateException.class,() -> {
            INDArray arr1 = Nd4j.ones(3,4,1);

            //Invalid op: y must match x/z dimensions 0 and 2
            INDArray arrInvalid = Nd4j.create(3,12);
            Nd4j.getExecutioner().exec(new BroadcastMulOp(arr1, arrInvalid, arr1, 0, 2));
            fail("Excepted exception on invalid input");
        });

    }

    protected static boolean arrayNotEquals(float[] arrayX, float[] arrayY, float delta) {
        if (arrayX.length != arrayY.length)
            return false;

        // on 2d arrays first & last elements will match regardless of order
        for (int i = 1; i < arrayX.length - 1; i++) {
            if (Math.abs(arrayX[i] - arrayY[i]) < delta) {
                log.info("ArrX[{}]: {}; ArrY[{}]: {}", i, arrayX[i], i, arrayY[i]);
                return false;
            }
        }

        return true;
    }
}
