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

package org.nd4j.linalg.shape;

import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Adam Gibson
 */
@NativeTag
@Tag(TagNames.NDARRAY_INDEXING)
public class ShapeTestsC extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();

    @AfterEach
    public void after() {
        Nd4j.setDataType(this.initialType);
    }




    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenZeroOne(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        assertEquals(4, baseArr.tensorsAlongDimension(0, 1));
        INDArray columnVectorFirst = Nd4j.create(new double[][] {{1, 5}, {9, 13}});
        INDArray columnVectorSecond = Nd4j.create(new double[][] {{2, 6}, {10, 14}});
        INDArray columnVectorThird = Nd4j.create(new double[][] {{3, 7}, {11, 15}});
        INDArray columnVectorFourth = Nd4j.create(new double[][] {{4, 8}, {12, 16}});
        INDArray[] assertions =
                new INDArray[] {columnVectorFirst, columnVectorSecond, columnVectorThird, columnVectorFourth};
        for (int i = 0; i < baseArr.tensorsAlongDimension(0, 1); i++) {
            INDArray test = baseArr.tensorAlongDimension(i, 0, 1);
            assertEquals( assertions[i], test,"Wrong at index " + i);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenSecondDim(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        INDArray[] assertions = new INDArray[] {Nd4j.create(new double[] {1, 3}), Nd4j.create(new double[] {2, 4}),
                Nd4j.create(new double[] {5, 7}), Nd4j.create(new double[] {6, 8}),
                Nd4j.create(new double[] {9, 11}), Nd4j.create(new double[] {10, 12}),
                Nd4j.create(new double[] {13, 15}), Nd4j.create(new double[] {14, 16}),


        };

        for (int i = 0; i < baseArr.tensorsAlongDimension(2); i++) {
            INDArray arr = baseArr.tensorAlongDimension(i, 2);
            assertEquals( assertions[i], arr,"Failed at index " + i);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThreeTwoTwo(Nd4jBackend backend) {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        INDArray[] assertions = new INDArray[] {Nd4j.create(new double[] {1, 3}), Nd4j.create(new double[] {2, 4}),
                Nd4j.create(new double[] {5, 7}), Nd4j.create(new double[] {6, 8}),
                Nd4j.create(new double[] {9, 11}), Nd4j.create(new double[] {10, 12}),

        };

        assertEquals(assertions.length, threeTwoTwo.tensorsAlongDimension(1));
        for (int i = 0; i < assertions.length; i++) {
            INDArray arr = threeTwoTwo.tensorAlongDimension(i, 1);
            assertEquals(assertions[i], arr);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThreeTwoTwoTwo(Nd4jBackend backend) {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        INDArray[] assertions = new INDArray[] {Nd4j.create(new double[] {1, 2}), Nd4j.create(new double[] {3, 4}),
                Nd4j.create(new double[] {5, 6}), Nd4j.create(new double[] {7, 8}),
                Nd4j.create(new double[] {9, 10}), Nd4j.create(new double[] {11, 12}),

        };

        assertEquals(assertions.length, threeTwoTwo.tensorsAlongDimension(2));
        for (int i = 0; i < assertions.length; i++) {
            assertEquals(assertions[i], threeTwoTwo.tensorAlongDimension(i, 2));
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRow(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        for (int i = 0; i < matrix.rows(); i++) {
            INDArray row = matrix.getRow(i);
//            System.out.println(matrix.getRow(i));
        }
        matrix.putRow(1, Nd4j.create(new double[] {1, 2}));
        assertEquals(matrix.getRow(0), matrix.getRow(1));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenFirstDim(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        INDArray[] assertions = new INDArray[] {Nd4j.create(new double[] {1, 5}), Nd4j.create(new double[] {2, 6}),
                Nd4j.create(new double[] {3, 7}), Nd4j.create(new double[] {4, 8}),
                Nd4j.create(new double[] {9, 13}), Nd4j.create(new double[] {10, 14}),
                Nd4j.create(new double[] {11, 15}), Nd4j.create(new double[] {12, 16}),


        };

        for (int i = 0; i < baseArr.tensorsAlongDimension(1); i++) {
            INDArray arr = baseArr.tensorAlongDimension(i, 1);
            assertEquals(assertions[i], arr,"Failed at index " + i);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapePermute(Nd4jBackend backend) {
        INDArray arrNoPermute = Nd4j.ones(DataType.DOUBLE,5, 3, 4);
        INDArray reshaped2dNoPermute = arrNoPermute.reshape(5 * 3, 4); //OK
        assertArrayEquals(reshaped2dNoPermute.shape(), new long[] {5 * 3, 4});

        INDArray arr = Nd4j.ones(DataType.DOUBLE,5, 4, 3);
        INDArray permuted = arr.permute(0, 2, 1);
        assertArrayEquals(arrNoPermute.shape(), permuted.shape());
        INDArray reshaped2D = permuted.reshape(5 * 3, 4); //NullPointerException
        assertArrayEquals(reshaped2D.shape(), new long[] {5 * 3, 4});
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEight(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(2, 2, 2);
        assertEquals(2, baseArr.tensorsAlongDimension(0, 1));
        INDArray columnVectorFirst = Nd4j.create(new double[][] {{1, 3}, {5, 7}});
        INDArray columnVectorSecond = Nd4j.create(new double[][] {{2, 4}, {6, 8}});
        INDArray test1 = baseArr.tensorAlongDimension(0, 0, 1);
        assertEquals(columnVectorFirst, test1);
        INDArray test2 = baseArr.tensorAlongDimension(1, 0, 1);
        assertEquals(columnVectorSecond, test2);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOtherReshape(Nd4jBackend backend) {
        INDArray nd = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6}, new long[] {2, 3});

        INDArray slice = nd.slice(1, 0);

        INDArray vector = slice;
//        for (int i = 0; i < vector.length(); i++) {
//            System.out.println(vector.getDouble(i));
//        }
        assertEquals(Nd4j.create(new double[] {4, 5, 6}), vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorAlongDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        INDArray assertion = Nd4j.create(new double[] {3, 4});
        INDArray vectorDimensionTest = arr.vectorAlongDimension(1, 2);
        assertEquals(assertion, vectorDimensionTest);
        val vectorsAlongDimension1 = arr.vectorsAlongDimension(1);
        assertEquals(8, vectorsAlongDimension1);
        INDArray zeroOne = arr.vectorAlongDimension(0, 1);
        assertEquals(zeroOne, Nd4j.create(new double[] {1, 3, 5}));

        INDArray testColumn2Assertion = Nd4j.create(new double[] {2, 4, 6});
        INDArray testColumn2 = arr.vectorAlongDimension(1, 1);

        assertEquals(testColumn2Assertion, testColumn2);


        INDArray testColumn3Assertion = Nd4j.create(new double[] {7, 9, 11});
        INDArray testColumn3 = arr.vectorAlongDimension(2, 1);
        assertEquals(testColumn3Assertion, testColumn3);


        INDArray v1 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(new long[] {2, 2});
        INDArray testColumnV1 = v1.vectorAlongDimension(0, 0);
        INDArray testColumnV1Assertion = Nd4j.create(new double[] {1, 3});
        assertEquals(testColumnV1Assertion, testColumnV1);

        INDArray testRowV1 = v1.vectorAlongDimension(1, 0);
        INDArray testRowV1Assertion = Nd4j.create(new double[] {2, 4});
        assertEquals(testRowV1Assertion, testRowV1);

        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray vectorOne = n.vectorAlongDimension(1, 2);
        INDArray assertionVectorOne = Nd4j.create(new double[] {3, 4});
        assertEquals(assertionVectorOne, vectorOne);


        INDArray oneThroughSixteen = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);

        assertEquals(8, oneThroughSixteen.vectorsAlongDimension(1));
        assertEquals(Nd4j.create(new double[] {1, 5}), oneThroughSixteen.vectorAlongDimension(0, 1));
        assertEquals(Nd4j.create(new double[] {2, 6}), oneThroughSixteen.vectorAlongDimension(1, 1));
        assertEquals(Nd4j.create(new double[] {3, 7}), oneThroughSixteen.vectorAlongDimension(2, 1));
        assertEquals(Nd4j.create(new double[] {4, 8}), oneThroughSixteen.vectorAlongDimension(3, 1));
        assertEquals(Nd4j.create(new double[] {9, 13}), oneThroughSixteen.vectorAlongDimension(4, 1));
        assertEquals(Nd4j.create(new double[] {10, 14}), oneThroughSixteen.vectorAlongDimension(5, 1));
        assertEquals(Nd4j.create(new double[] {11, 15}), oneThroughSixteen.vectorAlongDimension(6, 1));
        assertEquals(Nd4j.create(new double[] {12, 16}), oneThroughSixteen.vectorAlongDimension(7, 1));


        INDArray fourdTest = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        double[][] assertionsArr =
                new double[][] {{1, 3}, {2, 4}, {5, 7}, {6, 8}, {9, 11}, {10, 12}, {13, 15}, {14, 16},

                };

        assertEquals(assertionsArr.length, fourdTest.vectorsAlongDimension(2));

        for (int i = 0; i < assertionsArr.length; i++) {
            INDArray test = fourdTest.vectorAlongDimension(i, 2);
            INDArray assertionEntry = Nd4j.create(assertionsArr[i]);
            assertEquals(assertionEntry, test);
        }


    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumnSum(Nd4jBackend backend) {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600, DataType.FLOAT).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[] {44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar,getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowMean(Nd4jBackend backend) {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray rowMean = twoByThree.mean(1);
        INDArray assertion = Nd4j.create(new double[] {1.5, 3.5});
        assertEquals(assertion, rowMean,getFailureMessage(backend));


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowStd(Nd4jBackend backend) {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray rowStd = twoByThree.std(1);
        INDArray assertion = Nd4j.create(new double[] {0.7071067811865476f, 0.7071067811865476f});
        assertEquals(assertion, rowStd,getFailureMessage(backend));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumnSumDouble(Nd4jBackend backend) {
        DataType initialType = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        INDArray twoByThree = Nd4j.linspace(1, 600, 600, DataType.DOUBLE).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new double[] {44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar,getFailureMessage(backend));
        DataTypeUtil.setDTypeForContext(initialType);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumnVariance(Nd4jBackend backend) {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray columnVar = twoByThree.var(true, 0);
        INDArray assertion = Nd4j.create(new double[] {2, 2});
        assertEquals(assertion, columnVar);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCumSum(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {1, 4});
        INDArray cumSumAnswer = Nd4j.create(new double[] {1, 3, 6, 10}, new long[] {1, 4});
        INDArray cumSumTest = n.cumsum(0);
        assertEquals( cumSumAnswer, cumSumTest,getFailureMessage(backend));

        INDArray n2 = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);

        INDArray axis0assertion = Nd4j.create(new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0}, n2.shape());
        INDArray axis0Test = n2.cumsum(0);
        assertEquals(axis0assertion, axis0Test,getFailureMessage(backend));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumRow(Nd4jBackend backend) {
        INDArray rowVector10 = Nd4j.ones(DataType.DOUBLE,1,10);
        INDArray sum1 = rowVector10.sum(1);
        assertArrayEquals(new long[] {1}, sum1.shape());
        assertTrue(sum1.getDouble(0) == 10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumColumn(Nd4jBackend backend) {
        INDArray colVector10 = Nd4j.ones(10, 1);
        INDArray sum0 = colVector10.sum(0);
        assertArrayEquals( new long[] {1}, sum0.shape());
        assertTrue(sum0.getDouble(0) == 10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum2d(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(10, 10);
        INDArray sum0 = arr.sum(0);
        assertArrayEquals(new long[] {10}, sum0.shape());

        INDArray sum1 = arr.sum(1);
        assertArrayEquals(new long[] {10}, sum1.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum2dv2(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(10, 10);
        INDArray sumBoth = arr.sum(0, 1);
        assertArrayEquals(new long[0], sumBoth.shape());
        assertTrue(sumBoth.getDouble(0) == 100);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteReshape(Nd4jBackend backend) {
        INDArray arrTest = Nd4j.arange(60).reshape('c', 3, 4, 5);
        INDArray permute = arrTest.permute(2, 1, 0);
        assertArrayEquals(new long[] {5, 4, 3}, permute.shape());
        assertArrayEquals(new long[] {1, 5, 20}, permute.stride());
        INDArray reshapedPermute = permute.reshape(-1, 12);
        assertArrayEquals(new long[] {5, 12}, reshapedPermute.shape());
        assertArrayEquals(new long[] {12, 1}, reshapedPermute.stride());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRavel(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray asseriton = Nd4j.linspace(1, 4, 4);
        INDArray raveled = linspace.ravel();
        assertEquals(asseriton, raveled);

        INDArray tensorLinSpace = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        INDArray linspaced = Nd4j.linspace(1, 16, 16);
        INDArray tensorLinspaceRaveled = tensorLinSpace.ravel();
        assertEquals(linspaced, tensorLinspaceRaveled);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutScalar(Nd4jBackend backend) {
        //Check that the various putScalar methods have the same result...
        val shapes = new int[][] {{3, 4}, {1, 4}, {3, 1}, {3, 4, 5}, {1, 4, 5}, {3, 1, 5}, {3, 4, 1}, {1, 1, 5},
                {3, 4, 5, 6}, {1, 4, 5, 6}, {3, 1, 5, 6}, {3, 4, 1, 6}, {3, 4, 5, 1}, {1, 1, 5, 6},
                {3, 1, 1, 6}, {3, 1, 1, 1}};

        for (int[] shape : shapes) {
            int rank = shape.length;
            NdIndexIterator iter = new NdIndexIterator(shape);
            INDArray firstC = Nd4j.create(shape, 'c');
            INDArray firstF = Nd4j.create(shape, 'f');
            INDArray secondC = Nd4j.create(shape, 'c');
            INDArray secondF = Nd4j.create(shape, 'f');

            int i = 0;
            while (iter.hasNext()) {
                val currIdx = iter.next();
                firstC.putScalar(currIdx, i);
                firstF.putScalar(currIdx, i);

                switch (rank) {
                    case 2:
                        secondC.putScalar(currIdx[0], currIdx[1], i);
                        secondF.putScalar(currIdx[0], currIdx[1], i);
                        break;
                    case 3:
                        secondC.putScalar(currIdx[0], currIdx[1], currIdx[2], i);
                        secondF.putScalar(currIdx[0], currIdx[1], currIdx[2], i);
                        break;
                    case 4:
                        secondC.putScalar(currIdx[0], currIdx[1], currIdx[2], currIdx[3], i);
                        secondF.putScalar(currIdx[0], currIdx[1], currIdx[2], currIdx[3], i);
                        break;
                    default:
                        throw new RuntimeException();
                }
                i++;
            }
            assertEquals(firstC, firstF);
            assertEquals(firstC, secondC);
            assertEquals(firstC, secondF);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeToTrueScalar_1(Nd4jBackend backend) {
        val orig = Nd4j.create(new float[]{1.0f}, new int[]{1, 1});
        val exp = Nd4j.scalar(1.0f);

        assertArrayEquals(new long[]{1, 1}, orig.shape());

        val reshaped = orig.reshape();

        assertArrayEquals(exp.shapeInfoDataBuffer().asLong(), reshaped.shapeInfoDataBuffer().asLong());
        assertEquals(exp, reshaped);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeToTrueScalar_2(Nd4jBackend backend) {
        val orig = Nd4j.create(new float[]{1.0f}, new int[]{1});
        val exp = Nd4j.scalar(1.0f);

        assertArrayEquals(new long[]{1}, orig.shape());

        val reshaped = orig.reshape();

        assertArrayEquals(exp.shapeInfoDataBuffer().asLong(), reshaped.shapeInfoDataBuffer().asLong());
        assertEquals(exp, reshaped);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeToTrueScalar_3(Nd4jBackend backend) {
        val orig = Nd4j.create(new float[]{1.0f}, new int[]{1, 1});
        val exp = Nd4j.createFromArray(new float[]{1.0f});

        assertArrayEquals(new long[]{1, 1}, orig.shape());

        val reshaped = orig.reshape(1);

        assertArrayEquals(exp.shapeInfoDataBuffer().asLong(), reshaped.shapeInfoDataBuffer().asLong());
        assertEquals(exp, reshaped);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeToTrueScalar_4(Nd4jBackend backend) {
        val orig = Nd4j.create(new float[]{1.0f}, new int[]{1, 1});
        val exp = Nd4j.scalar(1.0f);

        assertArrayEquals(new long[]{1, 1}, orig.shape());

        val reshaped = orig.reshape(new int[0]);

        assertArrayEquals(exp.shapeInfoDataBuffer().asLong(), reshaped.shapeInfoDataBuffer().asLong());
        assertEquals(exp, reshaped);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewAfterReshape(Nd4jBackend backend) {
        val x = Nd4j.rand(3,4);
        val x2 = x.ravel();
        val x3 = x.reshape(6,2);

        assertFalse(x.isView());
        assertTrue(x2.isView());
        assertTrue(x3.isView());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
