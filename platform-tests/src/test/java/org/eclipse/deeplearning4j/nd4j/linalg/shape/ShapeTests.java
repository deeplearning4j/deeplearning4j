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

package org.eclipse.deeplearning4j.nd4j.linalg.shape;

import lombok.val;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Triple;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;

/**
 * @author Adam Gibson
 */
@NativeTag
@Tag(TagNames.NDARRAY_INDEXING)
public class ShapeTests extends BaseNd4jTestWithBackends {
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowColVectorVsScalar(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(2);
        assertTrue(arr.isRowVector());
        INDArray colVector = arr.reshape(2,1);
        assertTrue(colVector.isColumnVector());
        assertFalse(arr.isScalar());
        assertFalse(colVector.isScalar());

        INDArray arr3 = Nd4j.scalar(1.0);
        assertFalse(arr3.isColumnVector());
        assertFalse(arr3.isRowVector());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenZeroOne(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        long numTensors = baseArr.tensorsAlongDimension(0, 1);
        assertEquals(4, numTensors);

        for (int i = 0; i < numTensors; i++) {
            INDArray tensor = baseArr.tensorAlongDimension(i, 0, 1);
            assertArrayEquals(new long[]{2, 2}, tensor.shape(), "Tensor shape should be [2,2] at index " + i);

            // Each 2x2 tensor should match baseArr[:, :, d2, d3] for some d2, d3
            boolean matched = false;
            for (int d2 = 0; d2 < 2 && !matched; d2++) {
                for (int d3 = 0; d3 < 2 && !matched; d3++) {
                    boolean allMatch = true;
                    for (int d0 = 0; d0 < 2 && allMatch; d0++) {
                        for (int d1 = 0; d1 < 2 && allMatch; d1++) {
                            if (tensor.getDouble(d0, d1) != baseArr.getDouble(d0, d1, d2, d3)) {
                                allMatch = false;
                            }
                        }
                    }
                    if (allMatch) matched = true;
                }
            }
            assertTrue(matched, "Tensor " + i + " should match a valid [:,:,d2,d3] slice");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorAlongDimension1(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(1, 5, 5);
        assertEquals(arr.vectorsAlongDimension(0), 5);
        assertEquals(arr.vectorsAlongDimension(1), 5);
        for (int i = 0; i < arr.vectorsAlongDimension(0); i++) {
            if (i < arr.vectorsAlongDimension(0) - 1 && i > 0)
                assertEquals(25, arr.vectorAlongDimension(i, 0).length());
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenSecondDim(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        long numTensors = baseArr.tensorsAlongDimension(2);
        assertEquals(8, numTensors);

        for (int i = 0; i < numTensors; i++) {
            INDArray tensor = baseArr.tensorAlongDimension(i, 2);
            assertEquals(2, tensor.length(), "Tensor along dim 2 should have length 2");

            // Each vector should match baseArr[d0, d1, :, d3] for some d0, d1, d3
            boolean matched = false;
            for (int d0 = 0; d0 < 2 && !matched; d0++) {
                for (int d1 = 0; d1 < 2 && !matched; d1++) {
                    for (int d3 = 0; d3 < 2 && !matched; d3++) {
                        if (tensor.getDouble(0) == baseArr.getDouble(d0, d1, 0, d3) &&
                            tensor.getDouble(1) == baseArr.getDouble(d0, d1, 1, d3)) {
                            matched = true;
                        }
                    }
                }
            }
            assertTrue(matched, "Tensor " + i + " should match a valid [d0,d1,:,d3] slice");
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorAlongDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(4, 3, 2);

        // Verify vectorAlongDimension returns vectors of the correct length
        long numVectorsAlongDim2 = arr.vectorsAlongDimension(2);
        assertEquals(4 * 3, numVectorsAlongDim2); // 12 vectors of length 2
        for (int i = 0; i < numVectorsAlongDim2; i++) {
            INDArray v = arr.vectorAlongDimension(i, 2);
            assertEquals(2, v.length(), "Vector along dim 2 should have length 2");
        }

        long numVectorsAlongDim1 = arr.vectorsAlongDimension(1);
        assertEquals(4 * 2, numVectorsAlongDim1); // 8 vectors of length 3
        for (int i = 0; i < numVectorsAlongDim1; i++) {
            INDArray v = arr.vectorAlongDimension(i, 1);
            assertEquals(3, v.length(), "Vector along dim 1 should have length 3");
        }

        // 2x2 matrix: vectorAlongDimension along dim 0
        INDArray v1 = Nd4j.linspace(1, 4, 4, DataType.FLOAT).reshape(new long[] {2, 2});
        assertEquals(2, v1.vectorsAlongDimension(0));
        for (int i = 0; i < v1.vectorsAlongDimension(0); i++) {
            INDArray vec = v1.vectorAlongDimension(i, 0);
            assertEquals(2, vec.length(), "Vector along dim 0 of 2x2 should have length 2");
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThreeTwoTwo(Nd4jBackend backend) {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        long numTensors = threeTwoTwo.tensorsAlongDimension(1);
        assertEquals(6, numTensors);

        for (int i = 0; i < numTensors; i++) {
            INDArray tensor = threeTwoTwo.tensorAlongDimension(i, 1);
            assertEquals(2, tensor.length(), "Tensor along dim 1 should have length 2");

            // Each vector should match threeTwoTwo[d0, :, d2] for some d0, d2
            boolean matched = false;
            for (int d0 = 0; d0 < 3 && !matched; d0++) {
                for (int d2 = 0; d2 < 2 && !matched; d2++) {
                    if (tensor.getDouble(0) == threeTwoTwo.getDouble(d0, 0, d2) &&
                        tensor.getDouble(1) == threeTwoTwo.getDouble(d0, 1, d2)) {
                        matched = true;
                    }
                }
            }
            assertTrue(matched, "Tensor " + i + " should match a valid [d0,:,d2] slice");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoCopy(Nd4jBackend backend) {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);
        INDArray arr = Shape.newShapeNoCopy(threeTwoTwo, new long[] {3, 2, 2}, true);
        assertArrayEquals(arr.shape(), new long[] {3, 2, 2});
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThreeTwoTwoTwo(Nd4jBackend backend) {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        long numTensors = threeTwoTwo.tensorsAlongDimension(2);
        assertEquals(6, numTensors);

        for (int i = 0; i < numTensors; i++) {
            INDArray tensor = threeTwoTwo.tensorAlongDimension(i, 2);
            assertEquals(2, tensor.length(), "Tensor along dim 2 should have length 2");

            // Each vector should match threeTwoTwo[d0, d1, :] for some d0, d1
            boolean matched = false;
            for (int d0 = 0; d0 < 3 && !matched; d0++) {
                for (int d1 = 0; d1 < 2 && !matched; d1++) {
                    if (tensor.getDouble(0) == threeTwoTwo.getDouble(d0, d1, 0) &&
                        tensor.getDouble(1) == threeTwoTwo.getDouble(d0, d1, 1)) {
                        matched = true;
                    }
                }
            }
            assertTrue(matched, "Tensor " + i + " should match a valid [d0,d1,:] slice");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewAxis(Nd4jBackend backend) {
        INDArray tensor = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        INDArray assertion = tensor.reshape(1,3,2,2);
        INDArray tensorGet = tensor.get( NDArrayIndex.newAxis(), all(), all());
        assertEquals(assertion, tensorGet);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSixteenFirstDim(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        long numTensors = baseArr.tensorsAlongDimension(1);
        assertEquals(8, numTensors);

        for (int i = 0; i < numTensors; i++) {
            INDArray tensor = baseArr.tensorAlongDimension(i, 1);
            assertEquals(2, tensor.length(), "Tensor along dim 1 should have length 2");

            // Each vector should match baseArr[d0, :, d2, d3] for some d0, d2, d3
            boolean matched = false;
            for (int d0 = 0; d0 < 2 && !matched; d0++) {
                for (int d2 = 0; d2 < 2 && !matched; d2++) {
                    for (int d3 = 0; d3 < 2 && !matched; d3++) {
                        if (tensor.getDouble(0) == baseArr.getDouble(d0, 0, d2, d3) &&
                            tensor.getDouble(1) == baseArr.getDouble(d0, 1, d2, d3)) {
                            matched = true;
                        }
                    }
                }
            }
            assertTrue(matched, "Tensor " + i + " should match a valid [d0,:,d2,d3] slice");
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDimShuffle(Nd4jBackend backend) {
        INDArray scalarTest = Nd4j.scalar(0.0).reshape(1, -1);
        INDArray broadcast = scalarTest.dimShuffle(new Object[] {'x'}, new long[] {0, 1}, new boolean[] {true, true});
        assertTrue(broadcast.rank() == 3);
        INDArray rowVector = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(1, -1);
        assertEquals(rowVector,
                rowVector.dimShuffle(new Object[] {0, 1}, new int[] {0, 1}, new boolean[] {false, false}));
        //add extra dimension to row vector in middle
        INDArray rearrangedRowVector =
                rowVector.dimShuffle(new Object[] {0, 'x', 1}, new int[] {0, 1}, new boolean[] {true, true});
        assertArrayEquals(new long[] {1, 1, 4}, rearrangedRowVector.shape());

        INDArray dimshuffed = rowVector.dimShuffle(new Object[] {'x', 0, 'x', 'x'}, new long[] {0, 1},
                new boolean[] {true, true});
        assertArrayEquals(new long[] {1, 1, 1, 1, 4}, dimshuffed.shape());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEight(Nd4jBackend backend) {
        INDArray baseArr = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(2, 2, 2);
        assertEquals(2, baseArr.tensorsAlongDimension(0, 1));
        INDArray columnVectorFirst = Nd4j.create(new double[][] {{1, 3}, {2, 4}});
        INDArray columnVectorSecond = Nd4j.create(new double[][] {{5, 7}, {6, 8}});
        assertEquals(columnVectorFirst, baseArr.tensorAlongDimension(0, 0, 1));
        assertEquals(columnVectorSecond, baseArr.tensorAlongDimension(1, 0, 1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastShapes(){
        //Test cases: in1Shape, in2Shape, shapeOf(op(in1,in2))
        List<Triple<long[], long[], long[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new long[]{3,1}, new long[]{1,4}, new long[]{3,4}));
        testCases.add(new Triple<>(new long[]{3,1}, new long[]{3,4}, new long[]{3,4}));
        testCases.add(new Triple<>(new long[]{3,4}, new long[]{1,4}, new long[]{3,4}));
        testCases.add(new Triple<>(new long[]{3,4,1}, new long[]{1,1,5}, new long[]{3,4,5}));
        testCases.add(new Triple<>(new long[]{3,4,1}, new long[]{3,1,5}, new long[]{3,4,5}));
        testCases.add(new Triple<>(new long[]{3,1,5}, new long[]{1,4,1}, new long[]{3,4,5}));
        testCases.add(new Triple<>(new long[]{3,1,5}, new long[]{1,4,5}, new long[]{3,4,5}));
        testCases.add(new Triple<>(new long[]{3,1,5}, new long[]{3,4,5}, new long[]{3,4,5}));
        testCases.add(new Triple<>(new long[]{3,1,1,1}, new long[]{1,4,5,6}, new long[]{3,4,5,6}));
        testCases.add(new Triple<>(new long[]{1,1,1,6}, new long[]{3,4,5,6}, new long[]{3,4,5,6}));
        testCases.add(new Triple<>(new long[]{1,4,5,1}, new long[]{3,1,1,6}, new long[]{3,4,5,6}));
        testCases.add(new Triple<>(new long[]{1,6}, new long[]{3,4,5,1}, new long[]{3,4,5,6}));

        for(Triple<long[], long[], long[]> t : testCases){
            val x = t.getFirst();
            val y = t.getSecond();
            val exp = t.getThird();

            val act = Shape.broadcastOutputShape(x,y);
            assertArrayEquals(exp,act);
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
