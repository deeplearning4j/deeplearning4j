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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * gather reads slice indices[i] along the axis. With index validation off (as during DSP replay), an index outside
 * the axis gathers zeros on every backend. It used to be clamped to the nearest slice on some paths, which returned
 * another slice's data, and skipped on others, which left the output unwritten.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class GatherOutOfRangeTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void indicesOutsideTheAxisGatherZeros(Nd4jBackend backend) {
        INDArray input = matrix();
        assertExact("axis 0", new double[][]{{4, 5, 6}, {0, 0, 0}, {0, 0, 0}, {7, 8, 9}},
                gather(input, Nd4j.createFromArray(1L, 7L, -1L, 2L), 0));
        assertExact("axis 1", new double[][]{{3, 0, 1}, {6, 0, 4}, {9, 0, 7}, {12, 0, 10}},
                gather(input, Nd4j.createFromArray(2L, 5L, 0L), 1));
    }

    /** A vector input takes a fast path for float, double and long, and a generic one otherwise. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void vectorInputGathersZerosOutsideIt(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT64, DataType.INT32}) {
            INDArray input = Nd4j.createFromArray(10, 20, 30, 40, 50).castTo(type);
            INDArray out = gather(input, Nd4j.createFromArray(0L, 9L, -2L, 4L), 0);
            assertEquals(type, out.dataType());
            assertArrayEquals(new double[]{10, 0, 0, 50}, out.dup().data().asDouble(), 0.0, type.toString());
        }
    }

    /** Indices given as integer arguments follow the same rule. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integerArgumentIndicesGatherZerosOutsideTheAxis(Nd4jBackend backend) {
        INDArray out = Nd4j.exec(DynamicCustomOp.builder("gather").addInputs(matrix())
                .addIntegerArguments(0, 1, 7, 2).addBooleanArguments(false).build())[0];
        assertArrayEquals(new double[]{4, 5, 6, 0, 0, 0, 7, 8, 9}, out.dup().data().asDouble(), 0.0);
    }

    private static INDArray matrix() {
        return Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
    }

    private static INDArray gather(INDArray input, INDArray indices, int axis) {
        return Nd4j.exec(DynamicCustomOp.builder("gather").addInputs(input, indices)
                .addIntegerArguments(axis).addBooleanArguments(false).build())[0];
    }

    private static void assertExact(String label, double[][] expected, INDArray actual) {
        assertArrayEquals(new long[]{expected.length, expected[0].length}, actual.shape(), label + " shape");
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], actual.getDouble(i, j), 0.0, label + " [" + i + ", " + j + "]");
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
