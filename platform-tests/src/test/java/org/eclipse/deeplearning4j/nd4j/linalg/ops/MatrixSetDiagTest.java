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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * matrix_set_diag replaces the main diagonal of each matrix (the last two dimensions) with a vector and keeps the other
 * entries. On CPU every thread of its parallel loop went over all of the entries instead of its own chunk of them.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class MatrixSetDiagTest extends BaseNd4jTestWithBackends {

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    /** The same values in another layout: F order, or a stepped view into a larger array. */
    private static INDArray layout(INDArray c, int variant) {
        switch (variant) {
            case 1:
                return c.dup('f');
            case 2: {
                long[] shape = c.shape();
                long[] big = new long[shape.length];
                INDArrayIndex[] steps = new INDArrayIndex[shape.length];
                for (int i = 0; i < shape.length; i++) {
                    big[i] = 2 * shape[i];
                    steps[i] = NDArrayIndex.interval(0, 2, big[i]);
                }
                INDArray view = Nd4j.zeros(c.dataType(), big).addi(-7.0).get(steps);
                view.assign(c);
                return view;
            }
            default:
                return c.dup('c');
        }
    }

    private static String layoutName(int variant) {
        return variant == 0 ? "C order" : variant == 1 ? "F order" : "stepped view";
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void diagonalsAreReplacedInAnyLayout(Nd4jBackend backend) {
        long[][] shapes = {{5, 5}, {4, 6}, {6, 4}, {1, 7}, {7, 1}, {2, 3, 5, 5}, {3, 7, 9}, {64, 50, 50}};
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (long[] shape : shapes) {
                int rank = shape.length;
                int rows = (int) shape[rank - 2];
                int cols = (int) shape[rank - 1];
                int diagLength = Math.min(rows, cols);
                long[] diagShape = Arrays.copyOf(shape, rank - 1);
                diagShape[rank - 2] = diagLength;

                Nd4j.getRandom().setSeed(Arrays.hashCode(shape));
                INDArray input = Nd4j.rand(type, shape);
                INDArray diagonal = Nd4j.rand(type, diagShape).addi(10.0);

                double[] expected = logical(input);
                double[] diag = logical(diagonal);
                long batches = input.length() / ((long) rows * cols);
                for (int b = 0; b < batches; b++)
                    for (int i = 0; i < diagLength; i++)
                        expected[(int) (b * rows * cols + (long) i * cols + i)] = diag[b * diagLength + i];

                for (int variant = 0; variant < 3; variant++) {
                    INDArray result = Nd4j.exec(DynamicCustomOp.builder("matrix_set_diag")
                            .addInputs(layout(input, variant), layout(diagonal, variant)).build())[0];
                    assertArrayEquals(shape, result.shape());
                    assertArrayEquals(expected, logical(result), 0.0,
                            type + " " + Arrays.toString(shape) + " in " + layoutName(variant));
                }

                // an input and a diagonal in different layouts, into an output in a third
                INDArray out = Nd4j.createUninitialized(type, shape, 'f').assign(-1.0);
                Nd4j.exec(DynamicCustomOp.builder("matrix_set_diag").addInputs(layout(input, 2), layout(diagonal, 1))
                        .addOutputs(out).build());
                assertArrayEquals(expected, logical(out), 0.0,
                        type + " " + Arrays.toString(shape) + ", stepped input, F-order diagonal, F-order output");
            }
        }
    }
}
