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
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * tile_bp: the gradient of an entry of the input is the sum of the entries of the output's gradient it was tiled into.
 * The CPU helper added each entry of the gradient into gradI at the position of its memory offset taken as a logical
 * index, which is the entry it belongs to only when gradI is in C order (gradI has the layout of the input), and
 * added in the element type, which stops growing a HALF sum at 2048.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TileBpTest extends BaseNd4jTestWithBackends {

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    /** The sum over the repeats, from the definition: output[i] = input[i % shape] in each dimension. */
    private static double[] reference(long[] inShape, long[] reps, double[] gradO) {
        int rank = inShape.length;
        long[] outShape = new long[rank];
        long inLength = 1;
        for (int d = 0; d < rank; d++) {
            outShape[d] = inShape[d] * reps[d];
            inLength *= inShape[d];
        }
        double[] expected = new double[(int) inLength];
        long[] coords = new long[rank];
        for (int i = 0; i < gradO.length; i++) {
            long rest = i;
            for (int d = rank - 1; d >= 0; d--) {
                coords[d] = rest % outShape[d];
                rest /= outShape[d];
            }
            long inIndex = 0;
            for (int d = 0; d < rank; d++)
                inIndex = inIndex * inShape[d] + coords[d] % inShape[d];
            expected[(int) inIndex] += gradO[i];
        }
        return expected;
    }

    private static INDArray tileBp(INDArray input, INDArray gradO, long[] reps) {
        return Nd4j.exec(DynamicCustomOp.builder("tile_bp").addInputs(input, gradO).addIntegerArguments(reps)
                .build())[0];
    }

    /** gradO in C order, F order, or a stepped view of a larger array. */
    private static INDArray gradLayout(INDArray c, int variant) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gradientSumsTheRepeatsInAnyLayout(Nd4jBackend backend) {
        long[][][] cases = {
                {{2, 3}, {2, 3}},
                {{4}, {5}},
                {{3, 1, 2}, {2, 3, 1}},
                {{5, 4}, {1, 1}},
                {{1}, {7}},
                {{6, 5}, {3, 4}},
                {{30, 40}, {3, 2}},
                {{2, 3, 4, 5}, {2, 1, 2, 3}},
        };
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (long[][] c : cases) {
                long[] inShape = c[0];
                long[] reps = c[1];
                long[] outShape = new long[inShape.length];
                for (int d = 0; d < inShape.length; d++)
                    outShape[d] = inShape[d] * reps[d];

                Nd4j.getRandom().setSeed(Arrays.hashCode(inShape) + Arrays.hashCode(reps));
                INDArray gradO = Nd4j.rand(type, outShape);
                double[] expected = reference(inShape, reps, logical(gradO));
                for (char inputOrder : new char[]{'c', 'f'}) {
                    // gradI has the layout of the input
                    INDArray input = Nd4j.zeros(type, inShape).dup(inputOrder);
                    for (int variant = 0; variant < 3; variant++) {
                        INDArray gradI = tileBp(input, gradLayout(gradO, variant), reps);
                        assertArrayEquals(inShape, gradI.shape());
                        assertArrayEquals(expected, logical(gradI), type == DataType.DOUBLE ? 1e-12 : 1e-4,
                                type + " tile_bp of " + Arrays.toString(inShape) + " by " + Arrays.toString(reps)
                                        + ", input in order " + inputOrder + ", gradient layout " + variant);
                    }
                }
            }
        }
    }

    /** A sum of 4096 ones is 4096, not the 2048 (HALF) or 256 (BFLOAT16) where adding one stops changing the sum. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void lowPrecisionSumsAreAccumulatedWide(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.HALF, DataType.BFLOAT16}) {
            INDArray gradI = tileBp(Nd4j.zeros(type, 1), Nd4j.ones(type, 4096), new long[]{4096});
            assertEquals(4096.0, gradI.getDouble(0), 0.0, type + " sum of 4096 ones");
        }
    }
}
