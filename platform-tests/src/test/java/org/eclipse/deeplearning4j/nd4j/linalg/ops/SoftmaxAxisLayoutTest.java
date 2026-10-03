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
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * softmax along any axis of a dense array. The CPU helper indexed every TAD of a dense C-order array as consecutive
 * elements, which holds only for the last non-unit axis: along axis 2 of [2, 5, 3, 2] (attention's key axis) it read
 * the wrong elements, left the rest untouched and, in place, let neighbouring TADs overwrite each other. Its float
 * path also turned a row whose logits are all -Inf (every key masked) into NaN.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class SoftmaxAxisLayoutTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyAxisMatchesTheDefinition(Nd4jBackend backend) {
        long[] shape = {2, 5, 3, 2};
        Nd4j.getRandom().setSeed(5);
        INDArray logits = Nd4j.rand(DataType.DOUBLE, shape).muli(8).subi(4);
        for (DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT}) {
            INDArray input = logits.castTo(type);
            double tolerance = type == DataType.DOUBLE ? 1e-12 : 1e-6;
            for (int axis = 0; axis < shape.length; axis++) {
                double[] expected = reference(input.castTo(DataType.DOUBLE), axis);
                INDArray out = Nd4j.create(type, shape);
                Nd4j.exec(new SoftMax(input, out, axis));
                assertArrayEquals(expected, out.dup('c').data().asDouble(), tolerance, type + " axis " + axis);

                INDArray inPlace = input.dup('c');
                Nd4j.exec(new SoftMax(inPlace, inPlace, axis));
                assertArrayEquals(expected, inPlace.dup('c').data().asDouble(), tolerance,
                        type + " in place, axis " + axis);

                INDArray negative = Nd4j.create(type, shape);
                Nd4j.exec(new SoftMax(input, negative, axis - shape.length));
                assertArrayEquals(expected, negative.dup('c').data().asDouble(), tolerance,
                        type + " axis " + (axis - shape.length));
            }
        }
    }

    /** A row whose every logit is -Inf (an attention row with all its keys masked) gets weights of ~0, not NaN. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void fullyMaskedRowsGiveNoNaN(Nd4jBackend backend) {
        double inf = Double.NEGATIVE_INFINITY;
        INDArray logits = Nd4j.createFromArray(new double[][]{{1, 2, 3, 4}, {inf, inf, inf, inf},
                {0.5, inf, 0.5, inf}});
        double[] open = reference(logits.getRow(0, true), 1);
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.HALF, DataType.DOUBLE}) {
            INDArray out = Nd4j.create(type, 3, 4);
            Nd4j.exec(new SoftMax(logits.castTo(type), out, 1));
            double tolerance = type == DataType.HALF ? 1e-3 : 1e-6;
            for (int j = 0; j < 4; j++) {
                assertEquals(open[j], out.getDouble(0, j), tolerance, type + " open row [" + j + "]");
                double masked = out.getDouble(1, j);
                assertFalse(Double.isNaN(masked), type + " masked row [" + j + "] is NaN");
                assertTrue(Math.abs(masked) < 1e-6, type + " masked row [" + j + "] = " + masked);
                assertEquals(j % 2 == 0 ? 0.5 : 0.0, out.getDouble(2, j), tolerance,
                        type + " half-masked row [" + j + "]");
            }
        }
    }

    /**
     * log_softmax of a row whose every logit is -Inf is -Inf throughout (the log of the zero weights softmax gives it),
     * not log(0 / 0) = NaN, on the matrix path and the vector path, in place too; rows with a finite logit are as before.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void logSoftmaxOfFullyMaskedRowsIsMinusInfinity(Nd4jBackend backend) {
        double inf = Double.NEGATIVE_INFINITY;
        INDArray logits = Nd4j.createFromArray(new double[][]{{1, 2, 3, 4}, {inf, inf, inf, inf},
                {0.5, inf, 0.5, inf}});
        double[] open = reference(logits.getRow(0, true), 1);
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.HALF, DataType.DOUBLE}) {
            double tolerance = type == DataType.HALF ? 2e-3 : 1e-6;
            for (boolean inPlace : new boolean[]{false, true}) {
                String label = type + (inPlace ? " in place" : "");
                INDArray in = logits.castTo(type);
                INDArray out = inPlace ? in : Nd4j.create(type, 3, 4);
                logSoftmax(in, out, 1);
                for (int j = 0; j < 4; j++) {
                    assertEquals(Math.log(open[j]), out.getDouble(0, j), tolerance, label + " open row [" + j + "]");
                    assertEquals(inf, out.getDouble(1, j), label + " masked row [" + j + "]");
                    if (j % 2 == 0) {
                        assertEquals(Math.log(0.5), out.getDouble(2, j), tolerance, label + " half-masked row [" + j + "]");
                    } else {
                        assertEquals(inf, out.getDouble(2, j), label + " half-masked row [" + j + "]");
                    }
                }
                // a vector takes the vector path
                INDArray vector = Nd4j.valueArrayOf(new long[]{4}, inf, type);
                INDArray vectorOut = inPlace ? vector : Nd4j.create(type, 4);
                logSoftmax(vector, vectorOut, 0);
                for (int j = 0; j < 4; j++) {
                    assertEquals(inf, vectorOut.getDouble(j), label + " masked vector [" + j + "]");
                }
            }
        }
    }

    private static void logSoftmax(INDArray in, INDArray out, int dimension) {
        Nd4j.exec(DynamicCustomOp.builder("log_softmax").addInputs(in).addOutputs(out)
                .addIntegerArguments((long) dimension).build());
    }

    /** softmax along `axis` of a C-order view of `input`, from the definition with the maximum subtracted. */
    private static double[] reference(INDArray input, int axis) {
        long[] shape = input.shape();
        double[] in = input.dup('c').data().asDouble();
        double[] out = new double[in.length];
        long[] strides = new long[shape.length];
        strides[shape.length - 1] = 1;
        for (int d = shape.length - 2; d >= 0; d--) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        for (int start = 0; start < in.length; start++) {
            if ((start / strides[axis]) % shape[axis] != 0) continue;
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < shape[axis]; k++) {
                max = Math.max(max, in[(int) (start + k * strides[axis])]);
            }
            double sum = 0;
            for (int k = 0; k < shape[axis]; k++) {
                sum += Math.exp(in[(int) (start + k * strides[axis])] - max);
            }
            for (int k = 0; k < shape[axis]; k++) {
                int index = (int) (start + k * strides[axis]);
                out[index] = Math.exp(in[index] - max) / sum;
            }
        }
        return out;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
