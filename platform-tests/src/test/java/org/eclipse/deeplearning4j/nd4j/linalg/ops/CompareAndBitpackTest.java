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
 * compare_and_bitpack packs the comparisons of each 8 consecutive entries of the last dimension with a threshold into
 * one byte, the first entry in the highest bit. The CPU helper's general path (an F-ordered array, a view) stepped one
 * entry instead of 8 along the last dimension and never moved to the next row, so it packed the wrong entries and
 * wrote the wrong bytes; the CUDA launch took any array flagged as C order for a dense one.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class CompareAndBitpackTest extends BaseNd4jTestWithBackends {

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

    private static double[] expected(double[] values, double threshold) {
        double[] bytes = new double[values.length / 8];
        for (int j = 0; j < bytes.length; j++) {
            int bits = 0;
            for (int b = 0; b < 8; b++)
                if (values[8 * j + b] > threshold)
                    bits |= 1 << (7 - b);
            bytes[j] = bits;
        }
        return bytes;
    }

    private static INDArray values(DataType type, long... shape) {
        Nd4j.getRandom().setSeed(Arrays.hashCode(shape));
        INDArray random = Nd4j.rand(DataType.DOUBLE, shape);
        return (type == DataType.FLOAT || type == DataType.DOUBLE) ? random.castTo(type)
                : random.muli(10).castTo(type);
    }

    private static double thresholdOf(DataType type) {
        return (type == DataType.FLOAT || type == DataType.DOUBLE) ? 0.5 : 4.0;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyLayoutPacksTheSameBytes(Nd4jBackend backend) {
        long[][] shapes = {{8}, {64}, {1, 8}, {3, 16}, {5, 24}, {2, 3, 8}, {4, 64}, {1, 320}, {37, 72}, {2, 3, 4, 16},
                {3000, 64}};
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32}) {
            double threshold = thresholdOf(type);
            for (long[] shape : shapes) {
                INDArray c = values(type, shape);
                double[] expected = expected(logical(c), threshold);
                long[] outShape = shape.clone();
                outShape[outShape.length - 1] /= 8;
                for (int variant = 0; variant < 3; variant++) {
                    INDArray result = Nd4j.exec(DynamicCustomOp.builder("compare_and_bitpack")
                            .addInputs(layout(c, variant), Nd4j.scalar(type, threshold)).build())[0];
                    assertArrayEquals(outShape, result.shape());
                    assertArrayEquals(expected, logical(result), 0.0,
                            type + " " + Arrays.toString(shape) + " in " + layoutName(variant));
                }
            }
        }
    }

    /** The bytes go where the output's own strides put them, whatever the input's layout is. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bytesAreWrittenThroughTheOutputsStrides(Nd4jBackend backend) {
        long[][] shapes = {{3, 16}, {5, 24}, {2, 3, 8}, {37, 72}};
        for (long[] shape : shapes) {
            INDArray c = values(DataType.FLOAT, shape);
            double[] expected = expected(logical(c), 0.5);
            long[] outShape = shape.clone();
            outShape[outShape.length - 1] /= 8;
            for (int inputVariant = 0; inputVariant < 3; inputVariant++) {
                for (char outputOrder : new char[]{'c', 'f'}) {
                    INDArray out = Nd4j.createUninitialized(DataType.UINT8, outShape, outputOrder).assign(255);
                    Nd4j.exec(DynamicCustomOp.builder("compare_and_bitpack")
                            .addInputs(layout(c, inputVariant), Nd4j.scalar(DataType.FLOAT, 0.5)).addOutputs(out)
                            .build());
                    assertArrayEquals(expected, logical(out), 0.0, Arrays.toString(shape) + " in "
                            + layoutName(inputVariant) + " into an output in order " + outputOrder);
                }
            }
        }
    }
}
