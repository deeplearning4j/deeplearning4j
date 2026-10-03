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
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * bincount counts each value, or sums its weights, into the bin of that value: bins [0, n) where n is the largest
 * value plus one, raised to minLength and capped at maxLength. Values outside the bins add nothing. The values and
 * weights may be views of any layout, and the weights may have any numeric type.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class BincountTest extends BaseNd4jTestWithBackends {

    private static final long[][] VALUES = {{3, 0, 7, 3, 1}, {5, 3, 3, 0, 2}, {7, 6, 1, 1, 3}};

    private static final class Layout {
        final String name;
        final UnaryOperator<INDArray> of;

        Layout(String name, UnaryOperator<INDArray> of) {
            this.name = name;
            this.of = of;
        }
    }

    private static final Layout[] LAYOUTS = {
            new Layout("C order", x -> x.dup('c')),
            new Layout("F order", x -> x.dup('f')),
            new Layout("offset view", x -> {
                INDArray parent = Nd4j.create(x.dataType(), x.size(0) + 1, x.size(1));
                return parent.get(NDArrayIndex.interval(1, x.size(0) + 1), NDArrayIndex.all()).assign(x);
            }),
            new Layout("stepped view", x -> {
                INDArray parent = Nd4j.create(x.dataType(), x.size(0), 2 * x.size(1) + 1);
                return parent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2 * x.size(1) + 1)).assign(x);
            }),
    };

    private static double[] reference(long[][] values, double[][] weights, long bins) {
        double[] out = new double[(int) bins];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[i].length; j++) {
                long v = values[i][j];
                if (v >= 0 && v < bins)
                    out[(int) v] += weights == null ? 1 : weights[i][j];
            }
        return out;
    }

    private static void check(List<String> failures, String what, INDArray actual, double[] expected,
                              DataType expectedType) {
        if (actual.dataType() != expectedType) {
            failures.add(what + ": type " + actual.dataType() + ", expected " + expectedType);
            return;
        }
        double[] a = actual.castTo(DataType.DOUBLE).dup().data().asDouble();
        if (a.length != expected.length) {
            failures.add(what + ": " + a.length + " bins, expected " + expected.length);
            return;
        }
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - expected[i]) > 1e-5 * (1 + Math.abs(expected[i]))) {
                failures.add(what + ": " + Arrays.toString(a) + ", expected " + Arrays.toString(expected));
                return;
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void countsAndWeightsLandInTheirValuesBins(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        double[][] weights = new double[VALUES.length][VALUES[0].length];
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[i].length; j++)
                weights[i][j] = 0.5 + i + 0.25 * j;
        for (DataType valueType : new DataType[]{DataType.INT32, DataType.INT64}) {
            INDArray values = Nd4j.createFromArray(VALUES).castTo(valueType);
            for (Layout vl : LAYOUTS) {
                String where = valueType + " values, " + vl.name;
                INDArray counts = Nd4j.exec(DynamicCustomOp.builder("bincount").addInputs(vl.of.apply(values))
                        .build())[0];
                check(failures, where + ": counts", counts, reference(VALUES, null, 8), DataType.INT64);

                INDArray capped = Nd4j.exec(DynamicCustomOp.builder("bincount").addInputs(vl.of.apply(values))
                        .addIntegerArguments(0, 5).build())[0];
                check(failures, where + ": counts capped at 5 bins", capped, reference(VALUES, null, 5), DataType.INT64);

                INDArray widened = Nd4j.exec(DynamicCustomOp.builder("bincount").addInputs(vl.of.apply(values))
                        .addIntegerArguments(11, 20).build())[0];
                check(failures, where + ": counts in at least 11 bins", widened, reference(VALUES, null, 11),
                        DataType.INT64);

                for (DataType weightType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                    INDArray w = Nd4j.createFromArray(weights).castTo(weightType);
                    for (Layout wl : LAYOUTS) {
                        INDArray sums = Nd4j.exec(DynamicCustomOp.builder("bincount")
                                .addInputs(vl.of.apply(values), wl.of.apply(w)).build())[0];
                        check(failures, where + ", " + weightType + " weights, " + wl.name, sums,
                                reference(VALUES, weights, 8), weightType);
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void negativeValuesAddNothing(Nd4jBackend backend) {
        long[][] values = {{2, -1, 0}, {-3, 2, 1}};
        INDArray counts = Nd4j.exec(DynamicCustomOp.builder("bincount").addInputs(Nd4j.createFromArray(values))
                .build())[0];
        List<String> failures = new ArrayList<>();
        check(failures, "counts", counts, reference(values, null, 3), DataType.INT64);
        assertTrue(failures.isEmpty(), String.join("\n", failures));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
