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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ops whose helpers work on a temporary copy of an array leave their inputs and outputs as they should be. NDArray's
 * copy constructor gives a view sharing the buffer, and NDArray(NDArray*, bool, LaunchContext*) wraps the other
 * array's buffer, so helpers that "copied" an array either way wrote into it: nth_element sorted its input, sru and lstm
 * stored every step's state in c0 (and h0), static_bidirectional_rnn reversed its input and its backward outputs into
 * themselves, and lstmLayer's bidirectional sum ran the backward pass over the forward outputs. Each op runs twice on
 * the same inputs where it matters: the inputs must be unchanged and the runs must agree with each other and with a
 * reference.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class HelperCopyAliasingTest extends BaseNd4jTestWithBackends {

    private static INDArray[] exec(DynamicCustomOp op) {
        INDArray[] out = Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
        return out;
    }

    /** The values in a C-order array of the given shape; an empty shape is a scalar. */
    private static INDArray shaped(double[] values, long[] shape) {
        return shape.length == 0 ? Nd4j.scalar(values[0]) : Nd4j.createFromArray(values).reshape('c', shape);
    }

    private static void assertUnchanged(String what, INDArray before, INDArray after) {
        assertTrue(before.equalsWithEps(after, 0.0), what + " changed: " + after + " vs " + before);
    }

    private static void assertClose(String what, INDArray expected, INDArray actual, double tol) {
        assertTrue(Arrays.equals(expected.shape(), actual.shape()),
                what + ": shape " + Arrays.toString(actual.shape()) + " vs " + Arrays.toString(expected.shape()));
        double maxDiff = expected.castTo(DataType.DOUBLE).sub(actual.castTo(DataType.DOUBLE)).amaxNumber().doubleValue();
        assertTrue(maxDiff <= tol, what + ": max |diff| " + maxDiff + "\nactual " + actual + "\nexpected " + expected);
    }

    /** nth_element along the last axis: the n-th smallest (or largest) of each row of a sorted copy. */
    private static INDArray nthElementReference(INDArray x, int n, boolean reverse) {
        long last = x.size(x.rank() - 1);
        INDArray rows = x.dup('c').reshape('c', x.length() / last, last);
        double[] out = new double[(int) rows.size(0)];
        for (int r = 0; r < out.length; r++) {
            double[] row = rows.getRow(r).dup().toDoubleVector();
            Arrays.sort(row);
            out[r] = reverse ? row[row.length - 1 - n] : row[n];
        }
        return shaped(out, Arrays.copyOf(x.shape(), x.rank() - 1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nthElementSortsACopy(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(7);
        INDArray[] inputs = {Nd4j.rand(DataType.DOUBLE, 9), Nd4j.rand(DataType.DOUBLE, 4, 6),
                Nd4j.rand(DataType.DOUBLE, 2, 3, 5)};
        for (INDArray x : inputs) {
            for (boolean reverse : new boolean[]{false, true}) {
                int n = 2;
                INDArray before = x.dup();
                INDArray expected = nthElementReference(x, n, reverse);
                for (int run = 0; run < 2; run++) {
                    INDArray[] out = exec(DynamicCustomOp.builder("nth_element")
                            .addInputs(x, Nd4j.scalar(n)).addIntegerArguments(reverse ? 1 : 0).build());
                    String what = "nth_element " + Arrays.toString(x.shape()) + " reverse=" + reverse + " run " + run;
                    assertUnchanged(what + ": input", before, x);
                    assertClose(what, expected, out[0].reshape(expected.shape()), 0.0);
                }
            }
        }
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /**
     * sru over time, x [bS, inSize, time], w [3 * inSize, inSize] (rows: candidate, forget, reset), b [2 * inSize]
     * (forget, reset): z = x_t * w^T, f = sigmoid(z_f + b_f), r = sigmoid(z_r + b_r), c_t = f c_{t-1} + (1 - f) z_c,
     * h_t = r tanh(c_t) + (1 - r) x_t. Returns {h, c}, both [bS, inSize, time].
     */
    private static INDArray[] sruReference(INDArray x, INDArray w, INDArray b, INDArray c0) {
        int bS = (int) x.size(0), inSize = (int) x.size(1), time = (int) x.size(2);
        double[][][] h = new double[bS][inSize][time], c = new double[bS][inSize][time];
        double[][] prev = c0.toDoubleMatrix();
        for (int t = 0; t < time; t++) {
            double[][] next = new double[bS][inSize];
            for (int n = 0; n < bS; n++) {
                for (int k = 0; k < inSize; k++) {
                    double zc = 0, zf = 0, zr = 0;
                    for (int j = 0; j < inSize; j++) {
                        double xv = x.getDouble(n, j, t);
                        zc += xv * w.getDouble(k, j);
                        zf += xv * w.getDouble(inSize + k, j);
                        zr += xv * w.getDouble(2 * inSize + k, j);
                    }
                    double f = sigmoid(zf + b.getDouble(k)), r = sigmoid(zr + b.getDouble(inSize + k));
                    double ct = f * prev[n][k] + (1 - f) * zc;
                    next[n][k] = ct;
                    c[n][k][t] = ct;
                    h[n][k][t] = r * Math.tanh(ct) + (1 - r) * x.getDouble(n, k, t);
                }
            }
            prev = next;
        }
        return new INDArray[]{Nd4j.createFromArray(h), Nd4j.createFromArray(c)};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sruMatchesItsRecurrenceAndKeepsItsInitialState(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11);
        int bS = 2, inSize = 3, time = 4;
        INDArray x = Nd4j.rand(DataType.DOUBLE, bS, inSize, time).subi(0.5);
        INDArray w = Nd4j.rand(DataType.DOUBLE, 3 * inSize, inSize).subi(0.5);
        INDArray b = Nd4j.rand(DataType.DOUBLE, 2 * inSize).subi(0.5);
        INDArray c0 = Nd4j.rand(DataType.DOUBLE, bS, inSize).subi(0.5);
        INDArray mask = Nd4j.ones(DataType.DOUBLE, bS, inSize);
        INDArray c0Before = c0.dup();
        INDArray[] first = null;
        for (int run = 0; run < 2; run++) {
            INDArray[] out = exec(DynamicCustomOp.builder("sru").addInputs(x, w, b, c0, mask).build());
            assertUnchanged("sru run " + run + ": c0", c0Before, c0);
            INDArray[] expected = sruReference(x, w, b, c0Before);
            assertClose("sru h, run " + run, expected[0], out[0], 1e-12);
            assertClose("sru c, run " + run, expected[1], out[1], 1e-12);
            if (first == null) {
                first = new INDArray[]{out[0].dup(), out[1].dup()};
            } else {
                assertClose("sru h, second run", first[0], out[0], 0.0);
                assertClose("sru c, second run", first[1], out[1], 0.0);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void lstmKeepsItsInitialState(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(13);
        int time = 3, bS = 2, inSize = 3, numUnits = 4;
        INDArray x = Nd4j.rand(DataType.DOUBLE, time, bS, inSize).subi(0.5);
        INDArray h0 = Nd4j.rand(DataType.DOUBLE, bS, numUnits).subi(0.5);
        INDArray c0 = Nd4j.rand(DataType.DOUBLE, bS, numUnits).subi(0.5);
        INDArray wx = Nd4j.rand(DataType.DOUBLE, inSize, 4 * numUnits).subi(0.5);
        INDArray wh = Nd4j.rand(DataType.DOUBLE, numUnits, 4 * numUnits).subi(0.5);
        INDArray wc = Nd4j.rand(DataType.DOUBLE, 3 * numUnits).subi(0.5);
        INDArray wp = Nd4j.rand(DataType.DOUBLE, numUnits, numUnits).subi(0.5);
        INDArray b = Nd4j.rand(DataType.DOUBLE, 4 * numUnits).subi(0.5);
        INDArray h0Before = h0.dup(), c0Before = c0.dup();
        INDArray[] first = null;
        for (int run = 0; run < 2; run++) {
            INDArray[] out = exec(DynamicCustomOp.builder("lstm").addInputs(x, h0, c0, wx, wh, wc, wp, b)
                    .addIntegerArguments(0, 0).addFloatingPointArguments(0.0, 0.0, 1.0).build());
            assertUnchanged("lstm run " + run + ": h0", h0Before, h0);
            assertUnchanged("lstm run " + run + ": c0", c0Before, c0);
            if (first == null) {
                first = new INDArray[]{out[0].dup(), out[1].dup()};
            } else {
                assertClose("lstm h, second run", first[0], out[0], 0.0);
                assertClose("lstm c, second run", first[1], out[1], 0.0);
            }
        }
    }

    private static INDArray lstmLayerOutput(INDArray x, INDArray wx, INDArray wr, INDArray b, int directionMode) {
        // dataFormat 0 ([sL, bS, nIn]), sigmoid gates, tanh cell and output activations, biases, the full sequence out
        return exec(DynamicCustomOp.builder("lstmLayer").addInputs(x, wx, wr, b)
                .addIntegerArguments(0, directionMode, 2, 0, 0)
                .addBooleanArguments(true, false, false, false, false, true, false, false)
                .addFloatingPointArguments(0.0).build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void lstmLayerBidirectionalSumAddsBothDirections(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(43);
        int sL = 4, bS = 2, nIn = 3, nOut = 5;
        INDArray x = Nd4j.rand(DataType.DOUBLE, sL, bS, nIn).subi(0.5);
        INDArray wx = Nd4j.rand(DataType.DOUBLE, 2, nIn, 4 * nOut).subi(0.5);
        INDArray wr = Nd4j.rand(DataType.DOUBLE, 2, nOut, 4 * nOut).subi(0.5);
        INDArray b = Nd4j.rand(DataType.DOUBLE, 2, 4 * nOut).subi(0.5);
        INDArray concat = lstmLayerOutput(x, wx, wr, b, 3);
        INDArray expected = concat.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, nOut))
                .add(concat.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(nOut, 2 * nOut)));
        assertClose("lstmLayer bidirectional sum vs the halves of bidirectional concat", expected,
                lstmLayerOutput(x, wx, wr, b, 2), 1e-12);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void staticBidirectionalRnnMatchesTensorFlow(Nd4jBackend backend) {
        // libnd4j's static_bidir_rnn_test1 (TensorFlow's values): the backward outputs are reversed back in time
        int bS = 4, inSize = 4, numUnitsFW = 3, numUnitsBW = 3, time = 5;
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 0.01, 0.01, time * bS * inSize).reshape('c', time, bS, inSize);
        INDArray wx = Nd4j.valueArrayOf(new long[]{inSize, numUnitsFW}, 0.3, DataType.DOUBLE);
        INDArray wh = Nd4j.valueArrayOf(new long[]{numUnitsFW, numUnitsFW}, 0.4, DataType.DOUBLE);
        INDArray b = Nd4j.valueArrayOf(new long[]{2 * numUnitsFW}, 0.1, DataType.DOUBLE);
        INDArray h0FW = Nd4j.valueArrayOf(new long[]{bS, numUnitsFW}, 0.2, DataType.DOUBLE);
        INDArray h0BW = Nd4j.valueArrayOf(new long[]{bS, numUnitsBW}, 0.25, DataType.DOUBLE);
        INDArray maxTimeStep = Nd4j.createFromArray(new double[]{time - 1, time - 3, time - 4, 0});

        INDArray expH = Nd4j.createFromArray(new double[]{
                0.43819931, 0.43819931, 0.43819931, 0.86708881, 0.86708881, 0.86708881, 0.47615493, 0.47615493, 0.47615493,
                0.78347842, 0.78347842, 0.78347842, 0.51241561, 0.51241561, 0.51241561, 0.55529176, 0.55529176, 0.55529176,
                0., 0., 0., 0., 0., 0., 0.73880324, 0.73880324, 0.73880324,
                0.90935605, 0.90935605, 0.90935605, 0.77843476, 0.77843476, 0.77843476, 0.64692945, 0.64692945, 0.64692945,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.9052501, 0.9052501, 0.9052501, 0.9181592, 0.9181592, 0.9181592,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0.9555734, 0.9555734, 0.9555734, 0.8026439, 0.8026439, 0.8026439, 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0.}).reshape('c', time, bS, numUnitsFW + numUnitsBW);
        INDArray expHFWfinal = Nd4j.createFromArray(new double[]{0.9555734, 0.9555734, 0.9555734, 0.77843476,
                0.77843476, 0.77843476, 0.51241561, 0.51241561, 0.51241561, 0.2, 0.2, 0.2}).reshape('c', bS, numUnitsFW);
        INDArray expHBWfinal = Nd4j.createFromArray(new double[]{0.86708881, 0.86708881, 0.86708881, 0.78347842,
                0.78347842, 0.78347842, 0.55529176, 0.55529176, 0.55529176, 0.25, 0.25, 0.25}).reshape('c', bS, numUnitsBW);

        INDArray xBefore = x.dup();
        INDArray[] out = exec(DynamicCustomOp.builder("static_bidirectional_rnn")
                .addInputs(x, wx, wh, b, wx, wh, b, h0FW, h0BW, maxTimeStep).build());
        assertUnchanged("static_bidirectional_rnn: x", xBefore, x);
        assertClose("static_bidirectional_rnn h", expH, out[0], 1e-6);
        assertClose("static_bidirectional_rnn hFW final", expHFWfinal, out[1], 1e-6);
        assertClose("static_bidirectional_rnn hBW final", expHBWfinal, out[2], 1e-6);
    }
}
