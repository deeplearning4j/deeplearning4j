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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * sru_bi and sru_bi_bp, the bidirectional Simple Recurrent Unit (Lei et al., arXiv:1709.02755), against its
 * definition: for the features of x [time, bS, 2K] (the second K of them run backwards in time), with the input
 * x' = x * mask and U = x' * W [time, bS, 6K] holding feature j's candidate, forget and reset pre-activations at
 * columns 3j, 3j + 1 and 3j + 2,
 * <pre>
 *   f = sigmoid(U1 + bf), r = sigmoid(U2 + br), c = f * c_prev + (1 - f) * U0, h = r * (mask * tanh(c) - x') + x'
 * </pre>
 * The helpers indexed U = mmul(x, W), which comes out in F order, as if it were in C order, multiplied the mask into
 * the caller's x, summed the biases' gradients over the wrong rows and left the input gradient without the part that
 * flows through U. The backward pass is checked against numerical gradients of the forward pass. The mask is always
 * given (the Java wrapper checks the declared input counts): an all-ones mask stands for the unmasked cell.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class SruBiTest extends BaseNd4jTestWithBackends {

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double[][][] random3(Random rng, int a, int b, int c, double scale) {
        double[][][] values = new double[a][b][c];
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                for (int k = 0; k < c; k++)
                    values[i][j][k] = rng.nextGaussian() * scale;
        return values;
    }

    private static double[][] random2(Random rng, int a, int b, double scale) {
        double[][] values = new double[a][b];
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                values[i][j] = rng.nextGaussian() * scale;
        return values;
    }

    private static double[] random1(Random rng, int a, double scale) {
        double[] values = new double[a];
        for (int i = 0; i < a; i++)
            values[i] = rng.nextGaussian() * scale;
        return values;
    }

    /** A dropout mask: a feature is kept, scaled up, or dropped. */
    private static double[][] dropoutMask(Random rng, int bS, int d2) {
        double[][] mask = new double[bS][d2];
        for (int b = 0; b < bS; b++)
            for (int j = 0; j < d2; j++)
                mask[b][j] = rng.nextDouble() < 0.7 ? 1.0 / 0.7 : 0.0;
        return mask;
    }

    /** The inputs of one sru_bi call, and the gradients flowing into it for the backward pass. */
    private static final class Problem {
        final INDArray x, w, b, c0, mask, gradH, gradC;
        final int time, bS, k;

        Problem(int time, int bS, int k, boolean masked, long seed) {
            Random rng = new Random(seed);
            this.time = time;
            this.bS = bS;
            this.k = k;
            x = Nd4j.createFromArray(random3(rng, time, bS, 2 * k, 1.0));
            w = Nd4j.createFromArray(random2(rng, 2 * k, 6 * k, 0.5));
            b = Nd4j.createFromArray(random1(rng, 4 * k, 0.3));
            c0 = Nd4j.createFromArray(random2(rng, bS, 2 * k, 1.0));
            // the Java op wrapper requires the mask input (the op declares five and eight inputs): an all-ones mask is
            // the unmasked cell
            mask = masked ? Nd4j.createFromArray(dropoutMask(rng, bS, 2 * k)) : Nd4j.ones(DataType.DOUBLE, bS, 2 * k);
            gradH = Nd4j.createFromArray(random3(rng, time, bS, 2 * k, 1.0));
            gradC = Nd4j.createFromArray(random2(rng, bS, 2 * k, 1.0));
        }

        Problem(int time, int bS, int k, INDArray x, INDArray w, INDArray b, INDArray c0, INDArray mask,
                INDArray gradH, INDArray gradC) {
            this.time = time;
            this.bS = bS;
            this.k = k;
            this.x = x;
            this.w = w;
            this.b = b;
            this.c0 = c0;
            this.mask = mask;
            this.gradH = gradH;
            this.gradC = gradC;
        }
    }

    private static INDArray[] forwardOp(INDArray x, INDArray w, INDArray b, INDArray c0, INDArray mask) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(x);
        inputs.add(w);
        inputs.add(b);
        inputs.add(c0);
        if (mask != null)
            inputs.add(mask);
        return Nd4j.exec(DynamicCustomOp.builder("sru_bi").addInputs(inputs.toArray(new INDArray[0])).build());
    }

    private static INDArray[] backwardOp(Problem p, INDArray ct) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(p.x);
        inputs.add(p.w);
        inputs.add(p.b);
        inputs.add(p.c0);
        inputs.add(ct);
        inputs.add(p.gradC);
        inputs.add(p.gradH);
        if (p.mask != null)
            inputs.add(p.mask);
        return Nd4j.exec(DynamicCustomOp.builder("sru_bi_bp").addInputs(inputs.toArray(new INDArray[0])).build());
    }

    /** h and c of the definition, indexed [time][batch][feature]. */
    private static double[][][][] reference(Problem p) {
        int time = p.time, bS = p.bS, d2 = 2 * p.k;
        double[][][] xm = new double[time][bS][d2];
        double[][][] u = new double[time][bS][3 * d2];
        for (int t = 0; t < time; t++)
            for (int b = 0; b < bS; b++)
                for (int i = 0; i < d2; i++)
                    xm[t][b][i] = p.x.getDouble(t, b, i) * (p.mask == null ? 1.0 : p.mask.getDouble(b, i));
        for (int t = 0; t < time; t++)
            for (int b = 0; b < bS; b++)
                for (int o = 0; o < 3 * d2; o++) {
                    double sum = 0;
                    for (int i = 0; i < d2; i++)
                        sum += xm[t][b][i] * p.w.getDouble(i, o);
                    u[t][b][o] = sum;
                }

        double[][][] ht = new double[time][bS][d2];
        double[][][] ct = new double[time][bS][d2];
        for (int b = 0; b < bS; b++) {
            for (int j = 0; j < d2; j++) {
                boolean backwards = j >= p.k;
                double m = p.mask == null ? 1.0 : p.mask.getDouble(b, j);
                double cur = p.c0.getDouble(b, j);
                for (int s = 0; s < time; s++) {
                    int t = backwards ? time - 1 - s : s;
                    double f = sigmoid(u[t][b][3 * j + 1] + p.b.getDouble(j));
                    double r = sigmoid(u[t][b][3 * j + 2] + p.b.getDouble(j + d2));
                    cur = f * cur + (1 - f) * u[t][b][3 * j];
                    ct[t][b][j] = cur;
                    ht[t][b][j] = r * (m * Math.tanh(cur) - xm[t][b][j]) + xm[t][b][j];
                }
            }
        }
        return new double[][][][]{ht, ct};
    }

    /** The loss the backward pass differentiates: its gradient with respect to h is gradH, and with respect to the
     *  state each feature ends in gradC. */
    private static double loss(Problem p) {
        INDArray[] out = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
        double loss = 0;
        for (int t = 0; t < p.time; t++)
            for (int b = 0; b < p.bS; b++)
                for (int j = 0; j < 2 * p.k; j++) {
                    loss += p.gradH.getDouble(t, b, j) * out[0].getDouble(t, b, j);
                    int last = j >= p.k ? 0 : p.time - 1;
                    if (t == last)
                        loss += p.gradC.getDouble(b, j) * out[1].getDouble(t, b, j);
                }
        return loss;
    }

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    private static double[] logical(double[][][] array) {
        int a = array.length, b = array[0].length, c = array[0][0].length;
        double[] flat = new double[a * b * c];
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                for (int k = 0; k < c; k++)
                    flat[(i * b + j) * c + k] = array[i][j][k];
        return flat;
    }

    /** The derivative of the loss with respect to every element of an input, by central differences. */
    private static double[] numericalGradient(Problem p, INDArray input) {
        double eps = 1e-6;
        long[] shape = input.shape();
        double[] gradient = new double[(int) input.length()];
        long[] index = new long[shape.length];
        for (int e = 0; e < gradient.length; e++) {
            long rest = e;
            for (int d = shape.length - 1; d >= 0; d--) {
                index[d] = rest % shape[d];
                rest /= shape[d];
            }
            double original = input.getDouble(index);
            input.putScalar(index, original + eps);
            double up = loss(p);
            input.putScalar(index, original - eps);
            double down = loss(p);
            input.putScalar(index, original);
            gradient[e] = (up - down) / (2 * eps);
        }
        return gradient;
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void oddFeaturesAndNonFloatingInputsAreRejected(Nd4jBackend backend) {
        Problem p = new Problem(1, 1, 1, false, 7);
        assertThrows(RuntimeException.class, () -> forwardOp(p.x.castTo(DataType.INT32),
                p.w.castTo(DataType.INT32), p.b.castTo(DataType.INT32), p.c0.castTo(DataType.INT32),
                p.mask.castTo(DataType.INT32)));
        assertThrows(RuntimeException.class, () -> forwardOp(Nd4j.ones(DataType.DOUBLE, 1, 1, 3),
                p.w, p.b, p.c0, p.mask));
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void singletonCellHasTheClosedFormForwardAndGradient(Nd4jBackend backend) {
        Problem p = new Problem(1, 1, 1,
                Nd4j.createFromArray(2.0, -4.0).reshape(1, 1, 2),
                Nd4j.zeros(DataType.DOUBLE, 2, 6), Nd4j.zeros(DataType.DOUBLE, 4),
                Nd4j.zeros(DataType.DOUBLE, 1, 2), Nd4j.ones(DataType.DOUBLE, 1, 2),
                Nd4j.ones(DataType.DOUBLE, 1, 1, 2), Nd4j.ones(DataType.DOUBLE, 1, 2));
        INDArray[] fwd = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
        // Zero weights/bias/state give f=r=1/2, c=0 and h=x/2 for both directions.
        assertArrayEquals(new double[] {1, -2}, logical(fwd[0]), 0.0);
        assertArrayEquals(new double[] {0, 0}, logical(fwd[1]), 0.0);
        INDArray[] grads = backwardOp(p, fwd[1]);
        assertArrayEquals(new double[] {0.5, 0.5}, logical(grads[0]), 0.0);
        assertArrayEquals(new double[] {1.5, 0, -1, 1.5, 0, 2, -3, 0, 2, -3, 0, -4}, logical(grads[1]), 0.0);
        assertArrayEquals(new double[] {0, 0, -0.5, 1}, logical(grads[2]), 0.0);
        assertArrayEquals(new double[] {0.75, 0.75}, logical(grads[3]), 0.0);
        assertArrayEquals(new double[] {2, -4}, logical(p.x), 0.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardFollowsTheDefinitionAndLeavesItsInputs(Nd4jBackend backend) {
        int[][] sizes = {{1, 1, 1}, {4, 3, 2}, {5, 2, 3}, {3, 1, 4}};
        for (int[] size : sizes) {
            for (boolean masked : new boolean[]{false, true}) {
                Problem p = new Problem(size[0], size[1], size[2], masked, 100 + size[0] * 7 + size[2]);
                INDArray xBefore = p.x.dup();
                long[] xShape = p.x.shape();

                INDArray[] out = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
                double[][][][] expected = reference(p);
                String where = "time " + size[0] + ", batch " + size[1] + ", K " + size[2] + (masked ? ", masked" : "");
                assertArrayEquals(logical(expected[0]), logical(out[0]), 1e-10, "h, " + where);
                assertArrayEquals(logical(expected[1]), logical(out[1]), 1e-10, "c, " + where);

                // the input is the caller's: the mask used to be multiplied into it
                assertArrayEquals(xShape, p.x.shape(), where);
                assertArrayEquals(logical(xBefore), logical(p.x), 0.0, "x after the call, " + where);
                INDArray[] again = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
                assertArrayEquals(logical(out[0]), logical(again[0]), 0.0, "h of a second call, " + where);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardMatchesNumericalGradients(Nd4jBackend backend) {
        int[][] sizes = {{1, 1, 1}, {3, 2, 2}, {4, 3, 3}, {2, 4, 1}};
        for (int[] size : sizes) {
            for (boolean masked : new boolean[]{false, true}) {
                Problem p = new Problem(size[0], size[1], size[2], masked, 200 + size[0] * 5 + size[1]);
                String where = "time " + size[0] + ", batch " + size[1] + ", K " + size[2] + (masked ? ", masked" : "");

                INDArray[] fwd = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
                INDArray xBefore = p.x.dup();
                long[] xShape = p.x.shape();
                INDArray[] grads = backwardOp(p, fwd[1]);

                // gradients [gradI, gradW (per time step), gradB, gradC0]
                assertArrayEquals(new long[]{size[0], size[1], 2 * size[2]}, grads[0].shape(), where);
                assertArrayEquals(new long[]{size[0], 2 * size[2], 6 * size[2]}, grads[1].shape(), where);
                assertArrayEquals(new long[]{4 * size[2]}, grads[2].shape(), where);
                assertArrayEquals(new long[]{size[1], 2 * size[2]}, grads[3].shape(), where);

                assertArrayEquals(numericalGradient(p, p.x), logical(grads[0]), 1e-6, "gradient of x, " + where);
                // the weights are shared by every time step: their gradient is the sum of the per-step ones
                assertArrayEquals(numericalGradient(p, p.w), logical(grads[1].sum(0)), 1e-6,
                        "gradient of the weights, " + where);
                assertArrayEquals(numericalGradient(p, p.b), logical(grads[2]), 1e-6, "gradient of the biases, " + where);
                assertArrayEquals(numericalGradient(p, p.c0), logical(grads[3]), 1e-6,
                        "gradient of the initial state, " + where);

                // the call left x as it was, shape included
                assertArrayEquals(xShape, p.x.shape(), where);
                assertArrayEquals(logical(xBefore), logical(p.x), 0.0, "x after the backward pass, " + where);
            }
        }
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
                INDArray view = Nd4j.zeros(DataType.DOUBLE, big).addi(-7.0).get(steps);
                view.assign(c);
                return view;
            }
            default:
                return c.dup('c');
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyLayoutGivesTheCOrderResults(Nd4jBackend backend) {
        Problem c = new Problem(4, 3, 2, true, 321);
        INDArray[] fwd = forwardOp(c.x, c.w, c.b, c.c0, c.mask);
        INDArray[] grads = backwardOp(c, fwd[1]);

        for (int variant = 1; variant <= 2; variant++) {
            String where = variant == 1 ? "F order" : "stepped views";
            Problem p = new Problem(c.time, c.bS, c.k, layout(c.x, variant), layout(c.w, variant), layout(c.b, variant),
                    layout(c.c0, variant), layout(c.mask, variant), layout(c.gradH, variant), layout(c.gradC, variant));
            INDArray[] out = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
            assertArrayEquals(logical(fwd[0]), logical(out[0]), 1e-12, "h, " + where);
            assertArrayEquals(logical(fwd[1]), logical(out[1]), 1e-12, "c, " + where);

            INDArray[] g = backwardOp(p, layout(fwd[1], variant));
            for (int i = 0; i < 4; i++)
                assertArrayEquals(logical(grads[i]), logical(g[i]), 1e-12, "gradient " + i + ", " + where);
        }

        // and with only the input laid out otherwise, with a transposed weight matrix
        INDArray wT = layout(c.w.transpose().dup('c'), 0).transpose();
        Problem p = new Problem(c.time, c.bS, c.k, layout(c.x, 1), wT, c.b, c.c0, c.mask, c.gradH, c.gradC);
        INDArray[] out = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
        assertArrayEquals(logical(fwd[0]), logical(out[0]), 1e-12, "h, F-order x and a transposed w");
        INDArray[] g = backwardOp(p, fwd[1]);
        for (int i = 0; i < 4; i++)
            assertArrayEquals(logical(grads[i]), logical(g[i]), 1e-12, "gradient " + i + ", F-order x, transposed w");
    }

    /** A batch is its rows computed alone: the gradients of the shared weights and biases are the sums of theirs. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void largeBatchesEqualTheirSingleRows(Nd4jBackend backend) {
        // bS * 2K = 280 threads: more than one block of 128 or 256 would not cover with a single launch of that size
        Problem p = new Problem(3, 70, 2, true, 555);
        INDArray[] fwd = forwardOp(p.x, p.w, p.b, p.c0, p.mask);
        INDArray[] grads = backwardOp(p, fwd[1]);
        double[][][][] expected = reference(p);
        assertArrayEquals(logical(expected[0]), logical(fwd[0]), 1e-10, "h of the whole batch");
        assertArrayEquals(logical(expected[1]), logical(fwd[1]), 1e-10, "c of the whole batch");

        INDArray gradBSum = Nd4j.zeros(DataType.DOUBLE, 4 * p.k);
        INDArray gradWSum = Nd4j.zeros(DataType.DOUBLE, p.time, 2 * p.k, 6 * p.k);
        for (int b = 0; b < p.bS; b++) {
            INDArrayIndex row = NDArrayIndex.interval(b, b + 1);
            Problem single = new Problem(p.time, 1, p.k,
                    p.x.get(NDArrayIndex.all(), row, NDArrayIndex.all()).dup('c'), p.w, p.b,
                    p.c0.get(row, NDArrayIndex.all()).dup('c'), p.mask.get(row, NDArrayIndex.all()).dup('c'),
                    p.gradH.get(NDArrayIndex.all(), row, NDArrayIndex.all()).dup('c'),
                    p.gradC.get(row, NDArrayIndex.all()).dup('c'));
            INDArray[] out = forwardOp(single.x, single.w, single.b, single.c0, single.mask);
            assertArrayEquals(logical(out[0]), logical(fwd[0].get(NDArrayIndex.all(), row, NDArrayIndex.all())), 1e-12,
                    "h of row " + b);
            assertArrayEquals(logical(out[1]), logical(fwd[1].get(NDArrayIndex.all(), row, NDArrayIndex.all())), 1e-12,
                    "c of row " + b);

            INDArray[] g = backwardOp(single, out[1]);
            assertArrayEquals(logical(g[0]), logical(grads[0].get(NDArrayIndex.all(), row, NDArrayIndex.all())), 1e-12,
                    "gradient of x, row " + b);
            assertArrayEquals(logical(g[3]), logical(grads[3].get(row, NDArrayIndex.all())), 1e-12,
                    "gradient of c0, row " + b);
            gradWSum.addi(g[1]);
            gradBSum.addi(g[2]);
        }
        assertArrayEquals(logical(gradWSum), logical(grads[1]), 1e-9, "gradient of the weights");
        assertArrayEquals(logical(gradBSum), logical(grads[2]), 1e-9, "gradient of the biases");
    }
}
