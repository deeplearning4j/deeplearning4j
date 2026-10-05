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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRmsNormSwiGLU;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRmsNormSwiGLUBp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * fused_rms_norm_swiglu and its backprop, y = silu(rms_norm(x) * gamma @ wGate) * (rms_norm(x) * gamma @ wUp), against
 * a double-precision reference, on every hidden size, row count, operand layout and float type. The backprop was a stub
 * that threw on both backends ("Full kernel not yet implemented" on CUDA, "CPU backward not yet implemented" on CPU), so
 * training through the op was impossible. It is checked three ways: against the closed-form reference, against central
 * finite differences of the forward op, and through SameDiff's gradient.
 *
 * <p>The forward pass shares the bugs the fused layer norm had: both backends read the input and the gamma as dense rows
 * (CUDA rows of any layout but dense row-major, the CPU's loops the same), the CPU summed in float whatever the type, and
 * the shape function of the backprop gave each gradient the cached shape of the array it is the gradient of, strides
 * included, so the gradient of a stepped view addressed several times the elements its buffer holds. The gradients are
 * dense arrays of their own now, in the type of the array they belong to, and caller-provided outputs of any layout,
 * offset or type are written in place.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class FusedRmsNormSwigluTest extends BaseNd4jTestWithBackends {

    private static final float EPSILON = 1e-5f;
    /** exactly representable in every type, so untouched elements compare exactly */
    private static final double SENTINEL = 7.0;

    /** Values of the given type, uniform in [lo, hi): generated in FLOAT (DOUBLE for DOUBLE) and rounded to the type. */
    private static INDArray random(DataType type, long seed, double lo, double hi, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        DataType generated = type == DataType.DOUBLE ? DataType.DOUBLE : DataType.FLOAT;
        INDArray values = Nd4j.rand(generated, shape).muli(hi - lo).addi(lo);
        return values.dataType() == type ? values : values.castTo(type);
    }

    /** Activations [batch, seq, hidden], dense C order. */
    private static INDArray activations(DataType type, long seed, long batch, long seq, long hidden) {
        return random(type, seed, -2, 3, batch, seq, hidden);
    }

    private static INDArray gammaOf(DataType type, long seed, long hidden) {
        return random(type, seed, 0.5, 1.5, hidden);
    }

    /** The weights' range: the products of a unit-scale row with them are of a scale of 2, whatever the hidden size. */
    private static double scaleOf(long hidden) {
        return 3.0 / Math.sqrt(hidden);
    }

    /** A projection [hidden, inter] whose products with unit-scale rows are of a scale of 2. */
    private static INDArray weights(DataType type, long seed, long hidden, long inter) {
        double scale = scaleOf(hidden);
        return random(type, seed, -scale, scale, hidden, inter);
    }

    private static INDArray gradientOf(DataType type, long seed, long batch, long seq, long inter) {
        return random(type, seed, -1, 1, batch, seq, inter);
    }

    /** [length], every second element of a [2 * length] vector. */
    private static INDArray everySecond(DataType type, long seed, double lo, double hi, long length) {
        return random(type, seed, lo, hi, 2 * length).get(NDArrayIndex.interval(0, 2, 2 * length));
    }

    /** [rows, cols], every second column of a [rows, 2 * cols] array. */
    private static INDArray everySecondColumn(DataType type, long seed, double lo, double hi, long rows, long cols) {
        return random(type, seed, lo, hi, rows, 2 * cols).get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 2 * cols));
    }

    /** [batch, seq, cols], every second element of the last dimension of a [batch, seq, 2 * cols] array. */
    private static INDArray everySecondLast(DataType type, long seed, double lo, double hi, long batch, long seq,
                                            long cols) {
        return random(type, seed, lo, hi, batch, seq, 2 * cols).get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 2, 2 * cols));
    }

    /** The reference: [0] the output, [1] dx, [2] dgamma, [3] dwGate, [4] dwUp (the last four only with dy), flat. */
    private static double[][] reference(INDArray x, INDArray gamma, INDArray wGate, INDArray wUp, INDArray dy) {
        int hidden = (int) x.size(2);
        int rows = (int) (x.size(0) * x.size(1));
        int inter = (int) wGate.size(1);
        double[] xs = x.dup('c').data().asDouble();
        double[] gs = gamma.dup('c').data().asDouble();
        double[] wg = wGate.dup('c').data().asDouble();
        double[] wu = wUp.dup('c').data().asDouble();
        double[] dys = dy == null ? null : dy.dup('c').data().asDouble();
        double[] out = new double[rows * inter];
        double[] dx = new double[rows * hidden];
        double[] dgamma = new double[hidden];
        double[] dwg = new double[hidden * inter];
        double[] dwu = new double[hidden * inter];
        double epsilon = EPSILON;
        for (int r = 0; r < rows; r++) {
            double sumSq = 0;
            for (int i = 0; i < hidden; i++) sumSq += xs[r * hidden + i] * xs[r * hidden + i];
            double inverse = 1.0 / Math.sqrt(sumSq / hidden + epsilon);
            double[] normalized = new double[hidden];
            for (int i = 0; i < hidden; i++) normalized[i] = xs[r * hidden + i] * inverse * gs[i];
            double[] gate = new double[inter];
            double[] up = new double[inter];
            for (int j = 0; j < inter; j++) {
                double g = 0;
                double u = 0;
                for (int i = 0; i < hidden; i++) {
                    g += normalized[i] * wg[i * inter + j];
                    u += normalized[i] * wu[i * inter + j];
                }
                gate[j] = g;
                up[j] = u;
                double sigmoid = 1.0 / (1.0 + Math.exp(-g));
                out[r * inter + j] = g * sigmoid * u;
            }
            if (dys == null) continue;
            double[] dNormalized = new double[hidden];
            for (int j = 0; j < inter; j++) {
                double sigmoid = 1.0 / (1.0 + Math.exp(-gate[j]));
                double dy1 = dys[r * inter + j];
                double dUp = dy1 * gate[j] * sigmoid;
                double dGate = dy1 * up[j] * sigmoid * (1.0 + gate[j] * (1.0 - sigmoid));
                for (int i = 0; i < hidden; i++) {
                    dwg[i * inter + j] += normalized[i] * dGate;
                    dwu[i * inter + j] += normalized[i] * dUp;
                    dNormalized[i] += dGate * wg[i * inter + j] + dUp * wu[i * inter + j];
                }
            }
            double dot = 0;
            for (int i = 0; i < hidden; i++) dot += dNormalized[i] * gs[i] * xs[r * hidden + i];
            for (int i = 0; i < hidden; i++) {
                double xi = xs[r * hidden + i];
                dgamma[i] += dNormalized[i] * xi * inverse;
                dx[r * hidden + i] = inverse * (dNormalized[i] * gs[i] - xi * inverse * inverse * dot / hidden);
            }
        }
        return new double[][]{out, dx, dgamma, dwg, dwu};
    }

    private static void assertClose(String label, double[] expected, INDArray actual, double tolerance) {
        double[] values = actual.dup('c').data().asDouble();
        assertEquals(expected.length, values.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], tolerance * Math.max(1.0, Math.abs(expected[i])), label + " at " + i);
        }
    }

    private static double tolerance(DataType type) {
        return type == DataType.DOUBLE ? 1e-10 : 1e-4;
    }

    /**
     * The forward output is rounded to the type a few times (the normalized rows, the gate, the up projection, the
     * output): a few units in the last place of 11 or 8 bits, on the largest values of a few hundred.
     */
    private static double lowPrecisionForwardTolerance(DataType type) {
        return type == DataType.FLOAT16 ? 1e-2 : 6e-2;
    }

    /** The gradients are computed in float and rounded once: half a unit in the last place of 11 or 8 bits, four times. */
    private static double lowPrecisionBackwardTolerance(DataType type) {
        return type == DataType.FLOAT16 ? 2e-3 : 2e-2;
    }

    private static void assertForward(String label, INDArray x, INDArray gamma, INDArray wGate, INDArray wUp,
                                      double tolerance) {
        INDArray out = Nd4j.exec(new FusedRmsNormSwiGLU(x, gamma, wGate, wUp, EPSILON))[0];
        assertEquals(Arrays.toString(new long[]{x.size(0), x.size(1), wGate.size(1)}), Arrays.toString(out.shape()),
                label + " shape");
        assertEquals(x.dataType(), out.dataType(), label + " type");
        assertClose(label, reference(x, gamma, wGate, wUp, null)[0], out, tolerance);
    }

    private static DynamicCustomOp backwardOp(INDArray x, INDArray gamma, INDArray wGate, INDArray wUp, INDArray dy) {
        return DynamicCustomOp.builder("fused_rms_norm_swiglu_bp")
                .addFloatingPointArguments((double) EPSILON)
                .addInputs(x, gamma, wGate, wUp, dy)
                .build();
    }

    private static void assertBackward(String label, INDArray x, INDArray gamma, INDArray wGate, INDArray wUp,
                                       INDArray dy, double tolerance) {
        INDArray[] grads = Nd4j.exec(backwardOp(x, gamma, wGate, wUp, dy));
        assertEquals(4, grads.length, label + " gradients");
        double[][] expected = reference(x, gamma, wGate, wUp, dy);
        assertEquals(Arrays.toString(x.shape()), Arrays.toString(grads[0].shape()), label + " dx shape");
        assertEquals(Arrays.toString(gamma.shape()), Arrays.toString(grads[1].shape()), label + " dgamma shape");
        assertEquals(Arrays.toString(wGate.shape()), Arrays.toString(grads[2].shape()), label + " dwGate shape");
        assertEquals(Arrays.toString(wUp.shape()), Arrays.toString(grads[3].shape()), label + " dwUp shape");
        assertClose(label + " dx", expected[1], grads[0], tolerance);
        assertClose(label + " dgamma", expected[2], grads[1], tolerance);
        assertClose(label + " dwGate", expected[3], grads[2], tolerance);
        assertClose(label + " dwUp", expected[4], grads[3], tolerance);
    }

    /** The output shape the native shape function reports addresses exactly the elements it holds. */
    private static void assertDenseShape(String label, DataBuffer shapeInfo) {
        long[] info = shapeInfo.asLong();
        long[] shape = Shape.shape(info);
        long[] strides = Shape.stride(info);
        long length = 1;
        long farthest = 0;
        for (int i = 0; i < shape.length; i++) {
            length *= shape[i];
            farthest += (shape[i] - 1) * strides[i];
        }
        assertEquals(length - 1, farthest, label + ": shape " + Arrays.toString(shape) + " strides "
                + Arrays.toString(strides) + " must address exactly " + length + " elements");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardForEveryHiddenSize(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int hidden : new int[]{1, 31, 70, 96, 200, 256, 257, 1024, 1500}) {
                assertForward(type + " hidden " + hidden, activations(type, hidden, 2, 3, hidden),
                        gammaOf(type, hidden + 1, hidden), weights(type, hidden + 2, hidden, 5),
                        weights(type, hidden + 3, hidden, 5), tolerance(type));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardForEveryRowCount(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int rows : new int[]{1, 2, 31, 256, 257, 1500}) {
                assertForward(type + " rows " + rows, activations(type, rows, 1, rows, 33),
                        gammaOf(type, rows + 1, 33), weights(type, rows + 2, 33, 7), weights(type, rows + 3, 33, 7),
                        tolerance(type));
            }
            // a batch of single-position sequences, and one long sequence
            assertForward(type + " 1500 x 1", activations(type, 41, 1500, 1, 33), gammaOf(type, 42, 33),
                    weights(type, 43, 33, 7), weights(type, 44, 33, 7), tolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardForEveryLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            double scale = scaleOf(70);
            INDArray gamma = gammaOf(type, 51, 70);
            INDArray wGate = weights(type, 52, 70, 5);
            INDArray wUp = weights(type, 53, 70, 5);
            // activations whose rows are not one run apart: F order, a fully permuted view, permuted rows, every
            // second element of the last dimension, a nonzero base offset
            assertForward(type + " F order", activations(type, 54, 2, 3, 70).dup('f'), gamma, wGate, wUp, tolerance);
            assertForward(type + " fully permuted view", random(type, 55, -2, 3, 70, 3, 2).permute(2, 1, 0), gamma,
                    wGate, wUp, tolerance);
            assertForward(type + " permuted rows", random(type, 56, -2, 3, 3, 2, 70).permute(1, 0, 2), gamma, wGate,
                    wUp, tolerance);
            assertForward(type + " every second element of the last dimension", everySecondLast(type, 57, -2, 3, 2, 3, 70),
                    gamma, wGate, wUp, tolerance);
            assertForward(type + " offset rows",
                    random(type, 58, -2, 3, 4, 3, 70).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                            NDArrayIndex.all()), gamma, wGate, wUp, tolerance);
            // the gamma and the projections in other layouts
            INDArray x = activations(type, 59, 2, 3, 70);
            assertForward(type + " every second element of the gamma", x, everySecond(type, 60, 0.5, 1.5, 70), wGate, wUp,
                    tolerance);
            assertForward(type + " F order projections", x, gamma, wGate.dup('f'), wUp.dup('f'), tolerance);
            assertForward(type + " transposed projections", x, gamma, random(type, 61, -scale, scale, 5, 70).transpose(),
                    random(type, 62, -scale, scale, 5, 70).transpose(), tolerance);
            assertForward(type + " every second column of the projections", x, gamma,
                    everySecondColumn(type, 63, -scale, scale, 70, 5), everySecondColumn(type, 64, -scale, scale, 70, 5),
                    tolerance);
            assertForward(type + " offset projections", x, gamma,
                    random(type, 65, -scale, scale, 72, 5).get(NDArrayIndex.interval(1, 71), NDArrayIndex.all()),
                    random(type, 66, -scale, scale, 72, 5).get(NDArrayIndex.interval(1, 71), NDArrayIndex.all()),
                    tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardWithAGammaAndProjectionsOfAnotherType(Nd4jBackend backend) {
        // half weights with float activations (the op takes any float type for each input)
        assertForward("FLOAT input, HALF gamma and projections", activations(DataType.FLOAT, 71, 2, 3, 300),
                random(DataType.FLOAT16, 72, 0.5, 1.5, 300), weights(DataType.FLOAT16, 73, 300, 5),
                weights(DataType.FLOAT16, 74, 300, 5), 1e-4);
        assertForward("DOUBLE input, FLOAT gamma and projections", activations(DataType.DOUBLE, 75, 2, 3, 300),
                random(DataType.FLOAT, 76, 0.5, 1.5, 300), weights(DataType.FLOAT, 77, 300, 5),
                weights(DataType.FLOAT, 78, 300, 5), 1e-10);
        assertForward("FLOAT input, BFLOAT16 projections", activations(DataType.FLOAT, 79, 2, 3, 300),
                gammaOf(DataType.FLOAT, 80, 300), weights(DataType.BFLOAT16, 81, 300, 5),
                weights(DataType.BFLOAT16, 82, 300, 5), 1e-4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardOfHalfAndBfloat16(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT16, DataType.BFLOAT16}) {
            double tolerance = lowPrecisionForwardTolerance(type);
            for (int hidden : new int[]{1, 31, 70, 300}) {
                assertForward(type + " hidden " + hidden, activations(type, 91 + hidden, 2, 3, hidden),
                        gammaOf(type, 92 + hidden, hidden), weights(type, 93 + hidden, hidden, 5),
                        weights(type, 94 + hidden, hidden, 5), tolerance);
            }
            assertForward(type + " every second element of the last dimension", everySecondLast(type, 95, -2, 3, 2, 3, 70),
                    gammaOf(type, 96, 70), weights(type, 97, 70, 5), weights(type, 98, 70, 5), tolerance);
            assertForward(type + " F order", activations(type, 99, 2, 3, 70).dup('f'), gammaOf(type, 100, 70),
                    weights(type, 101, 70, 5), weights(type, 102, 70, 5), tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardForEveryHiddenSize(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int hidden : new int[]{1, 31, 70, 96, 200, 256, 257, 1024, 1500}) {
                assertBackward(type + " hidden " + hidden, activations(type, hidden, 2, 3, hidden),
                        gammaOf(type, hidden + 1, hidden), weights(type, hidden + 2, hidden, 5),
                        weights(type, hidden + 3, hidden, 5), gradientOf(type, hidden + 4, 2, 3, 5), tolerance(type));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardForEveryRowCount(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int rows : new int[]{1, 2, 31, 256, 257, 1500}) {
                assertBackward(type + " rows " + rows, activations(type, rows, 1, rows, 33),
                        gammaOf(type, rows + 1, 33), weights(type, rows + 2, 33, 7), weights(type, rows + 3, 33, 7),
                        gradientOf(type, rows + 4, 1, rows, 7), tolerance(type));
            }
            assertBackward(type + " 1500 x 1", activations(type, 141, 1500, 1, 33), gammaOf(type, 142, 33),
                    weights(type, 143, 33, 7), weights(type, 144, 33, 7), gradientOf(type, 145, 1500, 1, 7),
                    tolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardForEveryLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            double scale = scaleOf(70);
            INDArray gamma = gammaOf(type, 151, 70);
            INDArray wGate = weights(type, 152, 70, 5);
            INDArray wUp = weights(type, 153, 70, 5);
            INDArray dy = gradientOf(type, 154, 2, 3, 5);
            assertBackward(type + " F order", activations(type, 155, 2, 3, 70).dup('f'), gamma, wGate, wUp,
                    gradientOf(type, 156, 2, 3, 5).dup('f'), tolerance);
            assertBackward(type + " fully permuted views", random(type, 157, -2, 3, 70, 3, 2).permute(2, 1, 0), gamma,
                    wGate, wUp, random(type, 158, -1, 1, 5, 3, 2).permute(2, 1, 0), tolerance);
            assertBackward(type + " permuted rows", random(type, 159, -2, 3, 3, 2, 70).permute(1, 0, 2), gamma, wGate,
                    wUp, random(type, 160, -1, 1, 3, 2, 5).permute(1, 0, 2), tolerance);
            assertBackward(type + " every second element of the last dimension of x and dy",
                    everySecondLast(type, 161, -2, 3, 2, 3, 70), gamma, wGate, wUp,
                    everySecondLast(type, 162, -1, 1, 2, 3, 5), tolerance);
            assertBackward(type + " every second element of the last dimension of x only",
                    everySecondLast(type, 163, -2, 3, 2, 3, 70), gamma, wGate, wUp, dy, tolerance);
            assertBackward(type + " offset rows", random(type, 164, -2, 3, 4, 3, 70).get(NDArrayIndex.interval(1, 3),
                    NDArrayIndex.all(), NDArrayIndex.all()), gamma, wGate, wUp,
                    random(type, 165, -1, 1, 4, 3, 5).get(NDArrayIndex.interval(2, 4), NDArrayIndex.all(),
                            NDArrayIndex.all()), tolerance);
            INDArray x = activations(type, 166, 2, 3, 70);
            assertBackward(type + " every second element of the gamma", x, everySecond(type, 167, 0.5, 1.5, 70), wGate,
                    wUp, dy, tolerance);
            assertBackward(type + " F order projections", x, gamma, wGate.dup('f'), wUp.dup('f'), dy, tolerance);
            assertBackward(type + " transposed projections", x, gamma,
                    random(type, 168, -scale, scale, 5, 70).transpose(),
                    random(type, 169, -scale, scale, 5, 70).transpose(), dy, tolerance);
            assertBackward(type + " every second column of the projections", x, gamma,
                    everySecondColumn(type, 170, -scale, scale, 70, 5), everySecondColumn(type, 171, -scale, scale, 70, 5),
                    dy, tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardOfHalfAndBfloat16(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT16, DataType.BFLOAT16}) {
            double tolerance = lowPrecisionBackwardTolerance(type);
            for (int hidden : new int[]{1, 31, 70, 300}) {
                assertBackward(type + " hidden " + hidden, activations(type, 181 + hidden, 2, 3, hidden),
                        gammaOf(type, 182 + hidden, hidden), weights(type, 183 + hidden, hidden, 5),
                        weights(type, 184 + hidden, hidden, 5), gradientOf(type, 185 + hidden, 2, 3, 5), tolerance);
            }
            assertBackward(type + " every second element of the last dimension",
                    everySecondLast(type, 186, -2, 3, 2, 3, 70), gammaOf(type, 187, 70), weights(type, 188, 70, 5),
                    weights(type, 189, 70, 5), everySecondLast(type, 190, -1, 1, 2, 3, 5), tolerance);
            // many rows: the gradients of the gamma and the projections sum over all of them
            assertBackward(type + " 300 rows", activations(type, 191, 1, 300, 31), gammaOf(type, 192, 31),
                    weights(type, 193, 31, 5), weights(type, 194, 31, 5), gradientOf(type, 195, 1, 300, 5), tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardWithAGammaAndProjectionsOfAnotherType(Nd4jBackend backend) {
        // half weights with float activations: dx is float, the gradients of the gamma and the projections are half
        INDArray x = activations(DataType.FLOAT, 201, 2, 3, 300);
        INDArray dy = gradientOf(DataType.FLOAT, 202, 2, 3, 5);
        INDArray gamma = random(DataType.FLOAT16, 203, 0.5, 1.5, 300);
        INDArray wGate = weights(DataType.FLOAT16, 204, 300, 5);
        INDArray wUp = weights(DataType.FLOAT16, 205, 300, 5);
        INDArray[] grads = Nd4j.exec(backwardOp(x, gamma, wGate, wUp, dy));
        assertEquals(4, grads.length, "gradients");
        assertEquals(DataType.FLOAT, grads[0].dataType(), "dx type");
        assertEquals(DataType.FLOAT16, grads[1].dataType(), "dgamma type");
        assertEquals(DataType.FLOAT16, grads[2].dataType(), "dwGate type");
        assertEquals(DataType.FLOAT16, grads[3].dataType(), "dwUp type");
        double[][] expected = reference(x, gamma, wGate, wUp, dy);
        assertClose("dx", expected[1], grads[0], 1e-4);
        assertClose("dgamma", expected[2], grads[1], lowPrecisionBackwardTolerance(DataType.FLOAT16));
        assertClose("dwGate", expected[3], grads[2], lowPrecisionBackwardTolerance(DataType.FLOAT16));
        assertClose("dwUp", expected[4], grads[3], lowPrecisionBackwardTolerance(DataType.FLOAT16));

        // float weights with double activations: the gradients of the weights are float
        INDArray xd = activations(DataType.DOUBLE, 206, 2, 3, 70);
        INDArray dyd = gradientOf(DataType.DOUBLE, 207, 2, 3, 5);
        INDArray gammaF = gammaOf(DataType.FLOAT, 208, 70);
        INDArray wGateF = weights(DataType.FLOAT, 209, 70, 5);
        INDArray wUpF = weights(DataType.FLOAT, 210, 70, 5);
        INDArray[] gradsD = Nd4j.exec(backwardOp(xd, gammaF, wGateF, wUpF, dyd));
        assertEquals(DataType.DOUBLE, gradsD[0].dataType(), "dx type");
        assertEquals(DataType.FLOAT, gradsD[1].dataType(), "dgamma type");
        assertEquals(DataType.FLOAT, gradsD[2].dataType(), "dwGate type");
        assertEquals(DataType.FLOAT, gradsD[3].dataType(), "dwUp type");
        double[][] expectedD = reference(xd, gammaF, wGateF, wUpF, dyd);
        assertClose("dx", expectedD[1], gradsD[0], 1e-10);
        assertClose("dgamma", expectedD[2], gradsD[1], 1e-4);
        assertClose("dwGate", expectedD[3], gradsD[2], 1e-4);
        assertClose("dwUp", expectedD[4], gradsD[3], 1e-4);
    }

    /**
     * An independent check of the closed-form gradients: central finite differences of the forward op itself, loss =
     * sum(forward * dy), against every element of every gradient.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardMatchesFiniteDifferencesOfTheForwardOp(Nd4jBackend backend) {
        INDArray x = random(DataType.DOUBLE, 301, -2, 3, 2, 3, 6);
        INDArray gamma = random(DataType.DOUBLE, 302, 0.5, 1.5, 6);
        INDArray wGate = random(DataType.DOUBLE, 303, -0.5, 0.5, 6, 5);
        INDArray wUp = random(DataType.DOUBLE, 304, -0.5, 0.5, 6, 5);
        INDArray dy = random(DataType.DOUBLE, 305, -1, 1, 2, 3, 5);
        INDArray[] grads = Nd4j.exec(backwardOp(x, gamma, wGate, wUp, dy));
        String[] names = {"dx", "dgamma", "dwGate", "dwUp"};
        INDArray[] parameters = {x, gamma, wGate, wUp};
        double step = 1e-6;
        for (int p = 0; p < parameters.length; p++) {
            INDArray parameter = parameters[p];
            double[] analytic = grads[p].dup('c').data().asDouble();
            assertEquals(parameter.length(), analytic.length, names[p] + " length");
            for (int i = 0; i < parameter.length(); i++) {
                double original = parameter.getDouble(i);
                parameter.putScalar(i, original + step);
                double plus = loss(x, gamma, wGate, wUp, dy);
                parameter.putScalar(i, original - step);
                double minus = loss(x, gamma, wGate, wUp, dy);
                parameter.putScalar(i, original);
                double numeric = (plus - minus) / (2 * step);
                assertEquals(numeric, analytic[i], 1e-6 * Math.max(1.0, Math.abs(numeric)), names[p] + " at " + i);
            }
        }
    }

    private static double loss(INDArray x, INDArray gamma, INDArray wGate, INDArray wUp, INDArray dy) {
        INDArray out = Nd4j.exec(new FusedRmsNormSwiGLU(x, gamma, wGate, wUp, EPSILON))[0];
        return out.mul(dy).sumNumber().doubleValue();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffGradientsReachEveryInput(Nd4jBackend backend) {
        INDArray xArr = random(DataType.DOUBLE, 311, -2, 3, 2, 3, 70);
        INDArray gammaArr = random(DataType.DOUBLE, 312, 0.5, 1.5, 70);
        INDArray gateArr = weights(DataType.DOUBLE, 313, 70, 5);
        INDArray upArr = weights(DataType.DOUBLE, 314, 70, 5);
        INDArray dyArr = random(DataType.DOUBLE, 315, -1, 1, 2, 3, 5);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);
        SDVariable gamma = sd.var("gamma", gammaArr);
        SDVariable wGate = sd.var("wGate", gateArr);
        SDVariable wUp = sd.var("wUp", upArr);
        SDVariable out = new FusedRmsNormSwiGLU(sd, x, gamma, wGate, wUp, EPSILON).outputVariable();
        // the loss sum(out * w) has dLoss/dout = w
        out.mul(sd.constant("w", dyArr)).sum("loss").markAsLoss();
        Map<String, INDArray> grads = sd.calculateGradients(null, "x", "gamma", "wGate", "wUp");

        double[][] expected = reference(xArr, gammaArr, gateArr, upArr, dyArr);
        assertClose("SameDiff dx", expected[1], grads.get("x"), 1e-10);
        assertClose("SameDiff dgamma", expected[2], grads.get("gamma"), 1e-10);
        assertClose("SameDiff dwGate", expected[3], grads.get("wGate"), 1e-10);
        assertClose("SameDiff dwUp", expected[4], grads.get("wUp"), 1e-10);
    }

    /** The backprop's gradients are declared in the type of the array each one is the gradient of. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffDeclaresEachGradientInItsInputsType(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", activations(DataType.FLOAT, 321, 2, 3, 70));
        SDVariable gamma = sd.var("gamma", random(DataType.FLOAT16, 322, 0.5, 1.5, 70));
        SDVariable wGate = sd.var("wGate", weights(DataType.BFLOAT16, 323, 70, 5));
        SDVariable wUp = sd.var("wUp", weights(DataType.FLOAT16, 324, 70, 5));
        SDVariable dy = sd.var("dy", gradientOf(DataType.FLOAT, 325, 2, 3, 5));
        SDVariable[] grads = new FusedRmsNormSwiGLUBp(sd, x, gamma, wGate, wUp, dy, EPSILON).outputVariables();
        assertEquals(4, grads.length, "gradients");
        assertEquals(DataType.FLOAT, grads[0].dataType(), "dx type");
        assertEquals(DataType.FLOAT16, grads[1].dataType(), "dgamma type");
        assertEquals(DataType.BFLOAT16, grads[2].dataType(), "dwGate type");
        assertEquals(DataType.FLOAT16, grads[3].dataType(), "dwUp type");
    }

    /**
     * The root cause of every wrong value and every write past the end of a buffer a stepped view gave: the output
     * arrays are allocated with one element per element of the shapes the shape function returns, so those shapes must
     * not keep the strides of the arrays they are shaped like.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void outputShapesAreDenseForEveryOperandLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double scale = scaleOf(70);
            INDArray[] inputs = {
                    activations(type, 401, 2, 3, 70),
                    activations(type, 402, 2, 3, 70).dup('f'),
                    random(type, 403, -2, 3, 70, 3, 2).permute(2, 1, 0),
                    random(type, 404, -2, 3, 3, 2, 70).permute(1, 0, 2),
                    everySecondLast(type, 405, -2, 3, 2, 3, 70),
                    random(type, 406, -2, 3, 4, 3, 70).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                            NDArrayIndex.all()),
            };
            INDArray[] gammas = {gammaOf(type, 411, 70), everySecond(type, 412, 0.5, 1.5, 70)};
            INDArray[][] projections = {
                    {weights(type, 413, 70, 5), weights(type, 414, 70, 5)},
                    {weights(type, 415, 70, 5).dup('f'), weights(type, 416, 70, 5).dup('f')},
                    {random(type, 417, -scale, scale, 5, 70).transpose(), random(type, 418, -scale, scale, 5, 70).transpose()},
                    {everySecondColumn(type, 419, -scale, scale, 70, 5), everySecondColumn(type, 420, -scale, scale, 70, 5)},
            };
            for (INDArray x : inputs) {
                for (INDArray gamma : gammas) {
                    for (INDArray[] projection : projections) {
                        String layout = type + " input " + Arrays.toString(x.shape()) + " strides "
                                + Arrays.toString(x.stride()) + ", gamma strides " + Arrays.toString(gamma.stride())
                                + ", projection strides " + Arrays.toString(projection[0].stride());
                        List<DataBuffer> forward = Nd4j.getExecutioner().calculateOutputShape(
                                new FusedRmsNormSwiGLU(x, gamma, projection[0], projection[1], EPSILON));
                        assertEquals(1, forward.size(), layout + " forward outputs");
                        assertDenseShape(layout + " forward output", forward.get(0));

                        INDArray dy = gradientOf(type, 421, 2, 3, 5);
                        List<DataBuffer> backward = Nd4j.getExecutioner()
                                .calculateOutputShape(backwardOp(x, gamma, projection[0], projection[1], dy));
                        assertEquals(4, backward.size(), layout + " backward outputs");
                        for (int i = 0; i < backward.size(); i++) {
                            assertDenseShape(layout + " backward output " + i, backward.get(i));
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void callerProvidedOutputsAreFilledInPlace(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            INDArray x = activations(type, 431, 2, 3, 70);
            INDArray gamma = gammaOf(type, 432, 70);
            INDArray wGate = weights(type, 433, 70, 5);
            INDArray wUp = weights(type, 434, 70, 5);
            INDArray dy = gradientOf(type, 435, 2, 3, 5);
            double[] expected = reference(x, gamma, wGate, wUp, null)[0];
            double[][] expectedBackward = reference(x, gamma, wGate, wUp, dy);

            // forward into every second element of the last dimension of a sentinel-filled array: the others keep it
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 10}, SENTINEL, type);
            Nd4j.exec(new FusedRmsNormSwiGLU(x, gamma, wGate, wUp,
                    stepped.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 10)), EPSILON));
            double[] flat = stepped.dup('c').data().asDouble();
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 5; c++) {
                    assertEquals(expected[r * 5 + c], flat[r * 10 + 2 * c],
                            tolerance * Math.max(1.0, Math.abs(expected[r * 5 + c])),
                            type + " stepped output at " + r + ", " + c);
                    assertEquals(SENTINEL, flat[r * 10 + 2 * c + 1], 0.0,
                            type + " stepped output's neighbor at " + r + ", " + c);
                }
            }

            // forward into the middle batch entry of a sentinel-filled array (an offset view): the others keep it
            INDArray offset = Nd4j.valueArrayOf(new long[]{4, 3, 5}, SENTINEL, type);
            Nd4j.exec(new FusedRmsNormSwiGLU(x.get(NDArrayIndex.interval(0, 2), NDArrayIndex.all(), NDArrayIndex.all()),
                    gamma, wGate, wUp,
                    offset.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(), NDArrayIndex.all()), EPSILON));
            flat = offset.dup('c').data().asDouble();
            for (int i = 0; i < 4 * 3 * 5; i++) {
                if (i >= 15 && i < 45) {
                    double expectedValue = expected[i - 15];
                    assertEquals(expectedValue, flat[i], tolerance * Math.max(1.0, Math.abs(expectedValue)),
                            type + " offset output at " + i);
                } else {
                    assertEquals(SENTINEL, flat[i], 0.0, type + " offset output's surroundings at " + i);
                }
            }

            // forward into arrays of other float types: the op takes any float type for its output
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 5);
            Nd4j.exec(new FusedRmsNormSwiGLU(x, gamma, wGate, wUp, otherType, EPSILON));
            assertEquals(other, otherType.dataType(), type + " output type");
            assertClose(type + " output of type " + other, expected, otherType, 1e-4);
            INDArray half = Nd4j.create(DataType.FLOAT16, 2, 3, 5);
            Nd4j.exec(new FusedRmsNormSwiGLU(x, gamma, wGate, wUp, half, EPSILON));
            assertClose(type + " output of type HALF", expected, half, 5e-3);

            // backward into every second element of every gradient, and into every second one from the second on
            INDArray dxParent = Nd4j.valueArrayOf(new long[]{2, 3, 140}, SENTINEL, type);
            INDArray dgammaParent = Nd4j.valueArrayOf(new long[]{140}, SENTINEL, type);
            INDArray dwGateParent = Nd4j.valueArrayOf(new long[]{70, 10}, SENTINEL, type);
            INDArray dwUpParent = Nd4j.valueArrayOf(new long[]{70, 11}, SENTINEL, type);
            Nd4j.exec(DynamicCustomOp.builder("fused_rms_norm_swiglu_bp").addFloatingPointArguments((double) EPSILON)
                    .addInputs(x, gamma, wGate, wUp, dy)
                    .addOutputs(
                            dxParent.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)),
                            dgammaParent.get(NDArrayIndex.interval(0, 2, 140)),
                            dwGateParent.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 10)),
                            dwUpParent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 11)))
                    .build());
            double[] dx = dxParent.dup('c').data().asDouble();
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 70; c++) {
                    assertEquals(expectedBackward[1][r * 70 + c], dx[r * 140 + 2 * c],
                            tolerance * Math.max(1.0, Math.abs(expectedBackward[1][r * 70 + c])),
                            type + " dx at " + r + ", " + c);
                    assertEquals(SENTINEL, dx[r * 140 + 2 * c + 1], 0.0, type + " dx's neighbor at " + r + ", " + c);
                }
            }
            double[] dgamma = dgammaParent.dup('c').data().asDouble();
            for (int c = 0; c < 70; c++) {
                assertEquals(expectedBackward[2][c], dgamma[2 * c],
                        tolerance * Math.max(1.0, Math.abs(expectedBackward[2][c])), type + " dgamma at " + c);
                assertEquals(SENTINEL, dgamma[2 * c + 1], 0.0, type + " dgamma's neighbor at " + c);
            }
            double[] dwGate = dwGateParent.dup('c').data().asDouble();
            double[] dwUp = dwUpParent.dup('c').data().asDouble();
            for (int r = 0; r < 70; r++) {
                for (int c = 0; c < 5; c++) {
                    assertEquals(expectedBackward[3][r * 5 + c], dwGate[r * 10 + 2 * c],
                            tolerance * Math.max(1.0, Math.abs(expectedBackward[3][r * 5 + c])),
                            type + " dwGate at " + r + ", " + c);
                    assertEquals(SENTINEL, dwGate[r * 10 + 2 * c + 1], 0.0, type + " dwGate's neighbor at " + r + ", " + c);
                    assertEquals(expectedBackward[4][r * 5 + c], dwUp[r * 11 + 2 * c + 1],
                            tolerance * Math.max(1.0, Math.abs(expectedBackward[4][r * 5 + c])),
                            type + " dwUp at " + r + ", " + c);
                    assertEquals(SENTINEL, dwUp[r * 11 + 2 * c], 0.0, type + " dwUp's neighbor at " + r + ", " + c);
                }
                assertEquals(SENTINEL, dwUp[r * 11 + 10], 0.0, type + " dwUp's last element of row " + r);
            }

            // backward into arrays of another float type: the gradients are rounded to it once
            DataType gradientType = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray[] otherGradients = {Nd4j.create(gradientType, 2, 3, 70), Nd4j.create(gradientType, 70),
                    Nd4j.create(gradientType, 70, 5), Nd4j.create(gradientType, 70, 5)};
            Nd4j.exec(DynamicCustomOp.builder("fused_rms_norm_swiglu_bp").addFloatingPointArguments((double) EPSILON)
                    .addInputs(x, gamma, wGate, wUp, dy).addOutputs(otherGradients).build());
            for (int g = 0; g < 4; g++) {
                assertEquals(gradientType, otherGradients[g].dataType(), type + " gradient " + g + " type");
                assertClose(type + " gradient " + g + " of type " + gradientType, expectedBackward[g + 1],
                        otherGradients[g], 1e-4);
            }
        }
    }

    /**
     * A zero row normalizes to zero (not NaN: epsilon keeps 1 / rms finite): its gate and up projections are zero, and so
     * are its output and its gradients, in the 16-bit types too.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void zeroRowsStayFinite(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.FLOAT16}) {
            INDArray x = activations(type, 501, 2, 3, 70);
            // the second position of the first batch entry is all zero
            x.get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all()).assign(0.0);
            INDArray gamma = gammaOf(type, 502, 70);
            INDArray wGate = weights(type, 503, 70, 5);
            INDArray wUp = weights(type, 504, 70, 5);
            INDArray dy = gradientOf(type, 505, 2, 3, 5);
            double forwardTolerance = type == DataType.FLOAT ? 1e-4 : lowPrecisionForwardTolerance(type);
            double backwardTolerance = type == DataType.FLOAT ? 1e-4 : lowPrecisionBackwardTolerance(type);
            assertForward(type + " zero row", x, gamma, wGate, wUp, forwardTolerance);
            assertBackward(type + " zero row", x, gamma, wGate, wUp, dy, backwardTolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardOutputsDoNotAlias(Nd4jBackend backend) {
        // the four gradients are four arrays of their own: writing one must not change another
        INDArray x = activations(DataType.FLOAT, 511, 2, 3, 70);
        INDArray[] grads = Nd4j.exec(backwardOp(x, gammaOf(DataType.FLOAT, 512, 70), weights(DataType.FLOAT, 513, 70, 5),
                weights(DataType.FLOAT, 514, 70, 5), gradientOf(DataType.FLOAT, 515, 2, 3, 5)));
        double[][] before = new double[4][];
        for (int g = 0; g < 4; g++) before[g] = grads[g].dup('c').data().asDouble();
        for (int g = 0; g < 4; g++) {
            grads[g].assign(SENTINEL);
            for (int other = 0; other < 4; other++) {
                if (other == g) continue;
                double[] now = grads[other].dup('c').data().asDouble();
                boolean untouched = true;
                for (int i = 0; i < now.length; i++) {
                    // the gradients that were already overwritten hold the sentinel, the others their values
                    double expectedValue = other < g ? SENTINEL : before[other][i];
                    if (now[i] != expectedValue) untouched = false;
                }
                assertTrue(untouched, "writing gradient " + g + " changed gradient " + other);
            }
        }
    }
}
