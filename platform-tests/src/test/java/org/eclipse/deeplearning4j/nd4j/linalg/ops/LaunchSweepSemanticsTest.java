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
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ops whose CUDA launches, kernels or CPU loops went wrong past a size or for a layout give the Java reference result
 * there: trace of 513 matrices (3 threads a block and a halving reduction that dropped the third partial), the
 * monotonicity checks at 768 elements (the same reduction, one verdict in three lost), log_softmax of a vector (256
 * blocks raced in place), percentile along leading axes (a TAD read as if dense), dilation2d with several images and
 * channels (x read at a batch and channel never set), scatter_update with repeated indices (CUDA read 64-bit indices
 * from a 32-bit copy; CPU threads raced on a repeated row), barnes_gains and skipgram (launch keys missing on CUDA:
 * every call threw) and pooling3d's backprop past half a million gradients.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class LaunchSweepSemanticsTest extends BaseNd4jTestWithBackends {

    private static INDArray uniform(DataType type, long seed, double lo, double hi, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(type, shape).muli(hi - lo).addi(lo);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void traceOfEveryMatrix(Nd4jBackend backend) {
        for (long[] shape : new long[][]{{513, 3, 3}, {1000, 2, 5}, {3, 7, 4, 4}}) {
            INDArray x = uniform(DataType.DOUBLE, 7, -1, 1, shape);
            INDArray traces = Nd4j.exec(DynamicCustomOp.builder("trace").addInputs(x).build())[0];
            int rank = shape.length;
            long rows = shape[rank - 2];
            long cols = shape[rank - 1];
            long matrices = x.length() / (rows * cols);
            double[] values = x.dup('c').data().asDouble();
            double[] actual = traces.dup('c').data().asDouble();
            assertEquals(matrices, actual.length, "trace length of " + Arrays.toString(shape));
            for (long m = 0; m < matrices; m++) {
                double sum = 0;
                for (long d = 0; d < Math.min(rows, cols); d++) {
                    sum += values[(int) (m * rows * cols + d * cols + d)];
                }
                assertEquals(sum, actual[(int) m], 1e-12, "trace of matrix " + m + " of " + Arrays.toString(shape));
            }
        }
    }

    private static boolean nonDecreasing(double[] values) {
        return Nd4j.exec(new IsNonDecreasing(Nd4j.createFromArray(values).castTo(DataType.FLOAT)))[0].getDouble(0) != 0;
    }

    private static boolean strictlyIncreasing(double[] values) {
        return Nd4j.exec(new IsStrictlyIncreasing(Nd4j.createFromArray(values).castTo(DataType.FLOAT)))[0]
                .getDouble(0) != 0;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void monotonicityChecksSeeEveryPair(Nd4jBackend backend) {
        for (int length : new int[]{768, 1000, 5000}) {
            double[] increasing = new double[length];
            for (int i = 0; i < length; i++) increasing[i] = i * 0.5;
            assertTrue(nonDecreasing(increasing), "is_non_decreasing of an increasing " + length);
            assertTrue(strictlyIncreasing(increasing), "is_strictly_increasing of an increasing " + length);
            for (int at : new int[]{2, 5, length - 1}) {
                double[] decrease = increasing.clone();
                decrease[at] = decrease[at - 1] - 1;
                assertFalse(nonDecreasing(decrease), "is_non_decreasing, " + length + ", decrease at " + at);
                assertFalse(strictlyIncreasing(decrease), "is_strictly_increasing, " + length + ", decrease at " + at);
                double[] equal = increasing.clone();
                equal[at] = equal[at - 1];
                assertTrue(nonDecreasing(equal), "is_non_decreasing, " + length + ", equal pair at " + at);
                assertFalse(strictlyIncreasing(equal), "is_strictly_increasing, " + length + ", equal pair at " + at);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void logSoftmaxOfAVectorInAndOutOfPlace(Nd4jBackend backend) {
        for (int length : new int[]{2, 1000, 5000}) {
            INDArray x = uniform(DataType.FLOAT, length, -5, 5, length);
            double[] xs = x.data().asDouble();
            double max = Double.NEGATIVE_INFINITY;
            for (double v : xs) max = Math.max(max, v);
            double sum = 0;
            for (double v : xs) sum += Math.exp(v - max);
            double logSum = max + Math.log(sum);
            INDArray out = Nd4j.exec(DynamicCustomOp.builder("log_softmax").addInputs(x).build())[0];
            INDArray inPlace = x.dup();
            Nd4j.exec(DynamicCustomOp.builder("log_softmax").addInputs(inPlace).addOutputs(inPlace).build());
            for (int i = 0; i < length; i++) {
                assertEquals(xs[i] - logSum, out.getDouble(i), 1e-4, "log_softmax " + length + " at " + i);
                assertEquals(xs[i] - logSum, inPlace.getDouble(i), 1e-4,
                        "log_softmax in place " + length + " at " + i);
            }
        }
    }

    /** percentile over the given axes (all when empty), as the op picks: the sorted TAD's element at a position. */
    private static double[] percentileReference(double[] values, long[] shape, int[] axes, float q,
                                                int interpolation) {
        int rank = shape.length;
        boolean[] reduced = new boolean[rank];
        if (axes.length == 0) Arrays.fill(reduced, true);
        for (int axis : axes) reduced[axis] = true;
        int outLength = 1;
        int tadLength = 1;
        for (int d = 0; d < rank; d++) {
            if (reduced[d]) tadLength *= (int) shape[d];
            else outLength *= (int) shape[d];
        }
        List<List<Double>> tads = new ArrayList<>();
        for (int o = 0; o < outLength; o++) tads.add(new ArrayList<>());
        long[] strides = new long[rank];
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--) strides[d] = strides[d + 1] * shape[d + 1];
        for (int i = 0; i < values.length; i++) {
            long remainder = i;
            int outIndex = 0;
            for (int d = 0; d < rank; d++) {
                long coordinate = remainder / strides[d];
                remainder %= strides[d];
                if (!reduced[d]) outIndex = (int) (outIndex * shape[d] + coordinate);
            }
            tads.get(outIndex).add(values[i]);
        }
        // the op's float arithmetic: fraction = 1 - q / 100, scaled = (tadLength - 1) * fraction
        float fraction = (float) (1.0 - q / 100.0);
        float scaled = (tadLength - 1) * fraction;
        long position = interpolation == 0 ? (long) Math.ceil(scaled)
                : interpolation == 1 ? (long) Math.floor(scaled) : Math.round(scaled);
        position = tadLength - position - 1;
        double[] out = new double[outLength];
        for (int o = 0; o < outLength; o++) {
            double[] sorted = tads.get(o).stream().mapToDouble(Double::doubleValue).sorted().toArray();
            out[o] = sorted[(int) position];
        }
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void percentileAlongAnyAxes(Nd4jBackend backend) {
        long[] shape = {5, 7, 3};
        INDArray x = uniform(DataType.FLOAT, 11, -10, 10, shape);
        double[] values = x.dup('c').data().asDouble();
        float q = 30f;
        for (int[] axes : new int[][]{{0}, {1}, {2}, {0, 2}, {1, 2}, {}}) {
            for (int interpolation = 0; interpolation < 3; interpolation++) {
                DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("percentile").addInputs(x)
                        .addFloatingPointArguments((double) q, (double) interpolation);
                if (axes.length > 0) builder.addIntegerArguments(axes);
                INDArray out = Nd4j.exec(builder.build())[0];
                double[] expected = percentileReference(values, shape, axes, q, interpolation);
                double[] actual = out.dup('c').data().asDouble();
                assertEquals(expected.length, actual.length, "percentile length, axes " + Arrays.toString(axes));
                for (int o = 0; o < expected.length; o++) {
                    assertEquals(expected[o], actual[o], 0.0, "percentile, axes " + Arrays.toString(axes)
                            + ", interpolation " + interpolation + ", output " + o);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void dilation2dReadsEveryImageAndChannel(Nd4jBackend backend) {
        int bS = 3, iH = 9, iW = 10, iC = 4, kH = 2, kW = 3;
        INDArray x = uniform(DataType.FLOAT, 21, -1, 1, bS, iH, iW, iC);
        INDArray w = uniform(DataType.FLOAT, 22, -1, 1, kH, kW, iC);
        float[] xs = x.dup('c').data().asFloat();
        float[] ws = w.dup('c').data().asFloat();
        for (int[] rs : new int[][]{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 2, 2, 1, 1, 1, 2, 1}}) {
            // VALID padding, rates then strides
            long[] iArgs = {0, rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6], rs[7]};
            INDArray out = Nd4j.exec(DynamicCustomOp.builder("dilation2d").addInputs(x, w).addIntegerArguments(iArgs)
                    .build())[0];
            int dH = rs[1], dW = rs[2], sH = rs[5], sW = rs[6];
            int kHeff = kH + (kH - 1) * (dH - 1);
            int kWeff = kW + (kW - 1) * (dW - 1);
            int oH = (iH - kHeff + sH) / sH;
            int oW = (iW - kWeff + sW) / sW;
            assertEquals(Arrays.toString(new long[]{bS, oH, oW, iC}), Arrays.toString(out.shape()),
                    "dilation2d shape, rates and strides " + Arrays.toString(rs));
            for (int b = 0; b < bS; b++) {
                for (int oh = 0; oh < oH; oh++) {
                    for (int ow = 0; ow < oW; ow++) {
                        for (int c = 0; c < iC; c++) {
                            float max = -Float.MAX_VALUE;
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * sH + kh * dH;
                                    int iw = ow * sW + kw * dW;
                                    float v = xs[((b * iH + ih) * iW + iw) * iC + c] + ws[(kh * kW + kw) * iC + c];
                                    if (v > max) max = v;
                                }
                            }
                            assertEquals(max, out.getFloat(b, oh, ow, c), 0.0f, "dilation2d " + Arrays.toString(rs)
                                    + " at " + b + ", " + oh + ", " + ow + ", " + c);
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void scatterUpdateAppliesRepeatedIndicesInOrder(Nd4jBackend backend) {
        float[][] updateValues = new float[5][4];
        for (int k = 0; k < 5; k++) for (int c = 0; c < 4; c++) updateValues[k][c] = 4 * k + c + 1;
        INDArray updates = Nd4j.createFromArray(updateValues);
        long[] indices = {4, 0, 2, 0, 5};
        // add, reverse subtract (u - x: the order of the two updates of row 0 matters), assign (the last one stays)
        for (int op : new int[]{0, 4, 6}) {
            float[][] start = new float[6][4];
            for (int r = 0; r < 6; r++) for (int c = 0; c < 4; c++) start[r][c] = 10 * r - c;
            INDArray input = Nd4j.createFromArray(start);
            float[][] expected = new float[6][];
            for (int r = 0; r < 6; r++) expected[r] = start[r].clone();
            for (int k = 0; k < indices.length; k++) {
                float[] row = expected[(int) indices[k]];
                for (int c = 0; c < 4; c++) {
                    float u = updateValues[k][c];
                    row[c] = op == 0 ? row[c] + u : op == 4 ? u - row[c] : u;
                }
            }
            long[] iArgs = new long[4 + indices.length];
            iArgs[0] = op;
            iArgs[1] = 1;  // one TAD dimension: rows
            iArgs[2] = 1;
            iArgs[3] = indices.length;
            System.arraycopy(indices, 0, iArgs, 4, indices.length);
            Nd4j.exec(DynamicCustomOp.builder("scatter_update").addInputs(input, updates).addOutputs(input)
                    .addIntegerArguments(iArgs).build());
            assertEquals(Nd4j.createFromArray(expected), input, "scatter_update op " + op);
        }

        // two thousand updates of four rows: every update of a row arrives
        int count = 2000;
        float[][] many = new float[count][64];
        long[] manyIndices = new long[count];
        float[][] sums = new float[4][64];
        for (int k = 0; k < count; k++) {
            manyIndices[k] = k % 4;
            for (int c = 0; c < 64; c++) {
                many[k][c] = k + 1;
                sums[k % 4][c] += k + 1;
            }
        }
        INDArray rows = Nd4j.zeros(DataType.FLOAT, 4, 64);
        long[] iArgs = new long[4 + count];
        iArgs[1] = 1;
        iArgs[2] = 1;
        iArgs[3] = count;
        System.arraycopy(manyIndices, 0, iArgs, 4, count);
        Nd4j.exec(DynamicCustomOp.builder("scatter_update").addInputs(rows, Nd4j.createFromArray(many)).addOutputs(rows)
                .addIntegerArguments(iArgs).build());
        assertEquals(Nd4j.createFromArray(sums), rows, "scatter_update of 2000 updates to 4 rows");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void barnesGainsOfEveryElement(Nd4jBackend backend) {
        int n = 70000;
        INDArray x = uniform(DataType.FLOAT, 31, 0, 2, n);
        INDArray grad = uniform(DataType.FLOAT, 32, -1, 1, n);
        INDArray eps = uniform(DataType.FLOAT, 33, -1, 1, n);
        INDArray out = Nd4j.exec(DynamicCustomOp.builder("barnes_gains").addInputs(x, grad, eps)
                .addOutputs(Nd4j.create(DataType.FLOAT, n)).build())[0];
        float[] xs = x.data().asFloat();
        float[] gs = grad.data().asFloat();
        float[] es = eps.data().asFloat();
        float[] actual = out.data().asFloat();
        for (int i = 0; i < n; i++) {
            float result = Math.signum(gs[i]) != Math.signum(es[i]) ? xs[i] + 0.2f : xs[i] * 0.8f;
            if (result < 0.01) result = 0.01f;
            assertEquals(result, actual[i], 0.0f, "barnes_gains at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramHierarchicalSoftmaxRound(Nd4jBackend backend) {
        int vocab = 10, dim = 50, expLength = 1000, target = 3;
        int[] points = {1, 5, 7};
        byte[] codes = {0, 1, 1};
        double alpha = 0.025;
        float[] expTable = new float[expLength];
        for (int i = 0; i < expLength; i++) {
            double e = Math.exp((i / (double) expLength * 2 - 1) * 6.0);
            expTable[i] = (float) (e / (e + 1));
        }
        INDArray syn0 = uniform(DataType.FLOAT, 41, -0.5, 0.5, vocab, dim);
        INDArray syn1 = uniform(DataType.FLOAT, 42, -0.5, 0.5, vocab, dim);
        float[][] s0 = syn0.toFloatMatrix();
        float[][] s1 = syn1.toFloatMatrix();

        // word2vec's hierarchical softmax step (sg_cb.cpp hSoftmax_), then the target row takes the accumulated error
        float[] neu1e = new float[dim];
        for (int r = 0; r < points.length; r++) {
            float[] row = s1[points[r]];
            double dot = 0;
            for (int e = 0; e < dim; e++) dot += s0[target][e] * row[e];
            if (dot < -6 || dot >= 6) continue;
            int idx = (int) (((float) dot + 6.0f) * ((float) expLength / 6.0f / 2.0f));
            if (idx < 0 || idx >= expLength) continue;
            float g = (1.0f - codes[r] - expTable[idx]) * (float) alpha;
            for (int e = 0; e < dim; e++) {
                neu1e[e] += g * row[e];
                row[e] += g * s0[target][e];
            }
        }
        for (int e = 0; e < dim; e++) s0[target][e] += neu1e[e];

        SkipGramRound op = SkipGramRound.builder()
                .target(Nd4j.scalar(target))
                .ngStarter(Nd4j.empty(DataType.INT32))
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(Nd4j.empty(DataType.FLOAT))
                .expTable(Nd4j.createFromArray(expTable))
                .negTable(Nd4j.empty(DataType.FLOAT))
                .nsRounds(0)
                .indices(Nd4j.createFromArray(points))
                .codes(Nd4j.createFromArray(codes))
                .alpha(Nd4j.scalar(alpha))
                .randomValue(Nd4j.scalar(119L))
                .inferenceVector(Nd4j.empty(DataType.FLOAT))
                .preciseMode(false)
                .numWorkers(1)
                .iterations(1)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertTrue(Nd4j.createFromArray(s0).equalsWithEps(syn0, 1e-5), "skipgram syn0");
        assertTrue(Nd4j.createFromArray(s1).equalsWithEps(syn1, 1e-5), "skipgram syn1");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pooling3dBackpropPastHalfAMillionGradients(Nd4jBackend backend) {
        // pooling3d's backprop launched 512 blocks of ceil(n / 512) threads: past 524288 gradients it could not
        // launch. A 1x1x1 window passes every gradient through as it is.
        long[] shape = {3, 2, 16, 128, 128};
        INDArray input = uniform(DataType.FLOAT, 51, -1, 1, shape);
        INDArray gradO = uniform(DataType.FLOAT, 52, -1, 1, shape);
        long[] iArgs = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0};
        for (String op : new String[]{"avgpool3dnew_bp", "maxpool3dnew_bp"}) {
            INDArray gradI = Nd4j.exec(DynamicCustomOp.builder(op).addInputs(input, gradO).addIntegerArguments(iArgs)
                    .build())[0];
            assertEquals(gradO, gradI, op);
        }
    }
}
