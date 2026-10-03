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
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2DBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * depthwise_conv2d_bp against sums taken straight from the depthwise convolution's definition, with every output
 * starting as NaN and the input gradient a view into a NaN-filled parent whose first image must stay NaN. The
 * geometries are CNNGradientCheckTest#testDepthwiseConv2D's (a 5x5 image, three channels, two filters per channel)
 * plus SAME padding; a strided 1x1 kernel leaves every odd row and column of the image outside every window, so their
 * input gradient is zero.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class DepthwiseConv2dBackpropTest extends BaseNd4jTestWithBackends {

    private static final int BS = 3, IC = 3, MC = 2, OC = IC * MC, IH = 5, IW = 5;

    private static final class Geometry {
        final int k, s;
        final boolean same;

        Geometry(int k, int s, boolean same) {
            this.k = k;
            this.s = s;
            this.same = same;
        }

        int oH() { return same ? (IH + s - 1) / s : (IH - k) / s + 1; }
        int oW() { return same ? (IW + s - 1) / s : (IW - k) / s + 1; }
        // SAME pads the top and left by half the total, rounded down
        int pH() { return same ? Math.max(0, (oH() - 1) * s + k - IH) / 2 : 0; }
        int pW() { return same ? Math.max(0, (oW() - 1) * s + k - IW) / 2 : 0; }

        @Override
        public String toString() {
            return "kernel " + k + ", stride " + s + (same ? ", SAME" : ", VALID");
        }
    }

    /**
     * gradI [bS, iC, iH, iW], gradW [kH, kW, iC, mC] and gradB [iC*mC], summed in double from the definition: output
     * channel c*mC + m convolves input channel c with filter m.
     */
    private static double[][] reference(INDArray input, INDArray weights, INDArray gradO, Geometry g) {
        double[] x = input.dup('c').data().asDouble();
        double[] w = weights.dup('c').data().asDouble();
        double[] e = gradO.dup('c').data().asDouble();
        int oH = g.oH(), oW = g.oW(), k = g.k;
        double[] gradI = new double[BS * IC * IH * IW];
        double[] gradW = new double[k * k * IC * MC];
        double[] gradB = new double[OC];
        for (int b = 0; b < BS; b++) {
            for (int c = 0; c < IC; c++) {
                for (int m = 0; m < MC; m++) {
                    int o = c * MC + m;
                    for (int oh = 0; oh < oH; oh++) {
                        for (int ow = 0; ow < oW; ow++) {
                            double eps = e[((b * OC + o) * oH + oh) * oW + ow];
                            gradB[o] += eps;
                            for (int kh = 0; kh < k; kh++) {
                                int h = oh * g.s + kh - g.pH();
                                if (h < 0 || h >= IH) continue;
                                for (int kw = 0; kw < k; kw++) {
                                    int col = ow * g.s + kw - g.pW();
                                    if (col < 0 || col >= IW) continue;
                                    int xi = ((b * IC + c) * IH + h) * IW + col;
                                    int wi = ((kh * k + kw) * IC + c) * MC + m;
                                    gradI[xi] += w[wi] * eps;
                                    gradW[wi] += x[xi] * eps;
                                }
                            }
                        }
                    }
                }
            }
        }
        return new double[][]{gradI, gradW, gradB};
    }

    private static void compare(List<String> failures, String what, INDArray actual, double[] expected, double tol) {
        double[] a = actual.castTo(DataType.DOUBLE).dup('c').data().asDouble();
        int bad = 0;
        StringBuilder first = new StringBuilder();
        for (int i = 0; i < expected.length; i++) {
            if (!(Math.abs(a[i] - expected[i]) <= tol * (1.0 + Math.abs(expected[i])))) {
                if (bad++ < 4) {
                    first.append(" [").append(i).append("] ").append(a[i]).append(" vs ").append(expected[i]);
                }
            }
        }
        if (bad > 0) {
            failures.add(what + ": " + bad + " of " + expected.length + " wrong:" + first);
        }
    }

    private static void check(List<String> failures, DataType dt, boolean nhwc, Geometry g) {
        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.rand(DataType.DOUBLE, BS, IC, IH, IW).subi(0.5).castTo(dt);
        INDArray weights = Nd4j.rand(DataType.DOUBLE, g.k, g.k, IC, MC).subi(0.5).castTo(dt);
        INDArray bias = Nd4j.rand(DataType.DOUBLE, OC).subi(0.5).castTo(dt);
        INDArray gradO = Nd4j.rand(DataType.DOUBLE, BS, OC, g.oH(), g.oW()).subi(0.5).castTo(dt);
        double[][] expected = reference(input.castTo(DataType.DOUBLE), weights.castTo(DataType.DOUBLE),
                gradO.castTo(DataType.DOUBLE), g);

        INDArray opInput = nhwc ? input.permute(0, 2, 3, 1).dup('c') : input;
        INDArray opGradO = nhwc ? gradO.permute(0, 2, 3, 1).dup('c') : gradO;
        long[] s = opInput.shape();
        INDArray gradIParent = Nd4j.valueArrayOf(new long[]{BS + 1, s[1], s[2], s[3]}, Double.NaN, dt);
        INDArray gradI = gradIParent.get(NDArrayIndex.interval(1, BS + 1), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all());
        INDArray gradW = Nd4j.valueArrayOf(weights.shape(), Double.NaN, dt);
        INDArray gradB = Nd4j.valueArrayOf(bias.shape(), Double.NaN, dt);

        DepthwiseConv2DBp op = new DepthwiseConv2DBp();
        op.addInputArgument(opInput, weights, bias, opGradO);
        op.addOutputArgument(gradI, gradW, gradB);
        op.addIArgument(g.k, g.k, g.s, g.s, 0, 0, 1, 1, g.same ? 1 : 0, nhwc ? 1 : 0);
        Nd4j.getExecutioner().exec(op);

        double tol = dt == DataType.DOUBLE ? 1e-10 : dt == DataType.FLOAT ? 1e-5 : 3e-2;
        String where = dt + (nhwc ? " NHWC " : " NCHW ") + g;
        INDArray gradINchw = nhwc ? gradI.permute(0, 3, 1, 2) : gradI;
        compare(failures, where + " gradI", gradINchw, expected[0], tol);
        compare(failures, where + " gradW", gradW, expected[1], tol);
        compare(failures, where + " gradB", gradB, expected[2], tol);
        INDArray untouched = gradIParent.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all()).castTo(DataType.DOUBLE);
        if (untouched.isNaN().castTo(DataType.INT).sumNumber().longValue() != untouched.length()) {
            failures.add(where + ": gradI wrote outside its view");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gradientsFollowTheDefinition(Nd4jBackend backend) {
        Geometry[] geometries = {
                new Geometry(1, 1, false),
                new Geometry(3, 1, false),
                new Geometry(1, 2, false),
                new Geometry(3, 2, false),
                new Geometry(3, 1, true),
                new Geometry(2, 2, true),
        };
        List<String> failures = new ArrayList<>();
        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            for (boolean nhwc : new boolean[]{false, true}) {
                for (Geometry g : geometries) {
                    check(failures, dt, nhwc, g);
                }
            }
        }
        assertTrue(failures.isEmpty(), String.join("\n", failures));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
