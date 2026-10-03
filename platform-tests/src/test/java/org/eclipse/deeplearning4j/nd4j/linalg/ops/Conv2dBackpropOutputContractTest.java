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
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * conv2d_bp writes every element of its three outputs: none may depend on what the caller's arrays held. The CPU
 * input gradient came from col2im, which added the column entries into the output without clearing it first; conv2d_bp
 * hands it its gradI output as given, so a recycled buffer (workspace memory, for one) left its old contents in the
 * gradient. Here every output starts as NaN, the input gradient is a view into a NaN-filled parent whose first row must
 * stay NaN, and all three gradients are compared with sums taken straight from the convolution's definition.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class Conv2dBackpropOutputContractTest extends BaseNd4jTestWithBackends {

    private static final int BS = 2, IC = 3, OC = 4, IH = 7, IW = 6, KH = 3, KW = 2;

    private static final class Geometry {
        final int sH, sW, dH, dW;
        final PaddingMode mode;

        Geometry(int sH, int sW, int dH, int dW, PaddingMode mode) {
            this.sH = sH;
            this.sW = sW;
            this.dH = dH;
            this.dW = dW;
            this.mode = mode;
        }

        int ekH() { return dH * (KH - 1) + 1; }
        int ekW() { return dW * (KW - 1) + 1; }
        int oH() { return mode == PaddingMode.SAME ? (IH + sH - 1) / sH : (IH - ekH()) / sH + 1; }
        int oW() { return mode == PaddingMode.SAME ? (IW + sW - 1) / sW : (IW - ekW()) / sW + 1; }
        // SAME pads the top and left by half the total, rounded down
        int pH() { return mode == PaddingMode.SAME ? Math.max(0, (oH() - 1) * sH + ekH() - IH) / 2 : 0; }
        int pW() { return mode == PaddingMode.SAME ? Math.max(0, (oW() - 1) * sW + ekW() - IW) / 2 : 0; }

        @Override
        public String toString() {
            return "stride " + sH + "x" + sW + ", dilation " + dH + "x" + dW + ", " + mode;
        }
    }

    /** gradI [bS, iC, iH, iW], gradW [kH, kW, iC, oC] and gradB [oC], summed in double from the definition. */
    private static double[][] reference(INDArray input, INDArray weights, INDArray gradO, Geometry g) {
        double[] x = input.dup('c').data().asDouble();
        double[] w = weights.dup('c').data().asDouble();
        double[] e = gradO.dup('c').data().asDouble();
        int oH = g.oH(), oW = g.oW();
        double[] gradI = new double[BS * IC * IH * IW];
        double[] gradW = new double[KH * KW * IC * OC];
        double[] gradB = new double[OC];
        for (int b = 0; b < BS; b++) {
            for (int o = 0; o < OC; o++) {
                for (int oh = 0; oh < oH; oh++) {
                    for (int ow = 0; ow < oW; ow++) {
                        double eps = e[((b * OC + o) * oH + oh) * oW + ow];
                        gradB[o] += eps;
                        for (int kh = 0; kh < KH; kh++) {
                            int h = oh * g.sH + kh * g.dH - g.pH();
                            if (h < 0 || h >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int col = ow * g.sW + kw * g.dW - g.pW();
                                if (col < 0 || col >= IW) continue;
                                for (int c = 0; c < IC; c++) {
                                    int xi = ((b * IC + c) * IH + h) * IW + col;
                                    int wi = ((kh * KW + kw) * IC + c) * OC + o;
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

    private static void check(List<String> failures, DataType dt, String format, Geometry g) {
        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.rand(DataType.DOUBLE, BS, IC, IH, IW).subi(0.5).castTo(dt);
        INDArray weights = Nd4j.rand(DataType.DOUBLE, KH, KW, IC, OC).subi(0.5).castTo(dt);
        INDArray bias = Nd4j.rand(DataType.DOUBLE, OC).subi(0.5).castTo(dt);
        INDArray gradO = Nd4j.rand(DataType.DOUBLE, BS, OC, g.oH(), g.oW()).subi(0.5).castTo(dt);
        double[][] expected = reference(input.castTo(DataType.DOUBLE), weights.castTo(DataType.DOUBLE),
                gradO.castTo(DataType.DOUBLE), g);

        boolean nhwc = Conv2DConfig.NHWC.equals(format);
        INDArray opInput = nhwc ? input.permute(0, 2, 3, 1).dup('c') : input;
        INDArray opGradO = nhwc ? gradO.permute(0, 2, 3, 1).dup('c') : gradO;
        long[] s = opInput.shape();
        INDArray gradIParent = Nd4j.valueArrayOf(new long[]{BS + 1, s[1], s[2], s[3]}, Double.NaN, dt);
        INDArray gradI = gradIParent.get(NDArrayIndex.interval(1, BS + 1), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all());
        INDArray gradW = Nd4j.valueArrayOf(weights.shape(), Double.NaN, dt);
        INDArray gradB = Nd4j.valueArrayOf(bias.shape(), Double.NaN, dt);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(KH).kW(KW)
                .sH(g.sH).sW(g.sW)
                .pH(0).pW(0)
                .dH(g.dH).dW(g.dW)
                .paddingMode(g.mode)
                .dataFormat(format)
                .weightsFormat(WeightsFormat.YXIO)
                .build();
        Conv2DDerivative op = Conv2DDerivative.derivativeBuilder().config(config).build();
        op.addInputArgument(opInput, weights, bias, opGradO);
        op.addOutputArgument(gradI, gradW, gradB);
        Nd4j.getExecutioner().exec(op);

        double tol = dt == DataType.DOUBLE ? 1e-10 : dt == DataType.FLOAT ? 1e-5 : 3e-2;
        String where = dt + " " + format + " " + g;
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
    public void outputsDoNotDependOnTheirPriorContents(Nd4jBackend backend) {
        Geometry[] geometries = {
                new Geometry(1, 1, 1, 1, PaddingMode.VALID),
                new Geometry(2, 2, 1, 1, PaddingMode.VALID),
                new Geometry(1, 1, 2, 2, PaddingMode.VALID),
                new Geometry(1, 1, 1, 1, PaddingMode.SAME),
                new Geometry(2, 1, 1, 2, PaddingMode.SAME),
        };
        List<String> failures = new ArrayList<>();
        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            for (String format : new String[]{Conv2DConfig.NCHW, Conv2DConfig.NHWC}) {
                for (Geometry g : geometries) {
                    check(failures, dt, format, g);
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
