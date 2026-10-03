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
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2DBp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * conv2d, conv2d_bp, depthwise_conv2d and depthwise_conv2d_bp give the same results whatever the layout of their
 * activation inputs: C order, F order, the layout conv2d gives its own output (an F-order [oW, oH, bS, oC] array
 * permuted to [bS, oC, oH, oW], built in Java and taken from conv2d itself), an offset view, a stepped view with
 * gaps between rows, and a permuted view. Each
 * result is compared with the result for C-order inputs, which the definition-based tests check
 * (Conv2dBackpropOutputContractTest, DepthwiseConv2dBackpropTest).
 *
 * DL4J feeds layers exactly these layouts: a convolution layer's activations are conv2d's output, and Cropping2D
 * passes on a view of its input. im2col skipped every element of a view whose offset was not below the view's
 * length, and the depthwise backprop reshaped an F-ordered input's columns in F order.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class ConvolutionInputLayoutTest extends BaseNd4jTestWithBackends {

    private static final int BS = 3, IC = 3, IH = 7, IW = 6, MC = 2;

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
            new Layout("F-order [W, H, B, C] permuted to [B, C, H, W]", x -> {
                long[] s = x.shape();
                return Nd4j.create(x.dataType(), new long[]{s[3], s[2], s[0], s[1]}, 'f').permute(2, 3, 1, 0)
                        .assign(x);
            }),
            // conv2d's own output, with the layout and order its shape function gives: a 1x1 identity
            // convolution over the second axis reproduces the values exactly
            new Layout("conv2d output", x -> {
                long n = x.size(1);
                INDArray identity = Nd4j.eye(n).castTo(x.dataType()).reshape(1, 1, n, n);
                Conv2DConfig config = Conv2DConfig.builder().kH(1).kW(1).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                        .paddingMode(PaddingMode.VALID).dataFormat(Conv2DConfig.NCHW)
                        .weightsFormat(WeightsFormat.YXIO).build();
                return Nd4j.cnn().conv2d(x, identity, Nd4j.zeros(x.dataType(), n), config);
            }),
            new Layout("offset view", x -> {
                long[] s = x.shape();
                INDArray parent = Nd4j.valueArrayOf(new long[]{s[0] + 1, s[1], s[2], s[3]}, Double.NaN, x.dataType());
                return parent.get(NDArrayIndex.interval(1, s[0] + 1), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all()).assign(x);
            }),
            new Layout("stepped view", x -> {
                long[] s = x.shape();
                INDArray parent = Nd4j.valueArrayOf(new long[]{s[0], s[1], 2 * s[2] + 1, s[3] + 3}, Double.NaN,
                        x.dataType());
                return parent.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2 * s[2] + 1),
                        NDArrayIndex.interval(2, s[3] + 2)).assign(x);
            }),
            new Layout("permuted view", x -> {
                long[] s = x.shape();
                return Nd4j.create(x.dataType(), new long[]{s[0], s[2], s[3], s[1]}, 'c').permute(0, 3, 1, 2)
                        .assign(x);
            }),
    };

    private static final class Geometry {
        final int k, s;
        final PaddingMode mode;

        Geometry(int k, int s, PaddingMode mode) {
            this.k = k;
            this.s = s;
            this.mode = mode;
        }

        int oH() { return mode == PaddingMode.SAME ? (IH + s - 1) / s : (IH - k) / s + 1; }
        int oW() { return mode == PaddingMode.SAME ? (IW + s - 1) / s : (IW - k) / s + 1; }

        Conv2DConfig config(String format) {
            return Conv2DConfig.builder().kH(k).kW(k).sH(s).sW(s).pH(0).pW(0).dH(1).dW(1).paddingMode(mode)
                    .dataFormat(format).weightsFormat(WeightsFormat.YXIO).build();
        }

        @Override
        public String toString() {
            return "kernel " + k + ", stride " + s + ", " + mode;
        }
    }

    private static final Geometry[] GEOMETRIES = {
            new Geometry(1, 2, PaddingMode.VALID),
            new Geometry(2, 1, PaddingMode.SAME),
            new Geometry(3, 1, PaddingMode.VALID),
            new Geometry(3, 2, PaddingMode.SAME),
    };

    private static long[] activationShape(boolean nhwc, long c, long h, long w) {
        return nhwc ? new long[]{BS, h, w, c} : new long[]{BS, c, h, w};
    }

    private static void compare(List<String> failures, String what, INDArray expected, INDArray actual, double tol) {
        INDArray e = expected.castTo(DataType.DOUBLE), a = actual.castTo(DataType.DOUBLE);
        double maxDiff = e.sub(a).amaxNumber().doubleValue();
        if (!(maxDiff <= tol * (1 + e.amaxNumber().doubleValue())))
            failures.add(what + ": max |diff| " + maxDiff);
    }

    /** Runs an op (synchronized, so a device fault surfaces here) and compares every output with the expected one. */
    private static void check(List<String> failures, String what, Supplier<INDArray[]> expected,
                              Supplier<INDArray[]> op, double tol) {
        INDArray[] actual;
        try {
            actual = op.get();
            Nd4j.getExecutioner().commit();
        } catch (RuntimeException e) {
            failures.add(what + ": " + String.valueOf(e.getMessage()).split("\n")[0]);
            return;
        }
        INDArray[] exp = expected.get();
        String[] names = exp.length == 1 ? new String[]{""} : new String[]{" gradI", " gradW", " gradB"};
        for (int i = 0; i < exp.length; i++)
            compare(failures, what + names[i], exp[i], actual[i], tol);
    }

    private static INDArray[] conv2dBp(INDArray x, INDArray w, INDArray b, INDArray gradO, Conv2DConfig config) {
        INDArray gradI = Nd4j.create(x.dataType(), x.shape());
        INDArray gradW = Nd4j.create(w.dataType(), w.shape());
        INDArray gradB = Nd4j.create(b.dataType(), b.shape());
        Conv2DDerivative op = Conv2DDerivative.derivativeBuilder().config(config).build();
        op.addInputArgument(x, w, b, gradO);
        op.addOutputArgument(gradI, gradW, gradB);
        Nd4j.getExecutioner().exec(op);
        return new INDArray[]{gradI, gradW, gradB};
    }

    private static long[] depthwiseArgs(Geometry g, boolean nhwc) {
        return new long[]{g.k, g.k, g.s, g.s, 0, 0, 1, 1, g.mode == PaddingMode.SAME ? 1 : 0, nhwc ? 1 : 0};
    }

    private static INDArray depthwise(INDArray x, INDArray w, INDArray b, Geometry g, boolean nhwc) {
        INDArray out = Nd4j.create(x.dataType(), activationShape(nhwc, IC * MC, g.oH(), g.oW()));
        DepthwiseConv2D op = new DepthwiseConv2D();
        op.addInputArgument(x, w, b);
        op.addOutputArgument(out);
        op.addIArgument(depthwiseArgs(g, nhwc));
        Nd4j.getExecutioner().exec(op);
        return out;
    }

    private static INDArray[] depthwiseBp(INDArray x, INDArray w, INDArray b, INDArray gradO, Geometry g,
                                          boolean nhwc) {
        INDArray gradI = Nd4j.create(x.dataType(), x.shape());
        INDArray gradW = Nd4j.create(w.dataType(), w.shape());
        INDArray gradB = Nd4j.create(b.dataType(), b.shape());
        DepthwiseConv2DBp op = new DepthwiseConv2DBp();
        op.addInputArgument(x, w, b, gradO);
        op.addOutputArgument(gradI, gradW, gradB);
        op.addIArgument(depthwiseArgs(g, nhwc));
        Nd4j.getExecutioner().exec(op);
        return new INDArray[]{gradI, gradW, gradB};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void resultsDoNotDependOnTheInputLayout(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT}) {
            double tol = dt == DataType.DOUBLE ? 1e-12 : 1e-5;
            for (boolean nhwc : new boolean[]{false, true}) {
                String format = nhwc ? Conv2DConfig.NHWC : Conv2DConfig.NCHW;
                for (Geometry g : GEOMETRIES) {
                    Nd4j.getRandom().setSeed(42);
                    INDArray x = Nd4j.rand(DataType.DOUBLE, activationShape(nhwc, IC, IH, IW)).subi(0.5).castTo(dt);
                    INDArray w = Nd4j.rand(DataType.DOUBLE, g.k, g.k, IC, 4).subi(0.5).castTo(dt);
                    INDArray b = Nd4j.rand(DataType.DOUBLE, 4).subi(0.5).castTo(dt);
                    INDArray gradO = Nd4j.rand(DataType.DOUBLE, activationShape(nhwc, 4, g.oH(), g.oW())).subi(0.5).castTo(dt);
                    INDArray dw = Nd4j.rand(DataType.DOUBLE, g.k, g.k, IC, MC).subi(0.5).castTo(dt);
                    INDArray db = Nd4j.rand(DataType.DOUBLE, IC * MC).subi(0.5).castTo(dt);
                    INDArray dGradO = Nd4j.rand(DataType.DOUBLE, activationShape(nhwc, IC * MC, g.oH(), g.oW())).subi(0.5).castTo(dt);
                    Conv2DConfig config = g.config(format);

                    INDArray conv = Nd4j.cnn().conv2d(x, w, b, config);
                    INDArray[] convBp = conv2dBp(x, w, b, gradO, config);
                    INDArray dConv = depthwise(x, dw, db, g, nhwc);
                    INDArray[] dConvBp = depthwiseBp(x, dw, db, dGradO, g, nhwc);

                    for (Layout layout : LAYOUTS) {
                        String where = dt + " " + format + " " + g + ", " + layout.name;
                        // Each op is synchronized and checked on its own, so an asynchronous device fault is
                        // reported for the op that caused it
                        check(failures, where + " input: conv2d", () -> new INDArray[]{conv},
                                () -> new INDArray[]{Nd4j.cnn().conv2d(layout.of.apply(x), w, b, config)}, tol);
                        check(failures, where + " input: depthwise_conv2d", () -> new INDArray[]{dConv},
                                () -> new INDArray[]{depthwise(layout.of.apply(x), dw, db, g, nhwc)}, tol);
                        for (String operands : new String[]{"input", "gradO", "input and gradO"}) {
                            boolean inputInLayout = !operands.equals("gradO");
                            boolean gradOInLayout = !operands.equals("input");
                            check(failures, where + " " + operands + ": conv2d_bp", () -> convBp,
                                    () -> conv2dBp(inputInLayout ? layout.of.apply(x) : x, w, b,
                                            gradOInLayout ? layout.of.apply(gradO) : gradO, config), tol);
                            check(failures, where + " " + operands + ": depthwise_conv2d_bp", () -> dConvBp,
                                    () -> depthwiseBp(inputInLayout ? layout.of.apply(x) : x, dw, db,
                                            gradOInLayout ? layout.of.apply(dGradO) : dGradO, g, nhwc), tol);
                        }
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
