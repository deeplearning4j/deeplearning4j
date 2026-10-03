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
package org.eclipse.deeplearning4j.dl4jcore.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.gradientcheck.MLNConfig;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.function.ToDoubleBiFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A convolution layer's activations have the layout conv2d gives its output (an F-order array permuted to
 * [bS, oC, oH, oW]), and Cropping2D passes on a view of its input with gaps between its rows. im2col skipped every
 * element of such a view whose offset was not below the view's length, so a network with a cropping layer computed
 * different activations depending on whether a workspace had copied the view; and the depthwise backprop reshaped an
 * F-ordered input in F order, so a depthwise layer after a convolution layer passed it a wrong gradient. These are
 * CNNGradientCheckTest's cropping and strided depthwise networks, checked against their definitions in both workspace
 * modes: the score of each forward pass, the backprop gradient against differences of the definition's score, and a
 * gradient check.
 */
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
public class CnnActivationLayoutGradientTest extends BaseDL4JTest {

    private static final int CROP_H = 11, CROP_W = 12, CROP_C = 2;

    @Override
    public long getTimeoutMilliseconds() {
        return 300_000L;
    }

    /** Convolution (SAME, 2x2), cropping one row off the top and the bottom, convolution, dense softmax. */
    private static MultiLayerNetwork croppingNet(CNN2DFormat format, WorkspaceMode mode) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345).dataType(DataType.DOUBLE).updater(new NoOp())
                .trainingWorkspaceMode(mode).inferenceWorkspaceMode(mode)
                .activation(Activation.TANH).convolutionMode(ConvolutionMode.Same)
                .weightInit(new NormalDistribution(0, 0.3))
                .list()
                .layer(new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0})
                        .dataFormat(format).nIn(CROP_C).nOut(2).build())
                .layer(new Cropping2D.Builder(1, 1, 0, 0).dataFormat(format).build())
                .layer(new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0})
                        .dataFormat(format).nIn(2).nOut(2).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nOut(2).build())
                .setInputType(InputType.convolutional(CROP_H, CROP_W, CROP_C, format)).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    /** A 1x1 convolution, a depthwise 1x1 convolution of stride 2 with two filters per channel, dense softmax. */
    private static MultiLayerNetwork depthwiseNet(WorkspaceMode mode) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345).dataType(DataType.DOUBLE).updater(new NoOp())
                .trainingWorkspaceMode(mode).inferenceWorkspaceMode(mode)
                .activation(Activation.TANH).convolutionMode(ConvolutionMode.Truncate)
                .list()
                .layer(new Convolution2D.Builder().kernelSize(1, 1).stride(1, 1).nIn(3).nOut(3).build())
                .layer(new DepthwiseConvolution2D.Builder().kernelSize(1, 1).stride(2, 2).depthMultiplier(2)
                        .nIn(3).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nOut(6).build())
                .setInputType(InputType.convolutional(5, 5, 3, CNN2DFormat.NCHW)).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    private static double[] values(INDArray a) {
        return a.dup('c').data().asDouble();
    }

    /**
     * SAME 2x2 convolution of stride 1 (padding only below and to the right) with C-order weights [kH, kW, iC, oC]
     * of oC output channels, then tanh: [channels][height][width] for one example.
     */
    private static double[][][] sameConvTanh(double[][][] x, double[] w, double[] b, int oc) {
        int ic = x.length, ih = x[0].length, iw = x[0][0].length;
        double[][][] out = new double[oc][ih][iw];
        for (int o = 0; o < oc; o++)
            for (int h = 0; h < ih; h++)
                for (int col = 0; col < iw; col++) {
                    double z = b[o];
                    for (int kh = 0; kh < 2; kh++)
                        for (int kw = 0; kw < 2; kw++)
                            for (int i = 0; i < ic; i++)
                                if (h + kh < ih && col + kw < iw)
                                    z += w[((kh * 2 + kw) * ic + i) * oc + o] * x[i][h + kh][col + kw];
                    out[o][h][col] = Math.tanh(z);
                }
        return out;
    }

    /** Multi-class cross entropy of a softmax over logits, for a one-hot label row. */
    private static double crossEntropy(double[] logits, double[] labels, int row) {
        double max = Double.NEGATIVE_INFINITY;
        for (double z : logits)
            max = Math.max(max, z);
        double sum = 0;
        for (double z : logits)
            sum += Math.exp(z - max);
        double loss = 0;
        for (int j = 0; j < logits.length; j++)
            loss -= labels[row * logits.length + j] * (logits[j] - max - Math.log(sum));
        return loss;
    }

    /** The cropping network's score from its definition; activations flatten in C order of their own layout. */
    private static double croppingReferenceScore(MultiLayerNetwork net, INDArray input, INDArray labels,
                                                 boolean nhwc) {
        double[] w0 = values(net.getLayer(0).getParam("W")), b0 = values(net.getLayer(0).getParam("b"));
        double[] w2 = values(net.getLayer(2).getParam("W")), b2 = values(net.getLayer(2).getParam("b"));
        double[] w3 = values(net.getLayer(3).getParam("W")), b3 = values(net.getLayer(3).getParam("b"));
        double[] x = values(input), y = values(labels);
        int examples = (int) input.size(0);
        int ch = 2, hh = CROP_H - 2, ww = CROP_W;
        double loss = 0;
        for (int n = 0; n < examples; n++) {
            double[][][] image = new double[CROP_C][CROP_H][CROP_W];
            for (int c = 0; c < CROP_C; c++)
                for (int h = 0; h < CROP_H; h++)
                    for (int w = 0; w < CROP_W; w++)
                        image[c][h][w] = x[nhwc ? ((n * CROP_H + h) * CROP_W + w) * CROP_C + c
                                : ((n * CROP_C + c) * CROP_H + h) * CROP_W + w];
            double[][][] a0 = sameConvTanh(image, w0, b0, 2);
            double[][][] cropped = new double[2][hh][];
            for (int c = 0; c < 2; c++)
                for (int h = 0; h < hh; h++)
                    cropped[c][h] = a0[c][h + 1];
            double[][][] a2 = sameConvTanh(cropped, w2, b2, ch);
            double[] logits = b3.clone();
            for (int c = 0; c < ch; c++)
                for (int h = 0; h < hh; h++)
                    for (int w = 0; w < ww; w++) {
                        int k = nhwc ? (h * ww + w) * ch + c : (c * hh + h) * ww + w;
                        for (int j = 0; j < 2; j++)
                            logits[j] += a2[c][h][w] * w3[k * 2 + j];
                    }
            loss += crossEntropy(logits, y, n);
        }
        return loss / examples;
    }

    /**
     * The depthwise network's score from its definition: output channel c * 2 + m of the depthwise layer is filter m
     * of input channel c.
     */
    private static double depthwiseReferenceScore(MultiLayerNetwork net, INDArray input, INDArray labels) {
        double[] w0 = values(net.getLayer(0).getParam("W")), b0 = values(net.getLayer(0).getParam("b"));
        double[] w1 = values(net.getLayer(1).getParam("W")), b1 = values(net.getLayer(1).getParam("b"));
        double[] w2 = values(net.getLayer(2).getParam("W")), b2 = values(net.getLayer(2).getParam("b"));
        double[] x = values(input), y = values(labels);
        int examples = (int) input.size(0);
        double loss = 0;
        for (int n = 0; n < examples; n++) {
            double[][][] a0 = new double[3][5][5];
            for (int o = 0; o < 3; o++)
                for (int h = 0; h < 5; h++)
                    for (int w = 0; w < 5; w++) {
                        double z = b0[o];
                        for (int i = 0; i < 3; i++)
                            z += w0[i * 3 + o] * x[((n * 3 + i) * 5 + h) * 5 + w];
                        a0[o][h][w] = Math.tanh(z);
                    }
            double[] logits = b2.clone();
            for (int c = 0; c < 3; c++)
                for (int m = 0; m < 2; m++)
                    for (int oh = 0; oh < 3; oh++)
                        for (int ow = 0; ow < 3; ow++) {
                            int oc = c * 2 + m;
                            double a1 = Math.tanh(w1[c * 2 + m] * a0[c][2 * oh][2 * ow] + b1[oc]);
                            int k = (oc * 3 + oh) * 3 + ow;
                            for (int j = 0; j < 6; j++)
                                logits[j] += a1 * w2[k * 6 + j];
                        }
            loss += crossEntropy(logits, y, n);
        }
        return loss / examples;
    }

    /**
     * Every forward pass scores as the definition does, and every backprop gradient (divided by the minibatch, as the
     * updater divides it) equals the central difference of the definition's score.
     */
    private static void checkAgainstDefinition(MultiLayerNetwork net, INDArray input, INDArray labels,
                                               ToDoubleBiFunction<INDArray, INDArray> reference) {
        DataSet ds = new DataSet(input, labels);
        double expectedScore = reference.applyAsDouble(input, labels);
        assertEquals(expectedScore, net.score(ds, true), 1e-12, "score");
        net.setInput(input);
        net.setLabels(labels);
        net.computeGradientAndScore();
        assertEquals(expectedScore, net.score(), 1e-12, "score of the backprop pass");
        INDArray backprop = net.gradient().gradient().dup().divi(input.size(0));

        INDArray params = net.params();
        double eps = 1e-6;
        StringBuilder failures = new StringBuilder();
        for (long i = 0; i < params.length(); i++) {
            double orig = params.getDouble(i);
            params.putScalar(i, orig + eps);
            double plus = reference.applyAsDouble(input, labels);
            params.putScalar(i, orig - eps);
            double minus = reference.applyAsDouble(input, labels);
            params.putScalar(i, orig);
            double numeric = (plus - minus) / (2 * eps);
            double analytic = backprop.getDouble(i);
            if (!(Math.abs(analytic - numeric) <= 1e-6 * (1 + Math.abs(numeric))))
                failures.append(String.format("param %d: backprop %.10f, definition %.10f%n", i, analytic, numeric));
        }
        assertEquals("", failures.toString());
    }

    private static INDArray oneHot(int rows, int classes) {
        INDArray labels = Nd4j.zeros(DataType.DOUBLE, rows, classes);
        for (int i = 0; i < rows; i++)
            labels.putScalar(new int[]{i, (i + 1) % classes}, 1.0);
        return labels;
    }

    private static INDArray croppingInput(CNN2DFormat format) {
        Nd4j.getRandom().setSeed(12345);
        boolean nhwc = format == CNN2DFormat.NHWC;
        return Nd4j.rand(DataType.DOUBLE, nhwc ? new long[]{2, CROP_H, CROP_W, CROP_C} : new long[]{2, CROP_C, CROP_H, CROP_W});
    }

    private static INDArray depthwiseInput() {
        Nd4j.getRandom().setSeed(12345);
        return Nd4j.rand(DataType.DOUBLE, 3, 3, 5, 5);
    }

    @ParameterizedTest
    @EnumSource(CNN2DFormat.class)
    public void croppingNetworkMatchesItsDefinition(CNN2DFormat format) {
        for (WorkspaceMode mode : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {
            MultiLayerNetwork net = croppingNet(format, mode);
            boolean nhwc = format == CNN2DFormat.NHWC;
            checkAgainstDefinition(net, croppingInput(format), oneHot(2, 2),
                    (x, y) -> croppingReferenceScore(net, x, y, nhwc));
        }
    }

    @ParameterizedTest
    @EnumSource(value = WorkspaceMode.class, names = {"NONE", "ENABLED"})
    public void stridedDepthwiseNetworkMatchesItsDefinition(WorkspaceMode mode) {
        MultiLayerNetwork net = depthwiseNet(mode);
        checkAgainstDefinition(net, depthwiseInput(), oneHot(3, 6), (x, y) -> depthwiseReferenceScore(net, x, y));
    }

    @ParameterizedTest
    @EnumSource(value = WorkspaceMode.class, names = {"NONE", "ENABLED"})
    public void gradientChecksPass(WorkspaceMode mode) {
        for (CNN2DFormat format : CNN2DFormat.values()) {
            assertTrue(GradientCheckUtil.checkGradients(new MLNConfig().net(croppingNet(format, mode))
                    .input(croppingInput(format)).labels(oneHot(2, 2)).subset(true).maxPerParam(160)),
                    "cropping network, " + format + ", workspaces " + mode);
        }
        assertTrue(GradientCheckUtil.checkGradients(new MLNConfig().net(depthwiseNet(mode)).input(depthwiseInput())
                .labels(oneHot(3, 6)).subset(true).maxPerParam(256)), "depthwise network, workspaces " + mode);
    }
}
