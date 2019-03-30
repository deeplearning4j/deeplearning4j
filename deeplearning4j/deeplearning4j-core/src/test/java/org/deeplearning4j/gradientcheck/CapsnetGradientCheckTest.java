/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.gradientcheck;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleLayer;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PrimaryCapsules;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CapsnetGradientCheckTest extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testCapsNet() {

        int[] minibatchSizes = {1, 2, 5};

        int width = 28;
        int height = 28;
        int inputDepth = 1;

        int[] primaryCapsDims = {4, 8};
        int[] primaryCapsChannels = {16, 32};
        int[] capsules = {10, 12};
        int[] capsuleDims = {12, 16};
        int[] routings = {3, 5};

        Nd4j.getRandom().setSeed(12345);

        for (int routing : routings) {
            for (int primaryCapsDim : primaryCapsDims) {
                for (int primarpCapsChannel : primaryCapsChannels) {
                    for (int capsule : capsules) {
                        for (int capsuleDim : capsuleDims) {
                            for (int minibatchSize : minibatchSizes) {
                                INDArray input = Nd4j.rand(minibatchSize, inputDepth * height * width);
                                INDArray labels = Nd4j.zeros(minibatchSize, capsule);
                                for (int i = 0; i < minibatchSize; i++) {
                                    labels.putScalar(new int[]{i, i % capsule}, 1.0);
                                }

                                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .seed(123)
                                        .updater(new NoOp())
                                        .list()
                                        .layer(new ConvolutionLayer.Builder()
                                                .nOut(16)
                                                .kernelSize(9, 9)
                                                .stride(3, 3)
                                                .build())
                                        .layer(new PrimaryCapsules.Builder(primaryCapsDim, primarpCapsChannel)
                                                .kernelSize(7, 7)
                                                .stride(2, 2)
                                                .build())
                                        .layer(new CapsuleLayer.Builder(capsule, capsuleDim, routing).build())
                                        .layer(new CapsuleStrengthLayer.Builder().build())
                                        .setInputType(InputType.convolutionalFlat(28, 28, 1))
                                        .build();

                                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                                net.init();

                                for (int i = 0; i < 4; i++) {
                                    System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
                                }

                                String msg = "minibatch=" + minibatchSize +
                                        ", PrimaryCaps: " + primarpCapsChannel +
                                        " channels, " + primaryCapsDim + " dimensions, Capsules: " + capsule +
                                        " capsules with " + capsuleDim + "dimensions and " + routing + " routings";
                                System.out.println(msg);

                                boolean gradOK = GradientCheckUtil
                                        .checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input,
                                                labels);

                                assertTrue(msg, gradOK);

                                TestUtils.testModelSerialization(net);
                            }
                        }
                    }
                }
            }
        }
    }
}
