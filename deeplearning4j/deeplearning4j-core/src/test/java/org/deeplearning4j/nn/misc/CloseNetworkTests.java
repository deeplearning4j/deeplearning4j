/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.misc;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import static org.junit.Assert.assertTrue;

public class CloseNetworkTests extends BaseDL4JTest {

    public static MultiLayerNetwork getTestNet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder().nOut(5).kernelSize(3, 3).activation(Activation.TANH).build())
                .layer(new BatchNormalization.Builder().nOut(5).build())
                .layer(new SubsamplingLayer.Builder().build())
                .layer(new DenseLayer.Builder().nOut(10).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nOut(10).build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    @Test
    public void testCloseMLN() {
        for (boolean train : new boolean[]{false, true}) {
            for (boolean test : new boolean[]{false, true}) {
                MultiLayerNetwork net = getTestNet();

                INDArray f = Nd4j.rand(DataType.FLOAT, 16, 1, 28, 28);
                INDArray l = TestUtils.randomOneHot(16, 10);

                if (train) {
                    for (int i = 0; i < 3; i++) {
                        net.fit(f, l);
                    }
                }

                if (test) {
                    for (int i = 0; i < 3; i++) {
                        net.output(f);
                    }
                }

                net.close();

                assertTrue(net.params().wasClosed());
                if(train) {
                    assertTrue(net.getGradientsViewArray().wasClosed());
                    Updater u = net.getUpdater(false);
                    assertTrue(u.getStateViewArray().wasClosed());
                }

                //Make sure we don't get crashes etc when trying to use after closing
                try {
                    net.output(f);
                } catch (IllegalStateException e) {
                    String msg = e.getMessage();
                    assertTrue(msg, msg.contains("released"));
                }

                try {
                    net.fit(f, l);
                } catch (IllegalStateException e) {
                    String msg = e.getMessage();
                    assertTrue(msg, msg.contains("released"));
                }
            }
        }
    }

    @Test
    public void testCloseCG() {
        for (boolean train : new boolean[]{false, true}) {
            for (boolean test : new boolean[]{false, true}) {
                ComputationGraph net = getTestNet().toComputationGraph();

                INDArray f = Nd4j.rand(DataType.FLOAT, 16, 1, 28, 28);
                INDArray l = TestUtils.randomOneHot(16, 10);

                if (train) {
                    for (int i = 0; i < 3; i++) {
                        net.fit(new INDArray[]{f}, new INDArray[]{l});
                    }
                }

                if (test) {
                    for (int i = 0; i < 3; i++) {
                        net.output(f);
                    }
                }

                net.close();

                assertTrue(net.params().wasClosed());
                if(train) {
                    assertTrue(net.getGradientsViewArray().wasClosed());
                    Updater u = net.getUpdater(false);
                    assertTrue(u.getStateViewArray().wasClosed());
                }

                //Make sure we don't get crashes etc when trying to use after closing
                try {
                    net.output(f);
                } catch (IllegalStateException e) {
                    String msg = e.getMessage();
                    assertTrue(msg, msg.contains("released"));
                }

                try {
                    net.fit(new INDArray[]{f}, new INDArray[]{l});
                } catch (IllegalStateException e) {
                    String msg = e.getMessage();
                    assertTrue(msg, msg.contains("released"));
                }
            }
        }
    }
}
