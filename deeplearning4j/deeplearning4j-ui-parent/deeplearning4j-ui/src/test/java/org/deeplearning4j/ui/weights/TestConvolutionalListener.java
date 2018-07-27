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

package org.deeplearning4j.ui.weights;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 08/10/2016.
 */
public class TestConvolutionalListener {

    @Test
    @Ignore //Should be run manually
    public void testUI() throws Exception {

        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345) // Training iterations as above
                        .l2(0.0005).weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9)).list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                        //Note that nIn need not be specified in later layers
                                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nOut(outputNum).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ConvolutionalIterationListener(1), new ScoreIterationListener(1));

        for (int i = 0; i < 50; i++) {
            net.fit(mnistTrain.next());
            Thread.sleep(1000);
        }


        Thread.sleep(100000);
    }
}
