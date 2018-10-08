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

package org.deeplearning4j.nn.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.Assert.assertNotEquals;

/**
 * Test CenterLossOutputLayer.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class CenterLossOutputLayerTest extends BaseDL4JTest {

    private ComputationGraph getGraph(int numLabels, double lambda) {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).updater(new NoOp())
                        .graphBuilder().addInputs("input1")
                        .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.RELU).build(),
                                        "input1")
                        .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(numLabels)
                                        .lambda(lambda).activation(Activation.SOFTMAX).build(), "l1")
                        .setOutputs("lossLayer").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        return graph;
    }

    public ComputationGraph getCNNMnistConfig() {

        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345) // Training iterations as above
                        .l2(0.0005).weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .graphBuilder().addInputs("input")
                        .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                        .addLayer("0", new ConvolutionLayer.Builder(5, 5)
                                        //nIn and nOut specify channels. nIn here is the nChannels and nOut is the number of filters to be applied
                                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build(),
                                        "input")
                        .addLayer("1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build(), "0")
                        .addLayer("2", new ConvolutionLayer.Builder(5, 5)
                                        //Note that nIn need not be specified in later layers
                                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build(), "1")
                        .addLayer("3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build(), "2")
                        .addLayer("4", new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build(), "3")
                        .addLayer("output",
                                        new org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer.Builder(
                                                        LossFunction.MCXENT).nOut(outputNum)
                                                                        .activation(Activation.SOFTMAX).build(),
                                        "4")
                        .setOutputs("output").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        return graph;
    }

    @Test
    public void testLambdaConf() {
        double[] lambdas = new double[] {0.1, 0.01};
        double[] results = new double[2];
        int numClasses = 2;

        INDArray input = Nd4j.rand(150, 4);
        INDArray labels = Nd4j.zeros(150, numClasses);
        Random r = new Random(12345);
        for (int i = 0; i < 150; i++) {
            labels.putScalar(i, r.nextInt(numClasses), 1.0);
        }
        ComputationGraph graph;

        for (int i = 0; i < lambdas.length; i++) {
            graph = getGraph(numClasses, lambdas[i]);
            graph.setInput(0, input);
            graph.setLabel(0, labels);
            graph.computeGradientAndScore();
            results[i] = graph.score();
        }

        assertNotEquals(results[0], results[1]);
    }



    @Test
    @Ignore //Should be run manually
    public void testMNISTConfig() throws Exception {
        int batchSize = 64; // Test batch size
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        ComputationGraph net = getCNNMnistConfig();
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        for (int i = 0; i < 50; i++) {
            net.fit(mnistTrain.next());
            Thread.sleep(1000);
        }

        Thread.sleep(100000);
    }
}
