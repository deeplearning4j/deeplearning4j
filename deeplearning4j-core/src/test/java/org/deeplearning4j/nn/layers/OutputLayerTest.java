/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Collections;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/1/14.
 */
@Slf4j
public class OutputLayerTest {
    private static final ActivationsFactory af = ActivationsFactory.getInstance();
    private static final GradientsFactory gf = GradientsFactory.getInstance();

    @Test
    public void testSetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).iterations(100)
                        .updater(new Sgd(1e-1))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.ZERO).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer l = (OutputLayer) conf.getLayer().instantiate(conf,
                        Collections.<IterationListener>singletonList(new ScoreIterationListener(1)), 0, params, true);
        params = l.params();
        l.setParams(params);
        assertEquals(params, l.params());
    }

    @Test
    public void testOutputLayersRnnForwardPass() {
        //Test output layer with RNNs (
        //Expect all outputs etc. to be 2d
        int nIn = 2;
        int nOut = 5;
        int layerSize = 4;
        int timeSeriesLength = 6;
        int miniBatchSize = 3;

        Random r = new Random(12345L);
        INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < nIn; j++) {
                for (int k = 0; k < timeSeriesLength; k++) {
                    input.putScalar(new int[] {i, j, k}, r.nextDouble() - 0.5);
                }
            }
        }

        //As above, but for RnnOutputLayer. Expect all activations etc. to be 3d

        MultiLayerConfiguration confRnn = new NeuralNetConfiguration.Builder().seed(12345L).list()
                        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).activation(Activation.TANH)
                                        .updater(new NoOp()).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder(LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                        .updater(new NoOp()).build())
                        .build();

        MultiLayerNetwork mlnRnn = new MultiLayerNetwork(confRnn);
        mlnRnn.init();

        INDArray out3d = mlnRnn.feedForward(input).get(2);
        assertArrayEquals(out3d.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray outRnn = mlnRnn.output(input);
        assertArrayEquals(outRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

        mlnRnn.setInput(input);
        INDArray actRnn = mlnRnn.activate(false).get(0);
        assertArrayEquals(actRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});
    }
}
