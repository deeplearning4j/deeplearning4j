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

package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetConfigurationTest {

    final DataSet trainingSet = createData();

    public DataSet createData() {
        int numFeatures = 40;

        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);

        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        return new DataSet(input, labels);
    }



    @Test
    public void testLearningRateByParam() {
        double lr = 0.01;
        double biasLr = 0.02;
        int[] nIns = {4, 3, 3};
        int[] nOuts = {3, 3, 3};
        int oldScore = 1;
        int newScore = 1;
        int iteration = 3;
        INDArray gradientW = Nd4j.ones(nIns[0], nOuts[0]);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0])
                                        .updater(new Sgd(lr)).biasUpdater(new Sgd(biasLr)).build())
                        .layer(1, new BatchNormalization.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(new Sgd(0.7)).build())
                        .layer(2, new OutputLayer.Builder().nIn(nIns[2]).nOut(nOuts[2]).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(lr, ((Sgd)net.getLayer(0).conf().getUpdaterByParam("W")).getLearningRate(), 1e-4);
        assertEquals(biasLr, ((Sgd)net.getLayer(0).conf().getUpdaterByParam("b")).getLearningRate(), 1e-4);
        assertEquals(0.7, ((Sgd)net.getLayer(1).conf().getUpdaterByParam("gamma")).getLearningRate(), 1e-4);
        assertEquals(0.3, ((Sgd)net.getLayer(2).conf().getUpdaterByParam("W")).getLearningRate(), 1e-4); //From global LR
        assertEquals(0.3, ((Sgd)net.getLayer(2).conf().getUpdaterByParam("W")).getLearningRate(), 1e-4); //From global LR
    }

    @Test
    public void testLeakyreluAlpha() {
        //FIXME: Make more generic to use neuralnetconfs
        int sizeX = 4;
        int scaleX = 10;
        System.out.println("Here is a leaky vector..");
        INDArray leakyVector = Nd4j.linspace(-1, 1, sizeX);
        leakyVector = leakyVector.mul(scaleX);
        System.out.println(leakyVector);


        double myAlpha = 0.5;
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with alpha = 0.5 ..");
        System.out.println("======================");
        INDArray outDef = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(leakyVector.dup(), myAlpha));
        System.out.println(outDef);

        String confActivation = "leakyrelu";
        Object[] confExtra = {myAlpha};
        INDArray outMine = Nd4j.getExecutioner().execAndReturn(
                        Nd4j.getOpFactory().createTransform(confActivation, leakyVector.dup(), confExtra));
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with a value via getOpFactory");
        System.out.println("======================");
        System.out.println(outMine);

        //Test equality for ndarray elementwise
        //assertArrayEquals(..)
    }

    @Test
    public void testL1L2ByParam() {
        double l1 = 0.01;
        double l2 = 0.07;
        int[] nIns = {4, 3, 3};
        int[] nOuts = {3, 3, 3};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l1(l1)
                        .l2(l2).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).build())
                        .layer(1, new BatchNormalization.Builder().nIn(nIns[1]).nOut(nOuts[1]).l2(0.5).build())
                        .layer(2, new OutputLayer.Builder().nIn(nIns[2]).nOut(nOuts[2]).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(l1, net.getLayer(0).conf().getL1ByParam("W"), 1e-4);
        assertEquals(0.0, net.getLayer(0).conf().getL1ByParam("b"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("beta"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("gamma"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("mean"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("var"), 0.0);
        assertEquals(l2, net.getLayer(2).conf().getL2ByParam("W"), 1e-4);
        assertEquals(0.0, net.getLayer(2).conf().getL2ByParam("b"), 0.0);
    }
}
