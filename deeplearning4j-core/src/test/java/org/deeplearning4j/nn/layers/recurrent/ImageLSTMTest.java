/*
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

package org.deeplearning4j.nn.layers.recurrent;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.layers.ImageLSTM;
import org.junit.Ignore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 12/30/14.
 */
@Ignore
public class ImageLSTMTest {

    // TODO finish building out this test for image LSTM ...
    private static final Logger log = LoggerFactory.getLogger(ImageLSTMTest.class);

    @Ignore
    @Test
    public void testTraffic() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.ImageLSTM.Builder()
                        .nIn(4).nOut(4)
                        .activation("tanh").build())
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .build();
        Layer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(10)),0);
//        INDArray predict = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,3},4);
//        l.fit(predict);
//        INDArray out = l.activate(predict);
//        log.info("Out " + out);
    }


    @Test
    @Ignore
    public void testLSTMActivateResultsContained()  {
        INDArray input = getContainedData();
        Layer layer = getLSTMConfig(input.shape()[0], 10);
        INDArray expectedOutput = Nd4j.create(new double[] {
                1.33528078e-05,  -5.05130502e-03,  -3.94280548e-03,
                -9.52692528e-03, -2.13943459e-04,   3.29958488e-04,
                1.33345727e-03,   1.49443537e-03,   7.63499990e-03,
                -8.46091534e-03,   1.29049918e-03,  -2.02905897e-03,
                -2.05965719e-03,  -8.10794612e-03,  -2.94427043e-03,
                -3.07215801e-03,   3.74576943e-04,   2.52405324e-03,
                6.12143291e-03,  -6.46372699e-03,   1.16521495e-03,
                -6.27016936e-03,  -5.09448242e-04,  -4.08016428e-03,
                -6.02273857e-03,  -7.05439770e-03,  -3.19226720e-03,
                2.59527932e-03,   8.63829946e-03,  -7.76521200e-03,
                1.32506500e-03,   8.88431204e-04,  -2.92426303e-03,
                -7.36646599e-03,  -8.08439371e-03,  -5.97170557e-03,
                8.88601027e-04,   2.56400351e-03,   5.49041094e-03,
                -3.88451405e-03,   1.04079333e-03,  -9.01794065e-04,
                -2.69056009e-03,  -1.92756199e-03,  -6.82062829e-03,
                -5.56447686e-03,   9.74676430e-04,   1.41065843e-03,
                3.49398034e-03,  -2.32477587e-03,   4.93022156e-03,
                3.46329814e-04,   4.48665657e-03,  -8.14650031e-03,
                -5.20380651e-03,  -8.51394332e-03,   4.04197969e-04,
                4.26121057e-03,   8.59337350e-03,  -5.00470464e-03,
                3.00422186e-03,   3.79192699e-03,   6.95973303e-03,
                -3.42595537e-03,  -4.36002676e-03,  -1.01052914e-02,
                -4.22624609e-03,   4.74019012e-03,   5.23081966e-03,
                -3.60138721e-03,   4.38289620e-03,   1.03724512e-03,
                6.41595554e-03,  -8.22484502e-03,  -6.92261961e-03,
                -1.12504246e-02,  -4.12833222e-03,   4.23728583e-03,
                1.00947563e-02,  -6.76922645e-03
        },new int[]{8,10});

        INDArray lstmActivations = layer.activate(input);

//        assertArrayEquals(expectedOutput.shape(), lstmActivations.shape());
//        assertEquals(expectedOutput, lstmActivations);
    }

    //////////////////////////////////////////////////////////////////////////////////

    @Ignore
    @Test
    public void testBackpropResultsContained()  {
        INDArray input = getContainedData();
        Layer layer = getLSTMConfig(input.shape()[0], 10);
        INDArray epsilon = Nd4j.ones(8, 10);

//        INDArray expectedBiasGradient = Nd4j.create(new double[]{
//
//        }, new int[]{1, 10});
//
//        INDArray expectedWeightGradient = Nd4j.create(new double[] {
//
//        }, new int[]{8, 10});
//
//        INDArray expectedEpsilon = Nd4j.create(new double[] {
//
//        },new int[]{9,8});

//        layer.setInput(input);
// TODO if switch to calling activate in the backprop this can switch to set input
        layer.activate(input);
        Pair<Gradient, INDArray> pair = layer.backpropGradient(epsilon);

//        assertArrayEquals(expectedEpsilon.shape(), pair.getSecond().shape());
//        assertArrayEquals(expectedWeightGradient.shape(), pair.getFirst().getGradientFor("W").shape());
//        assertArrayEquals(expectedBiasGradient.shape(), pair.getFirst().getGradientFor("b").shape());
//        assertEquals(expectedEpsilon, pair.getSecond());
//        assertEquals(expectedWeightGradient, pair.getFirst().getGradientFor("W"));
//        assertEquals(expectedBiasGradient, pair.getFirst().getGradientFor("b"));
    }


    //////////////////////////////////////////////////////////////////////////////////

    private static Layer getLSTMConfig(int nIn, int nOut){

        ImageLSTM layer = new ImageLSTM.Builder()
                .nIn(nIn)
                .nOut(nOut)
                .activation("tanh")
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .layer(layer)
                .build();
        return LayerFactories.getFactory(conf).create(conf);

    }

    public INDArray getContainedData() {
        // represents xi & xs
        INDArray ret = Nd4j.create(new double[]{
                0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  0.,  0.,  0.,  1.,
                0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,
                0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  1.,  0.,  0.
        }, new int[]{9,8});
        return ret;
    }


}
