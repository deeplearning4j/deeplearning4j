/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.computationgraph;

import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestComputationGraphSpace {

    @Test
    public void testBasic() {

        ComputationGraphConfiguration expected = new NeuralNetConfiguration.Builder()
                        .updater(new Sgd(0.005))
                        .seed(12345)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).build(), "0").addLayer("2",
                                        new OutputLayer.Builder().lossFunction(LossFunction.MCXENT).nIn(10).nOut(5)
                                                        .build(),
                                        "1")
                        .setOutputs("2").backprop(true).pretrain(false).build();

        ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                        .updater(new Sgd(0.005))
                        .seed(12345).addInputs("in")
                        .addLayer("0", new DenseLayerSpace.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("1", new DenseLayerSpace.Builder().nIn(10).nOut(10).build(), "0")
                        .addLayer("2", new OutputLayerSpace.Builder().lossFunction(LossFunction.MCXENT).nIn(10).nOut(5)
                                        .build(), "1")
                        .setOutputs("2").backprop(true).pretrain(false).setInputTypes(InputType.feedForward(10))
                        .build();

        int nParams = cgs.numParameters();
        assertEquals(0, nParams);

        ComputationGraphConfiguration conf = cgs.getValue(new double[0]).getConfiguration();

        assertEquals(expected, conf);
    }

    @Test
    public void testBasic2() {

        ComputationGraphSpace mls = new ComputationGraphSpace.Builder()
                        .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                        .l2(new ContinuousParameterSpace(0.2, 0.5))
                        .addInputs("in").addLayer("0",
                                        new DenseLayerSpace.Builder().nIn(10).nOut(10)
                                                        .activation(new DiscreteParameterSpace<>(Activation.RELU,
                                                                        Activation.TANH))
                                                        .build(),
                                        "in")
                        .addLayer("1", new OutputLayerSpace.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX)
                                        .build(), "0")
                        .setOutputs("1").setInputTypes(InputType.feedForward(10)).pretrain(false).backprop(true)
                        .build();

        int nParams = mls.numParameters();
        assertEquals(3, nParams);

        //Assign numbers to each leaf ParameterSpace object (normally done by candidate generator)
        List<ParameterSpace> noDuplicatesList = LeafUtils.getUniqueObjects(mls.collectLeaves());

        //Second: assign each a number
        int c = 0;
        for (ParameterSpace ps : noDuplicatesList) {
            int np = ps.numParameters();
            if (np == 1) {
                ps.setIndices(c++);
            } else {
                int[] values = new int[np];
                for (int j = 0; j < np; j++)
                    values[c++] = j;
                ps.setIndices(values);
            }
        }

        int reluCount = 0;
        int tanhCount = 0;

        Random r = new Random(12345);

        for (int i = 0; i < 50; i++) {

            double[] rvs = new double[nParams];
            for (int j = 0; j < rvs.length; j++)
                rvs[j] = r.nextDouble();


            ComputationGraphConfiguration conf = mls.getValue(rvs).getConfiguration();
            assertEquals(false, conf.isPretrain());
            assertEquals(true, conf.isBackprop());

            int nLayers = conf.getVertexInputs().size();
            assertEquals(2, nLayers);

            for (int j = 0; j < nLayers; j++) {
                NeuralNetConfiguration layerConf =
                                ((LayerVertex) conf.getVertices().get(String.valueOf(j))).getLayerConf();

                double lr = ((Sgd)((BaseLayer) layerConf.getLayer()).getIUpdater()).getLearningRate();
                assertTrue(lr >= 0.0001 && lr <= 0.1);
                double l2 = ((BaseLayer) layerConf.getLayer()).getL2();
                assertTrue(l2 >= 0.2 && l2 <= 0.5);

                if (j == nLayers - 1) { //Output layer
                    assertEquals(Activation.SOFTMAX.getActivationFunction(),
                            ((BaseLayer) layerConf.getLayer()).getActivationFn());
                } else {
                    IActivation actFn = ((BaseLayer) layerConf.getLayer()).getActivationFn();
                    assertTrue(Activation.RELU.getActivationFunction().equals(actFn) ||
                            Activation.TANH.getActivationFunction().equals(actFn));
                    if (Activation.RELU.getActivationFunction().equals(actFn))
                        reluCount++;
                    else
                        tanhCount++;
                }
            }
        }

        System.out.println("ReLU vs. Tanh: " + reluCount + "\t" + tanhCount);
        assertTrue(reluCount > 0);
        assertTrue(tanhCount > 0);

    }


}
