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

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestDropout extends BaseDL4JTest {

    @Test
    public void testDropoutSimple() throws Exception {
        //Testing dropout with a single layer
        //Layer input: values should be set to either 0.0 or 2.0x original value

        int nIn = 8;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new Sgd())
                        .dropOut(0.5).list()
                        .layer(0, new OutputLayer.Builder().activation(Activation.IDENTITY)
                                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(nIn).nOut(nOut)
                                        .weightInit(WeightInit.XAVIER).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.getLayer(0).getParam("W").assign(Nd4j.eye(nIn));

        int nTests = 15;

        Nd4j.getRandom().setSeed(12345);
        int noDropoutCount = 0;
        for (int i = 0; i < nTests; i++) {
            INDArray in = Nd4j.rand(1, nIn);
            INDArray out = Nd4j.rand(1, nOut);
            INDArray inCopy = in.dup();

            List<INDArray> l = net.feedForward(in, true);

            INDArray postDropout = l.get(l.size() - 1);
            //Dropout occurred. Expect inputs to be either scaled 2x original, or set to 0.0 (with dropout = 0.5)
            for (int j = 0; j < inCopy.length(); j++) {
                double origValue = inCopy.getDouble(j);
                double doValue = postDropout.getDouble(j);
                if (doValue > 0.0) {
                    //Input was kept -> should be scaled by factor of (1.0/0.5 = 2)
                    assertEquals(origValue * 2.0, doValue, 0.0001);
                }
            }

            //Do forward pass
            //(1) ensure dropout ISN'T being applied for forward pass at test time
            //(2) ensure dropout ISN'T being applied for test time scoring
            //If dropout is applied at test time: outputs + score will differ between passes
            INDArray in2 = Nd4j.rand(1, nIn);
            INDArray out2 = Nd4j.rand(1, nOut);
            INDArray outTest1 = net.output(in2, false);
            INDArray outTest2 = net.output(in2, false);
            INDArray outTest3 = net.output(in2, false);
            assertEquals(outTest1, outTest2);
            assertEquals(outTest1, outTest3);

            double score1 = net.score(new DataSet(in2, out2), false);
            double score2 = net.score(new DataSet(in2, out2), false);
            double score3 = net.score(new DataSet(in2, out2), false);
            assertEquals(score1, score2, 0.0);
            assertEquals(score1, score3, 0.0);
        }

        if (noDropoutCount >= nTests / 3) {
            //at 0.5 dropout ratio and more than a few inputs, expect only a very small number of instances where
            //no dropout occurs, just due to random chance
            fail("Too many instances of dropout not being applied");
        }
    }
}
