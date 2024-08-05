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

package org.eclipse.deeplearning4j.dl4jcore.nn.multilayer;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
public class TestSetGetParameters extends BaseDL4JTest {

    @Test
    public void testSetParameters() {
        //Set up a MLN, then do set(get) on parameters. Results should be identical compared to before doing this.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(9).nOut(10)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(1, new DenseLayer.Builder().nIn(10).nOut(11)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(2, new AutoEncoder.Builder().corruptionLevel(0.5).nIn(11).nOut(12)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(3, new OutputLayer.Builder(LossFunction.MSE).nIn(12).nOut(12)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray initParams = net.params().dup();
        Map<String, INDArray> initParams2 = net.paramTable();

        net.setParams(net.params());

        INDArray initParamsAfter = net.params();
        Map<String, INDArray> initParams2After = net.paramTable();

        for (String s : initParams2.keySet()) {
            assertTrue( initParams2.get(s).equals(initParams2After.get(s)),"Params differ: " + s);
        }

        assertEquals(initParams, initParamsAfter);

        //Now, try the other way: get(set(random))
        INDArray randomParams = Nd4j.rand(initParams.dataType(), initParams.shape());
        net.setParams(randomParams.dup());

        assertEquals(net.params(), randomParams);
    }

    @Test
    public void testSetParametersRNN() {
        //Set up a MLN, then do set(get) on parameters. Results should be identical compared to before doing this.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new LSTM.Builder().nIn(9).nOut(10)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(1, new LSTM.Builder().nIn(10).nOut(11)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
                                        .dist(new NormalDistribution(0, 1)).nIn(11).nOut(12).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray initParams = net.params().dup();
        Map<String, INDArray> initParams2 = net.paramTable();

        net.setParams(net.params());

        INDArray initParamsAfter = net.params();
        Map<String, INDArray> initParams2After = net.paramTable();

        for (String s : initParams2.keySet()) {
            assertTrue( initParams2.get(s).equals(initParams2After.get(s)),"Params differ: " + s);
        }

        assertEquals(initParams, initParamsAfter);

        //Now, try the other way: get(set(random))
        INDArray randomParams = Nd4j.rand(initParams.dataType(), initParams.shape());
        net.setParams(randomParams.dup());

        assertEquals(net.params(), randomParams);
    }


}
