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

package org.deeplearning4j.nn.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Map;

import static org.junit.Assert.*;

public class TestSetGetParameters extends BaseDL4JTest {

    @Test
    public void testInitWithParamsCG() {

        Nd4j.getRandom().setSeed(12345);

        //Create configuration. Doesn't matter if this doesn't actually work for forward/backward pass here
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                        .addInputs("in").addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("1", new GravesLSTM.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("2", new GravesBidirectionalLSTM.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("3", new ConvolutionLayer.Builder().nIn(10).nOut(10).kernelSize(2, 2).stride(2, 2)
                                        .padding(2, 2).build(), "in")
                        .addLayer("4", new OutputLayer.Builder(LossFunction.MCXENT).nIn(10).nOut(10).build(), "3")
                        .addLayer("5", new OutputLayer.Builder(LossFunction.MCXENT).nIn(10).nOut(10).build(), "0")
                        .addLayer("6", new RnnOutputLayer.Builder(LossFunction.MCXENT).nIn(10).nOut(10).build(), "1",
                                        "2")
                        .setOutputs("4", "5", "6").pretrain(false).backprop(true).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        INDArray params = net.params();


        ComputationGraph net2 = new ComputationGraph(conf);
        net2.init(params, true);

        ComputationGraph net3 = new ComputationGraph(conf);
        net3.init(params, false);

        assertEquals(params, net2.params());
        assertEquals(params, net3.params());

        assertFalse(params == net2.params()); //Different objects due to clone
        assertTrue(params == net3.params()); //Same object due to clone


        Map<String, INDArray> paramsMap = net.paramTable();
        Map<String, INDArray> paramsMap2 = net2.paramTable();
        Map<String, INDArray> paramsMap3 = net3.paramTable();
        for (String s : paramsMap.keySet()) {
            assertEquals(paramsMap.get(s), paramsMap2.get(s));
            assertEquals(paramsMap.get(s), paramsMap3.get(s));
        }
    }
}
