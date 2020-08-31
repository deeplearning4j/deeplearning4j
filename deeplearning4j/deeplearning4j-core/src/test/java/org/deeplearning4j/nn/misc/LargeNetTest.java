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

package org.deeplearning4j.nn.misc;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Ignore //Ignored due to very large memory requirements
public class LargeNetTest extends BaseDL4JTest {

    @Ignore
    @Test
    public void testLargeMultiLayerNetwork(){
        Nd4j.setDataType(DataType.FLOAT);

        //More than 2.1 billion parameters
        //10M classes plus 300 vector size -> 3 billion elements

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new EmbeddingLayer.Builder().nIn(10_000_000).nOut(300).build())
                .layer(new OutputLayer.Builder().nIn(300).nOut(10).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray params = net.params();
        long paramsLength = params.length();
        long expParamsLength = 10_000_000L * 300 + 300 * 10 + 10;
        assertEquals(expParamsLength, paramsLength);

        long[] expW = new long[]{10_000_000, 300};
        assertArrayEquals(expW, net.getParam("0_W").shape());

        long[] expW1 = new long[]{300, 10};
        assertArrayEquals(expW1, net.getParam("1_W").shape());

        long[] expB1 = new long[]{1, 10};
        assertArrayEquals(expB1, net.getParam("1_b").shape());
    }

    @Ignore
    @Test
    public void testLargeCompGraph(){
        Nd4j.setDataType(DataType.FLOAT);

        //More than 2.1 billion parameters
        //10M classes plus 300 vector size -> 3 billion elements

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("0", new EmbeddingLayer.Builder().nIn(10_000_000).nOut(300).build(), "in")
                .layer("1", new OutputLayer.Builder().nIn(300).nOut(10).activation(Activation.SOFTMAX).build(), "0")
                .setOutputs("1")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray params = net.params();
        long paramsLength = params.length();
        long expParamsLength = 10_000_000L * 300 + 300 * 10 + 10;
        assertEquals(expParamsLength, paramsLength);

        long[] expW = new long[]{10_000_000, 300};
        assertArrayEquals(expW, net.getParam("0_W").shape());

        long[] expW1 = new long[]{300, 10};
        assertArrayEquals(expW1, net.getParam("1_W").shape());

        long[] expB1 = new long[]{1, 10};
        assertArrayEquals(expB1, net.getParam("1_b").shape());
    }
}
