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
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class CacheModeTest extends BaseDL4JTest {

    @Test
    public void testConvCacheModeSimple(){

        MultiLayerConfiguration conf1 = getConf(CacheMode.NONE);
        MultiLayerConfiguration conf2 = getConf(CacheMode.DEVICE);

        MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
        net1.init();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        INDArray in = Nd4j.rand(3, 28*28);
        INDArray labels = TestUtils.randomOneHot(3, 10);

        INDArray out1 = net1.output(in);
        INDArray out2 = net2.output(in);
        assertEquals(out1, out2);

        assertEquals(net1.params(), net2.params());
        net1.fit(in, labels);
        net2.fit(in, labels);
        assertEquals(net1.params(), net2.params());
    }

    private static MultiLayerConfiguration getConf(CacheMode cacheMode){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(12345)
                .cacheMode(cacheMode)
                .list()
                .layer(new ConvolutionLayer.Builder().nOut(3).build())
                .layer(new ConvolutionLayer.Builder().nOut(3).build())
                .layer(new OutputLayer.Builder().nOut(10).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        return conf;
    }

    @Test
    public void testLSTMCacheModeSimple(){

        for(boolean graves : new boolean[]{true, false}) {

            MultiLayerConfiguration conf1 = getConfLSTM(CacheMode.NONE, graves);
            MultiLayerConfiguration conf2 = getConfLSTM(CacheMode.DEVICE, graves);

            MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
            net1.init();
            MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
            net2.init();

            INDArray in = Nd4j.rand(new int[]{3, 3, 10});
            INDArray labels = TestUtils.randomOneHotTimeSeries(3, 10, 10);

            INDArray out1 = net1.output(in);
            INDArray out2 = net2.output(in);
            assertEquals(out1, out2);

            assertEquals(net1.params(), net2.params());
            net1.fit(in, labels);
            net2.fit(in, labels);
            assertEquals(net1.params(), net2.params());
        }
    }

    private static MultiLayerConfiguration getConfLSTM(CacheMode cacheMode, boolean graves){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(12345)
                .cacheMode(cacheMode)
                .list()
                .layer(graves ?
                        new GravesLSTM.Builder().nIn(3).nOut(3).build() :
                        new LSTM.Builder().nIn(3).nOut(3).build())
                .layer(graves ?
                        new GravesLSTM.Builder().nIn(3).nOut(3).build() :
                        new LSTM.Builder().nIn(3).nOut(3).build())
                .layer(new RnnOutputLayer.Builder().nOut(10).build())
                .build();

        return conf;
    }


    @Test
    public void testConvCacheModeSimpleCG(){

        ComputationGraphConfiguration conf1 = getConfCG(CacheMode.NONE);
        ComputationGraphConfiguration conf2 = getConfCG(CacheMode.DEVICE);

        ComputationGraph net1 = new ComputationGraph(conf1);
        net1.init();
        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();

        INDArray in = Nd4j.rand(3, 28*28);
        INDArray labels = TestUtils.randomOneHot(3, 10);

        INDArray out1 = net1.outputSingle(in);
        INDArray out2 = net2.outputSingle(in);
        assertEquals(out1, out2);

        assertEquals(net1.params(), net2.params());
        net1.fit(new DataSet(in, labels));
        net2.fit(new DataSet(in, labels));
        assertEquals(net1.params(), net2.params());
    }

    private static ComputationGraphConfiguration getConfCG(CacheMode cacheMode){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(12345)
                .cacheMode(cacheMode)
                .graphBuilder()
                .addInputs("in")
                .layer("0", new ConvolutionLayer.Builder().nOut(3).build(), "in")
                .layer("1", new ConvolutionLayer.Builder().nOut(3).build(), "0")
                .layer("2", new OutputLayer.Builder().nOut(10).build(), "1")
                .setOutputs("2")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .build();

        return conf;
    }

}
