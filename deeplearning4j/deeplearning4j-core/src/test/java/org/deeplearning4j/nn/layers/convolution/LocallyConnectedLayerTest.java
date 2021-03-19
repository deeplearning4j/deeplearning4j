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
package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.Arrays;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * @author Max Pumperla
 */
@DisplayName("Locally Connected Layer Test")
class LocallyConnectedLayerTest extends BaseDL4JTest {

    @BeforeEach
    void before() {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.factory().setDType(DataType.DOUBLE);
        Nd4j.EPS_THRESHOLD = 1e-4;
    }

    @Test
    @DisplayName("Test 2 d Forward")
    void test2dForward() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4).updater(new Nesterovs(0.9)).dropOut(0.5).list().layer(new LocallyConnected2D.Builder().kernelSize(8, 8).nIn(3).stride(4, 4).nOut(16).dropOut(0.5).convolutionMode(ConvolutionMode.Strict).setInputSize(28, 28).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build()).layer(// output layer
        new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutionalFlat(28, 28, 3));
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray input = Nd4j.ones(10, 3, 28, 28);
        INDArray output = network.output(input, false);
        assertArrayEquals(new long[] { 10, 10 }, output.shape());
    }

    @Test
    @DisplayName("Test 1 d Forward")
    void test1dForward() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4).updater(new Nesterovs(0.9)).dropOut(0.5).list().layer(new LocallyConnected1D.Builder().kernelSize(4).nIn(3).stride(1).nOut(16).dropOut(0.5).convolutionMode(ConvolutionMode.Strict).setInputSize(28).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build()).layer(// output layer
        new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.recurrent(3, 8));
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray input = Nd4j.ones(10, 3, 8);
        INDArray output = network.output(input, false);
        ;
        for (int i = 0; i < 100; i++) {
            // TODO: this falls flat for 1000 iterations on my machine
            output = network.output(input, false);
        }
        assertArrayEquals(new long[] { (8 - 4 + 1) * 10, 10 }, output.shape());
        network.fit(input, output);
    }

    @Test
    @DisplayName("Test Locally Connected")
    void testLocallyConnected() {
        for (DataType globalDtype : new DataType[] { DataType.DOUBLE, DataType.FLOAT, DataType.HALF }) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[] { DataType.DOUBLE, DataType.FLOAT, DataType.HALF }) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());
                for (int test = 0; test < 2; test++) {
                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", test=" + test;
                    ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder().dataType(networkDtype).seed(123).updater(new NoOp()).weightInit(WeightInit.XAVIER).convolutionMode(ConvolutionMode.Same).graphBuilder();
                    INDArray[] in;
                    INDArray label;
                    switch(test) {
                        case 0:
                            b.addInputs("in").addLayer("1", new LSTM.Builder().nOut(5).build(), "in").addLayer("2", new LocallyConnected1D.Builder().kernelSize(2).nOut(4).build(), "1").addLayer("out", new RnnOutputLayer.Builder().nOut(10).build(), "2").setOutputs("out").setInputTypes(InputType.recurrent(5, 4));
                            in = new INDArray[] { Nd4j.rand(networkDtype, 2, 5, 4) };
                            label = TestUtils.randomOneHotTimeSeries(2, 10, 4).castTo(networkDtype);
                            break;
                        case 1:
                            b.addInputs("in").addLayer("1", new ConvolutionLayer.Builder().kernelSize(2, 2).nOut(5).convolutionMode(ConvolutionMode.Same).build(), "in").addLayer("2", new LocallyConnected2D.Builder().kernelSize(2, 2).nOut(5).build(), "1").addLayer("out", new OutputLayer.Builder().nOut(10).build(), "2").setOutputs("out").setInputTypes(InputType.convolutional(8, 8, 1));
                            in = new INDArray[] { Nd4j.rand(networkDtype, 2, 1, 8, 8) };
                            label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                            break;
                        default:
                            throw new RuntimeException();
                    }
                    ComputationGraph net = new ComputationGraph(b.build());
                    net.init();
                    INDArray out = net.outputSingle(in);
                    assertEquals(networkDtype, out.dataType(),msg);
                    Map<String, INDArray> ff = net.feedForward(in, false);
                    for (Map.Entry<String, INDArray> e : ff.entrySet()) {
                        if (e.getKey().equals("in"))
                            continue;
                        String s = msg + " - layer: " + e.getKey();
                        assertEquals( networkDtype, e.getValue().dataType(),s);
                    }
                    net.setInputs(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();
                    net.fit(new MultiDataSet(in, new INDArray[] { label }));
                }
            }
        }
    }
}
