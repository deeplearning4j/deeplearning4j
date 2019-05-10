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

package org.deeplearning4j.nn.layers.custom;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.custom.testclasses.CustomLayer;
import org.deeplearning4j.nn.layers.custom.testclasses.CustomOutputLayer;
import org.deeplearning4j.nn.layers.custom.testclasses.CustomOutputLayerImpl;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 26/08/2016.
 */
public class TestCustomLayers extends BaseDL4JTest {

    @Test
    public void testJsonMultiLayerNetwork() {
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                                        .layer(1, new CustomLayer(3.14159)).layer(2,
                                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                                        .build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        MultiLayerConfiguration confFromYaml = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);
    }

    @Test
    public void testJsonComputationGraph() {
        //ComputationGraph with a custom layer; check JSON and YAML config actually works...

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder()
                        .addInputs("in").addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("1", new CustomLayer(3.14159), "0").addLayer("2",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                                                .nIn(10).nOut(10).build(),
                                        "1")
                        .setOutputs("2").build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        ComputationGraphConfiguration confFromJson = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        ComputationGraphConfiguration confFromYaml = ComputationGraphConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);
    }


    @Test
    public void checkInitializationFF() {
        //Actually create a network with a custom layer; check initialization and forward pass

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(9).nOut(10).build()).layer(1, new CustomLayer(3.14159)) //hard-coded nIn/nOut of 10
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10).nOut(11).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(9 * 10 + 10, net.getLayer(0).numParams());
        assertEquals(10 * 10 + 10, net.getLayer(1).numParams());
        assertEquals(10 * 11 + 11, net.getLayer(2).numParams());

        //Check for exceptions...
        net.output(Nd4j.rand(1, 9));
        net.fit(new DataSet(Nd4j.rand(1, 9), Nd4j.rand(1, 11)));
    }



    @Test
    public void testCustomOutputLayerMLN() {
        //Second: let's create a MultiLayerCofiguration with one, and check JSON and YAML config actually works...
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345).list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                                        .layer(1, new CustomOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX)
                                                        .nIn(10).nOut(10).build())
                                        .build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        MultiLayerConfiguration confFromYaml = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);

        //Third: check initialization
        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertTrue(net.getLayer(1) instanceof CustomOutputLayerImpl);

        //Fourth: compare to an equivalent standard output layer (should be identical)
        MultiLayerConfiguration conf2 =
                        new NeuralNetConfiguration.Builder().seed(12345).weightInit(WeightInit.XAVIER)
                                        .list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build()).layer(1,
                                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                                        .build();
        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        assertEquals(net2.params(), net.params());

        INDArray testFeatures = Nd4j.rand(1, 10);
        INDArray testLabels = Nd4j.zeros(1, 10);
        testLabels.putScalar(0, 3, 1.0);
        DataSet ds = new DataSet(testFeatures, testLabels);

        assertEquals(net2.output(testFeatures), net.output(testFeatures));
        assertEquals(net2.score(ds), net.score(ds), 1e-6);
    }


    @Test
    public void testCustomOutputLayerCG() {
        //Create a ComputationGraphConfiguration with custom output layer, and check JSON and YAML config actually works...
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in").addLayer("1",
                                        new CustomOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(10)
                                                        .nOut(10).activation(Activation.SOFTMAX).build(),
                                        "0")
                        .setOutputs("1").build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        ComputationGraphConfiguration confFromJson = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        ComputationGraphConfiguration confFromYaml = ComputationGraphConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);

        //Third: check initialization
        Nd4j.getRandom().setSeed(12345);
        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        assertTrue(net.getLayer(1) instanceof CustomOutputLayerImpl);

        //Fourth: compare to an equivalent standard output layer (should be identical)
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in").addLayer("1",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(10).nOut(10)
                                                .activation(Activation.SOFTMAX).build(),
                                        "0")
                        .setOutputs("1").build();
        Nd4j.getRandom().setSeed(12345);
        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();

        assertEquals(net2.params(), net.params());

        INDArray testFeatures = Nd4j.rand(1, 10);
        INDArray testLabels = Nd4j.zeros(1, 10);
        testLabels.putScalar(0, 3, 1.0);
        DataSet ds = new DataSet(testFeatures, testLabels);

        assertEquals(net2.output(testFeatures)[0], net.output(testFeatures)[0]);
        assertEquals(net2.score(ds), net.score(ds), 1e-6);
    }
}
