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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 2/5/17.
 */
@Slf4j
public class FrozenLayerTest extends BaseDL4JTest {

    /*
        A model with a few frozen layers ==
            Model with non frozen layers set with the output of the forward pass of the frozen layers
     */
    @Test
    public void testFrozen() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                        .activation(Activation.IDENTITY);

        FineTuneConfiguration finetune = new FineTuneConfiguration.Builder().updater(new Sgd(0.1)).build();

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.clone().list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(3).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build());

        modelToFineTune.init();
        List<INDArray> ff = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false);
        INDArray asFrozenFeatures = ff.get(2);

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune).fineTuneConfiguration(finetune)
                        .setFeatureExtractor(1).build();

        INDArray paramsLastTwoLayers =
                        Nd4j.hstack(modelToFineTune.getLayer(2).params(), modelToFineTune.getLayer(3).params());
        MultiLayerNetwork notFrozen = new MultiLayerNetwork(overallConf.clone().list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(3).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build(), paramsLastTwoLayers);

        //        assertEquals(modelNow.getLayer(2).conf(), notFrozen.getLayer(0).conf());  //Equal, other than names
        //        assertEquals(modelNow.getLayer(3).conf(), notFrozen.getLayer(1).conf());  //Equal, other than names

        //Check: forward pass
        INDArray outNow = modelNow.output(randomData.getFeatures());
        INDArray outNotFrozen = notFrozen.output(asFrozenFeatures);
        assertEquals(outNow, outNotFrozen);

        for (int i = 0; i < 5; i++) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
        }

        INDArray expected = Nd4j.hstack(modelToFineTune.getLayer(0).params(), modelToFineTune.getLayer(1).params(),
                        notFrozen.params());
        INDArray act = modelNow.params();
        assertEquals(expected, act);
    }


    @Test
    public void cloneMLNFrozen() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                        .activation(Activation.IDENTITY);
        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(3).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build());

        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false).get(2);
        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune).setFeatureExtractor(1).build();

        MultiLayerNetwork clonedModel = modelNow.clone();

        //Check json
        assertEquals(modelNow.getLayerWiseConfigurations().toJson(), clonedModel.getLayerWiseConfigurations().toJson());

        //Check params
        assertEquals(modelNow.params(), clonedModel.params());

        MultiLayerNetwork notFrozen = new MultiLayerNetwork(
                        overallConf.list().layer(0, new DenseLayer.Builder().nIn(2).nOut(3).build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build())
                                        .build(),
                        Nd4j.hstack(modelToFineTune.getLayer(2).params(), modelToFineTune.getLayer(3).params()));

        int i = 0;
        while (i < 5) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
            clonedModel.fit(randomData);
            i++;
        }

        INDArray expectedParams = Nd4j.hstack(modelToFineTune.getLayer(0).params(),
                        modelToFineTune.getLayer(1).params(), notFrozen.params());
        assertEquals(expectedParams, modelNow.params());
        assertEquals(expectedParams, clonedModel.params());

    }


    @Test
    public void testFrozenCompGraph() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                        .activation(Activation.IDENTITY);

        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder().addInputs("layer0In")
                        .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).build(), "layer0In")
                        .addLayer("layer1", new DenseLayer.Builder().nIn(3).nOut(2).build(), "layer0")
                        .addLayer("layer2", new DenseLayer.Builder().nIn(2).nOut(3).build(), "layer1")
                        .addLayer("layer3",
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build(),
                                        "layer2")
                        .setOutputs("layer3").build());

        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForward(randomData.getFeatures(), false).get("layer1");

        ComputationGraph modelNow =
                        new TransferLearning.GraphBuilder(modelToFineTune).setFeatureExtractor("layer1").build();

        ComputationGraph notFrozen = new ComputationGraph(overallConf.graphBuilder().addInputs("layer0In")
                        .addLayer("layer0", new DenseLayer.Builder().nIn(2).nOut(3).build(), "layer0In")
                        .addLayer("layer1",
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build(),
                                        "layer0")
                        .setOutputs("layer1").build());

        notFrozen.init();
        notFrozen.setParams(Nd4j.hstack(modelToFineTune.getLayer("layer2").params(),
                        modelToFineTune.getLayer("layer3").params()));

        int i = 0;
        while (i < 5) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
            i++;
        }

        assertEquals(Nd4j.hstack(modelToFineTune.getLayer("layer0").params(),
                        modelToFineTune.getLayer("layer1").params(), notFrozen.params()), modelNow.params());
    }

    @Test
    public void cloneCompGraphFrozen() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                        .activation(Activation.IDENTITY);

        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder().addInputs("layer0In")
                        .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).build(), "layer0In")
                        .addLayer("layer1", new DenseLayer.Builder().nIn(3).nOut(2).build(), "layer0")
                        .addLayer("layer2", new DenseLayer.Builder().nIn(2).nOut(3).build(), "layer1")
                        .addLayer("layer3",
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build(),
                                        "layer2")
                        .setOutputs("layer3").build());

        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForward(randomData.getFeatures(), false).get("layer1");
        ComputationGraph modelNow =
                        new TransferLearning.GraphBuilder(modelToFineTune).setFeatureExtractor("layer1").build();

        ComputationGraph clonedModel = modelNow.clone();

        //Check json
        assertEquals(clonedModel.getConfiguration().toJson(), modelNow.getConfiguration().toJson());

        //Check params
        assertEquals(modelNow.params(), clonedModel.params());

        ComputationGraph notFrozen = new ComputationGraph(overallConf.graphBuilder().addInputs("layer0In")
                        .addLayer("layer0", new DenseLayer.Builder().nIn(2).nOut(3).build(), "layer0In")
                        .addLayer("layer1",
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build(),
                                        "layer0")
                        .setOutputs("layer1").build());
        notFrozen.init();
        notFrozen.setParams(Nd4j.hstack(modelToFineTune.getLayer("layer2").params(),
                        modelToFineTune.getLayer("layer3").params()));


        int i = 0;
        while (i < 5) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
            clonedModel.fit(randomData);
            i++;
        }

        INDArray expectedParams = Nd4j.hstack(modelToFineTune.getLayer("layer0").params(),
                        modelToFineTune.getLayer("layer1").params(), notFrozen.params());
        assertEquals(expectedParams, modelNow.params());
        assertEquals(expectedParams, clonedModel.params());
    }


    @Test
    public void testFrozenLayerInstantiation() {
        //We need to be able to instantitate frozen layers from JSON etc, and have them be the same as if
        // they were initialized via the builder
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder().seed(12345).list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                                                        .nOut(10).build())
                        .build();

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).list().layer(0,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayer(new DenseLayer.Builder().nIn(10).nOut(10)
                                        .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build()))
                        .layer(1, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayer(
                                        new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                                        .weightInit(WeightInit.XAVIER).build()))
                        .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                                                        .nOut(10).build())
                        .build();

        MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
        net1.init();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        assertEquals(net1.params(), net2.params());


        String json = conf2.toJson();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(json);

        assertEquals(conf2, fromJson);

        MultiLayerNetwork net3 = new MultiLayerNetwork(fromJson);
        net3.init();

        INDArray input = Nd4j.rand(10, 10);

        INDArray out2 = net2.output(input);
        INDArray out3 = net3.output(input);

        assertEquals(out2, out3);
    }

    @Test
    public void testFrozenLayerInstantiationCompGraph() {

        //We need to be able to instantitate frozen layers from JSON etc, and have them be the same as if
        // they were initialized via the builder
        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                        .weightInit(WeightInit.XAVIER).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                        .weightInit(WeightInit.XAVIER).build(), "0")
                        .addLayer("2", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                                                        .nOut(10).build(),
                                        "1")
                        .setOutputs("2").build();

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.misc.FrozenLayer.Builder()
                                        .layer(new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                                        .weightInit(WeightInit.XAVIER).build())
                                        .build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.misc.FrozenLayer.Builder()
                                        .layer(new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                                        .weightInit(WeightInit.XAVIER).build())
                                        .build(), "0")
                        .addLayer("2", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                                                        .nOut(10).build(),
                                        "1")
                        .setOutputs("2").build();

        ComputationGraph net1 = new ComputationGraph(conf1);
        net1.init();
        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();

        assertEquals(net1.params(), net2.params());


        String json = conf2.toJson();
        ComputationGraphConfiguration fromJson = ComputationGraphConfiguration.fromJson(json);

        assertEquals(conf2, fromJson);

        ComputationGraph net3 = new ComputationGraph(fromJson);
        net3.init();

        INDArray input = Nd4j.rand(10, 10);

        INDArray out2 = net2.outputSingle(input);
        INDArray out3 = net3.outputSingle(input);

        assertEquals(out2, out3);
    }
}
