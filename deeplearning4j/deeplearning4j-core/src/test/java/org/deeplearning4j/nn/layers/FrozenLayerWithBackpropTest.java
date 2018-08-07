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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by Ugljesa Jovanovic (jovanovic.ugljesa@gmail.com) on 06/05/2018.
 * Reused instantiation tests from {@link FrozenLayerTest}
 */
@Slf4j
public class FrozenLayerWithBackpropTest extends BaseDL4JTest {

    @Test
    public void testFrozenWithBackpropLayerInstantiation() {
        //We need to be able to instantitate frozen layers from JSON etc, and have them be the same as if
        // they were initialized via the builder
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder().seed(12345).list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                        .nOut(10).build())
                .build();

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).list().layer(0,
                new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(new DenseLayer.Builder().nIn(10).nOut(10)
                        .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build()))
                .layer(1, new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                        new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                .weightInit(WeightInit.XAVIER).build()))
                .layer(2, new OutputLayer.Builder(
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
                .addLayer("2", new OutputLayer.Builder(
                                LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10)
                                .nOut(10).build(),
                        "1")
                .setOutputs("2").build();

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                .addInputs("in")
                .addLayer("0", new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                        new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                .weightInit(WeightInit.XAVIER).build()), "in")
                .addLayer("1", new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                        new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                .weightInit(WeightInit.XAVIER).build()), "0")
                .addLayer("2", new OutputLayer.Builder(
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

    @Test
    public void testMultiLayerNetworkFrozenLayerParamsAfterBackprop() {

        DataSet randomData = new DataSet(Nd4j.rand(100, 4, 12345), Nd4j.rand(100, 1, 12345));

        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(2))
                .list()
                .layer(0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build()
                )
                .layer(1,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(3)
                                        .nOut(4)
                                        .build()
                        )
                )
                .layer(2,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(4)
                                        .nOut(2)
                                        .build()
                        )
                ).layer(3,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                        .nIn(2)
                                        .nOut(1)
                                        .build()
                        )
                )
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf1);
        network.init();
        INDArray unfrozenLayerParams = network.getLayer(0).params().dup();
        INDArray frozenLayerParams1 = network.getLayer(1).params().dup();
        INDArray frozenLayerParams2 = network.getLayer(2).params().dup();
        INDArray frozenOutputLayerParams = network.getLayer(3).params().dup();

        for (int i = 0; i < 100; i++) {
            network.fit(randomData);
        }

        assertNotEquals(unfrozenLayerParams, network.getLayer(0).params());
        assertEquals(frozenLayerParams1, network.getLayer(1).params());
        assertEquals(frozenLayerParams2, network.getLayer(2).params());
        assertEquals(frozenOutputLayerParams, network.getLayer(3).params());

    }

    @Test
    public void testComputationGraphFrozenLayerParamsAfterBackprop() {

        DataSet randomData = new DataSet(Nd4j.rand(100, 4,12345), Nd4j.rand(100, 1, 12345));
        String frozenBranchName = "B1-";
        String unfrozenBranchName = "B2-";

        String initialLayer = "initial";

        String frozenBranchUnfrozenLayer0 = frozenBranchName + "0";
        String frozenBranchFrozenLayer1 = frozenBranchName + "1";
        String frozenBranchFrozenLayer2 = frozenBranchName + "2";
        String frozenBranchOutput = frozenBranchName + "Output";


        String unfrozenLayer0 = unfrozenBranchName + "0";
        String unfrozenLayer1 = unfrozenBranchName + "1";
        String unfrozenBranch2 = unfrozenBranchName + "Output";

        ComputationGraphConfiguration computationGraphConf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(2.0))
                .seed(12345)
                .graphBuilder()
                .addInputs("input")
                .addLayer(initialLayer,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        "input"
                )
                .addLayer(frozenBranchUnfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build(),
                        initialLayer
                )
                .addLayer(frozenBranchFrozenLayer1,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(3)
                                        .nOut(4)
                                        .build()
                        ),
                        frozenBranchUnfrozenLayer0
                )
                .addLayer(frozenBranchFrozenLayer2,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(4)
                                        .nOut(2)
                                        .build()
                        ),
                        frozenBranchFrozenLayer1
                )
                .addLayer(unfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        initialLayer
                )
                .addLayer(unfrozenLayer1,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(2)
                                .build(),
                        unfrozenLayer0
                )
                .addLayer(unfrozenBranch2,
                        new DenseLayer.Builder()
                                .nIn(2)
                                .nOut(1)
                                .build(),
                        unfrozenLayer1
                )
                .addVertex("merge",
                        new MergeVertex(), frozenBranchFrozenLayer2, unfrozenBranch2)
                .addLayer(frozenBranchOutput,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                        .nIn(3)
                                        .nOut(1)
                                        .build()
                        ),
                        "merge"
                )
                .setOutputs(frozenBranchOutput)
                .build();

        ComputationGraph computationGraph = new ComputationGraph(computationGraphConf);
        computationGraph.init();
        INDArray unfrozenLayerParams = computationGraph.getLayer(frozenBranchUnfrozenLayer0).params().dup();
        INDArray frozenLayerParams1 = computationGraph.getLayer(frozenBranchFrozenLayer1).params().dup();
        INDArray frozenLayerParams2 = computationGraph.getLayer(frozenBranchFrozenLayer2).params().dup();
        INDArray frozenOutputLayerParams = computationGraph.getLayer(frozenBranchOutput).params().dup();

        for (int i = 0; i < 100; i++) {
            computationGraph.fit(randomData);
        }

        assertNotEquals(unfrozenLayerParams, computationGraph.getLayer(frozenBranchUnfrozenLayer0).params());
        assertEquals(frozenLayerParams1, computationGraph.getLayer(frozenBranchFrozenLayer1).params());
        assertEquals(frozenLayerParams2, computationGraph.getLayer(frozenBranchFrozenLayer2).params());
        assertEquals(frozenOutputLayerParams, computationGraph.getLayer(frozenBranchOutput).params());

    }

    /**
     * Frozen layer should have same results as a layer with Sgd updater with learning rate set to 0
     */
    @Test
    public void testFrozenLayerVsSgd() {
        DataSet randomData = new DataSet(Nd4j.rand(100, 4, 12345), Nd4j.rand(100, 1, 12345));

        MultiLayerConfiguration confSgd = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(2))
                .list()
                .layer(0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build()
                )
                .layer(1,
                        new DenseLayer.Builder()
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(3)
                                .nOut(4)
                                .build()
                ).layer(2,
                        new DenseLayer.Builder()
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(4)
                                .nOut(2)
                                .build()

                ).layer(3,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(2)
                                .nOut(1)
                                .build()
                )
                .build();

        MultiLayerConfiguration confFrozen = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(2))
                .list()
                .layer(0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build()
                )
                .layer(1,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(3)
                                        .nOut(4)
                                        .build()
                        )
                )
                .layer(2,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(4)
                                        .nOut(2)
                                        .build()
                        )
                ).layer(3,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                        .nIn(2)
                                        .nOut(1)
                                        .build()
                        )
                )
                .build();
        MultiLayerNetwork frozenNetwork = new MultiLayerNetwork(confFrozen);
        frozenNetwork.init();
        INDArray unfrozenLayerParams = frozenNetwork.getLayer(0).params().dup();
        INDArray frozenLayerParams1 = frozenNetwork.getLayer(1).params().dup();
        INDArray frozenLayerParams2 = frozenNetwork.getLayer(2).params().dup();
        INDArray frozenOutputLayerParams = frozenNetwork.getLayer(3).params().dup();

        MultiLayerNetwork sgdNetwork = new MultiLayerNetwork(confSgd);
        sgdNetwork.init();
        INDArray unfrozenSgdLayerParams = sgdNetwork.getLayer(0).params().dup();
        INDArray frozenSgdLayerParams1 = sgdNetwork.getLayer(1).params().dup();
        INDArray frozenSgdLayerParams2 = sgdNetwork.getLayer(2).params().dup();
        INDArray frozenSgdOutputLayerParams = sgdNetwork.getLayer(3).params().dup();

        for (int i = 0; i < 100; i++) {
            frozenNetwork.fit(randomData);
        }
        for (int i = 0; i < 100; i++) {
            sgdNetwork.fit(randomData);
        }

        assertEquals(frozenNetwork.getLayer(0).params(), sgdNetwork.getLayer(0).params());
        assertEquals(frozenNetwork.getLayer(1).params(), sgdNetwork.getLayer(1).params());
        assertEquals(frozenNetwork.getLayer(2).params(), sgdNetwork.getLayer(2).params());
        assertEquals(frozenNetwork.getLayer(3).params(), sgdNetwork.getLayer(3).params());

    }

    @Test
    public void testComputationGraphVsSgd() {

        DataSet randomData = new DataSet(Nd4j.rand(100, 4, 12345), Nd4j.rand(100, 1, 12345));
        String frozenBranchName = "B1-";
        String unfrozenBranchName = "B2-";

        String initialLayer = "initial";

        String frozenBranchUnfrozenLayer0 = frozenBranchName + "0";
        String frozenBranchFrozenLayer1 = frozenBranchName + "1";
        String frozenBranchFrozenLayer2 = frozenBranchName + "2";
        String frozenBranchOutput = frozenBranchName + "Output";


        String unfrozenLayer0 = unfrozenBranchName + "0";
        String unfrozenLayer1 = unfrozenBranchName + "1";
        String unfrozenBranch2 = unfrozenBranchName + "Output";

        ComputationGraphConfiguration computationGraphConf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(2.0))
                .seed(12345)
                .graphBuilder()
                .addInputs("input")
                .addLayer(initialLayer,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        "input"
                )
                .addLayer(frozenBranchUnfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build(),
                        initialLayer
                )
                .addLayer(frozenBranchFrozenLayer1,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(3)
                                        .nOut(4)
                                        .build()
                        ),
                        frozenBranchUnfrozenLayer0
                )
                .addLayer(frozenBranchFrozenLayer2,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new DenseLayer.Builder()
                                        .nIn(4)
                                        .nOut(2)
                                        .build()
                        ),
                        frozenBranchFrozenLayer1
                )
                .addLayer(unfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        initialLayer
                )
                .addLayer(unfrozenLayer1,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(2)
                                .build(),
                        unfrozenLayer0
                )
                .addLayer(unfrozenBranch2,
                        new DenseLayer.Builder()
                                .nIn(2)
                                .nOut(1)
                                .build(),
                        unfrozenLayer1
                )
                .addVertex("merge",
                        new MergeVertex(), frozenBranchFrozenLayer2, unfrozenBranch2)
                .addLayer(frozenBranchOutput,
                        new org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop(
                                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                        .nIn(3)
                                        .nOut(1)
                                        .build()
                        ),
                        "merge"
                )
                .setOutputs(frozenBranchOutput)
                .build();

        ComputationGraphConfiguration computationSgdGraphConf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(2.0))
                .seed(12345)
                .graphBuilder()
                .addInputs("input")
                .addLayer(initialLayer,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        "input"
                )
                .addLayer(frozenBranchUnfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(3)
                                .build(),
                        initialLayer
                )
                .addLayer(frozenBranchFrozenLayer1,
                        new DenseLayer.Builder()
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(3)
                                .nOut(4)
                                .build(),
                        frozenBranchUnfrozenLayer0
                )
                .addLayer(frozenBranchFrozenLayer2,
                        new DenseLayer.Builder()
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(4)
                                .nOut(2)
                                .build()
                        ,
                        frozenBranchFrozenLayer1
                )
                .addLayer(unfrozenLayer0,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(4)
                                .build(),
                        initialLayer
                )
                .addLayer(unfrozenLayer1,
                        new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(2)
                                .build(),
                        unfrozenLayer0
                )
                .addLayer(unfrozenBranch2,
                        new DenseLayer.Builder()
                                .nIn(2)
                                .nOut(1)
                                .build(),
                        unfrozenLayer1
                )
                .addVertex("merge",
                        new MergeVertex(), frozenBranchFrozenLayer2, unfrozenBranch2)
                .addLayer(frozenBranchOutput,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .updater(new Sgd(0.0))
                                .biasUpdater(new Sgd(0.0))
                                .nIn(3)
                                .nOut(1)
                                .build()
                        ,
                        "merge"
                )
                .setOutputs(frozenBranchOutput)
                .build();

        ComputationGraph frozenComputationGraph = new ComputationGraph(computationGraphConf);
        frozenComputationGraph.init();
        INDArray unfrozenLayerParams = frozenComputationGraph.getLayer(frozenBranchUnfrozenLayer0).params().dup();
        INDArray frozenLayerParams1 = frozenComputationGraph.getLayer(frozenBranchFrozenLayer1).params().dup();
        INDArray frozenLayerParams2 = frozenComputationGraph.getLayer(frozenBranchFrozenLayer2).params().dup();
        INDArray frozenOutputLayerParams = frozenComputationGraph.getLayer(frozenBranchOutput).params().dup();

        ComputationGraph sgdComputationGraph = new ComputationGraph(computationSgdGraphConf);
        sgdComputationGraph.init();
        INDArray unfrozenSgdLayerParams = sgdComputationGraph.getLayer(frozenBranchUnfrozenLayer0).params().dup();
        INDArray frozenSgdLayerParams1 = sgdComputationGraph.getLayer(frozenBranchFrozenLayer1).params().dup();
        INDArray frozenSgdLayerParams2 = sgdComputationGraph.getLayer(frozenBranchFrozenLayer2).params().dup();
        INDArray frozenSgdOutputLayerParams = sgdComputationGraph.getLayer(frozenBranchOutput).params().dup();

        for (int i = 0; i < 100; i++) {
            frozenComputationGraph.fit(randomData);
        }
        for (int i = 0; i < 100; i++) {
            sgdComputationGraph.fit(randomData);
        }

        assertEquals(frozenComputationGraph.getLayer(frozenBranchUnfrozenLayer0).params(), sgdComputationGraph.getLayer(frozenBranchUnfrozenLayer0).params());
        assertEquals(frozenComputationGraph.getLayer(frozenBranchFrozenLayer1).params(), sgdComputationGraph.getLayer(frozenBranchFrozenLayer1).params());
        assertEquals(frozenComputationGraph.getLayer(frozenBranchFrozenLayer2).params(), sgdComputationGraph.getLayer(frozenBranchFrozenLayer2).params());
        assertEquals(frozenComputationGraph.getLayer(frozenBranchOutput).params(), sgdComputationGraph.getLayer(frozenBranchOutput).params());

    }


}
