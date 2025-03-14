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
package org.eclipse.deeplearning4j.dl4jcore.nn.conf.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import java.util.HashMap;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertThrows;

@DisplayName("Layer Config Validation Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class LayerConfigValidationTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Drop Connect")
    void testDropConnect() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).weightNoise(new DropConnect(0.5)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    @DisplayName("Test L 1 L 2 Not Set")
    void testL1L2NotSet() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    @Disabled
    @DisplayName("Test Reg Not Set L 1 Global")
    void testRegNotSetL1Global() {
        assertThrows(IllegalStateException.class, () -> {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).l1(0.5).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
        });
    }



    @Test
    @DisplayName("Test Weight Init Dist Not Set")
    void testWeightInitDistNotSet() {
        // Warning thrown only since global dist can be set with a different weight init locally
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).dist(new GaussianDistribution(1e-3, 2)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    @DisplayName("Test Nesterovs Not Set Global")
    void testNesterovsNotSetGlobal() {
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter))).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    @DisplayName("Test Comp Graph Null Layer")
    void testCompGraphNullLayer() {
        ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder().optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.01)).seed(42).miniBatch(false).l1(0.2).l2(0.2).updater(Updater.RMSPROP).graphBuilder().addInputs("in").addLayer("L" + 1, new LSTM.Builder().nIn(20).updater(Updater.RMSPROP).nOut(10).weightInit(WeightInit.XAVIER).dropOut(0.4).l1(0.3).activation(Activation.SIGMOID).build(), "in").addLayer("output", new RnnOutputLayer.Builder().nIn(20).nOut(10).activation(Activation.SOFTMAX).weightInit(WeightInit.RELU_UNIFORM).build(), "L" + 1).setOutputs("output");
        ComputationGraphConfiguration conf = gb.build();
        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();
    }

    @Test
    @DisplayName("Test Predefined Config Values")
    void testPredefinedConfigValues() {
        double expectedMomentum = 0.9;
        double expectedAdamMeanDecay = 0.9;
        double expectedAdamVarDecay = 0.999;
        double expectedRmsDecay = 0.95;
        Distribution expectedDist = new NormalDistribution(0, 1);
        double expectedL1 = 0.0;
        double expectedL2 = 0.0;
        // Nesterovs Updater
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(0.9)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new Nesterovs(0.3, 0.4)).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        BaseLayer layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(expectedMomentum, ((Nesterovs) layerConf.getIUpdater()).getMomentum(), 1e-3);
        assertNull(TestUtils.getL1Reg(layerConf.getRegularization()));
        assertEquals(0.5, TestUtils.getL2(layerConf), 1e-3);
        BaseLayer layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(0.4, ((Nesterovs) layerConf1.getIUpdater()).getMomentum(), 1e-3);
        // Adam Updater
        conf = new NeuralNetConfiguration.Builder().updater(new Adam(0.3)).weightInit(new WeightInitDistribution(expectedDist)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).l1(0.3).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();
        layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(0.3, TestUtils.getL1(layerConf), 1e-3);
        assertEquals(0.5, TestUtils.getL2(layerConf), 1e-3);
        layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(expectedAdamMeanDecay, ((Adam) layerConf1.getIUpdater()).getBeta1(), 1e-3);
        assertEquals(expectedAdamVarDecay, ((Adam) layerConf1.getIUpdater()).getBeta2(), 1e-3);
        assertEquals(new WeightInitDistribution(expectedDist), layerConf1.getWeightInitFn());
        assertNull(TestUtils.getL1Reg(layerConf1.getRegularization()));
        assertNull(TestUtils.getL2Reg(layerConf1.getRegularization()));
        // RMSProp Updater
        conf = new NeuralNetConfiguration.Builder().updater(new RmsProp(0.3)).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new RmsProp(0.3, 0.4, RmsProp.DEFAULT_RMSPROP_EPSILON)).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();
        layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(expectedRmsDecay, ((RmsProp) layerConf.getIUpdater()).getRmsDecay(), 1e-3);
        assertNull(TestUtils.getL1Reg(layerConf.getRegularization()));
        assertNull(TestUtils.getL2Reg(layerConf.getRegularization()));
        layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(0.4, ((RmsProp) layerConf1.getIUpdater()).getRmsDecay(), 1e-3);
    }
}
