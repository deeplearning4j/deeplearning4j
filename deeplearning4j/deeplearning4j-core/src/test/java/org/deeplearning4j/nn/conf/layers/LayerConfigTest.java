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

package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LayerConfigTest extends BaseDL4JTest {

    @Test
    public void testLayerName() {

        String name1 = "genisys";
        String name2 = "bill";

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).name(name1).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).name(name2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(name1, conf.getConf(0).getLayer().getLayerName());
        assertEquals(name2, conf.getConf(1).getLayer().getLayerName());

    }

    @Test
    public void testActivationLayerwiseOverride() {
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals("relu", ((BaseLayer) conf.getConf(0).getLayer()).getActivationFn().toString());
        assertEquals("relu", ((BaseLayer) conf.getConf(1).getLayer()).getActivationFn().toString());

        //With
        conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).activation(Activation.TANH).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals("relu", ((BaseLayer) conf.getConf(0).getLayer()).getActivationFn().toString());
        assertEquals("tanh", ((BaseLayer) conf.getConf(1).getLayer()).getActivationFn().toString());
    }


    @Test
    public void testWeightBiasInitLayerwiseOverride() {
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1.0)).biasInit(1).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(0).getLayer()).getWeightInit());
        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(1).getLayer()).getWeightInit());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(0).getLayer()).getDist().toString());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(1).getLayer()).getDist().toString());
        assertEquals(1, ((BaseLayer) conf.getConf(0).getLayer()).getBiasInit(), 0.0);
        assertEquals(1, ((BaseLayer) conf.getConf(1).getLayer()).getBiasInit(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1.0)).biasInit(1).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1,
                                        new DenseLayer.Builder().nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new UniformDistribution(0, 1)).biasInit(0).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(0).getLayer()).getWeightInit());
        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(1).getLayer()).getWeightInit());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(0).getLayer()).getDist().toString());
        assertEquals("UniformDistribution{lower=0.0, upper=1.0}",
                        ((BaseLayer) conf.getConf(1).getLayer()).getDist().toString());
        assertEquals(1, ((BaseLayer) conf.getConf(0).getLayer()).getBiasInit(), 0.0);
        assertEquals(0, ((BaseLayer) conf.getConf(1).getLayer()).getBiasInit(), 0.0);
    }

    /*
    @Test
    public void testLrL1L2LayerwiseOverride() {
        //Idea: Set some common values for all layers. Then selectively override
        // the global config, and check they actually work.

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(0.3).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.3, ((BaseLayer) conf.getConf(0).getLayer()).getLearningRate(), 0.0);
        assertEquals(0.3, ((BaseLayer) conf.getConf(1).getLayer()).getLearningRate(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder().learningRate(0.3).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).learningRate(0.2).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.3, ((BaseLayer) conf.getConf(0).getLayer()).getLearningRate(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(1).getLayer()).getLearningRate(), 0.0);

        //L1 and L2 without layerwise override:
        conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.2).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.1, ((BaseLayer) conf.getConf(0).getLayer()).getL1(), 0.0);
        assertEquals(0.1, ((BaseLayer) conf.getConf(1).getLayer()).getL1(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(0).getLayer()).getL2(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(1).getLayer()).getL2(), 0.0);

        //L1 and L2 with layerwise override:
        conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.2).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l1(0.9).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.8).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.9, ((BaseLayer) conf.getConf(0).getLayer()).getL1(), 0.0);
        assertEquals(0.1, ((BaseLayer) conf.getConf(1).getLayer()).getL1(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(0).getLayer()).getL2(), 0.0);
        assertEquals(0.8, ((BaseLayer) conf.getConf(1).getLayer()).getL2(), 0.0);
    }*/



    @Test
    public void testDropoutLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().dropOut(1.0).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(new Dropout(1.0), conf.getConf(0).getLayer().getIDropout());
        assertEquals(new Dropout(1.0), conf.getConf(1).getLayer().getIDropout());

        conf = new NeuralNetConfiguration.Builder().dropOut(1.0).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).dropOut(2.0).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(new Dropout(1.0), conf.getConf(0).getLayer().getIDropout());
        assertEquals(new Dropout(2.0), conf.getConf(1).getLayer().getIDropout());
    }

    @Test
    public void testMomentumLayerwiseOverride() {
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter)))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);

        Map<Integer, Double> testMomentumAfter2 = new HashMap<>();
        testMomentumAfter2.put(0, 0.2);

        conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter) ))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder()
                                        .nIn(2).nOut(2).updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter2))).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();
        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
        assertEquals(0.2, ((Nesterovs)((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
    }

    @Test
    public void testUpdaterRhoRmsDecayLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new AdaDelta(0.5, 0.9)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new AdaDelta(0.01,0.9)).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertTrue(((BaseLayer) conf.getConf(0).getLayer()).getIUpdater() instanceof AdaDelta);
        assertTrue(((BaseLayer) conf.getConf(1).getLayer()).getIUpdater() instanceof AdaDelta);
        assertEquals(0.5, ((AdaDelta)((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getRho(), 0.0);
        assertEquals(0.01, ((AdaDelta)((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getRho(), 0.0);

        conf = new NeuralNetConfiguration.Builder().updater(new RmsProp(1.0, 2.0, RmsProp.DEFAULT_RMSPROP_EPSILON)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).updater(new RmsProp(1.0, 1.0, RmsProp.DEFAULT_RMSPROP_EPSILON)).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new AdaDelta(0.5,AdaDelta.DEFAULT_ADADELTA_EPSILON)).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertTrue(((BaseLayer) conf.getConf(0).getLayer()).getIUpdater() instanceof RmsProp);
        assertTrue(((BaseLayer) conf.getConf(1).getLayer()).getIUpdater() instanceof AdaDelta);
        assertEquals(1.0, ((RmsProp) ((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getRmsDecay(), 0.0);
        assertEquals(0.5, ((AdaDelta) ((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getRho(), 0.0);
    }


    @Test
    public void testUpdaterAdamParamsLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1.0, 0.5, 0.5, 1e-8))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new Adam(1.0, 0.6, 0.7, 1e-8)).build())
                        .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.5, ((Adam) ((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getBeta1(), 0.0);
        assertEquals(0.6, ((Adam) ((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getBeta1(), 0.0);
        assertEquals(0.5, ((Adam) ((BaseLayer) conf.getConf(0).getLayer()).getIUpdater()).getBeta2(), 0.0);
        assertEquals(0.7, ((Adam) ((BaseLayer) conf.getConf(1).getLayer()).getIUpdater()).getBeta2(), 0.0);
    }

    @Test
    public void testGradientNormalizationLayerwiseOverride() {

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(0).getLayer()).getGradientNormalization());
        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(1).getLayer()).getGradientNormalization());
        assertEquals(10, ((BaseLayer) conf.getConf(0).getLayer()).getGradientNormalizationThreshold(), 0.0);
        assertEquals(10, ((BaseLayer) conf.getConf(1).getLayer()).getGradientNormalizationThreshold(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .gradientNormalization(GradientNormalization.None)
                                        .gradientNormalizationThreshold(2.5).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(0).getLayer()).getGradientNormalization());
        assertEquals(GradientNormalization.None, ((BaseLayer) conf.getConf(1).getLayer()).getGradientNormalization());
        assertEquals(10, ((BaseLayer) conf.getConf(0).getLayer()).getGradientNormalizationThreshold(), 0.0);
        assertEquals(2.5, ((BaseLayer) conf.getConf(1).getLayer()).getGradientNormalizationThreshold(), 0.0);
    }


    /*
    @Test
    public void testLearningRatePolicyExponential() {
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
                        .updater(Updater.SGD)
                        .learningRateDecayPolicy(LearningRatePolicy.Exponential).lrPolicyDecayRate(lrDecayRate).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(LearningRatePolicy.Exponential, conf.getConf(0).getLearningRatePolicy());
        assertEquals(LearningRatePolicy.Exponential, conf.getConf(1).getLearningRatePolicy());
        assertEquals(lrDecayRate, conf.getConf(0).getLrPolicyDecayRate(), 0.0);
        assertEquals(lrDecayRate, conf.getConf(1).getLrPolicyDecayRate(), 0.0);
    }

    @Test
    public void testLearningRatePolicyInverse() {
        double lr = 2;
        double lrDecayRate = 5;
        double power = 3;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(iterations).learningRate(lr)
                        .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(lrDecayRate)
                        .lrPolicyPower(power).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(LearningRatePolicy.Inverse, conf.getConf(0).getLearningRatePolicy());
        assertEquals(LearningRatePolicy.Inverse, conf.getConf(1).getLearningRatePolicy());
        assertEquals(lrDecayRate, conf.getConf(0).getLrPolicyDecayRate(), 0.0);
        assertEquals(lrDecayRate, conf.getConf(1).getLrPolicyDecayRate(), 0.0);
        assertEquals(power, conf.getConf(0).getLrPolicyPower(), 0.0);
        assertEquals(power, conf.getConf(1).getLrPolicyPower(), 0.0);
    }


    @Test
    public void testLearningRatePolicySteps() {
        double lr = 2;
        double lrDecayRate = 5;
        double steps = 4;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(iterations).learningRate(lr)
                        .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(lrDecayRate)
                        .lrPolicySteps(steps).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(LearningRatePolicy.Step, conf.getConf(0).getLearningRatePolicy());
        assertEquals(LearningRatePolicy.Step, conf.getConf(1).getLearningRatePolicy());
        assertEquals(lrDecayRate, conf.getConf(0).getLrPolicyDecayRate(), 0.0);
        assertEquals(lrDecayRate, conf.getConf(1).getLrPolicyDecayRate(), 0.0);
        assertEquals(steps, conf.getConf(0).getLrPolicySteps(), 0.0);
        assertEquals(steps, conf.getConf(1).getLrPolicySteps(), 0.0);
    }

    @Test
    public void testLearningRatePolicyPoly() {
        double lr = 2;
        double lrDecayRate = 5;
        double power = 3;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(iterations).learningRate(lr)
                        .learningRateDecayPolicy(LearningRatePolicy.Poly).lrPolicyDecayRate(lrDecayRate)
                        .lrPolicyPower(power).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(LearningRatePolicy.Poly, conf.getConf(0).getLearningRatePolicy());
        assertEquals(LearningRatePolicy.Poly, conf.getConf(1).getLearningRatePolicy());
        assertEquals(lrDecayRate, conf.getConf(0).getLrPolicyDecayRate(), 0.0);
        assertEquals(lrDecayRate, conf.getConf(1).getLrPolicyDecayRate(), 0.0);
        assertEquals(power, conf.getConf(0).getLrPolicyPower(), 0.0);
        assertEquals(power, conf.getConf(1).getLrPolicyPower(), 0.0);
    }

    @Test
    public void testLearningRatePolicySigmoid() {
        double lr = 2;
        double lrDecayRate = 5;
        double steps = 4;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(iterations).learningRate(lr)
                        .learningRateDecayPolicy(LearningRatePolicy.Sigmoid).lrPolicyDecayRate(lrDecayRate)
                        .lrPolicySteps(steps).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(LearningRatePolicy.Sigmoid, conf.getConf(0).getLearningRatePolicy());
        assertEquals(LearningRatePolicy.Sigmoid, conf.getConf(1).getLearningRatePolicy());
        assertEquals(lrDecayRate, conf.getConf(0).getLrPolicyDecayRate(), 0.0);
        assertEquals(lrDecayRate, conf.getConf(1).getLrPolicyDecayRate(), 0.0);
        assertEquals(steps, conf.getConf(0).getLrPolicySteps(), 0.0);
        assertEquals(steps, conf.getConf(1).getLrPolicySteps(), 0.0);
    }

*/
}
