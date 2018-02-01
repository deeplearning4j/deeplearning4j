package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class LayerConfigValidationTest extends BaseDL4JTest {


    @Test
    public void testDropConnect() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).weightNoise(new DropConnect(0.5))
                        .list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testL1L2NotSet() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3))
                        .list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    @Ignore //Old assumption: throw exception on l1 but no regularization. Current design: warn, not exception
    public void testRegNotSetL1Global() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).l1(0.5).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    @Ignore //Old assumption: throw exception on l1 but no regularization. Current design: warn, not exception
    public void testRegNotSetL2Local() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testWeightInitDistNotSet() {
        // Warning thrown only since global dist can be set with a different weight init locally
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Sgd(0.3)).dist(new GaussianDistribution(1e-3, 2))
                                        .list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetGlobal() {
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter))).list()
                                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testCompGraphNullLayer() {
        ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.01))
                        .seed(42).miniBatch(false).l1(0.2).l2(0.2)
                        /* Graph Builder */
                        .updater(Updater.RMSPROP).graphBuilder().addInputs("in")
                        .addLayer("L" + 1,
                                        new GravesLSTM.Builder().nIn(20).updater(Updater.RMSPROP).nOut(10)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .dropOut(0.4).l1(0.3).activation(Activation.SIGMOID).build(),
                                        "in")
                        .addLayer("output",
                                        new RnnOutputLayer.Builder().nIn(20).nOut(10).activation(Activation.SOFTMAX)
                                                        .weightInit(WeightInit.RELU_UNIFORM).build(),
                                        "L" + 1)
                        .setOutputs("output");
        ComputationGraphConfiguration conf = gb.build();
        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();
    }


    @Test
    public void testPredefinedConfigValues() {
        double expectedMomentum = 0.9;
        double expectedAdamMeanDecay = 0.9;
        double expectedAdamVarDecay = 0.999;
        double expectedRmsDecay = 0.95;
        Distribution expectedDist = new NormalDistribution(0, 1);
        double expectedL1 = 0.0;
        double expectedL2 = 0.0;

        // Nesterovs Updater
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(0.9))
                        .list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new Nesterovs(0.3, 0.4)).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        BaseLayer layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(expectedMomentum, ((Nesterovs)layerConf.getIUpdater()).getMomentum(), 1e-3);
        assertEquals(expectedL1, layerConf.getL1(), 1e-3);
        assertEquals(0.5, layerConf.getL2(), 1e-3);

        BaseLayer layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(0.4, ((Nesterovs)layerConf1.getIUpdater()).getMomentum(), 1e-3);

        // Adam Updater
        conf = new NeuralNetConfiguration.Builder().updater(new Adam(0.3))
                        .weightInit(WeightInit.DISTRIBUTION).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).l1(0.3).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(0.3, layerConf.getL1(), 1e-3);
        assertEquals(0.5, layerConf.getL2(), 1e-3);

        layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(expectedAdamMeanDecay, ((Adam)layerConf1.getIUpdater()).getBeta1(), 1e-3);
        assertEquals(expectedAdamVarDecay, ((Adam)layerConf1.getIUpdater()).getBeta2(), 1e-3);
        assertEquals(expectedDist, layerConf1.getDist());
        // l1 & l2 local should still be set whether regularization true or false
        assertEquals(expectedL1, layerConf1.getL1(), 1e-3);
        assertEquals(expectedL2, layerConf1.getL2(), 1e-3);

        //RMSProp Updater
        conf = new NeuralNetConfiguration.Builder().updater(new RmsProp(0.3)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new RmsProp(0.3, 0.4, RmsProp.DEFAULT_RMSPROP_EPSILON)).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        layerConf = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(expectedRmsDecay, ((RmsProp)layerConf.getIUpdater()).getRmsDecay(), 1e-3);
        assertEquals(expectedL1, layerConf.getL1(), 1e-3);
        assertEquals(expectedL2, layerConf.getL2(), 1e-3);

        layerConf1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(0.4, ((RmsProp)layerConf1.getIUpdater()).getRmsDecay(), 1e-3);


    }

}


