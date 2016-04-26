package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class LayerConfigValidationTest {


    @Test(expected = IllegalStateException.class)
    public void testDropConnect() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .useDropConnect(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testL1L2NotSet() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .regularization(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testRegNotSetL1Global() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .l1(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testRegNotSetL2Local() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testWeightInitDistNotSet() {
        // Warning thrown only since global dist can be set with a different weight init locally
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .dist(new GaussianDistribution(1e-3, 2))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetGlobal(){
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(1.0)
                .momentumAfter(testMomentumAfter)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetLocalMomentum(){
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).momentum(0.3).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetLocalMuAfter(){
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).momentumAfter(testMomentumAfter).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testAdaDeltaValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .rho(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).rho(0.01).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test
    public void testRmsPropValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .rmsDecay(2.0)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).rmsDecay(1.0).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.ADADELTA).rho(0.5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testAdamValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                        .adamMeanDecay(0.6).adamVarDecay(0.7).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingDecayRate(){
        double lr = 2;
        double power = 3;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Poly)
                .lrPolicyPower(power)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingPower(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingSteps(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingSchedule(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testCompGraphNullLayer(){
            ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(0.01).iterations(3)
                    .seed(42).miniBatch(false)
                    .l1(0.2).l2(0.2)
                    .rmsDecay(0.3)
                    .regularization(true)
		        /* Graph Builder */
                    .updater(Updater.RMSPROP)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("L" +1, new GravesLSTM.Builder().nIn(20)
                            .updater(Updater.RMSPROP).nOut(10)
                            .weightInit(WeightInit.XAVIER).dropOut(0.4)
                            .l1(0.3)
                            .activation("sigmoid").build(), "in")
                    .addLayer("output", new RnnOutputLayer.Builder().nIn(20)
                            .nOut(10).activation("softmax")
                            .weightInit(WeightInit.VI).build(), "L"+1).setOutputs("output");
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
        Distribution expectedDist = new NormalDistribution(1e-3, 1);
        double expectedL1 = 0.0;
        double expectedL2 = 0.0;

        // Nesterovs Updater
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .updater(Updater.NESTEROVS)
                .regularization(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).momentum(0.4).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Layer layerConf = net.getLayer(0).conf().getLayer();
        assertEquals(expectedMomentum, layerConf.getMomentum(), 1e-3);
        assertEquals(expectedL1, layerConf.getL1(), 1e-3);
        assertEquals(0.5, layerConf.getL2(), 1e-3);

        Layer layerConf1 = net.getLayer(1).conf().getLayer();
        assertEquals(0.4, layerConf1.getMomentum(), 1e-3);

        // Adam Updater
        conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.DISTRIBUTION)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).l1(0.3).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        layerConf = net.getLayer(0).conf().getLayer();
        assertEquals(0.3, layerConf.getL1(), 1e-3);
        assertEquals(0.5, layerConf.getL2(), 1e-3);

        layerConf1 = net.getLayer(1).conf().getLayer();
        assertEquals(expectedAdamMeanDecay, layerConf1.getAdamMeanDecay(), 1e-3);
        assertEquals(expectedAdamVarDecay, layerConf1.getAdamVarDecay(), 1e-3);
        assertEquals(expectedDist, layerConf1.getDist());
        // l1 & l2 local should still be set whether regularization true or false
        assertEquals(expectedL1, layerConf1.getL1(), 1e-3);
        assertEquals(expectedL2, layerConf1.getL2(), 1e-3);

        //RMSProp Updater
        conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).rmsDecay(0.4).build())
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        layerConf = net.getLayer(0).conf().getLayer();
        assertEquals(expectedRmsDecay, layerConf.getRmsDecay(), 1e-3);
        assertEquals(expectedL1, layerConf.getL1(), 1e-3);
        assertEquals(expectedL2, layerConf.getL2(), 1e-3);

        layerConf1 = net.getLayer(1).conf().getLayer();
        assertEquals(0.4, layerConf1.getRmsDecay(), 1e-3);


    }

}


