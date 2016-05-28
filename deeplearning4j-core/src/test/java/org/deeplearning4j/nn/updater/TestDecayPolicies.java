package org.deeplearning4j.nn.updater;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Test learning rate and momentum decay policies
 */


public class TestDecayPolicies {

    int nIn = 3;
    int nOut = 2;
    double epsilon = 1e-8;
    INDArray weightGradient = Nd4j.ones(nIn,nOut);
    INDArray biasGradient = Nd4j.ones(1,nOut);
    Gradient gradientSingle = new DefaultGradient();
    Gradient gradientMLN = new DefaultGradient();
    INDArray val, gradExpected, vPrev;
    String key;
    Map<String, INDArray> tmpStorage, tmpStorage2, tmpStorage3, tmpStorage4 = new HashMap<>();
    org.deeplearning4j.nn.conf.Updater[] updaters = {
            org.deeplearning4j.nn.conf.Updater.SGD,
            org.deeplearning4j.nn.conf.Updater.ADAGRAD,
            org.deeplearning4j.nn.conf.Updater.ADAM,
            org.deeplearning4j.nn.conf.Updater.RMSPROP,
    };

    @Before
    public void beforeDo(){
        int nLayers = 2;
        String wKey, bKey;

        gradientSingle.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientSingle.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int j=0; j < nLayers; j++){
            wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientMLN.setGradientFor(wKey, weightGradient.dup());
            bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY ;
            gradientMLN.setGradientFor(bKey, biasGradient.dup());
        }

        val = null;
        gradExpected = null;
        vPrev = null;
        tmpStorage = new HashMap<>();
        tmpStorage2 = new HashMap<>();
        tmpStorage3 = new HashMap<>();
        tmpStorage4 = new HashMap<>();

    }

    @Test
    public void testLearningRateExponentialDecaySingleLayer() {
        int iterations = 2;

        double lr = 1e-2;
        double decayRate = 2;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Exponential)
                .lrPolicyDecayRate(decayRate)
                .iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientActual = new DefaultGradient();
        gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        for (int i = 0; i < iterations; i++) {
            updater.update(layer, gradientActual, i, 1);
            double expectedLr = calcExponentialDecay(lr, decayRate, i);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("W"), 1e-4);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("b"), 1e-4);
        }
    }


    @Test
    public void testLearningRateInverseDecaySingleLayer() {
        int iterations = 2;

        double lr = 1e-2;
        double decayRate = 2;
        double power = 3;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse)
                .lrPolicyDecayRate(decayRate)
                .lrPolicyPower(power)
                .iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientActual = new DefaultGradient();
        gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < iterations; i++) {
            updater.update(layer, gradientActual, i, 1);
            double expectedLr = calcInverseDecay(lr, decayRate, i, power);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("W"), 1e-4);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("b"), 1e-4);
        }
    }

    @Test
    public void testLearningRateStepDecaySingleLayer() {
        int iterations = 2;

        double lr = 1e-2;
        double decayRate = 2;
        double steps = 3;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(decayRate)
                .lrPolicySteps(steps)
                .iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientActual = new DefaultGradient();
        gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < iterations; i++) {
            updater.update(layer, gradientActual, i, 1);
            double expectedLr = calcStepDecay(lr, decayRate, i, steps);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("W"), 1e-4);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("b"), 1e-4);
        }
    }


    @Test
    public void testLearningRatePolyDecaySingleLayer() {
        int iterations = 2;
        double lr = 1e-2;
        double power = 3;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Poly)
                .lrPolicyPower(power)
                .iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientActual = new DefaultGradient();
        gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < iterations; i++) {
            updater.update(layer, gradientActual, i, 1);
            double expectedLr = calcPolyDecay(lr, i, power, iterations);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("W"), 1e-4);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("b"), 1e-4);
        }
    }


    @Test
    public void testLearningRateSigmoidDecaySingleLayer() {
        int iterations = 2;
        double lr = 1e-2;
        double decayRate = 2;
        double steps = 3;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Sigmoid)
                .lrPolicyDecayRate(decayRate)
                .lrPolicySteps(steps)
                .iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0,params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientActual = new DefaultGradient();
        gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < iterations; i++) {
            updater.update(layer, gradientActual, i, 1);
            double expectedLr = calcSigmoidDecay(layer.conf().getLearningRateByParam("W"), decayRate, i, steps);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("W"), 1e-4);
            assertEquals(expectedLr, layer.conf().getLearningRateByParam("b"), 1e-4);
        }
    }


    @Test
    public void testLearningRateScheduleSingleLayer() {
        Map<Integer, Double> learningRateAfter = new HashMap<>();
        learningRateAfter.put(1, 0.2);
        int iterations = 2;

        for (org.deeplearning4j.nn.conf.Updater updaterFunc : updaters) {
            double lr = 1e-2;
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .learningRate(lr).learningRateSchedule(learningRateAfter)
                    .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                    .iterations(iterations)
                    .layer(new DenseLayer.Builder()
                            .nIn(nIn).nOut(nOut).updater(updaterFunc).build())
                    .build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
            Updater updater = UpdaterCreator.getUpdater(layer);

            Gradient gradientActual = new DefaultGradient();
            gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
            gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

            Gradient gradientExpected = new DefaultGradient();
            gradientExpected.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
            gradientExpected.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

            for (int i = 0; i < 2; i++) {
                updater.update(layer, gradientActual, i, 1);

                if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.SGD))
                    lr = testSGDComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAGRAD))
                    lr = testAdaGradComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAM))
                    lr = testAdamComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.RMSPROP))
                    lr = testRMSPropComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                assertEquals(lr, layer.conf().getLearningRateByParam("W"), 1e-4);
            }
        }
    }


    @Test
    public void testLearningRateScheduleMLN(){
        Map<Integer,Double> learningRateAfter = new HashMap<>();
        learningRateAfter.put(1, 0.2);
        int iterations = 2;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};

        for (org.deeplearning4j.nn.conf.Updater updaterFunc : updaters) {
            double lr = 1e-2;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .learningRate(lr).learningRateDecayPolicy(LearningRatePolicy.Schedule)
                    .learningRateSchedule(learningRateAfter).iterations(iterations)
                    .updater(updaterFunc)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).build())
                    .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).build())
                    .backprop(true).pretrain(false)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            Updater updater = UpdaterCreator.getUpdater(net);
            String wKey, bKey;

            for (int i = 0; i < 2; i++) {
                Gradient gradientActual = new DefaultGradient();
                Gradient gradientExpected = new DefaultGradient();
                for (int k = 0; k < net.getnLayers(); k++) {
                    wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
                    gradientActual.setGradientFor(wKey, weightGradient.dup());
                    gradientExpected.setGradientFor(wKey, weightGradient.dup());
                    bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY;
                    gradientActual.setGradientFor(bKey, biasGradient.dup());
                    gradientExpected.setGradientFor(bKey, biasGradient.dup());
                }

                updater.update(net, gradientActual, i, 1);
                if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.SGD))
                    lr = testSGDComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAGRAD))
                    lr = testAdaGradComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAM))
                    lr = testAdamComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.RMSPROP))
                    lr = testRMSPropComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);

                if(i == 0) assertEquals(lr, net.getLayer(1).conf().getLearningRateByParam("W"), lr);
                else assertEquals(lr, net.getLayer(1).conf().getLearningRateByParam("W"), learningRateAfter.get(1));
            }
        }
    }

    @Test
    public void testLearningRateScoreDecay(){
        double lr = 0.01;
        double lrScoreDecay = 0.10;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};
        int oldScore = 1;
        int newScore = 1;
        int iteration = 3;
        INDArray gradientW = Nd4j.ones(nIns[0], nOuts[0]);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Score)
                .lrPolicyDecayRate(lrScoreDecay)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        ConvexOptimizer opt = new StochasticGradientDescent(net.getDefaultConfiguration(), new NegativeDefaultStepFunction(), null, net);
        opt.checkTerminalConditions(gradientW, oldScore, newScore, iteration);
        assertEquals(lrScoreDecay, net.getLayer(0).conf().getLrPolicyDecayRate(), 1e-4);
        assertEquals(lr*(lrScoreDecay + Nd4j.EPS_THRESHOLD), net.getLayer(0).conf().getLearningRateByParam("W"), 1e-4);

    }

    @Test
    public void testOriginalLearningRateUnchanged() {
        // Confirm learning rate is unchanged while hash is updated

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();

        Nd4j.getRandom().setSeed(12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(false)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .learningRate(1.0)
                .learningRateDecayPolicy(LearningRatePolicy.Score)
                .lrPolicyDecayRate(0.10)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                .updater(org.deeplearning4j.nn.conf.Updater.SGD)
                .seed(12345L)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation("sigmoid")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("tanh")
                        .nIn(3).nOut(3)
                        .build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        //Run a number of iterations of learning
        mln.setInput(ds.getFeatureMatrix());
        mln.setLabels(ds.getLabels());
        mln.computeGradientAndScore();
        for( int j=0; j<1; j++ ) mln.fit(ds);
        mln.computeGradientAndScore();

        double lr0 = mln.getLayer(0).conf().getLayer().getLearningRate();
        double lr1 = mln.getLayer(1).conf().getLayer().getLearningRate();
        assertEquals(1.0, lr0, 0.0);
        assertEquals(1.0, lr1, 0.0);
    }

    @Test
    public void testMomentumScheduleSingleLayer(){
        double lr = 1e-2;
        double mu = 0.6;
        Map<Integer,Double> momentumAfter = new HashMap<>();
        momentumAfter.put(1, 0.2);
        int iterations = 2;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).momentum(mu)
                .momentumAfter(momentumAfter).iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientExpected = new DefaultGradient();
        gradientExpected.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientExpected.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradientSingle, i, 1);
            mu = testNesterovsComputation(gradientSingle, gradientExpected, lr, mu, momentumAfter, i);
            assertEquals(mu, layer.conf().getLayer().getMomentum(), 1e-4);
        }
    }

    @Test
    public void testMomentumScheduleMLN(){
        double lr = 1e-2;
        double mu = 0.6;
        Map<Integer,Double> momentumAfter = new HashMap<>();
        momentumAfter.put(1, 0.2);
        int iterations = 2;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).momentum(mu).momentumAfter(momentumAfter).iterations(iterations)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater updater = UpdaterCreator.getUpdater(net);

        String wKey, bKey;

        Gradient gradientExpected = new DefaultGradient();
        for (int k=0; k < net.getnLayers(); k++){
            wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientExpected.setGradientFor(wKey, weightGradient.dup());
            bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY ;
            gradientExpected.setGradientFor(bKey, biasGradient.dup());
        }

        for (int i = 0; i < 2; i++) {
            updater.update(net, gradientMLN, i, 1);
            mu = testNesterovsComputation(gradientMLN, gradientExpected, lr, mu, momentumAfter, i);
            assertEquals(mu, net.getLayer(1).conf().getLayer().getMomentum(), 1e-4);
        }
    }

///// Updater Calculations

    public double testSGDComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i){
        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            gradExpected = val.mul(lr);
            gradientExpected.setGradientFor(key, gradExpected);
            INDArray act = gradientActual.getGradientFor(key);
            assertEquals(gradExpected, act);
        }
        return lr;
    }

    public double testNesterovsComputation(Gradient gradientActual, Gradient gradientExpected, double lr, double mu, Map<Integer, Double> momentumAfter, int i) {

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if(momentumAfter !=null)
                mu = (momentumAfter.containsKey(i)) ? momentumAfter.get(i) : mu;
            key = entry.getKey();
            val = entry.getValue();
            INDArray vTmp = tmpStorage.get(key);

            if(vTmp == null)
                vTmp = Nd4j.zeros(val.shape());
            vPrev = vTmp;
            vTmp = vPrev.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(vTmp.mul(-mu - 1));
            gradientExpected.setGradientFor(key, gradExpected);

            INDArray act = gradientActual.getGradientFor(entry.getKey());
            assertEquals(gradExpected, act);
            tmpStorage.put(key, vTmp);
        }
        return mu;
    }


    public double testAdaGradComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray historicalGradient = tmpStorage.get(key);

            if(historicalGradient == null) historicalGradient = val.mul(val);
            else historicalGradient.addi(val.mul(val));

            gradExpected = Transforms.sqrt(historicalGradient.add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            gradientExpected.setGradientFor(key, gradExpected);
            tmpStorage.put(key, historicalGradient);
        }

        return lr;
    }

    public double testAdamComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {
        double beta1 = 0.9;
        double beta2 = 0.999;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();

            INDArray mTmp = tmpStorage2.get(key);
            INDArray vTmp = tmpStorage3.get(key);

            if(mTmp == null) mTmp = Nd4j.zeros(val.shape());
            if(vTmp == null) vTmp = Nd4j.zeros(val.shape());

            mTmp.muli(beta1).addi(val.mul(1.0-beta1));
            vTmp.muli(beta2).addi(val.mul(val).mul(1.0-beta2));

            double beta1t = FastMath.pow(beta1, i);
            double beta2t = FastMath.pow(beta2, i);
            double alphat = lr * FastMath.sqrt(1-beta2t)/(1-beta1t);
            if (Double.isNaN(alphat) || alphat == 0.0) alphat = epsilon;

            gradExpected = mTmp.mul(alphat).divi(Transforms.sqrt(vTmp).addi(epsilon));
            gradientExpected.setGradientFor(key, gradExpected);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));

            tmpStorage2.put(key, mTmp);
            tmpStorage3.put(key, vTmp);
        }
        return lr;
    }

    public double testRMSPropComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {
        double rmsDecay = 0.95;
        double epsilon = 1e-8;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray lastGTmp = tmpStorage4.get(key);

            if(lastGTmp==null)
                lastGTmp = Nd4j.zeros(val.shape());

            lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(epsilon)));
            gradientExpected.setGradientFor(key, gradExpected);

            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            tmpStorage4.put(key, lastGTmp);
        }

        return lr;
    }

    ///// Learning Rate Decay Policy Calculations

    public double calcExponentialDecay(double lr, double decayRate, double iteration){
        return lr * Math.pow(decayRate, iteration);
    }

    public double calcInverseDecay(double lr, double decayRate, double iteration, double power){
        return lr / Math.pow((1+decayRate * iteration), power);
    }

    public double calcStepDecay(double lr, double decayRate, double iteration, double steps){
        return lr * Math.pow(decayRate, Math.floor(iteration/steps));
    }

    public double calcPolyDecay(double lr, double iteration, double power, double maxIterations){
        return lr * Math.pow((1 - iteration/maxIterations), power);
    }
    public double calcSigmoidDecay(double lr, double decayRate, double iteration, double steps){
        return lr / (1 + Math.exp(-decayRate * (iteration - steps)));
    }

}

