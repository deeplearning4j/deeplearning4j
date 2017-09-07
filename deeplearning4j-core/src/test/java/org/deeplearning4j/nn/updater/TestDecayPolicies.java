package org.deeplearning4j.nn.updater;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test learning rate and momentum decay policies
 */


public class TestDecayPolicies {

    int nIn = 3;
    int nOut = 2;
    double epsilon = 1e-8;
    INDArray gradient;
    INDArray weightGradient; // = Nd4j.ones(nIn, nOut);
    INDArray biasGradient; // = Nd4j.ones(1, nOut);
    DefaultGradient gradientSingle = new DefaultGradient();
    DefaultGradient gradientMLN = new DefaultGradient();
    INDArray val, gradExpected, vPrev;
    String key;
    Map<String, INDArray> tmpStorage, tmpStorage2, tmpStorage3, tmpStorage4 = new HashMap<>();
    org.deeplearning4j.nn.conf.Updater[] updaters = {org.deeplearning4j.nn.conf.Updater.SGD,
                    org.deeplearning4j.nn.conf.Updater.ADAGRAD, org.deeplearning4j.nn.conf.Updater.ADAM,
                    org.deeplearning4j.nn.conf.Updater.RMSPROP, org.deeplearning4j.nn.conf.Updater.ADAMAX};

    @Before
    public void beforeDo() {
        Nd4j.getRandom().setSeed(12345);
        int nLayers = 2;
        String wKey, bKey;

        gradient = Nd4j.ones(1, nIn * nOut + nOut);
        gradient.addi(Nd4j.rand(gradient.shape()));
        weightGradient = gradient.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradient.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));

        gradientSingle.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientSingle.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradientSingle.setFlattenedGradient(gradient);

        for (int j = 0; j < nLayers; j++) {
            wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientMLN.setGradientFor(wKey, weightGradient.dup());
            bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY;
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
    public void testMomentumScheduleSingleLayer() {
        double lr = 1e-2;
        double mu = 0.9;
        Map<Integer, Double> momentumAfter = new HashMap<>();
        momentumAfter.put(0, mu);
        momentumAfter.put(1, 0.2);
        int iterations = 2;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(lr, new MapSchedule(ScheduleType.ITERATION, momentumAfter)))
                .iterations(iterations).layer(new DenseLayer.Builder().nIn(nIn)
                                        .nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradient);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientExpected = new DefaultGradient();
        gradientExpected.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientExpected.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradientSingle, i, 0, 1);
            mu = testNesterovsComputation(gradientSingle, gradientExpected, lr, mu, momentumAfter, i);
//            assertEquals(mu, ((BaseLayer) layer.conf().getLayer()).getMomentum(), 1e-4);
        }
        fail();
    }

    @Test
    public void testMomentumScheduleMLN() {
        double lr = 1e-2;
        double mu = 0.6;
        Map<Integer, Double> momentumAfter = new HashMap<>();
        momentumAfter.put(1, 0.2);
        int iterations = 2;
        int[] nIns = {4, 2};
        int[] nOuts = {2, 3};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(lr, new MapSchedule(ScheduleType.ITERATION,momentumAfter)))
                .iterations(iterations).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0])
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1])
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater updater = UpdaterCreator.getUpdater(net);
        int stateSize = (int) new Nesterovs().stateSize(net.numParams());
        updater.setStateViewArray(net, Nd4j.create(1, stateSize), true);

        String wKey, bKey;

        Gradient gradientMLN = new DefaultGradient();
        INDArray gradViewArr = net.getGradientsViewArray();
        int paramsSoFar = 0;
        for (int j = 0; j < 2; j++) {
            int nParams = net.getLayer(j).numParams();
            INDArray g = gradViewArr.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + nParams));
            int nW = nIns[j] * nOuts[j];
            int nB = nOuts[j];
            INDArray gw = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nW));
            INDArray gb = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nW, nW + nB));
            wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientMLN.setGradientFor(wKey, gw);
            bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY;
            gradientMLN.setGradientFor(bKey, gb);
            paramsSoFar += nParams;
        }

        Gradient gradientExpected = new DefaultGradient();
        gradViewArr = gradViewArr.dup();
        paramsSoFar = 0;
        for (int j = 0; j < net.getnLayers(); j++) {
            int nParams = net.getLayer(j).numParams();
            INDArray g = gradViewArr.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + nParams));
            int nW = nIns[j] * nOuts[j];
            int nB = nOuts[j];
            INDArray gw = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nW));
            INDArray gb = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nW, nW + nB));
            wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientExpected.setGradientFor(wKey, gw);
            bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY;
            gradientExpected.setGradientFor(bKey, gb);
        }



        for (int i = 0; i < 2; i++) {
            updater.update(net, gradientMLN, i, 0, 1);
            mu = testNesterovsComputation(gradientMLN, gradientExpected, lr, mu, momentumAfter, i);
//            assertEquals(mu, ((BaseLayer) net.getLayer(1).conf().getLayer()).getMomentum(), 1e-4);
        }
        fail();
    }

    ///// Updater Calculations

    public double testSGDComputation(Gradient gradientActual, Gradient gradientExpected, double lr,
                    Map<Integer, Double> learningRateAfter, int i) {
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

    public double testNesterovsComputation(Gradient gradientActual, Gradient gradientExpected, double lr, double mu,
                    Map<Integer, Double> momentumAfter, int i) {

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (momentumAfter != null)
                mu = (momentumAfter.containsKey(i)) ? momentumAfter.get(i) : mu;
            key = entry.getKey();
            val = entry.getValue();
            INDArray vTmp = tmpStorage.get(key);

            if (vTmp == null)
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


    public double testAdaGradComputation(Gradient gradientActual, Gradient gradientExpected, double lr,
                    Map<Integer, Double> learningRateAfter, int i) {

        double epsilon = AdaGrad.DEFAULT_ADAGRAD_EPSILON;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray historicalGradient = tmpStorage.get(key);

            if (historicalGradient == null)
                historicalGradient = val.mul(val);
            else
                historicalGradient.addi(val.mul(val));

            gradExpected = Transforms.sqrt(historicalGradient.add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            gradientExpected.setGradientFor(key, gradExpected);
            tmpStorage.put(key, historicalGradient);
        }

        return lr;
    }

    public double testAdamComputation(Gradient gradientActual, Gradient gradientExpected, double lr,
                    Map<Integer, Double> learningRateAfter, int i) {
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = Adam.DEFAULT_ADAM_EPSILON;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();

            INDArray mTmp = tmpStorage2.get(key);
            INDArray vTmp = tmpStorage3.get(key);

            if (mTmp == null)
                mTmp = Nd4j.zeros(val.shape());
            if (vTmp == null)
                vTmp = Nd4j.zeros(val.shape());

            mTmp.muli(beta1).addi(val.mul(1.0 - beta1));
            vTmp.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));

            double beta1t = FastMath.pow(beta1, i + 1);
            double beta2t = FastMath.pow(beta2, i + 1);
            double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
            if (Double.isNaN(alphat) || alphat == 0.0)
                alphat = epsilon;

            gradExpected = mTmp.mul(alphat).divi(Transforms.sqrt(vTmp).addi(epsilon));
            gradientExpected.setGradientFor(key, gradExpected);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));

            tmpStorage2.put(key, mTmp);
            tmpStorage3.put(key, vTmp);
        }
        return lr;
    }

    public double testAdaMaxComputation(Gradient gradientActual, Gradient gradientExpected, double lr,
                    Map<Integer, Double> learningRateAfter, int i) {

        double beta1 = 0.9;
        double beta2 = 0.999;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();

            INDArray mTmp = tmpStorage2.get(key);
            INDArray uTmp = tmpStorage3.get(key);

            if (mTmp == null)
                mTmp = Nd4j.zeros(val.shape());
            if (uTmp == null)
                uTmp = Nd4j.zeros(val.shape());

            mTmp.muli(beta1).addi(val.mul(1.0 - beta1));
            uTmp.assign(Transforms.max(uTmp.mul(beta2), Transforms.abs(val)));

            double beta1t = FastMath.pow(beta1, i + 1);
            double alphat = lr / (1 - beta1t);
            if (Double.isNaN(alphat) || alphat == 0.0)
                alphat = epsilon;

            gradExpected = mTmp.mul(alphat).divi(uTmp);
            gradientExpected.setGradientFor(key, gradExpected);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));

            tmpStorage2.put(key, mTmp);
            tmpStorage3.put(key, uTmp);
        }
        return lr;
    }

    public double testRMSPropComputation(Gradient gradientActual, Gradient gradientExpected, double lr,
                    Map<Integer, Double> learningRateAfter, int i) {
        double rmsDecay = RmsProp.DEFAULT_RMSPROP_RMSDECAY;
        double epsilon = RmsProp.DEFAULT_RMSPROP_EPSILON;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray lastGTmp = tmpStorage4.get(key);

            if (lastGTmp == null)
                lastGTmp = Nd4j.valueArrayOf(val.shape(), epsilon);

            lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(epsilon)));
            gradientExpected.setGradientFor(key, gradExpected);

            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            tmpStorage4.put(key, lastGTmp);
        }

        return lr;
    }

}

