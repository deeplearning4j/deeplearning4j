package org.deeplearning4j.nn.updater;


import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.Assert.*;

public class TestUpdaters {

    protected int nIn = 3;
    protected int nOut = 2;
    //    protected double epsilon = 1e-8;
    protected INDArray gradients;
    protected INDArray weightGradient;
    protected INDArray biasGradient;
    protected DefaultGradient gradient = new DefaultGradient();
    protected INDArray val, gradExpected;
    protected String key;


    @Before
    public void beforeDo() {
        gradients = Nd4j.ones(nIn * nOut + nOut);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setFlattenedGradient(gradients);
    }

    @Test
    public void testAdaDeltaUpdate() {
        //Here: test updaters manually vs. using updater
        INDArray dxSquared;
        Map<String, INDArray> msg = new HashMap<>();
        Map<String, INDArray> msdx = new HashMap<>();

        double rho = 0.85;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().rho(rho)
                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA)
                                        .epsilon(Nd4j.EPS_THRESHOLD).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradient, i, 0, 1);

            // calculations for one iteration / update

            for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
                key = entry.getKey();
                val = entry.getValue();
                INDArray msgTmp = msg.get(key);
                INDArray msdxTmp = msdx.get(key);

                if (msgTmp == null) {
                    msgTmp = Nd4j.zeros(val.shape());
                    msdxTmp = Nd4j.zeros(val.shape());
                }

                msgTmp.muli(rho);
                msgTmp.addi(val.mul(val).muli(1 - rho));

                gradExpected = Transforms.sqrt(msdxTmp.add(Nd4j.EPS_THRESHOLD))
                                .divi(Transforms.sqrt(msgTmp.add(Nd4j.EPS_THRESHOLD))).muli(val);
                gradientCopyPreUpdate.setGradientFor(key, gradExpected);

                assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));

                msdxTmp.muli(rho);
                dxSquared = gradExpected.mul(gradExpected);
                msdxTmp.addi(dxSquared.muli(1 - rho));

                msg.put(key, msgTmp);
                msdx.put(key, msdxTmp);
                count++;
            }
            assertEquals(rho, layer.layerConf().getRho(), 1e-4);
        }

        assertEquals(4, count);
    }

    @Test
    public void testAdaGradUpdater() {
        double lr = 1e-2;
        double epsilon = AdaGrad.DEFAULT_ADAGRAD_EPSILON;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = Transforms.sqrt(val.mul(val).add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }
        assertEquals(lr, layer.layerConf().getLearningRate(), 1e-4);
        assertEquals(2, count);
    }


    @Test
    public void testAdamUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = Adam.DEFAULT_ADAM_EPSILON;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
                        .iterations(iteration).adamMeanDecay(beta1).adamVarDecay(beta2).layer(new DenseLayer.Builder()
                                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, iteration, 0, 1);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);
        double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0 - beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(beta1, layer.layerConf().getAdamMeanDecay(), 1e-4);
        assertEquals(beta2, layer.layerConf().getAdamVarDecay(), 1e-4);
        assertEquals(2, count);
    }

    @Test
    public void testNadamUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = Nadam.DEFAULT_NADAM_EPSILON;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr).iterations(iteration)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(new Nadam.Builder().learningRate(lr).beta1(beta1)
                                                                        .beta2(beta2).epsilon(epsilon).build())
                                                        .build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);

        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        /*
        * Making update for layer
        * */
        updater.update(layer, gradient, iteration, 0,1);

        double beta1t = FastMath.pow(beta1, iteration + 1);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            INDArray oneMinusBeta1Grad = val.mul(1.0 - beta1);
            m.muli(beta1).addi(oneMinusBeta1Grad);

            INDArray oneMinusBeta2GradSquared = val.mul(val).muli(1.0 - beta2);
            v.muli(beta2).addi(oneMinusBeta2GradSquared);

            INDArray biasCorrectedEstimateOfMomentum = m.mul(beta1).divi(1.0 - beta1t);
            INDArray secondTerm = oneMinusBeta1Grad.divi(1.0 - beta1t);

            INDArray alphat = biasCorrectedEstimateOfMomentum.add(secondTerm).muli(lr);

            INDArray sqrtV = Transforms.sqrt(v, false).addi(epsilon);

            gradExpected = val.assign(alphat).divi(sqrtV);
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals("Count should be equal to 2, one for weight gradient and one for bias gradient", 2, count);

        /*
        * Check that we are not erroneously mutating moving avg gradient while calculating
        * `biasCorrectedEstimateOfMomentum = m * beta1 /(1.0 - beta1t);`
        * */
        BaseMultiLayerUpdater baseUpdater = (BaseMultiLayerUpdater) updater;
        UpdaterBlock ub = (UpdaterBlock) baseUpdater.getUpdaterBlocks().get(0);
        NadamUpdater nadamUpdater = (NadamUpdater) ub.getGradientUpdater();


        //Calculated for following setup: initialWeights are all equal to 1, beta1 = 0.8, beta2 = 0.888, learning rate = 0.01
        double calculatedByHandMScalar = 0.2;
        double[] expectedM = Nd4j.ones(1, numParams).mul(calculatedByHandMScalar).data().asDouble();

        double[] actualM = Arrays.copyOfRange(nadamUpdater.getM().data().asDouble(), 0, numParams);
        for (int i = 0; i < actualM.length; i++) {
            actualM[i] = Math.round(actualM[i] * 1e2) / 1e2;
        }

        assertEquals("Wrong weight gradient after first iteration's update", Arrays.equals(actualM, expectedM), true);

    }

    @Test
    public void testAdaMaxUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = AdaMax.DEFAULT_ADAMAX_EPSILON;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr)
                        .iterations(iteration).adamMeanDecay(beta1).adamVarDecay(beta2).layer(new DenseLayer.Builder()
                                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAMAX).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, iteration, 0, 1);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);
        double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0 - beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(beta1, layer.layerConf().getAdamMeanDecay(), 1e-4);
        assertEquals(beta2, layer.layerConf().getAdamVarDecay(), 1e-4);
        assertEquals(2, count);
    }

    @Test
    public void testNestorovsUpdater() {
        double lr = 1e-2;
        double mu = 0.6;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr).momentum(mu)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            INDArray val = entry.getValue();
            INDArray v = Nd4j.create(val.shape());
            INDArray vPrev = v.dup();
            v = v.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(v.mul(-mu - 1));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(mu, layer.layerConf().getMomentum(), 1e-4);
        assertEquals(2, count);
    }


    @Test
    public void testRMSPropUpdater() {
        double lr = 0.01;
        double rmsDecay = 0.25;
        Map<String, INDArray> lastG = new HashMap<>();


        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr).rmsDecay(rmsDecay)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);


        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1);

        double epsilon = 1e-8;

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            key = entry.getKey();
            val = entry.getValue();
            INDArray lastGTmp = lastG.get(key);

            if (lastGTmp == null)
                lastGTmp = Nd4j.zeros(val.shape());

            lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(epsilon)));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            lastG.put(key, lastGTmp);
        }
        assertEquals(rmsDecay, layer.layerConf().getRmsDecay(), 1e-4);
    }

    @Test
    public void testSGDUpdater() {
        double lr = 0.05;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1);

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.layerConf().getLearningRate(), 1e-4);
    }


    @Test
    public void testNoOpUpdater() {
        Random r = new Random(12345L);
        double lr = 0.5;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(lr)
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                                        .updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        for (int i = 0; i < weightGradient.length(); i++)
            weightGradient.putScalar(i, r.nextDouble());
        for (int i = 0; i < biasGradient.length(); i++)
            biasGradient.putScalar(i, r.nextDouble());

        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1);

        INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

        assertEquals(wg, weightGradActual);
        assertEquals(bg, biasGradActual);

    }

    @Test
    public void testMultiLayerUpdater() throws Exception {
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr).momentum(0.6).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(8)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS)
                                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE)
                                        .build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(Nd4j.create(1, 4), Nd4j.create(1, 8));

        Updater updater = net.getUpdater();
        assertNotNull(updater);
        assertTrue(updater.getClass() == MultiLayerUpdater.class);

        MultiLayerUpdater mlu = (MultiLayerUpdater) updater;

        int count = 0;
        for (UpdaterBlock u : mlu.getUpdaterBlocks()) {
            GradientUpdater gu = u.getGradientUpdater();
            switch (count) {
                case 0:
                    assertTrue(gu instanceof SgdUpdater);
                    break;
                case 1:
                    assertTrue(gu instanceof org.nd4j.linalg.learning.NoOpUpdater);
                    break;
                case 2:
                    assertTrue(gu instanceof AdaGradUpdater);
                    break;
                case 3:
                    assertTrue(gu instanceof NesterovsUpdater);
                    break;
                default:
                    throw new RuntimeException();
            }
            count++;
        }


        GradientUpdater[] uArr = new GradientUpdater[4];
        uArr[0] = new SgdUpdater(new Sgd(lr));
        uArr[1] = new NoOpUpdater(new NoOp());
        uArr[2] = new AdaGradUpdater(new AdaGrad(lr, AdaGrad.DEFAULT_ADAGRAD_EPSILON));
        INDArray updaterState = Nd4j.create(1, 6 * 7 + 7, 'f');
        uArr[2].setStateViewArray(updaterState, new int[] {1, 6 * 7 + 7}, 'f', true);

        uArr[3] = new NesterovsUpdater(new Nesterovs(lr, 0.6));
        //        updaterStateSize = uArr[3].stateSizeForLayer(net.getLayer(3));
        updaterState = Nd4j.create(1, 7 * 8 + 8, 'f');
        uArr[3].setStateViewArray(updaterState, new int[] {1, 7 * 8 + 8}, 'f', true);

        int[] nIns = {4, 5, 6, 7};
        int[] nOuts = {5, 6, 7, 8};

        for (int i = 0; i < 5; i++) {
            Gradient gradient = new DefaultGradient();
            Map<String, INDArray> expectedGradient = new LinkedHashMap<>();

            for (int j = 0; j < net.getnLayers(); j++) {
                //Generate test gradient:
                INDArray wGrad = Nd4j.rand(nIns[j], nOuts[j]);
                INDArray bGrad = Nd4j.rand(1, nOuts[j]);

                String wKey = j + "_" + DefaultParamInitializer.WEIGHT_KEY;
                String bKey = j + "_" + DefaultParamInitializer.BIAS_KEY;

                gradient.setGradientFor(wKey, wGrad);
                gradient.setGradientFor(bKey, bGrad);

                //Also put copy of gradient through separate layer updaters to compare
                Gradient layerGradient = new DefaultGradient();
                layerGradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wGrad.dup());
                layerGradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, bGrad.dup());

//                uArr[j].getConfig().applySchedules(0, net.getLayer(j).conf().getLearningRateByParam("W"));
                for (String s : layerGradient.gradientForVariable().keySet()) {
                    expectedGradient.put(j + "_" + s, layerGradient.getGradientFor(s));
                }
            }

            updater.update(net, gradient, i, 0, 1);
            assertEquals(gradient.gradientForVariable(), expectedGradient);
        }
    }


    @Test
    public void testSetGetUpdater() {

        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr).momentum(0.6).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(Nd4j.rand(5, nIn), Nd4j.rand(5, nOut)); //Fit, to initialize optimizer/updater

        Updater updater = net.getUpdater();
        assertTrue(updater instanceof MultiLayerUpdater);

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater()); //Should be identical object
    }

    @Test
    public void testSetGetUpdater2() {
        //Same as above test, except that we are doing setUpdater on a new network
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;
        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr).momentum(0.6).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater()); //Should be identical object
    }


    @Test
    public void testEpsilon() {
        //Test epsilon setting - adagrad
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).epsilon(0.123).build())
                        .layer(2, new OutputLayer.Builder().nIn(2).nOut(2).epsilon(0.456).build()).build();

        assertEquals(1e-6, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(0).getLayer()).getEpsilon(),
                        0.0);
        assertEquals(0.123, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(1).getLayer()).getEpsilon(),
                        0.0);
        assertEquals(0.456, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(2).getLayer()).getEpsilon(),
                        0.0);

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //        net.fit(Nd4j.create(1,2), Nd4j.create(1,2));
        MultiLayerUpdater updater = (MultiLayerUpdater) net.getUpdater();
        List<UpdaterBlock> l = updater.getUpdaterBlocks();

        AdaGrad adaGrad = (AdaGrad) l.get(0).getGradientUpdater().getConfig();
        assertEquals(1e-6, adaGrad.getEpsilon(), 0.0);

        AdaGrad adaGrad1 = (AdaGrad) l.get(1).getGradientUpdater().getConfig();
        assertEquals(0.123, adaGrad1.getEpsilon(), 0.0);

        AdaGrad adaGrad2 = (AdaGrad) l.get(2).getGradientUpdater().getConfig();
        assertEquals(0.456, adaGrad2.getEpsilon(), 0.0);


        //Test epsilon setting - adadelta
        conf = new NeuralNetConfiguration.Builder().updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).epsilon(0.123).build())
                        .layer(2, new OutputLayer.Builder().nIn(2).nOut(2).epsilon(0.456).build()).build();

        assertEquals(1e-6, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(0).getLayer()).getEpsilon(),
                        0.0);
        assertEquals(0.123, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(1).getLayer()).getEpsilon(),
                        0.0);
        assertEquals(0.456, ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(2).getLayer()).getEpsilon(),
                        0.0);

        net = new MultiLayerNetwork(conf);
        net.init();
        updater = (MultiLayerUpdater) net.getUpdater();
        l = updater.getUpdaterBlocks();

        AdaDelta adaDelta = (AdaDelta) l.get(0).getGradientUpdater().getConfig();
        assertEquals(1e-6, adaDelta.getEpsilon(), 0.0);

        AdaDelta adaDelta1 = (AdaDelta) l.get(1).getGradientUpdater().getConfig();
        assertEquals(0.123, adaDelta1.getEpsilon(), 0.0);

        AdaDelta adaDelta2 = (AdaDelta) l.get(2).getGradientUpdater().getConfig();
        assertEquals(0.456, adaDelta2.getEpsilon(), 0.0);
    }

    @Test
    public void testPretrain() {

        gradients = Nd4j.ones(nIn * nOut + nOut + nIn);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        INDArray vbiasGradient = gradients.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradient.setFlattenedGradient(gradients);


        //Test with pretrain = true
        double lr = 0.05;
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbiasGradient);


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(lr).seed(42)
                        .updater(org.deeplearning4j.nn.conf.Updater.SGD)
                        .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                        .lossFunction(LossFunctions.LossFunction.COSINE_PROXIMITY)
                                        .activation(Activation.IDENTITY).nIn(nIn).nOut(nOut).build())
                        .build();
        int numParams = conf.getLayer().initializer().numParams(conf);
        conf.setPretrain(true);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        DefaultGradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        INDArray vbg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);
        gradientCopyPreUpdate.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbg);

        updater.update(layer, gradient, -1, 0, 1);

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.layerConf().getLearningRate(), 1e-4);


        //Test with pretrain == false
        gradients = Nd4j.ones(nIn * nOut + nOut + nIn);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        vbiasGradient = gradients.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbiasGradient);
        gradient.setFlattenedGradient(gradients);

        gradientCopyPreUpdate = new DefaultGradient();
        g = gradients.dup();
        wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        vbg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);
        gradientCopyPreUpdate.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbg);
        gradientCopyPreUpdate.setFlattenedGradient(g);

        conf.setPretrain(false);
        params = Nd4j.create(1, numParams);
        layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        updater = UpdaterCreator.getUpdater(layer);

        updater.update(layer, gradient, -1, 0, 1);

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            //            System.out.println(entry.getKey());
            val = entry.getValue();
            if (!entry.getKey().equals("vb")) {
                gradExpected = val.mul(lr);
            } else {
                //With pretrain == false, we shouldn't be updating the pretrain params (vb)
                gradExpected = val;
            }
            //            System.out.println(gradExpected + "\t" + gradient.getGradientFor(entry.getKey()));
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.layerConf().getLearningRate(), 1e-4);
    }

    @Test
    public void testEpsilonAllUpdaters() {

        double e = 7e-2;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().epsilon(e).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build())
                        .layer(3, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(4, new OutputLayer.Builder().nIn(2).nOut(2)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAMAX).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.fit(Nd4j.create(1, 2), Nd4j.create(1, 2));


        MultiLayerUpdater updater = (MultiLayerUpdater) net.getUpdater();
        List<UpdaterBlock> l = updater.getUpdaterBlocks();

        Adam adam = (Adam) l.get(0).getGradientUpdater().getConfig(); //u0.updaterForVariable.get("W");
        assertEquals(e, adam.getEpsilon(), 0.0);

        RmsProp rmsProp = (RmsProp) l.get(1).getGradientUpdater().getConfig(); //u1.updaterForVariable.get("W");
        assertEquals(e, rmsProp.getEpsilon(), 0.0);

        AdaDelta adaDelta = (AdaDelta) l.get(2).getGradientUpdater().getConfig(); //u2.updaterForVariable.get("W");
        assertEquals(e, adaDelta.getEpsilon(), 0.0);

        AdaGrad adaGrad = (AdaGrad) l.get(3).getGradientUpdater().getConfig(); //u3.updaterForVariable.get("W");
        assertEquals(e, adaGrad.getEpsilon(), 0.0);

        AdaMax adaMax = (AdaMax) l.get(4).getGradientUpdater().getConfig(); //u3.updaterForVariable.get("W");
        assertEquals(e, adaMax.getEpsilon(), 0.0);
    }

    @Test
    public void testUpdaterBlockMlnAndCG() {
        for (int i = 0; i < 2; i++) {

            List<UpdaterBlock> blocks;
            if (i == 0) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(0.5).list()
                                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).name("l0")
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
                                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).name("l1")
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAM).biasLearningRate(0.25)
                                                .build())
                                .layer(2, new DenseLayer.Builder().nIn(10).nOut(10).name("l2")
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build())
                                .layer(3, new DenseLayer.Builder().nIn(10).nOut(10).name("l3")
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                                .layer(4, new OutputLayer.Builder().nIn(10).nOut(10).name("l4")
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAMAX).build())
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
                blocks = u.getUpdaterBlocks();
            } else {
                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(0.5)
                                .graphBuilder().addInputs("in")
                                .addLayer("l0", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAM).build(), "in")
                                .addLayer("l1", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAM).biasLearningRate(0.25)
                                                .build(), "l0")
                                .addLayer("l2", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build(), "l1")
                                .addLayer("l3", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build(), "l2")
                                .addLayer("l4", new OutputLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAMAX).build(), "l3")
                                .setOutputs("l4").build();

                ComputationGraph net = new ComputationGraph(conf);
                net.init();

                ComputationGraphUpdater u = net.getUpdater();
                blocks = u.getUpdaterBlocks();
            }


            //Expect 4 blocks: (layer0 W, layer0 B, layer 1 W], [layer 1 B], [layer 2 W, layer 2 B],
            // [layer 3 W, layer 3 B], [layer 4 W, layer 4 B]
            assertEquals(5, blocks.size());


            //Check first updater block:
            UpdaterBlock ub0 = blocks.get(0);
            assertEquals(3, ub0.getLayersAndVariablesInBlock().size());
            assertEquals("l0", ub0.getLayersAndVariablesInBlock().get(0).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub0.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l0", ub0.getLayersAndVariablesInBlock().get(1).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub0.getLayersAndVariablesInBlock().get(1).getParamName());
            assertEquals("l1", ub0.getLayersAndVariablesInBlock().get(2).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub0.getLayersAndVariablesInBlock().get(2).getParamName());

            int nParams0 = 10 * 10 + 10 + 10 * 10;
            assertEquals(0, ub0.getParamOffsetStart());
            assertEquals(nParams0, ub0.getParamOffsetEnd());
            int nUpdaterVals0 = 2 * nParams0; //2x for Adam
            assertEquals(0, ub0.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0, ub0.getUpdaterViewOffsetEnd());

            //Check second updater block:
            UpdaterBlock ub1 = blocks.get(1);
            assertEquals(1, ub1.getLayersAndVariablesInBlock().size());
            assertEquals("l1", ub1.getLayersAndVariablesInBlock().get(0).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub1.getLayersAndVariablesInBlock().get(0).getParamName());

            int nParams1 = 10;
            assertEquals(nParams0, ub1.getParamOffsetStart());
            assertEquals(nParams0 + nParams1, ub1.getParamOffsetEnd());
            int nUpdaterVals1 = 2 * nParams1; //2x for Adam
            assertEquals(nUpdaterVals0, ub1.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1, ub1.getUpdaterViewOffsetEnd());

            //Check third updater block:
            UpdaterBlock ub2 = blocks.get(2);
            assertEquals(2, ub2.getLayersAndVariablesInBlock().size());
            assertEquals("l2", ub2.getLayersAndVariablesInBlock().get(0).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub2.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l2", ub2.getLayersAndVariablesInBlock().get(1).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub2.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams2 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1, ub2.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2, ub2.getParamOffsetEnd());
            int nUpdaterVals2 = 2 * nParams2; //2x for Adadelta
            assertEquals(nUpdaterVals0 + nUpdaterVals1, ub2.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2, ub2.getUpdaterViewOffsetEnd());

            //Check fourth updater block:
            UpdaterBlock ub3 = blocks.get(3);
            assertEquals(2, ub3.getLayersAndVariablesInBlock().size());
            assertEquals("l3", ub3.getLayersAndVariablesInBlock().get(0).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub3.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l3", ub3.getLayersAndVariablesInBlock().get(1).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub3.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams3 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1 + nParams2, ub3.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3, ub3.getParamOffsetEnd());
            int nUpdaterVals3 = nParams3; //1x for AdaGrad
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2, ub3.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3, ub3.getUpdaterViewOffsetEnd());

            //Check fifth updater black
            UpdaterBlock ub4 = blocks.get(4);
            assertEquals(2, ub4.getLayersAndVariablesInBlock().size());
            assertEquals("l4", ub4.getLayersAndVariablesInBlock().get(0).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub4.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l4", ub4.getLayersAndVariablesInBlock().get(1).getLayer().conf().getLayer().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub4.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams4 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3, ub4.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3 + nParams4, ub4.getParamOffsetEnd());
            int nUpdaterVals4 = 2 * nParams4; //2x for AdaGrad
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3,
                            ub4.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3 + nUpdaterVals4,
                            ub4.getUpdaterViewOffsetEnd());
        }
    }


    @Test
    public void testUpdaterBlockVae() {

        List<UpdaterBlock> blocks;
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().learningRate(0.5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAM).list()
                                        .layer(0, new VariationalAutoencoder.Builder().nIn(8).nOut(12)
                                                        .encoderLayerSizes(10, 11).decoderLayerSizes(13, 14).build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
        blocks = u.getUpdaterBlocks();


        //Expect 2 blocks: Standard, and pretrain-only params
        assertEquals(2, blocks.size());


        //Check first updater block (all backprop-only params)
        UpdaterBlock ub0 = blocks.get(0);
        List<String> expParams = Arrays.asList("e0W", "e0b", "e1W", "e1b", "pZXMeanW", "pZXMeanb");
        List<String> actParams = new ArrayList<>();
        for (UpdaterBlock.ParamState vs : ub0.getLayersAndVariablesInBlock()) {
            actParams.add(vs.getParamName());
        }
        assertEquals(expParams, actParams);

        //Check second updater block
        UpdaterBlock ub1 = blocks.get(1);
        expParams = Arrays.asList("pZXLogStd2W", "pZXLogStd2b", "d0W", "d0b", "d1W", "d1b", "pXZW", "pXZb");
        actParams = new ArrayList<>();
        for (UpdaterBlock.ParamState vs : ub1.getLayersAndVariablesInBlock()) {
            actParams.add(vs.getParamName());
        }
        assertEquals(expParams, actParams);
    }


    @Test
    public void testUpdaterConfigDeprecatedMethods() {
        //.momentum(), .epsilon() etc - these are now deprecated, but we still want them to work as expected
        // until they are actually removed

        double lr = 0.75;
        double eps = 0.65;
        double adamMean = 0.1;
        double adamVar = 0.2;
        double momentum = 0.3;
        Map<Integer, Double> momentumSchedule = new HashMap<>();
        momentumSchedule.put(0, 0.35);
        momentumSchedule.put(10, 0.34);
        double rmsDecay = 0.4;

        for (boolean useEnum : new boolean[] {true, false}) {
            NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                            //Multiple updaters
                            .learningRate(lr).epsilon(eps)
                            //Adam
                            .adamMeanDecay(adamMean).adamVarDecay(adamVar)
                            //Momentum
                            .momentum(momentum).momentumAfter(momentumSchedule)
                            //RMSProp
                            .rmsDecay(rmsDecay).list();
            if (useEnum) {
                listBuilder.layer(0,
                                new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
                                .layer(2, new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build())
                                .layer(3, new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                                .layer(4, new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                                .layer(5, new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build());
            } else {
                listBuilder.layer(0, new DenseLayer.Builder().nIn(10).nOut(10).updater(new Sgd()).build())
                                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam()).build())
                                .layer(2, new DenseLayer.Builder().nIn(10).nOut(10).updater(new AdaDelta()).build())
                                .layer(3, new DenseLayer.Builder().nIn(10).nOut(10).updater(new Nesterovs()).build())
                                .layer(4, new DenseLayer.Builder().nIn(10).nOut(10).updater(new AdaGrad()).build())
                                .layer(5, new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp()).build());
            }


            MultiLayerConfiguration conf = listBuilder.build();

            Sgd sgd = (Sgd) ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(0).getLayer()).getIUpdater();
            assertEquals(lr, sgd.getLearningRate(), 1e-6);

            Adam adam = (Adam) ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(1).getLayer()).getIUpdater();
            assertEquals(lr, adam.getLearningRate(), 1e-6);
            assertEquals(eps, adam.getEpsilon(), 1e-6);
            assertEquals(adamMean, adam.getBeta1(), 1e-6);
            assertEquals(adamVar, adam.getBeta2(), 1e-6);

            //Adadelta: no params

            Nesterovs nesterovs = (Nesterovs) ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(3).getLayer())
                            .getIUpdater();
            assertEquals(lr, nesterovs.getLearningRate(), 1e-6);
            assertEquals(momentum, nesterovs.getMomentum(), 1e-6);
            assertEquals(momentumSchedule, nesterovs.getMomentumSchedule());

            AdaGrad adagrad = (AdaGrad) ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(4).getLayer())
                            .getIUpdater();
            assertEquals(lr, adagrad.getLearningRate(), 1e-6);
            assertEquals(eps, adagrad.getEpsilon(), 1e-6);

            RmsProp rmsProp = (RmsProp) ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getConf(5).getLayer())
                            .getIUpdater();
            assertEquals(lr, rmsProp.getLearningRate(), 1e-6);
            assertEquals(rmsDecay, rmsProp.getRmsDecay(), 1e-6);
            assertEquals(eps, rmsProp.getEpsilon(), 1e-6);
        }
    }
}
