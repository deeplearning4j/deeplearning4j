package org.deeplearning4j.nn.updater;


import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.optimize.terminations.EpsTermination;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.*;

public class TestUpdaters {

	int nIn = 3;
	int nOut = 2;
    double epsilon = 1e-8;
	INDArray weightGradient = Nd4j.ones(nIn,nOut);
	INDArray biasGradient = Nd4j.ones(1,nOut);
	Gradient gradient = new DefaultGradient();
    INDArray val, gradExpected;
    String key;


	@Before
	public void beforeDo(){
		gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
		gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
	}

	@Test
	public void testAdaDeltaUpdate(){
        INDArray dxSquared;
        Map<String, INDArray> msg = new HashMap<>();
        Map<String, INDArray> msdx = new HashMap<>();

		double rho = 0.85;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.rho(rho)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		Gradient gradientDup = new DefaultGradient();
		gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
		gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

		for (int i = 0; i < 2; i++) {
            updater.update(layer, gradient, i);

            // calculations for one iteration / update

            for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
                key = entry.getKey();
                val = entry.getValue();
				INDArray msgTmp = msg.get(key);
				INDArray msdxTmp = msdx.get(key);

                if(msgTmp == null) {
                    msgTmp = Nd4j.zeros(val.shape());
                    msdxTmp = Nd4j.zeros(val.shape());
                }

                msgTmp.muli(rho);
                msgTmp.addi(1 - rho).muli(val.mul(val));

                gradExpected = Transforms.sqrt(msdxTmp.add(Nd4j.EPS_THRESHOLD))
						.divi(Transforms.sqrt(msgTmp.add(Nd4j.EPS_THRESHOLD))).muli(val);
				gradientDup.setGradientFor(key, gradExpected);
                assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));

                msdxTmp.muli(rho);
                dxSquared = gradExpected.mul(gradExpected);
                msdxTmp.addi(dxSquared.muli(1 - rho));

				msg.put(key, msgTmp);
				msdx.put(key, msdxTmp);
            }
            assertEquals(rho, layer.conf().getLayer().getRho(), 1e-4);
        }

	}

	@Test
	public void testAdaGradUpdater() {
		double lr = 1e-2;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = Transforms.sqrt(val.mul(val).add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
		assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
	}


	@Test
	public void testAdamUpdater(){
        INDArray m, v;
        double lr = 0.01;
		int iteration = 0;
		double beta1 = 0.9; // TODO allow passing in betas and update test
		double beta2 = 0.999;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).iterations(iteration)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, iteration);

		double beta1t = FastMath.pow(beta1, iteration);
		double beta2t = FastMath.pow(beta2, iteration);
		double alphat = lr * FastMath.sqrt(((1-beta2t)/(1-beta1t)));

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0-beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0-beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }

        assertEquals(beta1, layer.conf().getLayer().getAdamMeanDecay(), 1e-4);
        assertEquals(beta2, layer.conf().getLayer().getAdamVarDecay(), 1e-4);

	}

	@Test
	public void testNestorovsUpdater(){
		double lr = 1e-2;
		double mu = 0.6;
        INDArray v, vPrev;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).momentum(mu)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            v = Nd4j.zeros(val.shape());
            vPrev = v;
            v = vPrev.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(v.mul(-mu - 1));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }

		assertEquals(mu, layer.conf().getLayer().getMomentum(), 1e-4);
	}

	@Test
	public void testmomentumAfterUpdaterSingleLayer(){
		double lr = 1e-2;
		double mu = 0.6;
		Map<Integer,Double> momentumAfter = new HashMap<>();
		momentumAfter.put(1, 0.2);
		int iterations = 2;
		Map<String, INDArray> v = new HashMap<>();

		INDArray vPrev;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).momentum(mu).momentumAfter(momentumAfter).schedules(true).iterations(iterations)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradient, i);

            for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
				if(momentumAfter !=null)
					mu = (momentumAfter.containsKey(i)) ? momentumAfter.get(i) : mu;
				key = entry.getKey();
				val = entry.getValue();
				INDArray vTmp = v.get(key);

				if(vTmp == null)
                    vTmp = Nd4j.zeros(val.shape());
                vPrev = vTmp;
                vTmp = vPrev.mul(mu).subi(val.mul(lr));
                gradExpected = vPrev.muli(mu).addi(vTmp.mul(-mu - 1));
                gradientDup.setGradientFor(key, gradExpected);

                assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
				v.put(key, vTmp);
            }

            assertEquals(momentumAfter, layer.conf().getLayer().getMomentumAfter());
        }
    }


	@Test
	public void testMomentumAfterMLN(){
		double lr = 1e-2;
		double mu = 0.6;
		Map<Integer,Double> momentumAfter = new HashMap<>();
		momentumAfter.put(1, 0.2);
		int iterations = 2;
		int nLayers = 2;
		int[] nIns = {4,2};
		int[] nOuts = {2,3};

		Map<String, INDArray> v = new HashMap<>();
		INDArray vPrev;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).momentum(mu).momentumAfter(momentumAfter).schedules(true).iterations(iterations)
				.list(nLayers)
				.layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.backprop(true).pretrain(false)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		Updater updater = UpdaterCreator.getUpdater(net);

		Gradient g = new DefaultGradient();
		String wKey, bKey;
		INDArray vTmp;

		for (int j=0; j < nLayers; j++){
			wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
			g.setGradientFor(wKey, weightGradient);
			bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY ;
			g.setGradientFor(bKey, biasGradient);
		}

		Gradient gDup = new DefaultGradient();
		for (int k=0; k < nLayers; k++){
			wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
			gDup.setGradientFor(wKey, weightGradient);
			bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY ;
			gDup.setGradientFor(bKey, biasGradient);
		}

		for (int i = 0; i < 2; i++) {
			updater.update(net, g, i);

			for (Map.Entry<String, INDArray> entry : gDup.gradientForVariable().entrySet()) {
				if(momentumAfter !=null)
					mu = (momentumAfter.containsKey(i)) ? momentumAfter.get(i) : mu;
				key = entry.getKey();
				val = entry.getValue();
				vTmp = v.get(key);

				if(vTmp == null)
					vTmp = Nd4j.zeros(val.shape());
				vPrev = vTmp;
				vTmp = vPrev.mul(mu).subi(val.mul(lr));
				gradExpected = vPrev.muli(mu).addi(vTmp.mul(-mu - 1));
				gDup.setGradientFor(key, gradExpected);

				assertEquals(gradExpected, g.getGradientFor(entry.getKey()));
				v.put(key, vTmp);
			}
			v =  new HashMap<>();
			assertEquals(lr, net.getLayer(1).conf().getLayer().getLearningRate(), 1e-4);
		}
	}


	@Test
	public void testRMSPropUpdater(){
		double lr = 0.01;
		double rmsDecay = 0.25;
		Map<String, INDArray> lastG = new HashMap<>();


		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.rmsDecay(rmsDecay)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
			key = entry.getKey();
			val = entry.getValue();
            INDArray lastGTmp = lastG.get(key);

			if(lastGTmp==null)
				lastGTmp = Nd4j.zeros(val.shape());

			lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(Nd4j.EPS_THRESHOLD)));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
			lastG.put(key, lastGTmp);
        }
		assertEquals(rmsDecay, layer.conf().getLayer().getRmsDecay(), 1e-4);
	}

	@Test
	public void testSGDUpdater(){
		double lr = 0.05;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.build();
		
		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
	}


	@Test
	public void testNoOpUpdater(){
		Random r = new Random(12345L);
		double lr = 0.5;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);
		
		for( int i=0; i<weightGradient.length(); i++ ) weightGradient.putScalar(i, r.nextDouble());
		for( int i=0; i<biasGradient.length(); i++ ) biasGradient.putScalar(i, r.nextDouble());

		updater.update(layer, gradient, -1);
		
		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);
		
		assertEquals(weightGradient, weightGradActual);
		assertEquals(biasGradient, biasGradActual);

	}

	@Test
	public void testLearningRateAfterSingleLayer(){
		double lr = 1e-2;
		Map<Integer,Double> learningRateAfter = new HashMap<>();
		learningRateAfter.put(1, 0.2);
		int iterations = 2;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).learningRateAfter(learningRateAfter).schedules(true).iterations(iterations)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		Gradient gradientDup = new DefaultGradient();
		gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
		gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

		for (int i = 0; i < 2; i++) {
			updater.update(layer, gradient, i);

			for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
				if(learningRateAfter !=null)
					lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
				key = entry.getKey();
				val = entry.getValue();
				gradExpected = val.mul(lr);
				gradientDup.setGradientFor(key, gradExpected);
				assertEquals(gradExpected, gradient.getGradientFor(key));
			}
			assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
		}
	}

	@Test
	public void testLearningRateAfterMLN(){
		double lr = 1e-2;
		Map<Integer,Double> learningRateAfter = new HashMap<>();
		learningRateAfter.put(1, 0.2);
		int iterations = 2;
		int nLayers = 2;
		int[] nIns = {4,2};
		int[] nOuts = {2,3};

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).learningRateAfter(learningRateAfter).schedules(true).iterations(iterations)
				.list(nLayers)
				.layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.backprop(true).pretrain(false)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		Updater updater = UpdaterCreator.getUpdater(net);

		Gradient g = new DefaultGradient();
		String wKey, bKey;

		for (int j=0; j < nLayers; j++){
			wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
			g.setGradientFor(wKey, weightGradient);
			bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY ;
			g.setGradientFor(bKey, biasGradient);
		}

		Gradient gDup = new DefaultGradient();
		for (int k=0; k < nLayers; k++){
			wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
			gDup.setGradientFor(wKey, weightGradient);
			bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY ;
			gDup.setGradientFor(bKey, biasGradient);
		}

		for (int i = 0; i < 2; i++) {
			updater.update(net, g, i);

			for (Map.Entry<String, INDArray> entry : gDup.gradientForVariable().entrySet()) {
				if(learningRateAfter !=null)
					lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
				key = entry.getKey();
				val = entry.getValue();
				gradExpected = val.mul(lr);
				gDup.setGradientFor(key, gradExpected);
				assertEquals(gradExpected, g.getGradientFor(key));
			}
			assertEquals(lr, net.getLayer(1).conf().getLayer().getLearningRate(), 1e-4);
		}
	}


	@Test
    public void testLearningRateScoreDecay(){
        double lr = 0.01;
        double lrScoreDecay = 0.10;
        int nLayers = 2;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};
		int oldScore = 1;
		int newScore = 1;
		int iteration = 3;
        INDArray gradientW = Nd4j.ones(nIns[0],nOuts[0]);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).learningRateScoreBasedDecayRate(lrScoreDecay)
                .list(nLayers)
                .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

		ConvexOptimizer opt = new StochasticGradientDescent(net.getDefaultConfiguration(), new NegativeDefaultStepFunction(), null, net);
        opt.checkTerminalConditions(gradientW, oldScore, newScore, iteration);
		assertEquals(lrScoreDecay, net.getLayer(0).conf().getLayer().getLrScoreBasedDecay(), 1e-4);
		assertEquals(lr*(lrScoreDecay + Nd4j.EPS_THRESHOLD), net.getLayer(0).conf().getLayer().getLearningRate(), 1e-4);

	}

    @Test
	public void testMultiLayerUpdater() throws Exception {
		Nd4j.getRandom().setSeed(12345L);
		int nLayers = 4;
		double lr = 0.03;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.learningRate(lr)
			.momentum(0.6)
			.list(nLayers)
			.layer(0, new DenseLayer.Builder().nIn(4).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
			.layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
			.layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
			.layer(3, new DenseLayer.Builder().nIn(7).nOut(8).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		Updater updater = UpdaterCreator.getUpdater(net);
		assertNotNull(updater);
		assertTrue(updater.getClass() == MultiLayerUpdater.class);
		
		Field f = MultiLayerUpdater.class.getDeclaredField("layerUpdaters");
		f.setAccessible(true);
		Updater[] updaters = (Updater[])f.get(updater);
		assertNotNull(updaters);
		assertTrue(updaters.length == nLayers);
		assertTrue(updaters[0] instanceof SgdUpdater );
		assertTrue(updaters[1] instanceof NoOpUpdater );
		assertTrue(updaters[2] instanceof AdaGradUpdater );
		assertTrue(updaters[3] instanceof NesterovsUpdater );
		
		Updater[] uArr = new Updater[4];
		uArr[0] = new SgdUpdater();
		uArr[1] = new NoOpUpdater();
		uArr[2] = new AdaGradUpdater();
		uArr[3] = new NesterovsUpdater();
		
		int[] nIns = {4,5,6,7};
		int[] nOuts = {5,6,7,8};
		
		for( int i=0; i<5; i++ ){
			Gradient gradient = new DefaultGradient();
			Map<String,INDArray> expectedGradient = new HashMap<>();
			
			for( int j=0; j<nLayers; j++ ){
				//Generate test gradient:
				INDArray wGrad = Nd4j.rand(nIns[j],nOuts[j]);
				INDArray bGrad = Nd4j.rand(1,nOuts[j]);
				
				String wKey = j + "_" + DefaultParamInitializer.WEIGHT_KEY;
				String bKey = j + "_" + DefaultParamInitializer.BIAS_KEY;
				
				gradient.setGradientFor(wKey, wGrad);
				gradient.setGradientFor(bKey, bGrad);
				
				//Also put copy of gradient through separate layer updaters to compare
				Gradient layerGradient = new DefaultGradient();
				layerGradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wGrad.dup());
				layerGradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, bGrad.dup());
				uArr[j].update(net.getLayer(j), layerGradient, i);
				for( String s : layerGradient.gradientForVariable().keySet() ){
					expectedGradient.put(j+"_"+s,layerGradient.getGradientFor(s));
				}
			}
			
			updater.update(net, gradient, i);
			assertTrue(gradient.gradientForVariable().equals(expectedGradient));
		}
	}



}
