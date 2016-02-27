package org.deeplearning4j.nn.updater;


import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
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
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
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

import static org.junit.Assert.*;

public class TestUpdaters {

	protected int nIn = 3;
	protected int nOut = 2;
	protected double epsilon = 1e-8;
	protected INDArray weightGradient = Nd4j.ones(nIn,nOut);
	protected INDArray biasGradient = Nd4j.ones(1,nOut);
	protected Gradient gradient = new DefaultGradient();
	protected INDArray val, gradExpected;
	protected String key;


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
            updater.update(layer, gradient, i, 1);

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

		updater.update(layer, gradient, -1, 1);

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
		double beta1 = 0.8;
		double beta2 = 0.888;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).iterations(iteration).adamMeanDecay(beta1).adamVarDecay(beta2)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, iteration, 1);

		double beta1t = FastMath.pow(beta1, iteration);
		double beta2t = FastMath.pow(beta2, iteration);
		double alphat = lr * FastMath.sqrt(1-beta2t)/(1-beta1t);

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

		updater.update(layer, gradient, -1, 1);

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

		updater.update(layer, gradient, -1, 1);

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

		updater.update(layer, gradient, -1, 1);

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

		updater.update(layer, gradient, -1, 1);
		
		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);
		
		assertEquals(weightGradient, weightGradActual);
		assertEquals(biasGradient, biasGradActual);

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
                .learningRate(lr).learningRateScoreBasedDecayRate(lrScoreDecay)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

		ConvexOptimizer opt = new StochasticGradientDescent(net.getDefaultConfiguration(), new NegativeDefaultStepFunction(), null, net);
        opt.checkTerminalConditions(gradientW, oldScore, newScore, iteration);
		assertEquals(lrScoreDecay, net.getLayer(0).conf().getLayer().getLrScoreBasedDecay(), 1e-4);
		assertEquals(lr*(lrScoreDecay + Nd4j.EPS_THRESHOLD), net.getLayer(0).conf().getLearningRateByParam("W"), 1e-4);

	}

	@Test
	public void testLearningRateScoreDecayLearningRateUnchanged() {

		DataSet ds = new IrisDataSetIterator(150,150).next();
		ds.normalizeZeroMeanZeroUnitVariance();

		Nd4j.getRandom().setSeed(12345);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.regularization(false)
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.learningRate(1.0)
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
	public void testMultiLayerUpdater() throws Exception {
		Nd4j.getRandom().setSeed(12345L);
		double lr = 0.03;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.learningRate(lr)
			.momentum(0.6)
			.list()
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
		assertTrue(updaters.length == net.getnLayers());
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
			
			for( int j=0; j< net.getnLayers(); j++ ){
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
				uArr[j].update(net.getLayer(j), layerGradient, i, 1);
				for( String s : layerGradient.gradientForVariable().keySet() ){
					expectedGradient.put(j+"_"+s,layerGradient.getGradientFor(s));
				}
			}
			
			updater.update(net, gradient, i, 1);
			assertTrue(gradient.gradientForVariable().equals(expectedGradient));
		}
	}


	@Test
	public void testSetGetUpdater(){

		Nd4j.getRandom().setSeed(12345L);
		double lr = 0.03;

		int nIn = 4;
		int nOut = 8;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.momentum(0.6)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
				.layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
				.layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.backprop(true).pretrain(false)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.fit(Nd4j.rand(5, nIn), Nd4j.rand(5, nOut));	//Fit, to initialize optimizer/updater

		Updater updater = net.getUpdater();
		assertTrue(updater instanceof MultiLayerUpdater);

		Updater newUpdater = UpdaterCreator.getUpdater(net);
		net.setUpdater(newUpdater);
		assertTrue(newUpdater == net.getUpdater());	//Should be identical object
	}

	@Test
	public void testSetGetUpdater2(){
		//Same as above test, except that we are doing setUpdater on a new network
		Nd4j.getRandom().setSeed(12345L);
		double lr = 0.03;
		int nIn = 4;
		int nOut = 8;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.momentum(0.6)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
				.layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
				.layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.backprop(true).pretrain(false)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		Updater newUpdater = UpdaterCreator.getUpdater(net);
		net.setUpdater(newUpdater);
		assertTrue(newUpdater == net.getUpdater());	//Should be identical object
	}

	@Test
	public void testUpdaterAggregationBasic(){

		Updater[] updaters = new Updater[]{
				new AdaDeltaUpdater(),
				new AdaGradUpdater(),
				new AdamUpdater(),
				new NesterovsUpdater(),
				new NoOpUpdater(),
				new RmsPropUpdater(),
				new SgdUpdater(),
		};

		org.deeplearning4j.nn.conf.Updater[] arr = new org.deeplearning4j.nn.conf.Updater[]{
				org.deeplearning4j.nn.conf.Updater.ADADELTA,
				org.deeplearning4j.nn.conf.Updater.ADAGRAD,
				org.deeplearning4j.nn.conf.Updater.ADAM,
				org.deeplearning4j.nn.conf.Updater.NESTEROVS,
				org.deeplearning4j.nn.conf.Updater.NONE,
				org.deeplearning4j.nn.conf.Updater.RMSPROP,
				org.deeplearning4j.nn.conf.Updater.SGD
		};

		DataSet dsTemp = new DataSet(Nd4j.rand(5,10), Nd4j.rand(5, 10));

		for(int i=0; i<updaters.length; i++ ){

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.iterations(1)
					.updater(arr[i])
					.list()
					.layer(0,new DenseLayer.Builder().nIn(10).nOut(10).build())
					.layer(1,new OutputLayer.Builder().nIn(10).nOut(10).build())
					.backprop(true).pretrain(false).build();

			MultiLayerNetwork net = new MultiLayerNetwork(conf);
			net.init();

			net.fit(dsTemp);

			Updater updater = net.getUpdater();

			System.out.println(i);
			assertNotNull(updater);
			assertTrue(updater instanceof MultiLayerUpdater);


			UpdaterAggregator ag = updater.getAggregator(true);
			Updater u2 = ag.getUpdater();

			assertEquals(u2,updater);

			UpdaterAggregator ag2 = updater.getAggregator(true);
			ag2.aggregate(updater);
			assertEquals(updater,ag2.getUpdater());
		}
	}

}
