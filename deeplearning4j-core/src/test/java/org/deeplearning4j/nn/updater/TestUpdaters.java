package org.deeplearning4j.nn.updater;


import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.*;

public class TestUpdaters {

	int nIn = 3;
	int nOut = 2;
	INDArray weightGradient = Nd4j.ones(nIn,nOut);
	INDArray biasGradient = Nd4j.ones(1,nOut);
	Gradient gradient = new DefaultGradient();

	@Before
	public void beforeDo(){
		gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
		gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
	}

	@Test
	public void testAdaDeltaUpdate(){
		double lr = 1e-2;
		double rho = 0.85;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).rho(rho)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

		// calculations for one iteration / update
		INDArray msgW, msdxW, dxSquaredW, weightGradExpected, weightGradExpected2,
				msgB, msdxB, dxSquaredB,biasGradExpected, biasGradExpected2;

		msgW = Nd4j.zeros(weightGradient.shape());
		msdxW = Nd4j.zeros(weightGradient.shape());
		msgW.muli(rho);
		msgW.addi(1-rho).muli(weightGradient.mul(weightGradient));

		weightGradExpected = Transforms.sqrt(msdxW.add(Nd4j.EPS_THRESHOLD))
				.divi(Transforms.sqrt(msgW.add(Nd4j.EPS_THRESHOLD))).muli(weightGradient);

		msgB = Nd4j.zeros(biasGradient.shape());
		msdxB = Nd4j.zeros(biasGradient.shape());
		msgB.muli(rho);
		msgB.addi(1-rho).muli(biasGradient.mul(biasGradient));

		biasGradExpected = Transforms.sqrt(msdxB.add(Nd4j.EPS_THRESHOLD))
				.divi(Transforms.sqrt(msgB.add(Nd4j.EPS_THRESHOLD))).muli(weightGradient);


		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(rho, layer.conf().getRho(), 1e-4);


		// calculations for two iterations / updates
		Gradient gradient2 = new DefaultGradient();
		gradient2.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradExpected);
		gradient2.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradExpected);

		updater.update(layer, gradient2, -1);

		msdxW.muli(rho);
		dxSquaredW = weightGradExpected.mul(weightGradExpected);
		msdxW.addi(dxSquaredW.muli(1 - rho));

		msgW.muli(rho);
		msgW.addi(1-rho).muli(weightGradExpected.mul(weightGradExpected));

		weightGradExpected2 = Transforms.sqrt(msdxW.add(Nd4j.EPS_THRESHOLD))
				.divi(Transforms.sqrt(msgW.add(Nd4j.EPS_THRESHOLD))).muli(weightGradExpected);

		msdxB.muli(rho);
		dxSquaredB = biasGradExpected.mul(biasGradExpected);
		msdxB.addi(dxSquaredB.muli(1 - rho));

		msgB.muli(rho);
		msgB.addi(1-rho).muli(biasGradExpected.mul(biasGradExpected));
		biasGradExpected2 = Transforms.sqrt(msdxB.add(Nd4j.EPS_THRESHOLD))
				.divi(Transforms.sqrt(msgB.add(Nd4j.EPS_THRESHOLD))).muli(biasGradExpected);

		INDArray weightGradActual2 = gradient2.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual2 = gradient2.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected2, weightGradActual2);
		assertEquals(biasGradExpected2, biasGradActual2);
		assertEquals(rho, layer.conf().getRho(), 1e-4);

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

		// calculations
		INDArray weightGradExpected = Transforms.sqrt(weightGradient.mul(weightGradient)
				.add(1e-8)).rdiv(lr).mul(weightGradient);

		INDArray biasGradExpected = Transforms.sqrt(biasGradient.mul(biasGradient)
				.add(1e-8)).rdiv(lr).mul(biasGradient);

		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(lr, layer.conf().getLr(), 1e-4);
	}

	@Test
	public void testAdamUpdater(){
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

		// calculations
		INDArray mW, vW, weightGradExpected, mB, vB, biasGradExpected;

		double beta1t = FastMath.pow(beta1, iteration);
		double beta2t = FastMath.pow(beta2, iteration);
		double alphat = lr * FastMath.sqrt(((1-beta2t)/(1-beta1t)));

		mW = Nd4j.zeros(weightGradient.shape());
		vW = Nd4j.zeros(weightGradient.shape());

		mW.muli(beta1).addi(weightGradient.mul(1.0-beta1));
		vW.muli(beta2).addi(weightGradient.mul(weightGradient).mul(1.0 - beta2));
		weightGradExpected = mW.mul(alphat).divi(Transforms.sqrt(vW).addi(1e-8));

		mB = Nd4j.zeros(biasGradient.shape());
		vB = Nd4j.zeros(biasGradient.shape());

		mB.muli(beta1).addi(biasGradient.mul(1.0-beta1));
		vB.muli(beta2).addi(biasGradient.mul(biasGradient).mul(1.0-beta2));
		biasGradExpected = mB.mul(alphat).divi(Transforms.sqrt(vB).addi(1e-8));

		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(lr, layer.conf().getLr(), 1e-4);

	}

	@Test
	public void testNestorovsUpdater(){
		double lr = 1e-2;
		double mu = 0.6;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr).momentum(mu)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

		// calculations
		INDArray vW, vPrevW, weightGradExpected, vB, vPrevB, biasGradExpected;
		vW = Nd4j.zeros(weightGradient.shape());
		vPrevW = vW;
		vW = vPrevW.mul(mu).subi(weightGradient.mul(lr));
		weightGradExpected = vPrevW.muli(mu).addi(vW.mul(-mu - 1));

		vB = Nd4j.zeros(biasGradient.shape());
		vPrevB = vB;
		vB = vPrevB.mul(mu).subi(biasGradient.mul(lr));
		biasGradExpected = vPrevB.muli(mu).addi(vB.mul(-mu - 1));

		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(mu, layer.conf().getMomentum(), 1e-4);
	}

	@Test
	public void testRMSPropUpdater(){
		double lr = 0.01;
		double rmsDecay = 0.25;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.rmsDecay(rmsDecay)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build())
				.build();

		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

		// calculations
		INDArray lastGW, lastGB;

		lastGW = Nd4j.zeros(weightGradient.shape());
		lastGB = Nd4j.zeros(biasGradient.shape());

		lastGW.muli(rmsDecay).addi(weightGradient.mul(weightGradient).muli(1 - rmsDecay));
		INDArray weightGradExpected = weightGradient.mul(lr).div(Transforms.sqrt(lastGW.add(Nd4j.EPS_THRESHOLD)));

		lastGB.muli(rmsDecay).addi(biasGradient.mul(biasGradient).muli(1 - rmsDecay));
		INDArray biasGradExpected = biasGradient.mul(lr).div(Transforms.sqrt(lastGB.add(Nd4j.EPS_THRESHOLD)));

		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(rmsDecay, layer.conf().getRmsDecay(), 1e-4);

	}

	@Test
	public void testSGDUpdater(){
		double lr = 0.01;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder().nIn(nIn)
						.nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
				.build();
		
		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);

		updater.update(layer, gradient, -1);

		// calculations
		INDArray weightGradExpected = weightGradient.mul(lr);
		INDArray biasGradExpected = biasGradient.mul(lr);
		
		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);
		
		assertEquals(weightGradExpected, weightGradActual);
		assertEquals(biasGradExpected, biasGradActual);
		assertEquals(lr, layer.conf().getLr(), 1e-4);

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
