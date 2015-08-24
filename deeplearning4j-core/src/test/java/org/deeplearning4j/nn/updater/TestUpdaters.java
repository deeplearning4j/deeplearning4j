package org.deeplearning4j.nn.updater;

import static org.junit.Assert.*;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

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
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestUpdaters {
	
	@Test
	public void testSGDUpdater(){
		int nIn = 3;
		int nOut = 5;
		double lr = 0.1;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut)
						.updater(org.deeplearning4j.nn.conf.Updater.SGD)
						.build())
				.build();
		
		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);
		
		INDArray weightGradient = Nd4j.ones(nIn,nOut);
		INDArray biasGradient = Nd4j.ones(1,nOut);
		
		Gradient gradient = new DefaultGradient();
		gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
		gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
		
		updater.update(layer, gradient, -1);
		
		INDArray weightGradExpected = Nd4j.ones(nIn,nOut).muli(lr);
		INDArray biasGradExpected = Nd4j.ones(1,nOut).muli(lr);
		
		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);
		
		assertTrue(weightGradExpected.equals(weightGradActual));
		assertTrue(biasGradExpected.equals(biasGradActual));
	}
	
	@Test
	public void testNoOpUpdater(){
		Random r = new Random(12345L);
		int nIn = 3;
		int nOut = 5;
		double lr = 0.1;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.learningRate(lr)
				.layer(new DenseLayer.Builder()
						.nIn(nIn).nOut(nOut)
						.updater(org.deeplearning4j.nn.conf.Updater.NONE)
						.build())
				.build();
		
		
		Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
		Updater updater = UpdaterCreator.getUpdater(layer);
		
		INDArray weightGradient = Nd4j.zeros(nIn,nOut);
		INDArray biasGradient = Nd4j.zeros(1,nOut);
		for( int i=0; i<weightGradient.length(); i++ ) weightGradient.putScalar(i, r.nextDouble());
		for( int i=0; i<biasGradient.length(); i++ ) biasGradient.putScalar(i, r.nextDouble());
		
		Gradient gradient = new DefaultGradient();
		gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
		gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());
		
		updater.update(layer, gradient, -1);
		
		INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
		INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);
		
		assertTrue(weightGradient.equals(weightGradActual));
		assertTrue(biasGradient.equals(biasGradActual));
	}
	
	@Test
	public void testMultiLayerUpdater() throws Exception {
		Nd4j.getRandom().setSeed(12345L);
		int nLayers = 4;
		double lr = 0.03;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.learningRate(lr)
			.momentum(0.5)
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
