package org.deeplearning4j.nn.updater;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
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
				.updater(org.deeplearning4j.nn.conf.Updater.SGD)
				.learningRate(lr)
				.layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
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
		
		System.out.println(weightGradActual);
		System.out.println(biasGradActual);
		assertTrue(weightGradExpected.equals(weightGradActual));
		assertTrue(biasGradExpected.equals(biasGradActual));
	}
	
	@Test
	public void testMultiLayerUpdater(){
		
		fail("Not implemented");
	}

}
