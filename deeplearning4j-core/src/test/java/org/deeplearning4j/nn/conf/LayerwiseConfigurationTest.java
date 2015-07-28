package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Alex Black 28/07/15
 *
 */
public class LayerwiseConfigurationTest {
	
	@Test
	public void testLayerConfigActivation(){
		//Idea: Set some common values for all layers. Then selectively override
		// the global config, and check they actually work.
		
		//Without layerwise override:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.activationFunction("sigmoid")
		.list(2)
		.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		assertTrue(conf.getConf(0).getActivationFunction().equals("sigmoid"));
		assertTrue(conf.getConf(1).getActivationFunction().equals("sigmoid"));
		
		//With:
		conf = new NeuralNetConfiguration.Builder()
			.activationFunction("sigmoid")
			.list(2)
			.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).activation("tanh").build() )
			.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).activation("relu").build() )
			.build();
		
		net = new MultiLayerNetwork(conf);
		net.init();
		
		assertTrue(conf.getConf(0).getActivationFunction().equals("tanh"));
		assertTrue(conf.getConf(1).getActivationFunction().equals("relu"));
	}
	
	@Test
	public void testLayerWeightInit(){
		//Without layerwise override:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.weightInit(WeightInit.ZERO)
		.list(2)
		.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		assertTrue(conf.getConf(0).getWeightInit() == WeightInit.ZERO);
		assertTrue(conf.getConf(1).getWeightInit() == WeightInit.ZERO);
		INDArray w0 = net.getLayer(0).getParam(DefaultParamInitializer.WEIGHT_KEY).linearView();
		for( int i=0; i<w0.length(); i++ ) assertTrue(w0.getDouble(i)==0.0);
		INDArray w1 = net.getLayer(1).getParam(DefaultParamInitializer.WEIGHT_KEY).linearView();
		for( int i=0; i<w1.length(); i++ ) assertTrue(w1.getDouble(i)==0.0);
		
		//With:
		conf = new NeuralNetConfiguration.Builder()
			.activationFunction("sigmoid")
			.weightInit(WeightInit.ZERO)
			.list(2)
			.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(10,11)).build() )
			.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(20,21)).build() )
			.build();
		
		net = new MultiLayerNetwork(conf);
		net.init();
		assertTrue(conf.getConf(0).getWeightInit() == WeightInit.DISTRIBUTION);
		assertTrue(conf.getConf(1).getWeightInit() == WeightInit.DISTRIBUTION);
		assertTrue(conf.getConf(0).getDist() instanceof UniformDistribution);
		assertTrue(conf.getConf(1).getDist() instanceof UniformDistribution);
		
		w0 = net.getLayer(0).getParam(DefaultParamInitializer.WEIGHT_KEY).linearView();
		for( int i=0; i<w0.length(); i++ ) assertTrue(w0.getDouble(i)>=10.0 && w0.getDouble(i)<=11.0);
		w1 = net.getLayer(1).getParam(DefaultParamInitializer.WEIGHT_KEY).linearView();
		for( int i=0; i<w1.length(); i++ ) assertTrue(w1.getDouble(i)>=20.0 && w0.getDouble(i)<=21.0);
	}

}
