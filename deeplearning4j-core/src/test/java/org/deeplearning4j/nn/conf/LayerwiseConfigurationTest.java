package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

/**@author Alex Black 28/07/15
 */
public class LayerwiseConfigurationTest {

	
	@Test
	public void testLayerUpdater(){
		//Without layerwise override:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.updater(Updater.NESTEROVS)
		.list(2)
		.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
		.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		assertTrue(conf.getConf(0).getUpdater() == Updater.NESTEROVS);
		assertTrue(conf.getConf(1).getUpdater() == Updater.NESTEROVS);
		
		//With:
		conf = new NeuralNetConfiguration.Builder()
			.updater(Updater.NESTEROVS)
			.list(4)
			.layer(0, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.SGD).build() )
			.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.RMSPROP).build() )
			.layer(2, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.NONE).build() )
			.layer(3, new DenseLayer.Builder().nIn(2).nOut(2).build() )
			.build();
		
		net = new MultiLayerNetwork(conf);
		net.init();
		
		assertTrue(conf.getConf(0).getUpdater()==Updater.SGD);
		assertTrue(conf.getConf(1).getUpdater()==Updater.RMSPROP);
		assertTrue(conf.getConf(2).getUpdater()==Updater.NONE);
		assertTrue(conf.getConf(3).getUpdater()==Updater.NESTEROVS);
	}

}
