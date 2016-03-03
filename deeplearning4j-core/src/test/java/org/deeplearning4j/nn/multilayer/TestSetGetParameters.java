package org.deeplearning4j.nn.multilayer;

import static org.junit.Assert.*;

import java.util.Map;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GRU;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TestSetGetParameters {
	
	@Test
	public void testSetParameters(){
		//Set up a MLN, then do set(get) on parameters. Results should be identical compared to before doing this.
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.list()
			.layer(0, new DenseLayer.Builder().nIn(9).nOut(10)
					.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).build())
			.layer(1, new RBM.Builder().nIn(10).nOut(11)
					.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).build())
			.layer(2, new AutoEncoder.Builder().corruptionLevel(0.5).nIn(11).nOut(12)
					.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).build())
			.layer(3, new OutputLayer.Builder(LossFunction.MSE).nIn(12).nOut(12)
					.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).build())
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		INDArray initParams = net.params().dup();
		Map<String,INDArray> initParams2 = net.paramTable();
		
		net.setParams(net.params());
		
		INDArray initParamsAfter = net.params();
		Map<String,INDArray> initParams2After = net.paramTable();
		
		for( String s : initParams2.keySet() ){
			assertTrue("Params differ: "+s, initParams2.get(s).equals(initParams2After.get(s)));
		}
		
		assertEquals(initParams,initParamsAfter);
		
		//Now, try the other way: get(set(random))
		INDArray randomParams = Nd4j.rand(initParams.shape());
		net.setParams(randomParams.dup());
		
		assertEquals(net.params(),randomParams);
	}
	
	@Test
	public void testSetParametersRNN(){
		//Set up a MLN, then do set(get) on parameters. Results should be identical compared to before doing this.
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(9).nOut(10).weightInit(WeightInit.DISTRIBUTION)
					.dist(new NormalDistribution(0,1)).build())
			.layer(1, new GravesLSTM.Builder().nIn(10).nOut(11).weightInit(WeightInit.DISTRIBUTION)
					.dist(new NormalDistribution(0,1)).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).weightInit(WeightInit.DISTRIBUTION)
					.dist(new NormalDistribution(0,1)).nIn(11).nOut(12).build())
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		INDArray initParams = net.params().dup();
		Map<String,INDArray> initParams2 = net.paramTable();
		
		net.setParams(net.params());
		
		INDArray initParamsAfter = net.params();
		Map<String,INDArray> initParams2After = net.paramTable();
		
		for( String s : initParams2.keySet() ){
			assertTrue("Params differ: "+s, initParams2.get(s).equals(initParams2After.get(s)));
		}
		
		assertEquals(initParams,initParamsAfter);
		
		//Now, try the other way: get(set(random))
		INDArray randomParams = Nd4j.rand(initParams.shape());
		net.setParams(randomParams.dup());
		
		assertEquals(net.params(),randomParams);
	}
}
