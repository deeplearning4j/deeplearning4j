package org.deeplearning4j.optimize.solver;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TestOptimizers {
	
	@Test
	public void testOptimizersBasicMLPBackprop(){
		//Basic tests of the 'does it throw an exception' variety.
		
		DataSetIterator iter = new IrisDataSetIterator(5,50);
		
		for( OptimizationAlgorithm oa : OptimizationAlgorithm.values() ){
			MultiLayerNetwork network = new MultiLayerNetwork(getMLPConfigIris(oa));
			network.init();
			
			iter.reset();
			network.fit(iter);
		}
	}
	
	private static MultiLayerConfiguration getMLPConfigIris( OptimizationAlgorithm oa ){
		MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
		.nIn(4).nOut(3)
		.weightInit(WeightInit.DISTRIBUTION)
		.dist(new NormalDistribution(0, 0.1))

		.activationFunction("sigmoid")
		.lossFunction(LossFunction.MCXENT)
		.optimizationAlgo(oa)
		.iterations(1)
		.batchSize(5)
		.constrainGradientToUnitNorm(false)
		.corruptionLevel(0.0)
		.layer(new RBM())
		.learningRate(0.1).useAdaGrad(false)
		.regularization(true)
		.l2(0.01)
		.applySparsity(false).sparsity(0.0)
		.seed(12345L)
		.list(4).hiddenLayerSizes(8,10,5)
		.backward(true).pretrain(false)
		.useDropConnect(false)

		.override(3, new ConfOverride() {
			@Override
			public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
				builder.activationFunction("softmax");
				builder.layer(new OutputLayer());
				builder.weightInit(WeightInit.DISTRIBUTION);
				builder.dist(new NormalDistribution(0, 0.1));
			}
		}).build();

		return c;
	}
}
