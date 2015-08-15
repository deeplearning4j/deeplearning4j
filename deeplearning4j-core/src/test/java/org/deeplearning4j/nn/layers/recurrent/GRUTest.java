package org.deeplearning4j.nn.layers.recurrent;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class GRUTest {
	
	@Test
	public void testGRUForwardBasic(){
		//Very basic test of forward pass for GRU layer.
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.
		
		int nIn = 13;
		int nHiddenUnits = 17;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.layer(new org.deeplearning4j.nn.conf.layers.GRU.Builder()
						.nIn(nIn)
						.nOut(nHiddenUnits)
						.build())
				.activationFunction("tanh")
				.build();
	
		GRU layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
		
		//Data: has shape [miniBatchSize,nIn,timeSeriesLength];
		//Output/activations has shape [miniBatchsize,nHiddenUnits,timeSeriesLength];
		
		INDArray dataSingleExampleTimeLength1 = Nd4j.ones(1,nIn,1);
		INDArray activations1 = layer.activate(dataSingleExampleTimeLength1);
		assertArrayEquals(activations1.shape(),new int[]{1,nHiddenUnits,1});
		
		INDArray dataMultiExampleLength1 = Nd4j.ones(10,nIn,1);
		INDArray activations2 = layer.activate(dataMultiExampleLength1);
		assertArrayEquals(activations2.shape(),new int[]{10,nHiddenUnits,1});
		
		INDArray dataSingleExampleLength12 = Nd4j.ones(1,nIn,12);
		INDArray activations3 = layer.activate(dataSingleExampleLength12);
		assertArrayEquals(activations3.shape(),new int[]{1,nHiddenUnits,12});
		
		INDArray dataMultiExampleLength15 = Nd4j.ones(10,nIn,15);
		INDArray activations4 = layer.activate(dataMultiExampleLength15);
		assertArrayEquals(activations4.shape(),new int[]{10,nHiddenUnits,15});
	}
	
	@Test
	public void testGRUBackwardBasic(){
		//Very basic test of backprop for mini-batch + time series
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape. 
		
		testGRUBackwardBasicHelper(13,3,17,10,7);
		testGRUBackwardBasicHelper(13,3,17,1,7);		//Edge case: miniBatchSize = 1
		testGRUBackwardBasicHelper(13,3,17,10,1);	//Edge case: timeSeriesLength = 1
		testGRUBackwardBasicHelper(13,3,17,1,1);		//Edge case: both miniBatchSize = 1 and timeSeriesLength = 1
	}
	
	private static void testGRUBackwardBasicHelper(int nIn, int nOut, int gruNHiddenUnits, int miniBatchSize, int timeSeriesLength ){
		
		INDArray inputData = Nd4j.ones(miniBatchSize,nIn,timeSeriesLength);
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.activationFunction("tanh")
				.dist(new UniformDistribution(0, 1))
                .layer(new org.deeplearning4j.nn.conf.layers.GRU.Builder()
						.nIn(nIn)
						.nOut(gruNHiddenUnits)
						.weightInit(WeightInit.DISTRIBUTION)
						.build())
				.build();
		
		GRU gru = LayerFactories.getFactory(conf.getLayer()).create(conf);
		//Set input, do a forward pass:
		gru.activate(inputData);
		assertNotNull(gru.input());

		INDArray epsilon = Nd4j.ones(miniBatchSize, gruNHiddenUnits, timeSeriesLength);

		Pair<Gradient,INDArray> out = gru.backpropGradient(epsilon);
		Gradient outGradient = out.getFirst();
		INDArray nextEpsilon = out.getSecond();

		INDArray biasGradient = outGradient.getGradientFor(GRUParamInitializer.BIAS);
		INDArray inWeightGradient = outGradient.getGradientFor(GRUParamInitializer.INPUT_WEIGHTS);
		INDArray recurrentWeightGradient = outGradient.getGradientFor(GRUParamInitializer.RECURRENT_WEIGHTS);
		assertNotNull(biasGradient);
		assertNotNull(inWeightGradient);
		assertNotNull(recurrentWeightGradient);

		assertArrayEquals(biasGradient.shape(),new int[]{1,3*gruNHiddenUnits});
		assertArrayEquals(inWeightGradient.shape(),new int[]{nIn,3*gruNHiddenUnits});
		assertArrayEquals(recurrentWeightGradient.shape(),new int[]{gruNHiddenUnits,3*gruNHiddenUnits});

		assertNotNull(nextEpsilon);
		assertArrayEquals(nextEpsilon.shape(),new int[]{miniBatchSize,nIn,timeSeriesLength});
	}
}
