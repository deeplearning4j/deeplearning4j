package org.deeplearning4j.nn.layers.recurrent;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class GravesLSTMTest {
	
	@Test
	public void testLSTMGravesForwardBasic(){
		//Very basic test of forward prop. of LSTM layer with a time series.
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.
		
		int nIn = 13;
		int nHiddenUnits = 17;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
						.nIn(nIn)
						.nOut(nHiddenUnits)
						.activation("tanh")
						.build())
				.build();
	
		GravesLSTM layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
		
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
	public void testLSTMGravesBackwardBasic(){
		//Very basic test of backprop for mini-batch + time series
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape. 
		
		testGravesBackwardBasicHelper(13,3,17,10,7);
		testGravesBackwardBasicHelper(13,3,17,1,7);		//Edge case: miniBatchSize = 1
		testGravesBackwardBasicHelper(13,3,17,10,1);	//Edge case: timeSeriesLength = 1
		testGravesBackwardBasicHelper(13,3,17,1,1);		//Edge case: both miniBatchSize = 1 and timeSeriesLength = 1
	}
	
	private static void testGravesBackwardBasicHelper(int nIn, int nOut, int lstmNHiddenUnits, int miniBatchSize, int timeSeriesLength ){
		
		INDArray inputData = Nd4j.ones(miniBatchSize,nIn,timeSeriesLength);
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
						.nIn(nIn)
						.nOut(lstmNHiddenUnits)
						.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
						.activation("tanh")
						.build())
				.build();
		
		GravesLSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf);
		//Set input, do a forward pass:
		lstm.activate(inputData);
		assertNotNull(lstm.input());

		INDArray epsilon = Nd4j.ones(miniBatchSize, lstmNHiddenUnits, timeSeriesLength);

		Pair<Gradient,INDArray> out = lstm.backpropGradient(epsilon);
		Gradient outGradient = out.getFirst();
		INDArray nextEpsilon = out.getSecond();

		INDArray biasGradient = outGradient.getGradientFor(GravesLSTMParamInitializer.BIAS_KEY);
		INDArray inWeightGradient = outGradient.getGradientFor(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
		INDArray recurrentWeightGradient = outGradient.getGradientFor(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
		assertNotNull(biasGradient);
		assertNotNull(inWeightGradient);
		assertNotNull(recurrentWeightGradient);

		assertArrayEquals(biasGradient.shape(),new int[]{1,4*lstmNHiddenUnits});
		assertArrayEquals(inWeightGradient.shape(),new int[]{nIn,4*lstmNHiddenUnits});
		assertArrayEquals(recurrentWeightGradient.shape(),new int[]{lstmNHiddenUnits,4*lstmNHiddenUnits+3});

		assertNotNull(nextEpsilon);
		assertArrayEquals(nextEpsilon.shape(),new int[]{miniBatchSize,nIn,timeSeriesLength});
		
		//Check update:
		for( String s : outGradient.gradientForVariable().keySet() ){
			lstm.update(outGradient.getGradientFor(s), s);
		}
	}
}
