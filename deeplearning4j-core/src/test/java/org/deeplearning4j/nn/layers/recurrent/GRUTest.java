package org.deeplearning4j.nn.layers.recurrent;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;


public class GRUTest {
	
	@Test
    @Ignore
	public void testGRUForwardBasic(){
		//Very basic test of forward pass for GRU layer.
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.
		
		int nIn = 13;
		int nHiddenUnits = 17;
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.layer(new org.deeplearning4j.nn.conf.layers.GRU.Builder()
						.nIn(nIn)
						.nOut(nHiddenUnits)
						.activation("tanh")
						.build())
				.build();

		int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
		INDArray params = Nd4j.create(1, numParams);
		GRU layer = LayerFactories.getFactory(conf.getLayer()).create(conf,null,0,params);
		
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
    @Ignore
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
                .layer(new org.deeplearning4j.nn.conf.layers.GRU.Builder()
						.nIn(nIn)
						.nOut(gruNHiddenUnits)
						.weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(0, 1))
						.activation("tanh")
						.build())
				.build();

		int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
		INDArray params = Nd4j.create(1, numParams);
		GRU gru = LayerFactories.getFactory(conf.getLayer()).create(conf,null,0,params);
		//Set input, do a forward pass:
		gru.activate(inputData);
		assertNotNull(gru.input());

		INDArray epsilon = Nd4j.ones(miniBatchSize, gruNHiddenUnits, timeSeriesLength);

		Pair<Gradient,INDArray> out = gru.backpropGradient(epsilon);
		Gradient outGradient = out.getFirst();
		INDArray nextEpsilon = out.getSecond();

		INDArray biasGradient = outGradient.getGradientFor(GRUParamInitializer.BIAS_KEY);
		INDArray inWeightGradient = outGradient.getGradientFor(GRUParamInitializer.INPUT_WEIGHT_KEY);
		INDArray recurrentWeightGradient = outGradient.getGradientFor(GRUParamInitializer.RECURRENT_WEIGHT_KEY);
		assertNotNull(biasGradient);
		assertNotNull(inWeightGradient);
		assertNotNull(recurrentWeightGradient);

		assertArrayEquals(biasGradient.shape(),new int[]{1,3*gruNHiddenUnits});
		assertArrayEquals(inWeightGradient.shape(),new int[]{nIn,3*gruNHiddenUnits});
		assertArrayEquals(recurrentWeightGradient.shape(),new int[]{gruNHiddenUnits,3*gruNHiddenUnits});

		assertNotNull(nextEpsilon);
		assertArrayEquals(nextEpsilon.shape(),new int[]{miniBatchSize,nIn,timeSeriesLength});
	}
	
	
	@Test
    @Ignore
	public void testForwardPassSanityCheck(){
		//Set up a basic GRU+OutputLayer network and do a sanity check on forward pass (i.e., check not NaN or Inf.)
		Random r = new Random(12345L);
		int timeSeriesLength = 20;
    	int nIn = 5;
    	int nOut = 4;
    	int gruNUnits = 7;
    	int miniBatchSize = 11;
		
		INDArray inputData = Nd4j.ones(miniBatchSize,nIn,timeSeriesLength);
		for( int i=0; i<miniBatchSize; i++ ){
			for( int j=0; j<nIn; j++ ){
				for( int k=0; k<timeSeriesLength; k++ ){
					inputData.putScalar(new int[]{i,j,k}, r.nextDouble()-0.5);
				}
			}
		}
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.regularization(false)
			.list()
			.layer(0, new org.deeplearning4j.nn.conf.layers.GRU.Builder().activation("tanh")
				.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,0.1))
            	.nIn(nIn).nOut(gruNUnits).build())
        	.layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")
        			.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,0.1))
        			.nIn(gruNUnits).nOut(nOut).build())
        	.build();
		
		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
		mln.init();
		
		List<INDArray> activations = mln.feedForward(inputData);
		
		INDArray gruActiv = activations.get(1);
		INDArray outActiv = activations.get(2);
		assertArrayEquals(gruActiv.shape(),new int[]{miniBatchSize,gruNUnits,timeSeriesLength});
		assertArrayEquals(outActiv.shape(),new int[]{miniBatchSize*timeSeriesLength,nOut});
		
		
		for( int i=0; i<gruActiv.length(); i++ ){
			double d = gruActiv.getDouble(i);
			assertTrue(!Double.isNaN(d) && !Double.isInfinite(d));
			assertTrue(d >= -1.0 && d <=1.0);	//Tanh
		}
		
		for( int i=0; i<outActiv.length(); i++ ){
			double d = outActiv.getDouble(i);
			assertTrue(!Double.isNaN(d) && !Double.isInfinite(d));
			assertTrue(d >= 0.0 && d <=1.0);	//Softmax
		}
	}
}
