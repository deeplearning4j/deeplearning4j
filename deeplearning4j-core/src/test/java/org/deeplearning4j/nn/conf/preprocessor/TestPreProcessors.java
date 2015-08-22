package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestPreProcessors {
	
	@Test
	public void testRnnToFeedForwardPreProcessor(){
		int miniBatchSize = 5;
		int hiddenLayerSize = 7;
		int timeSeriesLength = 9;
		
		RnnToFeedForwardPreProcessor proc = new RnnToFeedForwardPreProcessor();
		NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
		.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(hiddenLayerSize).build()).build();
		
		DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
		layer.setInputMiniBatchSize(miniBatchSize);
		
		INDArray activations3d = Nd4j.rand(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});
		for( int i=0; i<miniBatchSize; i++ ){
			for( int j=0; j<hiddenLayerSize; j++ ){
				for( int k=0; k<timeSeriesLength; k++ ){
					double value = 100*i + 10*j + k;	//value abc -> example=a, neuronNumber=b, time=c
					activations3d.putScalar(new int[]{i,j,k},value);
				}
			}
		}

		INDArray activations2d = proc.preProcess(activations3d,layer);
		assertArrayEquals(activations2d.shape(),new int[]{miniBatchSize*timeSeriesLength,hiddenLayerSize});

		//Expect each row in activations2d to have order:
		//(example=0,t=0), (example=0,t=1), (example=0,t=2), ..., (example=1,t=0), (example=1,t=1), ...
		int nRows = activations2d.rows();
		for( int i=0; i<nRows; i++ ){
			INDArray row = activations2d.getRow(i);
			assertArrayEquals(row.shape(),new int[]{1,hiddenLayerSize});
			int origExampleNum = i / timeSeriesLength;
			int time = i % timeSeriesLength;
			INDArray expectedRow = activations3d.tensorAlongDimension(time,1,0).getRow(origExampleNum);
			assertTrue(row.equals(expectedRow));
		}

		//Given that epsilons and activations have same shape, we can do this (even though it's not the intended use)
		//Basically backprop should be exact opposite of preProcess
		INDArray out = proc.backprop(activations2d,layer);
		
		assertTrue(out.equals(activations3d));
	}
	
	@Test
	public void testFeedForwardToRnnPreProcessor(){
		Nd4j.getRandom().setSeed(12345L);
		int miniBatchSize = 5;
		int hiddenLayerSize = 7;
		int timeSeriesLength = 9;

		FeedForwardToRnnPreProcessor proc = new FeedForwardToRnnPreProcessor();
		
		NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
		.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(hiddenLayerSize).build()).build();
		
		DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
		layer.setInputMiniBatchSize(miniBatchSize);

		INDArray activations2d = Nd4j.rand(miniBatchSize*timeSeriesLength,hiddenLayerSize);

		INDArray activations3d = proc.preProcess(activations2d,layer);
		assertArrayEquals(activations3d.shape(),new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});

		int nRows2D = miniBatchSize*timeSeriesLength;
		for( int i=0; i<nRows2D; i++ ){
			int time = i % timeSeriesLength;
			int example = i / timeSeriesLength;
			
			INDArray row2d = activations2d.getRow(i);
			INDArray row3d = activations3d.tensorAlongDimension(time,1,0).getRow(example);
			
			assertTrue(row2d.equals(row3d));
		}

		//Again epsilons and activations have same shape, we can do this (even though it's not the intended use)
		INDArray epsilon2d = proc.backprop(activations3d,layer);

		assertTrue(epsilon2d.equals(activations2d));
	}
}
