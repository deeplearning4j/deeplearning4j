package org.deeplearning4j.nn.conf.preprocessor.output;

import static org.junit.Assert.*;

import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TestRnnOutputProcessor {
	
	@Test
	public void testRnnOutputProcessor(){
		Nd4j.getRandom().setSeed(12345);
		
		int timeSeriesLength = 10;
		int nIn = 9;
		int lstmLayerSize = 12;
		int nOut = 7;
		int miniBatchSize = 13;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.list(2)
			.layer(0,new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize).build())
			.layer(1,new OutputLayer.Builder(LossFunction.MCXENT).nIn(lstmLayerSize).nOut(nOut).build())
			.inputPreProcessor(1, new RnnToFeedForwardPreProcessor())
			.outputProcessor(new RnnOutputProcessor())
			.pretrain(false).backprop(true)
			.build();
		
		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
		mln.init();
		
		assertNotNull(mln.getLayerWiseConfigurations().getOutputProcessor());
		assertTrue(mln.getLayerWiseConfigurations().getOutputProcessor() instanceof RnnOutputProcessor);
		
		INDArray input3d = Nd4j.rand(new int[]{miniBatchSize,nIn,timeSeriesLength});
		mln.setInput(input3d);
		
		INDArray labels3d = Nd4j.rand(new int[]{miniBatchSize,nOut,timeSeriesLength});
		
		RnnOutputProcessor proc = new RnnOutputProcessor();
		INDArray labels2d = proc.processLabels(labels3d,mln);
		assertArrayEquals(labels2d.shape(),new int[]{miniBatchSize*timeSeriesLength,nOut});
		INDArray labels3dv2 = proc.processOutput(labels2d,mln);
		assertTrue(labels3d.equals(labels3dv2));
		
		List<INDArray> activations = mln.feedForward(input3d);
		INDArray out = activations.get(2);
		assertArrayEquals(out.shape(),new int[]{miniBatchSize,nOut,timeSeriesLength});
		
		mln.fit(input3d, labels3d);	//Ok as long as it doesn't throw an exception (due to shapes etc)
	}
}
