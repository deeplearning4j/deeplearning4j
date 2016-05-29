package org.deeplearning4j.nn.layers.recurrent;

import static org.junit.Assert.*;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;


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

		int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
		INDArray params = Nd4j.create(1, numParams);
		GravesLSTM layer = LayerFactories.getFactory(conf.getLayer()).create(conf,null,0,params);
		
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

		int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
		INDArray params = Nd4j.create(1, numParams);
		GravesLSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf,null,0,params);
		lstm.setBackpropGradientsViewArray(Nd4j.create(1, LayerFactories.getFactory(conf.getLayer()).initializer().numParams(conf,true)));
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

	@Test
	public void testGravesLSTMForwardPassHelper() throws Exception {
		//GravesLSTM.activateHelper() has different behaviour (due to optimizations) when forBackprop==true vs false
		//But should otherwise provide identical activations
		Nd4j.getRandom().setSeed(12345);

		int nIn = 10;
		int layerSize = 15;
		int miniBatchSize = 4;
		int timeSeriesLength = 7;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
        .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
				.nIn(nIn).nOut(layerSize)
				.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
				.activation("tanh")
				.build())
		.build();

		int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
		INDArray params = Nd4j.create(1, numParams);
		GravesLSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf,null,0,params);
		INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});
		lstm.setInput(input);

		Method actHelper = GravesLSTM.class.getDeclaredMethod("activateHelper",boolean.class,INDArray.class,INDArray.class,boolean.class);
		actHelper.setAccessible(true);

		//Call activateHelper with both forBackprop == true, and forBackprop == false and compare
		Class<?> innerClass = Class.forName("org.deeplearning4j.nn.layers.recurrent.FwdPassReturn");

		Object oFalse = actHelper.invoke(lstm, false,null,null,false);	//GravesLSTM.FwdPassReturn object; want fwdPassOutput INDArray
		Object oTrue = actHelper.invoke(lstm, false,null,null,true);	//want fwdPassOutputAsArrays object

		Field fwdPassOutput = innerClass.getDeclaredField("fwdPassOutput");
		fwdPassOutput.setAccessible(true);

		Field fwdPassOutputAsArrays = innerClass.getDeclaredField("fwdPassOutputAsArrays");
		fwdPassOutputAsArrays.setAccessible(true);

		INDArray fwdPassFalse = (INDArray)fwdPassOutput.get(oFalse);
		INDArray[] fwdPassTrue = (INDArray[])fwdPassOutputAsArrays.get(oTrue);

		for( int i=0; i<timeSeriesLength; i++ ){
			INDArray sliceFalse = fwdPassFalse.tensorAlongDimension(i, 1,0);
			INDArray sliceTrue = fwdPassTrue[i];
			assertTrue(sliceFalse.equals(sliceTrue));
		}
	}

	@Test
	public void testSingleExample(){
		Nd4j.getRandom().setSeed(12345);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.updater(Updater.SGD)
				.learningRate(0.1)
				.seed(12345)
				.list()
				.layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).build())
				.layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).activation("tanh").build())
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		INDArray in1 = Nd4j.rand(new int[]{1,2,4});
		INDArray in2 = Nd4j.rand(new int[]{1, 2, 5});
		in2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)}, in1);

		assertEquals(in1, in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

		INDArray labels1 = Nd4j.rand(new int[]{1, 1, 4});
		INDArray labels2 = Nd4j.create(1, 1, 5);
		labels2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)}, labels1);
		assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

		INDArray out1 = net.output(in1);
		INDArray out2 = net.output(in2);

		System.out.println(Arrays.toString(net.output(in1).data().asFloat()));
		System.out.println(Arrays.toString(net.output(in2).data().asFloat()));

		List<INDArray> activations1 = net.feedForward(in1);
		List<INDArray> activations2 = net.feedForward(in2);

		for( int i=0; i<3; i++ ){
			System.out.println("-----\n"+i);
			System.out.println(Arrays.toString(activations1.get(i).dup().data().asDouble()));
			System.out.println(Arrays.toString(activations2.get(i).dup().data().asDouble()));

			System.out.println(activations1.get(i));
			System.out.println(activations2.get(i));
		}



		//Expect first 4 time steps to be indentical...
		for( int i=0; i<4; i++ ){
			double d1 = out1.getDouble(i);
			double d2 = out2.getDouble(i);
			assertEquals(d1,d2,0.0);
		}
	}
}
