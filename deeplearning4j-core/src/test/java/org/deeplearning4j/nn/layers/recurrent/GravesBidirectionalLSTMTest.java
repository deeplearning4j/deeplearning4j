package org.deeplearning4j.nn.layers.recurrent;

import com.google.common.collect.Lists;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;


public class GravesBidirectionalLSTMTest {
	
	@Test
	public void testBidirectionalLSTMGravesForwardBasic(){
		//Very basic test of forward prop. of LSTM layer with a time series.
		//Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.
		
		int nIn = 13;
		int nHiddenUnits = 17;
		
		final NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
						.nIn(nIn)
						.nOut(nHiddenUnits)
						.activation("tanh")
						.build())
				.build();

		final GravesBidirectionalLSTM layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
		
		//Data: has shape [miniBatchSize,nIn,timeSeriesLength];
		//Output/activations has shape [miniBatchsize,nHiddenUnits,timeSeriesLength];
		
		final INDArray dataSingleExampleTimeLength1 = Nd4j.ones(1,nIn,1);
		final INDArray activations1 = layer.activate(dataSingleExampleTimeLength1);
		assertArrayEquals(activations1.shape(),new int[]{1,nHiddenUnits,1});
		
		final INDArray dataMultiExampleLength1 = Nd4j.ones(10,nIn,1);
		final INDArray activations2 = layer.activate(dataMultiExampleLength1);
		assertArrayEquals(activations2.shape(),new int[]{10,nHiddenUnits,1});
		
		final INDArray dataSingleExampleLength12 = Nd4j.ones(1,nIn,12);
		final INDArray activations3 = layer.activate(dataSingleExampleLength12);
		assertArrayEquals(activations3.shape(),new int[]{1,nHiddenUnits,12});
		
		final INDArray dataMultiExampleLength15 = Nd4j.ones(10,nIn,15);
		final INDArray activations4 = layer.activate(dataMultiExampleLength15);
		assertArrayEquals(activations4.shape(),new int[]{10,nHiddenUnits,15});
	}
	
	@Test
	public void testBidirectionalLSTMGravesBackwardBasic(){
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
                .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
						.nIn(nIn)
						.nOut(lstmNHiddenUnits)
						.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
						.activation("tanh")
						.build())
				.build();
		
		GravesBidirectionalLSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf);
		//Set input, do a forward pass:
		lstm.activate(inputData);
		assertNotNull(lstm.input());

		INDArray epsilon = Nd4j.ones(miniBatchSize, lstmNHiddenUnits, timeSeriesLength);

		Pair<Gradient,INDArray> out = lstm.backpropGradient(epsilon);
		Gradient outGradient = out.getFirst();
		INDArray nextEpsilon = out.getSecond();

		INDArray biasGradientF = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS);
		INDArray inWeightGradientF = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS);
		INDArray recurrentWeightGradientF = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS);
		assertNotNull(biasGradientF);
		assertNotNull(inWeightGradientF);
		assertNotNull(recurrentWeightGradientF);

		INDArray biasGradientB = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS);
		INDArray inWeightGradientB = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS);
		INDArray recurrentWeightGradientB = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
		assertNotNull(biasGradientB);
		assertNotNull(inWeightGradientB);
		assertNotNull(recurrentWeightGradientB);

		assertArrayEquals(biasGradientF.shape(),new int[]{1,4*lstmNHiddenUnits});
		assertArrayEquals(inWeightGradientF.shape(),new int[]{nIn,4*lstmNHiddenUnits});
		assertArrayEquals(recurrentWeightGradientF.shape(),new int[]{lstmNHiddenUnits,4*lstmNHiddenUnits+3});

		assertArrayEquals(biasGradientB.shape(),new int[]{1,4*lstmNHiddenUnits});
		assertArrayEquals(inWeightGradientB.shape(),new int[]{nIn,4*lstmNHiddenUnits});
		assertArrayEquals(recurrentWeightGradientB.shape(),new int[]{lstmNHiddenUnits,4*lstmNHiddenUnits+3});

		assertNotNull(nextEpsilon);
		assertArrayEquals(nextEpsilon.shape(),new int[]{miniBatchSize,nIn,timeSeriesLength});
		
		//Check update:
		for( String s : outGradient.gradientForVariable().keySet() ){
			lstm.update(outGradient.getGradientFor(s), s);
		}
	}

	@Test
	public void testGravesBidirectionalLSTMForwardPassHelper() throws Exception {
		//GravesBidirectionalLSTM.activateHelper() has different behaviour (due to optimizations) when forBackprop==true vs false
		//But should otherwise provide identical activations
		Nd4j.getRandom().setSeed(12345);

		int nIn = 10;
		int layerSize = 15;
		int miniBatchSize = 4;
		int timeSeriesLength = 7;

		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
				.nIn(nIn).nOut(layerSize)
				.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
				.activation("tanh")
				.build())
		.build();

		GravesBidirectionalLSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf);
		INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});
		lstm.setInput(input);


		final INDArray fwdPassFalse = LSTMHelpers.activateHelper(
				lstm,
				lstm.conf(),
				lstm.input(),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS),
				false,null,null,false,true,GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS).fwdPassOutput;

		final INDArray[] fwdPassTrue = LSTMHelpers.activateHelper(
				lstm,
				lstm.conf(),
				lstm.input(),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
				lstm.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS),
				false,null,null,true,true,GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS).fwdPassOutputAsArrays;


		for( int i=0; i<timeSeriesLength; i++ ){
			INDArray sliceFalse = fwdPassFalse.tensorAlongDimension(i, 1,0);
			INDArray sliceTrue = fwdPassTrue[i];
			assertTrue(sliceFalse.equals(sliceTrue));
		}
	}

	@Test
	public void testConvergence(){
		Nd4j.getRandom().setSeed(12345);
        final int state1Len = 100;
        final int state2Len = 30;

        //segment by signal mean
        //Data: has shape [miniBatchSize,nIn,timeSeriesLength];

        final INDArray sig1 = Nd4j.randn(new int[]{1,2,state1Len}).mul(0.001);
        final INDArray sig2 = Nd4j.randn(new int[]{1,2,state2Len}).mul(0.001).add(Nd4j.ones(new int[]{1,2,state2Len}).mul(10.0));

        INDArray sig = Nd4j.concat(2,sig1,sig2);
        INDArray labels = Nd4j.zeros(new int[]{1,2,state1Len + state2Len});

        for (int t = 0; t < state1Len; t++) {
            labels.putScalar(new int[]{0,0,t},1.0);
        }

        for (int t = state1Len; t < state1Len + state2Len; t++) {
            labels.putScalar(new int[]{0,1,t},1.0);
        }

        for (int i = 0; i < 3; i++) {
            sig = Nd4j.concat(2,sig,sig);
            labels = Nd4j.concat(2,labels,labels);
        }

        final DataSet ds = new DataSet(sig,labels);

        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(5)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .regularization(true)
                .l2(0.001)
				.updater(Updater.ADAGRAD)
				.seed(12345)
				.list(3)
                .pretrain(false)
				.layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().activation("tanh").nIn(2).nOut(2).weightInit(WeightInit.UNIFORM).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().activation("tanh").nIn(2).nOut(2).weightInit(WeightInit.UNIFORM).build())
            //    .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).weightInit(WeightInit.UNIFORM).build())
            //    .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).weightInit(WeightInit.UNIFORM).build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(2).activation("tanh").build())
				.backprop(true)
                .build();

		final MultiLayerNetwork net = new MultiLayerNetwork(conf);

        net.setListeners(new ScoreIterationListener(1));

        net.init();
        for (int iEpoch = 0; iEpoch < 3; iEpoch++) {
            net.fit(ds);
            final INDArray output = net.output(ds.getFeatureMatrix());
            Evaluation evaluation = new Evaluation();
            evaluation.evalTimeSeries(ds.getLabels(),output);
            System.out.print(evaluation.stats() + "\n");
        }


        int foo = 3;
        foo++;

//		INDArray labels1 = Nd4j.rand(new int[]{1, 1, 4});
//		INDArray labels2 = Nd4j.create(n);
//		labels2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)}, labels1);
//		assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));
//
//		INDArray out1 = net.output(in1);
//		INDArray out2 = net.output(in2);
//
//		System.out.println(Arrays.toString(net.output(in1).data().asFloat()));
//		System.out.println(Arrays.toString(net.output(in2).data().asFloat()));
//
//		List<INDArray> activations1 = net.feedForward(in1);
//		List<INDArray> activations2 = net.feedForward(in2);
//
//		for( int i=0; i<3; i++ ){
//			System.out.println("-----\n"+i);
//			System.out.println(Arrays.toString(activations1.get(i).dup().data().asDouble()));
//			System.out.println(Arrays.toString(activations2.get(i).dup().data().asDouble()));
//
//			System.out.println(activations1.get(i));
//			System.out.println(activations2.get(i));
//		}
//
//
//
//		//Expect first 4 time steps to be indentical...
//		for( int i=0; i<4; i++ ){
//			double d1 = out1.getDouble(i);
//			double d2 = out2.getDouble(i);
//			assertEquals(d1,d2,0.0);
//		}
	}
}
