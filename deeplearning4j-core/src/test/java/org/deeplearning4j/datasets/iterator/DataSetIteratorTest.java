package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import java.util.*;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.loader.CifarLoader;
import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DataSetIteratorTest {
	
	@Test
	public void testBatchSizeOfOne() throws Exception {
		//Test for (a) iterators returning correct number of examples, and
		//(b) Labels are a proper one-hot vector (i.e., sum is 1.0)
		
		//Iris:
		DataSetIterator iris = new IrisDataSetIterator(1, 5);
		int irisC = 0;
		while(iris.hasNext()){
			irisC++;
			DataSet ds = iris.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(irisC==5);
		
		
		//MNIST:
		DataSetIterator mnist = new MnistDataSetIterator(1, 5);
		int mnistC = 0;
		while(mnist.hasNext()){
			mnistC++;
			DataSet ds = mnist.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(mnistC==5);
		
		//LFW:
		DataSetIterator lfw = new LFWDataSetIterator(1, 5);
		int lfwC = 0;
		while(lfw.hasNext()){
			lfwC++;
			DataSet ds = lfw.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(lfwC==5);
	}

	@Test
	public void testMnist() throws Exception {
		ClassPathResource cpr = new ClassPathResource("mnist_first_200.txt");
		CSVRecordReader rr = new CSVRecordReader(0,",");
		rr.initialize(new FileSplit(cpr.getFile()));
		RecordReaderDataSetIterator dsi = new RecordReaderDataSetIterator(rr,0,10);

		MnistDataSetIterator iter = new MnistDataSetIterator(10,200,false,true,false,0);

		while(dsi.hasNext()){
			DataSet dsExp = dsi.next();
			DataSet dsAct = iter.next();

			INDArray fExp = dsExp.getFeatureMatrix();
			fExp.divi(255);
			INDArray lExp = dsExp.getLabels();

			INDArray fAct = dsAct.getFeatureMatrix();
			INDArray lAct = dsAct.getLabels();

			assertEquals(fExp,fAct);
			assertEquals(lExp,lAct);
		}
		assertFalse(iter.hasNext());
	}

	@Test
	public void testLfwIterator() throws Exception {
		int numExamples = 10;
		int row = 28;
		int col = 28;
		int channels = 1;
		LFWDataSetIterator iter = new LFWDataSetIterator(numExamples, numExamples, new int[] {row,col,channels});
		assertTrue(iter.hasNext());
		DataSet data = iter.next();
		assertEquals(numExamples, data.getLabels().size(0));
		assertEquals(row*col*channels, data.getFeatureMatrix().size(1));
	}

	@Test
	public void testLfwModel() throws Exception{
		final int numRows = 28;
		final int numColumns = 28;
		int numChannels = 3;
		int outputNum = LFWLoader.SUB_NUM_LABELS;
		int numSamples = 4;
		int batchSize = 2;
		int iterations = 1;
		int seed = 123;
		int listenerFreq = iterations;

		LFWDataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[] {numRows,numColumns,numChannels}, outputNum, true, new Random(seed));

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list()
				.layer(0, new ConvolutionLayer.Builder(10, 10)
						.nIn(numChannels)
						.nOut(6)
						.weightInit(WeightInit.XAVIER)
						.activation("relu")
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.weightInit(WeightInit.XAVIER)
						.activation("softmax")
						.build())
				.backprop(true).pretrain(false);
		new ConvolutionLayerSetup(builder,numRows,numColumns,numChannels);

		MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
		model.init();

		model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

		model.fit(lfw.next());

		DataSet dataTest = lfw.next();
		INDArray output = model.output(dataTest.getFeatureMatrix());
		Evaluation eval = new Evaluation(outputNum);
		eval.eval(dataTest.getLabels(), output);
		System.out.println(eval.stats());


	}

	@Test
	public void testCifarIterator() throws Exception {
		int numExamples = 10;
		int row = 28;
		int col = 28;
		int channels = 1;
		CifarDataSetIterator iter = new CifarDataSetIterator(numExamples, numExamples, new int[] {row,col,channels});
		assertTrue(iter.hasNext());
		DataSet data = iter.next();
		assertEquals(numExamples, data.getLabels().size(0));
		assertEquals(row*col*channels, data.getFeatureMatrix().size(1));
	}


	@Test
	public void testCifarModel() throws Exception{
		final int height = 32;
		final int width = 32;
		int channels = 3;
		int outputNum = CifarLoader.NUM_LABELS;
		int numSamples = 100;
		int batchSize = 5;
		int iterations = 1;
		int seed = 123;
		int listenerFreq = iterations;

		CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples, "TRAIN");

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list()
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						.nIn(channels)
						.nOut(6)
						.weightInit(WeightInit.XAVIER)
						.activation("relu")
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.weightInit(WeightInit.XAVIER)
						.activation("softmax")
						.build())
				.backprop(true).pretrain(false)
				.cnnInputSize(height, width, channels);

		MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
		model.init();

		model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

		model.fit(cifar);

		cifar = new CifarDataSetIterator(batchSize, numSamples, "TEST");
		Evaluation eval = new Evaluation(cifar.getLabels());
		while(cifar.hasNext()) {
			DataSet testDS = cifar.next(batchSize);
			INDArray output = model.output(testDS.getFeatureMatrix());
			eval.eval(testDS.getLabels(), output);
		}
		System.out.println(eval.stats());


	}

}
