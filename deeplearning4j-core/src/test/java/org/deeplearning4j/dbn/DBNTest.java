package org.deeplearning4j.dbn;
import static org.junit.Assert.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.activation.HardTanh;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.deeplearning4j.nn.activation.Tanh;
import org.deeplearning4j.transformation.MatrixTransform;
import org.deeplearning4j.transformation.MultiplyScalar;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class DBNTest {

	private static Logger log = LoggerFactory.getLogger(DBNTest.class);


	@Test
	public void testDBN() {

		int n = 400;
		DataSet d = MatrixUtil.xorData(n);

		DoubleMatrix x = d.getFirst();
		DoubleMatrix y = d.getSecond();


		double preTrainLr = 0.1;
		int preTrainEpochs = 10000;
		int k = 1;
		int[] hiddenLayerSizes = new int[] {2,2,2};
		double fineTuneLr = 0.1;
		int fineTuneEpochs = 10000;

		CDBN dbn = new CDBN.Builder().useAdaGrad(true)
				.hiddenLayerSizes(hiddenLayerSizes)
				.numberOfInputs(d.numInputs())
				.useRegularization(false).withActivation(new HardTanh())
				.numberOfOutPuts(d.numOutcomes()).build();

		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);







		DoubleMatrix predict = dbn.predict(x);
		//log.info(predict.toString());

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		log.info(eval.stats());





	}

	@Test
	public void testCDBN() {
		DoubleMatrix x = new DoubleMatrix( new double[][] 
				{{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}});



		DoubleMatrix  y = new DoubleMatrix(new double[][]
				{{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1},
				{0, 1}});

		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.01;
		int preTrainEpochs = 10000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {3};
		double fineTuneLr = 0.01;
		int fineTuneEpochs = 1000;

		CDBN dbn = new CDBN.Builder().useAdaGrad(true).withActivation(new Tanh())
				.numberOfInputs(nIns).numberOfOutPuts(nOuts)
				.hiddenLayerSizes(hiddenLayerSizes).useRegularization(false)
				.withRng(rng)
				.build();
		
		
		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);


		DoubleMatrix testX = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.},
				{0.5, 0.5, 0.5, 0.5, 0.5, 0.}});

		DoubleMatrix predict =  dbn.predict(x);
		Evaluation eval = new Evaluation();
		eval.eval(y,predict);
		log.info(eval.stats());

	}


	@Test
	public void testIris() {
		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.01;
		int preTrainEpochs = 10000;
		int k = 1;
		int nIns = 4,nOuts = 3;
		int[] hiddenLayerSizes = new int[] {4,3,2};
		double fineTuneLr = 0.01;
		int fineTuneEpochs = 10000;

		CDBN dbn = new CDBN.Builder().useAdaGrad(true)
				.numberOfInputs(nIns).numberOfOutPuts(nOuts).withActivation(new Tanh())
				.hiddenLayerSizes(hiddenLayerSizes).useRegularization(true)
				.withRng(rng)
				.build();



		DataSetIterator iter = new IrisDataSetIterator(150, 150);

		DataSet next = iter.next(150);
		next.shuffle();

		List<DataSet> finetuneBatches = next.dataSetBatches(10);



		DataSetIterator sampling = new SamplingDataSetIterator(next, 150, 3000);

		List<DataSet> miniBatches = new ArrayList<DataSet>();

		while(sampling.hasNext()) {
			next = sampling.next();
			miniBatches.add(next.copy());
		}

		log.info("Training on " + miniBatches.size() + " minibatches");


		for(int i = 0; i < miniBatches.size(); i++) {
			DataSet curr = miniBatches.get(i);
			dbn.pretrain(curr.getFirst(),k, preTrainLr, preTrainEpochs);

		}


		Evaluation eval = new Evaluation();


		for(int i = 0; i < miniBatches.size(); i++) {
			DataSet curr = miniBatches.get(i);
			dbn.setInput(curr.getFirst());
			dbn.finetune(curr.getSecond(),fineTuneLr, fineTuneEpochs);

		}






		for(int i = 0; i < miniBatches.size(); i++) {
			DataSet test = miniBatches.get(i);
			DoubleMatrix predicted = dbn.predict(test.getFirst());
			DoubleMatrix real = test.getSecond();


			eval.eval(real, predicted);

		}

		log.info("Evaled " + eval.stats());


	}


	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher(true);
		fetcher.fetch(100);
		DataSet d = fetcher.next();
		d.filterAndStrip(new int[]{0,1});
		log.info("Training on " + d.numExamples());
		StopWatch watch = new StopWatch();
		
		log.info("Data set " + d);
		
		DBN dbn = new DBN.Builder()
		.withActivation(Activations.tanh())
		.hiddenLayerSizes(new int[]{500,250,100})
		.withMomentum(0)
		.numberOfInputs(784).useAdaGrad(true)
		.numberOfOutPuts(2)
		.useRegularization(false)
		.build();
		
		watch.start();
		
		dbn.pretrain(d.getFirst(), 1, 0.1, 300);
		dbn.finetune(d.getSecond(), 0.1, 100);
		
		watch.stop();
		
		log.info("Took " + watch.getTime());
		Evaluation eval = new Evaluation();
		DoubleMatrix predict = dbn.predict(d.getFirst());
		eval.eval(d.getSecond(), predict);
		log.info(eval.stats());
	}



}
