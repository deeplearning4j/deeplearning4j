package org.deeplearning4j.dbn;
import static org.junit.Assert.*;
import java.io.IOException;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.HardTanh;
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
	@Ignore
	public void testDBN() {

		int n = 50;
		DataSet d = MatrixUtil.xorData(n,500);
		DoubleMatrix x = d.getFirst();
		DoubleMatrix y = d.getSecond();


		double preTrainLr = 0.0001;
		int preTrainEpochs = 1000;
		int k = 1;
		int[] hiddenLayerSizes = new int[] {400,250,100};
		double fineTuneLr = 0.0001;
		int fineTuneEpochs = 1000;

		DBN dbn = new DBN.Builder().forceEpochs()
				.hiddenLayerSizes(hiddenLayerSizes)
				.numberOfInputs(d.numInputs())
				.useRegularization(false)
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
	@Ignore
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

		double preTrainLr = 0.001;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {20,10,5};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 200;

		CDBN dbn = new CDBN.Builder()
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


		Evaluation eval = new Evaluation();
		eval.eval(y, dbn.predict(x));
		log.info(eval.stats());

	}



	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(1000);
		DataSet d = fetcher.next();
		assertEquals(1000,d.numExamples());

		DBN dbn = new DBN.Builder().hiddenLayerSizes(new int[]{500,250,100}).withActivation(new HardTanh())
				.numberOfInputs(784).numberOfOutPuts(10).useRegularization(false).build();
		dbn.pretrain(d.getFirst(), 1, 0.0001, 30000);
		dbn.finetune(d.getSecond(), 0.0001, 10000);

		Evaluation eval = new Evaluation();
		DoubleMatrix predict = dbn.predict(d.getFirst());
		eval.eval(d.getSecond(), predict);
		log.info(eval.stats());
	}



}
