package org.deeplearning4j.dbn;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class DBNTest {

	private static Logger log = LoggerFactory.getLogger(DBNTest.class);


	@Test
	public void testDBN() {

		int n = 10;
		DataSet d = MatrixUtil.xorData(n);
		DoubleMatrix x = d.getFirst();
		DoubleMatrix y = d.getSecond();





		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {2,2,2};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 1000;

		DBN dbn = new DBN.Builder()
    .hiddenLayerSizes(hiddenLayerSizes)
		.numberOfInputs(nIns).renderWeights(0)
		.useRegularization(false).withMomentum(0).withDist(new NormalDistribution(0,0.001))
		.numberOfOutPuts(nOuts).withRng(rng).build();

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
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {5,5};
		double fineTuneLr = 0.01;
		int fineTuneEpochs = 200;

		CDBN dbn = new CDBN.Builder()
		.numberOfInputs(nIns).numberOfOutPuts(nOuts)
		.hiddenLayerSizes(hiddenLayerSizes).useRegularization(false)
		.withRng(rng).withL2(0.1).renderWeights(1000)
		.build();
		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);


		DoubleMatrix testX = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.},
				{0.5, 0.5, 0.5, 0.5, 0.5, 0.}});

		log.info(dbn.predict(testX).toString());

	}



}
