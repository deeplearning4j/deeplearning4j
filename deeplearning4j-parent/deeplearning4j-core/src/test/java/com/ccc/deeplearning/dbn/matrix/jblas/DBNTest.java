package com.ccc.deeplearning.dbn.matrix.jblas;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.dbn.matrix.jblas.CDBN;
import com.ccc.deeplearning.dbn.matrix.jblas.DBN;
import com.ccc.deeplearning.rbm.matrix.jblas.CRBM;


public class DBNTest  extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(DBNTest.class);


	@Test
	public void testDBN() {
		DoubleMatrix x = new DoubleMatrix(new double[][] 
				{{1,1,1,0,0,0},
				{1,0,1,0,0,0},
				{1,1,1,0,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,0,0},
				{0,0,1,1,1,0}});
		DoubleMatrix   y = new DoubleMatrix(new double[][]  
				{{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1},
				{0, 1}});


		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.1;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {3,3};
		double fineTuneLr = 0.1;
		int fineTuneEpochs = 200;

		DBN dbn = new DBN.Builder()
		.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns)
		.numberOfOutPuts(nOuts).withRng(rng).build();
		
		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);

		DoubleMatrix testX = new DoubleMatrix(new double[][]
				{{1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0},
				{1, 1, 1, 1, 1, 0}});


		DoubleMatrix predict = dbn.predict(testX);
		log.info(predict.toString());
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

		double preTrainLr = 0.1;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {5,5};
		double fineTuneLr = 0.1;
		int fineTuneEpochs = 200;

		CDBN dbn = new CDBN(nIns, hiddenLayerSizes, nOuts, 2, rng, x, y);
		dbn.pretrain(k, preTrainLr, preTrainEpochs);
		dbn.finetune(fineTuneLr, fineTuneEpochs);

		
		DoubleMatrix testX = new DoubleMatrix(new double[][]
                {{0.5, 0.5, 0., 0., 0., 0.},
                {0., 0., 0., 0.5, 0.5, 0.},
                {0.5, 0.5, 0.5, 0.5, 0.5, 0.}});
		
		log.info(dbn.predict(testX).toString());

	}
	

	
}
