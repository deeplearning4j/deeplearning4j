package com.ccc.deeplearning.dbn.matrix.jblas;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.plot.NeuralNetPlotter;
import com.ccc.deeplearning.rbm.CRBM;
import com.ccc.deeplearning.transformation.MultiplyScalar;


public class DBNTest  extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(DBNTest.class);


	@Test
	public void testDBN() {
		
		int n = 1000000;
		DoubleMatrix x = DoubleMatrix.rand(n,2);
		x = x.gti(0.5);
		
		DoubleMatrix y = DoubleMatrix.zeros(n,2);
		for(int i = 0; i < x.rows; i++) {
			if(x.get(i,0) == x.get(i,1))
				y.put(i,0,1);
			else
				y.put(i,1,1);
		}
		
		
	


		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.001;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 2,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {4,4,4};
		double fineTuneLr = 0.001;
		int fineTuneEpochs = 1000;

		DBN dbn = new DBN.Builder()
				.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns).renderWeights(100)
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
