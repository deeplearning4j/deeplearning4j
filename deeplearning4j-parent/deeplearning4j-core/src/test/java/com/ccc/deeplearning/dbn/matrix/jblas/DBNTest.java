package com.ccc.deeplearning.dbn.matrix.jblas;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Arrays;

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
				});
		DoubleMatrix   y = new DoubleMatrix(new double[][]  
				{{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1}});


		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.1;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {6,6,6};
		double fineTuneLr = 0.1;
		int fineTuneEpochs = 200;

		DBN dbn = new DBN.Builder().useRegularization(true).renderWeights(1)
				.hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns)
				.numberOfOutPuts(nOuts).withRng(rng).build();
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		//plotter.plot(dbn.layers[0]);

		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);

		plotter.plotNetworkGradient(dbn.layers[0],dbn.layers[0].getGradient(new Object[]{k,preTrainLr}));

		dbn.finetune(y,fineTuneLr, fineTuneEpochs);



		DoubleMatrix testX = new DoubleMatrix(new double[][]
				{{1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0},
				{1, 1, 1, 1, 1, 0}});


		DoubleMatrix testY = new DoubleMatrix(new double[][] {
				{1,0},
				{1,0},
				{0,1}
		});

		DoubleMatrix predict = dbn.predict(x);
		log.info(predict.toString());

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
