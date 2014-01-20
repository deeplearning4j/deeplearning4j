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
		int[] hiddenLayerSizes = new int[] {4,7,9};
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

		
		DoubleMatrix testY = new DoubleMatrix(new double[][] {
				{1,0},
				{1,0},
				{0,1}
		});

		DoubleMatrix predict = dbn.predict(testX);
		log.info(predict.toString());
		
		Evaluation eval = new Evaluation();
		eval.eval(testY, predict);
		log.info(eval.stats());
		
		
		DBN decoder = new DBN.Builder().buildEmpty();
		decoder.asDecoder(dbn);
		assertEquals(dbn.nOuts,decoder.nIns);
		assertEquals(dbn.nIns,decoder.nOuts);
		assertEquals(decoder.nLayers,dbn.nLayers);
		boolean e = Arrays.equals(new int[]{9,7,4},decoder.hiddenLayerSizes);
		
		assertEquals(true,e);
	/*	assertEquals(decoder.layers[0].getnHidden(),dbn.layers[dbn.layers.length - 1].getnVisible());
		assertEquals(decoder.layers[0].getnVisible(),dbn.layers[dbn.layers.length - 1].getnHidden());
		assertEquals(decoder.sigmoidLayers[0].n_in,dbn.sigmoidLayers[dbn.layers.length - 1].n_out);
		assertEquals(decoder.sigmoidLayers[0].n_out,dbn.sigmoidLayers[dbn.layers.length - 1].n_in);

		
		assertEquals(decoder.layers[1].getnHidden(),dbn.layers[dbn.layers.length - 2].getnVisible());
		assertEquals(decoder.layers[1].getnVisible(),dbn.layers[dbn.layers.length - 2].getnHidden());
		assertEquals(decoder.sigmoidLayers[1].n_in,dbn.sigmoidLayers[dbn.layers.length - 2].n_out);
		assertEquals(decoder.sigmoidLayers[1].n_out,dbn.sigmoidLayers[dbn.layers.length - 2].n_in);*/

		decoder.pretrain(predict, 1, 0.1, 1000);
		decoder.finetune(testX, 0.1, 1000);
	
		
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
