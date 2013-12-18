package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas.CDBN;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas.DBN;

public class MnistDbnTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistDbnTest.class);

	@Test
	public void testMnist() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> first = this.getMnistExample(1);
		int numIns = first.getFirst().columns;
		int numLabels = 10;
		int[] layerSizes = new int[4];
		Arrays.fill(layerSizes,500);

		double lr = 0.1;
		DBN dbn = new DBN.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();
	
		List<Pair<DoubleMatrix,DoubleMatrix>> list = this.getMnistExampleBatches(10, 500);
		for(int iter = 0; iter < list.size(); iter++) {
			Pair<DoubleMatrix,DoubleMatrix> curr = list.get(iter);
			int numCorrect = 0;

			if(curr.getFirst().rows != curr.getSecond().rows)
				throw new IllegalArgumentException("Rows are not the same");
			
			dbn.pretrain(curr.getFirst(),2, lr, 200);
			dbn.finetune(curr.getSecond(), lr,500);
			
			DoubleMatrix predicted = dbn.predict(curr.getFirst());
			DoubleMatrix y = curr.getSecond();
			
			for(int i = 0; i < curr.getFirst().rows; i++) {
				DoubleMatrix actualRow = y.getRow(i);
				DoubleMatrix predictedRow = predicted.getRow(i);
				int actualMax = SimpleBlas.iamax(actualRow);
				int predictedMax = SimpleBlas.iamax(predictedRow);
				if(actualMax == predictedMax)
					numCorrect++;
			}
			
			
			log.info("Correct " + numCorrect + " on iteration " + iter);

		}
		
		


	}

}
