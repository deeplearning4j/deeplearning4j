package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class MnistSdaTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistSdaTest.class);

	@Test
	public void testMnist() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> first = this.getMnistExample(1);
		int numIns = first.getFirst().columns;
		int numLabels = 10;
		int[] layerSizes = new int[3];
		Arrays.fill(layerSizes,300);
		double lr = 0.1;
		SdAMatrix sda = new SdAMatrix.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();

		int numCorrect = 0;
		Pair<DoubleMatrix,DoubleMatrix> curr = this.getMnistExampleBatch(2000);
		
		
		sda.pretrain(curr.getFirst(), lr, 0.3, 500);
		sda.finetune(curr.getSecond(), lr,200);
		
		DoubleMatrix predicted = sda.predict(curr.getFirst());
		
		for(int i = 0; i < curr.getFirst().rows; i++) {
			DoubleMatrix row = curr.getSecond().getRow(i);
			DoubleMatrix predictedRow = predicted.getRow(i);
			int actualMax = SimpleBlas.iamax(row);
			int predictedMax = SimpleBlas.iamax(predictedRow);
			if(actualMax == predictedMax)
				numCorrect++;
		}
		
		
		log.info("Correct " + numCorrect);



	}

}
